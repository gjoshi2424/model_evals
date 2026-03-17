import json
import logging

from inspect_ai.agent import AgentPrompt, AgentState, AgentSubmit, react, run
from inspect_ai.model import ChatMessage, ChatMessageTool, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool

from database import COST_ENQUIRY_ERROR_PREFIX, cost_enquiry
from prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    REFLECT_INSTRUCTION,
    REFLECTION_HEADER,
)

logger = logging.getLogger(__name__)


def msg_text(m: ChatMessage) -> str:
    """Safely extract text from any ChatMessage subtype."""
    if isinstance(m.content, str):
        return m.content
    if isinstance(m.content, list):
        return " ".join(getattr(part, "text", "") for part in m.content)
    return ""


# Maximum reflection retries for the Reflexion strategy.
REFLEXION_MAX_RETRIES: int = 3

# Maximum tool-call steps per agent run.
MAX_STEPS: int = 30

# Number of consecutive CostEnquiry failures before the react loop exits early
# and reflexion is triggered.
COST_ENQUIRY_ERROR_THRESHOLD: int = 3

@tool
def cost_enquiry_tool() -> Tool:
    """Tool that calculates the cost of a one-day travel sub-plan."""

    async def execute(plan_json: str) -> str:
        """Calculate the daily cost of a sub-plan.

        Args:
            plan_json: JSON string with keys: people_number, day, current_city,
                transportation, breakfast, attraction, lunch, dinner, accommodation.
                Example: '{"people_number": 7, "day": 1, "current_city": "from Ithaca to Charlotte",
                "transportation": "Flight Number: F3633413, from Ithaca to Charlotte",
                "breakfast": "Nagaland's Kitchen, Charlotte", "attraction": "-",
                "lunch": "Cafe Maple Street, Charlotte", "dinner": "Bombay Vada Pav, Charlotte",
                "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}'
        """
        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError:
            return "Invalid JSON. Please provide a valid JSON string for the plan."
        return cost_enquiry(plan)

    return execute


def make_react_agent(system_prompt: str) -> object:
    """Create a react() agent with the given system prompt.

    Args:
        system_prompt: System-level instructions injected into the agent via AgentPrompt.

    Returns:
        A configured react() Agent instance with at most MAX_STEPS tool-call iterations.
    """
    steps = 0

    async def on_continue(state: AgentState) -> bool | str:
        nonlocal steps
        steps += 1
        if steps >= MAX_STEPS:
            return False
        return True

    return react(
        prompt=AgentPrompt(instructions=system_prompt),
        tools=[cost_enquiry_tool()],
        on_continue=on_continue,
        submit=AgentSubmit(keep_in_messages=True),
    )


def make_reflexion_react_agent(system_prompt: str) -> tuple[object, list[int]]:
    """Create a react() agent that tracks consecutive CostEnquiry failures.

    Returns:
        A tuple of (agent, consecutive_errors) where consecutive_errors[0] holds
        the running error count. The agent exits early once
        COST_ENQUIRY_ERROR_THRESHOLD failures occur.
    """
    steps = 0
    consecutive_errors = [0]

    async def on_continue(state: AgentState) -> bool | str:
        nonlocal steps
        steps += 1
        if steps >= MAX_STEPS:
            return False
        last = state.messages[-1] if state.messages else None
        if isinstance(last, ChatMessageTool) and last.function == "cost_enquiry_tool":
            if last.error is not None or (
                isinstance(last.content, str)
                and last.content.startswith(COST_ENQUIRY_ERROR_PREFIX)
            ):
                consecutive_errors[0] += 1
            else:
                consecutive_errors[0] = 0  # successful call resets the counter
        if consecutive_errors[0] >= COST_ENQUIRY_ERROR_THRESHOLD:
            return False
        return True

    agent = react(
        prompt=AgentPrompt(instructions=system_prompt),
        tools=[cost_enquiry_tool()],
        on_continue=on_continue,
        submit=AgentSubmit(keep_in_messages=True),
    )
    return agent, consecutive_errors


@solver
def sole_planning() -> Solver:
    """Sole-planning solver for the direct and CoT strategies.

    Returns:
        Solver that generates a single completion for the planner prompt.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state)

    return solve


@solver
def sole_planning_react() -> Solver:
    """Sole-planning solver using the ReAct strategy via inspect_ai.agent.react().

    A new agent is created per sample so the step counter is not shared
    across samples.

    Returns:
        Solver implementing the ReAct planning strategy.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent = make_react_agent(REACT_AGENT_SYSTEM_PROMPT)
        agent_state = await run(agent, state.input)
        state.output = agent_state.output
        return state

    return solve


@solver
def sole_planning_reflexion() -> Solver:
    """Sole-planning solver using the Reflexion strategy.

    Returns:
        Solver implementing the Reflexion planning strategy.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        text: str = state.metadata["reference_information"]
        query: str = state.metadata["query"]
        user_input = state.input
        reflections: list[str] = []

        for retry in range(REFLEXION_MAX_RETRIES):
            # Build system prompt: base instructions + example, then reflections appended
            if reflections:
                reflection_block = REFLECTION_HEADER + "Reflections:\n- " + "\n- ".join(r.strip() for r in reflections)
                system_prompt = REACT_AGENT_SYSTEM_PROMPT + "\n\n" + reflection_block
            else:
                system_prompt = REACT_AGENT_SYSTEM_PROMPT

            agent, error_counter = make_reflexion_react_agent(system_prompt)
            agent_state = await run(agent, user_input)
            state.output = agent_state.output

            if error_counter[0] < COST_ENQUIRY_ERROR_THRESHOLD:
                logger.debug(
                    "Sample %s: Reflexion finished on attempt %d (threshold not reached).",
                    state.sample_id,
                    retry + 1,
                )
                return state

            # Reflexion step: ask the model to diagnose the failure
            history = "\n".join(
                f"[{m.role}] {msg_text(m)}" for m in agent_state.messages
            )
            reflect_prompt = REFLECT_INSTRUCTION.format(
                text=text,
                query=query,
                scratchpad=history,
            )
            reflection_output = await get_model().generate(
                [ChatMessageUser(content=reflect_prompt)]
            )
            reflection_text = reflection_output.completion.strip()
            reflections.append(reflection_text)
            logger.info(
                "Sample %s: Reflexion retry %d/%d (%d CostEnquiry errors). Reflection: %s",
                state.sample_id,
                retry + 1,
                REFLEXION_MAX_RETRIES,
                error_counter[0],
                reflection_text,
            )

        return state

    return solve
