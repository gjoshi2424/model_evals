import json
import logging

from inspect_ai.agent import AgentPrompt, AgentState, AgentSubmit, react, run
from inspect_ai.model import ChatMessage, ChatMessageTool, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool

from database import cost_enquiry
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

# Maximum tool-call steps per agent run,
MAX_STEPS: int = 30

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
            # Build system prompt: base instructions + accumulated reflections
            if reflections:
                reflection_block = REFLECTION_HEADER + "\n".join(reflections)
                system_prompt = reflection_block + "\n\n" + REACT_AGENT_SYSTEM_PROMPT
            else:
                system_prompt = REACT_AGENT_SYSTEM_PROMPT

            agent = make_react_agent(system_prompt)
            agent_state = await run(agent, user_input)
            state.output = agent_state.output

            # Check whether the agent actually called submit() successfully.
            submitted = any(
                isinstance(m, ChatMessageTool)
                and m.function == "submit"
                and m.error is None
                for m in agent_state.messages
            )
            if submitted:
                logger.debug(
                    "Sample %s: Reflexion finished on attempt %d.",
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
                "Sample %s: Reflexion retry %d/%d. Reflection: %s",
                state.sample_id,
                retry + 1,
                MAX_STEPS,
                reflection_text,
            )

        return state

    return solve
