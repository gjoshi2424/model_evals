"""
Solvers for TravelPlanner sole-planning strategies.

Four strategies are implemented, mirroring the original TravelPlanner repo:

- **direct**: Single generate call.  Original: Planner.run() in tools/planner/apis.py
- **cot**: Same as direct but uses the CoT prompt (ends "Let's think step by step. First, ").
- **react**: Uses inspect_ai.agent.react() with cost_enquiry as a real @tool.
  The model calls cost_enquiry(plan_json) to verify day costs, then submit(plan) to finish.
  Original: ReactPlanner.run() in tools/planner/apis.py
- **reflexion**: Wraps the react agent in a retry loop with a self-reflection step
  generated between attempts when the agent fails to produce a complete plan.
  Original: ReactReflectPlanner.run() in tools/planner/apis.py

All four strategies are "sole-planning": the model receives pre-collected reference
information together with the query and produces a complete travel plan.
"""

import json
import logging

from inspect_ai.agent import Agent, AgentPrompt, AgentState, react, run
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool

from database import cost_enquiry as _cost_enquiry_fn
from prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    REFLECT_INSTRUCTION,
    REFLECTION_HEADER,
)

logger = logging.getLogger(__name__)

# Maximum reflection retries for the Reflexion strategy.
_REFLEXION_MAX_RETRIES: int = 3

# Maximum tool-call steps per agent run, matching self.max_steps = 30 in the
# original ReactPlanner / ReactReflectPlanner (tools/planner/apis.py).
_MAX_STEPS: int = 30

_CONTINUE_MESSAGE = "Please proceed to the next step using your best judgement."



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
        return _cost_enquiry_fn(plan)

    return execute



def _make_react_agent(system_prompt: str) -> object:
    """Create a react() agent with the given system prompt and cost_enquiry tool.

    Uses AgentPrompt to supply only our instructions, suppressing inspect's
    default assistant/submit prompt additions. Halts after _MAX_STEPS tool calls,
    matching self.max_steps = 30 in the original ReactPlanner (tools/planner/apis.py).
    """
    steps = 0

    async def on_continue(state: AgentState) -> bool | str:
        nonlocal steps
        steps += 1
        if steps >= _MAX_STEPS:
            return False
        return _CONTINUE_MESSAGE

    return react(
        prompt=AgentPrompt(
            instructions=system_prompt
        ),
        tools=[cost_enquiry_tool()],
        on_continue=on_continue,
    )


@solver
def sole_planning_direct() -> Solver:
    """Sole-planning solver using the direct strategy.

    Passes the pre-formatted planner prompt (stored as the sample input)
    directly to the model and returns the completion as the answer.
    No system message is used, matching the original single-HumanMessage approach.

    Returns:
        Solver that generates a single completion for the planner prompt.
    """

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        return await generate_fn(state)

    return solve


@solver
def sole_planning_cot() -> Solver:
    """Sole-planning solver using the chain-of-thought (CoT) strategy.

    Identical to the direct strategy at inference time: a single generate call.
    The prompt itself drives the CoT behaviour — it ends with
    "Let's think step by step. First, " which elicits step-by-step reasoning.
    The prompt is set at the dataset level via COT_PLANNER_INSTRUCTION.

    Returns:
        Solver that generates a single completion for the CoT planner prompt.
    """

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        return await generate_fn(state)

    return solve



def sole_planning_react() -> Agent:
    """Sole-planning solver using the ReAct strategy via inspect_ai.agent.react().

    Returns an Agent directly (Task accepts Solver | Agent), which inspect_ai
    runs against the sample input automatically — no manual run() wrapper needed.
    This matches how cybench and cybergym use react() directly as a solver.

    The model uses inspect's native tool-calling mechanism:
    - ``cost_enquiry(plan_json)`` — calculates the daily cost from the local database.
    - ``submit(plan)`` — signals completion; the argument becomes the final answer.

    Returns:
        Agent implementing the ReAct planning strategy.
    """
    return _make_react_agent(REACT_AGENT_SYSTEM_PROMPT)



@solver
def sole_planning_reflexion() -> Solver:
    """Sole-planning solver using the Reflexion strategy.

    Wraps the react agent in a retry loop. When an attempt ends without a
    submitted plan, the model is asked to diagnose its failure via
    REFLECT_INSTRUCTION. The resulting reflection is prepended to the system
    prompt of the next attempt (matching REACT_REFLECT_PLANNER_INSTRUCTION
    from the original ReactReflectPlanner in tools/planner/apis.py).

    Up to 3 reflection retries are used (matching max_retry_step = 3).

    Returns:
        Solver implementing the Reflexion planning strategy.
    """

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        text: str = state.metadata["reference_information"]
        query: str = state.metadata["query"]
        user_input = state.input.text
        reflections: list[str] = []

        for retry in range(_REFLEXION_MAX_RETRIES):
            # Build system prompt: base instructions + accumulated reflections
            if reflections:
                reflection_block = REFLECTION_HEADER + "\n".join(reflections)
                system_prompt = reflection_block + "\n\n" + REACT_AGENT_SYSTEM_PROMPT
            else:
                system_prompt = REACT_AGENT_SYSTEM_PROMPT

            agent = _make_react_agent(system_prompt)
            agent_state = await run(agent, user_input)
            state.output = agent_state.output

            # If the agent submitted a plan (output has content), we're done.
            # A halted/empty-output agent will have a short or empty completion.
            plan = state.output.completion.strip()
            if plan:
                logger.debug(
                    f"Sample {state.sample_id}: Reflexion finished on attempt {retry + 1}."
                )
                return state

            # --- Reflexion step: ask the model to diagnose the failure ---
            history = "\n".join(
                str(m.content) for m in agent_state.messages if hasattr(m, "content")
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
                f"Sample {state.sample_id}: Reflexion retry {retry + 1}/"
                f"{_REFLEXION_MAX_RETRIES}. Reflection: {reflection_text[:100]}"
            )

        return state

    return solve
