"""
Solver for TravelPlanner sole-planning (direct strategy).

The direct planning strategy simply passes the full formatted prompt
(PLANNER_INSTRUCTION with reference_information and query) to the model
and collects its response. This mirrors the original Planner.run() method
in tools/planner/apis.py, which calls:

    self.llm([HumanMessage(content=self._build_agent_prompt(text, query))]).content

In Inspect terms this is a single generate() call — no system message,
no multi-turn loop — because the entire prompt is already in the sample input.
"""

from inspect_ai.solver import Generate, Solver, TaskState, solver


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
