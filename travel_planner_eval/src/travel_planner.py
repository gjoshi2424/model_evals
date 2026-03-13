from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.solver import Solver

from dataset import Split, travel_planner_dataset
from prompts import (
    COT_PLANNER_INSTRUCTION,
    DIRECT_PLANNER_INSTRUCTION,
    REACT_USER_TEMPLATE,
)
from scorer import travel_planner_scorer
from solver import (
    sole_planning,
    sole_planning_react,
    sole_planning_reflexion,
)

Strategy = Literal["direct", "cot", "react", "reflexion"]


@task
def travel_planner(
    strategy: Strategy = "direct",
    split: Split = "validation",
    parse_model: str | None = None,
) -> Task:
    """TravelPlanner sole-planning evaluation.

    Args:
        strategy: Planning strategy to use. Defaults to ``"direct"``.
        split: HuggingFace dataset split to evaluate.
            ``"validation"`` (180 samples) or ``"train"`` (45 samples).
            The test split is not supported (no ground-truth labels).
        parse_model: Model used to parse plan output into JSON for scoring.
            Defaults to the task model if None.

    Returns:
        Configured Inspect Task.
    """
    solvers_dict: dict[Strategy, Solver | Agent] = {
        "direct": sole_planning(),
        "cot": sole_planning(),
        "react": sole_planning_react(),
        "reflexion": sole_planning_reflexion(),
    }
    instructions_dict: dict[Strategy, str] = {
        "direct": DIRECT_PLANNER_INSTRUCTION,
        "cot": COT_PLANNER_INSTRUCTION,
        "react": REACT_USER_TEMPLATE,
        "reflexion": REACT_USER_TEMPLATE,
    }

    return Task(
        dataset=travel_planner_dataset(
            split=split,
            planner_instruction=instructions_dict[strategy],
        ),
        solver=solvers_dict[strategy],
        scorer=travel_planner_scorer(parse_model=parse_model),
    )
