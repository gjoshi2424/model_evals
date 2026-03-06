"""
TravelPlanner: A Benchmark for Real-World Planning with Language Agents.

Zheng et al., 2024 (https://arxiv.org/abs/2402.01622)

This module implements the sole-planning (direct strategy) task from
TravelPlanner. The model receives pre-collected reference information
alongside a travel query, and must produce a complete multi-day itinerary
covering transportation, meals, attractions, and accommodation.

Evaluation follows the original paper's constraint framework:
  - Commonsense constraints (city routing, diverse meals/attractions, etc.)
  - Hard constraints (budget, cuisine, room type, house rules, transportation)
"""

from inspect_ai import Task, task

from travel_planner_eval._version import EVAL_VERSION
from travel_planner_eval.dataset import Split, travel_planner_dataset
from travel_planner_eval.scorer import travel_planner_scorer
from travel_planner_eval.solver import sole_planning_direct


@task
def travel_planner_direct(
    split: Split = "validation",
    parse_model: str | None = None,
) -> Task:
    """TravelPlanner sole-planning evaluation (direct strategy).

    The model receives the full reference information (pre-collected travel data)
    along with a natural-language travel query and must output a complete
    day-by-day travel plan. This is the "direct" strategy from the original paper:
    a single-pass generation with no chain-of-thought or reflection.

    The scoring replicates the original two-step process:
      1. An LLM parses the natural-language plan into a structured JSON format.
      2. Commonsense and hard constraints are evaluated on the JSON plan.

    Args:
        split: HuggingFace dataset split to evaluate.
            "validation" (180 samples) or "train" (45 samples).
            The test split is not supported (no ground-truth labels).
        parse_model: Model used to parse natural-language plan output into
            structured JSON (the postprocessing step). Defaults to the
            task model if None. The original paper used gpt-4-1106-preview.

    Returns:
        Configured Inspect Task.
    """
    return Task(
        dataset=travel_planner_dataset(split=split),
        solver=sole_planning_direct(),
        scorer=travel_planner_scorer(parse_model=parse_model),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
