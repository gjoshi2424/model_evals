"""
Dataset loading for TravelPlanner evaluation.

Loads the osunlp/TravelPlanner dataset from HuggingFace and converts each
record into an Inspect Sample. The full planner prompt (PLANNER_INSTRUCTION
with reference_information and query) is used as the sample input, following
the original sole-planning direct strategy from tools/planner/sole_planning.py.
"""

from typing import Any, Literal

from inspect_ai.dataset import Dataset, Sample, hf_dataset

from travel_planner_eval.prompts import PLANNER_INSTRUCTION

# HuggingFace dataset identifier
DATASET = "osunlp/TravelPlanner"
DATASET_REVISION = "8736504ecfc31b7f8b7e40122873c337e83fff7c"

# Available evaluated splits (test has no labels)
Split = Literal["train", "validation"]


def travel_planner_dataset(split: Split = "validation") -> Dataset:
    """Load the TravelPlanner dataset from HuggingFace.

    Args:
        split: Dataset split to load ("train" has 45 samples,
            "validation" has 180). The test split is not supported
            because it has no ground-truth labels.

    Returns:
        Inspect Dataset of TravelPlanner samples.
    """
    return hf_dataset(
        DATASET,
        name=split,
        split=split,
        sample_fields=_record_to_sample,
        revision=DATASET_REVISION,
    )


def _record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a single HuggingFace dataset record into an Inspect Sample.

    Args:
        record: A single row from the HuggingFace dataset.

    Returns:
        Inspect Sample with the formatted planner prompt as input and all
        query constraint fields stored in metadata for the scorer.
    """
    # Parse local_constraint if it came through as a string (happens in some HF splits)
    local_constraint = record["local_constraint"]
    if isinstance(local_constraint, str):
        local_constraint = eval(local_constraint)  # noqa: S307

    # Build the full prompt as a single user message (matching original HumanMessage approach)
    input_text = PLANNER_INSTRUCTION.format(
        text=record["reference_information"],
        query=record["query"],
    )

    return Sample(
        input=input_text,
        metadata={
            # Query constraint fields needed by the scorer
            "query": record["query"],
            "org": record["org"],
            "dest": record["dest"],
            "days": record["days"],
            "visiting_city_number": record["visiting_city_number"],
            "people_number": record["people_number"],
            "budget": record["budget"],
            "local_constraint": local_constraint,
            "level": record["level"],
            "date": record["date"],
        },
    )
