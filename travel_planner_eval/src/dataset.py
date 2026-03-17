import ast
from typing import Any, Literal

from inspect_ai.dataset import Dataset, Sample, hf_dataset

from prompts import DIRECT_PLANNER_INSTRUCTION

DATASET = "osunlp/TravelPlanner"
DATASET_REVISION = "8736504ecfc31b7f8b7e40122873c337e83fff7c"

# Available evaluated splits
Split = Literal["train", "validation"]


def travel_planner_dataset(
    split: Split = "validation",
    planner_instruction: str = DIRECT_PLANNER_INSTRUCTION,
) -> Dataset:
    """Load the TravelPlanner dataset from HuggingFace.

    Args:
        split: Dataset split to load ("train" has 45 samples,
            "validation" has 180). The test split is not supported
            because it has no ground-truth labels.
        planner_instruction: Prompt template for the sample input.

    Returns:
        Inspect Dataset of TravelPlanner samples.
    """

    return hf_dataset(
        DATASET,
        name=split,
        split=split,
        sample_fields=lambda record: record_to_sample(record, planner_instruction),
        revision=DATASET_REVISION,
    )


def record_to_sample(record: dict[str, Any], planner_instruction: str) -> Sample:
    """Convert a single HuggingFace dataset record into an Inspect Sample.

    Args:
        record: A single row from the HuggingFace dataset.
        planner_instruction: Prompt template string with ``{text}`` and
            ``{query}`` placeholders.

    Returns:
        Inspect Sample with the formatted planner prompt as input and all
        query constraint fields stored in metadata for the scorer
    """
    local_constraint = record["local_constraint"]
    if isinstance(local_constraint, str):
        local_constraint = ast.literal_eval(local_constraint)

    text = record["reference_information"]
    query = record["query"]

    input_text = planner_instruction.format(text=text, query=query)

    return Sample(
        input=input_text,
        metadata={
            "reference_information": text,
            "query": query,
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
