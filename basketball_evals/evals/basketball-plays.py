"""Basketball plays evaluation using multiple choice questions.

This module evaluates LLM ability to identify basketball plays and schemes
from textual descriptions using a multiple choice format.
"""
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import multiple_choice, system_message
from inspect_ai.scorer import choice
from inspect_ai.model import GenerateConfig

SYSTEM_MESSAGE = """
Choose the correct basketball scheme or play based on the description given.
"""


def record_to_sample(record):
    """Convert a JSON record to an Inspect Sample.
    
    Args:
        record: Contains input, target, and choices fields.
        
    Returns:
        Sample object with input text, target answer, and multiple choice options.
    """
    return Sample(
        input=record["input"], target=record["target"], choices=record["choices"]
    )


def sample_to_fewshot(sample):
    """Format a sample as a few-shot example.
    
    Args:
        sample: Sample object with input, choices, and target.
        
    Returns:
        Formatted string showing the question, choices, and answer.
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(sample.choices)])
    return (
        f"{sample.input}\n\n"
        + f"{choices_text}\n\n"
        + f"\n\nANSWER: {sample.target}"
    )


@task
def basketball_mcq_eval(fewshot=True):
    """
    Tests knowledge recall of basketball schemes and plays using multiple choice
    questions.
    
    Args:
        fewshot: If True, includes few-shot examples in the system message.
        
    Returns:
        Task configured with dataset, solver, and scorer.
    """
    solver = [multiple_choice()]
    if fewshot:
        fewshot_data = json_dataset(
            "../data/plays-few-shot.jsonl", sample_fields=record_to_sample
        )
        fewshot_examples = [sample_to_fewshot(sample) for sample in fewshot_data]
        fewshot_text = "\n\n".join(fewshot_examples)
        full_message = f"{SYSTEM_MESSAGE}\n\nHere are some examples:\n\n{fewshot_text}"
        solver.insert(0, system_message(full_message))
    else:
        solver.insert(0, system_message(SYSTEM_MESSAGE))
    return Task(
        dataset=json_dataset("../data/plays.jsonl", sample_fields=record_to_sample),
        solver=solver,
        scorer=choice(),
        config=GenerateConfig(temperature=0.0)
    )
