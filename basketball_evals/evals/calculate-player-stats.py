from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import prompt_template, generate, use_tools, system_message
from inspect_ai.scorer import match
from inspect_ai.model import GenerateConfig
from inspect_ai.tool import tool


PROMPT_TEMPLATE_WITH_TOOLS = """
Solve the following basketball stats problem using the given tools. You can trust the result from them. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to 
the problem, and you do not need to use a \\boxed command.

""".strip()

PROMPT_TEMPLATE_WITHOUT_TOOLS = """
Solve the following basketball stats problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to 
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def record_to_sample(record):
    """Convert to Inspect Sample.
    
    Args:
        record: Contains input question, target answer, and reasoning.
        
    Returns:
        Sample object with input, target, and reasoning stored in metadata.
    """
    question = record["input"]
    answer = str(record['target'])
    metadata = {"reasoning": record['reasoning']}

    return Sample(
        input=question,
        target=answer,
        metadata=metadata
    )

def sample_to_fewshot(sample):
    """Format a sample as a few-shot example with reasoning.
    
    Args:
        sample: Sample object with input, target, and reasoning metadata.
        
    Returns:
        Formatted string showing the question, step-by-step reasoning, and answer.
    """
    return (
        f"{sample.input}\n\Reasoning:\n"
        + f"{sample.metadata['reasoning']}\n\n"
        + f"ANSWER: {sample.target}"
    )

@tool
def efg():
    async def execute(FG: str, FGA: str, three_P: str):
        """
        Calculate effective field goal percentage.

        Args:
            FG: Total field goals made.
            FGA: Total field goals attempted.
            three_P: Three-point field goals made.

        Returns:
            The effective field goal percentage rounded to 3 decimal places.
        """
        return round((float(FG) + 0.5 * float(three_P)) / float(FGA), 3)

    return execute

@tool
def true_shooting():
    async def execute(PTS: str, FGA: str, FTA: str):
        """
        Calculate true shooting percentage.

        Args:
            PTS: Total points scored.
            FGA: Total field goals attempted.
            FTA: Total free throws attempted.

        Returns:
            The true shooting percentage rounded to 3 decimal places.
        """
        pts = float(PTS)
        fga = float(FGA)
        fta = float(FTA)
        return round(pts / (2 * (fga + 0.44 * fta)), 3)

    return execute

@tool
def three_point_attempt_rate():
    async def execute(three_PA: str, FGA: str):
        """
        Calculate 3-point attempt rate.

        Args:
            three_PA: Three-point field goals attempted.
            FGA: Total field goals attempted.

        Returns:
            The 3-point attempt rate rounded to 3 decimal places.
        """
        return round(float(three_PA) / float(FGA), 3)

    return execute

@tool
def free_throw_rate():
    async def execute(FTA: str, FGA: str):
        """
        Calculate free throw rate.

        Args:
            FTA: Total free throws attempted.
            FGA: Total field goals attempted.

        Returns:
            The free throw rate rounded to 3 decimal places.
        """
        return round(float(FTA) / float(FGA), 3)

    return execute


@task
def basketball_stats(pass_tools = False, few_shot = False, limit=25):
    """
    Tests mathematical reasoning by asking models to compute stats
    from player data.
    
    Args:
        pass_tools: If True, provides calculator tools for each stat formula.
        few_shot: If True, includes few-shot examples with reasoning.
        
    Returns:
        Task configured with dataset, solver chain, and numeric matcher scorer.
        
    """
    solver = [generate()]
    dataset = json_dataset("../data/player-season-stats-questions.jsonl", sample_fields=record_to_sample, shuffle=True, seed=25, limit=limit)

    if(pass_tools):
        solver.insert(0, prompt_template(PROMPT_TEMPLATE_WITH_TOOLS))
        solver.insert(1, use_tools(efg(), true_shooting(), three_point_attempt_rate(), free_throw_rate()))
    else:
        solver.insert(0, prompt_template(PROMPT_TEMPLATE_WITHOUT_TOOLS))
    if(few_shot):
        fewshot_data = json_dataset("../data/player-season-stats-fewshot.jsonl", sample_fields=record_to_sample)
        sys_message = "\n\n".join([sample_to_fewshot(sample) for sample in fewshot_data])
        solver.insert(0, system_message(sys_message))
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=match(numeric=True),
        config=GenerateConfig(temperature=0.0)
    )
   