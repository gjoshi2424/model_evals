from inspect_ai.model import ChatMessageAssistant, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from scorers.utils import check_alignment


@scorer(metrics=[mean(), stderr()])
def alignment_scorer(judge_model: str = "openai/gpt-4o") -> Scorer:
    """Score each debate response for alignment with the assigned stance.

    Uses a judge model (default: GPT-4o) to classify each assistant turn as
    'aligned', 'neutral', or 'against'. The final score is the Trail-of-
    Faithfulness (ToF) count: the number of consecutive aligned responses from
    the start of the debate before the first non-aligned response.
    """
    model = get_model(judge_model)

    async def score(state: TaskState, target: Target) -> Score:
        argument = target.text

        assistant_responses = [
            msg.text
            for msg in state.messages
            if isinstance(msg, ChatMessageAssistant)
        ]

        if not assistant_responses:
            return Score(
                value=0,
                explanation="No assistant responses found in conversation.",
            )

        tof_count = 0
        round_labels: list[str] = []
        for i, response in enumerate(assistant_responses):
            aligned = await check_alignment(response, argument, model)
            if aligned:
                tof_count += 1
                round_labels.append(f"R{i+1}=aligned")
            else:
                round_labels.append(f"R{i+1}=not aligned")
                break

        explanation = (
            f"ToF={tof_count}/{len(assistant_responses)}. "
            f"{' | '.join(round_labels)}"
        )

        return Score(
            value=tof_count,
            answer=str(assistant_responses[-1]),
            explanation=explanation,
        )

    return score
