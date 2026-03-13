import logging
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from constraints import commonsense, hard
from prompts import FORMAT_CONVERSION_PREFIX
from utils import parse_json_plan

logger = logging.getLogger(__name__)


class ScoreExplanation:
    """Human-readable labels for score outcomes."""

    EMPTY_PLAN = "Model produced an empty or whitespace-only plan"
    PARSE_FAILED = "LLM-based plan parsing failed; could not extract JSON"
    CONSTRAINT_FAIL = "One or more constraint checks failed"
    ALL_PASS = "All available constraint checks passed"


@scorer(metrics=[accuracy(), stderr()])
def travel_planner_scorer(
    parse_model: str | None = None,
) -> Scorer:
    """Score a TravelPlanner sole-planning sample.
    Args:
        parse_model: Model to use for the JSON parsing step. If None, uses
            the same model that generated the plan (the task model).
            The original paper used gpt-4-1106-preview for this step.

    Returns:
        Scorer function that produces a binary Score (CORRECT / INCORRECT)
        with per-constraint details in Score.metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        plan_text: str = state.output.completion.strip()
        if not plan_text:
            logger.warning(f"Sample {state.sample_id}: empty plan output")
            return Score(
                value=INCORRECT,
                answer=plan_text,
                explanation=ScoreExplanation.EMPTY_PLAN,
                metadata={"delivered": False},
            )
        # Step 1: Parse natural-language plan → structured JSON
        parsing_prompt = FORMAT_CONVERSION_PREFIX + f"Text:\n{plan_text}\nJSON:\n"
        # get_model(None) returns the current task model; passing a string uses a
        # separate model. Either way, parsing costs appear in that model's log usage.
        model = get_model(parse_model)
        parse_response = await model.generate(
            input=[
                ChatMessageSystem(content="You are a helpful assistant."),
                ChatMessageUser(content=parsing_prompt),
            ]
        )
        parse_output = parse_response.completion.strip()

        parsed_plan: list[dict[str, Any]] | None = parse_json_plan(parse_output)

        if parsed_plan is None:
            logger.warning(
                f"Sample {state.sample_id}: JSON parsing failed. "
                f"Parser output: {parse_output[:200]}"
            )
            return Score(
                value=INCORRECT,
                answer=plan_text,
                explanation=ScoreExplanation.PARSE_FAILED,
                metadata={
                    "delivered": True,
                    "parse_failed": True,
                    "parser_output": parse_output[:500],
                },
            )

        # Step 2: Evaluate constraints
        # (hard constraints are only evaluated if the plan is not absent).
        query_data: dict[str, Any] = state.metadata
        commonsense_results = commonsense.evaluation(query_data, parsed_plan)
        not_absent_pass = commonsense_results["is_not_absent"][0]
        sandbox_pass = commonsense_results["is_valid_information_in_sandbox"][0]
        if not_absent_pass and sandbox_pass:
            hard_results = hard.evaluation(query_data, parsed_plan)
        else:
            hard_results = None

        # Determine final pass/fail
        commonsense_pass = all_pass(commonsense_results)
        hard_pass = all_pass(hard_results) if hard_results is not None else None

        final_pass = commonsense_pass and (hard_pass is not False)

        # Build flat metadata for Score
        constraint_metadata: dict[str, Any] = {
            "delivered": True,
            "parse_failed": False,
            "commonsense_pass": commonsense_pass,
            "hard_pass": hard_pass,
            "final_pass": final_pass,
        }

        # Store per-check result: True/False/None and failure reason
        for check_name, (result, reason) in commonsense_results.items():
            constraint_metadata[f"commonsense_{check_name}"] = result
            if reason:
                constraint_metadata[f"commonsense_{check_name}_reason"] = reason

        if hard_results is not None:
            for check_name, (result, reason) in hard_results.items():
                constraint_metadata[f"hard_{check_name}"] = result
                if reason:
                    constraint_metadata[f"hard_{check_name}_reason"] = reason

        explanation = (
            ScoreExplanation.ALL_PASS
            if final_pass
            else build_failure_explanation(commonsense_results, hard_results)
        )

        return Score(
            value=CORRECT if final_pass else INCORRECT,
            answer=plan_text,
            explanation=explanation,
            metadata=constraint_metadata,
        )

    return score


def all_pass(results: dict[str, tuple] | None) -> bool:
    """Return True if every check in results passed (True) or was N/A (None).
    Args:
        results: Dict mapping check name to ``(result, reason)`` tuples, or None
            (treated as all-pass).

    Returns:
        True if no check returned False, False otherwise.
    """
    if results is None:
        return True
    return all(result is not False for result, _ in results.values())


def build_failure_explanation(
    commonsense_results: dict[str, tuple],
    hard_results: dict[str, tuple] | None,
) -> str:
    """Build an explanation of which constraints failed.

    Args:
        commonsense_results: Dict mapping commonsense check name to
            ``(result, reason)`` tuples.
        hard_results: Dict mapping hard check name to ``(result, reason)`` tuples,
            or None if hard constraints were not evaluated.

    Returns:
        Single-line string listing every failed check and its reason.
    """
    failures: list[str] = []

    for check_name, (result, reason) in commonsense_results.items():
        if result is False:
            msg = f"[commonsense] {check_name}"
            if reason:
                msg += f": {reason}"
            failures.append(msg)

    if hard_results:
        for check_name, (result, reason) in hard_results.items():
            if result is False:
                msg = f"[hard] {check_name}"
                if reason:
                    msg += f": {reason}"
                failures.append(msg)

    return ScoreExplanation.CONSTRAINT_FAIL + " — " + "; ".join(failures)
