"""Tests for scorer helper functions in scorer.py."""

from scorer import ScoreExplanation, _all_pass, _build_failure_explanation

def test_all_pass_all_true():
    results = {"check1": (True, None), "check2": (True, None)}
    assert _all_pass(results) is True


def test_all_pass_with_not_applicable():
    # None means constraint not applicable — still counts as pass
    results = {"check1": (True, None), "optional": (None, None)}
    assert _all_pass(results) is True


def test_all_pass_all_none():
    results = {"c1": (None, None), "c2": (None, None)}
    assert _all_pass(results) is True


def test_all_pass_with_false():
    results = {"check1": (True, None), "check2": (False, "Something failed")}
    assert _all_pass(results) is False


def test_all_pass_all_false():
    results = {"check1": (False, "reason A"), "check2": (False, "reason B")}
    assert _all_pass(results) is False


def test_all_pass_none_input():
    # None as the whole dict is treated as all-pass (hard constraints skipped)
    assert _all_pass(None) is True


def test_all_pass_empty_dict():
    assert _all_pass({}) is True


def test_build_failure_explanation_commonsense_failure_with_reason():
    commonsense = {
        "is_valid_days": (False, "The number of days should be 3."),
        "is_valid_restaurants": (True, None),
    }
    explanation = _build_failure_explanation(commonsense, None)
    assert ScoreExplanation.CONSTRAINT_FAIL in explanation
    assert "is_valid_days" in explanation
    assert "The number of days should be 3." in explanation
    # Passing checks should not appear in the explanation
    assert "is_valid_restaurants" not in explanation


def test_build_failure_explanation_hard_failure():
    commonsense = {"is_valid_days": (True, None)}
    hard = {"valid_cost": (False, None)}
    explanation = _build_failure_explanation(commonsense, hard)
    assert "[hard]" in explanation
    assert "valid_cost" in explanation


def test_build_failure_explanation_failure_without_reason():
    commonsense = {"is_valid_days": (False, None)}
    explanation = _build_failure_explanation(commonsense, None)
    assert "is_valid_days" in explanation
    # Should not crash when reason is None


def test_build_failure_explanation_multiple_failures():
    commonsense = {
        "is_valid_days": (False, "Wrong day count."),
        "is_valid_attractions": (False, "Repeated attraction."),
    }
    hard = {"valid_cost": (False, "Over budget.")}
    explanation = _build_failure_explanation(commonsense, hard)
    assert "is_valid_days" in explanation
    assert "is_valid_attractions" in explanation
    assert "valid_cost" in explanation
    assert "[commonsense]" in explanation
    assert "[hard]" in explanation


def test_build_failure_explanation_no_failures_returns_prefix():
    # All checks pass — no failures listed, just the prefix
    commonsense = {"is_valid_days": (True, None)}
    explanation = _build_failure_explanation(commonsense, {})
    assert ScoreExplanation.CONSTRAINT_FAIL in explanation


def test_build_failure_explanation_hard_none_ignored():
    commonsense = {"is_valid_days": (False, "Bad.")}
    # hard=None means hard constraints were not evaluated — should be ignored
    explanation = _build_failure_explanation(commonsense, None)
    assert "[hard]" not in explanation
