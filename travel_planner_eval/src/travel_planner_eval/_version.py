"""Hardcoded version metadata for the TravelPlanner evaluation.

This replaces inspect_evals.metadata.load_eval_metadata for standalone use.
Version is sourced from eval.yaml: version: "1-A"
"""

from typing import Any


class _TaskVersion:
    """Minimal replica of inspect_evals.metadata.TaskVersion."""

    full_version = "1-A"
    comparability_version = 1
    interface_version = "A"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "full_task_version": self.full_version,
            "task_interface_version": self.interface_version,
            "task_comparability_version": self.comparability_version,
        }


EVAL_VERSION = _TaskVersion()
