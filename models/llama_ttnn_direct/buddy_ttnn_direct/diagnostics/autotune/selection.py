from __future__ import annotations

import copy
from typing import Any

from ...compiler.tuning import (
    LM_HEAD_DRAM_CONCAT,
    OFFICIAL_LINEAR_OUTPUTS,
    OFFICIAL_PROGRAM_CONFIG,
    SDPA_GRID_8X4_PROGRAM_CONFIG,
)

AUTOTUNE_LEVELS = (
    ("lm_head_split_count", "lm_head_split_count"),
    ("memory_config_layout", "memory_layout"),
    ("program_config_core_grid", "program_config"),
)
AUTOTUNE_METRIC = "tokens_per_second_per_user"


def level_values(state_key: str, current: Any) -> list[Any]:
    alternatives = {
        "lm_head_split_count": (8, 16),
        "memory_layout": (OFFICIAL_LINEAR_OUTPUTS, LM_HEAD_DRAM_CONCAT),
        "program_config": (
            OFFICIAL_PROGRAM_CONFIG,
            SDPA_GRID_8X4_PROGRAM_CONFIG,
        ),
    }[state_key]
    return [current] + [value for value in alternatives if value != current]


def select_winner(
    candidates: list[dict[str, Any]],
    *,
    dry_run: bool,
    min_relative_improvement: float,
) -> dict[str, Any] | None:
    if dry_run:
        return candidates[0]
    passed = [
        candidate
        for candidate in candidates
        if candidate.get("passed")
        and isinstance(candidate.get("metric_value"), (int, float))
    ]
    if not passed:
        return None
    best = max(passed, key=lambda candidate: float(candidate["metric_value"]))
    incumbent = candidates[0]
    if not (
        incumbent.get("passed")
        and isinstance(incumbent.get("metric_value"), (int, float))
    ):
        return best
    incumbent_metric = float(incumbent["metric_value"])
    required_metric = incumbent_metric * (1.0 + min_relative_improvement)
    if float(best["metric_value"]) < required_metric:
        return incumbent
    return best


def selection_summary(
    candidates: list[dict[str, Any]],
    winner: dict[str, Any] | None,
) -> dict[str, Any]:
    incumbent = candidates[0] if candidates else None
    incumbent_metric = (
        incumbent.get("metric_value") if isinstance(incumbent, dict) else None
    )
    winner_metric = winner.get("metric_value") if winner is not None else None
    relative_improvement = None
    if (
        isinstance(incumbent_metric, (int, float))
        and float(incumbent_metric) > 0.0
        and isinstance(winner_metric, (int, float))
    ):
        relative_improvement = float(winner_metric) / float(incumbent_metric) - 1.0
    return {
        "incumbent_candidate_id": (
            incumbent.get("candidate_id") if isinstance(incumbent, dict) else None
        ),
        "winner_candidate_id": (
            winner.get("candidate_id") if winner is not None else None
        ),
        "winner_relative_improvement_vs_incumbent": relative_improvement,
    }


def winner_summary(
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate["candidate_id"],
        "state": copy.deepcopy(candidate["state"]),
        "metric": AUTOTUNE_METRIC,
        "metric_value": candidate.get("metric_value"),
        "profile_report": candidate.get("profile_report"),
    }


def confirmation_promotion_decision(
    *,
    challenger: dict[str, Any],
    incumbent: dict[str, Any],
    min_relative_improvement: float,
) -> dict[str, Any]:
    challenger_metric = challenger.get("metric_value")
    incumbent_metric = incumbent.get("metric_value")
    challenger_passed = bool(challenger.get("passed"))
    incumbent_passed = bool(incumbent.get("passed"))
    relative_improvement = None
    if (
        challenger_passed
        and incumbent_passed
        and isinstance(challenger_metric, (int, float))
        and isinstance(incumbent_metric, (int, float))
        and float(incumbent_metric) > 0.0
    ):
        relative_improvement = float(challenger_metric) / float(incumbent_metric) - 1.0
        promoted = relative_improvement >= min_relative_improvement
        reason = (
            "confirmed improvement met promotion threshold"
            if promoted
            else "confirmed improvement was below promotion threshold"
        )
    elif challenger_passed and not incumbent_passed:
        promoted = True
        reason = "challenger passed confirmation and incumbent did not"
    else:
        promoted = False
        reason = "challenger did not pass confirmation"
    return {
        "status": "promoted" if promoted else "rejected",
        "promoted": promoted,
        "reason": reason,
        "challenger_metric": challenger_metric,
        "incumbent_metric": incumbent_metric,
        "relative_improvement": relative_improvement,
        "minimum_relative_improvement": min_relative_improvement,
    }
