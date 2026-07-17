from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass
from typing import Any, Mapping

FINAL_CAMPAIGN_SCHEMA_VERSION = 1

PERFORMANCE_MILESTONES = (
    ("M0", 34.0),
    ("M1", 35.0),
    ("M2", 36.0),
    ("M3", 37.34),
    ("M4", 38.0),
    ("M5", 40.0),
)

REQUIRED_PAPER_EVIDENCE = (
    "winner_discovery_path",
    "profiler_attribution",
    "contributions",
    "prefetch_contribution",
    "ablation",
    "failures",
)


class FinalCampaignError(ValueError):
    """Raised when final campaign evidence is malformed."""


@dataclass(frozen=True)
class FinalCampaignPolicy:
    target_tokens_per_second_per_user: float = 37.34
    target_decode_step_ms_p50: float = 26.79
    partial_success_tokens_per_second_per_user: float = 35.0
    minimum_relative_improvement: float = 0.01
    maximum_cv: float = 0.015
    warmup: int = 5
    iterations: int = 100
    repetitions: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.target_tokens_per_second_per_user,
            self.target_decode_step_ms_p50,
            self.partial_success_tokens_per_second_per_user,
        )
        if any(value <= 0.0 for value in positive):
            raise FinalCampaignError("performance thresholds must be positive")
        if self.minimum_relative_improvement < 0.0:
            raise FinalCampaignError(
                "minimum_relative_improvement must be non-negative"
            )
        if not 0.0 <= self.maximum_cv <= 1.0:
            raise FinalCampaignError("maximum_cv must be in [0, 1]")
        if (self.warmup, self.iterations, self.repetitions) != (5, 100, 3):
            raise FinalCampaignError(
                "final campaign confirmation is fixed to 5 warmup, "
                "100 iterations, and 3 repetitions"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens_per_second_per_user": (
                self.target_tokens_per_second_per_user
            ),
            "target_decode_step_ms_p50": self.target_decode_step_ms_p50,
            "partial_success_tokens_per_second_per_user": (
                self.partial_success_tokens_per_second_per_user
            ),
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "maximum_cv": self.maximum_cv,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "repetitions": self.repetitions,
        }


def build_final_campaign_report(
    *,
    confirmation: Mapping[str, Any],
    correctness: Mapping[str, Any],
    generalization: Mapping[str, Any],
    search_evidence: Mapping[str, Any],
    policy: FinalCampaignPolicy | None = None,
) -> dict[str, Any]:
    """Build the final promotion and paper-evidence report.

    Completing Phase 10 and meeting the +10% performance target are separate
    outcomes. This prevents a complete, reproducible partial success from
    being reported as target attainment.
    """

    active_policy = policy or FinalCampaignPolicy()
    contract = confirmation.get("measurement_contract")
    records = confirmation.get("records")
    if not isinstance(contract, Mapping):
        contract = {}
    if not isinstance(records, Mapping):
        records = {}

    incumbent = _summarize_arm(records, "incumbent", active_policy)
    challenger = _summarize_arm(records, "challenger", active_policy)
    relative_improvement = _relative_improvement(
        incumbent.get("median_tokens_per_second_per_user"),
        challenger.get("median_tokens_per_second_per_user"),
    )

    expected_order = _expected_order(active_policy.repetitions)
    observed_order = list(contract.get("order") or [])
    contract_checks = {
        "warmup": contract.get("warmup") == active_policy.warmup,
        "iterations": contract.get("iterations") == active_policy.iterations,
        "repetitions": contract.get("repetitions") == active_policy.repetitions,
        "paired_alternating_order": observed_order == expected_order,
        "execution_mode_trace": contract.get("execution_mode") == "trace",
        "runtime_inputs_persistent": (
            contract.get("runtime_input_mode") == "persistent"
        ),
        "after_prefill": contract.get("after_prefill") is True,
        "isolated_subprocess": contract.get("isolated_subprocess") is True,
    }

    precision = correctness.get("precision_contract")
    if not isinstance(precision, Mapping):
        precision = {}
    precision_unchanged = bool(
        precision.get("mutated") is False
        and precision.get("expected_hash")
        and precision.get("expected_hash") == precision.get("observed_hash")
    )

    evidence_summary = _summarize_search_evidence(search_evidence)
    checks = {
        "measurement_contract": all(contract_checks.values()),
        "incumbent_reports": incumbent["reports_valid"],
        "challenger_reports": challenger["reports_valid"],
        "incumbent_cv": incumbent["cv_passed"],
        "challenger_cv": challenger["cv_passed"],
        "minimum_relative_improvement": bool(
            relative_improvement is not None
            and relative_improvement >= active_policy.minimum_relative_improvement
        ),
        "full_depth_functional": _passed(correctness.get("full_depth_functional")),
        "all_bf16_regression": _passed(correctness.get("all_bf16_regression")),
        "performance_recipe_quality": _passed(
            correctness.get("performance_recipe_quality")
        ),
        "precision_contract_unchanged": precision_unchanged,
        "generalization": _passed(generalization),
        "paper_evidence_complete": evidence_summary["complete"],
    }
    phase_completion_checks = {
        name: passed
        for name, passed in checks.items()
        if name != "minimum_relative_improvement"
    }
    phase_completed = all(phase_completion_checks.values())
    promotion_allowed = bool(phase_completed and checks["minimum_relative_improvement"])

    challenger_throughput = challenger.get("median_tokens_per_second_per_user")
    challenger_latency = challenger.get("median_decode_step_ms_p50")
    throughput_target_met = bool(
        challenger_throughput is not None
        and challenger_throughput >= active_policy.target_tokens_per_second_per_user
    )
    latency_target_met = bool(
        challenger_latency is not None
        and challenger_latency <= active_policy.target_decode_step_ms_p50
    )
    target_met = bool(
        promotion_allowed and throughput_target_met and latency_target_met
    )
    partial_success = bool(
        promotion_allowed
        and not target_met
        and challenger_throughput is not None
        and challenger_throughput
        >= active_policy.partial_success_tokens_per_second_per_user
    )
    if target_met:
        status = "target_met"
    elif partial_success:
        status = "partial_success"
    elif phase_completed:
        status = "completed_no_promotion"
    else:
        status = "failed_acceptance"

    milestones = _performance_milestones(challenger_throughput)
    target_checks = {
        "throughput": throughput_target_met,
        "latency": latency_target_met,
    }
    all_checks = {**checks, **{f"target_{k}": v for k, v in target_checks.items()}}
    return {
        "schema_version": FINAL_CAMPAIGN_SCHEMA_VERSION,
        "stage": "doc7-phase10-full-campaign",
        "status": status,
        "passed": phase_completed,
        "goal_achieved": target_met,
        "policy": active_policy.to_dict(),
        "performance": {
            "incumbent": incumbent,
            "challenger": challenger,
            "relative_improvement": relative_improvement,
            "milestones": milestones,
            "target_checks": target_checks,
        },
        "measurement_contract": {
            "observed": copy.deepcopy(dict(contract)),
            "expected_order": expected_order,
            "checks": contract_checks,
        },
        "correctness": copy.deepcopy(dict(correctness)),
        "generalization": copy.deepcopy(dict(generalization)),
        "paper_evidence": evidence_summary,
        "acceptance": {
            "phase_completed": phase_completed,
            "promotion_allowed": promotion_allowed,
            "partial_success": partial_success,
            "target_met": target_met,
            "checks": all_checks,
            "failed_checks": [
                name for name, passed in all_checks.items() if not passed
            ],
        },
    }


def _summarize_arm(
    records: Mapping[str, Any],
    label: str,
    policy: FinalCampaignPolicy,
) -> dict[str, Any]:
    rows = []
    for repetition in range(policy.repetitions):
        key = f"{label}_{repetition}"
        raw = records.get(key)
        if not isinstance(raw, Mapping):
            rows.append({"key": key, "valid": False, "reason": "missing"})
            continue
        throughput = _positive_float(raw.get("tokens_per_second_per_user"))
        latency = _positive_float(raw.get("decode_step_ms_p50"))
        valid = bool(raw.get("passed") is True and throughput and latency)
        rows.append(
            {
                "key": key,
                "valid": valid,
                "tokens_per_second_per_user": throughput,
                "decode_step_ms_p50": latency,
                "report": raw.get("report"),
                "device_seconds": _nonnegative_float(raw.get("device_seconds")),
            }
        )
    valid_rows = [row for row in rows if row["valid"]]
    throughputs = [row["tokens_per_second_per_user"] for row in valid_rows]
    latencies = [row["decode_step_ms_p50"] for row in valid_rows]
    reports_valid = len(valid_rows) == policy.repetitions
    mean = statistics.fmean(throughputs) if throughputs else None
    standard_deviation = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    cv = standard_deviation / mean if mean else None
    return {
        "report_count": len(valid_rows),
        "reports_valid": reports_valid,
        "values_tokens_per_second_per_user": throughputs,
        "median_tokens_per_second_per_user": (
            statistics.median(throughputs) if throughputs else None
        ),
        "mean_tokens_per_second_per_user": mean,
        "standard_deviation_tokens_per_second_per_user": standard_deviation,
        "cv": cv,
        "cv_passed": bool(reports_valid and cv is not None and cv <= policy.maximum_cv),
        "median_decode_step_ms_p50": (
            statistics.median(latencies) if latencies else None
        ),
        "device_seconds": sum(row.get("device_seconds") or 0.0 for row in valid_rows),
        "records": rows,
    }


def _summarize_search_evidence(
    search_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    raw_counts = search_evidence.get("candidate_counts")
    categories: dict[str, Any] = {}
    complete = isinstance(raw_counts, Mapping) and bool(raw_counts)
    total_enumerated = 0
    total_legal = 0
    total_measured = 0
    if isinstance(raw_counts, Mapping):
        for name, raw in raw_counts.items():
            if not isinstance(raw, Mapping):
                complete = False
                continue
            enumerated = _nonnegative_int(raw.get("enumerated"))
            legal = _nonnegative_int(raw.get("legal"))
            measured = _nonnegative_int(raw.get("measured"))
            valid = bool(
                enumerated is not None
                and legal is not None
                and measured is not None
                and enumerated >= legal >= measured
            )
            complete = complete and valid
            categories[str(name)] = {
                "enumerated": enumerated,
                "legal": legal,
                "measured": measured,
                "valid": valid,
            }
            if valid:
                total_enumerated += enumerated
                total_legal += legal
                total_measured += measured

    missing = [name for name in REQUIRED_PAPER_EVIDENCE if name not in search_evidence]
    complete = bool(complete and not missing and total_enumerated > 0)
    device_seconds = _nonnegative_float(search_evidence.get("device_seconds"))
    if device_seconds is None:
        complete = False
    pruning_ratio = (
        1.0 - (total_measured / total_enumerated) if total_enumerated else None
    )
    return {
        "complete": complete,
        "candidate_counts": categories,
        "totals": {
            "enumerated": total_enumerated,
            "legal": total_legal,
            "measured": total_measured,
            "pruning_ratio": pruning_ratio,
            "device_seconds": device_seconds,
            "device_minutes": (
                device_seconds / 60.0 if device_seconds is not None else None
            ),
        },
        "missing_fields": missing,
        **{
            name: copy.deepcopy(search_evidence.get(name))
            for name in REQUIRED_PAPER_EVIDENCE
        },
        "device_time_basis": copy.deepcopy(search_evidence.get("device_time_basis")),
        "source_artifacts": copy.deepcopy(search_evidence.get("source_artifacts")),
        "ranking_model": copy.deepcopy(search_evidence.get("ranking_model")),
    }


def _performance_milestones(value: float | None) -> dict[str, Any]:
    rows = [
        {
            "id": name,
            "threshold": threshold,
            "passed": bool(value and value >= threshold),
        }
        for name, threshold in PERFORMANCE_MILESTONES
    ]
    passed = [row["id"] for row in rows if row["passed"]]
    return {
        "observed_tokens_per_second_per_user": value,
        "highest_passed": passed[-1] if passed else None,
        "milestones": rows,
    }


def _expected_order(repetitions: int) -> list[str]:
    result = []
    for repetition in range(repetitions):
        pair = [f"incumbent_{repetition}", f"challenger_{repetition}"]
        result.extend(pair if repetition % 2 == 0 else reversed(pair))
    return result


def _relative_improvement(
    incumbent: float | None,
    challenger: float | None,
) -> float | None:
    if incumbent is None or challenger is None or incumbent <= 0.0:
        return None
    return (challenger - incumbent) / incumbent


def _passed(value: Any) -> bool:
    return bool(isinstance(value, Mapping) and value.get("passed") is True)


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def _nonnegative_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0.0 else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None
