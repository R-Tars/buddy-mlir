from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import CandidateConfig, MeasurementContract, sha256_json

CONFIRMATION_SCHEMA_VERSION = 1


class ConfirmationError(ValueError):
    """Raised when a matched A/B confirmation request is malformed."""


@dataclass(frozen=True)
class ConfirmationPolicy:
    minimum_relative_improvement: float = 0.01
    maximum_cv: float = 0.015
    metric: str = "tokens_per_second_per_user"
    metric_direction: str = "maximize"
    warmup: int = 5
    iterations: int = 100
    repetitions: int = 3

    def __post_init__(self) -> None:
        if self.minimum_relative_improvement < 0.0:
            raise ConfirmationError("minimum_relative_improvement must be non-negative")
        if not 0.0 <= self.maximum_cv <= 1.0:
            raise ConfirmationError("maximum_cv must be in [0, 1]")
        if not self.metric:
            raise ConfirmationError("confirmation metric must be non-empty")
        if self.metric_direction not in {"maximize", "minimize"}:
            raise ConfirmationError("metric_direction must be 'maximize' or 'minimize'")
        expected = MeasurementContract.final_confirmation()
        if (self.warmup, self.iterations, self.repetitions) != (
            expected.warmup,
            expected.iterations,
            expected.repetitions,
        ):
            raise ConfirmationError(
                "final confirmation is fixed to 5 warmup, 100 iterations, "
                "and 3 repetitions"
            )

    @property
    def measurement_contract(self) -> MeasurementContract:
        return MeasurementContract.final_confirmation()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "metric_direction": self.metric_direction,
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "maximum_cv": self.maximum_cv,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "repetitions": self.repetitions,
        }


@dataclass(frozen=True)
class ConfirmationArm:
    label: str
    candidate: CandidateConfig
    reports: tuple[Mapping[str, Any], ...]
    correctness_passed: bool
    quality_passed: bool

    def __post_init__(self) -> None:
        if self.label not in {"incumbent", "challenger"}:
            raise ConfirmationError(
                "confirmation arm label must be incumbent or challenger"
            )
        if not self.reports:
            raise ConfirmationError("confirmation arm must contain reports")


def confirm_matched_ab(
    *,
    incumbent: ConfirmationArm,
    challenger: ConfirmationArm,
    policy: ConfirmationPolicy | None = None,
) -> dict[str, Any]:
    """Apply the strict long-run promotion gate to a matched A/B pair.

    Invalid measurements are represented in the returned report instead of
    raising, so callers can always persist a classified failure artifact.
    """

    active_policy = policy or ConfirmationPolicy()
    errors: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        errors,
        name="confirmation.arm_labels",
        passed=(incumbent.label == "incumbent" and challenger.label == "challenger"),
        message="matched A/B requires incumbent and challenger arms",
    )

    incumbent_identity = _frozen_candidate_identity(incumbent.candidate)
    challenger_identity = _frozen_candidate_identity(challenger.candidate)
    contracts_match = incumbent_identity == challenger_identity
    _check(
        checks,
        errors,
        name="confirmation.frozen_candidate_contracts_match",
        passed=contracts_match,
        message=(
            "incumbent and challenger differ outside tunable_state"
            if not contracts_match
            else None
        ),
    )

    required_contract = active_policy.measurement_contract.to_dict()
    for arm in (incumbent, challenger):
        observed_contract = arm.candidate.measurement_contract.to_dict()
        contract_matches = observed_contract == required_contract
        _check(
            checks,
            errors,
            name=f"confirmation.{arm.label}.measurement_contract",
            passed=contract_matches,
            message=(
                "candidate does not use the fixed final confirmation contract"
                if not contract_matches
                else None
            ),
            expected=required_contract,
            observed=observed_contract,
        )

    incumbent_summary = _summarize_arm(incumbent, active_policy)
    challenger_summary = _summarize_arm(challenger, active_policy)
    for summary in (incumbent_summary, challenger_summary):
        label = summary["label"]
        _check(
            checks,
            errors,
            name=f"confirmation.{label}.report_count",
            passed=summary["report_count_valid"],
            message=(
                f"{label} must contain exactly {active_policy.repetitions} reports"
                if not summary["report_count_valid"]
                else None
            ),
            expected=active_policy.repetitions,
            observed=summary["report_count"],
        )
        _check(
            checks,
            errors,
            name=f"confirmation.{label}.reports_valid",
            passed=summary["reports_valid"],
            message=(
                f"{label} has failed or contract-incompatible reports"
                if not summary["reports_valid"]
                else None
            ),
        )
        _check(
            checks,
            errors,
            name=f"confirmation.{label}.cv",
            passed=summary["cv_passed"],
            message=(
                f"{label} coefficient of variation exceeds the limit"
                if not summary["cv_passed"]
                else None
            ),
            limit=active_policy.maximum_cv,
            observed=summary["cv"],
        )
        _check(
            checks,
            errors,
            name=f"confirmation.{label}.correctness",
            passed=bool(summary["correctness_passed"]),
            message=(
                f"{label} correctness gate failed"
                if not summary["correctness_passed"]
                else None
            ),
        )
        _check(
            checks,
            errors,
            name=f"confirmation.{label}.quality",
            passed=bool(summary["quality_passed"]),
            message=(
                f"{label} quality gate failed"
                if not summary["quality_passed"]
                else None
            ),
        )

    relative_improvement = _relative_improvement(
        incumbent_summary.get("median"),
        challenger_summary.get("median"),
        active_policy.metric_direction,
    )
    measurements_valid = not errors
    improvement_passed = bool(
        measurements_valid
        and relative_improvement is not None
        and relative_improvement >= active_policy.minimum_relative_improvement
    )
    checks.append(
        {
            "name": "confirmation.minimum_relative_improvement",
            "passed": improvement_passed,
            "required": active_policy.minimum_relative_improvement,
            "observed": relative_improvement,
            "promotion_gate": True,
        }
    )

    promoted = bool(measurements_valid and improvement_passed)
    selected = challenger if promoted else incumbent
    return {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "status": "passed" if measurements_valid else "failed",
        "passed": measurements_valid,
        "strategy": "matched_ab_long_confirmation",
        "matched_ab": True,
        "policy": active_policy.to_dict(),
        "measurement_contract": required_contract,
        "frozen_identity_sha256": (
            sha256_json(incumbent_identity) if contracts_match else None
        ),
        "arms": {
            "incumbent": incumbent_summary,
            "challenger": challenger_summary,
        },
        "relative_improvement": relative_improvement,
        "promotion": {
            "promoted": promoted,
            "decision": ("promote_challenger" if promoted else "retain_incumbent"),
            "reason": _promotion_reason(
                measurements_valid=measurements_valid,
                improvement_passed=improvement_passed,
            ),
            "selected_candidate_fingerprint": sha256_json(
                selected.candidate.identity_payload()
            ),
            "selected_arm": selected.label,
        },
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "errors": errors,
    }


def write_confirmation_report(
    out: str | Path,
    report: Mapping[str, Any],
) -> Path:
    from .search import atomic_write_json

    destination = Path(out)
    atomic_write_json(destination, dict(report))
    return destination


def _summarize_arm(
    arm: ConfirmationArm,
    policy: ConfirmationPolicy,
) -> dict[str, Any]:
    report_summaries = [
        _validate_report(report, arm.label, index, policy)
        for index, report in enumerate(arm.reports)
    ]
    values = [
        float(summary["metric_value"])
        for summary in report_summaries
        if summary["valid"] and summary["metric_value"] is not None
    ]
    report_count_valid = len(arm.reports) == policy.repetitions
    reports_valid = bool(
        report_count_valid
        and len(values) == policy.repetitions
        and all(summary["valid"] for summary in report_summaries)
    )
    median = statistics.median(values) if values else None
    mean = statistics.fmean(values) if values else None
    standard_deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    cv = standard_deviation / abs(mean) if mean is not None and mean != 0.0 else None
    cv_passed = bool(reports_valid and cv is not None and cv <= policy.maximum_cv)
    return {
        "label": arm.label,
        "candidate_fingerprint": sha256_json(arm.candidate.identity_payload()),
        "tunable_state_sha256": sha256_json(arm.candidate.tunable_state),
        "report_count": len(arm.reports),
        "report_count_valid": report_count_valid,
        "reports_valid": reports_valid,
        "metric": policy.metric,
        "metric_direction": policy.metric_direction,
        "values": values,
        "median": median,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "cv": cv,
        "maximum_cv": policy.maximum_cv,
        "cv_passed": cv_passed,
        "correctness_passed": bool(arm.correctness_passed),
        "quality_passed": bool(arm.quality_passed),
        "reports": report_summaries,
    }


def _validate_report(
    report: Mapping[str, Any],
    arm: str,
    repetition: int,
    policy: ConfirmationPolicy,
) -> dict[str, Any]:
    errors: list[str] = []
    passed = bool(report.get("passed", _status_passed(report.get("status"))))
    if not passed:
        errors.append("report did not pass")

    observed_warmup = _integer_or_none(report.get("warmup"))
    observed_iterations = _integer_or_none(report.get("iterations"))
    if observed_warmup != policy.warmup:
        errors.append(f"warmup must be {policy.warmup}; observed {observed_warmup}")
    if observed_iterations != policy.iterations:
        errors.append(
            "iterations must be " f"{policy.iterations}; observed {observed_iterations}"
        )

    expected_execution = {
        "execution_mode": "trace",
        "runtime_input_mode": "persistent",
        "after_prefill": True,
    }
    for key, expected in expected_execution.items():
        if key in report and report.get(key) != expected:
            errors.append(f"{key} must be {expected!r}; observed {report.get(key)!r}")

    reported_arm = report.get("confirmation_arm")
    if reported_arm is not None and reported_arm != arm:
        errors.append(f"confirmation_arm must be {arm!r}; observed {reported_arm!r}")
    reported_repetition = report.get("confirmation_repetition")
    if reported_repetition is not None and int(reported_repetition) != repetition:
        errors.append(
            "confirmation_repetition must be "
            f"{repetition}; observed {reported_repetition}"
        )

    metric_value = _metric_value(report, policy.metric)
    if metric_value is None or not math.isfinite(metric_value) or metric_value <= 0.0:
        errors.append(f"metric {policy.metric!r} must be finite and positive")
        metric_value = None
    return {
        "arm": arm,
        "repetition": repetition,
        "valid": not errors,
        "passed": passed,
        "metric_value": metric_value,
        "warmup": observed_warmup,
        "iterations": observed_iterations,
        "report_sha256": sha256_json(copy.deepcopy(dict(report))),
        "errors": errors,
    }


def _metric_value(report: Mapping[str, Any], metric: str) -> float | None:
    candidates: Sequence[Any] = (
        report.get(metric),
        (
            (report.get("throughput_summary") or {}).get(metric)
            if isinstance(report.get("throughput_summary"), Mapping)
            else None
        ),
        report.get("metric_value") if report.get("metric") in {None, metric} else None,
        (
            (report.get("objective") or {}).get("value")
            if isinstance(report.get("objective"), Mapping)
            and (report.get("objective") or {}).get("name") == metric
            else None
        ),
    )
    for value in candidates:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _frozen_candidate_identity(candidate: CandidateConfig) -> dict[str, Any]:
    identity = candidate.identity_payload()
    identity.pop("tunable_state", None)
    return identity


def _relative_improvement(
    incumbent: Any,
    challenger: Any,
    direction: str,
) -> float | None:
    try:
        incumbent_value = float(incumbent)
        challenger_value = float(challenger)
    except (TypeError, ValueError):
        return None
    if incumbent_value <= 0.0 or challenger_value <= 0.0:
        return None
    if direction == "maximize":
        return (challenger_value - incumbent_value) / incumbent_value
    return (incumbent_value - challenger_value) / incumbent_value


def _check(
    checks: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    message: str | None,
    **details: Any,
) -> None:
    check = {"name": name, "passed": bool(passed), **details}
    checks.append(check)
    if not passed:
        errors.append(
            {
                "check": name,
                "message": message or "confirmation check failed",
                "details": copy.deepcopy(details),
            }
        )


def _promotion_reason(
    *,
    measurements_valid: bool,
    improvement_passed: bool,
) -> str:
    if not measurements_valid:
        return "matched A/B confirmation failed a contract, stability, or quality gate"
    if not improvement_passed:
        return "challenger improvement is below the promotion threshold"
    return "challenger passes matched A/B stability, quality, and improvement gates"


def _status_passed(status: Any) -> bool:
    return str(status).lower() in {
        "passed",
        "profiled",
        "completed",
        "success",
    }


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
