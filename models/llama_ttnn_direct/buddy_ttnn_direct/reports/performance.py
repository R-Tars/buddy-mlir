from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .schema import (
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    number_at_least as _number_at_least,
    numbers_equal as _numbers_equal,
    positive_number as _positive_number,
)


OFFICIAL_PERFORMANCE_PARITY_METRIC = "tokens_per_second_per_user"
PROFILE_GENERATE_MILESTONE_IDS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")


def default_performance_baselines_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "performance_baselines.json"
    )


def load_performance_baselines(
    path: str | Path | None = None,
) -> dict[str, Any]:
    baseline_path = (
        Path(path)
        if path is not None
        else default_performance_baselines_path()
    )
    payload = json.loads(baseline_path.read_text())
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"performance baseline file has no entries: {baseline_path}"
        )
    ids: set[str] = set()
    for entry in entries:
        if not performance_baseline_entry_complete(entry):
            raise ValueError(
                f"invalid performance baseline entry in {baseline_path}: "
                f"{entry!r}"
            )
        entry_id = str(entry["id"])
        if entry_id in ids:
            raise ValueError(
                f"duplicate performance baseline id in {baseline_path}: "
                f"{entry_id}"
            )
        ids.add(entry_id)
    payload["path"] = str(baseline_path)
    return payload


def resolve_performance_baseline(
    baseline_reference: str,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_performance_baselines(path)
    for entry in payload["entries"]:
        if entry["id"] == baseline_reference:
            resolved = dict(entry)
            resolved["baseline_file"] = payload["path"]
            resolved["metric"] = payload.get(
                "metric",
                "decode_tokens_per_second_per_user",
            )
            return resolved
    raise ValueError(
        f"unknown performance baseline reference {baseline_reference!r}; "
        f"available references: "
        f"{', '.join(str(entry['id']) for entry in payload['entries'])}"
    )


def performance_baseline_entry_complete(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    required_strings = (
        "id",
        "role",
        "implementation",
        "frontend",
        "model",
        "source",
    )
    return (
        all(_non_empty_string(entry.get(key)) for key in required_strings)
        and _positive_number(entry.get("batch_size"))
        and _positive_number(
            entry.get("decode_tokens_per_second_per_user")
        )
        and _positive_number(entry.get("aggregate_tokens_per_second"))
    )


def official_performance_baseline_entry_complete(entry: Any) -> bool:
    return (
        performance_baseline_entry_complete(entry)
        and entry.get("role") == "official_8b_target"
        and entry.get("model") == "Llama 3.1 8B"
        and _numbers_equal(entry.get("batch_size"), 32)
    )


def performance_baseline_entry_summary(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    fields = (
        "id",
        "role",
        "implementation",
        "frontend",
        "model",
        "batch_size",
        "decode_tokens_per_second_per_user",
        "aggregate_tokens_per_second",
        "source",
        "baseline_file",
        "metric",
    )
    return {field: entry.get(field) for field in fields if field in entry}


def throughput_baseline_summary(
    report: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    throughput = profile.get("throughput_summary") or {}
    observed = throughput.get("tokens_per_second_per_user")
    baseline = report.get("baseline_tokens_per_second_per_user")
    min_ratio = report.get("min_baseline_ratio")
    ratio = None
    if _positive_number(observed) and _positive_number(baseline):
        ratio = float(observed) / float(baseline)
    summary = {
        "metric": OFFICIAL_PERFORMANCE_PARITY_METRIC,
        "observed": observed,
        "baseline": baseline,
        "baseline_reference": report.get("baseline_reference"),
        "baseline_reference_entry": performance_baseline_entry_summary(
            report.get("baseline_reference_entry")
        ),
        "ratio": ratio,
        "min_ratio": min_ratio,
    }
    if min_ratio is not None:
        summary["passed"] = _number_at_least(ratio, min_ratio)
    return summary


def performance_gap_summary(
    report: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = throughput_baseline_summary(report, profile)
    observed = baseline_summary.get("observed")
    baseline = baseline_summary.get("baseline")
    min_ratio = baseline_summary.get("min_ratio")
    bottleneck = profile.get("bottleneck_summary") or {}
    sections = bottleneck.get("sections_ms")
    if not isinstance(sections, dict):
        sections = {}
    section_total = sum(
        float(value)
        for value in sections.values()
        if isinstance(value, (int, float))
    )
    max_section = bottleneck.get("max_section")
    max_section_ms = bottleneck.get("max_section_ms")
    max_section_share = None
    if _nonnegative_number(max_section_ms) and section_total > 0.0:
        max_section_share = float(max_section_ms) / section_total

    shortfall_to_baseline = None
    speedup_to_baseline = None
    if _positive_number(observed) and _positive_number(baseline):
        shortfall_to_baseline = max(0.0, float(baseline) - float(observed))
        speedup_to_baseline = float(baseline) / float(observed)

    required_for_min_ratio = None
    shortfall_to_min_ratio = None
    speedup_to_min_ratio = None
    if _positive_number(baseline) and min_ratio is not None:
        required_for_min_ratio = float(baseline) * float(min_ratio)
        if _positive_number(observed):
            shortfall_to_min_ratio = max(
                0.0,
                required_for_min_ratio - float(observed),
            )
            speedup_to_min_ratio = (
                required_for_min_ratio / float(observed)
                if required_for_min_ratio > 0.0
                else 0.0
            )

    return {
        "metric": OFFICIAL_PERFORMANCE_PARITY_METRIC,
        "status": (profile.get("throughput_summary") or {}).get("status"),
        "observed": observed,
        "baseline": baseline,
        "baseline_reference": baseline_summary.get("baseline_reference"),
        "ratio": baseline_summary.get("ratio"),
        "min_ratio": min_ratio,
        "passed_min_ratio": baseline_summary.get("passed"),
        "shortfall_to_baseline": shortfall_to_baseline,
        "required_speedup_to_baseline": speedup_to_baseline,
        "required_tokens_per_second_per_user_for_min_ratio": (
            required_for_min_ratio
        ),
        "shortfall_to_min_ratio": shortfall_to_min_ratio,
        "required_speedup_to_min_ratio": speedup_to_min_ratio,
        "bottleneck": {
            "max_section": max_section,
            "max_section_ms": max_section_ms,
            "max_section_share": max_section_share,
            "sections_ms": sections,
        },
    }


def validate_real_generate_milestones(
    profile_generate: dict[str, Any],
    generate_depth_sweep: dict[str, Any],
) -> dict[str, Any] | None:
    profile_milestones = profile_generate.get("performance_milestones")
    if not profile_generate_milestones_complete(profile_milestones):
        return None
    summary = copy.deepcopy(profile_milestones)
    summary["basis"] = (
        "validate-real-decode PR-7 generate performance milestone evidence"
    )
    summary["sources"] = {
        "profile_generate_report": profile_generate.get(
            "profile_generate_report"
        ),
        "generate_report": profile_generate.get("generate_report"),
        "generate_depth_sweep_report": generate_depth_sweep.get(
            "generate_depth_sweep_report"
        ),
    }
    observed = summary.setdefault("observed", {})
    if isinstance(observed, dict):
        observed["generate_depth_sweep"] = {
            "status": generate_depth_sweep.get("status"),
            "covered_full_depth": generate_depth_sweep.get(
                "covered_full_depth"
            ),
            "max_depth": generate_depth_sweep.get("max_depth"),
            "passed_depth_count": generate_depth_sweep.get(
                "passed_depth_count"
            ),
            "failed_depths": generate_depth_sweep.get("failed_depths", []),
        }

    sweep_acceptance = generate_depth_sweep.get("acceptance")
    full_depth_from_sweep = (
        generate_depth_sweep.get("status") == "pass"
        and generate_depth_sweep.get("covered_full_depth") is True
        and isinstance(sweep_acceptance, dict)
        and sweep_acceptance.get("passed") is True
    )
    milestones = summary.get("milestones")
    if isinstance(milestones, list):
        for milestone in milestones:
            if not isinstance(milestone, dict):
                continue
            if milestone.get("id") != "M1":
                continue
            milestone.setdefault("observed", {})
            if isinstance(milestone["observed"], dict):
                milestone["observed"]["generate_depth_sweep"] = {
                    "status": generate_depth_sweep.get("status"),
                    "covered_full_depth": generate_depth_sweep.get(
                        "covered_full_depth"
                    ),
                    "acceptance_passed": (
                        sweep_acceptance.get("passed")
                        if isinstance(sweep_acceptance, dict)
                        else None
                    ),
                }
            if not milestone.get("passed") and full_depth_from_sweep:
                milestone["passed"] = True
                milestone["status"] = "passed"
                milestone["evidence_source"] = "generate_depth_sweep"
                milestone.pop("reason", None)
        _refresh_generate_milestone_summary(summary)
    return summary


def _refresh_generate_milestone_summary(summary: dict[str, Any]) -> None:
    milestones = summary.get("milestones")
    if not isinstance(milestones, list):
        return
    highest_passed = None
    next_milestone = None
    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue
        if milestone.get("passed") is True:
            highest_passed = milestone.get("id")
            continue
        if next_milestone is None:
            next_milestone = {
                "id": milestone.get("id"),
                "name": milestone.get("name"),
                "reason": milestone.get("reason"),
            }
    summary["highest_passed"] = highest_passed
    summary["next_milestone"] = next_milestone


def profile_generate_milestones_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    milestones = value.get("milestones")
    if not isinstance(milestones, list):
        return False
    ids = [
        milestone.get("id")
        for milestone in milestones
        if isinstance(milestone, dict)
    ]
    return (
        ids == list(PROFILE_GENERATE_MILESTONE_IDS)
        and isinstance(value.get("official_reference"), dict)
        and isinstance(value.get("observed"), dict)
    )
