from __future__ import annotations

from typing import Any

from ..codegen.config_diff import PARITY_SECTIONS
from .schema import (
    field_keys as _field_keys,
    int_equal as _int_equal,
    non_empty_string as _non_empty_string,
)


def config_gap_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    status = summary.get("status")
    if status not in {"match", "diff_found"}:
        return False
    if not _int_equal(summary.get("section_count"), len(PARITY_SECTIONS)):
        return False
    counts = summary.get("issue_counts_by_section")
    if not isinstance(counts, dict):
        return False
    if set(counts) != set(PARITY_SECTIONS):
        return False
    try:
        issue_count = int(summary.get("issue_count"))
        section_counts = {
            section: int(counts[section])
            for section in PARITY_SECTIONS
        }
    except (TypeError, ValueError):
        return False
    if issue_count < 0 or any(count < 0 for count in section_counts.values()):
        return False
    if sum(section_counts.values()) != issue_count:
        return False
    sections_with_issues = summary.get("sections_with_issues")
    if not isinstance(sections_with_issues, list):
        return False
    if set(sections_with_issues) - set(PARITY_SECTIONS):
        return False
    expected_sections = [
        section
        for section in PARITY_SECTIONS
        if section_counts[section] > 0
    ]
    if sections_with_issues != expected_sections:
        return False
    top_issue_paths = summary.get("top_issue_paths")
    if not isinstance(top_issue_paths, list):
        return False
    if issue_count == 0:
        return (
            status == "match"
            and sections_with_issues == []
            and top_issue_paths == []
        )
    return status == "diff_found" and bool(top_issue_paths) and all(
        config_gap_issue_complete(issue) for issue in top_issue_paths
    )


def official_required_field_coverage_complete(coverage: Any) -> bool:
    if not isinstance(coverage, dict):
        return False
    try:
        required = int(coverage.get("required_field_count"))
        present = int(coverage.get("present_required_count"))
        missing = int(coverage.get("missing_required_count"))
    except (TypeError, ValueError):
        return False
    missing_paths = coverage.get("missing_required_paths")
    missing_sections = coverage.get("sections_missing_required_fields")
    return (
        coverage.get("status") == "complete"
        and required > 0
        and present == required
        and missing == 0
        and missing_paths == []
        and missing_sections == []
    )


def official_required_field_coverage_observed(
    coverage: Any,
) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "status": coverage.get("status"),
        "required_field_count": coverage.get("required_field_count"),
        "present_required_count": coverage.get("present_required_count"),
        "missing_required_count": coverage.get("missing_required_count"),
        "missing_required_paths": coverage.get("missing_required_paths", []),
        "sections_missing_required_fields": coverage.get(
            "sections_missing_required_fields",
            [],
        ),
    }


def config_gap_issue_complete(issue: Any) -> bool:
    if not isinstance(issue, dict):
        return False
    return (
        issue.get("kind") in {"missing", "mismatch", "extra"}
        and issue.get("section") in PARITY_SECTIONS
        and _non_empty_string(issue.get("path"))
    )


def config_gap_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    counts = summary.get("issue_counts_by_section")
    top_issue_paths = summary.get("top_issue_paths")
    return {
        "status": summary.get("status"),
        "issue_count": summary.get("issue_count"),
        "section_count": summary.get("section_count"),
        "sections_with_issues": summary.get("sections_with_issues"),
        "issue_count_sections": _field_keys(counts),
        "top_issue_count": len(top_issue_paths)
        if isinstance(top_issue_paths, list)
        else None,
    }
