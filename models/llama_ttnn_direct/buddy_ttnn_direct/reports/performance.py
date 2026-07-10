from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OFFICIAL_PERFORMANCE_PARITY_METRIC = "tokens_per_second_per_user"


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


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _numbers_equal(lhs: Any, rhs: Any) -> bool:
    try:
        return abs(float(lhs) - float(rhs)) <= 1.0e-9
    except (TypeError, ValueError):
        return False


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
