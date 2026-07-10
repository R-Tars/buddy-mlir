from __future__ import annotations

from pathlib import Path
from typing import Any


def artifact_index(paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def artifact_evidence(name: str, path: Path) -> dict[str, Any]:
    if path.is_file():
        kind = "file"
    elif path.is_dir():
        kind = "directory"
    else:
        kind = "missing"
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "kind": kind,
    }


def step_names_with_status(
    results: Any,
    *,
    status: str | None = None,
    failing: bool = False,
) -> list[str]:
    if not isinstance(results, dict):
        return []
    names = []
    passing_statuses = {"pass", "dry_run", "skipped", "pending"}
    for name, value in results.items():
        value = str(value)
        if status is not None and value == status:
            names.append(str(name))
        elif failing and value not in passing_statuses:
            names.append(str(name))
    return names
