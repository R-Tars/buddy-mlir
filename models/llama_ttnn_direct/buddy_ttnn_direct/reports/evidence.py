from __future__ import annotations

from pathlib import Path
from typing import Any

from ..codegen.artifacts import write_json


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


def tensorization_evidence(step: dict[str, Any]) -> dict[str, Any]:
    from .. import validation

    tensorization = validation._step_tensorization_summary(step)
    return {
        "status": tensorization.get("status"),
        "roles": tensorization.get("roles"),
        "tensor_count": tensorization.get("tensor_count"),
        "target_dtype_counts": tensorization.get("target_dtype_counts", {}),
        "layout_counts": tensorization.get("layout_counts", {}),
        "memory_config_counts": tensorization.get("memory_config_counts", {}),
        "transform_counts": tensorization.get("transform_counts", {}),
        "transform_paths_by_kind": tensorization.get(
            "transform_paths_by_kind",
            {},
        ),
        "ttnn_dtype_counts": tensorization.get("ttnn_dtype_counts", {}),
        "ttnn_layout_counts": tensorization.get("ttnn_layout_counts", {}),
        "ttnn_memory_config_counts": tensorization.get(
            "ttnn_memory_config_counts",
            {},
        ),
        "tensor_paths": tensorization.get("tensor_paths", []),
        "required_tensorized_tensor_paths": step.get(
            "required_tensorized_tensor_paths",
            [],
        ),
        "missing_required_tensorized_tensor_paths": step.get(
            "missing_required_tensorized_tensor_paths",
            [],
        ),
        "physical_shape_mismatches": (
            validation._tensorized_physical_shape_mismatches(tensorization)
        ),
        "key_paths": tensorization.get("key_paths", []),
        "key_tensors": tensorization.get("key_tensors", {}),
    }


def reference_summary(runtime_report: dict[str, Any]) -> dict[str, Any]:
    reference = runtime_report.get("reference") or {}
    checks = reference.get("checks") or []
    return {
        "reference_status": reference.get("status"),
        "reference_kind": reference.get("kind"),
        "reference_planned_ops": reference.get("planned_ops"),
        "reference_planned_observed_ops": reference.get(
            "planned_observed_ops"
        ),
        "reference_observed_ops": reference.get("observed_ops"),
        "reference_failed_checks": [
            check.get("name")
            for check in checks
            if isinstance(check, dict) and not check.get("passed")
        ],
    }


def candidate_reference_status_counts(
    report: dict[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in report.get("candidates", []):
        status = candidate.get("reference_status")
        if status is None:
            continue
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def dump_validation_report(report: dict[str, Any], out: str | Path) -> None:
    write_json_report(Path(out), report)


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
