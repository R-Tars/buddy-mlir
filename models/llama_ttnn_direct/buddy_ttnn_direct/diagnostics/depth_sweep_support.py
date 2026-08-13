from __future__ import annotations

import json
import subprocess
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


BLOCKED_DEPTH_REASON = "blocked by an earlier depth failure"
DepthValues = str | list[int | str] | tuple[int | str, ...] | None


def load_program_num_layers(program_root: Path) -> int:
    config = json.loads((program_root / "config.json").read_text())
    return int(config["num_layers"])


def resolve_depths(
    depths: DepthValues,
    *,
    program_num_layers: int,
    defaults: Sequence[int | str],
) -> list[int]:
    if program_num_layers <= 0:
        raise ValueError("program_num_layers must be positive")
    use_defaults = depths is None
    if use_defaults:
        tokens = list(defaults)
    elif isinstance(depths, str):
        tokens = [token.strip() for token in depths.split(",") if token.strip()]
    else:
        tokens = list(depths)
    if not tokens:
        raise ValueError("at least one depth is required")

    resolved: list[int] = []
    for token in tokens:
        if isinstance(token, str) and token.lower() in {"full", "max", "all"}:
            depth = program_num_layers
        else:
            try:
                depth = int(token)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid depth {token!r}") from exc
        if depth <= 0:
            raise ValueError("depths must be positive")
        if depth > program_num_layers:
            if use_defaults:
                continue
            raise ValueError(
                "depths must be <= generated config num_layers "
                f"({program_num_layers})"
            )
        if depth not in resolved:
            resolved.append(depth)
    if not resolved:
        raise ValueError("at least one depth is required")
    return resolved


def resolve_output_root(
    out_path: Path,
    requested: str | Path | None,
    *,
    suffix: str,
) -> Path:
    root = Path(requested) if requested is not None else out_path.parent / f"{out_path.stem}_{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def should_isolate_depth_steps(
    requested: bool,
    *,
    dry_run: bool,
    injected_modules: Sequence[Any | None],
) -> bool:
    return bool(requested) and not dry_run and all(module is None for module in injected_modules)


def run_depth_sequence(
    depths: Sequence[int],
    *,
    report_path_for: Callable[[int], Path],
    execute: Callable[[int, Path], dict[str, Any]],
    make_record: Callable[[int, dict[str, Any], Path], dict[str, Any]],
    make_skipped: Callable[[int, Path, str], dict[str, Any]],
    make_exception: Callable[[int, Path, Exception], dict[str, Any]],
    dry_run: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    blocked = False
    for depth in depths:
        report_path = report_path_for(depth)
        if blocked:
            records.append(make_skipped(depth, report_path, BLOCKED_DEPTH_REASON))
            continue
        try:
            record = make_record(depth, execute(depth, report_path), report_path)
            if record["status"] == "no_device":
                blocked = True
            elif not record["passed"] and not dry_run:
                blocked = True
        except Exception as exc:  # pragma: no cover - defensive CLI path.
            record = make_exception(depth, report_path, exc)
            blocked = True
        records.append(record)
    return records


def run_isolated_command(
    command: Sequence[str],
    *,
    report_path: Path,
    missing_report: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    result = subprocess.run(list(command), capture_output=True, text=True, check=False)
    payload = json.loads(report_path.read_text()) if report_path.is_file() else missing_report()
    if not report_path.is_file():
        write_json(report_path, payload)
    payload["isolated_subprocess"] = {
        "enabled": True,
        "returncode": result.returncode,
        "command": list(command),
        "stdout": diagnostic_excerpt(result.stdout),
        "stderr": diagnostic_excerpt(result.stderr),
    }
    return payload


def project_fields(payload: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: payload.get(field) for field in fields}


def ensure_report_written(report_path: Path, payload: dict[str, Any]) -> None:
    if not report_path.is_file():
        write_json(report_path, payload)


def depth_summary(depths: list[int], program_num_layers: int) -> dict[str, Any]:
    return {
        "program_num_layers": program_num_layers,
        "depths": depths,
        "depth_count": len(depths),
        "max_depth": max(depths) if depths else None,
        "covered_full_depth": program_num_layers in depths,
    }


def result_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passed_depth_count": sum(record.get("passed") is True for record in records),
        "failed_depths": [
            record["depth"]
            for record in records
            if record.get("passed") is False and record.get("status") != "skipped"
        ],
    }


def sweep_status(
    records: list[dict[str, Any]], *, dry_run: bool, accepted: bool
) -> str:
    if dry_run:
        return "dry_run"
    if any(record.get("status") == "no_device" for record in records):
        return "no_device"
    return "pass" if accepted else "fail"


def field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is not None:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def exception_details(exception: Exception) -> dict[str, str]:
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exc(),
    }


def diagnostic_excerpt(value: str | None, *, limit: int = 2000) -> str:
    text = str(value).strip() if value else ""
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
