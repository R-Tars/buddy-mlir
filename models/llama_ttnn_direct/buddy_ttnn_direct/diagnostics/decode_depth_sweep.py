from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from ..smoke_single_layer_decode import profile_decode_step


DEFAULT_DECODE_DEPTH_TARGETS = (1, 2, 4, "full")

PROFILE_SECTION_LATENCY_KEYS = (
    "embedding_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
)

PROFILE_LAYER_LATENCY_KEYS = (
    "rms_norm_attn_ms",
    "attention_ms",
    "residual_add_attn_ms",
    "rms_norm_mlp_ms",
    "mlp_ms",
    "residual_add_mlp_ms",
    "total_ms",
)


def run_decode_depth_sweep(
    *,
    program_dir: str | Path,
    out: str | Path,
    depths: str | list[int | str] | tuple[int | str, ...] | None = None,
    model_path: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    batch_size: int | None = None,
    cache_len: int | None = None,
    device: str = "p150a",
    device_id: int = 0,
    dtype_seed: str = "bf16",
    trace: bool = False,
    trace_iterations: int = 1,
    dry_run: bool = False,
    require_full_depth: bool = True,
    isolate_depth_steps: bool = False,
    prompt: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_module: Any | None = None,
    ttnn_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    program_root = Path(program_dir)
    out_path = Path(out)
    config = json.loads((program_root / "config.json").read_text())
    program_num_layers = int(config["num_layers"])
    if trace_iterations <= 0:
        raise ValueError("trace_iterations must be positive")

    resolved_depths = resolve_decode_depths(
        depths,
        program_num_layers=program_num_layers,
    )
    profile_root = (
        Path(profiles_dir)
        if profiles_dir is not None
        else out_path.parent / f"{out_path.stem}_profiles"
    )
    profile_root.mkdir(parents=True, exist_ok=True)
    model_path_for_profile = Path(model_path) if model_path is not None else None
    use_depth_isolation = (
        bool(isolate_depth_steps)
        and not dry_run
        and tokenizer_module is None
        and ttnn_module is None
        and torch_module is None
    )

    records = []
    stop_after_failure = False
    for depth in resolved_depths:
        report_path = profile_root / f"profile_depth_{depth}.json"
        if stop_after_failure:
            records.append(
                {
                    "depth": depth,
                    "status": "skipped",
                    "passed": False,
                    "profile_report": str(report_path),
                    "reason": "blocked by an earlier depth failure",
                }
            )
            continue
        try:
            if use_depth_isolation:
                profile = _run_isolated_profile_depth(
                    report_path=report_path,
                    program_root=program_root,
                    model_path=model_path_for_profile,
                    depth=depth,
                    device=device,
                    device_id=device_id,
                    batch_size=batch_size,
                    cache_len=cache_len,
                    dtype_seed=dtype_seed,
                    trace=trace,
                    trace_iterations=trace_iterations,
                    prompt=prompt,
                    tokenizer_path=tokenizer_path,
                )
            else:
                profile = profile_decode_step(
                    out=report_path,
                    program_dir=program_root,
                    layers=depth,
                    model_path=model_path_for_profile,
                    device=device,
                    device_id=device_id,
                    batch_size=batch_size,
                    cache_len=cache_len,
                    dtype_seed=dtype_seed,
                    trace=trace,
                    trace_iterations=trace_iterations,
                    dry_run=dry_run,
                    prompt=prompt,
                    tokenizer_path=tokenizer_path,
                    tokenizer_module=tokenizer_module,
                    ttnn_module=ttnn_module,
                    torch_module=torch_module,
                )
            record = _profile_record(
                depth=depth,
                profile=profile,
                report_path=report_path,
            )
            if record["status"] == "no_device":
                stop_after_failure = True
            elif not record["passed"] and not dry_run:
                stop_after_failure = True
        except Exception as exc:  # pragma: no cover - defensive CLI path.
            record = {
                "depth": depth,
                "status": "fail",
                "passed": False,
                "profile_report": str(report_path),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            stop_after_failure = True
        records.append(record)

    status_counts = _field_counts(records, "status")
    reference_status_counts = _field_counts(records, "reference_status")
    trace_status_counts = _field_counts(records, "trace_status")
    acceptance = _depth_sweep_acceptance(
        records,
        depths=resolved_depths,
        program_num_layers=program_num_layers,
        dry_run=dry_run,
        require_full_depth=require_full_depth,
    )
    if dry_run:
        status = "dry_run"
    elif any(record.get("status") == "no_device" for record in records):
        status = "no_device"
    elif acceptance["passed"]:
        status = "pass"
    else:
        status = "fail"

    report = {
        "schema_version": 1,
        "command": "decode-depth-sweep",
        "status": status,
        "passed": bool(acceptance["passed"]),
        "dry_run": bool(dry_run),
        "require_full_depth": bool(require_full_depth),
        "isolate_depth_steps": bool(use_depth_isolation),
        "program_dir": str(program_root),
        "model_path": str(model_path_for_profile) if model_path_for_profile else None,
        "profiles_dir": str(profile_root),
        "program_num_layers": program_num_layers,
        "depths": resolved_depths,
        "depth_count": len(resolved_depths),
        "max_depth": max(resolved_depths) if resolved_depths else None,
        "covered_full_depth": program_num_layers in resolved_depths,
        "batch_size": batch_size,
        "cache_len": cache_len,
        "device": device,
        "device_id": device_id,
        "dtype_seed": dtype_seed,
        "trace_enabled": trace,
        "trace_iterations": trace_iterations if trace else 0,
        "prompt_runtime_requested": prompt is not None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "status_counts": status_counts,
        "reference_status_counts": reference_status_counts,
        "trace_status_counts": trace_status_counts,
        "passed_depth_count": sum(
            1 for record in records if record.get("passed") is True
        ),
        "failed_depths": [
            record["depth"]
            for record in records
            if record.get("passed") is False
            and record.get("status") != "skipped"
        ],
        "records": records,
        "acceptance": acceptance,
    }
    _write_json(out_path, report)
    return report


def resolve_decode_depths(
    depths: str | list[int | str] | tuple[int | str, ...] | None,
    *,
    program_num_layers: int,
) -> list[int]:
    if program_num_layers <= 0:
        raise ValueError("program_num_layers must be positive")
    tokens: list[int | str]
    if depths is None:
        tokens = list(DEFAULT_DECODE_DEPTH_TARGETS)
    elif isinstance(depths, str):
        tokens = [
            token.strip()
            for token in depths.split(",")
            if token.strip()
        ]
    else:
        tokens = list(depths)
    if not tokens:
        raise ValueError("at least one depth is required")

    resolved = []
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
            if depths is None:
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


def _profile_record(
    *,
    depth: int,
    profile: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    throughput = profile.get("throughput_summary") or {}
    trace = profile.get("trace") or {}
    reference = profile.get("reference") or {}
    layer_profiles = profile.get("layer_profiles") or []
    bottleneck = profile.get("bottleneck_summary") or {}
    return {
        "depth": depth,
        "status": profile.get("status"),
        "passed": bool(profile.get("passed")),
        "profile_report": str(report_path),
        "layers": profile.get("layers"),
        "batch_size": profile.get("batch_size"),
        "cache_len": profile.get("cache_len"),
        "parameter_source": profile.get("parameter_source"),
        "input_source": profile.get("input_source"),
        "synthetic_runtime_input_tensor_count": (
            (profile.get("parameter_setup") or {}).get(
                "synthetic_runtime_input_tensor_count"
            )
        ),
        "prompt_runtime_input_tensor_count": (
            (profile.get("parameter_setup") or {}).get(
                "prompt_runtime_input_tensor_count"
            )
        ),
        "decode_runtime_state_input_tensor_count": (
            (profile.get("parameter_setup") or {}).get(
                "decode_runtime_state_input_tensor_count"
            )
        ),
        "rotary_runtime_input_tensor_count": (
            (profile.get("parameter_setup") or {}).get(
                "rotary_runtime_input_tensor_count"
            )
        ),
        "kv_cache_runtime_input_tensor_count": (
            (profile.get("parameter_setup") or {}).get(
                "kv_cache_runtime_input_tensor_count"
            )
        ),
        "prompt_tokenization": profile.get("prompt_tokenization"),
        "decode_runtime_state": profile.get("decode_runtime_state"),
        "rotary_runtime_state": profile.get("rotary_runtime_state"),
        "kv_cache_runtime_state": profile.get("kv_cache_runtime_state"),
        "synthetic_rotary_tensor_count": (
            (profile.get("parameter_setup") or {}).get(
                "synthetic_rotary_tensor_count"
            )
        ),
        "input_shapes": profile.get("input_shapes"),
        "kv_cache": profile.get("kv_cache"),
        "latency_ms": profile.get("latency_ms"),
        "tensor_conversion_count": profile.get("tensor_conversion_count"),
        "tensor_conversion_ms": profile.get("tensor_conversion_ms"),
        "section_latency_ms": profile.get("section_latency_ms"),
        "layer_profile_count": len(layer_profiles),
        "layer_profile_ids": [
            layer.get("layer_id")
            for layer in layer_profiles
            if isinstance(layer, dict)
        ],
        "layer_profiles": layer_profiles,
        "lm_head_profile": profile.get("lm_head_profile"),
        "output_shapes": profile.get("output_shapes"),
        "throughput_summary": throughput,
        "tokens_per_second_per_user": throughput.get(
            "tokens_per_second_per_user"
        ),
        "aggregate_tokens_per_second": throughput.get(
            "aggregate_tokens_per_second"
        ),
        "bottleneck_summary": bottleneck,
        "max_section": bottleneck.get("max_section"),
        "trace_status": trace.get("status"),
        "trace_iterations": trace.get("iterations"),
        "reference_status": reference.get("status"),
        "reference_kind": reference.get("kind"),
        "reference_failed_checks": [
            check.get("name")
            for check in reference.get("checks", [])
            if isinstance(check, dict) and not check.get("passed")
        ],
        "error": profile.get("error"),
        "isolated_subprocess": profile.get("isolated_subprocess"),
    }


def _run_isolated_profile_depth(
    *,
    report_path: Path,
    program_root: Path,
    model_path: Path | None,
    depth: int,
    device: str,
    device_id: int,
    batch_size: int | None,
    cache_len: int | None,
    dtype_seed: str,
    trace: bool,
    trace_iterations: int,
    prompt: str | None,
    tokenizer_path: str | Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        "profile-decode-step",
        "--program-dir",
        str(program_root),
        "--layers",
        str(depth),
        "--device",
        device,
        "--device-id",
        str(device_id),
        "--dtype-seed",
        dtype_seed,
        "--trace-iterations",
        str(trace_iterations),
        "--out",
        str(report_path),
    ]
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if cache_len is not None:
        command.extend(["--cache-len", str(cache_len)])
    if trace:
        command.append("--trace")
    if prompt is not None:
        command.extend(["--prompt", prompt])
    if tokenizer_path is not None:
        command.extend(["--tokenizer-path", str(tokenizer_path)])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    isolated = {
        "enabled": True,
        "returncode": result.returncode,
        "command": command,
        "stdout": _diagnostic_excerpt(result.stdout),
        "stderr": _diagnostic_excerpt(result.stderr),
    }
    if report_path.is_file():
        profile = json.loads(report_path.read_text())
    else:
        profile = {
            "schema_version": 1,
            "command": "profile-decode-step",
            "status": "fail",
            "passed": False,
            "program_dir": str(program_root),
            "model_path": str(model_path) if model_path else None,
            "layers": depth,
            "batch_size": batch_size,
            "cache_len": cache_len,
            "device": device,
            "device_id": device_id,
            "dtype_seed": dtype_seed,
            "error": "isolated profile depth step did not write a report",
        }
        _write_json(report_path, profile)
    profile["isolated_subprocess"] = isolated
    return profile


def _depth_sweep_acceptance(
    records: list[dict[str, Any]],
    *,
    depths: list[int],
    program_num_layers: int,
    dry_run: bool,
    require_full_depth: bool,
) -> dict[str, Any]:
    observed_depths = [record.get("depth") for record in records]
    checks = [
        _check(
            "decode_depth_sweep.depths",
            observed_depths == depths,
            observed=observed_depths,
            expected=depths,
        ),
        _check(
            "decode_depth_sweep.monotonic_depths",
            depths == sorted(depths) and len(depths) == len(set(depths)),
            observed=depths,
            expected="strictly increasing unique depths",
        ),
        _check(
            "decode_depth_sweep.all_depths_passed",
            all(record.get("passed") is True for record in records),
            observed=[
                {
                    "depth": record.get("depth"),
                    "status": record.get("status"),
                    "passed": record.get("passed"),
                }
                for record in records
            ],
            expected=True,
        ),
        _check(
            "decode_depth_sweep.layer_profile_counts",
            all(
                _record_layer_profiles_match_depth(record, dry_run=dry_run)
                for record in records
            ),
            observed=[
                {
                    "depth": record.get("depth"),
                    "layer_profile_count": record.get("layer_profile_count"),
                    "layer_profile_ids": record.get("layer_profile_ids"),
                }
                for record in records
            ],
            expected="layer profile ids [0..depth)",
        ),
        _check(
            "decode_depth_sweep.throughput_summary",
            all(_record_has_throughput(record, dry_run=dry_run) for record in records),
            observed=[
                {
                    "depth": record.get("depth"),
                    "throughput_status": (
                        record.get("throughput_summary") or {}
                    ).get("status"),
                    "tokens_per_second_per_user": record.get(
                        "tokens_per_second_per_user"
                    ),
                }
                for record in records
            ],
            expected="dry_run or measured throughput for each depth",
        ),
        _check(
            "decode_depth_sweep.profile_breakdown",
            all(
                _record_has_profile_breakdown(record)
                for record in records
            ),
            observed=[
                _record_profile_breakdown_observed(record)
                for record in records
            ],
            expected=(
                "section latency, per-layer latency, and LM-head/argmax "
                "profile evidence for each depth"
            ),
        ),
    ]
    if require_full_depth:
        checks.insert(
            2,
            _check(
                "decode_depth_sweep.full_depth",
                program_num_layers in depths,
                observed=max(depths) if depths else None,
                expected=program_num_layers,
            ),
        )
    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "failed_checks": [
            check["name"] for check in checks if not check["passed"]
        ],
    }


def _record_layer_profiles_match_depth(
    record: dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    if record.get("status") == "skipped":
        return False
    depth = record.get("depth")
    ids = record.get("layer_profile_ids")
    try:
        expected = list(range(int(depth)))
    except (TypeError, ValueError):
        return False
    return ids == expected and record.get("layer_profile_count") == len(expected)


def _record_has_throughput(
    record: dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    throughput = record.get("throughput_summary")
    if not isinstance(throughput, dict):
        return False
    if dry_run:
        return throughput.get("status") == "dry_run"
    return (
        throughput.get("status") == "measured"
        and _positive_number(throughput.get("tokens_per_second_per_user"))
        and _positive_number(throughput.get("aggregate_tokens_per_second"))
    )


def _record_has_profile_breakdown(
    record: dict[str, Any],
) -> bool:
    if record.get("status") == "skipped":
        return False
    depth = record.get("depth")
    try:
        expected_layer_ids = list(range(int(depth)))
    except (TypeError, ValueError):
        return False
    layer_profiles = record.get("layer_profiles")
    lm_head_profile = record.get("lm_head_profile")
    return (
        _has_nonnegative_fields(
            record.get("section_latency_ms"),
            PROFILE_SECTION_LATENCY_KEYS,
        )
        and _layer_profile_ids(layer_profiles) == expected_layer_ids
        and _layer_profiles_have_nonnegative_fields(
            layer_profiles,
            PROFILE_LAYER_LATENCY_KEYS,
        )
        and _lm_head_profile_complete(lm_head_profile)
    )


def _record_profile_breakdown_observed(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "depth": record.get("depth"),
        "section_latency_ms": _field_keys(record.get("section_latency_ms")),
        "layer_profile_ids": _layer_profile_ids(record.get("layer_profiles")),
        "layer_profile_fields": [
            _field_keys(profile)
            for profile in record.get("layer_profiles") or []
            if isinstance(profile, dict)
        ],
        "lm_head_profile": _lm_head_profile_observed(
            record.get("lm_head_profile")
        ),
    }


def _layer_profile_ids(layer_profiles: Any) -> list[int]:
    if not isinstance(layer_profiles, list):
        return []
    layer_ids = []
    for profile in layer_profiles:
        if not isinstance(profile, dict):
            return []
        try:
            layer_ids.append(int(profile["layer_id"]))
        except (KeyError, TypeError, ValueError):
            return []
    return layer_ids


def _layer_profiles_have_nonnegative_fields(
    layer_profiles: Any,
    fields: tuple[str, ...],
) -> bool:
    if not isinstance(layer_profiles, list) or not layer_profiles:
        return False
    return all(
        isinstance(profile, dict)
        and _has_nonnegative_fields(profile, fields)
        for profile in layer_profiles
    )


def _lm_head_profile_complete(profile: Any) -> bool:
    if not isinstance(profile, dict):
        return False
    return (
        _positive_number(profile.get("split_count"))
        and _nonnegative_number(profile.get("lm_head_ms"))
        and _nonnegative_number(profile.get("argmax_ms"))
        and profile.get("argmax_status") in {"profiled", "skipped"}
    )


def _lm_head_profile_observed(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return {
        "split_count": profile.get("split_count"),
        "lm_head_ms": profile.get("lm_head_ms"),
        "argmax_ms": profile.get("argmax_ms"),
        "argmax_status": profile.get("argmax_status"),
    }


def _has_nonnegative_fields(value: Any, fields: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    return all(_nonnegative_number(value.get(field)) for field in fields)


def _field_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    check = {
        "name": name,
        "passed": bool(passed),
    }
    check.update(details)
    return check


def _field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _diagnostic_excerpt(value: str | None, *, limit: int = 2000) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
