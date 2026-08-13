from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..reports.contracts import PROFILE_LAYER_LATENCY_KEYS, PROFILE_SECTION_LATENCY_KEYS
from .decode_step import profile_decode_step
from .depth_sweep_support import (
    check,
    depth_summary,
    exception_details,
    field_counts,
    load_program_num_layers,
    project_fields,
    resolve_depths,
    resolve_output_root,
    result_summary,
    run_depth_sequence,
    run_isolated_command,
    should_isolate_depth_steps,
    sweep_status,
    write_json,
)


DEFAULT_DECODE_DEPTH_TARGETS = (1, 2, 4, "full")
PROFILE_RECORD_FIELDS = """
layers batch_size cache_len parameter_source input_source prompt_tokenization
decode_runtime_state rotary_runtime_state kv_cache_runtime_state input_shapes kv_cache
latency_ms tensor_conversion_count tensor_conversion_ms section_latency_ms
lm_head_profile output_shapes error isolated_subprocess
""".split()


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
    program_root, out_path = Path(program_dir), Path(out)
    program_num_layers = load_program_num_layers(program_root)
    if trace_iterations <= 0:
        raise ValueError("trace_iterations must be positive")
    resolved_depths = resolve_depths(
        depths, program_num_layers=program_num_layers, defaults=DEFAULT_DECODE_DEPTH_TARGETS
    )
    profile_root = resolve_output_root(out_path, profiles_dir, suffix="profiles")
    model_root = Path(model_path) if model_path is not None else None
    isolate = should_isolate_depth_steps(
        isolate_depth_steps,
        dry_run=dry_run,
        injected_modules=(tokenizer_module, ttnn_module, torch_module),
    )

    def execute(depth: int, report_path: Path) -> dict[str, Any]:
        if isolate:
            return _run_isolated_profile_depth(
                report_path=report_path, program_root=program_root, model_path=model_root,
                depth=depth, device=device, device_id=device_id, batch_size=batch_size,
                cache_len=cache_len, dtype_seed=dtype_seed, trace=trace,
                trace_iterations=trace_iterations, prompt=prompt,
                tokenizer_path=tokenizer_path,
            )
        return profile_decode_step(
            out=report_path, program_dir=program_root, layers=depth,
            model_path=model_root, device=device, device_id=device_id,
            batch_size=batch_size, cache_len=cache_len, dtype_seed=dtype_seed,
            trace=trace, trace_iterations=trace_iterations, dry_run=dry_run,
            prompt=prompt, tokenizer_path=tokenizer_path,
            tokenizer_module=tokenizer_module, ttnn_module=ttnn_module,
            torch_module=torch_module,
        )

    records = run_depth_sequence(
        resolved_depths,
        report_path_for=lambda depth: profile_root / f"profile_depth_{depth}.json",
        execute=execute,
        make_record=lambda depth, payload, path: _profile_record(depth, payload, path),
        make_skipped=lambda depth, path, reason: {
            "depth": depth, "status": "skipped", "passed": False,
            "profile_report": str(path), "reason": reason,
        },
        make_exception=lambda depth, path, exc: {
            "depth": depth, "status": "fail", "passed": False,
            "profile_report": str(path), "error": exception_details(exc),
        },
        dry_run=dry_run,
    )
    acceptance = _depth_sweep_acceptance(
        records, depths=resolved_depths, program_num_layers=program_num_layers,
        dry_run=dry_run, require_full_depth=require_full_depth,
    )
    report = {
        "schema_version": 1,
        "command": "decode-depth-sweep",
        "status": sweep_status(records, dry_run=dry_run, accepted=acceptance["passed"]),
        "passed": bool(acceptance["passed"]),
        "dry_run": bool(dry_run),
        "require_full_depth": bool(require_full_depth),
        "isolate_depth_steps": isolate,
        "program_dir": str(program_root),
        "model_path": str(model_root) if model_root else None,
        "profiles_dir": str(profile_root),
        **depth_summary(resolved_depths, program_num_layers),
        "batch_size": batch_size, "cache_len": cache_len,
        "device": device, "device_id": device_id, "dtype_seed": dtype_seed,
        "trace_enabled": trace, "trace_iterations": trace_iterations if trace else 0,
        "prompt_runtime_requested": prompt is not None,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "status_counts": field_counts(records, "status"),
        "reference_status_counts": field_counts(records, "reference_status"),
        "trace_status_counts": field_counts(records, "trace_status"),
        **result_summary(records),
        "records": records,
        "acceptance": acceptance,
    }
    write_json(out_path, report)
    return report


def _profile_record(depth: int, profile: dict[str, Any], report_path: Path) -> dict[str, Any]:
    setup = profile.get("parameter_setup") or {}
    throughput, trace = profile.get("throughput_summary") or {}, profile.get("trace") or {}
    reference, layers = profile.get("reference") or {}, profile.get("layer_profiles") or []
    bottleneck = profile.get("bottleneck_summary") or {}
    record = {
        "depth": depth, "status": profile.get("status"),
        "passed": bool(profile.get("passed")), "profile_report": str(report_path),
        **project_fields(profile, PROFILE_RECORD_FIELDS),
        "synthetic_runtime_input_tensor_count": setup.get("synthetic_runtime_input_tensor_count"),
        "prompt_runtime_input_tensor_count": setup.get("prompt_runtime_input_tensor_count"),
        "decode_runtime_state_input_tensor_count": setup.get("decode_runtime_state_input_tensor_count"),
        "rotary_runtime_input_tensor_count": setup.get("rotary_runtime_input_tensor_count"),
        "kv_cache_runtime_input_tensor_count": setup.get("kv_cache_runtime_input_tensor_count"),
        "synthetic_rotary_tensor_count": setup.get("synthetic_rotary_tensor_count"),
        "layer_profile_count": len(layers),
        "layer_profile_ids": [layer.get("layer_id") for layer in layers if isinstance(layer, dict)],
        "layer_profiles": layers,
        "throughput_summary": throughput,
        "tokens_per_second_per_user": throughput.get("tokens_per_second_per_user"),
        "aggregate_tokens_per_second": throughput.get("aggregate_tokens_per_second"),
        "bottleneck_summary": bottleneck, "max_section": bottleneck.get("max_section"),
        "trace_status": trace.get("status"), "trace_iterations": trace.get("iterations"),
        "reference_status": reference.get("status"), "reference_kind": reference.get("kind"),
        "reference_failed_checks": [
            item.get("name") for item in reference.get("checks", [])
            if isinstance(item, dict) and not item.get("passed")
        ],
    }
    return record


def _run_isolated_profile_depth(
    *, report_path: Path, program_root: Path, model_path: Path | None,
    depth: int, device: str, device_id: int, batch_size: int | None,
    cache_len: int | None, dtype_seed: str, trace: bool,
    trace_iterations: int, prompt: str | None,
    tokenizer_path: str | Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "models.llama_ttnn_direct.buddy_ttnn_direct.cli",
        "diagnose", "--stage", "decode-step-profile", "--program-dir", str(program_root),
        "--layers", str(depth), "--device", device, "--device-id", str(device_id),
        "--dtype-seed", dtype_seed, "--trace-iterations", str(trace_iterations),
        "--out", str(report_path),
    ]
    for flag, value in (("--model-path", model_path), ("--batch-size", batch_size),
                        ("--cache-len", cache_len)):
        if value is not None:
            command.extend([flag, str(value)])
    if trace:
        command.append("--trace")
    for flag, value in (("--prompt", prompt), ("--tokenizer-path", tokenizer_path)):
        if value is not None:
            command.extend([flag, str(value)])

    def missing() -> dict[str, Any]:
        return {
            "schema_version": 1, "command": "profile-decode-step", "status": "fail",
            "passed": False, "program_dir": str(program_root),
            "model_path": str(model_path) if model_path else None, "layers": depth,
            "batch_size": batch_size, "cache_len": cache_len, "device": device,
            "device_id": device_id, "dtype_seed": dtype_seed,
            "error": "isolated profile depth step did not write a report",
        }

    return run_isolated_command(command, report_path=report_path, missing_report=missing)


def _depth_sweep_acceptance(
    records: list[dict[str, Any]], *, depths: list[int], program_num_layers: int,
    dry_run: bool, require_full_depth: bool,
) -> dict[str, Any]:
    observed_depths = [record.get("depth") for record in records]
    checks = [
        check("decode_depth_sweep.depths", observed_depths == depths,
              observed=observed_depths, expected=depths),
        check("decode_depth_sweep.monotonic_depths",
              depths == sorted(depths) and len(depths) == len(set(depths)),
              observed=depths, expected="strictly increasing unique depths"),
        check("decode_depth_sweep.all_depths_passed",
              all(record.get("passed") is True for record in records),
              observed=[project_fields(record, ("depth", "status", "passed")) for record in records],
              expected=True),
        check("decode_depth_sweep.layer_profile_counts",
              all(_record_layer_profiles_match_depth(record) for record in records),
              observed=[project_fields(record, ("depth", "layer_profile_count", "layer_profile_ids")) for record in records],
              expected="layer profile ids [0..depth)"),
        check("decode_depth_sweep.throughput_summary",
              all(_record_has_throughput(record, dry_run=dry_run) for record in records),
              observed=[{"depth": record.get("depth"),
                         "throughput_status": (record.get("throughput_summary") or {}).get("status"),
                         "tokens_per_second_per_user": record.get("tokens_per_second_per_user")}
                        for record in records],
              expected="dry_run or measured throughput for each depth"),
        check("decode_depth_sweep.profile_breakdown",
              all(_record_has_profile_breakdown(record) for record in records),
              observed=[_record_profile_breakdown_observed(record) for record in records],
              expected="section latency, per-layer latency, and LM-head/argmax profile evidence for each depth"),
    ]
    if require_full_depth:
        checks.insert(2, check("decode_depth_sweep.full_depth",
                              program_num_layers in depths,
                              observed=max(depths) if depths else None,
                              expected=program_num_layers))
    failed = [item["name"] for item in checks if not item["passed"]]
    return {"status": "passed" if not failed else "failed", "passed": not failed,
            "checks": checks, "failed_checks": failed}


def _record_layer_profiles_match_depth(record: dict[str, Any]) -> bool:
    if record.get("status") == "skipped":
        return False
    try:
        expected = list(range(int(record.get("depth"))))
    except (TypeError, ValueError):
        return False
    return record.get("layer_profile_ids") == expected and record.get("layer_profile_count") == len(expected)


def _record_has_throughput(record: dict[str, Any], *, dry_run: bool) -> bool:
    throughput = record.get("throughput_summary")
    if not isinstance(throughput, dict):
        return False
    if dry_run:
        return throughput.get("status") == "dry_run"
    return (throughput.get("status") == "measured"
            and _positive_number(throughput.get("tokens_per_second_per_user"))
            and _positive_number(throughput.get("aggregate_tokens_per_second")))


def _record_has_profile_breakdown(record: dict[str, Any]) -> bool:
    if record.get("status") == "skipped":
        return False
    try:
        expected = list(range(int(record.get("depth"))))
    except (TypeError, ValueError):
        return False
    layers, lm_head = record.get("layer_profiles"), record.get("lm_head_profile")
    return (_has_nonnegative_fields(record.get("section_latency_ms"), PROFILE_SECTION_LATENCY_KEYS)
            and _layer_profile_ids(layers) == expected
            and _layer_profiles_have_nonnegative_fields(layers, PROFILE_LAYER_LATENCY_KEYS)
            and _lm_head_profile_complete(lm_head))


def _record_profile_breakdown_observed(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth": record.get("depth"),
        "section_latency_ms": _field_keys(record.get("section_latency_ms")),
        "layer_profile_ids": _layer_profile_ids(record.get("layer_profiles")),
        "layer_profile_fields": [_field_keys(item) for item in record.get("layer_profiles") or [] if isinstance(item, dict)],
        "lm_head_profile": _lm_head_profile_observed(record.get("lm_head_profile")),
    }


def _layer_profile_ids(layer_profiles: Any) -> list[int]:
    if not isinstance(layer_profiles, list):
        return []
    try:
        if not all(isinstance(profile, dict) for profile in layer_profiles):
            return []
        return [int(profile["layer_id"]) for profile in layer_profiles]
    except (KeyError, TypeError, ValueError):
        return []


def _layer_profiles_have_nonnegative_fields(layer_profiles: Any, fields: tuple[str, ...]) -> bool:
    return bool(layer_profiles) and isinstance(layer_profiles, list) and all(
        isinstance(profile, dict) and _has_nonnegative_fields(profile, fields)
        for profile in layer_profiles
    )


def _lm_head_profile_complete(profile: Any) -> bool:
    return (isinstance(profile, dict) and _positive_number(profile.get("split_count"))
            and _nonnegative_number(profile.get("lm_head_ms"))
            and _nonnegative_number(profile.get("argmax_ms"))
            and profile.get("argmax_status") in {"profiled", "skipped"})


def _lm_head_profile_observed(profile: Any) -> dict[str, Any]:
    return project_fields(profile, ("split_count", "lm_head_ms", "argmax_ms", "argmax_status")) if isinstance(profile, dict) else {}


def _has_nonnegative_fields(value: Any, fields: tuple[str, ...]) -> bool:
    return isinstance(value, dict) and all(_nonnegative_number(value.get(field)) for field in fields)


def _field_keys(value: Any) -> list[str]:
    return sorted(str(key) for key in value) if isinstance(value, dict) else []


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
