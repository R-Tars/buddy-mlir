from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runtime_environment import collect_ttnn_environment


NO_TTNN_DEVICE_MESSAGE = "No TTNN device detected. Use --dry-run or run on P150A."
NUMERIC_REFERENCE_NOT_RUN_REASON = (
    "No torch numeric reference is executed by this smoke path yet; "
    "the reference evidence is limited to generated-path structure, "
    "shape, and dtype checks."
)
PROMPT_CONDITIONED_GENERATE_SEMANTICS = (
    "prompt_conditioned_prefill_decode"
)


def write_report(out: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_report = _json_safe_report_value(report)
    report.clear()
    report.update(safe_report)
    out_path.write_text(json.dumps(report, indent=2) + "\n")


def _json_safe_report_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_report_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_report_value(item) for item in value]
    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }


def dry_run_reference(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": "dry_run",
        "passed": None,
        "numeric_reference": {
            "status": "not_run",
            "reason": NUMERIC_REFERENCE_NOT_RUN_REASON,
        },
        "checks": [],
    }


def planned_cache_population(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "layer_id": layer_id,
            "status": "planned",
            "key_cache_shape": plan["expected_output_shapes"]["key_cache"],
            "value_cache_shape": plan["expected_output_shapes"]["value_cache"],
            "write_policy": "fill_cache_per_user",
            "update_shape_layout": "batch_heads_seq_head_dim",
            "planned_user_count": plan["batch_size"],
        }
        for layer_id in range(int(plan["layers"]))
    ]


def trace_report(
    *,
    requested: bool,
    status: str,
    iterations: int = 0,
    trace_id: Any | None = None,
    capture_latency_ms: float | None = None,
    execute_latency_ms: float | None = None,
    execute_samples_ms: list[float] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": requested,
        "status": status,
        "iterations": iterations,
    }
    if trace_id is not None:
        report["trace_id"] = trace_id
    if capture_latency_ms is not None:
        report["capture_latency_ms"] = capture_latency_ms
    if execute_latency_ms is not None:
        report["execute_latency_ms"] = execute_latency_ms
    if execute_samples_ms is not None:
        report["execute_samples_ms"] = execute_samples_ms
    if error is not None:
        report["error"] = error
    return report


def _shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _dtype(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    return str(dtype) if dtype is not None else None


def _generated_token_id_source(
    per_step_token_metadata: list[dict[str, Any]],
) -> str:
    sources = {
        str(step.get("token_materialization_source"))
        for step in per_step_token_metadata
    }
    if not sources:
        return "none"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _generated_token_materialization_status(
    per_step_token_metadata: list[dict[str, Any]],
) -> str:
    statuses = {
        str(step.get("token_materialization_status"))
        for step in per_step_token_metadata
    }
    if not statuses:
        return "not_run"
    if statuses == {"materialized"}:
        return "materialized"
    if statuses == {"unavailable"}:
        return "unavailable"
    return "partial"


def generated_token_budget(
    *,
    max_new_tokens: int,
    decode_steps: int,
) -> dict[str, Any]:
    prefill_first_token_count = 1 if max_new_tokens > 0 else 0
    return {
        "max_new_tokens": max_new_tokens,
        "prefill_first_token_count": prefill_first_token_count,
        "decode_loop_token_count": decode_steps,
        "decode_steps": decode_steps,
        "total_planned_generated_tokens": (
            prefill_first_token_count + decode_steps
        ),
        "decode_steps_formula": (
            "max_new_tokens - 1 because the first generated token is "
            "materialized from prefill output"
        ),
    }


def cache_population_summary(cache_population: Any) -> dict[str, Any]:
    if not isinstance(cache_population, list):
        return {
            "layer_count": 0,
            "layer_ids": [],
            "status_counts": {},
            "write_policies": [],
            "update_shape_layouts": [],
            "key_cache_shapes": [],
            "value_cache_shapes": [],
            "filled_user_count_total": 0,
            "planned_user_count_total": 0,
        }
    status_counts: dict[str, int] = {}
    layer_ids: list[int] = []
    write_policies: set[str] = set()
    update_shape_layouts: set[str] = set()
    key_cache_shapes: list[list[int]] = []
    value_cache_shapes: list[list[int]] = []
    filled_user_count_total = 0
    planned_user_count_total = 0
    for entry in cache_population:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        layer_id = entry.get("layer_id")
        if layer_id is not None:
            layer_ids.append(int(layer_id))
        write_policy = entry.get("write_policy")
        if write_policy:
            write_policies.add(str(write_policy))
        update_shape_layout = entry.get("update_shape_layout")
        if update_shape_layout:
            update_shape_layouts.add(str(update_shape_layout))
        key_shape = entry.get("key_cache_shape")
        if isinstance(key_shape, list):
            key_cache_shapes.append(key_shape)
        value_shape = entry.get("value_cache_shape")
        if isinstance(value_shape, list):
            value_cache_shapes.append(value_shape)
        filled_user_count_total += int(entry.get("filled_user_count") or 0)
        planned_user_count_total += int(entry.get("planned_user_count") or 0)
    return {
        "layer_count": len(layer_ids),
        "layer_ids": layer_ids,
        "status_counts": status_counts,
        "write_policies": sorted(write_policies),
        "update_shape_layouts": sorted(update_shape_layouts),
        "key_cache_shapes": key_cache_shapes,
        "value_cache_shapes": value_cache_shapes,
        "filled_user_count_total": filled_user_count_total,
        "planned_user_count_total": planned_user_count_total,
    }


def generate_reference_summary(
    *,
    prefill_reference: dict[str, Any],
    step_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_steps = [
        int(step["step_index"])
        for step in step_reports
        if not step.get("passed")
    ]
    passed = bool(prefill_reference.get("passed")) and not failed_steps
    return {
        "kind": "generate_prefill_then_decode_structural",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "prefill_status": prefill_reference.get("status"),
        "decode_step_count": len(step_reports),
        "failed_decode_steps": failed_steps,
    }


def generate_base_report(
    *,
    program_dir: Path,
    program_num_layers: int,
    layers: int,
    max_new_tokens: int,
    decode_steps: int,
    prefill_len: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    dry_run: bool,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
) -> dict[str, Any]:
    token_budget = generated_token_budget(
        max_new_tokens=max_new_tokens,
        decode_steps=decode_steps,
    )
    return {
        "schema_version": 1,
        "command": "generate",
        "mode": "generate",
        "template": "prefill_then_decode_generate",
        "program_dir": str(program_dir),
        "program_num_layers": program_num_layers,
        "layers": layers,
        "device": device,
        "device_id": device_id,
        "batch_size": batch_size,
        "prefill_len": prefill_len,
        "cache_len": cache_len,
        "max_new_tokens": max_new_tokens,
        "decode_steps": decode_steps,
        "generated_token_budget": token_budget,
        "prefill_first_token_counts_as_generated_token": True,
        "decode_steps_excludes_prefill_token": True,
        "dtype_seed": dtype_seed,
        "dtype": "bfloat16" if dtype_seed == "bf16" else "float32",
        "layout": "tile",
        "dry_run": dry_run,
        "prefill_plan": prefill_plan,
        "decode_plan": decode_plan,
        "prefill_op_sequence": prefill_plan["op_sequence"],
        "decode_op_sequence": decode_plan["op_sequence"],
        "model_semantics": PROMPT_CONDITIONED_GENERATE_SEMANTICS,
        "kv_cache_source": "prefill",
        "semantic_disclaimer": (
            "This generate path runs prefill before decode. It is a first "
            "functional bring-up path; decode reuses TTNN token tensors "
            "directly and host token materialization is kept for reporting and "
            "detokenization. Performance parity is not claimed."
        ),
        "ttnn_environment": collect_ttnn_environment(None),
    }


def generate_no_device_report(
    *,
    detail: str,
    ttnn_module: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return generate_failed_report(
        status="no_device",
        message=NO_TTNN_DEVICE_MESSAGE,
        detail=detail,
        ttnn_module=ttnn_module,
        **kwargs,
    )


def generate_failed_report(
    *,
    program_dir: Path,
    layers: int,
    max_new_tokens: int,
    decode_steps: int,
    prefill_len: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
    status: str,
    message: str,
    detail: str,
    ttnn_module: Any | None = None,
) -> dict[str, Any]:
    prefill_cache_population = planned_cache_population(prefill_plan)
    report = generate_base_report(
        program_dir=program_dir,
        program_num_layers=layers,
        layers=layers,
        max_new_tokens=max_new_tokens,
        decode_steps=decode_steps,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        decode_plan=decode_plan,
        prefill_plan=prefill_plan,
    )
    report.update(
        {
            "passed": False,
            "status": status,
            "runtime_status": status,
            "prefill_status": status,
            "decode_loop_runtime_owned": False,
            "generate_runtime_owned": False,
            "generated_token_ids": [],
            "generated_text": "",
            "generated_text_by_user": [],
            "generated_text_status": "not_run",
            "generated_text_source": status,
            "prefill": {
                "status": status,
                "cache_population": prefill_cache_population,
            },
            "prefill_cache_population": prefill_cache_population,
            "prefill_cache_population_summary": (
                cache_population_summary(prefill_cache_population)
            ),
            "step_reports": [],
            "per_step_token_metadata": [],
            "tensor_conversion_count": 0,
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "decode_token_runtime_handoff": "not_run",
            "decode_token_host_roundtrip_per_step": False,
            "host_token_materialization_for_reporting_only": False,
            "host_copy_profile": host_copy_not_run_profile(status),
            "section_profile": section_profile_not_run(status),
            "latency_ms": None,
            "trace": trace_report(requested=False, status="disabled"),
            "reference": {
                "kind": "generate_prefill_then_decode",
                "status": "not_run",
                "passed": False,
                "checks": [],
            },
            "error": message,
            "detail": detail,
            "ttnn_version": getattr(ttnn_module, "__version__", None),
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
        }
    )
    report["end_to_end_contract"] = generate_end_to_end_contract(report)
    return report


def generate_dry_run_report(
    *,
    program_dir: Path,
    program_num_layers: int,
    layers: int,
    max_new_tokens: int,
    decode_steps: int,
    prefill_len: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
) -> dict[str, Any]:
    prefill_cache_population = planned_cache_population(prefill_plan)
    report = generate_base_report(
        program_dir=program_dir,
        program_num_layers=program_num_layers,
        layers=layers,
        max_new_tokens=max_new_tokens,
        decode_steps=decode_steps,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=True,
        decode_plan=decode_plan,
        prefill_plan=prefill_plan,
    )
    report.update(
        {
            "passed": True,
            "status": "dry_run",
            "runtime_status": "dry_run",
            "prefill_status": "dry_run",
            "decode_loop_runtime_owned": False,
            "planned_decode_loop_runtime_owned": True,
            "kv_cache_source": "prefill",
            "runtime_owner": "TTNNDirectRuntimeContext",
            "generated_token_ids": [],
            "generated_text": "",
            "generated_text_by_user": [],
            "generated_text_status": "not_run",
            "generated_text_source": "dry_run",
            "prefill": {
                "status": "dry_run",
                "cache_population": prefill_cache_population,
            },
            "prefill_cache_population": prefill_cache_population,
            "prefill_cache_population_summary": (
                cache_population_summary(prefill_cache_population)
            ),
            "step_reports": [],
            "per_step_token_metadata": [],
            "tensor_conversion_count": (
                prefill_plan["tensor_conversion_count"]
                + decode_plan["tensor_conversion_count"]
            ),
            "runtime_context": {
                "class": "TTNNDirectRuntimeContext",
                "status": "planned",
                "owns": [
                    "parameters",
                    "kv_cache",
                    "page_table",
                    "rotary_state",
                    "tokenizer",
                    "generated_model",
                ],
                "parameter_tensorization_count_per_generate": 1,
                "parameter_tensorization_count_per_decode_step": 0,
                "kv_cache_initialization_count_per_generate": 1,
                "kv_cache_reinitialized_per_step": False,
                "decode_token_runtime_handoff": "device_tensor_direct",
                "decode_token_host_roundtrip_per_step": False,
                "host_token_materialization_for_reporting_only": True,
                "decode_step_count": decode_steps,
            },
            "parameter_tensorization_count_per_generate": 1,
            "parameter_tensorization_count_per_decode_step": 0,
            "kv_cache_initialization_count_per_generate": 1,
            "kv_cache_reinitialized_per_step": False,
            "decode_token_runtime_handoff": "device_tensor_direct",
            "decode_token_host_roundtrip_per_step": False,
            "host_token_materialization_for_reporting_only": True,
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "host_copy_profile": host_copy_not_run_profile("dry_run"),
            "section_profile": section_profile_not_run("dry_run"),
            "trace": trace_report(requested=False, status="disabled"),
            "reference": dry_run_reference("generate"),
            "error": None,
            "message": "Dry run only; TTNN device is not required.",
        }
    )
    report["end_to_end_contract"] = generate_end_to_end_contract(report)
    return report


def generate_success_report(
    *,
    program_dir: Path,
    program_num_layers: int,
    layers: int,
    max_new_tokens: int,
    decode_steps: int,
    prefill_len: int,
    device: str,
    device_id: int,
    batch_size: int,
    cache_len: int,
    dtype_seed: str,
    decode_plan: dict[str, Any],
    prefill_plan: dict[str, Any],
    context: Any,
    prefill_token: Any,
    first_token: Any,
    prefill_latency_ms: float,
    prefill_output_shapes: dict[str, Any],
    prefill_cache_population: list[dict[str, Any]],
    prefill_reference: dict[str, Any],
    step_reports: list[dict[str, Any]],
    per_step_token_metadata: list[dict[str, Any]],
    generated_token_ids_by_user: list[list[int]],
    token_materialization: Any,
    text_report: dict[str, Any],
    decode_runtime_state: dict[str, Any],
    rotary_runtime_state: dict[str, Any],
    tensor_conversion_count: int,
    decode_runtime_state_input_tensor_count: int,
    decode_rotary_runtime_input_tensor_count: int,
    latency_ms: float,
    section_profiler: Any,
    ttnn_module: Any,
) -> dict[str, Any]:
    decode_passed = all(step["passed"] for step in step_reports)
    passed = bool(prefill_reference["passed"] and decode_passed)
    parameter_setup = dict(context.parameter_setup)
    parameter_setup.update(
        {
            "generate_runtime_owned": passed,
            "decode_loop_runtime_owned": decode_steps == 0 or decode_passed,
            "prefill_prompt_runtime_input_tensor_count": (
                context.prefill_prompt_runtime_input_tensor_count
            ),
            "prefill_rotary_runtime_input_tensor_count": (
                context.prefill_rotary_runtime_input_tensor_count
            ),
            "prefill_first_token_tensor_conversion_count": (
                first_token.tensor_conversion_count
            ),
            "decode_runtime_state_input_tensor_count": (
                decode_runtime_state_input_tensor_count
            ),
            "decode_rotary_runtime_input_tensor_count": (
                decode_rotary_runtime_input_tensor_count
            ),
            "decode_loop_step_count": decode_steps,
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "parameter_tensorization_count_per_generate": (
                context.parameter_tensorization_count_per_generate
            ),
            "parameter_tensorization_count_per_decode_step": (
                context.parameter_tensorization_count_per_decode_step
            ),
            "kv_cache_initialization_count_per_generate": (
                context.kv_cache_initialization_count_per_generate
            ),
            "kv_cache_reinitialized_per_step": (
                context.kv_cache_reinitialized_per_step
            ),
            "decode_token_runtime_handoff": (
                context.decode_token_runtime_handoff
            ),
            "decode_token_host_roundtrip_per_step": (
                context.decode_token_host_roundtrip_per_step
            ),
            "host_token_materialization_for_reporting_only": (
                context.host_token_materialization_for_reporting_only
            ),
        }
    )
    host_copy = host_copy_profile(
        first_token_materialization_ms=(
            token_materialization.first_token_materialization_ms
        ),
        step_reports=step_reports,
    )
    report = generate_base_report(
        program_dir=program_dir,
        program_num_layers=program_num_layers,
        layers=layers,
        max_new_tokens=max_new_tokens,
        decode_steps=decode_steps,
        prefill_len=prefill_len,
        device=device,
        device_id=device_id,
        batch_size=batch_size,
        cache_len=cache_len,
        dtype_seed=dtype_seed,
        dry_run=False,
        decode_plan=decode_plan,
        prefill_plan=prefill_plan,
    )
    report.update(
        {
            "passed": passed,
            "status": "passed" if passed else "reference_mismatch",
            "runtime_status": "passed" if passed else "reference_mismatch",
            "prefill_status": (
                "passed"
                if prefill_reference["passed"]
                else "reference_mismatch"
            ),
            "decode_loop_runtime_owned": decode_steps == 0 or decode_passed,
            "generate_runtime_owned": passed,
            "kv_cache_source": "prefill",
            "input_source": "prompt_prefill",
            "runtime_owner": "TTNNDirectRuntimeContext",
            "parameter_source": context.parameter_source,
            "parameter_setup": parameter_setup,
            "prompt_tokenization": context.prefill_tokenization,
            "prefill_tokenization": context.prefill_tokenization,
            "decode_runtime_state": decode_runtime_state,
            "rotary_runtime_state": rotary_runtime_state,
            "kv_cache_runtime_state": context.kv_cache_runtime_state,
            "runtime_context": context.to_report(decode_step_count=decode_steps),
            "parameter_tensorization_count_per_generate": (
                context.parameter_tensorization_count_per_generate
            ),
            "parameter_tensorization_count_per_decode_step": (
                context.parameter_tensorization_count_per_decode_step
            ),
            "kv_cache_initialization_count_per_generate": (
                context.kv_cache_initialization_count_per_generate
            ),
            "kv_cache_reinitialized_per_step": (
                context.kv_cache_reinitialized_per_step
            ),
            "decode_token_runtime_handoff": (
                context.decode_token_runtime_handoff
            ),
            "decode_token_host_roundtrip_per_step": (
                context.decode_token_host_roundtrip_per_step
            ),
            "host_token_materialization_for_reporting_only": (
                context.host_token_materialization_for_reporting_only
            ),
            "prefill": {
                "status": (
                    "passed"
                    if prefill_reference["passed"]
                    else "reference_mismatch"
                ),
                "latency_ms": prefill_latency_ms,
                "output_shapes": prefill_output_shapes,
                "output": {
                    "kind": "token",
                    "shape": _shape(prefill_token),
                    "dtype": _dtype(prefill_token),
                    "repr": repr(prefill_token),
                },
                "first_token": {
                    "status": token_materialization.first_token_status,
                    "source": token_materialization.first_token_source,
                    "token_ids_by_user": (
                        token_materialization.first_token_ids_by_user
                    ),
                    "token_shape": _shape(first_token.token_ids),
                    "runtime_handoff": first_token.runtime_handoff,
                    "runtime_host_roundtrip": first_token.runtime_host_roundtrip,
                    "host_roundtrip": False,
                    "host_materialization_for_reporting": True,
                    "host_materialization_ms": (
                        token_materialization.first_token_materialization_ms
                    ),
                },
                "cache_population": prefill_cache_population,
                "reference": prefill_reference,
            },
            "prefill_cache_population": prefill_cache_population,
            "prefill_cache_population_summary": (
                cache_population_summary(prefill_cache_population)
            ),
            "step_reports": step_reports,
            "per_step_token_metadata": per_step_token_metadata,
            "generated_token_ids": generated_token_ids_by_user,
            "generated_token_id_source": _generated_token_id_source(
                per_step_token_metadata
            ),
            "token_materialization_status": (
                _generated_token_materialization_status(
                    per_step_token_metadata
                )
            ),
            "generated_text": text_report["generated_text"],
            "generated_text_by_user": text_report["generated_text_by_user"],
            "generated_text_status": text_report["status"],
            "generated_text_source": text_report["source"],
            "generated_text_report": text_report,
            "output_shapes": (
                step_reports[-1]["output_shapes"]
                if step_reports
                else prefill_output_shapes
            ),
            "output": step_reports[-1]["output"] if step_reports else None,
            "tensor_conversion_count": tensor_conversion_count,
            "synthetic_runtime_input_tensor_count": 0,
            "synthetic_rotary_tensor_count": 0,
            "synthetic_kv_cache_tensor_count": 0,
            "host_copy_profile": host_copy,
            "section_profile": section_profiler.to_report(
                host_copy_profile=host_copy,
            ),
            "latency_ms": latency_ms,
            "throughput_summary": generate_throughput_summary(
                latency_ms=latency_ms,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            ),
            "trace": trace_report(requested=False, status="disabled"),
            "reference": generate_reference_summary(
                prefill_reference=prefill_reference,
                step_reports=step_reports,
            ),
            "error": None
            if passed
            else "generate structural reference mismatch",
            "ttnn_version": getattr(ttnn_module, "__version__", None),
            "ttnn_environment": collect_ttnn_environment(ttnn_module),
        }
    )
    report["end_to_end_contract"] = generate_end_to_end_contract(report)
    return report


def generate_end_to_end_contract(report: dict[str, Any]) -> dict[str, Any]:
    dry_run = bool(report.get("dry_run"))
    runtime_context = report.get("runtime_context")
    if not isinstance(runtime_context, dict):
        runtime_context = {}

    def value_or_setup(key: str) -> Any:
        value = report.get(key)
        if value is not None:
            return value
        return _setup_count(report, key)

    generated_text_status = report.get("generated_text_status")
    generated_text_ready = (
        generated_text_status in {"decoded", "fallback", "placeholder"}
        and isinstance(report.get("generated_text"), str)
    )
    decode_runtime_owned = (
        bool(report.get("planned_decode_loop_runtime_owned"))
        if dry_run
        else bool(report.get("decode_loop_runtime_owned"))
    )
    prefill_ready = (
        report.get("prefill_status") == "dry_run"
        if dry_run
        else report.get("prefill_status") == "passed"
    )
    prefill = report.get("prefill")
    if not isinstance(prefill, dict):
        prefill = {}
    cache_population = prefill.get("cache_population")
    if not isinstance(cache_population, list):
        cache_population = []
    expected_user_count = optional_int(report.get("batch_size"))
    accepted_cache_write_policies = {
        "fill_cache_per_user",
        "paged_fill_cache_per_user",
    }
    cache_write_policy_ok = bool(cache_population) and all(
        isinstance(entry, dict)
        and entry.get("write_policy") in accepted_cache_write_policies
        for entry in cache_population
    )
    cache_user_count_ok = bool(cache_population)
    for entry in cache_population:
        if not isinstance(entry, dict):
            cache_user_count_ok = False
            continue
        observed_user_count = (
            entry.get("planned_user_count")
            if dry_run
            else entry.get("filled_user_count")
        )
        users = entry.get("users", [])
        if (
            expected_user_count is not None
            and observed_user_count != expected_user_count
        ):
            cache_user_count_ok = False
        if (
            not dry_run
            and expected_user_count is not None
            and len(users) != expected_user_count
        ):
            cache_user_count_ok = False
    runtime_context_ready = (
        runtime_context.get("status") == "planned"
        if dry_run
        else (
            runtime_context.get("status") == "built"
            and runtime_context.get("generated_model_initialized") is True
        )
    )
    checks = [
        {
            "name": "generate.mode",
            "passed": report.get("mode") == "generate",
            "observed": report.get("mode"),
            "expected": "generate",
        },
        {
            "name": "generate.prefill_status",
            "passed": prefill_ready,
            "observed": report.get("prefill_status"),
            "expected": "dry_run" if dry_run else "passed",
        },
        {
            "name": "generate.decode_loop_runtime_owned",
            "passed": decode_runtime_owned,
            "observed": report.get("decode_loop_runtime_owned"),
            "expected": True,
        },
        {
            "name": "generate.kv_cache_source",
            "passed": report.get("kv_cache_source") == "prefill",
            "observed": report.get("kv_cache_source"),
            "expected": "prefill",
        },
        {
            "name": "generate.model_semantics",
            "passed": (
                report.get("model_semantics")
                == PROMPT_CONDITIONED_GENERATE_SEMANTICS
            ),
            "observed": report.get("model_semantics"),
            "expected": PROMPT_CONDITIONED_GENERATE_SEMANTICS,
        },
        {
            "name": "generate.prefill_kv_cache_write_policy",
            "passed": cache_write_policy_ok,
            "observed": [
                entry.get("write_policy")
                for entry in cache_population
                if isinstance(entry, dict)
            ],
            "expected": sorted(accepted_cache_write_policies),
        },
        {
            "name": "generate.prefill_kv_cache_user_count",
            "passed": cache_user_count_ok,
            "observed": [
                {
                    "layer_id": entry.get("layer_id"),
                    "planned_user_count": entry.get("planned_user_count"),
                    "filled_user_count": entry.get("filled_user_count"),
                    "user_report_count": len(entry.get("users", [])),
                }
                for entry in cache_population
                if isinstance(entry, dict)
            ],
            "expected": expected_user_count,
        },
        {
            "name": "generate.input_source",
            "passed": dry_run or report.get("input_source") == "prompt_prefill",
            "observed": report.get("input_source"),
            "expected": "prompt_prefill",
        },
        {
            "name": "generate.generated_text_available",
            "passed": dry_run or generated_text_ready,
            "observed": {
                "generated_text_status": generated_text_status,
                "generated_text_type": type(report.get("generated_text")).__name__,
            },
            "expected": "decoded/fallback/placeholder generated_text",
        },
        {
            "name": "generate.synthetic_runtime_inputs",
            "passed": value_or_setup("synthetic_runtime_input_tensor_count") == 0,
            "observed": value_or_setup("synthetic_runtime_input_tensor_count"),
            "expected": 0,
        },
        {
            "name": "generate.synthetic_rotary_inputs",
            "passed": value_or_setup("synthetic_rotary_tensor_count") == 0,
            "observed": value_or_setup("synthetic_rotary_tensor_count"),
            "expected": 0,
        },
        {
            "name": "generate.synthetic_kv_cache_inputs",
            "passed": value_or_setup("synthetic_kv_cache_tensor_count") == 0,
            "observed": value_or_setup("synthetic_kv_cache_tensor_count"),
            "expected": 0,
        },
        {
            "name": "generate.parameter_tensorization_once",
            "passed": (
                value_or_setup("parameter_tensorization_count_per_generate")
                == 1
            ),
            "observed": value_or_setup(
                "parameter_tensorization_count_per_generate"
            ),
            "expected": 1,
        },
        {
            "name": "generate.no_decode_step_parameter_tensorization",
            "passed": (
                value_or_setup("parameter_tensorization_count_per_decode_step")
                == 0
            ),
            "observed": value_or_setup(
                "parameter_tensorization_count_per_decode_step"
            ),
            "expected": 0,
        },
        {
            "name": "generate.kv_cache_not_reinitialized_per_step",
            "passed": report.get("kv_cache_reinitialized_per_step") is False,
            "observed": report.get("kv_cache_reinitialized_per_step"),
            "expected": False,
        },
        {
            "name": "generate.decode_token_device_handoff",
            "passed": (
                value_or_setup("decode_token_runtime_handoff")
                == "device_tensor_direct"
                and value_or_setup("decode_token_host_roundtrip_per_step")
                is False
            ),
            "observed": {
                "decode_token_runtime_handoff": value_or_setup(
                    "decode_token_runtime_handoff"
                ),
                "decode_token_host_roundtrip_per_step": value_or_setup(
                    "decode_token_host_roundtrip_per_step"
                ),
            },
            "expected": {
                "decode_token_runtime_handoff": "device_tensor_direct",
                "decode_token_host_roundtrip_per_step": False,
            },
        },
        {
            "name": "generate.runtime_context",
            "passed": runtime_context_ready,
            "observed": {
                "status": runtime_context.get("status"),
                "generated_model_initialized": runtime_context.get(
                    "generated_model_initialized"
                ),
            },
            "expected": "planned" if dry_run else "built generated model",
        },
    ]
    failed_checks = [
        check["name"] for check in checks if not bool(check["passed"])
    ]
    status = "dry_run" if dry_run else ("passed" if not failed_checks else "failed")
    return {
        "schema_version": 1,
        "status": status,
        "passed": not failed_checks,
        "dry_run": dry_run,
        "checks": checks,
        "failed_checks": failed_checks,
        "runtime_input_summary": {
            "prefill_prompt_runtime_input_tensor_count": value_or_setup(
                "prefill_prompt_runtime_input_tensor_count"
            ),
            "prefill_page_table_runtime_input_tensor_count": value_or_setup(
                "prefill_page_table_runtime_input_tensor_count"
            ),
            "prefill_rotary_runtime_input_tensor_count": value_or_setup(
                "prefill_rotary_runtime_input_tensor_count"
            ),
            "decode_runtime_state_input_tensor_count": value_or_setup(
                "decode_runtime_state_input_tensor_count"
            ),
            "decode_rotary_runtime_input_tensor_count": value_or_setup(
                "decode_rotary_runtime_input_tensor_count"
            ),
            "kv_cache_runtime_input_tensor_count": value_or_setup(
                "kv_cache_runtime_input_tensor_count"
            ),
            "synthetic_runtime_input_tensor_count": value_or_setup(
                "synthetic_runtime_input_tensor_count"
            ),
            "synthetic_rotary_tensor_count": value_or_setup(
                "synthetic_rotary_tensor_count"
            ),
            "synthetic_kv_cache_tensor_count": value_or_setup(
                "synthetic_kv_cache_tensor_count"
            ),
            "decode_token_runtime_handoff": value_or_setup(
                "decode_token_runtime_handoff"
            ),
            "decode_token_host_roundtrip_per_step": value_or_setup(
                "decode_token_host_roundtrip_per_step"
            ),
            "host_token_materialization_for_reporting_only": value_or_setup(
                "host_token_materialization_for_reporting_only"
            ),
        },
        "semantic_disclaimer": report.get("semantic_disclaimer"),
    }


def generate_throughput_summary(
    *,
    latency_ms: float | None,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    total_tokens = batch_size * max_new_tokens
    summary: dict[str, Any] = {
        "batch_size": batch_size,
        "generated_tokens_per_user": max_new_tokens,
        "total_generated_tokens": total_tokens,
        "basis": "generate_latency_ms",
    }
    if latency_ms is None or latency_ms <= 0.0:
        summary.update(
            {
                "status": "unavailable",
                "latency_ms": latency_ms,
                "tokens_per_second_per_user": None,
                "aggregate_tokens_per_second": None,
            }
        )
        return summary
    tokens_per_second_per_user = 1000.0 * max_new_tokens / latency_ms
    summary.update(
        {
            "status": "measured",
            "latency_ms": latency_ms,
            "tokens_per_second_per_user": tokens_per_second_per_user,
            "aggregate_tokens_per_second": (
                tokens_per_second_per_user * batch_size
            ),
        }
    )
    return summary


def _setup_count(generate: dict[str, Any], key: str) -> Any:
    setup = generate.get("parameter_setup")
    if isinstance(setup, dict):
        return setup.get(key)
    return None


def host_copy_not_run_profile(status: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": status,
        "basis": "prefill/decode token materialization timing",
        "host_roundtrip_present": False,
        "runtime_host_roundtrip_present": False,
        "runtime_handoff": "device_tensor_direct",
        "host_materialization_for_reporting": False,
        "prefill_first_token_ms": None,
        "decode_token_materialization_ms_samples": [],
        "decode_token_materialization_ms_total": None,
        "decode_token_materialization_ms_mean": None,
        "total_ms": None,
    }


def section_profile_not_run(status: str) -> dict[str, Any]:
    section_names = (
        "embedding_ms",
        "prefill_attention_ms",
        "decode_attention_ms",
        "mlp_ms",
        "prefill_mlp_ms",
        "decode_mlp_ms",
        "final_norm_ms",
        "lm_head_ms",
        "argmax_ms",
        "host_copy_ms",
    )
    return {
        "status": "not_run",
        "reason": status,
        "basis": "generated model method wrappers",
        "sections_ms": {name: None for name in section_names},
        "prefill_layer_profiles": [],
        "decode_layer_profiles": [],
        "lm_head_argmax_total_ms": None,
        "host_copy_ms": None,
    }


def host_copy_profile(
    *,
    first_token_materialization_ms: float,
    step_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = [
        latency
        for latency in (
            float_or_none(step.get("token_materialization_ms"))
            for step in step_reports
        )
        if latency is not None
    ]
    decode_total = sum(samples) if samples else 0.0
    first_token_ms = float(first_token_materialization_ms)
    total_ms = first_token_ms + decode_total
    return {
        "status": "measured",
        "basis": (
            "token materialization for reporting and detokenization after "
            "runtime decode handoff"
        ),
        "host_roundtrip_present": False,
        "runtime_host_roundtrip_present": False,
        "runtime_handoff": "device_tensor_direct",
        "host_materialization_for_reporting": True,
        "prefill_first_token_ms": first_token_ms,
        "decode_token_materialization_ms_samples": samples,
        "decode_token_materialization_ms_total": decode_total,
        "decode_token_materialization_ms_mean": (
            decode_total / len(samples) if samples else None
        ),
        "total_ms": total_ms,
    }


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def default_generate_report_path(out: Path) -> Path:
    suffix = out.suffix or ".json"
    return out.with_name(f"{out.stem}_generate{suffix}")
