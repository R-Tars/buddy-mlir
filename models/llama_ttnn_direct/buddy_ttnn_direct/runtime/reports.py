from __future__ import annotations

from pathlib import Path
from typing import Any


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
