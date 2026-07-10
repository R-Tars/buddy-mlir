from __future__ import annotations

from typing import Any

from .schema import (
    field_keys as _field_keys,
    has_nonnegative_fields as _has_nonnegative_fields,
    int_equal as _int_equal,
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    positive_number as _positive_number,
)


PROFILE_BOTTLENECK_SECTION_KEYS = (
    "tensor_conversion_ms",
    "embedding_ms",
    "per_layer_attention_ms",
    "per_layer_mlp_ms",
    "layer_stack_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
    "trace_execute_ms",
)


def lm_head_profile_complete(
    profile: Any,
    *,
    output_kind: Any = "token",
) -> bool:
    if not isinstance(profile, dict):
        return False
    expected_argmax_status = (
        "skipped" if output_kind == "logits" else "profiled"
    )
    return (
        _positive_number(profile.get("split_count"))
        and _nonnegative_number(profile.get("lm_head_ms"))
        and _nonnegative_number(profile.get("argmax_ms"))
        and profile.get("argmax_status") == expected_argmax_status
    )


def lm_head_profile_observed(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return {
        "split_count": profile.get("split_count"),
        "lm_head_ms": profile.get("lm_head_ms"),
        "argmax_ms": profile.get("argmax_ms"),
        "argmax_status": profile.get("argmax_status"),
    }


def trace_profile_complete(
    trace: Any,
    throughput: Any,
    *,
    expected_iterations: Any,
) -> bool:
    if not isinstance(trace, dict) or not isinstance(throughput, dict):
        return False
    samples = trace.get("execute_samples_ms")
    return (
        trace.get("status") == "captured_and_executed"
        and _int_equal(trace.get("iterations"), expected_iterations)
        and trace_samples_complete(samples, expected_iterations)
        and _nonnegative_number(trace.get("capture_latency_ms"))
        and _positive_number(trace.get("execute_latency_ms"))
        and _positive_number(throughput.get("trace_execute_mean_ms"))
        and _positive_number(
            throughput.get("trace_execute_tokens_per_second_per_user")
        )
        and _positive_number(
            throughput.get("trace_execute_aggregate_tokens_per_second")
        )
        and _int_equal(throughput.get("trace_iterations"), expected_iterations)
    )


def trace_samples_complete(samples: Any, expected_iterations: Any) -> bool:
    if not isinstance(samples, list):
        return False
    try:
        expected_count = int(expected_iterations)
    except (TypeError, ValueError):
        return False
    return (
        len(samples) == expected_count
        and bool(samples)
        and all(_positive_number(sample) for sample in samples)
    )


def trace_profile_observed(
    trace: Any,
    throughput: Any,
) -> dict[str, Any]:
    trace_dict = trace if isinstance(trace, dict) else {}
    throughput_dict = throughput if isinstance(throughput, dict) else {}
    return {
        "status": trace_dict.get("status"),
        "iterations": trace_dict.get("iterations"),
        "execute_sample_count": trace_dict.get("execute_sample_count"),
        "capture_latency_ms": trace_dict.get("capture_latency_ms"),
        "execute_latency_ms": trace_dict.get("execute_latency_ms"),
        "execute_samples_ms": trace_dict.get("execute_samples_ms"),
        "trace_execute_mean_ms": throughput_dict.get(
            "trace_execute_mean_ms"
        ),
        "trace_execute_tokens_per_second_per_user": throughput_dict.get(
            "trace_execute_tokens_per_second_per_user"
        ),
        "trace_execute_aggregate_tokens_per_second": throughput_dict.get(
            "trace_execute_aggregate_tokens_per_second"
        ),
        "trace_iterations": throughput_dict.get("trace_iterations"),
    }


def throughput_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    return (
        summary.get("status") == "measured"
        and _positive_number(summary.get("latency_ms"))
        and _positive_number(summary.get("tokens_per_second_per_user"))
        and _positive_number(summary.get("aggregate_tokens_per_second"))
    )


def throughput_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "status": summary.get("status"),
        "latency_ms": summary.get("latency_ms"),
        "tokens_per_second_per_user": summary.get(
            "tokens_per_second_per_user"
        ),
        "aggregate_tokens_per_second": summary.get(
            "aggregate_tokens_per_second"
        ),
        "trace_execute_tokens_per_second_per_user": summary.get(
            "trace_execute_tokens_per_second_per_user"
        ),
    }


def layer_profile_ids(layer_profiles: Any) -> list[int]:
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


def layer_profiles_have_nonnegative_fields(
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


def layer_profile_field_keys(layer_profiles: Any) -> list[list[str]]:
    if not isinstance(layer_profiles, list):
        return []
    return [
        _field_keys(profile)
        for profile in layer_profiles
        if isinstance(profile, dict)
    ]


def bottleneck_summary_complete(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    sections = summary.get("sections_ms")
    max_section = summary.get("max_section")
    return (
        _non_empty_string(max_section)
        and isinstance(sections, dict)
        and str(max_section) in sections
        and _nonnegative_number(summary.get("max_section_ms"))
        and _has_nonnegative_fields(
            sections,
            PROFILE_BOTTLENECK_SECTION_KEYS,
        )
    )


def bottleneck_summary_observed(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    sections = summary.get("sections_ms")
    return {
        "max_section": summary.get("max_section"),
        "max_section_ms": summary.get("max_section_ms"),
        "sections_ms": _field_keys(sections),
    }


def step_trace_summary(step: dict[str, Any]) -> dict[str, Any]:
    trace = step.get("trace") or {}
    return trace if isinstance(trace, dict) else {}
