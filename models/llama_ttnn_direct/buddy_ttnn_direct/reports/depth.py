from __future__ import annotations

from typing import Any

from ..diagnostics.decode_depth_sweep import (
    PROFILE_LAYER_LATENCY_KEYS,
    PROFILE_SECTION_LATENCY_KEYS,
)
from .profiling import (
    bottleneck_summary_complete as _bottleneck_summary_complete,
    bottleneck_summary_observed as _bottleneck_summary_observed,
    layer_profile_field_keys as _layer_profile_field_keys,
    layer_profile_ids as _layer_profile_ids,
    layer_profiles_have_nonnegative_fields as _layer_profiles_have_nonnegative_fields,
    lm_head_profile_complete as _lm_head_profile_complete,
    lm_head_profile_observed as _lm_head_profile_observed,
)
from .runtime import (
    decode_output_shape_observed as _decode_output_shape_observed,
    decode_output_shapes_complete as _decode_output_shapes_complete,
    decode_runtime_input_observed as _decode_runtime_input_observed,
    decode_runtime_inputs_complete as _decode_runtime_inputs_complete,
    runtime_input_source_supported as _runtime_input_source_supported,
)
from .schema import (
    field_keys as _field_keys,
    has_nonnegative_fields as _has_nonnegative_fields,
    int_equal as _int_equal,
    int_list as _int_list,
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    positive_number as _positive_number,
)


def decode_depth_sweep_records_complete(
    records: Any,
    *,
    expected_depths: Any,
    batch_size: Any,
    cache_len: Any,
    seq_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any,
    page_block_size: Any,
    require_trace: bool,
    trace_iterations: Any,
) -> bool:
    if not isinstance(records, list) or not records:
        return False
    depths = _int_list(expected_depths)
    if len(records) != len(depths):
        return False
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return False
        depth = depths[index]
        if not decode_depth_sweep_record_complete(
            record,
            depth=depth,
            batch_size=batch_size,
            cache_len=cache_len,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            output_kind=output_kind,
            page_block_size=page_block_size,
            require_trace=require_trace,
            trace_iterations=trace_iterations,
        ):
            return False
    return True


def decode_depth_sweep_record_complete(
    record: dict[str, Any],
    *,
    depth: int,
    batch_size: Any,
    cache_len: Any,
    seq_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any,
    page_block_size: Any,
    require_trace: bool,
    trace_iterations: Any,
) -> bool:
    throughput = record.get("throughput_summary")
    if not isinstance(throughput, dict):
        return False
    if not (
        _int_equal(record.get("depth"), depth)
        and record.get("status") == "profiled"
        and record.get("passed") is True
        and _non_empty_string(record.get("profile_report"))
        and _int_equal(record.get("layers"), depth)
        and _int_equal(record.get("batch_size"), batch_size)
        and _int_equal(record.get("cache_len"), cache_len)
        and record.get("parameter_source") == "hf_model"
        and _runtime_input_source_supported(record)
        and _positive_number(record.get("latency_ms"))
        and _positive_number(record.get("tensor_conversion_count"))
        and _nonnegative_number(record.get("tensor_conversion_ms"))
        and _int_equal(record.get("layer_profile_count"), depth)
        and record.get("layer_profile_ids") == list(range(depth))
        and _has_nonnegative_fields(
            record.get("section_latency_ms"),
            PROFILE_SECTION_LATENCY_KEYS,
        )
        and _layer_profile_ids(record.get("layer_profiles"))
        == list(range(depth))
        and _layer_profiles_have_nonnegative_fields(
            record.get("layer_profiles"),
            PROFILE_LAYER_LATENCY_KEYS,
        )
        and _lm_head_profile_complete(
            record.get("lm_head_profile"),
            output_kind=output_kind,
        )
        and record.get("reference_status") == "passed"
        and record.get("reference_failed_checks") == []
        and throughput.get("status") == "measured"
        and _positive_number(record.get("tokens_per_second_per_user"))
        and _positive_number(record.get("aggregate_tokens_per_second"))
        and _positive_number(throughput.get("tokens_per_second_per_user"))
        and _positive_number(throughput.get("aggregate_tokens_per_second"))
        and _decode_runtime_inputs_complete(
            record,
            layer_count=depth,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
        )
        and _bottleneck_summary_complete(record.get("bottleneck_summary"))
        and _decode_output_shapes_complete(
            record.get("output_shapes"),
            layer_count=depth,
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len,
            vocab_size=vocab_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            output_kind=output_kind,
            page_block_size=page_block_size,
        )
    ):
        return False
    if require_trace:
        return (
            record.get("trace_status") == "captured_and_executed"
            and _int_equal(record.get("trace_iterations"), trace_iterations)
        )
    return True


def decode_depth_sweep_records_observed(
    records: Any,
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    observed = []
    for record in records:
        if not isinstance(record, dict):
            continue
        throughput = record.get("throughput_summary")
        bottleneck = record.get("bottleneck_summary")
        observed.append(
            {
                "depth": record.get("depth"),
                "status": record.get("status"),
                "passed": record.get("passed"),
                "layers": record.get("layers"),
                "batch_size": record.get("batch_size"),
                "cache_len": record.get("cache_len"),
                "parameter_source": record.get("parameter_source"),
                "input_source": record.get("input_source"),
                "latency_ms": record.get("latency_ms"),
                "tensor_conversion_count": record.get(
                    "tensor_conversion_count"
                ),
                "layer_profile_count": record.get("layer_profile_count"),
                "layer_profile_ids": record.get("layer_profile_ids"),
                "section_latency_ms": _field_keys(
                    record.get("section_latency_ms")
                ),
                "layer_profile_fields": _layer_profile_field_keys(
                    record.get("layer_profiles")
                ),
                "lm_head_profile": _lm_head_profile_observed(
                    record.get("lm_head_profile")
                ),
                "reference_status": record.get("reference_status"),
                "reference_failed_checks": record.get(
                    "reference_failed_checks"
                ),
                "throughput_status": throughput.get("status")
                if isinstance(throughput, dict)
                else None,
                "tokens_per_second_per_user": record.get(
                    "tokens_per_second_per_user"
                ),
                "aggregate_tokens_per_second": record.get(
                    "aggregate_tokens_per_second"
                ),
                "trace_status": record.get("trace_status"),
                "trace_iterations": record.get("trace_iterations"),
                "output_shapes": _decode_output_shape_observed(
                    record.get("output_shapes")
                ),
                "runtime_inputs": _decode_runtime_input_observed(record),
                "bottleneck": _bottleneck_summary_observed(bottleneck),
            }
        )
    return observed
