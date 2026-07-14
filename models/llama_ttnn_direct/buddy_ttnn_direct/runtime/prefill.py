from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from .decode import prefill_token_direct_handoff
from .inputs import build_decode_runtime_state
from .rotary import attach_prefill_rotary_parameters
from .structural import (
    generated_observed_op_sequence,
    observed_cache_population,
    prefill_reference,
)
from .tensor_meta import runtime_int_tensor, tensor_dtype, tensor_shape


# Compatibility names remain patchable for existing diagnostic tests.
_generated_observed_op_sequence = generated_observed_op_sequence
_observed_cache_population = observed_cache_population
_prefill_reference = prefill_reference


def build_prefill_page_table_tensor(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    prompt_token_count: int,
) -> SimpleNamespace:
    runtime_state = build_decode_runtime_state(
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        prompt_token_count=prompt_token_count,
    )
    kwargs = {"device": device}
    dtype = getattr(
        ttnn,
        "int32",
        getattr(ttnn, "uint32", getattr(ttnn, "bfloat16", None)),
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    page_table = ttnn.from_torch(
        runtime_int_tensor(
            torch,
            runtime_state.page_table,
            name="prefill_page_table",
        ),
        **kwargs,
    )
    report = runtime_state.to_report()
    report["source"] = "prefill_page_table_runtime_state"
    return SimpleNamespace(
        page_table=page_table,
        tensor_conversion_count=1,
        prefill_page_table_runtime_state=report,
    )


def prefill_token_ids_tensor(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    token_ids: list[list[int]],
) -> Any:
    kwargs = {"device": device}
    dtype = getattr(
        ttnn,
        "uint32",
        getattr(ttnn, "int32", getattr(ttnn, "bfloat16", None)),
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    return ttnn.from_torch(
        runtime_int_tensor(torch, token_ids, name="prefill_prompt_token_ids"),
        **kwargs,
    )


def run_prefill_prompt(
    *,
    context: Any,
    ttnn: Any,
    device: Any,
    prefill_plan: dict[str, Any],
    layer_count: int,
) -> SimpleNamespace:
    prefill_start = time.perf_counter()
    prefill_token, kv_cache, cache_reports = (
        context.generated_model.prefill_prompt(
            context.prefill_token_ids,
            context.kv_cache,
            context.prefill_page_table,
            valid_seq_len=context.prefill_tokenization.get(
                "effective_token_count_by_user",
                context.prefill_tokenization["effective_token_count"],
            ),
        )
    )
    context.update_kv_cache(kv_cache)
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
    latency_ms = (time.perf_counter() - prefill_start) * 1000.0
    output_shapes = {
        "token": tensor_shape(prefill_token),
        "key_cache": tensor_shape(kv_cache[0].k),
        "value_cache": tensor_shape(kv_cache[0].v),
        "kv_cache_layers": [
            {
                "layer_id": layer_id,
                "key_cache": tensor_shape(layer_cache.k),
                "value_cache": tensor_shape(layer_cache.v),
            }
            for layer_id, layer_cache in enumerate(kv_cache[:layer_count])
        ],
    }
    cache_population = _observed_cache_population(
        plan=prefill_plan,
        cache_reports=cache_reports,
        output_shapes=output_shapes,
    )
    observed_ops = _generated_observed_op_sequence(
        context.generated_model,
        ttnn,
    )
    observed_ops_source = "runtime_instrumentation"
    if observed_ops is None:
        observed_ops = list(prefill_plan["op_sequence"])
        observed_ops_source = "generated_execution_plan"
    reference = _prefill_reference(
        plan=prefill_plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output={
            "kind": "token",
            "shape": tensor_shape(prefill_token),
            "dtype": tensor_dtype(prefill_token),
        },
        observed_ops=observed_ops,
    )
    reference["observed_ops_source"] = observed_ops_source
    first_token = prefill_token_direct_handoff(prefill_token=prefill_token)
    context.update_decode_token(first_token.token_ids)
    prompt_token_counts = [
        int(value)
        for value in context.prefill_tokenization.get(
            "effective_token_count_by_user",
            [context.prefill_tokenization["effective_token_count"]],
        )
    ]
    prefill_event = {
        "step_index": "prefill",
        "token": first_token.token_ids,
        "runtime_handoff": first_token.runtime_handoff,
        "runtime_host_roundtrip": first_token.runtime_host_roundtrip,
        "cache_position_value": (
            context.prefill_tokenization["effective_token_count"] - 1
        ),
        "token_shape": tensor_shape(first_token.token_ids),
    }
    if len(set(prompt_token_counts)) > 1:
        prefill_event["cache_position_value"] = None
        prefill_event["cache_position_values"] = [
            value - 1 for value in prompt_token_counts
        ]
    generated_token_events = [prefill_event]
    return SimpleNamespace(
        prefill_token=prefill_token,
        kv_cache=kv_cache,
        cache_reports=cache_reports,
        latency_ms=latency_ms,
        output_shapes=output_shapes,
        cache_population=cache_population,
        reference=reference,
        first_token=first_token,
        generated_token_events=generated_token_events,
    )
