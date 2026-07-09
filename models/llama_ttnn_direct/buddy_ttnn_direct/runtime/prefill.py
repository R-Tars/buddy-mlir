from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from .decode import prefill_token_direct_handoff
from .inputs import build_decode_runtime_state
from ..smoke_decode_shell import _dtype, _runtime_int_tensor, _shape
from ..smoke_prefill import _observed_cache_population, _prefill_reference
from ..smoke_single_layer_decode import (
    _generated_observed_op_sequence,
    _synthetic_tensor_factory,
)


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
        _runtime_int_tensor(
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
        _runtime_int_tensor(torch, token_ids, name="prefill_prompt_token_ids"),
        **kwargs,
    )


def attach_prefill_rotary_parameters(
    *,
    parameters: Any,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    prefill_plan: dict[str, Any],
) -> SimpleNamespace:
    tensor, tensor_count = _synthetic_tensor_factory(
        ttnn=ttnn,
        torch=torch,
        device=device,
        dtype_seed=dtype_seed,
    )
    shapes = prefill_plan["layer_parameter_shapes"]
    for layer_id in range(int(prefill_plan["layers"])):
        layer = parameters.layers[layer_id]
        attention = getattr(layer, "attention", None)
        if attention is None:
            attention = SimpleNamespace()
            layer.attention = attention
        attention.rotary = SimpleNamespace(
            cos_matrix=tensor(
                shapes["rotary_cos_matrix"],
                name=f"prefill.layers.{layer_id}.rotary_cos",
            ),
            sin_matrix=tensor(
                shapes["rotary_sin_matrix"],
                name=f"prefill.layers.{layer_id}.rotary_sin",
            ),
            transformation_matrix=tensor(
                shapes["rotary_transformation_matrix"],
                name=f"prefill.layers.{layer_id}.rotary_transform",
            ),
        )
    return SimpleNamespace(tensor_conversion_count=tensor_count())


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
        )
    )
    context.update_kv_cache(kv_cache)
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(device)
    latency_ms = (time.perf_counter() - prefill_start) * 1000.0
    output_shapes = {
        "token": _shape(prefill_token),
        "key_cache": _shape(kv_cache[0].k),
        "value_cache": _shape(kv_cache[0].v),
        "kv_cache_layers": [
            {
                "layer_id": layer_id,
                "key_cache": _shape(layer_cache.k),
                "value_cache": _shape(layer_cache.v),
            }
            for layer_id, layer_cache in enumerate(kv_cache[:layer_count])
        ],
    }
    cache_population = _observed_cache_population(
        plan=prefill_plan,
        cache_reports=cache_reports,
        output_shapes=output_shapes,
    )
    reference = _prefill_reference(
        plan=prefill_plan,
        layer_count=layer_count,
        output_shapes=output_shapes,
        output={
            "kind": "token",
            "shape": _shape(prefill_token),
            "dtype": _dtype(prefill_token),
        },
        observed_ops=_generated_observed_op_sequence(
            context.generated_model,
            ttnn,
        ),
    )
    first_token = prefill_token_direct_handoff(prefill_token=prefill_token)
    context.update_decode_token(first_token.token_ids)
    generated_token_events = [
        {
            "step_index": "prefill",
            "token": first_token.token_ids,
            "runtime_handoff": first_token.runtime_handoff,
            "runtime_host_roundtrip": first_token.runtime_host_roundtrip,
            "cache_position_value": (
                context.prefill_tokenization["effective_token_count"] - 1
            ),
            "token_shape": _shape(first_token.token_ids),
        }
    ]
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
