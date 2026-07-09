from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .inputs import build_decode_runtime_state
from ..smoke_decode_shell import _runtime_int_tensor
from ..smoke_single_layer_decode import _synthetic_tensor_factory


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
