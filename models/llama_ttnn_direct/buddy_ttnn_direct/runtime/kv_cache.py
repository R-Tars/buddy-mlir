from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .inputs import build_decode_kv_cache_runtime_state


def build_prompt_decode_kv_cache_tensors(
    *,
    ttnn: Any,
    torch: Any,
    device: Any,
    dtype_seed: str,
    layer_count: int,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> SimpleNamespace:
    runtime_state = build_decode_kv_cache_runtime_state(
        layer_count=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    kwargs = {
        "device": device,
        "dtype": _ttnn_dtype(ttnn, dtype_seed),
    }
    memory_config = _runtime_dram_memory_config(ttnn)
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    layout = getattr(ttnn, "TILE_LAYOUT", None)
    if layout is not None:
        kwargs["layout"] = layout
    tensor_count = 0

    def cache_tensor(name: str) -> Any:
        nonlocal tensor_count
        tensor_count += 1
        return ttnn.from_torch(
            _runtime_float_tensor(
                torch,
                runtime_state.physical_shape,
                dtype_seed=dtype_seed,
                name=name,
            ),
            **kwargs,
        )

    kv_cache = []
    for layer_id in range(layer_count):
        kv_cache.append(
            SimpleNamespace(
                k=cache_tensor(f"runtime.layers.{layer_id}.key_cache"),
                v=cache_tensor(f"runtime.layers.{layer_id}.value_cache"),
            )
        )

    kv_cache_runtime_state = runtime_state.to_report()
    kv_cache_runtime_state["memory_config"] = "dram"
    kv_cache_runtime_state["ttnn_memory_config"] = _config_repr(memory_config)
    return SimpleNamespace(
        kv_cache=kv_cache,
        tensor_conversion_count=tensor_count,
        kv_cache_runtime_state=kv_cache_runtime_state,
    )


def _ttnn_dtype(ttnn: Any, dtype_seed: str) -> Any:
    if dtype_seed == "bf16":
        return getattr(ttnn, "bfloat16", None)
    return getattr(ttnn, "float32", None)


def _runtime_dram_memory_config(ttnn: Any) -> Any | None:
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def _config_repr(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _runtime_float_tensor(
    torch: Any,
    shape: list[int],
    *,
    dtype_seed: str,
    name: str,
) -> Any:
    zeros = getattr(torch, "zeros", None)
    if not callable(zeros):
        raise ValueError("torch module must provide zeros")
    dtype = (
        getattr(torch, "bfloat16", None)
        if dtype_seed == "bf16"
        else getattr(torch, "float32", None)
    )
    try:
        tensor = zeros(tuple(shape), dtype=dtype)
    except TypeError:
        tensor = zeros(tuple(shape))
    try:
        tensor.name = name
    except AttributeError:
        pass
    return tensor
