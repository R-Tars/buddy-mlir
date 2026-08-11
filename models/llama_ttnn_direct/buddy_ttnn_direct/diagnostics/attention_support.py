"""Shared, specialized support for isolated attention diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

from ..runtime.config_runtime import realize_ttnn_config
from ..runtime.inputs import build_decode_runtime_state
from ..runtime.plans import (
    decode_head_shape,
    decode_hidden_shape,
    linear_weight_shape,
)
from ..runtime.rotary import (
    decode_rotary_cos_sin_memory_config,
    decode_rotary_transform_memory_config,
)
from ..reports.contracts import ATTENTION_PRIMITIVES
from .support import (
    randn,
    runtime_index_tensor,
    ttnn_dtype,
    zeros,
)


def decode_rotary_cos_sin_shape(batch_size: int, head_dim: int) -> list[int]:
    return [1, batch_size, 1, head_dim]


def decode_rotary_transform_shape(batch_size: int, *, tile_size: int = 32) -> list[int]:
    return [1, 1, batch_size * tile_size, tile_size]


def memory_config(ttnn: Any) -> Any | None:
    return getattr(ttnn, "L1_MEMORY_CONFIG", None)


def height_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    head_dim: int,
) -> Any | None:
    return sharded_height_memory_config(
        ttnn, device, batch_size=batch_size, shard_shape=(int(getattr(ttnn, "TILE_SIZE", 32)), head_dim)
    ) or getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", memory_config(ttnn))


def rotary_cos_sin_height_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    head_dim: int,
) -> Any | None:
    return decode_rotary_cos_sin_memory_config(ttnn, device, batch_size=batch_size, head_dim=head_dim) or height_sharded_memory_config(
        ttnn, device, batch_size=batch_size, head_dim=head_dim
    )


def rotary_transform_height_sharded_memory_config(
    ttnn: Any, device: Any, *, batch_size: int
) -> Any | None:
    return decode_rotary_transform_memory_config(ttnn, device, batch_size=batch_size) or getattr(
        ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", memory_config(ttnn)
    )


def sharded_height_memory_config(
    ttnn: Any,
    device: Any,
    *,
    batch_size: int,
    shard_shape: tuple[int, int],
) -> Any | None:
    create = getattr(ttnn, "create_sharded_memory_config", None)
    grid_type = getattr(ttnn, "CoreGrid", None)
    strategy = getattr(getattr(ttnn, "ShardStrategy", None), "HEIGHT", None)
    orientation = getattr(getattr(ttnn, "ShardOrientation", None), "ROW_MAJOR", None)
    if not callable(create) or not callable(grid_type) or strategy is None:
        return None
    core_grid = batch_core_grid(ttnn, device, batch_size=batch_size)
    if core_grid is None:
        return None
    try:
        return create(
            shape=shard_shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=orientation,
            use_height_and_width_as_shard_shape=True,
        )
    except Exception:
        return None


def batch_core_grid(ttnn: Any, device: Any, *, batch_size: int) -> Any | None:
    grid_type = getattr(ttnn, "CoreGrid", None)
    if not callable(grid_type):
        return None
    size_fn = getattr(device, "compute_with_storage_grid_size", None)
    try:
        size = size_fn() if callable(size_fn) else None
    except Exception:
        size = None
    grid_x = max(1, min(batch_size, int(getattr(size, "x", 8) or 8)))
    while grid_x > 1 and batch_size % grid_x:
        grid_x -= 1
    grid_y = max(1, (batch_size + grid_x - 1) // grid_x)
    if grid_y > int(getattr(size, "y", 8) or 8):
        return None
    try:
        return grid_type(y=grid_y, x=grid_x)
    except TypeError:
        return grid_type(grid_y, grid_x)


def page_state_from_plan(plan: dict[str, Any]) -> Any | None:
    page_table = plan["input_shapes"].get("page_table")
    positions = plan["input_shapes"].get("cache_position")
    if not page_table or not positions:
        return None
    return build_decode_runtime_state(
        batch_size=int(page_table[0]),
        cache_len=int(page_table[1]) * int(plan["page_block_size"]),
        page_block_size=int(plan["page_block_size"]),
        prompt_token_count=1,
    )


def realize_sdpa_runtime_config(
    value: Mapping[str, Any] | None, ttnn: Any
) -> dict[str, Any]:
    if value is None:
        return {}
    allowed = {"program_config", "kernel_output_memory_config", "post_sdpa_output_memory_config"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown SDPA runtime config fields: {sorted(unknown)}")
    result: dict[str, Any] = {}
    for key, descriptor in value.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"SDPA runtime config {key} must be a descriptor")
        result[key] = realize_ttnn_config(dict(descriptor), ttnn)
    return result


def input_tensor_contracts(
    primitive: str, *, input_shapes: dict[str, list[int]]
) -> dict[str, dict[str, str]]:
    contracts = {
        name: {"dtype": "bfloat16_or_float32", "layout": "tile", "memory_config": "default_or_l1"}
        for name in input_shapes
    }
    for name in ("page_table", "cache_position"):
        if name in contracts:
            contracts[name] = {"dtype": "int32", "layout": "row_major", "memory_config": "default_or_dram"}
    height_inputs = {
        "rotary_embedding_decode": {"query", "key"},
        "paged_update_cache": {"update"},
        "paged_scaled_dot_product_attention_decode": {"query"},
        "nlp_concat_heads_decode": {"attention"},
    }.get(primitive, set())
    for name in height_inputs:
        if name in contracts:
            contracts[name]["memory_config"] = "height_sharded_l1"
    if primitive == "rotary_embedding_decode":
        for name in ("cos_matrix", "sin_matrix"):
            if name in contracts:
                contracts[name]["memory_config"] = "rotary_cos_sin_height_sharded_l1"
        if "transformation_matrix" in contracts:
            contracts["transformation_matrix"]["memory_config"] = "rotary_transform_height_sharded_l1"
    for name in ("cache", "key_cache", "value_cache"):
        if name in contracts:
            contracts[name]["memory_config"] = "dram"
    return contracts


def without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def validate_args(**kwargs: Any) -> None:
    primitive = kwargs["primitive"]
    if primitive not in ATTENTION_PRIMITIVES:
        raise ValueError(f"primitive must be one of {list(ATTENTION_PRIMITIVES)}")
    for name in ("batch_size", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "max_cache_len"):
        if kwargs[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if kwargs["num_heads"] % kwargs["num_kv_heads"]:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if kwargs["hidden_size"] != kwargs["num_heads"] * kwargs["head_dim"]:
        raise ValueError("hidden_size must equal num_heads * head_dim")
    if kwargs["dtype_seed"] not in {"bf16", "fp32"}:
        raise ValueError("dtype_seed must be one of: bf16, fp32")
