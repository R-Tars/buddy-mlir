from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DecodeRuntimeState:
    batch_size: int
    cache_len: int
    page_block_size: int
    page_count: int
    max_num_blocks: int
    cache_position_value: int | None
    cache_position_values: list[int]
    page_table: list[list[int]]
    cache_position: list[int]

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "built",
            "source": "decode_runtime_state",
            "batch_size": self.batch_size,
            "cache_len": self.cache_len,
            "page_block_size": self.page_block_size,
            "page_count": self.page_count,
            "max_num_blocks": self.max_num_blocks,
            "cache_position_value": self.cache_position_value,
            "cache_position_values": self.cache_position_values,
            "page_table_shape": [self.batch_size, self.page_count],
            "cache_position_shape": [self.batch_size],
        }


@dataclass(frozen=True)
class DecodeRotaryRuntimeState:
    layer_count: int
    batch_size: int
    head_dim: int
    cache_position_value: int
    cos_sin_shape: list[int]
    transformation_shape: list[int]
    tensor_count: int

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "built",
            "source": "rotary_runtime_state",
            "layer_count": self.layer_count,
            "batch_size": self.batch_size,
            "head_dim": self.head_dim,
            "cache_position_value": self.cache_position_value,
            "matrix_shape": list(self.cos_sin_shape),
            "cos_sin_shape": list(self.cos_sin_shape),
            "transformation_shape": list(self.transformation_shape),
            "tensors_per_layer": 3,
            "tensor_count": self.tensor_count,
            "tensor_roles": [
                "cos_matrix",
                "sin_matrix",
                "transformation_matrix",
            ],
        }


@dataclass(frozen=True)
class DecodeKVCacheRuntimeState:
    layer_count: int
    batch_size: int
    cache_len: int
    page_block_size: int
    page_count: int
    max_num_blocks: int
    num_kv_heads: int
    head_dim: int
    physical_shape: list[int]
    logical_shape: list[int]
    tensor_count: int

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "built",
            "source": "kv_cache_runtime_state",
            "layer_count": self.layer_count,
            "batch_size": self.batch_size,
            "cache_len": self.cache_len,
            "page_block_size": self.page_block_size,
            "page_count": self.page_count,
            "max_num_blocks": self.max_num_blocks,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "physical_shape": list(self.physical_shape),
            "logical_shape": list(self.logical_shape),
            "tensors_per_layer": 2,
            "tensor_count": self.tensor_count,
            "tensor_roles": ["key_cache", "value_cache"],
        }


def build_decode_runtime_state(
    *,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    prompt_token_count: int | Sequence[int],
) -> DecodeRuntimeState:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cache_len <= 0:
        raise ValueError("cache_len must be positive")
    if page_block_size <= 0:
        raise ValueError("page_block_size must be positive")
    if isinstance(prompt_token_count, int):
        prompt_token_counts = [prompt_token_count for _ in range(batch_size)]
    else:
        prompt_token_counts = [int(value) for value in prompt_token_count]
    if len(prompt_token_counts) != batch_size:
        raise ValueError(
            "prompt_token_count sequence length must match batch_size"
        )
    if any(value <= 0 for value in prompt_token_counts):
        raise ValueError("prompt_token_count values must be positive")

    page_count = max(1, math.ceil(cache_len / page_block_size))
    max_num_blocks = batch_size * page_count
    cache_position_values = [
        min(max(prompt_count - 1, 0), cache_len - 1)
        for prompt_count in prompt_token_counts
    ]
    cache_position_value = (
        cache_position_values[0]
        if len(set(cache_position_values)) == 1
        else None
    )
    page_table = [
        [batch_id * page_count + page_id for page_id in range(page_count)]
        for batch_id in range(batch_size)
    ]
    return DecodeRuntimeState(
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        page_count=page_count,
        max_num_blocks=max_num_blocks,
        cache_position_value=cache_position_value,
        cache_position_values=cache_position_values,
        page_table=page_table,
        cache_position=cache_position_values,
    )


def build_decode_rotary_runtime_state(
    *,
    layer_count: int,
    batch_size: int,
    head_dim: int,
    cache_position_value: int,
) -> DecodeRotaryRuntimeState:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if cache_position_value < 0:
        raise ValueError("cache_position_value must be non-negative")

    return DecodeRotaryRuntimeState(
        layer_count=layer_count,
        batch_size=batch_size,
        head_dim=head_dim,
        cache_position_value=cache_position_value,
        cos_sin_shape=[1, batch_size, 1, head_dim],
        transformation_shape=[1, 1, batch_size * 32, 32],
        tensor_count=3 * layer_count,
    )


def build_decode_kv_cache_runtime_state(
    *,
    layer_count: int,
    batch_size: int,
    cache_len: int,
    page_block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> DecodeKVCacheRuntimeState:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cache_len <= 0:
        raise ValueError("cache_len must be positive")
    if page_block_size <= 0:
        raise ValueError("page_block_size must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")

    page_count = max(1, math.ceil(cache_len / page_block_size))
    max_num_blocks = batch_size * page_count
    physical_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    logical_shape = [batch_size, cache_len, num_kv_heads, head_dim]
    return DecodeKVCacheRuntimeState(
        layer_count=layer_count,
        batch_size=batch_size,
        cache_len=cache_len,
        page_block_size=page_block_size,
        page_count=page_count,
        max_num_blocks=max_num_blocks,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        physical_shape=physical_shape,
        logical_shape=logical_shape,
        tensor_count=2 * layer_count,
    )
