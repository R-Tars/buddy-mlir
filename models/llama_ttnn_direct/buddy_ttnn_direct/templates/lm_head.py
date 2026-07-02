from __future__ import annotations

from typing import Any


OFFICIAL_SPLIT_LM_HEAD_OPS = [
    "linear.lm_head_split",
    "concat.lm_head_logits",
]

DEVICE_ARGMAX_GREEDY_OPS = [
    "argmax.greedy",
]


def official_split_lm_head_op_sequence(split_count: int) -> list[str]:
    if split_count <= 0:
        raise ValueError("split_count must be positive")
    return [
        f"linear.lm_head_split[{shard_id}]"
        for shard_id in range(split_count)
    ] + ["concat.lm_head_logits"]


def device_argmax_greedy_op_sequence() -> list[str]:
    return list(DEVICE_ARGMAX_GREEDY_OPS)


def build_lm_head_split_ranges(
    vocab_size: int | None, split_count: int
) -> list[dict[str, Any]]:
    if split_count <= 0:
        raise ValueError("split_count must be positive")

    if vocab_size is None:
        return [
            {
                "shard_id": shard_id,
                "vocab_start": None,
                "vocab_end": None,
            }
            for shard_id in range(split_count)
        ]

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    base = vocab_size // split_count
    remainder = vocab_size % split_count
    ranges: list[dict[str, Any]] = []
    cursor = 0
    for shard_id in range(split_count):
        shard_size = base + (1 if shard_id < remainder else 0)
        next_cursor = cursor + shard_size
        ranges.append(
            {
                "shard_id": shard_id,
                "vocab_start": cursor,
                "vocab_end": next_cursor,
            }
        )
        cursor = next_cursor
    return ranges
