from __future__ import annotations

from typing import Any


DECODE_LAYER_OPS = [
    "rms_norm.attn",
    "qkv_linear",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache.k",
    "paged_update_cache.v",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "o_proj_linear",
    "residual_add.attn",
    "rms_norm.mlp",
    "mlp_gate",
    "mlp_up",
    "mul_silu",
    "mlp_down",
    "residual_add.mlp",
]
DECODE_FINAL_OPS = ["rms_norm.final", "split_lm_head", "argmax_or_sampling"]
DECODE_FINAL_LOGITS_OPS = ["rms_norm.final", "split_lm_head"]
DECODE_PARAMETER_ROLES = ["embedding", "norm", "attention", "mlp", "lm_head"]
PREFILL_LAYER_OPS = [
    "rms_norm.attn",
    "qkv_linear",
    "split_query_key_value_heads_prefill",
    "rotary_embedding_prefill",
    "scaled_dot_product_attention",
    "fill_cache.k",
    "fill_cache.v",
    "concat_heads_prefill",
    "o_proj_linear",
    "residual_add.attn",
    "rms_norm.mlp",
    "mlp_gate",
    "mlp_up",
    "mul_silu",
    "mlp_down",
    "residual_add.mlp",
]
PREFILL_FINAL_OPS = ["rms_norm.final", "split_lm_head", "argmax_or_sampling"]


def decode_step_plan(
    *, layers: int, batch_size: int, cache_len: int, config: dict[str, Any]
) -> dict[str, Any]:
    dimensions = _dimensions(config)
    hidden_size = dimensions["hidden_size"]
    intermediate_size = dimensions["intermediate_size"]
    num_heads = dimensions["num_heads"]
    num_kv_heads = dimensions["num_kv_heads"]
    head_dim = dimensions["head_dim"]
    vocab_size = dimensions["vocab_size"]
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    kv = _kv_shapes(config, batch_size, cache_len, num_kv_heads, head_dim)
    lm_head_splits = lm_head_split_shapes(config, hidden_size, vocab_size)
    output_kind = decode_output_kind(config)
    expected_decode_output = (
        [batch_size, 1, vocab_size]
        if output_kind == "logits"
        else [batch_size, 1]
    )
    input_shapes = {
        "token_ids": [batch_size, 1],
        "page_table": [batch_size, kv["page_count"]],
        "cache_position": [batch_size],
        "key_cache": kv["physical_shape"],
        "value_cache": kv["physical_shape"],
    }
    layer_parameter_shapes = _layer_parameter_shapes(
        batch_size=batch_size,
        sequence_len=None,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qkv_size=qkv_size,
    )
    return {
        "layers": layers,
        "vocab_size": vocab_size,
        "input_shapes": input_shapes,
        "parameter_shapes": {
            "embedding": embedding_weight_shape(vocab_size, hidden_size),
            **layer_parameter_shapes,
            "final_norm": norm_weight_shape(hidden_size),
            "lm_head_splits": lm_head_splits,
        },
        "layer_parameter_shapes": layer_parameter_shapes,
        "rotary": dict(config.get("rotary") or {}),
        "expected_intermediate_shapes": {
            "embedding": decode_hidden_shape(batch_size, hidden_size),
            "qkv": decode_hidden_shape(batch_size, qkv_size),
            "query": decode_head_shape(batch_size, num_heads, head_dim),
            "key": decode_head_shape(batch_size, num_kv_heads, head_dim),
            "value": decode_head_shape(batch_size, num_kv_heads, head_dim),
            "attention": decode_head_shape(batch_size, num_heads, head_dim),
            "concat_heads": decode_hidden_shape(batch_size, num_heads * head_dim),
            "attention_output": decode_hidden_shape(batch_size, hidden_size),
            "mlp_intermediate": decode_hidden_shape(batch_size, intermediate_size),
        },
        "expected_output_shapes": {
            output_kind: expected_decode_output,
            "key_cache": input_shapes["key_cache"],
            "value_cache": input_shapes["value_cache"],
        },
        "output_kind": output_kind,
        "kv_cache": kv,
        "tensor_conversion_count": 5 + len(lm_head_splits) + 12 * layers,
        "op_sequence": decode_op_sequence(layers, output_kind=output_kind),
    }


def prefill_plan(
    *,
    layers: int,
    batch_size: int,
    prefill_len: int,
    cache_len: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    dimensions = _dimensions(config)
    hidden_size = dimensions["hidden_size"]
    intermediate_size = dimensions["intermediate_size"]
    num_heads = dimensions["num_heads"]
    num_kv_heads = dimensions["num_kv_heads"]
    head_dim = dimensions["head_dim"]
    vocab_size = dimensions["vocab_size"]
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    kv = _kv_shapes(config, batch_size, cache_len, num_kv_heads, head_dim)
    lm_head_splits = lm_head_split_shapes(config, hidden_size, vocab_size)
    layer_parameter_shapes = _layer_parameter_shapes(
        batch_size=batch_size,
        sequence_len=prefill_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qkv_size=qkv_size,
    )
    input_shapes = {
        "token_ids": [batch_size, prefill_len],
        "key_cache": kv["physical_shape"],
        "value_cache": kv["physical_shape"],
    }
    kv.update(
        {
            "source": "prefill",
            "write_policy": "fill_cache_per_user",
            "planned_user_count": batch_size,
        }
    )
    return {
        "layers": layers,
        "batch_size": batch_size,
        "prefill_len": prefill_len,
        "cache_len": cache_len,
        "vocab_size": vocab_size,
        "input_shapes": input_shapes,
        "parameter_shapes": {
            "embedding": embedding_weight_shape(vocab_size, hidden_size),
            **layer_parameter_shapes,
            "final_norm": norm_weight_shape(hidden_size),
            "lm_head_splits": lm_head_splits,
        },
        "layer_parameter_shapes": layer_parameter_shapes,
        "rotary": dict(config.get("rotary") or {}),
        "expected_intermediate_shapes": {
            "embedding": [batch_size, prefill_len, hidden_size],
            "qkv": [batch_size, prefill_len, qkv_size],
            "query": [batch_size, num_heads, prefill_len, head_dim],
            "key": [batch_size, num_kv_heads, prefill_len, head_dim],
            "value": [batch_size, num_kv_heads, prefill_len, head_dim],
            "attention": [batch_size, num_heads, prefill_len, head_dim],
            "concat_heads": [batch_size, prefill_len, num_heads * head_dim],
            "attention_output": [batch_size, prefill_len, hidden_size],
            "mlp_intermediate": [batch_size, prefill_len, intermediate_size],
        },
        "expected_output_shapes": {
            "token": [batch_size, 1],
            "key_cache": kv["physical_shape"],
            "value_cache": kv["physical_shape"],
        },
        "kv_cache": kv,
        "tensor_conversion_count": 4 + len(lm_head_splits) + 12 * layers,
        "op_sequence": prefill_op_sequence(layers),
    }


def decode_op_sequence(layers: int, *, output_kind: str = "token") -> list[str]:
    ops = ["embedding"]
    for _ in range(layers):
        ops.extend(DECODE_LAYER_OPS)
    ops.extend(DECODE_FINAL_LOGITS_OPS if output_kind == "logits" else DECODE_FINAL_OPS)
    return ops


def prefill_op_sequence(layers: int) -> list[str]:
    ops = ["embedding"]
    for _ in range(layers):
        ops.extend(PREFILL_LAYER_OPS)
    ops.extend(PREFILL_FINAL_OPS)
    return ops


def decode_output_kind(config: dict[str, Any]) -> str:
    lm_head = config.get("lm_head") if isinstance(config.get("lm_head"), dict) else {}
    generation = (
        config.get("generation")
        if isinstance(config.get("generation"), dict)
        else {}
    )
    if lm_head.get("retain_logits") or generation.get("retain_logits"):
        return "logits"
    if generation.get("mode") == "full_logits" or generation.get("template") == "full_logits":
        return "logits"
    return "token"


def lm_head_split_shapes(
    config: dict[str, Any], hidden_size: int, vocab_size: int
) -> list[list[int]]:
    lm_head = config.get("lm_head", {})
    split_configs = lm_head.get("splits")
    if split_configs:
        return [
            linear_weight_shape(
                hidden_size,
                int(split["vocab_end"]) - int(split["vocab_start"]),
            )
            for split in split_configs
        ]
    split_count = int(lm_head.get("split_count", 1))
    base, remainder = divmod(vocab_size, split_count)
    return [
        linear_weight_shape(hidden_size, base + (shard_id < remainder))
        for shard_id in range(split_count)
    ]


def decode_hidden_shape(batch_size: int, hidden_size: int) -> list[int]:
    return [1, 1, batch_size, hidden_size]


def decode_head_shape(batch_size: int, num_heads: int, head_dim: int) -> list[int]:
    return [1, batch_size, num_heads, head_dim]


def linear_weight_shape(in_features: int, out_features: int) -> list[int]:
    return [1, 1, in_features, out_features]


def embedding_weight_shape(vocab_size: int, hidden_size: int) -> list[int]:
    return [1, 1, vocab_size, hidden_size]


def norm_weight_shape(hidden_size: int) -> list[int]:
    return [1, 1, hidden_size // 32, 32] if hidden_size % 32 == 0 else [1, 1, 1, hidden_size]


def _dimensions(config: dict[str, Any]) -> dict[str, int]:
    return {
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_heads": int(config["num_attention_heads"]),
        "num_kv_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["head_dim"]),
        "vocab_size": int(config["vocab_size"]),
    }


def _kv_shapes(
    config: dict[str, Any],
    batch_size: int,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    kv_config = config.get("kv_cache") if isinstance(config.get("kv_cache"), dict) else {}
    page_block_size = int(kv_config.get("page_block_size", 32))
    page_count = max(1, (cache_len + page_block_size - 1) // page_block_size)
    max_num_blocks = batch_size * page_count
    physical_shape = [max_num_blocks, num_kv_heads, page_block_size, head_dim]
    return {
        "policy": kv_config.get("policy", "paged"),
        "template": kv_config.get("template", "paged_kv_cache"),
        "page_block_size": page_block_size,
        "page_count": page_count,
        "max_num_blocks": max_num_blocks,
        "physical_shape": physical_shape,
        "logical_shape": [batch_size, cache_len, num_kv_heads, head_dim],
    }


def _layer_parameter_shapes(
    *,
    batch_size: int,
    sequence_len: int | None,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qkv_size: int,
) -> dict[str, list[int]]:
    if sequence_len is None:
        cos_sin = [1, batch_size, 1, head_dim]
        transform = [1, 1, batch_size * 32, 32]
    else:
        cos_sin = [1, 1, sequence_len, head_dim]
        transform = [1, 1, 32, 32]
    return {
        "input_norm": norm_weight_shape(hidden_size),
        "post_attention_norm": norm_weight_shape(hidden_size),
        "attention_wqkv": linear_weight_shape(hidden_size, qkv_size),
        "attention_o_proj": linear_weight_shape(num_heads * head_dim, hidden_size),
        "rotary_cos_matrix": cos_sin,
        "rotary_sin_matrix": cos_sin,
        "rotary_transformation_matrix": transform,
        "mlp_gate": linear_weight_shape(hidden_size, intermediate_size),
        "mlp_up": linear_weight_shape(hidden_size, intermediate_size),
        "mlp_down": linear_weight_shape(intermediate_size, hidden_size),
    }
