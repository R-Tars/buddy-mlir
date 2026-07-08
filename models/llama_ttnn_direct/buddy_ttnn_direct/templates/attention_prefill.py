from __future__ import annotations


OFFICIAL_PREFILL_ATTENTION_OPS = [
    "linear.qkv_packed",
    "split_query_key_value_heads_prefill",
    "rotary_embedding_prefill",
    "scaled_dot_product_attention",
    "fill_cache.k",
    "fill_cache.v",
    "concat_heads_prefill",
    "linear.o_proj",
]


def official_prefill_attention_op_sequence() -> list[str]:
    return list(OFFICIAL_PREFILL_ATTENTION_OPS)
