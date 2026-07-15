"""Compatibility facade for the former template-owned TTNN wrappers."""

from ..ttnn_compat.errors import UnsupportedTTNNOp
from ..ttnn_compat.ops import (
    concat_heads_prefill,
    fill_cache,
    nlp_concat_heads_decode,
    nlp_create_qkv_heads_decode,
    paged_fill_cache,
    paged_fused_update_cache,
    paged_sdpa_decode,
    paged_update_cache,
    rotary_embedding_decode,
    rotary_embedding_fused_qk,
    rotary_embedding_prefill,
    scaled_dot_product_attention,
    split_qkv_heads_prefill,
)

__all__ = [
    "UnsupportedTTNNOp",
    "concat_heads_prefill",
    "fill_cache",
    "nlp_concat_heads_decode",
    "nlp_create_qkv_heads_decode",
    "paged_fill_cache",
    "paged_fused_update_cache",
    "paged_sdpa_decode",
    "paged_update_cache",
    "rotary_embedding_decode",
    "rotary_embedding_fused_qk",
    "rotary_embedding_prefill",
    "scaled_dot_product_attention",
    "split_qkv_heads_prefill",
]
