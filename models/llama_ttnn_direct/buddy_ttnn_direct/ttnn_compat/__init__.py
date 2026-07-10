"""TTNN API compatibility wrappers used by generated programs."""

from . import ops
from .errors import UnsupportedTTNNOp
from .model_ops import TTNNCompatOps
from .ops import (
    concat_heads_prefill,
    fill_cache,
    nlp_concat_heads_decode,
    nlp_create_qkv_heads_decode,
    paged_fill_cache,
    paged_sdpa_decode,
    paged_update_cache,
    rotary_embedding_decode,
    rotary_embedding_prefill,
    scaled_dot_product_attention,
    split_qkv_heads_prefill,
)

__all__ = [
    "TTNNCompatOps",
    "UnsupportedTTNNOp",
    "concat_heads_prefill",
    "fill_cache",
    "nlp_concat_heads_decode",
    "nlp_create_qkv_heads_decode",
    "ops",
    "paged_fill_cache",
    "paged_sdpa_decode",
    "paged_update_cache",
    "rotary_embedding_decode",
    "rotary_embedding_prefill",
    "scaled_dot_product_attention",
    "split_qkv_heads_prefill",
]
