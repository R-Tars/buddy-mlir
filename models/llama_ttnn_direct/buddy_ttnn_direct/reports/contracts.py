from __future__ import annotations


ATTENTION_PRIMITIVES = (
    "qkv_linear",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "o_proj_linear",
)

ATTENTION_LAYER_OPS = [
    "qkv_linear",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache.k",
    "paged_update_cache.v",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "o_proj_linear",
]

PROFILE_SECTION_LATENCY_KEYS = (
    "embedding_ms",
    "final_norm_ms",
    "lm_head_ms",
    "argmax_ms",
    "host_copy_ms",
)

PROFILE_LAYER_LATENCY_KEYS = (
    "rms_norm_attn_ms",
    "attention_ms",
    "residual_add_attn_ms",
    "rms_norm_mlp_ms",
    "mlp_ms",
    "residual_add_mlp_ms",
    "total_ms",
)
