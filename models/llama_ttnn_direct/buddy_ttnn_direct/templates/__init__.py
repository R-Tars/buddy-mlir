"""Template registry helpers for Buddy-TTNN Direct."""

from . import ttnn_ops
from .registry import (
    build_execution_plan,
    dump_execution_plan,
    load_execution_plan,
    load_template_config,
    validate_template_config,
)
from .diff import (
    diff_plan_against_official,
    dump_plan_diff,
    expand_plan_ops,
    load_official_template,
)
from .attention_decode import (
    dump_attention_decode_op_sequence,
    official_paged_attention_decode_op_sequence,
)
from .mlp_decode import official_gated_mlp_decode_op_sequence
from .lm_head import (
    build_lm_head_split_ranges,
    device_argmax_greedy_op_sequence,
    official_split_lm_head_op_sequence,
)
from .ttnn_ops import (
    UnsupportedTTNNOp,
    nlp_concat_heads_decode,
    nlp_create_qkv_heads_decode,
    paged_sdpa_decode,
    paged_update_cache,
    rotary_embedding_decode,
)

__all__ = [
    "build_execution_plan",
    "diff_plan_against_official",
    "dump_attention_decode_op_sequence",
    "dump_execution_plan",
    "dump_plan_diff",
    "expand_plan_ops",
    "load_execution_plan",
    "load_official_template",
    "load_template_config",
    "build_lm_head_split_ranges",
    "device_argmax_greedy_op_sequence",
    "official_gated_mlp_decode_op_sequence",
    "official_paged_attention_decode_op_sequence",
    "official_split_lm_head_op_sequence",
    "UnsupportedTTNNOp",
    "nlp_concat_heads_decode",
    "nlp_create_qkv_heads_decode",
    "paged_sdpa_decode",
    "paged_update_cache",
    "rotary_embedding_decode",
    "ttnn_ops",
    "validate_template_config",
]
