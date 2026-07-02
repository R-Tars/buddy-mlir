"""Template registry helpers for Buddy-TTNN Direct."""

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
from .mlp_decode import official_gated_mlp_decode_op_sequence
from .lm_head import (
    build_lm_head_split_ranges,
    device_argmax_greedy_op_sequence,
    official_split_lm_head_op_sequence,
)

__all__ = [
    "build_execution_plan",
    "diff_plan_against_official",
    "dump_execution_plan",
    "dump_plan_diff",
    "expand_plan_ops",
    "load_execution_plan",
    "load_official_template",
    "load_template_config",
    "build_lm_head_split_ranges",
    "device_argmax_greedy_op_sequence",
    "official_gated_mlp_decode_op_sequence",
    "official_split_lm_head_op_sequence",
    "validate_template_config",
]
