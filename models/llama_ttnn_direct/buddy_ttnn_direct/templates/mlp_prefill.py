from __future__ import annotations


OFFICIAL_GATED_MLP_PREFILL_OPS = [
    "linear.mlp_gate",
    "linear.mlp_up",
    "mul.silu",
    "linear.mlp_down",
]


def official_gated_mlp_prefill_op_sequence() -> list[str]:
    return list(OFFICIAL_GATED_MLP_PREFILL_OPS)
