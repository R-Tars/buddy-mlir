"""Numerical correctness reference and comparison helpers."""

from .artifacts import tensor_snapshot
from .hf_reference import (
    capture_hf_reference,
    load_hf_reference,
    write_hf_reference,
)
from .metrics import compare_snapshots
from .performance_recipe import (
    greedy_agreement,
    load_official_performance_reference,
    token_accuracy,
)

__all__ = [
    "capture_hf_reference",
    "compare_snapshots",
    "greedy_agreement",
    "load_hf_reference",
    "load_official_performance_reference",
    "tensor_snapshot",
    "token_accuracy",
    "write_hf_reference",
]
