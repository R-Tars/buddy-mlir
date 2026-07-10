"""Generated-model source fragments."""

from .attention import render_attention
from .constants import render_constants
from .core import render_model_core
from .lm_head import render_lm_head
from .mlp import render_mlp
from .preamble import render_preamble

__all__ = [
    "render_attention",
    "render_constants",
    "render_lm_head",
    "render_mlp",
    "render_model_core",
    "render_preamble",
]
