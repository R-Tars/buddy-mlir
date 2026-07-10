from __future__ import annotations

import textwrap
from typing import Any

from .config import build_codegen_config
from .templates import (
    render_attention,
    render_constants,
    render_lm_head,
    render_mlp,
    render_model_core,
    render_preamble,
)


def render_python_ttnn_model(plan: dict[str, Any]) -> str:
    config = build_codegen_config(plan)
    rms_norm_eps = config["rms_norm"]["eps"]
    if rms_norm_eps is None:
        rms_norm_eps = 1e-5
    return "".join(
        (
            render_preamble(),
            render_model_core(),
            render_attention(),
            render_mlp(),
            render_lm_head(
                generation_template=config["generation"]["template"],
            ),
            render_constants(
                num_layers=int(config["num_layers"]),
                lm_head_split_count=int(config["lm_head"]["split_count"]),
                rms_norm_eps=float(rms_norm_eps),
            ),
        )
    )


def render_codegen_readme(plan: dict[str, Any]) -> str:
    config = build_codegen_config(plan)
    return textwrap.dedent(
        f"""
        # Buddy-TTNN Direct Generated Skeleton

        This directory was generated from a Buddy-TTNN Direct execution plan.
        The generated `model.py` defines the decode program structure and
        template method boundaries. Attention decode, MLP decode, and LM-head
        templates emit official-like TTNN calls through a small compatibility
        wrapper. Embedding and RMSNorm also route through `TTNNCompatOps` and
        raise `UnsupportedTTNNOp` if the installed TTNN module does not expose
        the required primitive. Attention primitive wrappers raise
        `UnsupportedTTNNOp` when the installed TTNN module lacks a required
        decode API.

        Model: `{config["model_name"]}`
        Layers: `{config["num_layers"]}`
        Mode: `{config["mode"]}`
        Batch size: `{config["batch_size"]}`
        Decode sequence length: `{config["seq_len"]}`

        Expected files:

        ```text
        model.py
        config.json
        plan.json
        README.md
        ```

        Validate syntax with:

        ```bash
        python -m py_compile model.py
        ```
        """
    ).lstrip()
