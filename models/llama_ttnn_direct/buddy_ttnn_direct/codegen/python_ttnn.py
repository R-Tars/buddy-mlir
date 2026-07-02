from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path
from typing import Any

from .artifacts import (
    ensure_output_dir,
    planned_artifact_paths,
    write_json,
    write_text,
)
from ..templates.attention_decode import (
    official_paged_attention_decode_op_sequence,
)
from ..templates.lm_head import build_lm_head_split_ranges
from ..templates.registry import find_custom_fused_templates


class CustomFusedRegionNotImplemented(NotImplementedError):
    """Raised when codegen encounters a reserved custom fused template."""


def validate_execution_plan_for_codegen(plan: dict[str, Any]) -> None:
    errors: list[str] = []
    custom_templates = find_custom_fused_templates(plan)
    if custom_templates:
        joined = ", ".join(custom_templates)
        raise CustomFusedRegionNotImplemented(
            "CustomFusedRegionNotImplemented: codegen for reserved custom "
            f"fused template(s) is not implemented: {joined}"
        )
    if plan.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if plan.get("mode") != "decode":
        errors.append("Phase 3 codegen only supports decode plans")
    layers = plan.get("layers")
    if not isinstance(layers, list) or not layers:
        errors.append("plan must contain a non-empty layers list")
    else:
        for expected_id, layer in enumerate(layers):
            if layer.get("layer_id") != expected_id:
                errors.append(
                    f"layer_id mismatch at index {expected_id}: "
                    f"{layer.get('layer_id')}"
                )
            templates = layer.get("templates")
            if not isinstance(templates, list) or not templates:
                errors.append(f"layer {expected_id} has no templates")
    final = plan.get("final")
    if not isinstance(final, list) or not final:
        errors.append("plan must contain a non-empty final list")

    if errors:
        raise ValueError(
            "invalid execution plan for Python TTNN codegen:\n- "
            + "\n- ".join(errors)
        )


def build_codegen_config(plan: dict[str, Any]) -> dict[str, Any]:
    validate_execution_plan_for_codegen(plan)
    template_config = copy.deepcopy(plan["template_config"])
    lm_head_split_count = int(template_config["lm_head_split_count"])
    vocab_size = plan.get("vocab_size")
    head_dim = plan.get("head_dim")
    attention_scale = None
    if head_dim:
        attention_scale = float(head_dim) ** -0.5
    generation_template = (
        "device_argmax_greedy"
        if "device_argmax_greedy" in plan["final"]
        else "full_logits"
    )
    retain_logits = generation_template != "device_argmax_greedy"
    return {
        "schema_version": 1,
        "model_name": plan["model_name"],
        "num_layers": len(plan["layers"]),
        "mode": plan["mode"],
        "batch_size": plan["batch_size"],
        "seq_len": plan["seq_len"],
        "max_cache_len": plan["max_cache_len"],
        "hidden_size": plan.get("hidden_size"),
        "intermediate_size": plan.get("intermediate_size"),
        "num_attention_heads": plan.get("num_attention_heads"),
        "num_key_value_heads": plan.get("num_key_value_heads"),
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "template_config": template_config,
        "layers": copy.deepcopy(plan["layers"]),
        "final": copy.deepcopy(plan["final"]),
        "attention": {
            "template": "official_paged_attention_decode",
            "op_sequence": official_paged_attention_decode_op_sequence(),
            "scale": attention_scale,
            "qkv_output_memory_config": None,
            "qkv_program_config": None,
            "qkv_compute_kernel_config": None,
            "qkv_output_dtype": None,
            "qkv_heads_memory_config": None,
            "sdpa_output_memory_config": None,
            "sdpa_program_config": None,
            "sdpa_compute_kernel_config": None,
            "concat_heads_output_memory_config": None,
            "o_proj_output_memory_config": None,
            "o_proj_program_config": None,
            "o_proj_compute_kernel_config": None,
            "o_proj_output_dtype": None,
        },
        "lm_head": {
            "template": "official_split_lm_head",
            "split_count": lm_head_split_count,
            "split_axis": "vocab",
            "retain_logits": retain_logits,
            "output_memory_config": None,
            "concat_memory_config": None,
            "output_dtype": None,
            "compute_kernel_config": None,
            "program_configs": [None] * lm_head_split_count,
            "splits": build_lm_head_split_ranges(
                vocab_size, lm_head_split_count
            ),
        },
        "generation": {
            "template": generation_template,
            "mode": (
                "greedy"
                if generation_template == "device_argmax_greedy"
                else "full_logits"
            ),
            "retain_logits": retain_logits,
        },
    }


def render_python_ttnn_model(plan: dict[str, Any]) -> str:
    config = build_codegen_config(plan)
    model_class = "BuddyLlama31TTNN"
    num_layers = int(config["num_layers"])
    attention_template = _template_name(
        plan, "official_paged_attention_decode"
    )
    mlp_template = _template_name(plan, "official_gated_mlp_decode")
    lm_head_template = _template_name(plan, "official_split_lm_head")
    generation_template = config["generation"]["template"]
    lm_head_split_count = int(config["lm_head"]["split_count"])

    return (
        textwrap.dedent(
            f'''
            # Auto-generated by Buddy-TTNN Direct.
            from __future__ import annotations

            import json
            from types import SimpleNamespace

            import ttnn
            from models.llama_ttnn_direct.buddy_ttnn_direct.templates import ttnn_ops


            def _to_namespace(value):
                if isinstance(value, dict):
                    return SimpleNamespace(
                        **{{key: _to_namespace(item) for key, item in value.items()}}
                    )
                if isinstance(value, list):
                    return [_to_namespace(item) for item in value]
                return value


            def load_config(path):
                with open(path) as handle:
                    return _to_namespace(json.load(handle))


            def _optional_attr(obj, name, default=None):
                return getattr(obj, name, default)


            def _tensor_memory_config(tensor):
                memory_config = getattr(tensor, "memory_config", None)
                if callable(memory_config):
                    return memory_config()
                return None


            class TTNNCompatOps:
                def __init__(self, ttnn_module):
                    self.ttnn = ttnn_module

                def add(self, left, right, *, memory_config=None, dtype=None):
                    kwargs = {{}}
                    if memory_config is not None:
                        kwargs["memory_config"] = memory_config
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    return self.ttnn.add(left, right, **kwargs)

                def linear(
                    self,
                    activation,
                    weight,
                    *,
                    memory_config=None,
                    program_config=None,
                    compute_kernel_config=None,
                    dtype=None,
                ):
                    kwargs = {{}}
                    if memory_config is not None:
                        kwargs["memory_config"] = memory_config
                    if program_config is not None:
                        kwargs["program_config"] = program_config
                    if compute_kernel_config is not None:
                        kwargs["compute_kernel_config"] = compute_kernel_config
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    return self.ttnn.linear(activation, weight, **kwargs)

                def mul_silu(
                    self,
                    gate,
                    up,
                    *,
                    memory_config=None,
                    dtype=None,
                ):
                    kwargs = {{}}
                    activation = self._silu_activation()
                    if activation is not None:
                        kwargs["input_tensor_a_activations"] = [activation]
                    if memory_config is not None:
                        kwargs["memory_config"] = memory_config
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    mul = getattr(self.ttnn, "mul", None)
                    if mul is None:
                        mul = getattr(self.ttnn, "multiply", None)
                    if mul is None:
                        raise AttributeError("ttnn must provide mul or multiply")
                    return mul(gate, up, **kwargs)

                def _silu_activation(self):
                    unary_with_param = getattr(self.ttnn, "UnaryWithParam", None)
                    unary_op_type = getattr(self.ttnn, "UnaryOpType", None)
                    if unary_with_param is None or unary_op_type is None:
                        return None
                    silu = getattr(unary_op_type, "SILU", None)
                    if silu is None:
                        return None
                    return unary_with_param(silu)

                def nlp_create_qkv_heads_decode(
                    self,
                    qkv,
                    *,
                    num_heads,
                    num_kv_heads,
                    memory_config=None,
                ):
                    return ttnn_ops.nlp_create_qkv_heads_decode(
                        self.ttnn,
                        qkv,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        memory_config=memory_config,
                    )

                def rotary_embedding_decode(
                    self,
                    q,
                    k,
                    *,
                    cos_matrix,
                    sin_matrix,
                    transformation_matrix,
                    is_decode_mode=True,
                ):
                    return ttnn_ops.rotary_embedding_decode(
                        self.ttnn,
                        q,
                        k,
                        cos_matrix=cos_matrix,
                        sin_matrix=sin_matrix,
                        transformation_matrix=transformation_matrix,
                        is_decode_mode=is_decode_mode,
                    )

                def paged_update_cache(
                    self,
                    cache_tensor,
                    update_tensor,
                    *,
                    update_idxs_tensor=None,
                    update_idxs=None,
                    page_table=None,
                ):
                    return ttnn_ops.paged_update_cache(
                        self.ttnn,
                        cache_tensor,
                        update_tensor,
                        update_idxs_tensor=update_idxs_tensor,
                        update_idxs=update_idxs,
                        page_table=page_table,
                    )

                def paged_sdpa_decode(
                    self,
                    query,
                    key_cache,
                    value_cache,
                    page_table,
                    cache_position,
                    *,
                    scale=None,
                    memory_config=None,
                    program_config=None,
                    compute_kernel_config=None,
                ):
                    return ttnn_ops.paged_sdpa_decode(
                        self.ttnn,
                        query,
                        key_cache,
                        value_cache,
                        page_table,
                        cache_position,
                        scale=scale,
                        memory_config=memory_config,
                        program_config=program_config,
                        compute_kernel_config=compute_kernel_config,
                    )

                def nlp_concat_heads_decode(
                    self,
                    attn,
                    *,
                    num_heads,
                    memory_config=None,
                ):
                    return ttnn_ops.nlp_concat_heads_decode(
                        self.ttnn,
                        attn,
                        num_heads=num_heads,
                        memory_config=memory_config,
                    )

                def concat(self, tensors, *, dim=-1, memory_config=None):
                    kwargs = {{"dim": dim}}
                    if memory_config is not None:
                        kwargs["memory_config"] = memory_config
                    return self.ttnn.concat(tensors, **kwargs)

                def argmax(self, tensor, *, dim=-1):
                    return self.ttnn.argmax(tensor, dim=dim)


            class {model_class}:
                def __init__(self, device, parameters, config):
                    self.device = device
                    self.parameters = parameters
                    self.config = config
                    self.ops = TTNNCompatOps(ttnn)

                def decode_step(self, token_ids, page_table, cache_position, kv_cache):
                    hidden = self.embed(token_ids)
                    for layer_id in range(self.config.num_layers):
                        hidden = self.decode_layer(
                            layer_id,
                            hidden,
                            page_table,
                            cache_position,
                            kv_cache,
                        )
                    hidden = self.final_norm(hidden)
                    token = self.lm_head_argmax(hidden)
                    return token, kv_cache

                def decode_layer(self, layer_id, hidden, page_table, cache_position, kv_cache):
                    residual = hidden
                    hidden = self.rmsnorm(hidden, layer_id, kind="attn")
                    hidden = self.attention_decode(
                        layer_id,
                        hidden,
                        page_table,
                        cache_position,
                        kv_cache,
                    )
                    hidden = self.ops.add(residual, hidden)
                    residual = hidden
                    hidden = self.rmsnorm(hidden, layer_id, kind="mlp")
                    hidden = self.mlp_decode(layer_id, hidden)
                    hidden = self.ops.add(residual, hidden)
                    return hidden

                def embed(self, token_ids):
                    # TODO: ttnn.embedding for token ids.
                    raise NotImplementedError(
                        "Embedding template is not implemented in Phase 3."
                    )

                def rmsnorm(self, hidden, layer_id, kind):
                    # TODO: ttnn.rms_norm using per-layer RMSNorm weights.
                    raise NotImplementedError(
                        "RMSNorm template is not implemented in Phase 3."
                    )

                def final_norm(self, hidden):
                    # TODO: final ttnn.rms_norm before LM-head.
                    raise NotImplementedError(
                        "Final RMSNorm template is not implemented in Phase 3."
                    )

                def attention_decode(
                    self,
                    layer_id,
                    hidden,
                    page_table,
                    cache_position,
                    kv_cache,
                ):
                    # Template: {attention_template}
                    layer_params = self.parameters.layers[layer_id].attention
                    attention_config = self.config.attention

                    qkv = self.ops.linear(
                        hidden,
                        layer_params.wqkv_packed.weight,
                        memory_config=_optional_attr(
                            attention_config, "qkv_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            attention_config, "qkv_program_config"
                        ),
                        compute_kernel_config=_optional_attr(
                            attention_config, "qkv_compute_kernel_config"
                        ),
                        dtype=_optional_attr(
                            attention_config, "qkv_output_dtype"
                        ),
                    )

                    q, k, v = self.ops.nlp_create_qkv_heads_decode(
                        qkv,
                        num_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_key_value_heads,
                        memory_config=_optional_attr(
                            attention_config, "qkv_heads_memory_config"
                        ),
                    )

                    q, k = self.rotary_embedding_decode(
                        layer_id, q, k, cache_position
                    )
                    kv_cache = self.paged_update_kv_cache(
                        layer_id,
                        k,
                        v,
                        page_table,
                        cache_position,
                        kv_cache,
                    )
                    layer_cache = kv_cache[layer_id]

                    attn = self.ops.paged_sdpa_decode(
                        q,
                        layer_cache.k,
                        layer_cache.v,
                        page_table,
                        cache_position,
                        scale=_optional_attr(attention_config, "scale"),
                        memory_config=_optional_attr(
                            attention_config, "sdpa_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            attention_config, "sdpa_program_config"
                        ),
                        compute_kernel_config=_optional_attr(
                            attention_config, "sdpa_compute_kernel_config"
                        ),
                    )

                    attn = self.ops.nlp_concat_heads_decode(
                        attn,
                        num_heads=self.config.num_attention_heads,
                        memory_config=_optional_attr(
                            attention_config,
                            "concat_heads_output_memory_config",
                        ),
                    )

                    return self.ops.linear(
                        attn,
                        layer_params.o_proj.weight,
                        memory_config=_optional_attr(
                            attention_config, "o_proj_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            attention_config, "o_proj_program_config"
                        ),
                        compute_kernel_config=_optional_attr(
                            attention_config, "o_proj_compute_kernel_config"
                        ),
                        dtype=_optional_attr(
                            attention_config, "o_proj_output_dtype"
                        ),
                    )

                def rotary_embedding_decode(self, layer_id, q, k, cache_position):
                    rotary_params = _optional_attr(
                        self.parameters, "rotary", None
                    )
                    if rotary_params is None:
                        layer_attention = self.parameters.layers[
                            layer_id
                        ].attention
                        rotary_params = _optional_attr(
                            layer_attention, "rotary", None
                        )
                    rotary_config = _optional_attr(self.config, "rotary", None)
                    return self.ops.rotary_embedding_decode(
                        q,
                        k,
                        cos_matrix=_optional_attr(
                            rotary_params, "cos_matrix", None
                        ),
                        sin_matrix=_optional_attr(
                            rotary_params, "sin_matrix", None
                        ),
                        transformation_matrix=_optional_attr(
                            rotary_params, "transformation_matrix", None
                        ),
                        is_decode_mode=_optional_attr(
                            rotary_config, "is_decode_mode", True
                        ),
                    )

                def paged_update_kv_cache(
                    self,
                    layer_id,
                    k,
                    v,
                    page_table,
                    cache_position,
                    kv_cache,
                ):
                    layer_cache = kv_cache[layer_id]
                    self.ops.paged_update_cache(
                        layer_cache.k,
                        k,
                        update_idxs_tensor=cache_position,
                        page_table=page_table,
                    )
                    self.ops.paged_update_cache(
                        layer_cache.v,
                        v,
                        update_idxs_tensor=cache_position,
                        page_table=page_table,
                    )
                    return kv_cache

                def mlp_decode(self, layer_id, hidden):
                    # Template: {mlp_template}
                    layer_params = self.parameters.layers[layer_id].mlp
                    mlp_config = self.config.mlp
                    compute_kernel_config = _optional_attr(
                        mlp_config, "compute_kernel_config"
                    )
                    intermediate_dtype = _optional_attr(
                        mlp_config, "intermediate_dtype"
                    )

                    gate = self.ops.linear(
                        hidden,
                        layer_params.gate_proj.weight,
                        memory_config=_optional_attr(
                            mlp_config, "gate_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            mlp_config, "gate_program_config"
                        ),
                        compute_kernel_config=compute_kernel_config,
                        dtype=intermediate_dtype,
                    )
                    up = self.ops.linear(
                        hidden,
                        layer_params.up_proj.weight,
                        memory_config=_optional_attr(
                            mlp_config, "up_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            mlp_config, "up_program_config"
                        ),
                        compute_kernel_config=compute_kernel_config,
                        dtype=intermediate_dtype,
                    )
                    mid = self.ops.mul_silu(
                        gate,
                        up,
                        memory_config=_tensor_memory_config(gate),
                        dtype=intermediate_dtype,
                    )
                    return self.ops.linear(
                        mid,
                        layer_params.down_proj.weight,
                        memory_config=_optional_attr(
                            mlp_config, "down_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            mlp_config, "down_program_config"
                        ),
                        compute_kernel_config=compute_kernel_config,
                        dtype=_optional_attr(mlp_config, "output_dtype"),
                    )

                def lm_head_argmax(self, hidden):
                    # Template: {lm_head_template} + {generation_template}
                    lm_head_config = self.config.lm_head
                    split_count = int(
                        _optional_attr(
                            lm_head_config,
                            "split_count",
                            GENERATED_LM_HEAD_SPLIT_COUNT,
                        )
                    )
                    program_configs = _optional_attr(
                        lm_head_config, "program_configs", None
                    )
                    split_configs = _optional_attr(
                        lm_head_config, "splits", None
                    )
                    shard_logits = []
                    for shard_id in range(split_count):
                        split_params = self.parameters.lm_head.splits[shard_id]
                        split_config = None
                        if split_configs is not None and shard_id < len(split_configs):
                            split_config = split_configs[shard_id]
                        program_config = _optional_attr(
                            split_config, "program_config", None
                        )
                        if (
                            program_config is None
                            and program_configs is not None
                            and shard_id < len(program_configs)
                        ):
                            program_config = program_configs[shard_id]
                        logits_i = self.ops.linear(
                            hidden,
                            split_params.weight,
                            memory_config=_optional_attr(
                                lm_head_config, "output_memory_config"
                            ),
                            program_config=program_config,
                            compute_kernel_config=_optional_attr(
                                lm_head_config, "compute_kernel_config"
                            ),
                            dtype=_optional_attr(lm_head_config, "output_dtype"),
                        )
                        shard_logits.append(logits_i)

                    logits = self.ops.concat(
                        shard_logits,
                        dim=-1,
                        memory_config=_optional_attr(
                            lm_head_config, "concat_memory_config"
                        ),
                    )
                    generation_config = _optional_attr(
                        self.config, "generation", None
                    )
                    generation_mode = _optional_attr(
                        generation_config, "mode", "greedy"
                    )
                    retain_logits = bool(
                        _optional_attr(lm_head_config, "retain_logits", False)
                    )
                    if generation_mode == "greedy" and not retain_logits:
                        return self.ops.argmax(logits, dim=-1)
                    return logits


            GENERATED_NUM_LAYERS = {num_layers}
            GENERATED_LM_HEAD_SPLIT_COUNT = {lm_head_split_count}
            '''
        ).lstrip()
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
        wrapper; embedding and RMSNorm bodies still intentionally raise
        `NotImplementedError`. Attention primitive wrappers raise
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


def write_python_ttnn_skeleton(
    plan: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    validate_execution_plan_for_codegen(plan)
    root = ensure_output_dir(out_dir)
    paths = planned_artifact_paths(root)
    write_text(paths["model.py"], render_python_ttnn_model(plan))
    write_json(paths["config.json"], build_codegen_config(plan))
    write_json(paths["plan.json"], copy.deepcopy(plan))
    write_text(paths["README.md"], render_codegen_readme(plan))
    return paths


def dry_run_report(plan: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    config = build_codegen_config(plan)
    return {
        "dry_run": True,
        "model_name": config["model_name"],
        "num_layers": config["num_layers"],
        "out_dir": str(out_dir),
        "artifacts": sorted(planned_artifact_paths(out_dir)),
    }


def _template_name(plan: dict[str, Any], expected: str) -> str:
    for layer in plan.get("layers", []):
        for template in layer.get("templates", []):
            if template == expected:
                return expected
    for template in plan.get("final", []):
        if template == expected:
            return expected
    return expected
