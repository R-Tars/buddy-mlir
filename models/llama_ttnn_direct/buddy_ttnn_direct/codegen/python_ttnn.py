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
from ..templates.attention_prefill import (
    official_prefill_attention_op_sequence,
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
    kv_cache_template = template_config.get("kv_cache_template")
    kv_cache_policy = "paged" if kv_cache_template == "paged_kv_cache" else None
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
        "embedding": {
            "output_memory_config": None,
            "output_dtype": None,
        },
        "rms_norm": {
            "eps": plan.get("rms_norm_eps"),
            "input_memory_config": "dram",
            "output_memory_config": "dram",
            "output_dtype": None,
        },
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
        "prefill": {
            "template": template_config.get(
                "prefill_attention_template",
                "official_prefill_attention",
            ),
            "mlp_template": template_config.get(
                "prefill_mlp_template",
                "official_gated_mlp_prefill",
            ),
            "seq_len": int(template_config.get("prefill_seq_len", 128)),
            "attention_op_sequence": official_prefill_attention_op_sequence(),
            "attention_mask": "causal",
            "cache_write_policy": "fill_cache_per_user",
            "kv_cache_source": "prefill",
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
        "mlp": {
            "template": "official_gated_mlp_decode",
            "gate_output_memory_config": None,
            "gate_program_config": None,
            "up_output_memory_config": None,
            "up_program_config": None,
            "down_output_memory_config": None,
            "down_program_config": None,
            "compute_kernel_config": None,
            "intermediate_dtype": None,
            "output_dtype": None,
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
        "kv_cache": {
            "template": kv_cache_template,
            "policy": kv_cache_policy,
            "page_block_size": 32,
            "dtype": "bfloat8_b",
            "max_cache_len": plan["max_cache_len"],
            "num_kv_heads": plan.get("num_key_value_heads"),
            "head_dim": head_dim,
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
    rms_norm_eps = config["rms_norm"]["eps"]
    if rms_norm_eps is None:
        rms_norm_eps = 1e-5
    rms_norm_eps = float(rms_norm_eps)

    return (
        textwrap.dedent(
            f'''
            # Auto-generated by Buddy-TTNN Direct.
            from __future__ import annotations

            import json
            from types import SimpleNamespace

            import ttnn
            from models.llama_ttnn_direct.buddy_ttnn_direct.ttnn_compat import ops as ttnn_ops


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


            def _tensor_shape(tensor):
                shape = getattr(tensor, "shape", None)
                if shape is None:
                    return None
                return [int(dim) for dim in shape]


            class TTNNCompatOps:
                def __init__(self, ttnn_module):
                    self.ttnn = ttnn_module
                    self.op_log = []

                def _record(self, op_name):
                    self.op_log.append(op_name)

                def resolve_memory_config(self, memory_config):
                    if memory_config is None:
                        return None
                    if not isinstance(memory_config, str):
                        return memory_config
                    value = memory_config
                    if value in ("", "none", "None", "default"):
                        return None
                    if value.startswith("ttnn."):
                        value = value.split(".", 1)[1]
                    aliases = {{
                        "dram": "DRAM_MEMORY_CONFIG",
                        "l1": "L1_MEMORY_CONFIG",
                        "l1_interleaved": "L1_MEMORY_CONFIG",
                        "l1_width_sharded": "L1_MEMORY_CONFIG",
                        "l1_height_sharded": "L1_MEMORY_CONFIG",
                    }}
                    attr = aliases.get(value, value)
                    return getattr(self.ttnn, attr, memory_config)

                def add(
                    self,
                    left,
                    right,
                    *,
                    memory_config=None,
                    dtype=None,
                    op_name="residual_add",
                ):
                    self._record(op_name)
                    kwargs = {{}}
                    if memory_config is not None:
                        kwargs["memory_config"] = self.resolve_memory_config(
                            memory_config
                        )
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
                    op_name="linear",
                ):
                    self._record(op_name)
                    kwargs = {{}}
                    if memory_config is not None:
                        kwargs["memory_config"] = self.resolve_memory_config(
                            memory_config
                        )
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
                    op_name="mul_silu",
                ):
                    self._record(op_name)
                    kwargs = {{}}
                    activation = self._silu_activation()
                    if activation is not None:
                        kwargs["input_tensor_a_activations"] = [activation]
                    if memory_config is not None:
                        kwargs["memory_config"] = self.resolve_memory_config(
                            memory_config
                        )
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

                def embedding(
                    self,
                    token_ids,
                    weight,
                    *,
                    memory_config=None,
                    dtype=None,
                    op_name="embedding",
                ):
                    self._record(op_name)
                    op = getattr(self.ttnn, "embedding", None)
                    if op is None:
                        raise ttnn_ops.UnsupportedTTNNOp(
                            "embedding",
                            (("embedding",),),
                        )
                    kwargs = {{}}
                    if memory_config is not None:
                        kwargs["memory_config"] = self.resolve_memory_config(
                            memory_config
                        )
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    return op(token_ids, weight, **kwargs)

                def rms_norm(
                    self,
                    hidden,
                    weight,
                    *,
                    epsilon,
                    memory_config=None,
                    dtype=None,
                    op_name="rms_norm",
                ):
                    self._record(op_name)
                    op = getattr(self.ttnn, "rms_norm", None)
                    if op is None:
                        op = getattr(self.ttnn, "rmsnorm", None)
                    if op is None:
                        raise ttnn_ops.UnsupportedTTNNOp(
                            "rms_norm",
                            (("rms_norm",), ("rmsnorm",)),
                        )
                    hidden = self.ensure_tile_layout(
                        hidden,
                        op_name=f"to_layout.tile.{{op_name}}",
                    )
                    kwargs = {{"weight": weight, "epsilon": epsilon}}
                    if memory_config is not None:
                        kwargs["memory_config"] = self.resolve_memory_config(
                            memory_config
                        )
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    return op(hidden, **kwargs)

                def ensure_tile_layout(self, tensor, *, op_name):
                    to_layout = getattr(self.ttnn, "to_layout", None)
                    tile_layout = getattr(self.ttnn, "TILE_LAYOUT", None)
                    if to_layout is None or tile_layout is None:
                        return tensor
                    self._record(op_name)
                    return to_layout(tensor, tile_layout)

                def reshape_decode_hidden_for_layer(
                    self,
                    hidden,
                    *,
                    op_name="reshape_hidden_decode",
                ):
                    shape = getattr(hidden, "shape", None)
                    if shape is None:
                        return hidden
                    shape = [int(dim) for dim in shape]
                    if len(shape) == 3 and shape[1] == 1:
                        batch = shape[0]
                        feature_dim = shape[2]
                    elif (
                        len(shape) == 4
                        and shape[0] == 1
                        and shape[1] != 1
                        and shape[2] == 1
                    ):
                        batch = shape[1]
                        feature_dim = shape[3]
                    else:
                        return hidden
                    reshape = getattr(self.ttnn, "reshape", None)
                    if reshape is None:
                        return hidden
                    logical_shape = (1, 1, batch, feature_dim)
                    padded_batch = ((batch + 31) // 32) * 32
                    padded_shape = (1, 1, padded_batch, feature_dim)
                    self._record(op_name)
                    try:
                        return reshape(hidden, logical_shape, padded_shape)
                    except TypeError:
                        return reshape(hidden, logical_shape)

                def reshape_decode_qkv_for_heads(
                    self,
                    qkv,
                    *,
                    op_name="reshape_qkv_decode",
                ):
                    shape = getattr(qkv, "shape", None)
                    if shape is None or len(shape) != 4:
                        return qkv
                    shape = [int(dim) for dim in shape]
                    if not (shape[0] == 1 and shape[1] != 1 and shape[2] == 1):
                        return qkv
                    reshape = getattr(self.ttnn, "reshape", None)
                    if reshape is None:
                        return qkv
                    batch = shape[1]
                    feature_dim = shape[3]
                    logical_shape = (1, 1, batch, feature_dim)
                    padded_batch = ((batch + 31) // 32) * 32
                    padded_shape = (1, 1, padded_batch, feature_dim)
                    self._record(op_name)
                    try:
                        return reshape(qkv, logical_shape, padded_shape)
                    except TypeError:
                        return reshape(qkv, logical_shape)

                def reshape_prefill_qkv_for_heads(
                    self,
                    qkv,
                    *,
                    op_name="reshape_qkv_prefill",
                ):
                    shape = getattr(qkv, "shape", None)
                    if shape is None or len(shape) != 4:
                        return qkv
                    shape = [int(dim) for dim in shape]
                    if shape[0] == 1:
                        squeeze_dim = 0
                        logical_shape = (shape[1], shape[2], shape[3])
                    elif shape[1] == 1:
                        squeeze_dim = 1
                        logical_shape = (shape[0], shape[2], shape[3])
                    else:
                        return qkv
                    squeeze = getattr(self.ttnn, "squeeze", None)
                    if callable(squeeze):
                        self._record(op_name)
                        try:
                            return squeeze(qkv, squeeze_dim)
                        except TypeError:
                            pass
                    reshape = getattr(self.ttnn, "reshape", None)
                    if reshape is None:
                        return qkv
                    self._record(op_name)
                    return reshape(qkv, logical_shape)

                def to_memory_config(
                    self,
                    tensor,
                    *,
                    memory_config=None,
                    op_name="to_memory_config",
                ):
                    memory_config = self.resolve_memory_config(memory_config)
                    if memory_config is None:
                        return tensor
                    op = getattr(self.ttnn, "to_memory_config", None)
                    if op is None:
                        return tensor
                    self._record(op_name)
                    return op(tensor, memory_config=memory_config)

                def slice_batch_user(
                    self,
                    tensor,
                    user_id,
                    *,
                    op_name="slice.batch_user",
                ):
                    shape = _tensor_shape(tensor)
                    if shape is None or not shape or shape[0] <= 1:
                        return tensor
                    slice_op = getattr(self.ttnn, "slice", None)
                    if not callable(slice_op):
                        raise ttnn_ops.UnsupportedTTNNOp(
                            "slice_batch_user",
                            (("slice",),),
                        )
                    starts = [0 for _ in shape]
                    ends = list(shape)
                    steps = [1 for _ in shape]
                    starts[0] = int(user_id)
                    ends[0] = int(user_id) + 1
                    self._record(op_name)
                    try:
                        return slice_op(tensor, starts, ends, steps)
                    except TypeError as err:
                        try:
                            return slice_op(tensor, starts, ends, steps=steps)
                        except TypeError:
                            try:
                                return slice_op(tensor, starts, ends)
                            except TypeError:
                                raise err

                def nlp_create_qkv_heads_decode(
                    self,
                    qkv,
                    *,
                    num_heads,
                    num_kv_heads,
                    memory_config=None,
                    op_name="nlp_create_qkv_heads_decode",
                ):
                    self._record(op_name)
                    return ttnn_ops.nlp_create_qkv_heads_decode(
                        self.ttnn,
                        qkv,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        memory_config=self.resolve_memory_config(memory_config),
                    )

                def split_qkv_heads_prefill(
                    self,
                    qkv,
                    *,
                    num_heads,
                    num_kv_heads,
                    memory_config=None,
                    op_name="split_query_key_value_heads_prefill",
                ):
                    self._record(op_name)
                    return ttnn_ops.split_qkv_heads_prefill(
                        self.ttnn,
                        qkv,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        memory_config=self.resolve_memory_config(memory_config),
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
                    op_name="rotary_embedding_decode",
                ):
                    self._record(op_name)
                    return ttnn_ops.rotary_embedding_decode(
                        self.ttnn,
                        q,
                        k,
                        cos_matrix=cos_matrix,
                        sin_matrix=sin_matrix,
                        transformation_matrix=transformation_matrix,
                        is_decode_mode=is_decode_mode,
                    )

                def rotary_embedding_prefill(
                    self,
                    q,
                    k,
                    *,
                    cos_matrix,
                    sin_matrix,
                    transformation_matrix,
                    op_name="rotary_embedding_prefill",
                ):
                    self._record(op_name)
                    return ttnn_ops.rotary_embedding_prefill(
                        self.ttnn,
                        q,
                        k,
                        cos_matrix=cos_matrix,
                        sin_matrix=sin_matrix,
                        transformation_matrix=transformation_matrix,
                    )

                def paged_update_cache(
                    self,
                    cache_tensor,
                    update_tensor,
                    *,
                    update_idxs_tensor=None,
                    update_idxs=None,
                    page_table=None,
                    op_name="paged_update_cache",
                ):
                    self._record(op_name)
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
                    op_name="paged_scaled_dot_product_attention_decode",
                ):
                    self._record(op_name)
                    return ttnn_ops.paged_sdpa_decode(
                        self.ttnn,
                        query,
                        key_cache,
                        value_cache,
                        page_table,
                        cache_position,
                        scale=scale,
                        memory_config=self.resolve_memory_config(memory_config),
                        program_config=program_config,
                        compute_kernel_config=compute_kernel_config,
                    )

                def scaled_dot_product_attention(
                    self,
                    query,
                    key,
                    value,
                    *,
                    is_causal=True,
                    scale=None,
                    memory_config=None,
                    program_config=None,
                    compute_kernel_config=None,
                    attention_mask=None,
                    op_name="scaled_dot_product_attention",
                ):
                    self._record(op_name)
                    return ttnn_ops.scaled_dot_product_attention(
                        self.ttnn,
                        query,
                        key,
                        value,
                        is_causal=is_causal,
                        scale=scale,
                        memory_config=self.resolve_memory_config(memory_config),
                        program_config=program_config,
                        compute_kernel_config=compute_kernel_config,
                        attention_mask=attention_mask,
                    )

                def fill_cache(
                    self,
                    cache_tensor,
                    update_tensor,
                    *,
                    user_id=0,
                    page_table=None,
                    op_name="fill_cache",
                ):
                    self._record(op_name)
                    return ttnn_ops.fill_cache(
                        self.ttnn,
                        cache_tensor,
                        update_tensor,
                        user_id=user_id,
                        page_table=page_table,
                    )

                def paged_fill_cache(
                    self,
                    cache_tensor,
                    update_tensor,
                    page_table,
                    *,
                    batch_idx=0,
                    batch_idx_tensor=None,
                    compute_kernel_config=None,
                    op_name="paged_fill_cache",
                ):
                    self._record(op_name)
                    return ttnn_ops.paged_fill_cache(
                        self.ttnn,
                        cache_tensor,
                        update_tensor,
                        page_table,
                        batch_idx=batch_idx,
                        batch_idx_tensor=batch_idx_tensor,
                        compute_kernel_config=compute_kernel_config,
                    )

                def nlp_concat_heads_decode(
                    self,
                    attn,
                    *,
                    num_heads,
                    memory_config=None,
                    op_name="nlp_concat_heads_decode",
                ):
                    self._record(op_name)
                    return ttnn_ops.nlp_concat_heads_decode(
                        self.ttnn,
                        attn,
                        num_heads=num_heads,
                        memory_config=self.resolve_memory_config(memory_config),
                    )

                def concat_heads_prefill(
                    self,
                    attn,
                    *,
                    memory_config=None,
                    op_name="concat_heads_prefill",
                ):
                    self._record(op_name)
                    return ttnn_ops.concat_heads_prefill(
                        self.ttnn,
                        attn,
                        memory_config=self.resolve_memory_config(memory_config),
                    )

                def concat(
                    self,
                    tensors,
                    *,
                    dim=-1,
                    memory_config=None,
                    op_name="concat",
                ):
                    self._record(op_name)
                    kwargs = {{"dim": dim}}
                    if memory_config is not None:
                        kwargs["memory_config"] = self.resolve_memory_config(
                            memory_config
                        )
                    return self.ttnn.concat(tensors, **kwargs)

                def argmax(self, tensor, *, dim=-1, op_name="argmax_or_sampling"):
                    self._record(op_name)
                    return self.ttnn.argmax(tensor, dim=dim)

                def normalize_decode_token(
                    self,
                    token,
                    *,
                    batch_size,
                    op_name="normalize_argmax_token",
                ):
                    shape = getattr(token, "shape", None)
                    if shape is None:
                        return token
                    shape = [int(dim) for dim in shape]
                    if shape in ([batch_size, 1], [batch_size]):
                        return token
                    slice_op = getattr(self.ttnn, "slice", None)
                    squeeze_op = getattr(self.ttnn, "squeeze", None)
                    reshape_op = getattr(self.ttnn, "reshape", None)
                    if (
                        len(shape) == 3
                        and shape[0] == 1
                        and shape[1] == 1
                        and shape[2] == batch_size
                    ):
                        if callable(reshape_op):
                            self._record(f"{{op_name}}.reshape")
                            try:
                                return reshape_op(
                                    token,
                                    (batch_size, 1),
                                    (batch_size, 1),
                                )
                            except TypeError:
                                return reshape_op(token, (batch_size, 1))
                        if callable(squeeze_op):
                            self._record(f"{{op_name}}.squeeze.0")
                            token = squeeze_op(token, 0)
                            self._record(f"{{op_name}}.squeeze.1")
                            return squeeze_op(token, 0)
                        return token
                    if (
                        callable(slice_op)
                        and len(shape) == 2
                        and shape[0] == batch_size
                        and shape[1] > 1
                    ):
                        self._record(f"{{op_name}}.slice")
                        return slice_op(
                            token,
                            [0, shape[1] - 1],
                            [batch_size, shape[1]],
                        )
                    if not callable(slice_op) or not callable(squeeze_op):
                        return token
                    if (
                        len(shape) == 3
                        and shape[0] == 1
                        and shape[1] == batch_size
                        and shape[2] >= 1
                    ):
                        if shape[2] > 1:
                            self._record(f"{{op_name}}.slice")
                            token = slice_op(
                                token,
                                [0, 0, 0],
                                [1, batch_size, 1],
                            )
                        self._record(f"{{op_name}}.squeeze")
                        return squeeze_op(token, 0)
                    if (
                        len(shape) == 4
                        and shape[0] == 1
                        and shape[1] == 1
                        and shape[2] == batch_size
                        and shape[3] >= 1
                    ):
                        if shape[3] > 1:
                            self._record(f"{{op_name}}.slice")
                            token = slice_op(
                                token,
                                [0, 0, 0, 0],
                                [1, 1, batch_size, 1],
                            )
                        self._record(f"{{op_name}}.squeeze.0")
                        token = squeeze_op(token, 0)
                        self._record(f"{{op_name}}.squeeze.1")
                        return squeeze_op(token, 0)
                    return token


            class {model_class}:
                def __init__(self, device, parameters, config):
                    self.device = device
                    self.parameters = parameters
                    self.config = config
                    self.ops = TTNNCompatOps(ttnn)

                def _attention_heads_memory_config(self):
                    return self._height_sharded_memory_config(
                        batch_size=int(_optional_attr(self.config, "batch_size", 1) or 1),
                        head_dim=int(_optional_attr(self.config, "head_dim", 1) or 1),
                    )

                def _height_sharded_memory_config(self, *, batch_size, head_dim):
                    ttnn_module = self.ops.ttnn
                    create_sharded = getattr(
                        ttnn_module,
                        "create_sharded_memory_config",
                        None,
                    )
                    if callable(create_sharded):
                        core_grid = self._batch_core_grid(batch_size=batch_size)
                        shard_strategy = getattr(
                            getattr(ttnn_module, "ShardStrategy", None),
                            "HEIGHT",
                            None,
                        )
                        shard_orientation = getattr(
                            getattr(ttnn_module, "ShardOrientation", None),
                            "ROW_MAJOR",
                            None,
                        )
                        tile_size = int(getattr(ttnn_module, "TILE_SIZE", 32))
                        if core_grid is not None and shard_strategy is not None:
                            try:
                                return create_sharded(
                                    shape=(tile_size, head_dim),
                                    core_grid=core_grid,
                                    strategy=shard_strategy,
                                    orientation=shard_orientation,
                                    use_height_and_width_as_shard_shape=True,
                                )
                            except Exception:
                                pass
                    return getattr(
                        ttnn_module,
                        "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
                        None,
                    )

                def _batch_core_grid(self, *, batch_size):
                    ttnn_module = self.ops.ttnn
                    core_grid_type = getattr(ttnn_module, "CoreGrid", None)
                    if not callable(core_grid_type):
                        return None
                    compute_grid = None
                    compute_with_storage_grid_size = getattr(
                        self.device,
                        "compute_with_storage_grid_size",
                        None,
                    )
                    if callable(compute_with_storage_grid_size):
                        try:
                            compute_grid = compute_with_storage_grid_size()
                        except Exception:
                            compute_grid = None
                    physical_x = int(getattr(compute_grid, "x", 8) or 8)
                    physical_y = int(getattr(compute_grid, "y", 8) or 8)
                    grid_x = max(1, min(batch_size, physical_x))
                    while grid_x > 1 and batch_size % grid_x != 0:
                        grid_x -= 1
                    grid_y = max(1, (batch_size + grid_x - 1) // grid_x)
                    if grid_y > physical_y:
                        return None
                    try:
                        return core_grid_type(y=grid_y, x=grid_x)
                    except TypeError:
                        return core_grid_type(grid_y, grid_x)

                def prefill_prompt(self, token_ids, kv_cache, page_table=None):
                    hidden = self.embed(token_ids)
                    cache_reports = []
                    for layer_id in range(self.config.num_layers):
                        hidden, kv_cache, cache_report = self.prefill_layer(
                            layer_id,
                            hidden,
                            kv_cache,
                            page_table,
                        )
                        cache_reports.append(cache_report)
                    hidden = self.final_norm(hidden)
                    token = self.lm_head_argmax(hidden)
                    return token, kv_cache, cache_reports

                def prefill_layer(self, layer_id, hidden, kv_cache, page_table=None):
                    residual = hidden
                    hidden = self.rmsnorm(hidden, layer_id, kind="attn")
                    hidden, kv_cache, cache_report = self.attention_prefill(
                        layer_id,
                        hidden,
                        kv_cache,
                        page_table,
                    )
                    hidden = self.ops.add(
                        residual,
                        hidden,
                        op_name="residual_add.attn",
                    )
                    residual = hidden
                    hidden = self.rmsnorm(hidden, layer_id, kind="mlp")
                    hidden = self.mlp_decode(layer_id, hidden)
                    hidden = self.ops.add(
                        residual,
                        hidden,
                        op_name="residual_add.mlp",
                    )
                    return hidden, kv_cache, cache_report

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
                    hidden = self.ops.reshape_decode_hidden_for_layer(
                        hidden,
                        op_name="reshape_hidden_decode",
                    )
                    residual = hidden
                    hidden = self.rmsnorm(hidden, layer_id, kind="attn")
                    hidden = self.attention_decode(
                        layer_id,
                        hidden,
                        page_table,
                        cache_position,
                        kv_cache,
                    )
                    hidden = self.ops.add(
                        residual,
                        hidden,
                        op_name="residual_add.attn",
                    )
                    residual = hidden
                    hidden = self.rmsnorm(hidden, layer_id, kind="mlp")
                    hidden = self.mlp_decode(layer_id, hidden)
                    hidden = self.ops.add(
                        residual,
                        hidden,
                        op_name="residual_add.mlp",
                    )
                    return hidden

                def embed(self, token_ids):
                    embedding_config = _optional_attr(
                        self.config, "embedding", None
                    )
                    return self.ops.embedding(
                        token_ids,
                        self.parameters.embedding.weight,
                        memory_config=_optional_attr(
                            embedding_config, "output_memory_config"
                        ),
                        dtype=_optional_attr(
                            embedding_config, "output_dtype"
                        ),
                    )

                def rmsnorm(self, hidden, layer_id, kind):
                    layer_params = self.parameters.layers[layer_id]
                    if kind == "attn":
                        norm_params = layer_params.input_norm
                    elif kind == "mlp":
                        norm_params = layer_params.post_attention_norm
                    else:
                        raise ValueError(f"unknown RMSNorm kind: {{kind}}")
                    return self._rmsnorm_with_weight(
                        hidden,
                        norm_params.weight,
                        op_name=f"rms_norm.{{kind}}",
                    )

                def final_norm(self, hidden):
                    return self._rmsnorm_with_weight(
                        hidden,
                        self.parameters.final_norm.weight,
                        op_name="rms_norm.final",
                    )

                def _rmsnorm_with_weight(self, hidden, weight, op_name):
                    rms_config = _optional_attr(self.config, "rms_norm", None)
                    epsilon = _optional_attr(
                        rms_config, "eps", GENERATED_RMS_NORM_EPS
                    )
                    hidden = self.ops.to_memory_config(
                        hidden,
                        memory_config=_optional_attr(
                            rms_config, "input_memory_config"
                        ),
                        op_name=f"to_memory_config.{{op_name}}.input",
                    )
                    return self.ops.rms_norm(
                        hidden,
                        weight,
                        epsilon=epsilon,
                        memory_config=_optional_attr(
                            rms_config, "output_memory_config"
                        ),
                        dtype=_optional_attr(rms_config, "output_dtype"),
                        op_name=op_name,
                    )

                def attention_prefill(
                    self,
                    layer_id,
                    hidden,
                    kv_cache,
                    page_table=None,
                ):
                    layer_params = self.parameters.layers[layer_id].attention
                    prefill_config = _optional_attr(
                        self.config,
                        "prefill",
                        self.config.attention,
                    )

                    qkv = self.ops.linear(
                        hidden,
                        layer_params.wqkv_packed.weight,
                        memory_config=_optional_attr(
                            prefill_config, "qkv_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            prefill_config, "qkv_program_config"
                        ),
                        compute_kernel_config=_optional_attr(
                            prefill_config, "qkv_compute_kernel_config"
                        ),
                        dtype=_optional_attr(
                            prefill_config, "qkv_output_dtype"
                        ),
                        op_name="qkv_linear",
                    )
                    qkv = self.ops.reshape_prefill_qkv_for_heads(
                        qkv,
                        op_name="reshape_qkv_prefill",
                    )

                    q, k, v = self.ops.split_qkv_heads_prefill(
                        qkv,
                        num_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_key_value_heads,
                        memory_config=_optional_attr(
                            prefill_config, "qkv_heads_memory_config"
                        ),
                    )

                    q, k = self.rotary_embedding_prefill(layer_id, q, k)
                    attn = self.ops.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=True,
                        scale=_optional_attr(self.config.attention, "scale"),
                        memory_config=_optional_attr(
                            prefill_config, "sdpa_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            prefill_config, "sdpa_program_config"
                        ),
                        compute_kernel_config=_optional_attr(
                            prefill_config, "sdpa_compute_kernel_config"
                        ),
                    )
                    kv_cache, cache_report = self.fill_prefill_kv_cache(
                        layer_id,
                        k,
                        v,
                        kv_cache,
                        page_table=page_table,
                    )

                    attn = self.ops.concat_heads_prefill(
                        attn,
                        memory_config=_optional_attr(
                            prefill_config,
                            "concat_heads_output_memory_config",
                        ),
                    )

                    output = self.ops.linear(
                        attn,
                        layer_params.o_proj.weight,
                        memory_config=_optional_attr(
                            prefill_config, "o_proj_output_memory_config"
                        ),
                        program_config=_optional_attr(
                            prefill_config, "o_proj_program_config"
                        ),
                        compute_kernel_config=_optional_attr(
                            prefill_config, "o_proj_compute_kernel_config"
                        ),
                        dtype=_optional_attr(
                            prefill_config, "o_proj_output_dtype"
                        ),
                        op_name="o_proj_linear",
                    )
                    return output, kv_cache, cache_report

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
                        op_name="qkv_linear",
                    )
                    qkv = self.ops.reshape_decode_qkv_for_heads(
                        qkv,
                        op_name="reshape_qkv_decode",
                    )
                    attention_heads_memory_config = _optional_attr(
                        attention_config,
                        "qkv_heads_memory_config",
                    ) or self._attention_heads_memory_config()

                    q, k, v = self.ops.nlp_create_qkv_heads_decode(
                        qkv,
                        num_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_key_value_heads,
                        memory_config=attention_heads_memory_config,
                        op_name="nlp_create_qkv_heads_decode",
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

                    attn = self.ops.to_memory_config(
                        attn,
                        memory_config=_optional_attr(
                            attention_config,
                            "concat_heads_input_memory_config",
                        )
                        or attention_heads_memory_config,
                        op_name="to_memory_config.concat_heads_input",
                    )
                    attn = self.ops.nlp_concat_heads_decode(
                        attn,
                        num_heads=self.config.num_attention_heads,
                        memory_config=None,
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
                        op_name="o_proj_linear",
                    )

                def rotary_embedding_prefill(self, layer_id, q, k):
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
                    return self.ops.rotary_embedding_prefill(
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
                        op_name="rotary_embedding_decode",
                    )

                def fill_prefill_kv_cache(
                    self,
                    layer_id,
                    k,
                    v,
                    kv_cache,
                    page_table=None,
                ):
                    layer_cache = kv_cache[layer_id]
                    key_shape = _tensor_shape(k)
                    value_shape = _tensor_shape(v)
                    batch_user_count = 1
                    if key_shape is not None and len(key_shape) >= 1:
                        batch_user_count = max(batch_user_count, key_shape[0])
                    if value_shape is not None and len(value_shape) >= 1:
                        batch_user_count = max(batch_user_count, value_shape[0])
                    user_reports = []
                    for user_id in range(batch_user_count):
                        user_k = (
                            self.ops.slice_batch_user(
                                k,
                                user_id,
                                op_name="slice.prefill_k",
                            )
                            if batch_user_count > 1
                            else k
                        )
                        user_v = (
                            self.ops.slice_batch_user(
                                v,
                                user_id,
                                op_name="slice.prefill_v",
                            )
                            if batch_user_count > 1
                            else v
                        )
                        filled_k = self.ops.fill_cache(
                            layer_cache.k,
                            user_k,
                            user_id=user_id,
                            op_name="fill_cache.k",
                        ) if page_table is None else self.ops.paged_fill_cache(
                            layer_cache.k,
                            user_k,
                            page_table,
                            batch_idx=user_id,
                            op_name="fill_cache.k",
                        )
                        if filled_k is not None:
                            layer_cache.k = filled_k
                        filled_v = self.ops.fill_cache(
                            layer_cache.v,
                            user_v,
                            user_id=user_id,
                            op_name="fill_cache.v",
                        ) if page_table is None else self.ops.paged_fill_cache(
                            layer_cache.v,
                            user_v,
                            page_table,
                            batch_idx=user_id,
                            op_name="fill_cache.v",
                        )
                        if filled_v is not None:
                            layer_cache.v = filled_v
                        user_reports.append(
                            {{
                                "user_id": user_id,
                                "key_update_shape": _tensor_shape(user_k),
                                "value_update_shape": _tensor_shape(user_v),
                                "key_cache_memory_config": (
                                    _tensor_memory_config(layer_cache.k)
                                ),
                                "value_cache_memory_config": (
                                    _tensor_memory_config(layer_cache.v)
                                ),
                            }}
                        )
                    return kv_cache, {{
                        "layer_id": layer_id,
                        "write_policy": (
                            "fill_cache_per_user"
                            if page_table is None
                            else "paged_fill_cache_per_user"
                        ),
                        "page_table_shape": _tensor_shape(page_table),
                        "filled_user_count": batch_user_count,
                        "update_shape_layout": "batch_heads_seq_head_dim",
                        "key_update_shape": key_shape,
                        "value_update_shape": value_shape,
                        "users": user_reports,
                        "key_cache_memory_config": _tensor_memory_config(
                            layer_cache.k
                        ),
                        "value_cache_memory_config": _tensor_memory_config(
                            layer_cache.v
                        ),
                    }}

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
                        op_name="paged_update_cache.k",
                    )
                    self.ops.paged_update_cache(
                        layer_cache.v,
                        v,
                        update_idxs_tensor=cache_position,
                        page_table=page_table,
                        op_name="paged_update_cache.v",
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
                        op_name="mlp_gate",
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
                        op_name="mlp_up",
                    )
                    mid = self.ops.mul_silu(
                        gate,
                        up,
                        memory_config=_tensor_memory_config(gate),
                        dtype=intermediate_dtype,
                        op_name="mul_silu",
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
                        op_name="mlp_down",
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
                            op_name="split_lm_head",
                        )
                        shard_logits.append(logits_i)

                    logits = self.ops.concat(
                        shard_logits,
                        dim=-1,
                        memory_config=_optional_attr(
                            lm_head_config, "concat_memory_config"
                        ),
                        op_name="split_lm_head.concat",
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
                        token = self.ops.argmax(
                            logits,
                            dim=-1,
                            op_name="argmax_or_sampling",
                        )
                        return self.ops.normalize_decode_token(
                            token,
                            batch_size=int(
                                _optional_attr(
                                    self.config,
                                    "batch_size",
                                    1,
                                )
                                or 1
                            ),
                        )
                    return logits


            GENERATED_NUM_LAYERS = {num_layers}
            GENERATED_LM_HEAD_SPLIT_COUNT = {lm_head_split_count}
            GENERATED_RMS_NORM_EPS = {rms_norm_eps!r}
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
