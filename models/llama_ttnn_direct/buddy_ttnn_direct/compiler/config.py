from __future__ import annotations

import copy
from typing import Any

from ..templates.attention_decode import (
    official_paged_attention_decode_op_sequence,
)
from ..templates.attention_prefill import (
    official_prefill_attention_op_sequence,
)
from ..templates.lm_head import build_lm_head_split_ranges


def validate_execution_plan_for_codegen(plan: dict[str, Any]) -> None:
    errors: list[str] = []
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
