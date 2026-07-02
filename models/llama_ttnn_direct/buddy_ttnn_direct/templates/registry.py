from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..semantic.graph import LlamaModelGraph
from ..semantic.validate import validate_llama_graph


REQUIRED_CONFIG_KEYS = {
    "device",
    "model",
    "batch_size",
    "decode_seq_len",
    "prefill_seq_len",
    "max_cache_len",
    "attention_template",
    "mlp_template",
    "lm_head_template",
    "kv_cache_template",
    "generation_template",
    "lm_head_split_count",
    "dtype_recipe",
}

CUSTOM_FUSED_MLP_TEMPLATE = "custom_buddy_fused_mlp_decode"
CUSTOM_FUSED_LM_HEAD_TEMPLATE = "custom_buddy_lmhead_argmax_decode"
CUSTOM_FUSED_REGION_TEMPLATES = {
    CUSTOM_FUSED_MLP_TEMPLATE,
    CUSTOM_FUSED_LM_HEAD_TEMPLATE,
}

ALLOWED_ATTENTION_TEMPLATES = {"official_paged_attention_decode"}
ALLOWED_MLP_TEMPLATES = {
    "official_gated_mlp_decode",
    CUSTOM_FUSED_MLP_TEMPLATE,
}
ALLOWED_LM_HEAD_TEMPLATES = {
    "official_split_lm_head",
    CUSTOM_FUSED_LM_HEAD_TEMPLATE,
}
ALLOWED_KV_CACHE_TEMPLATES = {"paged_kv_cache"}
ALLOWED_GENERATION_TEMPLATES = {
    "device_argmax_greedy",
    "full_logits",
}


def load_template_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    validate_template_config(config)
    return config


def validate_template_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    errors: list[str] = []
    if missing:
        errors.append("missing config keys: " + ", ".join(missing))

    for key in (
        "batch_size",
        "decode_seq_len",
        "prefill_seq_len",
        "max_cache_len",
        "lm_head_split_count",
    ):
        if key in config:
            try:
                value = int(config[key])
            except (TypeError, ValueError):
                errors.append(f"{key} must be an integer")
                continue
            if value <= 0:
                errors.append(f"{key} must be positive")

    _check_allowed(
        config,
        "attention_template",
        ALLOWED_ATTENTION_TEMPLATES,
        errors,
    )
    _check_allowed(config, "mlp_template", ALLOWED_MLP_TEMPLATES, errors)
    _check_allowed(
        config,
        "lm_head_template",
        ALLOWED_LM_HEAD_TEMPLATES,
        errors,
    )
    _check_allowed(
        config,
        "kv_cache_template",
        ALLOWED_KV_CACHE_TEMPLATES,
        errors,
    )
    _check_allowed(
        config,
        "generation_template",
        ALLOWED_GENERATION_TEMPLATES,
        errors,
    )

    if errors:
        raise ValueError(
            "invalid Buddy-TTNN Direct template config:\n- "
            + "\n- ".join(errors)
        )


def build_execution_plan(
    graph: LlamaModelGraph, config: dict[str, Any]
) -> dict[str, Any]:
    validate_llama_graph(graph)
    validate_template_config(config)
    _validate_graph_config_compatibility(graph, config)

    layer_templates = [
        {
            "layer_id": layer.layer_id,
            "templates": [
                "rmsnorm",
                config["attention_template"],
                "residual_add",
                "rmsnorm",
                config["mlp_template"],
                "residual_add",
            ],
        }
        for layer in graph.layers
    ]

    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "mode": graph.mode,
        "batch_size": graph.batch_size,
        "seq_len": graph.seq_len,
        "max_cache_len": graph.max_cache_len,
        "hidden_size": graph.hidden_size,
        "intermediate_size": graph.intermediate_size,
        "num_attention_heads": graph.num_attention_heads,
        "num_key_value_heads": graph.num_key_value_heads,
        "head_dim": graph.head_dim,
        "vocab_size": graph.vocab_size,
        "rms_norm_eps": graph.rms_norm_eps,
        "template_config": {
            "device": config["device"],
            "model": config["model"],
            "dtype_recipe": config["dtype_recipe"],
            "kv_cache_template": config["kv_cache_template"],
            "lm_head_split_count": int(config["lm_head_split_count"]),
        },
        "layers": layer_templates,
        "final": [
            "rmsnorm",
            config["lm_head_template"],
            config["generation_template"],
        ],
    }


def dump_execution_plan(plan: dict[str, Any], out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2) + "\n")


def load_execution_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def find_custom_fused_templates(plan: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for layer in plan.get("layers", []):
        for template in layer.get("templates", []):
            if template in CUSTOM_FUSED_REGION_TEMPLATES:
                found.append(template)
    for template in plan.get("final", []):
        if template in CUSTOM_FUSED_REGION_TEMPLATES:
            found.append(template)
    return sorted(set(found))


def _check_allowed(
    config: dict[str, Any],
    key: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    if key not in config:
        return
    value = config[key]
    if value not in allowed:
        errors.append(
            f"{key} must be one of {sorted(allowed)}; got {value!r}"
        )


def _validate_graph_config_compatibility(
    graph: LlamaModelGraph, config: dict[str, Any]
) -> None:
    errors: list[str] = []
    if graph.mode != "decode":
        errors.append("Phase 2 template planning only supports decode graphs")
    if graph.batch_size != int(config["batch_size"]):
        errors.append(
            "batch_size mismatch: "
            f"semantic={graph.batch_size}, config={config['batch_size']}"
        )
    if graph.seq_len != int(config["decode_seq_len"]):
        errors.append(
            "decode_seq_len mismatch: "
            f"semantic={graph.seq_len}, config={config['decode_seq_len']}"
        )
    if graph.max_cache_len != int(config["max_cache_len"]):
        errors.append(
            "max_cache_len mismatch: "
            f"semantic={graph.max_cache_len}, config={config['max_cache_len']}"
        )
    if (
        graph.generation_mode != "greedy"
        and config["generation_template"] == "device_argmax_greedy"
    ):
        errors.append(
            "device_argmax_greedy requires semantic generation_mode=greedy"
        )

    if errors:
        raise ValueError(
            "semantic graph and template config are incompatible:\n- "
            + "\n- ".join(errors)
        )
