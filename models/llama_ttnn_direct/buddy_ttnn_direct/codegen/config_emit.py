from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..semantic.graph import LlamaModelGraph
from ..semantic.validate import validate_llama_graph
from ..templates.lm_head import build_lm_head_split_ranges


SEED_RECIPE = "official_like_performance_seed"


def emit_parameter_config(
    graph: LlamaModelGraph,
    *,
    recipe: str = SEED_RECIPE,
    lm_head_split_count: int = 8,
    kv_page_block_size: int = 32,
) -> dict[str, Any]:
    validate_llama_graph(graph)
    if recipe != SEED_RECIPE:
        raise ValueError(f"unsupported parameter metadata recipe: {recipe}")
    if lm_head_split_count <= 0:
        raise ValueError("lm_head_split_count must be positive")
    if kv_page_block_size <= 0:
        raise ValueError("kv_page_block_size must be positive")

    weights: dict[str, dict[str, Any]] = {}
    _add_weight(
        weights,
        "model.embed_tokens.weight",
        role="embedding",
        layer_id=None,
        target_dtype="bfloat16",
        packing="none",
        layout="row_major",
    )

    for layer in graph.layers:
        layer_id = layer.layer_id
        _add_weight(
            weights,
            layer.input_norm.weight,
            role="input_norm",
            layer_id=layer_id,
            target_dtype="bfloat16",
            packing="none",
            layout="row_major",
        )
        _add_weight(
            weights,
            layer.attention.q_proj,
            role="q_proj",
            layer_id=layer_id,
            target_dtype="bfloat8_b",
            packing="qkv_pack",
            layout="tile",
        )
        _add_weight(
            weights,
            layer.attention.k_proj,
            role="k_proj",
            layer_id=layer_id,
            target_dtype="bfloat8_b",
            packing="qkv_pack",
            layout="tile",
        )
        _add_weight(
            weights,
            layer.attention.v_proj,
            role="v_proj",
            layer_id=layer_id,
            target_dtype="bfloat8_b",
            packing="qkv_pack",
            layout="tile",
        )
        _add_weight(
            weights,
            layer.attention.o_proj,
            role="o_proj",
            layer_id=layer_id,
            target_dtype="bfloat8_b",
            packing="none",
            layout="tile",
        )
        _add_weight(
            weights,
            layer.post_attention_norm.weight,
            role="post_attention_norm",
            layer_id=layer_id,
            target_dtype="bfloat16",
            packing="none",
            layout="row_major",
        )
        _add_weight(
            weights,
            layer.mlp.gate_proj,
            role="mlp_gate",
            layer_id=layer_id,
            target_dtype="bfloat4_b",
            packing="gate_up_group",
            layout="tile",
        )
        _add_weight(
            weights,
            layer.mlp.up_proj,
            role="mlp_up",
            layer_id=layer_id,
            target_dtype="bfloat4_b",
            packing="gate_up_group",
            layout="tile",
        )
        _add_weight(
            weights,
            layer.mlp.down_proj,
            role="mlp_down",
            layer_id=layer_id,
            target_dtype="bfloat8_b",
            packing="none",
            layout="tile",
        )

    _add_weight(
        weights,
        graph.final_norm.weight,
        role="final_norm",
        layer_id=None,
        target_dtype="bfloat16",
        packing="none",
        layout="row_major",
    )
    _add_weight(
        weights,
        graph.lm_head.weight,
        role="lm_head",
        layer_id=None,
        target_dtype="bfloat8_b",
        packing="vocab_split",
        layout="tile",
        extra={"tied_to_embedding": graph.lm_head.tied_to_embedding},
    )

    return {
        "schema_version": 1,
        "model_name": graph.model_name,
        "mode": graph.mode,
        "recipe": recipe,
        "weights": weights,
        "activations": {
            "target_dtype": "bfloat16",
            "layout": "tile",
        },
        "lm_head": {
            "weight": graph.lm_head.weight,
            "split_count": lm_head_split_count,
            "split_axis": "vocab",
            "splits": build_lm_head_split_ranges(
                graph.vocab_size, lm_head_split_count
            ),
            "target_dtype": "bfloat8_b",
            "tied_to_embedding": graph.lm_head.tied_to_embedding,
        },
        "kv_cache": {
            "policy": "paged",
            "page_block_size": kv_page_block_size,
            "dtype": "bfloat8_b",
        },
    }


def dump_parameter_config(config: dict[str, Any], out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2) + "\n")


def load_parameter_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def parameter_config_dry_run_report(
    graph: LlamaModelGraph,
    *,
    recipe: str = SEED_RECIPE,
    lm_head_split_count: int = 8,
    kv_page_block_size: int = 32,
) -> dict[str, Any]:
    config = emit_parameter_config(
        graph,
        recipe=recipe,
        lm_head_split_count=lm_head_split_count,
        kv_page_block_size=kv_page_block_size,
    )
    return {
        "dry_run": True,
        "model_name": config["model_name"],
        "mode": config["mode"],
        "recipe": config["recipe"],
        "weight_count": len(config["weights"]),
        "lm_head_split_count": config["lm_head"]["split_count"],
        "kv_page_block_size": config["kv_cache"]["page_block_size"],
    }


def _add_weight(
    weights: dict[str, dict[str, Any]],
    name: str,
    *,
    role: str,
    layer_id: int | None,
    target_dtype: str,
    packing: str,
    layout: str,
    extra: dict[str, Any] | None = None,
) -> None:
    entry = {
        "role": role,
        "layer_id": layer_id,
        "target_dtype": target_dtype,
        "packing": packing,
        "layout": layout,
        "memory_config": "dram",
    }
    if extra:
        entry.update(extra)

    if name not in weights:
        weights[name] = entry
        return

    existing = weights[name]
    shared_roles = existing.setdefault("shared_roles", [existing["role"]])
    if role not in shared_roles:
        shared_roles.append(role)
    existing["role"] = "shared"
    existing.setdefault("target_dtype", target_dtype)
    existing.setdefault("packing", packing)
    existing.setdefault("layout", layout)
    existing.setdefault("memory_config", "dram")
    if extra:
        existing.update(extra)
