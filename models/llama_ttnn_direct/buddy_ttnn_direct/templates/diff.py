from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .mlp_decode import official_gated_mlp_decode_op_sequence


ATTENTION_DECODE_OPS = [
    "linear.qkv_packed",
    "nlp_create_qkv_heads_decode",
    "rotary_embedding_decode",
    "paged_update_cache",
    "paged_scaled_dot_product_attention_decode",
    "nlp_concat_heads_decode",
    "linear.o_proj",
]

GATED_MLP_DECODE_OPS = official_gated_mlp_decode_op_sequence()

TEMPLATE_TO_OPS = {
    "official_paged_attention_decode": ATTENTION_DECODE_OPS,
    "official_gated_mlp_decode": GATED_MLP_DECODE_OPS,
}

FINAL_TEMPLATE_TO_OPS = {
    "official_split_lm_head": ["split_lm_head"],
    "device_argmax_greedy": ["argmax_or_sampling"],
    "full_logits": ["argmax_or_sampling"],
}


def load_official_template(path: str | Path) -> dict[str, list[str]]:
    template = json.loads(Path(path).read_text())
    validate_official_template(template)
    return template


def validate_official_template(template: dict[str, Any]) -> None:
    errors: list[str] = []
    for key in ("layer_ops", "final_ops"):
        value = template.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{key} must be a non-empty list")
        elif not all(isinstance(item, str) for item in value):
            errors.append(f"{key} must contain only strings")
    if errors:
        raise ValueError(
            "invalid official decode template:\n- " + "\n- ".join(errors)
        )


def expand_plan_ops(plan: dict[str, Any]) -> dict[str, list[str]]:
    layers = plan.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("plan must contain a non-empty layers list")
    first_layer = layers[0]
    templates = first_layer.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError("plan layer must contain a non-empty templates list")

    return {
        "layer_ops": expand_layer_templates(templates),
        "final_ops": expand_final_templates(plan.get("final", [])),
    }


def expand_layer_templates(templates: list[str]) -> list[str]:
    ops: list[str] = []
    rmsnorm_count = 0
    for template in templates:
        if template == "rmsnorm":
            rmsnorm_count += 1
            if rmsnorm_count == 1:
                ops.append("rmsnorm.attn")
            elif rmsnorm_count == 2:
                ops.append("rmsnorm.mlp")
            else:
                ops.append("rmsnorm")
            continue
        if template in TEMPLATE_TO_OPS:
            ops.extend(TEMPLATE_TO_OPS[template])
            continue
        ops.append(template)
    return ops


def expand_final_templates(templates: list[str]) -> list[str]:
    ops: list[str] = []
    for template in templates:
        if template == "rmsnorm":
            ops.append("rmsnorm.final")
            continue
        if template in FINAL_TEMPLATE_TO_OPS:
            ops.extend(FINAL_TEMPLATE_TO_OPS[template])
            continue
        ops.append(template)
    return ops


def diff_plan_against_official(
    plan: dict[str, Any], official_template: dict[str, Any]
) -> dict[str, Any]:
    validate_official_template(official_template)
    expanded = expand_plan_ops(plan)
    layer_diff = _diff_ops(expanded["layer_ops"], official_template["layer_ops"])
    final_diff = _diff_ops(expanded["final_ops"], official_template["final_ops"])
    return {
        "schema_version": 1,
        "expanded": expanded,
        "reference": {
            "layer_ops": list(official_template["layer_ops"]),
            "final_ops": list(official_template["final_ops"]),
        },
        "missing_ops": layer_diff["missing_ops"] + final_diff["missing_ops"],
        "extra_ops": layer_diff["extra_ops"] + final_diff["extra_ops"],
        "order_mismatch": layer_diff["order_mismatch"]
        + final_diff["order_mismatch"],
        "sections": {
            "layer": layer_diff,
            "final": final_diff,
        },
    }


def dump_plan_diff(diff: dict[str, Any], out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diff, indent=2) + "\n")


def _diff_ops(ours: list[str], official: list[str]) -> dict[str, Any]:
    ours_counts = Counter(ours)
    official_counts = Counter(official)
    missing = _counter_difference(official_counts, ours_counts, official)
    extra = _counter_difference(ours_counts, official_counts, ours)
    order_mismatch = []
    if not missing and not extra and ours != official:
        for index, (ours_op, official_op) in enumerate(zip(ours, official)):
            if ours_op != official_op:
                order_mismatch.append(
                    {
                        "index": index,
                        "ours": ours_op,
                        "official": official_op,
                    }
                )
    return {
        "ours": list(ours),
        "official": list(official),
        "missing_ops": missing,
        "extra_ops": extra,
        "order_mismatch": order_mismatch,
    }


def _counter_difference(
    left: Counter[str], right: Counter[str], order: list[str]
) -> list[str]:
    remaining = left - right
    output: list[str] = []
    for op in order:
        if remaining[op] > 0:
            output.append(op)
            remaining[op] -= 1
    return output
