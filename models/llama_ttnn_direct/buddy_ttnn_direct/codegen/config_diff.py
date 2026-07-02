from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import write_json


PARITY_SECTIONS = (
    "dtype_recipe",
    "compute_fidelity",
    "program_config",
    "memory_config",
    "core_grid",
    "lm_head",
    "paged_attention",
)


def default_official_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "official_p150a_llama31_8b_config_seed.json"
    )


def diff_official_config(
    ours: str | Path | dict[str, Any],
    official: str | Path | dict[str, Any],
) -> dict[str, Any]:
    ours_raw = _load_json_or_dict(ours)
    official_raw = _load_json_or_dict(official)
    ours_view = build_config_parity_view(ours_raw)
    official_view = build_config_parity_view(official_raw)

    ours_flat = _flatten(ours_view["parity_config"])
    official_flat = _flatten(official_view["parity_config"])
    all_paths = sorted(set(ours_flat).union(official_flat))

    missing_fields = []
    mismatched_fields = []
    extra_fields = []
    matching_fields = []
    for path in all_paths:
        ours_value = ours_flat.get(path)
        official_value = official_flat.get(path)
        section = path.split(".", 1)[0]
        if _is_missing(ours_value) and not _is_missing(official_value):
            missing_fields.append(
                {
                    "section": section,
                    "path": path,
                    "ours": ours_value,
                    "official": official_value,
                }
            )
            continue
        if not _is_missing(ours_value) and _is_missing(official_value):
            extra_fields.append(
                {
                    "section": section,
                    "path": path,
                    "ours": ours_value,
                    "official": official_value,
                }
            )
            continue
        if ours_value != official_value:
            mismatched_fields.append(
                {
                    "section": section,
                    "path": path,
                    "ours": ours_value,
                    "official": official_value,
                }
            )
            continue
        matching_fields.append(path)

    section_summaries = {
        section: _section_summary(
            section,
            missing_fields,
            mismatched_fields,
            extra_fields,
            matching_fields,
        )
        for section in PARITY_SECTIONS
    }
    issue_count = (
        len(missing_fields) + len(mismatched_fields) + len(extra_fields)
    )
    return {
        "schema_version": 1,
        "status": "match" if issue_count == 0 else "diff_found",
        "summary": {
            "missing_count": len(missing_fields),
            "mismatch_count": len(mismatched_fields),
            "extra_count": len(extra_fields),
            "matching_count": len(matching_fields),
            "issue_count": issue_count,
        },
        "ours": {
            "source_format": ours_view["source_format"],
            "model_name": ours_view.get("model_name"),
            "parity_config": ours_view["parity_config"],
        },
        "official": {
            "source_format": official_view["source_format"],
            "model_name": official_view.get("model_name"),
            "source": official_raw.get("source"),
            "parity_config": official_view["parity_config"],
        },
        "missing_fields": missing_fields,
        "mismatched_fields": mismatched_fields,
        "extra_fields": extra_fields,
        "matching_fields": matching_fields,
        "sections": section_summaries,
    }


def dump_config_diff(diff: dict[str, Any], out: str | Path) -> None:
    write_json(out, diff)


def build_config_parity_view(config: dict[str, Any]) -> dict[str, Any]:
    if "parity_config" in config:
        parity_config = _normalize_parity_config(config["parity_config"])
        return {
            "source_format": "normalized_parity_config",
            "model_name": config.get("model_name") or config.get("model"),
            "parity_config": parity_config,
        }
    parity_config = _normalize_generated_config(config)
    return {
        "source_format": "generated_ttnn_direct_config",
        "model_name": config.get("model_name") or config.get("model"),
        "parity_config": parity_config,
    }


def _normalize_generated_config(config: dict[str, Any]) -> dict[str, Any]:
    template_config = _dict(config.get("template_config"))
    attention = _dict(config.get("attention"))
    mlp = _dict(config.get("mlp"))
    lm_head = _dict(config.get("lm_head"))
    embedding = _dict(config.get("embedding"))
    rms_norm = _dict(config.get("rms_norm"))
    generation = _dict(config.get("generation"))
    kv_cache = _dict(config.get("kv_cache"))

    return _normalize_parity_config(
        {
            "dtype_recipe": {
                "recipe": _first_non_missing(
                    template_config.get("dtype_recipe"),
                    config.get("dtype_recipe"),
                ),
                "embedding": embedding.get("output_dtype"),
                "rms_norm": rms_norm.get("output_dtype"),
                "attention_qkv": attention.get("qkv_output_dtype"),
                "attention_o_proj": attention.get("o_proj_output_dtype"),
                "mlp_intermediate": mlp.get("intermediate_dtype"),
                "mlp_output": mlp.get("output_dtype"),
                "lm_head": lm_head.get("output_dtype"),
                "kv_cache": _first_non_missing(
                    kv_cache.get("dtype"),
                    _get(config, "parameter_config.kv_cache.dtype"),
                ),
            },
            "compute_fidelity": {
                "attention_qkv": _compute_fidelity(
                    attention.get("qkv_compute_kernel_config")
                ),
                "attention_sdpa": _compute_fidelity(
                    attention.get("sdpa_compute_kernel_config")
                ),
                "attention_o_proj": _compute_fidelity(
                    attention.get("o_proj_compute_kernel_config")
                ),
                "mlp": _compute_fidelity(
                    mlp.get("compute_kernel_config")
                ),
                "lm_head": _compute_fidelity(
                    lm_head.get("compute_kernel_config")
                ),
            },
            "program_config": {
                "attention_qkv": attention.get("qkv_program_config"),
                "attention_sdpa": attention.get("sdpa_program_config"),
                "attention_o_proj": attention.get("o_proj_program_config"),
                "mlp_gate": mlp.get("gate_program_config"),
                "mlp_up": mlp.get("up_program_config"),
                "mlp_down": mlp.get("down_program_config"),
                "lm_head_shards": lm_head.get("program_configs"),
            },
            "memory_config": {
                "embedding": embedding.get("output_memory_config"),
                "rms_norm": rms_norm.get("output_memory_config"),
                "attention_qkv": attention.get("qkv_output_memory_config"),
                "attention_heads": attention.get("qkv_heads_memory_config"),
                "attention_sdpa": attention.get("sdpa_output_memory_config"),
                "attention_concat_heads": attention.get(
                    "concat_heads_output_memory_config"
                ),
                "attention_o_proj": attention.get("o_proj_output_memory_config"),
                "mlp_gate": mlp.get("gate_output_memory_config"),
                "mlp_up": mlp.get("up_output_memory_config"),
                "mlp_down": mlp.get("down_output_memory_config"),
                "lm_head_output": lm_head.get("output_memory_config"),
                "lm_head_concat": lm_head.get("concat_memory_config"),
                "kv_cache": kv_cache.get("memory_config"),
            },
            "core_grid": {
                "attention": _first_non_missing(
                    attention.get("core_grid"),
                    template_config.get("attention_core_grid"),
                ),
                "mlp": _first_non_missing(
                    mlp.get("core_grid"),
                    template_config.get("mlp_core_grid"),
                ),
                "lm_head": _first_non_missing(
                    lm_head.get("core_grid"),
                    template_config.get("lm_head_core_grid"),
                ),
            },
            "lm_head": {
                "template": lm_head.get("template"),
                "split_count": lm_head.get("split_count"),
                "split_axis": lm_head.get("split_axis"),
                "retain_logits": lm_head.get("retain_logits"),
                "shard_count": _list_len(lm_head.get("splits")),
                "argmax_strategy": _argmax_strategy(lm_head, generation),
            },
            "paged_attention": {
                "enabled": attention.get("template")
                == "official_paged_attention_decode",
                "scale": attention.get("scale"),
                "max_cache_len": config.get("max_cache_len"),
                "page_block_size": kv_cache.get("page_block_size"),
                "qkv_heads_memory_config": attention.get(
                    "qkv_heads_memory_config"
                ),
                "sdpa_program_config": attention.get("sdpa_program_config"),
                "sdpa_memory_config": attention.get("sdpa_output_memory_config"),
            },
        }
    )


def _normalize_parity_config(parity_config: Any) -> dict[str, Any]:
    source = _dict(parity_config)
    return {
        section: _dict(source.get(section))
        for section in PARITY_SECTIONS
    }


def _compute_fidelity(value: Any) -> Any:
    if isinstance(value, dict):
        return _first_non_missing(
            value.get("compute_fidelity"),
            value.get("math_fidelity"),
            value.get("fidelity"),
        )
    return value


def _argmax_strategy(
    lm_head: dict[str, Any],
    generation: dict[str, Any],
) -> str | None:
    mode = generation.get("mode")
    retain_logits = bool(lm_head.get("retain_logits", False))
    if mode == "greedy" and not retain_logits:
        return "full_logits_concat_argmax"
    if mode == "full_logits" or retain_logits:
        return "retain_full_logits"
    return None


def _section_summary(
    section: str,
    missing_fields: list[dict[str, Any]],
    mismatched_fields: list[dict[str, Any]],
    extra_fields: list[dict[str, Any]],
    matching_fields: list[str],
) -> dict[str, int]:
    prefix = section + "."
    return {
        "missing_count": sum(
            1 for field in missing_fields if field["section"] == section
        ),
        "mismatch_count": sum(
            1 for field in mismatched_fields if field["section"] == section
        ),
        "extra_count": sum(
            1 for field in extra_fields if field["section"] == section
        ),
        "matching_count": sum(
            1 for path in matching_fields if path == section or path.startswith(prefix)
        ),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return {prefix: value}
        output: dict[str, Any] = {}
        for index, item in enumerate(value):
            output.update(_flatten(item, f"{prefix}[{index}]"))
        return output
    return {prefix: value}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, list):
        return not value or all(_is_missing(item) for item in value)
    if isinstance(value, dict):
        return not value or all(_is_missing(item) for item in value.values())
    return False


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _get(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _first_non_missing(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return None


def _list_len(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def _load_json_or_dict(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return json.loads(Path(value).read_text())
