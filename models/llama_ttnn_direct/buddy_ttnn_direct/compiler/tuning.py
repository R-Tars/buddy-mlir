from __future__ import annotations

import copy
from typing import Any, Mapping


OFFICIAL_LINEAR_OUTPUTS = "official_l1_sharded"
LM_HEAD_DRAM_CONCAT = "lm_head_dram_concat"
SUPPORTED_MEMORY_LAYOUTS = frozenset(
    (OFFICIAL_LINEAR_OUTPUTS, LM_HEAD_DRAM_CONCAT)
)

OFFICIAL_PROGRAM_CONFIG = "official"
SDPA_GRID_8X4_PROGRAM_CONFIG = "sdpa_grid_8x4"
SUPPORTED_PROGRAM_CONFIGS = frozenset(
    (OFFICIAL_PROGRAM_CONFIG, SDPA_GRID_8X4_PROGRAM_CONFIG)
)


def normalize_tuning_config(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("template_config.autotune must be an object")

    config = {
        "schema_version": int(value.get("schema_version", 1)),
        "memory_layout": str(
            value.get("memory_layout", OFFICIAL_LINEAR_OUTPUTS)
        ),
        "program_config": str(
            value.get("program_config", OFFICIAL_PROGRAM_CONFIG)
        ),
    }
    if config["schema_version"] != 1:
        raise ValueError("template_config.autotune.schema_version must be 1")
    if config["memory_layout"] not in SUPPORTED_MEMORY_LAYOUTS:
        raise ValueError(
            "unsupported autotune memory layout: "
            f"{config['memory_layout']}"
        )
    if config["program_config"] not in SUPPORTED_PROGRAM_CONFIGS:
        raise ValueError(
            "unsupported autotune program config: "
            f"{config['program_config']}"
        )
    return config


def apply_runtime_tuning(
    config: dict[str, Any],
    tuning: Any,
) -> dict[str, Any]:
    normalized = normalize_tuning_config(tuning)
    if normalized is None:
        return config

    result = copy.deepcopy(config)
    if (
        (result.get("template_config") or {}).get("dtype_recipe")
        == "all_bf16_correctness"
    ):
        _remove_parameter_dtype_overrides(result)
    if normalized["memory_layout"] == LM_HEAD_DRAM_CONCAT:
        _use_lm_head_dram_concat(result)
    if normalized["program_config"] == SDPA_GRID_8X4_PROGRAM_CONFIG:
        _use_sdpa_grid_8x4(result)
    result["autotune"] = copy.deepcopy(normalized)
    return result


def _use_lm_head_dram_concat(config: dict[str, Any]) -> None:
    dram = {"kind": "ttnn_memory_config", "name": "DRAM_MEMORY_CONFIG"}
    lm_head = config.get("lm_head") or {}
    lm_head["shard_output_memory_config"] = copy.deepcopy(dram)
    lm_head["concat_memory_config"] = copy.deepcopy(dram)


def _use_sdpa_grid_8x4(config: dict[str, Any]) -> None:
    attention = config.get("attention") or {}
    program_config = attention.get("sdpa_program_config")
    if not isinstance(program_config, dict):
        raise ValueError(
            "sdpa_grid_8x4 requires an official SDPA program config"
        )
    if program_config.get("kind") != "ttnn_sdpa_program_config":
        raise ValueError(
            "sdpa_grid_8x4 requires a ttnn_sdpa_program_config descriptor"
        )
    program_config["core_grid"] = [8, 4]


def _remove_parameter_dtype_overrides(config: dict[str, Any]) -> None:
    layer_overrides = (config.get("mlp") or {}).get("layer_overrides") or {}
    for override in layer_overrides.values():
        if isinstance(override, dict):
            override.pop("parameter_intermediate_dtype", None)
