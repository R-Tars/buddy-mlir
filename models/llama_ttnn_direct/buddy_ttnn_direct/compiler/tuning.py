from __future__ import annotations

import copy
from typing import Any, Mapping

from ..autotune.space import (
    LM_HEAD_DRAM_CONCAT,
    OFFICIAL_LINEAR_OUTPUTS,
    OFFICIAL_PROGRAM_CONFIG,
    SDPA_GRID_8X4_PROGRAM_CONFIG,
    SearchSpaceConfig,
    adapt_legacy_presets,
)

SUPPORTED_MEMORY_LAYOUTS = frozenset((OFFICIAL_LINEAR_OUTPUTS, LM_HEAD_DRAM_CONCAT))

SUPPORTED_PROGRAM_CONFIGS = frozenset(
    (OFFICIAL_PROGRAM_CONFIG, SDPA_GRID_8X4_PROGRAM_CONFIG)
)


def normalize_tuning_config(
    value: Any,
) -> dict[str, Any] | SearchSpaceConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("template_config.autotune must be an object")

    schema_version = int(value.get("schema_version", 1))
    if schema_version == 2:
        return SearchSpaceConfig.from_dict(value)
    config = {
        "schema_version": schema_version,
        "memory_layout": str(value.get("memory_layout", OFFICIAL_LINEAR_OUTPUTS)),
        "program_config": str(value.get("program_config", OFFICIAL_PROGRAM_CONFIG)),
    }
    if config["schema_version"] != 1:
        raise ValueError("template_config.autotune.schema_version must be 1")
    if config["memory_layout"] not in SUPPORTED_MEMORY_LAYOUTS:
        raise ValueError(
            "unsupported autotune memory layout: " f"{config['memory_layout']}"
        )
    if config["program_config"] not in SUPPORTED_PROGRAM_CONFIGS:
        raise ValueError(
            "unsupported autotune program config: " f"{config['program_config']}"
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
    if (result.get("template_config") or {}).get(
        "dtype_recipe"
    ) == "all_bf16_correctness":
        _remove_parameter_dtype_overrides(result)
    if isinstance(normalized, SearchSpaceConfig):
        space = normalized
    else:
        space = adapt_legacy_presets(
            result,
            memory_layout=normalized["memory_layout"],
            program_config=normalized["program_config"],
        )
    return space.apply_to_runtime_config(result)


def _remove_parameter_dtype_overrides(config: dict[str, Any]) -> None:
    layer_overrides = (config.get("mlp") or {}).get("layer_overrides") or {}
    for override in layer_overrides.values():
        if isinstance(override, dict):
            override.pop("parameter_intermediate_dtype", None)
