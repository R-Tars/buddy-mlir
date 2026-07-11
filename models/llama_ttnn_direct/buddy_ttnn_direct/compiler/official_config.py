from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


P150A_LLAMA31_8B_B32_PERFORMANCE = "p150a_llama31_8b_b32_performance"
SUPPORTED_OFFICIAL_CONFIG_PROFILES = frozenset(
    (P150A_LLAMA31_8B_B32_PERFORMANCE,)
)


def official_config_reference_path(profile: str) -> Path:
    if profile not in SUPPORTED_OFFICIAL_CONFIG_PROFILES:
        raise ValueError(f"unsupported official config profile: {profile}")
    return (
        Path(__file__).resolve().parents[1]
        / "reference"
        / "official_p150a_llama31_8b_config_seed.json"
    )


def load_official_config_profile(profile: str) -> dict[str, Any]:
    path = official_config_reference_path(profile)
    payload = json.loads(path.read_text())
    if payload.get("profile_id") != profile:
        raise ValueError(
            f"official config profile mismatch: expected {profile}, "
            f"observed {payload.get('profile_id')}"
        )
    runtime_config = payload.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise ValueError("official config reference must contain runtime_config")
    return payload


def apply_official_config_profile(
    config: dict[str, Any],
    profile: str | None,
) -> dict[str, Any]:
    if not profile:
        return config
    official = load_official_config_profile(profile)
    merged = _deep_merge(config, official["runtime_config"])

    # Generation behavior belongs to the selected Buddy template, not the
    # imported hardware profile.
    merged["lm_head"].update(
        {
            key: copy.deepcopy(config["lm_head"][key])
            for key in (
                "template",
                "split_count",
                "split_axis",
                "retain_logits",
                "argmax_strategy",
                "splits",
            )
        }
    )
    merged["generation"] = copy.deepcopy(config["generation"])
    _adapt_lm_head_program_configs(merged["lm_head"])
    merged["official_config_profile"] = profile
    merged["official_config_source"] = copy.deepcopy(official["source"])
    merged["official_parity_config"] = copy.deepcopy(
        official["parity_config"]
    )
    return merged


def official_layer_dtype_overrides(
    profile: str | None,
    *,
    dtype_recipe: str | None = None,
) -> dict[int, dict[str, str]]:
    if not profile:
        return {}
    if dtype_recipe == "all_bf16_correctness":
        return {}
    official = load_official_config_profile(profile)
    mlp = official["runtime_config"].get("mlp") or {}
    raw_overrides = mlp.get("layer_overrides") or {}
    result: dict[int, dict[str, str]] = {}
    for layer_id, values in raw_overrides.items():
        parameter_dtype = values.get("parameter_intermediate_dtype")
        if parameter_dtype is not None:
            result[int(layer_id)] = {
                "mlp_intermediate": str(parameter_dtype)
            }
    return result


def official_weight_memory_overrides(
    profile: str | None,
    *,
    lm_head_split_count: int | None = None,
    vocab_size: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not profile:
        return {}
    official = load_official_config_profile(profile)
    parameter_config = official.get("parameter_config") or {}
    raw_configs = parameter_config.get("weight_memory_config") or {}
    role_sources = {
        "q_proj": "attention_qkv",
        "k_proj": "attention_qkv",
        "v_proj": "attention_qkv",
        "o_proj": "attention_o_proj",
        "mlp_gate": "mlp_gate",
        "mlp_up": "mlp_up",
        "mlp_down": "mlp_down",
        "lm_head": "lm_head",
    }
    result = {
        role: copy.deepcopy(raw_configs[source])
        for role, source in role_sources.items()
        if source in raw_configs
    }
    if (
        "lm_head" in result
        and lm_head_split_count is not None
        and vocab_size is not None
    ):
        result["lm_head"]["n"] = math.ceil(
            int(vocab_size) / int(lm_head_split_count)
        )
    return result


def _adapt_lm_head_program_configs(lm_head: dict[str, Any]) -> None:
    splits = lm_head.get("splits")
    program_configs = lm_head.get("program_configs")
    if not isinstance(splits, list) or not splits:
        return
    if not isinstance(program_configs, list) or not program_configs:
        return
    template = program_configs[0]
    if not isinstance(template, dict):
        return
    if template.get("kind") != "ttnn_matmul_dram_sharded_program_config":
        return
    core_grid = lm_head.get("core_grid", [8, 8])
    core_count = math.prod(int(value) for value in core_grid)

    adapted = []
    for split in splits:
        vocab_start = int(split["vocab_start"])
        vocab_end = int(split["vocab_end"])
        shard_width = vocab_end - vocab_start
        descriptor = copy.deepcopy(template)
        descriptor["per_core_N"] = math.ceil(shard_width / (32 * core_count))
        adapted.append(descriptor)
    lm_head["program_configs"] = adapted


def _deep_merge(
    base: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
