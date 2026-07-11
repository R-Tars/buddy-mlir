from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable


SEARCH_KEYS = ("lm_head_split_count", "generation_template")
OPTIONAL_SEARCH_DEFAULTS = {
    "mlp_intermediate_dtype": [None],
    "attention_sdpa_output_memory_config": [None],
    "attention_concat_heads_output_memory_config": [None],
}
DEFAULT_SEARCH_SPACE = {
    "lm_head_split_count": [1, 2, 4, 8, 16],
    "generation_template": ["full_logits", "device_argmax_greedy"],
    **OPTIONAL_SEARCH_DEFAULTS,
}
ALLOWED_GENERATION_TEMPLATES = {"full_logits", "device_argmax_greedy"}
ALLOWED_MLP_INTERMEDIATE_DTYPES = {
    None,
    "bfloat4_b",
    "bfloat8_b",
    "bfloat16",
}
ALLOWED_MEMORY_CONFIGS = {None, "default", "l1"}


def load_search_space(path: str | Path) -> dict[str, list[Any]]:
    space = json.loads(Path(path).read_text())
    validate_search_space(space)
    return {
        "lm_head_split_count": [
            int(value) for value in space["lm_head_split_count"]
        ],
        "generation_template": [
            str(value) for value in space["generation_template"]
        ],
        "mlp_intermediate_dtype": [
            _none_or_str(value)
            for value in space.get(
                "mlp_intermediate_dtype",
                OPTIONAL_SEARCH_DEFAULTS["mlp_intermediate_dtype"],
            )
        ],
        "attention_sdpa_output_memory_config": [
            _none_or_str(value)
            for value in space.get(
                "attention_sdpa_output_memory_config",
                OPTIONAL_SEARCH_DEFAULTS["attention_sdpa_output_memory_config"],
            )
        ],
        "attention_concat_heads_output_memory_config": [
            _none_or_str(value)
            for value in space.get(
                "attention_concat_heads_output_memory_config",
                OPTIONAL_SEARCH_DEFAULTS[
                    "attention_concat_heads_output_memory_config"
                ],
            )
        ],
    }


def validate_search_space(space: dict[str, Any]) -> None:
    errors: list[str] = []
    for key in SEARCH_KEYS:
        values = space.get(key)
        if not isinstance(values, list) or not values:
            errors.append(f"{key} must be a non-empty list")

    for value in space.get("lm_head_split_count", []):
        try:
            split_count = int(value)
        except (TypeError, ValueError):
            errors.append("lm_head_split_count values must be integers")
            continue
        if split_count <= 0:
            errors.append("lm_head_split_count values must be positive")

    for value in space.get("generation_template", []):
        if value not in ALLOWED_GENERATION_TEMPLATES:
            errors.append(
                "generation_template values must be one of "
                f"{sorted(ALLOWED_GENERATION_TEMPLATES)}; got {value!r}"
            )

    _check_optional_values(
        space,
        "mlp_intermediate_dtype",
        ALLOWED_MLP_INTERMEDIATE_DTYPES,
        errors,
    )
    _check_optional_values(
        space,
        "attention_sdpa_output_memory_config",
        ALLOWED_MEMORY_CONFIGS,
        errors,
    )
    _check_optional_values(
        space,
        "attention_concat_heads_output_memory_config",
        ALLOWED_MEMORY_CONFIGS,
        errors,
    )

    if errors:
        raise ValueError(
            "invalid Buddy-TTNN Direct search space:\n- "
            + "\n- ".join(errors)
        )


def enumerate_candidate_configs(
    base_config: dict[str, Any],
    space: dict[str, Iterable[Any]],
) -> list[dict[str, Any]]:
    normalized_space = {
        "lm_head_split_count": list(space["lm_head_split_count"]),
        "generation_template": list(space["generation_template"]),
        "mlp_intermediate_dtype": list(
            space.get(
                "mlp_intermediate_dtype",
                OPTIONAL_SEARCH_DEFAULTS["mlp_intermediate_dtype"],
            )
        ),
        "attention_sdpa_output_memory_config": list(
            space.get(
                "attention_sdpa_output_memory_config",
                OPTIONAL_SEARCH_DEFAULTS["attention_sdpa_output_memory_config"],
            )
        ),
        "attention_concat_heads_output_memory_config": list(
            space.get(
                "attention_concat_heads_output_memory_config",
                OPTIONAL_SEARCH_DEFAULTS[
                    "attention_concat_heads_output_memory_config"
                ],
            )
        ),
    }
    validate_search_space(normalized_space)
    candidates: list[dict[str, Any]] = []
    for (
        split_count,
        generation_template,
        mlp_intermediate_dtype,
        attention_sdpa_output_memory_config,
        attention_concat_heads_output_memory_config,
    ) in itertools.product(
        normalized_space["lm_head_split_count"],
        normalized_space["generation_template"],
        normalized_space["mlp_intermediate_dtype"],
        normalized_space["attention_sdpa_output_memory_config"],
        normalized_space["attention_concat_heads_output_memory_config"],
    ):
        config = dict(base_config)
        config["lm_head_split_count"] = int(split_count)
        config["generation_template"] = str(generation_template)
        config["mlp_intermediate_dtype"] = _none_or_str(
            mlp_intermediate_dtype
        )
        config["attention_sdpa_output_memory_config"] = _none_or_str(
            attention_sdpa_output_memory_config
        )
        config["attention_concat_heads_output_memory_config"] = _none_or_str(
            attention_concat_heads_output_memory_config
        )
        candidates.append(
            {
                "id": candidate_id(
                    int(split_count),
                    str(generation_template),
                    mlp_intermediate_dtype=_none_or_str(
                        mlp_intermediate_dtype
                    ),
                    attention_sdpa_output_memory_config=_none_or_str(
                        attention_sdpa_output_memory_config
                    ),
                    attention_concat_heads_output_memory_config=_none_or_str(
                        attention_concat_heads_output_memory_config
                    ),
                ),
                "config": config,
                "lm_head_split_count": int(split_count),
                "generation_template": str(generation_template),
                "mlp_intermediate_dtype": _none_or_str(
                    mlp_intermediate_dtype
                ),
                "attention_sdpa_output_memory_config": _none_or_str(
                    attention_sdpa_output_memory_config
                ),
                "attention_concat_heads_output_memory_config": _none_or_str(
                    attention_concat_heads_output_memory_config
                ),
            }
        )
    return candidates


def candidate_id(
    split_count: int,
    generation_template: str,
    *,
    mlp_intermediate_dtype: str | None = None,
    attention_sdpa_output_memory_config: str | None = None,
    attention_concat_heads_output_memory_config: str | None = None,
) -> str:
    suffix = (
        "argmax"
        if generation_template == "device_argmax_greedy"
        else "full_logits"
    )
    parts = [f"lm_split_{split_count}", suffix]
    if mlp_intermediate_dtype is not None:
        parts.append(f"mlp_{mlp_intermediate_dtype}")
    if attention_sdpa_output_memory_config not in (None, "default"):
        parts.append(f"sdpa_{attention_sdpa_output_memory_config}")
    if attention_concat_heads_output_memory_config not in (None, "default"):
        parts.append(
            f"concat_{attention_concat_heads_output_memory_config}"
        )
    return "_".join(parts)


def _check_optional_values(
    space: dict[str, Any],
    key: str,
    allowed: set[Any],
    errors: list[str],
) -> None:
    if key not in space:
        return
    values = space.get(key)
    if not isinstance(values, list) or not values:
        errors.append(f"{key} must be a non-empty list when provided")
        return
    for value in values:
        normalized = _none_or_str(value)
        if normalized not in allowed:
            errors.append(
                f"{key} values must be one of {sorted(map(str, allowed))}; "
                f"got {value!r}"
            )


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)
