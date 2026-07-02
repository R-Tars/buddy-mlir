from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable


SEARCH_KEYS = ("lm_head_split_count", "generation_template")
DEFAULT_SEARCH_SPACE = {
    "lm_head_split_count": [1, 2, 4, 8, 16],
    "generation_template": ["full_logits", "device_argmax_greedy"],
}
ALLOWED_GENERATION_TEMPLATES = {"full_logits", "device_argmax_greedy"}


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
    }
    validate_search_space(normalized_space)
    candidates: list[dict[str, Any]] = []
    for split_count, generation_template in itertools.product(
        normalized_space["lm_head_split_count"],
        normalized_space["generation_template"],
    ):
        config = dict(base_config)
        config["lm_head_split_count"] = int(split_count)
        config["generation_template"] = str(generation_template)
        candidates.append(
            {
                "id": candidate_id(
                    int(split_count), str(generation_template)
                ),
                "config": config,
                "lm_head_split_count": int(split_count),
                "generation_template": str(generation_template),
            }
        )
    return candidates


def candidate_id(split_count: int, generation_template: str) -> str:
    suffix = (
        "argmax"
        if generation_template == "device_argmax_greedy"
        else "full_logits"
    )
    return f"lm_split_{split_count}_{suffix}"
