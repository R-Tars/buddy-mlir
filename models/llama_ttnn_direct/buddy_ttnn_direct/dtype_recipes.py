from __future__ import annotations


PERFORMANCE_RECIPE = "official_like_performance_seed"
CORRECTNESS_RECIPE = "all_bf16_correctness"
SEED_RECIPE = PERFORMANCE_RECIPE
SUPPORTED_RECIPES = frozenset((PERFORMANCE_RECIPE, CORRECTNESS_RECIPE))


def recipe_dtypes(recipe: str) -> dict[str, str]:
    if recipe not in SUPPORTED_RECIPES:
        raise ValueError(f"unsupported parameter metadata recipe: {recipe}")
    if recipe == CORRECTNESS_RECIPE:
        return {
            "attention_qkv": "bfloat16",
            "attention_output": "bfloat16",
            "mlp_intermediate": "bfloat16",
            "mlp_output": "bfloat16",
            "lm_head": "bfloat16",
            "kv_cache": "bfloat16",
        }
    return {
        "attention_qkv": "bfloat8_b",
        "attention_output": "bfloat8_b",
        "mlp_intermediate": "bfloat4_b",
        "mlp_output": "bfloat8_b",
        "lm_head": "bfloat8_b",
        "kv_cache": "bfloat8_b",
    }
