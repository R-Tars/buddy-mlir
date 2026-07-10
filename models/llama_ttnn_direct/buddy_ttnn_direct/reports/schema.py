from __future__ import annotations

from pathlib import Path
from typing import Any


def acceptance_check(
    name: str,
    passed: bool,
    **details: Any,
) -> dict[str, Any]:
    check = {
        "name": name,
        "passed": bool(passed),
    }
    check.update(details)
    return check


def path_exists(path: Any) -> bool:
    if path is None:
        return False
    try:
        return Path(path).exists()
    except (TypeError, ValueError):
        return False


def paths_exist(paths: Any) -> bool:
    if not isinstance(paths, list) or not paths:
        return False
    return all(path_exists(path) for path in paths)


def path_exists_relative_to(path: Any, base_dir: Any) -> bool:
    if path_exists(path):
        return True
    if path is None or base_dir is None:
        return False
    try:
        candidate = Path(base_dir) / Path(path)
    except (TypeError, ValueError):
        return False
    return candidate.exists()


def paths_exist_relative_to(paths: Any, base_dir: Any) -> bool:
    if not isinstance(paths, list) or not paths:
        return False
    return all(path_exists_relative_to(path, base_dir) for path in paths)


def status_count_matches_total(
    counts: Any,
    status: str,
    total: Any,
) -> bool:
    if not isinstance(counts, dict):
        return False
    try:
        expected_total = int(total)
        observed_total = int(counts.get(status))
    except (TypeError, ValueError):
        return False
    return (
        expected_total > 0
        and counts == {status: observed_total}
        and observed_total == expected_total
    )


def positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def numbers_equal(lhs: Any, rhs: Any) -> bool:
    try:
        return abs(float(lhs) - float(rhs)) <= 1.0e-9
    except (TypeError, ValueError):
        return False


def has_nonnegative_fields(value: Any, fields: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    return all(
        field in value and nonnegative_number(value.get(field))
        for field in fields
    )


def field_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)


def number_at_least(value: Any, minimum: Any) -> bool:
    try:
        return float(value) >= float(minimum)
    except (TypeError, ValueError):
        return False


def int_equal(observed: Any, expected: Any) -> bool:
    try:
        return int(observed) == int(expected)
    except (TypeError, ValueError):
        return False


def int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        converted = safe_int(item)
        if converted is None:
            return []
        result.append(converted)
    return result


def int_list_contains(values: Any, expected: Any) -> bool:
    expected_int = safe_int(expected)
    if expected_int is None:
        return False
    return expected_int in int_list(values)


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def contains_all(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, list):
        return False
    return set(expected).issubset(set(observed))


def kv_cache_contract_from_template_config(
    template_config: dict[str, Any],
    *,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    template = template_config.get("kv_cache_template")
    return {
        "template": template,
        "policy": "paged" if template == "paged_kv_cache" else None,
        "page_block_size": 32,
        "dtype": "bfloat8_b",
        "max_cache_len": cache_len,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def decode_step_contract(
    *,
    layer_count: int,
    batch_size: int,
    seq_len: int,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    page_block_size = safe_int(kv_cache.get("page_block_size")) or 32
    page_count = max(1, (cache_len + page_block_size - 1) // page_block_size)
    max_num_blocks = batch_size * page_count
    physical_kv_cache_shape = [
        max_num_blocks,
        num_kv_heads,
        page_block_size,
        head_dim,
    ]
    logical_kv_cache_shape = [
        batch_size,
        cache_len,
        num_kv_heads,
        head_dim,
    ]
    kv_policy = kv_cache.get("policy")
    kv_template = kv_cache.get("template")
    generation_template = generation.get("template")
    retain_logits = bool(generation.get("retain_logits", False))
    output_kind = "logits" if retain_logits else "token"
    return {
        "schema_version": 1,
        "source": "generated_program_config",
        "layers": layer_count,
        "batch_size": batch_size,
        "decode_seq_len": seq_len,
        "cache_len": cache_len,
        "token_input_shape": [batch_size, seq_len],
        "kv_cache_policy": kv_policy,
        "kv_cache_template": kv_template,
        "uses_paged_kv_cache": (
            kv_policy == "paged" or kv_template == "paged_kv_cache"
        ),
        "kv_page_block_size": page_block_size,
        "page_count": page_count,
        "max_num_blocks": max_num_blocks,
        "page_table_shape": [batch_size, page_count],
        "cache_position_shape": [batch_size],
        "kv_cache_shape": physical_kv_cache_shape,
        "kv_cache_physical_shape": physical_kv_cache_shape,
        "kv_cache_logical_shape": logical_kv_cache_shape,
        "kv_cache_layer_ids": list(range(layer_count)),
        "generation_template": generation_template,
        "output_kind": output_kind,
        "accepted_output_kinds": ["token", "logits"],
    }


def positive_count(counts: Any) -> bool:
    if not isinstance(counts, dict):
        return False
    total = 0
    for count in counts.values():
        try:
            total += int(count)
        except (TypeError, ValueError):
            return False
    return total > 0
