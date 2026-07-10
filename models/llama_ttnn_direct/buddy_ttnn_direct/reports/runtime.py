from __future__ import annotations

from typing import Any

from .schema import (
    int_equal as _int_equal,
    int_list as _int_list,
    non_empty_string as _non_empty_string,
    nonnegative_number as _nonnegative_number,
    number_at_least as _number_at_least,
    numbers_equal as _numbers_equal,
    positive_number as _positive_number,
    safe_int as _safe_int,
)


def paged_kv_cache_shape(
    *,
    batch_size: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> list[int]:
    batch = _safe_int(batch_size)
    cache = _safe_int(cache_len)
    kv_heads = _safe_int(num_kv_heads)
    dim = _safe_int(head_dim)
    block = _safe_int(page_block_size) or 32
    if None in (batch, cache, kv_heads, dim) or cache < 0 or block <= 0:
        return []
    pages_per_user = max(1, (cache + block - 1) // block)
    return [batch * pages_per_user, kv_heads, block, dim]


def expected_attention_layer_output_shape_summary(
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    hidden = _safe_int(hidden_size)
    attention_output = (
        [1, 1, batch, hidden] if None not in (batch, hidden) else []
    )
    kv_shape = paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    return {
        "attention_output": attention_output,
        "kv_cache_shape": kv_shape,
    }


def attention_layer_output_shapes_complete(
    output_shapes: Any,
    *,
    batch_size: Any,
    cache_len: Any,
    hidden_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any = 32,
) -> bool:
    expected = expected_attention_layer_output_shape_summary(
        batch_size=batch_size,
        cache_len=cache_len,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    if not isinstance(output_shapes, dict):
        return False
    return (
        _int_list(output_shapes.get("attention_output"))
        == expected["attention_output"]
        and _int_list(output_shapes.get("key_cache"))
        == expected["kv_cache_shape"]
        and _int_list(output_shapes.get("value_cache"))
        == expected["kv_cache_shape"]
    )


def attention_layer_output_shape_observed(
    output_shapes: Any,
) -> dict[str, Any]:
    if not isinstance(output_shapes, dict):
        return {}
    return {
        "attention_output": _int_list(
            output_shapes.get("attention_output")
        ),
        "key_cache": _int_list(output_shapes.get("key_cache")),
        "value_cache": _int_list(output_shapes.get("value_cache")),
    }


def expected_decode_output_shape_summary(
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any = "token",
    page_block_size: Any = 32,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    seq = _safe_int(seq_len)
    vocab = _safe_int(vocab_size)
    layers = _safe_int(layer_count)
    token_shape = [batch, seq] if batch is not None and seq is not None else []
    token_vector = [batch] if batch is not None else []
    normalized_output_kind = (
        "logits" if output_kind == "logits" else "token"
    )
    logits_shape = (
        [batch, seq, vocab] if None not in (batch, seq, vocab) else []
    )
    output_shape = (
        logits_shape if normalized_output_kind == "logits" else token_shape
    )
    kv_shape = paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    layer_ids = list(range(layers)) if layers is not None and layers > 0 else []
    return {
        "output_kind": normalized_output_kind,
        "output_shape": output_shape,
        "accepted_output_shapes": [
            shape for shape in (token_shape, token_vector) if shape
        ]
        if normalized_output_kind == "token"
        else [output_shape],
        "kv_cache_shape": kv_shape,
        "kv_cache_layer_ids": layer_ids,
    }


def decode_output_shapes_complete(
    output_shapes: Any,
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    vocab_size: Any,
    num_kv_heads: Any,
    head_dim: Any,
    output_kind: Any = "token",
    page_block_size: Any = 32,
) -> bool:
    expected = expected_decode_output_shape_summary(
        layer_count=layer_count,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_len=cache_len,
        vocab_size=vocab_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        output_kind=output_kind,
        page_block_size=page_block_size,
    )
    if not isinstance(output_shapes, dict):
        return False
    output_kind = expected["output_kind"]
    output_shape = _int_list(output_shapes.get(output_kind))
    if output_kind == "token":
        if output_shape not in expected["accepted_output_shapes"]:
            return False
    elif output_shape != expected["output_shape"]:
        return False
    if _int_list(output_shapes.get("key_cache")) != expected["kv_cache_shape"]:
        return False
    if _int_list(output_shapes.get("value_cache")) != expected["kv_cache_shape"]:
        return False
    layers = output_shapes.get("kv_cache_layers")
    if not isinstance(layers, list):
        return False
    if len(layers) != len(expected["kv_cache_layer_ids"]):
        return False
    observed_layer_ids = []
    for layer in layers:
        if not isinstance(layer, dict):
            return False
        try:
            layer_id = int(layer["layer_id"])
        except (KeyError, TypeError, ValueError):
            return False
        observed_layer_ids.append(layer_id)
        if _int_list(layer.get("key_cache")) != expected["kv_cache_shape"]:
            return False
        if _int_list(layer.get("value_cache")) != expected["kv_cache_shape"]:
            return False
    return observed_layer_ids == expected["kv_cache_layer_ids"]


def decode_output_shape_observed(output_shapes: Any) -> dict[str, Any]:
    if not isinstance(output_shapes, dict):
        return {}
    layers = output_shapes.get("kv_cache_layers")
    return {
        "output_kind": "logits" if "logits" in output_shapes else "token",
        "token": _int_list(output_shapes.get("token")),
        "logits": _int_list(output_shapes.get("logits")),
        "key_cache": _int_list(output_shapes.get("key_cache")),
        "value_cache": _int_list(output_shapes.get("value_cache")),
        "kv_cache_layer_ids": [
            _safe_int(layer.get("layer_id"))
            for layer in layers
            if isinstance(layer, dict)
        ]
        if isinstance(layers, list)
        else [],
        "kv_cache_layer_shapes": [
            {
                "layer_id": _safe_int(layer.get("layer_id")),
                "key_cache": _int_list(layer.get("key_cache")),
                "value_cache": _int_list(layer.get("value_cache")),
            }
            for layer in layers
            if isinstance(layer, dict)
        ]
        if isinstance(layers, list)
        else [],
    }


def shape_dict_has_int_lists(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(_int_list(shape) for shape in value.values())


def decode_shell_numeric_reference_complete(
    decode_shell: Any,
    *,
    expected_pcc_threshold: Any,
) -> bool:
    if not isinstance(decode_shell, dict):
        return False
    threshold = decode_shell.get("pcc_threshold")
    return (
        decode_shell.get("numeric_reference_status") == "passed"
        and decode_shell.get("numeric_reference_kind") == "torch_decode_shell"
        and decode_shell.get("numeric_reference_passed") is True
        and _number_at_least(decode_shell.get("pcc"), threshold)
        and _numbers_equal(threshold, expected_pcc_threshold)
        and decode_shell.get("numeric_reference_failed_checks") == []
    )


def decode_shell_numeric_reference_observed(
    decode_shell: Any,
) -> dict[str, Any]:
    if not isinstance(decode_shell, dict):
        return {}
    return {
        "status": decode_shell.get("numeric_reference_status"),
        "kind": decode_shell.get("numeric_reference_kind"),
        "passed": decode_shell.get("numeric_reference_passed"),
        "pcc": decode_shell.get("pcc"),
        "pcc_threshold": decode_shell.get("pcc_threshold"),
        "failed_checks": decode_shell.get(
            "numeric_reference_failed_checks",
            [],
        ),
    }


def observed_ops_cover_planned(planned_ops: Any, observed_ops: Any) -> bool:
    if not isinstance(planned_ops, list) or not isinstance(observed_ops, list):
        return False
    planned_index = 0
    for observed in observed_ops:
        if (
            planned_index < len(planned_ops)
            and str(observed) == str(planned_ops[planned_index])
        ):
            planned_index += 1
    return planned_index == len(planned_ops)


def decode_runtime_inputs_complete(
    step: Any,
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
) -> bool:
    if not isinstance(step, dict):
        return False
    expected = expected_decode_runtime_input_summary(
        layer_count=layer_count,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    input_shapes = step.get("input_shapes")
    kv_cache = step.get("kv_cache")
    if not isinstance(input_shapes, dict) or not isinstance(kv_cache, dict):
        return False
    input_source = step.get("input_source")
    expected_synthetic_runtime_count = expected[
        "synthetic_runtime_input_tensor_count"
    ]
    expected_prompt_runtime_count = 0
    expected_decode_runtime_state_count = 0
    expected_rotary_runtime_count = 0
    expected_kv_cache_runtime_count = 0
    expected_synthetic_rotary_count = expected[
        "synthetic_rotary_tensor_count"
    ]
    if input_source == "prompt_runtime":
        expected_synthetic_runtime_count = 0
        expected_prompt_runtime_count = 1
        expected_decode_runtime_state_count = 2
        expected_rotary_runtime_count = expected_prompt_rotary_runtime_count(
            step,
            fallback=expected_synthetic_rotary_count,
        )
        expected_kv_cache_runtime_count = expected[
            "kv_cache_runtime_input_tensor_count"
        ]
        expected_synthetic_rotary_count = 0
    return (
        _int_list(input_shapes.get("token_ids"))
        == expected["token_ids"]
        and _int_list(input_shapes.get("page_table"))
        == expected["page_table"]
        and _int_list(input_shapes.get("cache_position"))
        == expected["cache_position"]
        and _int_list(input_shapes.get("key_cache"))
        == expected["kv_cache_shape"]
        and _int_list(input_shapes.get("value_cache"))
        == expected["kv_cache_shape"]
        and _int_list(kv_cache.get("physical_shape"))
        == expected["kv_cache_shape"]
        and _int_list(kv_cache.get("logical_shape"))
        == expected["kv_cache_logical_shape"]
        and _int_equal(kv_cache.get("page_block_size"), page_block_size)
        and _int_equal(kv_cache.get("page_count"), expected["page_count"])
        and _int_equal(
            kv_cache.get("max_num_blocks"),
            expected["max_num_blocks"],
        )
        and runtime_input_source_supported(step)
        and _int_equal(
            step.get("synthetic_runtime_input_tensor_count"),
            expected_synthetic_runtime_count,
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("prompt_runtime_input_tensor_count"),
                expected_prompt_runtime_count,
            )
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("decode_runtime_state_input_tensor_count"),
                expected_decode_runtime_state_count,
            )
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("rotary_runtime_input_tensor_count"),
                expected_rotary_runtime_count,
            )
        )
        and (
            input_source != "prompt_runtime"
            or _int_equal(
                step.get("kv_cache_runtime_input_tensor_count"),
                expected_kv_cache_runtime_count,
            )
        )
        and _int_equal(
            step.get("synthetic_rotary_tensor_count"),
            expected_synthetic_rotary_count,
        )
    )


def expected_prompt_rotary_runtime_count(
    step: dict[str, Any],
    *,
    fallback: Any,
) -> Any:
    rotary_state = step.get("rotary_runtime_state")
    if not isinstance(rotary_state, dict):
        return fallback
    tensor_count = _safe_int(rotary_state.get("tensor_count"))
    if rotary_state.get("shared_across_layers") is True:
        return tensor_count if tensor_count is not None else 3
    return tensor_count if tensor_count is not None else fallback


def runtime_input_source_supported(step: Any) -> bool:
    return (
        isinstance(step, dict)
        and step.get("input_source") in {"synthetic", "prompt_runtime"}
    )


def synthetic_runtime_inputs_accepted(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if step.get("input_source") == "prompt_runtime":
        return _nonnegative_number(
            step.get("synthetic_runtime_input_tensor_count")
        )
    return _positive_number(step.get("synthetic_runtime_input_tensor_count"))


def decode_shell_runtime_inputs_accepted(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    source = step.get("input_source")
    if source == "prompt_runtime":
        return (
            _int_equal(step.get("synthetic_runtime_input_tensor_count"), 0)
            and _int_equal(step.get("runtime_input_tensor_count"), 0)
            and _int_equal(step.get("prompt_runtime_input_tensor_count"), 1)
        )
    if source == "synthetic":
        return _positive_number(step.get("runtime_input_tensor_count"))
    return False


def expected_decode_runtime_input_summary(
    *,
    layer_count: Any,
    batch_size: Any,
    seq_len: Any,
    cache_len: Any,
    num_kv_heads: Any,
    head_dim: Any,
    page_block_size: Any,
) -> dict[str, Any]:
    batch = _safe_int(batch_size)
    seq = _safe_int(seq_len)
    cache = _safe_int(cache_len)
    layers = _safe_int(layer_count)
    block = _safe_int(page_block_size) or 32
    page_count = None
    max_num_blocks = None
    if batch is not None and cache is not None and block > 0:
        page_count = max(1, (cache + block - 1) // block)
        max_num_blocks = batch * page_count
    kv_shape = paged_kv_cache_shape(
        batch_size=batch_size,
        cache_len=cache_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )
    logical_shape = (
        [batch, cache, _safe_int(num_kv_heads), _safe_int(head_dim)]
        if None
        not in (
            batch,
            cache,
            _safe_int(num_kv_heads),
            _safe_int(head_dim),
        )
        else []
    )
    return {
        "token_ids": [batch, seq]
        if None not in (batch, seq)
        else [],
        "page_table": [batch, page_count]
        if None not in (batch, page_count)
        else [],
        "cache_position": [batch] if batch is not None else [],
        "kv_cache_shape": kv_shape,
        "kv_cache_logical_shape": logical_shape,
        "page_count": page_count,
        "max_num_blocks": max_num_blocks,
        "synthetic_runtime_input_tensor_count": (
            3 + 2 * layers if layers is not None else None
        ),
        "kv_cache_runtime_input_tensor_count": (
            2 * layers if layers is not None else None
        ),
        "synthetic_rotary_tensor_count": (
            3 * layers if layers is not None else None
        ),
    }


def decode_runtime_input_observed(step: Any) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    input_shapes = step.get("input_shapes")
    kv_cache = step.get("kv_cache")
    return {
        "input_source": step.get("input_source"),
        "synthetic_runtime_input_tensor_count": step.get(
            "synthetic_runtime_input_tensor_count"
        ),
        "prompt_runtime_input_tensor_count": step.get(
            "prompt_runtime_input_tensor_count"
        ),
        "prompt_tokenization": step.get("prompt_tokenization"),
        "decode_runtime_state_input_tensor_count": step.get(
            "decode_runtime_state_input_tensor_count"
        ),
        "decode_runtime_state": step.get("decode_runtime_state"),
        "rotary_runtime_input_tensor_count": step.get(
            "rotary_runtime_input_tensor_count"
        ),
        "rotary_runtime_state": step.get("rotary_runtime_state"),
        "kv_cache_runtime_input_tensor_count": step.get(
            "kv_cache_runtime_input_tensor_count"
        ),
        "kv_cache_runtime_state": step.get("kv_cache_runtime_state"),
        "synthetic_rotary_tensor_count": step.get(
            "synthetic_rotary_tensor_count"
        ),
        "input_shapes": {
            "token_ids": _int_list((input_shapes or {}).get("token_ids")),
            "page_table": _int_list((input_shapes or {}).get("page_table")),
            "cache_position": _int_list(
                (input_shapes or {}).get("cache_position")
            ),
            "key_cache": _int_list((input_shapes or {}).get("key_cache")),
            "value_cache": _int_list(
                (input_shapes or {}).get("value_cache")
            ),
        }
        if isinstance(input_shapes, dict)
        else {},
        "kv_cache": {
            "page_block_size": (kv_cache or {}).get("page_block_size"),
            "page_count": (kv_cache or {}).get("page_count"),
            "max_num_blocks": (kv_cache or {}).get("max_num_blocks"),
            "physical_shape": _int_list(
                (kv_cache or {}).get("physical_shape")
            ),
            "logical_shape": _int_list((kv_cache or {}).get("logical_shape")),
        }
        if isinstance(kv_cache, dict)
        else {},
    }


def step_ttnn_environment(step: dict[str, Any]) -> dict[str, Any]:
    environment = step.get("ttnn_environment") or {}
    return environment if isinstance(environment, dict) else {}


def ttnn_runtime_identity_available(environment: Any) -> bool:
    if not isinstance(environment, dict):
        return False
    return _non_empty_string(environment.get("version")) or _non_empty_string(
        environment.get("module_file")
    )


def ttnn_runtime_identity_observed(environment: Any) -> dict[str, Any]:
    if not isinstance(environment, dict):
        return {}
    return {
        "version": environment.get("version"),
        "module_file": environment.get("module_file"),
    }
