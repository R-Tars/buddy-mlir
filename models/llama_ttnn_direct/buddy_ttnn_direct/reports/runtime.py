from __future__ import annotations

from typing import Any

from .schema import int_list as _int_list, safe_int as _safe_int


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
