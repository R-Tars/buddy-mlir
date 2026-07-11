from __future__ import annotations

from typing import Any, Sequence

from .errors import UnsupportedTTNNOp


def nlp_create_qkv_heads_decode(
    ttnn_module: Any,
    fused_qkv: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    memory_config: Any | None = None,
) -> tuple[Any, Any, Any]:
    # Signature: ttnn.experimental.nlp_create_qkv_heads_decode(
    #   fused_qkv, num_heads, num_kv_heads, memory_config=None
    # )
    op_name = "nlp_create_qkv_heads_decode"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        memory_config=memory_config,
    )
    return op(fused_qkv, **kwargs)


def split_qkv_heads_prefill(
    ttnn_module: Any,
    fused_qkv: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    memory_config: Any | None = None,
) -> tuple[Any, Any, Any]:
    # Llama RoPE requires K to remain [B, H, S, D]. The TTNN default
    # transposes K, and seq_len == head_dim can hide that error in shapes.
    op_name = "split_query_key_value_and_split_heads"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("transformer", op_name),
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_key=False,
        memory_config=memory_config,
    )
    return op(fused_qkv, **kwargs)


def rotary_embedding_decode(
    ttnn_module: Any,
    query: Any,
    key: Any,
    *,
    cos_matrix: Any,
    sin_matrix: Any,
    transformation_matrix: Any,
    is_decode_mode: bool = True,
) -> tuple[Any, Any]:
    # Signature: ttnn.experimental.rotary_embedding_llama(
    #   tensor, cos_matrix, sin_matrix, transformation_matrix,
    #   is_decode_mode=True
    # )
    op_name = "rotary_embedding_llama"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = {"is_decode_mode": is_decode_mode}
    return (
        op(query, cos_matrix, sin_matrix, transformation_matrix, **kwargs),
        op(key, cos_matrix, sin_matrix, transformation_matrix, **kwargs),
    )


def rotary_embedding_prefill(
    ttnn_module: Any,
    query: Any,
    key: Any,
    *,
    cos_matrix: Any,
    sin_matrix: Any,
    transformation_matrix: Any,
) -> tuple[Any, Any]:
    return rotary_embedding_decode(
        ttnn_module,
        query,
        key,
        cos_matrix=cos_matrix,
        sin_matrix=sin_matrix,
        transformation_matrix=transformation_matrix,
        is_decode_mode=False,
    )


def paged_update_cache(
    ttnn_module: Any,
    cache_tensor: Any,
    update_tensor: Any,
    *,
    update_idxs_tensor: Any | None = None,
    update_idxs: Any | None = None,
    page_table: Any | None = None,
) -> Any:
    # Signature: ttnn.experimental.paged_update_cache(
    #   cache_tensor, update_tensor, update_idxs_tensor=None,
    #   update_idxs=None, page_table=None
    # )
    op_name = "paged_update_cache"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(
        update_idxs_tensor=update_idxs_tensor,
        update_idxs=update_idxs,
        page_table=page_table,
    )
    return op(cache_tensor, update_tensor, **kwargs)


def paged_sdpa_decode(
    ttnn_module: Any,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    page_table: Any,
    cache_position: Any,
    *,
    scale: float | None = None,
    memory_config: Any | None = None,
    program_config: Any | None = None,
    compute_kernel_config: Any | None = None,
    attention_mask: Any | None = None,
    attention_sink: Any | None = None,
    sliding_window_size: int | None = None,
) -> Any:
    # Signature: ttnn.transformer.paged_scaled_dot_product_attention_decode(
    #   query, key_cache, value_cache, cur_pos_tensor, page_table_tensor,
    #   scale=None, program_config=None, compute_kernel_config=None,
    #   memory_config=None
    # )
    op_name = "paged_scaled_dot_product_attention_decode"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("transformer", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(
        cur_pos_tensor=cache_position,
        page_table_tensor=page_table,
        scale=scale,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        attention_mask=attention_mask,
        attention_sink=attention_sink,
        sliding_window_size=sliding_window_size,
    )
    return op(query, key_cache, value_cache, **kwargs)


def scaled_dot_product_attention(
    ttnn_module: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    memory_config: Any | None = None,
    program_config: Any | None = None,
    compute_kernel_config: Any | None = None,
    attention_mask: Any | None = None,
) -> Any:
    op_name = "scaled_dot_product_attention"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("transformer", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(
        is_causal=is_causal,
        scale=scale,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        attention_mask=attention_mask,
    )
    return op(query, key, value, **kwargs)


def fill_cache(
    ttnn_module: Any,
    cache_tensor: Any,
    update_tensor: Any,
    *,
    user_id: int = 0,
    page_table: Any | None = None,
) -> Any:
    op_name = "fill_cache"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("kv_cache", "fill_cache_for_user_"),
            ("kv_cache", op_name),
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(user_id=user_id, page_table=page_table)
    try:
        return op(cache_tensor, update_tensor, **kwargs)
    except TypeError as err:
        if page_table is not None:
            raise
        try:
            return op(cache_tensor, update_tensor, user_id)
        except TypeError:
            raise err


def paged_fill_cache(
    ttnn_module: Any,
    cache_tensor: Any,
    update_tensor: Any,
    page_table: Any,
    *,
    batch_idx: int = 0,
    batch_idx_tensor: Any | None = None,
    compute_kernel_config: Any | None = None,
) -> Any:
    op_name = "paged_fill_cache"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(
        batch_idx=batch_idx,
        batch_idx_tensor=batch_idx_tensor,
        compute_kernel_config=compute_kernel_config,
    )
    return op(cache_tensor, update_tensor, page_table, **kwargs)


def nlp_concat_heads_decode(
    ttnn_module: Any,
    attention: Any,
    *,
    num_heads: int,
    memory_config: Any | None = None,
) -> Any:
    # Signature: ttnn.experimental.nlp_concat_heads_decode(
    #   attention, num_heads
    # )
    if memory_config is not None:
        to_memory_config = _resolve_op(
            ttnn_module,
            "to_memory_config",
            (("to_memory_config",),),
        )
        attention = to_memory_config(attention, memory_config=memory_config)
    op_name = "nlp_concat_heads_decode"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("experimental", op_name),
            (op_name,),
        ),
    )
    return op(attention, num_heads=num_heads)


def concat_heads_prefill(
    ttnn_module: Any,
    attention: Any,
    *,
    memory_config: Any | None = None,
) -> Any:
    op_name = "concatenate_heads"
    op = _resolve_op(
        ttnn_module,
        op_name,
        (
            ("transformer", op_name),
            ("experimental", op_name),
            (op_name,),
        ),
    )
    kwargs = _without_none(memory_config=memory_config)
    return op(attention, **kwargs)


def _resolve_op(
    ttnn_module: Any,
    op_name: str,
    paths: Sequence[Sequence[str]],
) -> Any:
    for path in paths:
        candidate = _resolve_path(ttnn_module, path)
        if callable(candidate):
            return candidate
    raise UnsupportedTTNNOp(op_name, paths)


def _resolve_path(root: Any, path: Sequence[str]) -> Any | None:
    current = root
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _without_none(**kwargs: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in kwargs.items()
        if value is not None
    }
