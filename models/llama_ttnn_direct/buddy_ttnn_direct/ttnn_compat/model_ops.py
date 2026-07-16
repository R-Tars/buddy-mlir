from __future__ import annotations

from . import ops as ttnn_ops


def _tensor_memory_config(tensor):
    memory_config = getattr(tensor, "memory_config", None)
    if callable(memory_config):
        return memory_config()
    return memory_config


def _tensor_layout(tensor):
    layout = getattr(tensor, "layout", None)
    if callable(layout):
        return layout()
    return layout


def _tensor_shape(tensor):
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


class TTNNCompatOps:
    def __init__(self, ttnn_module, *, record_ops=False):
        self.ttnn = ttnn_module
        self.op_log = [] if record_ops else None

    def _record(self, op_name):
        if self.op_log is not None:
            self.op_log.append(op_name)

    def enable_recording(self):
        if self.op_log is None:
            self.op_log = []

    def resolve_memory_config(self, memory_config):
        if memory_config is None:
            return None
        if not isinstance(memory_config, str):
            return memory_config
        value = memory_config
        if value in ("", "none", "None", "default"):
            return None
        if value.startswith("ttnn."):
            value = value.split(".", 1)[1]
        aliases = {
            "dram": "DRAM_MEMORY_CONFIG",
            "l1": "L1_MEMORY_CONFIG",
            "l1_interleaved": "L1_MEMORY_CONFIG",
            "l1_width_sharded": "L1_MEMORY_CONFIG",
            "l1_height_sharded": "L1_MEMORY_CONFIG",
        }
        attr = aliases.get(value, value)
        return getattr(self.ttnn, attr, memory_config)

    def add(
        self,
        left,
        right,
        *,
        memory_config=None,
        dtype=None,
        op_name="residual_add",
    ):
        self._record(op_name)
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self.ttnn.add(left, right, **kwargs)

    def linear(
        self,
        input_tensor,
        weight,
        *,
        memory_config=None,
        program_config=None,
        compute_kernel_config=None,
        dtype=None,
        activation=None,
        op_name="linear",
    ):
        self._record(op_name)
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        if program_config is not None:
            kwargs["program_config"] = program_config
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        if dtype is not None:
            kwargs["dtype"] = dtype
        if activation is not None:
            kwargs["activation"] = activation
        return self.ttnn.linear(input_tensor, weight, **kwargs)

    def mul_silu(
        self,
        gate,
        up,
        *,
        memory_config=None,
        dtype=None,
        op_name="mul_silu",
    ):
        self._record(op_name)
        kwargs = {}
        activation = self._silu_activation()
        if activation is not None:
            kwargs["input_tensor_a_activations"] = [activation]
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        if dtype is not None:
            kwargs["dtype"] = dtype
        mul = getattr(self.ttnn, "mul", None)
        if mul is None:
            mul = getattr(self.ttnn, "multiply", None)
        if mul is None:
            raise AttributeError("ttnn must provide mul or multiply")
        return mul(gate, up, **kwargs)

    def multiply(
        self,
        left,
        right,
        *,
        memory_config=None,
        dtype=None,
        op_name="multiply",
    ):
        self._record(op_name)
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        if dtype is not None:
            kwargs["dtype"] = dtype
        op = getattr(self.ttnn, "mul", None)
        if op is None:
            op = getattr(self.ttnn, "multiply", None)
        if op is None:
            raise ttnn_ops.UnsupportedTTNNOp(
                "multiply",
                (("mul",), ("multiply",)),
            )
        return op(left, right, **kwargs)

    def split_last_dim(
        self,
        tensor,
        *,
        split_size,
        strategy="auto",
        memory_config=None,
        op_name="split_last_dim",
    ):
        split_size = int(split_size)
        if strategy not in {"auto", "split", "slice"}:
            raise ValueError(
                "split_last_dim strategy must be 'auto', 'split', or 'slice'"
            )
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        split = getattr(self.ttnn, "split", None)
        if strategy != "slice" and callable(split):
            self._record(op_name)
            result = split(tensor, split_size, dim=-1, **kwargs)
            if len(result) != 2:
                raise ValueError("packed gate/up split must produce two tensors")
            return result[0], result[1]
        if strategy == "split":
            raise ttnn_ops.UnsupportedTTNNOp("split_last_dim", (("split",),))
        slice_op = getattr(self.ttnn, "slice", None)
        shape = _tensor_shape(tensor)
        if not callable(slice_op) or shape is None:
            raise ttnn_ops.UnsupportedTTNNOp(
                "split_last_dim",
                (("split",), ("slice",)),
            )
        if shape[-1] != 2 * split_size:
            raise ValueError("packed gate/up width must equal twice the split size")
        starts = [0] * len(shape)
        middle = list(shape)
        middle[-1] = split_size
        second = list(starts)
        second[-1] = split_size
        self._record(op_name)
        return (
            slice_op(tensor, starts, middle, **kwargs),
            slice_op(tensor, second, shape, **kwargs),
        )

    def _silu_activation(self):
        unary_with_param = getattr(self.ttnn, "UnaryWithParam", None)
        unary_op_type = getattr(self.ttnn, "UnaryOpType", None)
        if unary_with_param is None or unary_op_type is None:
            return None
        silu = getattr(unary_op_type, "SILU", None)
        if silu is None:
            return None
        return unary_with_param(silu)

    def embedding(
        self,
        token_ids,
        weight,
        *,
        memory_config=None,
        dtype=None,
        op_name="embedding",
    ):
        self._record(op_name)
        op = getattr(self.ttnn, "embedding", None)
        if op is None:
            raise ttnn_ops.UnsupportedTTNNOp(
                "embedding",
                (("embedding",),),
            )
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return op(token_ids, weight, **kwargs)

    def rms_norm(
        self,
        hidden,
        weight,
        *,
        epsilon,
        program_config=None,
        compute_kernel_config=None,
        memory_config=None,
        dtype=None,
        op_name="rms_norm",
    ):
        self._record(op_name)
        op = getattr(self.ttnn, "rms_norm", None)
        if op is None:
            op = getattr(self.ttnn, "rmsnorm", None)
        if op is None:
            raise ttnn_ops.UnsupportedTTNNOp(
                "rms_norm",
                (("rms_norm",), ("rmsnorm",)),
            )
        hidden = self.ensure_tile_layout(
            hidden,
            op_name=f"to_layout.tile.{op_name}",
        )
        kwargs = {"weight": weight, "epsilon": epsilon}
        if program_config is not None:
            kwargs["program_config"] = program_config
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        result = op(hidden, **kwargs)
        result_dtype = getattr(result, "dtype", None)
        if dtype is None or result_dtype is None or result_dtype == dtype:
            return result
        typecast = getattr(self.ttnn, "typecast", None)
        if typecast is None:
            return result
        return typecast(result, dtype)

    def ensure_tile_layout(self, tensor, *, op_name):
        to_layout = getattr(self.ttnn, "to_layout", None)
        tile_layout = getattr(self.ttnn, "TILE_LAYOUT", None)
        if to_layout is None or tile_layout is None:
            return tensor
        if _tensor_layout(tensor) == tile_layout:
            return tensor
        self._record(op_name)
        return to_layout(tensor, tile_layout)

    def reshape_decode_hidden_for_layer(
        self,
        hidden,
        *,
        op_name="reshape_hidden_decode",
    ):
        shape = getattr(hidden, "shape", None)
        if shape is None:
            return hidden
        shape = [int(dim) for dim in shape]
        if len(shape) == 3 and shape[1] == 1:
            batch = shape[0]
            feature_dim = shape[2]
        elif len(shape) == 4 and shape[0] == 1 and shape[1] != 1 and shape[2] == 1:
            batch = shape[1]
            feature_dim = shape[3]
        else:
            return hidden
        reshape = getattr(self.ttnn, "reshape", None)
        if reshape is None:
            return hidden
        logical_shape = (1, 1, batch, feature_dim)
        padded_batch = ((batch + 31) // 32) * 32
        padded_shape = (1, 1, padded_batch, feature_dim)
        self._record(op_name)
        try:
            return reshape(hidden, logical_shape, padded_shape)
        except TypeError:
            return reshape(hidden, logical_shape)

    def reshape_decode_qkv_for_heads(
        self,
        qkv,
        *,
        op_name="reshape_qkv_decode",
    ):
        shape = getattr(qkv, "shape", None)
        if shape is None or len(shape) != 4:
            return qkv
        shape = [int(dim) for dim in shape]
        if not (shape[0] == 1 and shape[1] != 1 and shape[2] == 1):
            return qkv
        reshape = getattr(self.ttnn, "reshape", None)
        if reshape is None:
            return qkv
        batch = shape[1]
        feature_dim = shape[3]
        logical_shape = (1, 1, batch, feature_dim)
        padded_batch = ((batch + 31) // 32) * 32
        padded_shape = (1, 1, padded_batch, feature_dim)
        self._record(op_name)
        try:
            return reshape(qkv, logical_shape, padded_shape)
        except TypeError:
            return reshape(qkv, logical_shape)

    def reshape_prefill_qkv_for_heads(
        self,
        qkv,
        *,
        op_name="reshape_qkv_prefill",
    ):
        shape = getattr(qkv, "shape", None)
        if shape is None or len(shape) != 4:
            return qkv
        shape = [int(dim) for dim in shape]
        if shape[0] == 1:
            squeeze_dim = 0
            logical_shape = (shape[1], shape[2], shape[3])
        elif shape[1] == 1:
            squeeze_dim = 1
            logical_shape = (shape[0], shape[2], shape[3])
        else:
            return qkv
        squeeze = getattr(self.ttnn, "squeeze", None)
        if callable(squeeze):
            self._record(op_name)
            try:
                return squeeze(qkv, squeeze_dim)
            except TypeError:
                pass
        reshape = getattr(self.ttnn, "reshape", None)
        if reshape is None:
            return qkv
        self._record(op_name)
        return reshape(qkv, logical_shape)

    def reshape_prefill_hidden_for_layer(
        self,
        hidden,
        *,
        op_name="reshape_prefill_hidden_for_layer",
    ):
        shape = _tensor_shape(hidden)
        if shape is None:
            return hidden
        if len(shape) == 3:
            target_shape = (1, shape[0], shape[1], shape[2])
        elif len(shape) == 4 and shape[0] == 1:
            return hidden
        elif len(shape) == 4 and shape[1] == 1:
            target_shape = (1, shape[0], shape[2], shape[3])
        else:
            return hidden
        reshape = getattr(self.ttnn, "reshape", None)
        if not callable(reshape):
            raise ttnn_ops.UnsupportedTTNNOp(
                "reshape_prefill_hidden_for_layer",
                (("reshape",),),
            )
        self._record(op_name)
        try:
            return reshape(hidden, target_shape, target_shape)
        except TypeError:
            return reshape(hidden, target_shape)

    def reshape_prefill_hidden_like(
        self,
        hidden,
        reference,
        *,
        op_name="reshape_prefill_hidden_like",
    ):
        hidden_shape = _tensor_shape(hidden)
        reference_shape = _tensor_shape(reference)
        if hidden_shape is None or reference_shape is None:
            return hidden
        if hidden_shape == reference_shape:
            return hidden
        hidden_elements = 1
        for dim in hidden_shape:
            hidden_elements *= dim
        reference_elements = 1
        for dim in reference_shape:
            reference_elements *= dim
        if hidden_elements != reference_elements:
            raise ValueError(
                "prefill residual tensors must have the same element count: "
                f"hidden={hidden_shape}, reference={reference_shape}"
            )
        reshape = getattr(self.ttnn, "reshape", None)
        if not callable(reshape):
            raise ttnn_ops.UnsupportedTTNNOp(
                "reshape_prefill_hidden_like",
                (("reshape",),),
            )
        self._record(op_name)
        try:
            return reshape(hidden, tuple(reference_shape), tuple(reference_shape))
        except TypeError:
            return reshape(hidden, tuple(reference_shape))

    def to_memory_config(
        self,
        tensor,
        *,
        memory_config=None,
        op_name="to_memory_config",
    ):
        memory_config = self.resolve_memory_config(memory_config)
        if memory_config is None:
            return tensor
        if _tensor_memory_config(tensor) == memory_config:
            return tensor
        op = getattr(self.ttnn, "to_memory_config", None)
        if op is None:
            return tensor
        self._record(op_name)
        return op(tensor, memory_config=memory_config)

    def slice_batch_user(
        self,
        tensor,
        user_id,
        *,
        op_name="slice.batch_user",
    ):
        shape = _tensor_shape(tensor)
        if shape is None or not shape or shape[0] <= 1:
            return tensor
        slice_op = getattr(self.ttnn, "slice", None)
        if not callable(slice_op):
            raise ttnn_ops.UnsupportedTTNNOp(
                "slice_batch_user",
                (("slice",),),
            )
        starts = [0 for _ in shape]
        ends = list(shape)
        steps = [1 for _ in shape]
        starts[0] = int(user_id)
        ends[0] = int(user_id) + 1
        self._record(op_name)
        try:
            return slice_op(tensor, starts, ends, steps)
        except TypeError as err:
            try:
                return slice_op(tensor, starts, ends, steps=steps)
            except TypeError:
                try:
                    return slice_op(tensor, starts, ends)
                except TypeError:
                    raise err from None

    def select_sequence_position(
        self,
        tensor,
        position,
        *,
        op_name="select_sequence_position",
    ):
        shape = _tensor_shape(tensor)
        if shape is None or len(shape) not in (3, 4):
            return tensor
        sequence_dim = 1 if len(shape) == 3 else 2
        sequence_length = shape[sequence_dim]
        position = int(position)
        if position < 0 or position >= sequence_length:
            raise ValueError(
                f"sequence position {position} is outside length " f"{sequence_length}"
            )
        slice_op = getattr(self.ttnn, "slice", None)
        if not callable(slice_op):
            raise ttnn_ops.UnsupportedTTNNOp(
                "select_sequence_position",
                (("slice",),),
            )
        starts = [0 for _ in shape]
        ends = list(shape)
        steps = [1 for _ in shape]
        starts[sequence_dim] = position
        ends[sequence_dim] = position + 1
        self._record(op_name)
        try:
            return slice_op(tensor, starts, ends, steps)
        except TypeError as err:
            try:
                return slice_op(tensor, starts, ends, steps=steps)
            except TypeError:
                try:
                    return slice_op(tensor, starts, ends)
                except TypeError:
                    raise err from None

    def select_sequence_positions(
        self,
        tensor,
        positions,
        *,
        op_name="select_sequence_positions",
    ):
        shape = _tensor_shape(tensor)
        if shape is None or len(shape) not in (3, 4):
            return tensor
        sequence_dim = 1 if len(shape) == 3 else 2
        batch_dim = 0 if len(shape) == 3 or shape[0] != 1 else 1
        batch_size = shape[batch_dim]
        positions = [int(position) for position in positions]
        if len(positions) != batch_size:
            raise ValueError("sequence position count must match tensor batch size")
        if len(set(positions)) == 1:
            return self.select_sequence_position(
                tensor,
                positions[0],
                op_name=op_name,
            )

        slice_op = getattr(self.ttnn, "slice", None)
        concat_op = getattr(self.ttnn, "concat", None)
        if not callable(slice_op) or not callable(concat_op):
            raise ttnn_ops.UnsupportedTTNNOp(
                "select_sequence_positions",
                (("slice", "concat"),),
            )
        selected = []
        for user_id, position in enumerate(positions):
            if position < 0 or position >= shape[sequence_dim]:
                raise ValueError(
                    f"sequence position {position} is outside length "
                    f"{shape[sequence_dim]} for user {user_id}"
                )
            starts = [0 for _ in shape]
            ends = list(shape)
            steps = [1 for _ in shape]
            starts[batch_dim] = user_id
            ends[batch_dim] = user_id + 1
            starts[sequence_dim] = position
            ends[sequence_dim] = position + 1
            try:
                user_hidden = slice_op(tensor, starts, ends, steps)
            except TypeError:
                try:
                    user_hidden = slice_op(tensor, starts, ends, steps=steps)
                except TypeError:
                    user_hidden = slice_op(tensor, starts, ends)
            selected.append(user_hidden)
        self._record(op_name)
        return concat_op(selected, dim=batch_dim)

    def nlp_create_qkv_heads_decode(
        self,
        qkv,
        *,
        num_heads,
        num_kv_heads,
        overlap_qk_coregrid=None,
        memory_config=None,
        op_name="nlp_create_qkv_heads_decode",
    ):
        self._record(op_name)
        return ttnn_ops.nlp_create_qkv_heads_decode(
            self.ttnn,
            qkv,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            overlap_qk_coregrid=overlap_qk_coregrid,
            memory_config=self.resolve_memory_config(memory_config),
        )

    def split_qkv_heads_prefill(
        self,
        qkv,
        *,
        num_heads,
        num_kv_heads,
        memory_config=None,
        op_name="split_query_key_value_heads_prefill",
    ):
        self._record(op_name)
        return ttnn_ops.split_qkv_heads_prefill(
            self.ttnn,
            qkv,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            memory_config=self.resolve_memory_config(memory_config),
        )

    def rotary_embedding_decode(
        self,
        q,
        k,
        *,
        cos_matrix,
        sin_matrix,
        transformation_matrix,
        is_decode_mode=True,
        op_name="rotary_embedding_decode",
    ):
        self._record(op_name)
        return ttnn_ops.rotary_embedding_decode(
            self.ttnn,
            q,
            k,
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            transformation_matrix=transformation_matrix,
            is_decode_mode=is_decode_mode,
        )

    def rotary_embedding_fused_qk(
        self,
        q,
        k,
        *,
        cos_matrix,
        sin_matrix,
        transformation_matrix,
        compute_kernel_config=None,
        op_name="rotary_embedding_llama_fused_qk",
    ):
        self._record(op_name)
        return ttnn_ops.rotary_embedding_fused_qk(
            self.ttnn,
            q,
            k,
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            transformation_matrix=transformation_matrix,
            compute_kernel_config=compute_kernel_config,
        )

    def rotary_embedding_prefill(
        self,
        q,
        k,
        *,
        cos_matrix,
        sin_matrix,
        transformation_matrix,
        op_name="rotary_embedding_prefill",
    ):
        self._record(op_name)
        return ttnn_ops.rotary_embedding_prefill(
            self.ttnn,
            q,
            k,
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            transformation_matrix=transformation_matrix,
        )

    def paged_update_cache(
        self,
        cache_tensor,
        update_tensor,
        *,
        update_idxs_tensor=None,
        update_idxs=None,
        page_table=None,
        op_name="paged_update_cache",
    ):
        self._record(op_name)
        return ttnn_ops.paged_update_cache(
            self.ttnn,
            cache_tensor,
            update_tensor,
            update_idxs_tensor=update_idxs_tensor,
            update_idxs=update_idxs,
            page_table=page_table,
        )

    def paged_fused_update_cache(
        self,
        key_cache,
        key,
        value_cache,
        value,
        *,
        update_idxs_tensor=None,
        update_idxs=None,
        page_table=None,
        op_name="paged_fused_update_cache",
    ):
        self._record(op_name)
        return ttnn_ops.paged_fused_update_cache(
            self.ttnn,
            key_cache,
            key,
            value_cache,
            value,
            update_idxs_tensor=update_idxs_tensor,
            update_idxs=update_idxs,
            page_table=page_table,
        )

    def paged_sdpa_decode(
        self,
        query,
        key_cache,
        value_cache,
        page_table,
        cache_position,
        *,
        scale=None,
        memory_config=None,
        program_config=None,
        compute_kernel_config=None,
        op_name="paged_scaled_dot_product_attention_decode",
    ):
        self._record(op_name)
        return ttnn_ops.paged_sdpa_decode(
            self.ttnn,
            query,
            key_cache,
            value_cache,
            page_table,
            cache_position,
            scale=scale,
            memory_config=self.resolve_memory_config(memory_config),
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        *,
        is_causal=True,
        scale=None,
        memory_config=None,
        program_config=None,
        compute_kernel_config=None,
        attention_mask=None,
        op_name="scaled_dot_product_attention",
    ):
        self._record(op_name)
        return ttnn_ops.scaled_dot_product_attention(
            self.ttnn,
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale,
            memory_config=self.resolve_memory_config(memory_config),
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            attention_mask=attention_mask,
        )

    def fill_cache(
        self,
        cache_tensor,
        update_tensor,
        *,
        user_id=0,
        page_table=None,
        op_name="fill_cache",
    ):
        self._record(op_name)
        return ttnn_ops.fill_cache(
            self.ttnn,
            cache_tensor,
            update_tensor,
            user_id=user_id,
            page_table=page_table,
        )

    def paged_fill_cache(
        self,
        cache_tensor,
        update_tensor,
        page_table,
        *,
        batch_idx=0,
        batch_idx_tensor=None,
        compute_kernel_config=None,
        op_name="paged_fill_cache",
    ):
        self._record(op_name)
        return ttnn_ops.paged_fill_cache(
            self.ttnn,
            cache_tensor,
            update_tensor,
            page_table,
            batch_idx=batch_idx,
            batch_idx_tensor=batch_idx_tensor,
            compute_kernel_config=compute_kernel_config,
        )

    def nlp_concat_heads_decode(
        self,
        attn,
        *,
        num_heads,
        memory_config=None,
        op_name="nlp_concat_heads_decode",
    ):
        self._record(op_name)
        return ttnn_ops.nlp_concat_heads_decode(
            self.ttnn,
            attn,
            num_heads=num_heads,
            memory_config=self.resolve_memory_config(memory_config),
        )

    def concat_heads_prefill(
        self,
        attn,
        *,
        memory_config=None,
        op_name="concat_heads_prefill",
    ):
        self._record(op_name)
        return ttnn_ops.concat_heads_prefill(
            self.ttnn,
            attn,
            memory_config=self.resolve_memory_config(memory_config),
        )

    def concat(
        self,
        tensors,
        *,
        dim=-1,
        memory_config=None,
        op_name="concat",
    ):
        self._record(op_name)
        kwargs = {"dim": dim}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        return self.ttnn.concat(tensors, **kwargs)

    def argmax(self, tensor, *, dim=-1, op_name="argmax_or_sampling"):
        self._record(op_name)
        return self.ttnn.argmax(tensor, dim=dim)

    def force_argmax(
        self,
        tensor,
        *,
        op_name="argmax_or_sampling",
    ):
        untilize = getattr(self.ttnn, "untilize", None)
        argmax = getattr(self.ttnn, "argmax", None)
        if not callable(untilize) or not callable(argmax):
            raise ttnn_ops.UnsupportedTTNNOp(
                "force_argmax",
                (("untilize",), ("argmax",)),
            )
        self._record(op_name)
        shape = _tensor_shape(tensor)
        reshape = getattr(self.ttnn, "reshape", None)
        if (
            callable(reshape)
            and shape is not None
            and len(shape) == 4
            and shape[0] == 1
            and shape[1] > 1
            and shape[2] == 1
        ):
            logical_shape = (1, 1, shape[1], shape[3])
            self._record(f"{op_name}.reshape")
            try:
                tensor = reshape(tensor, logical_shape, logical_shape)
            except TypeError:
                tensor = reshape(tensor, logical_shape)
        self._record(f"{op_name}.untilize")
        row_major = untilize(tensor, use_multicore=True)
        self._record(f"{op_name}.multicore")
        return argmax(
            row_major,
            dim=-1,
            keepdim=False,
            use_multicore=True,
        )

    def local_argmax(
        self,
        tensor,
        *,
        vocab_start,
        op_name="lm_head.local_argmax",
    ):
        topk = getattr(self.ttnn, "topk", None)
        typecast = getattr(self.ttnn, "typecast", None)
        add = getattr(self.ttnn, "add", None)
        if not callable(topk) or not callable(typecast) or not callable(add):
            raise ttnn_ops.UnsupportedTTNNOp(
                "local_argmax",
                (("topk",), ("typecast",), ("add",)),
            )
        uint32 = getattr(self.ttnn, "uint32", None)
        if uint32 is None:
            raise ttnn_ops.UnsupportedTTNNOp(
                "local_argmax.uint32",
                (("uint32",),),
            )
        self._record(f"{op_name}.topk")
        values, indices = topk(
            tensor,
            k=1,
            dim=-1,
            largest=True,
            sorted=False,
        )
        self._record(f"{op_name}.typecast")
        indices = typecast(indices, uint32)
        if int(vocab_start) != 0:
            self._record(f"{op_name}.offset")
            indices = add(indices, int(vocab_start))
        return values, indices

    def global_argmax(
        self,
        candidate_values,
        candidate_indices,
        *,
        memory_config=None,
        op_name="argmax_or_sampling",
    ):
        if not candidate_values or len(candidate_values) != len(candidate_indices):
            raise ValueError(
                "global_argmax requires matching non-empty value/index candidates"
            )
        topk = getattr(self.ttnn, "topk", None)
        gather = getattr(self.ttnn, "gather", None)
        if not callable(topk) or not callable(gather):
            raise ttnn_ops.UnsupportedTTNNOp(
                "global_argmax",
                (("topk",), ("gather",)),
            )
        self._record(op_name)
        kwargs = {"dim": -1}
        if memory_config is not None:
            kwargs["memory_config"] = self.resolve_memory_config(memory_config)
        self._record(f"{op_name}.concat_values")
        values = self.ttnn.concat(candidate_values, **kwargs)
        self._record(f"{op_name}.concat_indices")
        indices = self.ttnn.concat(candidate_indices, **kwargs)
        self._record(f"{op_name}.topk")
        _, winner_slot = topk(
            values,
            k=1,
            dim=-1,
            largest=True,
            sorted=False,
        )
        self._record(f"{op_name}.gather")
        return gather(indices, -1, winner_slot)

    def normalize_decode_token(
        self,
        token,
        *,
        batch_size,
        op_name="normalize_argmax_token",
    ):
        shape = getattr(token, "shape", None)
        if shape is None:
            return token
        shape = [int(dim) for dim in shape]
        if shape in ([batch_size, 1], [batch_size]):
            return token
        slice_op = getattr(self.ttnn, "slice", None)
        squeeze_op = getattr(self.ttnn, "squeeze", None)
        reshape_op = getattr(self.ttnn, "reshape", None)
        if len(shape) == 3 and shape[1] == 1 and shape[2] == 1:
            logical_batch = shape[0]
            if callable(reshape_op):
                self._record(f"{op_name}.reshape")
                try:
                    return reshape_op(
                        token,
                        (logical_batch, 1),
                        (logical_batch, 1),
                    )
                except TypeError:
                    return reshape_op(token, (logical_batch, 1))
            return token
        if (
            len(shape) == 3
            and shape[0] == 1
            and shape[1] == 1
            and shape[2] == batch_size
        ):
            if callable(reshape_op):
                self._record(f"{op_name}.reshape")
                try:
                    return reshape_op(
                        token,
                        (batch_size, 1),
                        (batch_size, 1),
                    )
                except TypeError:
                    return reshape_op(token, (batch_size, 1))
            if callable(squeeze_op):
                self._record(f"{op_name}.squeeze.0")
                token = squeeze_op(token, 0)
                self._record(f"{op_name}.squeeze.1")
                return squeeze_op(token, 0)
            return token
        if (
            callable(slice_op)
            and len(shape) == 2
            and shape[0] == batch_size
            and shape[1] > 1
        ):
            self._record(f"{op_name}.slice")
            return slice_op(
                token,
                [0, shape[1] - 1],
                [batch_size, shape[1]],
            )
        if (
            callable(reshape_op)
            and len(shape) == 4
            and shape[0] == 1
            and shape[2] == 1
            and shape[3] == 1
        ):
            logical_batch = shape[1]
            self._record(f"{op_name}.reshape")
            try:
                return reshape_op(
                    token,
                    (logical_batch, 1),
                    (logical_batch, 1),
                )
            except TypeError:
                return reshape_op(token, (logical_batch, 1))
        if (
            callable(reshape_op)
            and len(shape) == 4
            and shape[0] == 1
            and shape[1] == 1
            and shape[3] == 1
        ):
            logical_batch = shape[2]
            self._record(f"{op_name}.reshape")
            try:
                return reshape_op(
                    token,
                    (logical_batch, 1),
                    (logical_batch, 1),
                )
            except TypeError:
                return reshape_op(token, (logical_batch, 1))
        if not callable(slice_op) or not callable(squeeze_op):
            return token
        if (
            len(shape) == 3
            and shape[0] == 1
            and shape[1] == batch_size
            and shape[2] >= 1
        ):
            if shape[2] > 1:
                self._record(f"{op_name}.slice")
                token = slice_op(
                    token,
                    [0, 0, 0],
                    [1, batch_size, 1],
                )
            self._record(f"{op_name}.squeeze")
            return squeeze_op(token, 0)
        if (
            len(shape) == 4
            and shape[0] == 1
            and shape[1] == 1
            and shape[2] == batch_size
            and shape[3] >= 1
        ):
            if shape[3] > 1:
                self._record(f"{op_name}.slice")
                token = slice_op(
                    token,
                    [0, 0, 0, 0],
                    [1, 1, batch_size, 1],
                )
            self._record(f"{op_name}.squeeze.0")
            token = squeeze_op(token, 0)
            self._record(f"{op_name}.squeeze.1")
            return squeeze_op(token, 0)
        return token
