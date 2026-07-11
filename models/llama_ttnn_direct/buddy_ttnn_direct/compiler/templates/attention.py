from __future__ import annotations


_SOURCE = """\
    def attention_prefill(
        self,
        layer_id,
        hidden,
        kv_cache,
        page_table=None,
        valid_seq_len=None,
    ):
        layer_params = self.parameters.layers[layer_id].attention
        prefill_config = _optional_attr(
            self.config,
            "prefill",
            self.config.attention,
        )

        qkv = self.ops.linear(
            hidden,
            layer_params.wqkv_packed.weight,
            memory_config=_optional_attr(
                prefill_config, "qkv_output_memory_config"
            ),
            program_config=_optional_attr(
                prefill_config, "qkv_program_config"
            ),
            compute_kernel_config=_optional_attr(
                prefill_config, "qkv_compute_kernel_config"
            ),
            dtype=_optional_attr(
                prefill_config, "qkv_output_dtype"
            ),
            op_name="qkv_linear",
        )
        qkv = self.ops.reshape_prefill_qkv_for_heads(
            qkv,
            op_name="reshape_qkv_prefill",
        )

        q, k, v = self.ops.split_qkv_heads_prefill(
            qkv,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            memory_config=_optional_attr(
                prefill_config, "qkv_heads_memory_config"
            ),
        )
        self._observe(
            f"prefill.layer.{layer_id}.q_pre_rope",
            q,
            valid_seq_len=valid_seq_len,
        )
        self._observe(
            f"prefill.layer.{layer_id}.k_pre_rope",
            k,
            valid_seq_len=valid_seq_len,
        )
        self._observe(
            f"prefill.layer.{layer_id}.v_pre_rope",
            v,
            valid_seq_len=valid_seq_len,
        )

        q, k = self.rotary_embedding_prefill(layer_id, q, k)
        self._observe(
            f"prefill.layer.{layer_id}.q_post_rope",
            q,
            valid_seq_len=valid_seq_len,
        )
        self._observe(
            f"prefill.layer.{layer_id}.k_post_rope",
            k,
            valid_seq_len=valid_seq_len,
        )
        attn = self.ops.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=_optional_attr(self.config.attention, "scale"),
            memory_config=_optional_attr(
                prefill_config, "sdpa_output_memory_config"
            ),
            program_config=_optional_attr(
                prefill_config, "sdpa_program_config"
            ),
            compute_kernel_config=_optional_attr(
                prefill_config, "sdpa_compute_kernel_config"
            ),
        )
        kv_cache, cache_report = self.fill_prefill_kv_cache(
            layer_id,
            k,
            v,
            kv_cache,
            page_table=page_table,
        )

        attn = self.ops.concat_heads_prefill(
            attn,
            memory_config=_optional_attr(
                prefill_config,
                "concat_heads_output_memory_config",
            ),
        )

        output = self.ops.linear(
            attn,
            layer_params.o_proj.weight,
            memory_config=_optional_attr(
                prefill_config, "o_proj_output_memory_config"
            ),
            program_config=_optional_attr(
                prefill_config, "o_proj_program_config"
            ),
            compute_kernel_config=_optional_attr(
                prefill_config, "o_proj_compute_kernel_config"
            ),
            dtype=_optional_attr(
                prefill_config, "o_proj_output_dtype"
            ),
            op_name="o_proj_linear",
        )
        return output, kv_cache, cache_report

    def attention_decode(
        self,
        layer_id,
        hidden,
        page_table,
        cache_position,
        kv_cache,
    ):
        # Template: official_paged_attention_decode
        layer_params = self.parameters.layers[layer_id].attention
        attention_config = self.config.attention

        qkv = self.ops.linear(
            hidden,
            layer_params.wqkv_packed.weight,
            memory_config=_optional_attr(
                attention_config, "qkv_output_memory_config"
            ),
            program_config=_optional_attr(
                attention_config, "qkv_program_config"
            ),
            compute_kernel_config=_optional_attr(
                attention_config, "qkv_compute_kernel_config"
            ),
            dtype=_optional_attr(
                attention_config, "qkv_output_dtype"
            ),
            op_name="qkv_linear",
        )
        qkv = self.ops.reshape_decode_qkv_for_heads(
            qkv,
            op_name="reshape_qkv_decode",
        )
        attention_heads_memory_config = _optional_attr(
            attention_config,
            "qkv_heads_memory_config",
        ) or self._attention_heads_memory_config()

        q, k, v = self.ops.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            memory_config=attention_heads_memory_config,
            op_name="nlp_create_qkv_heads_decode",
        )

        q, k = self.rotary_embedding_decode(
            layer_id, q, k, cache_position
        )
        kv_cache = self.paged_update_kv_cache(
            layer_id,
            k,
            v,
            page_table,
            cache_position,
            kv_cache,
        )
        layer_cache = kv_cache[layer_id]

        attn = self.ops.paged_sdpa_decode(
            q,
            layer_cache.k,
            layer_cache.v,
            page_table,
            cache_position,
            scale=_optional_attr(attention_config, "scale"),
            memory_config=_optional_attr(
                attention_config, "sdpa_output_memory_config"
            ),
            program_config=_optional_attr(
                attention_config, "sdpa_program_config"
            ),
            compute_kernel_config=_optional_attr(
                attention_config, "sdpa_compute_kernel_config"
            ),
        )

        attn = self.ops.to_memory_config(
            attn,
            memory_config=_optional_attr(
                attention_config,
                "concat_heads_input_memory_config",
            )
            or attention_heads_memory_config,
            op_name="to_memory_config.concat_heads_input",
        )
        attn = self.ops.nlp_concat_heads_decode(
            attn,
            num_heads=self.config.num_attention_heads,
            memory_config=None,
        )

        return self.ops.linear(
            attn,
            layer_params.o_proj.weight,
            memory_config=_optional_attr(
                attention_config, "o_proj_output_memory_config"
            ),
            program_config=_optional_attr(
                attention_config, "o_proj_program_config"
            ),
            compute_kernel_config=_optional_attr(
                attention_config, "o_proj_compute_kernel_config"
            ),
            dtype=_optional_attr(
                attention_config, "o_proj_output_dtype"
            ),
            op_name="o_proj_linear",
        )

    def rotary_embedding_prefill(self, layer_id, q, k):
        rotary_params = _optional_attr(
            self.parameters, "rotary", None
        )
        if rotary_params is None:
            layer_attention = self.parameters.layers[
                layer_id
            ].attention
            rotary_params = _optional_attr(
                layer_attention, "rotary", None
            )
        return self.ops.rotary_embedding_prefill(
            q,
            k,
            cos_matrix=_optional_attr(
                rotary_params, "cos_matrix", None
            ),
            sin_matrix=_optional_attr(
                rotary_params, "sin_matrix", None
            ),
            transformation_matrix=_optional_attr(
                rotary_params, "transformation_matrix", None
            ),
        )

    def rotary_embedding_decode(self, layer_id, q, k, cache_position):
        rotary_params = _optional_attr(
            self.parameters, "rotary", None
        )
        if rotary_params is None:
            layer_attention = self.parameters.layers[
                layer_id
            ].attention
            rotary_params = _optional_attr(
                layer_attention, "rotary", None
            )
        rotary_config = _optional_attr(self.config, "rotary", None)
        return self.ops.rotary_embedding_decode(
            q,
            k,
            cos_matrix=_optional_attr(
                rotary_params, "cos_matrix", None
            ),
            sin_matrix=_optional_attr(
                rotary_params, "sin_matrix", None
            ),
            transformation_matrix=_optional_attr(
                rotary_params, "transformation_matrix", None
            ),
            is_decode_mode=_optional_attr(
                rotary_config, "is_decode_mode", True
            ),
            op_name="rotary_embedding_decode",
        )

    def fill_prefill_kv_cache(
        self,
        layer_id,
        k,
        v,
        kv_cache,
        page_table=None,
    ):
        layer_cache = kv_cache[layer_id]
        key_shape = _tensor_shape(k)
        value_shape = _tensor_shape(v)
        batch_user_count = 1
        if key_shape is not None and len(key_shape) >= 1:
            batch_user_count = max(batch_user_count, key_shape[0])
        if value_shape is not None and len(value_shape) >= 1:
            batch_user_count = max(batch_user_count, value_shape[0])
        user_reports = []
        for user_id in range(batch_user_count):
            user_k = (
                self.ops.slice_batch_user(
                    k,
                    user_id,
                    op_name="slice.prefill_k",
                )
                if batch_user_count > 1
                else k
            )
            user_v = (
                self.ops.slice_batch_user(
                    v,
                    user_id,
                    op_name="slice.prefill_v",
                )
                if batch_user_count > 1
                else v
            )
            filled_k = self.ops.fill_cache(
                layer_cache.k,
                user_k,
                user_id=user_id,
                op_name="fill_cache.k",
            ) if page_table is None else self.ops.paged_fill_cache(
                layer_cache.k,
                user_k,
                page_table,
                batch_idx=user_id,
                op_name="fill_cache.k",
            )
            if filled_k is not None:
                layer_cache.k = filled_k
            filled_v = self.ops.fill_cache(
                layer_cache.v,
                user_v,
                user_id=user_id,
                op_name="fill_cache.v",
            ) if page_table is None else self.ops.paged_fill_cache(
                layer_cache.v,
                user_v,
                page_table,
                batch_idx=user_id,
                op_name="fill_cache.v",
            )
            if filled_v is not None:
                layer_cache.v = filled_v
            user_reports.append(
                {
                    "user_id": user_id,
                    "key_update_shape": _tensor_shape(user_k),
                    "value_update_shape": _tensor_shape(user_v),
                    "key_cache_memory_config": (
                        _tensor_memory_config(layer_cache.k)
                    ),
                    "value_cache_memory_config": (
                        _tensor_memory_config(layer_cache.v)
                    ),
                }
            )
        return kv_cache, {
            "layer_id": layer_id,
            "write_policy": (
                "fill_cache_per_user"
                if page_table is None
                else "paged_fill_cache_per_user"
            ),
            "page_table_shape": _tensor_shape(page_table),
            "filled_user_count": batch_user_count,
            "update_shape_layout": "batch_heads_seq_head_dim",
            "key_update_shape": key_shape,
            "value_update_shape": value_shape,
            "users": user_reports,
            "key_cache_memory_config": _tensor_memory_config(
                layer_cache.k
            ),
            "value_cache_memory_config": _tensor_memory_config(
                layer_cache.v
            ),
        }

    def paged_update_kv_cache(
        self,
        layer_id,
        k,
        v,
        page_table,
        cache_position,
        kv_cache,
    ):
        layer_cache = kv_cache[layer_id]
        self.ops.paged_update_cache(
            layer_cache.k,
            k,
            update_idxs_tensor=cache_position,
            page_table=page_table,
            op_name="paged_update_cache.k",
        )
        self.ops.paged_update_cache(
            layer_cache.v,
            v,
            update_idxs_tensor=cache_position,
            page_table=page_table,
            op_name="paged_update_cache.v",
        )
        return kv_cache

"""


def render_attention() -> str:
    return _SOURCE
