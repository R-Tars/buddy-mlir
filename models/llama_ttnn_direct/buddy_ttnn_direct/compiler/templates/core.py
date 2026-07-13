from __future__ import annotations


_SOURCE = """\
class BuddyLlama31TTNN:
    def __init__(self, device, parameters, config, observer=None):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.ops = TTNNCompatOps(ttnn)
        self.observer = observer

    def _observe(self, name, tensor, **metadata):
        if self.observer is None:
            return
        observe = getattr(self.observer, "observe", None)
        if callable(observe):
            observe(name, tensor, ops=self.ops, **metadata)
        elif callable(self.observer):
            self.observer(name, tensor, ops=self.ops, **metadata)

    def _attention_heads_memory_config(self):
        return self._height_sharded_memory_config(
            batch_size=int(_optional_attr(self.config, "batch_size", 1) or 1),
            head_dim=int(_optional_attr(self.config, "head_dim", 1) or 1),
        )

    def _height_sharded_memory_config(self, *, batch_size, head_dim):
        ttnn_module = self.ops.ttnn
        create_sharded = getattr(
            ttnn_module,
            "create_sharded_memory_config",
            None,
        )
        if callable(create_sharded):
            core_grid = self._batch_core_grid(batch_size=batch_size)
            shard_strategy = getattr(
                getattr(ttnn_module, "ShardStrategy", None),
                "HEIGHT",
                None,
            )
            shard_orientation = getattr(
                getattr(ttnn_module, "ShardOrientation", None),
                "ROW_MAJOR",
                None,
            )
            tile_size = int(getattr(ttnn_module, "TILE_SIZE", 32))
            if core_grid is not None and shard_strategy is not None:
                try:
                    return create_sharded(
                        shape=(tile_size, head_dim),
                        core_grid=core_grid,
                        strategy=shard_strategy,
                        orientation=shard_orientation,
                        use_height_and_width_as_shard_shape=True,
                    )
                except Exception:
                    pass
        return getattr(
            ttnn_module,
            "L1_HEIGHT_SHARDED_MEMORY_CONFIG",
            None,
        )

    def _batch_core_grid(self, *, batch_size):
        ttnn_module = self.ops.ttnn
        core_grid_type = getattr(ttnn_module, "CoreGrid", None)
        if not callable(core_grid_type):
            return None
        compute_grid = None
        compute_with_storage_grid_size = getattr(
            self.device,
            "compute_with_storage_grid_size",
            None,
        )
        if callable(compute_with_storage_grid_size):
            try:
                compute_grid = compute_with_storage_grid_size()
            except Exception:
                compute_grid = None
        physical_x = int(getattr(compute_grid, "x", 8) or 8)
        physical_y = int(getattr(compute_grid, "y", 8) or 8)
        grid_x = max(1, min(batch_size, physical_x))
        while grid_x > 1 and batch_size % grid_x != 0:
            grid_x -= 1
        grid_y = max(1, (batch_size + grid_x - 1) // grid_x)
        if grid_y > physical_y:
            return None
        try:
            return core_grid_type(y=grid_y, x=grid_x)
        except TypeError:
            return core_grid_type(grid_y, grid_x)

    def prefill_prompt(
        self,
        token_ids,
        kv_cache,
        page_table=None,
        valid_seq_len=None,
    ):
        hidden = self.embed(token_ids)
        cache_reports = []
        for layer_id in range(self.config.num_layers):
            hidden, kv_cache, cache_report = self.prefill_layer(
                layer_id,
                hidden,
                kv_cache,
                page_table,
                valid_seq_len=valid_seq_len,
            )
            cache_reports.append(cache_report)
            self._observe(
                f"prefill.layer.{layer_id}.hidden",
                hidden,
                layer_id=layer_id,
                valid_seq_len=valid_seq_len,
            )
        if valid_seq_len is not None:
            if isinstance(valid_seq_len, (list, tuple)):
                hidden = self.ops.select_sequence_positions(
                    hidden,
                    [int(value) - 1 for value in valid_seq_len],
                    op_name="select_last_prompt_hidden_by_user",
                )
            else:
                hidden = self.ops.select_sequence_position(
                    hidden,
                    int(valid_seq_len) - 1,
                    op_name="select_last_prompt_hidden",
                )
            hidden = self.ops.reshape_decode_hidden_for_layer(
                hidden,
                op_name="reshape_prefill_selected_hidden",
            )
        hidden = self.final_norm(hidden)
        self._observe("prefill.final_hidden", hidden)
        token = self.lm_head_argmax(hidden, stage="prefill")
        return token, kv_cache, cache_reports

    def prefill_layer(
        self,
        layer_id,
        hidden,
        kv_cache,
        page_table=None,
        valid_seq_len=None,
    ):
        residual = hidden
        hidden = self.rmsnorm(
            hidden, layer_id, kind="attn", stage="prefill"
        )
        hidden, kv_cache, cache_report = self.attention_prefill(
            layer_id,
            hidden,
            kv_cache,
            page_table,
            valid_seq_len=valid_seq_len,
        )
        hidden = self.ops.add(
            residual,
            hidden,
            op_name="residual_add.attn",
        )
        residual = hidden
        hidden = self.rmsnorm(
            hidden, layer_id, kind="mlp", stage="prefill"
        )
        hidden = self.mlp_decode(layer_id, hidden, stage="prefill")
        hidden = self.ops.add(
            residual,
            hidden,
            op_name="residual_add.mlp",
        )
        return hidden, kv_cache, cache_report

    def decode_step(self, token_ids, page_table, cache_position, kv_cache):
        hidden = self.embed(token_ids)
        for layer_id in range(self.config.num_layers):
            hidden = self.decode_layer(
                layer_id,
                hidden,
                page_table,
                cache_position,
                kv_cache,
            )
            self._observe(
                f"decode.layer.{layer_id}.hidden",
                hidden,
                layer_id=layer_id,
            )
        hidden = self.final_norm(hidden)
        self._observe("decode.final_hidden", hidden)
        token = self.lm_head_argmax(hidden, stage="decode")
        return token, kv_cache

    def decode_layer(self, layer_id, hidden, page_table, cache_position, kv_cache):
        hidden = self.ops.reshape_decode_hidden_for_layer(
            hidden,
            op_name="reshape_hidden_decode",
        )
        residual = hidden
        hidden = self.rmsnorm(hidden, layer_id, kind="attn")
        hidden = self.attention_decode(
            layer_id,
            hidden,
            page_table,
            cache_position,
            kv_cache,
        )
        hidden = self.ops.add(
            residual,
            hidden,
            op_name="residual_add.attn",
        )
        residual = hidden
        hidden = self.rmsnorm(hidden, layer_id, kind="mlp")
        hidden = self.mlp_decode(layer_id, hidden)
        hidden = self.ops.add(
            residual,
            hidden,
            op_name="residual_add.mlp",
        )
        return hidden

    def embed(self, token_ids):
        embedding_config = _optional_attr(
            self.config, "embedding", None
        )
        return self.ops.embedding(
            token_ids,
            self.parameters.embedding.weight,
            memory_config=_optional_attr(
                embedding_config, "output_memory_config"
            ),
            dtype=_optional_attr(
                embedding_config, "output_dtype"
            ),
        )

    def rmsnorm(self, hidden, layer_id, kind, stage="decode"):
        layer_params = self.parameters.layers[layer_id]
        if kind == "attn":
            norm_params = layer_params.input_norm
        elif kind == "mlp":
            norm_params = layer_params.post_attention_norm
        else:
            raise ValueError(f"unknown RMSNorm kind: {kind}")
        return self._rmsnorm_with_weight(
            hidden,
            norm_params.weight,
            config_kind="attention" if kind == "attn" else kind,
            use_kind_config=stage == "decode",
            op_name=f"rms_norm.{kind}",
        )

    def final_norm(self, hidden):
        return self._rmsnorm_with_weight(
            hidden,
            self.parameters.final_norm.weight,
            config_kind="final",
            use_kind_config=True,
            op_name="rms_norm.final",
        )

    def _rmsnorm_with_weight(
        self,
        hidden,
        weight,
        config_kind,
        use_kind_config,
        op_name,
    ):
        rms_config = _optional_attr(self.config, "rms_norm", None)
        kind_config = (
            _optional_attr(rms_config, config_kind, None)
            if use_kind_config
            else None
        )
        epsilon = _optional_attr(
            rms_config, "eps", GENERATED_RMS_NORM_EPS
        )
        hidden = self.ops.to_memory_config(
            hidden,
            memory_config=_optional_attr(
                kind_config, "input_memory_config"
            ) or _optional_attr(rms_config, "input_memory_config"),
            op_name=f"to_memory_config.{op_name}.input",
        )
        return self.ops.rms_norm(
            hidden,
            weight,
            epsilon=epsilon,
            program_config=_optional_attr(
                kind_config, "program_config"
            ),
            compute_kernel_config=_optional_attr(
                kind_config,
                "compute_kernel_config",
                _optional_attr(rms_config, "compute_kernel_config"),
            ),
            memory_config=_optional_attr(
                kind_config,
                "output_memory_config",
                _optional_attr(rms_config, "output_memory_config"),
            ),
            dtype=_optional_attr(rms_config, "output_dtype"),
            op_name=op_name,
        )

"""


def render_model_core() -> str:
    return _SOURCE
