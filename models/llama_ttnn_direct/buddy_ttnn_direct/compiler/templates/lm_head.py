from __future__ import annotations

_SOURCE = """\
    def lm_head_argmax(self, hidden, stage="decode"):
        # Template: official_split_lm_head + __GENERATION_TEMPLATE__
        lm_head_config = self.config.lm_head
        generation_config = _optional_attr(
            self.config, "generation", None
        )
        generation_mode = _optional_attr(
            generation_config, "mode", "greedy"
        )
        retain_logits = bool(
            _optional_attr(lm_head_config, "retain_logits", False)
        )
        greedy_token_output = generation_mode == "greedy" and not retain_logits
        argmax_strategy = _optional_attr(
            lm_head_config,
            "argmax_strategy",
            "full_logits_untilize_multicore_argmax",
        )
        local_global_argmax = (
            greedy_token_output and argmax_strategy == "local_global_argmax"
        )
        official_force_argmax = (
            greedy_token_output
            and argmax_strategy == "full_logits_untilize_multicore_argmax"
        )
        if greedy_token_output and not (
            local_global_argmax or official_force_argmax
        ):
            raise ValueError(f"unsupported LM-head argmax strategy: {argmax_strategy}")
        split_count = int(
            _optional_attr(
                lm_head_config,
                "split_count",
                GENERATED_LM_HEAD_SPLIT_COUNT,
            )
        )
        program_configs = _optional_attr(
            lm_head_config, "program_configs", None
        )
        split_configs = _optional_attr(
            lm_head_config, "splits", None
        )
        shard_logits = (
            []
            if (not local_global_argmax or self.observer is not None)
            else None
        )
        candidate_values = []
        candidate_indices = []
        for shard_id in range(split_count):
            split_params = self.parameters.lm_head.splits[shard_id]
            split_config = None
            if split_configs is not None and shard_id < len(split_configs):
                split_config = split_configs[shard_id]
            program_config = _optional_attr(
                split_config, "program_config", None
            )
            if (
                program_config is None
                and program_configs is not None
                and shard_id < len(program_configs)
            ):
                program_config = program_configs[shard_id]
            logits_i = self.ops.linear(
                hidden,
                split_params.weight,
                memory_config=_optional_attr(
                    lm_head_config, "output_memory_config"
                ),
                program_config=program_config,
                compute_kernel_config=_optional_attr(
                    lm_head_config, "compute_kernel_config"
                ),
                dtype=_optional_attr(lm_head_config, "output_dtype"),
                op_name="split_lm_head",
            )
            if shard_logits is not None:
                shard_logits.append(logits_i)
            if local_global_argmax:
                vocab_start = _optional_attr(
                    split_config, "vocab_start", None
                )
                if vocab_start is None:
                    raise ValueError(
                        "local/global LM-head argmax requires vocab_start "
                        f"for shard {shard_id}"
                    )
                value_i, index_i = self.ops.local_argmax(
                    logits_i,
                    vocab_start=int(vocab_start),
                    op_name=f"lm_head.local_argmax[{shard_id}]",
                )
                candidate_values.append(value_i)
                candidate_indices.append(index_i)

        logits = None
        if shard_logits is not None:
            logits = self.ops.concat(
                shard_logits,
                dim=-1,
                memory_config=_optional_attr(
                    lm_head_config, "concat_memory_config"
                ),
                op_name="split_lm_head.concat",
            )
            self._observe(f"{stage}.logits", logits)
        if official_force_argmax:
            token = self.ops.force_argmax(
                logits,
                op_name="argmax_or_sampling",
            )
            return self.ops.normalize_decode_token(
                token,
                batch_size=int(
                    _optional_attr(
                        self.config,
                        "batch_size",
                        1,
                    )
                    or 1
                ),
            )
        if local_global_argmax:
            token = self.ops.global_argmax(
                candidate_values,
                candidate_indices,
                memory_config=_optional_attr(
                    lm_head_config, "concat_memory_config"
                ),
                op_name="argmax_or_sampling",
            )
            return self.ops.normalize_decode_token(
                token,
                batch_size=int(
                    _optional_attr(
                        self.config,
                        "batch_size",
                        1,
                    )
                    or 1
                ),
            )
        if logits is None:
            raise RuntimeError("LM-head logits were not retained")
        return logits


"""


def render_lm_head(*, generation_template: str) -> str:
    return _SOURCE.replace(
        "__GENERATION_TEMPLATE__",
        generation_template,
    )
