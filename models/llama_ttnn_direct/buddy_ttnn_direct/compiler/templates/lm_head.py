from __future__ import annotations


_SOURCE = """\
    def lm_head_argmax(self, hidden):
        # Template: official_split_lm_head + __GENERATION_TEMPLATE__
        lm_head_config = self.config.lm_head
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
        shard_logits = []
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
            shard_logits.append(logits_i)

        logits = self.ops.concat(
            shard_logits,
            dim=-1,
            memory_config=_optional_attr(
                lm_head_config, "concat_memory_config"
            ),
            op_name="split_lm_head.concat",
        )
        generation_config = _optional_attr(
            self.config, "generation", None
        )
        generation_mode = _optional_attr(
            generation_config, "mode", "greedy"
        )
        retain_logits = bool(
            _optional_attr(lm_head_config, "retain_logits", False)
        )
        if generation_mode == "greedy" and not retain_logits:
            token = self.ops.argmax(
                logits,
                dim=-1,
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
        return logits


"""


def render_lm_head(*, generation_template: str) -> str:
    return _SOURCE.replace(
        "__GENERATION_TEMPLATE__",
        generation_template,
    )
