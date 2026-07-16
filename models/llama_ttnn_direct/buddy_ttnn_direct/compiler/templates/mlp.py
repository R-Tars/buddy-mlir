from __future__ import annotations

_SOURCE = """\
    def mlp_decode(self, layer_id, hidden, stage="decode"):
        # Template: official_gated_mlp_decode
        layer_params = self.parameters.layers[layer_id].mlp
        mlp_config = (
            self.config.mlp
            if stage == "decode"
            else _optional_attr(self.config, "prefill", self.config.mlp)
        )
        layer_overrides = _optional_attr(
            mlp_config, "layer_overrides", None
        )
        layer_config = _optional_attr(
            layer_overrides, str(layer_id), None
        )
        gate_up_compute_kernel_config = _optional_attr(
            layer_config,
            "gate_up_compute_kernel_config",
            _optional_attr(
                mlp_config,
                "gate_up_compute_kernel_config",
                _optional_attr(mlp_config, "compute_kernel_config"),
            ),
        )
        down_compute_kernel_config = _optional_attr(
            layer_config,
            "down_compute_kernel_config",
            _optional_attr(
                mlp_config,
                "down_compute_kernel_config",
                _optional_attr(mlp_config, "compute_kernel_config"),
            ),
        )
        intermediate_dtype = _optional_attr(
            mlp_config, "intermediate_dtype"
        )
        gate_up_template = _template_choice(
            self.config,
            "mlp.gate_up",
            "separate_gate_up",
        )
        activation_template = _template_choice(
            self.config,
            "mlp.activation_placement",
            "mul_fused_silu",
        )
        gate_program_config = _optional_attr(
            mlp_config, "gate_program_config"
        )
        packed_gate_up_program_config = _optional_attr(
            layer_config,
            "packed_gate_up_program_config",
            _optional_attr(
                mlp_config, "packed_gate_up_program_config"
            ),
        )
        packed_gate_up_output_memory_config = _optional_attr(
            layer_config,
            "packed_gate_up_output_memory_config",
            _optional_attr(
                mlp_config, "packed_gate_up_output_memory_config"
            ),
        )
        packed_gate_up_split_strategy = _optional_attr(
            layer_config,
            "packed_gate_up_split_strategy",
            _optional_attr(
                mlp_config, "packed_gate_up_split_strategy", "split"
            ),
        )
        packed_gate_up_split_output_memory_config = _optional_attr(
            layer_config,
            "packed_gate_up_split_output_memory_config",
            _optional_attr(
                mlp_config,
                "packed_gate_up_split_output_memory_config",
            ),
        )
        packed_gate_up_mul_input_memory_config = _optional_attr(
            layer_config,
            "packed_gate_up_mul_input_memory_config",
            _optional_attr(
                mlp_config, "packed_gate_up_mul_input_memory_config"
            ),
        )
        packed_gate_up_mul_conversion = _optional_attr(
            layer_config,
            "packed_gate_up_mul_conversion",
            _optional_attr(
                mlp_config, "packed_gate_up_mul_conversion", False
            ),
        )
        if gate_up_template in ("packed_gate_up", "packed_projection"):
            packed = self.ops.linear(
                hidden,
                layer_params.gate_up_proj.weight,
                memory_config=packed_gate_up_output_memory_config,
                program_config=packed_gate_up_program_config,
                compute_kernel_config=gate_up_compute_kernel_config,
                dtype=intermediate_dtype,
                op_name="mlp_gate_up_packed",
            )
            gate, up = self.ops.split_last_dim(
                packed,
                split_size=int(self.config.intermediate_size),
                strategy=packed_gate_up_split_strategy,
                memory_config=packed_gate_up_split_output_memory_config,
                op_name="split_gate_up",
            )
            if packed_gate_up_mul_conversion:
                gate = self.ops.to_memory_config(
                    gate,
                    memory_config=packed_gate_up_mul_input_memory_config,
                    op_name="to_memory_config.packed_gate_to_mul",
                )
                up = self.ops.to_memory_config(
                    up,
                    memory_config=packed_gate_up_mul_input_memory_config,
                    op_name="to_memory_config.packed_up_to_mul",
                )
        else:
            gate = self.ops.linear(
                hidden,
                layer_params.gate_proj.weight,
                memory_config=_optional_attr(
                    mlp_config, "gate_output_memory_config"
                ),
                program_config=gate_program_config,
                compute_kernel_config=gate_up_compute_kernel_config,
                dtype=intermediate_dtype,
                activation=(
                    "silu"
                    if (
                        activation_template == "gate_linear_fused_silu"
                        and gate_program_config is None
                    )
                    else None
                ),
                op_name="mlp_gate",
            )
            up = self.ops.linear(
                hidden,
                layer_params.up_proj.weight,
                memory_config=_optional_attr(
                    mlp_config, "up_output_memory_config"
                ),
                program_config=_optional_attr(
                    mlp_config, "up_program_config"
                ),
                compute_kernel_config=gate_up_compute_kernel_config,
                dtype=intermediate_dtype,
                op_name="mlp_up",
            )
        if activation_template == "gate_linear_fused_silu":
            mid = self.ops.multiply(
                gate,
                up,
                memory_config=_tensor_memory_config(gate),
                dtype=intermediate_dtype,
                op_name="mul_gate_up",
            )
        else:
            mid = self.ops.mul_silu(
                gate,
                up,
                memory_config=(
                    packed_gate_up_mul_input_memory_config
                    if gate_up_template in (
                        "packed_gate_up",
                        "packed_projection",
                    )
                    else _tensor_memory_config(gate)
                ),
                dtype=intermediate_dtype,
                op_name="mul_silu",
            )
        return self.ops.linear(
            mid,
            layer_params.down_proj.weight,
            memory_config=_optional_attr(
                mlp_config, "down_output_memory_config"
            ),
            program_config=_optional_attr(
                mlp_config, "down_program_config"
            ),
            compute_kernel_config=down_compute_kernel_config,
            dtype=_optional_attr(mlp_config, "output_dtype"),
            op_name="mlp_down",
        )

"""


def render_mlp() -> str:
    return _SOURCE
