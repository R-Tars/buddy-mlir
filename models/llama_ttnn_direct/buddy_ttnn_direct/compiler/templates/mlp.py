from __future__ import annotations


_SOURCE = """\
    def mlp_decode(self, layer_id, hidden):
        # Template: official_gated_mlp_decode
        layer_params = self.parameters.layers[layer_id].mlp
        mlp_config = self.config.mlp
        compute_kernel_config = _optional_attr(
            mlp_config, "compute_kernel_config"
        )
        intermediate_dtype = _optional_attr(
            mlp_config, "intermediate_dtype"
        )

        gate = self.ops.linear(
            hidden,
            layer_params.gate_proj.weight,
            memory_config=_optional_attr(
                mlp_config, "gate_output_memory_config"
            ),
            program_config=_optional_attr(
                mlp_config, "gate_program_config"
            ),
            compute_kernel_config=compute_kernel_config,
            dtype=intermediate_dtype,
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
            compute_kernel_config=compute_kernel_config,
            dtype=intermediate_dtype,
            op_name="mlp_up",
        )
        mid = self.ops.mul_silu(
            gate,
            up,
            memory_config=_tensor_memory_config(gate),
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
            compute_kernel_config=compute_kernel_config,
            dtype=_optional_attr(mlp_config, "output_dtype"),
            op_name="mlp_down",
        )

"""


def render_mlp() -> str:
    return _SOURCE
