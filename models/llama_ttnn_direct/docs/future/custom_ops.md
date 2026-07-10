# Future TTNN Direct Custom Ops

No C++ device operation or Python wrapper is currently implemented for custom
TTNN Direct fused regions. This document records possible future work only and
is not part of the template registry or product configuration schema.

## Reserved Templates

```text
custom_buddy_fused_mlp_decode
custom_buddy_lmhead_argmax_decode
```

These names are design placeholders. The planner deliberately rejects them;
main configs must use implemented official templates until a real operation,
wrapper, numerical reference, and device test exist.

## Planned Operations

```text
buddy_fused_mlp_decode:
  inputs: hidden, gate_weight, up_weight, down_weight
  output: hidden_out
  internal: gate linear, up linear, silu_mul, down linear

buddy_lmhead_argmax_decode:
  inputs: hidden, split lm_head weights
  output: token ids
  internal: local shard matmul + local argmax + global argmax
```

## Future TTNN Shape

The intended implementation path is a TTNN custom operation that either wraps a
small sequence of existing TTNN ops or defines a device operation with program
construction, circular buffers, kernels, compile-time arguments, and runtime
argument patching handled by TTNN's operation framework.

Before either placeholder can enter the registry:

1. Implement the TTNN operation and Python wrapper.
2. Add a torch numerical reference and tolerance gate.
3. Add a focused P150A device test and report schema.
4. Demonstrate a measured benefit over the composed TTNN path.
5. Add the template to an explicit experimental config before any main config.
