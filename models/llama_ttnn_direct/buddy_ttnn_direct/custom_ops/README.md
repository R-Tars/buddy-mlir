# Buddy-TTNN Direct Custom Ops

Phase 15 reserves semantic template names for future custom fused TTNN regions.
No C++ device operation or Python wrapper is implemented in this phase.

## Reserved Templates

```text
custom_buddy_fused_mlp_decode
custom_buddy_lmhead_argmax_decode
```

The planner accepts these names so search/config layers can preserve intent.
Python TTNN codegen raises `CustomFusedRegionNotImplemented` when either
template is selected, rather than silently falling back to an official template.

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
argument patching handled by TTNN's operation framework. The first integration
should keep the semantic template names stable and add hardware smoke tests
before enabling these templates in default configs.
