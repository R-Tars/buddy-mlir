# TTNN Direct Phase 12.1 Fused Profiler Trace Fix Report

## Status

```text
PHASE_12_1_STATUS=ACCEPTED
PHASE_13_ALLOWED=false
```

Phase 12.1 fixes only the two blockers identified by the Phase 12 audit: the
profiler can now analyze the frozen fused attention winner, and the historical
Phase 11 baseline handoff reject set is restored. Runtime, compiler, codegen,
correctness, the winner, precision, and execution are unchanged. Phase 13 has
not started.

## Phase 12 Failure

The Phase 12 parser assumed two separate Q/K RoPE operations followed by two
separate paged cache updates. The frozen winner instead selects:

```text
attention.rope      = fused_qk_rope
attention.kv_update = fused_paged_update
```

Its trace contains layout conversion costs before the fused operations. The
old state machine therefore failed closed at exact trace index 14:

```text
expected: RotaryEmbeddingLlamaDeviceOperation
observed: ReshardDeviceOperation

11 LayerNormDeviceOperation
12 MatmulDeviceOperation
13 NLPCreateQKVHeadsDecodeDeviceOperation
14 ReshardDeviceOperation
15 RotaryEmbeddingLlamaFusedQKDeviceOperation
16 ReshardDeviceOperation
17 ReshardDeviceOperation
18 PagedFusedUpdateCacheDeviceOperation
19 SdpaDecodeDeviceOperation
```

The failed capture was retained without replacement. Its source identities
remain:

```text
raw CSV:       74f1d2a883820171a3cf3588fd449228bab58c95471e98072f6c34912cbc4420
profile:       6b649c02b27499bb3914a6eafdb3668f0b0a484e34bd1081b39e18242ddd57e1
program config:b5f9365f4111b6ac1f2bbb164db04bec87b6e32e7ddc0e4cefe0356478f9a621
```

## Exact Fused Contract

The parser resolves both attention choices through the canonical
`autotune.templates` registry. It does not copy template names into a second
registry. The only new ownership edge is the exact
`diagnostics/autotune_profiler_audit.py -> autotune.templates` import;
`autotune -> diagnostics` remains empty.

Pinned TT-Metal source and the frozen CSV agree on the exact device OP CODE
identities:

| Semantic operation | Python API | Profiler OP CODE |
| --- | --- | --- |
| Fused Q/K RoPE | `ttnn.experimental.rotary_embedding_llama_fused_qk` | `RotaryEmbeddingLlamaFusedQKDeviceOperation` |
| Fused paged K/V update | `ttnn.experimental.paged_fused_update_cache` | `PagedFusedUpdateCacheDeviceOperation` |

The source commit is
`61e690c25202111b52cbc1fbc9148b6524070c6f`. The corresponding headers and
their hashes are frozen in `fused_device_op_identity.json`.

For fused Q/K RoPE, the state machine consumes only existing exact
`_LAYOUT_OPS` immediately after create-heads, assigns those conversions to
`rope`, and then requires the exact fused Q/K OP CODE. For fused KV update it
does the same immediately after RoPE, assigns conversions to `kv_update`, and
then requires the exact fused update OP CODE.

The current winner observes one `ReshardDeviceOperation` per layer in `rope`
and two per layer in `kv_update`. The parser's exact conversion whitelist also
contains `InterleavedToShardedDeviceOperation`,
`ShardedToInterleavedDeviceOperation`, and `TilizeDeviceOperation`; arbitrary
operations are not skipped. There is no substring matching, fuzzy matching,
or search-forward behavior.

Separate templates remain strict: exactly two
`RotaryEmbeddingLlamaDeviceOperation` and exactly two
`PagedUpdateCacheDeviceOperation` instances are required per layer. The same
Phase 12 separate capture still yields 676 assignments, 21 nonempty region
records, five budget answers, and a device/e2e ratio of
`0.9929633572708255`. Its Phase 12.1 report is byte-identical to the Phase 12
supported-program report.

Malformed and hybrid unit cases cover fused config with a separate trace,
separate config with a fused trace, missing fused Q/K, missing fused KV, and an
unknown operation before either fused operation. Every case fails at the exact
state-machine position.

## Frozen Capture Reanalysis

The same Phase 12 failed fused CSV, profile, and program config were analyzed
directly by the fixed code. No recapture or input substitution was used. It
now passes with:

```text
trace replay ops:             708
model regions:                 20 / 20 nonempty
region records with runtime:   21 / 21 nonempty
all trace ops assigned:        exactly once
device/e2e ratio:              0.994646019161457
budget-answer keys:            5 / 5 exact
```

The `rope` region contains `ReshardDeviceOperation` plus
`RotaryEmbeddingLlamaFusedQKDeviceOperation`. The `kv_update` region contains
`ReshardDeviceOperation` plus `PagedFusedUpdateCacheDeviceOperation`. Public
region names remain `rope` and `kv_update`; no fused-only region was added.

## Fresh P150A Acceptance

A fresh `autotune-profiler-audit` capture used the actual frozen fused winner
and the same pinned TT-Metal commit. It passed with:

```text
raw CSV SHA-256:               b528f84bc1dfaabbe55db155ca8a8fa7c3e4b7a975562e9e850d467b419bc9b5
status / passed:               pass / true
trace replay ops:              708
model regions:                  20 / 20 nonempty
region records with runtime:    21 / 21 nonempty
device/e2e ratio:               0.9952410885076554
decode p50:                     28.12155499123037 ms
capture / replay:               1 / 1
persistent inputs:              7
post-capture compilation:       0
budget-answer keys:             5 / 5 exact
```

Every observed bottleneck class is in the frozen public enum, JSON and CSV
outputs exist, and all trace operations receive exactly one assignment. This
one-iteration profiler run is an integration acceptance check, not a formal
throughput benchmark.

## Baseline Handoff Parity

Phase 12 accidentally allowed profiler-style
`runtime_inputs.token_update=captured_device_to_device_copy` to replace the
Phase 11 baseline's mandatory `runtime_context` evidence. The shared validator
now takes two small consumer policies:

```text
baseline: require_runtime_input_stability=false, handoff_evidence=runtime_context
profiler: require_runtime_input_stability=true,  handoff_evidence=runtime_inputs
```

This shares validation mechanics while preserving each consumer's historical
reject semantics. The baseline matrix is:

| Mutation | Phase 11 | Phase 12.1 |
| --- | --- | --- |
| `runtime_context` missing | FAIL | FAIL |
| `runtime_context` non-object | FAIL | FAIL |
| direct-device handoff missing | FAIL | FAIL |
| host handoff | FAIL | FAIL |
| host roundtrip `true` | FAIL | FAIL |
| valid direct-device context, no `runtime_inputs` | PASS | PASS |

Thus a correct profiler token-update field cannot substitute for missing or
invalid baseline context evidence. The profiler still requires the captured
device-to-device copy and all runtime-input stability fields.

## Frozen Product Identity

Phase 10.1 provenance remains
`c529fbd1c9be9a927a86985a3a79595977d440eba237c85341300651f926f87b`.
Phase 11 depth identity remains
`6dd526b9c64f7bb192c1f334203478534d5e77a06172572c6986c174f910d0b2`.
Runtime/compiler/codegen remains
`4d0c0e332fe33fea70fdfb0f0ff17394eb15ad3b7b888acacfe39ad3709806c9`.

A fresh build and dry-run validation passed. `model.py`, `config.json`,
`semantic_graph.json`, `execution_plan.json`, `weights_manifest.json`,
`run_decode.py`, and `README.md` all match Phase 12 byte-for-byte.
`run_decode.py` remains 140 lines. Best config, precision, execution, template,
program, memory, and SDPA identities are unchanged. No autotune campaign ran
and no winner was promoted.

These frozen identities permit reuse of the Phase 10.1 formal result:

```text
throughput:                 35.8723317007 tokens/s/user
decode p50:                 28.0515274952 ms
CV:                         0.0211981%
Buddy top-1 / top-5:        0.910 / 0.982
minimum greedy agreement:   0.964
```

## Regression And Size

```text
compileall:    pass
product:       282 passed / 718 subtests
diagnostics:   175 passed / 46 subtests
focused:       285 passed / 730 subtests
diff --check:  pass
```

Production changes add 100 and remove 61 lines, for a net `+39`; this is below
the 180-new-line implementation limit. Scoped tests add 160 and remove 16
lines, for a net `+144`; this is below the 220-new-line test limit. Phase 12.1
is a compatibility fix and has no deletion target.

## Evidence And Stop

Frozen inputs and complete machine-readable acceptance evidence are under:

```text
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase12_1_before/
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase12_1_after/
```

Phase 12.1 is complete. Phase 13 remains explicitly unstarted.
