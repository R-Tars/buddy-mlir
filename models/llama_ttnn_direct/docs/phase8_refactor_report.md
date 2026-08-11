# TTNN Direct Phase 8 Smoke Runtime Consolidation Report

Status: Phase 8 only. Phase 9 has not started.

## Scope And Outcome

Phase 8 retires the six top-level `smoke_*.py` implementation modules and
moves the retained user-facing stages under `diagnostics/`. It consolidates
generic device lifecycle in `runtime/device.py`, reusable MLP probe execution
in `autotune/microbench.py`, and diagnostic-only common support in
`diagnostics/support.py` and `diagnostics/attention_support.py`.

The refactor does not change compiler/codegen, correctness, precision,
autotune candidate enumeration/search/ranking/promotion, or product
prefill/decode/trace semantics. No autotune campaign was run and no winner was
promoted.

## Frozen Before State

The exact Phase 8 before canonical LOC is `89176`. The metric enumerates UTF-8
text files from `git ls-files -- models/llama_ttnn_direct`, counts each with
`len(text.splitlines())`, and counts binary files as zero text LOC.

The implementation inventory was:

| Before smoke implementation | Before LOC | Retained diagnostic owner | After LOC |
| --- | ---: | --- | ---: |
| `smoke_mlp.py` | 460 | `diagnostics/mlp.py` | 244 |
| `smoke_attention_primitive.py` | 1187 | `diagnostics/attention_primitive.py` | 688 |
| `smoke_attention_layer.py` | 1019 | `diagnostics/attention_layer.py` | 913 |
| `smoke_decode_shell.py` | 1299 | `diagnostics/decode_shell.py` | 965 |
| `smoke_prefill.py` | 856 | `diagnostics/prefill.py` | 598 |
| `smoke_single_layer_decode.py` | 2986 | `diagnostics/decode_step.py` | 1299 |
| **Total** | **7807** | | **4707** |

The six old smoke-specific test files were:

| Before smoke test | LOC |
| --- | ---: |
| `test_smoke_attention_layer.py` | 379 |
| `test_smoke_attention_primitive.py` | 665 |
| `test_smoke_decode_shell.py` | 1052 |
| `test_smoke_mlp.py` | 94 |
| `test_smoke_prefill.py` | 464 |
| `test_smoke_single_layer_decode.py` | 1779 |
| **Total** | **4433** |

The replacement contract suite is `tests_diagnostics/fakes.py` at 792 LOC and
`tests_diagnostics/test_diagnostic_contracts.py` at 684 LOC, totaling 1476
LOC or `33.295737%` of the old tests. The machine mapping preserves all 36 old
behavioral tests; no test was discarded as private-helper-only.

## Symbol Decisions

The before inventory classified 205 symbols: 7 stage entrypoints, 15
specialized probes, 2 autotune-shared probes, 133 generic diagnostic-support
symbols, and 48 cross-smoke private helpers.

Runtime duplicates were the smoke-local families for generated-model loading
and namespace conversion, tensor shape/dtype/runtime-integer conversion,
`decode_step_reference` and `prefill_reference`, generated observed-op and
shape/dtype/value checks, prefill execution, decode timing, trace management,
and segmented generate profiling. They now use the canonical runtime owners
listed below rather than keeping aliases or copied bodies.

The retained specialized probes are:

- `run_smoke_mlp`, `MLP_SMOKE_OPS`, and its PCC probe;
- `run_smoke_attention_primitive`, all seven isolated primitives, and
  `PRIMITIVE_EXPECTED_OBSERVED_OPS`;
- `run_smoke_attention_layer` and `ATTENTION_LAYER_EXPECTED_OBSERVED_OPS`;
- `run_smoke_decode_shell`, `DECODE_SHELL_OPS`, and the MLP-only numeric and
  structural reference path;
- `run_smoke_prefill`, `PREFILL_LAYER_OPS`, and `PREFILL_FINAL_OPS`;
- `run_smoke_single_layer_decode`, `run_smoke_decode_step`,
  `profile_decode_step`, and the decode op/parameter-role contracts.

The old `prepare_mlp_smoke_on_device` probe moved unchanged in responsibility
to `autotune.microbench`; generic `_managed_ttnn_device` behavior moved to
`runtime.device.managed_ttnn_device`. The duplicate smoke-local
`NoTTNNDeviceError` was removed in favor of `runtime.errors.NoTTNNDeviceError`.
No inventory symbol was classified `TEST_ONLY` or `DEAD`.

All six top-level `smoke_*.py` files are gone. There are zero compatibility
facades and no `KEEP_PUBLIC_API` or `KEEP_SPECIALIZED` exception.

## Ownership Convergence

Cross-smoke private import edges fell from 10 to 0. The two former autotune to
`smoke_mlp` imports fell to zero, while autotune to diagnostics remains zero.
Product runtime/compiler/codegen to diagnostics and correctness to diagnostics
also remain zero; only the CLI's lazy `diagnose` dispatch is allowed.

The retained diagnostics reuse these canonical owners:

| Concern | Canonical owner reused |
| --- | --- |
| Model loading and namespace conversion | `runtime.model_loader` |
| Tensor metadata and runtime integer tensors | `runtime.tensor_meta` |
| Shape, dtype, value, op, prefill, and decode references | `runtime.structural` |
| Prefill/decode plans and input/cache/config state | `runtime.plans`, `runtime.inputs`, `runtime.kv_cache`, `runtime.config_runtime` |
| Product prefill execution | `runtime.prefill` |
| Eager decode, trace, and profiling | `runtime.decode`, `runtime.trace`, `runtime.profile` |
| Generic device lifecycle and errors | `runtime.device`, `runtime.errors` |
| Diagnostic report/tensor factories | `diagnostics.support` |
| Attention-only shape/memory/page setup | `diagnostics.attention_support` |
| Reusable MLP measurement probe | `autotune.microbench` |

The managed-device contract opens and closes exactly once and closes on both
success and exception paths. Prefill preserves synthetic and injected state
plus cache population; decode-step preserves synthetic, injected, and
model-backed eager/trace execution; decode-step-profile preserves segmented
profiling; decode-shell remains attention-disabled and MLP-only.

## Stable Contracts

The exact ordered 16-stage tuple remains:

```text
mlp, attention-primitive, attention-layer, prefill, decode-shell,
decode-step, decode-step-profile, decode-loop-legacy, depth-sweep,
generate-depth-sweep, autotune, autotune-profiler-audit, benchmark-parity,
execution-graph-diff, performance-correctness, template-profile
```

All 16 stage regressions pass. The eight smoke-derived report contracts for
MLP, attention primitive, attention layer, prefill, decode shell, decode step,
decode-step profile, and template profile remain stable after ephemeral-field
normalization. Seven are exactly identical. Template profile differs only in
the expected ownership field:
`smoke_mlp.prepare_mlp_smoke_on_device` became
`autotune.microbench.prepare_mlp_smoke_on_device`.

Template profile retains one-device semantics. The eager hardware run opens
and closes one device with trace disabled. The trace run opens and closes one
device, performs one real capture, 5 warmup plus 20 measured replays (25
total), and one release. Both pass PCC with minimum `0.9998626112937927`.

## LOC Accounting

The final source implementation diff is 1695 added and 4253 deleted lines,
net `-2558`. Of those changes, 3573 unchanged lines are accounted as moves,
not deletions. Runtime device support adds 80 lines, below the 120-line gate,
and total new/changed source implementation is 1695 lines, below the 1800-line
gate.

The test diff is 1240 added and 4201 deleted lines, net `-2961`; the 1476-line
replacement smoke suite is below the 1600-line and 55% gates. Documentation
and final total accounting are regenerated from the staged tree:

```text
Phase 8 after canonical LOC: 83862
Phase 8 canonical net:       -5314
Total Git diff net:          -5314 (3144 added / 8458 deleted)
Cumulative removed:          35315 / 119177
Cumulative reduction:        29.632395513%
Lines above <=35753 target:  48109
```

## Validation

Static validation passes:

- `compileall`: pass;
- product tests: `280 passed, 712 subtests passed`;
- diagnostics tests: `135 passed, 21 subtests passed`;
- exact root-command, validate-suite, stage, import-boundary, canonical
  autotune/layout, report-schema, and product-import guards: pass;
- `git diff --check`: pass.

The fresh generated bundle contains all seven required artifacts. Its
`run_decode.py` is 140 lines, compiles, and the dry-run validation suite passes.

P150A smoke-derived regression passes all 14 checks, including seven attention
primitive coverage through hardware plus fake contracts, one attention layer,
prefill, MLP-only decode shell, eager and real-trace decode-step, segmented
eager and trace profile, and eager and real-trace template profile.

The frozen winner is unchanged: config, precision, execution, template,
program, memory, SDPA, and all seven generated-artifact hashes match the
before evidence. No campaign or promotion occurred.

Three matched full-model P150A runs produced `35.595057`, `35.576564`, and
`35.583623` tokens/s/user. Median throughput is `35.583623` tokens/s/user,
median p50 is `28.054850` ms, CV is `0.021412%`, and regression from the
Phase 7 `35.598902` baseline is `0.042920%`. Every run retains four captures,
105 executes, seven persistent inputs, zero post-capture compilation, and
released traces. The 1% regression and 1.5% CV gates pass.

Full-depth validation passes 9/9 checks. All-BF16 passes 99 comparisons with
98 PCC checks, minimum PCC `0.9925982836726808`, and top token `4999 == 4999`.
The 500-token batch-32 quality run passes: official top-1/top-5 is
`0.910/0.980`, Buddy aggregate and minimum-user are both `0.910/0.982`, and
minimum greedy agreement is `0.964`.

Phase 8 stops here. Phase 9 diagnostic/evidence consolidation is explicitly
out of scope.
