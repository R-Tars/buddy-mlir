# TTNN Direct Phase 5.1 Canonical Autotune Fix Report

Status: accepted. Phase 6 has not started.

## Scope

Phase 5.1 starts from `e9958f6d3309de2faabc85d779ef0b43b0d825f1` and
fixes only canonical autotune integration. It does not change model math,
precision, the product runtime hot path, the frozen best config, or search and
promotion algorithms. No autotuning campaign or new winner was produced. The
unrelated dirty `llvm` submodule is excluded.

The implementation delta is 249 additions and 9 deletions: 108 lines for the
diagnostics-side candidate gate and 141 additions across the evaluator,
campaign, CLI, and rebuild artifact. This stays within the 250-line limit; the
candidate gate stays within its 120-line limit.

## Default Confirmation Repair

The previous default evaluator normalized throughput but omitted top-level
`warmup` and `iterations`. `ConfirmationPolicy` therefore observed `None`
instead of `5` and `100`, so every real default long-confirmation report was
invalid even when the device profile succeeded.

`ModelCandidateEvaluator` now writes these fields from the canonical
`MeasurementContract`, and writes `execution_mode`, `runtime_input_mode`,
`after_prefill`, `sampling`, and `page_table` from the canonical
`CandidateConfig`. It also records `measurement_contract` and `program_dir`.
Subprocess report claims cannot override these values. Cached reports missing
the normalized contract are invalidated.

The default evaluator test now passes directly through `_normalize`,
`confirmation_runner`, `ConfirmationPolicy`, and `confirm_matched_ab`.
Matched `5x100x3` reports are valid with trace execution, persistent inputs,
and after-prefill timing.

## Candidate Quality Gate

A successful performance profile now leaves:

```text
gate_status = not_evaluated
correctness_passed = false
quality_passed = false
```

It can no longer impersonate correctness or token quality. The canonical
campaign accepts an injected `CandidateGateRunner`; only full-model finalists
invoke it. `autotune/**` does not import diagnostics.

The default non-dry diagnostics CLI builds the gate and injects it into the
campaign. It requires `--prompt` and `--official-tt-metal-root` before device
work. Dry-run remains prompt-free, official-root-free, and device-free.

The gate reuses `diagnostics/performance_correctness.py` and derives booleans
from the complete acceptance report. Correctness requires official sample
count, complete Buddy batch, aggregate top-1/top-5, and minimum-user
top-1/top-5 checks. Quality requires minimum-user official/Buddy greedy
agreement. Evidence is reused only when a passed report has an exact identity
match for candidate fingerprint, reference hash, 500-token count, runtime
identity, and workload. A test proves that a 5% faster challenger with a
failed gate is retained rather than promoted.

## Search Artifacts And Ownership

Generated `reproduce_build.sh` now calls the public root command `build`, not
the retired `build-program` command. A device-free execution returned zero and
produced all seven required program-bundle artifacts. The campaign retains its
six-stage schema and separately reports actual stage states; the current
`layout_beam` is truthfully `skipped` because Phase 5.1 does not wire a layout
proposal group.

The ownership audit passed:

```text
diagnostics/autotune directory: absent
autotune -> diagnostics imports: 0
diagnostics -> autotune imports: diagnostics/cli.py -> autotune/campaign.py only
retired layered-tuner symbols in Python source: 0
```

## Correctness And P150A Gate

The fresh 500-token gate used the frozen full-depth program, official
TT-Metal commit `61e690c25202111b52cbc1fbc9148b6524070c6f`, and the fixed
official reference corpus. It passed every acceptance check:

| Metric | Result |
| --- | ---: |
| Official sample count | 500 / 500 |
| Official top-1 / top-5 | 0.910 / 0.980 |
| Buddy top-1 / top-5 | 0.910 / 0.982 |
| Buddy minimum-user top-1 / top-5 | 0.910 / 0.982 |
| Minimum-user greedy agreement | 0.964 |
| `correctness_passed` / `quality_passed` | true / true |

The accepted Phase 5 full-depth functional and All-BF16 evidence remains
applicable because runtime math, precision, program artifacts, and the winner
did not change. All-BF16 minimum PCC remains `0.992598` against `0.99`, and
the top token remains matched.

## Frozen Winner And Performance

The frozen winner identity remains unchanged:

```text
config:             1ed82e5b0f2f2ce8444fa881ce725f62a37d6f556fa2cee493cda6dc7e5fde5d
precision contract: 4c8d2ca23c0d6dc7fd1f07a9b96680ee534dc3e4a30b37c9fe1fe5501e41ff11
execution contract: b1cbda955efffc534afead92b7267c4875fbcedba45e8db0ae5a5c9faae96b8f
```

Three fresh P150A runs used batch 32, prefill 256, cache 1024, five warmups,
100 measured iterations, full trace, persistent inputs, after-prefill timing,
and force argmax:

| Metric | Phase 5 reference | Phase 5.1 |
| --- | ---: | ---: |
| Median tokens/s/user | 35.583117 | 35.589377 |
| Median p50 decode latency | 28.054182 ms | 28.043237 ms |
| Throughput CV | 0.01825% | 0.02712% |
| Change | - | +0.01759% |

Raw throughput was `35.595258`, `35.572485`, and `35.589377`
tokens/s/user. Every run retained seven persistent inputs, four trace
captures, 105 trace executions, and zero program compilations after capture.
The <=1% regression and <=1.5% CV gates passed.

## Software Regression

| Check | Result |
| --- | --- |
| `compileall -q models/llama_ttnn_direct` | Pass |
| Product tests | 277 passed, 712 subtests passed |
| Diagnostics tests | 232 passed, 27 subtests passed |
| Focused Phase 5.1 autotune tests | 5 passed |
| `git diff --check` | Pass |

Machine-readable source, confirmation, gate, cache, rebuild, ownership,
winner, performance, and correctness evidence is under
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase5_1/`.
