# TTNN Direct Phase 4 Refactor Report

Status: quarantined historical search removal and all regression gates passed.

## Scope

Phase 4 started from commit
`ad3637e598676b3d33a4c823d44afcb8ca38b289`. It removes the quarantined
phase-era search implementation and its validation-only compatibility surface.
It does not rewrite or move either current autotuner, change model math,
precision, configuration, generated model code, or runtime execution. Phase 5
has not started. The unrelated dirty `llvm` path is excluded.

## Inventory and Dependency Audit

The frozen `future/historical_search` inventory contained eight tracked files
and 967 tracked lines:

| Component | Files | Lines |
| --- | ---: | ---: |
| Search implementation and package | 5 | 950 |
| Static search-space JSON | 2 | 11 |
| Quarantine README | 1 | 6 |
| Total | 8 | 967 |

The before scan found 202 textual references: 4 source imports, 141 test
references, 2 documentation references, 2 package exports, and 53 other
references. Product and current-autotuner imports were zero. All four source
imports were in the old diagnostics validation workflow and loaded only the
phase-era decode-step search, report, runner, and search-space helpers. The
current `diagnose --stage autotune` path instead resolves through
`diagnostics/autotune` and the semantic `autotune` package.

## Removal

All eight historical-search files and all 967 lines were deleted, for a 100%
historical tracked-line deletion rate. The now-empty `future/__init__.py` was
also deleted, so the `future` package no longer exists.

The 765-line `tests_diagnostics/test_search.py` was entirely specific to the
retired stack and was deleted. The 503-line `reports/autotune.py` schema helper
was also deleted because its only consumer was the same retired validation
workflow; it was not used by the current autotuner or its diagnostics stage.
In total, eleven complete files and 2,236 tracked lines were removed.

The validation workflow and report compatibility modules no longer emit or
validate the retired `search_dry_run`, `decode_step_autotune_dry_run`, or
`decode_step_autotune` sections. Their historical-only tests were removed from
`test_validate_direct.py`. No helper was migrated because the dependency audit
found no reusable helper imported by product code or either current tuner.

The exact file list, line counts, and pre-removal hashes are recorded in
`phase4_after/removed_historical_files.json`.

## Ownership Boundary

The import-boundary test now requires the `future` package to be absent and
scans every package Python source and test for either a future-package import or
the retired search marker. The after scan covers 204 Python files and reports:

```text
source historical-search references: 0
test historical-search references:   0
future-package imports:               0
```

This guard covers semantic, compiler, codegen, runtime, reports, TTNN
compatibility, CLI, diagnostics, autotune, and correctness ownership roots.

## Current Autotuner Freeze

Neither current tuner tree changed. Before and after identities match exactly:

| Tree | Files | Lines | Content SHA-256 |
| --- | ---: | ---: | --- |
| `autotune/**` | 26 | 22,315 | `0a3c164cae12b36313b04699d507ba77d554ed0f9c92456ebfc900de0c3c5f5c` |
| `diagnostics/autotune/**` | 4 | 1,085 | `bbe02cb29df5074539bb60af0d41c57720ae905b6737c2546b08a0c7ca78b52f` |

Current autotune and ownership regression coverage passed 149 tests and 677
subtests. Schema-v2 contracts, legality, active measurement planning, MatMul,
SDPA, packed MLP, layout, prefetch, ranking, confirmation, generalization,
final-campaign planning, and best-config handling remain covered.
`diagnose --stage autotune --dry-run` also passed.

## Best Config Identity

The frozen winner remains unchanged. Before and after hashes are identical:

```text
config:             1ed82e5b0f2f2ce8444fa881ce725f62a37d6f556fa2cee493cda6dc7e5fde5d
precision contract: 4c8d2ca23c0d6dc7fd1f07a9b96680ee534dc3e4a30b37c9fe1fe5501e41ff11
execution contract: b1cbda955efffc534afead92b7267c4875fbcedba45e8db0ae5a5c9faae96b8f
template choices:   8ee3b05ec01ef8c0854b60119c1a02f71a448379821e90aef4bb010e98416cc9
program configs:    c87dbd92cd9a8d9877dac9ff2c338f837375186211ec2e6a5dffa16716cbd5d8
memory configs:     5ccad7b894b69f4a3ba8f4973d89b798700b64326a94e24e6434bab1d0337c22
SDPA config:        ba9c0498d380fcc70bc97b2417c39bf5ed26cafc617d6545eb45cd4d08d776c2
```

Persistent runtime inputs remain selected and no new tuning campaign or winner
was produced.

## Size

Before this report, the Phase 4 source tree changed from 238 files / 112,150
lines to 227 files / 108,126 lines. The implementation, tests, and existing
documentation delta was +45 / -4,069 lines, net -4,024. Including this report,
the final Phase 4 tree is 228 files / 108,285 lines, a net Phase 4 reduction of
3,865 lines.

Against the Phase 1 frozen baseline of 253 files / 119,177 lines, the repository
has cumulatively removed 25 files and 10,892 tracked lines.

## Software and Bundle Validation

| Check | Result |
| --- | --- |
| `compileall -q models/llama_ttnn_direct` | Pass |
| Targeted historical validation tests | 77 passed |
| Product tests | 275 passed, 712 subtests passed |
| Diagnostics tests | 227 passed, 27 subtests passed |
| Six root commands | Pass |
| Current autotune dry-runs and tests | Pass |
| CMake `llama31_ttnn_direct_program` target | Pass |

The fresh generated bundle contains all seven required artifacts. Its
`run_decode.py` is 140 lines, compiles, and passes generate, profile, validate,
inspect, diagnose, and generated-runner dry-runs.

## P150A Regression

The fresh performance runs use batch 32, prefill 256, cache 1024, five warmups,
100 measured iterations, three repetitions, full trace, persistent inputs,
after-prefill timing, and force argmax. Raw throughput is `35.600501`,
`35.560901`, and `35.555811` tokens/s/user.

| Metric | Phase 3.2 | Phase 4 | Change |
| --- | ---: | ---: | ---: |
| Median tokens/s/user | 35.538711 | 35.560901 | +0.06244% |
| Median p50 decode latency | 28.087311 ms | 28.079519 ms | -0.007792 ms |
| Throughput CV | 0.08688% | 0.05616% | Pass |

Every run retained four trace captures, 105 trace replays, seven persistent
inputs, and zero compilation after capture. The <=1% regression and <=1.5% CV
gates passed.

## Correctness

Fresh full-depth generation passed with prompt-conditioned prefill, prefill
populated KV cache, generated text `!\nWe`, and user-zero tokens `[4999, 1687]`.
All-BF16 validation passed 99 comparisons and 98 numeric PCC checks; minimum
PCC was 0.992598 against the 0.99 threshold, and top token 4999 matched.

The fresh 500-token performance-recipe quality run also passed: official
top-1/top-5 was 0.91/0.98, Buddy and minimum-user top-1/top-5 were
0.91/0.982, and greedy agreement was 0.964.

Machine-readable before/after inventories, identities, ownership audit,
performance summary, correctness summary, and raw reports are under
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase4_*`.
