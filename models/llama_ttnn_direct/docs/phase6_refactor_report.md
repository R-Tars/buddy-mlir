# TTNN Direct Phase 6 Legacy Validation Retirement Report

Status: accepted. Phase 7 has not started.

## Scope And Ownership

Phase 6 retires phase-era validation orchestration and leaves the supported
product surface on `build`, `generate`, `profile`, `validate`, `inspect`, and
`diagnose`. It does not change `runtime/**`, compiler templates,
`ttnn_compat/**`, canonical `autotune/**`, `correctness/**`, model math,
precision, or the frozen winner. The unrelated dirty `llvm` submodule is
excluded.

`reports/validation.py` remains the product validation owner and now contains
only `_product_validation_result` plus:

- `validate_dryrun`
- `validate_functional`
- `validate_device`
- `validate_performance`

The correctness suite continues to call `correctness.run` directly. The five
public suites remain `dryrun`, `functional`, `device`, `performance`, and
`correctness`.

## Reachability Audit

The before/after scan covered Python source, tests, CMake, shell and generated
runner source, docs, package exports, and `__all__`. No product, public CLI,
current diagnose, canonical autotune, correctness, CMake, generated-runner, or
unknown caller was found. Remaining textual references are classified as:

| Classification | Count |
| --- | ---: |
| `DOC_ONLY` | 10 |
| `LEGACY_INTERNAL` | 9 |
| `LEGACY_TEST_ONLY` | 6 |
| Active callers | 0 |

This permitted retirement of `diagnostics/validation_workflow.py`,
`diagnostics/legacy_validation.py`, and the legacy-only acceptance machinery
in `reports/validation.py`.

## Retired Stack And LOC

| File | Before LOC | After LOC | Change |
| --- | ---: | ---: | ---: |
| `diagnostics/validation_workflow.py` | 4670 | 0 | -4670 |
| `diagnostics/legacy_validation.py` | 555 | 0 | -555 |
| `reports/validation.py` | 3357 | 263 | -3094 |
| `tests_diagnostics/test_validate_direct.py` | 9019 | 0 | -9019 |
| `tests_diagnostics/test_report_helpers.py` | 0 | 178 | +178 |

The target files shrink by 17,160 lines. Across the complete tracked model
tree, source implementation shrinks by 8,315 lines, tests shrink by 8,821
lines, and documentation grows by 193 lines. Total tracked LOC changes from
108,850 to 91,907, a net deletion of 16,943 lines. Relative to the Phase 1
baseline of 119,177 lines, the cumulative reduction is 27,270 lines.

The retired-symbol inventory contains 63 exact entries: 32 from
`validation_workflow.py`, 10 from `legacy_validation.py`, 15 from legacy
`reports/validation.py`, and six class/helper definitions from
`test_validate_direct.py`. It includes `validate_direct`,
`preflight_real_decode`, `validate_real_decode`, both recovery functions,
`VALIDATION_STEPS`, `REAL_DECODE_VALIDATION_STEPS`, the legacy acceptance
functions, and the synthetic runtime-input helpers. The machine-readable
inventory records every name and original line span.

The deleted test module contained 77 tests. Sixty-two legacy workflow tests
and two legacy recovery tests were deleted. Thirteen generic report-helper
tests were preserved either by direct existing helper coverage or by the new
178-line `test_report_helpers.py`; no current diagnostic or product validation
contract test was discarded.

## Capability Migration

| Legacy capability | Modern path |
| --- | --- |
| semantic import / plan / program generation | `build` |
| bundle structural validation | `validate --suite dryrun` |
| plan/config comparison | `inspect` |
| prompt-conditioned real generation | `generate` |
| device/full-depth validation | `validate --suite device --require-full-depth` |
| performance validation | `validate --suite performance` |
| HF numerical correctness | `validate --suite correctness` |
| performance-recipe quality | `diagnose --stage performance-correctness` |
| MLP/attention/prefill/decode bring-up | current `diagnose --stage ...` paths |
| decode depth sweep | `diagnose --stage depth-sweep` |
| generate depth sweep | `diagnose --stage generate-depth-sweep` |
| benchmark parity | `diagnose --stage benchmark-parity` |
| graph comparison | `diagnose --stage execution-graph-diff` |

Phase-era validation orchestration was retired after
`build`/`generate`/`profile`/`validate`/`inspect`/`diagnose` became the
supported surfaces.

## Contract And Boundary Verification

Twelve frozen synthetic cases cover pass/fail behavior for dryrun, functional,
device, device full-depth, performance, and performance full-depth. Before and
after results are identical for schema version, command, suite, status, pass
flag, check count and names, failed checks, and expected/observed semantics.

The import boundary proves both retired files are absent and no source imports
them. All six root command help paths pass, the root command set is unchanged,
and the validate suite set is unchanged. All eleven required current diagnose
stages parse and complete their dry run. A current-HEAD fresh bundle contains
all seven required artifacts, passes `validate --suite dryrun`, and its
140-line `run_decode.py` compiles.

## Retained Report Helpers

Phase 6 does not broadly delete report helpers. `contracts.py`, `profiling.py`,
`runtime.py`, `schema.py`, and `validation.py` remain current;
`performance.py` and `runtime_diagnostics.py` remain test-covered. The orphan
inventory records these Phase 7 candidates without deleting them:

- `reports/artifacts.py`
- `reports/attention.py`
- `reports/config.py`
- `reports/depth.py`
- `reports/evidence.py`
- `reports/tensorization.py`

`reports/evidence.py` retains its phase-era formatter for that inventory, but
its retired acceptance imports are lazy so the current package import boundary
is not blocked.

## Frozen Winner And Fresh Bundle

The frozen winner remains the bundle generated in the `dbf5330` era. Its
seven artifact hashes and the config, precision, execution, memory, program,
template, and SDPA identities are byte-for-byte identical to Phase 5.1. All
device, correctness, quality, and performance regressions below use that
frozen bundle; no tuning campaign or winner promotion was run.

A fresh bundle generated from the current HEAD is structurally valid but is
not claimed to have seven hashes identical to the older frozen bundle.
`model.py`, `semantic_graph.json`, and `weights_manifest.json` are byte
identical. `README.md`, `config.json`, `execution_plan.json`, and
`run_decode.py` differ because the earlier `aaebc2a` compatibility-boundary
refactor changed the generated config/plan/runner representation. This is a
generation-schema difference predating Phase 6, not a Phase 6 runtime, model
math, precision, correctness, or winner change.

## P150A Regression

Fresh full-depth device validation passed all 9 checks at 32 layers and batch
32, with prompt-conditioned prefill/decode and KV cache populated by prefill.
Fresh All-BF16 validation passed 99/99 comparisons and 98 PCC checks; minimum
PCC is `0.9925982836726808` against `0.99`, and top token is `4999 == 4999`.

The fresh 500-token quality gate used official TT-Metal commit
`61e690c25202111b52cbc1fbc9148b6524070c6f` and passed:

| Metric | Official | Buddy |
| --- | ---: | ---: |
| Top-1 accuracy | 0.910 | 0.910 |
| Top-5 accuracy | 0.980 | 0.982 |
| Minimum-user top-1 / top-5 | - | 0.910 / 0.982 |
| Minimum-user greedy agreement | - | 0.964 |

Three fresh P150A performance runs used batch 32, prefill 256, cache 1024,
five warmups, 100 measured iterations, trace execution, persistent inputs,
after-prefill timing, and force argmax:

| Metric | Result |
| --- | ---: |
| Raw tokens/s/user | 35.588814, 35.563819, 35.587857 |
| Median tokens/s/user | 35.587857 |
| Median p50 decode latency | 28.058942 ms |
| Throughput CV | 0.032501% |
| Change from Phase 5.1 | -0.004269% |

Every run retained four captures, 105 executes, seven persistent inputs, and
zero program compilations after capture. The 1% regression and 1.5% CV gates
pass.

## Software Regression

| Check | Result |
| --- | --- |
| Python 3.12 `compileall -q models/llama_ttnn_direct` | Pass |
| Product tests | 278 passed, 712 subtests passed |
| Diagnostics tests | 161 passed, 27 subtests passed |
| Product validation contract | 12/12 identical |
| Six root command help paths | Pass |
| Eleven current diagnose dry runs | Pass |
| Fresh bundle dryrun and runner compile | Pass |
| `git diff --check` | Pass |

Machine-readable before/after evidence is under
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase6_before/`
and `phase6_after/`. Phase 7 has not started.
