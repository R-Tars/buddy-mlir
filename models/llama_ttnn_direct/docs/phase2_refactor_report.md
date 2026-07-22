# TTNN Direct Phase 2 Refactor Report

## Scope

Phase 2 converged the TTNN Direct product entry points and generated runner.
The implementation commits are `1c2a5a2` (`refactor: converge TTNN Direct
product CLI`) and `c2e134a` (`test: migrate TTNN Direct CLI regression
coverage`). The frozen before commit is
`736699d7eb488469818f29bb762af7c134010da9`.

This phase changes CLI/parser ownership, generated-runner forwarding, and
regression-test entry points only. It does not change model math, generated
model code, runtime decode/prefill/trace behavior, persistent inputs, TTNN
operations, configuration, precision recipes, autotune logic, or correctness
logic. No Phase 3 work is included.

## Tracked Size

The source refactor delta is measured against the Phase 2 before snapshot. The
final after inventory includes this small documentation artifact; large raw
evidence remains outside the source tree in the ignored build tree.

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Tracked files | 237 | 238 | +1 |
| Tracked lines | 116757 | 113132 | -3625 |
| `cli.py` | 3503 | 791 | -2712 |
| `codegen/program.py` | 753 | 255 | -498 |
| Generated `run_decode.py` | 1153 | 140 | -1013 |

The four production implementation files add 429 lines in the
before-to-after diff. Test changes are migration/regression coverage and are
reported separately. The implementation-only source reduction is 3,836 lines;
the final tracked reduction including this report is 3,625 lines. Both are
above the 2,500-line minimum, without deleting protected runtime, autotune,
diagnostics, or correctness modules.

## Product CLI

The root parser now exposes exactly these six commands:

```text
build generate profile validate inspect diagnose
```

`ProductCommandRegistry`, `legacy_parsers`, hidden parser construction, and
CLI-only legacy handlers/default paths are gone. The removed before snapshot
contained 28 hidden root commands and 33 top-level CLI functions were removed.
The hidden command list and handler list are recorded in:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/removed_cli_handlers.json
```

`diagnose` registers its stages through `diagnostics/cli.py` and imports stage
implementations lazily. The retained stages include MLP, attention primitive
and layer, prefill, decode step, legacy decode loop, both depth sweeps,
autotune, benchmark parity, execution-graph diff,
performance-correctness, and template profile.

The detailed command arguments, defaults, required fields, and help checks are
in `phase2_after/public_surface.json`. All six help commands exit with code 0.

## Capability Migration

| Legacy capability | Phase 2 entry |
| --- | --- |
| `import-llama`, `plan`, `codegen-python`, `emit-config`, `prepare-artifacts` | `build` internal steps |
| `build-program` | `build` |
| `package-program` | `build` plus the existing package API/CMake target |
| `materialize-parameters`, `tensorize-parameters` | product runtime initialization; program metadata through `inspect` |
| `diff-plan` | `inspect --official-template` |
| `diff-official-config` | `inspect --official-config` |
| smoke MLP/attention/prefill/decode stages | `diagnose --stage ...` |
| `prompt-decode-loop` | `diagnose --stage decode-loop-legacy` |
| `decode-depth-sweep` | `diagnose --stage depth-sweep` |
| `generate-depth-sweep` | `diagnose --stage generate-depth-sweep` |
| `profile-template` | `diagnose --stage template-profile` |
| `search`, `autotune-decode-step` | `diagnose --stage autotune` |
| `benchmark-parity` | `diagnose --stage benchmark-parity` |
| `execution-graph-diff` | `diagnose --stage execution-graph-diff` |
| `performance-correctness` | `diagnose --stage performance-correctness` |
| `validate-direct` | `validate --suite dryrun` plus the retained diagnostics workflow API |
| `validate-real-decode` | `validate --suite functional/device/performance/correctness` |

The complete machine-readable table is
`phase2_after/legacy_capability_migration.json`.

## Generated Runner

The generated `run_decode.py` remains in `PROGRAM_ARTIFACTS`, but is now a
140-line facade. It locates the repository import path, forwards to the six
command product CLI, and injects its own `--program-dir`. Its compatibility
translation layer is 61 lines and maps the historical modes without
duplicating product argument definitions.

The runner has no direct imports of smoke modules, `decode_loop`,
`profile_template`, legacy validation, or autotune runners. It compiles and
supports both product commands and the documented legacy-mode translations.
Evidence:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/generated_runner_surface.json
${BUDDY_BUILD}/models/llama31_ttnn_direct/program/
```

## Validation

Software validation after the final implementation/test commits:

| Check | Result |
| --- | --- |
| `compileall -q models/llama_ttnn_direct` | Pass |
| Product tests | 269 passed, 710 subtests passed |
| Diagnostics tests | 243 passed, 17 subtests passed |
| Six product help paths | Pass |
| Build bundle and seven artifacts | Pass |
| Product generate/profile/validate/inspect dry-runs | Pass |
| All documented diagnose dry-runs | Pass |
| Generated runner compile and facade checks | Pass |

The fresh bundle contains `model.py`, `config.json`,
`semantic_graph.json`, `execution_plan.json`, `weights_manifest.json`,
`run_decode.py`, and `README.md`.

## Performance

The P150A runs use the same final performance candidate, model, TTNN binary,
prompt corpus, and measurement contract as the Phase 2 before snapshot:

```text
tt-metal commit: 61e690c25202111b52cbc1fbc9148b6524070c6f
batch: 32
prefill: 256
cache: 1024
warmup: 5
iterations: 100
repetitions: 3
execution: full trace
runtime inputs: persistent
timing: after prefill
sampling: force argmax
```

The refactor does not alter the generated model/config used for this isolated
measurement. Raw reports are:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/performance_runs/repetition_1.json
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/performance_runs/repetition_2.json
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/performance_runs/repetition_3.json
```

| Metric | Phase 2 before | Phase 2 after | Change |
| --- | ---: | ---: | ---: |
| Median tokens/s/user | 35.587352 | 35.597434 | +0.0283% |
| Median p50 decode latency | 28.062326 ms | 28.039165 ms | -0.023161 ms |
| Throughput CV | 0.0213% | 0.0082% | Pass |

After raw throughput is `35.602899`, `35.597434`, and `35.596169`
tokens/s/user. Every run retained seven persistent inputs and zero compilation
after trace capture. The performance summary, including hashes and contract
checks, is in `phase2_after/performance_summary.json`.

## Correctness

Fresh full-depth functional validation used the real Llama 3.1 8B model,
32 layers, batch 32, prompt `Hello from TTNN Direct`, prefill 128, and cache
1024. It passed with prompt-conditioned prefill, paged KV cache populated from
prefill, and generated text `!\nWe`.

The protected numeric evidence was revalidated without modifying or rerunning
the precision/runtime implementation:

| Gate | Result |
| --- | --- |
| Full-depth functional generation | Pass |
| All-BF16 | 99 comparisons, minimum numeric PCC 0.992598, threshold 0.99 |
| Performance-recipe quality | Official top-1/top-5 0.91/0.98; Buddy top-1/top-5 0.91/0.982; greedy agreement 0.964 |

The correctness summary records source hashes and protected-file audit:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/correctness_summary.json
```

No protected runtime, autotune, correctness, compiler-template, or TTNN
compatibility file changed between the Phase 2 before commit and the
implementation commits.

## Evidence and Scope Audit

All machine-readable Phase 2 evidence is under the ignored build tree:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_before/
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase2_after/
```

The after inventory includes tracked LOC, public surface, CLI imports,
generated runner surface, removed handlers, migration table, dependency
report, source identity, performance, and correctness summaries. The existing
`llvm` submodule change was deliberately excluded from both commits.

## Phase 3 Boundary

Phase 2 is complete and stops here. The next suitable refactor should first
audit the diagnostics/autotune ownership boundary and their public import
facades, with the same dependency, correctness, and performance freeze. It
must not begin by moving or deleting runtime/model code without a new frozen
contract.
