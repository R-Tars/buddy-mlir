# TTNN Direct Phase 9 Refactor Report

## Status

`PHASE_9_STATUS=ACCEPTED`

Phase 9 changed tests and this report only. Production implementation stayed
byte-identical, no autotune campaign ran, no winner was promoted, and Phase 10
was not started.

## Test Ownership

`tests_diagnostics` changed from 8,189 to 4,689 LOC, a net deletion of 3,500
LOC. Python LOC changed from 8,170 to 4,670. The suite retains all 135 test
methods; pytest subtests increased from 21 to 46.

The two depth-sweep files were merged into `test_depth_sweeps.py`:

- deleted `test_diagnostics_depth_sweep.py` (417 LOC)
- deleted `test_generate_depth_sweep.py` (504 LOC)
- added `test_depth_sweeps.py` (276 LOC)

The following current contract owners were compacted with shared fixtures,
tables, field subsets, and `subTest` parameterization:

| Test owner | Before | After | Net |
| --- | ---: | ---: | ---: |
| `test_codegen_skeleton.py` | 1,211 | 318 | -893 |
| `test_runtime_modules.py` | 916 | 292 | -624 |
| `test_benchmark_parity.py` | 542 | 197 | -345 |
| `test_prepare_artifacts.py` | 321 | 88 | -233 |
| `test_config_emit.py` | 277 | 84 | -193 |
| `test_package_program.py` | 201 | 62 | -139 |
| `test_performance_correctness.py` | 219 | 139 | -80 |
| `test_plan_diff.py` | 220 | 152 | -68 |
| `test_template_registry.py` | 213 | 145 | -68 |
| `test_runtime_environment.py` | 230 | 164 | -66 |
| `test_autotune_profiler_audit.py` | 266 | 205 | -61 |
| `test_decode_loop.py` | 167 | 129 | -38 |
| `test_execution_graph_diff.py` | 163 | 128 | -35 |
| `test_attention_decode_template.py` | 51 | 39 | -12 |

All 135 old methods have an after owner. Migration status is 102
`PRESERVED_PARAMETERIZED` and 33 `PRESERVED_EXACT`; there are no `UNKNOWN`,
`DELETED_FOR_LOC`, `NOT_NEEDED`, or unmapped rows. The 168 before behavior IDs
equal the 168 after behavior IDs, with no missing or extra IDs.

`fakes.py` remains 792 LOC (growth zero), no `fixtures.py` was added, and no
product test was migrated or added.

## Protected Contracts

The exact Phase 8 diagnostic contract remains checked in, including the 16
ordered diagnose stages, all attention primitives, MLP, attention layer,
prefill, model-backed session, decode shell, decode step eager/trace/segmented,
and managed-device cleanup behavior.

The complete Phase 3.2 template-profile matrix remains covered: dry-run eager
and trace, no-device, one-device eager with persistent tensors, normal capture
and replay counts, unavailable API, begin failure, capture-body failure with
both cleanup outcomes, end failure, replay failure, PCC failure, release
failure, release-before-fallback, and one device enter/exit.

Phase 5/5.1 autotune fail-closed finalist and confirmation coverage, Phase 6
product validation, benchmark parity, performance correctness, execution graph
normalization/capture selection, profiler trace/clock/region/reconciliation,
depth sweeps, runtime trace/persistent-input behavior, and codegen contracts all
remain owned by checked-in tests.

The observed CLI surface exactly matches the frozen contract:

- root commands: 6 (`build`, `generate`, `profile`, `validate`, `inspect`, `diagnose`)
- validate suites: 5 (`dryrun`, `functional`, `device`, `performance`, `correctness`)
- diagnose stages: 16

## Identity Gates

The production tree hash is unchanged:

```text
372b39028a95c828b1066dff0a3204ba0d7a69c9301d6e794ec0cd3efb6aaaa7
```

Production changed files are zero. A fresh bundle was generated from the
frozen Phase 8 semantic graph, execution plan, canonical template config, and
the documented Llama 3.1 model path. All seven artifact hashes match the Phase
8 bundle. `run_decode.py` remains 140 LOC (within the 180 LOC limit), and both
generated Python files compile.

The frozen best-config identity is unchanged for config, precision, execution,
template choices, program configs, memory configs, and SDPA config. No tuning
or promotion occurred.

## Regression

The required software regression passed:

```text
python -m compileall -q models/llama_ttnn_direct
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests -q
  280 passed, 712 subtests passed
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
  135 passed, 46 subtests passed
focused mandatory contract selection
  212 passed, 696 subtests passed
git diff --check
  pass
```

Before adding this report, canonical tracked LOC was 83,862 -> 80,362, net
-3,500. Final canonical LOC including this required report is 80,500, so the
total Phase 9 tracked net is -3,362. Phase 1 cumulative removal is 38,677 LOC
(32.45%), with 44,747 LOC remaining to the 35,753 target.

## Hardware Policy

Hardware was not repeated. This is valid because Phase 9 is a test-only
refactor and the production tree, all seven generated artifacts, best config,
precision, and execution identities are byte-identical. The inherited Phase 8
P150A evidence remains applicable: 3-run median throughput 35.5836 tokens/s/user,
full-depth correctness passed, All-BF16 minimum PCC was 0.992598, and the
500-token quality gate passed.

`hardware_not_repeated_reason = "test-only refactor with byte-identical production and generated artifacts"`

## Evidence

Machine evidence is under:

```text
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase9_before/
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase9_after/
```

The after directory contains method and import inventories, behavior and
migration maps, exact test LOC/count comparisons, CLI and protected-contract
matrices, production/bundle/winner identities, software regression, and
hardware reuse evidence.
