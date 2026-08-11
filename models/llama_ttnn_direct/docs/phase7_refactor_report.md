# TTNN Direct Phase 7 Orphan Report Pruning Report

Status: Phase 7 only. Phase 8 has not started.

## Scope

This phase removes only the report-helper dependency closure proven unreachable
after Phase 6. It does not modify runtime behavior, compiler/codegen,
templates, TTNN compatibility, canonical autotune, correctness, precision,
the frozen winner, or current smoke/diagnostic implementations.

The current report core is preserved:

- `reports/contracts.py`
- `reports/profiling.py`
- `reports/runtime.py`
- `reports/schema.py`
- `reports/validation.py`

No helper migration was needed. The deleted modules had no current generic
helper consumed outside the orphan closure.

## Canonical LOC

The release metric is defined as follows: enumerate UTF-8 text files returned
by `git ls-files -- models/llama_ttnn_direct`, count each with
`len(text.splitlines())`, and count binary files as zero text LOC. The same
algorithm is used for historical Git trees and the working tree. This is the
canonical metric; Git gross additions/deletions are retained only as a
reconciliation reference.

Replaying the Phase 1 tree at `dbf5330c296425e8c596ab960b692056db268715`
produces exactly `253 files / 119177 lines`. Phase 6 before is
`108850 lines`; Phase 7 before is `91907 lines`. The Phase 6 raw
`--no-renames` diff is `418 additions / 17361 deletions`, net `-16943`, which
matches the canonical `108850 -> 91907` delta.

Before this report was tracked, the Phase 7 working tree was `89020 lines`, a
net reduction of `2887` lines. The interim raw diff is `39 additions / 2926
deletions`, also net `-2887`; its additions do not include this untracked
report, while canonical LOC does. After the report is tracked, the final
Phase 7 tree is `89176 lines`. Its raw diff is `195 additions / 2926
deletions`, net `-2731`, which matches the canonical `91907 -> 89176` delta.
The raw and canonical metrics use different implementations (Git's diff line
accounting versus UTF-8 `splitlines()`), so the report keeps both values and
records the reconciliation explicitly rather than substituting one for the
other.

## Reachability And Closure

The six Phase 6 primary orphan roots were re-audited with an AST import graph,
reverse-import scan, tests/docs/CMake text scan, and generated-bundle scan:

| Candidate | LOC | Current non-test importers | Decision |
| --- | ---: | --- | --- |
| `reports/artifacts.py` | 42 | none | delete |
| `reports/attention.py` | 251 | none | delete |
| `reports/config.py` | 127 | none | delete |
| `reports/depth.py` | 223 | none | delete |
| `reports/evidence.py` | 1158 | none | delete |
| `reports/tensorization.py` | 473 | none | delete |

Fixpoint recomputation after removing `evidence.py` found:

| Transitive candidate | Last non-test importer | Decision |
| --- | --- | --- |
| `reports/performance.py` | `reports/evidence.py` | delete |
| `reports/runtime_diagnostics.py` | `reports/evidence.py` | delete |

The closure contains eight modules and removes `2737` implementation lines
(`2274` primary plus `463` transitive). `evidence.py` was not retained merely
because it contained generic-looking functions: its only source parent was the
retired validation stack, and its remaining importer was the helper-only test
file.

`runtime_environment.py` was audited separately and is retained. It has seven
current source importers: `runtime/reports.py`, `runtime/steady_profile.py`,
`runtime/prefill_profile.py`, and four current smoke/diagnostic modules. It
also has its dedicated environment test. Deleting it would break current
runtime report and diagnostic device-environment behavior.

The exact importer records, public symbols, parent deletion causes, and
fixpoint decisions are in the Phase 7 before/after machine evidence.

## Tests And Ownership

`tests_diagnostics/test_report_helpers.py` contained only tests for the deleted
report modules. It is deleted together with its production dependencies. The
dedicated `test_runtime_environment.py` is retained.

The existing checked-in `tests/test_validation_schema.py` remains untouched and
continues to protect the four product validation functions, CLI dryrun and
correctness wiring, failed-check behavior, and the six root commands. The
12-case product validation contract is regenerated before and after pruning;
all case schemas, statuses, check names, failed checks, and observed/expected
semantics are identical.

The ownership guard now proves every retired report module is absent and that
no source or test imports any retired report module. It separately asserts the
five current report-core modules remain present and that
`runtime_environment.py` remains present.

## Documentation

Architecture and diagnostics test documentation now describe only the current
report core and current diagnostic ownership. No Phase 8 smoke or diagnostics
consolidation was started.

## LOC Accounting

| Area | Change |
| --- | ---: |
| Retired report implementation | -2737 |
| Retired orphan helper tests | -178 |
| Documentation net | -1 |
| New boundary guard lines | +29 |
| Net before final phase report | -2887 |
| Phase 7 report | +156 |
| Final Phase 7 net | -2731 |

Relative to the Phase 1 baseline, the final canonical tree is `89176` lines:
`30001` lines or `25.173481460%` have been removed cumulatively. The 70%
target is `35753` tracked lines, so `53423` lines remain above that target.
These values are generated from the committed tree and are also recorded in
`phase7_after/canonical_tracked_loc.json` and
`phase7_after/loc_metric_reconciliation.json`.

## Regression Results

The product validation contract is `12/12` identical. The six root command
names and five validate suites remain unchanged, and all 11 required diagnose
stages still complete dry-run. The fresh generated bundle contains all seven
artifacts; its `run_decode.py` is `140` lines and compiles.

Phase 7 reuses the frozen winner without an autotune campaign or promotion.
The config, precision, execution, template, program, memory, and SDPA hashes
remain unchanged. Fresh full-depth validation passed `9/9`; All-BF16 passed
`99/99` comparisons with `98` PCC checks, minimum PCC
`0.9925982836726808`, and top token `4999 == 4999`.

The isolated fresh 500-token quality pair used TT-Metal commit
`61e690c25202111b52cbc1fbc9148b6524070c6f`, the fixed official corpus, batch
32, static Buddy prefill 256, prompt replay 256, cache 1024, and force-argmax
teacher forcing. Official accuracy was `0.910 / 0.980` (top-1/top-5); Buddy
was `0.910 / 0.982`, with minimum-user `0.910 / 0.982` and greedy agreement
`0.964`. Official and Buddy ran in separate fresh Python processes.

Three fresh P150A performance runs measured `35.596763`, `35.598902`, and
`35.635491` tokens/s/user. The median is `35.598902` tokens/s/user, p50 is
`28.039627` ms, and CV is `0.049912%`; every run retained four captures, 105
executes, seven persistent inputs, and zero compile after capture. The 1%
regression and 1.5% CV gates pass.

Phase 7 stops here. Phase 8 smoke/diagnostics implementation consolidation,
autotune changes, runtime hot-path work, generated-model work, and new
performance optimization are explicitly out of scope.
