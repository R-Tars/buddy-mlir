# TTNN Direct Phase 11 Depth-Sweep Consolidation Report

## Status

`PHASE_11_STATUS=ACCEPTED`

Phase 11 consolidated only the shared orchestration of `depth-sweep` and
`generate-depth-sweep`. It did not start Phase 12, change product execution,
run an autotune campaign, or promote a winner.

## Scope And Ownership

The new neutral owner is `diagnostics/depth_sweep_support.py` at 202 LOC,
below the 320 LOC limit. It owns only:

- generated-program `num_layers` loading;
- depth parsing, deduplication, validation, and `full`/`max`/`all` aliases;
- output-root resolution and isolation eligibility;
- the stop-after-failure iteration policy;
- isolated subprocess execution, metadata, and output excerpts;
- common counts, summaries, check construction, field projection, exception
  details, report persistence, and JSON writing.

Decode keeps its profile record schema, missing-report payload, throughput,
reference, trace, section/layer breakdown, and LM-head acceptance semantics.
Generate keeps its record and base report schemas, skipped and exception
report persistence, cache and failed-step diagnostics, and prefill, text,
runtime-ownership, token-budget, and full-depth acceptance semantics.

No generic diagnostics framework was introduced. The former private
`resolve_decode_depths` helper was deleted; the neutral `resolve_depths` is its
only replacement. Cross-stage imports changed from one
`generate_depth_sweep -> decode_depth_sweep` edge to zero edges in either
direction. Both stages now depend directly on the neutral support module.

## Stable Contracts

Normalized before/after reports match exactly for both public stages. This
comparison covers top-level keys, status-specific record keys, acceptance
keys, ordered acceptance check names, command, status, pass state, resolved
depths, and full-depth coverage.

The decode contract retains default depths `1,2,4,full`, trace validation,
direct and isolated execution, profile observability, reference and trace
status, and the existing `require_full_depth` default. The generate contract
retains default depths `1,2,4,8,16,full`, prompt/tokenizer and prefill inputs,
runtime ownership, detailed failure diagnostics, and per-depth report files.

For both stages, the first real failure or `no_device` result prevents later
execution while retaining later records with reason
`blocked by an earlier depth failure`. Generate still writes each skipped
depth's report file. Isolation still records `enabled`, `returncode`,
`command`, `stdout`, and `stderr`; decode still targets
`diagnose --stage decode-step-profile`, and generate still targets `generate`
with equivalent argument order and values.

The ordered 16-stage diagnose surface, product CLI surface, runtime/compiler/
codegen and autotune import boundaries, Phase 10.1 provenance behavior,
candidate-quality gates, product validation, and canonical autotune contracts
all pass their explicit regression selections.

## P150A Validation

Fresh hardware diagnostics used the pinned product TT-Metal commit
`61e690c25202111b52cbc1fbc9148b6524070c6f` with a 95,000,000 KB virtual-memory
limit.

The decode sweep ran depths `1,2,4,32` with trace enabled and two trace
iterations. All 4 depths passed, all references passed, all traces reported
`captured_and_executed`, full depth was covered, all isolated subprocesses
returned zero, and `failed_depths` was empty.

The generate sweep ran depths `1,2,4,32`, batch 32, prefill 256, cache 1024,
and two generated tokens per user. All 4 depths passed prefill, generated the
planned 32-user token budget, retained persistent runtime ownership, covered
full depth, returned zero from every isolated subprocess, and had no failed
depths.

Independent depth-1 decode and generate hardware isolation sanities also
passed with return code zero and their child reports present. These are
diagnostic regressions, not formal throughput measurements.

## Identity Gates

All six frozen Phase 10.1 official comparison/provenance files are
byte-identical; their combined identity remains
`c529fbd1c9be9a927a86985a3a79595977d440eba237c85341300651f926f87b`.
Runtime/compiler/codegen remains
`4d0c0e332fe33fea70fdfb0f0ff17394eb15ad3b7b888acacfe39ad3709806c9`,
and the protected runtime/compiler/codegen/autotune/correctness identity
remains
`8eaab7d50687572cfe0c77eaccb3aad374c263474b45010be76c1a1af18f0696`.

A fresh frozen-config bundle was built and dry-run validated. `model.py`,
`config.json`, `semantic_graph.json`, `execution_plan.json`,
`weights_manifest.json`, `run_decode.py`, and `README.md` all byte-match Phase
10.1. Generated Python compiles and `run_decode.py` remains 140 LOC. Best
config, precision, execution, template, program, memory, and SDPA identities
are unchanged. No autotune campaign ran and no winner was promoted.

Because these product identities are unchanged, Phase 10.1 formal evidence is
inherited: 35.8723317007 tokens/s/user, p50 28.0515274952 ms, CV 0.0211981%,
Buddy top1/top5 0.910/0.982, and minimum greedy agreement 0.964.

## Regression

The final unchanged implementation passed:

```text
python -m compileall -q models/llama_ttnn_direct
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests -q
  281 passed, 712 subtests passed
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
  157 passed, 46 subtests passed
focused Phase 11 contract selection
  261 passed, 724 subtests passed
git diff --check
  pass
```

## LOC Accounting

Canonical Phase 11 before LOC was 80,008. Final canonical LOC, including this
required report, is 79,625, for a total Phase 11 net of -383 LOC.

| Owner | Before | After | Net |
| --- | ---: | ---: | ---: |
| `decode_depth_sweep.py` | 721 | 343 | -378 |
| `generate_depth_sweep.py` | 800 | 354 | -446 |
| `depth_sweep_support.py` | 0 | 202 | +202 |
| **Combined source** | **1,521** | **899** | **-622** |
| `test_depth_sweeps.py` | 276 | 369 | +93 |

Relative to the Phase 1 baseline of 119,177 LOC, cumulative removal is
39,552 LOC (33.1876%). The tree remains 43,872 LOC above the 35,753 target.

## Evidence

Machine-readable before and after evidence, exact contracts, ownership and
identity reports, fresh bundle logs, software regression, and P150A reports
are under:

```text
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase11_before/
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase11_after/
```
