# TTNN Direct Phase 1 Refactor Report

## Scope

Phase 1 froze the product contract, removed checked-in historical evidence and
duplicate status text, and deleted only a Python file proven unreachable. It
did not change generated model logic, prefill, decode, tracing, persistent
inputs, the production config, or autotune behavior.

The behavior baseline is Buddy commit
`dbf5330c296425e8c596ab960b692056db268715`, tt-metal commit
`61e690c25202111b52cbc1fbc9148b6524070c6f`, and best-config SHA-256
`1ed82e5b0f2f2ce8444fa881ce725f62a37d6f556fa2cee493cda6dc7e5fde5d`.

## Tracked Size

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Tracked files | 253 | 237 | -16 |
| Tracked lines | 119177 | 116757 | -2420 |
| Relative line reduction | - | - | 2.0306% |

New Python lines are `307`, below the Phase 1 limit of `500`: the 194-line
inventory tool, two regression tests totaling 87 lines, and 26 replacement
test lines. The replacement test also removes 146 historical-evidence lines.

## Removed Files

All line counts are from the frozen `phase1_before/tracked_files.json`.

| File | Lines | Reason |
| --- | ---: | --- |
| `REFACTOR_BASELINE.md` | 61 | Stale duplicate status merged into README and latest summary |
| `buddy_ttnn_direct/templates/mlp_prefill.py` | 13 | All dependency-report reachability guards are false |
| `docs/evidence/hf_correctness_reference_manifest_20260710.json` | 64 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_decode_steady_evidence_20260711.json` | 85 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_generate_depth_evidence_20260709.json` | 236 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_generate_profile_evidence_20260709.json` | 68 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal1_dependency_inversion_evidence_20260714.json` | 70 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal2_persistent_inputs_evidence_20260714.json` | 71 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal3_full_trace_evidence_20260714.json` | 88 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal4_execution_graph_evidence_20260714.json` | 113 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal5_prefill_evidence_20260714.json` | 145 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal6_prefetcher_decision_20260714.json` | 97 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal7_correctness_performance_evidence_20260714.json` | 140 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_goal8_custom_op_decision_20260714.json` | 94 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_layered_autotune_evidence_20260711.json` | 155 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_lm_head_argmax_evidence_20260711.json` | 116 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_matched_parity_evidence_20260714.json` | 119 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_numerical_correctness_evidence_20260711.json` | 123 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_official_config_parity_evidence_20260711.json` | 117 | Historical run evidence; no runtime/build dependency |
| `docs/evidence/p150a_semantic_autotune_paper_evidence_20260715.json` | 430 | Historical run evidence; no runtime/build dependency |

The 18 evidence JSON files account for 2,331 deleted lines. Their current
result is represented by the 51-line `docs/evidence/latest_summary.json`;
raw reports remain under the ignored Buddy build tree.

## Dependency Evidence

The before report records imports, reverse imports, CLI/CMake/package exports,
tests, documentation commands, and generated-source references for every
Python module. After standalone `__main__` entrypoints were recognized,
`templates/mlp_prefill.py` was the only non-test candidate with every guard
false. The after unreferenced-file report is empty.

`docs/future/custom_ops.md` remains because diagnostics tests reference it.
The following Phase 1 protected areas also remain because they are reachable
or explicitly deferred: hidden CLI handlers, generated-runner legacy modes,
smoke modules, diagnostics tests, autotune modules, and runtime compatibility
facades.

## Correctness

| Gate | Result | Evidence |
| --- | --- | --- |
| Full-depth functional generation | Pass | Fresh P150A validation after refactor |
| All-BF16 full-depth PCC | Pass | 99 comparisons, minimum PCC 0.992598 at threshold 0.99; source report hash revalidated |
| Performance-recipe quality | Pass | Buddy top-1/top-5 0.91/0.982 and official-Buddy greedy agreement 0.964; source report hash revalidated |

The fresh functional result generated non-empty text. Hash reuse for the two
numeric gates is deliberate: no execution, precision, config, or generated
model file changed in Phase 1.

## Performance

The before and after runs use the same final generated program, model, prompt
corpus, TTNN binary, and measurement contract: batch 32, prefill 256, cache
1024, force argmax, 5 warmups, 100 measured iterations, 3 repetitions, full
trace, persistent inputs, and post-prefill timing.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Median tokens/s/user | 35.579552 | 35.587352 | +0.0219% |
| Median p50 decode latency | 28.062528 ms | 28.062326 ms | -0.000202 ms |
| Throughput CV | 0.0690% | 0.0213% | Pass |

After raw throughput is `35.577165`, `35.587352`, and `35.595740`
tokens/s/user. Regression is `-0.0219%`, so the no-more-than-1% gate passes.
All runs retain seven persistent inputs and zero program compilation after
trace capture.

## Validation

- `python -m compileall -q models/llama_ttnn_direct`: passed.
- Product tests: 266 passed, 683 subtests passed.
- Diagnostics tests: 245 passed, 17 subtests passed.
- Six product-command help parses: passed.
- Build, generate, profile, and validate dry-runs: passed.
- Matched three-repetition P150A steady decode: passed.
- Fresh full-depth P150A functional validation: passed.

## Phase 1 Limit

The 5,000-line target cannot be reached safely in this phase. Historical JSON
provides only 2,331 lines, duplicate documentation provides 61 lines, and the
dependency report proves only one additional 13-line Python file completely
unreachable. Deleting more would require crossing into reachable or expressly
protected Phase 2-5 work. This report therefore records the actual safe
reduction and stops without performing CLI/runtime/autotune restructuring.

## Artifacts

Machine-readable before and after LOC, dependency, public-surface,
removed-file, correctness, and performance reports are under:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase1_before/
${BUDDY_BUILD}/models/llama31_ttnn_direct/ttnn_direct_refactor/phase1_after/
```

These paths are covered by the repository's `/build*` ignore rule.

## Next Phase

Phase 2 should isolate the six product CLI commands from hidden handlers and
then remove generated-runner legacy modes in dependency order. Later phases
can move the 23,394-line autotune core, 12,071-line diagnostics package, and
22,136-line diagnostics test suite out of the product package before deciding
what can be deleted. No Phase 2 work is included here.
