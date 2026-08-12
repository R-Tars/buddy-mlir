# TTNN Direct Phase 10 Refactor Report
`PHASE_10_STATUS=ACCEPTED`
Phase 10 consolidates the official TTNN comparison harness only. Product math, runtime,
compiler, codegen, autotune, correctness, CLI, and Phase 11 are unchanged.

## Ownership
Canonical LOC started at 80,500. The three stage owners changed as follows:
| Owner | Before | After | Net |
| --- | ---: | ---: | ---: |
| `benchmark_parity.py` | 1,869 | 893 | -976 |
| `performance_correctness.py` | 567 | 341 | -226 |
| `execution_graph_diff.py` | 818 | 498 | -320 |
| shared support | 0 | 360 | +360 |
`official_support.py` (291 LOC) owns prompts, pytest environment/hooks, latency and
accuracy parsers, page parameters, official source setup, cache observation, and graph
capture. `process_support.py` (69 LOC) owns path, git/hash, and logged subprocess helpers;
the runner preserves combined output, `shell=False`, timeout, optional `RLIMIT_AS`,
integer return code, and no retry.
Parity planning/resume/statistics, correctness corpus/teacher forcing/minimum-user gates,
and graph normalization/selection/focus diff remain stage-local domain semantics.
Cross-stage imports fell from 2 edges / 12 symbols / 3 private symbols to 0 / 0 / 0.

## Frozen Contracts
The public pytest plugin path remains
`models.llama_ttnn_direct.buddy_ttnn_direct.diagnostics.benchmark_parity`; hooks are re-exported from the stage. Public signatures,
three report schemas/check names, root CLI (6), validation suites (5), and diagnose
stages (16) are unchanged. AST tests enforce zero comparison-stage cross imports.
`candidate_quality_gate.py` remains byte-identical at SHA-256:
`9aa59e0fcd493f88c62cab2d34890b50cb777bbc3fe600375d904acb57efe029`; focused regression and the 500-token
minimum-user fail-closed semantics remain.

## Fresh P150A Integration

Final-source correctness passed: 500 official samples, official top1/top5 0.910/0.980,
Buddy aggregate/minimum-user 0.906/0.982, and minimum-user agreement 0.972. All nine
checks passed.

Fresh 1x `(warmup=1, iterations=3)` parity ran without resume: official-demo 20.7680,
official-greedy 21.8446, and Buddy trace+persistent 34.0799 tokens/s/user. Each supplied
three exact samples; same-commit was true and official-greedy remained primary. This
short run validates the harness, not formal performance.

Execution-graph-diff passed at batch 32 / prefill 256 / cache 1024 with full decode
captures (official 1,442 ops, Buddy 708), focus diffs, official cache 141 -> 141 with zero
misses, and Buddy cache 96 -> 128 -> 128 with zero post-capture compilation.

## Identity, Regression, and LOC

The 50-file runtime/compiler/codegen identity remains:
`4d0c0e332fe33fea70fdfb0f0ff17394eb15ad3b7b888acacfe39ad3709806c9`; seven artifacts and all winner identities are
byte-identical. No tuning/promotion ran, so Phase 8 performance (35.583623 tokens/s/user,
p50 28.054850 ms, CV 0.021412%) and All-BF16/full-depth evidence remain reusable under
`product_runtime_not_modified_and_bundle_byte_identical`.

Regression passed: compileall; product 281/712; diagnostics 143/46; focused comparison,
candidate gate, autotune integration, validation, import boundaries; `git diff --check`.

Target source LOC is 3,254 -> 2,092 (-1,162); tests are 464 -> 646 (+182). Final
canonical LOC is 79,589: Phase 10 net -911. Cumulative removal is 39,588 (33.22%),
leaving 43,836 LOC to the 35,753 target.

Machine evidence is under `build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase10_{before,after}/`.
