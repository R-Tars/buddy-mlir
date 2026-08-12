# TTNN Direct Phase 10.1 Performance Provenance Fix

```text
PHASE_10_1_STATUS=ACCEPTED
PHASE_11_ALLOWED=false
```

Phase 10.1 repairs the provenance boundary around official TT-Metal
comparisons. It does not change Buddy product math, runtime behavior,
compiler/codegen output, precision, the frozen winner, or autotune ownership.
The work stops here; Phase 11 has not started.

## Scope

The implementation is limited to version-aware comparison metadata and its
tests:

- `diagnostics/benchmark_parity.py` records the local runtime identity,
  force-argmax ancestry, comparison scope, and fail-closed comparability.
- `diagnostics/official_support.py` defines the three official reference
  scopes and their explicit comparability booleans.
- The diagnostics tests cover same-runtime, pinned-release, current-main,
  SHA-mismatch, and force-argmax provenance cases.

The production change is `+137/-4` lines and the new tests add `96` lines,
within the Phase 10.1 limits of `250` and `300` lines. The unrelated dirty
`llvm` submodule was left untouched and is not part of this change.

## TT-Metal Lineage

Phase 8 formal performance and the Phase 10 short parity harness both used:

```text
SHA:     61e690c25202111b52cbc1fbc9148b6524070c6f
describe: v0.70.0-dev20260505-15-g61e690c2520
```

Therefore there was no Phase 8 to Phase 10 TT-Metal version drift. Neither
checkout contains the single-chip P150 force-argmax fix:

```text
0a0510e7fc197f599c9d5974a0f1f6776378db03
```

The ancestry checks returned exit status `1` for both historical checkouts.
The Phase 10 short official greedy result is consequently classified as a
same-runtime-commit result on the legacy top-k sampling path, not as current
upstream performance. This explains the observed `21.8446 t/s/user` slow
class without claiming that the comparison harness or Buddy product regressed.

Historical measurement-time fields that were not captured, including checkout
status, shared-library hashes, some environment variables, and firmware/KMD
runtime values, remain explicitly `unknown` in
`phase10_1/toolchain_provenance_before.json`. The binary hashes and hardware
tool versions recorded there as current observations are not retroactively
claimed as historical measurement identities.

## Reference Scopes

Phase 10.1 keeps the following references separate:

| Reference | Identity | Result | Comparable use |
| --- | --- | ---: | --- |
| Same-runtime local official | `61e690c2520...` | `21.8435 t/s/user` median | Apples-to-apples with Buddy's pinned runtime; legacy sampling path |
| Pinned historical release | `v0.64.0-dev20251030`, `b76035f...` | `33.1 t/s/user` published reference | Cross-version context only |
| Current-main upstream official | `30023337d7fa...` | `33.22 t/s/user` average | Current upstream reference; not same-version comparable |

The same-runtime official measurement used three repetitions and had values
`21.8434566833`, `21.8667131990`, and `21.8356487491`. Its force-argmax fix
was absent, and `temperature=0` therefore selected the expected legacy top-k
path.

The current-main measurement was taken from a clean, timestamped checkout
snapshot. It contains the force-argmax fix, uses one P150 device, and its log
confirmed `force_argmax=True`. It measured `33.22 t/s/user`, `53.32 ms` TTFT,
`33.99 t/s/user` for the first token, and `33.05 t/s/user` at token 128.
The official resolver was run against `models/model_targets.yaml` rather than
the deprecated `PERF.md`. For batch 32 and sequence length 1024 it returned
`TARGET_NOT_DEFINED`; the registered sequence-length-712 value of `33.0`
with 15% tolerance was retained only as a neighboring sanity reference and
was not used as the exact acceptance target.

The pinned local official SHA and current-main SHA differ. Accordingly:

```text
same_runtime_commit_comparable = true
pinned_release_comparable       = false
current_main_comparable         = false
```

`same_runtime_commit=true` means only that Buddy and the local official
checkout used the same TT-Metal commit. It does not mean current-main or
latest-upstream equivalence.

## Buddy Formal Performance

Buddy was freshly measured with the established formal contract:

```text
batch size:       32
prefill length:   256
cache length:     1024
execution:        trace
inputs:           persistent
warmup:           5
measured:         100
repetitions:      3
force argmax:     true
```

The three successful throughputs were:

```text
35.8723317007 t/s/user
35.8873322381 t/s/user
35.8702657976 t/s/user
```

The median is `35.8723317007 t/s/user`, median p50 latency is
`28.0515274952 ms`, and CV is `0.0211981%`. The acceptance floor is
`35.227787 t/s/user`; the formal result passes both the floor and the
`CV <= 1.5%` gate. Each run recorded four captures, 105 executes, seven
persistent inputs, and zero post-capture compilation. One temporarily busy
device attempt was retried and is retained as a failed-attempt record, not
silently discarded.

This is a `0.8114%` increase over the retained Phase 8 median of
`35.5836228411 t/s/user`, with the frozen product identity unchanged. It is
not an optimization claim from Phase 10.1.

The earlier Phase 10 value of `34.0799 t/s/user` came from a
`1 repetition / 1 warmup / 3 measured` harness smoke run. It has a different
sample count and purpose, so no direct regression percentage is computed
between it and the formal `3 / 5 / 100` baseline.

## Correctness and Identity

The fresh 500-token performance-correctness run passed:

```text
official top-1/top-5:       0.910 / 0.980
Buddy aggregate top-1/top-5: 0.910 / 0.982
Buddy minimum-user top-1:    0.910
minimum greedy agreement:   0.964
```

The corpus, prompt IDs, teacher-forcing targets, top-5 IDs, replay IDs, and
official prediction identity match the retained reference. Phase 10's lower
`0.906` minimum-user top-1 prediction hash differs from Phase 8 and Phase
10.1's `0.910` result, while the product source and TT-Metal SHA remain
identical. The retained Phase 10 evidence does not capture enough historical
runtime/build/process state to identify a narrower low-level cause; it is
therefore conservatively classified as historical runtime/build/process-state
variation rather than attributed to a source change.

The 50-file runtime/compiler/codegen identity is unchanged:

```text
4d0c0e332fe33fea70fdfb0f0ff17394eb15ad3b7b888acacfe39ad3709806c9
```

All seven generated program artifacts byte-match Phase 9. The winner,
precision, execution, template, program, memory, and SDPA identities are
unchanged. No autotune campaign ran and no winner was promoted.

## Software Validation

The complete results are frozen in the ignored machine evidence file
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase10_1/software_regression.json`:

```text
compileall:    passed
product:       281 passed / 712 subtests
diagnostics:   151 passed / 46 subtests
focused:       74 passed
diff --check:  passed
```

The evidence directory also contains the complete provenance, performance,
correctness, identity, target-resolution, and official-run artifacts. Build
evidence remains under the ignored `build-tenstorrent` tree; only this report
and the four scoped source/test files are intended for the Git commit.

## Final Stop

Phase 10.1 is accepted because the historical lineage is recovered, the old
official slow result is explained by the absent force-argmax fix, current-main
official provenance is independently frozen, Buddy's formal performance
passes, correctness passes, and product identity is unchanged. Phase 11 is
explicitly not started.
