# TTNN Direct Phase 12 Autotune Profiler Contract Convergence Report

## Status

`PHASE_12_STATUS=PARTIAL_SAFE_BOUNDARY`

Phase 12 converged the duplicated decode measurement contract, hashing, and
profiler JSON persistence without changing the unique profiler algorithms.
It does not satisfy the requested LOC deletion gate, and the frozen profiler
region assignment still does not support the later fused QK-RoPE/KV-update
winner trace. The document's stop conditions prohibit changing that assignment
in this phase, so semantic safety takes precedence over artificial deletion.
Phase 13 has not started.

## Canonical Ownership

`autotune/measurement.py` now owns
`validate_decode_profile_contract(...)` and `force_argmax_enabled(...)`. The
same validator is actively called by both:

- `autotune/baseline.py` for formal 3x5x100 evidence;
- `diagnostics/autotune_profiler_audit.py` for one-trace profiler capture.

The common validator fails closed on profiled/pass state, full depth, trace and
persistent execution, after-prefill mode, capture/replay counts, positive
persistent inputs, zero per-step device allocation and host transfer, page-table
reuse, device token handoff, zero post-capture compilation, and force argmax.
It uses the existing `MeasurementContract`; no second contract type or generic
diagnostics framework was introduced.

The dependency direction is only `diagnostics -> autotune.measurement ->
autotune.schema`. Autotune has no reverse diagnostics dependency. The import
boundary test allows only this explicit stage-to-owner edge.

## Baseline Contract

Baseline-only semantics remain in `autotune/baseline.py`:

- exactly three distinct reports with warmup 5 and 100 measured iterations;
- exactly 100 positive raw samples and positive throughput per report;
- identical trace keys, program directory, and TT-Metal commit;
- acceptance evidence, median throughput, CV, and artifact manifests.

The baseline file changed from 777 to 754 LOC, net -23. Its full diff is
27 additions and 50 deletions: the shared execution checks and local file hash
were removed while baseline-only checks remained. Rebuilding the historical
Phase 0 baseline produced six byte-identical JSON artifacts, including the
decode report, config, op graph, runtime/source identities, and manifest.
The normalized public baseline contract is exactly unchanged.

## Profiler Contract

Profiler capture remains full-depth Llama 3.1 8B at batch 32, prefill 256,
cache 1024, eager prefill, trace execution, persistent inputs, and
after-prefill measurement. Its canonical contract remains:

```text
kind=profiler_audit_capture
warmup=0
iterations=1
repetitions=1
```

The Tracy command, environment contract, top-level report keys, measurement
fields, region fields, five budget keys, precision summary, and CSV schema are
unchanged. The program-derived precision summary was retained; no incomplete
`PrecisionContract` fields were fabricated.

Profiler-local `_sha256_json`, `_file_sha256`, `_force_argmax`,
`_validate_measurement_contract`, and `_atomic_write_json` were removed.
Canonical owners are `autotune.schema.sha256_json`,
`autotune.measurement.file_sha256`, the shared validator, and
`runtime.reports.write_report`. The local object reader remains input-specific,
and the minimal atomic text writer remains necessary for CSV output. Baseline's
multi-artifact JSON writer remains local to preserve its byte contract.

## Unique Analysis Freeze

The following unique analysis remains in place and was not rewritten or moved:

- trace replay selection;
- clock-offset normalization and latency reconciliation;
- decode op-to-region assignment;
- hardware metric aggregation;
- bottleneck classification;
- five budget answers and metric availability;
- region CSV rendering.

The same persisted CSV/profile/config was evaluated before and after. Trace
selection, clock normalization, all 676 semantic assignments, 21 regions,
budget answers, bottleneck ranking, metric availability, and latency are exact.
The complete parsed reports are equal and both are 86,275 bytes; JSON key order
differs because persistence now uses the canonical runtime writer.

## P150A Validation

Fresh hardware capture used pinned TT-Metal
`61e690c25202111b52cbc1fbc9148b6524070c6f` and the profiler's existing
separate-QK-RoPE/separate-paged-update contract program. It passed with:

```text
status/pass:          pass / true
trace replay ops:     676
nonempty regions:     21
device/e2e ratio:     0.9929633573
budget answers:       5
capture/replay:       1 / 1
persistent inputs:    7
post-capture compile: 0
```

All observed bottleneck classes are valid, and both JSON and CSV outputs exist.
This is a one-iteration profiler integration check, not a formal throughput
measurement.

A second fresh capture used the current frozen fused winner. The device profile
itself passed the shared measurement contract, but the audit failed closed at
trace op 14: it expected `RotaryEmbeddingLlamaDeviceOperation` and observed
`ReshardDeviceOperation` before the fused QK-RoPE operation. Supporting that
trace requires changing the unique region assignment, which Phase 12 explicitly
forbids. Both the passing supported-contract evidence and this limitation are
retained in the Phase 12 machine evidence.

## Identity And Performance

Phase 10.1 provenance files remain byte-identical with combined SHA-256
`c529fbd1c9be9a927a86985a3a79595977d440eba237c85341300651f926f87b`.
Phase 11 depth files remain byte-identical with combined SHA-256
`6dd526b9c64f7bb192c1f334203478534d5e77a06172572c6986c174f910d0b2`.
Runtime/compiler/codegen remains
`4d0c0e332fe33fea70fdfb0f0ff17394eb15ad3b7b888acacfe39ad3709806c9`.

All seven freshly generated artifacts match Phase 11 byte-for-byte;
`run_decode.py` remains 140 LOC and compiles. Best config, precision, execution,
template, program, memory, and SDPA identities are unchanged. No search or
autotune campaign ran, and no winner was promoted.

The unchanged product identity therefore inherits the Phase 10.1 formal result:
35.8723317007 tokens/s/user, p50 28.0515274952 ms, and CV 0.0211981%.
Correctness evidence remains Buddy top1/top5 0.910/0.982 with minimum greedy
agreement 0.964.

## Regression

```text
compileall:    pass
product:       282 passed / 712 subtests
diagnostics:   168 passed / 46 subtests
focused:       278 passed / 724 subtests
diff --check:  pass
```

Fail-closed tests cover wrong execution/runtime mode, missing after-prefill,
per-step allocation/H2D, page-table reuse, token handoff, post-capture compile,
missing trace replay, and force-argmax. Existing trace selection, normalization,
region assignment, truncated trace, budget, CSV, and segmented-prefill tests
remain.

## LOC Accounting

Canonical Phase 12 before LOC was 79,625. Final canonical LOC is 79,988, for a
total Phase 12 net of +363 LOC.

| Owner | Before | After | Net |
| --- | ---: | ---: | ---: |
| `autotune_profiler_audit.py` | 1,256 | 1,204 | -52 |
| `measurement.py` | 241 | 382 | +141 |
| `baseline.py` | 777 | 754 | -23 |
| **Combined implementation** | **2,274** | **2,340** | **+66** |
| Three scoped test owners | 703 | 812 | +109 |
| This report | 0 | 188 | +188 |

The requested source -300 and total -200 gates are not met. Reaching them in
the allowed scope would require deleting validation/tests or rewriting the
protected unique profiler analysis, so the safe-boundary status is mandatory.
Relative to the Phase 1 baseline of 119,177 LOC, cumulative removal is 39,189
LOC (32.8830%). The tree remains 44,235 LOC above the 35,753 target.

## Evidence

Machine-readable inventories, contracts, before/after analysis, baseline
rebuild, frozen identities, software regression, and both P150A captures are
under:

```text
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase12_before/
build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase12_after/
```
