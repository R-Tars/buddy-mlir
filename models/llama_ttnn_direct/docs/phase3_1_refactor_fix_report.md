# TTNN Direct Phase 3.1 Refactor Fix Report

Status: eager/trace template-profile semantics and all regression gates passed.

## Fix

Phase 3 was not equivalent because every warmup/measured sample called
`run_smoke_mlp`, reopening the device and recreating tensors, while `--trace`
reported an eager worker without calling the TTNN trace APIs.

Eager profiling now opens one managed device, creates the MLP inputs and weights
once, and reuses them and the program cache for all warmup and measured runs.
The report records one open/close and the persistent tensor lifecycle.

Trace profiling performs one compile run, one capture, the requested warmup and
measured `execute_trace` replays, PCC before release, and one release. Missing
APIs and capture failures explicitly report `trace_api_unavailable_fell_back_to_eager`
and `trace_failed_fell_back_to_eager`; neither fallback is labeled captured.

## Validation

Mocked lifecycle tests passed: one device enter/exit, one tensor preparation,
correct eager execution counts, and trace counts of 1 capture, 3 replays, and 1
release for the 1-warmup/2-iteration test. Existing dry-run, no-device, and CLI
coverage passed. Product tests passed 274 tests/712 subtests; diagnostics passed
242 tests/21 subtests; compileall, 140-line runner compatibility, and CMake passed.

P150A eager passed at PCC 0.999863 and p50 3.291806 ms. Trace was captured once,
replayed 5 warmups plus 20 measurements, released once, and reached p50 3.288330 ms.
Full-model 5x100x3 throughput was 35.615811 t/s/u median, p50 28.035265 ms,
CV 0.0402%, and +0.1066% versus Phase 3. Full-depth generation, 99-check all-BF16
(minimum PCC 0.992598), and 500-token quality gates passed.
## Size and Scope

Tracked implementation/tests/report delta is +340/-220 lines, net +120; the
profiler is 180 lines. Product runtime, model math, configs, precision, generated
model, and Phase 4 areas are unchanged. Raw evidence and source identity are under
`build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase3_1/`.
