# TTNN Direct Phase 3.2 Refactor Fix Report

Status: trace capture/replay cleanup, safe fallback, and all regression gates passed.

## Failure Behavior
Capture-body failures now best-effort end capture and always release an allocated trace; eager fallback runs only after both operations succeed. End/cleanup failures block fallback because capture state is uncertain, replay failures fall back only after successful release, PCC exceptions report `runtime_error`, and release failures report `trace_release_failed` without fallback.

## Validation
Mock tests cover begin, capture body, capture cleanup, direct end, warmup replay, PCC, and release failures with lifecycle/count/order assertions. Targeted tests passed 7/10 subtests; product 274/712 and diagnostics 243/27 passed, plus compileall, CMake, and the 140-line runner checks. P150A eager/trace passed at PCC 0.999863; trace retained 1 capture, 5+20 replays, and 1 release. Full-model 5x100x3 median was 35.538711 t/s/u, p50 28.087311 ms, CV 0.08688%, and 0.21648% below Phase 3.1. Fresh full-depth, 99-check all-BF16 (minimum PCC 0.992598), and 500-token quality gates passed.

Scope: +105/-16 implementation and tests plus this 11-line report is net +100; runtime math, precision, configs, generated model, and Phase 4 are unchanged. Raw evidence is under `build-tenstorrent/models/llama31_ttnn_direct/ttnn_direct_refactor/phase3_2/`.
