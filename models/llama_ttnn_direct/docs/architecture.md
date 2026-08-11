# TTNN Direct Architecture

## Product Pipeline

TTNN Direct keeps one user-facing execution path:

```text
build
  HF Llama metadata
    -> semantic graph
    -> official-like template plan
    -> generated model.py and program bundle

generate
  program bundle + HF weights + prompt
    -> materialize and tensorize parameters once
    -> create TTNNDirectRuntimeContext
    -> tokenize and prefill prompt
    -> populate paged KV cache
    -> run generated decode steps
    -> detokenize and write report

profile
  generate
    -> section and layer timing
    -> throughput summary
    -> profile report

validate
  build artifacts + generate/profile reports
    -> compact suite checks
    -> validation report
```

The existing `models/llama31_tt` path remains independent and continues to be
the TTIR baseline.

## Build-Tree Ownership

All reproducible TTNN Direct artifacts live beneath the active Buddy binary
directory:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/
  program/
  correctness_program/
  llama31_ttnn_direct_package/
  reports/
  autotune/
  references/
  runtime_artifacts/generated/{inspector,watcher}/
  evidence_archive/
```

The source tree owns generators and checked-in compact evidence only. It does
not own generated model programs or TT-Metal logs. Device commands execute
from `runtime_artifacts/` and set `TT_METAL_LOGS_PATH` there so generic
TT-Metal outputs cannot create a top-level `generated/` directory. System
temporary storage remains reserved for disposable process-level scratch.

## Package Boundaries

### Semantic model

`buddy_ttnn_direct/semantic/` owns the Llama graph dataclasses, Hugging Face
importer, graph validation, and JSON I/O. It does not import TTNN or open a
device.

### Planning and code generation

`buddy_ttnn_direct/templates/` maps semantic operations to official-like TTNN
templates. `buddy_ttnn_direct/compiler/` validates plans, builds generated
config, renders generated Python source, and writes compiler artifacts.
`compiler/official_config.py` imports the extracted hardware profile and keeps
its provenance attached to generated config. `runtime/config_runtime.py`
materializes JSON descriptors into TTNN dtype, memory, program, grid, and
compute-kernel objects without making compiler modules import TTNN.
`buddy_ttnn_direct/codegen/` retains artifact, parameter, tensorization, and
package helpers. Compiler entry points are imported directly from
`buddy_ttnn_direct/compiler/`.

The generated bundle contains at least:

```text
README.md
config.json
execution_plan.json
model.py
run_decode.py
semantic_graph.json
weights_manifest.json
```

### Runtime

`buddy_ttnn_direct/runtime/` owns the product generate path:

- `context.py`: `TTNNDirectRuntimeContext` and persistent runtime ownership.
- `plans.py`: canonical prefill/decode execution plans shared with diagnostics.
- `model_loader.py`: generated Python model loading and config namespaces.
- `tensor_meta.py`: shape, dtype, and integer host-tensor helpers.
- `tokenizer.py`: prompt tokenization and generated text decoding.
- `inputs.py`: token, page-table, cache-position, and rotary inputs.
- `decode_inputs.py`: A/B-selectable persistent decode input ownership. The
  persistent path allocates page table, current position, rotary index, full
  cos/sin caches, and rotary transformation once per runtime session.
- `trace.py`: full decode trace key/session ownership, compile run, capture,
  nonblocking replay, device token feedback, and program-cache accounting.
- `prefill_trace.py`: shape-specific prefill trace keys and experimental
  capture/replay ownership.
- `rotary.py`: HF-compatible Llama RoPE values and TTNN tensor placement.
- `kv_cache.py`: paged KV-cache allocation and metadata.
- `prefill.py`: prompt prefill orchestration.
- `prefill_profile.py`: repeated batched-prefill timing and TTFT/reference
  comparison reports.
- `decode.py`: decode-step and token handoff orchestration.
- `generate.py`: high-level generate entry point.
- `profile.py`: generate section profiling and profile report assembly.
- `reports.py`: optional compact/full generate schemas, atomic JSON writing,
  and streamed JSONL diagnostics.

Repository code imports these runtime entry points directly; there is no
package-root generate compatibility facade.

### Reports and validation

`buddy_ttnn_direct/reports/` owns report-only logic. The modules do not open a
device, execute model stages, or import `buddy_ttnn_direct/diagnostics/`.

- `schema.py`: shared field, path, number, and acceptance helpers.
- `validation.py`: compact product validation suites.
- `performance.py`: baseline and milestone summaries.
- `evidence.py`: compact reproducibility/evidence manifests.
- `runtime.py`, `profiling.py`, `tensorization.py`: report contract checks.
- `attention.py`, `depth.py`: diagnostics report checks.
- `config.py`, `artifacts.py`: config coverage and artifact checks.

`buddy_ttnn_direct/correctness/` owns numerical reference artifacts and
comparison metrics. HF reference capture runs on CPU, supports a truncated
layer count, and records deterministic samples for layer hidden states,
final hidden state, logits, and prefill KV cache.

`correctness/performance_recipe.py` owns the lower-precision token contract.
The corresponding diagnostic loads TT-Transformers' fixed `.refpt` corpus,
measures top-1/top-5 teacher-forced accuracy, and compares multi-token greedy
predictions. When the 512-token reference prompt exceeds Buddy's static
prefill bucket, the first 256 tokens run through prefill and the remaining 256
are replayed exactly through eager decode before target-token observations are
collected. This does not alter normal generation.

The product CLI uses five compact suites:

- `dryrun`: required bundle artifacts plus generate/profile dry-runs.
- `functional`: successful prefill/decode, prefilled KV cache, and generated
  text.
- `device`: functional checks plus TTNN identity and optional full depth.
- `performance`: positive throughput and complete profile sections, with
  optional full depth.
- `correctness`: HF top-token, logits, hidden-state, and sampled prefill
  KV-cache comparisons against observations captured from P150A.

Phase-era validation orchestration was retired after
`build`/`generate`/`profile`/`validate`/`inspect`/`diagnose` became the
supported surfaces.

### Diagnostics

Smoke, sweep, and legacy decode workflows are development tools. The visible
CLI exposes them only through `diagnose --stage ...`; their tests live under
`tests_diagnostics/` and are excluded from the default product suite.
`diagnostics/cli.py` is loaded only when `diagnose` executes and exposes the
autotune diagnostic entrypoint without owning tuning semantics. All tuning
semantics belong to the canonical `buddy_ttnn_direct/autotune/` package.

### Semantic autotune contracts

`buddy_ttnn_direct/autotune/` owns the schema-v2 candidate identity. Its frozen
precision contract is derived from the checked-in official performance profile
and includes weight/activation/KV dtypes, per-layer overrides, math fidelity,
FP32 accumulation, approximation, and packer settings. Execution is fixed to
post-prefill full decode trace with persistent inputs, fixed page tables, and
force-argmax sampling. Candidate fingerprints include these contracts plus the
semantic graph, model config, weights/recipe identity, runtime commit, device,
workload, tunable state, and measurement counts. The legacy layered runner is
gone; the canonical campaign composes schema-v2 enumeration, active
measurement, hierarchical search, and matched A/B confirmation directly.

`autotune/space.py` defines the lossless schema-v2 execution space: template
choices, typed matmul and SDPA descriptors, memory placement and sharding,
core grids, and producer-consumer edges. Runtime descriptor extensions are
preserved during serialization. Schema-v1 preset names are accepted only by a
compatibility adapter; generated programs and candidate fingerprints contain
the resulting structured fields.

`autotune/legality.py` validates schema-v2 candidates before measurement. Its
static pass checks shape and tile divisibility, program-family constraints,
physical grid and worker limits, shard coverage, dtype/layout compatibility,
template-specific inputs, CB page count, and conservative L1 usage. The P150A
descriptor models an `11x10` logical worker grid, `110` workers, and
`1,572,864` bytes of worker L1. The default usable limit is `0.8` of L1.

L1 estimates report input, output, double-buffered CB, intermediate, and
program-local scratch components for every supported matmul and SDPA program.
Static rejection prevents compile launch. Candidates that pass may run a
representative compile-only command in an isolated process; its output is
classified as `shape_incompatible`, `l1_overflow`, `unsupported_layout`,
`invalid_program_config`, `invalid_core_grid`, `api_unavailable`,
`compile_error`, or `runtime_error`. A legality report is complete when every
candidate has an accepted or rejected result; rejected search candidates do
not make report generation itself fail.

`autotune/templates.py` is the existing-API semantic template registry. It
defines four independent axes and eight canonical choices:

```text
attention.kv_update: separate_paged_update | fused_paged_update
attention.rope: separate_qk_rope | fused_qk_rope
mlp.activation_placement: mul_fused_silu | gate_linear_fused_silu
mlp.gate_up: separate_gate_up | packed_gate_up
```

Each definition owns its TTNN API availability groups, static predicate,
runtime-config hook, schema fields, operation sequence, cost metadata, and CPU
reference evaluator. Template application occurs after program, memory, and
grid descriptors are installed so hooks always modify the final candidate.
`gate_linear_fused_silu` writes SILU into a sharded matmul program's
`fused_activation`; it uses `ttnn.linear(..., activation="silu")` only when no
program config is present. `packed_gate_up` is incompatible with that partial
activation placement and is never a default.

Fused QK RoPE follows TT-Transformers runtime preparation: decode rotary
indices and transformation shards use twice the logical batch, while Q and K
are moved to disjoint core ranges before the fused TTNN call. Packed gate/up
weights are concatenated offline on the output-feature axis, tensorized once,
projected once, and split through the public `ttnn.split` API. No C++ or Metal
implementation is introduced by the registry.

`autotune/transfer.py` defines representative-layer ownership as a complete,
non-overlapping partition. For Llama 3.1 8B, `group_default` measures layer 0
and transfers the selected state to layers 0-30. `group_override` measures
layer 31 only because its frozen MLP precision recipe differs. Expansion emits
the representative, destination layers, and state hash for every assignment.

`autotune/microbench.py` measures an `op` or `region` target through a versioned
JSON worker protocol. Every repetition runs in a fresh subprocess, explicitly
acknowledges the trace/persistent execution contract and warmup count, and
returns exactly the configured number of synchronized samples. Reports retain
raw per-repetition samples, mean, p50, p90, population standard deviation, CV,
process logs, program-cache count, trace-capture count, and replay allocation
count.

The measurement-cache key covers the complete candidate fingerprint, target,
representative group, worker protocol, worker callable, and payload hash. The
candidate fingerprint already includes the TTNN/tt-metal runtime commit and
measurement settings, so a runtime change cannot reuse an old result. Cache
writes are atomic, and failed, timed-out, malformed, or partial measurements
are never cached.

`autotune/matmul.py` enumerates the six decode MatMul regions: QKV, O
projection, gate, up, down, and the LM-head shard vector. The official program
vector is always proposal zero. Additional DRAM-sharded programs are derived
from common K-tile divisors, padded N tiles, the device DRAM width, and bounded
worker-core targets. The family does not expose a compute-grid argument, so
the candidate records its derived worker count and the device grid rather than
inventing an unsupported runtime field.

Every proposal replaces only one schema-v2 operator and reruns the complete
Phase 2 legality engine, including shape divisibility and conservative L1/CB
estimation. Illegal proposals retain stable error evidence; legal proposals
retain only target-local L1 evidence and a hash of the materializable search
space. Microbenchmark selection ranks passed Phase 4 reports, but labels its
winner as representative-only and never grants promotion to the default.

`autotune/sdpa.py` enumerates decode SDPA as a bounded, cache-aware space. It
always inserts the exact official program and then covers logical grids,
physical sub-core placements, q/k chunks, per-head-batch core caps, and kernel
output memory. Cache length limits k chunks directly: the 128, 512, and 1024
workloads add 128, 256, and 512-sized terminal choices respectively. Candidate
identity includes the complete SDPA workload and device descriptor, so results
cannot collide across cache lengths or targets.

Sub-core rectangles use physical device coordinates. Their union must be
disjoint, lie inside the device descriptor, and contain exactly the logical
program-grid core count; an offset `8x8` placement is therefore valid on the
P150A `11x10` worker grid. Runtime realization converts those rectangles to a
real `ttnn.CoreRangeSet`. The enumerator also updates the duplicated schema-v2
grid and memory paths when materializing a candidate, preventing later config
application from silently restoring official values.

`exp_approx_mode` remains precision-frozen and is absent from the tunable
identity. Every proposal passes the shared legality/L1 engine before exposure.
The active TTNN binding materializes every retained descriptor, and the smoke
attention APIs accept an explicit candidate runtime config for isolated SDPA
and full-attention validation without changing the product default.

`autotune/layout_graph.py` performs the next cross-op step as three explicit
region DAGs: Attention, MLP, and LM-head. Node candidates carry measured op
costs and producer/consumer memory ports; edges carry tensor shapes and an
optional measured conversion cost. A topological beam of width 4-16 rejects
unmeasured transitions and illegal explicit shards before ranking paths by
operator plus conversion latency. Reports name removed, retained, and already
elided conversions separately and retain every rejected transition.

The P150A Llama 3.1 workload exposes an important device constraint here.
TTNN decode SDPA rejects sharded output when query and KV head counts differ,
so the tempting direct SDPA-to-concat height-sharded path is illegal for the
32/8 GQA shape. The shared legality engine records
`SDPA_GQA_SHARDED_OUTPUT_UNSUPPORTED`; the layout beam therefore retains the
measured DRAM-to-height-sharded conversion. Whole-layer confirmation always
selects either a passing challenger or the incumbent, so a runtime rejection
or latency regression cannot be promoted.

`autotune/search.py` composes the earlier search layers without constructing a
Cartesian product. Each template axis first retains an independent top-K,
each operator group retains a latency/L1/conversion Pareto frontier, and each
layout group contributes compatible path mutations. These local JSON-path
mutations are applied to the current beam instead of replacing a complete
candidate snapshot, so an SDPA choice cannot silently erase an earlier
template choice and a layout choice can intentionally refine producer memory.
The beam is capped to 4-16 states after every group and always protects the
root incumbent for later matched comparison.

The orchestrator then ranks representative whole-layer runs, sends only the
configured top candidates to a complete 32-layer post-prefill trace, and
compares the top non-incumbent with the root incumbent. Candidate cache keys
cover the full search request plus frozen `CandidateConfig` identity, including
TTNN commit, device descriptor, precision, graph, model, weights, prompt, and
target. Each callback result is atomically complete or absent. Checkpoints
retain candidate and device-minute usage; a matching completed campaign or
individual result can be resumed, while an identity mismatch is rejected.
Exceptions and budget exhaustion update `search_report.json` before returning.

`autotune/confirmation.py` owns the final promotion decision. It requires a
matched pair of final `CandidateConfig` objects and exactly three reports per
arm under the frozen 5-warmup/100-iteration contract. It validates trace,
persistent inputs, post-prefill execution, correctness, and quality; computes
per-arm median and population CV; and promotes only when both CV values are at
most 1.5% and the configured metric improves by at least 1%. The campaign runs
arms in A-B, B-A, A-B order. A valid but sub-threshold candidate produces a
passed confirmation that explicitly selects the incumbent.

The Phase 8 P150A campaign exercised all six stages with batch 32, cache 1024,
prefill bucket 256, the official 32-prompt corpus, and full 32-layer traces.
The incumbent reached a three-run median of `33.9425 t/s/u` with `0.0365%` CV;
the legal SDPA `q_chunk_size=32` challenger reached `33.9103 t/s/u` with
`0.3255%` CV. Its `-0.0948%` relative change failed the 1% improvement gate,
so `best_config.json` retained the official q-chunk and rebuilt directly into
a complete generated program bundle.

`autotune/generalization.py` coordinates the same hierarchical algorithm
across distinct model/target shapes. Every workload records a shape and
campaign fingerprint, uses the same template/op/layout group structure, and
either executes or imports a matched search report with a compatible frozen
target. Official hand configurations may be the root incumbent only; they
cannot enter as challenger proposals. Suite acceptance also requires no
Cartesian exhaustive search, a passing directly buildable config for every
workload, and publication breadth of either two models or one model with at
least three workloads. Interrupted suites atomically retain the active
workload and reuse completed search reports on resume.

The Phase 9 P150A campaign met the one-model/three-workload criterion with
Llama 3.1 8B batch32/prefill256 and cache lengths 512, 1024, and 2048. The two
new shapes independently enumerated 224 and 226 legal SDPA candidates after
static pruning, then selected the same shape-derived `q_chunk_size=32` axis
without an official challenger seed. Under matched `5x100x3` confirmation,
the incumbent medians were `34.0405`, `33.9425`, and `34.0052 t/s/u`; the
challengers changed throughput by `-0.0833%`, `-0.0948%`, and `-0.0090%`.
All arm CVs were below 0.33%, so the conservative 1% gate retained each
incumbent and all three final configs rebuilt successfully.

`autotune/artifact.py` closes the experiment by loading the Phase 3-9 reports
as immutable inputs. It rejects missing or failed evidence, inconsistent
enumeration counts, incomplete accuracy/generalization gates, absent final
configs, and any ablation matrix that omits or mislabels a required row. The
output bundle contains canonical JSON, an ablation CSV, a Markdown result
table, a normalized rebuild spec, source and output SHA256 manifests, compact
checked-in evidence, and a non-device verification/rebuild script. Atomic
failure reports remain available even when the input spec is invalid.

The Phase 10 artifact records 456 total search entities: 8 template variants,
174 MatMul programs, 256 SDPA programs, and 18 layout states. Analytical
checks reject 79 before timing, including 78 of 430 program candidates, for a
program pruning ratio of `18.14%`. The campaign accounts for 8 microbenchmark
experiments, 2 short and 6 strict full-model profiles, and `775.54 s` of
candidate subprocess device time. The selected `33.9425 t/s/u` result is
`101.866%` of the locally reproduced corresponding release. Fixed-corpus
top-1/top-5 and greedy agreement plus logits, hidden, and KV PCC remain
passing. Nine ablation rows cover every required search-axis combination,
representative-layer transfer on/off, and analytical pruning on/off; reused
performance is allowed only when the selected config SHA256 is identical.

## Runtime Ownership

`TTNNDirectRuntimeContext` owns the generated model, tensorized parameters,
paged KV cache, prompt token IDs, page table, cache positions, rotary state,
and tokenizer metadata for one generate run.

The intended invariants are:

- parameters are tensorized once per generate call;
- the paged KV cache is initialized once and filled by prefill;
- decode steps reuse the same KV cache;
- prefill and decode RoPE tensors are derived from the HF model configuration;
- prefill selects the last valid prompt position before final norm and LM-head;
- generated token tensors are handed directly to the next decode step;
- persistent decode mode updates current position and rotary index on device,
  gathers cos/sin from full device caches, and performs no per-step
  `ttnn.from_torch` calls;
- trace decode mode clones one stable token input, captures rotary lookup,
  embedding, all decoder layers, final norm, split LM-head, force-argmax,
  token feedback, and position increments in one device trace;
- trace replay performs one nonblocking `execute_trace` and one device
  synchronization per step without recreating model, weights, KV cache, or
  trace inputs;
- host token materialization is limited to reporting and detokenization.

`runtime_input_mode` selects `recreate` or `persistent`. The P150A performance
configuration defaults to `persistent`; `recreate` remains available as a
matched A/B control. Generate and steady-profile reports expose per-step
device-tensor creation, host-to-device updates, page-table updates,
cache-position updates, rotary updates, and token device copies.

`execution_mode` selects `eager` or `trace`. Trace mode requires persistent
inputs and opens P150A with the official Llama 3.1 8B trace-region reservation
of 52,000,000 bytes. Steady profiling leaves generated tokens entirely on the
device. Generate mode materializes token IDs only for reporting while the
captured device-to-device feedback remains the autoregressive input. Per-op
section profiling is disabled during trace capture because TTNN forbids event
synchronization inside a trace.

Trace reports expose the complete `DecodeTraceKey`, capture/replay/compile-run
counts, persistent input and update counts, and the number of program-cache
entries added after capture. A successful steady run has one capture, one
compile run, `warmup + iterations` replays, and zero post-capture programs.

Prefill profiling is independent from steady decode. It times one complete
batched prefill and reports average TTFT per user as batch latency divided by
batch size, matching the official demo's convention. Reference provenance is
explicit: the published P150 target is tied to
`v0.64.0-dev20251030` (`b76035f`), and a newer same-commit local official run is
reported separately. The eager path is the production default. Prefill trace
keys include prefill length, batch size, and a configuration hash, but the
current TTNN runtime synchronizes at the large residual add during trace
capture; this candidate therefore remains disabled instead of falling back.

The parity harness records prefetcher, global-CB, sub-device, trace, and
sampling controls for every official profile. The corresponding published
release does not expose the Llama prefetcher path. The newer same-commit source
does expose it, but its fixture and command-line defaults are disabled and the
matched benchmark does not request it. Buddy therefore keeps prefetcher and
sub-device execution disabled unless a future matched official A/B clears the
documented one-percent promotion threshold.

Final parity runs keep three comparison scopes explicit: corresponding-release
official, same-commit current official, and Buddy. Each run writes a contract
beside its raw log/profile so a matching passed run can be resumed safely. A
damaged top-level report can recover only artifacts with complete sample counts
and, for pytest runs, a zero-error JUnit result. Release execution isolates
`TT_METAL_HOME`, `TT_METAL_BUILD_HOME`, and `TT_METAL_RUNTIME_ROOT`; clean
release model source precedes the same-commit compiled runtime on `PYTHONPATH`.

Execution-graph capture is diagnostic-only. `execution-graph-diff` wraps the
corresponding-release official trace through the parity pytest plugin and asks
the Buddy trace session to emit TTNN graph plus Python I/O metadata. Normal
generate/profile runs do not create graph files. `TTNNCompatOps` semantic op
recording is also disabled by default and enabled only by observers and smoke
diagnostics.

Custom operations remain outside the production path. After Goal 7 reaches
M8, the Goal 8 audit finds no residual region that satisfies the required
greater-than-one-millisecond measurement and unsupported-official-composition
criteria. LM-head force-argmax, residual/RMSNorm, and SDPA/concat-heads remain
composed from the matched official TTNN operations.

## Evidence Boundary

Runtime reference data lives under `buddy_ttnn_direct/reference/` and may be
loaded by product code. Historical run evidence lives under `docs/evidence/`
and must never be imported by runtime code.

## Compatibility Boundaries

Generated-model source fragments live under `compiler/templates/`, divided
into preamble, model core, attention, MLP, LM-head, and constants. The small
`compiler/source_templates.py` renderer composes those fragments. Generated
programs import `TTNNCompatOps` from `ttnn_compat/model_ops.py` instead of
embedding a private copy of the compatibility layer.

The product parser registers exactly `build`, `generate`, `profile`,
`validate`, `inspect`, and `diagnose`. Product modules must not import
diagnostics, autotune, smoke, historical search, or legacy diagnostic adapters;
AST boundary tests enforce this rule and the reports-to-diagnostics prohibition.
