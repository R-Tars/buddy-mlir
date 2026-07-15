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
package helpers; `codegen/python_ttnn.py` is a compatibility facade for the
canonical compiler modules.

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

`buddy_ttnn_direct/generate.py` remains a small compatibility facade that
re-exports the public runtime entry points.

### Reports and validation

`buddy_ttnn_direct/reports/` owns report-only logic. The modules do not open a
device, execute model stages, or import `buddy_ttnn_direct/diagnostics/`.

- `schema.py`: shared field, path, number, and acceptance helpers.
- `validation.py`: product suites and retained legacy acceptance assembly.
- `performance.py`: baseline and milestone summaries.
- `evidence.py`: compact reproducibility/evidence manifests.
- `runtime.py`, `profiling.py`, `tensorization.py`: report contract checks.
- `attention.py`, `depth.py`, `autotune.py`: diagnostics report checks.
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

Legacy phase gates remain available to diagnostics, but they are not registered
as hidden top-level commands. Their orchestration lives in
`diagnostics/validation_workflow.py`; the package-root `validation.py` is a
compatibility module alias. Importing or parsing the product CLI does not load
the legacy workflow.

### Diagnostics

Smoke, sweep, layered autotune, and legacy decode workflows are development
tools. The visible CLI exposes them only through `diagnose --stage ...`; their
tests live under `tests_diagnostics/` and are excluded from the default product
suite. `diagnostics/cli.py` is loaded only when `diagnose` executes. Current
autotune runs one axis at a time with isolated post-prefill
steady-decode subprocesses. Phase-era Cartesian search is quarantined under
`future/historical_search/` and is loaded only by diagnostics compatibility
paths.

### Semantic autotune contracts

`buddy_ttnn_direct/autotune/` owns the schema-v2 candidate identity. Its frozen
precision contract is derived from the checked-in official performance profile
and includes weight/activation/KV dtypes, per-layer overrides, math fidelity,
FP32 accumulation, approximation, and packer settings. Execution is fixed to
post-prefill full decode trace with persistent inputs, fixed page tables, and
force-argmax sampling. Candidate fingerprints include these contracts plus the
semantic graph, model config, weights/recipe identity, runtime commit, device,
workload, tunable state, and measurement counts. The legacy layered runner is
an adapter to this contract and cannot search dtype or fidelity fields.

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

Legacy validation remains available under diagnostics for compatibility but
is not part of product validation or default tests. It can be simplified
without changing the visible command or report contracts described here.

The product parser registers exactly `build`, `generate`, `profile`,
`validate`, `inspect`, and `diagnose`. Product runtime modules must not import
`smoke_*`, `decode_loop`, or `profile_template`; boundary tests enforce this
rule along with the reports-to-diagnostics prohibition.
