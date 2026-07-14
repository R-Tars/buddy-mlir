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
- `rotary.py`: HF-compatible Llama RoPE values and TTNN tensor placement.
- `kv_cache.py`: paged KV-cache allocation and metadata.
- `prefill.py`: prompt prefill orchestration.
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
