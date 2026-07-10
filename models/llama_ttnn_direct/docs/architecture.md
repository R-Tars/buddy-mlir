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

## Package Boundaries

### Semantic model

`buddy_ttnn_direct/semantic/` owns the Llama graph dataclasses, Hugging Face
importer, graph validation, and JSON I/O. It does not import TTNN or open a
device.

### Planning and code generation

`buddy_ttnn_direct/templates/` maps semantic operations to official-like TTNN
templates. `buddy_ttnn_direct/codegen/` builds the program bundle, emits
normalized config and parameter metadata, materializes weights, tensorizes
parameters, and writes generated Python.

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
- `tokenizer.py`: prompt tokenization and generated text decoding.
- `inputs.py`: token, page-table, cache-position, and rotary inputs.
- `kv_cache.py`: paged KV-cache allocation and metadata.
- `prefill.py`: prompt prefill orchestration.
- `decode.py`: decode-step and token handoff orchestration.
- `generate.py`: high-level generate entry point.
- `profile.py`: generate section profiling and profile report assembly.
- `reports.py`: generate report schemas and JSON writing.

`buddy_ttnn_direct/generate.py` remains a small compatibility facade that
re-exports the public runtime entry points.

### Reports and validation

`buddy_ttnn_direct/reports/` owns report-only logic. The modules do not open a
device and do not execute model stages.

- `schema.py`: shared field, path, number, and acceptance helpers.
- `validation.py`: product suites and retained legacy acceptance assembly.
- `performance.py`: baseline and milestone summaries.
- `evidence.py`: compact reproducibility/evidence manifests.
- `runtime.py`, `profiling.py`, `tensorization.py`: report contract checks.
- `attention.py`, `depth.py`, `autotune.py`: diagnostics report checks.
- `config.py`, `artifacts.py`: config coverage and artifact checks.

The product CLI uses four compact suites:

- `dryrun`: required bundle artifacts plus generate/profile dry-runs.
- `functional`: successful prefill/decode, prefilled KV cache, and generated
  text.
- `device`: functional checks plus TTNN identity and optional full depth.
- `performance`: positive throughput and complete profile sections, with
  optional full depth.

Legacy phase gates remain reachable only through hidden compatibility commands
and diagnostics. Importing or parsing the product CLI does not load the legacy
`validation.py` module.

### Diagnostics

Smoke, sweep, search, and legacy decode workflows are development tools. The
visible CLI exposes them only through `diagnose --stage ...`; their tests live
under `tests_diagnostics/` and are excluded from the default product suite.

## Runtime Ownership

`TTNNDirectRuntimeContext` owns the generated model, tensorized parameters,
paged KV cache, prompt token IDs, page table, cache positions, rotary state,
and tokenizer metadata for one generate run.

The intended invariants are:

- parameters are tensorized once per generate call;
- the paged KV cache is initialized once and filled by prefill;
- decode steps reuse the same KV cache;
- generated token tensors are handed directly to the next decode step;
- host token materialization is limited to reporting and detokenization.

## Evidence Boundary

Runtime reference data lives under `buddy_ttnn_direct/reference/` and may be
loaded by product code. Historical run evidence lives under `docs/evidence/`
and must never be imported by runtime code.

## Remaining Refactor Work

The runtime split and report split are complete enough for the product path,
but two large compatibility areas remain:

- `codegen/python_ttnn.py` still combines generated source and TTNN wrappers;
- legacy validation and diagnostic implementations remain available for
  compatibility.

These are isolated from the visible product workflow and can be simplified
without changing the command or report contracts described here.
