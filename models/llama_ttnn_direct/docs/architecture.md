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
templates. `buddy_ttnn_direct/compiler/` validates plans, builds generated
config, renders generated Python source, and writes compiler artifacts.
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
- `tokenizer.py`: prompt tokenization and generated text decoding.
- `inputs.py`: token, page-table, cache-position, and rotary inputs.
- `rotary.py`: HF-compatible Llama RoPE values and TTNN tensor placement.
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
and diagnostics. Their orchestration lives in
`diagnostics/validation_workflow.py`; the package-root `validation.py` is a
compatibility module alias. Importing or parsing the product CLI does not load
the legacy workflow.

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
- prefill and decode RoPE tensors are derived from the HF model configuration;
- prefill selects the last valid prompt position before final norm and LM-head;
- generated token tensors are handed directly to the next decode step;
- host token materialization is limited to reporting and detokenization.

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
