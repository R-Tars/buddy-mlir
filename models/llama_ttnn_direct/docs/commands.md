# TTNN Direct Command Reference

Initialize paths from the Buddy MLIR repository root:

```bash
export BUDDY_REPO_ROOT=/wafer/zhuxinye/buddy-mlir
export BUDDY_BUILD="${BUDDY_BUILD:-$BUDDY_REPO_ROOT/build-tenstorrent}"
export TTNN_DIRECT_BUILD="$BUDDY_BUILD/models/llama31_ttnn_direct"
```

The visible command set is `build`, `generate`, `profile`, `validate`,
`inspect`, and `diagnose`.

## Environment

Use the built Python environment and TT-Metal setup documented in
[`docs/TenstorrentEnvironment.md`](../../../docs/TenstorrentEnvironment.md).
The device-free commands and `--dry-run` modes do not open a P150A.

The examples below use:

```bash
export MODEL=/wafer/share/models/Llama-3.1-8B-Instruct
export CONFIG="$BUDDY_REPO_ROOT/models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json"
export CORRECTNESS_CONFIG="$BUDDY_REPO_ROOT/models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32_correctness.json"
export PROGRAM="$TTNN_DIRECT_BUILD/program"
export CORRECTNESS_PROGRAM="$TTNN_DIRECT_BUILD/correctness_program"
export REPORTS="$TTNN_DIRECT_BUILD/reports"
export AUTOTUNE="$TTNN_DIRECT_BUILD/autotune"
export HF_REFERENCES="$TTNN_DIRECT_BUILD/references/hf"
export RUNTIME_ARTIFACTS="$TTNN_DIRECT_BUILD/runtime_artifacts"
export OFFICIAL_PROMPTS=/wafer/zhuxinye/tt-metal-official-repro/models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json

mkdir -p \
  "$PROGRAM" \
  "$CORRECTNESS_PROGRAM" \
  "$REPORTS/inspect" \
  "$REPORTS/dryrun" \
  "$REPORTS/generate" \
  "$REPORTS/performance" \
  "$REPORTS/validation" \
  "$REPORTS/correctness" \
  "$REPORTS/diagnostics" \
  "$AUTOTUNE/candidates" \
  "$HF_REFERENCES" \
  "$RUNTIME_ARTIFACTS"

export PYTHONPATH="$BUDDY_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TT_METAL_LOGS_PATH="$RUNTIME_ARTIFACTS"
cd "$RUNTIME_ARTIFACTS"

python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli --help
```

The runtime working directory is deliberate. Inspector honors
`TT_METAL_LOGS_PATH`, while the pinned watcher also writes relative to the
process working directory. Together these settings keep both under
`$RUNTIME_ARTIFACTS/generated` and leave the source tree clean.

The production `CONFIG` imports the extracted P150 TT-Transformers performance
profile, including compressed dtypes, compute kernels, DRAM-sharded weights,
memory layouts, and program grids. `CORRECTNESS_CONFIG` intentionally remains
all-BF16 and does not import the performance profile.

Full-depth 8B execution needs enough virtual address space to map all weight
shards and TTNN runtime objects. Check `ulimit -v`; when the shell has a lower
finite limit, use at least the benchmark harness default before launching:

```bash
ulimit -v 95000000
```

A low limit can raise `mmap ... Cannot allocate memory` even when `free -h`
shows ample available RAM.

## Build

Build imports the HF Llama graph, creates the template plan, emits normalized
config and manifests, and writes the generated Python program bundle.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli build \
  --model-path "$MODEL" \
  --config "$CONFIG" \
  --out-dir "$PROGRAM"
```

Use `inspect` to verify the required bundle files:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli inspect \
  --program-dir "$PROGRAM" \
  --out "$REPORTS/inspect/inspect.json"
```

## Generate

### Device-free dry-run

Dry-run validates plans without loading weights or opening a device. This
example explicitly requests a compact report:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli generate \
  --program-dir "$PROGRAM" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --max-new-tokens 2 \
  --dry-run \
  --report-level summary \
  --out "$REPORTS/dryrun/generate.json"
```

### P150A execution

Non-dry-run execution requires a model, prompt, tokenizer, and available
device:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli generate \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --max-new-tokens 2 \
  --device p150a \
  --device-id 0
```

`max-new-tokens` includes the first token produced by prefill. Remaining tokens
come from decode steps. Generation prints decoded text to the terminal and
does not create a JSON report unless `--out` is provided.

### Official batch32 prompts

Use the same JSON and instruct formatting as the TT-Metal text demo when
comparing batch output. Buddy reads the first 32 prompt objects, applies
`tokenizer.apply_chat_template` with one user turn and an assistant generation
prompt, and keeps each user's prompt and decode positions independent:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli generate \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --input-prompts "$OFFICIAL_PROMPTS" \
  --instruct \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 256 \
  --cache-len 1024 \
  --max-new-tokens 200 \
  --device p150a \
  --device-id 0 \
  --report-level summary \
  --out "$REPORTS/generate/official_batch32.json"
```

The official instruct prompts are longer than 128 tokens for some users, so a
128-token Buddy prefill would not be an input-parity run. Prompt-file mode
rejects truncation and reports the required minimum. The summary records the
input file SHA256, per-user prompt SHA256 values, and per-user token counts.

Use one of the optional report modes when an artifact is needed:

```bash
# Compact result, timing, throughput, and cache-capacity summary.
--report-level summary --out "$REPORTS/generate/generate.json"

# Runtime evidence plus streamed per-step diagnostics.
--report-level full --out "$REPORTS/generate/generate.json"
```

For compatibility, supplying `--out` without `--report-level` selects `full`.
Full mode writes the main report atomically and streams decode details to the
sibling `generate.steps.jsonl`; repeated structural references are stored once
in `generate.references.jsonl` and linked by ID. Cache capacity is checked
before opening the device: `effective_prompt_tokens + max_new_tokens - 1`
must not exceed `cache-len`.

## Profile

Generate mode runs the same generate path and derives throughput and section
timing from its report.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli profile \
  --mode generate \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --max-new-tokens 2 \
  --device p150a \
  --generate-report "$REPORTS/performance/generate.json" \
  --out "$REPORTS/performance/generate_profile.json"
```

Add `--dry-run` for schema validation without device execution.

The profile report includes:

- prefill and decode-step latency;
- per-section and per-layer timings;
- host materialization timing;
- tokens per second per user and aggregate throughput;
- performance milestones and the underlying generate report path.

The command does not claim official parity.

### Post-prefill steady decode

Steady mode materializes and tensorizes parameters once, runs prompt prefill
once, excludes warmup iterations, and measures repeated decode iterations. The
timed region includes decode metadata preparation, generated `decode_step`, and
device synchronization. It excludes prefill, warmup, host token copies, and the
per-op diagnostic profiler.

When `--layers` is omitted in this mode, all generated layers are used.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli profile \
  --mode decode-steady \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --warmup 5 \
  --iterations 50 \
  --after-prefill \
  --device p150a \
  --out "$REPORTS/performance/decode_steady.json"
```

The report records `prefill_ms`, `decode_step_ms_p50`,
`decode_step_ms_mean`, `tokens_per_second_per_user`, and
`aggregate_tokens_per_second`. Add `--dry-run` to validate the report contract
without loading weights or opening a device.

For a steady-decode comparison initialized from the official batch32 prompts,
replace `--prompt` in the command above with:

```bash
--input-prompts "$OFFICIAL_PROMPTS" --instruct --prefill-len 256
```

### Batched prefill and TTFT

Prefill steady mode reuses one loaded model session, excludes warmup runs, and
measures the complete batch32 prefill. It reports both the batch latency and
the official demo's average-TTFT convention:

```text
average_ttft_ms_per_user = batch_prefill_latency_ms / batch_size
```

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli profile \
  --mode prefill-steady \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --input-prompts "$OFFICIAL_PROMPTS" \
  --instruct \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 256 \
  --cache-len 1024 \
  --warmup 1 \
  --iterations 3 \
  --prefill-execution-mode eager \
  --device p150a \
  --out "$REPORTS/performance/prefill_steady.json"
```

Official references in this report are versioned. The published P150 numbers
belong to `tt-metal v0.64.0-dev20251030` (`b76035f`), while the same-commit
local comparison uses the Buddy runtime's `tt-metal` commit (`61e690c`). These
are separate comparison scopes. The corresponding-release local TTFT, the
same-commit local TTFT, and the published `57 ms` reference are all reported.

`--prefill-execution-mode trace` is an explicit experiment. On the current
runtime, TTNN synchronizes at the large prefill residual add during capture, so
the production default remains eager and the failed trace candidate is retained
as evidence rather than silently falling back.

## Layered Autotune

Autotune is a development diagnostic, not a product validation gate. Its
compatibility runner varies one axis at a time in this order: LM-head split
count, memory layout, then program config/core grid. The production precision
recipe and all compute fidelity fields are frozen by the candidate contract.
Every candidate explicitly uses post-prefill steady decode, full trace, and
persistent inputs; repeated incumbents reuse the same measurement rather than
expanding a Cartesian product.

The compatibility CLI still accepts its historical preset progression, but
each preset is immediately expanded to schema v2. Candidate artifacts expose
typed `templates`, `operators`, `memory_configs`, `core_grids`, and `edges`;
generated runtime configs do not retain the old preset strings.

The schema-v2 legality layer statically checks candidate shapes, grids,
programs, sharding, layouts, and conservative L1/CB capacity before a device
command can launch. Legal candidates may then use the isolated compile-only
validator. Reports retain accepted and rejected candidates with stable error
classes. The command below remains the historical compatibility runner; the
hierarchical semantic search orchestrator consumes the same legality API.

### Existing-API Template Registry

The semantic registry contains eight candidates across KV update, QK RoPE,
MLP activation placement, and gate/up projection. It changes no precision
field and uses only public APIs from the active TTNN runtime. The default gate
and up projections remain separate.

Run all template hooks without opening a device against an existing program:

```bash
python -c '
import json
from models.llama_ttnn_direct.buddy_ttnn_direct.autotune import (
    dry_run_template,
    list_template_definitions,
)
runtime = json.load(open("build-tenstorrent/models/llama31_ttnn_direct/program/config.json"))
for definition in list_template_definitions():
    report = dry_run_template(definition.name, runtime)
    print(definition.name, report["status"])
'
```

Availability probing accepts the imported `ttnn` module and checks the exact
API groups without opening a device. Generated single-layer regression covers
each template independently; P150A acceptance reports belong under
`$AUTOTUNE`, not in the source tree. First-call diagnostic latency includes
compilation and must not be reported as template performance.

### Representative Microbenchmark Engine

The product API in `autotune/microbench.py` accepts op and region targets. A
worker is named as `module:callable`; the parent writes a frozen request and
runs one isolated Python process per repetition. Workers should construct the
response with `make_worker_response(...)` so warmup, trace/persistent mode,
sample count, and instrumentation are validated before caching.

Llama 3.1 8B uses the transfer plan returned by
`build_llama31_8b_transfer_plan()`: layer 0 represents layers 0-30 and layer 31
is a singleton override. The short, confirmation, and final measurement
presets are respectively `3x10-20x1`, `5x50x2`, and `5x100x3`.

Run the device-free protocol and cache acceptance tests:

```bash
python -m pytest -q \
  models/llama_ttnn_direct/buddy_ttnn_direct/tests/test_autotune_microbench.py \
  models/llama_ttnn_direct/buddy_ttnn_direct/tests/test_autotune_contracts.py
```

Retained search runs should place `measurement_cache/`, `process_logs/`, and
`microbench_report.json` beneath the active model build tree. Cached reports
are valid only for their exact key; failed or partial subprocess runs are not
reused.

### MatMul Program Enumeration

`enumerate_all_matmul_programs(...)` consumes the schema-v2 baseline,
representative workload, P150A descriptor, and frozen precision contract. It
returns legal and rejected candidates for QKV, O projection, MLP gate/up/down,
and LM-head shards. The official program vector is included exactly once for
each operator. Candidate spaces can be materialized with
`candidate.search_space.apply_to_runtime_config(...)`.

Run the device-free enumerator and Phase 4 selection integration tests:

```bash
python -m pytest -q \
  models/llama_ttnn_direct/buddy_ttnn_direct/tests/test_autotune_matmul.py
```

The selected microbenchmark winner is a pruning/ranking result only. It must
still pass whole-layer and full-model confirmation before promotion.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage autotune \
  --model-path "$MODEL" \
  --config "$CONFIG" \
  --official-tt-metal-root "$OFFICIAL_TT_METAL_ROOT" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --warmup 5 \
  --iterations 10 \
  --repetitions 3 \
  --min-relative-improvement 0.01 \
  --candidates-dir "$AUTOTUNE/candidates" \
  --out "$AUTOTUNE/report.json"
```

Hardware candidates run in isolated subprocesses and are resumable by the
schema-v2 candidate fingerprint. Legal MatMul and SDPA candidates first pass
active successive halving; hierarchical search then composes template,
operator, and layout proposals. The default `1%` minimum relative improvement
is applied by the final matched `5x100x3` A/B confirmation. Add `--dry-run` to
serialize the canonical search plan and contracts without opening a device.

### Paper Artifact

The Phase 10 artifact CLI is a development tool and does not open a device. It
loads completed raw reports, validates their pass state, records all source
hashes, derives the paper metrics and required ablations, and creates a
self-verifying bundle beneath the active build tree:

```bash
export PAPER_ARTIFACT="$AUTOTUNE/doc6_phase10_artifact/bundle"

python -m models.llama_ttnn_direct.buddy_ttnn_direct.autotune.artifact_cli \
  verify \
  --artifact-dir "$PAPER_ARTIFACT"
```

Rebuild from the normalized, hash-pinned spec without overwriting the captured
bundle:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.autotune.artifact_cli \
  build \
  --spec "$PAPER_ARTIFACT/spec.json" \
  --out-dir "$AUTOTUNE/doc6_phase10_artifact/manual-rebuild"

python -m models.llama_ttnn_direct.buddy_ttnn_direct.autotune.artifact_cli \
  verify \
  --artifact-dir "$AUTOTUNE/doc6_phase10_artifact/manual-rebuild"
```

`paper_artifact.json` is the machine-readable result, `ablation.csv` and
`RESULTS.md` are presentation views, and both source and generated files are
covered by SHA256 manifests. `reproduce.sh` verifies the original first and
then writes each rebuild to a unique child directory.

## Validate

Validation emits `/OUT_DIR/validation_report.json` with named checks and a flat
`failed_checks` list.

### Dry-run suite

Checks required program artifacts and runs device-free generate/profile
dry-runs:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite dryrun \
  --program-dir "$PROGRAM" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --max-new-tokens 2 \
  --out-dir "$REPORTS/validation/dryrun"
```

### Functional suite

Runs generate and requires successful prefill/decode, prompt-conditioned model
semantics, `kv_cache_source=prefill`, and non-empty generated text:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite functional \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --out-dir "$REPORTS/validation/functional"
```

### Device suite

Adds TTNN runtime identity and device-target checks. Add
`--require-full-depth` to require `layers == program_num_layers`.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite device \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --device p150a \
  --require-full-depth \
  --out-dir "$REPORTS/validation/device"
```

### Performance suite

Runs profile and requires a passed profile, positive tokens/s/user, and all
required section records. It does not require or claim official parity.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite performance \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --device p150a \
  --require-full-depth \
  --out-dir "$REPORTS/validation/performance"
```

### Correctness suite

Runs a CPU Hugging Face reference followed by the same truncated layer count
on P150A. It compares the top token, complete last-position logits, per-layer
last-position hidden states, and deterministic prefill KV-cache coordinates:

Build the dedicated high-accuracy program first. This keeps the all-BF16
correctness contract separate from the compressed performance recipe:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli build \
  --model-path "$MODEL" \
  --config "$CORRECTNESS_CONFIG" \
  --out-dir "$CORRECTNESS_PROGRAM"
```

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite correctness \
  --program-dir "$CORRECTNESS_PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 1 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --check top_token,logits_pcc,hidden_pcc,kv_cache_pcc \
  --pcc-threshold 0.99 \
  --hf-reference "$HF_REFERENCES/depth_1/hf_reference.json" \
  --out-dir "$REPORTS/correctness/depth_1"
```

The Step A contract requires exact top-token matching at depth 1. Repeat with
`--layers 2`, `4`, and `32` and use
`--check logits_pcc,hidden_pcc,kv_cache_pcc` for those sampled-depth PCC gates.
The output directory contains the compact comparison report plus separate
`hf_reference.json`,
`ttnn_observations.json`, and `generate.json` evidence files.
Omit `--hf-reference` to capture the CPU reference during the command. A
provided artifact is rejected unless its model config, prompt, depth, prefill
length, dtype, and checkpoint count match the request.

## Diagnose

Diagnostics are explicit development tools and are not part of the product
acceptance path. See
[`buddy_ttnn_direct/diagnostics/README.md`](../buddy_ttnn_direct/diagnostics/README.md)
for stages and examples.

Phase-era validation orchestration was retired after
`build`/`generate`/`profile`/`validate`/`inspect`/`diagnose` became the
supported surfaces.

## Exit Codes

- `0`: command or validation passed.
- `1`: validation, input, or runtime failure.
- `2`: a device-executing path could not access a TTNN device.
