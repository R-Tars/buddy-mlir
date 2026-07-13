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

## Layered Autotune

Autotune is a development diagnostic, not a product validation gate. It varies
one axis at a time in this order: LM-head split count, dtype recipe, memory
layout, then program config/core grid. Every candidate uses post-prefill steady
decode; repeated incumbents reuse the same measurement rather than expanding a
Cartesian product.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage autotune \
  --model-path "$MODEL" \
  --config "$CONFIG" \
  --prompt "Hello from TTNN Direct" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --warmup 5 \
  --iterations 10 \
  --confirm-warmup 5 \
  --confirm-iterations 50 \
  --min-relative-improvement 0.01 \
  --candidates-dir "$AUTOTUNE/candidates" \
  --out "$AUTOTUNE/report.json"
```

Hardware candidates run in isolated subprocesses and are resumable by state
fingerprint. The default `1%` minimum relative improvement applies both during
each short-measurement level and to matched 5/50 confirmations of the
provisional winner and root incumbent. This prevents a noisy short-run delta
from replacing the default. Add `--dry-run` to generate the five unique
candidate bundles without opening a device.

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

## Exit Codes

- `0`: command or validation passed.
- `1`: validation, input, or runtime failure.
- `2`: a device-executing path could not access a TTNN device.
