# TTNN Direct Command Reference

Run commands from the Buddy MLIR repository root:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli --help
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
export CONFIG=models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json
export PROGRAM=/tmp/llama31_ttnn_direct
```

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
  --out /tmp/ttnn_direct_inspect.json
```

## Generate

### Device-free dry-run

Dry-run validates plans and writes the generate report schema without loading
weights or opening a device:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli generate \
  --program-dir "$PROGRAM" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --max-new-tokens 2 \
  --dry-run \
  --out /tmp/ttnn_direct_generate_dryrun.json
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
  --device-id 0 \
  --out /tmp/ttnn_direct_generate.json
```

`max-new-tokens` includes the first token produced by prefill. Remaining tokens
come from decode steps.

## Profile

Profile runs the same generate path and derives throughput and section timing
from its report.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli profile \
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
  --generate-report /tmp/ttnn_direct_profile_generate.json \
  --out /tmp/ttnn_direct_profile.json
```

Add `--dry-run` for schema validation without device execution.

The profile report includes:

- prefill and decode-step latency;
- per-section and per-layer timings;
- host materialization timing;
- tokens per second per user and aggregate throughput;
- performance milestones and the underlying generate report path.

The command does not claim official parity.

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
  --out-dir /tmp/ttnn_direct_validate_dryrun
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
  --out-dir /tmp/ttnn_direct_validate_functional
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
  --out-dir /tmp/ttnn_direct_validate_device
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
  --out-dir /tmp/ttnn_direct_validate_performance
```

### Correctness suite

Runs a CPU Hugging Face reference followed by the same truncated layer count
on P150A. It compares the top token, complete last-position logits, per-layer
last-position hidden states, and deterministic prefill KV-cache coordinates:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite correctness \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --prompt "Hello from TTNN Direct" \
  --layers 1 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --check top_token,logits_pcc,hidden_pcc,kv_cache_pcc \
  --pcc-threshold 0.99 \
  --hf-reference /wafer/zhuxinye/tmp/ttnn_direct_hf_references/depth_1/hf_reference.json \
  --out-dir /tmp/ttnn_direct_validate_correctness_l1
```

Repeat with `--layers 2`, `4`, and `32`. The output directory contains the
compact comparison report plus separate `hf_reference.json`,
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
