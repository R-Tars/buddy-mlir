# Buddy-TTNN Direct

Buddy-TTNN Direct generates an official-style Python TTNN program from a
Hugging Face Llama model description. It is an additive path and does not
replace the existing `models/llama31_tt` TTIR baseline.

The product workflow is:

```text
HF config and weights
  -> semantic graph
  -> TTNN execution plan and generated Python program
  -> parameter materialization and TTNN tensorization
  -> prompt prefill
  -> paged-KV decode
  -> generated text and profile reports
```

## Current Status

- Full-depth prompt-conditioned prefill, decode, and text generation has
  passed on P150A for Llama 3.1 8B, batch 32.
- The generated decode path reuses a prefilled paged KV cache and keeps token
  handoff on device; host token materialization is for reporting and text
  decoding.
- The frozen pre-refactor profile measured about `0.1898 tokens/s/user` versus
  the recorded official target of `33.1 tokens/s/user`. This is functional
  evidence, not performance parity.
- The dedicated all-BF16 correctness recipe passes the Hugging Face reference
  gates at depths `1,2,4,32`; full-depth logits PCC is `0.99891` and the
  minimum sampled hidden/KV PCC is `0.99260` at a `0.99` threshold.
- The frozen profile's full-logits argmax was the dominant measured
  bottleneck. The current path selects the final valid prompt position before
  final norm and LM-head; a post-fix device profile is still pending. Split
  LM-head local argmax plus global reduction remains the next decode target.

See [REFACTOR_BASELINE.md](REFACTOR_BASELINE.md) and
[docs/evidence/README.md](docs/evidence/README.md) for the frozen measurements.

## Support Matrix

| Area | Status |
| --- | --- |
| Llama semantic import | Supported |
| Decode and prefill plan generation | Supported |
| Python TTNN program generation | Supported |
| HF weight materialization | Supported |
| TTNN parameter tensorization | Supported |
| Prompt-conditioned prefill and decode | Functional on P150A |
| Paged KV cache | Functional |
| Product dry-run workflow | Device-free |
| Full-model numerical correctness | Proven on P150A with all-BF16 recipe |
| Official performance parity | Not achieved |

## Quick Start

Set up TT-Metal and the Buddy toolchain as described in
[`docs/TenstorrentEnvironment.md`](../../docs/TenstorrentEnvironment.md).

Define paths used by the examples:

```bash
export MODEL=/wafer/share/models/Llama-3.1-8B-Instruct
export CONFIG=models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json
export CORRECTNESS_CONFIG=models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32_correctness.json
export PROGRAM=/tmp/llama31_ttnn_direct
```

Build a generated program bundle:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli build \
  --model-path "$MODEL" \
  --config "$CONFIG" \
  --out-dir "$PROGRAM"
```

Run a device-free validation of the generated bundle:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli validate \
  --suite dryrun \
  --program-dir "$PROGRAM" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --max-new-tokens 2 \
  --out-dir /tmp/ttnn_direct_validate
```

Run prompt-conditioned generation on an available P150A:

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
  --out /tmp/ttnn_direct_generate.json
```

Profile the same workflow with the `profile` command. Detailed examples and
suite semantics are in [docs/commands.md](docs/commands.md).

## Commands

The user-facing CLI exposes only:

- `build`: create the generated TTNN program bundle.
- `generate`: run or dry-run prompt prefill followed by decode.
- `profile`: profile generate and emit section/throughput data.
- `validate`: run `dryrun`, `functional`, `device`, `performance`, or
  `correctness` gates.
- `inspect`: inspect required program artifacts and normalized config.
- `diagnose`: run explicitly selected development diagnostics.

Bring-up commands are documented separately in
[diagnostics/README.md](buddy_ttnn_direct/diagnostics/README.md).

## Reports

All commands emit JSON with a stable top-level `schema_version`, `status`, and
`passed` contract.

- Generate reports record prompt/prefill/decode ownership, KV-cache source,
  generated token IDs and text, runtime environment, latency, and reference
  checks.
- Profile reports record prefill/decode timings, per-section and per-layer
  timing data, throughput, and the path to the underlying generate report.
- Validation reports contain compact named checks and `failed_checks`; the
  product path does not run smoke, search, or autotune gates.

Historical evidence under `docs/evidence/` is documentation only and is never
imported by runtime code.

## Known Limitations

- LM-head shards are concatenated into full logits before argmax.
- Numerical correctness is proven with the dedicated all-BF16 recipe. The
  compressed performance recipe has a separate, lower-precision acceptance
  profile and is not claimed to pass the `0.99` full-depth PCC gate.
- The current profile mixes prefill and a short decode run; a steady-state
  decode benchmark is still needed for official comparison.
- Official dtype, memory, program, and core-grid parity remains incomplete.
- The primary executable is the Python CLI; there is no integrated
  `buddy-cli` runner for this path.

## Development

Run the eight-file product test set:

```bash
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests -q
```

Run retained bring-up and legacy coverage explicitly:

```bash
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
```

Additional references:

- [Architecture](docs/architecture.md)
- [Command reference](docs/commands.md)
- [Historical evidence](docs/evidence/README.md)
- [Diagnostics](buddy_ttnn_direct/diagnostics/README.md)
