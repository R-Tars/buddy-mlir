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
- The Step B mixed prefill plus one-decode profile reached `1.3583
  tokens/s/user` and milestone M2 with official force-argmax, a 48% improvement
  over the comparable tiled-argmax run.
- Step C post-prefill steady decode passed on P150A with 5 warmup and 50 measured
  iterations. Mean decode latency is `33.962 ms`, throughput is `29.445
  tokens/s/user` (`942.23` aggregate), and the run reaches `88.96%` of the
  recorded official `33.1 tokens/s/user` target.
- Step D imports the P150/Llama 3.1 8B TT-Transformers performance profile at
  tt-metal commit `61e690c2`. All 55 fields across dtype, fidelity, memory,
  program, grid, LM-head, and paged-attention sections match the extracted
  reference. DRAM-sharded weights and the layer-31 precision override execute
  successfully at full depth.
- The Step D full-depth profile measures `34.933 ms` mean decode latency and
  `28.627 tokens/s/user` (`86.49%` of target). This is `2.78%` slower than the
  Step C baseline, so config parity is complete but performance parity remains
  a Step E optimization target.
- The dedicated all-BF16 correctness recipe passes the Hugging Face reference
  gates at depths `1,2,4,32`; full-depth logits PCC is `0.99891` and the
  minimum sampled hidden/KV PCC is `0.99260` at a `0.99` threshold.
- The production greedy path follows TT-Transformers force-argmax: concatenate
  LM-head logits, untilize with multicore, then run multicore argmax. A composed
  shard-local/global reduction remains available for diagnostics but was slower
  on P150A. The steady benchmark reaches milestone M5 but remains below the M6
  greater-than-90% threshold, so official performance parity is not claimed.

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
| Official TT-Transformers config parity | 55/55 compared fields match; full-depth execution passed |
| Steady decode benchmark | 28.627 tokens/s/user with imported profile, 86.49% of target |
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

Profile the same workflow or post-prefill steady decode with the `profile`
command. Detailed examples and suite semantics are in
[docs/commands.md](docs/commands.md).

## Commands

The user-facing CLI exposes only:

- `build`: create the generated TTNN program bundle.
- `generate`: run or dry-run prompt prefill followed by decode.
- `profile`: profile generate or post-prefill steady decode.
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
- Generate-mode profile reports record per-section/per-layer timing and the
  underlying generate report. Decode-steady reports record warmup samples,
  measured p50/mean decode latency, separate prefill latency, and throughput.
- Validation reports contain compact named checks and `failed_checks`; the
  product path does not run smoke, search, or autotune gates.

Historical evidence under `docs/evidence/` is documentation only and is never
imported by runtime code.

## Known Limitations

- The composed local/global LM-head reduction needs a fused TTNN operation to
  become competitive with the official force-argmax path.
- Numerical correctness is proven with the dedicated all-BF16 recipe. The
  compressed performance recipe has a separate, lower-precision acceptance
  profile and is not claimed to pass the `0.99` full-depth PCC gate.
- The steady decode path still creates five runtime metadata/rotary tensors per
  iteration and the imported profile reaches `86.49%`, not the M6
  greater-than-90% milestone.
- Buddy prefill represents 32 users in one tensor, so the imported QKV and WO
  prefill configs disable TT-Transformers' single-sequence batch fusion while
  retaining the extracted grid and block geometry.
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
