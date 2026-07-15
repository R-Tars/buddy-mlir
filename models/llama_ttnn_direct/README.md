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
- Historical Step E used four progressive levels instead of a Cartesian
  search. The precision-preserving semantic-autotune contract now freezes the
  production dtype/fidelity recipe, so the compatibility runner varies only
  LM-head splits, memory layout, and program/grid. Every candidate uses full
  decode trace, persistent inputs, and post-prefill steady decode in an
  isolated process. A `1%` promotion threshold keeps the incumbent when a
  challenger is within run-to-run noise.
- Autotune candidate state now uses a lossless schema-v2 representation for
  template choices, matmul/SDPA programs, memory and sharding, core grids, and
  layout edges. Historical preset strings are expanded by a compatibility
  adapter before fingerprinting or code generation.
- The schema-v2 legality engine rejects incompatible shapes, grids, sharding,
  program configs, layouts, and conservative L1/CB footprints before device
  execution. It covers the current P150A matmul/SDPA configuration plus paged
  fused cache update and fused QK RoPE, and exposes isolated compile-only
  validation with stable error classes.
- Step E retained split 8, the compressed performance recipe, official L1
  sharding, and the official SDPA 8x8 grid. LM-head DRAM concat advanced after
  a `1.61%` short-run gain, but matched 5/50 confirmations reduced that gain to
  `0.64%`, below the `1%` promotion threshold. The selected incumbent measured
  `28.436 tokens/s/user` (`85.91%` of target); the default config is unchanged
  and performance parity is still not claimed.
- The dedicated all-BF16 correctness recipe passes the Hugging Face reference
  gates at depths `1,2,4,32`; full-depth logits PCC is `0.99891` and the
  minimum sampled hidden/KV PCC is `0.99260` at a `0.99` threshold.
- The compressed performance recipe passes a separate fixed-corpus contract:
  over 500 teacher-forced target tokens, official TT-Transformers reaches
  `0.91/0.98` top-1/top-5 accuracy and all 32 Buddy users reach
  `0.908/0.98`, with `0.97` greedy agreement against official.
- The final P150A trace benchmark uses five warmups, 100 measured decode
  iterations, and three repetitions. The corresponding
  `v0.64.0-dev20251030` release reproduces `33.321 tokens/s/user`; Buddy
  reaches `33.938 tokens/s/user`, or `101.85%`, with `0.0139%` repetition CV.
  Goal 7 therefore reaches the M8 performance parity band.
- Goal 8 audits the optional LM-head/argmax, residual/RMSNorm, and
  SDPA/concat-heads custom-op candidates. None satisfies the conditional
  bottleneck threshold, so no custom kernel is added.
- The production greedy path follows TT-Transformers force-argmax: concatenate
  LM-head logits, untilize with multicore, then run multicore argmax. A composed
  shard-local/global reduction remains available for diagnostics but was slower
  on P150A. The retained force-argmax path now reaches M8 with full decode
  trace and persistent inputs. Same-commit and corresponding-release results
  remain separately labeled because their official runtimes differ.

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
| Layered autotune | Frozen contracts, schema-v2 space, and legality engine complete; semantic search orchestration pending |
| Steady decode benchmark | Goal 7: 33.938 tokens/s/user median, 29.463 ms mean |
| Official performance parity | M8 reached: 101.85% of corresponding release, CV 0.0139% |

## Quick Start

Set up TT-Metal and the Buddy toolchain as described in
[`docs/TenstorrentEnvironment.md`](../../docs/TenstorrentEnvironment.md).

Define paths used by the examples:

```bash
export BUDDY_REPO_ROOT=/wafer/zhuxinye/buddy-mlir
export BUDDY_BUILD="${BUDDY_BUILD:-$BUDDY_REPO_ROOT/build-tenstorrent}"
export TTNN_DIRECT_BUILD="$BUDDY_BUILD/models/llama31_ttnn_direct"
export MODEL=/wafer/share/models/Llama-3.1-8B-Instruct
export CONFIG="$BUDDY_REPO_ROOT/models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json"
export CORRECTNESS_CONFIG="$BUDDY_REPO_ROOT/models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32_correctness.json"
export PROGRAM="$TTNN_DIRECT_BUILD/program"
export REPORTS="$TTNN_DIRECT_BUILD/reports"
export RUNTIME_ARTIFACTS="$TTNN_DIRECT_BUILD/runtime_artifacts"
export OFFICIAL_PROMPTS=/wafer/zhuxinye/tt-metal-official-repro/models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json

mkdir -p "$PROGRAM" "$REPORTS" "$RUNTIME_ARTIFACTS"
export PYTHONPATH="$BUDDY_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TT_METAL_LOGS_PATH="$RUNTIME_ARTIFACTS"
cd "$RUNTIME_ARTIFACTS"
```

Running from `RUNTIME_ARTIFACTS` keeps TT-Metal inspector and watcher output
under the Buddy build tree instead of creating `generated/` in the source
tree.

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
  --out-dir "$REPORTS/validation/dryrun"
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
  --device p150a
```

Generation prints decoded text directly and does not write a report by
default. Add `--report-level summary --out "$REPORTS/generate/generate.json"`
for a compact JSON report. `--report-level full` additionally streams
per-step diagnostics to sibling `generate.steps.jsonl` and deduplicated
`generate.references.jsonl` files.

For an apples-to-apples batch32 input comparison with the TT-Metal demo, use
`--input-prompts "$OFFICIAL_PROMPTS" --instruct --prefill-len 256` instead of
`--prompt`. Prompt-file mode rejects truncation and records the source file
hash plus per-user token lengths in the report.

Profile the same workflow or post-prefill steady decode with the `profile`
command. Detailed examples and suite semantics are in
[docs/commands.md](docs/commands.md).

Run the official-style full decode trace benchmark with persistent inputs:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli profile \
  --mode decode-steady \
  --execution-mode trace \
  --runtime-input-mode persistent \
  --program-dir "$PROGRAM" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --input-prompts "$OFFICIAL_PROMPTS" \
  --instruct \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 256 \
  --cache-len 1024 \
  --warmup 5 \
  --iterations 50 \
  --device p150a \
  --after-prefill \
  --out "$REPORTS/performance/decode_trace.json"
```

Use `--execution-mode eager` as the matched control. Trace mode captures one
complete decode step and reports trace identity, capture/replay counts,
persistent inputs, and post-capture program-cache growth.

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

Report-producing commands emit JSON with a stable top-level `schema_version`,
`status`, and `passed` contract. Generate reports are optional.

- Summary generate reports record generated text and token IDs, KV-cache
  capacity, latency, throughput, and compact status. Full reports retain
  runtime ownership evidence and stream detailed decode-step references to
  JSONL so repeated operations do not accumulate in host memory.
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
  compressed performance recipe is intentionally judged by its separate
  fixed-corpus token-accuracy and greedy-agreement contract, which passes.
- Full decode trace reaches a post-cleanup three-run median of `33.997 t/s/u`
  on the matched P150A workload (`101.35%` of the corresponding-release local
  official median), with zero per-step input allocation and zero post-capture
  program compilation. Execution-graph parity removes 64 redundant RMSNorm
  layout calls while retaining one required initial tilize.
- Buddy prefill represents 32 users in one tensor, so the imported QKV and WO
  prefill configs disable TT-Transformers' single-sequence batch fusion while
  retaining the extracted grid and block geometry.
- The primary executable is the Python CLI; there is no integrated
  `buddy-cli` runner for this path.

## Development

Run the product test set:

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
