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

- Full-depth prompt-conditioned prefill, paged-KV decode, and text generation
  pass on P150A for Llama 3.1 8B at batch 32.
- The production decode path uses persistent device inputs, one full decode
  trace, context-aware SDPA, and device-to-device token handoff.
- The phase-1 frozen benchmark uses 5 warmups, 100 measured post-prefill decode
  iterations, and 3 repetitions. Median throughput is `35.580 tokens/s/user`,
  median p50 latency is `28.063 ms`, and CV is `0.0690%`.
- The all-BF16 full-depth gate passes 99 comparisons at PCC `>= 0.99`; its
  minimum PCC is `0.99260`. Full-depth functional generation also passes.
- The compressed performance recipe passes the fixed-corpus quality gate:
  official and Buddy top-1 accuracy are both `0.91`, Buddy top-5 is
  `0.982`, and greedy agreement with official is `0.964`.
- The current best generated config SHA-256 is
  `1ed82e5b0f2f2ce8444fa881ce725f62a37d6f556fa2cee493cda6dc7e5fde5d`.
  Historical campaigns and raw samples live in the ignored build tree.

See [docs/evidence/README.md](docs/evidence/README.md) for the evidence storage
contract and [docs/evidence/latest_summary.json](docs/evidence/latest_summary.json)
for the compact current baseline.

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
| Semantic autotune | Hierarchical search, strict matched A/B, one-model/three-workload generalization, and a hash-verified paper artifact with all required ablations are complete |
| Steady decode benchmark | Phase-1 baseline: 35.580 tokens/s/user median, 28.063 ms p50, CV 0.0690% |
| Official performance parity | M8 remains established; matched raw comparison reports stay in the build tree |

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

Autotune development includes bounded per-op enumeration, cross-op layout beam
search, and a hierarchical campaign orchestrator. The orchestrator evaluates
template groups, per-op Pareto frontiers, and layout mutations one group at a
time with a bounded beam; it never materializes the full Cartesian product.
Every device result is written atomically under its candidate fingerprint,
budget counters survive resume, and callback failures retain classified
reports. `best_config.json` is a template config accepted directly by
`buddy-ttnn-direct build`.

Final promotion uses alternating matched A/B order with exactly 5 warmup, 100
iterations, and 3 repetitions. Both arms must preserve the frozen runtime,
device, precision, semantic graph, model, weights, prompt, and measurement
identity; correctness and quality must pass, each arm must have CV at most
1.5%, and the challenger median must improve by at least 1%. On the Phase 8
P150A batch32/cache1024 campaign, an enumerated SDPA `q_chunk_size=32`
challenger measured `33.9103 t/s/u` median versus the incumbent's
`33.9425 t/s/u`, so the orchestrator correctly retained the incumbent.

`autotune/generalization.py` applies that same orchestrator to distinct model
shapes and rejects campaigns that change the template/op/layout group
structure, inject an official hand-tuned challenger, use Cartesian exhaustive
search, omit a directly buildable final config, or fail to cover either two
models or one model with three workloads. The Phase 9 P150A campaign used
Llama 3.1 8B batch32/prefill256 at cache lengths 512, 1024, and 2048. The
incumbent medians were `34.0405`, `33.9425`, and `34.0052 t/s/u`; independently
enumerated `q_chunk_size=32` challengers changed them by `-0.0833%`,
`-0.0948%`, and `-0.0090%`. Every arm met the 1.5% CV bound, all three
campaigns retained the incumbent under the 1% promotion threshold, and every
selected config rebuilt directly.

Phase 10 packages the complete experiment with `autotune/artifact.py`. The
builder validates and hashes every Phase 3-9 source report, derives search
time, enumeration/pruning, microbench and full-model counts, performance,
official ratio, and accuracy, then writes JSON, CSV, Markdown, source and
artifact manifests, and a rebuild script. The captured campaign records 456
search entities, 79 analytical rejections, 8 microbench experiments, 8
full-model profiles, and `775.54 s` of orchestrator-accounted device process
time. Its selected incumbent measures `33.9425 t/s/u`, or `101.866%` of the
corresponding release, while preserving the 500-token accuracy and full-depth
PCC gates. The required config/template/layout, transfer on/off, and pruning
on/off ablations are all present; rows that select an identical config are
explicitly marked as evidence reuse or analytical cost studies.

Layout reports preserve measured conversion costs, removed and retained
conversions, rejected sharding constraints, and the whole-layer
incumbent/challenger decision. For the Llama 3.1 8B GQA shape, the current TTNN
runtime requires interleaved SDPA output, so the SDPA-to-concat conversion is
retained rather than replaced by an illegal sharded producer output.

Additional references:

- [Architecture](docs/architecture.md)
- [Command reference](docs/commands.md)
- [Historical evidence](docs/evidence/README.md)
- [Diagnostics](buddy_ttnn_direct/diagnostics/README.md)
