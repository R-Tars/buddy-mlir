# TTNN Direct Diagnostics

Diagnostics preserve focused bring-up tools without adding them to the product
workflow. They are selected through one visible command:

The examples assume the build-tree variables from `docs/commands.md`,
including `PROGRAM`, `REPORTS`, `AUTOTUNE`, and `RUNTIME_ARTIFACTS`.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage STAGE \
  --out "$REPORTS/diagnostics/diagnose.json"
```

Available stages are:

- `mlp`
- `attention-primitive`
- `attention-layer`
- `prefill`
- `decode-shell`
- `decode-step`
- `decode-step-profile`
- `decode-loop-legacy`
- `depth-sweep`
- `generate-depth-sweep`
- `autotune`
- `autotune-profiler-audit`
- `benchmark-parity`
- `execution-graph-diff`
- `performance-correctness`
- `template-profile`

Use `--dry-run` whenever the selected stage supports it and no device should be
opened.

## Examples

MLP schema dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage mlp \
  --batch-size 32 \
  --hidden-size 4096 \
  --intermediate-size 14336 \
  --dry-run \
  --out "$REPORTS/diagnostics/mlp.json"
```

Attention primitive dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage attention-primitive \
  --primitive paged_scaled_dot_product_attention_decode \
  --batch-size 32 \
  --hidden-size 4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --dry-run \
  --out "$REPORTS/diagnostics/attention.json"
```

Depth sweep dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage depth-sweep \
  --program-dir "$PROGRAM" \
  --depths 1,2,4,8,16,32 \
  --batch-size 32 \
  --cache-len 1024 \
  --dry-run \
  --out "$REPORTS/diagnostics/depth_sweep.json"
```

Canonical semantic autotune dry-run or P150A execution:

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
  --repetitions 3 \
  --candidates-dir "$AUTOTUNE/candidates" \
  --out "$AUTOTUNE/report.json"
```

The command dispatches lazily to `buddy_ttnn_direct/autotune/campaign.py`.
The campaign keeps schema-v2 candidate identity and frozen precision/execution
contracts, measures legal operator proposals with the canonical active
scheduler, and uses hierarchical search followed by matched 5x100x3
confirmation. Hardware profile evaluations run in isolated subprocesses.
Non-dry runs also require `--official-tt-metal-root` for the candidate quality
gate; add `--dry-run` to serialize the search plan without opening a device or
requiring the official checkout.

Full decode execution-graph comparison:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage execution-graph-diff \
  --buddy-program "$PROGRAM" \
  --official-tt-metal-root "$OFFICIAL_TT_METAL_ROOT" \
  --official-python "$OFFICIAL_PYTHON" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --input-prompts "$OFFICIAL_PROMPTS" \
  --batch-size 32 \
  --prefill-len 256 \
  --cache-len 1024 \
  --out "$REPORTS/diagnostics/execution_graph_diff.json"
```

The diagnostic runs official and Buddy sequentially. It captures compile-time
trace graphs only when requested, normalizes release and current TTNN graph
formats, and compares operation counts plus tensor/layout/memory metadata.

Final corresponding-release and same-commit parity benchmark:

```bash
export OFFICIAL_TT_METAL_ROOT="$BUDDY_REPO_ROOT/thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal"
export OFFICIAL_PYTHON="$TTNN_DIRECT_BUILD/official-benchmark-venv/bin/python"
export OFFICIAL_RELEASE_CLEAN_ROOT="$RUNTIME_ARTIFACTS/official_release_clean_b76035f"
export OFFICIAL_RELEASE_RUNTIME_ROOT=/wafer/zhuxinye/tt-metal-official-repro
export OFFICIAL_RELEASE_PYTHON="$OFFICIAL_RELEASE_RUNTIME_ROOT/python_env_wheel_0_64_0/bin/python"
export OFFICIAL_RELEASE_PROMPTS="$OFFICIAL_RELEASE_CLEAN_ROOT/models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"

python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage benchmark-parity \
  --buddy-program "$PROGRAM" \
  --official-tt-metal-root "$OFFICIAL_TT_METAL_ROOT" \
  --official-python "$OFFICIAL_PYTHON" \
  --official-release-root "$OFFICIAL_RELEASE_CLEAN_ROOT" \
  --official-release-python "$OFFICIAL_RELEASE_PYTHON" \
  --official-release-runtime-root "$OFFICIAL_RELEASE_RUNTIME_ROOT" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --input-prompts "$OFFICIAL_RELEASE_PROMPTS" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 128 \
  --cache-len 1024 \
  --page-block-size 32 \
  --warmup 5 \
  --iterations 100 \
  --repetitions 3 \
  --out "$REPORTS/diagnostics/final_parity.json"
```

The release model root must be a clean checkout at the published commit. A
separate runtime root is allowed only for compiled `ttnn`/`tt_eager` artifacts
from that same commit; the report records source status, commits, paths, and
binary hashes. Passed runs are resumable only when their contract matches.

Performance-recipe token correctness:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli diagnose \
  --stage performance-correctness \
  --buddy-program "$PROGRAM" \
  --official-tt-metal-root "$OFFICIAL_TT_METAL_ROOT" \
  --official-python "$OFFICIAL_PYTHON" \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --layers 32 \
  --batch-size 32 \
  --prefill-len 512 \
  --cache-len 1024 \
  --accuracy-tokens 500 \
  --out "$REPORTS/diagnostics/performance_correctness.json"
```

This diagnostic is intentionally eager and teacher-forced. It does not change
the normal autoregressive generate path.

Phase-era validation orchestration was retired after
`build`/`generate`/`profile`/`validate`/`inspect`/`diagnose` became the
supported surfaces. Shared reports are limited to current product validation
contracts; each diagnose stage owns its stage-specific output.

## Tests

Run diagnostics coverage explicitly:

```bash
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
```

These tests use fake or injected TTNN modules. Passing them does not establish
P150A numerical correctness or performance.
