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
- `decode-step`
- `decode-loop-legacy`
- `depth-sweep`
- `generate-depth-sweep`
- `autotune`

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

Layered autotune dry-run or P150A execution:

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
  --candidates-dir "$AUTOTUNE/candidates" \
  --out "$AUTOTUNE/report.json"
```

Autotune checks device ownership before each unique candidate and runs every
hardware profile in a separate process so 8B parameter mappings and TTNN state
are released between candidates. Add `--dry-run` to build all candidate bundles
without opening a device.

## Legacy Compatibility

Historical low-level subcommands remain internally callable while refactoring
continues, but they are hidden from top-level help and are not documented as
user workflows. New automation should use the six product commands and
`diagnose --stage ...`.

## Tests

Run diagnostics and legacy compatibility coverage explicitly:

```bash
pytest models/llama_ttnn_direct/buddy_ttnn_direct/tests_diagnostics -q
```

These tests use fake or injected TTNN modules. Passing them does not establish
P150A numerical correctness or performance.
