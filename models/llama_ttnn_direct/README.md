# Buddy-TTNN Direct

This directory hosts the additive Buddy-TTNN Direct path. The goal is to
generate official-style TTNN programs from a Buddy LLM semantic graph, without
replacing the existing `models/llama31_tt` TTIR baseline.

## Phase 1: Llama Semantic Graph

Phase 1 imports Hugging Face Llama configuration and weight metadata into a
dataclass-based semantic graph. It does not import TTNN, lower to TTIR, run a
device, or load full tensor payloads.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  import-llama \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --mode decode \
  --batch-size 32 \
  --seq-len 1 \
  --max-cache-len 1024 \
  --dry-run \
  --out /tmp/llama_semantic.json
```

The generated JSON records model dimensions, decode settings, all decoder
layers, attention projection weight names, MLP projection weight names, RMSNorm
weight names, and the LM-head weight name.

The importer reads local `config.json` directly. If present, it also reads
`model.safetensors.index.json`, `pytorch_model.bin.index.json`, or safetensors
file metadata for state-dict keys. If metadata is missing, canonical HF Llama
weight names are inferred from the config so dry-run CI can still exercise the
path with a tiny fake model directory.

## Phase 2: Template Plan

Phase 2 maps each semantic graph node to an official-style TTNN template name.
It still does not import TTNN, require a Tenstorrent device, or load full
weights.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  plan \
  --semantic-json /tmp/llama_semantic.json \
  --config models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json \
  --out /tmp/llama_ttnn_plan.json
```

The generated plan has a deterministic per-layer template sequence:

```text
rmsnorm
official_paged_attention_decode
residual_add
rmsnorm
official_gated_mlp_decode
residual_add
```

The final sequence is:

```text
rmsnorm
official_split_lm_head
device_argmax_greedy
```

## Phase 3: Python TTNN Skeleton

Phase 3 turns an execution plan into a readable Python TTNN skeleton. The
generated program imports `ttnn` and exposes the decode dataflow. In Phase 3
the template bodies are intentionally emitted as TODO stubs; later phases fill
selected templates with official-like TTNN calls.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  codegen-python \
  --plan-json /tmp/llama_ttnn_plan.json \
  --out-dir /tmp/buddy_ttnn_codegen
```

Dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  codegen-python \
  --plan-json /tmp/llama_ttnn_plan.json \
  --out-dir /tmp/buddy_ttnn_codegen \
  --dry-run
```

Generated files:

```text
/tmp/buddy_ttnn_codegen/model.py
/tmp/buddy_ttnn_codegen/config.json
/tmp/buddy_ttnn_codegen/plan.json
/tmp/buddy_ttnn_codegen/README.md
```

Validate the generated Python syntax with:

```bash
python -m py_compile /tmp/buddy_ttnn_codegen/model.py
```

## Phase 4: Parameter Metadata

Phase 4 emits dtype, layout, packing, split, and KV-cache metadata for Llama
weights. It does not read or convert tensor payloads.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  emit-config \
  --semantic-json /tmp/llama_semantic.json \
  --lm-head-split-count 8 \
  --kv-page-block-size 32 \
  --out /tmp/llama_parameter_config.json
```

Dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  emit-config \
  --semantic-json /tmp/llama_semantic.json \
  --out /tmp/llama_parameter_config.json \
  --dry-run
```

The seed recipe is:

```text
attention q/k/v/o weights -> bfloat8_b, tile
MLP gate/up weights        -> bfloat4_b, tile, gate_up_group
MLP down weights           -> bfloat8_b, tile
RMSNorm weights            -> bfloat16, row_major
embedding weights          -> bfloat16, row_major
LM-head weights            -> bfloat8_b, tile, vocab_split
activations                -> bfloat16
KV cache                   -> bfloat8_b, paged
```

## Phase 16: Official Plan Diff

Phase 16 compares a Buddy-TTNN Direct execution plan against a hand-written
official-like Llama decode template. The diff is structural only; it does not
run TTNN or measure performance.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  diff-plan \
  --ours /tmp/llama_ttnn_plan.json \
  --official-template models/llama_ttnn_direct/buddy_ttnn_direct/reference/official_llama31_decode_template.json \
  --out /tmp/plan_diff.json
```

The high-level plan templates are expanded before comparison. For example,
`official_paged_attention_decode` expands to:

```text
linear.qkv_packed
nlp_create_qkv_heads_decode
rotary_embedding_decode
paged_update_cache
paged_scaled_dot_product_attention_decode
nlp_concat_heads_decode
linear.o_proj
```

The output records `missing_ops`, `extra_ops`, and `order_mismatch`.

## Phase 5: Gated MLP TTNN Template

Phase 5 replaces the generated `mlp_decode()` stub with real TTNN call
structure:

```text
gate = ttnn.linear(...)
up   = ttnn.linear(...)
mid  = ttnn.mul(gate, up, input_tensor_a_activations=[SILU])
out  = ttnn.linear(mid, ...)
```

The generated model routes these calls through `TTNNCompatOps`, a small wrapper
around `ttnn.linear`, `ttnn.mul`, and `ttnn.add`. This keeps API-version
adaptation localized and lets tests mock TTNN without a device.

At the end of Phase 5, attention, embedding, RMSNorm, and LM-head codegen
remain explicit `NotImplementedError` boundaries.

## Phase 6: Split LM-head + Greedy Argmax

Phase 6 replaces the generated `lm_head_argmax()` stub with conservative TTNN
call structure:

```text
for each vocab shard:
  logits_i = ttnn.linear(hidden, lm_head_split_i, ...)
logits = ttnn.concat(shard_logits, dim=-1)
token  = ttnn.argmax(logits, dim=-1)
```

The generated `config.json` now records `lm_head.split_count`,
`lm_head.splits[*].vocab_start`, and `lm_head.splits[*].vocab_end`. Changing
`lm_head_split_count` to 1, 2, or 8 changes both generated config metadata and
the generated model constant.

This phase still materializes full logits before argmax. It does not add a
custom fused local/global argmax region.

## Phase 7: AttentionDecode Skeleton + Official Op Names

Phase 7 replaces the generated `attention_decode()` stub with an official-like
TTNN decode sequence:

```text
linear.qkv_packed
nlp_create_qkv_heads_decode
rotary_embedding_decode
paged_update_cache
paged_scaled_dot_product_attention_decode
nlp_concat_heads_decode
linear.o_proj
```

Experimental and transformer TTNN calls go through `TTNNCompatOps` wrappers so
API-name differences stay localized. Rotary embedding and paged KV-cache update
remain explicit template boundaries in this phase; attention is not expected to
run end-to-end on hardware yet.

## Phase 8: Offline Artifact Manifests

Phase 8 adds `prepare-artifacts`, which reads model weight metadata and writes
offline manifests without requiring a Tenstorrent device or loading full tensor
payloads:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  prepare-artifacts \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --semantic-json /tmp/llama_semantic.json \
  --config /tmp/llama_parameter_config.json \
  --out-dir /tmp/buddy_ttnn_artifacts
```

Generated files:

```text
weights_manifest.json
packed_qkv_manifest.json
mlp_manifest.json
lm_head_splits_manifest.json
kv_cache_manifest.json
```

The first implementation parses safetensors headers directly and records lazy
manifest entries for packed QKV, MLP weights, LM-head vocab splits, and paged
KV-cache metadata. Tensor conversion and materialized TTNN files remain future
work.

## Phase 9: MLP Smoke Test

Phase 9 adds `smoke-mlp`, a focused TTNN smoke path for the generated gated MLP
template. It does not run full Llama.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-mlp \
  --device p150a \
  --batch-size 32 \
  --hidden-size 1024 \
  --intermediate-size 4096 \
  --dtype-seed bf16 \
  --dry-run \
  --out /tmp/mlp_smoke_report.json
```

Dry-run writes the report schema without opening a TTNN device. Without
`--dry-run`, the command attempts a small `linear, linear, mul_silu, linear`
TTNN run and compares against a torch reference by PCC. If no device is
available, it writes a failed report with:

```text
No TTNN device detected. Use --dry-run or run on P150A.
```

## Phase 10: Template Profiling Reports

Phase 10 adds `profile-template` for MLP profiling reports with warmup,
iteration count, optional trace mode, and dry-run schema output:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  profile-template \
  --template mlp_decode \
  --config /tmp/buddy_ttnn_codegen/config.json \
  --warmup 5 \
  --iterations 20 \
  --trace \
  --dry-run \
  --out /tmp/profile_mlp.json
```

The report records `latency_ms.mean/p50/p90`, the expected MLP op counts, and a
trace status. With TTNN hardware available, the command attempts eager MLP
profiling and uses TTNN trace capture/execute APIs when present. Without
hardware, `--dry-run` still writes the full JSON schema.

## Phase 11: Full Decode Program Builder

Phase 11 adds `build-program`, which wires semantic import, template planning,
Python TTNN codegen, and weight manifest generation into one decode bundle:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  build-program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --config models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json \
  --out-dir /tmp/llama31_ttnn_program
```

Generated files:

```text
model.py
config.json
semantic_graph.json
execution_plan.json
weights_manifest.json
run_decode.py
README.md
```

`run_decode.py --dry-run` prints the expanded per-layer decode op sequence and
final ops. Full TTNN decode execution remains intentionally disabled at this
phase, but missing attention op wrappers now report the exact template boundary
that needs implementation.

## Phase 12: Attention TTNN Op Wrappers

Phase 12 moves official decode attention primitives into
`templates/ttnn_ops.py`. The wrappers are thin TTNN API adapters for:

```text
ttnn.experimental.nlp_create_qkv_heads_decode
ttnn.experimental.rotary_embedding_llama
ttnn.experimental.paged_update_cache
ttnn.transformer.paged_scaled_dot_product_attention_decode
ttnn.experimental.nlp_concat_heads_decode
```

The generated `TTNNCompatOps` delegates attention calls to those wrappers. The
wrapper module does not import `ttnn` at module import time, so generated code
can still be imported with a fake or unavailable TTNN module for offline tests.
If the installed TTNN version lacks a required API, wrappers raise
`UnsupportedTTNNOp` with the official decode template op name and searched API
paths.

## Phase 13: Program Package Directory

Phase 13 packages the generated Python TTNN program and manifests into a
directory without changing the existing `llama31_tt_rax` flatbuffer package
path.

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  package-program \
  --program-dir /tmp/llama31_ttnn_program \
  --out-dir /tmp/llama31_ttnn_direct_package
```

Dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  package-program \
  --program-dir /tmp/llama31_ttnn_program \
  --out-dir /tmp/llama31_ttnn_direct_package \
  --dry-run
```

The package writes `manifest.json` with:

```json
{
  "backend": "tenstorrent-ttnn-direct",
  "program_type": "python-ttnn",
  "entrypoint": "model.py",
  "semantic_graph": "semantic_graph.json",
  "execution_plan": "execution_plan.json",
  "weights_manifest": "weights_manifest.json"
}
```

CMake exposes an additive target behind
`BUDDY_BUILD_LLAMA31_TTNN_DIRECT_MODEL=ON`:

```bash
cmake --build "$BUDDY_BUILD" --target llama31_ttnn_direct_program
```

The target produces:

```text
$BUDDY_BUILD/models/llama31_ttnn_direct/llama31_ttnn_direct_package/
```

By default the target uses a config-only Llama 3.1 8B model description so it
does not require weights or TTNN hardware. Set
`BUDDY_LLAMA31_TTNN_DIRECT_MODEL_PATH` to package from a local Hugging Face
model directory.
