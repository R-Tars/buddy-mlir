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
generated program imports `ttnn` and exposes the decode dataflow, but template
bodies remain TODO stubs with `NotImplementedError`.

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
