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
