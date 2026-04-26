# Buddy Llama-3.1-8B on P150A (end-to-end via TTIR)

Work-in-progress example that compiles
`meta-llama/Llama-3.1-8B-Instruct` (HF, bf16) through Buddy Compiler
to TTIR, then to TTNN flatbuffer, and runs prefill + greedy decode on
a Tenstorrent P150A (Blackhole).

Blueprint: `examples/BuddyDeepSeekR1` (already working end-to-end on
the same hardware). Every script here mirrors its DeepSeek counterpart
with `llama31` naming and Llama-3.1-specific sizes.

## Baseline to beat

Official `tt_transformers` stack on the same P150A, measured on
2026-04-24 (see `/wafer/zhuxinye/llama_tt_transformers/BASELINE.md`):

| Case | TTFT (ms) | Decode (t/s/u) |
|---|---|---|
| `batch-1 performance` | 65.5 | **23.2** |
| `ci-token-matching` (Instruct) | 134 | 22.6 |

## Milestones

- **P0** ✅ — lower prefill TTIR (seq=32, bf16) ⇒ MLIR on disk, `ttmlir-opt` clean.
- **P1** ✅ — prefill-only flatbuffer on P150A, logits match HF bf16 golden
  (PCC 0.9928, Top-1 match).
- **P2** ✅ — static-cache prefill+decode lowered and driven end-to-end
  with `llama31_chat_run.py` (prompt `"What is 2+2?"` → `"2 + 2 = 4."`
  + natural EOS; 128-token perf run produces a coherent 200-word
  description of the transformer architecture).
- **P3** ✅ — first decode t/s/u numbers captured (see below).
- **P4** — optimization (flash-attn fusion, grid tuning, quantization,
  keep-KV-on-device round-trip to close the ~127 ms/tok host gap).

## TTIR results (2026-04-24)

Single P150A, `--max-cache-len 1024`, bf16, prompt of 50 tokens,
`--ignore-eos --max-new-tokens 128`:

| Metric | Value |
|---|---|
| Prefill (1024 seq) | 1.11 s → **920 tok/s** |
| Decode wall steady (126 tokens in 23.23 s) | **5.42 tok/s** |
| Decode **device-only** steady (`rt.submit+wait`) | **17.31 tok/s** |
| Host round-trip overhead | 126.6 ms/tok |
| First decode step (JIT warmup) | 0.86 s wall, 0.73 s submit |

Compared to the official `tt_transformers` baseline (23.2 t/s/u):

- **Device compute** alone is at **75 %** of the official throughput on
  a completely unoptimized Buddy → TTIR → TTNN path (no
  flash-attention fusion, no prefetcher, no GQA fusion on P150A).
- The wall-clock gap (5.4 vs 23.2 t/s/u) is almost entirely
  host-side: each decode step spends ~127 ms outside `rt.submit+wait`.
  KV outputs are already kept device-resident (via `rt.to_layout`),
  so the remaining cost is the per-step `to_layout` + `deallocate`
  sync plus a 64-entry Python loop. P4 will batch these into a
  single runtime call (or reuse pre-allocated KV slots) to close
  the gap.

## Layout (after all phases)

```
import-llama31.py                         # HF -> Buddy graph sanity
buddy-llama31-lower-ttir.py               # --mode {prefill,decode} --static-cache ...
llama31_chat_prepare.py                   # weights + slot_roles.json
llama31_chat_run.py                       # ttrt.runtime driver (prefill + greedy decode)
export_golden_logits.py                   # HF bf16 reference
compare_golden_bundle.py                  # HF vs device comparison
run_llama31_p150_e2e.sh                   # prefill-only wrapper
run_llama31_p150_chat.sh                  # full chat wrapper
```

## P0 quick check

```
source /wafer/zhuxinye/miniconda3/etc/profile.d/conda.sh && conda activate tt-mlir
export BUDDY_BUILD=/wafer/zhuxinye/buddy-mlir/build
export TTMLIR_BUILD=/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20
export PYTHONPATH="$BUDDY_BUILD/python_packages:$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"
export PATH="$TTMLIR_BUILD/bin:$PATH"
export HF_HOME=/wafer/zhuxinye/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LLAMA31_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

python buddy-llama31-lower-ttir.py --mode prefill --seq 32 \
    --ttmlir-opt "$(command -v ttmlir-opt)"
```

Expected: `ttir_out/llama31_ttir_prefill.mlir` on disk, plus
`ttmlir-opt: OK` at the end.
