# TTNN Direct Refactor Baseline

This baseline freezes the state before the product cleanup refactor described
in `/wafer/zhuxinye/推进文档4.md`.

## Functional Baseline

- Branch: `ttnn-direct-phase1`
- Latest pre-refactor evidence commit: `8776ca9`
- Target device: P150A
- Model: `/wafer/share/models/Llama-3.1-8B-Instruct`
- Prompt: `Hello from TTNN Direct`
- Batch size: 32
- Prefill length: 128
- Cache length: 1024
- Generated tokens per user: 2
- Full-depth prompt-conditioned `prefill -> decode -> text`: passed
- Depth ladder passed: `1,2,4,8,16,32`
- Full-depth required gate: passed
- KV cache source: `prefill`
- Model semantics: `prompt_conditioned_prefill_decode`
- Parameter tensorization count per generate: 1
- Parameter tensorization count per decode step: 0
- KV cache reinitialized per step: false

Evidence:

- `docs/evidence/p150a_generate_depth_evidence_20260709.json`
- `/wafer/zhuxinye/tmp/ttnn_direct_pr5_full_20260709_143626/generate_full_report.json`

## Performance Baseline

- Full-depth `profile-generate`: passed
- Observed throughput: `0.1897960392015765 t/s/u`
- Observed aggregate throughput: `6.073473254450448 tokens/s`
- Official reference target: `33.1 t/s/u`
- Ratio of official reference: `0.005734019311225876`
- Highest performance milestone passed: `M1`
- Next milestone: `M2`, batch32 decode tokens/s/user greater than 1
- Official performance parity claimed: false

Evidence:

- `docs/evidence/p150a_generate_profile_evidence_20260709.json`
- `/wafer/zhuxinye/tmp/ttnn_direct_pr7_profile_full_20260709_144258/generate_profile_report.json`

## Bottleneck Baseline

The first full-depth profile shows full-logits `argmax` as the dominant
bottleneck:

- Generate total latency: `10537.627699784935 ms`
- Argmax latency: `9348.338863346726 ms`
- Argmax fraction of generate latency: `0.8871388446886882`
- Runtime token handoff: `device_tensor_direct`
- Runtime host roundtrip present: false
- Host materialization is for reporting/detokenization only: true

The next performance step after the cleanup refactor should avoid full logits
materialization by implementing split LM-head local argmax plus global argmax
reduce, then measuring a decode-only steady-state benchmark.
