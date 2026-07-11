# TTNN Direct Evidence

This directory stores historical P150A evidence and reproducibility manifests.
These files are documentation artifacts only. Runtime code must not import
them.

The current baseline evidence is:

- `p150a_generate_depth_evidence_20260709.json`: prompt-conditioned generate
  passed for depths `1,2,4,8,16,32` with batch 32, prefill length 128, cache
  length 1024, and two generated tokens per user.
- `p150a_generate_profile_evidence_20260709.json`: full-depth `profile-generate`
  passed with about `0.1898 t/s/u`, `highest_passed=M1`, and
  `official_performance_parity_claimed=false`.
- `hf_correctness_reference_manifest_20260710.json`: CPU BF16 Hugging Face
  reference artifacts were captured for depths `1,2,4,32`, including hidden,
  logits, and sampled KV checkpoints. Q/K references use the Meta interleaved
  RoPE layout expected by TTNN. This manifest does not claim TTNN numerical
  correctness.
- `p150a_numerical_correctness_evidence_20260711.json`: the dedicated all-BF16
  correctness recipe passed the Step A checks at depths `1,2,4,32`, including
  full-depth logits, hidden-state, and sampled KV PCC at threshold `0.99`.
- `p150a_lm_head_argmax_evidence_20260711.json`: the Step B strategy study
  rejected composed local/global top-k for the default path and selected the
  TT-Transformers force-argmax sequence. The mixed two-token profile reached
  `1.3583 t/s/u` and milestone M2 while 1/32-layer correctness remained green.
- `p150a_decode_steady_evidence_20260711.json`: the Step C full-depth benchmark
  excluded five warmup iterations and measured 50 post-prefill decode steps.
  Mean throughput was `29.445 t/s/u` (`88.96%` of the recorded official target),
  reaching M5 while remaining below the greater-than-90% M6 threshold.
- `p150a_official_config_parity_evidence_20260711.json`: Step D replaced the
  hand-written seed with an extracted TT-Transformers performance profile. All
  55 compared fields across seven parity sections match, depth-1 generation and
  the full 32-layer steady benchmark pass, and the imported profile measures
  `28.627 t/s/u` (`86.49%` of target). Config parity is proven; performance
  parity is not claimed.
- `p150a_layered_autotune_evidence_20260711.json`: Step E replaced the old
  Cartesian decode-step search with four progressive post-prefill steady-decode
  levels. LM-head DRAM concat advanced after a `1.61%` short-run gain, but its
  matched 5/50 confirmation improved on the root incumbent by only `0.64%`,
  below the `1%` promotion threshold. The official config therefore remains the
  default; its final confirmation measured `28.436 t/s/u` (`85.91%` of target).

Keep runtime reference files under `buddy_ttnn_direct/reference/`. Historical
run evidence belongs here so refactors do not accidentally turn local evidence
records into runtime dependencies.
