# TTNN Direct Evidence

This directory stores historical P150A evidence and reproducibility manifests.
These files are documentation artifacts only. Runtime code must not import
them.

Compact manifests remain checked in here. Their untracked raw programs,
reports, candidates, and HF reference arrays live under the active build tree:

```text
${BUDDY_BUILD}/models/llama31_ttnn_direct/evidence_archive/phase_history/
${BUDDY_BUILD}/models/llama31_ttnn_direct/references/hf/
```

The manifests currently record the concrete build path used for the captured
2026-07-09 through 2026-07-11 evidence. Cleaning the Buddy build tree also
removes those raw files, but does not remove these compact checked-in records.

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
- `p150a_matched_parity_evidence_20260714.json`: Goal 0 separates the published
  release reference from the same-commit comparison. The corresponding
  `v0.64.0-dev20251030` release reproduces `33.546 t/s/u`, while the current
  same-commit official greedy median is `21.657 t/s/u` and Buddy is
  `27.342 t/s/u`. Buddy is `126.25%` of the same-commit official result but
  `81.51%` of the cross-version published release; only the former is a matched
  parity ratio. All profiles pass the three-run `CV <= 1.5%` gate.
- `p150a_goal1_dependency_inversion_evidence_20260714.json`: Goal 1 removes
  product runtime dependencies on smoke/legacy modules and the reports-to-
  diagnostics dependency. Its matched eager 5/50 regression run measures a
  `27.610 t/s/u` median across three repetitions (`CV 1.09%`), a `0.98%`
  increase over the pre-change median, so the no-more-than-1% regression gate
  passes.

Keep runtime reference files under `buddy_ttnn_direct/reference/`. Historical
run evidence belongs here so refactors do not accidentally turn local evidence
records into runtime dependencies.
