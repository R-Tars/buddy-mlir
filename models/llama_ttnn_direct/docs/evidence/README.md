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

Keep runtime reference files under `buddy_ttnn_direct/reference/`. Historical
run evidence belongs here so refactors do not accidentally turn local evidence
records into runtime dependencies.
