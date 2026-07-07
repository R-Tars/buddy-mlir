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

Final templates preserve generation semantics: `device_argmax_greedy` expands
to `argmax_or_sampling`, while `full_logits` stops after `split_lm_head`.

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

`run_decode.py` prints the expanded per-layer decode op sequence and final ops.
It also wraps the same generated decode bring-up gates used by the repository
CLI:

```bash
python /tmp/llama31_ttnn_program/run_decode.py \
  --mode smoke \
  --layers 1 \
  --device p150a \
  --out /tmp/decode_step_smoke_report.json

python /tmp/llama31_ttnn_program/run_decode.py \
  --mode profile \
  --layers 1 \
  --device p150a \
  --out /tmp/decode_step_profile_report.json

python /tmp/llama31_ttnn_program/run_decode.py \
  --mode validate-real \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --trace \
  --trace-iterations 10 \
  --require-trace \
  --require-full-depth \
  --require-program-runtime-shape \
  --require-batch32-decode-step \
  --min-tokens-per-second-per-user 1.0 \
  --baseline-reference tt_metal_official_llama31_8b_b32 \
  --min-baseline-ratio 0.1 \
  --decode-shell-pcc-threshold 0.99 \
  --require-decode-shell-numeric-reference \
  --out-dir /tmp/validate_ttnn_direct_real
```

Add `--dry-run` to any execution mode to write the report schema without
opening a TTNN device. Missing attention op wrappers still report the exact
template boundary that needs implementation.

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
  "weights_manifest": "weights_manifest.json",
  "runtime": {
    "buddy_cli_supported": false,
    "python_runner": "run_decode.py",
    "python_runner_supported": true,
    "runner_modes": [
      "inspect",
      "smoke",
      "profile",
      "decode-loop",
      "validate-real"
    ]
  }
}
```

The packaged `PACKAGE_README.md` points at the same runner modes exposed by the
generated bundle. Use `python run_decode.py --mode smoke`,
`--mode profile`, `--mode decode-loop`, or `--mode validate-real` from the
package directory for Python TTNN bring-up; `buddy-cli` dispatch remains out
of scope for this phase.

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

## Phase 14: Minimal Semantic Autotune

Phase 14 adds a small reusable search path that only enumerates LM-head split
count and generation template candidates:

```json
{
  "lm_head_split_count": [1, 2, 4, 8, 16],
  "generation_template": ["full_logits", "device_argmax_greedy"]
}
```

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  search \
  --semantic-json /tmp/llama_semantic.json \
  --base-config models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json \
  --space models/llama_ttnn_direct/buddy_ttnn_direct/search/spaces/lm_head_minimal.json \
  --metric latency_ms \
  --out /tmp/search_report.json \
  --dry-run
```

The command writes `/tmp/search_report.json` and a sibling
`/tmp/search_report_candidates/` directory. Each candidate contains:

```text
config.json
execution_plan.json
codegen/
```

Phase 14 does not run device measurements yet. Reports set `best` to `null`
and mark dry-run candidates as `dry_run_generated`.

## Phase 15: Custom Fused Region Hooks

Phase 15 reserves template names for future custom TTNN fused regions:

```json
{
  "mlp_template": "custom_buddy_fused_mlp_decode",
  "lm_head_template": "custom_buddy_lmhead_argmax_decode"
}
```

The template registry accepts these names and preserves them in the execution
plan. Python TTNN codegen deliberately fails with
`CustomFusedRegionNotImplemented` when either reserved template is selected, so
there is no silent fallback to the official MLP or LM-head templates.

The future op shapes are documented in:

```text
models/llama_ttnn_direct/buddy_ttnn_direct/custom_ops/README.md
```

## Phase 2 Kickoff: Direct Path Validation

The first follow-up from the Phase 1 review is a unified `validate-direct`
command. It runs the existing device-free checks in one place and now also
covers the follow-up no-device dry-run gates added during Phase 2 bring-up:

```text
import-llama
plan
diff-plan
emit-config
prepare-artifacts
build-program
py_compile generated Python
diff-official-config
tensorize-parameters --dry-run
smoke-decode-shell --dry-run
smoke-attention-primitive --dry-run
smoke-attention-layer --dry-run
smoke-single-layer-decode --dry-run
smoke-decode-step --dry-run --trace
profile-decode-step --dry-run --trace
search --dry-run
autotune-decode-step --dry-run --trace
package-program
```

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  validate-direct \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --config models/llama_ttnn_direct/buddy_ttnn_direct/configs/p150a_llama31_8b_b32.json \
  --out-dir /tmp/validate_ttnn_direct
```

The command writes `/tmp/validate_ttnn_direct/validation_report.json` plus the
intermediate semantic graph, execution plan, plan diff, parameter config,
offline manifests, generated program, official config diff, tensorization
dry-run report, decode/attention smoke dry-run reports, search/autotune
artifacts, and package directory. It does not import TTNN, open a Tenstorrent
device, load full tensor payloads, or run `materialize-parameters`; real
parameter materialization remains an explicit command because it needs local
weight files.

The final report includes a device-free `acceptance` block in addition to the
per-step results. The acceptance checks require the dry-run chain to finish,
the structural plan diff to stay clean, compiled generated Python files to
exist, no-device tensorization and smoke reports to stay in `dry_run` status,
search/autotune candidate counts to be positive, and the bundled default
decode-step autotune search space to vary every review knob. If these evidence
checks fail after the steps have run, `validate-direct` reports
`acceptance_failed` instead of `pass`.

For the real-weight path, use `validate-real-decode` after `build-program`.
This validation gate first writes an official-config parity diff for the
generated program, materializes selected HF safetensors, then runs the
attention-disabled decode shell, an independent attention primitive sweep, an
independent single-layer attention decode smoke, a real-weight single-layer
generated decode smoke, real-weight
`smoke-decode-step`, `profile-decode-step`, an integrated decode-depth sweep,
and optionally
`autotune-decode-step` against the existing generated program:

Preflight the exact final-acceptance arguments first. This writes a
`real_decode_preflight_report.json` without loading tensor payloads, opening a
TTNN device, or running runtime gates. The preflight report records the same
throughput, baseline-ratio, and decode-shell PCC thresholds that the final
validation command will use. It also records a `final_acceptance_plan` block
that names the requested acceptance scope, effective requirement flags, final
gate names, thresholds, and baseline source before runtime execution starts:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  validate-real-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --layers 32 \
  --cache-len 1024 \
  --device p150a \
  --prompt "Hello from TTNN Direct" \
  --tokenizer-path /path/to/Llama-3.1-8B-Instruct \
  --official-config models/llama_ttnn_direct/buddy_ttnn_direct/reference/official_p150a_llama31_8b_config_seed.json \
  --trace-iterations 10 \
  --require-model-end-to-end \
  --require-official-performance-parity \
  --min-tokens-per-second-per-user 1.0 \
  --baseline-reference tt_metal_official_llama31_8b_b32 \
  --min-baseline-ratio 0.1 \
  --decode-shell-pcc-threshold 0.99 \
  --preflight-only \
  --out-dir /tmp/validate_ttnn_direct_real
```

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  validate-real-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --layers 32 \
  --cache-len 1024 \
  --device p150a \
  --prompt "Hello from TTNN Direct" \
  --tokenizer-path /path/to/Llama-3.1-8B-Instruct \
  --official-config models/llama_ttnn_direct/buddy_ttnn_direct/reference/official_p150a_llama31_8b_config_seed.json \
  --trace-iterations 10 \
  --require-model-end-to-end \
  --require-official-performance-parity \
  --min-tokens-per-second-per-user 1.0 \
  --baseline-reference tt_metal_official_llama31_8b_b32 \
  --min-baseline-ratio 0.1 \
  --decode-shell-pcc-threshold 0.99 \
  --out-dir /tmp/validate_ttnn_direct_real
```

The report at
`/tmp/validate_ttnn_direct_real/real_decode_validation_report.json` links the
official config diff, materialization, attention-disabled decode shell,
standalone attention primitive sweep, standalone attention layer, single-layer
generated decode, full decode-step smoke/profile, decode-depth sweep, and
autotune subreports. The
decode shell gate runs before full attention, then the primitive sweep records
one report per official attention decode wrapper with status, error,
input/output shapes, expected output shapes, dtype/layout/memory config,
structural call coverage, TTNN version, and tt-metal git commit. The
standalone attention layer then records layer0 primitive latency and
tensor/memory-config conversion counts before the single-layer gate proves the
generated
`embedding -> layer0 attention -> layer0 MLP -> final norm -> LM-head` path
before the validation scales to the requested layer count. When a torch
reference can run, `--decode-shell-pcc-threshold` gates the shell final-hidden
PCC.
The integrated decode-depth sweep reuses `profile-decode-step` for the
review ladder. By default it covers `1`, `2`, and `4` where those depths are
not greater than the requested `--layers`, and always includes the requested
depth. Passing `--require-full-depth` also adds the generated program's full
layer count and gates full-depth coverage.
Use `--require-full-decode-step` for final functional acceptance. It enables
trace capture, full generated depth, generated batch/cache shape, the batch32
decode contract, and decode-shell numeric reference gates together.
The validation also writes
`/tmp/validate_ttnn_direct_real/real_decode_evidence_manifest.json`, a compact
evidence bundle index that records artifact existence, TTNN environment,
materialization/tensorization summaries, trace status, throughput summary,
autotune status, failed runtime steps, skipped follow-up steps, and failed
acceptance checks. If a runtime gate stops early, the manifest is still written
with `status=incomplete` so the failed bring-up attempt has an inspectable
evidence bundle. Use this manifest as the primary attachment for P150A
acceptance runs.
The validation report and evidence manifest also include a `reproducibility`
block with canonical `validate-real-decode` and preflight CLI commands plus a
machine-readable index of the key report/artifact paths.
When a prompt is provided, those canonical commands preserve `--prompt` and
`--tokenizer-path`; if `--tokenizer-path` is omitted, preflight records
`effective_tokenizer_path=<model-path>`, matching the runtime fallback.
They carry the same `final_acceptance_plan` block, so reviewers can distinguish
bring-up, full decode-step, and official performance-parity runs from the
artifact bundle alone.
The manifest also writes an `acceptance_gate_matrix` that maps the planned
final gate names to the actual acceptance-check status, making failed or
missing final gates visible without manually comparing report sections. For
official performance parity, the matrix records the official baseline
reference, the positive `--min-baseline-ratio` floor, and the observed ratio
gate as separate checks.
The manifest also includes a `model_end_to_end_readiness` block. This is
stricter than `full_decode_step_ready`: it remains false while runtime inputs
such as token ids, page tables, cache positions, paged KV cache, or rotary
tensors are supplied by the smoke/profile harness as synthetic tensors. Use it
to avoid confusing generated decode-step evidence with a real
tokenizer/prompt driven model decode loop. Even when prompt mode owns all
runtime tensors, the block remains false until a non-smoke decode loop records
`decode_loop_runtime_owned=true`.
The smoke/profile commands and generated runner accept `--prompt` plus
`--tokenizer-path` to build decode `token_ids` through the Hugging Face
tokenizer and derive paged decode metadata for page table and cache position.
They record `input_source=prompt_runtime`, remove the synthetic token-id,
page-table, and cache-position inputs from those reports, and add
`decode_runtime_state` evidence. In prompt mode the per-layer rotary tensors
are also emitted from `rotary_runtime_state` instead of being counted as
synthetic rotary inputs, and paged KV-cache tensors are emitted from
`kv_cache_runtime_state` instead of being counted as synthetic runtime inputs.
Use `prompt-decode-loop` or generated `run_decode.py --mode decode-loop` to
run the separate multi-step prompt loop. That loop initializes paged KV cache
once, advances page table/cache position and rotary runtime tensors per step,
feeds each generated token into the next step, and records
`decode_loop_runtime_owned=true` when the structural checks pass.
Use `--require-model-end-to-end` when final validation should fail unless that
block reports `model_end_to_end_ready=true`; the flag also enables the full
decode-step acceptance requirements. Preflight treats a non-empty prompt as
required for this scope, because the prompt decode loop is the evidence that
runtime token ids, page table/cache position, rotary tensors, and paged KV cache
are owned by the decode loop rather than by smoke/profile harnesses.
The manifest also includes an `acceptance_scope` block. `status=accepted`
means the requested gates passed, while
`acceptance_scope.full_decode_step_ready=true` is reserved for stricter runs
that prove full generated depth, generated batch/cache shape, batch32 decode
contract, trace capture/execute, and decode-shell numeric reference evidence.
`acceptance_scope.official_performance_parity_ready=true` additionally requires
model end-to-end readiness, official config match against a normalized external
parity reference, and a baseline-ratio gate, so a small bring-up run,
self-referenced generated config, or smoke/profile path with synthetic runtime
inputs cannot be mistaken for final end-to-end or performance-parity evidence.
Use `--require-official-performance-parity` for final performance acceptance.
It enables `--require-model-end-to-end`, `--require-full-decode-step`, and
`--require-official-config-match`, and it requires an explicit
`--baseline-reference` plus `--min-baseline-ratio` so final evidence is tied to
an auditable official baseline.
By default, the official-config diff is evidence-only: `diff_found` is
acceptable because the bundled official config is a seed reference with known
gaps. Use `--require-official-config-match` when a curated official parity
config is available and the real decode acceptance run should fail on any
config mismatch. Strong official-config acceptance also requires the
`--official-config` input to be a normalized parity JSON with a top-level
`parity_config`; passing the generated program's own `config.json` is still
allowed for exploratory diffs, but it fails the strong acceptance gate.
Use `--require-full-depth` and `--require-program-runtime-shape` separately
when debugging one strict requirement at a time. Prefer
`--require-full-decode-step` for final acceptance runs that must prove the
generated program's full layer count, configured batch/cache dimensions,
batch32 contract, trace, and shell numeric reference together.
Use `--require-batch32-decode-step` for the review Step 3 batch-32 decode
contract; this is also exposed by generated bundle `run_decode.py --mode
validate-real`.
Use `--baseline-reference tt_metal_official_llama31_8b_b32` to record the
observed throughput ratio against the TT-Metal official Llama 3.1 8B batch-32
baseline from `reference/performance_baselines.json`, and
`--min-baseline-ratio` to fail acceptance when the observed/baseline ratio
falls below the current phase's floor. `--baseline-tokens-per-second-per-user`
remains available for ad hoc baselines, but final evidence should prefer a
reference id so the source/model/batch are auditable.
When `--require-official-performance-parity` is enabled, validation also
requires that baseline reference to resolve to an official Llama 3.1 8B
batch-32 target; flow references, current Buddy baselines, and 3B official
targets remain useful comparison points but cannot satisfy final 8B parity.
The official parity floor must also be a positive `--min-baseline-ratio`; use
zero only for exploratory non-parity bring-up runs that record a baseline ratio
without claiming a final performance gate.
Use `--require-decode-shell-numeric-reference` for acceptance runs that should
fail instead of accepting a `numeric_reference.status=not_run` shell report.
When enabled, acceptance requires `numeric_reference.kind=torch_decode_shell`,
`status=passed`, `passed=true`, a final-hidden PCC greater than or equal to the
recorded `--decode-shell-pcc-threshold`, a matching threshold value, and no
numeric-reference failed checks.
Use `--skip-autotune` to stop after materialize/shell/smoke/profile during
bring-up, or `--dry-run` to write the schema without loading safetensors or
opening a TTNN device. `--require-official-performance-parity` rejects
`--skip-autotune`, because final performance evidence must include the
decode-step autotune leaderboard and best-candidate summary. With
`--require-trace`, `--require-decode-shell-numeric-reference`, and/or
`--min-tokens-per-second-per-user`, the final report includes an
`acceptance` block that checks materialized tensor count, real-weight
`hf_model` parameter sources, required materialized tensor paths, materialized
weight shapes against the generated config, resolved
synthetic/runtime input source/count evidence for token ids, page tables,
cache position, paged KV cache, and per-layer rotary tensors,
runtime input shape evidence for token/page/cache-position/KV tensors,
official config diff evidence including required parity-field coverage,
layer/batch/cache runtime shape, TTNN module
availability, TTNN version and tt-metal git commit evidence, successful
shell/attention-primitive/attention-layer/single-layer/smoke/profile runtime
status, attention primitive report completeness, decode-step tensor conversion
counts, token-or-logits output and paged KV-cache shapes, optional full-depth
and program runtime-shape requirements, shell
numeric/structural references, single-layer and decode-step
tensorization roles and memory config evidence, required tensorized decode
weight paths for single-layer/smoke/profile, tensorized key-weight physical
shapes derived from each transform, LM-head source metadata-reference and
sliced-read evidence, generated linear weight transform evidence,
embedding/RMSNorm weight shape-transform evidence, LM-head split tensor
transform evidence, decode-step structural references, empty structural
reference failed-check lists, generated observed op sequence coverage for the
shell/single-layer/smoke/profile paths, trace capture/execute status plus
requested execute iteration/sample-count evidence, trace execute latency
samples and derived trace throughput, measured profile latency, complete
profile attribution sections, LM-head split/argmax profile evidence, per-layer
attention/MLP timing records, bottleneck summary, positive profile throughput,
and, unless
`--skip-autotune` is used, real-weight decode-step autotune knob coverage and
a complete `output_kind_counts` summary for the generation templates under
test, every profiled autotune candidate's real-weight/profile/reference/shape
evidence, plus a best candidate with a passed structural reference. When a
throughput baseline is supplied, the evidence also records the
observed/baseline ratio, the baseline reference id when provided, and the
baseline's source/model/batch summary before gating it against the requested
floor.

## Phase 2 PR-B: Torch-Side Parameter Materialization

`materialize-parameters` turns a generated program's `weights_manifest.json`
into the nested host-side parameter object expected by generated `model.py`.
This step still does not create TTNN device tensors.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  materialize-parameters \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --backend torch \
  --layers 0 \
  --out /tmp/parameter_report.json
```

The first implementation supports:

```text
params.embedding.weight
params.layers[i].attention.{q_proj,k_proj,v_proj,o_proj}.weight
params.layers[i].attention.wqkv_packed.weight
params.layers[i].mlp.{gate_proj,up_proj,down_proj}.weight
params.layers[i].input_norm.weight
params.layers[i].post_attention_norm.weight
params.final_norm.weight
params.lm_head.weight
params.lm_head.splits[j].weight
```

Use `--layers 0` or another comma-separated layer list while bringing up real
models. The materializer dynamically imports `torch` and `safetensors` only
when the torch backend is used, reads individual safetensors keys through
`safe_open` when available, packs QKV on output-feature axis `0`, and slices
the LM-head on vocab axis `0`. When `safe_open.get_slice` is available,
`params.lm_head.weight` is kept as a metadata reference with shape/dtype while
each `params.lm_head.splits[j].weight` is read directly from its vocab range,
avoiding a full LM-head tensor load during layer-limited bring-up. The output
report records each tensor path's source key, shape, materialization mode, and
whether LM-head splits came from sliced tensor reads.
`validate-real-decode` consumes those shape records and fails acceptance when
embedding, QKV/O projection, packed QKV, MLP, RMSNorm, final norm, or LM-head
split shapes do not match the generated program config.

## Phase 2 PR-C: TTNN Tensorization Seed

`tensorize-parameters` converts materialized host-side parameters into TTNN
tensors for selected role groups. The default remains the conservative
`mlp,lm_head` seed used for early bring-up, and the command now also supports
`embedding`, `norm`, and `attention` role groups so the generated decode model
can receive TTNN tensors for embedding, RMSNorm/final norm, packed QKV,
attention output projection, MLP, and LM-head split weights. KV-cache tensors
are still created by the smoke paths rather than materialized from weights.
Real-decode validation checks the key tensorized physical shapes implied by
these transforms, such as linear weights `[1, 1, in_features, out_features]`,
embedding weights `[1, 1, vocab, hidden]`, and RMSNorm weights in 4D form.

Dry-run, no device or TTNN import:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  tensorize-parameters \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --roles mlp,lm_head \
  --layers 0 \
  --device p150a \
  --dry-run \
  --out /tmp/tensorize_report.json
```

Full generated-decode role dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  tensorize-parameters \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --roles embedding,norm,attention,mlp,lm_head \
  --layers 0 \
  --device p150a \
  --dry-run \
  --out /tmp/tensorize_decode_roles_report.json
```

Device mode first materializes torch parameters, then calls `ttnn.from_torch`
with role-based dtype/layout and conservative DRAM weight placement from the
emitted parameter config:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  tensorize-parameters \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --roles mlp,lm_head \
  --layers 0 \
  --device p150a \
  --out /tmp/tensorize_report.json
```

The report records each planned or converted tensor path, target dtype, layout,
memory config, resolved TTNN dtype/layout/memory config in device mode,
source shape, converted shape, and tensor transform when host tensors are
available. Generated linear weights are transposed from Hugging Face
`[out_features, in_features]` convention and reshaped to the official
TT-Transformers physical tensor shape `[1, 1, in_features, out_features]`
before TTNN conversion. This applies to packed QKV, attention output
projection, MLP gate/up/down, and LM-head split tensors. Embedding weights are
reshaped from
`[vocab, hidden]` to `[1, 1, vocab, hidden]`, and RMSNorm/final-norm weights
are reshaped from `[hidden]` to the 4D tile-aligned TT-Transformers convention
`[1, 1, hidden // 32, 32]` when possible, with a 4D fallback for synthetic
non-tile test shapes. Real decode validation records `transform_counts` plus
`transform_paths_by_kind`, requires all generated linear weight paths to
report `transpose_2d_to_4d`, requires embedding/RMSNorm paths to report their 4D
reshape transforms, and separately requires each LM-head split to report the
same tensorization evidence.

## Phase 2 PR-D: Decode Shell Without Attention

Generated `model.py` now routes embedding, per-layer RMSNorm, and final
RMSNorm through `TTNNCompatOps` instead of raising `NotImplementedError`.
Missing TTNN primitives fail explicitly with `UnsupportedTTNNOp`.

The first decode-shell smoke path runs the generated program with attention
disabled:

```text
token ids
embedding
per-layer RMSNorm + MLP + residual add
final RMSNorm
split LM-head + greedy argmax
```

Dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-decode-shell \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 1 \
  --disable-attention \
  --device p150a \
  --dry-run \
  --out /tmp/decode_shell_report.json
```

Device mode additionally needs the local model path so host parameters can be
materialized and converted for the shell. If token ids are not injected by a
test harness or provided through `--prompt`, the smoke synthesizes a row-major
TTNN `token_ids` tensor instead of passing a Python placeholder through
generated embedding:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-decode-shell \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 1 \
  --batch-size 32 \
  --disable-attention \
  --device p150a \
  --prompt "Hello from TTNN Direct" \
  --tokenizer-path /path/to/Llama-3.1-8B-Instruct \
  --pcc-threshold 0.99 \
  --out /tmp/decode_shell_report.json
```

This smoke command intentionally does not execute attention. Its purpose is to
validate that generated embedding/RMSNorm/MLP/LM-head code can be loaded and
composed before attention primitive bring-up.

Successful non-dry-run reports now include a `reference` block with
`kind=structural_shape_dtype`. It checks the expected layer count, disabled
attention status, per-layer hidden shape/dtype, and final token shape/dtype.
When host torch parameters are available, the same block also includes
`numeric_reference.kind=torch_decode_shell`, a final-hidden PCC check against a
torch reference, the PCC threshold used by the check, and an optional token
match after LM-head argmax. The torch
reference accepts either Hugging Face source weights or tensorized physical
TT-Transformers weights such as `[1, 1, in_features, out_features]`, so the
shell can validate the generated path after TTNN tensorization. If host
parameters or `ttnn.to_torch` output conversion are unavailable,
`numeric_reference.status` remains `not_run` with the reason recorded. Reports
also record `input_source`, `input_shapes.token_ids`, and
`runtime_input_tensor_count` so real device bring-up can distinguish injected
inputs from synthesized runtime inputs. When `--model-path` is used, the shell
report also records compact materialization and tensorization evidence,
including tensorized role groups, required tensor paths, transform counts, and
key 4D weight shapes.

## Phase 2 PR-E: Attention Primitive Smoke

`smoke-attention-primitive` validates one official decode attention primitive
at a time. It is meant for API signature, shape/layout, and memory-config
bring-up before attempting a full attention layer.

Supported primitives:

```text
qkv_linear
nlp_create_qkv_heads_decode
rotary_embedding_decode
paged_update_cache
paged_scaled_dot_product_attention_decode
nlp_concat_heads_decode
o_proj_linear
```

Dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-attention-primitive \
  --primitive paged_scaled_dot_product_attention_decode \
  --device p150a \
  --batch-size 32 \
  --hidden-size 4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --dry-run \
  --out /tmp/primitive_report.json
```

On a P150A system, drop `--dry-run` to execute the selected primitive with
synthetic tensors. The report records input shapes, expected/output shapes,
dtype seed, layout, memory-config placeholder, host-to-device tensor conversion
count, TTNN environment metadata (`ttnn_environment.version`, module file, and
detected tt-metal git commit when available), and explicit `api_mismatch`
errors when a wrapper cannot find the expected TTNN API. This command
deliberately does not compose a full attention layer. Successful non-dry-run
reports include `reference.kind=structural_shape_op_sequence`, which checks
observed output shapes against the primitive plan and verifies the expected
raw TTNN wrapper call sequence, while keeping
`numeric_reference.status=not_run`.
Decode attention activation shapes follow the official TT-Transformers 4D
physical convention: hidden/QKV tensors are `[1, 1, batch, hidden]`, head
tensors are `[1, batch, heads, head_dim]`, concat-head output is
`[1, 1, batch, hidden]`, and paged K/V cache tensors use
`[max_num_blocks, num_kv_heads, page_block_size, head_dim]`.
`validate-real-decode` also runs the full primitive list as a real acceptance
gate before `smoke-attention-layer`; the evidence manifest records the
primitive report directory, per-primitive statuses, TTNN environment, and
failed checks when a wrapper has an API mismatch or incomplete report schema.

## Phase 2 PR-F: One-Layer Attention Smoke

`smoke-attention-layer` composes the validated primitive wrappers into a single
synthetic attention decode layer:

```text
hidden
qkv_linear
nlp_create_qkv_heads_decode
rotary_embedding_decode
paged_update_cache.k
paged_update_cache.v
paged_scaled_dot_product_attention_decode
nlp_concat_heads_decode
o_proj_linear
```

Dry-run:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-attention-layer \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layer 0 \
  --device p150a \
  --batch-size 32 \
  --cache-len 1024 \
  --dry-run \
  --out /tmp/attention_layer_report.json
```

On a P150A system, drop `--dry-run` to execute the synthetic layer. The report
records per-primitive latency, input/output shapes, expected output shapes,
dtype, layout, memory config, explicit per-primitive `error` status,
host-to-device tensor conversion count,
memory-config conversion count, and TTNN environment metadata when available.
This smoke path is still independent from full generated decode execution so
individual attention issues stay easier to isolate. Successful non-dry-run reports include
`reference.kind=structural_shape_op_sequence` for the final attention output,
paged KV cache shapes, and expected raw TTNN wrapper call coverage. Each
primitive report also records `expected_output_shapes`, and the reference
checks include intermediate QKV, rotary, SDPA, concat-heads, and O-projection
shapes. These shape checks use the official decode physical activation
convention (`[1, 1, batch, hidden]` and `[1, batch, heads, head_dim]`) rather
than the older logical `[batch, seq, hidden]` summary. This is still not a
torch PCC check. `validate-real-decode` requires every attention-layer
primitive report to have `status=passed`, nonnegative latency, matching output
shapes, and `error=null`; a stale or masked per-primitive API error fails
acceptance even if the top-level layer status is later rewritten.

## Performance Step 1: Official Config Diff

`diff-official-config` compares a generated TTNN Direct `config.json` against
an official or official-like parity JSON. The comparison normalizes both sides
into these sections:

```text
dtype_recipe
compute_fidelity
program_config
memory_config
core_grid
lm_head
paged_attention
```

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  diff-official-config \
  --ours /tmp/llama31_ttnn_direct_program/config.json \
  --official models/llama_ttnn_direct/buddy_ttnn_direct/reference/official_p150a_llama31_8b_config_seed.json \
  --out /tmp/official_config_diff.json
```

The bundled official config is a hand-written seed reference, not a measured
TT-Transformers export. Replace `--official` with an imported or hand-curated
official JSON when available. The report records missing fields, mismatches,
extra fields, matching fields, per-section summaries, and a `gap_summary`
with the affected sections plus top missing/mismatched/extra paths. Its purpose
is to make parity gaps explicit before tuning dtype, compute fidelity, program
config, memory config, core grid, LM-head strategy, or paged attention config.
The report also records `required_field_coverage` for both the generated config
and the official reference. `validate-direct` and `validate-real-decode` require
the official/reference side to cover every required parity field, so an
incomplete seed cannot be used as strong parity evidence. When
`--require-official-config-match` or `--require-official-performance-parity` is
enabled, the validation gate additionally requires the official side to have
`source_format=normalized_parity_config`; a generated TTNN Direct config can
match itself structurally, but that self-reference is not accepted as official
parity proof.

## Performance Step 2: Generated Single-Layer Decode Smoke

`smoke-single-layer-decode` composes the generated decode program into a
single-layer execution path:

```text
embedding -> layer0 attention -> residual add -> layer0 MLP -> residual add
-> final norm -> split LM-head -> argmax
```

The non-dry-run path uses synthetic TTNN tensors for parameters, token ids,
page table, cache position, and KV cache. It is meant to validate generated
`decode_step()` control flow and TTNN API/shape composition before loading real
Llama weights or attempting full 32-layer decode.

Example:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-single-layer-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --out /tmp/single_layer_decode_report.json
```

Dry-run mode writes the same report schema without importing TTNN or opening a
device:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-single-layer-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --batch-size 32 \
  --cache-len 1024 \
  --dry-run \
  --out /tmp/single_layer_decode_report.json
```

The report records the expanded generated op sequence, synthetic input and
parameter shapes, expected intermediate shapes, output shapes, tensor
conversion count, TTNN environment metadata, and explicit `api_mismatch` /
`no_device` status when a required TTNN op or device is unavailable.
Expected intermediate attention shapes in the report use official decode
physical axes (`[1, 1, batch, hidden]` and `[1, batch, heads, head_dim]`);
token input/output contract fields remain logical batch-facing shapes.

When a local HF model directory is available, add `--model-path` in device mode
to materialize real weights through `materialize-parameters` and tensorize the
generated decode roles through `tensorize-parameters`. By default token ids,
page table, cache position, paged KV cache, and rotary matrices remain
synthetic for this bring-up step. Add `--prompt` and `--tokenizer-path` to
source token ids from the tokenizer and page table/cache position from the
decode runtime state. Prompt mode also creates per-layer rotary tensors from
`rotary_runtime_state` and paged KV cache tensors from
`kv_cache_runtime_state`:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-single-layer-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --prompt "Hello from TTNN Direct" \
  --tokenizer-path /path/to/Llama-3.1-8B-Instruct \
  --out /tmp/single_layer_decode_real_weights_report.json
```

The report sets `parameter_source` to `hf_model` for this path and
`input_source` to `synthetic` or `prompt_runtime`. It also includes a compact
`parameter_setup` summary with materialized layer ids,
materialized/tensorized tensor counts, tensorized role groups, complete
tensorized weight paths, dtype/layout/memory-config counts, key tensor
dtype/layout/memory-config records, prompt tokenization metadata when present,
decode runtime state metadata for page table/cache position when present, and
rotary/KV-cache runtime-state metadata when present, plus the number of any
remaining synthetic runtime tensors added around the real weights.

## Performance Step 2b: Generated Decode Layer Stack Smoke

`smoke-decode-step` extends the generated decode smoke path from one layer to a
configurable layer stack. By default it uses synthetic TTNN tensors for both
parameters and runtime inputs, but it can also take `--model-path` to
materialize/tensorize real generated decode weights. Add `--prompt` and
`--tokenizer-path` to use tokenizer-owned token ids plus runtime-owned page
table/cache position, rotary tensors, and paged KV cache tensors. The owning
multi-step decode loop is handled by `prompt-decode-loop`; this smoke command
still calls a single generated `decode_step()` with
`config.num_layers = --layers`.

Use it to walk the review plan from 2 layers to 4 layers and finally the full
generated layer count:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-decode-step \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 2 \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --out /tmp/decode_step_2l_report.json
```

For inspection without TTNN:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-decode-step \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 4 \
  --batch-size 32 \
  --cache-len 1024 \
  --dry-run \
  --out /tmp/decode_step_4l_report.json
```

The report records the repeated generated op sequence, layer count, planned
per-layer parameter shapes, expected output shapes, tensor conversion count,
parameter/input source, and per-layer KV cache output shapes. Successful
non-dry-run reports also include a `reference` block with structural
shape/dtype checks for token and KV-cache outputs, plus observed fake-op
sequences when the injected test TTNN module exposes them. This remains a
functional-path smoke; loading real weights does not yet claim numeric
correctness or official performance parity.

`prompt-decode-loop` records the first non-smoke loop ownership evidence for
model end-to-end readiness. It requires a prompt and tokenizer, runs multiple
generated decode steps, keeps the KV cache returned by one step for the next
step, and rebuilds runtime page table/cache position plus rotary tensors as
the cache position advances:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  prompt-decode-loop \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --prompt "Hello from TTNN Direct" \
  --decode-steps 2 \
  --layers 1 \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --out /tmp/prompt_decode_loop_report.json
```

`validate-real-decode` runs this step automatically when `--prompt` is
provided. Without a prompt it is marked skipped; with
`--require-model-end-to-end`, readiness also requires this loop to report
`decode_loop_runtime_owned=true`. The `--preflight-only` path fails early for
`--require-model-end-to-end` if no non-empty prompt is present, and the
reproducibility block preserves prompt/tokenizer arguments for the final run.

`decode-depth-sweep` automates the same bring-up ladder and writes a single
summary report while preserving each depth's `profile-decode-step` report:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  decode-depth-sweep \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --depths 1,2,4,full \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --out /tmp/decode_depth_sweep_report.json
```

When `--depths` is omitted, the default is `1,2,4,full`, clipped to the
generated program's layer count. The report records per-depth status, profile
report paths, latency, section latency, full per-layer timing records, LM-head
split/argmax profile evidence, throughput, output shapes, trace status,
reference status, and acceptance checks for increasing unique depths,
full-depth coverage, per-depth pass status, layer profile counts, profile
breakdown completeness, and throughput availability. Use `--dry-run` to
generate the same schema without opening a TTNN device.
When invoked through `validate-real-decode`, full-depth coverage is required
only when that command is run with `--require-full-depth`; otherwise the sweep
is scoped to the requested validation depth so small bring-up runs stay
lightweight. The real-decode acceptance gate also validates every depth record,
not just the sweep summary: each profiled depth must match the requested
batch/cache shape, use HF parameters and expose runtime input ownership for
token ids `[B, 1]`, page table `[B, page_count]`, cache position `[B]`, and
paged K/V cache `[max_num_blocks, num_kv_heads, page_block_size, head_dim]`
input shapes, account for `3 + 2 * depth` synthetic runtime-input tensors and
`3 * depth` synthetic rotary tensors in synthetic mode, or zero synthetic
runtime-input tensors plus prompt/runtime-state/rotary-runtime/KV-runtime
counts in `prompt_runtime` mode, expose layer-profile ids for `[0..depth)`, pass the
reference checks, report measured throughput, include decode output/KV-cache
shapes, include complete section latency fields, per-layer attention/MLP
timing records, LM-head split/argmax profile evidence, and complete bottleneck
timing sections. When `--require-trace` is used, each depth record must also
show
`captured_and_executed` trace status with the requested iteration count.

## Performance Step 3: Batch32 Decode-Step Contract Gate

`validate-real-decode` now records a `decode_step_contract` block for the
generated decode path. The block makes the review Step 3 contract explicit:
token input `[B, 1]`, decode `seq_len = 1`, paged KV-cache metadata, page
table shape, cache-position shape, per-layer KV-cache shape, and whether the
generated output is a token or retained logits. For paged attention, the
contract distinguishes logical cache length from the physical TT-Transformers
cache tensor shape: `page_count = ceil(cache_len / page_block_size)`,
`max_num_blocks = batch * page_count`, and each K/V cache tensor is planned as
`[max_num_blocks, num_kv_heads, page_block_size, head_dim]`.

For the P150A Llama 3.1 8B batch32 target, run the real validation without a
small-batch override and require the batch32 contract:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  validate-real-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --layers 1 \
  --cache-len 1024 \
  --device p150a \
  --skip-autotune \
  --require-program-runtime-shape \
  --require-batch32-decode-step \
  --out-dir /tmp/validate_real_decode_b32
```

The acceptance report always checks decode `seq_len = 1`, paged KV-cache
status, page-table/cache-position/KV-cache shapes, token ids `[B, 1]`,
the exact synthetic runtime input count, the exact synthetic rotary tensor
count, and token-or-logits output kind and shape.
`--require-batch32-decode-step` adds an explicit batch-size-32 gate while still
allowing small-batch smoke tests when the flag is omitted.

## Performance Step 4: Decode-Step Trace Smoke

`smoke-decode-step` can also exercise TTNN trace capture and execution for the
generated decode path:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-decode-step \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 2 \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --trace \
  --trace-iterations 10 \
  --out /tmp/decode_step_trace_report.json
```

When TTNN exposes `begin_trace_capture`, `end_trace_capture`, `execute_trace`,
and optionally `release_trace`, the smoke captures one generated decode step
and executes the captured trace `--trace-iterations` times. If trace APIs are
unavailable or capture fails, the report records an explicit fallback status
instead of silently pretending trace was used.

## Performance Step 5: Decode-Step Bottleneck Profile

`profile-decode-step` runs the generated decode path with synthetic TTNN
tensors, or with real generated decode weights when `--model-path` is provided,
and records section-level latency for bottleneck attribution:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  profile-decode-step \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 2 \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --trace \
  --trace-iterations 10 \
  --out /tmp/decode_step_profile_report.json
```

The report includes tensor conversion time, embedding time, per-layer
attention time, per-layer MLP time, final norm time, LM-head split/concat time,
argmax time, host-copy time, trace execute time, and a `bottleneck_summary`
that identifies the largest measured section. It also records a
`throughput_summary` with decode-step latency, aggregate tokens/sec, and
tokens/sec/user for the one-token-per-user decode step; when trace execution
samples are available, trace-execute throughput is reported separately. LM-head
profiling mirrors the generated split LM-head code but separates device argmax
into its own timed section. When the generated config retains full logits, the
profile report records `output_kind=logits`, checks logits shape
`[batch, seq_len, vocab]`, and marks argmax as skipped. Non-dry-run profile
reports reuse the generated decode-step
`reference.kind=structural_shape_dtype` checks, so output or KV-cache shape
mismatches are reported as `reference_mismatch` instead of being treated as
valid latency measurements.
`validate-real-decode` copies the profile throughput and bottleneck data into
the evidence manifest's `performance_gap_summary`, including observed/baseline
t/s/u, required speedup, min-ratio shortfall, and the largest profile section.

## Performance Step 6: Minimal Decode-Step Autotune

`autotune-decode-step` evaluates a small generated decode-step search space
using `profile-decode-step` as the measurement backend:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  autotune-decode-step \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --space models/llama_ttnn_direct/buddy_ttnn_direct/search/spaces/decode_step_minimal.json \
  --layers 2 \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --metric tokens_per_second_per_user \
  --trace \
  --trace-iterations 10 \
  --out /tmp/decode_step_autotune_report.json
```

The bundled minimal space covers LM-head split count, device argmax versus full
logits, MLP intermediate dtype, attention SDPA output memory config, and concat
heads output memory config. Use `--dry-run` to materialize candidate configs
without running TTNN profiles. The report records every candidate's knobs,
profile report path, metric, bottleneck summary, status/reference/trace
summaries, output kind, output shapes, LM-head split/argmax profile summary,
and the best candidate when measurements are available. It also writes a
ranked `leaderboard` that covers every candidate plus a compact
`best_candidate_summary` for the winning candidate, so P150A evidence can be
reviewed without opening every nested profile report. `latency_ms` is
minimized; `tokens_per_second_per_user` and `aggregate_tokens_per_second` are
maximized. Top-level `status_counts`, `reference_status_counts`,
`trace_status_counts`, and `output_kind_counts` make failed, skipped, token,
or full-logits candidate classes visible without opening every nested profile
report. `knob_coverage` also records `varied_knobs`,
`missing_varied_knobs`, and `all_knobs_varied`; the bundled default
`decode_step_minimal.json` validation gate requires all five review knobs to
vary at least once. Candidates whose profile report does not pass the
structural reference gate are not considered for `best`. In
`validate-real-decode`, autotune acceptance also checks every candidate
summary, not just `best`: each candidate must be profiled with HF parameters,
pass its structural reference, report a valid output kind and decode/KV-cache
shape summary, include LM-head and bottleneck profile evidence, and, when
trace is required, show `captured_and_executed` trace status. The same
acceptance gate requires the leaderboard to cover all candidates, rank the
selected best candidate first, and preserve measured throughput, bottleneck,
and LM-head summaries for the winning candidate.

Add `--model-path /path/to/Llama-3.1-8B-Instruct` in device mode to pass real
HF weights through each candidate's `profile-decode-step` run. Candidate
directories copy the generated program metadata needed by real-weight profile
(`semantic_graph.json`, `weights_manifest.json`, and `execution_plan.json`),
while token ids, page table, and cache position can be runtime-owned when
`--prompt` is supplied; rotary tensors and paged KV cache tensors are also
runtime-owned in prompt mode. Candidate records forward the
`parameter_source` and compact `parameter_setup` summary, including
tensorization dtype/layout/memory-config evidence, from their profile report.
