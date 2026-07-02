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
  --min-tokens-per-second-per-user 1.0 \
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
    "runner_modes": ["inspect", "smoke", "profile", "validate-real"]
  }
}
```

The packaged `PACKAGE_README.md` points at the same runner modes exposed by the
generated bundle. Use `python run_decode.py --mode smoke/profile/validate-real`
from the package directory for Python TTNN bring-up; `buddy-cli` dispatch remains
out of scope for this phase.

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

For the real-weight path, use `validate-real-decode` after `build-program`.
This validation gate materializes selected HF safetensors, then runs
real-weight `smoke-decode-step`, `profile-decode-step`, and optionally
`autotune-decode-step` against the existing generated program:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  validate-real-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --layers 1 \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --trace \
  --trace-iterations 10 \
  --require-trace \
  --min-tokens-per-second-per-user 1.0 \
  --decode-shell-pcc-threshold 0.99 \
  --require-decode-shell-numeric-reference \
  --out-dir /tmp/validate_ttnn_direct_real
```

The report at
`/tmp/validate_ttnn_direct_real/real_decode_validation_report.json` links the
materialization, attention-disabled decode shell, smoke, profile, and autotune
subreports. The decode shell gate runs before full decode-step smoke/profile
so embedding/RMSNorm/MLP/LM-head correctness can fail early; when a torch
reference can run, `--decode-shell-pcc-threshold` gates the final-hidden PCC.
The validation also writes
`/tmp/validate_ttnn_direct_real/real_decode_evidence_manifest.json`, a compact
evidence bundle index that records artifact existence, TTNN environment,
materialization/tensorization summaries, trace status, throughput summary,
autotune status, and failed acceptance checks. Use this manifest as the
primary attachment for P150A acceptance runs.
Use `--require-decode-shell-numeric-reference` for acceptance runs that should
fail instead of accepting a `numeric_reference.status=not_run` shell report.
Use `--skip-autotune` to stop after materialize/shell/smoke/profile during
bring-up, or `--dry-run` to write the schema without loading safetensors or
opening a TTNN device. With `--require-trace`, `--require-decode-shell-numeric-reference`,
and/or `--min-tokens-per-second-per-user`, the final report includes an
`acceptance` block that checks materialized tensor count, real-weight
`hf_model` parameter sources, TTNN module availability, TTNN version and
tt-metal git commit evidence, decode-step tensor conversion counts, shell
numeric/structural references, decode-step tensorization roles and memory
config evidence, decode-step structural references, trace capture/execute
status, and profile throughput before marking the validation as accepted.

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
the LM-head on vocab axis `0`. The output report records each materialized
tensor's source key and shape.

## Phase 2 PR-C: TTNN Tensorization Seed

`tensorize-parameters` converts materialized host-side parameters into TTNN
tensors for selected role groups. The default remains the conservative
`mlp,lm_head` seed used for early bring-up, and the command now also supports
`embedding`, `norm`, and `attention` role groups so the generated decode model
can receive TTNN tensors for embedding, RMSNorm/final norm, packed QKV,
attention output projection, MLP, and LM-head split weights. KV-cache tensors
are still created by the smoke paths rather than materialized from weights.

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
memory config, resolved TTNN dtype/layout/memory config in device mode, and
shape when host tensors are available.

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
test harness, the smoke synthesizes a row-major TTNN `token_ids` tensor instead
of passing a Python placeholder through generated embedding:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-decode-shell \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --layers 1 \
  --disable-attention \
  --device p150a \
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
torch reference, and an optional token match after LM-head argmax. If host
parameters or `ttnn.to_torch` output conversion are unavailable,
`numeric_reference.status` remains `not_run` with the reason recorded. Reports
also record `input_source`, `input_shapes.token_ids`, and
`runtime_input_tensor_count` so real device bring-up can distinguish injected
inputs from synthesized runtime inputs.

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
reports include
`reference.kind=structural_shape`, which checks observed output shapes against
the primitive plan while keeping `numeric_reference.status=not_run`.

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
dtype, layout, memory config, host-to-device tensor conversion count,
memory-config conversion count, and TTNN environment metadata when available.
This smoke path is still independent from full generated decode execution so
individual attention issues stay easier to isolate. Successful non-dry-run reports include
`reference.kind=structural_shape` for the final attention output and paged KV
cache shapes; each primitive report also records `expected_output_shapes`, and
the reference checks include intermediate QKV, rotary, SDPA, concat-heads, and
O-projection shapes. This is still not a torch PCC check.

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
extra fields, matching fields, and per-section summaries. Its purpose is to
make parity gaps explicit before tuning dtype, compute fidelity, program
config, memory config, core grid, LM-head strategy, or paged attention config.

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

When a local HF model directory is available, add `--model-path` in device mode
to materialize real weights through `materialize-parameters` and tensorize the
generated decode roles through `tensorize-parameters`. Token ids, page table,
cache position, paged KV cache, and rotary matrices remain synthetic for this
bring-up step:

```bash
python -m models.llama_ttnn_direct.buddy_ttnn_direct.cli \
  smoke-single-layer-decode \
  --program-dir /tmp/llama31_ttnn_direct_program \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --batch-size 32 \
  --cache-len 1024 \
  --device p150a \
  --out /tmp/single_layer_decode_real_weights_report.json
```

The report sets `parameter_source` to `hf_model` for this path and
`input_source` to `synthetic`. It also includes a compact `parameter_setup`
summary with materialized layer ids, materialized/tensorized tensor counts,
tensorized role groups, dtype/layout/memory-config counts, key tensor
dtype/layout/memory-config records, and the number of synthetic
rotary/runtime-input tensors added around the real weights.

## Performance Step 2b: Generated Decode Layer Stack Smoke

`smoke-decode-step` extends the generated decode smoke path from one layer to a
configurable layer stack. By default it uses synthetic TTNN tensors for both
parameters and runtime inputs, but it can also take `--model-path` to
materialize/tensorize real generated decode weights while keeping token ids,
page table, cache position, rotary matrices, and per-layer paged KV cache
synthetic. It then calls generated `decode_step()` with
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
into its own timed section. Non-dry-run profile reports reuse the generated decode-step
`reference.kind=structural_shape_dtype` checks, so output or KV-cache shape
mismatches are reported as `reference_mismatch` instead of being treated as
valid latency measurements.

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
summaries, and the best candidate when measurements are available. `latency_ms`
is minimized; `tokens_per_second_per_user` and `aggregate_tokens_per_second`
are maximized. Top-level `status_counts`, `reference_status_counts`, and
`trace_status_counts` make failed or skipped candidate classes visible without
opening every nested profile report. Candidates whose profile report does not
pass the structural reference gate are not considered for `best`.

Add `--model-path /path/to/Llama-3.1-8B-Instruct` in device mode to pass real
HF weights through each candidate's `profile-decode-step` run. Candidate
directories copy the generated program metadata needed by real-weight profile
(`semantic_graph.json`, `weights_manifest.json`, and `execution_plan.json`),
while token ids, page table, cache position, rotary matrices, and paged KV
cache remain synthetic at this stage. Candidate records forward the
`parameter_source` and compact `parameter_setup` summary, including
tensorization dtype/layout/memory-config evidence, from their profile report.
