#!/usr/bin/env bash
# ===- run_llama31_p150_chat.sh -----------------------------------------------
# Full Llama-3.1-8B-Instruct chat on P150A with 1024-token static cache.
# Pipeline:
#   1) lower prefill TTIR (--static-cache --max-cache-len 1024) -> TTNN -> FB
#   2) lower decode  TTIR (same)                                -> TTNN -> FB
#   3) prepare chat artifacts (weights + slot roles) for both phases
#   4) launch llama31_chat_run.py (interactive, reads one prompt from stdin)
#
# Use MAX_CACHE_LEN (default 1024) to control cache length.
# Set SKIP_LOWER=1 / SKIP_PREPARE=1 to rerun only the chat step with
# existing artifacts. Set MAX_NEW_TOKENS=N to cap generation length.
#
# Memory note: Llama-3.1-8B's bf16 weights are ~16 GB; the chat artifacts
# are loaded twice (prefill + decode contexts), so the script raises the
# virtual-memory soft limit to 95 GB before running anything.
# ===---------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

ulimit -v 95000000 || true

cd "${SCRIPT_DIR}"
export LLAMA31_MODEL_PATH="${LLAMA31_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-1024}"
SKIP_LOWER="${SKIP_LOWER:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
TTIR_OUT="${SCRIPT_DIR}/ttir_out_static"
CHAT_ART="${SCRIPT_DIR}/chat_artifacts"
mkdir -p "${TTIR_OUT}" "${CHAT_ART}"

SYS_DIR="/tmp/ttrt_sys_llama31"
if [[ ! -f "${SYS_DIR}/system_desc.ttsys" ]]; then
  echo "=== query system descriptor ==="
  python -m ttrt query --save-artifacts --artifact-dir "${SYS_DIR}" 2>&1 | tail -3
fi
SYS="${SYS_DIR}/system_desc.ttsys"

PREFILL_TTNN="${TTIR_OUT}/llama31_prefill_static.ttnn"
DECODE_TTNN="${TTIR_OUT}/llama31_decode_static.ttnn"

if [[ "${SKIP_LOWER}" != "1" ]]; then
  if [[ ! -f "${PREFILL_TTNN}" ]]; then
    echo "=== [1/4] Lower prefill TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-llama31-lower-ttir.py \
      --mode prefill --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --element-dtype bf16 \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [1b/4] prefill TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/llama31_ttir_prefill.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}" \
      -o "${TTIR_OUT}/llama31_prefill_static_ttnn.mlir"
    echo "=== [1c/4] prefill TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/llama31_prefill_static_ttnn.mlir" \
      -o "${PREFILL_TTNN}"
  else
    echo "=== [1/4] prefill flatbuffer already exists: ${PREFILL_TTNN} ==="
  fi

  if [[ ! -f "${DECODE_TTNN}" ]]; then
    echo "=== [2/4] Lower decode TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-llama31-lower-ttir.py \
      --mode decode --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --element-dtype bf16 \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [2b/4] decode TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/llama31_ttir_decode.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}" \
      -o "${TTIR_OUT}/llama31_decode_static_ttnn.mlir"
    echo "=== [2c/4] decode TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/llama31_decode_static_ttnn.mlir" \
      -o "${DECODE_TTNN}"
  else
    echo "=== [2/4] decode flatbuffer already exists: ${DECODE_TTNN} ==="
  fi
else
  echo "=== skipping lower/ttnn/flatbuffer (SKIP_LOWER=1) ==="
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  if [[ ! -f "${CHAT_ART}/prefill/slot_roles.json" \
        || ! -f "${CHAT_ART}/decode/slot_roles.json" ]]; then
    echo "=== [3/4] Prepare chat artifacts (weights + roles) ==="
    python llama31_chat_prepare.py \
      --max-cache-len "${MAX_CACHE_LEN}" \
      -o "${CHAT_ART}"
  else
    echo "=== [3/4] chat artifacts already exist: ${CHAT_ART} ==="
  fi
else
  echo "=== skipping chat prepare (SKIP_PREPARE=1) ==="
fi

echo "=== [4/4] Interactive chat on P150A ==="
EXTRA_ARGS=()
if [[ "${MAX_NEW_TOKENS}" != "0" ]]; then
  EXTRA_ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
python llama31_chat_run.py \
  --prefill-ttnn "${PREFILL_TTNN}" \
  --decode-ttnn "${DECODE_TTNN}" \
  --artifacts "${CHAT_ART}" \
  --max-cache-len "${MAX_CACHE_LEN}" \
  --ignore-system-desc \
  "${EXTRA_ARGS[@]}"
