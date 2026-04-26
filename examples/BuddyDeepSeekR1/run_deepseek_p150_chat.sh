#!/usr/bin/env bash
# ===- run_deepseek_p150_chat.sh ----------------------------------------------
# Full interactive DeepSeek-R1 chat on P150A with 1024-token static cache.
# Pipeline:
#   1) lower prefill TTIR (--static-cache --max-cache-len 1024) -> TTNN -> FB
#   2) lower decode  TTIR (same)                                -> TTNN -> FB
#   3) prepare chat artifacts (weights + slot roles) for both phases
#   4) launch deepseek_chat_run.py (interactive)
#
# Use MAX_CACHE_LEN (default 1024) / PHASES (prefill,decode) to control steps.
# Set SKIP_LOWER=1 / SKIP_PREPARE=1 to rerun only the interactive step with
# existing artifacts.
# ===---------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "/wafer/zhuxinye/miniconda3/etc/profile.d/conda.sh"
conda activate tt-mlir

export TTMLIR_BUILD="${TTMLIR_BUILD:-/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20}"
export BUDDY_BUILD="${BUDDY_BUILD:-/wafer/zhuxinye/buddy-mlir/build}"
export TT_METAL_HOME="${TT_METAL_HOME:-${TTMLIR_BUILD}/../third_party/tt-metal/src/tt-metal}"
export PATH="${TTMLIR_BUILD}/bin:${PATH}"
export LD_LIBRARY_PATH="${TTMLIR_BUILD}/lib:${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${BUDDY_BUILD}/python_packages:${TTMLIR_BUILD}/python_packages${PYTHONPATH:+:$PYTHONPATH}"

# Offline HF to avoid ~40s x 5 retry stalls when network is flaky; the model is
# cached locally already.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Copy latest Buddy frontend/ops changes into the install dir (required so
# _runtime_inputs_ref is preserved and EqTensorOp lowers to TTIR).
if [[ -f "${SCRIPT_DIR}/../../frontend/Python/frontend.py" ]]; then
  cp -f "${SCRIPT_DIR}/../../frontend/Python/frontend.py" \
    "${BUDDY_BUILD}/python_packages/buddy/compiler/frontend.py"
fi
if [[ -f "${SCRIPT_DIR}/../../frontend/Python/ops/ttir_llm.py" ]]; then
  cp -f "${SCRIPT_DIR}/../../frontend/Python/ops/ttir_llm.py" \
    "${BUDDY_BUILD}/python_packages/buddy/compiler/ops/ttir_llm.py"
fi
if [[ -f "${SCRIPT_DIR}/../../frontend/Python/ops/ttir.py" ]]; then
  cp -f "${SCRIPT_DIR}/../../frontend/Python/ops/ttir.py" \
    "${BUDDY_BUILD}/python_packages/buddy/compiler/ops/ttir.py"
fi

cd "${SCRIPT_DIR}"
export DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-1024}"
SKIP_LOWER="${SKIP_LOWER:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
TTIR_OUT="${SCRIPT_DIR}/ttir_out_static"
CHAT_ART="${SCRIPT_DIR}/chat_artifacts"
mkdir -p "${TTIR_OUT}" "${CHAT_ART}"

SYS_DIR="/tmp/ttrt_sys_deepseek"
if [[ ! -f "${SYS_DIR}/system_desc.ttsys" ]]; then
  echo "=== query system descriptor ==="
  python -m ttrt query --save-artifacts --artifact-dir "${SYS_DIR}" 2>&1 | tail -3
fi
SYS="${SYS_DIR}/system_desc.ttsys"

PREFILL_TTNN="${TTIR_OUT}/deepseek_prefill_static.ttnn"
DECODE_TTNN="${TTIR_OUT}/deepseek_decode_static.ttnn"

if [[ "${SKIP_LOWER}" != "1" ]]; then
  if [[ ! -f "${PREFILL_TTNN}" ]]; then
    echo "=== [1/4] Lower prefill TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-deepseek-r1-lower-ttir.py \
      --mode prefill --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [1b/4] prefill TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/deepseek_r1_ttir_prefill.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}" \
      -o "${TTIR_OUT}/deepseek_prefill_static_ttnn.mlir"
    echo "=== [1c/4] prefill TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/deepseek_prefill_static_ttnn.mlir" \
      -o "${PREFILL_TTNN}"
  else
    echo "=== [1/4] prefill flatbuffer already exists: ${PREFILL_TTNN} ==="
  fi

  if [[ ! -f "${DECODE_TTNN}" ]]; then
    echo "=== [2/4] Lower decode TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-deepseek-r1-lower-ttir.py \
      --mode decode --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [2b/4] decode TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/deepseek_r1_ttir_decode.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}" \
      -o "${TTIR_OUT}/deepseek_decode_static_ttnn.mlir"
    echo "=== [2c/4] decode TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/deepseek_decode_static_ttnn.mlir" \
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
    python deepseek_chat_prepare.py \
      --max-cache-len "${MAX_CACHE_LEN}" \
      -o "${CHAT_ART}"
  else
    echo "=== [3/4] chat artifacts already exist: ${CHAT_ART} ==="
  fi
else
  echo "=== skipping chat prepare (SKIP_PREPARE=1) ==="
fi

echo "=== [4/4] Interactive chat on P150A ==="
python deepseek_chat_run.py \
  --prefill-ttnn "${PREFILL_TTNN}" \
  --decode-ttnn "${DECODE_TTNN}" \
  --artifacts "${CHAT_ART}" \
  --max-cache-len "${MAX_CACHE_LEN}" \
  --ignore-system-desc
