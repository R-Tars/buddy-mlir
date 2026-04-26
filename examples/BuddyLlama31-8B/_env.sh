#!/usr/bin/env bash
# Shared environment for the Llama-3.1-8B TTIR example. Source it, don't exec.
# Mirrors the exports inside BuddyDeepSeekR1/run_deepseek_p150_chat.sh so the
# downstream pipeline (ttmlir-opt / ttmlir-translate / ttrt) stays identical.

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

export HF_HOME="${HF_HOME:-/wafer/zhuxinye/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export LLAMA31_MODEL_PATH="${LLAMA31_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

# Copy latest Buddy frontend/ops changes into the install dir (needed so
# _runtime_inputs_ref is preserved and EqTensorOp lowers to TTIR).
BUDDY_SRC_FRONTEND="${SCRIPT_DIR}/../../frontend/Python"
for f in frontend.py ops/ttir_llm.py ops/ttir.py; do
  src="${BUDDY_SRC_FRONTEND}/${f}"
  dst="${BUDDY_BUILD}/python_packages/buddy/compiler/${f}"
  if [[ -f "${src}" && -f "${dst}" ]]; then
    cp -f "${src}" "${dst}"
  fi
done
