#!/usr/bin/env bash
# ===- run_deepseek_p150_e2e.sh -----------------------------------------------
# DeepSeek TTIR：lower（含 @forward）→ TTNN → flatbuffer → P150A 运行 → golden 对比
# 使用前请按本机路径修改 TTMLIR_BUILD / BUDDY_BUILD；conda 环境名为 tt-mlir（与 LeNet 指南一致）。
# ===---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "/wafer/zhuxinye/miniconda3/etc/profile.d/conda.sh"
conda activate tt-mlir

# 与交互环境里 ``proxy_on`` 一致（便于 HuggingFace 下载）
export http_proxy="${http_proxy:-http://192.168.15.159:7890}"
export https_proxy="${https_proxy:-http://192.168.15.159:7890}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

export TTMLIR_BUILD="${TTMLIR_BUILD:-/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20}"
export BUDDY_BUILD="${BUDDY_BUILD:-/wafer/zhuxinye/buddy-mlir/build}"
export TT_METAL_HOME="${TT_METAL_HOME:-${TTMLIR_BUILD}/../third_party/tt-metal/src/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
export PATH="${TTMLIR_BUILD}/bin:${PATH}"
export LD_LIBRARY_PATH="${TTMLIR_BUILD}/lib:${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${BUDDY_BUILD}/python_packages:${TTMLIR_BUILD}/python_packages${PYTHONPATH:+:$PYTHONPATH}"

# 与 lowering 一致：同步前端源码到 build（若已 ninja 安装可省略）
if [[ -f "${SCRIPT_DIR}/../../frontend/Python/graph/ttir_import.py" ]]; then
  cp -f "${SCRIPT_DIR}/../../frontend/Python/graph/ttir_import.py" \
    "${BUDDY_BUILD}/python_packages/buddy/compiler/graph/ttir_import.py" || true
fi

cd "${SCRIPT_DIR}"
# 勿导出空串，否则 Python 的 os.environ.get(..., default) 会读到 "" 而跳过默认 HF id
export DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
SEQ="${SEQ:-8}"

echo "=== 1) Lower TTIR + @forward ==="
python buddy-deepseek-r1-lower-ttir.py \
  --mode prefill --seq "${SEQ}" \
  --packed-forward \
  --use-proxy \
  --ttmlir-opt "$(command -v ttmlir-opt)" \
  -o ./ttir_out

MOD="./ttir_out/deepseek_r1_ttir_prefill_module.mlir"

echo "=== 2) System descriptor ==="
ART="/tmp/ttrt_sys_deepseek"
python -m ttrt query --save-artifacts --artifact-dir "${ART}"
SYS="${ART}/system_desc.ttsys"

echo "=== 3) TTIR → TTNN ==="
ttmlir-opt "${MOD}" \
  --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}" \
  -o ./ttir_out/deepseek_prefill_ttnn.mlir

echo "=== 4) TTNN → flatbuffer ==="
ttmlir-translate --ttnn-to-flatbuffer ./ttir_out/deepseek_prefill_ttnn.mlir \
  -o ./ttir_out/deepseek_prefill.ttnn

echo "=== 5) E2E artifacts (packed weights + golden) ==="
python deepseek_ttir_e2e_prepare.py --seq "${SEQ}" -o ./ttir_e2e_artifacts --use-proxy

echo "=== 6) Device run + compare（需 P150A 在线；首次编译可能较久）==="
python deepseek_ttrt_run_compare.py \
  --ttnn ./ttir_out/deepseek_prefill.ttnn \
  --e2e-dir ./ttir_e2e_artifacts \
  --program-index 1 \
  --ignore-system-desc \
  --save-device-logits ./ttir_out/device_prefill_logits.npy

echo "=== 完成 ==="
