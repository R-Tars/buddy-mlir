#!/usr/bin/env bash
# Buddy Dynamo + ``buddy_mlir`` (Python 解释器 A) 与 ``ttmlir`` (Python 解释器 B) 必须 **同一版本**，
# 否则扩展 ``.so`` 无法加载。本机已验证路径：使用 conda 环境 ``tt-mlir``（Python 3.12），并：
#   - 将 buddy-mlir 配到该 Python（cmake ``-DPython3_EXECUTABLE=.../tt-mlir/bin/python``）
#   - ``pip install PyYAML nanobind``（若尚未有）
#   - ``ninja python-package-buddy-mlir`` 后 ``cmake -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`` 再 ``ninja python-package-buddy``
#
# 用法：
#   source "$(dirname "$0")/env_buddy_ttir.sh"
#   python buddy-lenet-lower-ttir.py --random-init --ttmlir-opt "$(which ttmlir-opt)"
#   # 默认写入本目录下 ttir_out/；也可用 -o /你的路径
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TTMLIR_BUILD="${TTMLIR_BUILD:-/wafer/zhuxinye/tt-mlir/build-runtime-clang20}"
BUDDY_BUILD="${BUDDY_BUILD:-$REPO_ROOT/build}"
export PATH="${TTMLIR_BUILD}/bin:${PATH}"
export PYTHONPATH="${BUDDY_BUILD}/python_packages:${TTMLIR_BUILD}/python_packages${PYTHONPATH:+:$PYTHONPATH}"
# 可选：避免无驱动 GPU 的 torch CUDA 警告
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
echo "[env_buddy_ttir] PYTHONPATH -> buddy + tt-mlir python_packages"
echo "[env_buddy_ttir] 请使用与 buddy_mlir 构建 **相同** 的 Python，例如:"
echo "    conda activate tt-mlir && which python"

# --- LeNet f32 端到端（PyTorch 参考）常用命令（在 examples/BuddyLeNet 下）---
# 训练并写入 tests/Models/BuddyLeNet/lenet-model.pth（需 torchvision，如 conda env torch2.10）:
#   python pytorch-lenet-train.py --epochs 3 --output ../tests/Models/BuddyLeNet/lenet-model.pth
# 单图分类（默认读 tests 下权重与 images/1.png）:
#   python pytorch-lenet-inference.py
# 导出 TTIR @forward 用的权重/输入/golden（需 Pillow；可用 torch2.10 env）:
#   python lenet_ttir_e2e_prepare.py -o ./ttir_e2e_artifacts
# 生成 f32 TTIR 模块并跑 ttmlir-opt / TTNN pipeline 检查:
#   LENET_MODEL_PATH=../tests/Models/BuddyLeNet/lenet-model.pth python buddy-lenet-lower-ttir.py --element-dtype f32 --packed-forward --ttmlir-opt \"\$(which ttmlir-opt)\" --ttnn-pipeline-check
# Tenstorrent 上跑 .ttnn 并与 golden 对比（应用 **本机** system_desc 重编译，勿用 mock wormhole）:
#   python lenet_ttir_e2e_prepare.py -o ./ttir_e2e_artifacts
#   python -m ttrt query --save-artifacts --artifact-dir /tmp/ttrt_sys && SYS=/tmp/ttrt_sys/system_desc.ttsys
#   ttmlir-opt ttir_out/lenet_ttir_module.mlir --ttir-to-ttnn-backend-pipeline=system-desc-path=$SYS -o ttir_out/lenet_ttnn.mlir
#   ttmlir-translate --ttnn-to-flatbuffer ttir_out/lenet_ttnn.mlir -o ttir_out/lenet.ttnn
#   conda activate tt-mlir   # 与 ttrt 的 cpython-312 一致
#   python lenet_ttrt_run_compare.py run --ttnn ./ttir_out/lenet.ttnn --e2e-dir ./ttir_e2e_artifacts --verbose
# 仅对比已有 device 输出: python lenet_ttrt_run_compare.py compare --device-output …/device_output_0.pt --e2e-dir ./ttir_e2e_artifacts
