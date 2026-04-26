# LeNet 端到端测试指南：PyTorch → Buddy → TTIR → P150A 设备

> 最后验证时间：2026-04-17，服务器 P150A（Blackhole 架构）  
> 用户：zhuxinye，工作目录 `/wafer/zhuxinye`

---

## 目录

1. [整体流程概览](#1-整体流程概览)
2. [前置条件](#2-前置条件)
3. [环境激活（每次开新终端必做）](#3-环境激活每次开新终端必做)
4. [Step 1：PyTorch → Buddy Graph → TTIR MLIR](#4-step-1pytorch--buddy-graph--ttir-mlir)
5. [Step 2：TTIR → TTNN（使用真实设备描述符）](#5-step-2ttir--ttnn使用真实设备描述符)
6. [Step 3：TTNN MLIR → Flatbuffer 二进制](#6-step-3ttnn-mlir--flatbuffer-二进制)
7. [Step 4：在 P150A 上运行（快速验证）](#7-step-4在-p150a-上运行快速验证)
8. [Step 5：端到端精度对比（完整 E2E）](#8-step-5端到端精度对比完整-e2e)
9. [用不同测试图片运行](#9-用不同测试图片运行)
10. [常见问题与故障排除](#10-常见问题与故障排除)
11. [关键文件说明](#11-关键文件说明)
12. [流程架构图](#12-流程架构图)

---

## 1. 整体流程概览

```
PyTorch LeNet 模型 (.pth)
        │
        ▼  torch._dynamo 捕获计算图
  DynamoCompiler (buddy.compiler.frontend)
        │
        ▼  ops_registry=tosa → Buddy Graph
  Graph.lower_to_ttir()
        │
        ▼  ttmlir Python bindings 生成 TTIR dialect
  TTIR MLIR (.mlir)      ← 本步产物：lenet_ttir_module.mlir
        │
        ▼  ttmlir-opt --ttir-to-ttnn-backend-pipeline
  TTNN MLIR (.mlir)      ← 本步产物：lenet_ttnn.mlir
        │
        ▼  ttmlir-translate --ttnn-to-flatbuffer
  Flatbuffer (.ttnn)     ← 本步产物：lenet.ttnn
        │
        ▼  ttrt run / lenet_ttrt_run_compare.py
  P150A Blackhole 设备执行 → 输出 logits → 与 PyTorch golden 对比
```

---

## 2. 前置条件

以下已在当前服务器上就绪，无需重复操作：

| 组件 | 路径 | 说明 |
|------|------|------|
| Miniconda | `/wafer/zhuxinye/miniconda3` | Python 环境管理 |
| conda 环境 `tt-mlir` | `/wafer/zhuxinye/miniconda3/envs/tt-mlir` | Python 3.12.13 |
| tt-mlir 构建 | `/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20` | 含 ttmlir-opt、ttmlir-translate、ttrt |
| buddy-mlir 构建 | `/wafer/zhuxinye/buddy-mlir/build` | 含 buddy.compiler Python 包 |
| LeNet 权重 | `/wafer/zhuxinye/buddy-mlir/tests/Models/BuddyLeNet/lenet-model.pth` | 预训练权重 |
| 测试图片 | `/wafer/zhuxinye/buddy-mlir/examples/BuddyLeNet/images/` | 0-9 数字图片 |

### conda 环境中已安装的关键包

```
torch           2.11.0
torchvision     0.26.0
numpy           2.4.4
pillow          12.2.0
loguru          0.7.3
graphviz        0.21
nanobind        2.10.2
pybind11        3.0.3
PyYAML          6.0.3
```

---

## 3. 环境激活（每次开新终端必做）

**复制以下整段到终端执行**，设置好所有需要的环境变量：

```bash
# ====== 环境激活脚本 ======
source /wafer/zhuxinye/miniconda3/etc/profile.d/conda.sh
conda activate tt-mlir

export TTMLIR_BUILD=/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20
export BUDDY_BUILD=/wafer/zhuxinye/buddy-mlir/build
export TT_METAL_HOME=/wafer/zhuxinye/gitprojects/tt-mlir/third_party/tt-metal/src/tt-metal
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"

export PATH="${TTMLIR_BUILD}/bin:${PATH}"
export PYTHONPATH="${BUDDY_BUILD}/python_packages:${TTMLIR_BUILD}/python_packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${TTMLIR_BUILD}/lib:${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"

# 进入 LeNet 示例目录
cd /wafer/zhuxinye/buddy-mlir/examples/BuddyLeNet
```

### 验证环境是否正确

```bash
# 验证 Python 版本
python --version
# 期望输出: Python 3.12.13

# 验证 ttmlir-opt 可用
which ttmlir-opt
# 期望输出: /wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20/bin/ttmlir-opt

# 验证关键 Python 模块都能导入
python -c "
from buddy.compiler.frontend import DynamoCompiler; print('buddy OK')
import ttmlir; print('ttmlir OK')
from ttmlir.dialects import ttir; print('ttir dialect OK')
import torch; print('torch', torch.__version__)
"
# 期望输出:
#   buddy OK
#   ttmlir OK
#   ttir dialect OK
#   torch 2.11.0
```

---

## 4. Step 1：PyTorch → Buddy Graph → TTIR MLIR

> 这一步将 PyTorch LeNet 模型通过 Buddy 前端下降到 TTIR dialect 的 MLIR。

```bash
python buddy-lenet-lower-ttir.py \
  --element-dtype f32 \
  --packed-forward \
  --ttmlir-opt "$(which ttmlir-opt)"
```

### 参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--element-dtype` | `f32` | TTIR 张量元素类型，f32 与 PyTorch f32 权重对齐 |
| `--packed-forward` | （标志） | 生成 `@forward` 函数，将打包权重解包后调用 `@subgraph0` |
| `--ttmlir-opt` | `$(which ttmlir-opt)` | 用 ttmlir-opt 做语法验证 |

### 可选参数

| 参数 | 说明 |
|------|------|
| `--random-init` | 不加载权重，使用随机初始化（快速测试 op lowering 是否正确） |
| `--element-dtype bf16` | 使用 bf16 类型（默认值） |
| `--ttnn-pipeline-check` | 额外用 mock system desc 做 TTNN pipeline 烟雾测试 |
| `-o /路径` | 指定输出目录（默认 `./ttir_out/`） |

### 期望输出

```
Wrote TTIR module: /wafer/zhuxinye/buddy-mlir/examples/BuddyLeNet/ttir_out/lenet_ttir_module.mlir
module {
  func.func @subgraph0(%arg0: tensor<1x1x28x28xf32>, ...) -> tensor<1x10xf32> {
    ... ttir.conv2d ... ttir.relu ... ttir.max_pool2d ... ttir.matmul ... ttir.add ...
  }
  func.func @forward(%arg0: tensor<44426xf32>, %arg1: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
    ... ttir.slice_static ... ttir.reshape ... call @subgraph0 ...
  }
}
Running: .../ttmlir-opt .../lenet_ttir_module.mlir -o /dev/null
ttmlir-opt: OK
```

### 产物

- `ttir_out/lenet_ttir_module.mlir` — 含 `@subgraph0`（计算逻辑）和 `@forward`（权重解包+调用）的 TTIR MLIR 模块

### 检查产物

```bash
cat ttir_out/lenet_ttir_module.mlir
```

---

## 5. Step 2：TTIR → TTNN（使用真实设备描述符）

> 这一步使用 P150A 的真实 system descriptor 将 TTIR 下降到 TTNN dialect。

### 5.1 获取设备描述符

```bash
python -m ttrt query --save-artifacts --artifact-dir /tmp/ttrt_sys
```

期望输出中包含 `"arch": "Blackhole"` 和 `PASS: getting system_desc passed`。

### 5.2 执行 TTIR → TTNN 转换

```bash
SYS=/tmp/ttrt_sys/system_desc.ttsys

ttmlir-opt ttir_out/lenet_ttir_module.mlir \
  --ttir-to-ttnn-backend-pipeline="system-desc-path=$SYS" \
  -o ttir_out/lenet_ttnn.mlir
```

### 期望输出

```
ttir_out/lenet_ttir_module.mlir:1:1: warning: Empty argument type map provided. ...
```

只有一个无害 warning，无报错即成功。

### 检查产物

```bash
# 查看 TTNN MLIR（会包含 ttnn.conv2d, ttnn.relu, ttnn.matmul 等 TTNN dialect ops）
head -50 ttir_out/lenet_ttnn.mlir
```

---

## 6. Step 3：TTNN MLIR → Flatbuffer 二进制

```bash
ttmlir-translate --ttnn-to-flatbuffer ttir_out/lenet_ttnn.mlir \
  -o ttir_out/lenet.ttnn
```

### 验证

```bash
ls -la ttir_out/lenet.ttnn
# 期望: 约 140KB 的二进制文件
```

---

## 7. Step 4：在 P150A 上运行（快速验证）

> 使用 `ttrt run` 快速验证 `.ttnn` 能否在设备上执行（输入为全零/随机）。

```bash
python -m ttrt run ttir_out/lenet.ttnn
```

### 期望输出

```
... evaluating binary=ttir_out/lenet.ttnn
... e2e_duration_nanoseconds_submit = ...
... PASS: test case=ttir_out/lenet.ttnn
```

看到 `PASS` 即设备执行成功。

---

## 8. Step 5：端到端精度对比（完整 E2E）

> 使用真实权重和测试图片，在设备上运行并与 PyTorch golden 结果对比。

### 8.1 准备 E2E 数据（权重 + 输入 + golden）

```bash
python lenet_ttir_e2e_prepare.py -o ./ttir_e2e_artifacts
```

期望输出：

```
[OK] artifacts in .../ttir_e2e_artifacts
     golden class (PyTorch): 1
     files: arg0.data, input_1x1x28x28_f32.npy, golden_logits_f32.npy, ...
```

产物说明：

| 文件 | 内容 |
|------|------|
| `arg0.data` | f32 打包权重（44426 个 float32，177704 字节） |
| `input_1x1x28x28_f32.npy` | 测试输入图像（默认 `images/1.png`，经 Normalize(0.5,0.5) 预处理） |
| `golden_logits_f32.npy` | PyTorch 推理的 logits 输出 shape=(1,10) |
| `golden_class.txt` | PyTorch argmax 分类结果 |

### 8.2 在设备上运行并对比

```bash
python lenet_ttrt_run_compare.py run \
  --ttnn ./ttir_out/lenet.ttnn \
  --e2e-dir ./ttir_e2e_artifacts \
  --program-index 1 \
  --verbose \
  --save-device-output ./ttir_out/device_output.pt
```

### 参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--ttnn` | `./ttir_out/lenet.ttnn` | 编译产物路径 |
| `--e2e-dir` | `./ttir_e2e_artifacts` | 权重/输入/golden 目录 |
| `--program-index` | `1` | 选择 `@forward` 函数（0=`@subgraph0`，1=`@forward`） |
| `--verbose` | （标志） | 打印各阶段进度，便于排查卡住位置 |
| `--save-device-output` | `./ttir_out/device_output.pt` | 保存设备 logits 到文件 |

### 期望输出

```
[lenet_ttrt] loading flatbuffer …
[lenet_ttrt] flatbuffer loaded
... PASS: getting system_desc passed
[lenet_ttrt] query OK
[lenet_ttrt] set_compatible_device_runtime …
[lenet_ttrt] get_program …
[lenet_ttrt] load arg0 + image tensors …
[lenet_ttrt] populate_inputs/outputs …
[lenet_ttrt] open_mesh_device …
[lenet_ttrt] create_tensor(host) …
[lenet_ttrt] convert_input_layouts …
[lenet_ttrt] submit (首次可能编译内核，耗时数分钟) …
[lenet_ttrt] submit done
{
  "golden_argmax": 1,           ← PyTorch 参考分类
  "device_argmax": 1,           ← 设备输出分类（应一致）
  "max_abs_diff": 0.078...,     ← 最大绝对误差
  "mean_abs_diff": 0.032...,    ← 平均绝对误差
  "pcc": 0.999999...,           ← 皮尔逊相关系数（>0.99 为优秀）
  "pcc_msg": "..."
}
```

### 判断是否成功

- `golden_argmax == device_argmax`：分类结果一致 ✓
- `pcc > 0.99`：PCC 极高，数值精度优秀 ✓

### 8.3 仅做离线对比（已有设备输出时）

如果之前已保存了 `device_output.pt`，可以不连设备直接对比：

```bash
python lenet_ttrt_run_compare.py compare \
  --device-output ./ttir_out/device_output.pt \
  --e2e-dir ./ttir_e2e_artifacts
```

---

## 9. 用不同测试图片运行

`images/` 目录下有 0-9 数字图片。要换图片，只需在 prepare 步骤指定 `--image`：

```bash
# 例如用数字 3 的图片
python lenet_ttir_e2e_prepare.py -o ./ttir_e2e_artifacts --image ./images/3.png

# 然后重新在设备上运行
python lenet_ttrt_run_compare.py run \
  --ttnn ./ttir_out/lenet.ttnn \
  --e2e-dir ./ttir_e2e_artifacts \
  --program-index 1 \
  --verbose
```

注意：换图片后 `.ttnn` 不需要重新编译，因为模型结构和权重不变。

---

## 10. 常见问题与故障排除

### Q1：`ttrt run` 卡住不动

设备可能处于脏状态（之前进程被 kill -9 等）。

```bash
# 重置设备
tt-smi -r 0

# 等待几秒后重试
python -m ttrt run ttir_out/lenet.ttnn
```

### Q2：`ModuleNotFoundError: No module named 'xxx'`

确保环境变量设置正确。重新执行 [第3节](#3-环境激活每次开新终端必做) 的激活脚本。

### Q3：`ttmlir-opt` 报错 MLIR 语法错误

检查 Step 1 的 TTIR lowering 是否成功。可单独跑验证：

```bash
ttmlir-opt ttir_out/lenet_ttir_module.mlir -o /dev/null
```

### Q4：system desc 不匹配

如果 `.ttnn` 是在其他机器编译的，需用当前机器的 system desc 重新编译：

```bash
python -m ttrt query --save-artifacts --artifact-dir /tmp/ttrt_sys
SYS=/tmp/ttrt_sys/system_desc.ttsys
ttmlir-opt ttir_out/lenet_ttir_module.mlir \
  --ttir-to-ttnn-backend-pipeline="system-desc-path=$SYS" \
  -o ttir_out/lenet_ttnn.mlir
ttmlir-translate --ttnn-to-flatbuffer ttir_out/lenet_ttnn.mlir \
  -o ttir_out/lenet.ttnn
```

### Q5：CUDA 相关 warning

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old ...
```

这是 torch CUDA 初始化的警告，**完全不影响**我们的流程（我们使用 Tenstorrent 硬件，不用 NVIDIA GPU）。

### Q6：首次 submit 耗时很长

首次在设备上运行会编译内核（JIT），可能需要 10-30 秒。后续运行会快得多。

---

## 11. 关键文件说明

### 核心源码

| 文件 | 作用 |
|------|------|
| `buddy-lenet-lower-ttir.py` | 主脚本：PyTorch → Buddy Graph → TTIR MLIR |
| `lenet_ttir_e2e_prepare.py` | 准备 E2E 数据：导出权重、输入、golden |
| `lenet_ttrt_run_compare.py` | 设备运行 + 对比：在 P150A 上跑并校验精度 |
| `env_buddy_ttir.sh` | 环境变量设置辅助脚本 |
| `model.py` | LeNet 模型定义 |

### TTIR Lowering 实现

| 文件 | 作用 |
|------|------|
| `frontend/Python/ops/ttir.py` | 算子 lowering：Conv2d、MaxPool2d、ReLU、AddMM、Reshape 等 → TTIR ops |
| `frontend/Python/graph/ttir_import.py` | 图级 lowering：遍历 Buddy Graph 节点，构建 `ttmlir.ir.Module` |
| `frontend/Python/graph/graph.py` | `Graph.lower_to_ttir()` 方法入口 |

### 生成产物

| 文件 | 说明 |
|------|------|
| `ttir_out/lenet_ttir_module.mlir` | TTIR MLIR（含 @subgraph0 + @forward） |
| `ttir_out/lenet_ttnn.mlir` | TTNN MLIR（设备特定） |
| `ttir_out/lenet.ttnn` | Flatbuffer 二进制（设备可执行） |
| `ttir_out/device_output.pt` | 设备输出 logits |
| `ttir_e2e_artifacts/arg0.data` | 打包权重 |
| `ttir_e2e_artifacts/golden_logits_f32.npy` | PyTorch 参考 logits |

---

## 12. 流程架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST (x86_64)                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  PyTorch     │    │  buddy.compiler  │    │  ttmlir       │  │
│  │  LeNet model │───▶│  DynamoCompiler  │───▶│  Python API   │  │
│  │  (.pth)      │    │  Buddy Graph     │    │  TTIR Module  │  │
│  └──────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                      │          │
│                                              ┌───────▼───────┐  │
│                                              │  ttmlir-opt   │  │
│                                              │  TTIR → TTNN  │  │
│                                              └───────┬───────┘  │
│                                                      │          │
│                                           ┌──────────▼────────┐ │
│                                           │ ttmlir-translate  │ │
│                                           │ TTNN → Flatbuffer │ │
│                                           └──────────┬────────┘ │
│                                                      │          │
│  ┌───────────────────────────────────────────────────▼────────┐ │
│  │                    ttrt runtime                            │ │
│  │  打包权重 + 输入图像 → submit(.ttnn) → wait → 读取输出    │ │
│  └───────────────────────────────────┬───────────────────────┘ │
│                                      │                         │
└──────────────────────────────────────┼─────────────────────────┘
                                       │ PCIe
                              ┌────────▼────────┐
                              │   P150A 设备     │
                              │   Blackhole 架构  │
                              │   10x11 核心网格  │
                              │   4GB DRAM x8     │
                              └─────────────────┘
```

---

## 一键复现（完整命令序列）

如果你想从头到尾快速复现，以下是完整命令序列：

```bash
# === 0. 激活环境 ===
source /wafer/zhuxinye/miniconda3/etc/profile.d/conda.sh
conda activate tt-mlir
export TTMLIR_BUILD=/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20
export BUDDY_BUILD=/wafer/zhuxinye/buddy-mlir/build
export TT_METAL_HOME=/wafer/zhuxinye/gitprojects/tt-mlir/third_party/tt-metal/src/tt-metal
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
export PATH="${TTMLIR_BUILD}/bin:${PATH}"
export PYTHONPATH="${BUDDY_BUILD}/python_packages:${TTMLIR_BUILD}/python_packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${TTMLIR_BUILD}/lib:${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
cd /wafer/zhuxinye/buddy-mlir/examples/BuddyLeNet

# === 1. PyTorch → TTIR MLIR ===
python buddy-lenet-lower-ttir.py \
  --element-dtype f32 --packed-forward \
  --ttmlir-opt "$(which ttmlir-opt)"

# === 2. 获取设备描述符 ===
python -m ttrt query --save-artifacts --artifact-dir /tmp/ttrt_sys
SYS=/tmp/ttrt_sys/system_desc.ttsys

# === 3. TTIR → TTNN ===
ttmlir-opt ttir_out/lenet_ttir_module.mlir \
  --ttir-to-ttnn-backend-pipeline="system-desc-path=$SYS" \
  -o ttir_out/lenet_ttnn.mlir

# === 4. TTNN → Flatbuffer ===
ttmlir-translate --ttnn-to-flatbuffer ttir_out/lenet_ttnn.mlir \
  -o ttir_out/lenet.ttnn

# === 5. 准备 E2E 数据 ===
python lenet_ttir_e2e_prepare.py -o ./ttir_e2e_artifacts

# === 6. 设备运行 + 精度对比 ===
python lenet_ttrt_run_compare.py run \
  --ttnn ./ttir_out/lenet.ttnn \
  --e2e-dir ./ttir_e2e_artifacts \
  --program-index 1 \
  --verbose \
  --save-device-output ./ttir_out/device_output.pt
```

预期最终输出：

```json
{
  "golden_argmax": 1,
  "device_argmax": 1,
  "max_abs_diff": 0.078,
  "mean_abs_diff": 0.032,
  "pcc": 0.9999993
}
```

`golden_argmax == device_argmax` 且 `pcc > 0.999`，端到端验证通过。
