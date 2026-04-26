# Buddy → TTIR → P150A 端到端运行指南

> 最后更新：2026-04-26  
> 适用硬件：Tenstorrent P150A（Blackhole）  
> 适用环境：`/wafer/zhuxinye` 服务器，conda env `tt-mlir`

本指南汇总三个 TTIR 端到端例子的运行步骤、当前进度与性能数据。三个例子的复杂度依次递增，建议按顺序熟悉：

| 例子 | 模型 | 流程 | 状态 |
|---|---|---|---|
| `examples/BuddyLeNet` | LeNet（CNN，~60K 参数） | f32 forward，单图分类，golden 对比 | ✅ 已通 |
| `examples/BuddyDeepSeekR1` | DeepSeek-R1-Distill-Qwen-1.5B (bf16) | static-cache 1024，prefill+decode 交互 chat，golden 对比 | ✅ 已通 |
| `examples/BuddyLlama31-8B` | meta-llama/Llama-3.1-8B-Instruct (bf16) | static-cache 1024，prefill+decode 交互 chat，对标官方 `tt_transformers` | ✅ 已通 |

---

## 0. 公共前置条件（一次性）

| 组件 | 路径 |
|---|---|
| miniconda env | `/wafer/zhuxinye/miniconda3/envs/tt-mlir`（Python 3.12） |
| tt-mlir build | `/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20`（`ttmlir-opt` / `ttmlir-translate` / `ttrt`） |
| buddy-mlir build | `/wafer/zhuxinye/buddy-mlir/build`（`build/python_packages/buddy/compiler/...`） |
| HF cache | `/wafer/zhuxinye/.cache/huggingface`（DeepSeek-R1-Distill-Qwen-1.5B 与 Llama-3.1-8B-Instruct 已离线快照） |
| LeNet 权重 | `tests/Models/BuddyLeNet/lenet-model.pth` |

每个例子目录都自带一个 `env_*.sh` / `_env.sh` 用来激活环境 + 同步前端 Python 改动到 build 目录。

P150A 设备 sanity：

```bash
source /wafer/zhuxinye/miniconda3/etc/profile.d/conda.sh && conda activate tt-mlir
export PATH=/wafer/zhuxinye/gitprojects/tt-mlir/build-runtime-clang20/bin:$PATH
ttrt query | tail -3        # 应该看到 P150A 描述符被 dump 到 /tmp/ttrt-artifacts
```

> ⚠️ Llama-3.1-8B 在权重导出和 chat 启动阶段会临时占用 30+ GB 虚拟内存，必须先 `ulimit -v 95000000`。所有 Llama wrapper 已经内置这一行。

---

## 1. LeNet（最快上手）

详细文档：`examples/BuddyLeNet/LENET_TTIR_E2E_GUIDE.md`（已包含 477 行的完整步骤、故障排除、架构图）。

### 一键命令

```bash
cd examples/BuddyLeNet
source ./env_buddy_ttir.sh        # 激活 conda + 设 PYTHONPATH/PATH/LD_LIBRARY_PATH

# 1) PyTorch → TTIR MLIR
python buddy-lenet-lower-ttir.py \
    --pth ../../tests/Models/BuddyLeNet/lenet-model.pth \
    -o ttir_out

# 2) TTIR → TTNN
ttmlir-opt ttir_out/lenet_ttir_module.mlir \
    --ttir-to-ttnn-backend-pipeline="system-desc-path=$(ttrt query --save-artifacts --artifact-dir /tmp/ttrt_sys_lenet 2>/dev/null; echo /tmp/ttrt_sys_lenet/system_desc.ttsys)" \
    -o ttir_out/lenet_ttnn.mlir

# 3) TTNN → flatbuffer
ttmlir-translate --ttnn-to-flatbuffer ttir_out/lenet_ttnn.mlir -o ttir_out/lenet.ttnn

# 4) 在 P150A 上跑并对比 PyTorch golden
python lenet_ttir_e2e_prepare.py \
    --pth ../../tests/Models/BuddyLeNet/lenet-model.pth \
    --image images/3.png \
    -o ttir_e2e_artifacts
python lenet_ttrt_run_compare.py \
    --ttnn ttir_out/lenet.ttnn \
    --e2e-dir ttir_e2e_artifacts \
    --ignore-system-desc
```

### 期望输出

```
[Compare] max_abs_diff=...  PCC=1.0000  argmax_match=True (golden=3, device=3)
```

---

## 2. DeepSeek-R1-Distill-Qwen-1.5B

提供两条流水线：

- **E2E sanity**：seq=8 prefill-only 跑一次，验证 prefill logits 与 HF bf16 golden 数值一致（PCC > 0.99）。
- **Chat**：static-cache 1024，prefill + decode 交互式生成。

### Sanity（一键）

```bash
cd examples/BuddyDeepSeekR1
./run_deepseek_p150_e2e.sh        # SEQ=8 默认；可 SEQ=32 ./run_deepseek_p150_e2e.sh
```

会按顺序：lower TTIR → TTNN → flatbuffer → 导出 packed weights + golden → P150A 跑 → 对比。

### Chat（一键）

```bash
cd examples/BuddyDeepSeekR1
./run_deepseek_p150_chat.sh       # 等候 “Please send a message:” 然后输入
```

第二次起追加 `SKIP_LOWER=1 SKIP_PREPARE=1 ./run_deepseek_p150_chat.sh` 可绕过 ~5 分钟的 lowering + artifact 阶段。

### 性能（参考）

| Phase | 数值 |
|---|---|
| Prefill (seq=1024, 1.5B bf16) | ~0.6 s |
| Decode device-only steady | ~25 t/s/u |
| Decode wall steady | ~12 t/s/u（host 有 KV round-trip 开销） |

---

## 3. Llama-3.1-8B-Instruct（对标官方栈）

### 基线（要超越的）

来自服务器上 Tenstorrent 官方 `tt_transformers` stack（详情见 `/wafer/zhuxinye/llama_tt_transformers/BASELINE.md`）：

| Case | TTFT (ms) | Decode (t/s/u) |
|---|---|---|
| `batch-1 performance` | 65.5 | **23.2** |
| `ci-token-matching` (Instruct) | 134 | 22.6 |

### 一键 chat

```bash
cd examples/BuddyLlama31-8B
./run_llama31_p150_chat.sh        # 内部已 ulimit -v 95000000
# 等候 “Please send a message:” 后输入
```

子步骤拆解（首次跑各阶段大致耗时）：

| 步骤 | 内容 | 耗时 | 峰值 RAM |
|---|---|---|---|
| [1/4] | prefill TTIR + opt + flatbuffer (`llama31_prefill_static.ttnn`, 4.1 MB) | ~90 s | 16 GB |
| [2/4] | decode TTIR + opt + flatbuffer (`llama31_decode_static.ttnn`, 4.3 MB) | ~20 s | 16 GB |
| [3/4] | `llama31_chat_prepare.py` 导出 prefill+decode `weights.npz`（各 ~15 GB） | ~5 min | 33 GB |
| [4/4] | `llama31_chat_run.py` 加载 → 上传 → JIT → decode 循环 | 上传 ~1 min；首 token ~1.7 s；steady ~0.18 s/tok | 33 GB |

### 性能基线（首次量化，2026-04-24）

prompt 50 tokens，`--ignore-eos --max-new-tokens 128`：

| Metric | 值 |
|---|---|
| Prefill 1024 seq | 1.11 s → **920 tok/s** |
| Decode wall steady (126 tok in 23.23 s) | **5.42 tok/s** |
| Decode **device-only** steady (`rt.submit+wait`) | **17.31 tok/s** |
| Host round-trip overhead | 126.6 ms/tok |

device 计算已达官方 75 %（零优化起点，无 flash-attn / GQA fusion / prefetcher）；wall 差距几乎全在 host 的 64 条 tensor `to_layout`+`deallocate` 同步循环——P4 优化的主要靶子。

### Sanity 对比（仅 prefill，可选）

```bash
ulimit -v 95000000
python buddy-llama31-lower-ttir.py --mode prefill --seq 32 \
    --ttmlir-opt "$(command -v ttmlir-opt)" -o ttir_out
ttmlir-opt ttir_out/llama31_ttir_prefill.mlir \
    --ttir-to-ttnn-backend-pipeline="system-desc-path=/tmp/ttrt_sys_llama31/system_desc.ttsys" \
    -o ttir_out/llama31_prefill_seq32_ttnn.mlir
ttmlir-translate --ttnn-to-flatbuffer ttir_out/llama31_prefill_seq32_ttnn.mlir \
    -o ttir_out/llama31_prefill_seq32.ttnn
python llama31_ttir_e2e_prepare.py --seq 32 -o ttir_e2e_artifacts
python llama31_ttrt_run_subgraph.py \
    --ttnn ttir_out/llama31_prefill_seq32.ttnn \
    --e2e-dir ttir_e2e_artifacts --ignore-system-desc
```

期望：PCC ≈ 0.99，Top-1 token 与 HF bf16 golden 一致。

---

## 4. 三个例子保留下来的脚本一览

### `examples/BuddyLeNet`

| 文件 | 作用 |
|---|---|
| `env_buddy_ttir.sh` | 激活环境 + 同步 frontend 改动到 build 包 |
| `buddy-lenet-import.py` | 旧的 import 入口（保留，PyTorch → fake-lenet.mlir） |
| `buddy-lenet-lower-ttir.py` | PyTorch → TTIR MLIR（本流程入口） |
| `pytorch-lenet-train.py` | argparse 化训练脚本（默认 3 epoch） |
| `pytorch-lenet-inference.py` | 单图推理（自动从 `tests/Models/...` 找权重） |
| `lenet_ttir_e2e_prepare.py` | 导出 packed weight + golden logits |
| `lenet_ttrt_run_compare.py` | 上 P150A 跑 + 与 golden 对比 |
| `LENET_TTIR_E2E_GUIDE.md` | 详细步骤文档 |

### `examples/BuddyDeepSeekR1`

| 文件 | 作用 |
|---|---|
| `buddy-deepseek-r1-lower-ttir.py` | PyTorch → TTIR（支持 `--mode {prefill,decode}` + `--static-cache`） |
| `deepseek_ttir_e2e_prepare.py` | seq=8/32 prefill sanity 流程的 packed weights + golden 导出 |
| `deepseek_ttrt_run_compare.py` | seq=8/32 prefill sanity 设备运行 + 对比 |
| `deepseek_chat_prepare.py` | static-cache 1024 chat 流程的 weights + slot_roles 导出 |
| `deepseek_chat_run.py` | chat 流程主驱动（device-resident KV，host overhead 已优化） |
| `run_deepseek_p150_e2e.sh` | E2E sanity wrapper |
| `run_deepseek_p150_chat.sh` | Chat wrapper |

### `examples/BuddyLlama31-8B`

| 文件 | 作用 |
|---|---|
| `_env.sh` | 激活环境 + 同步 frontend 改动 |
| `buddy-llama31-lower-ttir.py` | PyTorch → TTIR（同 DeepSeek 接口，模型 id 默认 Llama-3.1-8B-Instruct） |
| `llama31_ttir_e2e_prepare.py` | seq=32 prefill sanity 的 packed weights + golden 导出（streaming write 防 OOM） |
| `llama31_ttrt_run_subgraph.py` | seq=32 prefill sanity 设备运行 + 对比（np.memmap 防 OOM） |
| `llama31_chat_prepare.py` | static-cache 1024 chat 流程 prepare（动态 KV shape，兼容 GQA） |
| `llama31_chat_run.py` | chat 流程主驱动（含 force=True KV dealloc 防碎片化、`--ignore-eos`、`--max-new-tokens`） |
| `run_llama31_p150_chat.sh` | Chat wrapper（已内置 ulimit -v 95 GB） |
| `README.md` | 进度 + 基线表 |

---

## 5. 涉及的前端 / 测试改动

### `frontend/Python`

| 文件 | 作用 |
|---|---|
| `frontend.py`（modified） | DynamoCompiler 增加 `_runtime_inputs_ref`、`_params_ref` 元数据，供 chat_prepare 反查每个 placeholder 对应的实际 tensor |
| `graph/graph.py`（modified） | Graph.fuse_ops、子图遍历相关小改动，配合 TTIR lowering |
| `graph/ttir_import.py`（new） | 新增 TTIR 导入 / lowering 主入口 |
| `ops/ttir.py`（new） | TTIR 算子注册（matmul、reshape、permute、SDPA 等） |
| `ops/ttir_llm.py`（new） | LLM 专用 TTIR 算子（GQA / RoPE / static-cache scatter via `where`） |

### `tests/Python/dialects`

| 文件 | 作用 |
|---|---|
| `test_lenet_ttir_op_inventory.py` | LeNet TTIR 模型的算子 inventory 测试 |
| `test_lenet_ttir_ops_coverage.py` | LeNet TTIR 算子覆盖度测试 |
| `test_ttir_ops_registry.py` | TTIR 算子注册表 sanity 测试 |

---

## 6. 排查清单

| 现象 | 可能原因 / 处置 |
|---|---|
| `ttmlir-opt: ERROR: ...op not registered` | 没把 frontend 改动 cp 到 build 目录；执行 `source env_buddy_ttir.sh` 或 `_env.sh`，里面会自动同步 |
| Python OOM（Llama 流程） | 没加 `ulimit -v 95000000`；wrapper 默认会加，手工跑要自己加 |
| `Out of Memory: Not enough space to allocate ... DRAM buffer` | DeepSeek/Llama chat 第 3 步起爆 DRAM：检查 `chat_run.py` 里 KV swap 那段是 `force=True` |
| HF `OSError: ... not a local folder` | `HF_HUB_OFFLINE=1` 导致离线模式找不到本地缓存；wrapper 已默认开启离线，本地确实缓存过 Llama-3.1 / DeepSeek-R1 |
| `tokenizer.apply_chat_template` 抛 `TemplateError` | 模型 id 不对；Llama 默认 `meta-llama/Llama-3.1-8B-Instruct`，DeepSeek 默认 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| device 跑得通但 logits 全是 NaN | 前端 cp 没生效 / TTIR 改动没拉进 build；同 1 |

---

## 7. 当前里程碑总览

| 例子 | P0 lower | P1 prefill device | P2 decode chat | P3 perf | 备注 |
|---|---|---|---|---|---|
| LeNet | ✅ | ✅ PCC=1.0 | n/a | n/a | 简单分类，无 KV |
| DeepSeek-R1-1.5B | ✅ | ✅ PCC>0.99 | ✅ 流畅生成 | 25 t/s/u device-only | chat wrapper 工作 |
| Llama-3.1-8B | ✅ | ✅ PCC=0.9928 | ✅ "What is 2+2?" → "2 + 2 = 4." | **17.3 t/s/u device-only** vs 官方 23.2 | 200-word transformer 解释生成正确 |

后续优化（P4）方向已记录在 `examples/BuddyLlama31-8B/README.md`。
