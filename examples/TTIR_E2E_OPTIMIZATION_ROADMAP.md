# TTIR end-to-end optimization roadmap (Llama-3.1-8B on P150A)

> 本文档定义如何用 Buddy 前端图基础设施优化 TTIR 路径，使其在 P150A
> 上**超越** Tenstorrent 原生 `tt_transformers` 软件栈。
>
> 起点（commit `19a49ff`，2026-04-24）：Llama-3.1-8B 端到端打通，
> wall 5.4 t/s/u / device-only 17.3 t/s/u。
>
> 基线（要超越的）：官方 `tt_transformers` 在同一台 P150A 上跑出
> **23.2 t/s/u**（详情见 `/wafer/zhuxinye/llama_tt_transformers/BASELINE.md`）。

---

## 1. 目标分级

| Tier | wall t/s/u | 含义 |
|---|---|---|
| T0 当前 | 5.4 | TTIR 端到端 PoC 跑通 |
| T1 host 收敛 | ≥17 | wall 持平 device-only |
| T2 持平官方 | ≥23 | 与官方 `tt_transformers` 打成平手 |
| **T3 超越官方** | **≥28** | **本路线图的最终目标** |
| T4 挑战 PERF.md | ≥33 | 远期目标，可能受硬件限制 |

---

## 2. 当前瓶颈分解（数据驱动）

```
单步 decode wall = 0.184 s/tok = 5.4 t/s/u
                       │
        ├── rt.submit + wait =  57 ms  (device 计算 17.3 t/s/u)
        └── host 同步开销     = 127 ms  ← 最大单点瓶颈
                │
                ├── 64 次 rt.to_layout(KV)        ≈ 80~90 ms
                ├── 64 次 rt.deallocate_tensor   ≈ 20~30 ms
                ├── ~99 次 dict 查找 + numpy + tensor 构造 ≈ 10 ms
                └── tokenizer.decode + argmax + ...   ≈ 5 ms

官方 tt_transformers stack: 23.2 t/s/u = 43 ms/tok
        ├── KV stays put（in-place update，无 to_layout 循环）
        ├── flash attention + DRAM prefetcher
        └── bfp4_b weight-only quantization
```

**两个完全独立的方向都要做**：

- **Host 同步**：184 ms/tok → ~50 ms/tok（让 wall ≥18 t/s/u，逼近 device-only）
- **Device 计算**：17.3 → ≥25 t/s/u（超越官方）

任一孤立做都不够：host 不解决，device 再快 wall 也只有 5；device 不解决，host 再快也只有 17。

---

## 3. Buddy 前端图相对 ttmlir-opt / ttnn 的独到杠杆

| 能力 | Buddy 前端图 | ttmlir-opt | ttnn runtime |
|---|:---:|:---:|:---:|
| 模型结构感知（"32 个 transformer block"） | **✅** | ❌ | ❌ |
| Python fx pattern matching | **✅** | ❌ | ❌ |
| 跨子图分析（prefill / decode 共享 weight） | **✅** | ❌ | ❌ |
| Tensor 生命周期 / alias 分析 → in-place hint | **✅** | 部分 | ❌ |
| 量化（dtype 重写、scale/zp 注入） | **✅** | 部分 | ❌ |
| 算子级 fusion（permute+matmul、softmax+matmul） | 部分 | **✅** | ❌ |
| 内存放置（DRAM/L1/sharding） | ❌ | **✅** | 部分 |
| Kernel 调度 / aiclk | ❌ | 部分 | **✅** |

**结论**：本路线图集中在前 5 行——这是 Buddy 独到的领域；后 3 行让给
ttmlir-opt 与 ttnn runtime。

---

## 4. 现状梳理：为什么 Llama 没走 GQA fusion

`frontend/Python/graph/transform/fuse_ops.py::gqa_attention_fusion_check`
当前要求 K/V 分支必须是：

```
View ← Clone ← Expand ← Unsqueeze ← IndexPut
```

但我们为绕过 buddy 没注册的 `aten.index_copy_`，把 StaticCache 的
update 改写成了 `where(mask, expanded_new_kv, past_kv)` 形式（见
`buddy-llama31-lower-ttir.py::_patch_static_cache_for_buddy`）。

→ Llama-3.1 decode TTIR 里 32 层 SDPA **全部 miss fusion**，每层都是
裸的 SDPA + 多 view + clone + where——device-only 17.3 t/s/u 的最大根源。

这条信息直接决定了 P5 的优先级。

---

## 5. 优化路线（6 个 Phase，按 ROI 排序）

> 每个 Phase 包含：根因 / Buddy 前端要做的事 / 预期收益 / 工作量 / 风险 / 验证准则。

### 🔥 P4. KV alias 标注 + host-side runtime 批处理

| 项 | 内容 |
|---|---|
| 目标 | wall 5.4 → **≥15 t/s/u**，device-only 不变 |
| 工作量 | ~2 天 |
| 风险 | 低 |

**根因**：每步 decode 后端把 64 个 KV output 拷回 host layout（`rt.to_layout`）
+ 强制 dealloc 上一步 KV（防 DRAM 碎片）+ 重新上传作为下一步 input——
但实际上 input/output KV **完全 alias**（同一个 cache slot 写入并读出）。

**Buddy 前端要做的事**：

1. `frontend/Python/graph/ttir_import.py`：lowering decode subgraph 时，
   对每个 KV output tensor 加 attribute `tt.in_place_with = #arg{N}`，
   N 是对应 input KV 的 slot 序号。这是给 ttmlir-opt allocator 的 alias
   hint，让它复用同一 DRAM buffer。
2. `examples/BuddyLlama31-8B/llama31_chat_prepare.py`：输出新文件
   `kv_alias_map.json`，格式 `{output_idx: input_slot}`。
3. `examples/BuddyLlama31-8B/llama31_chat_run.py::_decode_once`：input
   KV 直接复用上一步 output KV 的 device handle，跳过 `rt.to_layout`
   和 `rt.deallocate_tensor`。

**验证**：跑 200-token decode，wall t/s/u ≥ device-only × 0.95，
端到端 PCC 与 HF golden 不退化。

**回退**：如果 ttmlir-opt 不认 `tt.in_place_with` attr，至少能在 chat_run
层面跳过冗余 to_layout（次优但仍有 30~50 % wall 提升）。

---

### 🔥 P5. 扩展 GQA fusion，让 Llama 走 fused op

| 项 | 内容 |
|---|---|
| 目标 | device-only 17.3 → **~22 t/s/u** |
| 工作量 | ~3-4 天 |
| 风险 | 中（fused op 数值正确性需谨慎守门） |

**根因**：见第 4 节——`where`-based scatter pattern 不被识别。

**Buddy 前端要做的事**：

1. `fuse_ops.py::gqa_attention_fusion_check` 增加新 pattern：
   ```
   View ← (where(mask, expand(unsqueeze(new_k)), past_k))
       | View ← Clone ← Expand ← Unsqueeze ← IndexPut   (现有路径)
   ```
2. `GQAAttentionFusedOp` 的 lowering（`ops/ttir_llm.py`）同时支持两种
   input shape（DeepSeek IndexPut + Llama where）。
3. `tests/Python/dialects/test_llama31_gqa_fusion.py`：单层 fixture，
   验证 fused vs unfused PCC ≥ 0.99。

**预期产出**：32 个 `ttir.scaled_dot_product_attention` + 96 个
`ttir.where` + 大量 reshape/permute → 32 个 `ttir.gqa_attention_fused`，
合并 ~1024 个中间 op。

**验证**：端到端 200-token decode top-1 token-match 率 ≥ 95%；
device-only ≥ 22 t/s/u。

---

### 🔥 P6. Decode 子图 CSE：合并冗余 cache_position / mask

| 项 | 内容 |
|---|---|
| 目标 | device-only +5~10 % |
| 工作量 | ~2 天 |
| 风险 | 低 |

**根因**：当前 decode TTIR 有 **34 个 `tensor<1xi64>` cache_position
placeholder**。每个独立 broadcast 成 `arange(L) == cache_position` mask。
32 倍重复工作。同理 `1.0 / sqrt(head_dim)` scale、`causal_mask` 等所有
layer-invariant 常量。

**Buddy 前端要做的事**：

1. lowering 时识别"所有 i64[1] placeholder 实际来自同一个
   `cache_position` python arg"，合并为单个 placeholder。
2. 把 mask 计算 hoist 到 subgraph 入口，32 层共享。
3. 同理 scale / causal_mask 的 CSE。

**实现位置**：`frontend/Python/graph/transform/decode_cse.py`（新建），
或扩展 `useless_op_eliminate.py`。

**验证**：decode TTIR 文件大小下降；端到端 PCC 不变；device-only +5~10 %。

---

### 🟡 P7. Weight-only 量化 (W8A16 / W4A16)

| 项 | 内容 |
|---|---|
| 目标 | device-only → **30+ t/s/u** |
| 工作量 | ~5-7 天 |
| 风险 | 中-高（量化误差累积 32 层） |

**根因**：DRAM 带宽是 decode 的硬瓶颈，weight 体积 ÷2/÷4 直接换吞吐。
官方栈 33.6 t/s/u 就是靠 bfp4_b 量化达成的。

**起点**：`frontend/Python/graph/transform/quantization/weight_only_channel_wise.py`
框架已具备 weight-only channel-wise 量化能力。

**Buddy 前端要做的事**：

1. 把 `weight_only_channel_wise` 应用到 Llama 32 层
   q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj。
2. `ops/ttir.py`：实现 `ttir.dequantize`（i8/i4 → bf16，per-channel scale）。
3. 阶梯式验证：先 W8A16（PCC 通常 > 0.98），再 W4A16（> 0.95）。
4. 如果 W4A16 整模型 PCC 不达标，至少量化 lm_head 单层（1 GB → 500 MB
   或 250 MB），收益已显著。

**验证**：端到端 PCC ≥ 0.95（w.r.t. HF bf16 golden）；
chat 200-token 生成的语义连贯性人工检查；device-only ≥ 30 t/s/u。

---

### 🟡 P8. Prefill / Decode 共享 weight buffer

| 项 | 内容 |
|---|---|
| 目标 | DRAM 32 GB → 16 GB；chat 启动 5min → 2min |
| 工作量 | ~2 天 |
| 风险 | 低 |

**根因**：当前 prefill_cache 与 decode_cache **各上传一份 16 GB weight
到 DRAM**，但两套 weight tensor 完全相同。

**Buddy 前端要做的事**：

1. `chat_prepare`：给 prefill / decode weight slot 加 `weight_id`（hash
   of tensor data），相同 hash 共享同一 `w_xxxx.npy`。
2. `chat_run`：weight 只上传一次，给两个 program 的对应 slot 都 retain
   同一 device handle。
3. 如果 prefill / decode 需要不同 layout，提前生成两份 layout 的 device
   tensor（仍只占 weight 一次的 DRAM）。

**收益不直接体现在 t/s/u**，但是 P9（更大 context / trace）的前置条件。

---

### 🟢 P9. Static-shape 特化 + sharding hint

| 项 | 内容 |
|---|---|
| 目标 | device-only +5~10 % |
| 工作量 | ~3 天 |
| 风险 | 高（容易选错 sharding 反而变慢） |

**Buddy 前端要做的事**：

1. cache_len=1024 是已知静态值，可以更激进特化：把
   `[1, 32, 1, 1024]` attention scores 直接 partition 到 8 个 core。
2. 给关键 matmul（q_proj / o_proj / down_proj）加
   `tt.sharding = #...` attr，绕过 ttmlir-opt 的 default 选择。

放在最后是因为这是 ttmlir 强项，Buddy 前端只能"建议"——需要 microbenchmark
逐 op 验证，否则容易反向优化。

---

### 🔵 P10（远期）. Speculative decoding / Multi-stream pipelining

只在 P4-P9 都做完且离 30 t/s/u 仍差距明显时才考虑。需要训一个 1.5B
draft model 对齐 8B 的 tokenizer，工程量大。

---

## 6. 推荐执行顺序与里程碑

| 顺序 | Phase | 预期 wall t/s/u | 预期 device t/s/u | 累计 | 状态 |
|---|---|---|---|---|---|
| 当前 | — | 5.4 | 17.3 | 0 d | ✅ 已 commit `19a49ff` |
| 1 | P4: host 同步收敛 | **17~18** | 17.3 | +2 d | ⏳ 待启动 |
| 2 | P5: GQA fusion | **22~23** | 22~23 | +5 d | ⏳ |
| 3 | P6: decode CSE | 23~24 | 23~24 | +7 d | ⏳ |
| 4 | P8: shared weight | 23~24 | 23~24 | +9 d | ⏳ |
| 5 | P7: W8A16 量化 | **28~32** | **28~32** | +14 d | ⏳ |
| 6 | P9: sharding 调优 | **30~35** | 30~35 | +17 d | ⏳ |

**关键里程碑**：

- **里程碑 A（步骤 2 完成）**：wall 首次 **持平或超过官方栈 23.2 t/s/u**，
  即 device 已用满（75 % → 95 %）+ host 收敛。
- **里程碑 B（步骤 5 完成）**：理论 28-32 t/s/u，**显著超越官方
  baseline 23.2**（路线图的核心目标 T3）。
- **里程碑 C（步骤 6 完成）**：尝试摸到官方 PERF.md 的 33.6 t/s/u。

---

## 7. Sprint 1（首周可动手）

为了快速验证整条路线可行，Sprint 1 只做最小闭环：

1. **Profile 127 ms/tok host 开销精确分解**（半天）：
   - 用 `time.perf_counter_ns` 在 `_decode_once` 每步插桩。
   - 区分 `to_layout` / `dealloc` / numpy / tokenizer.decode / 其它。
   - 输出 `examples/BuddyLlama31-8B/profile_decode_step.csv`。
2. **P4 实施**（2 天）：按上述方案改 `chat_run` + 输出 alias hint。
3. **P5 fixture-only**（1 天）：单层 GQA fusion 反例，先不动 Llama 主流程，
   只在 `tests/Python/dialects/` 加最小 reproduce。

**Sprint 1 出口标准**：

- ✅ wall ≥ 15 t/s/u（P4 见效）
- ✅ GQA fusion fixture PCC ≥ 0.99（P5 路线可行）

任一不达标 → 路线需要重新评估。

---

## 8. 进度追踪表

| 日期 | Phase | commit | wall t/s/u | device t/s/u | 备注 |
|---|---|---|---|---|---|
| 2026-04-24 | T0 起点 | `19a49ff` | 5.4 | 17.3 | TTIR e2e PoC |
| TBD | Sprint1 profile | | 5.4 | 17.3 | host 开销精确分解 |
| TBD | P4 | | | | KV alias |
| TBD | P5 | | | | GQA fusion |
| ... | ... | ... | ... | ... | ... |

每个 Phase 完成后追加一行，commit hash + 实测数据。

---

## 9. 风险登记 & 退路

| 风险 | 影响 | 退路 |
|---|---|---|
| ttmlir-opt 不认 `tt.in_place_with` attr | P4 device alias 失效 | 仅做 chat_run 层 host 优化（仍能拿 30~50 % wall 提升） |
| Llama 的 `where`-scatter pattern 在 fused op lowering 出现数值偏差 | P5 PCC 退化 | 退回未 fused 版本，转向 P6/P7 |
| W4A16 量化整模型 PCC < 0.9 | P7 chat 输出质量不可接受 | 降级到 W8A16 或仅量化 lm_head |
| 官方栈 23.2 t/s/u 实际偏低（aiclk 不稳） | T2 里程碑虚高 | 复测官方栈每周 1 次，记录在 BASELINE.md |
| ttnn runtime 升级导致 API 变更 | 全局 | 在 `_env.sh` 锁定 ttmlir-opt / ttrt 版本 commit |

---

## 10. 引用与依赖

- 当前 baseline 数据：`/wafer/zhuxinye/llama_tt_transformers/BASELINE.md`
- 当前 TTIR e2e 实现：`examples/BuddyLlama31-8B/`（commit `19a49ff`）
- 总览运行指南：`examples/TTIR_E2E_RUN_GUIDE.md`
- Buddy 前端图基础设施：`frontend/Python/graph/`、
  `frontend/Python/graph/transform/`、`frontend/Python/ops/`
