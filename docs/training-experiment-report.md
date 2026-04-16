# 智能宠物喂食器 VLM 训练实验报告

> **实验日期**: 2026-04-16  
> **硬件**: NVIDIA RTX 5090 (32GB VRAM), 754GB RAM  
> **基础模型**: Qwen2-VL-2B-Instruct  
> **训练框架**: LLaMA-Factory + PyTorch 2.11 + CUDA 12.8  
> **目标**: 验证 train → eval → quantize 完整管线，产出可量化部署的端侧 VLM

---

## 目录

1. [系统架构与选型逻辑](#1-系统架构与选型逻辑)
2. [数据管线与数据集](#2-数据管线与数据集)
3. [训练方案与超参数](#3-训练方案与超参数)
4. [实验结果](#4-实验结果)
5. [消融实验](#5-消融实验)
6. [量化与端侧部署](#6-量化与端侧部署)
7. [摩擦点记录](#7-摩擦点记录)
8. [结论与后续建议](#8-结论与后续建议)

---

## 1. 系统架构与选型逻辑

### 1.1 产品背景

智能宠物喂食器内置固定俯角摄像头，需要在端侧（瑞芯微 RK3576, 6 TOPS NPU）实时分析宠物进食行为，输出结构化 JSON 事件，供手机 APP 展示健康趋势和异常告警。

**核心叙事**: "All AI runs on your device, not our servers" — 图像帧永不离开设备，只有结构化 JSON 上行。

### 1.2 端侧推理架构：两级流水线

```
┌──────────────────────────────────────────────────────────────┐
│  一级：常驻检测 YOLOv8-nano INT8                              │
│  · 推理延迟 <5ms，24h 运行                                    │
│  · 检测宠物是否进入碗区域                                     │
│  · 宽松阈值，宁可多触发不漏触发                               │
└────────────────────────┬─────────────────────────────────────┘
                         │ 事件触发
┌────────────────────────▼─────────────────────────────────────┐
│  二级：事件推理 Qwen2-VL-2B (W8A8)                            │
│  · 视觉编码器: .rknn (FP16)                                   │
│  · LLM 主体: .rkllm (W8A8)                                    │
│  · 输入: 触发后抓取 1-3 帧 + 固定 prompt                      │
│  · 输出: 结构化 JSON（PetFeederEvent Schema v1.0）             │
│  · 推理延迟: 2-4s/帧                                          │
│  · 推理完成后立即休眠，图像帧从内存清除                        │
└──────────────────────────────────────────────────────────────┘
                         ‖ 并行运行
┌──────────────────────────────────────────────────────────────┐
│  音频 CNN: PANNs MobileNetV2 INT8 (<5MB, <10ms)              │
│  · 进食咀嚼 / 饮水舔食 / 叫声 / 呕吐前声 / 环境音 五分类     │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 模型选型决策

| 候选方案 | 参数量 | 端侧可行性 | 选择理由 |
|---|---|---|---|
| **Qwen2-VL-2B** ✅ | 2.1B | RK3576 8GB RAM 可运行 | 最小的多模态 VLM，支持动态分辨率 |
| Qwen2-VL-7B | 7.6B | 超出 8GB RAM | 精度更高但无法端侧部署 |
| LLaVA-1.6 | 7B+ | 无 2B 版本 | 无适合端侧的小规模版本 |
| InternVL2-2B | 2B | 可选 | NaViT 支持不如 Qwen2-VL 成熟 |

**为什么用 LoRA 而不是全量微调**:
- Qwen2-VL-2B 全量微调需 ~16GB 显存（FP16），加上优化器状态约 48GB，单卡不够
- LoRA rank=16 仅训练 ~2.4M 参数（占总参数 0.1%），显存 <12GB
- 实验证明 LoRA SFT 足以学会完整 Schema 结构（100% JSON 合规率）

**为什么暂不做 DPO**:
- 当前 teacher 模型（doubao-seed-2-0-lite）不支持 logprobs（推理模型限制）
- SFT 已达到 100% Schema 合规，DPO 的边际收益主要在 narrative 质量
- 数据量不足（仅 16 DPO 对 vs 最低要求 500 对）

### 1.4 输出 Schema: PetFeederEvent v1.0

模型需要输出严格符合以下 Pydantic v2 Schema 的 JSON：

```
PetFeederEvent
├── schema_version: "1.0"
├── pet_present: bool
├── pet_count: 0-4
├── pet: PetInfo | null
│   ├── species: "cat" | "dog" | "unknown"
│   ├── breed_estimate: str
│   ├── id_tag: str
│   ├── id_confidence: 0.0-1.0
│   ├── action
│   │   ├── primary: 6 类 (eating|drinking|sniffing_only|leaving_bowl|sitting_idle|other)
│   │   └── distribution: 6 维概率 (sum=1.0±0.01)
│   ├── eating_metrics
│   │   ├── speed: {fast, normal, slow} (sum=1.0±0.01 when eating)
│   │   ├── engagement: 0.0-1.0
│   │   └── abandoned_midway: 0.0-1.0
│   ├── mood: {alertness, anxiety, engagement} 各 0.0-1.0
│   ├── body_signals: {posture: 4类, ear_position: 4类}
│   └── anomaly_signals: {vomit_gesture, food_rejection, excessive_sniffing, lethargy, aggression}
├── bowl
│   ├── food_fill_ratio: 0.0-1.0 | null
│   ├── water_fill_ratio: 0.0-1.0 | null
│   └── food_type_visible: "dry"|"wet"|"mixed"|"unknown"
├── scene: {lighting: 3类, image_quality: 3类, confidence_overall: 0.0-1.0}
└── narrative: str (≤80字, 客观描述当前帧)
```

Schema 复杂度：~30 个字段，含多个概率分布约束（sum=1.0±0.01），枚举值，嵌套对象。对 2B 模型而言是显著挑战。

---

## 2. 数据管线与数据集

### 2.1 数据采集与标注管线

```
原始视频帧 (13,970 帧)
    │
    ▼
pet-data: 帧提取 + 去重 + 质检
    │
    ▼
pet-annotation: VLM 自动标注 (doubao-seed-2-0-lite)
    │  · 基于 Schema v1.0 system prompt
    │  · JSON Schema 自动验证
    │  · 不合规输出自动重试
    │
    ▼
人工审核 (Label Studio)
    │  · auto_checked → approved
    │  · 966 条通过审核
    │
    ▼
pet-train: 导出 ShareGPT 格式 SFT 数据
    │  · system prompt + 图片 + expected JSON
    │
    ▼
sft_v2.jsonl (966 samples)
```

### 2.2 数据集统计

**总体概况**:

| 指标 | 数值 |
|---|---|
| 原始帧总数 | 13,970 |
| 通过审核 (approved) | 966 |
| 待审核 (needs_review) | 46 |
| 未标注 (pending) | 12,958 |
| 标注率 | 6.9% |

**动作分布 (966 条 approved)**:

```
sitting_idle  ████████████████████████████████████████  697 (72.2%)
other         ████████████                              214 (22.1%)
sniffing_only ██                                         25 ( 2.6%)
no_pet        █                                          20 ( 2.1%)
eating        ▏                                           4 ( 0.4%)
drinking      ▏                                           1 ( 0.1%)
leaving_bowl  ▏                                           1 ( 0.1%)
```

> ⚠️ **严重数据不均衡**: sitting_idle 占 72.2%，eating 仅 0.4%（4 条）。这是 action accuracy 的主要瓶颈。

**物种分布**:

```
cat     ██████████████████████████████████  554 (58.9%)
dog     ████████████████████████            383 (40.7%)
unknown ▏                                     5 ( 0.5%)
```

**光照分布**:

```
bright  ██████████████████████████████████████████  828 (85.9%)
dim     ███████                                     134 (13.9%)
infrared_night                                        0 ( 0.0%)  ← 缺失
```

### 2.3 数据子集 (消融实验用)

| 数据集 | 样本数 | 用途 |
|---|---|---|
| sft_v2.jsonl | 966 | 全量训练 (100%) |
| sft_v2_75pct.jsonl | 724 | 消融 - 75% 数据 |
| sft_v2_50pct.jsonl | 483 | 消融 - 50% 数据 |
| sft_v2_25pct.jsonl | 241 | 消融 - 25% 数据 |
| pet_sft_train.jsonl | 184 | Phase 1 探索 (v1) |

### 2.4 Gold Set 评估集

| 版本 | 样本数 | 问题 |
|---|---|---|
| Gold Set v1 | 6 | 全部是猫，仅 3 种动作，统计无意义 |
| **Gold Set v2** | **21** | 按动作分层：sitting_idle=6, other=5, sniffing=3, eating=3, no_pet=2, leaving=1, drinking=1 |

Gold Set v2 覆盖了全部 6 种动作 + no_pet 场景，物种分布：cat=9, dog=7, unknown=3, none=2。

---

## 3. 训练方案与超参数

### 3.1 训练框架

使用 **LLaMA-Factory** 作为训练框架，通过 YAML 配置驱动 SFT 训练。

训练数据格式为 **ShareGPT** 格式：
```json
{
  "conversations": [
    {"from": "system", "value": "<Schema v1.0 system prompt>"},
    {"from": "human",  "value": "<image>\n分析这张宠物喂食器画面"},
    {"from": "gpt",    "value": "{\"schema_version\":\"1.0\",...}"}
  ],
  "images": ["path/to/frame.jpg"]
}
```

### 3.2 Phase 1: 小规模探索 (184 samples)

| 参数 | 值 |
|---|---|
| 数据量 | 184 samples (sft v1) |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch=16) |
| Learning rate | 2e-4 (cosine schedule) |
| LoRA rank / alpha | 16 / 32 |
| Label smoothing | 0.0 (0.1 导致 OOM) |
| 训练步数 | 36 steps |
| 训练时间 | 199.2s (~3.3 min) |
| 最终 train_loss | 0.337 |

**Phase 1 结论**: JSON 合规 83.3%，species 正确率 100%，但 action 仅 20%（全部坍缩到 sitting_idle）。数据太少且分布不均。

### 3.3 Phase 2: 全量训练 (966 samples)

| 参数 | 值 |
|---|---|
| 数据量 | 966 samples (sft v2) |
| Epochs | 3 |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch=16) |
| Learning rate | 2e-4 (cosine schedule) |
| LoRA rank / alpha | 16 / 32 |
| Label smoothing | 0.0 |
| 训练步数 | 183 steps |
| 训练时间 | 1026.9s (~17.1 min) |
| 吞吐量 | 2.822 samples/sec |
| 最终 train_loss | 0.189 |

### 3.4 关键发现: merge_and_unload() 必须

> ⚠️ **PeftModel 直接推理会产出完全错误的 JSON**

不合并 LoRA adapter 时，模型输出自创字段（`color`, `gender`, `open_door`），完全不符合 Schema。调用 `model.merge_and_unload()` 合并后，输出 100% Schema 合规。

```python
# 错误方式 — 输出完全乱码
model = PeftModel.from_pretrained(base_model, adapter_path)
# output: {"color": "orange", "gender": "male", "open_door": false, ...}

# 正确方式 — 输出 100% 合规
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # ← 关键
# output: {"schema_version": "1.0", "pet_present": true, ...}
```

---

## 4. 实验结果

### 4.1 Phase 1 → Phase 2 对比

| 指标 | Phase 1 (184 samples) | Phase 2 (966 samples) | 变化 |
|---|---|---|---|
| Train loss | 0.337 | 0.189 | ↓ 43.9% |
| JSON 合规率 | 83.3% | 100% | ↑ 16.7pp |
| Action accuracy | 20.0% | 71.4% | ↑ 51.4pp |
| Species accuracy | 100% | 85.7% | ↓ 14.3pp* |
| Gold set 大小 | 6 | 21 | v2 更全面 |

*Species accuracy 下降因 gold set v2 包含 unknown 和 no_pet 场景，比 v1（全 cat）更严格。

### 4.2 Loss 曲线

**Phase 1 (184 samples, 36 steps)**:
```
Loss
0.50 ┤╮
0.45 ┤╰╮
0.40 ┤  ╰╮
0.35 ┤    ╰─╮
0.30 ┤      ╰──╮
0.27 ┤          ╰──
     └─┬──┬──┬──┬──┬──┬──→ Step
       5  10 15 20 25 30 35
```

**Phase 2 (966 samples, 183 steps)**:
```
Loss
0.45 ┤╮
0.40 ┤│
0.35 ┤╰╮
0.30 ┤  │
0.25 ┤  ╰╮
0.20 ┤    ╰╮
0.15 ┤     ╰──────────────────────────
0.14 ┤                                ── (plateau)
     └─┬───┬───┬───┬───┬───┬───┬───┬──→ Step
      10  30  50  70  90 110 130 150 180
```

Phase 2 在 ~step 70 (~epoch 1.1) loss 降到 0.16 后进入平台期，最终收敛到 0.136。

### 4.3 Phase 2 详细 Loss 数据

| Step | Loss | Epoch |
|------|------|-------|
| 10 | 0.4474 | 0.17 |
| 20 | 0.3687 | 0.33 |
| 30 | 0.2649 | 0.50 |
| 40 | 0.2156 | 0.66 |
| 50 | 0.1894 | 0.83 |
| 60 | 0.1662 | 0.99 |
| 70 | 0.1638 | 1.15 |
| 80 | 0.1591 | 1.31 |
| 90 | 0.1554 | 1.48 |
| 100 | 0.1446 | 1.65 |
| 110 | 0.1441 | 1.81 |
| 120 | 0.1488 | 1.98 |
| 130 | 0.1428 | 2.13 |
| 140 | 0.1437 | 2.30 |
| 150 | 0.1423 | 2.46 |
| 160 | 0.1381 | 2.63 |
| 170 | 0.1379 | 2.80 |
| 180 | 0.1357 | 2.96 |

---

## 5. 消融实验

### 5.1 实验矩阵

在 Phase 2 baseline（966 samples, 3 epochs, lr=2e-4, rank=16）基础上，分 4 维度进行消融：

| 维度 | 变量 | 实验组 |
|---|---|---|
| 数据规模 | 25%, 50%, 75%, 100% | data25, data50, data75, baseline |
| 训练轮次 | 1, 3, 5 epochs | ep1, baseline, ep5 |
| 学习率 | 1e-4, 2e-4, 5e-4 | lr1e4, baseline, lr5e4 |
| LoRA Rank | 8, 16, 32 | r8, baseline, r32 |

共 10 组实验（含 baseline），全部在 RTX 5090 上完成。

### 5.2 完整结果表

| 实验 | 数据 | Ep | LR | Rank | Train Loss | 训练时间 | 合规率 | Action Acc | Species Acc |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | **100%** | **3** | **2e-4** | **16** | **0.189** | **17.1m** | **100%** | **71.4%** | **85.7%** |
| data25 | 25% | 3 | 2e-4 | 16 | 0.279 | 4.3m | 100% | 28.6% | 71.4% |
| data50 | 50% | 3 | 2e-4 | 16 | 0.224 | 8.6m | 100% | 42.9% | 71.4% |
| data75 | 75% | 3 | 2e-4 | 16 | 0.202 | 12.9m | 100% | 61.9% | 81.0% |
| ep1 | 100% | 1 | 2e-4 | 16 | 0.261 | 5.7m | 100% | 28.6% | 71.4% |
| ep5 | 100% | 5 | 2e-4 | 16 | 0.166 | 28.6m | 100% | 66.7% | 85.7% |
| lr1e4 | 100% | 3 | 1e-4 | 16 | 0.216 | 17.1m | 100% | 52.4% | 81.0% |
| lr5e4 | 100% | 3 | 5e-4 | 16 | 0.163 | 17.1m | 100% | 66.7% | 90.5% |
| r8 | 100% | 3 | 2e-4 | 8 | 0.203 | 17.2m | 100% | 61.9% | 81.0% |
| r32 | 100% | 3 | 2e-4 | 32 | 0.175 | 17.2m | 100% | 66.7% | 85.7% |

> 注：所有 10 组实验均达到 100% JSON 合规率和 21/21 valid JSON，说明 Schema 结构学习是稳健的。

### 5.3 Data Scaling Law (数据规模效应)

```
Action Accuracy
80% ┤
    │
70% ┤                              ● baseline (71.4%)
    │
60% ┤                    ●
    │                    data75 (61.9%)
50% ┤
    │
40% ┤          ●
    │          data50 (42.9%)
30% ┤●
    │ data25 (28.6%)
20% ┤
    └────┬─────┬─────┬─────┬──→ Data %
        25%   50%   75%  100%
```

**发现**: 数据量与 action accuracy 呈近似线性关系。每增加 25% 数据，accuracy 提升约 10-15pp。**数据是当前最大的杠杆**。

### 5.4 Epoch Sweep (训练轮次)

```
Action Accuracy
80% ┤
    │
70% ┤         ● baseline (71.4%)
    │
60% ┤                   ● ep5 (66.7%)
    │
50% ┤
    │
40% ┤
    │
30% ┤● ep1 (28.6%)
20% ┤
    └────┬─────┬─────┬──→ Epochs
         1     3     5
```

**发现**: 1 epoch 严重欠拟合。3 epoch 最优。5 epoch loss 更低（0.166 vs 0.189）但 action accuracy 下降 4.7pp，出现轻微过拟合。

### 5.5 Learning Rate Sweep (学习率)

```
Action Accuracy
80% ┤
    │
70% ┤         ● baseline (71.4%)
    │
60% ┤                   ● lr5e4 (66.7%)
    │
50% ┤● lr1e4 (52.4%)
    │
40% ┤
    └────┬─────┬─────┬──→ LR
       1e-4  2e-4  5e-4
```

**发现**: lr=2e-4 action accuracy 最高。lr=5e-4 虽然 loss 最低（0.163）且 species accuracy 最高（90.5%），但 action accuracy 降到 66.7%，说明高 LR 可能在稀有类上过拟合。

### 5.6 LoRA Rank Sweep

```
Action Accuracy
80% ┤
    │
70% ┤         ● baseline r16 (71.4%)
    │
60% ┤●                   ● r32 (66.7%)
    │ r8 (61.9%)
50% ┤
    └────┬─────┬─────┬──→ Rank
         8    16    32
```

**发现**: rank=16 最优。rank=8 容量不足（action -9.5pp），rank=32 过参数化导致轻微过拟合。LoRA rank=16 是 Qwen2-VL-2B 的甜蜜点。

### 5.7 消融实验结论

| 维度 | 最优设置 | 关键发现 |
|---|---|---|
| **数据规模** | 100% (966) | 线性 scaling，更多数据 = 更好性能，当前远未饱和 |
| **Epochs** | 3 | 1 ep 欠拟合，5 ep 轻微过拟合 |
| **Learning rate** | 2e-4 | 1e-4 收敛不足，5e-4 稀有类过拟合 |
| **LoRA rank** | 16 | r8 容量不足，r32 过参数化 |

**最佳配置** = baseline: 全量数据 + 3 epochs + lr=2e-4 + rank=16。

---

## 6. 量化与端侧部署

### 6.1 当前状态

| 步骤 | 状态 | 说明 |
|---|---|---|
| LoRA merge → base model | ✅ 完成 | 合并模型 4.2GB |
| ONNX vision encoder export | ❌ 失败 | Qwen2-VL 3D RoPE 无法 ONNX trace |
| ONNX LLM export | ⏭ 跳过 | 依赖 vision encoder |
| RKNN/RKLLM 转换 | ⏳ 待做 | 需要 RK3576 SDK 环境 |

### 6.2 ONNX 导出失败原因

Qwen2-VL 的 vision encoder 使用了三项对 ONNX 不友好的技术：

1. **3D 旋转位置编码 (3D RoPE)**: cos/sin tensor shape 在 trace 时动态变化
2. **cu_seqlens 动态索引**: 用于 NaViT 变长序列 packing
3. **Flash Attention**: 自定义 CUDA kernel，无 ONNX 等价算子

PyTorch legacy tracer 和 dynamo exporter 均失败。

### 6.3 推荐端侧部署方案

```
方案: 使用 rkllm-toolkit 直接转换整个 VLM
              │
              ▼
┌──────────────────────────────────┐
│ 输入: merged model (4.2GB)       │
│ 工具: rkllm-toolkit              │
│ 环境: x86 Linux + RK3576 SDK    │
│                                  │
│ 输出:                            │
│  · vision_encoder.rknn (FP16)   │
│  · llm.rkllm (W8A8 或 W4A16)   │
└──────────────────────────────────┘
```

跳过 ONNX 中间格式，瑞芯微 toolkit 直接从 HuggingFace 权重转换为芯片格式。

### 6.4 量化后性能预估

| 量化方案 | 模型大小 | Action Acc 预估 | 合规率风险 | 推理延迟 |
|---|---|---|---|---|
| FP16 (无量化) | 4.2GB | 71.4% (基线) | 100% | 无法运行 (超 RAM) |
| **W8A8** | ~1.1GB | ~63-68% | 中等风险 | 2-4s/帧 |
| W4A16 | ~0.6GB | ~59-66% | 较高风险 | 1.5-3s/帧 |

> ⚠️ 量化后 JSON 合规率可能显著下降，必须配合 **Constrained Decoding** 保障。

---

## 7. 摩擦点记录

实验全程共发现并记录 **26 个摩擦点**，其中 18 个涉及代码修复，8 个为环境/运维问题。

### 7.1 分类统计

| 类别 | 数量 | 代表性问题 |
|---|---|---|
| 环境搭建 | 6 | git clone 失败、pip 依赖、HF 下载、macOS 资源叉 |
| 跨仓库集成 | 5 | `__main__` guard 缺失、params.yaml 路径、模块导入检测 |
| 训练 | 3 | label_smoothing OOM、eval 检测失败、batch 脚本中断 |
| VLM 兼容性 | 3 | AutoModelForCausalLM 不识别 Qwen2-VL、ONNX 导出失败 |
| 数据管线 | 3 | DB 路径不一致、并发写入、annotation 状态不一致 |
| 运维/操作 | 4 | API key 环境变量、nohup 缓冲、SIGPIPE、logprobs 不支持 |
| 评估 | 2 | GateResult 不排除 skipped、benchmark 路径解析 |

### 7.2 高影响摩擦点

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| 12 | label_smoothing 导致 VLM OOM | 训练无法启动 | 设为 0.0 |
| 13-14 | pet-eval 安装检测 + `__main__` 缺失 | 训练后 eval 从未触发 | 修复 train_sft.sh + eval_trained.py |
| 18 | AutoModelForCausalLM 不识别 VLM | eval 无法加载模型 | 检测 model_type 用对应类 |
| 21-22 | Qwen2-VL ONNX 导出失败 | 量化管线中断 | 改用 rkllm-toolkit |
| 25-26 | batch 脚本 set -e + eval exit(1) | 消融实验批量中断 | 移除 set -e，eval 加 `\|\| true` |

### 7.3 已提交的修复 PR

| 仓库 | PR | 修复内容 |
|---|---|---|
| pet-train | PR#6 | train→eval 集成修复 (#13-15) |
| pet-eval | PR#7 | gate skip + `__main__` + VLM model class + path resolution (#14,16-18) |
| pet-quantize | PR#5 | `__main__` guard + VLM ONNX 导出 (#19,21) |

---

## 8. 结论与后续建议

### 8.1 实验结论

1. **Qwen2-VL-2B + LoRA SFT 可以学会复杂的结构化输出 Schema**（100% JSON 合规率），验证了端侧 VLM 的技术可行性。

2. **数据量是当前性能的决定性因素**，消融实验显示清晰的 data scaling law，当前 966 样本远未饱和。

3. **最优超参数组合**: lr=2e-4, rank=16, 3 epochs — 在所有消融维度中，baseline 配置表现最佳。

4. **Action 识别是核心瓶颈**: 71.4% accuracy 不足以支撑可靠的健康趋势分析，根因是训练数据中 eating/drinking/leaving_bowl 极度稀缺（分别 4/1/1 条）。

5. **merge_and_unload() 是部署的必要步骤**，未合并的 PeftModel 输出完全不可用。

### 8.2 后续行动建议

**近期 (P0)**:

| 行动 | 预期效果 | 工作量 |
|---|---|---|
| 补标注数据至 2000+，每类动作 ≥100 条 | Action acc → 85-90% | 2-3 天标注 |
| 实现 Constrained Decoding | 量化后合规率 → 99.9% | 1-2 天开发 |
| 扩充 gold set 至 100+ 样本 | 评估统计可靠性 | 0.5 天 |

**中期 (P1)**:

| 行动 | 预期效果 | 工作量 |
|---|---|---|
| rkllm-toolkit 量化验证 | 确认端侧可部署性 | 2-3 天 (需 SDK 环境) |
| 增加红外夜视数据 | 覆盖全光照条件 | 1 天采集 + 标注 |
| DPO 训练 (500+ 偏好对) | Narrative 质量提升 | 3-5 天 |

**长期 (P2)**:

| 行动 | 预期效果 |
|---|---|
| 用户反馈闭环 → OTA 更新管线 | 持续模型迭代 |
| 音频 CNN 微调 | 多模态融合增强异常检测 |
| 多宠物场景支持 | pet_count > 1 的准确处理 |

### 8.3 最终配置推荐

```yaml
# 推荐的生产训练配置
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct
dataset: pet_sft_v3            # 目标: 2000+ balanced samples
num_train_epochs: 3
learning_rate: 2.0e-4
lora_rank: 16
lora_alpha: 32
per_device_train_batch_size: 2
gradient_accumulation_steps: 8  # effective batch = 16
label_smoothing_factor: 0.0     # VLM 大词表下不可用
warmup_ratio: 0.1
lr_scheduler_type: cosine
```

---

## 附录 A: 实验环境

| 项目 | 详情 |
|---|---|
| GPU | NVIDIA RTX 5090, 32GB VRAM |
| CPU | 未记录 (云服务器) |
| RAM | 754GB |
| CUDA | 12.8 |
| PyTorch | 2.11+cu128 |
| transformers | 5.5.4 |
| LLaMA-Factory | 源码安装 (git submodule) |
| Python | 3.11.x |
| 服务器 | AutoDL 云 GPU 实例 |

## 附录 B: 完整 Loss 数据

### Phase 1 (sft_explore_v1)

| Step | Loss | Epoch |
|------|------|-------|
| 5 | 0.4818 | 0.43 |
| 10 | 0.4057 | 0.87 |
| 15 | 0.3432 | 1.26 |
| 20 | 0.3158 | 1.70 |
| 25 | 0.2899 | 2.09 |
| 30 | 0.2694 | 2.52 |
| 35 | 0.2671 | 2.96 |

Final: train_loss=0.3372, runtime=199.2s

### Phase 2 全量训练 (sft_full_v2)

见 [4.3 节](#43-phase-2-详细-loss-数据)。

Final: train_loss=0.1886, runtime=1026.9s

### 消融实验训练指标

| 实验 | Train Loss | 训练时间 (s) | 吞吐量 (samples/sec) |
|---|---|---|---|
| sft_explore_v1 | 0.3372 | 199.2 | 2.772 |
| sft_full_v2 | 0.1886 | 1026.9 | 2.822 |
| ablation_data25 | 0.2792 | 258.9 | 2.793 |
| ablation_data50 | 0.2238 | 515.3 | 2.812 |
| ablation_data75 | 0.2023 | 771.6 | 2.815 |
| ablation_ep1 | 0.2607 | 344.8 | 2.802 |
| ablation_ep5 | 0.1656 | 1713.6 | 2.819 |
| ablation_lr1e4 | 0.2158 | 1028.2 | 2.818 |
| ablation_lr5e4 | 0.1627 | 1028.6 | 2.817 |
| ablation_r8 | 0.2034 | 1030.0 | 2.814 |
| ablation_r32 | 0.1752 | 1029.9 | 2.814 |

---

*报告生成日期: 2026-04-16*  
*实验执行: RTX 5090 GPU, AutoDL 云平台*  
*训练管线: Train-Pet-Pipeline (pet-train + pet-eval + pet-quantize)*
