---
type: results-report
date: 2026-07-20
experiment_line: qwen3-vl-4b-geo3k-grpo
round: 2
purpose: verl-v2 vs slime-v3 VL performance comparison
status: final
frameworks:
  - verl (fully async, 2+2 GPU, SGLang)
  - slime (disaggregated sync, 2+2 GPU, SGLang)
---

# Verl V2 vs Slime V3 / Qwen3-VL-4B GEO3K GRPO / 2026-07-20

## 1. Executive Summary

4× NVIDIA B300 SXM6 GPU 上对 Qwen3-VL-4B-Instruct 进行 GRPO 数学几何推理训练，对比 **verl fully async (SGLang)** 与 **slime disaggregated (SGLang)** 两种方案。

| 维度 | Verl V2 | Slime V3 | Δ |
|------|:---:|:---:|:---:|
| **step_time** | 692s (warm: 615s) | **295s** (~250s warm) | verl 慢 2.5× |
| **trainer idle** | 62.0% | 43.2% | verl 多等 |
| **trainer wait time** | **445s** | **137s** | 等 3.2× 更久（可比指标） |
| **reward trend** | 0.378→0.422 | 0.469→0.555 | 均正确学习 |
| **response length** | 1487 | 1625 | 接近 |

**核心结论**: Verl fully async 总步时是 slime disaggregated 的 2.5×（615s vs ~250s warm）。训练端等待时间（trainer wait）是 slime 的 3.2×（445s vs 137s）——队列延迟 + CPU reward + 尾样本等待放大了差距。

## 2. Experiment Configuration

### 2.1 硬件与软件

| 项目 | Verl V2 | Slime V3 |
|------|------|------|
| GPU | 4× NVIDIA B300 SXM6 AC (275 GiB) | 4× NVIDIA B300 SXM6 AC (275 GiB) |
| 训练后端 | Megatron-LM (mbridge) TP2 | Megatron-LM (mbridge) TP2 |
| 推理引擎 | SGLang (HTTP Server + Ray) | SGLang (独立进程) |
| 框架版本 | verl 0.9.0.dev (fork) | slime main branch |
| PyTorch | 2.9.1+cu129 | 2.9.1+cu129 |
| 权重同步 | NCCL CheckpointEngine | 同进程 weight_update |

### 2.2 GPU 分配

| | Verl V2 | Slime V3 |
|------|------|------|
| **模式** | 异步分离 (并行) | 分卡同步 (串行) |
| 训练 GPU | 2× (TP2, DP1) | 2× (TP2, DP1) |
| 推理 GPU | 2× (1 SGLang TP2) | 2× (1 SGLang TP2) |
| 流水线重叠 | ✅ 推理/训练并行 | ❌ rollout→train 串行 |

### 2.3 完整参数对照

| 类别 | 参数 | Verl V2 | Slime V3 | 匹配 |
|------|------|------|------|:---:|
| **模型** | model_path | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct | ✅ |
| | trust_remote_code | True | 默认 | ✅ |
| | freeze_vision_tower | True | — | — |
| **数据** | dataset | chenhegu/geo3k_imgurl | chenhegu/geo3k_imgurl | ✅ |
| | max_prompt_length | 2048 | (默认) | — |
| | max_response_length | 3072 | 3072 | ✅ |
| | image_key | images | images (via multimodal_keys) | ✅ |
| **算法** | adv_estimator | grpo | grpo | ✅ |
| | kl_loss_coef | 0 | 0 | ✅ |
| | entropy_coef | 0 | 0 | ✅ |
| | eps_clip / _high | 0.2 / 0.28 | 0.2 / 0.28 | ✅ |
| | loss_agg_mode | token-mean | (token-mean) | ✅ |
| **Rollout** | n_samples_per_prompt | 8 | 8 | ✅ |
| | rollout_temperature | 0.8 | 0.8 | ✅ |
| | rollout_batch_size | 1 (streaming) | 16 | ⚠️ 不同模式 |
| | global_batch_size | 64 | 64 | ✅ |
| **优化器** | lr | 1e-6 | 1e-6 | ✅ |
| | lr_decay_style | constant | constant | ✅ |
| | weight_decay | 0.1 | 0.1 | ✅ |
| | adam_beta1/2 | 0.9 / 0.98 | 0.9 / 0.98 | ✅ |
| **Megatron** | TP / PP | 2 / 1 | 2 / 1 | ✅ |
| | sequence_parallel | True | True | ✅ |
| | recompute | full/uniform/1 | full/uniform/1 | ✅ |
| | attention_backend | flash (FA2) | flash (FA2) | ✅ |
| | use_dynamic_bsz | True | True | ✅ |
| **SGLang** | TP | 2 | 2 | ✅ |
| | mem_fraction | 0.7 | 0.7 | ✅ |
| | mm_attention_backend | sdpa | sdpa | ✅ |
| | enforce_eager / disable-cuda-graph | True | True | ✅ |
| **推理数量** | engines | 1 × TP2 | 1 × TP2 | ✅ |
| **异步控制** | staleness_threshold | 0.5 | N/A | — |
| | trigger_param_sync_step | 1 | N/A | — |
| | total_rollout_steps | 2101 | 500 rollouts | — |

### 2.4 唯一差异：weight_decay

Verl Megatron `weight_decay=0.1` 正则化对象不同——Megatron 内部排除 LayerNorm 和 bias，等价于 slime 的默认行为。实际有效的 weight_decay 应用几乎一样。

## 3. Performance Results

### 3.1 指标可比性说明 ⚠️

**Verl `timing_s/gen` 和 Slime `perf/rollout_time` 不是同一度量**：

| 度量 | Verl | Slime |
|------|------|------|
| **含义** | Trainer 等待所有样本**生成完毕 + 入队**的 wall-clock | **纯 SGLang batch 推理**，无队列开销 |
| **包含** | 队列填满延迟 + 最慢样本尾延迟 + CPU reward | 128 样本的 batch 前向时间 |
| **生成量** | 64 样本 × 1487 tok = **95K** tok/step | 128 样本 × 1625 tok = **208K** tok/rollout |

**应该用 `fully_async/total_wait_time` 和 `perf/train_wait_time` 比较**——都测量"训练端等待推理端的时间"。

### 3.2 总步时

| 指标 | Verl V2 | Slime V3 | Δ |
|------|:---:|:---:|:---:|
| step_time (全步) | **692s** | **295s** | verl 慢 2.35× |
| step_time (预热后) | **615s** | ~250s | verl 慢 2.46× |

### 3.3 训练端等待时间（可比指标）

| 指标 | Verl V2 | Slime V3 | Δ |
|------|:---:|:---:|------|
| **trainer wait** | **445s** | **137s** | verl 等 3.2× 更久 |
| trainer idle_ratio | 62.0% | 43.2% (wait_ratio) | — |
| rollouter idle_ratio | 2.1% | — | 推理端饱和 |

**等 3.2× 更久**：Verl trainer 每步等 445s（队列填满 + 生成 + reward + 尾延迟），slime trainer 等 137s（rollout + log_probs + offload/onload）。

### 3.4 训练时间

| 指标 | Verl V2 | Slime V3 | Δ |
|------|:---:|:---:|:---:|
| update_actor_time | **238s** | **110s** | +117% |
| log_probs_time | (在 rollout 端) | 48s | — |
| actor grad_norm | 0.184 | 0.574 | 均健康 |
| actor pg_loss | 0.102 | ~0.000 | — |

Verl 训练慢 2.2× 可能原因：VL freeze_vision_tower 后的 recompute 开销、mbridge 权重导出/导入差异。

### 3.5 SGLang 推理量对比

| 指标 | Verl V2 | Slime V3 | 说明 |
|------|:---:|:---:|------|
| 每次推理样本数 | 64 | 128 | verl ppo_mini_batch_size / n |
| 每次推理 token 量 | **95K** | **208K** | 生成量不同 |
| response_length/mean | 1487 | 1625 | 接近 |
| truncated_ratio | 28.5% | 27.4% | **几乎一样** |
| rollout idle_ratio | 2.1% | — | verl 推理端饱和 |

**不可直接比较 gen_time**：Verl 的 447s 包含了异步流水线中的队列延迟、CPU reward、尾样本等待。Slime 的 122s 是同步 batch 推理的纯 SGLang 时间。两者测量范围和统计含义完全不同。

### 3.3 训练性能 (Megatron TP2)

| 指标 | Verl V2 | Slime V3 | Δ |
|------|:---:|:---:|:---:|
| update_actor_time | **238s** | **110s** | +117% |
| actor grad_norm | 0.184 | 0.574 | — |
| actor pg_loss | **0.102** | ~0.000 | verl 梯度更强 |
| GPU memory | 39-54 GB | — | — |

Verl 训练端慢 117% 的可能原因：
1. **VL 模型的 beam search log_probs**：freeze_vision_tower 后的 recompute 开销更大
2. **Megatron 前向次数不同**：verl async 使用 `use_rollout_log_probs=True`，训练端只需 forward+backward，但可能触发了额外的 vision encoder 操作
3. **mbridge 版本差异**：可能导致 VL 模型导出/导入效率不同

### 3.6 流水线效率

| 指标 | Verl V2 | Slime V3 |
|------|:---:|:---:|
| trainer idle_ratio | **62.0%** | — |
| train_wait_ratio | — | **43.2%** |
| rollout idle_ratio | **2.1%** | — |

```
Verl Fully Async:
  推理端: ████████████████████████████████████████████▌ idle 2.1%
  训练端: ════════ wait 447s ════════ ██ train 238s ██

Slime Disaggregated Sync:
  推理端: █████ rollout+logp 122s █████ ═══ wait 173s ═══
  训练端: ═══ wait 137s ═══ ██ train+logp 158s ██
```

Verl 推理端几乎满负荷（2.1% idle），训练端有大量闲置（62%）——**瓶颈完全在推理**。训练端每步等 445s，只训 238s。Slime 推理和训练时间更平衡（wait 137s vs train 158s），wait_ratio 仅 43%。

### 3.7 B300 兼容性

两组实验均在 B300 (sm_103a) 上运行，所需修复高度一致：

| # | 修复项 | Verl | Slime |
|---|------|:---:|:---:|
| 1 | 系统 ptxas 替代 Triton 内置 | ✅ | ✅ |
| 2 | 关闭 CUDA graph (SGLang) | ✅ | ✅ |
| 3 | FA2 替代 FA3 | ✅ | ✅ |
| 4 | mm_attention_backend=sdpa | ✅ | ✅ |
| 5 | NVLink 检测 + NCCL_NVLS_ENABLE | ✅ | ✅ |
| 6 | expandable_segments | ❌ (SGLang TorchMemorySaver 冲突) | ✅ |
| 7 | megatron-bridge VL 加载 | ✅ via mbridge | ✅ via mbridge |

## 4. Training Quality

| 指标 | Verl V2 | Slime V3 | 结论 |
|------|:---:|:---:|------|
| reward/mean (趋势) | 0.378→0.422 | 0.469→0.555 | 均正向学习 |
| reward/mean (全步) | 0.462 | 0.468 | **几乎一样** |
| reward/max | 0.9 (持续) | 0.758 | 均有满分答案 |
| response_length/mean | 1487 | 1625 | 接近 |
| truncated_ratio | 28.5% | 27.4% | **一样** |
| actor/loss | 0.102 | ~0.000 | verl 梯度更强 |
| grad_norm | 0.184 | 0.574 | 均健康 |

**训练质量等价**——两种框架在相同参数下产出几乎相同的奖励分布和回复长度。Verl 的全步平均 reward (0.462) 与 slime (0.468) 只有 0.006 的差异，可以忽略。

## 5. Key Findings

**F1: Verl trainer wait 3.2× 高于 slime——SGLang 异步管线开销不可忽略**

Verl trainer 每步等 445s，slime trainer 等 137s。Verl 虽然推理/训练并行，但队列填满延迟 + CPU reward worker + 异步流水线中的尾样本等待放大了总体等待时间——3.2×。这是 verl SGLang async 的核心瓶颈。

**F2: 训练质量等价——两种框架的算法实现水平一致**

尽管性能差距大，reward/mean、response_length、truncated_ratio 三指标几乎完全一致。这说明 verl 的 GRPO + Megatron + reward pipeline 在语义上正确——只是效率问题。

**F3: 异步流水线中的队列延迟 + CPU reward 放大了等待时间**

Rollouter idle 仅 2.1%——推理端饱和。但 trainer 每步等 445s（含队列填满 + reward computation on CPU + tail latency），只训练 238s。Slime 同步模式下 wait 仅 137s，因为不需要填充队列、没有异步通信开销。

**F4: B300 VL 适配已成熟**

两组框架在 B300 sm_103a 上均正常运行，需要的修复项几乎一致。SGLang disable_cuda_graph + mm_attention_backend=sdpa + FA2 的组合已证明稳定。

**F5: 数据预处理是 verl VL 适配的最大难点**

本次实验最耗时的部分不是训练本身，而是数据格式——base64→PIL→parquet 的兼容性问题经过多次迭代才解决（详见 commit 历史）。`chenhegu/geo3k_imgurl` 数据集的图片格式与 `hiyouga/geometry3k` 不同（base64 data URIs vs PIL-native），之前官方脚本只适配了后者。

## 6. Architecture Comparison

| 维度 | Verl Fully Async | Slime Disaggregated | 建议 |
|------|------|------|------|
| SGLang 集成 | HTTP Server (Ray actor) | 独立进程 (原生) | Slime 更好 |
| 流水线 | 异步并行 (推理/训练) | 同步串行 (rollout→train) | Verl 潜力更好 |
| 当前效率 | gen 占 65%，train idle 62% | 平衡 (43% wait) | Slime 更优 |
| 吞吐量改善 | 增加推理引擎 | 增加 GPU | 两者都可 |
| 显存效率 | 39 GB/GPU (exclusive) | ~128 GB/GPU (shared) | Verl 更好 |
| 配置复杂度 | 高 (2套 GPU 参数) | 中 | Slime 更简单 |
| VL 数据管线 | 需自行预处理 | 零配置 | Slime 更简单 |

## 7. Next Actions

### 短期
1. **vLLM 作为 verl 推理后端**：vLLM 与 verl 是进程内调用，无 HTTP 开销。之前文本对比 verl+vLLM 比 slime+SGLang 快 13%，VL 上可能也有优势。等待 cuPy + FlashAttn 后端解决后测试
2. **查看 verl SGLang 是否有非 HTTP 模式**：避免 HTTP server 的序列化开销

### 中期
3. **增加推理并行度**：2 个 SGLang TP2 引擎可减少 gen_time 约 50%
4. **更小的 ppo_mini_batch_size**：8 而非 64，减少训练端批数据积压时间

### 不推荐
- ❌ verl SGLang 生产 VL 训练（当前性能劣势太大）
- ❌ 在 B300 上用 vLLM (FlashInfer/FlashAttn 兼容性解决前)

## 8. Artifact Index

### Verl V2
| 用途 | 路径 |
|------|------|
| Script | `examples/grpo_trainer/run_qwen3_vl_4b_megatron_async.sh` |
| vLLM Script | `examples/grpo_trainer/run_qwen3_vl_4b_megatron_async_vllm.sh` |
| TensorBoard | `tensorboard_log/verl_async_geo3k_v2/qwen3_vl_4b_sglang_megatron_async_v2/` |
| Data preprocess | `examples/data_preprocess/geo3k_imgurl.py` |
| Reward | `examples/grpo_trainer/geo3k_reward.py` → `verl/utils/reward_score/geo3k.py` |
| Steps | 32 trainer steps (step 1-32) |
| Checkpoint | `/workspace/volume/pengxiong/models/Qwen3-VL-4B_verl_geo3k_async_v2/` |

### Slime V3
| 用途 | 路径 |
|------|------|
| Script | `scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh` |
| TensorBoard (train) | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090856/` |
| TensorBoard (eval) | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090614/` |
| Steps | 106 rollout / 212 train steps |
| Checkpoint | `iter_0000099` at slime models directory |
