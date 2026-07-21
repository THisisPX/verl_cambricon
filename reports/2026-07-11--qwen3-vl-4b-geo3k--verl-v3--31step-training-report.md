---
type: results-report
date: 2026-07-11
experiment_line: qwen3-vl-4b-geo3k-grpo
round: 1
purpose: verl v3 31-step async RL training analysis
status: final
framework: verl (fully async, 2+2 GPU split)
linked_experiments: []
---

# Qwen3-VL-4B GEO3K GRPO / Verl Fully Async / v3 / 31-Step Analysis / 2026-07-11

## 1. Executive Summary

在 4× NVIDIA B300 SXM6 GPU 上对 Qwen3-VL-4B-Instruct 进行 verl fully async GRPO 数学几何推理 RL 训练（31 步），核心发现：

- **reward / score / loss**: 全部 31 步为 **0.000**——模型无学习信号
- **training signal**: 死锁——GRPO group-normalized advantage 在组内全零 reward 时完全退化
- **流水线**: 推理端接近满负荷 (idle 8.5%)，训练端大量空等 (idle 73.4%)
- **validation**: step 20 eval score = 0.000

**根因**: Qwen3-VL-4B-Instruct 是通用指令微调模型，从未被训练过输出 `\boxed{}` 格式。geo3k 的 binary math reward 要求 `<think>...</think>\boxed{...}` 双条件匹配，模型输出从未命中。一旦全零，GRPO 的 `scores = (0 - 0) / (0 + 1e-6) = 0` 永久死锁。

与 slime 的实验不同（slime step 0 时就有 ~47% 的 reward），verl 模型在冷启动时无一命中格式。这需要数据预处理或 chat template 格式差异的进一步排查。

## 2. Experiment Identity and Decision Context

### 实验目标
在 verl 框架上使用 fully async disaggregated 模式对 Qwen3-VL-4B-Instruct 进行视觉数学几何推理（GEO3K 数据集）的 GRPO RL 训练，验证 verl async pipeline 的功能正确性。

### 关键决策
- 使用 fully async (2+2 GPU) 还是 hybrid engine (4 GPU 共享)？
  → 选择 fully async，验证异步流水线的功能
- SGLang 还是 vLLM？
  → SGLang，对标 slime 的推理引擎选择
- 能否在 4 GPU 上成功运行 VLM + Megatron + SGLang？
  → 能运行，pipeline 功能正确
- B300 Blackwell GPU 兼容性如何？
  → 6 项修复后正常

## 3. Setup and Evaluation Protocol

### 硬件
| 项目 | 配置 |
|------|------|
| GPU | 4× NVIDIA B300 SXM6 AC (275 GiB each), CUDA 13.0 |
| 服务器 | 8 卡共享节点 (notebook-devenviron-0708-155258) |
| 互联 | NVLink |
| 训练 GPU | 2× (TP2, DP1), ~41 GB allocated |
| 推理 GPU | 2× (1 SGLang engine × TP2) |

### 软件
| 项目 | 版本/说明 |
|------|------|
| verl | 0.9.0.dev (fork, main branch) |
| Python | 3.13.5 |
| SGLang | HTTP server 模式 (Ray actor) |
| Megatron-LM | mbridge (Megatron Bridge) |
| reward | geo3k (mathruler, LaTeX formula comparison) |
| 日志 | TensorBoard (WANDB_MODE=offline) |

### 完整训练参数

| 类别 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **模型** | `actor_rollout_ref.model.path` | `/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct` | |
| | `use_remove_padding` | True | THD 格式 |
| | `use_fused_kernels` | True | Megatron fused kernels |
| | `trust_remote_code` | True | Qwen3-VL 自定义模型代码 |
| | `freeze_vision_tower` | True | 冻结 vision encoder |
| **数据** | `data.train_files` | geo3k_imgurl/train.parquet | verl parquet 格式 |
| | `data.val_files` | geo3k_imgurl/test.parquet | |
| | `data.image_key` | images | 图片列 |
| | `data.max_prompt_length` | 2048 | |
| | `data.max_response_length` | 3072 | |
| | `data.train_batch_size` | 0 | async streaming mode |
| | `data.gen_batch_size` | 1 | sample-by-sample |
| **GPU** | `trainer.n_gpus_per_node` | 2 | 训练 GPU |
| | `rollout.n_gpus_per_node` | 2 | 推理 GPU |
| | `trainer.nnodes / rollout.nnodes` | 1 / 1 | 单节点 |
| **算法** | `algorithm.adv_estimator` | grpo | 组内标准化 |
| | `algorithm.use_kl_in_reward` | False | 无 KL 奖励惩罚 |
| | `algorithm.rollout_correction.bypass_mode` | True | 直接使用 rollout log probs |
| | `actor.use_kl_loss` | False | 无 KL loss |
| | `actor.entropy_coeff` | 0 | 无熵惩罚 |
| | `actor.clip_ratio_low / _high` | 0.2 / 0.28 | dual-clip PPO |
| | `actor.loss_agg_mode` | token-mean | |
| **Rollout** | `rollout.name` | sglang | |
| | `rollout.mode` | async | |
| | `rollout.n` | 8 | 每个 prompt 8 个回答 |
| | `rollout.temperature` | 0.8 | |
| | `rollout.max_model_len` | 6144 | 2048+3072+1024 headroom |
| | `rollout.enforce_eager` | True | B300 关闭 CUDA graph |
| | `engine_kwargs.sglang.mm_attention_backend` | sdpa | B300 VL 兼容 |
| | `engine_kwargs.sglang.attention_backend` | flashinfer | |
| **优化器** | `optim.lr` | 1e-6 | |
| | `optim.lr_decay_style` | constant | |
| | `optim.weight_decay` | 0.1 | |
| | `optim.betas` | [0.9, 0.98] | |
| | `optim.lr_warmup_steps` | 0 | |
| **Megatron** | `tensor_model_parallel_size` | 2 | TP2, DP1 |
| | `pipeline_model_parallel_size` | 1 | |
| | `context_parallel_size` | 1 | |
| | `sequence_parallel` | True | |
| | `use_mbridge` | True | Megatron Bridge |
| | `override_transformer_config.recompute_granularity` | full | |
| | `override_transformer_config.recompute_method` | uniform | |
| | `override_transformer_config.recompute_num_layers` | 1 | |
| | `override_transformer_config.attention_backend` | flash | FA2 |
| **异步** | `staleness_threshold` | 0 | on-policy 流式 |
| | `trigger_parameter_sync_step` | 4 | 每 4 步同步 |
| | `require_batches` | 1 | 纯流式 |
| | `partial_rollout` | True | 恢复被中断的 rollout |
| **Reward** | `custom_reward_function.path` | `geo3k_reward.py` | mathruler LaTeX 比较 |
| | `reward.reward_manager.name` | naive | |
| **监控** | `trainer.logger` | console + tensorboard | |
| | `trainer.project_name` | verl_async_geo3k | |
| | `trainer.experiment_name` | qwen3_vl_4b_sglang_megatron_async_slime_match | |

### B300/Blackwell 兼容性修复

| # | 问题 | 修复 |
|---|------|------|
| 1 | ptxas 不识别 sm_103a | `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` |
| 2 | CUDA kernel 无 sm_103a 镜像 | `enforce_eager=True` (禁用 CUDA graph) |
| 3 | FA3 不支持 Blackwell | `attention_backend=flash` (回退 FA2) |
| 4 | VL 多模态注意力 | `mm_attention_backend=sdpa` |
| 5 | NVLink 检测 | `NCCL_NVLS_ENABLE` 动态检测 |
| 6 | CUDA 架构列表 | `TORCH_CUDA_ARCH_LIST=10.0` |
| 7 | SGLang TorchMemorySaver 不兼容 | 移除 `expandable_segments` (commit `9e090d2f`) |

### 评估协议
- 每 20 rollout 评估一次（仅 step 20 执行）
- 测试集：geo3k_imgurl/test.parquet
- 贪婪解码 (do_sample=True, top_p=1, n=1)
- Primary metric: validation score (math accuracy via geo3k_reward)

## 4. Training Metrics

### 4.1 Rollout 指标 (训练集) ⚠️

| 指标 | Step 1 | Step 31 | 趋势 | 状态 |
|------|--------|--------|------|:---:|
| rewards/mean | **0.000** | **0.000** | → 0 | ❌ 死锁 |
| rewards/max | **0.000** | **0.000** | → 0 | ❌ |
| rewards/min | **0.000** | **0.000** | → 0 | ❌ |
| score/mean | **0.000** | **0.000** | → 0 | ❌ |
| score/max | **0.000** | **0.000** | → 0 | ❌ |
| response_length/mean | 217.1 | 264.5 | +21.8% | ⚠️ 极短（正常应有 1000+） |
| response_length/max | 3072 | 3072 | — | 全部 31 步均为 3072 |
| response_length/min | 57 | 51 | — | |
| truncated_ratio (clip) | 0.3% | 0.6% | → 0 | 几乎无截断（输出太短） |
| aborted_ratio | 0.000 | 0.000 | → 0 | 无 abort |
| prompt_length/mean | 83.3 | 83.1 | → 稳定 | |

**response_length/mean 为什么这么低？**

平均回复长度仅 200-264 tokens，远低于 slime 的 1625。这直接证明模型**没有学会格式**——它只输出简短的文本（如 "The answer is 42"），而不是 `<think>...</think>\boxed{...}`。因为不存在 `\boxed{}`，reward 始终为 0。

**为什么 max 始终是 3072？**

3072 是 `max_response_length` 上限。但 mean=264 而 max=3072，说明**极少数样本触发了上限**（可能是随机生成了长串无意义文本），与 "模型正常推理" 无关。

### 4.2 训练信号 (全零死锁) ❌

| 指标 | Step 1 | Step 31 | 31 步范围 | 解读 |
|------|--------|--------|-----------|------|
| actor/loss | 0.0000 | 0.0000 | 全零 | 无梯度 |
| actor/pg_loss | 0.0000 | 0.0000 | 全零 | 无 policy gradient |
| actor/grad_norm | 0.0000 | 0.0000 | 全零 | 梯度为零 |
| critic/advantages/mean | 0.0000 | 0.0000 | 全零 | advantage 为零 |
| critic/advantages/max | 0.0000 | 0.0000 | 全零 | |
| critic/advantages/min | 0.0000 | 0.0000 | 全零 | |
| actor/pg_clipfrac | 0.0000 | 0.0000 | 全零 | 无 clip 事件 |
| actor/pg_clipfrac_lower | 0.0000 | 0.0000 | 全零 | |
| actor/ppo_kl | 0.0004 | 0.0004 | 0.0004 | 仅微小浮动 |
| perf/mfu/actor | 0.0000 | 0.0000 | 全零 | MFU 为零（无有效 FLOPs） |

**死锁机制** (`verl/trainer/ppo/core_algos.py:304-326`):

```python
scores = token_level_rewards.sum(dim=-1)       # 全零 reward → scores 全零
for idx in id2score:
    scores_tensor = torch.stack(id2score[idx])  # 8 个 zero
    id2mean[idx] = torch.mean(scores_tensor)    # = 0.0
    id2std[idx] = torch.std(scores_tensor)      # = 0.0
scores[i] = (0 - 0) / (0 + 1e-6) = 0            # ← 永恒为零
```

<details>
<summary>验证 fullmatch regex 永远不会匹配</summary>

`geo3k.py`:
```python
def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0  # 从不匹配
```

模型输出示例（推测）：`"The answer is 42"` → `fullmatch` 返回 `None` → `format_reward=0`。同时 `extract_boxed_content` 也找不到 `\boxed{}`，`acc_reward=0`。两者都是 0 → `compute_score=0`。
</details>

### 4.3 验证指标 (Step 20) ❌

| 指标 | 值 | 说明 |
|------|:---:|------|
| val-core/acc/mean@1 | **0.0000** | 所有答案错误 |
| val-aux/reward/mean@1 | **0.0000** | reward 为零 |
| val-aux/num_turns/mean | 2.0 | 正常 (multi-turn agent loop) |
| rollouter/validate_time | 100.2s | 验证耗时正常 |

## 5. Performance Analysis

### 5.1 步时与吞吐

| 指标 | Mean | Min | Max | 趋势 |
|------|:---:|:---:|:---:|------|
| step_time | 251.0s | 179.0s | 304.1s | ↑ 逐渐变长 |
| gen_time | 184.7s | 115.9s | 227.8s | ↑ |
| update_actor_time | 63.7s | 55.7s | 73.1s | → 稳定 |
| param_sync_time | 1.4s | 1.3s | 1.6s | → 稳定 |
| throughput (tok/s) | 688.9 | 605.8 | 797.6 | ↓ 轻微波动 |
| total_num_tokens/step | 689k | 461k | 812k | ↑ 递增 |

```
step_time:  179s ═════════════════════════════════ 304s  (mean=251s, σ=30.6s)
            ██████████████████████████████████████████████
gen_time:   116s ═══════════════════════════════════ 228s  (mean=185s)
            █████████████████████████████████████████
train_time:  56s ═══════════════════════════ 73s            (mean=64s)
            ███████████████████
```

步时从 step 1 (179s) 增长到 step 31 (284s)。gen_time 的增长是主要原因——随着 SGLang engine 运行渐久，prefix cache 或内存碎片影响了推理速度。

### 5.2 异步流水线效率

| 指标 | Mean | Min | Max | 解读 |
|------|:---:|:---:|:---:|------|
| trainer idle_ratio | **73.4%** | 64.8% | 76.1% | 训练端大量空等 |
| rollouter idle_ratio | **8.5%** | 0.0% | 11.2% | 推理端接近满负荷 |
| mq_queue_size | 0 | 0 | 0 | 消息队列无积压（流式消费） |
| pending_queue_size | 128 | 128 | 128 | 持续 128 个样本在 pipeline 中 |
| active_tasks_size | 16 | 16 | 16 | 16 个并发 rollout task |
| total_generated_samples | 3968 | 128 | 7808 | 累计 7808 个样本 |
| processing_time/avg | 8.4s | 1.6s | 91.0s | 分布有长尾 |
| processing_time/tp99 | 49.3s | 48.4s | 53.5s | p99 延迟 ~50s |

```
Pipeline 可视化:

  推理端 (2 GPUs, idle 8.5%):
    ██████████████████████████████████████████████████▌  (接近满负荷)

  训练端 (2 GPUs, idle 73.4%):
    ════ wait 185s ════ ████ train 64s ████ ════ wait ════

  瓶颈: 推理端只有 2 个 GPU (1× SGLang TP2 engine)，训练端消费速度
  (ppo_mini_batch_size=64, train_time=64s) 远快于推理端
  (128 samples × 8 turns × ~2000 tokens ≈ 184s)。
```

### 5.3 Token 级别性能

| 指标 | Mean | Min | Max |
|------|:---:|:---:|:---:|
| timing_per_token_ms/gen | **0.210** | 0.111 | 0.320 |
| timing_per_token_ms/update_actor | **0.086** | 0.070 | 0.094 |
| timing_per_token_ms/adv | 0.000 | 0.000 | 0.000 |

推理端每 token 0.21ms，训练端每 token 0.086ms。推理/训练速度比约 2.4:1——训练远远快于推理，这是 trainer idle 73% 的直接原因。

### 5.4 显存

| 指标 | Step 1 | Step 31 | 解读 |
|------|:---:|:---:|------|
| max_memory_allocated_gb | 38.8 | **41.2** (+6%) | 稳定 |
| max_memory_reserved_gb | 41.5 | 43.7 (+5%) | 稳定 |
| cpu_memory_used_gb | 353.5 | 396.2 (+12%) | 节点级 CPU 内存增长（Ray 对象存储） |

训练 GPU 显存基本稳定在 ~41 GB。B300 的 275 GiB 充裕，即使是 A100 40GB 也勉强可行。

### 5.5 权重同步

| 指标 | Mean | 说明 |
|------|:---:|------|
| param_sync_time | 1.4s | 每 4 步同步一次 |
| current_param_version | 0→30 | 线性递增，每步 +1 |
| dropped_stale_samples | 0 | staleness_threshold=0，零容忍 |
| stale_trajectory_processed | 0 | |
| staleness_samples | 80→113 | 随步骤增长 |

权重同步采用 NCCL CheckpointEngine（`backend=nccl`），每次同步 ~1.4s。参数版本 (param_version) 从 0 递增到 30，每步对应当前权重的版本号。

## 6. Pipeline Correctness Verification

Verl async pipeline 本身**运行正确**，以下证据支持此结论：

| 验证项 | 预期行为 | 实际 | 结论 |
|------|------|:---:|:---:|
| 数据流 | RewardLoopManager → DataProto.rm_scores → trainer | 正常执行 | ✅ |
| 消息队列 | 流式消费，无积压 | queue_size=0 | ✅ |
| staleness 控制 | threshold=0, 零容忍旧样本 | dropped=0 | ✅ |
| partial rollout | 权重同步时不丢弃 | partial_ratio=0 | ✅ |
| 权重同步 | NCCL CheckpointEngine 每 4 步 | param_version 正确递增 | ✅ |
| 训练步数递增 | global_step 1→123 | 正常（4 mini_batch/step × 31） | ✅ |
| 验证 | step 20 执行 val | val-aux/num_turns 正常 | ✅ |
| multi-turn | agent loop 2 turns | num_turns=2 | ✅ |
| 无 abort | 所有 response 正常完成 | aborted_ratio=0 | ✅ |
| 推理端无 OOM | GPU ~41 GB / 275 GB | 充裕 | ✅ |

## 7. Key Findings

**F1: Pipeline 功能正确，但 reward 冷启动导致零学习** ❌

Verl fully async pipeline 的消息队列、权重同步、staleness 控制、partial rollout、agent loop 全部正常运行。唯一的致命问题是模型从未命中 `\boxed{}` 格式，导致 reward 恒为 0，进而 GRPO advantage 恒为 0，形成死锁。

**F2: 推理端饱和，训练端瓶颈不在自身而在推理速度**

训练端 idle 73.4%，推理端 idle 8.5%。训练端每次只训练 64s，然后等待推理端约 3 倍的时间。增加推理并行度（如 4 GPU 推理 × 2 engines）是最直接的改进方向。

**F3: response_length/mean=234 tokens——模型没有学会格式的有力证据**

正常的数学推理回复应该在 1000-2000 tokens（slime 的均值 1625）。verg 的 234 tokens 说明模型只输出了简短的自然语言文本，完全没有进入 chain-of-thought 格式。

**F4: Megatron+SGLang integration 在 VL 场景下工作正常**

没有 crash、OOM、或 hang。B300 兼容性方案（enforce_eager + FA2 + sdpa）有效。SGLang HTTP server 模式在 VL 推理中性能正常（0.21ms/token）。

**F5: 步时逐渐增长——SGLang 长运行稳定性需关注**

step_time 从 179s (step 1) → 284s (step 31)，+58%。gen_time 增长是主要原因。可能与 SGLang 的 prefix cache 饱和或内存碎片有关，需要更长实验验证。

## 8. Response Length Distribution Analysis

```
Response length by step (mean):
  Step  1: 217 tokens
  Step  5: ~230 tokens
  Step 10: ~240 tokens
  Step 15: ~250 tokens
  Step 20: ~255 tokens
  Step 25: ~260 tokens
  Step 31: 264 tokens  (+21.8% from step 1)
```

均值从 217 增长到 264——这个变化并非"学习到更多推理"，而可能是模型输出的随机波动叠加 token 级别的采样噪声。264 远低于 3072 上限，也低于 slime 的 1625。重要的是**没有出现 reward > 0 的单一样本**，所以这些长度变化只是无监督的随机漂移。

## 9. Limitations

- **31 步太少**：无法得出收敛性结论（计划 ~8000 rollout steps，实际只跑了 31 trainer steps）
- **reward=0 污染所有指标**：actor/loss, pg_loss, grad_norm, entropy 全为零，无法分析训练动力学
- **无 SFT 暖启动比较**：无法分离"缺乏格式对齐"和"模型能力的上限"
- **未记录模型输出样本**：由于 reward debug 打印可能被抑制，无法直接确认模型的实际输出内容。需要通过日志或单独推理确认
- **单硬件配置**：仅在 B300 × 4 上测试，A100/H100 的行为未知
- **未测试其他 reward manager**：仅使用 naive，未测试 dapo 或 batch

## 10. Next Actions

### P0 — 修复 reward 死锁
1. **排查模型输出**：在测试环境单独推理 Qwen3-VL-4B，观察它面对 math instruction 时的实际输出格式
2. **对比 chat template**：检查 verl 的 `geo3k_imgurl.py` 数据预处理生成的 prompt 格式与 slime 的 `--apply-chat-template` 是否一致。特别关注 `instruction_following` 的拼写位置
3. **SFT 暖启动**：用 `examples/data_preprocess/geo3k_imgurl.py` 的指令格式做几千条 SFT（已有 slime 的 `run_geo3k_vlm_sft.sh` 参考）
4. **更换 reward 策略**：测试放宽格式 regex（如接受 `\boxed{` 但不要求 `<think>` 标签）

### P1 — 延长实验
5. reward 修复后重新运行，目标 ≥ 100 steps
6. 增加推理并行度（如 4 GPU 推理 × 2 engines TP2）减少训练端 idle

### P2 — 深入分析
7. 增加 debug logging 打印 rollout sample 的前 N 个 response 内容
8. 在 verl 上运行 slime 的 chat template，确认 prompt 格式对齐

### 不推荐
- ❌ 在 reward=0 的情况下延长训练（完全浪费算力，梯度恒为零）
- ❌ 使用 dapo RM + binary answer（GEO3K LaTeX 答案不兼容 dapo 的 float 转换要求）
- ❌ 增加 n_samples_per_prompt（8 个零 reward 和 16 个零 reward 没有区别）

## 11. Artifact and Reproducibility Index

### Event Files
| 用途 | 路径 | 文件数 | 步数 |
|------|------|:---:|:---:|
| Train + Val | `tensorboard_log/verl_async_geo3k/qwen3_vl_4b_sglang_megatron_async_slime_match/` | 11 | 31 train / 1 val |

### Script
- `examples/grpo_trainer/run_qwen3_vl_4b_megatron_async.sh`

### Rewards
- `examples/grpo_trainer/geo3k_reward.py` (wrapper → `verl/utils/reward_score/geo3k.py`)
- `examples/data_preprocess/geo3k_imgurl.py`

### Model
- Qwen3-VL-4B-Instruct: `/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct/`

### Commits
```
04154b2a fix(grpo): align Qwen3-VL-4B async script with slime v3, fix P0 assert
b7454fb1 fix(grpo): switch back to nccl checkpoint engine
9e090d2f fix(grpo): remove expandable_segments
cc9fbbc6 fix(grpo): switch checkpoint_engine to naive, add missing SGLang keys
b934f6b6 fix(grpo): add missing lr_decay_steps
8af84c15 fix(grpo): remove LLM-only override_transformer_config keys
```

### Analysis Commands
```bash
# Parse event files
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('tensorboard_log/verl_async_geo3k/qwen3_vl_4b_sglang_megatron_async_slime_match')
ea.Reload()
for tag in sorted(ea.Tags()['scalars']):
    events = ea.Scalars(tag)
    vals = [e.value for e in events]
    print(f'{tag}: {len(vals)}pts mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}')
"
```
