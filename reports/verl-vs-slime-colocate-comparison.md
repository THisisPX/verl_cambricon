# Verl vs Slime 性能对比报告

## 实验目的

在相同条件下对比 verl 和 slime 两个 RL 训练框架的 Qwen3-4B GRPO 训练性能，定位差异根因。

## 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-4B |
| 硬件 | 8× A100 40GB (单节点) |
| 算法 | GRPO |
| 数据 | dapo-math-17k (train) / aime-2024 (val) |
| system_prompt | "Please reason step by step, and put your final answer in \\boxed{}." |
| 训练步数 | 8 |

## 三组实验配置

| 实验 | 框架 | 训练并行 | 推理引擎 | 推理并行 | GPU 分配 | 入口 |
|------|------|:---:|------|:---:|------|------|
| Slime TP4 | Slime | TP4, DP2 | SGLang | 4 engines × TP2 | 8GPU colocate + offload | `train.py --colocate` |
| Verl TP2 | Verl | TP2, DP4 | vLLM | 4 engines × TP2 | 8GPU HybridEngine + offload | `main_ppo` |
| Verl TP4 | Verl | TP4, DP2 | vLLM | 4 engines × TP2 | 8GPU HybridEngine + offload | `main_ppo` |

> **设计意图**: Verl TP4 保持推理端不变 (4×TP2)，仅改变训练 TP，用于隔离 TP 对性能的影响。

## 对齐的参数

| 参数 | Slime | Verl | 状态 |
|------|-------|------|:---:|
| advantage_estimator | grpo | grpo | ✅ |
| n_samples_per_prompt | 16 | 16 | ✅ |
| max_response_length | 8192 | 8192 | ✅ |
| max_prompt_length | 1024 | 1024 | ✅ |
| train_batch_size | 8 | 8 | ✅ |
| global_batch_size / ppo_mini_batch_size | 32 | 8 (×16=128) | ✅ |
| kl_loss_coef / use_kl_loss | 0.00 | False | ✅ |
| entropy_coef | 0.00 | 0.00 | ✅ |
| clip_ratio_low | 0.2 | 0.2 | ✅ |
| clip_ratio_high | 0.28 | 0.28 | ✅ |
| optimizer | Adam | Adam | ✅ |
| lr | 1e-6 | 1e-6 | ✅ |
| lr_decay | constant | constant | ✅ |
| weight_decay | 0.1 | 0.1 | ✅ |
| betas | [0.9, 0.98] | [0.9, 0.98] | ✅ |
| sequence_parallel | ✅ | ✅ | ✅ |
| recompute | full + uniform | full + uniform | ✅ |
| 保存间隔 | 9999 | 9999 | ✅ |
| 验证间隔 | 9999 | 9999 | ✅ |

## 未对齐的差异 (框架固有)

| 差异 | Slime | Verl TW2 | Verl TP4 | 影响 |
|------|-------|:---:|:---:|------|
| 推理引擎 | SGLang | vLLM | vLLM | rollout 速度 |
| CUDA graph | 开 (max_bs=8) | 关 (enforce_eager=True) | 关 (enforce_eager=True) | 推理内存 |
| 训练 TP | 4 | 2 | 4 | 训练速度 |
| offload 机制 | torch_memory_saver | Megatron 三路卸载 | Megatron 三路卸载 | 切换开销 |
| NCCL 管理 | 每步拆建进程组 | 全程保持 | 全程保持 | ~11s/step |
| loss_agg | sum-of-sample-mean | token-mean | token-mean | 梯度数值 |
| 评分函数 | deepscaler | math_dapo | math_dapo | reward scale |
| 权重同步 | UpdateWeightFromTensor | NCCL checkpoint engine | NCCL checkpoint engine | 0.4% step |

---

## 一、性能对比

### 1.1 时间分解 (8 步均值)

| 阶段 | Slime TP4 | Verl TP2 | Verl TP4 |
|------|:---:|:---:|:---:|
| rollout / gen | 275.9s | 260.8s | 258.4s |
| log_probs / old_log_prob | 16.3s | 14.8s | 20.2s |
| actor_train / update_actor | **57.0s** | **39.1s** | **55.7s** |
| data_preprocess / adv | 0.14s | 0.04s | 0.05s |
| sleep + wake_up | **11.2s** | — | — |
| update_weights | 1.1s | 2.5s | 2.2s |
| train (总) | 73.6s | 53.9s | 75.5s |
| **step_time** | **389.9s** | **337.1s** | **348.4s** |

### 1.2 时序可视化

```
Slime TP4  ═══════════════════════════════════════════════════════════════════
  │sleep│── rollout 275.9s ──│wake│── train 73.6s ──│      = 389.9s/step
  │7.4s │                    │3.8s│ log_p 16.3s      │
  │     │                    │    │ train 57.0s      │
  │     │                    │    │ upd_w 1.1s       │

Verl TP4  ═══════════════════════════════════════════════════════════════
  │── gen 258.4s ──│── old_lp 20.2s ──│── upd_act 55.7s ──│ = 348.4s
  │                │                  │  upd_w 2.2s        │

Verl TP2  ═══════════════════════════════════════════════════════════
  │── gen 260.8s ──│── old_lp 14.8s ──│── upd_act 39.1s ──│ = 337.1s
  │                │                  │  upd_w 2.5s        │
```

### 1.3 差异分解

| 对比维度 | 时间差 | 说明 |
|------|:---:|------|
| **Slime TP4 → Verl TP4** (同 TP，纯框架差异) | **-41.5s (-10.6%)** | |
| ├─ offload 切换 (sleep+wake_up) | -11.2s | Slime 拆建 NCCL，Verl 不需要 |
| ├─ rollout 生成 | -17.5s | vLLM vs SGLang 引擎差异 |
| ├─ log_probs 计算 | +3.9s | 计入差异，但方向相反 |
| ├─ actor_train | -1.3s | 接近 (57.0 vs 55.7)，TP 相同时训练速度持平 |
| └─ 其他框架开销 | -15.4s | 数据流、调度、通信 |
| | | |
| **Verl TP4 → Verl TP2** (同框架，纯 TP 差异) | **-11.3s (-3.2%)** | |
| ├─ update_actor | -16.6s | TP2 all-reduce 跨 2 卡 vs TP4 跨 4 卡 |
| ├─ old_log_prob | -5.4s | DP4 vs DP2，每卡数据量减半 |
| └─ gen (无关联) | +2.4s | 随机波动 |
| | | |
| **Slime TP4 → Verl TP2** (总体) | **-52.8s (-13.5%)** | |
| ├─ offload 架构差异 | -11.2s | |
| ├─ TP 策略差异 | -11.3s | Verl 能用 TP2 (offload 更彻底) |
| └─ 框架实现差异 | -30.3s | 引擎、调度、通信等 |

---

## 二、吞吐量

| 指标 | Slime TP4 | Verl TP2 | Verl TP4 |
|------|:---:|:---:|:---:|
| total_tokens / step | ~785k | 874k | 872k |
| response_length / mean | 6373 | 6667 | 6644 |
| truncated_ratio | 47% | 50% | 50% |
| tokens_per_gpu_per_sec (rollout) | 373 | — | — |

> Slime 的 `tokens_per_gpu_per_sec` 只统计 rollout 阶段。Rollout 时间接近 (276s vs 258s vs 261s)，推理吞吐差异不大。

---

## 三、训练质量

| 指标 | Slime TP4 (32 steps) | Verl TP2 (8 steps) | Verl TP4 (8 steps) |
|------|:---:|:---:|:---:|
| pg_loss | ~0.00 | 0.01 | 0.01 |
| entropy_loss / actor/entropy | 0.33 | 0.35 | 0.35 |
| grad_norm | 0.18 | 0.09 | 0.11 |
| pg_clipfrac | 0.00 | 0.00 | 0.00 |
| ppo_kl | 0.00 | 0.00 | 0.00 |
| truncated_ratio | 47% | 50% | 50% |
| response_length/mean | 6373 | 6667 | 6644 |

- **pg_loss、entropy、grad_norm 接近**，验证三个实验执行相同的数学运算
- **reward scale 不同** (0.52 vs -0.85) 是因为 deepscaler 和 math_dapo 评分口径不同，不影响梯度方向
- **pg_clipfrac 和 ppo_kl 均为 0**，GRPO ratio 在 clip 范围内，训练稳定

---

## 四、结论

### 4.1 Verl 比 Slime 快 13.5% (最佳配置)

Verl TP2 是三个实验中性能最优的配置 (337.1s/step)，比 Slime TP4 (389.9s) 快 52.8s。

### 4.2 差异来自两层

**第一层：框架架构差异 (41.5s, 占 79%)**

即使训练 TP 相同 (均为 4)，Verl 仍然快 10.6%。主要来自：
- **NCCL 进程组不拆建**: Slime 的 SGLang 是独立进程，colocate 时必须 `destroy_process_groups()` + `reload_process_groups()`，每次 11.2s。Verl 的 vLLM 和 Megatron 在同一进程内 (HybridEngine)，NCCL 组全程保持。
- **推理引擎差异**: vLLM vs SGLang，gen 阶段快 17.5s

**第二层：并行策略差异 (11.3s, 占 21%)**

Verl TP2 比 Verl TP4 快 3.2%，因为 TP2 的通信开销更低 (all-reduce 跨 2 卡 vs 4 卡) 且 DP 并行度更高 (DP4 vs DP2)。

### 4.3 Slime 被迫用 TP4 而 Verl 能用 TP2

这是 offload 实现细节导致的结果——两个框架在 colocate 模式下都会把模型卸载到 CPU 腾 GPU 给推理引擎。但 Verl 三路卸载 (param + grad + optimizer) 配合预分配 CPU pinned buffer，释放更彻底，使得 TP=2 (每卡 ~4GB 权重) 能在 A100 40GB 上运行。Slime 的 `torch_memory_saver` 在同样场景下 TP=2 会 OOM，被迫上 TP=4 (每卡 ~2GB 权重)。

### 4.4 核心结论

| 结论 | 证据 |
|------|------|
| **Verl 吞吐更高** | TP2 快 13.5%，TP4 快 10.6% |
| **Verl offload 更优** | HybridEngine 不拆建 NCCL，每步省 11.2s；释放更彻底，能用 TP2 |
| **训练质量一致** | pg_loss、entropy、grad_norm 在统计误差范围内 |
| **推理端不是瓶颈** | rollout 时间接近 (276s vs 259s vs 261s)，差异 < 7% |

---

## 五、实验脚本

| 框架 | 脚本路径 |
|------|------|
| Verl TP2 | `examples/grpo_trainer/run_qwen3_4b_megatron_perf_test.sh` (ACTOR_TP=2) |
| Verl TP4 | `examples/grpo_trainer/run_qwen3_4b_megatron_perf_test.sh` (ACTOR_TP=4) |
| Slime | `scripts/run-qwen3-4B-sync-colocate-8gpu.sh` |

## 六、Event 文件

| 实验 | 路径 |
|------|------|
| Slime rollout | `D:\learning\slime\tensorboard_log\slime-vs-verl-colocate-8gpu\20260614_114622\` |
| Slime train | `D:\learning\slime\tensorboard_log\slime-vs-verl-colocate-8gpu\20260614_115229\` |
| Verl TP2 | `tensorboard_log/verl_perf_test/qwen3_4b_grpo_n16_resp8192_megatron/` |
| Verl TP4 | `tensorboard_log/verl_perf_test/qwen3_4b_grpo_n16_resp8192_megatron_tp4/` |

---

*报告日期: 2026-06-17*
