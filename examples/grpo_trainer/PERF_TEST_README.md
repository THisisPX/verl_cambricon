# Verl 性能测试：Qwen3-4B GRPO（与 slime 框架对比）

## 概览

本测试在 8× A100 40GB 上运行 Qwen3-4B GRPO 训练，用于与 slime 框架的性能对比。

## 测试配置

| 配置 | 说明 |
|------|------|
| 模型 | Qwen3-4B |
| GPU | 8× A100 40GB |
| 算法 | GRPO |
| n_samples | 16 |
| max_response | 8192 tokens |
| 数据 | dapo-math-17k |
| 奖励 | math_dapo（基于规则判分，匹配 slime 的 deepscaler） |

## 两个测试脚本

### 1. 同步测试（Hybrid Engine）

```bash
bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test.sh
```

- **入口**: `verl.trainer.main_ppo`
- **架构**: FSDP hybrid engine，8 GPU 共享训练和推理
- **推理后端**: vLLM
- **对比标的**: slime 同步模式（4 卡训练 + 4 卡推理，串行）

### 2. 异步测试（Fully Async Policy）

```bash
bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test_async.sh
```

- **入口**: `verl.experimental.fully_async_policy.fully_async_main`
- **架构**: FSDP2，4 GPU 训练 + 4 GPU 推理（分离，流水线并行）
- **推理后端**: vLLM V1 引擎（异步模式）
- **对比标的**: slime 异步模式（4 卡训练 + 4 卡推理，重叠执行）

## 数据准备

```bash
# 下载模型
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B

# 下载数据集
huggingface-cli download zhuzilin/dapo-math-17k --local-dir /workspace/datasets/dapo-math-17k
huggingface-cli download zhuzilin/aime-2024 --local-dir /workspace/datasets/aime-2024

# 转换为 parquet 格式（如果下载的是 jsonl）
python3 -c "
import json, pandas as pd
data = [json.loads(l) for l in open('/workspace/datasets/dapo-math-17k/dapo-math-17k.jsonl')]
pd.DataFrame(data).to_parquet('/workspace/datasets/dapo-math-17k/train.parquet')
"
```

## 运行

```bash
# 同步测试
TRAIN_FILE=/workspace/datasets/dapo-math-17k/train.parquet \
TEST_FILE=/workspace/datasets/aime-2024/test.parquet \
MODEL_PATH=/workspace/models/Qwen3-4B \
bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test.sh

# 异步测试
TRAIN_FILE=/workspace/datasets/dapo-math-17k/train.parquet \
TEST_FILE=/workspace/datasets/aime-2024/test.parquet \
MODEL_PATH=/workspace/models/Qwen3-4B \
bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test_async.sh
```

## 查看日志

```bash
# TensorBoard
tensorboard --logdir checkpoints/verl_perf_test/ --port 6006

# Wandb（如果配置了）
# 在 wandb 控制台查看
```

## Metric 映射

| slime metric | verl metric | 来源 | 说明 |
|---|---|---|---|
| rollout_time | `timing_s/gen` | ray_trainer.py marked_timer | 生成阶段 |
| log_probs_time | `timing_s/old_log_prob` | ray_trainer.py marked_timer | 旧策略 log_prob |
| actor_train_time | `timing_s/update_actor` | ray_trainer.py marked_timer | Actor 更新 |
| ref_log_probs | `timing_s/Role.RefPolicy` | ray_trainer.py marked_timer | 参考策略 |
| update_weights | `timing_s/update_weights` | ray_trainer.py marked_timer | 权重同步 |
| data_preprocess | 不单独统计 | — | 包含在 train 中 |
| step_time | `perf/time_per_step` / `timing_s/step` | metric_utils.py | 每步总时间 |
| tokens/s/GPU | `perf/throughput` | metric_utils.py | token/s/GPU |
| response_len | `response_length/mean` | metric_utils.py | 响应长度 |
| truncated_ratio | `response/aborted_ratio` | metric_utils.py | 截断比例 |
| reward | `critic/rewards/mean` | metric_utils.py | 奖励均值 |
| pg_loss | `actor/pg_loss` | engine_workers.py | 策略梯度损失 |
| entropy | `actor/entropy` | engine_workers.py | 熵 |
| grad_norm | `actor/grad_norm` | engine_workers.py | 梯度范数 |
| approx_kl | `actor/approx_kl` | engine_workers.py | 近似 KL |

## 对比表格模板

运行后，从 wandb/TensorBoard 收集数据填入：

### 时间分解对比（同步模式）

| 阶段 | verl sync (s) | slime sync (s) | 差异 |
|------|--------------|---------------|------|
| rollout_time | | 224.2 | |
| log_probs_time | | 16.8 | |
| actor_train_time | | 113.9 | |
| update_weights | | 0.53 | |
| **step_time** | | **386.0** | |
| **throughput** (tok/s/GPU) | | **~908** | |

### 时间分解对比（异步模式）

| 阶段 | verl async (s) | slime async (s) | 差异 |
|------|---------------|----------------|------|
| rollout_time | | 223.1 | |
| log_probs_time | | 16.9 | |
| actor_train_time | | 115.0 | |
| update_weights | | 0.45 | |
| **step_time** | | **239.2** | |
| **throughput** (tok/s/GPU) | | **~923** | |

### 训练质量对比

| 指标 | verl | slime |
|------|------|-------|
| pg_loss 范围 | | -0.09 ~ +0.09 |
| entropy_loss 起始→结束 | | 0.50 → 0.24 |
| grad_norm 均值 | | ~0.24 |
| reward 均值 | | 51-53% |
| response_len 均值 | | ~6400 |

## 架构差异（解读结果时注意）

| 维度 | slime | verl |
|------|-------|------|
| **训练后端** | Megatron (TP2+DP2) | FSDP / FSDP2 |
| **推理后端** | SGLang | vLLM |
| **GPU分配** | 4训练+4推理（分离） | 同步: 8共享 / 异步: 4+4分离 |
| **权重同步** | 直接 GPU 间传输 | NCCL / checkpoint-engine |
| **模型加载** | Megatron torch_dist | HuggingFace + FSDP |
| **序列并行** | 开启 | 默认关闭（FSDP 场景） |
| **重计算** | full + uniform | gradient_checkpointing |

这些架构差异可能导致：
- verl 同步模式使用 8 GPU（混合引擎），slime 同步使用 4+4（分离）。verl 混合引擎的 GPU 利用率可能更高，因为 FSDP 对所有 GPU 进行统一调度
- slime 使用 Megatron TP=2 + sequence parallel，verl FSDP 不使用 TP。对于 4B 模型，FSDP 的开销可能低于 TP
- 推理引擎不同（vLLM vs SGLang），可能导致 rollout 速度差异（通常在 10-20% 以内）
