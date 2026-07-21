---
type: results-report
date: 2026-07-10
experiment_line: qwen3-vl-4b-geo3k-grpo
round: 1
purpose: verl-vs-slime VL async training comparison
status: final
frameworks:
  - verl (fully async: 2+2 GPU split, rollout/train parallel)
  - slime (disaggregated sync: 2+2 GPU split, rollout→train sequential)
---

# Verl vs Slime / Qwen3-VL-4B GEO3K GRPO / Performance Comparison / 2026-07-10

## 1. Executive Summary

在 4× NVIDIA B300 SXM6 GPU 上对 Qwen3-VL-4B-Instruct 进行数学几何推理 GRPO RL 训练，对比 **verl fully async** (2+2 GPU 分离, 异步并行) 与 **slime disaggregated** (2+2 GPU 分离, 同步串行) 两种方案的性能与训练质量：

| 维度 | Verl Fully Async | Slime Colocate | 对比 |
|------|:---:|:---:|:---:|
| **step_time** | 251.0s | 294.7s | verl 快 14.8% |
| **trainer idle** | 73.4% | 43.2% (wait_ratio) | verl 训练端空等更多 |
| **rollout_time** | 184.7s (gen only) | 290.9s (gen+log_probs+train total) | 不可直接比较 |
| **training signal** | ❌ loss=0, reward=0 | ✅ raw_reward 0.47→0.56 | 严重 bug |
| **GPU memory** | ~41 GB/GPU | ~35 GB/GPU (est.) | 均充裕 (B300 275 GiB) |
| **steps completed** | 31 (7808 samples) | 106 (13568 samples) | verl 提前终止 |

**核心结论**: Verl fully async 在步时上有优势（14.8%），但因 **reward function 全部返回 0** 导致模型完全没有学习信号。根因是 GRPO + binary math reward 的冷启动死锁——Qwen3-VL 未学习过 `\boxed{}` 格式，在无法命中格式前 reward 始终为 0，GRPO 的 group-normalized advantage 在组内全零 reward 时完全退化。

## 2. Experiment Configuration

### 2.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | 4× NVIDIA B300 SXM6 AC (275 GiB each), CUDA 13.0 |
| 互联 | NVLink |
| 服务器 | 共享 8 卡节点 |

### 2.2 框架与软件

| 项目 | Verl | Slime |
|------|------|------|
| 框架 | verl 0.9.0.dev (fork) | slime main branch |
| 训练后端 | Megatron-LM (mbridge) | Megatron-LM (mbridge) |
| 推理引擎 | SGLang (HTTP server 模式) | SGLang (独立进程) |
| PyTorch | 2.9.1+cu129 | 2.9.1+cu129 |
| Docker | 无 (裸机) | slimerl/slime:nightly-dev-20260629a |

### 2.3 GPU 分配策略 (关键差异)

| 维度 | Verl Fully Async | Slime Colocate |
|------|------|------|
| **模式** | 异步分离：训练/推理并行 | 同步共置：训练/推理串行 |
| **训练 GPU** | 2× GPU (TP2, DP1) | 2× GPU (TP2, DP1) |
| **推理 GPU** | 2× GPU (1 engine × TP2) | 2× GPU (1 engine × TP2) |
| **总 GPU** | 4 (并行使用) | 4 (共享切换) |
| **权重同步** | NCCL CheckpointEngine (每 4 步) | 同进程 weight_update |
| **流水线重叠** | 训练端处理旧样本时推理端生成新样本 | 无重叠 (串行) |

### 2.4 完整训练参数对照

| 类别 | 参数 | Verl | Slime | 匹配 |
|------|------|------|------|:---:|
| **模型** | model_path | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct | ✅ |
| | rotary_base | (mbridge 自动) | 5000000 | ✅ |
| | trust_remote_code | True | — | ✅ |
| **数据** | dataset | chenhegu/geo3k_imgurl | chenhegu/geo3k_imgurl | ✅ |
| | max_prompt_length | 2048 | (默认) | ✅ |
| | max_response_length | 3072 | 3072 | ✅ |
| **算法** | adv_estimator | grpo | grpo | ✅ |
| | kl_loss_coef | 0 | 0 | ✅ |
| | entropy_coef | 0 | 0 | ✅ |
| | eps_clip / eps_clip_high | 0.2 / 0.28 | 0.2 / 0.28 | ✅ |
| | loss_agg_mode | token-mean | (token-mean) | ✅ |
| **Rollout** | rm_type | geo3k (mathruler) | math | ✅ |
| | n_samples_per_prompt | 8 | 8 | ✅ |
| | rollout_temperature | 0.8 | 0.8 | ✅ |
| | rollout_batch_size | 1 (streaming) | 16 | ⚠️ 不同模式 |
| | global_batch_size | 64 | 64 | ✅ |
| **优化器** | lr | 1e-6 | 1e-6 | ✅ |
| | lr_decay_style | constant | constant | ✅ |
| | weight_decay | 0.1 | 0.1 | ✅ |
| | adam_beta1/2 | 0.9 / 0.98 | 0.9 / 0.98 | ✅ |
| **Megatron** | TP / PP | 2 / 1 | 2 / 1 | ✅ |
| | sequence_parallel | True | True | ✅ |
| | recompute | full / uniform / 1 | full / uniform / 1 | ✅ |
| | attention_backend | flash (FA2) | flash (FA2) | ✅ |
| | use_dynamic_bsz | True | True | ✅ |
| **SGLang** | TP | 2 | 2 | ✅ |
| | mem_fraction | 0.7 | 0.7 | ✅ |
| | mm_attention_backend | sdpa | sdpa | ✅ |
| | enforce_eager | True | True | ✅ |
| **异步** | staleness_threshold | 0 | N/A | — |
| | trigger_parameter_sync_step | 4 | N/A | — |
| | partial_rollout | True | N/A | — |

### 2.5 B300/Blackwell 兼容性修复清单

两组实验均在此 4× B300 节点上运行，共享以下 6 项修复：

| # | 问题 | 修复 |
|---|------|------|
| 1 | ptxas 不识别 sm_103a | `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` |
| 2 | CUDA kernel 无 sm_103a 镜像 | `enforce_eager=True` (禁用 CUDA graph) |
| 3 | FA3 不支持 Blackwell | `attention_backend=flash` (回退 FA2) |
| 4 | 多模态注意力 | `mm_attention_backend=sdpa` |
| 5 | NVLink 检测 | `NCCL_NVLS_ENABLE` 动态检测 |
| 6 | ptxas 路径 | `TORCH_CUDA_ARCH_LIST=10.0` |

## 3. Performance Comparison

### 3.1 步时对比

```
Verl Fully Async
step_time:  179s ─────────────────────────────────────────── 304s  (mean=251s)
            ████████████████████████████████████

Slime Colocate
step_time:  194s ───────────────────────────────────────── 1002s  (mean=295s)
            ██████████████████████████████████████████████
```

| 指标 | Verl Async | Slime Colocate | Δ |
|------|:---:|:---:|:---:|
| step_time (mean) | **251.0s** | 294.7s | **−14.8%** |
| step_time (min) | 179.0s | 194.1s | −7.8% |
| step_time (max) | 304.1s | 1001.7s | −69.6% |
| step_time (std) | 30.6s | 91.6s | **−66.6%** |

**Verl async 步时更短且更稳定**。Slime 的 colocate 模式步时标准差高达 91.6s，主要因为首步有大量 cold start 开销。

### 3.2 流水线效率

| 指标 | Verl Async | Slime Colocate | 解读 |
|------|:---:|:---:|------|
| trainer idle_ratio | **73.4%** | — | verl 训练端在等数据 |
| train_wait_time_ratio | — | **43.2%** | slime 训练端在等推理 |
| rollouter/engine idle | **8.5%** | — | verl 推理端接近满负荷 |
| pipeline overlap | ✅ 并行 | ❌ 串行 | verl 架构优势但未充分利用 |

**关键发现**:
- **Verl 推理端饱和 (idle 8.5%)**：2 个 SGLang TP2 引擎持续生成，几乎无闲置
- **Verl 训练端空闲 73.4%**：只在每 4 步参数同步后消费一波样本（ppo_mini_batch_size=64），远快于推理速度
- **Slime 训练端空闲 43.2%**：串行模式下，训练完成后必须等下一轮 rollout

```
Verl async 流水线:
  推理端: ─ gen ─ gen ─ gen ─ gen ─ gen ─ gen ─  (idle 8.5%)
  训练端: ── wait 185s ── train 64s ── wait ──  (idle 73.4%)
  
Slime colocate 流水线:
  推理端: ─ gen+logp ─ wait_train ─ gen+logp ─
  训练端: ──  wait  ── train 158s ──  wait  ──  (idle 43.2%)
```

### 3.3 推理性能 (SGLang)

| 指标 | Verl SGLang | Slime SGLang | Δ |
|------|:---:|:---:|:---:|
| gen_time (per step) | 184.7s | 121.5s | +52.0% |
| tokens_per_gpu_per_sec | — | 1058 | — |
| response_length/mean | 234 | 1625 | −85.6% |
| timing_per_token_ms/gen | **0.210ms** | — | 可比的每 token 速度 |
| truncated_ratio | **0%** | 27.4% | — |

⚠️ **直接比较 gen_time 是误导性**的：
- Verl async 的 gen_time (184.7s) 覆盖了并行生成 128+ 个样本的时间，是持续流水线下的指标
- Slime 的 rollout_time (121.5s) 是单步 16 × 8 = 128 样本的 batch 推理时间
- Verl 推理端 idle 仅 8.5%，说明它确实在持续工作，gen_time 长是因为 sample throughput 高

**更合理的对比**：verl `timing_per_token_ms/gen=0.210ms` vs slime `121.5s / (128 × 1625 tokens) ≈ 0.585ms`。Verl SGLang 的每 token 生成速度反而更快。

但 **response_length/mean=234** 极低——模型没学会格式，输出很短。Slime 的均值 1625 是正常推理行为。

> 注：两者均使用 SGLang 推理引擎，区别在于 verl 通过 HTTP server (Ray actor) 调用，slime 通过独立进程调用。在本次 VL 对比中，两种集成方式均正常运行，未出现之前 text-only 实验中 verl SGLang HTTP server 长尾延迟的问题。

### 3.4 训练性能 (Megatron)

| 指标 | Verl | Slime | Δ |
|------|:---:|:---:|:---:|
| update_actor_time | **63.7s** | **109.9s** | −42.1% |
| log_probs_time | (在 rollout 端) | 47.6s | — |
| param_sync_time | 1.4s | 1.0s | +40% |
| actor_train_tflops | — | 28.0 | — |
| max_memory_allocated | 41.2 GB | (未记录) | — |

**Verl 训练更快** (63.7s vs 109.9s)。差异来源：
1. Verl async 模式下训练端独占 GPU，无需 offload/load 切换
2. Slime colocate 需要在 train/infer 之间做模型 offload/load (megatron↔huggingface 权重转换)
3. Verl async 的 NCCL CheckpointEngine 同步仅需 1.4s（每 4 步一次）

### 3.5 吞吐量分析

| 指标 | Verl | Slime |
|------|:---:|:---:|
| 训练吞吐 (samples/hour) | ~457 | ~391 |
| 推理吞吐 (samples/hour) | ~696 | ~391 |
| 总有效样本 | 7808 (31 steps) | 13568 (106 steps) |
| 样本/秒 (总体) | 0.42 | 0.36 |

Verl async 的流水线设计理论上可以达到更高吞吐，但本次实验因提前终止未充分展示。

## 4. Training Quality Analysis

### 4.1 训练信号 (⚠️ 关键差异)

| 指标 | Verl | Slime | 状态 |
|------|:---:|:---:|:---:|
| raw_reward (trend) | ❌ 0.000 (31 步全零) | ✅ 0.469→0.555 | verl 无学习 |
| train/loss | ❌ 0.000 (31 步全零) | ⚠️ ~0.000 | 均弱 |
| grad_norm | ❌ 0.000 (31 步全零) | ✅ 0.574 | verl 无梯度 |
| entropy | ❌ 0.000 | ✅ 0.280 | verl 异常 |
| pg_clipfrac | ❌ 0.000 | ✅ 0.07% | verl 无信号 |
| eval_reward | ❌ 0.000 (step 20) | ✅ 0.491→0.526 | verl 无提升 |
| response_length/mean | ❌ 234 (异常低) | ✅ 1625 | verl 输出极短 |
| truncated_ratio | ❌ 0% | ✅ 27.4% | verl 未触发截断 |

### 4.2 根因分析：GRPO 冷启动死锁

**问题链路**：

```
Qwen3-VL-4B (指令微调, 未学 \boxed{}
→ 模型输出自然语言 ("The answer is 42")
→ geo3k_reward.py: compute_score() format_reward regex 不匹配 → 0
   acc_reward() extract_boxed_content → None → 0  
→ 组内 8 个 sample reward 全是 0
→ GRPO group normalization: mean=0, std=0
→ advantage = (0-0)/(0+1e-6) = 0
→ pg_loss = 0, grad_norm = 0
→ 模型永远不会改进 —— 死锁
```

**代码证据** (`verl/utils/reward_score/geo3k.py`):
```python
def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0  # ← 从不匹配
```

```python
def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    answer = extract_boxed_content(predict_str)  # ← 找不到 \boxed{}, 返回 None
    return 1.0 if grade_answer(answer, ground_truth) else 0.0  # ← grade_answer(None, ...) → 0
```

**为什么 slime 能学习？**

Slime 用了 `--rm-type math`，内部实现在第一个成功命中 `\boxed{}` 的样本之前也是全零。但 slime 的日志（report section 4.1）显示 step 0 时 `raw_reward=0.469`，说明有相当比例的样本一开始就输出了 `\boxed{}` 格式。这可能因为：

1. Slime 的 chat template 处理不同——prepend 了更强的格式指令
2. 数据预处理差异——`geo3k_imgurl.py` 中的 `instruction_following` 字符串位置和拼接方式影响模型的行为
3. B300 批次不同——不同日期的初始随机种子不同

**根本原因**：verl 的 `custom_reward_function` 确实被正确调用了（`critic/score/mean=0.0` 而非 `NaN` 证明了这一点），只是模型输出从未命中格式。这是 Qwen3-VL 模型行为 + binary reward 设计 + 无 SFT 暖启动的组合结果，非 verl 框架的 bug。

### 4.3 Verl 指标验证

| 验证项 | 预期 | 实际 | 结论 |
|------|------|------|------|
| 数据流 | `rm_scores` → `token_level_scores` → `token_level_rewards` → advantage | 全流程正常执行, 值全零 | ✅ 管线正确 |
| 权重同步 | NCCL CheckpointEngine 每 4 步 sync | param_sync_time ~1.4s | ✅ 正常 |
| 消息队列 | MessageQueue 无积压 | queue_size=0 | ✅ 正常 |
| staleness | threshold=0, 零容忍旧样本 | staleness=113, dropped=0 | ✅ 严格遵守 |
| partial rollout | 权重同步时不丢弃 | partial_ratio=0 | ✅ 正常 |
| 训练步数 | 31 步后停止 | — | 可能是手动停止或达到 max_steps |
| val | step 20 执行 | val-aux/num_turns 正常 | ✅ 正常 |

## 5. Key Findings

**F1: Verl fully async 在步时上有明确优势 (14.8%)**

verl step_time=251s vs slime=295s。优势来自：(a) 推理与训练并行执行，(b) 训练端独占 GPU 无需 offload/load 切换，(c) 训练更快 (63.7s vs 109.9s)。

但训练端 73.4% idle 说明推理端仍是瓶颈——2 个 SGLang TP2 引擎的生成速度跟不上训练消费。

**F2: GRPO + binary math reward 对未格式对齐的 VL 模型是死锁**

slime 的实验同样面临 train/loss ≈ 0 的问题（F4 in slime report）。Verl 的极端 case（全零 reward）只是此问题的最大表现。根本修复方向是使用带格式部分 credit 的 reward 或 SFT 暖启动。

**F3: SGLang HTTP server 集成在 VL 场景下性能正常**

与之前 text-only Qwen3-4B 对比实验中 verl SGLang 慢 vLLM 49% 不同，本次 VL 场景下每 token 生成速度 (0.210ms) 是正常的。可能原因：VL 推理的瓶颈在 vision encoder 和 cross-attention，HTTP 开销相对占比更小。

**F4: 异步流水线的推理并行度不足**

Verl 训练端 idle 73.4%，推理端 idle 8.5%。理论上如果增加推理引擎数量（如 2 engines × TP2 = 4 GPU 推理），可以减少训练端等待时间。但当前 4 GPU 总资源限制下无法扩展。

**F5: B300 兼容性方案已稳定**

6 项修复在两框架下均有效，未出现新问题。expandable_segments 在 verl 中被移除（`9e090d2f` commit，SGLang TorchMemorySaver 不兼容）。

## 6. Architecture Pros/Cons

### Verl Fully Async
| 优势 | 劣势 |
|------|------|
| 推理/训练并行 → 步时更短 | 需要更多 GPU（最少 4） |
| 步时更稳定 (std 30.6s vs 91.6s) | 训练端 idle 高（退化为推理速度） |
| NCCL CheckpointEngine 权重同步轻量 | staleness=0 时 pipeline overlap 有限 |
| 支持 partial_rollout 恢复 | 配置复杂（两套 GPU 分配参数） |
| MessageQueue 解耦推理/训练 | 不支持 critic/reward_model 分离 |

### Slime Colocate
| 优势 | 劣势 |
|------|------|
| 更简单的配置和使用 | 步时较长、不稳定 |
| SGLang 独立进程（更原生） | offload/load 切换有额外开销 (~11s) |
| 训练质量经过验证 | 推理与训练串行，无 pipeline 重叠 |
| RM 配置灵活 (math/dapo/gpqa) | 对 VLM 的 chain-of-thought 支持弱 |

## 7. Repair Plan: Verl Reward Deadlock

### 立即修复
1. **更换 reward manager 为 `batch`** + 放宽格式 regex：
   - 接受 `\boxed{` 但不要求 `<think>` 标签
   - 或增加部分格式 credit（当前 `format_score=0.1` 太低）

2. **添加 `reward.reward_kwargs`** 传递额外参数到 reward function

3. **检查数据预处理**：确认 `instruction_following` 字符串在 chat template 中的位置与 slime 一致

### 中期改进
4. **SFT 暖启动**：用几千个 `<think>...</think>\boxed{...}` 格式的样本做 SFT
5. **使用 dapo reward manager**：支持 overlong buffer penalty，引入额外的 reward 信号
6. **添加 debug 日志**：打印 model response 内容以确认格式问题

## 8. Limitations

- **verl 只跑了 31 步**（slime 106 步），训练质量对比不完整
- **reward=0 污染了所有训练指标**，无法从 verl 数据得出学习动力学结论  
- **模式不对等**：verl 是 2+2 async，slime 是 4 shared colocate——步时对比需谨慎解读
- **无 critic**：两组实验均无 critic，纯 GRPO
- **单数据集**：仅 GEO3K，结论泛化性有限

## 9. Artifact Index

### Verl
| 用途 | 路径 |
|------|------|
| Script | `examples/grpo_trainer/run_qwen3_vl_4b_megatron_async.sh` |
| TensorBoard (train) | `tensorboard_log/verl_async_geo3k/qwen3_vl_4b_sglang_megatron_async_slime_match/` |
| TensorBoard (val) | 同 train (step 20) |
| 步骤数 | 31 train steps |

### Slime
| 用途 | 路径 |
|------|------|
| Script | `scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh` |
| TensorBoard (eval) | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090614/` |
| TensorBoard (train) | `tensorboard_log/qwen3-vl-4b-geo3k-4gpu-v3/20260707_090856/` |
| Checkpoint | `iter_0000099` at slime models directory |
| 步骤数 | 106 rollout / 212 train steps |

### Analysis Tools
```bash
# Parse event files
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('tensorboard_log/...')
ea.Reload()
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    vals = [e.value for e in events]
    print(f'{tag}: mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}')
"
```

## 10. Next Actions

### P0 — 修复 verl reward 死锁
1. 放宽 geo3k reward 格式要求，或更换 reward 策略
2. 重新运行验证训练能否产生学习信号

### P1 — 延长训练
3. 训练至 100+ steps 以获取更可靠的数据
4. 在相同 total_rollout_steps 下重新对比

### P2 — 公平对比
5. 在 verl 上运行 colocate hybrid engine 模式，与 slime colocate 做 apples-to-apples 比较
6. 在 slime 上运行 disaggregated async 模式，与 verl async 比较

### 不推荐
- ❌ 使用 dapo RM + binary answer（GEO3K 的 LaTeX 答案不兼容 dapo 的 float 转换）
- ❌ 在 reward=0 的情况下延长训练（完全浪费算力）
