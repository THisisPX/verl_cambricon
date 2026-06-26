# Verl vs Slime 性能差异根因分析

> 基于 Qwen3-4B GRPO colocate 模式的 8 步同步训练数据

## 数据回顾

| 阶段 | Slime TP4 | Verl TP4 | Δ | 占比 |
|------|:---:|:---:|:---:|:---:|
| rollout/gen | 275.9s | 258.4s | -17.5s | 42% |
| actor_train/update_actor | 57.0s | 55.7s | -1.3s | 3% |
| log_probs/old_log_prob | 16.3s | 20.2s | +3.9s | — |
| sleep + wake_up | 11.2s | 0s | -11.2s | 27% |
| update_weights | 1.1s | 2.2s | +1.1s | — |
| data_preprocess | 0.14s | 0s | -0.14s | — |
| **step_time** | **389.9s** | **348.4s** | **-41.5s** | **100%** |

---

## 差异 1: Offload 切换机制 (11.2s, 占 27%)

### Slime

Slime 的 SGLang 推理引擎是**独立进程**——不在 Megatron 训练进程内，有自己的 NCCL 通信组做 TP2 推理。

当训练结束、准备推理时 (`slime/backends/megatron_utils/actor.py:156-173`)：

```python
@timer
def sleep(self) -> None:
    clear_memory(clear_host_memory=True)
    # SGLang 是独立进程，如果不拆进程组，Megatorn 的 NCCL rank
    # 和 SGLang 的 NCCL rank 可能在同一个 GPU 上冲突
    destroy_process_groups()           # ← 拆 NCCL: 所有 GPU 的通信连接全部断开
    torch_memory_saver.pause()         # ← 卸载模型到 CPU
```

推理结束后准备训练 (`actor.py:175-184`)：

```python
@timer
def wake_up(self) -> None:
    torch_memory_saver.resume()        # ← 加载模型回 GPU
    clear_memory()
    reload_process_groups()            # ← 重建 NCCL: 重新协商所有 rank 的连接
```

`destroy_process_groups` + `reload_process_groups` 需要：
- 所有 GPU 上的所有 rank 同步等待
- 重新建立 CUDA IPC 共享内存
- 重建 NCCL 环形拓扑

实测耗时：**sleep 7.4s + wake_up 3.8s = 11.2s/步**。

### Verl

Verl 的 vLLM 和 Megatron 在**同一进程**内。NCCL 通信组在训练启动时建立一次，全程保持。offload 只需要搬运数据 (`verl/workers/engine/base.py:259-265`)：

```python
def __enter__(self):
    self.engine.mode = self.mode
    self._context_switch("cuda")       # 模型→GPU
    # NCCL 组不变！

def __exit__(self, exc_type, exc_val, exc_tb):
    self._context_switch("cpu")        # 模型→CPU
    # NCCL 组不变！
```

`_context_switch` 调用 `offload_megatron_model_to_cpu` / `load_megatron_model_to_gpu` (`megatron_utils.py`)，只是搬运 tensor 数据：`param.data.cpu_data.copy_(...)` + `storage.resize_(0)`。不需要拆建任何进程组。

**结论**: Verl 每步比 Slime 省 11.2s，因为 HybridEngine 将训练和推理放在同一进程内，避免了跨进程 colocate 必需的 NCCL 拆建开销。

---

## 差异 2: 推理引擎 (17.5s, 占 42%)

| | Slime SGLang | Verl vLLM |
|---|---|---|
| rollout 时间 | 275.9s | 258.4s |
| engines 数量 | 4 × TP2 | 4 × TP2 |
| 显存比例 | 0.35 | 0.35 |
| max_seqs | 32 | 32 |
| CUDA graph | on (max_bs=8) | off (enforce_eager=True) |

17.5s 的差异 (6.3%) 来自 SGLang 和 vLLM 的引擎实现差异。尽管推理配置相同，但：

- **SGLang 的内部调度开销**：SGLang 使用 RadixAttention + prefix caching + 独立的 continuous batching 调度器。slime 的数据显示 prefix_cache_hit_rate 仅 21%，说明大多数请求没有享受到缓存。
- **vLLM 的 chunked_prefill 开启**：verl 脚本设置了 `enable_chunked_prefill=True`，大 prompt 被切分预填充，减少长请求阻塞短请求。
- **SGLang 的 CUDA graph 开启**：slime 有 `--sglang-cuda-graph-max-bs 8`（CUDA graph 最大 batch size 为 8），而 verl 设了 `enforce_eager=True`（禁用 CUDA graph）。CUDA graph 理应加速，但 max_bs=8 对 batch size=16 的场景帮助有限。

**结论**: vLLM 在此配置下比 SGLang 快 6.3%，主要由 chunked_prefill 和内部调度差异导致。这不代表 vLLM 在所有场景都更好，也不代表 SGLang 差——只是在此特定配置下引擎实现效率的差异。

---

## 差异 3: Offload 数据搬运策略

这里不是时间差异，但解释了为什么 **verl 能用 TP2 而 slime 不能**。

### Slime 的 `torch_memory_saver`

`torch_memory_saver` 是 PyTorch 层面的 CUDA allocator hook。它拦截 CUDA 内存分配请求，当 GPU 显存不足时将**已存在的 tensor** 搬到 CPU，释放空间后再恢复。

问题在于 Megatron 的 `param_data` 是连续的大 buffer（DDP buffer 包含 TP 组内所有参数）。`torch_memory_saver` 搬运这个大 buffer 时，如果要重新分配 CPU pinned memory，会产生 **transient 2x peak**——旧 buffer 还没释放，新 buffer 已经分配，峰值内存翻倍。A100 40GB 下这个峰值足以 OOM。

Slime 被迫用 TP4（每卡权重 2GB 而非 4GB），把单卡权重压到一半来避免这个峰值。

### Verl 的三路卸载 + 预分配 CPU buffer

`offload_megatron_model_to_cpu` (`verl/utils/megatron_utils.py:494-573`) 采用了不同的策略：

```python
# 只分配一次 CPU buffer，后续复用
existing = getattr(buffer.param_data, "cpu_data", None)
if existing is None:
    buffer.param_data.cpu_data = torch.empty(
        buffer.param_data.size(),
        dtype=buffer.param_data.dtype,
        device="cpu", pin_memory=True
    )
# 拷贝到预分配的 buffer，然后释放 GPU 存储
buffer.param_data.cpu_data.copy_(buffer.param_data.data, non_blocking=False)
buffer.param_data.storage().resize_(0)   # 彻底释放 GPU 显存
```

同时卸载三个东西：
1. **param (bf16)**: `param_data.storage().resize_(0)` — 释放 GPU 存储
2. **grad (fp32)**: `grad_data.storage().resize_(0)` — 释放 GPU 存储
3. **optimizer state (fp32)**: `exp_avg.to("cpu")` + `exp_avg_sq.to("cpu")` — Adam 状态移到 CPU

卸载后还清理：
```python
gc.collect()
torch.cuda.empty_cache()
get_global_memory_buffer().buffer.clear()    # Megatron 全局通信 buffer
_dummy_wgrads.clear()                        # TransformerEngine 临时梯度
```

**结论**: Verl 用预分配的 CPU pinned buffer 避免了 transient 2x peak，加上同时卸载 param + grad + optimizer state + 清理 Megatron/TransformerEngine 全局 buffer，释放得比 slime 更彻底。因此 verl TP=2 能在 A100 40GB 上运行，而 slime 需要 TP=4。

---

## 差异 4: 训练循环结构

### Slime (actor.py:408-511)

```python
def train_actor(self, rollout_id, rollout_data, external_data=None):
    data_iterator, num_microbatches = get_data_iterator(...)
    
    # 1. log_probs (可选, 如果 keep_old_actor 或 mismatch check)
    compute_log_prob(...)
    
    # 2. 计算 advantage
    compute_advantages_and_returns(args, rollout_data)
    
    # 3. 训练
    train(rollout_id, model, optimizer, scheduler, data_iterator, num_microbatches)
    
    # 4. 备份权重到 CPU
    self.weights_backuper.backup("actor")
```

Slime 的 `train()` 是 Megatron 原生训练循环——每次需要重新构建 data iterator，且内部包含 `TrainProfiler.step()` 和 routing replay 检查（`os.environ["ROUTING_REPLAY_STAGE"]`）。

### Verl (ray_trainer.py:1293-1333)

```python
def _update_actor(self, batch):
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)
    
    # 传参: mini_batch_size 已在 driver 上计算好
    tu.assign_non_tensor(batch_td, 
        global_batch_size=ppo_mini_batch_size,
        mini_batch_size=ppo_mini_batch_size,
        epochs=ppo_epochs,
        compute_loss=True
    )
    actor_output = self.actor_rollout_wg.update_actor(batch_td)
```

Verl 的更新更轻量——advantage 在 driver 上算（CPU 0.05s，不计入 GPU 时间），data iterator 在 worker 内部通过 `tu.make_iterator` 创建，不需要反复重建。

**结论**: TP 相同时 (均为 4)，训练时间接近 (57.0s vs 55.7s, 差异 < 3%)。训练循环实现差异在此不是主要因素。

---

## 差异 5: 权重同步

| 方面 | Slime | Verl |
|------|-------|------|
| 方法 | `UpdateWeightFromTensor` | NCCL checkpoint engine |
| 机制 | 直接 GPU→GPU 拷贝 | 序列化 → NCCL broadcast → 反序列化 |
| 耗时 | 1.1s | 2.2s |
| 原因 | 训练和推理同 GPU，直接拷贝 | 需要 Megatron TP→vLLM TP reshard |

Slime colocate 下用 `UpdateWeightFromTensor`——权重从训练端拷贝后直接写到 SGLang 引擎的参数。Verl 需要做 reshard（Megatron 的参数分布在 TP 组间，而 vLLM 的参数分布在另一个 TP 布局），因此多了序列化和重建参数 tensor 的步骤。

绝对差异只有 1.1s，占 step_time 的 0.3%，可以忽略。

---

## 总结

```
Slime 389.9s
  │
  ├── 11.2s (2.9%)  ─→ NCCL 拆建    │ 架构差异  
  ├── 17.5s (4.5%)  ─→ 推理引擎     │ (共 28.7s)
  └──  1.3s (0.3%)  ─→ 训练/调度    │
  │
  ▼
Verl TP4 348.4s
  │
  ├── 11.3s (3.2%)  ─→ TP2 vs TP4   │ 并行策略
  │
  ▼
Verl TP2 337.1s
```

| 根因 | 大小 | 分类 | 可避免？ |
|------|:---:|------|:---:|
| **NCCL 拆建** | 11.2s | Slime 架构限制 | 如果 Slime 将 SGLang 嵌入到同一进程可避免 |
| **推理引擎** | 17.5s | vLLM vs SGLang | 切换到相同引擎可对齐，但框架绑定自身推理引擎 |
| **并行策略** | 11.3s | TP2 vs TP4 | Slime 用 TP2 会 OOM，需要优化 offload |
| **权重同步** | 1.1s | reshard 开销 | 此场景可忽略 |
| **训练循环** | 1.3s | 实现细节 | 同 TP 时差异可忽略 |

核心建议：
1. **Slime 优化 offload 取消 NCCL 拆建** — 如果能像 verl 一样把推理引擎嵌入训练进程，省 11.2s/步
2. **Slime 优化 offload buffer 管理** — 采用预分配 CPU buffer 避免 transient 2x peak，可能让 TP2 可运行，省 11.3s/步
3. **两个优化叠加**: slime 可以从 390s 降到 ~367s，缩小与 verl 的差距

---

*分析日期: 2026-06-17*
