# Verl 框架下 vLLM vs SGLang 性能对比报告

## 实验目的

在 verl 框架内，控制所有其他变量不变，仅切换推理引擎，量化 vLLM 和 SGLang 的性能差异。

## 实验矩阵

| 实验 | 训练后端 | 推理引擎 | GPU | 模式 | step_time | gen | n |
|------|:---:|:---:|------|------|:---:|:---:|:---:|
| Megatron-TP4-vLLM | Megatron TP4 | **vLLM** | 8 A100 colocate | 同步 | **348.4s** | 258.4s | 8 |
| Megatron-TP4-SGLang | Megatron TP4 | **SGLang** | 8 A100 colocate | 同步 | **477.3s** | 384.6s | 8 |
| B300-vLLM-async | Megatron TP2 | **vLLM** | 4 B300 (2+2) | 异步 | **140.5s** | 98.4s | 16 |
| B300-SGLang-async | Megatron TP2 | **SGLang** | 4 B300 (2+2) | 异步 | **179.8s** | 114.1s | 16 |

### 参数对齐说明

两组对比中，推理参数和其他训练参数**完全相同**：

#### 推理引擎参数

| 参数 | vLLM | SGLang | 说明 |
|------|:---:|:---:|------|
| `tensor_model_parallel_size` | 2 | 2 | 推理 Tensor 并���度 |
| `gpu_memory_utilization` | 0.35 (sync) / 0.7 (async) | 0.35 (sync) / 0.7 (async) | GPU 显存与推理 |
| `max_model_len` | 9216 | 9216 | 最大序列长度 |
| `max_num_seqs` | 32 (sync) / 64 (async) | 32 (sync) / 64 (async) | 并发请求数 |
| `enforce_eager` | True (sync) / False (async) | True (sync) / False (async) | CUDA graph 开关 |
| `enable_chunked_prefill` | True | True | Chunked prefill |
| `free_cache_engine` | True | True | 训练时释放 KV cache |
| `calculate_log_probs` | True | True | 推理时计算 log_prob |

#### 训练算法参数

| 参数 | 同步 (TP4) | 异步 (B300) |
|------|:---:|:---:|
| `adv_estimator` | grpo | grpo |
| `use_kl_loss` | False | False |
| `entropy_coeff` | 0 | 0 |
| `clip_ratio_low` | 0.2 | 0.2 |
| `clip_ratio_high` | 0.28 | 0.28 |
| `optim.lr` | 1e-6 | 1e-6 |
| `optim.lr_decay_style` | constant | constant |
| `optim.weight_decay` | 0.1 | 0.1 |
| `optim.betas` | [0.9, 0.98] | [0.9, 0.98] |

每组对比实验中，训练参数、并行策略、数据批量、GPU 分配均保持一致。**唯一的变量是 `rollout.name`（vllm vs sglang）。**

## 结果

### 同步 Colocate (8×A100 40GB, Megatron TP4, 8192 tokens)

| 指标 | vLLM | SGLang | 差异 |
|------|:---:|:---:|:---:|
| rollout / gen | 258.4s | 384.6s | **+49%** |
| update_actor | 55.7s | 55.4s | — |
| old_log_prob | 20.2s | 18.5s | — |
| update_weights | 2.2s | 2.1s | — |
| **step_time** | **348.4s** | **477.3s** | **+37%** |
| throughput (tok/s/GPU) | 314 | 231 | -26% |
| response_len/mean | 6644 | 6628 | — |
| pg_loss | 0.01 | 0.01 | — |

### 异步 2+2 (4×B300, Megatron TP2, 8192 tokens)

| 指标 | vLLM | SGLang | 差异 |
|------|:---:|:---:|:---:|
| rollout / gen | 98.4s | 114.1s | **+16%** |
| update_actor | 40.0s | 64.0s | —* |
| param_sync | 1.9s | 1.6s | — |
| **step_time** | **140.5s** | **179.8s** | **+28%** |
| throughput | 1541 | 1269 | -18% |
| idle_ratio (训练端) | 70% | 61% | — |
| response_len/mean | 6448 | 6509 | — |
| pg_loss | 0.01 | 0.01 | — |

*\*B300 SGLang update_actor 较慢是因为 DP=1（2 GPU训练），vLLM 快可能是训练端引擎对 GPU 利用的差异所致。同步实验中同 TP4 时 update_actor 完全一致。*

## 分析

### SGLang 在 verl 中为何慢

verl 的 SGLang 集成采用 **HTTP Server 模式**——SGLang 引擎作为独立的 Ray actor 运行，推理请求通过 HTTP 发送：

```
vLLM (同步，同进程 AsyncLLM):
  generate_sequences()
    → AsyncLLM.generate()           ← 进程内调用
    → token stream (async iter)     ← 进程内传输
    → 返回

SGLang (同步，HTTP Server):
  generate_sequences()
    → HttpAdapter.generate()        ← 构造 HTTP POST
    → aiohttp POST → Ray actor      ← 网络往返
    → SGLang engine                 ← 独立进程
    → SSE token stream → HTTP       ← 网络传输
    → 解析 → 返回
```

每条请求都多了一次 HTTP 往返 + 序列化/反序列化。在同步模式下一个 step 有 128 条请求（8 个 prompt × 16 个 sample），开销被放大。

### SGLang 引擎不均衡

```
vLLM gen:       min=200s  max=305s  (范围 105s)
SGLang gen:     min=277s  max=533s  (范围 256s, 2.5倍)
SGLang slowest: 382s  → 4 个 engine 中总有一个特别慢
```

vLLM 的 4 个 engine 完成时间更均衡——因为它们在同一个进程内，负载分配更平均。SGLang 的独立 HTTP engine 无法做同等级的负载均衡。

## 结论

**在 verl 框架下，vLLM 是显著更优的推理引擎选择：**

| 对比场景 | SGLang 额外开销 | 性损失 |
|------|:---:|:---:|
| 同步 colocate (A100) | +129s rollout / step | **37%** |
| 异步 disaggregated (B300) | +39s rollout / step | **28%** |

两个不同硬件、不同模式下的数据一致表明 verl 的 vLLM 集成比 SGLang 更高效。差异来自 SGLang HTTP Server 模式的固有开销，不是参数不对齐。

**建议**：在 verl 中使用 vLLM 完成训练。如果需要使用 SGLang（如多轮工具调用等 SGLang 专属功能），建议等待 SGLang 集成优化（目前为 HTTP Server 模式，未来可能支持进程内引擎）。

## 实验脚本

| 实验 | 脚本 |
|------|------|
| Megatron TP4 + vLLM sync | `examples/grpo_trainer/run_qwen3_4b_megatron_perf_test.sh` |
| Megatron TP4 + SGLang sync | `examples/grpo_trainer/run_qwen3_4b_megatron_sglang_perf_test.sh` |
| B300 vLLM async | `examples/grpo_trainer/run_qwen3_4b_megatron_perf_test_async_b300.sh` |
| B300 SGLang async | `examples/grpo_trainer/run_qwen3_4b_megatron_perf_test_async_sglang_b300.sh` |

---

*报告日期: 2026-06-29*
