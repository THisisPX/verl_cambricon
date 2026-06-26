# Verl vs Slime 最终对比报告

## 实验总览

| 实验 | 框架 | 训练后端 | 推理引擎 | GPU | 模式 | max_resp | step_time | 数据点 |
|------|------|:---:|------|------|------|:---:|:---:|:---:|
| Slime | Slime | Megatron TP4 | SGLang 4×TP2 | 8 colocate | 同步 | 8192 | **389.9s** | 8 |
| verl-vLLM-TP4 | Verl | Megatron TP4 | vLLM 2×TP4 | 8 HybdEng | 同步 | 8192 | **348.4s** | 8 |
| verl-vLLM-TP2 | Verl | Megatron TP2 | vLLM 4×TP2 | 8 HybdEng | 同步 | 8192 | **337.1s** | 8 |
| verl-SGLang-TP4 | Verl | Megatron TP4 | SGLang 4×TP2 | 8 HybdEng | 同步 | 8192 | **477.3s** | 8 |
| verl-Async-4096 | Verl | Megatron TP2 | vLLM 2×TP2 | 4+4 分离 | **异步** | **4096** | **234.3s** | 32 |
| Slime | Slime | Megatron TP2 | SGLang 2×TP2 | 4+4 分离 | 异步 | 8192 | 239.2s* | 8 |

*\*参考 slime 之前 disaggregated async 报告（非 colocate）*

---

## 一、同步 Colocate 对比（8 GPU 共享, 8192 token）

### 时间分解

| 阶段 | Slime TP4 | Verl vLLM TP4 | Verl vLLM TP2 | Verl SGLang TP4 |
|------|:---:|:---:|:---:|:---:|
| rollout / gen | 275.9s | **258.4s** | 260.8s | **384.6s** |
| update_actor | 57.0s | 55.7s | **39.1s** | 55.4s |
| log_probs / old_log_prob | 16.3s | 20.2s | 14.8s | 18.5s |
| sleep + wake_up | **11.2s** | — | — | — |
| update_weights | 1.1s | 2.2s | 2.5s | 2.1s |
| **step_time** | **389.9s** | **348.4s** | **337.1s** | **477.3s** |

### 结论

| 排名 | 实验 | step_time | vs Slime |
|:---:|------|:---:|:---:|
| **🥇** | **Verl vLLM TP2** | **337.1s** | **-13.5%** |
| 🥈 | Verl vLLM TP4 | 348.4s | -10.6% |
| 🥉 | Slime TP4 | 389.9s | — |
| 4 | Verl SGLang TP4 | 477.3s | +22.4% |

**两个核心因素**：

1. **Offload 架构（Slime → Verl TP4：-41.5s）**: Slime 的 SGLang 是独立进程，colocate 时必须拆建 NCCL 进程组（sleep 7.4s + wake_up 3.8s = 11.2s）。Verl HybridEngine 的 vLLM 在同一进程内，不拆建 NCCL。

2. **TP 策略（Verl TP4 → TP2：-11.3s）**: Verl offload 更彻底（三路卸载 + 预分配 CPU buffer），使 TP2 可在 A100 40GB 上运行。TP2 的 all-reduce 跨 2 卡而非 4 卡，训练快 30%。

### Verl SGLang 为什么慢？

**rollout 384.6s vs vLLM 258.4s（+49%）**。根因是 verl SGLang 的 HTTP Server 模式——每条请求都要走 HTTP POST 而非进程内调用，并且四条引擎中总有一条最慢（slowest=382s），并行效率差。

### SGLang 在谁手里更好？

| | Slime SGLang | Verl SGLang |
|------|:---:|:---:|
| rollout | 275.9s | 384.6s |
| SGLang 集成方式 | 独立进程，SGLang 原生 | HTTP Server + Ray actor |
| 结论 | **Slime + SGLang 更快** | verl SGLang 集成有优化空间 |

---

## 二、异步 4+4 分离对比

### Verl Async 4096（A100 40GB, 4+4, Megatron TP2）

| 指标 | 值 | 说明 |
|------|:---:|------|
| gen | 192.4s (104-304) | 4096 token 生成，理论比 8192 快 |
| update_actor | 40.2s | TP2，与同步 TP2 接近 |
| param_sync | 1.4s | NCCL checkpoint engine 同步权重 |
| **step_time** | **234.3s** | |
| idle_ratio (训练端) | **82%** | ⚠️ **训练端大部分时间在等推理** |
| idle_ratio (推理端) | 6% | 推理端几乎满负荷 |
| clip_ratio | **79%** | 4096 token 被大量截断 |
| response_length/mean | 3847 | 非常接近 4096 上限 |
| throughput | 282 tok/s/GPU | |

### 流水线瓶颈

```
训练端: ── wait 192s ── train 40s ── wait 192s ──  (idle 82%)
推理端: ── gen ── gen ── gen ── gen ── gen ──  (idle 6%)
```

训练端 82% 时间在等推理——只有 2 个 vLLM 引擎（4 GPUs TP2），但要喂饱 4 张训练卡。**推理并行度太低**。

### Slime Async 参考（8192, 4+4）

| 指标 | Slime async | Verl async 4096 |
|------|:---:|:---:|
| step_time | 239.2s | 234.3s |
| rollout_time | 223.1s | 192.4s |
| idle 训练端 | — | **82%** |
| max_resp | 8192 | **4096** |

Verl 在 4096 token 下的步时接近 slime 在 8192 下的结果——但如果 verl 也要跑 8192 token，步时会远高于 slime 的 239s。

---

## 三、训练质量验证

| 指标 | Slime | Verl-vLLM-TP2 | Verl-vLLM-TP4 | Verl-SGLang | Verl-Async-4096 |
|------|:---:|:---:|:---:|:---:|:---:|
| pg_loss | ~0.00 | 0.01 | 0.01 | 0.01 | 0.01 |
| entropy | 0.33 | 0.35 | 0.35 | 0.35 | — |
| grad_norm | 0.18 | 0.09 | 0.11 | 0.10 | 0.10 |
| clip_ratio | 47% | 50% | 50% | 48% | **79%** |
| response_len/mean | 6373 | 6667 | 6644 | 6628 | **3847** |

训练动力学在所有实验中一致，验证算法实现正确。

---

## 四、最终总结

### 性能排名（同步 colocate）

```
🥇 Verl Megatron TP2 + vLLM    337s  (快 13.5% vs slime)
🥈 Verl Megatron TP4 + vLLM    348s  (快 10.6%)
🥉 Slime Megatron TP4 + SGLang 390s  (baseline)
   Verl Megatron TP4 + SGLang  477s  (慢 22%, HTTP Server 模式瓶颈)
```

### 异步模式现状

- 4096 token 能跑通但 **82% pipeline idle**（推理引擎太少）
- 8192 token 在 A100 40GB 上 OOM（4 卡训练 TP2/DP2 显存不够）
- 需要 B300 或更多 GPU 才能公平对比异步性能

### 推荐的生产组合

| 场景 | 推荐方案 |
|------|------|
| 8×A100 40GB 同步训练 | **Verl + Megatron TP2 + vLLM** |
| 8×A100 40GB 异步训练 | Slime（SGLang 集成更高效） |
| 推理引擎 | vLLM（verl 内比 SGLang 快 49%） |
| B300 / 大显存 | Verl 或 Slime 均可（待验证） |

---

*报告日期: 2026-06-26*
