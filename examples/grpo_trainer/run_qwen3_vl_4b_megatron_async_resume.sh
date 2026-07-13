#!/usr/bin/env bash
# ==============================================================================
# Resume wrapper for run_qwen3_vl_4b_megatron_async.sh
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_vl_4b_megatron_async_resume.sh
#
# Key differences from the main script:
#   1. resume_mode=auto         → 从最新 checkpoint 继续训练
#   2. total_rollout_steps 更大  → rollouter 不会提前退出
#   3. total_training_steps 更大 → 可以跑足够多步收集性能数据
#
# Override with env vars:
#   TRAINER_STEPS=24  bash ...  (跑 24 步)
#   TRAINER_STEPS=12  bash ...  (跑 12 步)
# ==============================================================================

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- new params for resume (overrides main script defaults) ----
TRAINER_STEPS="${TRAINER_STEPS:-24}"

# total_rollout_steps 必须足够大以保证 rollouter 持续生成直到 trainer 跑完。
# trainer 每步吃 ppo_mini_batch_size=64 个样本, n=8 → 8 prompts/step
# 24 steps × 8 prompts × 3 buffer = 576
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-576}"

# ---- invoke main script with resume overrides ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

bash "${SCRIPT_DIR}/run_qwen3_vl_4b_megatron_async.sh" \
    trainer.total_training_steps="${TRAINER_STEPS}" \
    trainer.resume_mode=auto \
    "rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}" \
    "$@"
