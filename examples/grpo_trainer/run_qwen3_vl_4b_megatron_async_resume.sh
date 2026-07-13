#!/usr/bin/env bash
# ==============================================================================
# Resume wrapper for run_qwen3_vl_4b_megatron_async.sh
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_vl_4b_megatron_async_resume.sh
#   TRAINER_STEPS=16 bash examples/grpo_trainer/run_qwen3_vl_4b_megatron_async_resume.sh
#
# Key differences from the main script:
#   1. resume_mode=auto         → 从最新 checkpoint 继续训练
#   2. total_rollout_steps=2101 → 全数据集, rollouter 不断供
#   3. total_training_steps     → 可控
# ==============================================================================

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- override defaults for resume ----
export TRAINER_STEPS="${TRAINER_STEPS:-16}"
export TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-2101}"

# ---- invoke main script ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

bash "${SCRIPT_DIR}/run_qwen3_vl_4b_megatron_async.sh" \
    trainer.total_training_steps="${TRAINER_STEPS}" \
    trainer.resume_mode=auto \
    "$@"
