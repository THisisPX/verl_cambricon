#!/usr/bin/env bash
# Performance Test: Qwen3-4B | GRPO | FSDP + vLLM
#
# Designed to compare with slime framework's Qwen3-4B GRPO results.
# Reference: E:/learning/slime/reports/sync-vs-async-comparison.md
#
# Slime: 8×A100 40GB, Megatron TP2×DP2 (train) + SGLang 2engines×TP2 (rollout)
# Verl:  8×A100 40GB, FSDP hybrid engine (all GPUs shared for train + rollout)
#
# Key differences to note in comparison:
# - Slime uses 4 dedicated GPUs for training + 4 for rollout (disaggregated)
# - Verl FSDP uses hybrid engine: all 8 GPUs do both training and rollout
# - Slime backend: Megatron; Verl backend: FSDP
# - Slime inference: SGLang; Verl inference: vLLM
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test.sh
#
# Override env vars:
#   TRAIN_FILE, TEST_FILE, MODEL_PATH, NNODES, NGPUS_PER_NODE, etc.

set -xeuo pipefail

# ======================== slime-matching defaults ========================
MODEL_PATH=${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- batch / rollout (matching slime's parameters) ---
# slime: rollout_batch_size=8, n_samples=16, global_batch_size=32
# verl:  train_batch_size = prompts per step
#        rollout.n = responses per prompt
#        total trajectories per step = train_batch_size * n
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

# --- algorithm ---
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# --- rollout (vLLM) ---
ROLLOUT_TP=${ROLLOUT_TP:-2}
# Lower mem util because FSDP hybrid engine shares GPUs with training
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
ROLLOUT_N=${ROLLOUT_N:-16}

# --- experiment tracking ---
PROJECT_NAME=${PROJECT_NAME:-verl_perf_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192}
# Disable checkpointing and validation for perf test
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
SAVE_FREQ=${SAVE_FREQ:-9999}
TEST_FREQ=${TEST_FREQ:-9999}

# ======================== parameter arrays ========================

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # Dynamic batch: max tokens per GPU must accommodate max_prompt + max_response
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12000
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    # vLLM-specific: larger bucket for faster weight sync
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12000
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    # Performance test: skip checkpointing and validation
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=False
)

EXTRA=(
    reward.num_workers=1
)

# ======================== launch ========================
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
