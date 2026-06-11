#!/usr/bin/env bash
# Performance Test (Megatron): Qwen3-4B | GRPO | Megatron + vLLM
#
# Megatron-based version for comparison with slime framework.
# Slime uses Megatron TP2×DP2 (4 GPUs) + SGLang (4 GPUs).
# Verl Megatron uses hybrid engine (all 8 GPUs, TP2×DP4).
#
# This uses the SAME training backend (Megatron) as slime, making it
# a more apples-to-apples comparison than the FSDP version.
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_megatron_perf_test.sh
#
# Override env vars:
#   TRAIN_FILE, TEST_FILE, MODEL_PATH, NNODES, NGPUS_PER_NODE, etc.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ======================== slime-matching defaults ========================
MODEL_PATH=${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k-verl.parquet}
TEST_FILE=${TEST_FILE:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024-verl.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
INFER_BACKEND=${INFER_BACKEND:-vllm}

# --- batch / rollout (matching slime) ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
# Token budget per GPU for dynamic batch packing (12000 is enough for 1 full
# sequence of 9216 tokens with some headroom)
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12000}

# --- algorithm ---
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

# --- Megatron parallelism (matching slime: TP=2) ---
# Qwen3-4B is small enough for TP=2, PP=1
# Total train GPUs = TP * PP * DP = 2 * 1 * 4 = 8 (hybrid engine)
ACTOR_TP=${ACTOR_TP:-2}
ACTOR_PP=${ACTOR_PP:-1}

# --- rollout (vLLM) ---
ROLLOUT_TP=${ROLLOUT_TP:-2}
# Lower gpu_memory_util than FSDP because Megatron training uses more GPU memory
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.3}
ROLLOUT_N=${ROLLOUT_N:-16}

# --- Megatron-specific: sequence parallel + full recompute (matching slime) ---
SEQUENCE_PARALLEL=True

# Full recompute (matching slime: recompute-granularity=full, recompute-method=uniform)
# Single GPU can afford more recompute layers for 4B model
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

# --- experiment tracking ---
PROJECT_NAME=${PROJECT_NAME:-verl_perf_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192_megatron}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
SAVE_FREQ=${SAVE_FREQ:-9999}
TEST_FREQ=${TEST_FREQ:-9999}

# ======================== parameter arrays ========================

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}
    actor_rollout_ref.actor.optim.betas=[${ADAM_BETA1},${ADAM_BETA2}]
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    # Dynamic batch: let Megatron handle micro-batch sizing
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    # Megatron parallelism (TP=2 matches slime)
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    # Sequence parallelism (matches slime's --sequence-parallel)
    ++actor_rollout_ref.actor.megatron.sequence_parallel=${SEQUENCE_PARALLEL}
    # Full recompute (matches slime's --recompute-granularity full --recompute-method uniform)
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=${RECOMPUTE_GRANULARITY}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=${RECOMPUTE_METHOD}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${RECOMPUTE_NUM_LAYERS}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=9216
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.balance_batch=True
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

# Critical: enable Megatron engine
EXTRA=(
    model_engine=megatron
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
