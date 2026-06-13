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
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}        # slime: --rollout-batch-size 8
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8} # slime: --global-batch-size 32, but verl multiplies by rollout.n=16
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
# Must accommodate max_prompt(1024) + max_response(8192) = 9216 tokens per sequence
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

# --- algorithm (matching slime: no KL loss, no entropy bonus) ---
ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
# Dual-clip PPO (matching slime: --eps-clip 0.2 --eps-clip-high 0.28)
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

# --- Megatron parallelism (matching slime: TP=2) ---
# Qwen3-4B is small enough for TP=2, PP=1
# Total train GPUs = TP * PP * DP = 2 * 1 * 4 = 8 (hybrid engine)
ACTOR_TP=${ACTOR_TP:-2}
ACTOR_PP=${ACTOR_PP:-1}

# Megatron offloading: essential for HybridEngine — frees GPU memory for vLLM wake_up
# (same as slime's forced --offload in colocate mode)
OFFLOAD=${OFFLOAD:-True}

# --- rollout (vLLM) ---
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
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
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-8}   # slime: --num-rollout 8
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
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}
    actor_rollout_ref.actor.optim.betas=[${ADAM_BETA1},${ADAM_BETA2}]
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    # Dynamic batch: let Megatron handle micro-batch sizing
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    # No KL loss — matching slime (--kl-loss-coef 0.00, no ref model needed)
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    # Dual-clip PPO (matching slime: --eps-clip 0.2 --eps-clip-high 0.28)
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    # Megatron parallelism (TP=2 matches slime)
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    # Offloading: essential for HybridEngine to free GPU memory for vLLM wake_up
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
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
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=True
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

# ======================== log dir ========================
LOG_DIR=${LOG_DIR:-"logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

# ======================== launch ========================
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
