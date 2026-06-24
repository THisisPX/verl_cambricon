#!/usr/bin/env bash
# Performance Test (Megatron): Qwen3-4B | GRPO | Megatron + SGLang (colocate)
#
# SGLang variant — identical to run_qwen3_4b_megatron_perf_test.sh except rollout engine.
# Purpose: isolate performance difference between vLLM and SGLang in verl HybridEngine.
#
#   Verl vLLM:  Megatron TP4×DP2 hybrid engine + vLLM 4engines×TP2
#   Verl SGLang: Megatron TP4×DP2 hybrid engine + SGLang 4engines×TP2
#   Slime:       Megatron TP4×DP2 colocate        + SGLang 4engines×TP2
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_megatron_sglang_perf_test.sh
#
# Override env vars:
#   TRAIN_FILE, TEST_FILE, MODEL_PATH, NNODES, NGPUS_PER_NODE, etc.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONWARNINGS=ignore

# ======================== slime-matching defaults ========================
MODEL_PATH=${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k-verl.parquet}
TEST_FILE=${TEST_FILE:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024-verl.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- batch / rollout (matching slime) ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}        # slime: --rollout-batch-size 8
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}  # verl multiplies by rollout.n=16 internally
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

# --- algorithm (matching slime: no KL loss, no entropy bonus) ---
ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

# --- Megatron parallelism ---
ACTOR_TP=${ACTOR_TP:-4}
ACTOR_PP=${ACTOR_PP:-1}
OFFLOAD=${OFFLOAD:-True}

# --- rollout (SGLang) ---
# TP=2 for 4 inference engines (matches slime: 4 engines × TP2)
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
ROLLOUT_N=${ROLLOUT_N:-16}

# --- Megatron-specific: sequence parallel + full recompute (matching slime) ---
SEQUENCE_PARALLEL=True
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}

# --- experiment tracking ---
PROJECT_NAME=${PROJECT_NAME:-verl_perf_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192_megatron_sglang}
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
    data.dataloader_num_workers=0
    data.val_batch_size=8
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
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
    ++actor_rollout_ref.actor.megatron.sequence_parallel=${SEQUENCE_PARALLEL}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=${RECOMPUTE_GRANULARITY}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=${RECOMPUTE_METHOD}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${RECOMPUTE_NUM_LAYERS}
)

# SGLang rollout — key difference from vLLM variant
ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
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
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=False
    trainer.resume_mode=disable
)

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
