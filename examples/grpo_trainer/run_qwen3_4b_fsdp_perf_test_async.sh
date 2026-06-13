#!/usr/bin/env bash
# Performance Test (Fully Async): Qwen3-4B | GRPO | Megatron + vLLM
#
# Fully async deployment with SEPARATE GPU groups:
#   - 4 GPUs for Megatron training (TP=2, DP=2)
#   - 4 GPUs for vLLM rollout (TP=2, DP=2)
#
# This matches slime's GPU layout (4 train + 4 inference) and uses
# the same training backend (Megatron), making it the fairest comparison.
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test_async.sh
#
# Override env vars:
#   TRAIN_FILE, TEST_FILE, MODEL_PATH, NNODES, NGPUS_PER_NODE, etc.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1

# ======================== slime-matching defaults ========================
MODEL_PATH=${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k-verl.parquet}
TEST_FILE=${TEST_FILE:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024-verl.parquet}

# GPU allocation: 4 train + 4 inference on 1 node
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
N_GPUS_TRAIN=${N_GPUS_TRAIN:-4}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}

# --- batch / rollout (matching slime) ---
TRAIN_BATCH_SIZE=0           # 0 = async mode (streaming)
GEN_BATCH_SIZE=1             # only 1 supported for async
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
# Must accommodate max_prompt(1024) + max_response(8192) = 9216 tokens per sequence
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

# --- algorithm (matching slime: no KL loss, no KL penalty) ---
ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
USE_KL_IN_REWARD=False
USE_KL_LOSS=False
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

# --- Megatron parallelism (training: 4 GPUs, TP=2, PP=1, DP=2) ---
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=${TRAIN_PP:-1}

# --- rollout parallelism (inference: 4 GPUs, TP=2, DP=2) ---
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

# --- Megatron offloading (essential on 40GB GPUs) ---
OFFLOAD=True  # param/grad/optimizer offloading

# --- dynamic batch ---
use_dynamic_bsz=True
actor_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU}
infer_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU}

# --- experiment tracking ---
PROJECT_NAME=${PROJECT_NAME:-verl_perf_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192_megatron_async}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-8}   # match slime: --num-rollout 8
TEST_FREQ=${TEST_FREQ:-9999}
SAVE_FREQ=${SAVE_FREQ:--1}

# --- async training ---
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
TRIGGER_PARAM_SYNC_STEP=${TRIGGER_PARAM_SYNC_STEP:-4}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}

# ======================== log dir ========================
LOG_DIR=${LOG_DIR:-"logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

# ======================== launch ========================
# Note: Hydra @hydra.main decorator in fully_async_main.py resolves
# config_path relative to the file, so --config-path is not needed
# when running from any CWD.

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.truncation='error' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.rollout_correction.bypass_mode=True \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
    actor_rollout_ref.actor.optim.betas="[${ADAM_BETA1},${ADAM_BETA2}]" \
    actor_rollout_ref.actor.optim.lr_decay_steps="${TOTAL_ROLLOUT_STEPS}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.megatron.grad_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${OFFLOAD}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len=9216 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAIN}" \
    trainer.val_before_train=False \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.n="${N_RESP_PER_PROMPT}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
    async_training.trigger_parameter_sync_step="${TRIGGER_PARAM_SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
