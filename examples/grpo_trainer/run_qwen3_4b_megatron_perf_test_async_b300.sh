#!/usr/bin/env bash
# Performance Test (Fully Async): Qwen3-4B | GRPO | Megatron + vLLM | 4×B300
#
# GPU: 2 train (TP2, DP1) + 2 inference (1 engine × TP2)
# B300 VRAM: no memory constraints — best params matching slime.
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_megatron_perf_test_async_b300.sh

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
# B300: pip-installed Triton bundles a stale ptxas that doesn't know sm_103a.
# Point to the system CUDA 13.0 ptxas instead (supports sm_103).
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ==================== 路径 ====================
MODEL_PATH=${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k-verl.parquet}
TEST_FILE=${TEST_FILE:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024-verl.parquet}

# ==================== GPU (4 total: 2+2) ====================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
N_GPUS_TRAIN=${N_GPUS_TRAIN:-2}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-2}

# ==================== 数据 & Rollout（匹配 slime）====================
TRAIN_BATCH_SIZE=0
GEN_BATCH_SIZE=1
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

# 16 steps × 8 samples/step = 128 total
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-128}

# ==================== 算法（匹配 slime）====================
ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
USE_KL_IN_REWARD=False
USE_KL_LOSS=False

# ==================== 优化器 ====================
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

# ==================== Megatron 并行 ====================
# 2 GPU: TP2 (model split across 2 GPUs), DP1
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=${TRAIN_PP:-1}
OFFLOAD=False

# ==================== vLLM 推理 ====================
# 2 GPU: 1 engine × TP2
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# ==================== Batch（B300 无压力）====================
PPO_MINI_BATCH_SIZE=8
PPO_MICRO_BATCH_SIZE_PER_GPU=4

actor_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU}
infer_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU}

# ==================== 实验追踪 ====================
PROJECT_NAME=${PROJECT_NAME:-verl_async_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192_megatron_async_b300}
TEST_FREQ=${TEST_FREQ:-9999}
SAVE_FREQ=${SAVE_FREQ:--1}

# ==================== 异步配置 ====================
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
TRIGGER_PARAM_SYNC_STEP=${TRIGGER_PARAM_SYNC_STEP:-1}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}

# ==================== 日志 ====================
LOG_DIR=${LOG_DIR:-"logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

# ==================== 启动 ====================
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
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
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
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
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
