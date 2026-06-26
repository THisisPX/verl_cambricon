#!/usr/bin/env bash
# Performance Test (Fully Async): Qwen3-4B | GRPO | FSDP2 + SGLang | 8×A100 40GB
#
# SGLang variant — identical to run_qwen3_4b_fsdp2_perf_test_async.sh
# except rollout engine is SGLang instead of vLLM.
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_fsdp2_perf_test_async_sglang.sh

set -xeuo pipefail

# ==================== 路径 ====================
MODEL_PATH=${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k-verl.parquet}
TEST_FILE=${TEST_FILE:-/workspace/volume/pengxiong/datasets/aime-2024/aime-2024-verl.parquet}

NNODES=${NNODES:-1}
NGPUS=${NGPUS:-8}
N_GPUS_TRAIN=${N_GPUS_TRAIN:-4}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}

# ==================== 实验追踪 ====================
PROJECT_NAME=${PROJECT_NAME:-verl_async_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192_fsdp2_async_sglang}

# ==================== 日志 ====================
LOG_DIR=${LOG_DIR:-"logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

# ==================== 启动 ====================
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_trainer' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=0 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=9216 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    ++actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.98]" \
    critic.strategy=fsdp2 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.max_resp_len=8192 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAIN}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.n=16 \
    rollout.total_rollout_steps=256 \
    async_training.staleness_threshold=0 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=1 \
    async_training.partial_rollout=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    "$@" 2>&1 | tee "${LOG_FILE}"
