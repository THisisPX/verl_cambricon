#!/usr/bin/env bash
# Performance Test (Async): Qwen3-4B | GRPO | FSDP2 + vLLM Fully Async
#
# Uses verl's fully_async_policy for separated rollout/trainer with 4+4 GPU split.
# Designed to compare with slime framework's async mode (4 train + 4 rollout, overlapped).
# Reference: E:/learning/slime/reports/sync-vs-async-comparison.md
#
# Slime async: 4GPUs Megatron train + 4GPUs SGLang rollout, overlapped
# Verl async:  4GPUs FSDP2 train + 4GPUs vLLM rollout, overlapped (fully_async_policy)
#
# Architecture: hybrid_engine=False, separate resource pools for train/rollout
# Overlap: rollout N+1 runs in parallel with train N (pipeline parallelism)
#
# Usage:
#   bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test_async.sh
#
# Override env vars:
#   TRAIN_FILE, TEST_FILE, MODEL_PATH, NNODES, NGPUS_PER_NODE, etc.
#
# Note: vLLM must use V1 engine for async mode (VLLM_USE_V1=1)

set -xeuo pipefail

# ======================== slime-matching defaults ========================
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/path/to/dapo-math-17k/train.parquet}
TEST_FILE=${TEST_FILE:-/path/to/aime-2024/test.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- GPU split (4 train + 4 rollout, matching slime's resource allocation) ---
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
N_GPUS_TRAINING=$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))

# --- batch / rollout (matching slime) ---
# In async mode, train_batch_size=0 (streaming), gen_batch_size=1
# total_rollout_steps controls total samples generated
# Each step processes require_batches * ppo_mini_batch_size samples
TRAIN_BATCH_SIZE=0
GEN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}  # per GPU
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-16}     # slime: n_samples=16
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

# --- algorithm (GRPO) ---
ADV_ESTIMATOR=grpo
ACTOR_LR=1e-6
KL_COEF=0.001
KL_LOSS_COEF=0.001
ENTROPY_COEFF=0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28  # slime: eps_clip_high=0.28
USE_KL_IN_REWARD=False
USE_KL_LOSS=True

# --- rollout (vLLM async mode) ---
ROLLOUT_NAME=vllm
ROLLOUT_MODE=async  # must be async for fully_async_policy
GEN_TP=1            # vLLM TP=1 for 4 GPUs (4 engines) - different from slime's 2 engines×TP2
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
# vLLM V1 engine required for async, enables prefix caching
export VLLM_USE_V1=1
return_raw_chat="True"

# --- dynamic batch ---
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
infer_ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))

# --- FSDP2 config ---
# hybrid sharding with group size = 2
fsdp_size=2

# --- experiment tracking ---
PROJECT_NAME=${PROJECT_NAME:-verl_perf_test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_n16_resp8192_async}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
TEST_FREQ=${TEST_FREQ:-9999}

# --- fully_async_policy specific ---
# staleness_threshold=0: synchronous (no stale samples)
# staleness_threshold=0.5: allows up to 50% stale samples in training
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0}
TRIGGER_PARAM_SYNC_STEP=${TRIGGER_PARAM_SYNC_STEP:-8}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-4}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-False}

# Total rollout samples = train_batch_size * total_steps (in equivalent sync mode)
# For perf test, just produce enough for 15 steps
# total = ppo_mini_batch_size * require_batches * trigger_param_sync_step * total_steps / n_resp_per_prompt
TOTAL_ROLLOUT_STEPS=$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TRIGGER_PARAM_SYNC_STEP * TOTAL_TRAINING_STEPS / N_RESP_PER_PROMPT))

# ======================== launch ========================
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.return_raw_chat="${return_raw_chat}" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    +actor_rollout_ref.actor.fsdp_config.fsdp_size="${fsdp_size}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    algorithm.rollout_correction.bypass_mode=True \
    critic.strategy=fsdp2 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=False \
    trainer.save_freq="${TEST_FREQ}" \
    trainer.default_local_dir="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    trainer.total_epochs=1 \
    trainer.test_freq="${TEST_FREQ}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    rollout.test_freq="${TEST_FREQ}" \
    async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
    async_training.trigger_parameter_sync_step="${TRIGGER_PARAM_SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    "$@"
