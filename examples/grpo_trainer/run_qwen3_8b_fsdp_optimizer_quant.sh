#!/usr/bin/env bash
# GRPO | Qwen3-8B | FSDP training | Optimizer State Quantization experiment
#
# This is the minimum viable experiment for the "RL-aware low-bit learner"
# project. It runs GRPO on GSM8K + MATH with configurable optimizer state
# quantization, supporting BF16 baseline / 8-bit / 4-bit comparison.
#
# Usage:
#   # BF16 baseline
#   QUANT_DTYPE=none bash run_qwen3_8b_fsdp_optimizer_quant.sh
#
#   # 8-bit optimizer states (int8)
#   QUANT_DTYPE=int8 bash run_qwen3_8b_fsdp_optimizer_quant.sh
#
#   # 4-bit optimizer states (int4 via 4-bit grad quant)
#   QUANT_DTYPE=int8 GRAD_QUANT_BITS=4 bash run_qwen3_8b_fsdp_optimizer_quant.sh
#
#   # 8-bit optimizer + scale staleness (recalibrate every 10 steps)
#   QUANT_DTYPE=int8 RECALIBRATE_FREQ=10 bash run_qwen3_8b_fsdp_optimizer_quant.sh
#
# Knobs:
#   QUANT_DTYPE      quantization dtype: none | int8 | fp8_e4m3  (default: none)
#   GRAD_QUANT_BITS  gradient quantization bits: '' | 4 | 8      (default: '')
#   RECALIBRATE_FREQ scale recalibration frequency: '' | N      (default: '')
#   STOCHASTIC_ROUND stochastic rounding: true | false           (default: true)
#   INFER_BACKEND    rollout backend: vllm | sglang              (default: vllm)

set -xeuo pipefail

########################### user-adjustable ###########################
QUANT_DTYPE=${QUANT_DTYPE:-none}
GRAD_QUANT_BITS=${GRAD_QUANT_BITS:-}
RECALIBRATE_FREQ=${RECALIBRATE_FREQ:-}
STOCHASTIC_ROUND=${STOCHASTIC_ROUND:-true}
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-5}
sp_size=${SP_SIZE:-1}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_optimizer_quant}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_quant${QUANT_DTYPE}_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
case "${INFER_BACKEND}" in
    vllm | sglang) ;;
    *) echo "INFER_BACKEND must be vllm or sglang" >&2; exit 1 ;;
esac

actor_param_offload=False
actor_optimizer_offload=False

########################### quantization config ###########################
QUANT_EXTRA=()
if [ "${QUANT_DTYPE}" != "none" ]; then
    QUANT_EXTRA+=(
        actor_rollout_ref.actor.optim.optimizer_quant.enable=true
        actor_rollout_ref.actor.optim.optimizer_quant.quant_dtype="${QUANT_DTYPE}"
        actor_rollout_ref.actor.optim.optimizer_quant.stochastic_round="${STOCHASTIC_ROUND}"
        actor_rollout_ref.actor.optim.optimizer_quant.log_diagnostics=true
    )
    if [ -n "${GRAD_QUANT_BITS}" ]; then
        QUANT_EXTRA+=(
            actor_rollout_ref.actor.optim.optimizer_quant.grad_quant_bits="${GRAD_QUANT_BITS}"
        )
    fi
    if [ -n "${RECALIBRATE_FREQ}" ]; then
        QUANT_EXTRA+=(
            actor_rollout_ref.actor.optim.optimizer_quant.recalibrate_freq="${RECALIBRATE_FREQ}"
        )
    fi
    # Add quantization info to experiment name
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_gs8k_math"
fi

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$HOME/data/gsm8k/train.parquet', '$HOME/data/math/train.parquet']"
    data.val_files="['$HOME/data/gsm8k/test.parquet', '$HOME/data/math/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${QUANT_EXTRA[@]}" \
    "$@"
