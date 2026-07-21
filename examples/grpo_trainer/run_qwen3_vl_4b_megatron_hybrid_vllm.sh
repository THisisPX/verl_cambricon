#!/usr/bin/env bash
# ==============================================================================
# verl Hybrid Engine GRPO | Qwen3-VL-4B | vLLM rollout | Megatron | 4× GPU
#
# 参数 1:1 对标 slime: scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh
#
# vLLM variant of run_qwen3_vl_4b_megatron_hybrid_sglang.sh
# 与 SGLang 版本唯一的推理引擎差异:
#   - rollout.name=vllm (vs sglang)
#   - VLLM_USE_V1=1
#   - mm_processor_cache_gb=0 (VL compatibility)
#   - CUDA graph 可用 (enforce_eager=False, B300 + system ptxas)
# ==============================================================================

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
# B300/Blackwell: system ptxas instead of stale Triton bundle
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

# ==================== 路径配置 ====================
MODEL_PATH="${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct}"
SAVE_DIR="${SAVE_DIR:-/workspace/volume/pengxiong/models/Qwen3-VL-4B_verl_hybrid_vllm_v1}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/workspace/volume/pengxiong/datasets/geo3k_imgurl}"
DATA_DIR="${DATA_DIR:-/workspace/volume/pengxiong/datasets/geo3k_imgurl-verl}"
# =====================================================================

# ---- user-adjustable: 1:1 对标 slime v3 ----
total_training_steps=${TOTAL_TRAINING_STEPS:-12}       # 性能测试用
train_batch_size=${TRAIN_BATCH_SIZE:-64}                # slime --global-batch-size 64
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}              # slime --n-samples-per-prompt 8
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-3072}        # slime --rollout-max-response-len 3072
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}

actor_lr=${ACTOR_LR:-1e-6}                             # slime --lr 1e-6
weight_decay=${WEIGHT_DECAY:-0.1}
adam_beta1=${ADAM_BETA1:-0.9}
adam_beta2=${ADAM_BETA2:-0.98}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}                  # slime --eps-clip 0.2
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}               # slime --eps-clip-high 0.28

actor_tp=${ACTOR_TP:-2}                                # slime --tensor-model-parallel-size 2
actor_pp=${ACTOR_PP:-1}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}      # vLLM on shared GPU: conservative
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.8}

save_freq=${SAVE_FREQ:-9999}
test_freq=${TEST_FREQ:-9999}

project_name=${PROJECT_NAME:-verl_hybrid_geo3k_v1}
experiment_name=${EXPERIMENT_NAME:-qwen3_vl_4b_vllm_megatron_hybrid_v1}
# ---- end user-adjustable ----

########################### 数据预处理 ###########################
VERL_DATA_DIR="${DATA_DIR}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

if [ ! -f "${VERL_DATA_DIR}/train.parquet" ]; then
    echo "Preprocessing ${RAW_DATA_DIR} to verl format..."
    python3 "${REPO_ROOT}/examples/data_preprocess/geo3k_imgurl.py" \
        --local_dataset_path "${RAW_DATA_DIR}" \
        --local_save_dir "${VERL_DATA_DIR}"
    echo "Preprocessing done: ${VERL_DATA_DIR}"
fi

########################### 参数组装 ###########################
export WANDB_MODE=offline

# NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${VERL_DATA_DIR}/train.parquet"
    data.val_files="${VERL_DATA_DIR}/test.parquet"
    data.image_key=images
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    custom_reward_function.path="${REPO_ROOT}/examples/grpo_trainer/geo3k_reward.py"
    custom_reward_function.name=compute_score
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    actor_rollout_ref.actor.optim.betas="[${adam_beta1},${adam_beta2}]"
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.shuffle=True
    actor_rollout_ref.actor.freeze_vision_tower=True
    # Megatron (匹配 slime: TP2 PP1 SP, full recompute)
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
)

ROLLOUT_VLLM=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.max_num_seqs=64
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.top_p=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_training_steps=${total_training_steps}
    trainer.default_local_dir="${SAVE_DIR}"
)

EXTRA=(
    model_engine=megatron
)

LOG_DIR="${LOG_DIR:-logs/${project_name}/${experiment_name}}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "============================================================"
echo "verl Hybrid Engine GRPO | Qwen3-VL-4B | vLLM + Megatron"
echo "GPU: 4× shared (TP2 train + TP2 inference, hybrid engine)"
echo "Dataset: ${VERL_DATA_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Save: ${SAVE_DIR}"
echo "Steps: ${total_training_steps} | Batch: ${train_batch_size} | n: ${n_resp_per_prompt}"
echo "LR: ${actor_lr} | TP: ${actor_tp} | PP: ${actor_pp}"
echo "Log: ${LOG_FILE}"
echo "============================================================"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT_VLLM[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
