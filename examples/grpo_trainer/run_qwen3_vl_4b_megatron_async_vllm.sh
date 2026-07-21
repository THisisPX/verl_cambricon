#!/usr/bin/env bash
# ==============================================================================
# verl Fully Async GRPO | Qwen3-VL-4B | vLLM rollout | Megatron training
# 4× GPU 分卡异步: 训练 2 卡 (TP2) + 推理 2 卡 (TP2) — 并行运行
#
# vLLM variant of run_qwen3_vl_4b_megatron_async.sh — for comparing
# vLLM vs SGLang inference performance on VL workloads.
#
# 参数匹配 slime: scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh
# ==============================================================================

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
# B300: pip-installed Triton bundles a stale ptxas that doesn't know sm_103a.
# Point to the system CUDA 13.0 ptxas instead (supports sm_103).
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ==================== 路径配置 (请根据实际环境修改) ====================
MODEL_PATH="${MODEL_PATH:-/workspace/volume/distributed-training-softdata/models/Qwen3-VL-4B-Instruct}"
SAVE_DIR="${SAVE_DIR:-/workspace/volume/pengxiong/models/Qwen3-VL-4B_verl_geo3k_async_v2_vllm}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/workspace/volume/pengxiong/datasets/geo3k_imgurl}"
DATA_DIR="${DATA_DIR:-/workspace/volume/pengxiong/datasets/geo3k_imgurl-verl}"
# =====================================================================

# ---- user-adjustable ----
trainer_steps=${TRAINER_STEPS:-12}
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-2101}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-3072}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}

actor_lr=${ACTOR_LR:-1e-6}
weight_decay=${WEIGHT_DECAY:-0.1}
adam_beta1=${ADAM_BETA1:-0.9}
adam_beta2=${ADAM_BETA2:-0.98}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

# Megatron 训练并行 (2 卡: TP2 DP1)
train_tp=${TRAIN_TP:-2}
train_pp=${TRAIN_PP:-1}

# vLLM 推理并行 (2 卡: 1 engine × TP2)
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}       # 4B VL 模型不需要太多显存
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.8}

# 异步控制参数
staleness_threshold=${STALENESS_THRESHOLD:-0.5}
trigger_parameter_sync_step=${TRIGGER_PARAM_SYNC_STEP:-1}
require_batches=${REQUIRE_BATCHES:-1}
partial_rollout=${PARTIAL_ROLLOUT:-True}

# 日志 & 保存
test_freq=${TEST_FREQ:-9999}
save_freq=${SAVE_FREQ:-9999}

project_name=${PROJECT_NAME:-verl_async_geo3k_v2_vllm}
experiment_name=${EXPERIMENT_NAME:-qwen3_vl_4b_vllm_megatron_async_v2}
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

########################### 日志 ###########################

LOG_DIR="${LOG_DIR:-logs/${project_name}/${experiment_name}}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Logging to: ${LOG_FILE}"

########################### 环境变量 ###########################

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

# B300/Blackwell 兼容
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"

# WandB 离线
export WANDB_MODE=offline

########################### launch ###########################

echo "============================================================"
echo "verl Fully Async GRPO | Qwen3-VL-4B | vLLM + Megatron"
echo "GPU: 2 train (TP2) + 2 inference (TP2) = 4 total"
echo "Dataset: ${VERL_DATA_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Save: ${SAVE_DIR}"
echo "Rollout steps: ${total_rollout_steps} | n: ${n_resp_per_prompt}"
echo "Mini batch: ${ppo_mini_batch_size} | LR: ${actor_lr}"
echo "Log: ${LOG_FILE}"
echo "============================================================"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    data.train_files="${VERL_DATA_DIR}/train.parquet" \
    data.val_files="${VERL_DATA_DIR}/test.parquet" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.image_key=images \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff="${entropy_coeff}" \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ppo_max_token_len_per_gpu}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.lr_decay_steps="${total_rollout_steps}" \
    actor_rollout_ref.actor.optim.weight_decay="${weight_decay}" \
    actor_rollout_ref.actor.optim.betas="[${adam_beta1},${adam_beta2}]" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${train_tp}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${train_pp}" \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_mem_util}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.temperature="${rollout_temperature}" \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ppo_max_token_len_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024)) \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    custom_reward_function.path="${REPO_ROOT}/examples/grpo_trainer/geo3k_reward.py" \
    custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.total_training_steps="${trainer_steps}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq="${save_freq}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=2 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
