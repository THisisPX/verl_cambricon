#!/usr/bin/env bash
# ==============================================================================
# verl GRPO | Qwen3-VL-4B | SGLang rollout | Megatron training | 4× GPU
#
# Parameter-matched with slime: scripts/run-qwen3-VL-4B-geo3k-4gpu-v3.sh
#
# GPU 分配: verl hybrid engine — 4 卡共享训练和推理 (3D-HybridEngine resharding)
#   slime 分卡模式: 训练 2 卡 (TP2) + 推理 2 卡 (TP2)
#   verl 等价配置: Megatron TP=2, PP=1 + SGLang TP=2, 4 GPU hybrid engine
#
# 数据集: chenhegu/geo3k_imgurl (几何推理, LaTeX 答案)
# Reward: geo3k (mathruler LaTeX 公式比较, 等价 slime 的 --rm-type math)
# ==============================================================================

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ==================== 路径配置 (请根据实际环境修改) ====================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
SAVE_DIR="${SAVE_DIR:-/workspace/volume/pengxiong/models/Qwen3-VL-4B_verl_geo3k_slime_match}"
DATA_DIR="${DATA_DIR:-$HOME/data}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/workspace/volume/pengxiong/datasets}"
# =====================================================================

# ---- user-adjustable ----
# 匹配 slime 的超参
total_training_steps=${TOTAL_TRAINING_STEPS:-500}
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-3072}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}

actor_lr=${ACTOR_LR:-1e-6}
weight_decay=${WEIGHT_DECAY:-0.1}
adam_beta1=${ADAM_BETA1:-0.9}
adam_beta2=${ADAM_BETA2:-0.98}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio=${CLIP_RATIO:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.7}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.8}

save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-20}

project_name=${PROJECT_NAME:-verl_grpo_geo3k}
experiment_name=${EXPERIMENT_NAME:-qwen3_vl_4b_sglang_megatron_slime_match}
# ---- end user-adjustable ----

########################### 数据预处理 ###########################

# slime 数据集 (chenhegu/geo3k_imgurl) 需要转换为 verl 格式
VERL_DATA_DIR="${DATA_DIR}/geo3k_imgurl"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

if [ ! -f "${VERL_DATA_DIR}/train.parquet" ]; then
    echo "Preprocessing chenhegu/geo3k_imgurl to verl format..."
    # 如果 raw data 目录已有 slime 下载的数据，直接使用
    RAW_PATH="${RAW_DATA_DIR}/chenhegu/geo3k_imgurl"
    if [ -d "${RAW_PATH}" ]; then
        python3 "${REPO_ROOT}/examples/data_preprocess/geo3k_imgurl.py" \
            --local_dataset_path "${RAW_PATH}" \
            --local_save_dir "${VERL_DATA_DIR}"
    else
        python3 "${REPO_ROOT}/examples/data_preprocess/geo3k_imgurl.py" \
            --local_save_dir "${VERL_DATA_DIR}"
    fi
    echo "Preprocessing done: ${VERL_DATA_DIR}"
fi

########################### 参数组装 ###########################

# 日志和保存
NOW=$(date +%Y%m%d_%H%M%S)
export WANDB_MODE=offline  # 仅使用 TensorBoard, 匹配 slime

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
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    actor_rollout_ref.actor.optim.betas="[${adam_beta1}, ${adam_beta2}]"
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.clip_ratio=${clip_ratio}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.shuffle=True
    # Megatron 训练并行配置 (匹配 slime: TP2 PP1 SP)
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    # Recompute (匹配 slime: full, uniform, 1 layer)
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    # Attention 配置 (匹配 slime: FA2, 无 dropout, FP32 allreduce/softmax)
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
    actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
    actor_rollout_ref.actor.megatron.override_transformer_config.accumulate_allreduce_grads_in_fp32=True
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    # VLM: 冻结 vision encoder
    actor_rollout_ref.actor.freeze_vision_tower=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # SGLang 额外参数 (匹配 slime)
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=sdpa
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
    # 验证参数 (匹配 slime: eval n=1, top_p=1)
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.top_p=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
)

# 不需要 ref 模型 (KL coef=0), 不配置 REF 参数

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

########################### 环境变量 (B300/Blackwell 兼容) ###########################

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

# Triton PTX 兼容 sm_103a
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"
# SGLang 多模态注意力后端
export SGLANG_MM_ATTENTION_BACKEND="${SGLANG_MM_ATTENTION_BACKEND:-sdpa}"

########################### launch ###########################

echo "============================================"
echo "verl GRPO | Qwen3-VL-4B | SGLang + Megatron"
echo "Dataset: ${VERL_DATA_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Save: ${SAVE_DIR}"
echo "Steps: ${total_training_steps} | Batch: ${train_batch_size} | n: ${n_resp_per_prompt}"
echo "LR: ${actor_lr} | TP: ${actor_tp} | PP: ${actor_pp}"
echo "============================================"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
