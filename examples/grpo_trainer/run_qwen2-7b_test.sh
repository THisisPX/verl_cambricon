#!/bin/bash

# IMPORTANT: See and modify  all [] placeholders in this file.
# Requires 8 GPUs with 80GB memory.

model_path="/workspace/dataset/favorite/soft-data-models/v1/Qwen2.5-7B-Instruct"
project_name="CodeRL+"
experiment_name="test"

train_file=/workspace/volume/pengxiong/codepluas-data/train_set_debug.parquet
val_files=/workspace/volume/pengxiong/codepluas-data/validation_set.parquet

LOG_DIR="/workspace/volume/pengxiong/verl/logs/$project_name"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/${experiment_name}.log"

custom_reward_path=/workspace/volume/pengxiong/verl/recipe/reward/reward_exec.py

export REWARD_LOGLEVEL=INFO
export REWARD_MAX_TESTS=-1



python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +data.rollout_integration.enable=True \
    +data.rollout_integration.ratio=0.4 \
    +data.rollout_integration.buffer_size=20000 \
    +data.rollout_integration.sampling_strategy=recent \
    +data.rollout_integration.prompt_template=default \
    data.train_files="$train_file" \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="$custom_reward_path" \
    custom_reward_function.name=compute_reward \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False\
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=1000 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=/workspace/volume/pengxiong/checkpoints/${project_name}/${experiment_name} \
    trainer.val_before_train=True \
    2>&1 | tee "$LOG_PATH"

echo "======================================================="
echo "Training completed!"
echo "Check results in log file: $LOG_PATH"


# set -x

# train_file=/workspace/CODERLPLUS/data/train_set.parquet
# val_files=/workspace/CODERLPLUS/data/validation_set.parquet
# custom_reward_path=/workspace/CODERLPLUS/recipe/reward/reward_function_coderlplus.py


# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$train_file \
#     data.val_files=$val_files \
#     data.train_batch_size=1024 \
#     data.max_prompt_length=512 \
#     data.max_response_length=1024 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/workspace/dataset/favorite/soft-data-models/v1/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     reward_model.enable=False \
#     reward_model.launch_reward_fn_async=False \
#     custom_reward_function.path="$custom_reward_path" \
#     custom_reward_function.name=compute_score \
#     trainer.default_local_dir=checkpoints/verl_grpo_codeplus/Qwen2.5-7B-Instruct-RL \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console"]' \
#     trainer.project_name='verl_grpo_example_gsm8k' \
#     trainer.experiment_name='qwen2_7b_function_rm' \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=15 $@
