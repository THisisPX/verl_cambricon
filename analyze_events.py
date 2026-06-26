"""Analyze tensorboard event files for verl vs slime comparison."""
import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==== Verl sync ====
verl_base = 'tensorboard_log/verl_perf_test'
verl_runs = {
    'verl_Megatron_vLLM': glob.glob(f'{verl_base}/qwen3_4b_grpo_n16_resp8192_megatron_tp4/events.out.*'),
    'verl_Megatron_SGLang': glob.glob(f'{verl_base}/qwen3_4b_grpo_n16_resp8192_megatron_sglang/events.out.*'),
}

# ==== Verl async ====
verl_a_base = 'tensorboard_log/verl_async_test'
verl_async_runs = {
    'verl_async_FSDP2_SGLang': glob.glob(f'{verl_a_base}/qwen3_4b_grpo_n16_resp8192_fsdp2_async_sglang/events.out.*'),
}

key_metrics = [
    'timing_s/gen', 'timing_s/update_actor', 'timing_s/old_log_prob',
    'timing_s/step', 'timing_s/update_weights', 'timing_s/timing_s/param_sync',
    'timing_s/agent_loop/slowest/generate_sequences',
    'perf/throughput', 'perf/total_num_tokens', 'perf/time_per_step',
    'actor/pg_loss', 'actor/grad_norm', 'actor/entropy',
    'response_length/mean', 'response_length/clip_ratio', 'critic/score/mean',
    'fully_async/trainer/idle_ratio',
]

def analyze(name, path, metrics):
    kb = os.path.getsize(path) / 1024
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()['scalars']
    print(f'\n--- {name} ({os.path.basename(path)[:55]}..., {kb:.0f}KB) ---')

    for tk in metrics:
        if tk in tags:
            vals = [s.value for s in ea.Scalars(tk)]
            if vals:
                avg = sum(vals) / len(vals)
                mn = min(vals)
                mx = max(vals)
                print(f'  {tk:50s}  avg={avg:10.2f}  min={mn:10.2f}  max={mx:10.2f}  n={len(vals)}')

    # print ALL available tags for async runs (to discover metric names)
    for t in sorted(tags):
        if t not in metrics and any(kw in t.lower() for kw in ['time', 'step', 'rollout', 'gen', 'train', 'throughput', 'token', 'loss', 'idle', 'queue', 'count']):
            vals = [s.value for s in ea.Scalars(t)]
            if vals:
                avg = sum(vals) / len(vals)
                print(f'  {t:50s}  avg={avg:10.2f}  n={len(vals)}')

print("=" * 90)
print("  VERL SYNC (Megatron)")
print("=" * 90)
for name, files in verl_runs.items():
    if files:
        files.sort(key=lambda f: os.path.getsize(f))
        analyze(name, files[-1], key_metrics)
    else:
        print(f'\n{name}: NO FILES')

print("\n" + "=" * 90)
print("  VERL ASYNC (FSDP2)")
print("=" * 90)
for name, files in verl_async_runs.items():
    if files:
        files.sort(key=lambda f: os.path.getsize(f))
        analyze(name, files[-1], key_metrics)
    else:
        print(f'\n{name}: NO FILES')

# ==== Slime ====
slime_base = r'D:\learning\slime\tensorboard_log\slime-vs-verl-colocate-8gpu'
slime_runs = {
    'slime_rollout': glob.glob(f'{slime_base}/20260614_114622/events.out.*'),
    'slime_train':   glob.glob(f'{slime_base}/20260614_115229/events.out.*'),
}
slime_metrics = [
    'perf/rollout_time', 'perf/step_time', 'perf/actor_train_time',
    'perf/log_probs_time', 'perf/sleep_time', 'perf/wake_up_time',
    'perf/train_time', 'perf/train_wait_time', 'perf/update_weights_time',
    'perf/tokens_per_gpu_per_sec', 'perf/wait_time_ratio',
    'rollout/truncated_ratio', 'rollout/response_len/mean',
    'train/pg_loss', 'train/entropy_loss', 'train/grad_norm',
]
print("\n" + "=" * 90)
print("  SLIME")
print("=" * 90)
for name, files in slime_runs.items():
    if files:
        analyze(name, files[-1], slime_metrics)
    else:
        print(f'\n{name}: NO FILES')
