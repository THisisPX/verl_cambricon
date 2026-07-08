# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready RL training library for large language models. It's developed by ByteDance Seed team and published as HybridFlow (EuroSys 2025). Version: 0.9.0.dev.

This is a fork with extended support for Cambricon MLU hardware, B300 testing, and async training benchmarks.

## Build & Install

```bash
# Editable install (dev) — prefer uv for env management
uv pip install -e .[test,vllm]

# Or with SGLang backend
uv pip install -e .[test,sglang]

# Install with pip as fallback
pip install -e .[test,vllm]

# Optional extras: test, prime, geo, gpu, math, vllm, sglang, trl, mcore, trtllm
```

## Lint & Format

```bash
# Install pre-commit hooks
pip install pre-commit hydra-core
pre-commit install

# Run on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

Ruff config: line-length=120, lint rules include E/F/UP/B/I/G with several ignores.

## Testing

```bash
# CPU tests (scripts ending in _on_cpu.py)
pytest tests/test_protocol_on_cpu.py
pytest tests/test_base_config_on_cpu.py

# Run all CPU unit tests
pytest tests/ -k "on_cpu"

# GPU tests (require GPUs)
pytest tests/ -k "not on_cpu"

# Specific test categories (in tests/):
# - tests/trainer/       - trainer unit tests
# - tests/models/        - model tests
# - tests/workers/       - worker tests
# - tests/utils/         - utility tests
# - tests/special_distributed/  - multi-GPU unit tests
# - tests/special_e2e/   - end-to-end training/generation tests
# - tests/special_npu/   - NPU-specific tests
# - tests/special_sanity/ - quick sanity checks
# - tests/special_standalone/ - dedicated environment tests

# CI workflows are in .github/workflows/:
# cpu_unit_tests.yml, gpu_unit_tests.yml, vllm.yml, sgl.yml, etc.
```

## Entry Points

There are two fundamentally different training modes with different entry points:

**Synchronous (hybrid engine) — `verl.trainer.main_ppo`:**
```bash
# Shared GPU pool: training and inference share the same GPUs
python3 -m verl.trainer.main_ppo --config-name=ppo_trainer.yaml
python3 -m verl.trainer.main_ppo --config-name=ppo_megatron_trainer.yaml
```
Training and rollout alternate on the same GPUs via 3D-HybridEngine resharding. Configurable through `verl/trainer/config/ppo_trainer.yaml` and `ppo_megatron_trainer.yaml`.

**Asynchronous (disaggregated) — `verl.experimental.fully_async_policy.fully_async_main`:**
```bash
# Separate GPU pools: trainer and rollouter run independently on different GPUs
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=fully_async_ppo_megatron_trainer.yaml
```
Uses separate Ray resource pools. Trainer and rollouter communicate via a MessageQueue (sample streaming) and NCCL CheckpointEngine (weight sync). Config at `verl/experimental/fully_async_policy/config/`.

Key constraint for async: `actor_rollout_ref.hybrid_engine=False` (must be off), `data.train_batch_size=0` (streaming), `data.gen_batch_size=1` (sample-by-sample).

## GPU Allocation (Async/Disaggregated Mode)

Training and inference GPUs are configured **independently** via two sets of parameters:

```bash
# Training GPUs (Megatron TP/PP/DP)
trainer.nnodes="${NNODES:-1}"
trainer.n_gpus_per_node="${N_GPUS_TRAIN:-2}"
actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP:-2}"

# Rollout/Inference GPUs (vLLM or SGLang TP)
rollout.nnodes="${NNODES:-1}"
rollout.n_gpus_per_node="${N_GPUS_ROLLOUT:-2}"
actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}"
```

**Uneven splits are fully supported.** Examples from existing scripts:
- 2 train + 2 inference (4 total) — B300 perf tests
- 4 train + 4 inference (8 total) — A100 40GB async tests
- 4 train + 12 inference (16 total) — DAPO 7B

The only constraint is total GPUs must be available in the Ray cluster. Training uses `trainer_pool`, rollout uses a separate pool managed by `LLMServerManager`.

**Practical constraints:**
- **Megatron training**: TP size must divide the number of attention heads. Qwen3-4B has 16 heads → TP=1,2,4,8 viable. 3 GPUs for Megatron training requires TP=1 (no tensor parallelism).
- **FSDP2 training**: Any number of GPUs works (no TP constraint).
- **vLLM/SGLang inference**: TP should divide the model's attention heads. 5 GPUs → TP=1 or TP=5.

## Training Modes

### Synchronous (Hybrid Engine)
- Entry: `verl.trainer.main_ppo`
- Single GPU pool, trainer and rollout share GPUs
- `actor_rollout_ref.hybrid_engine=True`
- 3D-HybridEngine resharding between train and generation phases
- Simpler setup, lower GPU count

### Fully Async (Disaggregated)
- Entry: `verl.experimental.fully_async_policy.fully_async_main`
- Separate GPU pools, trainer and rollouter run concurrently
- `actor_rollout_ref.hybrid_engine=False`
- Three Ray actors: `FullyAsyncRollouter` → `MessageQueue` → `FullyAsyncTrainer`
- Weight sync via NCCL CheckpointEngine (from MoonshotAI/checkpoint-engine)
- 1.72x-2.67x speedup documented for 7B models
- Parameters: `staleness_threshold` (max stale samples before pause), `trigger_parameter_sync_step` (local steps before sync), `partial_rollout` (resume interrupted rollouts)

### One-Step Off-Policy
- Entry: `verl.experimental.one_step_off_policy`
- Intermediate approach: generation runs asynchronously during training of previous batch
- Predecessor to fully async

## Platform & Hardware Abstraction

The codebase supports multiple hardware platforms through a plugin architecture (`verl/plugin/platform/`). All device-specific logic goes through `PlatformBase` obtained via `get_platform()` — the codebase never calls `torch.cuda.*` (or `torch.npu.*`) directly.

- **NVIDIA CUDA** (`platform_cuda.py`) — built-in, default
- **Huawei Ascend NPU** (`platform_npu.py`) — built-in, auto-detected via `torch.npu`
- **Cambricon MLU** — external plugin via `verl-hardware-plugin` package (loaded through `VERL_USE_EXTERNAL_MODULES`)
- **AMD ROCm**, **Intel XPU**, **MetaX** — addable via external plugins

Platform is selected by `VERL_PLATFORM` env var or auto-detected. Detection order: `VERL_PLATFORM` env var → probe all registered platforms via `is_available()` → fall back to `cuda`.

### Adding External Hardware Plugins

External plugins are loaded via two mechanisms:

1. **`VERL_USE_EXTERNAL_MODULES`** — comma-separated Python module paths to import:
   ```bash
   export VERL_USE_EXTERNAL_MODULES=my_plugin.platform_xpu
   ```

2. **setuptools entry_points** — auto-discovered under the `verl.plugins` group:
   ```toml
   [project.entry-points."verl.plugins"]
   my_hardware = "my_plugin.platform_xpu"
   ```
   Control with `VERL_USE_EXTERNAL_PLUGINS`:
   - `auto` — load all entry_points (default)
   - `none` — disable discovery
   - `pkg1,pkg2` — only load named entry_points

A plugin registers a platform by decorating a `PlatformBase` subclass with `@PlatformRegistry.register(platform="name")`. See `verl/plugin/platform/README.md` for the full interface.

### EngineRegistry

Engine selection uses two-dimensional lookup `(device, vendor)` in `EngineRegistry` (`verl/workers/engine/base.py`). When adding a new training/inference engine, register it for the appropriate `(device, vendor)` pair.

Engine backends by platform:
- **NVIDIA**: FSDP, FSDP2, Megatron-LM (via mbridge), VeOmni, TorchTitan, AutoModel
- **Ascend NPU**: MindSpeed (extends Megatron with NPU ops), MindSpeed-Megatron, FSDP2
- **Cambricon MLU**: FSDP MLU (external plugin)

Override engine device/vendor selection with `VERL_ENGINE_DEVICE` and `VERL_ENGINE_VENDOR` env vars.

Key env vars for NPU:
- `ASCEND_RT_VISIBLE_DEVICES` — NPU device selection
- `VERL_PLATFORM=huawei` — force Huawei platform
- `HCCL_*` variables — Huawei collective communication config

## Code Architecture

### High-Level Design

verl uses a **hybrid-controller programming model** that decouples computation and data dependencies. It's built on **Ray** for distributed orchestration. The core data transfer protocol is `DataProto` (`verl/protocol.py`), which wraps `TensorDict` for zero-copy structured data passing between components.

### Package Structure

```
verl/
  protocol.py              - DataProto: core data transfer abstraction (TensorDict-based)
  single_controller/        - HybridFlow controller programming model (Worker, WorkerGroup, ResourcePool)
  trainer/                  - Training loop implementations
    main_ppo.py             - Sync PPO entry point
    main_ppo_sync.py        - Synchronous PPO trainer (main loop)
    config/                 - Hydra/OmegaConf YAML configs (actor/, rollout/, algorithm/, etc.)
    ppo/                    - PPO algorithm losses and advantage estimators (core_algos.py)
  workers/
    engine/                 - Training engines: fsdp/, megatron/, mindspeed/, veomni/, torchtitan/
    rollout/                - Inference backends: vllm_rollout/, sglang_rollout/, hf_rollout.py
    reward_manager/         - Reward computation
    config/                 - Worker-level dataclass configs
  models/                   - Model wrappers (transformers/, mcore/) and registry
  utils/                    - Shared utilities (dataset, reward_score, checkpoint, device, etc.)
  experimental/
    fully_async_policy/     - Fully async training (Meituan contribution)
    separation/             - Base class for disaggregated trainer/rollouter (SeparateRayPPOTrainer)
    one_step_off_policy/    - One-step async training
    agent_loop/             - Multi-turn agent loop for async rollout
    reward_loop/            - Streaming reward computation
  plugin/                   - Plugin system via entry_points
    platform/               - Multi-chip platform abstraction (CUDA, NPU, MLU)
  checkpoint_engine/        - NCCL-based weight sync for async training
```

### Configuration System

Uses **Hydra/OmegaConf** with YAML configs. Override any parameter via command-line:
```
key=value                    # Simple override
+nested.key=value            # Add new key not in config
algorithm.adv_estimator=grpo # Nested override
```

Config search path priority: `verl/trainer/config/` → command-line overrides. For async training, configs extend the base Megatron trainer config (see `fully_async_ppo_megatron_trainer.yaml` defaults).

### Key Environment Variables

- `VERL_USE_EXTERNAL_MODULES` — comma-separated external modules to import
- `VERL_USE_EXTERNAL_PLUGINS` — plugin discovery policy (`auto`/`none`/comma-separated)
- `VERL_USE_MODELSCOPE` — use ModelScope hub instead of HuggingFace
- `VERL_AUTO_PADDING` — enable auto-padding in DataProto
- `VERL_PLATFORM` — force hardware platform (`nvidia`, `huawei`)
- `VERL_ENGINE_DEVICE` / `VERL_ENGINE_VENDOR` — override engine registry device/vendor lookup
- `VLLM_USE_V1=1` — enable vLLM V1 engine (recommended for async)
- `CUDA_DEVICE_MAX_CONNECTIONS=1` — required for Megatron

### RL Algorithms Supported

PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, VAPO, PF-PPO, SPPO, and more. Algorithm losses and advantage estimators are in `verl/trainer/ppo/core_algos.py`.

### Examples vs Recipe

**`examples/`** — minimal-dependency scripts driving `verl.trainer.main_ppo` with the current Hydra API. No custom entry points or reward code.

**`recipe/`** — git submodule (verl-recipe). Algorithm-specific extensions requiring custom trainer entry points, loss functions, or reward code (DAPO, PRIME, ReTool, R1, SPIN, FlowRL, etc.).

```
git submodule update --init --recursive  # get recipe code
```

Add new algorithms to `recipe/` if they need a custom entry point or reward code. Add to `examples/` if they only use existing `verl.trainer.main_ppo` features.

### Example Script Conventions

Every example script follows strict conventions (enforced by the `check-example-naming` pre-commit hook):

1. **Canonical filename**: `run_<model>_<train-backend>.sh`
   - `<train-backend>` must be the last underscore-separated token before `.sh`: `fsdp`, `fsdp2`, `megatron`, `mindspeed`, `automodel`, `veomni`
   - Feature toggles (inference backend, platform, LoRA, FP8) go inside the script as env vars, NOT in the filename
   - Do NOT create `_npu`, `_amd`, `_vllm`, `_sglang`, `_trtllm`, or `_fp8` script variants

2. **User-adjustable region** at the top with UPPERCASE env vars:
   ```bash
   DEVICE=${DEVICE:-gpu}
   INFER_BACKEND=${INFER_BACKEND:-vllm}
   MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
   NNODES=${NNODES:-1}
   # ---- end user-adjustable ----
   ```

3. **Defaults**: GSM8K + MATH datasets, `use_dynamic_bsz=True`, `balance_batch=True`, wandb logging.

4. GPU and NPU paths share the same `PROJECT_NAME` / `EXPERIMENT_NAME` (don't append `_npu`).

### Deprecated Config Knobs

These patterns are removed/deprecated — do NOT use in new scripts:
- `ppo_megatron_trainer.yaml` → use `actor_rollout_ref.actor.model_engine=megatron`
- `actor_rollout_ref.rollout.mode=async` — removed
- `actor_rollout_ref.hybrid_engine=True` — removed; the trainer enforces it internally
- `ppo_micro_batch_size` / `log_prob_micro_batch_size` → use `_per_gpu` suffix
- `data.val_batch_size` — removed
- Top-level `reward_model.*` → use `reward_model.reward_model.*` / `reward.reward_model.*`
- `actor.ulysses_sequence_parallel_size` → use `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size`

### Pre-commit Hooks

Key hooks beyond ruff:
- `autogen-trainer-cfg` — auto-generates trainer config files
- `check-example-naming` — enforces `run_<model>_<train-backend>.sh` naming convention
- Standard: trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, detect-private-key

### Testing Conventions

- Tests mirror the package structure: `tests/trainer/` ↔ `verl/trainer/`, `tests/models/` ↔ `verl/models/`
- `tests/special_distributed/` — requires multiple GPUs
- `tests/special_e2e/` — end-to-end training/generation runs
- `tests/special_npu/` — NPU-specific tests
- `tests/special_sanity/` — quick checks (includes `check_example_naming.py`)
- `tests/special_standalone/` — dedicated environment tests
- Files named `*_on_cpu.py` run on CPU in CI; all others require GPU
- CI: `cpu_unit_tests.yml` runs `test_*_on_cpu.py`; `gpu_unit_tests.yml` runs everything else

### Optional Dependency Extras

| Extra | Contents |
|-------|----------|
| `test` | pytest, pre-commit, py-spy, pytest-asyncio, pytest-rerunfailures |
| `vllm` | vllm>=0.8.5,<=0.12.0, tensordict |
| `sglang` | sglang[srt,openai]==0.5.8, torch==2.9.1 |
| `mcore` | mbridge (Megatron-LM bridge) |
| `trtllm` | tensorrt-llm (TensorRT-LLM inference) |
| `gpu` | liger-kernel, flash-attn |
| `math` | math-verify |
| `prime` | pyext |
| `geo` | mathruler, torchvision, qwen_vl_utils |
| `trl` | trl<=0.9.6 |

### B300-Specific Notes

This fork targets NVIDIA B300 (sm_103a). Known issues and workarounds:
- **Triton PTX crash on sm_103a**: Use `enforce_eager=True` in vLLM config for async vLLM; for Triton compilation, use system ptxas
- **CUDA graphs**: Re-enabled after switching to system ptxas for Triton
- **vLLM vs SGLang comparison**: See `reports/verl-vllm-vs-sglang-comparison.md`
- **verl vs slime comparison**: See `reports/verl-vs-slime-root-cause-analysis.md` and `reports/verl-vs-slime-colocate-comparison.md`

### Reports

The `reports/` directory contains analysis and comparison documents for this fork:
- `verl-vllm-vs-sglang-comparison.md` — vLLM vs SGLang inference backend comparison
- `verl-vs-slime-root-cause-analysis.md` — root cause analysis of verl vs slime performance
- `verl-vs-slime-colocate-comparison.md` — colocate mode comparison

### Important AGENTS.md Rules

- Prefer `uv` for Python env management (uv venv, uv pip install)
- Use `Co-authored-by:` trailers for AI-assisted commits
- Do not open low-value busywork PRs
- PR descriptions must include test commands and results
- No pure code-agent PRs; human must review every line
