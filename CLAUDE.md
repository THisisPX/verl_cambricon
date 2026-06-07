# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready RL training library for large language models. It's developed by ByteDance Seed team and published as HybridFlow (EuroSys 2025). Version: 0.9.0.dev.

## Build & Install

```bash
# Editable install (dev)
pip install -e .[test,vllm]

# Or with SGLang backend
pip install -e .[test,sglang]

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

## Code Architecture

### High-Level Design

verl uses a **hybrid-controller programming model** that decouples computation and data dependencies. It's built on **Ray** for distributed orchestration. The core data transfer protocol is `DataProto` (`verl/protocol.py`), which wraps `TensorDict` for zero-copy structured data passing between components.

### Package Structure

```
verl/
  protocol.py           - DataProto: core data transfer abstraction (TensorDict-based)
  single_controller/    - HybridFlow controller programming model
    base/               - Worker, WorkerGroup, ResourcePool abstractions
    ray/                - Ray-based distributed implementation
  trainer/
    main_ppo.py         - PPO trainer entry point (Ray-based)
    main_ppo_sync.py    - Synchronous PPO trainer (main training loop)
    sft_trainer.py      - SFT trainer
    sft_trainer_ray.py  - SFT trainer (Ray-based)
    config/             - Hydra/OmegaConf YAML configs for all components
    ppo/                - PPO core algorithm implementations (core_algos.py)
  workers/
    engine/             - Training engines (BaseEngine abstract class)
      base.py           - BaseEngine interface
      fsdp/             - FSDP/FSDP2 engine implementation
      megatron/         - Megatron-LM engine implementation
      automodel/        - AutoModel engine
      veomni/           - VeOmni engine
      torchtitan/       - TorchTitan engine
      mindspeed/        - Mindspeed engine
    rollout/            - Rollout generation backends
      vllm_rollout/     - vLLM integration
      sglang_rollout/   - SGLang integration
      trtllm_rollout/   - TensorRT-LLM integration
      hf_rollout.py     - HuggingFace transformers rollout
    reward_manager/     - Reward computation
    config/             - Worker-specific configs
  models/               - Model definitions
    transformers/       - HuggingFace model wrappers
    mcore/              - Megatron-Core model wrappers
    registry.py         - Model registration
  utils/                - Shared utilities
    fsdp_utils.py       - FSDP helper functions
    megatron_utils.py   - Megatron helper functions
    tensordict_utils.py - TensorDict manipulation
    model.py            - Model loading/saving/utilities
    dataset/            - Dataset loading and preprocessing
    reward_score/       - Reward function implementations
    checkpoint/         - Checkpoint save/load
    vllm/               - vLLM-specific utilities
    sglang/             - SGLang-specific utilities
  checkpoint_engine/    - Checkpoint engine abstraction
  experimental/         - Experimental features (async policy, off-policy, VLA)
  plugin/               - Plugin system via entry_points
```

### Key Backend Options

- **Training**: FSDP, FSDP2, Megatron-LM (via mbridge), TorchTitan, VeOmni, Mindspeed
- **Rollout/Inference**: vLLM (>=0.8.5), SGLang (0.5.8), HF Transformers, TensorRT-LLM
- **Hardware**: NVIDIA (CUDA), AMD (ROCm), Ascend NPU

### Configuration System

Uses **Hydra/OmegaConf** with YAML configs in `verl/trainer/config/`. The main entry configs are `ppo_trainer.yaml`, `ppo_megatron_trainer.yaml`, and generated defaults in `_generated_*.yaml`. Configs are split by component: `actor/`, `algorithm/`, `rollout/`, `critic/`, `data/`, `model/`, `reward/`, `ref/`, `optim/`, etc.

### RL Algorithms Supported

PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, VAPO, PF-PPO, SPPO, and more. Algorithm losses and advantage estimators are in `verl/trainer/ppo/core_algos.py`.

### Examples

`examples/` contains runnable training scripts per algorithm:
- `examples/ppo_trainer/`, `examples/grpo_trainer/`, etc.
- Each contains shell scripts (e.g., `run_qwen3_8b_fsdp.sh`) showing how to launch training.
- `examples/tutorial/` has getting-started walkthroughs.

### Key Environment Variables

- `VERL_USE_EXTERNAL_MODULES` - comma-separated external modules to import
- `VERL_USE_EXTERNAL_PLUGINS` - plugin discovery policy (`auto`/`none`/comma-separated)
- `VERL_USE_MODELSCOPE` - use ModelScope hub instead of HuggingFace
- `VERL_AUTO_PADDING` - enable auto-padding in DataProto

### Important AGENTS.md Rules

- Prefer `uv` for Python env management (uv venv, uv pip install)
- Use `Co-authored-by:` trailers for AI-assisted commits
- Do not open low-value busywork PRs
- PR descriptions must include test commands and results
- No pure code-agent PRs; human must review every line
