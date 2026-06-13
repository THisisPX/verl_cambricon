#!/usr/bin/env python3
"""Preprocess dapo-math-17k and aime-2024 parquet files for verl training.

Usage:
    python3 examples/grpo_trainer/preprocess_data.py

Override paths via env vars:
    TRAIN_RAW=/path/to/dapo-math-17k.parquet
    TEST_RAW=/path/to/aime-2024.parquet
    TRAIN_OUT=/path/to/dapo-math-17k-verl.parquet
    TEST_OUT=/path/to/aime-2024-verl.parquet
"""
import os
import json
import pandas as pd


def preprocess(raw_path: str, out_path: str, data_source: str, system_prompt: str = None):
    """Read raw parquet (columns: prompt, label) and write verl-format parquet."""
    df = pd.read_parquet(raw_path)
    print(f"Reading {raw_path}: {len(df)} rows, columns={df.columns.tolist()}")
    if system_prompt:
        print(f"Using system prompt: {system_prompt}")

    records = []
    for _, row in df.iterrows():
        # Ensure prompt is a native Python string, not numpy
        prompt_text = str(row["prompt"]) if not isinstance(row["prompt"], str) else row["prompt"]
        label_text = str(row["label"]) if not isinstance(row["label"], str) else row["label"]

        # Build prompt as native Python list of dicts
        prompt = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": prompt_text})

        # Build reward_model as native Python dict
        reward_model = {"style": "rule", "ground_truth": label_text}

        records.append({
            "data_source": data_source,
            "prompt": prompt,
            "ability": "math",
            "reward_model": reward_model,
        })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(out_path, index=False)
    print(f"Written {out_path}: {len(out_df)} rows")

    # Verify using datasets library (same as verl's RLHFDataset)
    try:
        from datasets import Dataset
        hf_ds = Dataset.from_parquet(out_path)
        row0 = hf_ds[0]
        prompt_type = type(row0["prompt"]).__name__
        prompt_content_type = type(row0["prompt"][0]["content"]).__name__
        rm_type = type(row0["reward_model"]).__name__
        print(f"  Verify (datasets) - prompt type={prompt_type}, content type={prompt_content_type}, reward_model type={rm_type}")
        assert prompt_type == "list", f"Expect list, got {prompt_type}"
        assert prompt_content_type == "str", f"Expect str, got {prompt_content_type}"
        assert rm_type == "dict", f"Expect dict, got {rm_type}"
        print("  OK")
    except ImportError:
        # Fallback: verify with pandas
        verify = pd.read_parquet(out_path)
        row0 = verify.iloc[0]
        prompt_type = type(row0["prompt"]).__name__
        prompt_content_type = type(row0["prompt"][0]["content"]).__name__
        rm_type = type(row0["reward_model"]).__name__
        print(f"  Verify (pandas) - prompt type={prompt_type}, content type={prompt_content_type}, reward_model type={rm_type}")
        print(f"  WARNING: datasets library not available, verify with pandas only")


if __name__ == "__main__":
    train_raw = os.environ.get("TRAIN_RAW", "/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k.parquet")
    test_raw = os.environ.get("TEST_RAW", "/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.parquet")
    train_out = os.environ.get("TRAIN_OUT", "/workspace/volume/pengxiong/datasets/dapo-math-17k/dapo-math-17k-verl.parquet")
    test_out = os.environ.get("TEST_OUT", "/workspace/volume/pengxiong/datasets/aime-2024/aime-2024-verl.parquet")
    # Match slime: --rollout-system-prompt "Please reason step by step, and put your final answer in \boxed{}."
    system_prompt = os.environ.get("SYSTEM_PROMPT", None)

    preprocess(train_raw, train_out, "math_dapo", system_prompt)
    preprocess(test_raw, test_out, "aime2024", system_prompt)

    print("\nDone! Run training with:")
    print(f"  TRAIN_FILE={train_out} \\")
    print(f"  TEST_FILE={test_out} \\")
    print(f"  MODEL_PATH=... \\")
    print(f"  bash examples/grpo_trainer/run_qwen3_4b_fsdp_perf_test.sh")
