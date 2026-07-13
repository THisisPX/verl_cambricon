# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the chenhegu/geo3k_imgurl dataset (same format as slime) to verl parquet format.

Usage:
    python examples/data_preprocess/geo3k_imgurl.py \
        --local_save_dir ~/data/geo3k_imgurl
"""

import argparse
import os
from io import BytesIO

import datasets
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/geo3k_imgurl"))
    parser.add_argument("--local_dataset_path", default=None, help="Local path to already-downloaded dataset.")
    args = parser.parse_args()

    data_source = "chenhegu/geo3k_imgurl"

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    def _load_image(img):
        """Convert a raw image entry to PIL.Image.

        The local chenhegu/geo3k_imgurl dataset stores images as file paths (str).
        verl RHLFDataset._build_messages requires PIL.Image or dict — str raises TypeError.
        """
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, dict):
            if "image" in img:
                return img
            if "bytes" in img:
                img["image"] = Image.open(BytesIO(img["bytes"]))
                return img
        if isinstance(img, bytes):
            return Image.open(BytesIO(img))
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(img)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            # <image> placeholder is required by verl's RLHFDataset._build_messages
            # to correctly embed images into the message content list.
            # Without it, images are silently dropped and the model never sees
            # the geometry diagram (causing 0% reward / 234-token mean responses).
            prompt = "<image>" + problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")
            # Convert string paths (or other formats) to PIL.Image list
            images = [_load_image(img) for img in images]

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))
    print(f"Preprocessed data saved to {args.local_save_dir}")
