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
import base64
import os
import re
from io import BytesIO

import datasets
from PIL import Image


def _decode_data_uri(uri: str) -> Image.Image:
    """Decode a ``data:image/...;base64,...`` URI to a PIL Image."""
    _, b64 = uri.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_pil(img) -> Image.Image:
    """Normalize any image entry to a PIL Image.

    ``chenhegu/geo3k_imgurl`` stores images as ``{"bytes": "data:image/...;base64,..."}``.
    ``load_dataset`` returns these as-is — we must decode them here so the
    ``images`` column contains PIL.Image objects, which datasets can serialize
    to parquet natively (arrow binary type).
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if "image" in img and isinstance(img["image"], Image.Image):
            return img["image"].convert("RGB")
        if "bytes" in img:
            if isinstance(img["bytes"], str):
                return _decode_data_uri(img["bytes"])
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    if isinstance(img, str):
        if img.startswith("data:"):
            return _decode_data_uri(img)
        return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


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

    # Some geo3k problem texts may contain a literal "<image>" substring
    # (e.g. HTML markup).  verl's RLHFDataset._build_messages uses
    # re.split("(<image>)") to locate images — a literal "<image>" would
    # be counted as a second image placeholder and crash with an assertion
    # error.  Strip them from the problem text.
    _RE_LITERAL_IMAGE = re.compile(r"<image>", re.IGNORECASE)

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = _RE_LITERAL_IMAGE.sub("", example.pop("problem"))
            answer = example.pop("answer")
            raw_images = example.pop("images")
            pil_images = [_to_pil(img) for img in raw_images]

            prompt_text = "<image>" + problem + " " + instruction_following

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                "images": pil_images,
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
