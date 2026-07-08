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
Custom reward function for chenhegu/geo3k_imgurl dataset.

Uses mathruler.grader (LaTeX formula comparison) — matches slime's --rm-type math behavior.
"""

from verl.utils.reward_score.geo3k import compute_score as _geo3k_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Wrapper matching verl's custom_reward_function interface.

    Args:
        data_source: Dataset identifier string (unused, delegated to geo3k scorer).
        solution_str: The model's generated response text.
        ground_truth: The ground truth answer (LaTeX formula).
        extra_info: Optional extra info dict.
    """
    return _geo3k_compute_score(solution_str, ground_truth)
