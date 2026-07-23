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
"""Configuration for optimizer state quantization."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizerStateQuantConfig:
    """Configuration for quantizing optimizer states (exp_avg, exp_avg_sq).

    This applies block-wise stochastic rounding quantization to Adam/AdamW
    optimizer states after each optimizer step, reducing memory footprint at
    the cost of precision.

    Args:
        enable: Whether to enable optimizer state quantization.
        quant_dtype: Target dtype for quantized states ("int8", "fp8_e4m3").
        block_size: Block size for block-wise quantization. Default 256.
        quantize_exp_avg: Whether to quantize first momentum (exp_avg).
        quantize_exp_avg_sq: Whether to quantize second momentum (exp_avg_sq).
        stochastic_round: Use stochastic rounding instead of nearest rounding.
        recalibrate_freq: Frequency (in optimizer steps) to recalibrate block-wise
            quantization scales. None means every step. Higher values
            introduce scale staleness but reduce overhead.
        grad_quant_bits: If set, quantize gradients to this bit-width before
            optimizer step. None means no gradient quantization.
            Supported: 4, 8.
        log_diagnostics: Whether to log diagnostic metrics (scale drift, ESS, etc.).
    """

    enable: bool = False
    quant_dtype: str = "int8"  # "int8" or "fp8_e4m3"
    block_size: int = 256
    quantize_exp_avg: bool = True
    quantize_exp_avg_sq: bool = True
    stochastic_round: bool = True
    recalibrate_freq: Optional[int] = None  # None = every step
    grad_quant_bits: Optional[int] = None  # 4 or 8, None = no grad quant
    log_diagnostics: bool = True
