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

from .config import OptimizerStateQuantConfig
from .diagnostics import OptimizerQuantDiagnostics
from .quantization import dequantize_tensor, quantize_tensor, quantize_tensor_with_scale, stochastic_round
from .wrapper import QuantizedOptimizerWrapper

__all__ = [
    "OptimizerStateQuantConfig",
    "OptimizerQuantDiagnostics",
    "QuantizedOptimizerWrapper",
    "quantize_tensor",
    "quantize_tensor_with_scale",
    "dequantize_tensor",
    "stochastic_round",
]
