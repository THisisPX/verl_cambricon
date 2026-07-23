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
"""Block-wise quantization and dequantization with stochastic rounding."""

from typing import Optional

import torch


def stochastic_round(tensor: torch.Tensor, random: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Stochastic rounding: floor(x) + Bernoulli(x - floor(x)).

    For quantizing to integers, this provides unbiased estimates of the
    expected value, unlike nearest rounding which introduces systematic bias.

    Args:
        tensor: Floating-point tensor to round.
        random: Optional pre-generated random tensor in [0,1). If None,
            generated internally.

    Returns:
        Rounded integer tensor (in floating-point dtype of input).
    """
    if random is None:
        random = torch.rand_like(tensor)
    floor_val = torch.floor(tensor)
    frac = tensor - floor_val
    return floor_val + (random < frac).to(tensor.dtype)


def _compute_block_scale(
    tensor: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, int, int]:
    """Compute per-block amax for block-wise quantization.

    Args:
        tensor: Input tensor to quantize (1D or can be reshaped).
        block_size: Number of elements per block.

    Returns:
        Tuple of (block_scales, num_blocks, padded_size).
    """
    flat = tensor.reshape(-1).float()
    numel = flat.numel()
    num_blocks = (numel + block_size - 1) // block_size
    padded_size = num_blocks * block_size

    if padded_size > numel:
        flat = torch.nn.functional.pad(flat, (0, padded_size - numel))

    # Reshape to [num_blocks, block_size], compute per-block amax
    blocked = flat.view(num_blocks, block_size)
    amax = blocked.abs().amax(dim=-1, keepdim=True)  # [num_blocks, 1]
    amax = amax.clamp(min=1e-12)  # avoid division by zero
    return amax, num_blocks, padded_size


def quantize_tensor(
    tensor: torch.Tensor,
    block_size: int = 256,
    quant_dtype: str = "int8",
    use_stochastic_round: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Quantize a tensor to low-bit integer representation with block-wise scaling.

    Args:
        tensor: Input tensor (any shape and dtype).
        block_size: Block size for block-wise quantization.
        quant_dtype: Target quantization type ("int8" or "fp8_e4m3").
        use_stochastic_round: Whether to use stochastic rounding.

    Returns:
        Tuple of (quantized_tensor, scales, original_shape).
        - quantized_tensor: int8 tensor (for int8) or fp8 tensor (for fp8_e4m3),
          shape = [num_blocks, block_size] (padded).
        - scales: float32 tensor of shape [num_blocks, 1].
        - original_shape: shape of the input tensor, for dequantize.
    """
    original_shape = tensor.shape
    numel = tensor.numel()

    if numel == 0:
        return (
            torch.zeros(0, dtype=torch.int8, device=tensor.device),
            torch.ones(0, 1, dtype=torch.float32, device=tensor.device),
            original_shape,
        )

    amax, num_blocks, padded_size = _compute_block_scale(tensor, block_size)
    flat = tensor.reshape(-1).float()
    if padded_size > numel:
        flat = torch.nn.functional.pad(flat, (0, padded_size - numel))
    blocked = flat.view(num_blocks, block_size)

    if quant_dtype == "int8":
        # int8: range [-127, 127], symmetric
        max_val = 127.0
        scale = amax / max_val  # [num_blocks, 1]
        normalized = blocked / scale.clamp(min=1e-12)

        if use_stochastic_round:
            quantized = stochastic_round(normalized)
        else:
            quantized = torch.round(normalized)

        quantized = quantized.clamp(-max_val, max_val).to(torch.int8)

    elif quant_dtype == "fp8_e4m3":
        # FP8 E4M3: use torch's native fp8 types if available
        if hasattr(torch, "float8_e4m3fn"):
            # Use native FP8 cast (no block-wise scaling for simplicity;
            # block scaling is applied at dequantization time via amax)
            # For now, quantize to FP8 directly and store per-block scales
            blocked_fp8 = blocked.to(torch.float8_e4m3fn)
            quantized = blocked_fp8.view(torch.int8)
            scale = amax
        else:
            # Fallback: use int8 with dynamic range
            max_val = 127.0
            scale = amax / max_val
            normalized = blocked / scale.clamp(min=1e-12)
            if use_stochastic_round:
                quantized_arr = stochastic_round(normalized)
            else:
                quantized_arr = torch.round(normalized)
            quantized = quantized_arr.clamp(-max_val, max_val).to(torch.int8)
    else:
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")

    return quantized, scale.to(torch.float32), original_shape


def dequantize_tensor(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
    quant_dtype: str = "int8",
    block_size: int = 256,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dequantize a tensor from block-wise quantized representation.

    Args:
        quantized: Quantized tensor of shape [num_blocks, block_size].
        scales: Per-block scales of shape [num_blocks, 1].
        original_shape: Original tensor shape before quantization.
        quant_dtype: Quantization type used.
        block_size: Block size used for quantization.
        output_dtype: Output dtype. If None, defaults to float32.

    Returns:
        Dequantized tensor in original_shape.
    """
    numel = 1
    for d in original_shape:
        numel *= d

    if numel == 0:
        return torch.zeros(original_shape, dtype=output_dtype or torch.float32, device=quantized.device)

    if quant_dtype == "fp8_e4m3" and hasattr(torch, "float8_e4m3fn"):
        # FP8: cast back from int8 view -> fp8 -> float32, then scale
        fp8_data = quantized.view(torch.float8_e4m3fn)
        result = fp8_data.float() * scales
    else:
        # int8: scale * quantized_value
        result = quantized.float() * scales

    # Unpad and reshape to original shape
    result_flat = result.reshape(-1)[:numel]
    result = result_flat.reshape(original_shape)

    if output_dtype is not None:
        result = result.to(output_dtype)

    return result
