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
"""QuantizedOptimizerWrapper: wraps any torch optimizer to quantize states."""

import logging
from typing import Optional

import torch

from .config import OptimizerStateQuantConfig
from .diagnostics import OptimizerQuantDiagnostics
from .quantization import dequantize_tensor, quantize_tensor

logger = logging.getLogger(__file__)


class QuantizedOptimizerWrapper:
    """Wraps a torch optimizer, quantizing Adam states after each step.

    The wrapper intercepts optimizer.step() calls and:
    1. Optionally quantizes gradients before the step (if grad_quant_bits is set).
    2. Calls the underlying optimizer.step().
    3. Quantizes exp_avg and exp_avg_sq to 8-bit / 4-bit.
    4. Tracks scale drift and other diagnostics.

    Only Adam/AdamW-style optimizers with per-parameter state dicts
    containing 'exp_avg' and 'exp_avg_sq' keys are supported.

    Args:
        optimizer: The wrapped torch optimizer.
        config: Quantization configuration.
        diagnostics: Optional diagnostics collector.
    """

    STATE_KEYS = ("exp_avg", "exp_avg_sq")

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerStateQuantConfig,
        diagnostics: Optional[OptimizerQuantDiagnostics] = None,
    ):
        self._optimizer = optimizer
        self._config = config
        self._diagnostics = diagnostics or (
            OptimizerQuantDiagnostics(block_size=config.block_size)
            if config.log_diagnostics
            else None
        )
        self._step_count = 0

    # Delegate attribute access to the wrapped optimizer.
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._optimizer, name)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        """Quantize-aware optimizer step.

        1. Quantize gradients (if enabled).
        2. Dequantize optimizer states for the update.
        3. Call underlying optimizer.step().
        4. Quantize optimizer states back.
        5. Optionally collect diagnostics.
        """
        config = self._config
        step = self._step_count
        self._step_count += 1

        # --- Step 1: Gradient quantization (optional) ---
        if config.grad_quant_bits is not None:
            self._quantize_gradients(config.grad_quant_bits)

        # --- Step 2: Dequantize optimizer states for update ---
        if step > 0:  # states are quantized after first step
            self._restore_optimizer_states()

        # --- Step 3: Run the optimizer step ---
        result = self._optimizer.step(*args, **kwargs)

        # --- Step 4: Re-quantize optimizer states ---
        # Recalibrate scales if recalibrate_freq is set
        needs_recalibration = (
            config.recalibrate_freq is None
            or step % config.recalibrate_freq == 0
        )
        self._quantize_optimizer_states(needs_recalibration)

        return result

    def _quantize_gradients(self, bits: int) -> None:
        """Quantize gradients in-place to reduce precision.

        Uses shared-exponent block-wise quantization: for each parameter,
        g_quant = round(g / scale) * scale, where scale = amax / (2^(bits-1) - 1).

        Args:
            bits: Bit width for quantized gradients (4 or 8).
        """
        max_val = 2 ** (bits - 1) - 1
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                with torch.no_grad():
                    amax = grad.abs().max().clamp(min=1e-12)
                    scale = amax / max_val
                    if self._config.stochastic_round:
                        from .quantization import stochastic_round

                        normalized = grad.float() / scale
                        quantized = stochastic_round(normalized)
                    else:
                        quantized = torch.round(grad.float() / scale)
                    quantized = quantized.clamp(-max_val, max_val)
                    p.grad = (quantized * scale).to(grad.dtype)

    def _restore_optimizer_states(self) -> None:
        """Dequantize optimizer states in-place before optimizer.step()."""
        config = self._config
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                if p not in self._optimizer.state:
                    continue
                state = self._optimizer.state[p]
                for key in self.STATE_KEYS:
                    if key not in state:
                        continue
                    val = state[key]
                    if val is None:
                        continue
                    # If it's quantized (stored as a tuple), dequantize
                    if isinstance(val, tuple) and len(val) == 3:
                        q, scales, orig_shape = val
                        restored = dequantize_tensor(
                            q, scales, orig_shape,
                            quant_dtype=config.quant_dtype,
                            block_size=config.block_size,
                            output_dtype=p.dtype,
                        )
                        state[key] = restored

    def _quantize_optimizer_states(self, needs_recalibration: bool) -> None:
        """Quantize optimizer states after optimizer.step().

        Args:
            needs_recalibration: Whether to compute fresh block-wise scales.
                If False, reuse existing scales (from previous step) for
                the quantization — this introduces scale staleness deliberately
                to study its effect on training stability.
        """
        config = self._config
        scale_drift_values: list[float] = []

        for param_id, group in enumerate(self._optimizer.param_groups):
            for p in group["params"]:
                if p not in self._optimizer.state:
                    continue
                state = self._optimizer.state[p]
                for key in self.STATE_KEYS:
                    if key not in state:
                        continue
                    val = state[key]
                    if val is None:
                        continue
                    # Skip if already quantized (should not happen after restore)
                    if isinstance(val, tuple):
                        continue

                    do_quantize = (
                        (key == "exp_avg" and config.quantize_exp_avg)
                        or (key == "exp_avg_sq" and config.quantize_exp_avg_sq)
                    )
                    if not do_quantize:
                        continue

                    q, scales, orig_shape = quantize_tensor(
                        val,
                        block_size=config.block_size,
                        quant_dtype=config.quant_dtype,
                        use_stochastic_round=config.stochastic_round,
                    )

                    # Track scale drift for diagnostics
                    if needs_recalibration and self._diagnostics is not None:
                        drift = self._diagnostics.update_and_compute_scale_drift(
                            param_id, scales
                        )
                        if drift is not None:
                            scale_drift_values.append(drift)

                    state[key] = (q, scales, orig_shape)

        # --- Collect diagnostics ---
        if self._diagnostics is not None and config.log_diagnostics:
            diagnostics_kwargs: dict = {}
            if scale_drift_values:
                diagnostics_kwargs["scale_drift_values"] = scale_drift_values

            metrics = self._diagnostics.get_diagnostics(
                global_step=self._step_count,
                **diagnostics_kwargs,
            )

    def get_diagnostics_for_logging(
        self,
        advantages: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        is_weights: Optional[torch.Tensor] = None,
        log_prob_q: Optional[torch.Tensor] = None,
        log_prob_bf16: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Compute remaining diagnostics from training data (called externally).

        Args:
            advantages: Per-token advantages for effective_grad_ratio.
            response_mask: Valid response token mask.
            is_weights: Importance sampling weights for ESS.
            log_prob_q: Quantized-path logprobs for KL gap.
            log_prob_bf16: BF16 reference logprobs for KL gap.

        Returns:
            Dict of diagnostic metrics for logging.
        """
        if self._diagnostics is None:
            return {}

        diagnostics = self._diagnostics
        kwargs: dict = {}

        if advantages is not None and response_mask is not None:
            kwargs["effective_grad_ratio"] = diagnostics.compute_per_group_effective_grad_ratio(
                advantages, response_mask, threshold=0.01
            )

        if is_weights is not None:
            kwargs["ess_value"] = diagnostics.compute_ess(is_weights)

        if log_prob_q is not None and log_prob_bf16 is not None and response_mask is not None:
            kwargs["quant_kl_gap"] = diagnostics.compute_quant_induced_kl_gap(
                log_prob_q, log_prob_bf16, response_mask
            )

        return diagnostics.get_diagnostics(**kwargs)
