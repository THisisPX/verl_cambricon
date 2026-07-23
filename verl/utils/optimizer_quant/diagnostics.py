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
"""Diagnostics for optimizer state quantization under RL training.

Four proxy metrics for the failure mechanisms:
1. Strategy drift: decomposed into quant-induced gap + optimization drift
2. Optimizer state scale staleness: block-wise scale drift ratio
3. Gradient structural sparsity: per-group effective gradient ratio
4. IS noise amplification: Effective Sample Size (ESS)
"""

from typing import Optional

import torch


class OptimizerQuantDiagnostics:
    """Collects and computes diagnostic metrics for low-bit optimizer training.

    All metrics are designed to be computed with minimal overhead during
    normal training — no per-sample gradient storage required.

    Args:
        block_size: Block size used for quantization (for scale drift).
        log_prefix: Prefix for logging keys.
    """

    def __init__(self, block_size: int = 256, log_prefix: str = "optim_quant"):
        self.block_size = block_size
        self.log_prefix = log_prefix

        # Stored per-step values for scale drift tracking
        self._last_scales: dict[int, torch.Tensor] = {}  # param_id -> scales tensor

        # Per-step counters
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Metric 1: Strategy drift decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def compute_quant_induced_kl_gap(
        log_prob_q: torch.Tensor, log_prob_bf16: torch.Tensor, loss_mask: torch.Tensor
    ) -> float:
        """Quant-induced policy gap: D_KL(pi_Q(theta) || pi(theta)).

        For a fixed theta, this measures the KL divergence between logprobs
        computed under quantized vs BF16 model weights (rollout-side gap).
        In a learner-side context, use this with the same batch to measure
        how optimizer state quantization rotates the update direction.

        Args:
            log_prob_q: Logprobs from quantized path.
            log_prob_bf16: Logprobs from BF16 reference path.
            loss_mask: Mask for valid tokens.

        Returns:
            Mean KL divergence (scalar float).
        """
        diff = log_prob_bf16 - log_prob_q  # log(p/q)
        masked_diff = diff * loss_mask
        kl = masked_diff.sum() / loss_mask.sum().clamp(min=1)
        return kl.item()

    @staticmethod
    def compute_optimization_drift(
        log_prob_theta_current: torch.Tensor,
        log_prob_theta_prev: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> float:
        """Optimization drift: D_KL(pi(theta_t) || pi(theta_{t-1})).

        Measures the policy change purely from parameter updates,
        independent of quantization.

        Args:
            log_prob_theta_current: Logprobs with current parameters.
            log_prob_theta_prev: Logprobs with previous parameters.
            loss_mask: Mask for valid tokens.

        Returns:
            Mean KL divergence (scalar float).
        """
        diff = log_prob_theta_prev - log_prob_theta_current
        masked_diff = diff * loss_mask
        kl = masked_diff.sum() / loss_mask.sum().clamp(min=1)
        return kl.item()

    # ------------------------------------------------------------------
    # Metric 2: Optimizer state scale staleness
    # ------------------------------------------------------------------

    def update_and_compute_scale_drift(
        self, param_id: int, scales: torch.Tensor
    ) -> Optional[float]:
        """Compute scale drift ratio for optimizer state blocks.

        Ratio = ||current_amax - last_calibrated_amax|| / ||last_calibrated_amax||
        averaged over blocks. This measures how much the optimizer state
        distribution has shifted since the last quantization scale calibration.

        Args:
            param_id: Unique identifier for this parameter tensor.
            scales: Current block-wise scales (amax) of shape [num_blocks, 1].

        Returns:
            Scale drift ratio (scalar), or None if no previous scales stored.
        """
        if param_id not in self._last_scales:
            self._last_scales[param_id] = scales.detach().clone()
            return None

        last = self._last_scales[param_id]
        current = scales.detach()

        # Per-block relative change
        drift = (current - last).abs() / last.clamp(min=1e-12)
        mean_drift = drift.mean().item()

        self._last_scales[param_id] = current.clone()
        return mean_drift

    # ------------------------------------------------------------------
    # Metric 3: Gradient structural sparsity (per-GRPO-group)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_per_group_effective_grad_ratio(
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        num_groups: int = 1,
        threshold: float = 0.01,
    ) -> float:
        """Fraction of groups with |advantage| above threshold.

        In GRPO, groups where all samples are correct or all wrong have
        advantage ≈ 0, contributing near-zero gradient for the entire group.
        This is a structural sparsity pattern distinct from random sparsity.

        Args:
            advantages: Per-token advantages of shape [batch, seq_len].
            response_mask: Valid token mask of shape [batch, seq_len].
            num_groups: Number of groups (prompts). Each group has
                rollout_n samples.
            threshold: Advantage magnitude below which a group is "sparse".

        Returns:
            Fraction of groups with effective (non-sparse) gradients.
        """
        if advantages.numel() == 0:
            return 0.0

        # Mean absolute advantage per sample
        masked_adv = advantages * response_mask
        per_sample_mean = masked_adv.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)

        effective = (per_sample_mean.abs() > threshold).float().mean().item()
        return effective

    @staticmethod
    def compute_grad_sparsity_below_threshold(
        grad: torch.Tensor, quant_step: float
    ) -> float:
        """Fraction of gradient elements below the quantization step size.

        Elements with |g| < quant_step/2 are quantized to zero, losing all
        gradient information. High fraction → severe SNR degradation.

        Args:
            grad: Gradient tensor.
            quant_step: Quantization step size (e.g., amax/127 for int8).

        Returns:
            Fraction of elements below threshold.
        """
        if grad.numel() == 0:
            return 1.0
        below = (grad.abs() < quant_step / 2).float().mean().item()
        return below

    # ------------------------------------------------------------------
    # Metric 4: IS noise amplification via ESS
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ess(is_weights: torch.Tensor) -> float:
        """Effective Sample Size from importance sampling weights.

        ESS = (sum w_i)^2 / sum(w_i^2)

        Low ESS → high variance in IS-corrected gradient estimates.
        This is the standard off-policy diagnostic that maps directly to
        Var(IS_corrected_grad) / Var(raw_grad) ≈ ESS_ref / ESS_actual.

        Args:
            is_weights: Importance sampling weights of shape [batch, ...].

        Returns:
            ESS as a scalar float.
        """
        if is_weights.numel() == 0:
            return 1.0

        w = is_weights.float()
        sum_w = w.sum()
        if sum_w == 0:
            return 0.0

        sum_w_sq = (w ** 2).sum()
        ess = (sum_w ** 2) / sum_w_sq.clamp(min=1e-12)
        # Normalize to [0, 1] range relative to batch size
        normalized_ess = ess.item() / w.numel()
        return min(normalized_ess, 1.0)

    # ------------------------------------------------------------------
    # Aggregate logging
    # ------------------------------------------------------------------

    def get_diagnostics(
        self,
        scale_drift_values: Optional[list[float]] = None,
        effective_grad_ratio: Optional[float] = None,
        grad_below_threshold: Optional[float] = None,
        ess_value: Optional[float] = None,
        quant_kl_gap: Optional[float] = None,
        opt_drift: Optional[float] = None,
        global_step: int = 0,
    ) -> dict[str, float]:
        """Package all diagnostics into a dict for logging.

        Returns:
            Dict of metric_name -> value, with self.log_prefix prepended.
        """
        metrics: dict[str, float] = {
            f"{self.log_prefix}/step": float(self._step_count),
        }
        self._step_count += 1

        if scale_drift_values:
            metrics[f"{self.log_prefix}/scale_drift_mean"] = float(
                sum(scale_drift_values) / max(len(scale_drift_values), 1)
            )
            metrics[f"{self.log_prefix}/scale_drift_max"] = float(
                max(scale_drift_values)
            )

        if effective_grad_ratio is not None:
            metrics[f"{self.log_prefix}/effective_grad_ratio"] = float(effective_grad_ratio)

        if grad_below_threshold is not None:
            metrics[f"{self.log_prefix}/grad_below_quant_threshold"] = float(grad_below_threshold)

        if ess_value is not None:
            metrics[f"{self.log_prefix}/ess"] = float(ess_value)

        if quant_kl_gap is not None:
            metrics[f"{self.log_prefix}/quant_kl_gap"] = float(quant_kl_gap)

        if opt_drift is not None:
            metrics[f"{self.log_prefix}/opt_drift"] = float(opt_drift)

        return metrics
