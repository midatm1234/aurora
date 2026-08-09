"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Bias-aware auxiliary objectives for stochastic residual refinement.

The primary loss of each refiner stays mathematically correct for its own
generative process (diffusion or flow matching). These optional terms target
Aurora's *systematic* rollout bias more directly::

    total_loss = generative_loss
               + reconstruction_weight * reconstruction_loss
               + bias_weight           * bias_loss
               + gradient_weight       * gradient_loss
               + pattern_correlation_weight * pattern_correlation_loss

All auxiliary terms are computed from a **clean-residual estimate**:

* diffusion reconstructs the clean residual from the configured prediction
  parameterisation (``epsilon`` / ``velocity`` / ``sample``) with the same
  conversion helpers the sampler uses;
* flow matching derives an endpoint estimate consistently with the selected
  interpolation path (directly for the ``existing_aurora`` data
  parameterisation, and as ``x_t + (1 - t) * v`` for the ``rectified_flow``
  velocity parameterisation).

Errors are never allowed to cancel across unrelated quantities: the mean bias is
accumulated **separately** per configured group (variable, pressure level,
forecast lead time) and only then aggregated. Every component is returned
separately so it can be logged on its own.

All weights default to ``0.0``; a configuration that does not enable them keeps
the pure generative objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from finetune.refinement.config import LossConfig
from finetune.refinement.packing import FieldPacking

__all__ = ["BiasLossTerms", "area_weights_from_latitudes", "compute_auxiliary_losses"]


@dataclass
class BiasLossTerms:
    """Individually logged auxiliary loss components."""

    reconstruction: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    gradient: torch.Tensor | None = None
    pattern_correlation: torch.Tensor | None = None

    def weighted_total(self, config: LossConfig, reference: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=reference.device, dtype=torch.float32)
        if self.reconstruction is not None:
            total = total + config.reconstruction_weight * self.reconstruction
        if self.bias is not None:
            total = total + config.bias_weight * self.bias
        if self.gradient is not None:
            total = total + config.gradient_weight * self.gradient
        if self.pattern_correlation is not None:
            total = total + config.pattern_correlation_weight * self.pattern_correlation
        return total


def area_weights_from_latitudes(
    lat: Sequence[float] | torch.Tensor | None,
    height: int,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor | None:
    """Cosine-latitude area weights shaped ``[1, 1, H, 1]``.

    Returns ``None`` when the latitude vector is unavailable or does not match
    the field height, so callers fall back to unweighted statistics rather than
    silently using a wrong weighting.
    """
    if lat is None:
        return None
    values = torch.as_tensor(list(lat) if not torch.is_tensor(lat) else lat, dtype=torch.float64)
    if values.numel() != height:
        return None
    weights = torch.cos(values * math.pi / 180.0).clamp(min=0.0)
    if float(weights.sum()) <= 0.0:
        return None
    return weights.to(device=device, dtype=dtype).view(1, 1, -1, 1)


def _group_ids(
    packing: FieldPacking | None,
    channels: int,
    config: LossConfig,
    lead_index: torch.Tensor | None,
    batch: int,
    device,
) -> torch.Tensor:
    """``[N, C]`` integer group id used to keep unrelated errors from cancelling."""
    if packing is not None and len(packing.channels) == channels:
        if config.separate_by_level:
            channel_group = list(range(channels))
        elif config.separate_by_variable:
            variables = packing.variables
            lookup = {name: idx for idx, name in enumerate(variables)}
            channel_group = [lookup[spec.aurora_name] for spec in packing.channels]
        else:
            channel_group = [0] * channels
    else:
        channel_group = list(range(channels)) if config.separate_by_level else [0] * channels

    channel_tensor = torch.tensor(channel_group, dtype=torch.long, device=device)
    num_channel_groups = int(channel_tensor.max().item()) + 1 if channels else 1

    if config.separate_by_lead_time and lead_index is not None:
        lead = lead_index.to(device=device, dtype=torch.long).reshape(-1)
        if lead.numel() != batch:
            raise ValueError(
                f"lead_index has {lead.numel()} entries but the effective batch is {batch}."
            )
    else:
        lead = torch.zeros(batch, dtype=torch.long, device=device)
    return lead.view(-1, 1) * num_channel_groups + channel_tensor.view(1, -1)


def _grouped_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean of ``values`` inside each group of ``groups``.

    ``values`` / ``weights`` are ``[N, C]`` reductions over the spatial axes and
    ``groups`` is the matching ``[N, C]`` integer group id.
    """
    flat_groups = groups.reshape(-1)
    size = int(flat_groups.max().item()) + 1 if flat_groups.numel() else 1
    numerator = torch.zeros(size, dtype=values.dtype, device=values.device)
    denominator = torch.zeros(size, dtype=values.dtype, device=values.device)
    numerator = numerator.index_add(0, flat_groups, values.reshape(-1))
    denominator = denominator.index_add(0, flat_groups, weights.reshape(-1))
    present = denominator > 0
    means = torch.zeros_like(numerator)
    means[present] = numerator[present] / denominator[present]
    return means, present


def compute_auxiliary_losses(
    residual_estimate: torch.Tensor,
    residual_target: torch.Tensor,
    *,
    config: LossConfig,
    mask: torch.Tensor | None = None,
    packing: FieldPacking | None = None,
    area_weight: torch.Tensor | None = None,
    lead_index: torch.Tensor | None = None,
    rollout_normalized: torch.Tensor | None = None,
) -> BiasLossTerms:
    """Compute the enabled auxiliary terms.

    Args:
        residual_estimate: clean-residual estimate in normalized target space,
            ``[N, C, H, W]``.
        residual_target: normalized residual target, same shape.
        config: resolved :class:`LossConfig`.
        mask: boolean validity mask, same shape.
        packing: channel metadata used to group the bias by variable / level.
        area_weight: ``[1, 1, H, 1]`` cosine-latitude weights, or ``None``.
        lead_index: ``[N]`` rollout-step index used to group the bias by
            forecast lead time. This is bookkeeping only; it never enters the
            network.
        rollout_normalized: deterministic rollout in normalized space, required
            for the pattern-correlation term.
    """
    if not config.has_auxiliary_terms:
        return BiasLossTerms()

    estimate = residual_estimate.float()
    target = residual_target.float()
    error = estimate - target
    weight = torch.ones_like(error) if mask is None else mask.to(error.dtype)
    if config.area_weighted and area_weight is not None:
        weight = weight * area_weight.to(device=error.device, dtype=error.dtype)

    terms = BiasLossTerms()
    denom = weight.sum().clamp(min=1.0)

    if config.reconstruction_weight > 0.0:
        terms.reconstruction = (error.pow(2) * weight).sum() / denom

    if config.bias_weight > 0.0:
        groups = _group_ids(
            packing, error.shape[1], config, lead_index, error.shape[0], error.device
        )
        numerator = (error * weight).sum(dim=(-2, -1))
        denominator = weight.sum(dim=(-2, -1))
        group_bias, present = _grouped_mean(numerator, denominator, groups)
        terms.bias = (group_bias[present] ** 2).mean() if bool(present.any()) else error.sum() * 0.0

    if config.gradient_weight > 0.0:
        terms.gradient = _gradient_loss(error, weight)

    if config.pattern_correlation_weight > 0.0:
        if rollout_normalized is None:
            raise ValueError(
                "refinement.loss.pattern_correlation_weight requires the deterministic "
                "rollout in normalized space."
            )
        base = rollout_normalized.float()
        terms.pattern_correlation = _pattern_correlation_loss(
            base + estimate, base + target, weight
        )
    return terms


def _gradient_loss(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Masked mean-square of the spatial finite differences of the error."""
    total = torch.zeros((), device=error.device, dtype=error.dtype)
    count = torch.zeros((), device=error.device, dtype=error.dtype)
    for dim in (-2, -1):
        diff = error.diff(dim=dim)
        pair_weight = torch.minimum(
            weight.narrow(dim, 0, weight.shape[dim] - 1),
            weight.narrow(dim, 1, weight.shape[dim] - 1),
        )
        total = total + (diff.pow(2) * pair_weight).sum()
        count = count + pair_weight.sum()
    return total / count.clamp(min=1.0)


def _pattern_correlation_loss(
    refined: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``1 - r`` per (sample, channel), averaged over well-conditioned entries."""
    dims = (-2, -1)
    norm = weight.sum(dim=dims).clamp(min=1.0)
    refined_mean = (refined * weight).sum(dim=dims) / norm
    target_mean = (target * weight).sum(dim=dims) / norm
    refined_anom = (refined - refined_mean[..., None, None]) * weight
    target_anom = (target - target_mean[..., None, None]) * weight
    cov = (refined_anom * target_anom).sum(dim=dims)
    refined_var = refined_anom.pow(2).sum(dim=dims)
    target_var = target_anom.pow(2).sum(dim=dims)
    # Guard against degenerate (constant) fields, which have no defined
    # correlation: they are excluded rather than contributing a fake 1.0.
    valid = (refined_var > 1e-12) & (target_var > 1e-12)
    if not bool(valid.any()):
        return refined.sum() * 0.0
    corr = cov[valid] / (refined_var[valid].sqrt() * target_var[valid].sqrt())
    return (1.0 - corr.clamp(-1.0, 1.0)).mean()
