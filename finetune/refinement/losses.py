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
from typing import Callable, Sequence

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
    deterministic: torch.Tensor | None = None
    mae: torch.Tensor | None = None
    extreme: torch.Tensor | None = None
    peak: torch.Tensor | None = None
    quantile: torch.Tensor | None = None
    variance: torch.Tensor | None = None
    spectral: torch.Tensor | None = None
    degradation: torch.Tensor | None = None
    magnitude: torch.Tensor | None = None

    #: term name -> the ``LossConfig`` weight that scales it.
    _WEIGHTS = {
        "reconstruction": "reconstruction_weight",
        "bias": "bias_weight",
        "gradient": "gradient_weight",
        "pattern_correlation": "pattern_correlation_weight",
        "deterministic": "deterministic_weight",
        "mae": "mae_weight",
        "extreme": "extreme_weight",
        "peak": "peak_weight",
        "quantile": "quantile_weight",
        "variance": "variance_weight",
        "spectral": "spectral_weight",
        "degradation": "degradation_weight",
        "magnitude": "magnitude_weight",
    }

    def weighted_total(self, config: LossConfig, reference: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=reference.device, dtype=torch.float32)
        for name, weight_key in self._WEIGHTS.items():
            value = getattr(self, name)
            if value is not None:
                total = total + float(getattr(config, weight_key)) * value
        return total

    def scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in self._WEIGHTS
            if getattr(self, name) is not None
        }


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
    deterministic_estimate: torch.Tensor | None = None,
    decode: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> BiasLossTerms:
    """Compute the enabled auxiliary terms.

    Two spaces are in play and the distinction matters for how the weights
    behave:

    * **scaled generative space** - ``residual_estimate``, ``residual_target``
      and ``deterministic_estimate`` arrive here already standardized, i.e. in
      the same space as the generative loss they are added to. Every term that
      is a function of the residual error alone (reconstruction, mae,
      deterministic, degradation, magnitude, bias, gradient) is computed there,
      so a weight of ``1.0`` really does mean "as important as the generative
      objective" regardless of how small the physical residual happens to be.
    * **normalized target space** - terms that need the refined *field*
      (``rollout + residual``) are computed after ``decode``, because adding a
      standardized residual to an un-standardized rollout would be meaningless.

    Args:
        residual_estimate: clean-residual estimate, ``[N, C, H, W]``, scaled.
        residual_target: residual target, same shape, scaled.
        config: resolved :class:`LossConfig`.
        mask: boolean validity mask, same shape.
        packing: channel metadata used to group the bias by variable / level.
        area_weight: ``[1, 1, H, 1]`` cosine-latitude weights, or ``None``.
        lead_index: ``[N]`` rollout-step index used to group the bias by
            forecast lead time. This is bookkeeping only; it never enters the
            network.
        rollout_normalized: deterministic rollout in normalized target space.
        deterministic_estimate: the residual the deterministic inference path
            would produce, required by ``deterministic_weight``. Supplying it
            costs one extra forward pass and is what ties the training objective
            to the quantity the forecast actually uses.
        decode: maps the scaled residual back to normalized target space.
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

    if config.mae_weight > 0.0:
        terms.mae = (error.abs() * weight).sum() / denom

    if config.magnitude_weight > 0.0:
        terms.magnitude = (estimate.pow(2) * weight).sum() / denom

    if config.degradation_weight > 0.0:
        # Zero while the correction is at least as good as no correction at all,
        # positive exactly where it makes a point worse.
        hinge = torch.relu(error.pow(2) - target.pow(2))
        terms.degradation = (hinge * weight).sum() / denom

    if config.deterministic_weight > 0.0:
        if deterministic_estimate is None:
            raise ValueError(
                "refinement.loss.deterministic_weight requires the deterministic "
                "residual estimate; the refiner did not provide one."
            )
        deterministic_error = deterministic_estimate.float() - target
        terms.deterministic = (deterministic_error.pow(2) * weight).sum() / denom

    if config.bias_weight > 0.0:
        groups = _group_ids(
            packing, error.shape[1], config, lead_index, error.shape[0], error.device
        )
        numerator = (error * weight).sum(dim=(-2, -1))
        denominator = weight.sum(dim=(-2, -1))
        group_bias, present = _grouped_mean(numerator, denominator, groups)
        terms.bias = (group_bias[present] ** 2).mean() if bool(present.any()) else error.sum() * 0.0

    if config.gradient_weight > 0.0:
        terms.gradient = _gradient_loss(
            error,
            weight,
            lon_periodic=bool(getattr(packing, "lon_periodic", False)),
        )

    if config.needs_rollout:
        if rollout_normalized is None:
            raise ValueError(
                "refinement.loss terms evaluated on the refined field "
                "(pattern_correlation, extreme, peak, quantile, variance, spectral) "
                "require the deterministic rollout in normalized space."
            )
        to_physical = decode if decode is not None else (lambda x: x)
        # Score the field the forecast will actually emit. The generative
        # estimate is drawn at a random noise level and is far noisier than the
        # deterministic estimate, so its spread/tails/spectrum are not the ones
        # a user sees.
        field_source = (
            deterministic_estimate
            if config.aux_on_deterministic and deterministic_estimate is not None
            else estimate
        )
        base = rollout_normalized.float()
        refined = base + to_physical(field_source.float()).float()
        truth = base + to_physical(target).float()

        if config.pattern_correlation_weight > 0.0:
            terms.pattern_correlation = _pattern_correlation_loss(refined, truth, weight)
        if config.extreme_weight > 0.0:
            terms.extreme = _extreme_loss(
                refined,
                truth,
                weight,
                quantile=config.extreme_quantile,
                intensity=config.extreme_intensity,
            )
        if config.peak_weight > 0.0:
            terms.peak = _peak_loss(refined, truth, weight)
        if config.quantile_weight > 0.0:
            terms.quantile = _quantile_loss(refined, truth, weight)
        if config.variance_weight > 0.0:
            terms.variance = _variance_loss(refined, truth, weight)
        if config.spectral_weight > 0.0:
            terms.spectral = _spectral_loss(refined, truth, weight)
    return terms


def _gradient_loss(
    error: torch.Tensor,
    weight: torch.Tensor,
    *,
    lon_periodic: bool = False,
) -> torch.Tensor:
    """Masked MSE of spatial differences, including the periodic longitude seam."""
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
    if lon_periodic and error.shape[-1] > 1:
        seam_diff = error[..., :1] - error[..., -1:]
        seam_weight = torch.minimum(weight[..., :1], weight[..., -1:])
        total = total + (seam_diff.pow(2) * seam_weight).sum()
        count = count + seam_weight.sum()
    return total / count.clamp(min=1.0)


def _pattern_correlation_loss(
    refined: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``1 - r`` per (sample, channel), averaged over well-conditioned entries."""
    dims = (-2, -1)
    norm = weight.sum(dim=dims).clamp(min=1.0)
    refined_mean = (refined * weight).sum(dim=dims) / norm
    target_mean = (target * weight).sum(dim=dims) / norm
    refined_anom = refined - refined_mean[..., None, None]
    target_anom = target - target_mean[..., None, None]
    # Apply the spatial/area weight once. Multiplying both anomalies by it
    # would square the requested weight in the covariance and variances.
    cov = (weight * refined_anom * target_anom).sum(dim=dims)
    refined_var = (weight * refined_anom.pow(2)).sum(dim=dims)
    target_var = (weight * target_anom.pow(2)).sum(dim=dims)
    # Guard against degenerate (constant) fields, which have no defined
    # correlation: they are excluded rather than contributing a fake 1.0.
    valid = (refined_var > 1e-12) & (target_var > 1e-12)
    if not bool(valid.any()):
        return refined.sum() * 0.0
    corr = cov[valid] / (refined_var[valid].sqrt() * target_var[valid].sqrt())
    return (1.0 - corr.clamp(-1.0, 1.0)).mean()


def _extreme_loss(
    refined: torch.Tensor,
    truth: torch.Tensor,
    weight: torch.Tensor,
    *,
    quantile: float,
    intensity: float,
) -> torch.Tensor:
    """Squared error re-weighted towards the tails of the *observed* field.

    The weight ramps from 1 to ``1 + intensity`` as the truth moves from the
    configured quantile to the field maximum, and symmetrically for the lower
    tail. Anchoring the weights on the truth (not the prediction) keeps the term
    from rewarding a model that simply inflates its own extremes.
    """
    flat_refined = refined.flatten(start_dim=-2).reshape(-1, refined.shape[-2] * refined.shape[-1])
    flat_truth = truth.flatten(start_dim=-2).reshape(-1, truth.shape[-2] * truth.shape[-1])
    flat_weight = weight.flatten(start_dim=-2).reshape(-1, weight.shape[-2] * weight.shape[-1])
    numerator = refined.sum() * 0.0
    denominator = weight.sum() * 0.0
    for refined_map, truth_map, map_weight in zip(flat_refined, flat_truth, flat_weight):
        valid = map_weight > 0
        if not bool(valid.any()):
            continue
        observed = truth_map[valid].float()
        upper = torch.quantile(observed, quantile).to(truth.dtype)
        lower = torch.quantile(observed, 1.0 - quantile).to(truth.dtype)
        maximum = observed.max().to(truth.dtype)
        minimum = observed.min().to(truth.dtype)
        span_up = (maximum - upper).clamp(min=1e-6)
        span_down = (lower - minimum).clamp(min=1e-6)
        excess_up = ((truth_map[valid] - upper) / span_up).clamp(min=0.0, max=1.0)
        excess_down = ((lower - truth_map[valid]) / span_down).clamp(min=0.0, max=1.0)
        tail_weight = 1.0 + intensity * torch.maximum(excess_up, excess_down)
        combined = map_weight[valid] * tail_weight
        numerator = numerator + ((refined_map[valid] - truth_map[valid]).pow(2) * combined).sum()
        denominator = denominator + combined.sum()
    return numerator / denominator.clamp(min=1.0)


def _peak_loss(
    refined: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Match the spatial maximum and minimum of every (sample, channel) map.

    Masked cells are pushed to -inf / +inf before the reduction so they cannot
    become a spurious extremum.
    """
    dims = (-2, -1)
    valid = weight > 0
    if not bool(valid.any()):
        return refined.sum() * 0.0
    neg_inf = torch.finfo(refined.dtype).min
    pos_inf = torch.finfo(refined.dtype).max
    refined_max = torch.where(valid, refined, torch.full_like(refined, neg_inf)).amax(dim=dims)
    truth_max = torch.where(valid, truth, torch.full_like(truth, neg_inf)).amax(dim=dims)
    refined_min = torch.where(valid, refined, torch.full_like(refined, pos_inf)).amin(dim=dims)
    truth_min = torch.where(valid, truth, torch.full_like(truth, pos_inf)).amin(dim=dims)
    present = valid.any(dim=dims)
    if not bool(present.any()):
        return refined.sum() * 0.0
    high = (refined_max[present] - truth_max[present]).pow(2)
    low = (refined_min[present] - truth_min[present]).pow(2)
    return 0.5 * (high.mean() + low.mean())


def _quantile_loss(
    refined: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Sorted-value (1-D Wasserstein-2) distance per (sample, channel).

    Comparing the two fields after sorting removes all spatial information and
    leaves only the value distribution, which is exactly what quantile/PDF
    agreement measures. Masked cells are removed before sorting. Filling them
    with a shared constant is not equivalent: the inserted values change rank
    alignment and dilute the distance even though their pointwise difference is
    zero.
    """
    flat_refined = refined.flatten(start_dim=-2).reshape(-1, refined.shape[-2] * refined.shape[-1])
    flat_truth = truth.flatten(start_dim=-2).reshape(-1, truth.shape[-2] * truth.shape[-1])
    flat_valid = (weight > 0).flatten(start_dim=-2).reshape(
        -1, weight.shape[-2] * weight.shape[-1]
    )
    losses: list[torch.Tensor] = []
    for refined_map, truth_map, valid in zip(flat_refined, flat_truth, flat_valid):
        if not bool(valid.any()):
            continue
        refined_sorted = refined_map[valid].sort().values
        truth_sorted = truth_map[valid].sort().values
        losses.append((refined_sorted - truth_sorted).pow(2).mean())
    return torch.stack(losses).mean() if losses else refined.sum() * 0.0


def _variance_loss(
    refined: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Match the weighted spatial standard deviation of each map."""
    dims = (-2, -1)
    norm = weight.sum(dim=dims).clamp(min=1.0)
    refined_mean = (refined * weight).sum(dim=dims) / norm
    truth_mean = (truth * weight).sum(dim=dims) / norm
    refined_var = (weight * (refined - refined_mean[..., None, None]).pow(2)).sum(dim=dims) / norm
    truth_var = (weight * (truth - truth_mean[..., None, None]).pow(2)).sum(dim=dims) / norm
    return (refined_var.clamp(min=0).sqrt() - truth_var.clamp(min=0).sqrt()).pow(2).mean()


def _spectral_loss(
    refined: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Match the radially averaged log power spectrum of each map.

    Oversmoothing shows up as a deficit at high wavenumbers and excessive noise
    as a surplus, so a log-spectrum match penalises both without forcing any
    particular spatial phase. Masked cells are zero-filled after mean removal,
    which is a windowing choice, not a value substitution.
    """
    dims = (-2, -1)
    norm = weight.sum(dim=dims, keepdim=True).clamp(min=1.0)
    refined_anom = (refined - (refined * weight).sum(dim=dims, keepdim=True) / norm) * weight
    truth_anom = (truth - (truth * weight).sum(dim=dims, keepdim=True) / norm) * weight

    refined_power = torch.fft.rfft2(refined_anom.float(), norm="ortho").abs().pow(2)
    truth_power = torch.fft.rfft2(truth_anom.float(), norm="ortho").abs().pow(2)

    bins = _radial_bins(refined_power.shape[-2], refined_power.shape[-1], refined_power.device)
    num_bins = int(bins.max().item()) + 1
    flat = bins.reshape(-1)
    shape = (*refined_power.shape[:-2], num_bins)

    def _radial(power: torch.Tensor) -> torch.Tensor:
        flat_power = power.reshape(*power.shape[:-2], -1)
        counts = torch.zeros(num_bins, device=power.device, dtype=power.dtype).index_add(
            0, flat, torch.ones_like(flat, dtype=power.dtype)
        )
        summed = torch.zeros(shape, device=power.device, dtype=power.dtype).index_add(
            -1, flat, flat_power
        )
        return summed / counts.clamp(min=1.0)

    eps = 1e-12
    return (
        torch.log(_radial(refined_power) + eps) - torch.log(_radial(truth_power) + eps)
    ).pow(2).mean()


_RADIAL_CACHE: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _radial_bins(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Integer radial wavenumber bin index for an ``rfft2`` output grid."""
    key = (height, width, device)
    cached = _RADIAL_CACHE.get(key)
    if cached is not None:
        return cached
    ky = torch.fft.fftfreq(height, device=device).reshape(-1, 1) * height
    kx = torch.arange(width, device=device, dtype=torch.float32).reshape(1, -1)
    radius = torch.sqrt(ky.float().pow(2) + kx.pow(2)).round().long()
    _RADIAL_CACHE[key] = radius
    return radius
