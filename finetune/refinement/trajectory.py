"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Trajectory-level objectives and scores for joint spatiotemporal refinement.

Marginal scores at individual valid times cannot distinguish a forecast that
evolves correctly from one that is right on average at every lead but wrong
about *when* an ozone episode starts, peaks and decays. Equally, a dependence
score alone cannot detect a biased forecast. This module therefore supplements
-- never replaces -- the marginal CRPS / twCRPS already implemented in
:mod:`finetune.refinement.evaluation`.

Two families are provided.

Objectives (differentiable, used in training)
---------------------------------------------
:func:`tendency_loss` penalizes an incorrect *rate of change* between adjacent
leads. It is applied to the declared **point** product only. It deliberately
does not penalize change itself: the reference tendency appears on both sides
of the difference, so a forecast that reproduces a sharp real event exactly
incurs zero loss while a forecast that smooths it incurs a large one. Nothing
here asks every stochastic member to follow the verifying trajectory.

Scores (non-differentiable, used in evaluation)
-----------------------------------------------
Every event functional is computed **member-wise first** and only then scored as
an ensemble. The maximum of an ensemble mean is not the ensemble distribution of
maxima, and the difference is exactly what matters for extreme ozone episodes.

* :func:`member_window_maximum` / :func:`member_time_weighted_mean` reduce a
  member's trajectory to a scalar event functional per grid cell.
* :func:`ensemble_crps` scores any such functional with the fair estimator.
* :func:`threshold_weighted_crps` applies ``v(x) = max(x, u)`` (upper tail) or
  ``v(x) = min(x, u)`` (lower tail) to members **and** truth alike, and scores
  every case rather than only exceedances.
* :func:`exceedance_brier` scores "does the episode exceed ``u`` anywhere in the
  declared window" as a probability forecast.
* :func:`residual_autocorrelation` and :func:`peak_timing_error` diagnose the
  temporal structure a smooth-looking animation can otherwise hide.

Thresholds
----------
Every threshold argument is a value in **absolute ozone space**, supplied by the
caller from *training-data* statistics. Nothing in this module estimates a
threshold from the data it is scoring.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "ensemble_crps",
    "exceedance_brier",
    "member_time_weighted_mean",
    "member_window_maximum",
    "peak_timing_error",
    "residual_autocorrelation",
    "tendency_loss",
    "threshold_weighted_crps",
]


def _check_trajectory(name: str, value: torch.Tensor, dims: int) -> None:
    if value.ndim != dims:
        raise ValueError(f"{name} must have {dims} dimensions, got {tuple(value.shape)}.")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weight = mask.to(values.dtype)
    return torch.where(mask.to(torch.bool), values, 0.0).sum() / weight.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def tendency_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    delta_hours: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    penalty: str = "huber",
    huber_delta: float = 1.0,
    scale: torch.Tensor | float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Tendency-reconstruction loss on the declared point product.

    .. math::

        L = \operatorname{mean}\; \rho\!\left(
            \frac{(\hat y_j - \hat y_{j-1}) - (y_j - y_{j-1})}{\Delta t_j}
        \right)

    Args:
        prediction: ``[B, S, C, H, W]`` point forecast trajectory.
        reference: ``[B, S, C, H, W]`` verifying trajectory, same units.
        delta_hours: ``[B, S]`` actual hours between adjacent frames. Element
            ``j`` is the gap **ending** at ``j``; element ``0`` is unused
            because there is no preceding frame. Using the real elapsed time
            rather than a step index is what keeps an irregular or gappy cadence
            honest.
        mask: ``[B, S, C, H, W]`` validity. A tendency pair is scored only when
            *both* of its frames are valid.
        penalty: ``huber`` (robust, default), ``l2`` or ``l1``.
        huber_delta: transition point of the Huber penalty.
        scale: per-channel normalization applied to the tendency difference so
            mixed units (``kg kg-1`` profiles and a ``kg m-2`` column) can be
            combined without the largest-magnitude field dominating. Must come
            from training data. Broadcast against ``[C]`` or a scalar.
        weight: optional ``[B, S, C, H, W]`` weights, for example cosine
            latitude.

    Returns:
        Scalar loss. ``0`` when no adjacent valid pair exists.
    """
    _check_trajectory("prediction", prediction, 5)
    _check_trajectory("reference", reference, 5)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and reference "
            f"{tuple(reference.shape)} must match."
        )
    batch, steps = prediction.shape[0], prediction.shape[1]
    pair_mask = None
    if mask is not None:
        if mask.shape != prediction.shape:
            raise ValueError(
                f"mask must match the trajectory shape {tuple(prediction.shape)}."
            )
        valid = mask.to(device=prediction.device, dtype=torch.bool)
        pair_mask = valid[:, 1:] & valid[:, :-1]
        # Mask before differencing and squaring: NaN * 0 in the loss, or in
        # its backward pass, cannot remove an invalid verifying observation.
        prediction = torch.where(valid, prediction, 0.0)
        reference = torch.where(valid, reference, 0.0)
    if steps < 2:
        return prediction[:, :0].sum()
    if delta_hours.shape != (batch, steps):
        raise ValueError(
            f"delta_hours must be [{batch}, {steps}], got {tuple(delta_hours.shape)}."
        )
    gaps = delta_hours[:, 1:].to(device=prediction.device, dtype=prediction.dtype)
    if not bool(torch.isfinite(gaps).all()) or bool((gaps <= 0).any()):
        raise ValueError(
            "delta_hours must be finite and strictly positive for every adjacent "
            "pair; a non-positive gap would mean the trajectory is not ordered."
        )

    predicted_tendency = (prediction[:, 1:] - prediction[:, :-1]) / gaps[:, :, None, None, None]
    reference_tendency = (reference[:, 1:] - reference[:, :-1]) / gaps[:, :, None, None, None]

    scale_tensor = torch.as_tensor(scale, dtype=prediction.dtype, device=prediction.device)
    if scale_tensor.ndim == 1:
        scale_tensor = scale_tensor.view(1, 1, -1, 1, 1)
    if not bool(torch.isfinite(scale_tensor).all()) or bool((scale_tensor <= 0).any()):
        raise ValueError("tendency_loss scale must be finite and strictly positive.")
    difference = (predicted_tendency - reference_tendency) / scale_tensor

    kind = str(penalty).lower()
    if kind == "huber":
        delta = float(huber_delta)
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta!r}.")
        absolute = difference.abs()
        elementwise = torch.where(
            absolute <= delta,
            0.5 * difference.pow(2),
            delta * (absolute - 0.5 * delta),
        )
    elif kind == "l2":
        elementwise = difference.pow(2)
    elif kind == "l1":
        elementwise = difference.abs()
    else:
        raise ValueError(f"Unsupported tendency penalty {penalty!r}; use huber, l2 or l1.")

    if weight is not None:
        if weight.shape != prediction.shape:
            raise ValueError("weight must match the trajectory shape.")
        pair_weight = weight[:, 1:].to(device=elementwise.device, dtype=elementwise.dtype)
        if pair_mask is not None:
            pair_weight = torch.where(pair_mask, pair_weight, 0.0)
        elementwise = elementwise * pair_weight
    return _masked_mean(elementwise, pair_mask)


# ---------------------------------------------------------------------------
# Member-wise event functionals
# ---------------------------------------------------------------------------


def member_window_maximum(
    members: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    window: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Per-member maximum over the lead axis of ``[B, M, S, C, H, W]``.

    Computing the maximum for every member separately, before any ensemble
    reduction, is what makes the result an ensemble *distribution of maxima*.

    ``window`` selects a half-open ``[start, stop)`` slice of the lead axis, the
    "declared window" of the event definition.
    """
    _check_trajectory("members", members, 6)
    selected = members
    selected_mask = mask
    if window is not None:
        start, stop = int(window[0]), int(window[1])
        if not 0 <= start < stop <= members.shape[2]:
            raise ValueError(
                f"window {window!r} is not a valid [start, stop) slice of "
                f"{members.shape[2]} leads."
            )
        selected = members[:, :, start:stop]
        selected_mask = None if mask is None else mask[:, :, start:stop]
    if selected_mask is not None:
        neutral = torch.finfo(selected.dtype).min
        selected = torch.where(selected_mask.to(torch.bool), selected, neutral)
    return selected.amax(dim=2)


def member_time_weighted_mean(
    members: torch.Tensor,
    delta_hours: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-member time-weighted mean over the lead axis of ``[B, M, S, C, H, W]``.

    Weighting by the actual hours each frame represents keeps the functional
    meaningful when the cadence is irregular. ``delta_hours`` is ``[B, S]``.
    """
    _check_trajectory("members", members, 6)
    batch, _, steps = members.shape[0], members.shape[1], members.shape[2]
    if delta_hours.shape != (batch, steps):
        raise ValueError(
            f"delta_hours must be [{batch}, {steps}], got {tuple(delta_hours.shape)}."
        )
    weights = delta_hours.to(members.dtype)[:, None, :, None, None, None]
    if mask is not None:
        weights = weights * mask.to(members.dtype)
    total = weights.sum(dim=2).clamp(min=torch.finfo(members.dtype).tiny)
    return (members * weights).sum(dim=2) / total


# ---------------------------------------------------------------------------
# Ensemble scores
# ---------------------------------------------------------------------------


def ensemble_crps(
    members: torch.Tensor,
    truth: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    estimator: str = "fair",
    member_dim: int = 1,
) -> torch.Tensor:
    r"""CRPS of an ensemble against truth, reduced over all remaining axes.

    ``empirical``:

    .. math:: \frac1M\sum_m |x_m - y| - \frac1{2M^2}\sum_{m,n}|x_m - x_n|

    ``fair`` (unbiased, needs ``M >= 2``):

    .. math:: \frac1M\sum_m |x_m - y| - \frac1{2M(M-1)}\sum_{m\ne n}|x_m - x_n|

    ``members`` may be any shape with an ensemble axis at ``member_dim``;
    ``truth`` has the same shape without that axis. Passing an already-reduced
    event functional (a window maximum, say) is the intended use.

    With ``M = 1`` the empirical estimator equals the MAE, which says nothing
    about calibration; the fair estimator is therefore rejected in that case
    rather than silently degenerating.
    """
    if members.ndim != truth.ndim + 1:
        raise ValueError(
            f"members {tuple(members.shape)} must have exactly one more axis than "
            f"truth {tuple(truth.shape)}."
        )
    count = int(members.shape[member_dim])
    kind = str(estimator).lower()
    if kind not in {"fair", "empirical"}:
        raise ValueError(f"Unsupported CRPS estimator {estimator!r}; use fair or empirical.")
    if kind == "fair" and count < 2:
        raise ValueError(
            "The fair CRPS estimator requires at least two independently sampled "
            f"members, got {count}. Use estimator='empirical' (which equals the MAE "
            "for one member) and do not describe it as a calibration measure."
        )

    moved = members.movedim(member_dim, 0)
    expanded_truth = truth.unsqueeze(0)
    absolute = (moved - expanded_truth).abs().mean(dim=0)
    pairwise = (moved.unsqueeze(0) - moved.unsqueeze(1)).abs().sum(dim=(0, 1))
    if kind == "fair":
        spread = pairwise / (2.0 * count * (count - 1))
    else:
        spread = pairwise / (2.0 * count * count)
    return _masked_mean(absolute - spread, mask)


def threshold_weighted_crps(
    members: torch.Tensor,
    truth: torch.Tensor,
    threshold: torch.Tensor | float,
    *,
    tail: str = "upper",
    mask: torch.Tensor | None = None,
    estimator: str = "fair",
    member_dim: int = 1,
) -> torch.Tensor:
    """Threshold-weighted CRPS with ``v(x) = max(x, u)`` or ``min(x, u)``.

    The same chaining transformation is applied to members *and* truth, and
    **every** case is scored -- including cases where the observation never
    exceeds ``u``. Restricting the score to observed exceedances would not be a
    proper scoring rule.

    ``threshold`` must come from training data, in absolute ozone units, and is
    broadcast against the scored shape.

    ``tail='lower'`` is the counterpart relevant to total-column ozone, where
    anomalously *low* values are the event of interest.
    """
    direction = str(tail).lower()
    if direction not in {"upper", "lower"}:
        raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}.")
    u = torch.as_tensor(threshold, dtype=members.dtype, device=members.device)
    if direction == "upper":
        chained_members = torch.maximum(members, u)
        chained_truth = torch.maximum(truth, u)
    else:
        chained_members = torch.minimum(members, u)
        chained_truth = torch.minimum(truth, u)
    return ensemble_crps(
        chained_members,
        chained_truth,
        mask=mask,
        estimator=estimator,
        member_dim=member_dim,
    )


def exceedance_brier(
    members: torch.Tensor,
    truth: torch.Tensor,
    threshold: torch.Tensor | float,
    *,
    mask: torch.Tensor | None = None,
    member_dim: int = 1,
    return_reliability: bool = False,
    num_bins: int = 10,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Brier score for "exceeds ``threshold``", with optional reliability bins.

    The forecast probability is the fraction of members whose (already reduced)
    event functional exceeds the threshold, so calling this with
    :func:`member_window_maximum` scores "exceedance anywhere in the window".

    Reliability is returned as per-bin mean forecast probability, observed
    frequency and count, which is the input a reliability diagram needs.
    """
    u = torch.as_tensor(threshold, dtype=members.dtype, device=members.device)
    probability = (members > u).to(torch.float32).mean(dim=member_dim)
    observed = (truth > u).to(torch.float32)
    brier = _masked_mean((probability - observed).pow(2), mask)
    if not return_reliability:
        return brier

    bins = int(num_bins)
    if bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins!r}.")
    flat_probability = probability.reshape(-1)
    flat_observed = observed.reshape(-1)
    if mask is not None:
        keep = mask.reshape(-1).to(torch.bool)
        flat_probability = flat_probability[keep]
        flat_observed = flat_observed[keep]
    index = torch.clamp((flat_probability * bins).long(), max=bins - 1)
    counts = torch.zeros(bins, dtype=torch.float64)
    forecast_sum = torch.zeros(bins, dtype=torch.float64)
    observed_sum = torch.zeros(bins, dtype=torch.float64)
    counts.index_add_(0, index.cpu(), torch.ones_like(flat_probability, dtype=torch.float64).cpu())
    forecast_sum.index_add_(0, index.cpu(), flat_probability.to(torch.float64).cpu())
    observed_sum.index_add_(0, index.cpu(), flat_observed.to(torch.float64).cpu())
    safe = counts.clamp(min=1.0)
    return brier, {
        "count": counts,
        "forecast_probability": forecast_sum / safe,
        "observed_frequency": observed_sum / safe,
    }


# ---------------------------------------------------------------------------
# Temporal-structure diagnostics
# ---------------------------------------------------------------------------


def residual_autocorrelation(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    lag: int = 1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lag-``lag`` autocorrelation of the forecast error along the lead axis.

    A refinement that removes the instantaneous bias but leaves strongly
    autocorrelated errors has not learned the evolution; a value near zero means
    the remaining error behaves like noise rather than persistent drift.
    Inputs are ``[B, S, C, H, W]``.
    """
    _check_trajectory("prediction", prediction, 5)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have the same shape.")
    step = int(lag)
    if step < 1 or step >= prediction.shape[1]:
        raise ValueError(
            f"lag must be in [1, {prediction.shape[1] - 1}], got {lag!r}."
        )
    error = (prediction - reference).float()
    if mask is not None:
        valid = mask.to(torch.bool)
        error = torch.where(valid, error, torch.zeros_like(error))
        pair_mask = valid[:, step:] & valid[:, :-step]
    else:
        pair_mask = None

    left, right = error[:, step:], error[:, :-step]
    if pair_mask is not None:
        weight = pair_mask.to(error.dtype)
        total = weight.sum().clamp(min=1.0)
        left_mean = (left * weight).sum() / total
        right_mean = (right * weight).sum() / total
        covariance = ((left - left_mean) * (right - right_mean) * weight).sum() / total
        left_var = ((left - left_mean).pow(2) * weight).sum() / total
        right_var = ((right - right_mean).pow(2) * weight).sum() / total
    else:
        left_mean, right_mean = left.mean(), right.mean()
        covariance = ((left - left_mean) * (right - right_mean)).mean()
        left_var = (left - left_mean).pow(2).mean()
        right_var = (right - right_mean).pow(2).mean()
    denominator = (left_var * right_var).sqrt().clamp(min=torch.finfo(error.dtype).tiny)
    return covariance / denominator


def peak_timing_error(
    members: torch.Tensor,
    truth: torch.Tensor,
    lead_hours: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    truth_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error, in hours, of each member's sampled peak time.

    ``members`` is ``[B, M, S, C, H, W]`` and ``truth`` is ``[B, S, C, H, W]``.
    The peak lead is located per member and compared with the observed peak
    lead, so this reports the timing skill of *realizations* rather than of a
    smoothed ensemble mean.

    Args:
        mask: member-shaped ``[B, M, S, C, H, W]`` validity for ``members``.
        truth_mask: truth-shaped ``[B, S, C, H, W]`` validity for ``truth``. It
            is a **separate** argument on purpose: inferring the truth mask by
            reducing ``mask`` over an axis chosen by comparing ``M`` with ``S``
            silently takes the peak over the member axis whenever an ensemble
            happens to be the same size as the rollout.

    With a 12-hour verification cadence the result is quantized to 12 hours and
    must not be presented as hourly peak-timing skill.
    """
    _check_trajectory("members", members, 6)
    _check_trajectory("truth", truth, 5)
    batch, _, steps = members.shape[0], members.shape[1], members.shape[2]
    if lead_hours.shape != (batch, steps):
        raise ValueError(
            f"lead_hours must be [{batch}, {steps}], got {tuple(lead_hours.shape)}."
        )
    neutral = torch.finfo(members.dtype).min
    member_values = members
    if mask is not None:
        if mask.shape != members.shape:
            raise ValueError(
                f"mask must match the member shape {tuple(members.shape)}, got "
                f"{tuple(mask.shape)}."
            )
        member_values = torch.where(mask.to(torch.bool), members, neutral)
    truth_values = truth
    if truth_mask is not None:
        if truth_mask.shape != truth.shape:
            raise ValueError(
                f"truth_mask must match the truth shape {tuple(truth.shape)}, got "
                f"{tuple(truth_mask.shape)}."
            )
        truth_values = torch.where(truth_mask.to(torch.bool), truth, neutral)

    member_peak = member_values.argmax(dim=2)
    truth_peak = truth_values.argmax(dim=1)
    leads = lead_hours.to(torch.float32)

    def _gather(index: torch.Tensor) -> torch.Tensor:
        flat = index.reshape(index.shape[0], -1)
        picked = torch.gather(leads, 1, flat.clamp(min=0, max=steps - 1))
        return picked.reshape(index.shape)

    return (_gather(member_peak) - _gather(truth_peak).unsqueeze(1)).abs().mean()
