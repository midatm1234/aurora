"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Evaluation metrics for raw-versus-refined Aurora rollouts.

Metrics are always computed **separately** by variable, pressure level and
forecast lead time (and, when a region mask is supplied, by region), and only
then aggregated. Averaging unrelated variables, levels or lead times together
would hide degradation, so the per-group table is the primary output and the
aggregate is derived from it.

Supported scores
----------------
``bias``                    mean error (refined - truth)
``mae``                     mean absolute error
``rmse``                    root-mean-square error
``pattern_correlation``     centred spatial (anomaly) correlation
``ensemble_mean_bias``      bias of the ensemble mean
``ensemble_spread``         mean unbiased ensemble standard deviation
``spread_skill_ratio``      ensemble spread / ensemble-mean RMSE
``crps``                    fair (unbiased) ensemble CRPS

Additional deterministic diagnostics cover centred error, anomaly
correlation, spatial standard deviation, quantile/tail/exceedance calibration,
empirical 1-D Wasserstein distance and distribution skewness.

All spatial moment/error metrics optionally use cosine-latitude area weighting.
Masked or non-finite cells are excluded rather than being filled.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from finetune.refinement.losses import area_weights_from_latitudes
from finetune.refinement.packing import FieldPacking

__all__ = ["compare_raw_and_refined", "evaluate_packed", "summarize"]

_EPS = 1e-12
_QUANTILES = (0.90, 0.95, 0.99)


def _weights(
    reference: torch.Tensor,
    packing: FieldPacking,
    mask: torch.Tensor | None,
    area_weighted: bool,
    region_mask: torch.Tensor | None,
) -> torch.Tensor:
    weight = torch.ones_like(reference)
    if mask is not None:
        weight = weight * mask.to(reference.dtype)
    if region_mask is not None:
        weight = weight * region_mask.to(device=reference.device, dtype=reference.dtype)
    if area_weighted:
        area = area_weights_from_latitudes(
            packing.lat, reference.shape[-2], device=reference.device, dtype=reference.dtype
        )
        if area is not None:
            weight = weight * area
    return torch.where(
        torch.isfinite(weight) & (weight > 0), weight, torch.zeros_like(weight)
    )


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # IEEE-754 defines NaN * 0 as NaN, so masking only through zero weights is
    # insufficient. Sanitize excluded cells before multiplication and report
    # NaN (not a misleading zero) when a slice contains no valid cells.
    safe_values = torch.where(weight > 0, values, torch.zeros_like(values))
    denominator = weight.sum(dim=(-2, -1))
    result = (safe_values * weight).sum(dim=(-2, -1)) / denominator.clamp(min=_EPS)
    return torch.where(
        denominator > 0,
        result,
        torch.full_like(result, float("nan")),
    )


def _scalar_weighted_moments(
    values: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return population mean/variance for an already selected group.

    Computation is in float64 so trace-gas fields around 1e-9 retain useful
    variance. No absolute variance floor is used: physical scale must not decide
    whether a spatial pattern is considered constant.
    """
    values = values.double().reshape(-1)
    weight = weight.double().reshape(-1)
    selected = torch.isfinite(values) & torch.isfinite(weight) & (weight > 0)
    values = values[selected]
    weight = weight[selected]
    nan = torch.full((), float("nan"), dtype=torch.float64, device=values.device)
    if values.numel() == 0:
        return nan, nan
    total = weight.sum()
    if not bool(total > 0):
        return nan, nan
    mean = (values * weight).sum() / total
    variance = ((values - mean).pow(2) * weight).sum() / total
    return mean, variance.clamp(min=0.0)


def _group_scientific_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    """Distribution and amplitude diagnostics for one variable/lead group.

    Spatial moments, tail errors and exceedance frequencies honour the supplied
    (mask/region/area) weights. Quantiles and empirical Wasserstein distance use
    the valid grid-cell sample directly, matching the benchmark evaluator's
    established quantile convention.
    """
    pred = prediction.double().reshape(-1)
    obs = truth.double().reshape(-1)
    flat_weight = weight.double().reshape(-1)
    selected = (
        torch.isfinite(pred)
        & torch.isfinite(obs)
        & torch.isfinite(flat_weight)
        & (flat_weight > 0)
    )
    pred = pred[selected]
    obs = obs[selected]
    flat_weight = flat_weight[selected]

    metric_names = (
        "centered_rmse",
        "prediction_std",
        "truth_std",
        "std_ratio",
        "prediction_skewness",
        "truth_skewness",
        "wasserstein_distance",
        *(f"prediction_p{int(q * 100)}" for q in _QUANTILES),
        *(f"truth_p{int(q * 100)}" for q in _QUANTILES),
        *(f"p{int(q * 100)}_bias" for q in _QUANTILES),
        *(f"p{int(q * 100)}_error" for q in _QUANTILES),
        "p95_tail_mae",
        "p99_tail_mae",
        *(f"p{int(q * 100)}_prediction_exceedance_frequency" for q in _QUANTILES),
        *(f"p{int(q * 100)}_truth_exceedance_frequency" for q in _QUANTILES),
        *(f"p{int(q * 100)}_exceedance_frequency_bias" for q in _QUANTILES),
        *(f"p{int(q * 100)}_exceedance_frequency_ratio" for q in _QUANTILES),
    )
    if pred.numel() == 0:
        return {name: float("nan") for name in metric_names}

    _, error_variance = _scalar_weighted_moments(pred - obs, flat_weight)
    pred_mean, pred_variance = _scalar_weighted_moments(pred, flat_weight)
    obs_mean, obs_variance = _scalar_weighted_moments(obs, flat_weight)
    pred_std = pred_variance.sqrt()
    obs_std = obs_variance.sqrt()
    std_ratio = (
        pred_std / obs_std
        if bool(torch.isfinite(obs_std) & (obs_std > 0))
        else torch.full_like(obs_std, float("nan"))
    )

    def _skewness(
        values: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor
    ) -> float:
        if values.numel() < 3 or not bool(
            torch.isfinite(variance) & (variance > 0)
        ):
            return float("nan")
        third_moment = (
            (values - mean).pow(3) * flat_weight
        ).sum() / flat_weight.sum()
        return float(third_moment / variance.pow(1.5))

    quantiles = torch.tensor(_QUANTILES, dtype=torch.float64, device=pred.device)
    pred_quantiles = torch.quantile(pred, quantiles)
    obs_quantiles = torch.quantile(obs, quantiles)
    result = {
        "centered_rmse": float(error_variance.sqrt()),
        "prediction_std": float(pred_std),
        "truth_std": float(obs_std),
        "std_ratio": float(std_ratio),
        "prediction_skewness": _skewness(pred, pred_mean, pred_variance),
        "truth_skewness": _skewness(obs, obs_mean, obs_variance),
        # Both empirical distributions have the same number of values because
        # validity is a joint prediction/truth mask. Sorted pairing is therefore
        # the exact equal-mass 1-D Wasserstein-1 distance.
        "wasserstein_distance": float(
            (pred.sort().values - obs.sort().values).abs().mean()
        ),
    }

    total_weight = flat_weight.sum()
    for index, q in enumerate(_QUANTILES):
        percentile = int(q * 100)
        pred_quantile = pred_quantiles[index]
        obs_quantile = obs_quantiles[index]
        quantile_bias = pred_quantile - obs_quantile
        result[f"prediction_p{percentile}"] = float(pred_quantile)
        result[f"truth_p{percentile}"] = float(obs_quantile)
        result[f"p{percentile}_bias"] = float(quantile_bias)
        result[f"p{percentile}_error"] = float(quantile_bias.abs())

        pred_exceeds = pred >= obs_quantile
        obs_exceeds = obs >= obs_quantile
        pred_frequency = (flat_weight * pred_exceeds).sum() / total_weight
        obs_frequency = (flat_weight * obs_exceeds).sum() / total_weight
        result[f"p{percentile}_prediction_exceedance_frequency"] = float(
            pred_frequency
        )
        result[f"p{percentile}_truth_exceedance_frequency"] = float(obs_frequency)
        result[f"p{percentile}_exceedance_frequency_bias"] = float(
            pred_frequency - obs_frequency
        )
        result[f"p{percentile}_exceedance_frequency_ratio"] = (
            float(pred_frequency / obs_frequency)
            if bool(obs_frequency > 0)
            else float("nan")
        )

        if percentile in (95, 99):
            tail_weight = flat_weight * obs_exceeds
            tail_total = tail_weight.sum()
            result[f"p{percentile}_tail_mae"] = (
                float(((pred - obs).abs() * tail_weight).sum() / tail_total)
                if bool(tail_total > 0)
                else float("nan")
            )
    return result


def _require_packed_shape(name: str, value: torch.Tensor, packing: FieldPacking) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must be [N, C, H, W], got {tuple(value.shape)}.")
    if value.shape[1] != packing.num_channels:
        raise ValueError(
            f"{name} has {value.shape[1]} channels but packing declares "
            f"{packing.num_channels}."
        )
    if packing.lat and len(packing.lat) != value.shape[-2]:
        raise ValueError(
            f"{name} height {value.shape[-2]} does not match the packing latitude "
            f"coordinate ({len(packing.lat)})."
        )
    if packing.lon and len(packing.lon) != value.shape[-1]:
        raise ValueError(
            f"{name} width {value.shape[-1]} does not match the packing longitude "
            f"coordinate ({len(packing.lon)})."
        )


def _require_sample_metadata(name: str, value: torch.Tensor | None, batch: int) -> None:
    if value is not None and value.numel() != batch:
        raise ValueError(
            f"{name} must contain one value per sample ({batch}), got "
            f"shape {tuple(value.shape)}."
        )


def evaluate_packed(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    packing: FieldPacking,
    lead_index: torch.Tensor | None = None,
    lead_hours: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    members: torch.Tensor | None = None,
    area_weighted: bool = True,
    region_mask: torch.Tensor | None = None,
    region_name: str = "all",
) -> list[dict[str, Any]]:
    """Per (variable, level, lead time) metrics for one packed comparison.

    Args:
        prediction: ``[N, C, H, W]`` physical-unit deterministic or refined field.
        truth: ``[N, C, H, W]`` physical-unit ground truth.
        packing: channel layout providing the variable/level metadata.
        lead_index: ``[N]`` rollout-step index (0-based) used to group results.
        lead_hours: ``[N]`` physical forecast lead time in hours, reported as-is.
        mask: ``[N, C, H, W]`` boolean validity mask.
        members: optional ``[N, M, C, H, W]`` ensemble members, in draw order.
        area_weighted: apply cosine-latitude weighting.
        region_mask: optional broadcastable spatial region mask.
        region_name: label recorded with every row.

    Returns:
        One row per ``(variable, level, lead time)`` group.
    """
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and truth {tuple(truth.shape)} "
            "must have the same shape."
        )
    if prediction.device != truth.device:
        raise ValueError(
            f"prediction is on {prediction.device} but truth is on {truth.device}."
        )
    _require_packed_shape("prediction", prediction, packing)
    _require_packed_shape("truth", truth, packing)
    batch = prediction.shape[0]
    _require_sample_metadata("lead_index", lead_index, batch)
    _require_sample_metadata("lead_hours", lead_hours, batch)
    if mask is not None and mask.shape != prediction.shape:
        raise ValueError(
            f"mask must exactly match prediction shape {tuple(prediction.shape)}, got "
            f"{tuple(mask.shape)}."
        )
    if members is not None:
        expected_members = (
            prediction.shape[0],
            prediction.shape[1],
            prediction.shape[2],
            prediction.shape[3],
        )
        if members.ndim != 5:
            raise ValueError(
                "members must be [N, M, C, H, W], got "
                f"{tuple(members.shape)}."
            )
        observed_members = (
            members.shape[0],
            members.shape[2],
            members.shape[3],
            members.shape[4],
        )
        if observed_members != expected_members or members.shape[1] < 1:
            raise ValueError(
                "members must have non-member shape "
                f"{expected_members} and at least one member, got {tuple(members.shape)}."
            )
    prediction = prediction.float()
    truth = truth.float()
    finite = torch.isfinite(prediction) & torch.isfinite(truth)
    valid = (
        finite
        if mask is None
        else (
            finite
            & mask.to(device=prediction.device, dtype=torch.bool)
        )
    )
    weight = _weights(prediction, packing, valid, area_weighted, region_mask)

    error = torch.where(valid, prediction - truth, torch.zeros_like(prediction))
    bias = _weighted_mean(error, weight)
    mae = _weighted_mean(error.abs(), weight)
    mse = _weighted_mean(error.pow(2), weight)

    pred_mean = _weighted_mean(prediction, weight)[..., None, None]
    truth_mean = _weighted_mean(truth, weight)[..., None, None]
    pred_anom = torch.where(valid, prediction - pred_mean, torch.zeros_like(prediction))
    truth_anom = torch.where(valid, truth - truth_mean, torch.zeros_like(truth))
    covariance = (pred_anom * truth_anom * weight).sum(dim=(-2, -1))
    pred_var = (pred_anom.pow(2) * weight).sum(dim=(-2, -1))
    truth_var = (truth_anom.pow(2) * weight).sum(dim=(-2, -1))
    correlation_denominator = pred_var.sqrt() * truth_var.sqrt()
    correlation = covariance / torch.where(
        correlation_denominator > 0,
        correlation_denominator,
        torch.ones_like(correlation_denominator),
    )
    correlation = torch.where(
        (pred_var > 0) & (truth_var > 0),
        correlation,
        torch.full_like(correlation, float("nan")),
    )

    spread = crps = None
    if members is not None:
        member_stack = members.to(device=prediction.device, dtype=torch.float32)
        count = member_stack.shape[1]
        member_finite = torch.isfinite(member_stack).all(dim=1)
        ensemble_valid = valid & member_finite
        ensemble_weight = _weights(
            prediction, packing, ensemble_valid, area_weighted, region_mask
        )
        safe_members = torch.where(
            torch.isfinite(member_stack), member_stack, torch.zeros_like(member_stack)
        )
        member_mean = safe_members.mean(dim=1)
        if count > 1:
            variance = safe_members.var(dim=1, unbiased=True)
            spread = _weighted_mean(
                variance.clamp(min=0.0).sqrt(), ensemble_weight
            )
            # Fair (unbiased) ensemble CRPS:
            #   E|X - y| - 1/(2 M (M-1)) * sum_{i,j} |X_i - X_j|
            absolute = (safe_members - truth.unsqueeze(1)).abs().mean(dim=1)
            pairwise = (
                safe_members.unsqueeze(1) - safe_members.unsqueeze(2)
            ).abs().sum(dim=(1, 2))
            crps_field = absolute - pairwise / (2.0 * count * (count - 1))
            crps = _weighted_mean(
                torch.where(
                    ensemble_valid, crps_field, torch.zeros_like(crps_field)
                ),
                ensemble_weight,
            )
        ensemble_error = torch.where(
            ensemble_valid, member_mean - truth, torch.zeros_like(member_mean)
        )
        ensemble_bias = _weighted_mean(ensemble_error, ensemble_weight)
        ensemble_mse = _weighted_mean(ensemble_error.pow(2), ensemble_weight)
    else:
        ensemble_bias = ensemble_mse = None

    if lead_index is None:
        groups = torch.zeros(batch, dtype=torch.long)
    else:
        groups = lead_index.reshape(-1).cpu().long()

    rows: list[dict[str, Any]] = []
    for lead in sorted(set(groups.tolist())):
        selection = (groups == lead).nonzero(as_tuple=True)[0]
        hours = None
        if lead_hours is not None:
            selected_hours = lead_hours.reshape(-1)[selection].float()
            if not torch.isfinite(selected_hours).all():
                raise ValueError(f"lead_hours contains a non-finite value for lead group {lead}.")
            if not torch.allclose(
                selected_hours,
                selected_hours[:1].expand_as(selected_hours),
                atol=1.0e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"lead_index group {lead} mixes physical lead times: "
                    f"{selected_hours.tolist()}."
                )
            hours = float(selected_hours[0])
        for spec in packing.channels:
            column = spec.index
            pattern_correlation = float(
                torch.nanmean(correlation[selection, column])
            )
            scientific_metrics = _group_scientific_metrics(
                prediction[selection, column],
                truth[selection, column],
                weight[selection, column],
            )
            row: dict[str, Any] = {
                "variable": spec.aurora_name,
                "dataset_name": spec.dataset_name,
                "kind": spec.kind,
                "level": spec.level,
                "units": spec.units,
                "rollout_step": int(lead) + 1,
                "lead_time_hours": hours,
                "region": region_name,
                "area_weighted": bool(area_weighted),
                "samples": int(selection.numel()),
                "bias": float(bias[selection, column].mean()),
                "mae": float(mae[selection, column].mean()),
                "rmse": float(mse[selection, column].mean().sqrt()),
                "pattern_correlation": pattern_correlation,
                # Explicit meteorological name while preserving the established
                # key consumed by existing summaries and notebooks.
                "anomaly_correlation": pattern_correlation,
                **scientific_metrics,
            }
            if ensemble_bias is not None:
                row["ensemble_mean_bias"] = float(ensemble_bias[selection, column].mean())
                row["ensemble_mean_rmse"] = float(ensemble_mse[selection, column].mean().sqrt())
            if spread is not None:
                row["ensemble_spread"] = float(spread[selection, column].mean())
                row["spread_skill_ratio"] = float(
                    spread[selection, column].mean()
                    / max(float(ensemble_mse[selection, column].mean().sqrt()), _EPS)
                )
            if crps is not None:
                row["crps"] = float(crps[selection, column].mean())
            rows.append(row)
    return rows


def compare_raw_and_refined(
    truth: torch.Tensor,
    *,
    packing: FieldPacking,
    candidates: Mapping[str, torch.Tensor],
    ensembles: Mapping[str, torch.Tensor] | None = None,
    lead_index: torch.Tensor | None = None,
    lead_hours: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    area_weighted: bool = True,
    regions: Mapping[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate several candidates (raw Aurora and each refiner) side by side.

    ``candidates`` maps a label (e.g. ``"raw"``, ``"flow_matching_unet"``) to a
    packed physical field; ``ensembles`` optionally maps the same labels to
    ``[N, M, C, H, W]`` member stacks.
    """
    ensembles = dict(ensembles or {})
    region_items: Iterable[tuple[str, torch.Tensor | None]] = (
        list(regions.items()) if regions else [("all", None)]
    )
    rows: list[dict[str, Any]] = []
    for label, prediction in candidates.items():
        for region_name, region_mask in region_items:
            for row in evaluate_packed(
                prediction,
                truth,
                packing=packing,
                lead_index=lead_index,
                lead_hours=lead_hours,
                mask=mask,
                members=ensembles.get(label),
                area_weighted=area_weighted,
                region_mask=region_mask,
                region_name=region_name,
            ):
                rows.append({"model": label, **row})
    return rows


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "raw",
    metrics: Sequence[str] = ("bias", "mae", "rmse", "pattern_correlation"),
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Aggregate per-group rows and flag tradeoffs against a baseline model.

    A model is only reported as an improvement when the absolute bias decreases
    **and** no other reported metric degrades by more than ``tolerance``
    (relative). Every degraded group is listed explicitly so a bias reduction
    can never hide an RMSE, MAE or correlation regression.

    Correlations are higher-is-better; signed bias diagnostics target zero;
    spread/exceedance ratios target one; prediction standard deviation and
    skewness target their truth counterparts; error/distance metrics are
    lower-is-better. metric_improvements reports mean baseline-minus-candidate
    error (or candidate-minus-baseline correlation), so positive is always
    better. Existing default metrics and summary fields are unchanged.
    """
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    if baseline not in by_model:
        raise KeyError(f"Baseline model {baseline!r} is not present in the rows.")

    def _key(row: Mapping[str, Any]) -> tuple:
        return (row["variable"], row["level"], row["rollout_step"], row["region"])

    def _is_correlation(metric: str) -> bool:
        return metric in {"pattern_correlation", "anomaly_correlation"}

    def _is_zero_target(metric: str) -> bool:
        return metric == "bias" or metric.endswith("_bias")

    def _paired_truth_metric(metric: str) -> str | None:
        if metric == "prediction_std":
            return "truth_std"
        if metric == "prediction_skewness":
            return "truth_skewness"
        if metric.startswith("prediction_p"):
            return metric.replace("prediction_", "truth_", 1)
        return None

    def _comparison_cost(metric: str, row: Mapping[str, Any]) -> float:
        value = float(row[metric])
        if _is_correlation(metric):
            return -value
        if _is_zero_target(metric):
            return abs(value)
        if metric.endswith("_ratio"):
            return abs(value - 1.0)
        truth_metric = _paired_truth_metric(metric)
        if truth_metric is not None and truth_metric in row:
            return abs(value - float(row[truth_metric]))
        return value

    def _cost_tolerance(metric: str, before_cost: float) -> float:
        # Dimensionless diagnostics need an absolute tolerance near their ideal
        # value (zero cost); otherwise harmless floating-point skew/ratio noise
        # would be classified as a scientific degradation.
        dimensionless = (
            metric.endswith("_ratio")
            or "skewness" in metric
            or "exceedance_frequency" in metric
        )
        scale = max(abs(before_cost), 1.0 if dimensionless else _EPS)
        return tolerance * scale

    baseline_rows = {_key(row): row for row in by_model[baseline]}
    summary: dict[str, Any] = {"baseline": baseline, "models": {}}
    for model, model_rows in by_model.items():
        aggregate = {
            metric: float(
                sum(
                    abs(r[metric]) if _is_zero_target(metric) else r[metric]
                    for r in model_rows
                )
                / max(len(model_rows), 1)
            )
            for metric in metrics
            if all(metric in r for r in model_rows)
        }
        degraded: list[dict[str, Any]] = []
        metric_improvements: dict[str, list[float]] = {
            metric: [] for metric in metrics
        }
        if model != baseline:
            for row in model_rows:
                reference = baseline_rows.get(_key(row))
                if reference is None:
                    continue
                for metric in metrics:
                    if metric not in row or metric not in reference:
                        continue
                    before, after = reference[metric], row[metric]
                    if not math.isfinite(float(before)):
                        continue
                    if not math.isfinite(float(after)):
                        worse = True
                    elif _is_correlation(metric):
                        worse = after < before - tolerance * max(abs(before), _EPS)
                    else:
                        before_cost = _comparison_cost(metric, reference)
                        after_cost = _comparison_cost(metric, row)
                        worse = (
                            after_cost
                            > before_cost + _cost_tolerance(metric, before_cost)
                        )
                    if math.isfinite(float(after)):
                        metric_improvements[metric].append(
                            _comparison_cost(metric, reference)
                            - _comparison_cost(metric, row)
                        )
                    if worse:
                        degraded.append(
                            {
                                "variable": row["variable"],
                                "level": row["level"],
                                "rollout_step": row["rollout_step"],
                                "region": row["region"],
                                "metric": metric,
                                "baseline": before,
                                "candidate": after,
                            }
                        )
        bias_improved = (
            model != baseline
            and "bias" in aggregate
            and aggregate["bias"]
            < float(
                sum(abs(r["bias"]) for r in by_model[baseline]) / max(len(by_model[baseline]), 1)
            )
        )
        summary["models"][model] = {
            "aggregate": aggregate,
            "groups": len(model_rows),
            "degraded_groups": degraded,
            "bias_improved": bias_improved,
            "metric_improvements": {
                metric: float(sum(values) / len(values))
                for metric, values in metric_improvements.items()
                if values
            },
            # An improvement claim requires a bias reduction with no material
            # degradation anywhere else.
            "improvement": bool(bias_improved and not degraded),
        }
    return summary
