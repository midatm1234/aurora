"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Strict paired evaluation for temporal-Mamba refinement ablations.

The evaluator compares raw Aurora, Mamba-off, ordered-Mamba, and optionally
shuffled-lead rollouts only after requiring exact initialization, valid-time,
grid, and seed alignment plus matching checkpoint, raw-corpus, split, and
trajectory-policy assertions. It hashes the supplied phase-1 artifact but
cannot independently prove that a legacy raw-rollout corpus came from that
artifact unless the raw corpus has its own source manifest. Statistical
uncertainty uses a hierarchical seed then chronological-initialization-block
bootstrap, never spatial pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.generate_overall_evaluation_maps import (
    Selection,
    aligned_2d_values,
    assert_same_coordinate,
    file_safe,
    load_settings,
    match_coordinate_indices,
    match_valid_time_index,
    resolve_forecast_variable,
    rollout_files_by_initialization,
    select_level_index,
    target_selections,
    validate_rollout_time_metadata,
)

_QUANTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Region:
    """Named latitude/longitude evaluation region."""

    name: str
    latitude_min: float
    latitude_max: float
    longitude_min: float | None = None
    longitude_max: float | None = None

    def mask(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """Return a two-dimensional mask, supporting dateline-crossing boxes."""
        latitude = np.asarray(latitude, dtype=float)
        longitude = np.asarray(longitude, dtype=float)
        lat_ok = (latitude >= self.latitude_min) & (latitude <= self.latitude_max)
        if self.longitude_min is None or self.longitude_max is None:
            lon_ok = np.ones(longitude.shape, dtype=bool)
        else:
            width = float(self.longitude_max - self.longitude_min)
            if abs(width) >= 360.0:
                lon_ok = np.ones(longitude.shape, dtype=bool)
            else:
                start = (float(self.longitude_min) + 180.0) % 360.0 - 180.0
                span = width % 360.0
                normalized = (longitude + 180.0) % 360.0 - 180.0
                lon_ok = (normalized - start) % 360.0 <= span
        result = lat_ok[:, None] & lon_ok[None, :]
        if not np.any(result):
            raise ValueError(f"Region {self.name!r} contains no evaluation grid cells.")
        return result


@dataclass(frozen=True)
class MetricSpec:
    """One paired metric and the direction in which skill improves."""

    name: str
    label: str
    higher_is_better: bool
    pool_as_root_mean_square: bool = False


def parse_region(value: str) -> Region:
    """Parse NAME,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX."""
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 5 or not parts[0]:
        raise ValueError("Region must be NAME,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX, " f"got {value!r}.")
    lat_min, lat_max, lon_min, lon_max = map(float, parts[1:])
    if not all(np.isfinite([lat_min, lat_max, lon_min, lon_max])):
        raise ValueError(f"Region bounds must be finite: {value!r}.")
    if lat_min > lat_max or lat_min < -90.0 or lat_max > 90.0:
        raise ValueError(f"Invalid latitude bounds in region {value!r}.")
    return Region(parts[0], lat_min, lat_max, lon_min, lon_max)


def default_regions(latitude: np.ndarray, longitude: np.ndarray) -> list[Region]:
    """Return the domain and standard latitude zones for a global grid."""
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    regions = [
        Region(
            "domain",
            float(latitude.min()),
            float(latitude.max()),
            float(longitude.min()),
            float(longitude.max()),
        )
    ]
    if float(np.ptp(latitude)) >= 120.0 and float(np.ptp(longitude)) >= 300.0:
        regions.extend(
            [
                Region("southern_extratropics", -90.0, -30.0),
                Region("tropics", -30.0, 30.0),
                Region("northern_extratropics", 30.0, 90.0),
            ]
        )
    return regions


def area_weights(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Cosine-latitude cell weights on a regular latitude/longitude grid."""
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    weights = np.cos(np.deg2rad(latitude)).clip(min=0.0)
    return np.broadcast_to(weights[:, None], (latitude.size, longitude.size)).copy()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0.0 else math.nan


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    if not np.isfinite(mean):
        return math.nan
    return math.sqrt(max(0.0, _weighted_mean((values - mean) ** 2, weights)))


def _weighted_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Stable weighted Pearson correlation without an absolute scale floor."""
    first_centered = first - _weighted_mean(first, weights)
    second_centered = second - _weighted_mean(second, weights)
    first_scale = float(np.max(np.abs(first_centered)))
    second_scale = float(np.max(np.abs(second_centered)))
    if first.size < 2 or first_scale == 0.0 or second_scale == 0.0:
        return math.nan
    first_centered = first_centered / first_scale
    second_centered = second_centered / second_scale
    covariance = _weighted_mean(first_centered * second_centered, weights)
    first_variance = _weighted_mean(first_centered**2, weights)
    second_variance = _weighted_mean(second_centered**2, weights)
    denominator = math.sqrt(first_variance * second_variance)
    return covariance / denominator if denominator > 0.0 else math.nan


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan
    positions = (np.cumsum(weights) - 0.5 * weights) / total
    return float(np.interp(percentile / 100.0, positions, values))


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    percentiles: Sequence[float],
) -> dict[float, float]:
    """Compute several weighted quantiles with a single stable ordering."""
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(np.sum(ordered_weights))
    if total <= 0.0:
        return {float(percentile): math.nan for percentile in percentiles}
    positions = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / total
    return {
        float(percentile): float(np.interp(float(percentile) / 100.0, positions, ordered_values))
        for percentile in percentiles
    }


def _weighted_distribution_distances(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return weighted empirical W1 and Kolmogorov CDF distances."""
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan, math.nan

    def ordered(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(values, kind="stable")
        return values[order], np.cumsum(weights[order]) / total

    first_values, first_cumulative = ordered(first)
    second_values, second_cumulative = ordered(second)
    support = np.unique(np.concatenate((first_values, second_values)))

    def cdf_at(values: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(values, support, side="right")
        result = np.zeros(support.shape, dtype=np.float64)
        positive = indices > 0
        result[positive] = cumulative[indices[positive] - 1]
        return result

    first_cdf = cdf_at(first_values, first_cumulative)
    second_cdf = cdf_at(second_values, second_cumulative)
    separation = np.abs(first_cdf - second_cdf)
    wasserstein = float(np.sum(separation[:-1] * np.diff(support))) if support.size > 1 else 0.0
    return wasserstein, float(np.max(separation, initial=0.0))


def _weighted_gradient_rmse(
    truth: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    *,
    longitude_periodic: bool,
) -> float:
    """Return a first-difference structural error on a two-dimensional grid."""
    if truth.ndim != 2:
        return math.nan
    squared_errors: list[np.ndarray] = []
    component_weights: list[np.ndarray] = []
    if truth.shape[0] > 1:
        squared_errors.append((np.diff(candidate, axis=0) - np.diff(truth, axis=0)) ** 2)
        component_weights.append(np.minimum(weights[:-1], weights[1:]))
    if truth.shape[1] > 1:
        if longitude_periodic:
            squared_errors.append(
                (np.roll(candidate, -1, axis=1) - candidate - np.roll(truth, -1, axis=1) + truth)
                ** 2
            )
            component_weights.append(np.minimum(weights, np.roll(weights, -1, axis=1)))
        else:
            squared_errors.append((np.diff(candidate, axis=1) - np.diff(truth, axis=1)) ** 2)
            component_weights.append(np.minimum(weights[:, :-1], weights[:, 1:]))
    denominator = float(sum(np.sum(value) for value in component_weights))
    if denominator <= 0.0:
        return math.nan
    numerator = float(
        sum(
            np.sum(error * component_weight)
            for error, component_weight in zip(
                squared_errors,
                component_weights,
                strict=True,
            )
        )
    )
    return math.sqrt(max(0.0, numerator / denominator))


def _quantile_label(percentile: float) -> str:
    return f"p{percentile:g}".replace(".", "_")


def _common_values(
    truth: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    arrays = {"truth": np.asarray(truth, dtype=np.float64)}
    arrays.update({name: np.asarray(value, dtype=np.float64) for name, value in candidates.items()})
    expected = arrays["truth"].shape
    if np.asarray(weights).shape != expected or any(
        value.shape != expected for value in arrays.values()
    ):
        raise ValueError("Truth, candidates, and weights must have identical shapes.")
    mask = np.asarray(weights) > 0.0
    for values in arrays.values():
        mask &= np.isfinite(values)
    if not np.any(mask):
        return np.asarray([]), {name: np.asarray([]) for name in candidates}, np.asarray([])
    return (
        arrays["truth"][mask],
        {name: arrays[name][mask] for name in candidates},
        np.asarray(weights, dtype=np.float64)[mask],
    )


def paired_field_metrics(
    truth: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    longitude_periodic: bool = False,
) -> dict[str, float | int]:
    """Area-weighted spatial, distribution, tail, and cell-pair metrics."""
    raw_truth = np.asarray(truth, dtype=np.float64)
    raw_candidates = {
        name: np.asarray(value, dtype=np.float64) for name, value in candidates.items()
    }
    raw_weights = np.asarray(weights, dtype=np.float64)
    common_mask = np.isfinite(raw_truth) & (raw_weights > 0.0)
    for value in raw_candidates.values():
        common_mask &= np.isfinite(value)
    structural_weights = np.where(common_mask, raw_weights, 0.0)

    truth, candidates, weights = _common_values(truth, candidates, weights)
    if truth.size == 0:
        return {"number_of_valid_points": 0}
    truth_std = _weighted_std(truth, weights)
    result: dict[str, float | int] = {
        "number_of_valid_points": int(truth.size),
        "truth_mean": _weighted_mean(truth, weights),
        "truth_spatial_std": truth_std,
    }
    truth_quantiles = _weighted_quantiles(truth, weights, _QUANTILES)
    for percentile, value in truth_quantiles.items():
        result[f"truth_{_quantile_label(percentile)}"] = value
    result["truth_maximum"] = float(np.max(truth))

    absolute_errors: dict[str, np.ndarray] = {}
    for name, candidate in candidates.items():
        error = candidate - truth
        absolute_error = np.abs(error)
        absolute_errors[name] = absolute_error
        mse = _weighted_mean(error**2, weights)
        bias = _weighted_mean(error, weights)
        centered_rmse = math.sqrt(max(0.0, _weighted_mean((error - bias) ** 2, weights)))
        candidate_std = _weighted_std(candidate, weights)
        spatial_correlation = _weighted_correlation(truth, candidate, weights)
        wasserstein, cdf_distance = _weighted_distribution_distances(
            truth,
            candidate,
            weights,
        )
        result.update(
            {
                f"{name}_mean": _weighted_mean(candidate, weights),
                f"{name}_mae": _weighted_mean(absolute_error, weights),
                f"{name}_mse": mse,
                f"{name}_rmse": math.sqrt(mse),
                f"{name}_bias": bias,
                f"{name}_abs_bias": abs(bias),
                f"{name}_centered_rmse": centered_rmse,
                f"{name}_pattern_rmse": centered_rmse,
                f"{name}_spatial_correlation": spatial_correlation,
                f"{name}_anomaly_correlation": spatial_correlation,
                f"{name}_gradient_rmse": _weighted_gradient_rmse(
                    raw_truth,
                    raw_candidates[name],
                    structural_weights,
                    longitude_periodic=longitude_periodic,
                ),
                f"{name}_wasserstein_distance": wasserstein,
                f"{name}_cdf_distance": cdf_distance,
                f"{name}_spatial_std": candidate_std,
                f"{name}_std_ratio": (candidate_std / truth_std if truth_std > 0.0 else math.nan),
                f"{name}_std_ratio_error": (
                    abs(candidate_std / truth_std - 1.0) if truth_std > 0.0 else math.nan
                ),
            }
        )
        candidate_quantiles = _weighted_quantiles(candidate, weights, _QUANTILES)
        for percentile, threshold in truth_quantiles.items():
            label = _quantile_label(percentile)
            candidate_quantile = candidate_quantiles[percentile]
            truth_tail = truth >= threshold
            truth_tail_weights = weights[truth_tail]
            candidate_frequency = _weighted_mean((candidate >= threshold).astype(float), weights)
            truth_frequency = _weighted_mean(truth_tail.astype(float), weights)
            result.update(
                {
                    f"{name}_{label}": candidate_quantile,
                    f"{name}_{label}_bias": candidate_quantile - threshold,
                    f"{name}_{label}_abs_error": abs(candidate_quantile - threshold),
                    f"{name}_{label}_tail_mae": _weighted_mean(
                        absolute_error[truth_tail],
                        truth_tail_weights,
                    ),
                    f"{name}_{label}_exceedance_frequency_bias": (
                        candidate_frequency - truth_frequency
                    ),
                    f"{name}_{label}_exceedance_frequency_abs_bias": abs(
                        candidate_frequency - truth_frequency
                    ),
                }
            )
        candidate_maximum = float(np.max(candidate))
        result[f"{name}_maximum"] = candidate_maximum
        result[f"{name}_maximum_bias"] = candidate_maximum - result["truth_maximum"]
        result[f"{name}_maximum_abs_error"] = abs(candidate_maximum - result["truth_maximum"])

    comparisons: list[tuple[str, str]] = []
    if "off" in candidates:
        comparisons.extend(("off", name) for name in candidates if name != "off")
    if "aurora" in candidates:
        comparisons.extend(("aurora", name) for name in candidates if name != "aurora")
    if "on" in candidates and "shuffled" in candidates:
        comparisons.append(("shuffled", "on"))
    for reference, challenger in dict.fromkeys(comparisons):
        reference_error = absolute_errors[reference]
        challenger_error = absolute_errors[challenger]
        tolerance = (
            8.0
            * np.finfo(np.float64).eps
            * np.maximum(
                reference_error,
                challenger_error,
            )
        )
        better = challenger_error < reference_error - tolerance
        worse = challenger_error > reference_error + tolerance
        result[f"fraction_cells_{challenger}_better_than_{reference}"] = _weighted_mean(
            better.astype(float), weights
        )
        result[f"fraction_cells_{challenger}_worse_than_{reference}"] = _weighted_mean(
            worse.astype(float), weights
        )
    return result


def paired_tendency_metrics(
    previous_truth: np.ndarray,
    previous_candidates: Mapping[str, np.ndarray],
    truth: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    weights: np.ndarray,
    lead_interval_hours: float,
) -> dict[str, float | int]:
    """Evaluate per-hour changes between consecutive valid forecast leads."""
    if not np.isfinite(lead_interval_hours) or lead_interval_hours <= 0.0:
        raise ValueError("lead_interval_hours must be finite and positive.")
    all_candidates = {f"previous_{name}": value for name, value in previous_candidates.items()}
    all_candidates.update(candidates)
    common_truth, common, common_weights = _common_values(
        previous_truth,
        {"current_truth": truth, **all_candidates},
        weights,
    )
    if common_truth.size == 0:
        return {"number_of_tendency_valid_points": 0}
    previous_truth = common_truth
    truth = common.pop("current_truth")
    previous = {
        name.removeprefix("previous_"): value
        for name, value in tuple(common.items())
        if name.startswith("previous_")
    }
    current = {name: value for name, value in common.items() if not name.startswith("previous_")}
    scale = 1.0 / lead_interval_hours
    truth_tendency = (truth - previous_truth) * scale
    candidate_tendencies = {name: (current[name] - previous[name]) * scale for name in current}
    truth_std = _weighted_std(truth_tendency, common_weights)
    result: dict[str, float | int] = {
        "number_of_tendency_valid_points": int(truth_tendency.size),
        "tendency_lead_interval_hours": float(lead_interval_hours),
        "truth_tendency_rms": math.sqrt(_weighted_mean(truth_tendency**2, common_weights)),
    }
    for name, tendency in candidate_tendencies.items():
        error = tendency - truth_tendency
        mse = _weighted_mean(error**2, common_weights)
        std = _weighted_std(tendency, common_weights)
        result.update(
            {
                f"{name}_tendency_mse": mse,
                f"{name}_tendency_rmse": math.sqrt(mse),
                f"{name}_tendency_spatial_correlation": _weighted_correlation(
                    truth_tendency, tendency, common_weights
                ),
                f"{name}_tendency_std_ratio": (std / truth_std if truth_std > 0.0 else math.nan),
                f"{name}_tendency_std_ratio_error": (
                    abs(std / truth_std - 1.0) if truth_std > 0.0 else math.nan
                ),
            }
        )
    if "off" in candidate_tendencies:
        residual_tendency = truth_tendency - candidate_tendencies["off"]
        residual_rms = math.sqrt(_weighted_mean(residual_tendency**2, common_weights))
        for name in (item for item in candidate_tendencies if item not in {"off", "aurora"}):
            correction_tendency = candidate_tendencies[name] - candidate_tendencies["off"]
            correction_rms = math.sqrt(_weighted_mean(correction_tendency**2, common_weights))
            amplitude_ratio = correction_rms / residual_rms if residual_rms > 0.0 else math.nan
            result[f"{name}_correction_tendency_amplitude_ratio"] = amplitude_ratio
            result[f"{name}_correction_tendency_amplitude_ratio_error"] = abs(amplitude_ratio - 1.0)
            result[f"{name}_correction_tendency_residual_correlation"] = _weighted_correlation(
                correction_tendency,
                residual_tendency,
                common_weights,
            )
    return result


def _column_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first - np.mean(first, axis=0, keepdims=True)
    second = second - np.mean(second, axis=0, keepdims=True)
    first_scale = np.max(np.abs(first), axis=0)
    second_scale = np.max(np.abs(second), axis=0)
    valid = (first_scale > 0.0) & (second_scale > 0.0)
    result = np.full(first_scale.shape, np.nan, dtype=np.float64)
    first[:, valid] /= first_scale[valid]
    second[:, valid] /= second_scale[valid]
    numerator = np.sum(first[:, valid] * second[:, valid], axis=0)
    denominator = np.sqrt(
        np.sum(first[:, valid] ** 2, axis=0) * np.sum(second[:, valid] ** 2, axis=0)
    )
    nonzero = denominator > 0.0
    values = np.full(denominator.shape, np.nan, dtype=np.float64)
    values[nonzero] = numerator[nonzero] / denominator[nonzero]
    result[valid] = values
    return result


def temporal_sequence_metrics(
    truth: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float | int]:
    """Area-weighted per-cell temporal correlation and lag-one autocorrelation."""
    truth = np.asarray(truth, dtype=np.float64)
    candidates = {name: np.asarray(value, dtype=np.float64) for name, value in candidates.items()}
    if truth.ndim != 3 or truth.shape[0] < 3:
        return {"number_of_temporal_cells": 0}
    if any(value.shape != truth.shape for value in candidates.values()):
        raise ValueError("Temporal truth and candidate sequences must have identical shapes.")
    finite = np.all(np.isfinite(truth), axis=0)
    for value in candidates.values():
        finite &= np.all(np.isfinite(value), axis=0)
    finite &= np.asarray(weights) > 0.0
    if not np.any(finite):
        return {"number_of_temporal_cells": 0}
    cell_weights = np.asarray(weights, dtype=float)[finite]
    truth_cells = truth[:, finite]
    truth_autocorrelation = _column_correlation(truth_cells[:-1], truth_cells[1:])
    result: dict[str, float | int] = {
        "number_of_temporal_cells": int(np.count_nonzero(finite)),
        "truth_lag1_autocorrelation": _weighted_mean(
            truth_autocorrelation[np.isfinite(truth_autocorrelation)],
            cell_weights[np.isfinite(truth_autocorrelation)],
        ),
    }
    for name, value in candidates.items():
        cells = value[:, finite]
        temporal_correlation = _column_correlation(truth_cells, cells)
        autocorrelation = _column_correlation(cells[:-1], cells[1:])
        temporal_valid = np.isfinite(temporal_correlation)
        autocorrelation_valid = np.isfinite(autocorrelation) & np.isfinite(truth_autocorrelation)
        candidate_autocorrelation = _weighted_mean(
            autocorrelation[autocorrelation_valid],
            cell_weights[autocorrelation_valid],
        )
        result.update(
            {
                f"{name}_temporal_correlation": _weighted_mean(
                    temporal_correlation[temporal_valid],
                    cell_weights[temporal_valid],
                ),
                f"{name}_lag1_autocorrelation": candidate_autocorrelation,
                f"{name}_lag1_autocorrelation_abs_error": abs(
                    candidate_autocorrelation - result["truth_lag1_autocorrelation"]
                ),
            }
        )
    return result


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a provenance artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def composite_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered, JSON-serializable provenance record list."""
    payload = json.dumps(
        list(records),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def select_initializations(
    initializations: Sequence[int],
    maximum: int | None,
    *,
    mode: str,
) -> list[int]:
    """Select reproducibly without making a capped sample a seasonal prefix."""
    values = sorted(int(value) for value in initializations)
    if maximum is None or maximum >= len(values):
        return values
    if maximum < 1:
        raise ValueError("maximum must be at least one.")
    if mode == "first":
        return values[:maximum]
    if mode != "uniform":
        raise ValueError("selection mode must be 'uniform' or 'first'.")
    indices = np.rint(np.linspace(0, len(values) - 1, maximum)).astype(int)
    if np.unique(indices).size != maximum:
        raise RuntimeError("Uniform initialization selection produced duplicate indices.")
    return [values[index] for index in indices]


def chronological_purged_split(
    initializations: Sequence[int],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_hours: float = 72.0,
) -> dict[str, list[int]]:
    """Group by UTC date and create chronological train/validation/test splits."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one.")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError(
            "validation_fraction must be positive and leave a non-empty test fraction."
        )
    if not np.isfinite(purge_hours) or purge_hours < 0.0:
        raise ValueError("purge_hours must be finite and non-negative.")
    values = np.asarray(sorted(set(int(value) for value in initializations)), dtype=np.int64)
    if values.size < 3:
        raise ValueError("At least three initializations are required.")
    times = values.astype("datetime64[ns]")
    dates = times.astype("datetime64[D]")
    unique_dates = np.unique(dates)
    if unique_dates.size < 3:
        raise ValueError("At least three UTC date groups are required.")
    validation_index = max(
        1,
        min(
            unique_dates.size - 2,
            int(math.floor(train_fraction * unique_dates.size)),
        ),
    )
    test_index = max(
        validation_index + 1,
        min(
            unique_dates.size - 1,
            int(math.floor((train_fraction + validation_fraction) * unique_dates.size)),
        ),
    )
    validation_start = unique_dates[validation_index].astype("datetime64[ns]")
    test_start = unique_dates[test_index].astype("datetime64[ns]")
    purge = np.timedelta64(int(round(purge_hours * 3600.0)), "s").astype("timedelta64[ns]")
    split = {
        "train": values[times < validation_start - purge].tolist(),
        "validation": values[(times >= validation_start) & (times < test_start - purge)].tolist(),
        "test": values[times >= test_start].tolist(),
    }
    if any(not members for members in split.values()):
        raise ValueError(
            "Purge/fraction settings produced an empty split; use more dates or "
            "reduce purge_hours."
        )
    return split


_REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "mode",
    "case_name",
    "spatial_head",
    "seed",
    "phase1_checkpoint_sha256",
    "spatial_checkpoint_sha256",
    "raw_rollout_corpus_sha256",
    "raw_rollout_corpus_hash_method",
    "phase1_rollout_linkage",
    "data_split_sha256",
    "evaluation_split",
    "selected_test_initializations",
    "trajectory_policy",
}


def load_run_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "mamba_ablation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing strict provenance manifest: {path}. "
            "Use --unsafe-skip-provenance-checks only for exploratory output."
        )
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    value["_manifest_path"] = str(path.resolve())
    value["_manifest_sha256"] = sha256_file(path)
    return value


def validate_study_manifests(
    manifests: Sequence[Mapping[str, Any]],
    *,
    require_shuffled: bool,
) -> dict[int, dict[str, Mapping[str, Any]]]:
    """Validate strict two-stage fairness and return manifests by seed/mode."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for manifest in manifests:
        missing = sorted(_REQUIRED_MANIFEST_FIELDS - set(manifest))
        if missing:
            raise ValueError(f"Ablation manifest is missing fields: {missing}.")
        if int(manifest["schema_version"]) != 1:
            raise ValueError("Only Mamba ablation manifest schema_version=1 is supported.")
        mode = str(manifest["mode"])
        if mode not in {"off", "on", "shuffled"}:
            raise ValueError(f"Invalid Mamba ablation mode {mode!r}.")
        trajectory_policy = manifest["trajectory_policy"]
        if (
            not isinstance(trajectory_policy, Mapping)
            or not str(trajectory_policy.get("source") or "").strip()
            or not isinstance(trajectory_policy.get("aurora_autoregressive_feedback"), bool)
        ):
            raise ValueError(
                "trajectory_policy must explicitly record its source and Aurora "
                "autoregressive-feedback boolean."
            )
        if not str(manifest["raw_rollout_corpus_hash_method"]).strip():
            raise ValueError("raw_rollout_corpus_hash_method must be explicit.")
        linkage = manifest["phase1_rollout_linkage"]
        if not isinstance(linkage, Mapping) or linkage.get("status") not in {
            "proven",
            "asserted_not_proven",
        }:
            raise ValueError(
                "phase1_rollout_linkage.status must be 'proven' or " "'asserted_not_proven'."
            )
        for name in (
            "phase1_checkpoint_sha256",
            "spatial_checkpoint_sha256",
            "raw_rollout_corpus_sha256",
            "data_split_sha256",
        ):
            if not _HASH_PATTERN.fullmatch(str(manifest[name])):
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        seed = int(manifest["seed"])
        if mode in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate {mode!r} manifest for seed {seed}.")
        by_seed[seed][mode] = manifest

    required_modes = {"off", "on"} | ({"shuffled"} if require_shuffled else set())
    for seed, modes in by_seed.items():
        if set(modes) != required_modes:
            raise ValueError(
                f"Seed {seed} has modes {sorted(modes)}; expected " f"{sorted(required_modes)}."
            )
        off = modes["off"]
        on = modes["on"]
        invariant_fields = (
            "case_name",
            "spatial_head",
            "seed",
            "phase1_checkpoint_sha256",
            "spatial_checkpoint_sha256",
            "raw_rollout_corpus_sha256",
            "raw_rollout_corpus_hash_method",
            "phase1_rollout_linkage",
            "data_split_sha256",
            "evaluation_split",
            "selected_test_initializations",
            "trajectory_policy",
        )
        for mode, manifest in modes.items():
            for name in invariant_fields:
                if manifest[name] != off[name]:
                    raise ValueError(f"Seed {seed} {mode}/{name} does not match Mamba-off.")
        if str(off["evaluation_split"]) != "test":
            raise ValueError("Ablation evaluation must use the held-out test split.")
        if on.get("spatial_parameters_frozen") is not True:
            raise ValueError("Mamba-on must record spatial_parameters_frozen=true.")
        if on.get("temporal_training_split") != "train":
            raise ValueError("Mamba-on temporal parameters must be tuned only on train.")
        if on.get("checkpoint_selection_split") != "validation":
            raise ValueError("Mamba-on checkpoint selection must use validation.")
        phase2_hash = str(on.get("phase2_checkpoint_sha256", ""))
        if not _HASH_PATTERN.fullmatch(phase2_hash):
            raise ValueError("Mamba-on must record phase2_checkpoint_sha256.")
        if "shuffled" in modes:
            shuffled = modes["shuffled"]
            if shuffled.get("spatial_parameters_frozen") is not True:
                raise ValueError("Shuffled-lead run must retain frozen spatial parameters.")
            if shuffled.get("phase2_checkpoint_sha256") != phase2_hash:
                raise ValueError(
                    "Ordered and shuffled arms must use the identical phase-2 checkpoint."
                )
            permutation = shuffled.get("lead_permutation")
            if not isinstance(permutation, list) or len(permutation) < 2:
                raise ValueError("Shuffled arm must record a non-trivial lead_permutation.")
    return by_seed


def _moving_block_bootstrap_indices(
    size: int,
    *,
    resamples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block bootstrap indices over ordered initializations."""
    if size < 1:
        return np.empty((0, 0), dtype=np.int64)
    if resamples < 1 or block_length < 1:
        raise ValueError("resamples and block_length must be at least one.")
    block_length = min(block_length, size)
    blocks = int(math.ceil(size / block_length))
    starts = rng.integers(0, size, size=(resamples, blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % size
    return indices.reshape(resamples, -1)[:, :size]


def _moving_block_bootstrap_means(
    values: np.ndarray,
    *,
    resamples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block bootstrap means over ordered initializations."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return np.asarray([], dtype=np.float64)
    indices = _moving_block_bootstrap_indices(
        values.size,
        resamples=resamples,
        block_length=block_length,
        rng=rng,
    )
    return np.mean(values[indices], axis=1)


def _hierarchical_block_bootstrap_means(
    paired: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    resamples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample seeds, then chronological initialization blocks within seed."""
    if paired.empty:
        return np.empty((0, len(value_columns)), dtype=np.float64)
    grouped = {
        seed: group.sort_values("initialization_time")[list(value_columns)].to_numpy(
            dtype=np.float64
        )
        for seed, group in paired.groupby("seed", sort=True)
    }
    seeds = np.asarray(list(grouped), dtype=object)
    output = np.empty((resamples, len(value_columns)), dtype=np.float64)
    for resample in range(resamples):
        sampled_seeds = rng.choice(seeds, size=seeds.size, replace=True)
        chunks: list[np.ndarray] = []
        for sampled_seed in sampled_seeds:
            values = grouped[sampled_seed]
            indices = _moving_block_bootstrap_indices(
                len(values),
                resamples=1,
                block_length=block_length,
                rng=rng,
            )[0]
            chunks.append(values[indices])
        output[resample] = np.mean(np.concatenate(chunks, axis=0), axis=0)
    return output


def paired_bootstrap_summary(
    frame: pd.DataFrame,
    specs: Sequence[MetricSpec],
    *,
    group_keys: Sequence[str],
    reference: str,
    candidate: str,
    resamples: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize positive-is-better paired deltas, clustered by initialization."""
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for group_key, group in frame.groupby(list(group_keys), dropna=False, sort=True):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_keys, group_key))
        for spec in specs:
            reference_column = f"{reference}_{spec.name}"
            candidate_column = f"{candidate}_{spec.name}"
            if reference_column not in group or candidate_column not in group:
                continue
            source_columns = [
                "initialization_time",
                reference_column,
                candidate_column,
            ]
            if "seed" in group:
                source_columns.insert(0, "seed")
            source = group[source_columns].dropna()
            if "seed" not in source:
                source.insert(0, "seed", 0)
            if source.empty:
                continue
            if spec.pool_as_root_mean_square:
                source = source.assign(
                    _reference_squared=source[reference_column] ** 2,
                    _candidate_squared=source[candidate_column] ** 2,
                )
                paired = (
                    source.groupby(["seed", "initialization_time"], sort=True)[
                        ["_reference_squared", "_candidate_squared"]
                    ]
                    .mean()
                    .reset_index()
                )
                reference_squared = paired["_reference_squared"].to_numpy(dtype=float)
                candidate_squared = paired["_candidate_squared"].to_numpy(dtype=float)
                reference_values = np.sqrt(reference_squared)
                candidate_values = np.sqrt(candidate_squared)
                bootstrap_squared = _hierarchical_block_bootstrap_means(
                    paired,
                    ["_reference_squared", "_candidate_squared"],
                    resamples=resamples,
                    block_length=block_length,
                    rng=rng,
                )
                bootstrap_reference = np.sqrt(bootstrap_squared[:, 0])
                bootstrap_candidate = np.sqrt(bootstrap_squared[:, 1])
                reference_mean = math.sqrt(float(np.mean(reference_squared)))
                candidate_mean = math.sqrt(float(np.mean(candidate_squared)))
                bootstrap = (
                    bootstrap_candidate - bootstrap_reference
                    if spec.higher_is_better
                    else bootstrap_reference - bootstrap_candidate
                )
            else:
                paired = (
                    source.groupby(["seed", "initialization_time"], sort=True)[
                        [reference_column, candidate_column]
                    ]
                    .mean()
                    .reset_index()
                )
                reference_values = paired[reference_column].to_numpy(dtype=float)
                candidate_values = paired[candidate_column].to_numpy(dtype=float)
                reference_mean = float(np.mean(reference_values))
                candidate_mean = float(np.mean(candidate_values))
                paired["_improvement"] = (
                    candidate_values - reference_values
                    if spec.higher_is_better
                    else reference_values - candidate_values
                )
                bootstrap = _hierarchical_block_bootstrap_means(
                    paired,
                    ["_improvement"],
                    resamples=resamples,
                    block_length=block_length,
                    rng=rng,
                )[:, 0]
            improvement = (
                candidate_values - reference_values
                if spec.higher_is_better
                else reference_values - candidate_values
            )
            mean_improvement = (
                candidate_mean - reference_mean
                if spec.higher_is_better
                else reference_mean - candidate_mean
            )
            scale = np.maximum(np.abs(reference_values), np.abs(candidate_values))
            tolerance = 8.0 * np.finfo(np.float64).eps * scale
            better = improvement > tolerance
            worse = improvement < -tolerance
            output.append(
                {
                    **base,
                    "comparison": f"{candidate}_vs_{reference}",
                    "metric": spec.name,
                    "metric_label": spec.label,
                    "higher_is_better": spec.higher_is_better,
                    "number_of_seeds": int(paired["seed"].nunique()),
                    "number_of_initializations": int(paired["initialization_time"].nunique()),
                    "number_of_seed_initialization_pairs": int(len(paired)),
                    "number_of_rows": int(len(group)),
                    "reference_mean": reference_mean,
                    "candidate_mean": candidate_mean,
                    "mean_improvement": mean_improvement,
                    "ci_lower": (
                        float(np.percentile(bootstrap, 2.5)) if len(paired) >= 2 else math.nan
                    ),
                    "ci_upper": (
                        float(np.percentile(bootstrap, 97.5)) if len(paired) >= 2 else math.nan
                    ),
                    "fraction_cases_candidate_better": float(np.mean(better)),
                    "fraction_cases_candidate_worse": float(np.mean(worse)),
                    "fraction_cases_tied": float(np.mean(~better & ~worse)),
                }
            )
    return pd.DataFrame(output)


def single_metric_bootstrap_summary(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    group_keys: Sequence[str],
    resamples: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap direct per-case fractions over initialization clusters."""
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for group_key, group in frame.groupby(list(group_keys), dropna=False, sort=True):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_keys, group_key))
        for column in columns:
            if column not in group:
                continue
            source_columns = ["initialization_time", column]
            if "seed" in group:
                source_columns.insert(0, "seed")
            source = group[source_columns].dropna()
            if "seed" not in source:
                source.insert(0, "seed", 0)
            paired = (
                source.groupby(["seed", "initialization_time"], sort=True)[column]
                .mean()
                .reset_index()
            )
            values = paired[column].to_numpy(dtype=float)
            if values.size == 0:
                continue
            bootstrap = _hierarchical_block_bootstrap_means(
                paired,
                [column],
                resamples=resamples,
                block_length=block_length,
                rng=rng,
            )[:, 0]
            output.append(
                {
                    **base,
                    "metric": column,
                    "number_of_seeds": int(paired["seed"].nunique()),
                    "number_of_initializations": int(paired["initialization_time"].nunique()),
                    "number_of_seed_initialization_pairs": int(values.size),
                    "mean": float(np.mean(values)),
                    "ci_lower": (
                        float(np.percentile(bootstrap, 2.5)) if values.size >= 2 else math.nan
                    ),
                    "ci_upper": (
                        float(np.percentile(bootstrap, 97.5)) if values.size >= 2 else math.nan
                    ),
                }
            )
    return pd.DataFrame(output)


FIELD_METRIC_SPECS = (
    MetricSpec("mae", "MAE", False),
    MetricSpec("rmse", "RMSE", False, True),
    MetricSpec("centered_rmse", "centered RMSE", False, True),
    MetricSpec("pattern_rmse", "pattern RMSE", False, True),
    MetricSpec("abs_bias", "absolute mean bias", False),
    MetricSpec("spatial_correlation", "spatial correlation", True),
    MetricSpec("anomaly_correlation", "anomaly correlation", True),
    MetricSpec("gradient_rmse", "gradient RMSE", False, True),
    MetricSpec("std_ratio_error", "|standard-deviation ratio - 1|", False),
    MetricSpec("wasserstein_distance", "Wasserstein-1 distance", False),
    MetricSpec("cdf_distance", "maximum CDF distance", False),
    MetricSpec("p50_abs_error", "P50 absolute error", False),
    MetricSpec("p75_abs_error", "P75 absolute error", False),
    MetricSpec("p90_abs_error", "P90 absolute error", False),
    MetricSpec("p95_abs_error", "P95 absolute error", False),
    MetricSpec("p99_abs_error", "P99 absolute error", False),
    MetricSpec("p99_9_abs_error", "P99.9 absolute error", False),
    MetricSpec("p95_tail_mae", "P95 truth-tail MAE", False),
    MetricSpec("p99_tail_mae", "P99 truth-tail MAE", False),
    MetricSpec("p99_9_tail_mae", "P99.9 truth-tail MAE", False),
    MetricSpec(
        "p99_9_exceedance_frequency_abs_bias",
        "P99.9 exceedance-frequency absolute bias",
        False,
    ),
    MetricSpec("maximum_abs_error", "maximum absolute error", False),
)

TENDENCY_METRIC_SPECS = (
    MetricSpec("tendency_rmse", "tendency RMSE", False, True),
    MetricSpec("tendency_spatial_correlation", "tendency spatial correlation", True),
    MetricSpec(
        "tendency_std_ratio_error",
        "|tendency standard-deviation ratio - 1|",
        False,
    ),
)

TEMPORAL_METRIC_SPECS = (
    MetricSpec("temporal_correlation", "temporal correlation", True),
    MetricSpec(
        "lag1_autocorrelation_abs_error",
        "lag-one autocorrelation absolute error",
        False,
    ),
)


@dataclass(frozen=True)
class StudyPair:
    """One seed's exact Mamba-off/on(/shuffled) matched artifact set."""

    seed: int
    directories: Mapping[str, Path]
    manifests: Mapping[str, Mapping[str, Any]]
    phase1_checkpoint: Path
    phase1_checkpoint_sha256: str
    spatial_checkpoint: Path | None
    spatial_checkpoint_sha256: str
    phase2_checkpoint: Path | None
    phase2_checkpoint_sha256: str | None


def _manifest_initialization_keys(values: Any) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError("selected_test_initializations must be a non-empty list.")
    result: list[int] = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            result.append(int(value))
            continue
        text = str(value).strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            result.append(int(text))
            continue
        try:
            parsed = np.datetime64(text, "ns")
        except ValueError as exc:
            raise ValueError(f"Invalid selected test initialization {value!r}.") from exc
        if np.isnat(parsed):
            raise ValueError("Selected test initializations cannot contain NaT.")
        result.append(int(parsed.astype(np.int64)))
    if len(set(result)) != len(result):
        raise ValueError("selected_test_initializations contains duplicates.")
    return sorted(result)


def _rollout_index(directory: Path, pattern: str) -> dict[int, Path]:
    files = rollout_files_by_initialization(directory, pattern)
    if not files:
        raise FileNotFoundError(
            f"No timestamped rollout NetCDFs matching {pattern!r} under {directory}."
        )
    return files


def _resolve_checkpoint_artifact(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    path_field: str,
    fallback_name: str,
) -> Path:
    configured = manifest.get(path_field)
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = directory / path
    else:
        path = directory.parent / fallback_name
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Strict provenance requires {path_field} or sibling " f"{fallback_name}: {path}."
        )
    return path


def build_study_pairs(
    off_directories: Sequence[Path],
    on_directories: Sequence[Path],
    shuffled_directories: Sequence[Path],
    phase1_checkpoints: Sequence[Path],
    *,
    unsafe_skip_provenance_checks: bool,
) -> list[StudyPair]:
    """Resolve seed pairs and validate manifests plus checkpoint bytes."""
    if not off_directories or len(off_directories) != len(on_directories):
        raise ValueError("--off-dir and --on-dir must be repeated equally.")
    if shuffled_directories and len(shuffled_directories) != len(off_directories):
        raise ValueError("--shuffled-dir must be omitted or repeated once for every seed.")
    if len(phase1_checkpoints) != len(off_directories):
        raise ValueError("--phase1-checkpoint must be repeated once for every paired seed.")

    modes_and_directories: list[tuple[str, Path]] = []
    for mode, directories in (
        ("off", off_directories),
        ("on", on_directories),
        ("shuffled", shuffled_directories),
    ):
        modes_and_directories.extend(
            (mode, Path(directory).expanduser().resolve()) for directory in directories
        )
    checkpoints = [Path(path).expanduser().resolve() for path in phase1_checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing phase-1 checkpoint: {checkpoint}")

    digest_cache: dict[Path, str] = {}
    for checkpoint in checkpoints:
        digest_cache.setdefault(checkpoint, sha256_file(checkpoint))

    if unsafe_skip_provenance_checks:
        output: list[StudyPair] = []
        for index, (off_dir, on_dir, checkpoint) in enumerate(
            zip(off_directories, on_directories, checkpoints, strict=True)
        ):
            directories = {
                "off": Path(off_dir).expanduser().resolve(),
                "on": Path(on_dir).expanduser().resolve(),
            }
            if shuffled_directories:
                directories["shuffled"] = Path(shuffled_directories[index]).expanduser().resolve()
            manifests = {
                mode: {
                    "schema_version": 1,
                    "mode": mode,
                    "seed": index,
                    "case_name": "unverified",
                    "spatial_head": "unverified",
                    "phase1_checkpoint_sha256": digest_cache[checkpoint],
                    "spatial_checkpoint_sha256": "unverified",
                    "raw_rollout_corpus_sha256": "unverified",
                    "raw_rollout_corpus_hash_method": "unverified",
                    "phase1_rollout_linkage": {
                        "status": "asserted_not_proven",
                    },
                    "data_split_sha256": "unverified",
                    "evaluation_split": "unverified",
                    "selected_test_initializations": [],
                    "trajectory_policy": {
                        "source": "unverified",
                        "aurora_autoregressive_feedback": False,
                    },
                }
                for mode in directories
            }
            output.append(
                StudyPair(
                    index,
                    directories,
                    manifests,
                    checkpoint,
                    digest_cache[checkpoint],
                    None,
                    "unverified",
                    None,
                    None,
                )
            )
        return output

    directory_by_seed_mode: dict[tuple[int, str], Path] = {}
    checkpoint_by_seed: dict[int, Path] = {}
    manifests: list[Mapping[str, Any]] = []
    for expected_mode, directory in modes_and_directories:
        manifest = load_run_manifest(directory)
        mode = str(manifest.get("mode"))
        if mode != expected_mode:
            raise ValueError(
                f"{directory} was passed as {expected_mode!r} but its manifest "
                f"declares mode={mode!r}."
            )
        seed = int(manifest["seed"])
        key = (seed, mode)
        if key in directory_by_seed_mode:
            raise ValueError(f"Duplicate directory for seed={seed}, mode={mode}.")
        directory_by_seed_mode[key] = directory
        manifests.append(manifest)
    for directory, checkpoint in zip(
        off_directories,
        checkpoints,
        strict=True,
    ):
        off_manifest = load_run_manifest(Path(directory).expanduser().resolve())
        seed = int(off_manifest["seed"])
        if seed in checkpoint_by_seed:
            raise ValueError(f"Duplicate --off-dir seed {seed}.")
        checkpoint_by_seed[seed] = checkpoint

    by_seed = validate_study_manifests(
        manifests,
        require_shuffled=bool(shuffled_directories),
    )
    study_invariants = (
        "case_name",
        "spatial_head",
        "raw_rollout_corpus_sha256",
        "raw_rollout_corpus_hash_method",
        "phase1_rollout_linkage",
        "data_split_sha256",
        "selected_test_initializations",
        "trajectory_policy",
    )
    first_off = next(iter(by_seed.values()))["off"]
    spatial_checkpoint_by_seed: dict[int, Path] = {}
    phase2_checkpoint_by_seed: dict[int, Path] = {}
    for seed, seed_manifests in by_seed.items():
        checkpoint = checkpoint_by_seed.get(seed)
        if checkpoint is None:
            raise ValueError(f"No phase-1 checkpoint was supplied for seed {seed}.")
        digest = digest_cache[checkpoint]
        for mode, manifest in seed_manifests.items():
            if manifest["phase1_checkpoint_sha256"] != digest:
                raise ValueError(
                    f"Seed {seed} {mode} manifest phase-1 hash does not match "
                    f"the supplied checkpoint bytes: {checkpoint}."
                )

        off_directory = directory_by_seed_mode[(seed, "off")]
        spatial_checkpoint = _resolve_checkpoint_artifact(
            off_directory,
            seed_manifests["off"],
            path_field="spatial_checkpoint_path",
            fallback_name="spatial_checkpoint.pt",
        )
        digest_cache.setdefault(
            spatial_checkpoint,
            sha256_file(spatial_checkpoint),
        )
        if digest_cache[spatial_checkpoint] != seed_manifests["off"]["spatial_checkpoint_sha256"]:
            raise ValueError(
                f"Seed {seed} spatial checkpoint bytes do not match its manifests: "
                f"{spatial_checkpoint}."
            )
        spatial_checkpoint_by_seed[seed] = spatial_checkpoint

        on_directory = directory_by_seed_mode[(seed, "on")]
        phase2_checkpoint = _resolve_checkpoint_artifact(
            on_directory,
            seed_manifests["on"],
            path_field="phase2_checkpoint_path",
            fallback_name="temporal_checkpoint.pt",
        )
        digest_cache.setdefault(
            phase2_checkpoint,
            sha256_file(phase2_checkpoint),
        )
        if digest_cache[phase2_checkpoint] != seed_manifests["on"]["phase2_checkpoint_sha256"]:
            raise ValueError(
                f"Seed {seed} phase-2 checkpoint bytes do not match its manifests: "
                f"{phase2_checkpoint}."
            )
        phase2_checkpoint_by_seed[seed] = phase2_checkpoint

        for name in study_invariants:
            if seed_manifests["off"][name] != first_off[name]:
                raise ValueError(
                    f"Seed {seed} changes study invariant {name!r}; all seeds must "
                    "use the same case, corpus, split, test dates, and trajectory."
                )

    return [
        StudyPair(
            seed=seed,
            directories={
                mode: directory_by_seed_mode[(seed, mode)] for mode in sorted(seed_manifests)
            },
            manifests=seed_manifests,
            phase1_checkpoint=checkpoint_by_seed[seed],
            phase1_checkpoint_sha256=digest_cache[checkpoint_by_seed[seed]],
            spatial_checkpoint=spatial_checkpoint_by_seed[seed],
            spatial_checkpoint_sha256=digest_cache[spatial_checkpoint_by_seed[seed]],
            phase2_checkpoint=phase2_checkpoint_by_seed[seed],
            phase2_checkpoint_sha256=digest_cache[phase2_checkpoint_by_seed[seed]],
        )
        for seed, seed_manifests in sorted(by_seed.items())
    ]


def _selected_rollout_files(
    pair: StudyPair,
    *,
    pattern: str,
    initialization_stride: int,
    maximum_initializations: int | None,
    initialization_selection: str,
    strict_provenance: bool,
) -> dict[str, dict[int, Path]]:
    if initialization_stride < 1:
        raise ValueError("initialization_stride must be at least one.")
    indexed = {
        mode: _rollout_index(directory, pattern) for mode, directory in pair.directories.items()
    }
    off_keys = sorted(indexed["off"])
    for mode, files in indexed.items():
        if sorted(files) != off_keys:
            missing = sorted(set(off_keys) - set(files))
            extra = sorted(set(files) - set(off_keys))
            raise ValueError(
                f"Seed {pair.seed} {mode} rollout initialization set differs "
                f"from Mamba-off; missing={missing[:5]}, extra={extra[:5]}."
            )
    if strict_provenance:
        declared = _manifest_initialization_keys(
            pair.manifests["off"]["selected_test_initializations"]
        )
        if off_keys != declared:
            raise ValueError(
                f"Seed {pair.seed} files do not exactly equal the manifest's held-out "
                "test initialization list."
            )
    strided = off_keys[::initialization_stride]
    selected = select_initializations(
        strided,
        maximum_initializations,
        mode=initialization_selection,
    )
    return {mode: {key: files[key] for key in selected} for mode, files in indexed.items()}


def _select_field(
    dataset: xr.Dataset,
    selection: Selection,
    *,
    step_dim: str,
    step_index: int,
    level_tolerance: float,
    context: str,
    latitude_indices: np.ndarray | None = None,
    longitude_indices: np.ndarray | None = None,
) -> np.ndarray:
    variable = resolve_forecast_variable(dataset, selection.variable)
    if step_dim not in variable.dims:
        raise ValueError(
            f"{context} variable {variable.name!r} does not use forecast step "
            f"dimension {step_dim!r}; dims={variable.dims}."
        )
    indexers: dict[str, Any] = {step_dim: step_index}
    if latitude_indices is not None:
        indexers["latitude"] = latitude_indices
    if longitude_indices is not None:
        indexers["longitude"] = longitude_indices
    field = variable.isel(indexers)
    if selection.level is not None:
        if "level" not in field.coords:
            raise ValueError(f"{context} has no level coordinate.")
        level_index = select_level_index(
            np.asarray(field["level"].values, dtype=float),
            selection.level,
            level_tolerance,
        )
        field = field.isel(level=level_index)
    if "member" in field.dims:
        field = field.mean("member", skipna=True)
    return aligned_2d_values(field, context=context)


def _select_truth_field(
    truth: xr.Dataset,
    selection: Selection,
    *,
    time_index: int,
    latitude_indices: np.ndarray,
    longitude_indices: np.ndarray,
    level_tolerance: float,
    context: str,
) -> np.ndarray:
    if selection.variable not in truth:
        raise KeyError(f"CAMS truth is missing {selection.variable!r}.")
    field = truth[selection.variable].isel(
        time=time_index,
        latitude=latitude_indices,
        longitude=longitude_indices,
    )
    if selection.level is not None:
        level_index = select_level_index(
            np.asarray(field["level"].values, dtype=float),
            selection.level,
            level_tolerance,
        )
        field = field.isel(level=level_index)
    return aligned_2d_values(field, context=context)


def collect_paired_metrics(
    settings: Mapping[str, Any],
    pairs: Sequence[StudyPair],
    *,
    rollout_pattern: str,
    initialization_stride: int,
    maximum_initializations: int | None,
    initialization_selection: str,
    custom_regions: Sequence[Region],
    include_default_regions: bool,
    strict_provenance: bool,
    aurora_directory: Path | None = None,
    representative_fields: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stream exactly matched test rollouts and return per-initialization metrics."""
    if not pairs:
        raise ValueError("At least one paired seed is required.")
    selections = target_selections(settings["raw"])
    selected_files = {
        pair.seed: _selected_rollout_files(
            pair,
            pattern=rollout_pattern,
            initialization_stride=initialization_stride,
            maximum_initializations=maximum_initializations,
            initialization_selection=initialization_selection,
            strict_provenance=strict_provenance,
        )
        for pair in pairs
    }
    selected_by_seed = {seed: sorted(files["off"]) for seed, files in selected_files.items()}
    first_dates = next(iter(selected_by_seed.values()))
    if any(dates != first_dates for dates in selected_by_seed.values()):
        raise ValueError("All seeds must evaluate the identical selected initialization dates.")
    if not first_dates:
        raise ValueError("Initialization selection is empty.")
    aurora_directory = (
        Path(settings["baseline_dir"] if aurora_directory is None else aurora_directory)
        .expanduser()
        .resolve()
    )
    aurora_index = _rollout_index(aurora_directory, rollout_pattern)
    missing_aurora = [value for value in first_dates if value not in aurora_index]
    if missing_aurora:
        missing_labels = [
            np.datetime_as_string(np.datetime64(value, "ns"), unit="s")
            for value in missing_aurora[:5]
        ]
        raise ValueError(
            "Raw Aurora baseline is missing selected test initializations: " f"{missing_labels}."
        )

    first_pair = pairs[0]
    first_file = selected_files[first_pair.seed]["off"][first_dates[0]]
    with xr.open_dataset(first_file, decode_times=True) as first:
        latitude = np.asarray(first["latitude"].values, dtype=float)
        longitude = np.asarray(first["longitude"].values, dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Evaluation latitude and longitude must be one-dimensional.")
    base_weights = area_weights(latitude, longitude)
    regions: list[Region] = []
    if include_default_regions:
        regions.extend(default_regions(latitude, longitude))
    regions.extend(custom_regions)
    if not regions:
        raise ValueError("At least one evaluation region is required.")
    names = [region.name for region in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"Evaluation region names must be unique: {names}.")
    region_weights = {
        region.name: base_weights * region.mask(latitude, longitude) for region in regions
    }
    longitude_periodic = float(np.ptp(longitude)) >= 300.0 and longitude.size > 2

    tolerance = float(settings["coordinate_tolerance"])
    level_tolerance = float(settings["level_tolerance"])
    requested_leads = np.asarray(settings["lead_hours"], dtype=float)
    if requested_leads.ndim != 1 or requested_leads.size == 0:
        raise ValueError("At least one forecast lead must be requested.")
    if not np.isfinite(requested_leads).all():
        raise ValueError("Requested forecast leads must be finite.")

    field_rows: list[dict[str, Any]] = []
    tendency_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    skipped_without_truth = 0
    case_name = str(settings["case_name"])
    for pair in pairs:
        manifest_case = str(pair.manifests["off"].get("case_name", case_name))
        if strict_provenance and manifest_case != case_name:
            raise ValueError(
                f"Seed {pair.seed} manifest case {manifest_case!r} does not match "
                f"configuration case {case_name!r}."
            )
    spatial_head = str(pairs[0].manifests["off"].get("spatial_head", "unverified"))

    with xr.open_dataset(settings["truth_path"], decode_times=True) as truth:
        truth_times = np.asarray(truth["time"].values).astype("datetime64[ns]")
        if np.isnat(truth_times).any() or np.unique(truth_times).size != truth_times.size:
            raise ValueError("CAMS truth times must be unique and non-NaT.")
        truth_lookup = {
            int(value.astype(np.int64)): index for index, value in enumerate(truth_times)
        }
        truth_latitude_indices = match_coordinate_indices(
            np.asarray(truth["latitude"].values, dtype=float),
            latitude,
            tolerance,
            name="truth latitude",
        )
        truth_longitude_indices = match_coordinate_indices(
            np.asarray(truth["longitude"].values, dtype=float),
            longitude,
            tolerance,
            name="truth longitude",
        )

        for pair in pairs:
            files_by_mode = selected_files[pair.seed]
            for initialization_ns in tqdm(
                selected_by_seed[pair.seed],
                desc=f"Paired Mamba evaluation seed {pair.seed}",
                unit="initialization",
            ):
                initialization = np.datetime64(initialization_ns, "ns")
                initialization_label = np.datetime_as_string(
                    initialization,
                    unit="s",
                )
                with ExitStack() as stack:
                    case_files = {
                        mode: files[initialization_ns] for mode, files in files_by_mode.items()
                    }
                    case_files["aurora"] = aurora_index[initialization_ns]
                    datasets = {
                        mode: stack.enter_context(xr.open_dataset(path, decode_times=True))
                        for mode, path in case_files.items()
                    }
                    valid_times: dict[str, np.ndarray] = {}
                    step_dims: dict[str, str] = {}
                    coordinate_indices: dict[
                        str,
                        tuple[np.ndarray | None, np.ndarray | None],
                    ] = {}
                    for mode, dataset in datasets.items():
                        context = str(case_files[mode])
                        times, step_dim = validate_rollout_time_metadata(
                            dataset,
                            initialization,
                            context=context,
                        )
                        valid_times[mode] = times
                        step_dims[mode] = step_dim
                        if mode == "aurora":
                            coordinate_indices[mode] = (
                                match_coordinate_indices(
                                    np.asarray(dataset["latitude"].values, dtype=float),
                                    latitude,
                                    tolerance,
                                    name="Aurora latitude",
                                ),
                                match_coordinate_indices(
                                    np.asarray(dataset["longitude"].values, dtype=float),
                                    longitude,
                                    tolerance,
                                    name="Aurora longitude",
                                ),
                            )
                        else:
                            assert_same_coordinate(
                                latitude,
                                np.asarray(dataset["latitude"].values, dtype=float),
                                tolerance,
                                name="latitude",
                                context=context,
                            )
                            assert_same_coordinate(
                                longitude,
                                np.asarray(dataset["longitude"].values, dtype=float),
                                tolerance,
                                name="longitude",
                                context=context,
                            )
                            coordinate_indices[mode] = (None, None)
                    off_valid_times = valid_times["off"]
                    for mode, times in valid_times.items():
                        if mode != "aurora" and not np.array_equal(times, off_valid_times):
                            raise ValueError(
                                f"Seed {pair.seed} {mode} valid times do not exactly "
                                "match Mamba-off."
                            )

                    lead_entries: list[tuple[float, int, int]] = []
                    for off_index, valid_time in enumerate(off_valid_times):
                        lead = float((valid_time - initialization) / np.timedelta64(1, "h"))
                        if not np.any(
                            np.isclose(
                                lead,
                                requested_leads,
                                atol=1.0e-6,
                                rtol=0.0,
                            )
                        ):
                            continue
                        truth_index = truth_lookup.get(int(valid_time.astype(np.int64)))
                        if truth_index is None:
                            skipped_without_truth += 1
                            continue
                        lead_entries.append((lead, off_index, truth_index))
                    lead_entries.sort(key=lambda value: value[0])
                    if not lead_entries:
                        continue

                    for selection in selections:
                        units = str(truth[selection.variable].attrs.get("units") or "")
                        level_label = (
                            "surface" if selection.level is None else f"{selection.level:g}"
                        )
                        sequences: dict[
                            str,
                            list[tuple[float, np.ndarray, dict[str, np.ndarray]]],
                        ] = {region.name: [] for region in regions}
                        for lead, off_index, truth_index in lead_entries:
                            valid_time = off_valid_times[off_index]
                            truth_values = _select_truth_field(
                                truth,
                                selection,
                                time_index=truth_index,
                                latitude_indices=truth_latitude_indices,
                                longitude_indices=truth_longitude_indices,
                                level_tolerance=level_tolerance,
                                context=(f"CAMS {selection.variable!r} at {valid_time}"),
                            )
                            candidates = {}
                            for mode, dataset in datasets.items():
                                mode_index = match_valid_time_index(
                                    valid_times[mode],
                                    valid_time,
                                    context=str(case_files[mode]),
                                )
                                candidates[mode] = _select_field(
                                    dataset,
                                    selection,
                                    step_dim=step_dims[mode],
                                    step_index=mode_index,
                                    level_tolerance=level_tolerance,
                                    context=(f"{mode} {selection.variable!r} at " f"{valid_time}"),
                                    latitude_indices=coordinate_indices[mode][0],
                                    longitude_indices=coordinate_indices[mode][1],
                                )
                            expected_shape = (latitude.size, longitude.size)
                            if truth_values.shape != expected_shape or any(
                                value.shape != expected_shape for value in candidates.values()
                            ):
                                raise ValueError(
                                    f"{selection.label} at {valid_time} does not "
                                    f"match grid shape {expected_shape}."
                                )
                            if (
                                representative_fields is not None
                                and pair.seed == first_pair.seed
                                and initialization_ns == first_dates[0]
                            ):
                                representative_key = f"{selection.variable}_{level_label}"
                                representative = representative_fields.get(representative_key)
                                if representative is None:
                                    representative = {
                                        "case_name": case_name,
                                        "spatial_head": spatial_head,
                                        "variable": selection.variable,
                                        "level": level_label,
                                        "units": units,
                                        "initialization_time": initialization_label,
                                        "latitude": latitude.copy(),
                                        "longitude": longitude.copy(),
                                        "lead_fields": [],
                                    }
                                    representative_fields[representative_key] = representative
                                lead_fields = representative["lead_fields"]
                                if lead_fields and lead <= float(
                                    lead_fields[-1]["lead_time_hours"]
                                ):
                                    raise RuntimeError(
                                        "Representative lead fields must be captured "
                                        "in strictly chronological order."
                                    )
                                lead_record = {
                                    "valid_time": np.datetime_as_string(
                                        valid_time,
                                        unit="s",
                                    ),
                                    "lead_time_hours": lead,
                                    "fields": {
                                        "truth": truth_values.copy(),
                                        **{
                                            name: value.copy() for name, value in candidates.items()
                                        },
                                    },
                                }
                                lead_fields.append(lead_record)
                                if "lead_time_hours" not in representative or lead > float(
                                    representative["lead_time_hours"]
                                ):
                                    representative.update(lead_record)

                            base = {
                                "case_name": case_name,
                                "spatial_head": spatial_head,
                                "seed": pair.seed,
                                "initialization_time": initialization_label,
                                "initialization_time_ns": initialization_ns,
                                "valid_time": np.datetime_as_string(
                                    valid_time,
                                    unit="s",
                                ),
                                "lead_time_hours": lead,
                                "variable": selection.variable,
                                "level": level_label,
                                "units": units,
                            }
                            for region in regions:
                                weights = region_weights[region.name]
                                metrics = paired_field_metrics(
                                    truth_values,
                                    candidates,
                                    weights,
                                    longitude_periodic=longitude_periodic,
                                )
                                field_rows.append(
                                    {
                                        **base,
                                        "region": region.name,
                                        **metrics,
                                    }
                                )
                                sequences[region.name].append((lead, truth_values, candidates))

                        for region in regions:
                            entries = sequences[region.name]
                            weights = region_weights[region.name]
                            sequence_base = {
                                "case_name": case_name,
                                "spatial_head": spatial_head,
                                "seed": pair.seed,
                                "initialization_time": initialization_label,
                                "initialization_time_ns": initialization_ns,
                                "variable": selection.variable,
                                "level": level_label,
                                "units": units,
                                "region": region.name,
                            }
                            for previous, current in zip(
                                entries[:-1],
                                entries[1:],
                                strict=True,
                            ):
                                previous_lead, previous_truth, previous_candidates = previous
                                lead, truth_values, candidates = current
                                tendency_rows.append(
                                    {
                                        **sequence_base,
                                        "lead_time_hours": lead,
                                        **paired_tendency_metrics(
                                            previous_truth,
                                            previous_candidates,
                                            truth_values,
                                            candidates,
                                            weights,
                                            lead - previous_lead,
                                        ),
                                    }
                                )
                            if len(entries) >= 3:
                                temporal_rows.append(
                                    {
                                        **sequence_base,
                                        "number_of_leads": len(entries),
                                        "first_lead_time_hours": entries[0][0],
                                        "last_lead_time_hours": entries[-1][0],
                                        **temporal_sequence_metrics(
                                            np.stack([entry[1] for entry in entries]),
                                            {
                                                mode: np.stack(
                                                    [entry[2][mode] for entry in entries]
                                                )
                                                for mode in datasets
                                            },
                                            weights,
                                        ),
                                    }
                                )

    if not field_rows:
        raise ValueError("No paired forecast cases matched configured leads and CAMS truth.")
    provenance = {
        "selected_initializations_by_seed": {
            str(seed): [
                np.datetime_as_string(np.datetime64(value, "ns"), unit="s") for value in values
            ]
            for seed, values in selected_by_seed.items()
        },
        "aurora_rollout_directory": str(aurora_directory),
        "aurora_selected_files": [
            {
                "initialization_time": np.datetime_as_string(np.datetime64(value, "ns"), unit="s"),
                "path": str(aurora_index[value]),
                "size_bytes": aurora_index[value].stat().st_size,
                "mtime_ns": aurora_index[value].stat().st_mtime_ns,
            }
            for value in first_dates
        ],
        "initialization_selection": initialization_selection,
        "initialization_stride": initialization_stride,
        "maximum_initializations": maximum_initializations,
        "skipped_forecast_steps_without_truth": skipped_without_truth,
        "longitude_periodic_for_gradient": longitude_periodic,
        "regions": [
            {
                "name": region.name,
                "latitude_min": region.latitude_min,
                "latitude_max": region.latitude_max,
                "longitude_min": region.longitude_min,
                "longitude_max": region.longitude_max,
            }
            for region in regions
        ],
    }
    return (
        pd.DataFrame(field_rows),
        pd.DataFrame(tendency_rows),
        pd.DataFrame(temporal_rows),
        provenance,
    )


_SUMMARY_KEYS = (
    "case_name",
    "spatial_head",
    "variable",
    "level",
    "units",
    "region",
)


def build_evaluation_summaries(
    field_frame: pd.DataFrame,
    tendency_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    bootstrap_block_length: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    """Build lead and overall paired summaries for all available controls."""
    modes = [mode for mode in ("aurora", "off", "on", "shuffled") if f"{mode}_mae" in field_frame]
    comparisons = [("aurora", "off"), ("aurora", "on"), ("off", "on")]
    if "shuffled" in modes:
        comparisons.extend((("aurora", "shuffled"), ("off", "shuffled"), ("shuffled", "on")))

    by_lead: list[pd.DataFrame] = []
    overall: list[pd.DataFrame] = []
    random_offset = 0
    diagnostic_frames = (
        ("field", field_frame, FIELD_METRIC_SPECS, "lead_time_hours"),
        ("tendency", tendency_frame, TENDENCY_METRIC_SPECS, "lead_time_hours"),
        ("temporal_sequence", temporal_frame, TEMPORAL_METRIC_SPECS, None),
    )
    for diagnostic, frame, specs, lead_column in diagnostic_frames:
        if frame.empty:
            continue
        for reference, candidate in comparisons:
            if not any(
                f"{candidate}_{spec.name}" in frame and f"{reference}_{spec.name}" in frame
                for spec in specs
            ):
                continue
            if lead_column is not None:
                lead_summary = paired_bootstrap_summary(
                    frame,
                    specs,
                    group_keys=(*_SUMMARY_KEYS, lead_column),
                    reference=reference,
                    candidate=candidate,
                    resamples=bootstrap_resamples,
                    block_length=bootstrap_block_length,
                    seed=bootstrap_seed + random_offset,
                )
                if not lead_summary.empty:
                    lead_summary.insert(0, "diagnostic", diagnostic)
                    by_lead.append(lead_summary)
            aggregate = paired_bootstrap_summary(
                frame,
                specs,
                group_keys=_SUMMARY_KEYS,
                reference=reference,
                candidate=candidate,
                resamples=bootstrap_resamples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed + random_offset + 1,
            )
            if not aggregate.empty:
                aggregate.insert(0, "diagnostic", diagnostic)
                overall.append(aggregate)
            random_offset += 2

    fraction_columns = [
        column for column in field_frame.columns if column.startswith("fraction_cells_")
    ]
    cell_by_lead = single_metric_bootstrap_summary(
        field_frame,
        fraction_columns,
        group_keys=(*_SUMMARY_KEYS, "lead_time_hours"),
        resamples=bootstrap_resamples,
        block_length=bootstrap_block_length,
        seed=bootstrap_seed + 1000,
    )
    cell_overall = single_metric_bootstrap_summary(
        field_frame,
        fraction_columns,
        group_keys=_SUMMARY_KEYS,
        resamples=bootstrap_resamples,
        block_length=bootstrap_block_length,
        seed=bootstrap_seed + 1001,
    )
    correction_columns = [
        column for column in tendency_frame.columns if "_correction_tendency_" in column
    ]
    correction_by_lead = (
        single_metric_bootstrap_summary(
            tendency_frame,
            correction_columns,
            group_keys=(*_SUMMARY_KEYS, "lead_time_hours"),
            resamples=bootstrap_resamples,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed + 1002,
        )
        if not tendency_frame.empty
        else pd.DataFrame()
    )
    correction_overall = (
        single_metric_bootstrap_summary(
            tendency_frame,
            correction_columns,
            group_keys=_SUMMARY_KEYS,
            resamples=bootstrap_resamples,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed + 1003,
        )
        if not tendency_frame.empty
        else pd.DataFrame()
    )
    return {
        "paired_summary_by_lead": (
            pd.concat(by_lead, ignore_index=True) if by_lead else pd.DataFrame()
        ),
        "paired_summary_overall": (
            pd.concat(overall, ignore_index=True) if overall else pd.DataFrame()
        ),
        "cell_fraction_summary_by_lead": cell_by_lead,
        "cell_fraction_summary_overall": cell_overall,
        "correction_tendency_summary_by_lead": correction_by_lead,
        "correction_tendency_summary_overall": correction_overall,
    }


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
) -> list[str]:
    if frame.empty:
        return ["No matched results were available."]
    headers = [label for _, label in columns]
    rows: list[list[str]] = []
    for _, item in frame.iterrows():
        cells: list[str] = []
        for name, _ in columns:
            value = item.get(name)
            if isinstance(value, (float, np.floating)):
                cells.append("—" if not np.isfinite(value) else f"{value:.5g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        rows.append(cells)
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _target_column(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["variable"].astype(str)
        + " ("
        + frame["level"].astype(str)
        + ", "
        + frame["region"].astype(str)
        + ")"
    )


def write_evaluation_summary(
    path: Path,
    summaries: Mapping[str, pd.DataFrame],
    provenance: Mapping[str, Any],
    pairs: Sequence[StudyPair],
    *,
    strict_provenance: bool,
) -> None:
    """Write a compact human-readable scientific and provenance summary."""
    overall = summaries["paired_summary_overall"].copy()
    if not overall.empty:
        overall["target"] = _target_column(overall)
        overall["ci"] = overall.apply(
            lambda row: (
                "—"
                if not np.isfinite(row["ci_lower"])
                else f"[{row['ci_lower']:.4g}, {row['ci_upper']:.4g}]"
            ),
            axis=1,
        )
    key_metrics = {
        "mae",
        "rmse",
        "abs_bias",
        "spatial_correlation",
        "gradient_rmse",
        "wasserstein_distance",
        "p99_abs_error",
        "p99_9_abs_error",
        "tendency_rmse",
        "temporal_correlation",
        "lag1_autocorrelation_abs_error",
    }
    report_overall = overall
    if not overall.empty and "domain" in set(overall["region"]):
        report_overall = overall.loc[overall["region"] == "domain"]
    off_baseline = report_overall.loc[
        (report_overall.get("comparison") == "off_vs_aurora")
        & report_overall.get("metric", pd.Series(dtype=str)).isin(key_metrics)
    ].copy()
    on_baseline = report_overall.loc[
        (report_overall.get("comparison") == "on_vs_aurora")
        & report_overall.get("metric", pd.Series(dtype=str)).isin(key_metrics)
    ].copy()
    ordered = report_overall.loc[
        (report_overall.get("comparison") == "on_vs_off")
        & report_overall.get("metric", pd.Series(dtype=str)).isin(key_metrics)
    ].copy()
    shuffled = report_overall.loc[
        (report_overall.get("comparison") == "on_vs_shuffled")
        & report_overall.get("metric", pd.Series(dtype=str)).isin(key_metrics)
    ].copy()

    correction = summaries["correction_tendency_summary_overall"].copy()
    if not correction.empty:
        correction["target"] = _target_column(correction)
        if "domain" in set(correction["region"]):
            correction = correction.loc[correction["region"] == "domain"]
        correction["ci"] = correction.apply(
            lambda row: (
                "—"
                if not np.isfinite(row["ci_lower"])
                else f"[{row['ci_lower']:.4g}, {row['ci_upper']:.4g}]"
            ),
            axis=1,
        )

    selected = provenance["selected_initializations_by_seed"]
    first_seed_dates = next(iter(selected.values()))
    raw_link_verified = all(
        pair.manifests["off"]["phase1_rollout_linkage"].get("status") == "proven" for pair in pairs
    )
    lines = [
        "# Paired temporal-Mamba ablation",
        "",
        (
            "Positive improvement means the candidate is better. Confidence "
            "intervals use a paired hierarchical bootstrap: seeds are resampled, "
            "then circular chronological initialization blocks within each seed—not "
            "pixels. RMSE-family point estimates and intervals pool squared errors "
            "before taking a square root."
        ),
        "",
        "## Study design and provenance",
        "",
        f"- Provenance enforcement: {'strict' if strict_provenance else 'UNSAFE / exploratory'}",
        f"- Seeds: {', '.join(str(pair.seed) for pair in pairs)}",
        f"- Selected initializations per seed: {len(first_seed_dates)}",
        f"- Selected date range: {first_seed_dates[0]} through {first_seed_dates[-1]}",
        f"- Initialization selection: {provenance['initialization_selection']}",
        f"- Initialization stride: {provenance['initialization_stride']}",
        f"- Raw Aurora baseline directory: {provenance['aurora_rollout_directory']}",
        (
            "- Raw-rollout corpus fingerprint agrees across arms and seeds: "
            f"{pairs[0].manifests['off']['raw_rollout_corpus_sha256']}"
        ),
        (
            "- Raw-rollout fingerprint method: "
            f"{pairs[0].manifests['off']['raw_rollout_corpus_hash_method']}"
        ),
        (
            "- Data split hash agrees across arms and seeds: "
            f"{pairs[0].manifests['off']['data_split_sha256']}"
        ),
        (
            "- Raw-rollout-to-phase-1 checkpoint linkage status: "
            f"{pairs[0].manifests['off']['phase1_rollout_linkage']['status']}"
        ),
        (
            "- Trajectory policy: "
            + json.dumps(
                pairs[0].manifests["off"]["trajectory_policy"],
                sort_keys=True,
            )
        ),
        "",
    ]
    if not raw_link_verified:
        lines.extend(
            [
                "> Provenance limitation: paired manifests declare the same raw-rollout "
                "corpus hash, but they do not independently prove that legacy raw "
                "rollouts were generated from the supplied phase-1 checkpoint. Do not "
                "claim that linkage without a source manifest carrying the checkpoint "
                "hash.",
                "",
            ]
        )
    lines.extend(
        [
            "## Mamba-off versus raw Aurora",
            "",
            *_markdown_table(
                off_baseline,
                (
                    ("target", "Target"),
                    ("metric_label", "Metric"),
                    ("reference_mean", "Aurora"),
                    ("candidate_mean", "Off"),
                    ("mean_improvement", "Improvement"),
                    ("ci", "95% CI"),
                    ("fraction_cases_candidate_better", "Cases better"),
                    ("fraction_cases_candidate_worse", "Cases worse"),
                ),
            ),
            "",
            "## Ordered Mamba-on versus raw Aurora",
            "",
            *_markdown_table(
                on_baseline,
                (
                    ("target", "Target"),
                    ("metric_label", "Metric"),
                    ("reference_mean", "Aurora"),
                    ("candidate_mean", "On"),
                    ("mean_improvement", "Improvement"),
                    ("ci", "95% CI"),
                    ("fraction_cases_candidate_better", "Cases better"),
                    ("fraction_cases_candidate_worse", "Cases worse"),
                ),
            ),
            "",
            "## Ordered Mamba-on versus Mamba-off",
            "",
            *_markdown_table(
                ordered,
                (
                    ("target", "Target"),
                    ("metric_label", "Metric"),
                    ("reference_mean", "Off"),
                    ("candidate_mean", "On"),
                    ("mean_improvement", "Improvement"),
                    ("ci", "95% CI"),
                    ("fraction_cases_candidate_better", "Cases better"),
                    ("fraction_cases_candidate_worse", "Cases worse"),
                ),
            ),
            "",
        ]
    )
    if not shuffled.empty:
        lines.extend(
            [
                "## Ordered versus shuffled lead history",
                "",
                *_markdown_table(
                    shuffled,
                    (
                        ("target", "Target"),
                        ("metric_label", "Metric"),
                        ("reference_mean", "Shuffled"),
                        ("candidate_mean", "Ordered"),
                        ("mean_improvement", "Improvement"),
                        ("ci", "95% CI"),
                    ),
                ),
                "",
            ]
        )
    if not correction.empty:
        lines.extend(
            [
                "## Predicted-correction tendency versus true-residual tendency",
                "",
                *_markdown_table(
                    correction,
                    (
                        ("target", "Target"),
                        ("metric", "Metric"),
                        ("mean", "Mean"),
                        ("ci", "95% CI"),
                    ),
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- A confidence interval spanning zero is inconclusive, not evidence of improvement.",
            (
                "- Ordered-on should beat both off and shuffled history before "
                "attributing skill to temporal ordering."
            ),
            "- Review case-worse and cell-worse fractions alongside aggregate errors.",
            "- The held-out test set must not be used for temporal checkpoint selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def plot_lead_improvements(
    summary: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """Plot positive-is-better paired improvements and confidence bands by lead."""
    if summary.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_metrics = (
        "rmse",
        "spatial_correlation",
        "gradient_rmse",
        "wasserstein_distance",
        "p99_9_abs_error",
        "tendency_rmse",
    )
    labels = {
        "rmse": "RMSE",
        "spatial_correlation": "Spatial correlation",
        "gradient_rmse": "Gradient RMSE",
        "wasserstein_distance": "Wasserstein-1",
        "p99_9_abs_error": "P99.9 error",
        "tendency_rmse": "Tendency RMSE",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    group_keys = (
        "comparison",
        "case_name",
        "spatial_head",
        "variable",
        "level",
        "region",
    )
    allowed_comparisons = {
        "off_vs_aurora",
        "on_vs_aurora",
        "on_vs_off",
        "on_vs_shuffled",
    }
    selected_region = (
        "domain"
        if "domain" in set(summary["region"])
        else sorted(set(summary["region"].astype(str)))[0]
    )
    useful = summary.loc[
        summary["metric"].isin(selected_metrics)
        & summary["comparison"].isin(allowed_comparisons)
        & (summary["region"].astype(str) == selected_region)
    ]
    for group_key, group in useful.groupby(
        list(group_keys),
        dropna=False,
        sort=True,
    ):
        (
            comparison,
            case_name,
            spatial_head,
            variable,
            level,
            region,
        ) = group_key
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for axis, metric in zip(axes.flat, selected_metrics, strict=True):
            values = group.loc[group["metric"] == metric].sort_values("lead_time_hours")
            if values.empty:
                axis.set_visible(False)
                continue
            x = values["lead_time_hours"].to_numpy(dtype=float)
            y = values["mean_improvement"].to_numpy(dtype=float)
            lower = values["ci_lower"].to_numpy(dtype=float)
            upper = values["ci_upper"].to_numpy(dtype=float)
            axis.plot(x, y, marker="o")
            finite = np.isfinite(lower) & np.isfinite(upper)
            if np.any(finite):
                axis.fill_between(
                    x[finite],
                    lower[finite],
                    upper[finite],
                    alpha=0.2,
                )
            axis.axhline(0.0, color="black", linewidth=0.9)
            axis.set_title(labels[metric])
            axis.set_xlabel("Forecast lead (hours)")
            axis.set_ylabel("Positive-is-better improvement")
            axis.grid(alpha=0.2)
        fig.suptitle(f"{case_name}: {spatial_head}, {variable} {level}, {region}\n" f"{comparison}")
        filename = (
            f"{file_safe(str(comparison))}_{file_safe(str(variable))}_"
            f"{file_safe(str(level))}_{file_safe(str(region))}.png"
        )
        path = output_directory / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_limits(
    arrays: Sequence[np.ndarray],
    *,
    symmetric: bool,
) -> tuple[float, float]:
    finite_arrays = [
        np.asarray(array, dtype=np.float64)[np.isfinite(array)]
        for array in arrays
        if np.any(np.isfinite(array))
    ]
    if not finite_arrays:
        return -1.0, 1.0
    values = np.concatenate(finite_arrays)
    if symmetric:
        extent = float(np.percentile(np.abs(values), 99.0))
        if extent == 0.0:
            extent = float(np.spacing(1.0))
        return -extent, extent
    lower, upper = np.percentile(values, [1.0, 99.0])
    if lower == upper:
        delta = float(np.spacing(abs(lower) if lower != 0.0 else 1.0))
        return float(lower - delta), float(upper + delta)
    return float(lower), float(upper)


def plot_representative_maps(
    representative_fields: Mapping[str, Mapping[str, Any]],
    output_directory: Path,
) -> list[Path]:
    """Plot one longest-lead CAMS/Aurora/off/on map and difference panel per target."""
    if not representative_fields:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for target, record in sorted(representative_fields.items()):
        fields = record["fields"]
        required = ("truth", "aurora", "off", "on")
        if any(name not in fields for name in required):
            continue
        absolute = [np.asarray(fields[name], dtype=float) for name in required]
        differences = [
            absolute[1] - absolute[0],
            absolute[2] - absolute[0],
            absolute[3] - absolute[0],
            absolute[3] - absolute[2],
        ]
        absolute_limits = _plot_limits(absolute, symmetric=False)
        difference_limits = _plot_limits(differences, symmetric=True)
        longitude = np.asarray(record["longitude"], dtype=float)
        latitude = np.asarray(record["latitude"], dtype=float)
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
        for axis, name, values in zip(
            axes[0],
            ("CAMS truth", "Raw Aurora", "Mamba-off", "Mamba-on"),
            absolute,
            strict=True,
        ):
            image = axis.pcolormesh(
                longitude,
                latitude,
                values,
                shading="auto",
                cmap="viridis",
                vmin=absolute_limits[0],
                vmax=absolute_limits[1],
            )
            axis.set_title(name)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            fig.colorbar(image, ax=axis, shrink=0.8)
        for axis, name, values in zip(
            axes[1],
            ("Aurora − CAMS", "Off − CAMS", "On − CAMS", "On − Off"),
            differences,
            strict=True,
        ):
            image = axis.pcolormesh(
                longitude,
                latitude,
                values,
                shading="auto",
                cmap="RdBu_r",
                vmin=difference_limits[0],
                vmax=difference_limits[1],
            )
            axis.set_title(name)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            fig.colorbar(image, ax=axis, shrink=0.8)
        fig.suptitle(
            f"{record['case_name']}: {record['variable']} {record['level']}, "
            f"lead {record['lead_time_hours']:g} h\n"
            f"initialization {record['initialization_time']}; units {record['units'] or 'unknown'}"
        )
        path = output_directory / f"representative_map_{file_safe(target)}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _hotspot_trajectory(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a chronological series at the longest-lead CAMS maximum."""
    raw_leads = record.get("lead_fields")
    if not isinstance(raw_leads, list) or not raw_leads:
        return None
    leads = sorted(raw_leads, key=lambda item: float(item["lead_time_hours"]))
    lead_hours = np.asarray(
        [float(item["lead_time_hours"]) for item in leads],
        dtype=np.float64,
    )
    if lead_hours.size > 1 and np.any(np.diff(lead_hours) <= 0.0):
        raise ValueError("Representative lead times must be unique and chronological.")
    longest_truth = np.asarray(leads[-1]["fields"]["truth"], dtype=np.float64)
    finite = np.isfinite(longest_truth)
    if longest_truth.ndim != 2 or not np.any(finite):
        return None
    hotspot_flat = int(np.argmax(np.where(finite, longest_truth, -np.inf)))
    latitude_index, longitude_index = np.unravel_index(
        hotspot_flat,
        longest_truth.shape,
    )
    latitude = np.asarray(record["latitude"], dtype=np.float64)
    longitude = np.asarray(record["longitude"], dtype=np.float64)
    if longest_truth.shape != (latitude.size, longitude.size):
        raise ValueError("Representative fields do not match their latitude/longitude grid.")

    values: dict[str, np.ndarray] = {}
    for mode in ("truth", "aurora", "off", "on", "shuffled"):
        if not all(mode in item["fields"] for item in leads):
            continue
        fields = [np.asarray(item["fields"][mode], dtype=np.float64) for item in leads]
        if any(field.shape != longest_truth.shape for field in fields):
            raise ValueError("Representative lead fields must retain one common grid.")
        values[mode] = np.asarray(
            [field[latitude_index, longitude_index] for field in fields],
            dtype=np.float64,
        )
    return {
        "latitude_index": int(latitude_index),
        "longitude_index": int(longitude_index),
        "latitude": float(latitude[latitude_index]),
        "longitude": float(longitude[longitude_index]),
        "lead_time_hours": lead_hours,
        "values": values,
    }


def plot_hotspot_trajectories(
    representative_fields: Mapping[str, Mapping[str, Any]],
    output_directory: Path,
) -> list[Path]:
    """Plot each candidate at the longest-lead CAMS-maximum grid cell."""
    if not representative_fields:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    labels = {
        "truth": "CAMS",
        "aurora": "Raw Aurora",
        "off": "Mamba-off",
        "on": "Mamba-on",
        "shuffled": "Shuffled history",
    }
    paths: list[Path] = []
    for target, record in sorted(representative_fields.items()):
        trajectory = _hotspot_trajectory(record)
        if trajectory is None:
            continue
        fig, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
        for mode in ("truth", "aurora", "off", "on", "shuffled"):
            values = trajectory["values"].get(mode)
            if values is None:
                continue
            axis.plot(
                trajectory["lead_time_hours"],
                values,
                marker="o",
                label=labels[mode],
            )
        axis.set_xlabel("Forecast lead (hours)")
        axis.set_ylabel(str(record["units"]) or "value")
        axis.grid(alpha=0.2)
        axis.legend()
        axis.set_title(
            f"{record['case_name']}: {record['variable']} {record['level']}\n"
            "CAMS maximum at longest lead; "
            f"lat {trajectory['latitude']:.3f}, lon {trajectory['longitude']:.3f}"
        )
        path = output_directory / f"hotspot_trajectory_{file_safe(target)}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_temporal_trajectories(
    field_frame: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """Plot domain-mean evolution and RMSE for one representative initialization."""
    if field_frame.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_region = (
        "domain"
        if "domain" in set(field_frame["region"])
        else sorted(set(field_frame["region"].astype(str)))[0]
    )
    selected_seed = sorted(set(field_frame["seed"]))[0]
    seed_frame = field_frame.loc[
        (field_frame["seed"] == selected_seed)
        & (field_frame["region"].astype(str) == selected_region)
    ]
    selected_initialization = sorted(set(seed_frame["initialization_time"]))[0]
    representative = seed_frame.loc[seed_frame["initialization_time"] == selected_initialization]
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (variable, level), group in representative.groupby(
        ["variable", "level"],
        dropna=False,
        sort=True,
    ):
        group = group.sort_values("lead_time_hours")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        axes[0].plot(
            group["lead_time_hours"],
            group["truth_mean"],
            marker="o",
            label="CAMS",
        )
        for mode, label in (
            ("aurora", "Raw Aurora"),
            ("off", "Mamba-off"),
            ("on", "Mamba-on"),
        ):
            column = f"{mode}_mean"
            if column in group:
                axes[0].plot(
                    group["lead_time_hours"],
                    group[column],
                    marker="o",
                    label=label,
                )
            rmse_column = f"{mode}_rmse"
            if rmse_column in group:
                axes[1].plot(
                    group["lead_time_hours"],
                    group[rmse_column],
                    marker="o",
                    label=label,
                )
        axes[0].set_title("Area-weighted domain mean")
        axes[0].set_ylabel(str(group["units"].iloc[0]) or "value")
        axes[1].set_title("Area-weighted RMSE")
        axes[1].set_ylabel(str(group["units"].iloc[0]) or "value")
        for axis in axes:
            axis.set_xlabel("Forecast lead (hours)")
            axis.grid(alpha=0.2)
            axis.legend()
        fig.suptitle(
            f"{group['case_name'].iloc[0]}: {variable} {level}; "
            f"initialization {selected_initialization}"
        )
        path = output_directory / (
            f"temporal_trajectory_{file_safe(str(variable))}_" f"{file_safe(str(level))}.png"
        )
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def evaluate_mamba_ablation(
    config_path: Path,
    off_directories: Sequence[Path],
    on_directories: Sequence[Path],
    shuffled_directories: Sequence[Path],
    phase1_checkpoints: Sequence[Path],
    output_directory: Path,
    *,
    rollout_pattern: str = "*.nc",
    aurora_directory: Path | None = None,
    initialization_stride: int = 1,
    maximum_initializations: int | None = None,
    initialization_selection: str = "uniform",
    custom_regions: Sequence[Region] = (),
    include_default_regions: bool = True,
    bootstrap_resamples: int = 2000,
    bootstrap_block_length: int = 6,
    bootstrap_seed: int = 0,
    unsafe_skip_provenance_checks: bool = False,
    make_plots: bool = True,
) -> dict[str, Path]:
    """Run the complete strict paired evaluation and write reusable artifacts."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least one.")
    if bootstrap_block_length < 1:
        raise ValueError("bootstrap_block_length must be at least one.")
    config_path = Path(config_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. "
            "Use a new path; paired evaluation never overwrites prior evidence."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    settings = load_settings(config_path)
    pairs = build_study_pairs(
        off_directories,
        on_directories,
        shuffled_directories,
        phase1_checkpoints,
        unsafe_skip_provenance_checks=unsafe_skip_provenance_checks,
    )
    representative_fields: dict[str, dict[str, Any]] = {}
    field_frame, tendency_frame, temporal_frame, provenance = collect_paired_metrics(
        settings,
        pairs,
        rollout_pattern=rollout_pattern,
        initialization_stride=initialization_stride,
        maximum_initializations=maximum_initializations,
        initialization_selection=initialization_selection,
        custom_regions=custom_regions,
        include_default_regions=include_default_regions,
        strict_provenance=not unsafe_skip_provenance_checks,
        aurora_directory=aurora_directory,
        representative_fields=representative_fields,
    )
    summaries = build_evaluation_summaries(
        field_frame,
        tendency_frame,
        temporal_frame,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_block_length=bootstrap_block_length,
        bootstrap_seed=bootstrap_seed,
    )

    paths: dict[str, Path] = {}
    frames = {
        "metrics_by_case": field_frame,
        "tendency_metrics_by_case": tendency_frame,
        "temporal_sequence_metrics_by_case": temporal_frame,
        **summaries,
    }
    for name, frame in frames.items():
        path = output_directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    summary_path = output_directory / "evaluation_summary.md"
    write_evaluation_summary(
        summary_path,
        summaries,
        provenance,
        pairs,
        strict_provenance=not unsafe_skip_provenance_checks,
    )
    paths["evaluation_summary"] = summary_path
    figure_paths: list[Path] = []
    if make_plots:
        figure_directory = output_directory / "figures"
        figure_paths.extend(
            plot_lead_improvements(
                summaries["paired_summary_by_lead"],
                figure_directory,
            )
        )
        figure_paths.extend(plot_representative_maps(representative_fields, figure_directory))
        figure_paths.extend(plot_hotspot_trajectories(representative_fields, figure_directory))
        figure_paths.extend(plot_temporal_trajectories(field_frame, figure_directory))

    raw_rollout_link_verified = all(
        pair.manifests["off"]["phase1_rollout_linkage"].get("status") == "proven" for pair in pairs
    )
    run_records = []
    for pair in pairs:
        run_records.append(
            {
                "seed": pair.seed,
                "phase1_checkpoint": str(pair.phase1_checkpoint),
                "phase1_checkpoint_sha256": pair.phase1_checkpoint_sha256,
                "spatial_checkpoint": (
                    None if pair.spatial_checkpoint is None else str(pair.spatial_checkpoint)
                ),
                "spatial_checkpoint_sha256": pair.spatial_checkpoint_sha256,
                "phase2_checkpoint": (
                    None if pair.phase2_checkpoint is None else str(pair.phase2_checkpoint)
                ),
                "phase2_checkpoint_sha256": pair.phase2_checkpoint_sha256,
                "directories": {
                    mode: str(directory) for mode, directory in pair.directories.items()
                },
                "manifests": {
                    mode: {
                        "path": manifest.get("_manifest_path"),
                        "sha256": manifest.get("_manifest_sha256"),
                        "phase2_checkpoint_sha256": manifest.get("phase2_checkpoint_sha256"),
                    }
                    for mode, manifest in pair.manifests.items()
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "truth_path": str(settings["truth_path"]),
        "case_name": settings["case_name"],
        "spatial_head": pairs[0]
        .manifests["off"]
        .get(
            "spatial_head",
            "unverified",
        ),
        "strict_provenance": not unsafe_skip_provenance_checks,
        "raw_rollout_corpus_sha256": pairs[0].manifests["off"]["raw_rollout_corpus_sha256"],
        "raw_rollout_corpus_hash_method": pairs[0].manifests["off"][
            "raw_rollout_corpus_hash_method"
        ],
        "phase1_rollout_linkage": pairs[0].manifests["off"]["phase1_rollout_linkage"],
        "data_split_sha256": pairs[0].manifests["off"]["data_split_sha256"],
        "trajectory_policy": pairs[0].manifests["off"]["trajectory_policy"],
        "raw_rollout_phase1_link_verified": raw_rollout_link_verified,
        "aurora_baseline": {
            "directory": provenance["aurora_rollout_directory"],
            "selected_files": provenance["aurora_selected_files"],
        },
        "bootstrap": {
            "unit": "seed_and_forecast_initialization_id",
            "method": "paired_hierarchical_seed_then_circular_moving_block",
            "resamples": bootstrap_resamples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
        "selection": provenance,
        "paired_runs": run_records,
        "output_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "figures": [{"path": str(path), "sha256": sha256_file(path)} for path in figure_paths],
    }
    manifest_path = output_directory / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["evaluation_manifest"] = manifest_path
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--off-dir",
        required=True,
        action="append",
        type=Path,
        help="Mamba-off rollout directory; repeat once per seed.",
    )
    parser.add_argument(
        "--on-dir",
        required=True,
        action="append",
        type=Path,
        help="Ordered Mamba-on rollout directory; repeat once per seed.",
    )
    parser.add_argument(
        "--shuffled-dir",
        action="append",
        default=[],
        type=Path,
        help="Shuffled-history rollout directory; omit or repeat once per seed.",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        required=True,
        action="append",
        type=Path,
        help=(
            "Phase-1 artifact asserted by a pair; repeat in --off-dir order. "
            "Its bytes are hashed and checked against every arm manifest, but "
            "this alone does not prove legacy raw-rollout lineage."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rollout-pattern", default="*.nc")
    parser.add_argument(
        "--aurora-dir",
        type=Path,
        help=(
            "Raw Aurora rollout corpus. Defaults to evaluation.baseline_rollout_path "
            "from the case YAML; selected dates/leads are matched exactly without "
            "interpolation."
        ),
    )
    parser.add_argument(
        "--initialization-stride",
        type=int,
        default=1,
        help=(
            "Evaluate every Nth chronological held-out initialization. Use 6 for "
            "a practical season-spanning global-O3 pass."
        ),
    )
    parser.add_argument(
        "--max-initializations",
        type=int,
        help="Optional cap applied reproducibly after striding.",
    )
    parser.add_argument(
        "--initialization-selection",
        choices=("uniform", "first"),
        default="uniform",
        help=(
            "Capped selection policy. Uniform spans all eligible dates; first is "
            "provided only for legacy reproduction."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        type=parse_region,
        default=[],
        metavar="NAME,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX",
    )
    parser.add_argument(
        "--no-default-regions",
        action="store_true",
        help="Evaluate only explicitly supplied --region boxes.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--bootstrap-block-length",
        type=int,
        default=6,
        help=(
            "Initialization block length. Use 1 when initialization-stride already "
            "makes cases approximately independent."
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/Markdown/JSON without lead, map, or trajectory PNGs.",
    )
    parser.add_argument(
        "--unsafe-skip-provenance-checks",
        action="store_true",
        help=(
            "Exploratory only: bypass strict arm manifests. Alignment checks still "
            "run and the report is visibly marked unsafe."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.unsafe_skip_provenance_checks:
        print(
            "WARNING: provenance checks are disabled; results cannot support a "
            "controlled Mamba attribution claim.",
            file=sys.stderr,
        )
    paths = evaluate_mamba_ablation(
        args.config,
        args.off_dir,
        args.on_dir,
        args.shuffled_dir,
        args.phase1_checkpoint,
        args.output_dir,
        rollout_pattern=args.rollout_pattern,
        aurora_directory=args.aurora_dir,
        initialization_stride=args.initialization_stride,
        maximum_initializations=args.max_initializations,
        initialization_selection=args.initialization_selection,
        custom_regions=args.region,
        include_default_regions=not args.no_default_regions,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_seed=args.bootstrap_seed,
        unsafe_skip_provenance_checks=args.unsafe_skip_provenance_checks,
        make_plots=not args.skip_plots,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
