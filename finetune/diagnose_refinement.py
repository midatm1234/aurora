"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Diagnose whether CAMS refinement corrects or amplifies baseline residuals.

The script aligns ground truth, baseline rollouts, and refined rollouts using
valid time and named coordinates. It writes per-case and per-lead metrics,
aggregate correction maps, and predicted-versus-true correction scatterplots.
No interpolation or array resizing is performed.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    robust_limits,
    rollout_files_by_initialization,
    select_level_index,
    target_selections,
    validate_rollout_time_metadata,
)
from finetune.refinement.io import resolve_forecast_variable


def _pyplot():
    """Import optional plotting support only when a figure is requested."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError(
            "Matplotlib is required only to render refinement diagnostic figures."
        ) from exc
    return plt


@dataclass
class SpatialSums:
    """Streaming sums for aggregate diagnostic maps."""

    truth: np.ndarray
    baseline: np.ndarray
    refined: np.ndarray
    baseline_error: np.ndarray
    refined_error: np.ndarray
    true_residual: np.ndarray
    predicted_correction: np.ndarray
    remaining_residual: np.ndarray
    count: np.ndarray
    forecasts: int = 0

    @classmethod
    def create(cls, shape: tuple[int, int]) -> SpatialSums:
        def zeros() -> np.ndarray:
            return np.zeros(shape, dtype=np.float64)

        return cls(
            truth=zeros(),
            baseline=zeros(),
            refined=zeros(),
            baseline_error=zeros(),
            refined_error=zeros(),
            true_residual=zeros(),
            predicted_correction=zeros(),
            remaining_residual=zeros(),
            count=np.zeros(shape, dtype=np.int64),
        )

    def update(
        self,
        truth: np.ndarray,
        baseline: np.ndarray,
        refined: np.ndarray,
    ) -> None:
        mask = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(refined)
        if not np.any(mask):
            return
        baseline_error = baseline - truth
        refined_error = refined - truth
        true_residual = truth - baseline
        predicted_correction = refined - baseline
        remaining_residual = truth - refined
        for accumulator, values in (
            (self.truth, truth),
            (self.baseline, baseline),
            (self.refined, refined),
            (self.baseline_error, baseline_error),
            (self.refined_error, refined_error),
            (self.true_residual, true_residual),
            (self.predicted_correction, predicted_correction),
            (self.remaining_residual, remaining_residual),
        ):
            accumulator[mask] += values[mask]
        self.count[mask] += 1
        self.forecasts += 1

    def means(self) -> dict[str, np.ndarray]:
        """Return mean fields, with NaN where no triplet was valid."""
        result: dict[str, np.ndarray] = {}
        for name in (
            "truth",
            "baseline",
            "refined",
            "baseline_error",
            "refined_error",
            "true_residual",
            "predicted_correction",
            "remaining_residual",
        ):
            values = np.full(self.count.shape, np.nan, dtype=np.float64)
            np.divide(
                getattr(self, name),
                self.count,
                out=values,
                where=self.count > 0,
            )
            result[name] = values
        return result


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


_QUANTILES = (90, 95, 99)


def _skewness(values: np.ndarray) -> float:
    """Population skewness without a physical-scale-dependent variance floor."""
    centered = values - np.mean(values)
    variance = float(np.mean(centered**2))
    if values.size < 3 or variance <= 0.0:
        return math.nan
    return float(np.mean(centered**3) / variance**1.5)


def _excess_kurtosis(values: np.ndarray) -> float:
    """Population excess kurtosis with scale-independent degeneracy handling."""
    if values.size < 4:
        return math.nan
    centered = values - np.mean(values)
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        return math.nan
    normalized = centered / scale
    variance = float(np.mean(normalized**2))
    return float(np.mean(normalized**4) / variance**2 - 3.0)


def _distribution_metrics(
    prefix: str,
    candidate: np.ndarray,
    truth: np.ndarray,
    truth_quantiles: dict[int, float],
) -> dict[str, float]:
    """Candidate distribution/tail diagnostics relative to the same truth cells."""
    result = {
        f"{prefix}_wasserstein_distance": float(
            np.mean(np.abs(np.sort(candidate) - np.sort(truth)))
        ),
        f"{prefix}_skewness": _skewness(candidate),
        f"{prefix}_excess_kurtosis": _excess_kurtosis(candidate),
    }
    for percentile in _QUANTILES:
        truth_quantile = truth_quantiles[percentile]
        candidate_quantile = float(np.percentile(candidate, percentile))
        truth_tail = truth >= truth_quantile
        candidate_frequency = float(np.mean(candidate >= truth_quantile))
        truth_frequency = float(np.mean(truth_tail))
        result.update(
            {
                f"{prefix}_p{percentile}": candidate_quantile,
                f"{prefix}_p{percentile}_bias": candidate_quantile - truth_quantile,
                f"{prefix}_p{percentile}_error": abs(candidate_quantile - truth_quantile),
                f"{prefix}_p{percentile}_tail_mae": float(
                    np.mean(np.abs(candidate[truth_tail] - truth[truth_tail]))
                ),
                f"{prefix}_p{percentile}_exceedance_frequency": candidate_frequency,
                f"{prefix}_p{percentile}_exceedance_frequency_bias": (
                    candidate_frequency - truth_frequency
                ),
            }
        )
    return result


def case_metrics(
    truth: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
) -> dict[str, float | int]:
    """Calculate magnitude, pattern, and correction diagnostics for one field."""
    mask = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(refined)
    truth = truth[mask].astype(np.float64)
    baseline = baseline[mask].astype(np.float64)
    refined = refined[mask].astype(np.float64)
    if truth.size == 0:
        return {"number_of_valid_points": 0}

    baseline_error = baseline - truth
    refined_error = refined - truth
    true_residual = truth - baseline
    correction = refined - baseline
    baseline_mse = float(np.mean(baseline_error**2))
    refined_mse = float(np.mean(refined_error**2))
    residual_rms = math.sqrt(float(np.mean(true_residual**2)))
    correction_rms = math.sqrt(float(np.mean(correction**2)))
    nonzero_sign = (np.abs(true_residual) > 0) | (np.abs(correction) > 0)
    sign_accuracy = (
        float(
            np.mean(np.signbit(correction[nonzero_sign]) == np.signbit(true_residual[nonzero_sign]))
        )
        if np.any(nonzero_sign)
        else math.nan
    )
    correction_energy = float(np.sum(correction**2))
    optimal_scale = (
        float(np.sum(correction * true_residual) / correction_energy)
        if correction_energy > 0
        else math.nan
    )

    baseline_absolute_error = np.abs(baseline_error)
    refined_absolute_error = np.abs(refined_error)
    comparison_tolerance = np.finfo(np.float64).eps * np.maximum(1.0, baseline_absolute_error)
    improved = refined_absolute_error < baseline_absolute_error - comparison_tolerance
    worsened = refined_absolute_error > baseline_absolute_error + comparison_tolerance

    truth_quantiles = {
        percentile: float(np.percentile(truth, percentile)) for percentile in _QUANTILES
    }
    distribution = {
        "truth_skewness": _skewness(truth),
        "truth_excess_kurtosis": _excess_kurtosis(truth),
        **{f"truth_p{percentile}": value for percentile, value in truth_quantiles.items()},
        **{
            f"truth_p{percentile}_exceedance_frequency": float(np.mean(truth >= value))
            for percentile, value in truth_quantiles.items()
        },
    }
    distribution.update(_distribution_metrics("baseline", baseline, truth, truth_quantiles))
    distribution.update(_distribution_metrics("refined", refined, truth, truth_quantiles))

    return {
        "number_of_valid_points": int(truth.size),
        "baseline_bias": float(np.mean(baseline_error)),
        "refined_bias": float(np.mean(refined_error)),
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "refined_mae": float(np.mean(np.abs(refined_error))),
        "baseline_mse": baseline_mse,
        "refined_mse": refined_mse,
        "baseline_rmse": math.sqrt(baseline_mse),
        "refined_rmse": math.sqrt(refined_mse),
        "baseline_centered_rmse": float(np.std(baseline_error)),
        "refined_centered_rmse": float(np.std(refined_error)),
        "baseline_spatial_correlation": _correlation(truth, baseline),
        "refined_spatial_correlation": _correlation(truth, refined),
        "truth_spatial_std": float(np.std(truth)),
        "baseline_spatial_std": float(np.std(baseline)),
        "refined_spatial_std": float(np.std(refined)),
        "baseline_amplitude_ratio": (
            float(np.std(baseline) / np.std(truth)) if np.std(truth) > 0 else math.nan
        ),
        "refined_amplitude_ratio": (
            float(np.std(refined) / np.std(truth)) if np.std(truth) > 0 else math.nan
        ),
        "correction_to_residual_correlation": _correlation(
            correction,
            true_residual,
        ),
        "correction_amplitude_ratio": (
            correction_rms / residual_rms if residual_rms > 0 else math.nan
        ),
        "correction_sign_accuracy": sign_accuracy,
        "fraction_points_improved": float(np.mean(improved)),
        "fraction_points_worsened": float(np.mean(worsened)),
        "fraction_points_unchanged": float(np.mean(~improved & ~worsened)),
        "optimal_correction_scale": optimal_scale,
        **distribution,
    }


_TEMPORAL_LINEAR_METRICS = (
    "temporal_lead_interval_hours",
    "truth_tendency_spatial_std",
    "baseline_tendency_spatial_std",
    "refined_tendency_spatial_std",
    "baseline_tendency_spatial_correlation",
    "refined_tendency_spatial_correlation",
    "baseline_tendency_std_ratio",
    "refined_tendency_std_ratio",
    "correction_tendency_to_residual_tendency_correlation",
    "correction_tendency_amplitude_ratio",
)
_TEMPORAL_MEAN_SQUARE_METRICS = (
    "truth_tendency_mean_square",
    "baseline_tendency_mean_square",
    "refined_tendency_mean_square",
    "baseline_tendency_mse",
    "refined_tendency_mse",
)


def _empty_temporal_metrics() -> dict[str, float | int]:
    """Return the temporal schema for a lead without a valid predecessor."""
    return {
        "number_of_temporal_valid_points": 0,
        **{name: math.nan for name in (*_TEMPORAL_LINEAR_METRICS, *_TEMPORAL_MEAN_SQUARE_METRICS)},
        "truth_tendency_rms": math.nan,
        "baseline_tendency_rms": math.nan,
        "refined_tendency_rms": math.nan,
        "baseline_tendency_rmse": math.nan,
        "refined_tendency_rmse": math.nan,
    }


def temporal_case_metrics(
    previous_truth: np.ndarray,
    previous_baseline: np.ndarray,
    previous_refined: np.ndarray,
    truth: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    *,
    lead_interval_hours: float,
) -> dict[str, float | int]:
    """Compare per-hour changes between two consecutive valid forecast leads."""
    if not np.isfinite(lead_interval_hours) or lead_interval_hours <= 0.0:
        raise ValueError("lead_interval_hours must be finite and strictly positive.")
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            previous_truth,
            previous_baseline,
            previous_refined,
            truth,
            baseline,
            refined,
        )
    )
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError("Consecutive temporal fields must have identical shapes.")
    mask = np.logical_and.reduce(tuple(np.isfinite(values) for values in arrays))
    result = _empty_temporal_metrics()
    if not np.any(mask):
        return result

    previous_truth, previous_baseline, previous_refined, truth, baseline, refined = (
        values[mask] for values in arrays
    )
    scale = 1.0 / lead_interval_hours
    truth_tendency = (truth - previous_truth) * scale
    baseline_tendency = (baseline - previous_baseline) * scale
    refined_tendency = (refined - previous_refined) * scale
    baseline_tendency_error = baseline_tendency - truth_tendency
    refined_tendency_error = refined_tendency - truth_tendency
    true_residual_tendency = truth_tendency - baseline_tendency
    predicted_correction_tendency = refined_tendency - baseline_tendency

    def mean_square(values: np.ndarray) -> float:
        return float(np.mean(values**2))

    truth_tendency_mean_square = mean_square(truth_tendency)
    baseline_tendency_mean_square = mean_square(baseline_tendency)
    refined_tendency_mean_square = mean_square(refined_tendency)
    baseline_tendency_mse = mean_square(baseline_tendency_error)
    refined_tendency_mse = mean_square(refined_tendency_error)
    true_residual_tendency_rms = math.sqrt(mean_square(true_residual_tendency))
    predicted_correction_tendency_rms = math.sqrt(mean_square(predicted_correction_tendency))
    truth_tendency_std = float(np.std(truth_tendency))

    result.update(
        {
            "number_of_temporal_valid_points": int(truth_tendency.size),
            "temporal_lead_interval_hours": float(lead_interval_hours),
            "truth_tendency_mean_square": truth_tendency_mean_square,
            "baseline_tendency_mean_square": baseline_tendency_mean_square,
            "refined_tendency_mean_square": refined_tendency_mean_square,
            "truth_tendency_rms": math.sqrt(truth_tendency_mean_square),
            "baseline_tendency_rms": math.sqrt(baseline_tendency_mean_square),
            "refined_tendency_rms": math.sqrt(refined_tendency_mean_square),
            "baseline_tendency_mse": baseline_tendency_mse,
            "refined_tendency_mse": refined_tendency_mse,
            "baseline_tendency_rmse": math.sqrt(baseline_tendency_mse),
            "refined_tendency_rmse": math.sqrt(refined_tendency_mse),
            "truth_tendency_spatial_std": truth_tendency_std,
            "baseline_tendency_spatial_std": float(np.std(baseline_tendency)),
            "refined_tendency_spatial_std": float(np.std(refined_tendency)),
            "baseline_tendency_spatial_correlation": _correlation(
                truth_tendency, baseline_tendency
            ),
            "refined_tendency_spatial_correlation": _correlation(truth_tendency, refined_tendency),
            "baseline_tendency_std_ratio": (
                float(np.std(baseline_tendency) / truth_tendency_std)
                if truth_tendency_std > 0.0
                else math.nan
            ),
            "refined_tendency_std_ratio": (
                float(np.std(refined_tendency) / truth_tendency_std)
                if truth_tendency_std > 0.0
                else math.nan
            ),
            "correction_tendency_to_residual_tendency_correlation": _correlation(
                predicted_correction_tendency,
                true_residual_tendency,
            ),
            "correction_tendency_amplitude_ratio": (
                predicted_correction_tendency_rms / true_residual_tendency_rms
                if true_residual_tendency_rms > 0.0
                else math.nan
            ),
        }
    )
    return result


def aggregate_case_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case rows by variable, level, and lead."""
    output: list[dict[str, Any]] = []
    keys = ["variable", "level", "lead_time_hours", "units"]
    for group_key, group in rows.groupby(keys, dropna=False, sort=True):
        weights = group["number_of_valid_points"].to_numpy(dtype=float)
        total = float(weights.sum())

        def weighted(
            column: str,
            current_group: pd.DataFrame = group,
            current_weights: np.ndarray = weights,
        ) -> float:
            values = current_group[column].to_numpy(dtype=float)
            valid = np.isfinite(values) & (current_weights > 0)
            return (
                float(np.average(values[valid], weights=current_weights[valid]))
                if np.any(valid)
                else math.nan
            )

        baseline_bias = weighted("baseline_bias")
        refined_bias = weighted("refined_bias")
        baseline_mse = weighted("baseline_mse")
        refined_mse = weighted("refined_mse")
        row: dict[str, Any] = dict(zip(keys, group_key))
        row.update(
            {
                "number_of_forecasts": int(len(group)),
                "number_of_valid_points": int(total),
                "fraction_forecasts_rmse_improved": float(
                    np.mean(group["refined_rmse"] < group["baseline_rmse"])
                ),
                "fraction_forecasts_rmse_worsened": float(
                    np.mean(group["refined_rmse"] > group["baseline_rmse"])
                ),
                "baseline_bias": baseline_bias,
                "refined_bias": refined_bias,
                "bias_improvement": abs(baseline_bias) - abs(refined_bias),
                "baseline_mae": weighted("baseline_mae"),
                "refined_mae": weighted("refined_mae"),
                "baseline_rmse": math.sqrt(baseline_mse),
                "refined_rmse": math.sqrt(refined_mse),
                "baseline_centered_rmse": math.sqrt(max(0.0, baseline_mse - baseline_bias**2)),
                "refined_centered_rmse": math.sqrt(max(0.0, refined_mse - refined_bias**2)),
            }
        )
        for name in (
            "baseline_spatial_correlation",
            "refined_spatial_correlation",
            "truth_spatial_std",
            "baseline_spatial_std",
            "refined_spatial_std",
            "baseline_amplitude_ratio",
            "refined_amplitude_ratio",
            "correction_to_residual_correlation",
            "correction_amplitude_ratio",
            "correction_sign_accuracy",
            "fraction_points_improved",
            "fraction_points_worsened",
            "fraction_points_unchanged",
            "optimal_correction_scale",
        ):
            row[name] = weighted(name)

        distribution_columns = [
            name
            for name in group.columns
            if name.startswith(("truth_p", "baseline_p", "refined_p"))
            or name
            in {
                "truth_skewness",
                "baseline_skewness",
                "refined_skewness",
                "truth_excess_kurtosis",
                "baseline_excess_kurtosis",
                "refined_excess_kurtosis",
                "baseline_wasserstein_distance",
                "refined_wasserstein_distance",
            }
        ]
        for name in distribution_columns:
            row[name] = weighted(name)
        if "number_of_temporal_valid_points" in group:
            temporal_weights = group["number_of_temporal_valid_points"].to_numpy(dtype=float)
            temporal_valid = temporal_weights > 0
            row["number_of_temporal_forecasts"] = int(np.count_nonzero(temporal_valid))
            row["number_of_temporal_valid_points"] = int(temporal_weights.sum())
            for name in _TEMPORAL_MEAN_SQUARE_METRICS:
                row[name] = weighted(name, current_weights=temporal_weights)
            for name in _TEMPORAL_LINEAR_METRICS:
                row[name] = weighted(name, current_weights=temporal_weights)
            for prefix in ("truth", "baseline", "refined"):
                row[f"{prefix}_tendency_rms"] = math.sqrt(row[f"{prefix}_tendency_mean_square"])
            for prefix in ("baseline", "refined"):
                row[f"{prefix}_tendency_rmse"] = math.sqrt(row[f"{prefix}_tendency_mse"])
            comparable = (
                temporal_valid
                & np.isfinite(group["baseline_tendency_rmse"].to_numpy(dtype=float))
                & np.isfinite(group["refined_tendency_rmse"].to_numpy(dtype=float))
            )
            row["fraction_forecasts_tendency_rmse_improved"] = (
                float(
                    np.mean(
                        group.loc[comparable, "refined_tendency_rmse"]
                        < group.loc[comparable, "baseline_tendency_rmse"]
                    )
                )
                if np.any(comparable)
                else math.nan
            )
            row["temporal_rmse_improvement_percent"] = (
                100.0
                * (row["baseline_tendency_rmse"] - row["refined_tendency_rmse"])
                / row["baseline_tendency_rmse"]
                if row["baseline_tendency_rmse"] > 0.0
                else math.nan
            )
            row["tendency_spatial_correlation_improvement"] = (
                row["refined_tendency_spatial_correlation"]
                - row["baseline_tendency_spatial_correlation"]
            )
        row["baseline_skewness_error"] = abs(row["baseline_skewness"] - row["truth_skewness"])
        row["refined_skewness_error"] = abs(row["refined_skewness"] - row["truth_skewness"])
        row["baseline_excess_kurtosis_error"] = abs(
            row["baseline_excess_kurtosis"] - row["truth_excess_kurtosis"]
        )
        row["refined_excess_kurtosis_error"] = abs(
            row["refined_excess_kurtosis"] - row["truth_excess_kurtosis"]
        )
        for metric in (
            "wasserstein_distance",
            "skewness_error",
            "excess_kurtosis_error",
            "p90_error",
            "p95_error",
            "p99_error",
            "p95_tail_mae",
            "p99_tail_mae",
        ):
            row[f"{metric}_improvement"] = row[f"baseline_{metric}"] - row[f"refined_{metric}"]
        for percentile in _QUANTILES:
            metric = f"p{percentile}_exceedance_frequency_bias"
            row[f"{metric}_improvement"] = abs(row[f"baseline_{metric}"]) - abs(
                row[f"refined_{metric}"]
            )
        row["mae_improvement_percent"] = (
            100.0 * (row["baseline_mae"] - row["refined_mae"]) / row["baseline_mae"]
            if row["baseline_mae"] > 0.0
            else math.nan
        )
        row["rmse_improvement_percent"] = (
            100.0 * (row["baseline_rmse"] - row["refined_rmse"]) / row["baseline_rmse"]
            if row["baseline_rmse"] > 0.0
            else math.nan
        )
        row["spatial_correlation_improvement"] = (
            row["refined_spatial_correlation"] - row["baseline_spatial_correlation"]
        )
        output.append(row)
    return pd.DataFrame(output)


def plot_diagnostic_map(
    fields: dict[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
    selection: Selection,
    units: str,
    forecasts: int,
    path: Path,
) -> None:
    """Write the requested eight-panel overall diagnostic map."""
    plt = _pyplot()
    field_limits = robust_limits([fields["truth"], fields["baseline"], fields["refined"]])
    error_limit = (
        max(
            abs(value)
            for value in robust_limits(
                [
                    fields["baseline_error"],
                    fields["refined_error"],
                    fields["true_residual"],
                    fields["predicted_correction"],
                    fields["remaining_residual"],
                ]
            )
        )
        or 1.0
    )
    panels = [
        ("Ground truth", "truth", "viridis", field_limits),
        ("Baseline", "baseline", "viridis", field_limits),
        ("Refined", "refined", "viridis", field_limits),
        ("Baseline error", "baseline_error", "RdBu_r", (-error_limit, error_limit)),
        ("Refined error", "refined_error", "RdBu_r", (-error_limit, error_limit)),
        ("True residual: truth − baseline", "true_residual", "RdBu_r", (-error_limit, error_limit)),
        (
            "Predicted correction: refined − baseline",
            "predicted_correction",
            "RdBu_r",
            (-error_limit, error_limit),
        ),
        (
            "Remaining residual: truth − refined",
            "remaining_residual",
            "RdBu_r",
            (-error_limit, error_limit),
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), constrained_layout=True)
    for axis, (title, name, cmap, limits) in zip(axes.flat, panels):
        mesh = axis.pcolormesh(
            longitude,
            latitude,
            fields[name],
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            shading="auto",
            rasterized=True,
        )
        fig.colorbar(mesh, ax=axis, shrink=0.78, label=units or None)
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        if longitude.size:
            axis.set_xlim(float(np.nanmin(longitude)), float(np.nanmax(longitude)))
    level = "surface" if selection.level is None else f"{selection.level:g} hPa"
    fig.suptitle(
        f"Refinement diagnosis: {selection.variable} ({level}); " f"{forecasts} matched forecasts"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    true_residual: np.ndarray,
    correction: np.ndarray,
    selection: Selection,
    units: str,
    path: Path,
) -> None:
    """Plot sampled predicted corrections against true residuals."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.hexbin(
        true_residual,
        correction,
        gridsize=90,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    limit = float(np.nanpercentile(np.abs(np.concatenate([true_residual, correction])), 99))
    limit = limit or 1.0
    axis.plot([-limit, limit], [-limit, limit], color="black", linewidth=1.2, label="ideal")
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel(f"True residual: truth − baseline ({units})".strip())
    axis.set_ylabel(f"Predicted correction: refined − baseline ({units})".strip())
    level = "surface" if selection.level is None else f"{selection.level:g} hPa"
    axis.set_title(
        f"{selection.variable} ({level}) correction diagnostic\n"
        f"correlation={_correlation(true_residual, correction):.3f}, "
        f"RMS ratio={np.sqrt(np.mean(correction**2)) / np.sqrt(np.mean(true_residual**2)):.3f}"
    )
    axis.legend()
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_comparison(
    truth: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    selection: Selection,
    units: str,
    path: Path,
) -> None:
    """Plot histogram/PDF, CDF, QQ agreement, and upper-tail exceedances."""
    plt = _pyplot()
    arrays = {
        "CAMS": np.asarray(truth, dtype=np.float64),
        "Aurora": np.asarray(baseline, dtype=np.float64),
        "Refined": np.asarray(refined, dtype=np.float64),
    }
    pooled = np.concatenate(list(arrays.values()))
    low, high = np.percentile(pooled, [0.5, 99.5])
    if not high > low:
        high = low + 1.0
    bins = np.linspace(low, high, 61)
    colors = {"CAMS": "black", "Aurora": "#d95f02", "Refined": "#1b9e77"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for label, values in arrays.items():
        axes[0, 0].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=colors[label],
        )
        ordered = np.sort(values)
        probability = (np.arange(ordered.size) + 0.5) / ordered.size
        axes[0, 1].plot(
            ordered,
            probability,
            label=label,
            color=colors[label],
        )
    axes[0, 0].set_title("Empirical PDF")
    axes[0, 0].set_ylabel("Density")
    axes[0, 1].set_title("Empirical CDF")
    axes[0, 1].set_ylabel("Cumulative probability")
    for axis in axes[0]:
        axis.set_xlabel(units or "concentration")
        axis.grid(alpha=0.2)
        axis.legend()

    probabilities = np.linspace(0.01, 0.99, 99)
    truth_quantiles = np.quantile(arrays["CAMS"], probabilities)
    for label in ("Aurora", "Refined"):
        candidate_quantiles = np.quantile(arrays[label], probabilities)
        axes[1, 0].plot(
            truth_quantiles,
            candidate_quantiles,
            label=label,
            color=colors[label],
        )
    diagonal_low = float(min(truth_quantiles.min(), low))
    diagonal_high = float(max(truth_quantiles.max(), high))
    axes[1, 0].plot(
        [diagonal_low, diagonal_high],
        [diagonal_low, diagonal_high],
        color="black",
        linestyle="--",
        linewidth=1,
        label="ideal",
    )
    axes[1, 0].set_title("Quantile-quantile agreement")
    axes[1, 0].set_xlabel(f"CAMS quantile ({units})".strip())
    axes[1, 0].set_ylabel(f"Forecast quantile ({units})".strip())
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    percentiles = np.asarray(_QUANTILES)
    thresholds = np.percentile(arrays["CAMS"], percentiles)
    expected = np.asarray([np.mean(arrays["CAMS"] >= value) for value in thresholds])
    x = np.arange(len(percentiles), dtype=float)
    width = 0.25
    axes[1, 1].bar(x - width, expected, width, label="CAMS", color=colors["CAMS"])
    for offset, label in ((0.0, "Aurora"), (width, "Refined")):
        frequencies = [np.mean(arrays[label] >= threshold) for threshold in thresholds]
        axes[1, 1].bar(
            x + offset,
            frequencies,
            width,
            label=label,
            color=colors[label],
        )
    axes[1, 1].set_xticks(x, [f"P{value}" for value in percentiles])
    axes[1, 1].set_ylabel("Exceedance frequency")
    axes[1, 1].set_title("CAMS-threshold exceedance calibration")
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend()

    level = "surface" if selection.level is None else f"{selection.level:g} hPa"
    fig.suptitle(f"{selection.variable} ({level}) distribution diagnostics")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_lead(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot magnitude, pattern, amplitude, and correction metrics by lead."""
    plt = _pyplot()
    for (variable, level), group in frame.groupby(["variable", "level"], sort=True):
        group = group.sort_values("lead_time_hours")
        stem = f"{file_safe(variable)}_{file_safe(level)}"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].plot(group.lead_time_hours, group.baseline_rmse, marker="o", label="baseline")
        axes[0, 0].plot(group.lead_time_hours, group.refined_rmse, marker="o", label="refined")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].legend()
        axes[0, 1].plot(
            group.lead_time_hours, group.baseline_centered_rmse, marker="o", label="baseline"
        )
        axes[0, 1].plot(
            group.lead_time_hours, group.refined_centered_rmse, marker="o", label="refined"
        )
        axes[0, 1].set_ylabel("Centered RMSE")
        axes[0, 1].legend()
        axes[1, 0].plot(
            group.lead_time_hours, group.baseline_amplitude_ratio, marker="o", label="baseline"
        )
        axes[1, 0].plot(
            group.lead_time_hours, group.refined_amplitude_ratio, marker="o", label="refined"
        )
        axes[1, 0].axhline(1, color="black", linewidth=1)
        axes[1, 0].set_ylabel("Forecast / truth spatial std")
        axes[1, 0].legend()
        axes[1, 1].plot(
            group.lead_time_hours,
            group.correction_amplitude_ratio,
            marker="o",
            label="correction RMS / residual RMS",
        )
        axes[1, 1].plot(
            group.lead_time_hours,
            group.correction_to_residual_correlation,
            marker="o",
            label="correction–residual corr",
        )
        axes[1, 1].plot(
            group.lead_time_hours,
            group.fraction_points_improved,
            marker="o",
            label="fraction points improved",
        )
        axes[1, 1].plot(
            group.lead_time_hours,
            group.fraction_points_worsened,
            marker="o",
            label="fraction points worsened",
        )
        axes[1, 1].axhline(1, color="black", linewidth=1)
        axes[1, 1].set_ylabel("Correction diagnostic")
        axes[1, 1].legend()
        for axis in axes.flat:
            axis.set_xlabel("Forecast lead time (hours)")
            axis.grid(alpha=0.25)
        fig.suptitle(f"Refinement diagnostics: {variable} ({level})")
        fig.savefig(output_dir / f"{stem}_metrics_by_lead.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_temporal_metrics_by_lead(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot lead-to-lead temporal consistency diagnostics by forecast lead."""
    if "number_of_temporal_valid_points" not in frame:
        return
    plt = _pyplot()
    for (variable, level), group in frame.groupby(["variable", "level"], sort=True):
        group = group.loc[group["number_of_temporal_valid_points"] > 0].sort_values(
            "lead_time_hours"
        )
        if group.empty:
            continue
        stem = f"{file_safe(variable)}_{file_safe(level)}"
        units = str(group.iloc[0].get("units") or "")
        tendency_units = f"{units} h$^{{-1}}$".strip()
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

        for prefix, label in (
            ("truth", "CAMS"),
            ("baseline", "Aurora"),
            ("refined", "refined"),
        ):
            axes[0, 0].plot(
                group.lead_time_hours,
                group[f"{prefix}_tendency_rms"],
                marker="o",
                label=label,
            )
        axes[0, 0].set_ylabel(f"Tendency RMS ({tendency_units})")
        axes[0, 0].legend()

        for prefix, label in (("baseline", "Aurora"), ("refined", "refined")):
            axes[0, 1].plot(
                group.lead_time_hours,
                group[f"{prefix}_tendency_rmse"],
                marker="o",
                label=label,
            )
        axes[0, 1].set_ylabel(f"Tendency error RMSE ({tendency_units})")
        axes[0, 1].legend()

        for prefix, label in (("baseline", "Aurora"), ("refined", "refined")):
            axes[0, 2].plot(
                group.lead_time_hours,
                group[f"{prefix}_tendency_spatial_correlation"],
                marker="o",
                label=label,
            )
        axes[0, 2].set_ylabel("Tendency spatial correlation")
        axes[0, 2].legend()

        for prefix, label in (("baseline", "Aurora"), ("refined", "refined")):
            axes[1, 0].plot(
                group.lead_time_hours,
                group[f"{prefix}_tendency_std_ratio"],
                marker="o",
                label=label,
            )
        axes[1, 0].axhline(1.0, color="black", linewidth=1)
        axes[1, 0].set_ylabel("Tendency std / CAMS tendency std")
        axes[1, 0].legend()

        axes[1, 1].plot(
            group.lead_time_hours,
            group.correction_tendency_amplitude_ratio,
            marker="o",
            color="tab:purple",
        )
        axes[1, 1].axhline(1.0, color="black", linewidth=1)
        axes[1, 1].set_ylabel("Correction-tendency RMS / residual-tendency RMS")

        axes[1, 2].plot(
            group.lead_time_hours,
            group.correction_tendency_to_residual_tendency_correlation,
            marker="o",
            color="tab:green",
        )
        axes[1, 2].set_ylabel("Correction/residual tendency correlation")

        for axis in axes.flat:
            axis.set_xlabel("Forecast lead time (hours)")
            axis.grid(alpha=0.25)
        fig.suptitle(f"Temporal refinement diagnostics: {variable} ({level})")
        fig.savefig(
            output_dir / f"{stem}_temporal_metrics_by_lead.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_metrics_by_case(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot case-to-case RMSE and correction-amplitude variation."""
    plt = _pyplot()
    frame = frame.copy()
    frame["initialization_time"] = pd.to_datetime(frame["initialization_time"])
    frame["rmse_improvement_percent"] = np.where(
        frame["baseline_rmse"] > 0.0,
        100.0 * (frame["baseline_rmse"] - frame["refined_rmse"]) / frame["baseline_rmse"],
        np.nan,
    )
    for (variable, level), group in frame.groupby(
        ["variable", "level"],
        sort=True,
    ):
        stem = f"{file_safe(variable)}_{file_safe(level)}"
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for lead, cases in group.groupby("lead_time_hours", sort=True):
            cases = cases.sort_values("initialization_time")
            label = f"{lead:g} h"
            axes[0].plot(
                cases.initialization_time,
                cases.rmse_improvement_percent,
                linewidth=0.9,
                alpha=0.8,
                label=label,
            )
            axes[1].plot(
                cases.initialization_time,
                cases.correction_amplitude_ratio,
                linewidth=0.9,
                alpha=0.8,
                label=label,
            )
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_ylabel("RMSE improvement (%)")
        axes[1].axhline(1, color="black", linewidth=1)
        axes[1].set_ylabel("Correction RMS / residual RMS")
        axes[1].set_xlabel("Forecast initialization")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(ncol=3, fontsize=8)
        fig.suptitle(f"Refinement by case: {variable} ({level})")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{stem}_metrics_by_case.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def _aggregate_target_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool lead rows without averaging already-normalized percentages."""
    if frame.empty:
        return frame.copy()

    output: list[dict[str, Any]] = []
    keys = ["variable", "level", "units"]
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        point_weights = group["number_of_valid_points"].to_numpy(dtype=float)
        forecast_weights = group["number_of_forecasts"].to_numpy(dtype=float)

        def weighted(
            column: str,
            weights: np.ndarray,
            *,
            squared: bool = False,
            current_group: pd.DataFrame = group,
        ) -> float:
            values = current_group[column].to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if not np.any(valid):
                return math.nan
            if squared:
                values = values**2
            return float(
                np.average(
                    values[valid],
                    weights=weights[valid],
                )
            )

        row: dict[str, Any] = dict(zip(keys, group_key))
        numeric_columns = [
            name
            for name in group.columns
            if name not in {*keys, "lead_time_hours"}
            and "tendency" not in name
            and not name.startswith(("temporal_", "number_of_temporal"))
            and pd.api.types.is_numeric_dtype(group[name])
        ]
        for name in numeric_columns:
            row[name] = weighted(name, point_weights)
        row["number_of_forecasts"] = int(group["number_of_forecasts"].sum())
        row["number_of_valid_points"] = int(point_weights.sum())
        for name in (
            "baseline_spatial_correlation",
            "refined_spatial_correlation",
            "fraction_forecasts_rmse_improved",
            "fraction_forecasts_rmse_worsened",
        ):
            row[name] = weighted(name, forecast_weights)
        for prefix in ("baseline", "refined"):
            row[f"{prefix}_rmse"] = math.sqrt(
                weighted(f"{prefix}_rmse", point_weights, squared=True)
            )
        row["bias_improvement"] = abs(row["baseline_bias"]) - abs(row["refined_bias"])
        row["mae_improvement_percent"] = (
            100.0 * (row["baseline_mae"] - row["refined_mae"]) / row["baseline_mae"]
            if row["baseline_mae"] > 0.0
            else math.nan
        )
        row["rmse_improvement_percent"] = (
            100.0 * (row["baseline_rmse"] - row["refined_rmse"]) / row["baseline_rmse"]
            if row["baseline_rmse"] > 0.0
            else math.nan
        )
        row["spatial_correlation_improvement"] = (
            row["refined_spatial_correlation"] - row["baseline_spatial_correlation"]
        )
        if "number_of_temporal_valid_points" in group:
            temporal_weights = group["number_of_temporal_valid_points"].to_numpy(dtype=float)
            temporal_forecast_weights = group["number_of_temporal_forecasts"].to_numpy(dtype=float)
            row["number_of_temporal_forecasts"] = int(temporal_forecast_weights.sum())
            row["number_of_temporal_valid_points"] = int(temporal_weights.sum())
            for name in _TEMPORAL_MEAN_SQUARE_METRICS:
                row[name] = weighted(name, temporal_weights)
            for name in _TEMPORAL_LINEAR_METRICS:
                row[name] = weighted(name, temporal_weights)
            for prefix in ("truth", "baseline", "refined"):
                row[f"{prefix}_tendency_rms"] = math.sqrt(row[f"{prefix}_tendency_mean_square"])
            for prefix in ("baseline", "refined"):
                row[f"{prefix}_tendency_rmse"] = math.sqrt(row[f"{prefix}_tendency_mse"])
            row["fraction_forecasts_tendency_rmse_improved"] = weighted(
                "fraction_forecasts_tendency_rmse_improved",
                temporal_forecast_weights,
            )
            row["temporal_rmse_improvement_percent"] = (
                100.0
                * (row["baseline_tendency_rmse"] - row["refined_tendency_rmse"])
                / row["baseline_tendency_rmse"]
                if row["baseline_tendency_rmse"] > 0.0
                else math.nan
            )
            row["tendency_spatial_correlation_improvement"] = (
                row["refined_tendency_spatial_correlation"]
                - row["baseline_tendency_spatial_correlation"]
            )
        output.append(row)
    return pd.DataFrame(output)


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    """Write absolute and baseline-relative scientific forecast diagnostics."""
    target = _aggregate_target_metrics(frame).sort_values(
        "rmse_improvement_percent", ascending=False
    )

    def number(value: float, digits: int = 4) -> str:
        return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"

    lines = [
        "# Refinement evaluation summary",
        "",
        "Every row compares CAMS truth, the unrefined Aurora rollout, and the "
        "refined rollout on the same finite grid cells. Positive improvement "
        "and correlation-change values are better. Tail MAE uses cells at or "
        "above the CAMS percentile; Wasserstein is the empirical full-"
        "distribution distance.",
        "",
        "## Absolute forecast skill and change from Aurora",
        "",
        "| variable | level | Aurora MAE | refined MAE | MAE improvement % | "
        "Aurora RMSE | refined RMSE | RMSE improvement % | Aurora abs bias | "
        "refined abs bias | Aurora corr | refined corr | corr change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in target.itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | {number(row.baseline_mae)} | "
            f"{number(row.refined_mae)} | {row.mae_improvement_percent:.3f} | "
            f"{number(row.baseline_rmse)} | {number(row.refined_rmse)} | "
            f"{row.rmse_improvement_percent:.3f} | "
            f"{number(abs(row.baseline_bias))} | {number(abs(row.refined_bias))} | "
            f"{row.baseline_spatial_correlation:.4f} | "
            f"{row.refined_spatial_correlation:.4f} | "
            f"{row.spatial_correlation_improvement:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Distribution and upper-tail agreement",
            "",
            "| variable | level | Aurora std ratio | refined std ratio | "
            "Aurora P95 error | refined P95 error | Aurora P99 error | "
            "refined P99 error | Aurora P99 tail MAE | refined P99 tail MAE | "
            "Aurora Wasserstein | refined Wasserstein |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target.itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | "
            f"{row.baseline_amplitude_ratio:.3f} | "
            f"{row.refined_amplitude_ratio:.3f} | "
            f"{number(row.baseline_p95_error)} | "
            f"{number(row.refined_p95_error)} | "
            f"{number(row.baseline_p99_error)} | "
            f"{number(row.refined_p99_error)} | "
            f"{number(row.baseline_p99_tail_mae)} | "
            f"{number(row.refined_p99_tail_mae)} | "
            f"{number(row.baseline_wasserstein_distance)} | "
            f"{number(row.refined_wasserstein_distance)} |"
        )

    lines.extend(
        [
            "",
            "### Distribution shape",
            "",
            "Population excess kurtosis is reported directly; positive error "
            "improvement means refinement moved it closer to CAMS.",
            "",
            "| variable | level | CAMS excess kurtosis | Aurora excess kurtosis | "
            "refined excess kurtosis | Aurora abs error | refined abs error | "
            "error improvement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target.itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | "
            f"{number(row.truth_excess_kurtosis)} | "
            f"{number(row.baseline_excess_kurtosis)} | "
            f"{number(row.refined_excess_kurtosis)} | "
            f"{number(row.baseline_excess_kurtosis_error)} | "
            f"{number(row.refined_excess_kurtosis_error)} | "
            f"{number(row.excess_kurtosis_error_improvement)} |"
        )

    lines.extend(
        [
            "",
            "## Correction behavior",
            "",
            "A correction-amplitude ratio above one means the predicted correction "
            "has more RMS energy than the true Aurora-to-CAMS residual.",
            "",
            "| variable | level | correction amplitude ratio | correction/residual "
            "correlation | points improved | points worsened | forecasts improved | "
            "forecasts worsened | optimal correction scale |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target.itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | "
            f"{row.correction_amplitude_ratio:.3f} | "
            f"{row.correction_to_residual_correlation:.3f} | "
            f"{row.fraction_points_improved:.1%} | "
            f"{row.fraction_points_worsened:.1%} | "
            f"{row.fraction_forecasts_rmse_improved:.1%} | "
            f"{row.fraction_forecasts_rmse_worsened:.1%} | "
            f"{row.optimal_correction_scale:.3f} |"
        )

    if "number_of_temporal_valid_points" in target:
        temporal_target = target.loc[target["number_of_temporal_valid_points"] > 0]
        if not temporal_target.empty:
            lines.extend(
                [
                    "",
                    "## Temporal consistency",
                    "",
                    "Tendencies are per-hour changes between consecutive evaluated valid "
                    "leads. Error RMSE, correlation, and standard-deviation ratios compare "
                    "Aurora/refined tendencies with CAMS. The final two columns compare the "
                    "change in the predicted correction with the change in the true "
                    "Aurora-to-CAMS residual.",
                    "",
                    "| variable | level | CAMS tendency RMS | Aurora tendency RMS | "
                    "refined tendency RMS | Aurora tendency error RMSE | refined tendency "
                    "error RMSE | error RMSE improvement % | Aurora tendency corr | "
                    "refined tendency corr | corr change | Aurora tendency std ratio | "
                    "refined tendency std ratio | correction-tendency amplitude ratio | "
                    "correction/residual tendency corr |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                    "---:|---:|---:|",
                ]
            )
            for row in temporal_target.itertuples(index=False):
                lines.append(
                    f"| {row.variable} | {row.level} | "
                    f"{number(row.truth_tendency_rms)} | "
                    f"{number(row.baseline_tendency_rms)} | "
                    f"{number(row.refined_tendency_rms)} | "
                    f"{number(row.baseline_tendency_rmse)} | "
                    f"{number(row.refined_tendency_rmse)} | "
                    f"{number(row.temporal_rmse_improvement_percent, 3)} | "
                    f"{number(row.baseline_tendency_spatial_correlation, 3)} | "
                    f"{number(row.refined_tendency_spatial_correlation, 3)} | "
                    f"{number(row.tendency_spatial_correlation_improvement, 3)} | "
                    f"{number(row.baseline_tendency_std_ratio, 3)} | "
                    f"{number(row.refined_tendency_std_ratio, 3)} | "
                    f"{number(row.correction_tendency_amplitude_ratio, 3)} | "
                    f"{number(row.correction_tendency_to_residual_tendency_correlation, 3)} |"
                )

    lines.extend(
        [
            "",
            "## Lead-time metrics",
            "",
            "| variable | level | lead h | Aurora MAE | refined MAE | MAE imp % | "
            "Aurora RMSE | refined RMSE | RMSE imp % | corr change | "
            "refined std ratio | refined P99 error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frame.sort_values(["variable", "level", "lead_time_hours"]).itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | {row.lead_time_hours:g} | "
            f"{number(row.baseline_mae)} | {number(row.refined_mae)} | "
            f"{row.mae_improvement_percent:.3f} | "
            f"{number(row.baseline_rmse)} | {number(row.refined_rmse)} | "
            f"{row.rmse_improvement_percent:.3f} | "
            f"{row.spatial_correlation_improvement:+.4f} | "
            f"{row.refined_amplitude_ratio:.3f} | "
            f"{number(row.refined_p99_error)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def diagnose(
    config_path: Path,
    *,
    finetuned_dir: Path | None = None,
    output_dir: Path | None = None,
    max_initializations: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Run the aligned refinement diagnostics and return the output directory."""
    settings = load_settings(config_path)
    if finetuned_dir is not None:
        settings["finetuned_dir"] = finetuned_dir.resolve()
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else settings["output_dir"] / "refinement_diagnostics"
    )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Diagnostic output directory is not empty: {destination}. "
            "Choose a new directory or pass --overwrite."
        )
    figure_dir = destination / "figures"
    map_dir = figure_dir / "maps"
    scatter_dir = figure_dir / "scatter"
    distribution_dir = figure_dir / "distributions"
    for directory in (destination, figure_dir, map_dir, scatter_dir, distribution_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selections = target_selections(settings["raw"])
    baseline_files = rollout_files_by_initialization(
        settings["baseline_dir"],
        "rollout_*.nc",
    )
    refined_files = rollout_files_by_initialization(
        settings["finetuned_dir"],
        "rollout_predictions_init_*.nc",
    )
    common = sorted(set(baseline_files) & set(refined_files))
    if max_initializations is not None:
        common = common[: max(0, max_initializations)]
    if not common:
        raise FileNotFoundError("No common baseline/refined rollout initializations.")

    rows: list[dict[str, Any]] = []
    scatter_samples: dict[Selection, list[tuple[np.ndarray, np.ndarray]]] = {
        selection: [] for selection in selections
    }
    distribution_samples: dict[Selection, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        selection: [] for selection in selections
    }
    rng = np.random.default_rng(42)
    truth = xr.open_dataset(settings["truth_path"], decode_times=True)
    try:
        truth_times = np.asarray(truth.time.values).astype("datetime64[ns]")
        if np.isnat(truth_times).any() or np.unique(truth_times).size != truth_times.size:
            raise ValueError("CAMS truth time must contain unique, non-NaT valid times.")
        truth_lookup = {
            int(value.astype(np.int64)): index for index, value in enumerate(truth_times)
        }
        with xr.open_dataset(refined_files[common[0]]) as first_refined:
            latitude = np.asarray(first_refined.latitude.values, dtype=float)
            longitude = np.asarray(first_refined.longitude.values, dtype=float)
            tolerance = settings["coordinate_tolerance"]
            truth_lat_idx = match_coordinate_indices(
                np.asarray(truth.latitude.values, dtype=float),
                latitude,
                tolerance,
                name="latitude",
            )
            truth_lon_idx = match_coordinate_indices(
                np.asarray(truth.longitude.values, dtype=float),
                longitude,
                tolerance,
                name="longitude",
            )

        missing_truth_variables = [
            selection.variable
            for selection in selections
            if selection.variable not in truth.data_vars
        ]
        if missing_truth_variables:
            raise KeyError(
                "CAMS truth is missing configured target variable(s): "
                + ", ".join(sorted(set(missing_truth_variables)))
            )

        sums = {
            selection: SpatialSums.create((len(latitude), len(longitude)))
            for selection in selections
        }
        skipped_without_truth = 0
        for initialization_ns in tqdm(
            common,
            desc="Refinement diagnostics",
            unit="initialization",
        ):
            with xr.open_dataset(
                baseline_files[initialization_ns],
                decode_times=True,
            ) as baseline, xr.open_dataset(
                refined_files[initialization_ns],
                decode_times=True,
            ) as refined:
                initialization = np.datetime64(initialization_ns, "ns")
                baseline_context = str(baseline_files[initialization_ns])
                refined_context = str(refined_files[initialization_ns])
                valid_times, refined_step_dim = validate_rollout_time_metadata(
                    refined,
                    initialization,
                    context=refined_context,
                )
                baseline_valid_times, baseline_step_dim = validate_rollout_time_metadata(
                    baseline,
                    initialization,
                    context=baseline_context,
                )
                assert_same_coordinate(
                    latitude,
                    np.asarray(refined.latitude.values, dtype=float),
                    tolerance,
                    name="latitude",
                    context=refined_context,
                )
                assert_same_coordinate(
                    longitude,
                    np.asarray(refined.longitude.values, dtype=float),
                    tolerance,
                    name="longitude",
                    context=refined_context,
                )
                baseline_lat_idx = match_coordinate_indices(
                    np.asarray(baseline.latitude.values, dtype=float),
                    latitude,
                    tolerance,
                    name=f"{baseline_context} latitude",
                )
                baseline_lon_idx = match_coordinate_indices(
                    np.asarray(baseline.longitude.values, dtype=float),
                    longitude,
                    tolerance,
                    name=f"{baseline_context} longitude",
                )
                previous_fields: dict[
                    Selection,
                    tuple[float, np.ndarray, np.ndarray, np.ndarray],
                ] = {}
                for refined_time_index, valid_time in enumerate(valid_times):
                    lead = float((valid_time - initialization) / np.timedelta64(1, "h"))
                    if not any(
                        np.isclose(lead, requested, atol=1.0e-6, rtol=0)
                        for requested in settings["lead_hours"]
                    ):
                        continue
                    truth_index = truth_lookup.get(int(valid_time.astype(np.int64)))
                    if truth_index is None:
                        skipped_without_truth += 1
                        continue
                    baseline_time_index = match_valid_time_index(
                        baseline_valid_times,
                        valid_time,
                        context=baseline_context,
                    )
                    for selection in selections:
                        if selection.variable not in baseline.data_vars:
                            raise KeyError(
                                f"{baseline_context} is missing configured variable "
                                f"{selection.variable!r}."
                            )
                        refined_variable = resolve_forecast_variable(refined, selection.variable)
                        if refined_step_dim not in refined_variable.dims:
                            raise ValueError(
                                f"{refined_context} forecast variable "
                                f"{refined_variable.name!r} does not use time step "
                                f"dimension {refined_step_dim!r}."
                            )
                        truth_field = truth[selection.variable].isel(
                            time=truth_index,
                            latitude=truth_lat_idx,
                            longitude=truth_lon_idx,
                        )
                        baseline_field = baseline[selection.variable].isel(
                            {
                                baseline_step_dim: baseline_time_index,
                                "latitude": baseline_lat_idx,
                                "longitude": baseline_lon_idx,
                            }
                        )
                        refined_field = refined_variable.isel(
                            {refined_step_dim: refined_time_index}
                        )
                        if selection.level is not None:
                            truth_field = truth_field.isel(
                                level=select_level_index(
                                    np.asarray(truth_field.level.values, dtype=float),
                                    selection.level,
                                    settings["level_tolerance"],
                                )
                            )
                            baseline_field = baseline_field.isel(
                                level=select_level_index(
                                    np.asarray(baseline_field.level.values, dtype=float),
                                    selection.level,
                                    settings["level_tolerance"],
                                )
                            )
                            refined_field = refined_field.isel(
                                level=select_level_index(
                                    np.asarray(refined_field.level.values, dtype=float),
                                    selection.level,
                                    settings["level_tolerance"],
                                )
                            )
                        if "member" in refined_field.dims:
                            refined_field = refined_field.mean("member", skipna=True)
                        truth_values = aligned_2d_values(
                            truth_field,
                            context=f"CAMS {selection.variable!r} at {valid_time}",
                        )
                        baseline_values = aligned_2d_values(
                            baseline_field,
                            context=f"Aurora {selection.variable!r} at {valid_time}",
                        )
                        refined_values = aligned_2d_values(
                            refined_field,
                            context=f"refined {selection.variable!r} at {valid_time}",
                        )
                        expected_shape = (len(latitude), len(longitude))
                        for field_name, values in (
                            ("CAMS", truth_values),
                            ("Aurora", baseline_values),
                            ("refined", refined_values),
                        ):
                            if values.shape != expected_shape:
                                raise ValueError(
                                    f"{field_name} {selection.variable!r} at "
                                    f"{valid_time} has shape {values.shape}; "
                                    f"expected {expected_shape}."
                                )
                        metrics = case_metrics(
                            truth_values,
                            baseline_values,
                            refined_values,
                        )
                        previous = previous_fields.get(selection)
                        temporal_metrics = _empty_temporal_metrics()
                        if previous is not None:
                            (
                                previous_lead,
                                previous_truth,
                                previous_baseline,
                                previous_refined,
                            ) = previous
                            temporal_metrics = temporal_case_metrics(
                                previous_truth,
                                previous_baseline,
                                previous_refined,
                                truth_values,
                                baseline_values,
                                refined_values,
                                lead_interval_hours=lead - previous_lead,
                            )
                        metrics.update(temporal_metrics)
                        previous_fields[selection] = (
                            lead,
                            truth_values.copy(),
                            baseline_values.copy(),
                            refined_values.copy(),
                        )
                        metrics.update(
                            {
                                "initialization_time": str(initialization),
                                "valid_time": str(valid_time),
                                "variable": selection.variable,
                                "level": (
                                    "surface" if selection.level is None else f"{selection.level:g}"
                                ),
                                "lead_time_hours": lead,
                                "units": str(truth[selection.variable].attrs.get("units") or ""),
                            }
                        )
                        rows.append(metrics)
                        sums[selection].update(
                            truth_values,
                            baseline_values,
                            refined_values,
                        )
                        flat_truth = truth_values.ravel()
                        flat_base = baseline_values.ravel()
                        flat_refined = refined_values.ravel()
                        valid = (
                            np.isfinite(flat_truth)
                            & np.isfinite(flat_base)
                            & np.isfinite(flat_refined)
                        )
                        indices = np.flatnonzero(valid)
                        if indices.size:
                            chosen = rng.choice(
                                indices,
                                size=min(300, indices.size),
                                replace=False,
                            )
                            scatter_samples[selection].append(
                                (
                                    flat_truth[chosen] - flat_base[chosen],
                                    flat_refined[chosen] - flat_base[chosen],
                                )
                            )
                            distribution_samples[selection].append(
                                (
                                    flat_truth[chosen],
                                    flat_base[chosen],
                                    flat_refined[chosen],
                                )
                            )

        if skipped_without_truth:
            print(
                "Skipped "
                f"{skipped_without_truth} requested forecast steps outside the CAMS "
                "truth valid-time range; no nearest-time matching was attempted."
            )

        case_frame = pd.DataFrame(rows)
        lead_frame = aggregate_case_metrics(case_frame)
        case_frame.to_csv(destination / "metrics_by_case.csv", index=False)
        lead_frame.to_csv(destination / "metrics_by_lead.csv", index=False)
        plot_metrics_by_lead(lead_frame, figure_dir)
        plot_temporal_metrics_by_lead(lead_frame, figure_dir)
        plot_metrics_by_case(case_frame, figure_dir)
        for summary_name in ("evaluation_summary.md", "diagnostic_summary.md"):
            write_summary(lead_frame, destination / summary_name)

        normalized_lon = ((longitude + 180.0) % 360.0) - 180.0
        lon_order = np.argsort(normalized_lon)
        normalized_lon = normalized_lon[lon_order]
        datasets: list[xr.Dataset] = []
        for selection in selections:
            fields = {name: value[:, lon_order] for name, value in sums[selection].means().items()}
            units = str(truth[selection.variable].attrs.get("units") or "")
            plot_diagnostic_map(
                fields,
                latitude,
                normalized_lon,
                selection,
                units,
                sums[selection].forecasts,
                map_dir / f"{selection.label}_correction_diagnostics.png",
            )
            samples = scatter_samples[selection]
            if samples:
                true_residual = np.concatenate([pair[0] for pair in samples])
                correction = np.concatenate([pair[1] for pair in samples])
                plot_scatter(
                    true_residual,
                    correction,
                    selection,
                    units,
                    scatter_dir / f"{selection.label}_correction_scatter.png",
                )
            samples = distribution_samples[selection]
            if samples:
                truth_sample = np.concatenate([triple[0] for triple in samples])
                baseline_sample = np.concatenate([triple[1] for triple in samples])
                refined_sample = np.concatenate([triple[2] for triple in samples])
                plot_distribution_comparison(
                    truth_sample,
                    baseline_sample,
                    refined_sample,
                    selection,
                    units,
                    distribution_dir / f"{selection.label}_distribution_diagnostics.png",
                )
            ds = xr.Dataset(
                {
                    name: (
                        ("latitude", "longitude"),
                        values.astype(np.float32),
                    )
                    for name, values in fields.items()
                },
                coords={
                    "latitude": latitude,
                    "longitude": normalized_lon,
                },
            )
            ds["valid_count"] = (
                ("latitude", "longitude"),
                sums[selection].count[:, lon_order].astype(np.int32),
            )
            ds = ds.expand_dims(selection=[selection.label]).assign_coords(
                variable=("selection", [selection.variable]),
                level=(
                    "selection",
                    ["surface" if selection.level is None else f"{selection.level:g}"],
                ),
                units=("selection", [units]),
                number_of_forecasts=(
                    "selection",
                    [sums[selection].forecasts],
                ),
            )
            datasets.append(ds)
        aggregate = xr.concat(datasets, dim="selection", join="exact")
        aggregate.to_netcdf(destination / "aggregate_correction_fields.nc")
        aggregate.close()
        for dataset in datasets:
            dataset.close()
    finally:
        truth.close()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--finetuned-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-initializations", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = diagnose(
        args.config.expanduser().resolve(),
        finetuned_dir=(
            args.finetuned_dir.expanduser().resolve() if args.finetuned_dir is not None else None
        ),
        output_dir=(
            args.output_dir.expanduser().resolve() if args.output_dir is not None else None
        ),
        max_initializations=args.max_initializations,
        overwrite=args.overwrite,
    )
    print(f"Saved refinement diagnostics: {output}")


if __name__ == "__main__":
    main()
