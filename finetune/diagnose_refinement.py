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
    def create(cls, shape: tuple[int, int]) -> "SpatialSums":
        zeros = lambda: np.zeros(shape, dtype=np.float64)
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
            np.mean(
                np.signbit(correction[nonzero_sign])
                == np.signbit(true_residual[nonzero_sign])
            )
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
    comparison_tolerance = np.finfo(np.float64).eps * np.maximum(
        1.0, baseline_absolute_error
    )
    improved = refined_absolute_error < baseline_absolute_error - comparison_tolerance
    worsened = refined_absolute_error > baseline_absolute_error + comparison_tolerance

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
            float(np.std(baseline) / np.std(truth))
            if np.std(truth) > 0
            else math.nan
        ),
        "refined_amplitude_ratio": (
            float(np.std(refined) / np.std(truth))
            if np.std(truth) > 0
            else math.nan
        ),
        "correction_to_residual_correlation": _correlation(
            correction, true_residual,
        ),
        "correction_amplitude_ratio": (
            correction_rms / residual_rms if residual_rms > 0 else math.nan
        ),
        "correction_sign_accuracy": sign_accuracy,
        "fraction_points_improved": float(
            np.mean(improved)
        ),
        "fraction_points_worsened": float(np.mean(worsened)),
        "fraction_points_unchanged": float(np.mean(~improved & ~worsened)),
        "optimal_correction_scale": optimal_scale,
    }


def aggregate_case_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case rows by variable, level, and lead."""
    output: list[dict[str, Any]] = []
    keys = ["variable", "level", "lead_time_hours", "units"]
    for group_key, group in rows.groupby(keys, dropna=False, sort=True):
        weights = group["number_of_valid_points"].to_numpy(dtype=float)
        total = float(weights.sum())

        def weighted(column: str) -> float:
            values = group[column].to_numpy(dtype=float)
            valid = np.isfinite(values) & (weights > 0)
            return (
                float(np.average(values[valid], weights=weights[valid]))
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
                "baseline_centered_rmse": math.sqrt(
                    max(0.0, baseline_mse - baseline_bias**2)
                ),
                "refined_centered_rmse": math.sqrt(
                    max(0.0, refined_mse - refined_bias**2)
                ),
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
        row["mae_improvement_percent"] = (
            100.0 * (row["baseline_mae"] - row["refined_mae"])
            / row["baseline_mae"]
            if row["baseline_mae"] > 1.0e-12
            else math.nan
        )
        row["rmse_improvement_percent"] = (
            100.0 * (row["baseline_rmse"] - row["refined_rmse"])
            / row["baseline_rmse"]
            if row["baseline_rmse"] > 1.0e-12
            else math.nan
        )
        row["spatial_correlation_improvement"] = (
            row["refined_spatial_correlation"]
            - row["baseline_spatial_correlation"]
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
    field_limits = robust_limits(
        [fields["truth"], fields["baseline"], fields["refined"]]
    )
    error_limit = max(
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
    ) or 1.0
    panels = [
        ("Ground truth", "truth", "viridis", field_limits),
        ("Baseline", "baseline", "viridis", field_limits),
        ("Refined", "refined", "viridis", field_limits),
        ("Baseline error", "baseline_error", "RdBu_r", (-error_limit, error_limit)),
        ("Refined error", "refined_error", "RdBu_r", (-error_limit, error_limit)),
        ("True residual: truth − baseline", "true_residual", "RdBu_r", (-error_limit, error_limit)),
        ("Predicted correction: refined − baseline", "predicted_correction", "RdBu_r", (-error_limit, error_limit)),
        ("Remaining residual: truth − refined", "remaining_residual", "RdBu_r", (-error_limit, error_limit)),
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
        f"Refinement diagnosis: {selection.variable} ({level}); "
        f"{forecasts} matched forecasts"
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
    limit = float(
        np.nanpercentile(np.abs(np.concatenate([true_residual, correction])), 99)
    )
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
        axes[0, 1].plot(group.lead_time_hours, group.baseline_centered_rmse, marker="o", label="baseline")
        axes[0, 1].plot(group.lead_time_hours, group.refined_centered_rmse, marker="o", label="refined")
        axes[0, 1].set_ylabel("Centered RMSE")
        axes[0, 1].legend()
        axes[1, 0].plot(group.lead_time_hours, group.baseline_amplitude_ratio, marker="o", label="baseline")
        axes[1, 0].plot(group.lead_time_hours, group.refined_amplitude_ratio, marker="o", label="refined")
        axes[1, 0].axhline(1, color="black", linewidth=1)
        axes[1, 0].set_ylabel("Forecast / truth spatial std")
        axes[1, 0].legend()
        axes[1, 1].plot(group.lead_time_hours, group.correction_amplitude_ratio, marker="o", label="correction RMS / residual RMS")
        axes[1, 1].plot(group.lead_time_hours, group.correction_to_residual_correlation, marker="o", label="correction–residual corr")
        axes[1, 1].plot(group.lead_time_hours, group.fraction_points_improved, marker="o", label="fraction points improved")
        axes[1, 1].plot(group.lead_time_hours, group.fraction_points_worsened, marker="o", label="fraction points worsened")
        axes[1, 1].axhline(1, color="black", linewidth=1)
        axes[1, 1].set_ylabel("Correction diagnostic")
        axes[1, 1].legend()
        for axis in axes.flat:
            axis.set_xlabel("Forecast lead time (hours)")
            axis.grid(alpha=0.25)
        fig.suptitle(f"Refinement diagnostics: {variable} ({level})")
        fig.savefig(output_dir / f"{stem}_metrics_by_lead.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_metrics_by_case(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot case-to-case RMSE and correction-amplitude variation."""
    plt = _pyplot()
    frame = frame.copy()
    frame["initialization_time"] = pd.to_datetime(frame["initialization_time"])
    frame["rmse_improvement_percent"] = np.where(
        frame["baseline_rmse"] > 1.0e-12,
        100.0
        * (frame["baseline_rmse"] - frame["refined_rmse"])
        / frame["baseline_rmse"],
        np.nan,
    )
    for (variable, level), group in frame.groupby(
        ["variable", "level"], sort=True,
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


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    """Write a concise Markdown interpretation of aggregate diagnostics."""
    target = (
        frame.groupby(["variable", "level"], as_index=False)
        .agg(
            rmse_improvement_percent=("rmse_improvement_percent", "mean"),
            mae_improvement_percent=("mae_improvement_percent", "mean"),
            correlation_change=("spatial_correlation_improvement", "mean"),
            correction_amplitude_ratio=("correction_amplitude_ratio", "mean"),
            correction_residual_correlation=(
                "correction_to_residual_correlation",
                "mean",
            ),
            fraction_points_improved=("fraction_points_improved", "mean"),
            fraction_points_worsened=("fraction_points_worsened", "mean"),
            fraction_forecasts_improved=(
                "fraction_forecasts_rmse_improved",
                "mean",
            ),
            fraction_forecasts_worsened=(
                "fraction_forecasts_rmse_worsened",
                "mean",
            ),
            optimal_correction_scale=("optimal_correction_scale", "mean"),
        )
        .sort_values("rmse_improvement_percent", ascending=False)
    )
    lines = [
        "# Refinement diagnostic summary",
        "",
        "Positive MAE/RMSE improvement means refinement reduced error. "
        "A correction amplitude ratio above one means the predicted correction "
        "has more RMS energy than the true baseline residual.",
        "",
        "| variable | level | RMSE improvement % | MAE improvement % | "
        "correlation change | correction amplitude ratio | correction/residual "
        "correlation | points improved | points worsened | forecasts improved | "
        "forecasts worsened | "
        "optimal correction scale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in target.itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | {row.rmse_improvement_percent:.3f} | "
            f"{row.mae_improvement_percent:.3f} | {row.correlation_change:+.4f} | "
            f"{row.correction_amplitude_ratio:.3f} | "
            f"{row.correction_residual_correlation:.3f} | "
            f"{row.fraction_points_improved:.1%} | "
            f"{row.fraction_points_worsened:.1%} | "
            f"{row.fraction_forecasts_improved:.1%} | "
            f"{row.fraction_forecasts_worsened:.1%} | "
            f"{row.optimal_correction_scale:.3f} |"
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
    for directory in (destination, figure_dir, map_dir, scatter_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selections = target_selections(settings["raw"])
    baseline_files = rollout_files_by_initialization(
        settings["baseline_dir"], "rollout_*.nc",
    )
    refined_files = rollout_files_by_initialization(
        settings["finetuned_dir"], "rollout_predictions_init_*.nc",
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
    rng = np.random.default_rng(42)
    truth = xr.open_dataset(settings["truth_path"], decode_times=True)
    try:
        truth_times = np.asarray(truth.time.values).astype("datetime64[ns]")
        if np.isnat(truth_times).any() or np.unique(truth_times).size != truth_times.size:
            raise ValueError("CAMS truth time must contain unique, non-NaT valid times.")
        truth_lookup = {
            int(value.astype(np.int64)): index
            for index, value in enumerate(truth_times)
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
            common, desc="Refinement diagnostics", unit="initialization",
        ):
            with xr.open_dataset(
                baseline_files[initialization_ns], decode_times=True,
            ) as baseline, xr.open_dataset(
                refined_files[initialization_ns], decode_times=True,
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
                        refined_variable = resolve_forecast_variable(
                            refined, selection.variable
                        )
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
                            truth_values, baseline_values, refined_values,
                        )
                        metrics.update(
                            {
                                "initialization_time": str(initialization),
                                "valid_time": str(valid_time),
                                "variable": selection.variable,
                                "level": (
                                    "surface"
                                    if selection.level is None
                                    else f"{selection.level:g}"
                                ),
                                "lead_time_hours": lead,
                                "units": str(
                                    truth[selection.variable].attrs.get("units") or ""
                                ),
                            }
                        )
                        rows.append(metrics)
                        sums[selection].update(
                            truth_values, baseline_values, refined_values,
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
        plot_metrics_by_case(case_frame, figure_dir)
        write_summary(lead_frame, destination / "diagnostic_summary.md")

        normalized_lon = ((longitude + 180.0) % 360.0) - 180.0
        lon_order = np.argsort(normalized_lon)
        normalized_lon = normalized_lon[lon_order]
        datasets: list[xr.Dataset] = []
        for selection in selections:
            fields = {
                name: value[:, lon_order]
                for name, value in sums[selection].means().items()
            }
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
                    [
                        "surface"
                        if selection.level is None
                        else f"{selection.level:g}"
                    ],
                ),
                units=("selection", [units]),
                number_of_forecasts=(
                    "selection", [sums[selection].forecasts],
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
            args.finetuned_dir.expanduser().resolve()
            if args.finetuned_dir is not None
            else None
        ),
        output_dir=(
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else None
        ),
        max_initializations=args.max_initializations,
        overwrite=args.overwrite,
    )
    print(f"Saved refinement diagnostics: {output}")


if __name__ == "__main__":
    main()
