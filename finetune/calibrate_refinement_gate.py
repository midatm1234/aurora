"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Fit and apply validation-selected refinement correction strengths.

For predicted correction ``c = refined - baseline`` and true residual
``r = truth - baseline``, the least-squares scalar is

    alpha = sum(c * r) / sum(c**2).

The fitted value is clipped to a configured safe interval (default [0, 1]).
Because zero is in that interval, leaving the baseline unchanged is an
explicit candidate when a correction is anti-correlated or unhelpful on the
calibration data. Scales are fitted independently by variable, level, and
forecast lead and must be estimated from validation/training-tail data, never
from the test cases used to report final performance.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.generate_overall_evaluation_maps import (  # noqa: E402
    Selection,
    load_settings,
    match_coordinate_indices,
    rollout_files_by_initialization,
    select_level_index,
    target_selections,
)


def least_squares_scale(
    correction_residual_sum: float,
    correction_squared_sum: float,
    *,
    minimum_scale: float = 0.0,
    maximum_scale: float = 1.0,
) -> tuple[float, float]:
    """Return raw and safely clipped least-squares correction strengths."""
    raw = (
        correction_residual_sum / correction_squared_sum
        if correction_squared_sum > 0
        else math.nan
    )
    selected = (
        float(np.clip(raw, minimum_scale, maximum_scale))
        if np.isfinite(raw)
        else 0.0
    )
    return raw, selected


def _time_indices(reference: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Match requested datetimes exactly and return reference indices."""
    lookup = {
        int(value.astype("datetime64[ns]").astype(np.int64)): index
        for index, value in enumerate(reference)
    }
    indices: list[int] = []
    for value in requested:
        key = int(value.astype("datetime64[ns]").astype(np.int64))
        if key not in lookup:
            raise ValueError(f"Forecast valid time {value} is absent from baseline.")
        indices.append(lookup[key])
    return np.asarray(indices, dtype=int)


def _aligned_baseline_variable(
    baseline: xr.Dataset,
    refined: xr.Dataset,
    variable: str,
    tolerance: float,
    level_tolerance: float,
) -> xr.DataArray:
    """Align a baseline variable to refined named coordinates without interpolation."""
    if variable not in baseline or variable not in refined:
        raise KeyError(f"{variable!r} must exist in baseline and refined rollouts.")
    candidate = baseline[variable]
    reference = refined[variable]
    indexers: dict[str, Any] = {
        "time": _time_indices(
            np.asarray(baseline.time.values),
            np.asarray(refined.time.values),
        ),
        "latitude": match_coordinate_indices(
            np.asarray(baseline.latitude.values, dtype=float),
            np.asarray(refined.latitude.values, dtype=float),
            tolerance,
            name="baseline latitude",
        ),
        "longitude": match_coordinate_indices(
            np.asarray(baseline.longitude.values, dtype=float),
            np.asarray(refined.longitude.values, dtype=float),
            tolerance,
            name="baseline longitude",
        ),
    }
    if "level" in reference.dims:
        indexers["level"] = np.asarray(
            [
                select_level_index(
                    np.asarray(baseline.level.values, dtype=float),
                    float(level),
                    level_tolerance,
                )
                for level in refined.level.values
            ],
            dtype=int,
        )
    aligned = candidate.isel(indexers).transpose(*reference.dims)
    return aligned.assign_coords(
        {
            dimension: reference.coords[dimension]
            for dimension in reference.dims
            if dimension in reference.coords
        }
    )


def _field(
    dataset: xr.Dataset,
    selection: Selection,
    time_index: int,
    level_tolerance: float,
) -> np.ndarray:
    """Extract one configured field using level coordinate values."""
    values = dataset[selection.variable].isel(time=time_index)
    if selection.level is not None:
        values = values.isel(
            level=select_level_index(
                np.asarray(dataset.level.values, dtype=float),
                selection.level,
                level_tolerance,
            )
        )
    if "member" in values.dims:
        values = values.mean("member", skipna=True)
    return np.asarray(values.values, dtype=np.float64)


def fit_scales(
    config_path: Path,
    calibration_truth: Path,
    calibration_refined_dir: Path,
    *,
    minimum_scale: float = 0.0,
    maximum_scale: float = 1.0,
) -> pd.DataFrame:
    """Fit per-variable/level/lead least-squares correction scales."""
    if not minimum_scale <= 0 <= maximum_scale:
        raise ValueError("Scale interval must include zero as the identity safeguard.")
    settings = load_settings(config_path)
    selections = target_selections(settings["raw"])
    baseline_files = rollout_files_by_initialization(
        settings["baseline_dir"], "rollout_*.nc",
    )
    refined_files = rollout_files_by_initialization(
        calibration_refined_dir, "rollout_predictions_init_*.nc",
    )
    common = sorted(set(baseline_files) & set(refined_files))
    if not common:
        raise FileNotFoundError(
            "No common baseline/calibration-refined rollout initializations."
        )

    keys = [
        (
            selection.variable,
            "surface" if selection.level is None else f"{selection.level:g}",
            float(lead),
        )
        for selection in selections
        for lead in settings["lead_hours"]
    ]
    sums: dict[tuple[str, str, float], dict[str, float]] = {
        key: {
            "correction_residual": 0.0,
            "correction_squared": 0.0,
            "residual_squared": 0.0,
            "valid_points": 0.0,
            "forecasts": 0.0,
        }
        for key in keys
    }
    truth = xr.open_dataset(calibration_truth, decode_times=True)
    try:
        truth_times = np.asarray(truth.time.values).astype("datetime64[ns]")
        truth_lookup = {
            int(value.astype(np.int64)): index
            for index, value in enumerate(truth_times)
        }
        for initialization_ns in tqdm(
            common, desc="Fitting refinement gate", unit="initialization",
        ):
            with xr.open_dataset(
                baseline_files[initialization_ns], decode_times=True,
            ) as baseline, xr.open_dataset(
                refined_files[initialization_ns], decode_times=True,
            ) as refined:
                initialization = np.datetime64(initialization_ns, "ns")
                valid_times = np.asarray(refined.time.values).astype("datetime64[ns]")
                truth_lat_idx = match_coordinate_indices(
                    np.asarray(truth.latitude.values, dtype=float),
                    np.asarray(refined.latitude.values, dtype=float),
                    settings["coordinate_tolerance"],
                    name="truth latitude",
                )
                truth_lon_idx = match_coordinate_indices(
                    np.asarray(truth.longitude.values, dtype=float),
                    np.asarray(refined.longitude.values, dtype=float),
                    settings["coordinate_tolerance"],
                    name="truth longitude",
                )
                aligned_baseline = {
                    selection.variable: _aligned_baseline_variable(
                        baseline,
                        refined,
                        selection.variable,
                        settings["coordinate_tolerance"],
                        settings["level_tolerance"],
                    )
                    for selection in selections
                }
                for time_index, valid_time in enumerate(valid_times):
                    lead = float(
                        (valid_time - initialization) / np.timedelta64(1, "h")
                    )
                    requested = next(
                        (
                            value
                            for value in settings["lead_hours"]
                            if np.isclose(lead, value, atol=1.0e-6, rtol=0)
                        ),
                        None,
                    )
                    if requested is None:
                        continue
                    truth_index = truth_lookup.get(int(valid_time.astype(np.int64)))
                    if truth_index is None:
                        raise ValueError(
                            f"Calibration truth lacks valid time {valid_time}."
                        )
                    for selection in selections:
                        truth_values = truth[selection.variable].isel(
                            time=truth_index,
                            latitude=truth_lat_idx,
                            longitude=truth_lon_idx,
                        )
                        if selection.level is not None:
                            truth_values = truth_values.isel(
                                level=select_level_index(
                                    np.asarray(truth.level.values, dtype=float),
                                    selection.level,
                                    settings["level_tolerance"],
                                )
                            )
                        base_values = _field(
                            aligned_baseline[selection.variable].to_dataset(
                                name=selection.variable
                            ),
                            selection,
                            time_index,
                            settings["level_tolerance"],
                        )
                        refined_values = _field(
                            refined,
                            selection,
                            time_index,
                            settings["level_tolerance"],
                        )
                        truth_array = np.asarray(truth_values.values, dtype=np.float64)
                        mask = (
                            np.isfinite(truth_array)
                            & np.isfinite(base_values)
                            & np.isfinite(refined_values)
                        )
                        if not np.any(mask):
                            continue
                        residual = truth_array[mask] - base_values[mask]
                        correction = refined_values[mask] - base_values[mask]
                        level = (
                            "surface"
                            if selection.level is None
                            else f"{selection.level:g}"
                        )
                        state = sums[(selection.variable, level, float(requested))]
                        state["correction_residual"] += float(
                            np.dot(correction, residual)
                        )
                        state["correction_squared"] += float(
                            np.dot(correction, correction)
                        )
                        state["residual_squared"] += float(np.dot(residual, residual))
                        state["valid_points"] += int(mask.sum())
                        state["forecasts"] += 1
    finally:
        truth.close()

    rows: list[dict[str, Any]] = []
    for (variable, level, lead), state in sums.items():
        denominator = state["correction_squared"]
        raw_scale, selected_scale = least_squares_scale(
            state["correction_residual"],
            denominator,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        baseline_mse = (
            state["residual_squared"] / state["valid_points"]
            if state["valid_points"] > 0
            else math.nan
        )
        calibrated_sse = (
            state["residual_squared"]
            - 2 * selected_scale * state["correction_residual"]
            + selected_scale**2 * state["correction_squared"]
        )
        calibrated_mse = (
            max(0.0, calibrated_sse) / state["valid_points"]
            if state["valid_points"] > 0
            else math.nan
        )
        rows.append(
            {
                "variable": variable,
                "level": level,
                "lead_time_hours": lead,
                "number_of_forecasts": int(state["forecasts"]),
                "number_of_valid_points": int(state["valid_points"]),
                "raw_least_squares_scale": raw_scale,
                "selected_scale": selected_scale,
                "calibration_baseline_rmse": math.sqrt(baseline_mse),
                "calibration_calibrated_rmse": math.sqrt(calibrated_mse),
                "calibration_rmse_improvement_percent": (
                    100
                    * (math.sqrt(baseline_mse) - math.sqrt(calibrated_mse))
                    / math.sqrt(baseline_mse)
                    if baseline_mse > 0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["variable", "level", "lead_time_hours"]
    )


def apply_scales(
    config_path: Path,
    scales: pd.DataFrame,
    application_refined_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Apply fitted scales to rollouts without modifying source files."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Calibrated rollouts never overwrite existing results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_path)
    selections = target_selections(settings["raw"])
    baseline_files = rollout_files_by_initialization(
        settings["baseline_dir"], "rollout_*.nc",
    )
    refined_files = rollout_files_by_initialization(
        application_refined_dir, "rollout_predictions_init_*.nc",
    )
    common = sorted(set(baseline_files) & set(refined_files))
    if not common:
        raise FileNotFoundError(
            "No common baseline/application-refined rollout initializations."
        )
    scale_lookup = {
        (str(row.variable), str(row.level), float(row.lead_time_hours)): float(
            row.selected_scale
        )
        for row in scales.itertuples(index=False)
    }
    written: list[Path] = []
    for initialization_ns in tqdm(
        common, desc="Applying refinement gate", unit="initialization",
    ):
        with xr.open_dataset(
            baseline_files[initialization_ns], decode_times=True,
        ) as baseline, xr.open_dataset(
            refined_files[initialization_ns], decode_times=True,
        ) as source:
            refined = source.load()
            initialization = np.datetime64(initialization_ns, "ns")
            valid_times = np.asarray(refined.time.values).astype("datetime64[ns]")
            leads = [
                float((valid_time - initialization) / np.timedelta64(1, "h"))
                for valid_time in valid_times
            ]
            for variable in sorted({selection.variable for selection in selections}):
                aligned_baseline = _aligned_baseline_variable(
                    baseline,
                    refined,
                    variable,
                    settings["coordinate_tolerance"],
                    settings["level_tolerance"],
                )
                for selection in [
                    item for item in selections if item.variable == variable
                ]:
                    level_label = (
                        "surface"
                        if selection.level is None
                        else f"{selection.level:g}"
                    )
                    level_index = (
                        None
                        if selection.level is None
                        else select_level_index(
                            np.asarray(refined.level.values, dtype=float),
                            selection.level,
                            settings["level_tolerance"],
                        )
                    )
                    for time_index, lead in enumerate(leads):
                        matched_lead = next(
                            (
                                requested
                                for requested in settings["lead_hours"]
                                if np.isclose(
                                    lead, requested, atol=1.0e-6, rtol=0
                                )
                            ),
                            None,
                        )
                        if matched_lead is None:
                            continue
                        scale = scale_lookup.get(
                            (variable, level_label, float(matched_lead)), 0.0
                        )
                        indexer: dict[str, int] = {"time": time_index}
                        if level_index is not None:
                            indexer["level"] = level_index
                        base_values = aligned_baseline.isel(indexer)
                        source_values = refined[variable].isel(indexer)
                        refined[variable].loc[
                            {
                                dimension: refined[dimension].values[index]
                                for dimension, index in indexer.items()
                            }
                        ] = base_values + scale * (source_values - base_values)
            refined.attrs.update(
                {
                    "refinement_gate": "validation-selected least-squares scale",
                    "refinement_gate_formula": "clip(sum(c*r)/sum(c^2), 0, 1)",
                    "refinement_gate_calibration": "external held-out data",
                }
            )
            destination = output_dir / refined_files[initialization_ns].name
            refined.to_netcdf(destination)
            refined.close()
            written.append(destination)
    return written


def write_summary(scales: pd.DataFrame, path: Path) -> None:
    """Document the fitted scales and calibration-set effect."""
    lines = [
        "# Validation-selected refinement gate",
        "",
        "For correction `c = refined - baseline` and residual "
        "`r = truth - baseline`, the fitted strength is "
        "`alpha = clip(sum(c*r) / sum(c^2), 0, 1)`.",
        "",
        "The calibration data are separate from the final test cases. A zero "
        "scale means the validation residual and correction were orthogonal, "
        "anti-correlated, missing, or otherwise favored the unchanged baseline.",
        "",
        "| variable | level | lead (h) | raw alpha | selected alpha | "
        "calibration RMSE improvement % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scales.itertuples(index=False):
        lines.append(
            f"| {row.variable} | {row.level} | {row.lead_time_hours:g} | "
            f"{row.raw_least_squares_scale:.4f} | {row.selected_scale:.4f} | "
            f"{row.calibration_rmse_improvement_percent:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--calibration-truth", required=True, type=Path)
    parser.add_argument("--calibration-refined-dir", required=True, type=Path)
    parser.add_argument("--application-refined-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-scale", type=float, default=0.0)
    parser.add_argument("--maximum-scale", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Calibration output directory is not empty: {output_dir}. "
            "Choose a new directory; results are never overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_path = output_dir / "correction_scales.csv"
    scales = fit_scales(
        args.config.expanduser().resolve(),
        args.calibration_truth.expanduser().resolve(),
        args.calibration_refined_dir.expanduser().resolve(),
        minimum_scale=args.minimum_scale,
        maximum_scale=args.maximum_scale,
    )
    scales.to_csv(scale_path, index=False)
    write_summary(scales, output_dir / "calibration_summary.md")
    rollout_dir = output_dir / "rollouts"
    written = apply_scales(
        args.config.expanduser().resolve(),
        scales,
        args.application_refined_dir.expanduser().resolve(),
        rollout_dir,
    )
    print(f"Saved scales: {scale_path}")
    print(f"Saved {len(written)} calibrated rollouts: {rollout_dir}")


if __name__ == "__main__":
    main()
