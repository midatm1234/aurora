"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Trace one exactly matched CAMS/Aurora/refinement batch to JSON, NetCDF, and PNGs.

This diagnostic is deliberately bounded: it reads one initialization and one
forecast lead, performs no interpolation, and never modifies rollout files.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr
import yaml
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.diagnose_refinement import case_metrics
from finetune.generate_overall_evaluation_maps import (
    Selection,
    aligned_2d_values,
    assert_same_coordinate,
    load_settings,
    match_coordinate_indices,
    match_valid_time_index,
    rollout_files_by_initialization,
    select_level_index,
    target_selections,
    validate_rollout_time_metadata,
)
from finetune.refinement.io import resolve_forecast_variable


def _parse_initialization(value: str) -> np.datetime64:
    for pattern in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H"):
        try:
            return np.datetime64(dt.datetime.strptime(value, pattern), "ns")
        except ValueError:
            pass
    try:
        parsed = np.datetime64(value, "ns")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid initialization {value!r}; use YYYYMMDDTHHMMSS or ISO time."
        ) from exc
    if np.isnat(parsed):
        raise argparse.ArgumentTypeError("Initialization must not be NaT.")
    return parsed


def _initialization_key(value: np.datetime64) -> int:
    return int(np.datetime64(value, "ns").astype(np.int64))


def _field_stats(
    values: np.ndarray,
    *,
    dimensions: tuple[str, ...] = ("latitude", "longitude"),
) -> dict[str, Any]:
    values = np.asarray(values)
    finite = np.isfinite(values)
    valid = values[finite].astype(np.float64)
    result: dict[str, Any] = {
        "shape": list(values.shape),
        "dimensions": list(dimensions),
        "dimension_meanings": {
            name: {
                "member": "stochastic ensemble member in stored draw order",
                "latitude": "north-to-south grid row",
                "longitude": "west-to-east grid column",
            }.get(name, name)
            for name in dimensions
        },
        "dtype": str(values.dtype),
        "device": "cpu",
        "nan_count": int(np.isnan(values).sum()),
        "nonfinite_count": int(values.size - finite.sum()),
        "valid_count": int(valid.size),
    }
    if valid.size:
        result.update(
            {
                "minimum": float(valid.min()),
                "maximum": float(valid.max()),
                "mean": float(valid.mean()),
                "standard_deviation": float(valid.std()),
                "p01": float(np.percentile(valid, 1)),
                "p50": float(np.percentile(valid, 50)),
                "p99": float(np.percentile(valid, 99)),
            }
        )
    return result


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first).ravel()
    second = np.asarray(second).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2 or np.std(first[valid]) == 0 or np.std(second[valid]) == 0:
        return math.nan
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def _level_key(level: float) -> str:
    rounded = round(float(level), 3)
    text = str(int(rounded)) if rounded.is_integer() else str(rounded)
    return text.replace(".", "_")


def _load_aurora_scales(repository: Path) -> dict[str, float]:
    """Read the literal scale table without importing Aurora's model package."""
    path = repository / "aurora" / "normalisation.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "scales"
            for target in node.targets
        ):
            value_node = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "scales"
        ):
            value_node = node.value
        if value_node is not None:
            values = ast.literal_eval(value_node)
            return {str(name): float(value) for name, value in values.items()}
    raise RuntimeError(f"Could not find the Aurora scale table in {path}.")


def _normalization_scale(
    selection: Selection,
    *,
    scales: Mapping[str, float],
) -> tuple[str, float]:
    key = (
        selection.variable
        if selection.level is None
        else f"{selection.variable}_{_level_key(selection.level)}"
    )
    if key not in scales:
        raise KeyError(f"Aurora normalization scale {key!r} is unavailable.")
    scale = float(scales[key])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Aurora normalization scale {key!r} is invalid: {scale}.")
    return key, scale


def _select_level(field: xr.DataArray, selection: Selection, tolerance: float) -> xr.DataArray:
    if selection.level is None:
        return field
    if "level" not in field.dims:
        raise ValueError(
            f"Atmospheric selection {selection.variable}@{selection.level:g} hPa has no "
            f"level dimension in {field.name!r}."
        )
    index = select_level_index(
        np.asarray(field.level.values, dtype=float), selection.level, tolerance
    )
    return field.isel(level=index)


def _render_map(
    values: np.ndarray,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    title: str,
    units: str,
    path: Path,
    limits: tuple[float, float],
    diverging: bool,
) -> None:
    """Write a nearest-neighbour pixel-grid PNG without smoothing."""
    values = np.asarray(values, dtype=np.float64)
    low, high = limits
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = (low - 0.5, high + 0.5) if np.isfinite(low) else (0.0, 1.0)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if diverging:
        # Blue -> near-white -> red, centered by the caller at zero.
        rgb = np.empty((*values.shape, 3), dtype=np.float64)
        left = normalized <= 0.5
        fraction = np.where(left, normalized * 2.0, (normalized - 0.5) * 2.0)
        blue = np.array([33.0, 102.0, 172.0])
        white = np.array([247.0, 247.0, 247.0])
        red = np.array([178.0, 24.0, 43.0])
        rgb[left] = blue + fraction[left][:, None] * (white - blue)
        rgb[~left] = white + fraction[~left][:, None] * (red - white)
    else:
        # Dark purple -> teal -> yellow.
        purple = np.array([68.0, 1.0, 84.0])
        teal = np.array([33.0, 145.0, 140.0])
        yellow = np.array([253.0, 231.0, 37.0])
        rgb = np.empty((*values.shape, 3), dtype=np.float64)
        left = normalized <= 0.5
        fraction = np.where(left, normalized * 2.0, (normalized - 0.5) * 2.0)
        rgb[left] = purple + fraction[left][:, None] * (teal - purple)
        rgb[~left] = teal + fraction[~left][:, None] * (yellow - teal)
    rgb[~np.isfinite(values)] = np.array([170.0, 170.0, 170.0])
    raster = Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")
    available_width, available_height = 820, 500
    scale = min(available_width / values.shape[1], available_height / values.shape[0])
    raster_size = (
        max(1, int(round(values.shape[1] * scale))),
        max(1, int(round(values.shape[0] * scale))),
    )
    raster = raster.resize(raster_size, resample=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (960, 680), "white")
    draw = ImageDraw.Draw(canvas)
    left_px = (canvas.width - raster.width) // 2
    top_px = 70
    canvas.paste(raster, (left_px, top_px))
    draw.rectangle(
        (left_px, top_px, left_px + raster.width, top_px + raster.height),
        outline="black",
        width=2,
    )
    draw.text((20, 18), title, fill="black")
    draw.text(
        (left_px, top_px + raster.height + 10),
        f"lon {longitude[0]:g} to {longitude[-1]:g}",
        fill="black",
    )
    draw.text(
        (left_px + raster.width - 180, top_px + raster.height + 10),
        f"lat {latitude[0]:g} to {latitude[-1]:g}",
        fill="black",
    )
    bar_left, bar_top, bar_width, bar_height = 140, 625, 680, 18
    for offset in range(bar_width):
        # Reuse the same palette through a tiny recursive-free interpolation.
        position = offset / max(bar_width - 1, 1)
        if diverging:
            if position <= 0.5:
                color = blue + position * 2.0 * (white - blue)
            else:
                color = white + (position - 0.5) * 2.0 * (red - white)
        elif position <= 0.5:
            color = purple + position * 2.0 * (teal - purple)
        else:
            color = teal + (position - 0.5) * 2.0 * (yellow - teal)
        draw.line(
            (bar_left + offset, bar_top, bar_left + offset, bar_top + bar_height),
            fill=tuple(np.rint(color).astype(np.uint8)),
        )
    draw.rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + bar_height),
        outline="black",
    )
    draw.text((bar_left, bar_top + 23), f"{low:.5g}", fill="black")
    draw.text((bar_left + bar_width - 90, bar_top + 23), f"{high:.5g}", fill="black")
    draw.text((430, bar_top + 23), units or "unitless", fill="black")
    canvas.save(path)


def _robust_limits(arrays: list[np.ndarray], *, symmetric: bool) -> tuple[float, float]:
    finite = [array[np.isfinite(array)] for array in arrays]
    finite = [array for array in finite if array.size]
    if not finite:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    pooled = np.concatenate(finite)
    if symmetric:
        limit = float(np.percentile(np.abs(pooled), 99)) or 1.0
        return -limit, limit
    low, high = np.percentile(pooled, [1, 99])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _resolved_run_config(finetuned_dir: Path, fallback: Mapping[str, Any]) -> tuple[dict, str]:
    path = finetuned_dir / "resolved_config.yaml"
    if not path.exists():
        return dict(fallback), "requested config (run snapshot unavailable)"
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Resolved run config is not a mapping: {path}")
    return loaded, str(path)


def trace_batch(
    config_path: Path,
    *,
    initialization: np.datetime64,
    lead_hours: float,
    output_dir: Path,
    finetuned_dir: Path | None = None,
    legacy_refined_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Trace one matched forecast case and return the trace directory."""
    settings = load_settings(config_path)
    if finetuned_dir is not None:
        settings["finetuned_dir"] = finetuned_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Trace output directory is not empty: {output_dir}; pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "maps"
    map_dir.mkdir(exist_ok=True)

    key = _initialization_key(initialization)
    baseline_files = rollout_files_by_initialization(
        settings["baseline_dir"], "rollout_*.nc"
    )
    refined_files = rollout_files_by_initialization(
        settings["finetuned_dir"], "rollout_predictions_init_*.nc"
    )
    if key not in baseline_files or key not in refined_files:
        raise FileNotFoundError(
            f"Initialization {initialization} is not present in both baseline and "
            "refined rollout directories."
        )
    legacy_files = None
    if legacy_refined_dir is not None:
        legacy_files = rollout_files_by_initialization(
            legacy_refined_dir, "rollout_predictions_init_*.nc"
        )
        if key not in legacy_files:
            raise FileNotFoundError(
                f"Legacy directory has no rollout for initialization {initialization}."
            )

    run_config, run_config_source = _resolved_run_config(
        settings["finetuned_dir"], settings["raw"]
    )
    target_space = (((run_config.get("model") or {}).get("refinement") or {}).get(
        "target_space"
    ) or {})
    residual_scaling = (((run_config.get("model") or {}).get("refinement") or {}).get(
        "residual_scaling"
    ) or {})
    residual_scaling_method = str(residual_scaling.get("method", "none")).lower()
    if residual_scaling_method not in {"", "none", "identity"}:
        raise ValueError(
            "This trace can reconstruct normalized corrections exactly only for the "
            "legacy Aurora field-normalized residual space. The resolved run uses "
            f"residual_scaling.method={residual_scaling_method!r}; provide a trace "
            "implementation that loads that checkpoint's saved correction statistics."
        )
    scales = _load_aurora_scales(settings["root"])
    selections = target_selections(run_config)
    tolerance = settings["coordinate_tolerance"]
    level_tolerance = settings["level_tolerance"]
    requested_valid_time = np.datetime64(initialization, "ns") + np.timedelta64(
        int(round(float(lead_hours) * 3_600_000_000_000)), "ns"
    )

    trace: dict[str, Any] = {
        "correction_convention": "true_correction_physical = CAMS - Aurora",
        "application_convention": (
            "refined_forecast = Aurora + predicted_correction_physical"
        ),
        "initialization_time": str(np.datetime64(initialization, "ns")),
        "lead_time_hours": float(lead_hours),
        "valid_time": str(requested_valid_time),
        "normalization": {
            "run_config_source": run_config_source,
            "target_space": target_space,
            "residual_scaling": residual_scaling,
            "formula": "correction_normalized = correction_physical / Aurora field scale",
            "warning": (
                "The failed rollout did not serialize the raw sampler correction. "
                "predicted_correction_normalized is reconstructed from the final "
                "physical forecast after constraints."
            ),
        },
        "paths": {
            "config": str(config_path),
            "truth": str(settings["truth_path"]),
            "aurora": str(baseline_files[key]),
            "refined": str(refined_files[key]),
            "legacy_refined": (
                None if legacy_files is None else str(legacy_files[key])
            ),
        },
        "channel_order": [
            {
                "index": index,
                "variable": selection.variable,
                "level_hpa": selection.level,
                "kind": selection.kind,
            }
            for index, selection in enumerate(selections)
        ],
        "selections": {},
    }

    output_fields: dict[str, list[np.ndarray]] = {}
    selection_labels: list[str] = []
    selection_variables: list[str] = []
    selection_levels: list[str] = []
    selection_units: list[str] = []

    with xr.open_dataset(settings["truth_path"], decode_times=True) as truth, xr.open_dataset(
        baseline_files[key], decode_times=True
    ) as baseline, xr.open_dataset(refined_files[key], decode_times=True) as refined:
        refined_times, refined_step_dim = validate_rollout_time_metadata(
            refined, initialization, context=str(refined_files[key])
        )
        baseline_times, baseline_step_dim = validate_rollout_time_metadata(
            baseline, initialization, context=str(baseline_files[key])
        )
        refined_time_index = match_valid_time_index(
            refined_times, requested_valid_time, context=str(refined_files[key])
        )
        baseline_time_index = match_valid_time_index(
            baseline_times, requested_valid_time, context=str(baseline_files[key])
        )
        truth_times = np.asarray(truth.time.values).astype("datetime64[ns]")
        truth_time_index = match_valid_time_index(
            truth_times, requested_valid_time, context=str(settings["truth_path"])
        )
        latitude = np.asarray(refined.latitude.values, dtype=float)
        longitude = np.asarray(refined.longitude.values, dtype=float)
        truth_latitude_indices = match_coordinate_indices(
            np.asarray(truth.latitude.values, dtype=float), latitude, tolerance, name="latitude"
        )
        truth_longitude_indices = match_coordinate_indices(
            np.asarray(truth.longitude.values, dtype=float), longitude, tolerance, name="longitude"
        )
        baseline_latitude_indices = match_coordinate_indices(
            np.asarray(baseline.latitude.values, dtype=float), latitude, tolerance, name="latitude"
        )
        baseline_longitude_indices = match_coordinate_indices(
            np.asarray(baseline.longitude.values, dtype=float), longitude, tolerance, name="longitude"
        )
        trace["coordinates"] = {
            "latitude": latitude.tolist(),
            "latitude_order": (
                "descending" if np.all(np.diff(latitude) < 0) else "ascending"
            ),
            "longitude": longitude.tolist(),
            "longitude_order": (
                "ascending" if np.all(np.diff(longitude) > 0) else "non-monotonic"
            ),
            "pressure_level_order": {
                "CAMS": np.asarray(truth.level.values, dtype=float).tolist()
                if "level" in truth.coords else [],
                "Aurora": np.asarray(baseline.level.values, dtype=float).tolist()
                if "level" in baseline.coords else [],
                "refined": np.asarray(refined.level.values, dtype=float).tolist()
                if "level" in refined.coords else [],
            },
        }
        trace["source_dataset_schemas"] = {
            role: {
                "dimensions": {str(name): int(size) for name, size in dataset.sizes.items()},
                "time_dimensions": list(dataset.time.dims),
                "time_dtype": str(dataset.time.dtype),
                "latitude_dtype": str(dataset.latitude.dtype),
                "longitude_dtype": str(dataset.longitude.dtype),
            }
            for role, dataset in (
                ("CAMS", truth),
                ("Aurora", baseline),
                ("refined", refined),
            )
        }

        legacy = (
            xr.open_dataset(legacy_files[key], decode_times=True)
            if legacy_files is not None
            else None
        )
        try:
            legacy_time_index = legacy_step_dim = None
            if legacy is not None:
                assert_same_coordinate(
                    latitude,
                    np.asarray(legacy.latitude.values, dtype=float),
                    tolerance,
                    name="latitude",
                    context=str(legacy_files[key]),
                )
                assert_same_coordinate(
                    longitude,
                    np.asarray(legacy.longitude.values, dtype=float),
                    tolerance,
                    name="longitude",
                    context=str(legacy_files[key]),
                )
                legacy_times, legacy_step_dim = validate_rollout_time_metadata(
                    legacy, initialization, context=str(legacy_files[key])
                )
                legacy_time_index = match_valid_time_index(
                    legacy_times, requested_valid_time, context=str(legacy_files[key])
                )

            for selection in selections:
                if selection.variable not in truth or selection.variable not in baseline:
                    raise KeyError(
                        f"Selection {selection.variable!r} is absent from CAMS or Aurora."
                    )
                truth_field = truth[selection.variable].isel(
                    {
                        "time": truth_time_index,
                        "latitude": truth_latitude_indices,
                        "longitude": truth_longitude_indices,
                    }
                )
                baseline_field = baseline[selection.variable].isel(
                    {
                        baseline_step_dim: baseline_time_index,
                        "latitude": baseline_latitude_indices,
                        "longitude": baseline_longitude_indices,
                    }
                )
                refined_variable = resolve_forecast_variable(refined, selection.variable)
                refined_field = refined_variable.isel(
                    {refined_step_dim: refined_time_index}
                )
                truth_field = _select_level(truth_field, selection, level_tolerance)
                baseline_field = _select_level(baseline_field, selection, level_tolerance)
                refined_field = _select_level(refined_field, selection, level_tolerance)
                member_count = int(refined_field.sizes.get("member", 1))
                refined_member_values = None
                ensemble_spread = None
                if "member" in refined_field.dims:
                    refined_member_values = np.asarray(
                        refined_field.transpose(
                            "member", "latitude", "longitude"
                        ).values,
                        dtype=float,
                    )
                    if member_count > 1:
                        ensemble_spread = np.nanstd(
                            refined_member_values, axis=0, ddof=1
                        )
                    refined_field = refined_field.mean("member", skipna=True)
                cams = aligned_2d_values(truth_field, context="CAMS trace field")
                aurora = aligned_2d_values(baseline_field, context="Aurora trace field")
                refined_values = aligned_2d_values(
                    refined_field, context="refined trace field"
                )
                true_correction = cams - aurora
                predicted_correction = refined_values - aurora
                remaining_error = refined_values - cams
                scale_key, scale = _normalization_scale(selection, scales=scales)
                normalized_true = true_correction / scale
                normalized_predicted = predicted_correction / scale
                fields = {
                    "aurora": aurora,
                    "cams": cams,
                    "true_correction_physical": true_correction,
                    "true_correction_normalized": normalized_true,
                    "predicted_correction_physical": predicted_correction,
                    "predicted_correction_normalized": normalized_predicted,
                    "refined": refined_values,
                    "remaining_error_refined_minus_cams": remaining_error,
                }
                if ensemble_spread is not None:
                    fields["ensemble_spread_physical"] = ensemble_spread
                label = selection.label
                units = str(truth[selection.variable].attrs.get("units") or "")
                field_limits = _robust_limits([cams, aurora, refined_values], symmetric=False)
                correction_limits = _robust_limits(
                    [true_correction, predicted_correction, remaining_error], symmetric=True
                )
                normalized_limits = _robust_limits(
                    [normalized_true, normalized_predicted], symmetric=True
                )
                for field_name, values in fields.items():
                    output_fields.setdefault(field_name, []).append(values.astype(np.float32))
                    normalized = "normalized" in field_name
                    correction = (
                        "correction" in field_name or "remaining_error" in field_name
                    )
                    if normalized:
                        limits = normalized_limits
                    elif correction:
                        limits = correction_limits
                    elif "spread" in field_name:
                        limits = _robust_limits([values], symmetric=False)
                    else:
                        limits = field_limits
                    _render_map(
                        values,
                        latitude=latitude,
                        longitude=longitude,
                        title=(
                            f"{selection.variable} "
                            f"({'surface' if selection.level is None else f'{selection.level:g} hPa'}) "
                            f"{field_name}; init={initialization}; lead={lead_hours:g} h"
                        ),
                        units="1" if normalized else units,
                        path=map_dir / f"{label}_{field_name}.png",
                        limits=limits,
                        diverging=correction,
                    )
                entry: dict[str, Any] = {
                    "variable": selection.variable,
                    "level_hpa": selection.level,
                    "units": units,
                    "refined_data_variable": str(refined_variable.name),
                    "refined_variable_role": str(
                        refined_variable.attrs.get("aurora_refinement_role", "legacy_unspecified")
                    ),
                    "ensemble_members": member_count,
                    "ensemble_member_tensor": (
                        None
                        if refined_member_values is None
                        else _field_stats(
                            refined_member_values,
                            dimensions=("member", "latitude", "longitude"),
                        )
                    ),
                    "fraction_member_cells_equal_zero": (
                        None
                        if refined_member_values is None
                        else float(np.mean(refined_member_values == 0))
                    ),
                    "normalization_scale_key": scale_key,
                    "normalization_scale": scale,
                    "source_tensor_metadata": {
                        "CAMS": {
                            "name": str(truth_field.name),
                            "dimensions": list(truth_field.dims),
                            "shape": list(truth_field.shape),
                            "dtype": str(truth_field.dtype),
                            "device": "cpu",
                        },
                        "Aurora": {
                            "name": str(baseline_field.name),
                            "dimensions": list(baseline_field.dims),
                            "shape": list(baseline_field.shape),
                            "dtype": str(baseline_field.dtype),
                            "device": "cpu",
                        },
                        "refined_full_variable": {
                            "name": str(refined_variable.name),
                            "dimensions": list(refined_variable.dims),
                            "shape": list(refined_variable.shape),
                            "dtype": str(refined_variable.dtype),
                            "device": "cpu",
                        },
                        "refined_selected_ensemble_mean": {
                            "name": str(refined_field.name),
                            "dimensions": list(refined_field.dims),
                            "shape": list(refined_field.shape),
                            "dtype": str(refined_field.dtype),
                            "device": "cpu",
                        },
                    },
                    "fraction_final_cells_equal_zero": float(
                        np.mean(refined_values == 0)
                    ),
                    "predicted_to_true_correction_correlation": _correlation(
                        predicted_correction, true_correction
                    ),
                    "predicted_correction_to_aurora_correlation": _correlation(
                        predicted_correction, aurora
                    ),
                    "metrics": case_metrics(cams, aurora, refined_values),
                    "tensors": {
                        name: _field_stats(values) for name, values in fields.items()
                    },
                }
                if legacy is not None:
                    legacy_variable = resolve_forecast_variable(
                        legacy, selection.variable
                    )
                    legacy_field = legacy_variable.isel(
                        {legacy_step_dim: legacy_time_index}
                    )
                    legacy_field = _select_level(
                        legacy_field, selection, level_tolerance
                    )
                    if "member" in legacy_field.dims:
                        legacy_field = legacy_field.mean("member", skipna=True)
                    legacy_values = aligned_2d_values(
                        legacy_field, context="legacy refined trace field"
                    )
                    legacy_correction = legacy_values - aurora
                    legacy_remaining_error = legacy_values - cams
                    legacy_fields = {
                        "legacy_refined": legacy_values,
                        "legacy_predicted_correction_physical": legacy_correction,
                        "legacy_remaining_error_refined_minus_cams": (
                            legacy_remaining_error
                        ),
                    }
                    legacy_field_limits = _robust_limits(
                        [cams, aurora, refined_values, legacy_values], symmetric=False
                    )
                    legacy_correction_limits = _robust_limits(
                        [
                            true_correction,
                            predicted_correction,
                            legacy_correction,
                            remaining_error,
                            legacy_remaining_error,
                        ],
                        symmetric=True,
                    )
                    for field_name, values in legacy_fields.items():
                        output_fields.setdefault(field_name, []).append(
                            values.astype(np.float32)
                        )
                        correction = "correction" in field_name or "error" in field_name
                        _render_map(
                            values,
                            latitude=latitude,
                            longitude=longitude,
                            title=(
                                f"{selection.variable} "
                                f"({'surface' if selection.level is None else f'{selection.level:g} hPa'}) "
                                f"{field_name}; init={initialization}; lead={lead_hours:g} h"
                            ),
                            units=units,
                            path=map_dir / f"{label}_{field_name}.png",
                            limits=(
                                legacy_correction_limits
                                if correction
                                else legacy_field_limits
                            ),
                            diverging=correction,
                        )
                    entry["legacy_comparison"] = {
                        "data_variable": str(legacy_variable.name),
                        "metrics": case_metrics(cams, aurora, legacy_values),
                        "tensors": {
                            name: _field_stats(values)
                            for name, values in legacy_fields.items()
                        },
                        "refined_difference": _field_stats(
                            refined_values - legacy_values
                        ),
                    }
                trace["selections"][label] = entry
                selection_labels.append(label)
                selection_variables.append(selection.variable)
                selection_levels.append(
                    "surface" if selection.level is None else f"{selection.level:g}"
                )
                selection_units.append(units)
        finally:
            if legacy is not None:
                legacy.close()

    dataset = xr.Dataset(
        {
            name: (("selection", "latitude", "longitude"), np.stack(values))
            for name, values in output_fields.items()
        },
        coords={
            "selection": selection_labels,
            "variable": ("selection", selection_variables),
            "level": ("selection", selection_levels),
            "units": ("selection", selection_units),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "initialization_time": str(np.datetime64(initialization, "ns")),
            "valid_time": str(requested_valid_time),
            "lead_time_hours": float(lead_hours),
            "correction_convention": "CAMS - Aurora",
            "application_convention": "Aurora + predicted_correction_physical",
        },
    )
    dataset["true_correction_normalized"].attrs["units"] = "1"
    dataset["predicted_correction_normalized"].attrs["units"] = "1"
    dataset.to_netcdf(output_dir / "trace_fields.nc")
    dataset.close()
    (output_dir / "trace.json").write_text(
        json.dumps(_json_safe(trace), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--initialization", required=True, type=_parse_initialization)
    parser.add_argument("--lead-hours", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--finetuned-dir", type=Path)
    parser.add_argument("--legacy-refined-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = trace_batch(
        args.config.expanduser().resolve(),
        initialization=args.initialization,
        lead_hours=args.lead_hours,
        output_dir=args.output_dir.expanduser().resolve(),
        finetuned_dir=(
            None if args.finetuned_dir is None else args.finetuned_dir.expanduser().resolve()
        ),
        legacy_refined_dir=(
            None
            if args.legacy_refined_dir is None
            else args.legacy_refined_dir.expanduser().resolve()
        ),
        overwrite=args.overwrite,
    )
    print(f"Saved refinement batch trace: {output}")


if __name__ == "__main__":
    main()
