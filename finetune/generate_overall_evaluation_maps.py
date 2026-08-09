"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Generate overall spatial-difference maps for CAMS rollout evaluation.

The aggregation streams matched forecast cases and requested lead times. For
each configured target variable and level it writes:

* mean ground truth, baseline, and fine-tuned fields;
* baseline and fine-tuned mean signed error;
* spatial MAE improvement, defined as baseline MAE minus fine-tuned MAE.

Positive MAE improvement therefore means that fine-tuning reduced local
absolute error. No interpolation is performed.
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm


@dataclass(frozen=True)
class Selection:
    """One configured target variable and vertical level."""

    variable: str
    level: float | None
    kind: str | None

    @property
    def label(self) -> str:
        """Return a file-safe selection label."""
        level = "surface" if self.level is None else f"{self.level:g}"
        return f"{file_safe(self.variable)}_{file_safe(level)}"


@dataclass
class Accumulator:
    """Streaming spatial sums over a common finite mask."""

    truth_sum: np.ndarray
    baseline_sum: np.ndarray
    finetuned_sum: np.ndarray
    baseline_error_sum: np.ndarray
    finetuned_error_sum: np.ndarray
    baseline_abs_error_sum: np.ndarray
    finetuned_abs_error_sum: np.ndarray
    valid_count: np.ndarray
    forecast_count: int = 0

    @classmethod
    def create(cls, shape: tuple[int, int]) -> "Accumulator":
        """Create zero-filled sums for one horizontal grid."""
        zeros = lambda: np.zeros(shape, dtype=np.float64)
        return cls(
            truth_sum=zeros(),
            baseline_sum=zeros(),
            finetuned_sum=zeros(),
            baseline_error_sum=zeros(),
            finetuned_error_sum=zeros(),
            baseline_abs_error_sum=zeros(),
            finetuned_abs_error_sum=zeros(),
            valid_count=np.zeros(shape, dtype=np.int64),
        )

    def update(
        self,
        truth: np.ndarray,
        baseline: np.ndarray,
        finetuned: np.ndarray,
    ) -> None:
        """Add one aligned forecast case using a common finite mask."""
        mask = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(finetuned)
        if not np.any(mask):
            return
        baseline_error = baseline - truth
        finetuned_error = finetuned - truth
        self.truth_sum[mask] += truth[mask]
        self.baseline_sum[mask] += baseline[mask]
        self.finetuned_sum[mask] += finetuned[mask]
        self.baseline_error_sum[mask] += baseline_error[mask]
        self.finetuned_error_sum[mask] += finetuned_error[mask]
        self.baseline_abs_error_sum[mask] += np.abs(baseline_error[mask])
        self.finetuned_abs_error_sum[mask] += np.abs(finetuned_error[mask])
        self.valid_count[mask] += 1
        self.forecast_count += 1

    def means(self) -> dict[str, np.ndarray]:
        """Return aggregate fields, with NaN where no valid case exists."""
        count = self.valid_count

        def divide(values: np.ndarray) -> np.ndarray:
            output = np.full(values.shape, np.nan, dtype=np.float64)
            np.divide(values, count, out=output, where=count > 0)
            return output

        baseline_mae = divide(self.baseline_abs_error_sum)
        finetuned_mae = divide(self.finetuned_abs_error_sum)
        return {
            "mean_ground_truth": divide(self.truth_sum),
            "mean_baseline": divide(self.baseline_sum),
            "mean_finetuned": divide(self.finetuned_sum),
            "baseline_mean_error": divide(self.baseline_error_sum),
            "finetuned_mean_error": divide(self.finetuned_error_sum),
            "baseline_mae": baseline_mae,
            "finetuned_mae": finetuned_mae,
            "mae_improvement": baseline_mae - finetuned_mae,
        }


def file_safe(value: Any) -> str:
    """Convert a value to a conservative file-safe token."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("._-") or "unnamed"


def repository_root(start: Path) -> Path:
    """Return the closest parent containing `.git`."""
    start = start.resolve()
    return next(
        (candidate for candidate in (start, *start.parents) if (candidate / ".git").exists()),
        start,
    )


def resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_from_base(value: str | Path, base: Path) -> Path:
    """Resolve a training path relative to its configured project directory."""
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def configured_map_extent(
    raw: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    """Return the YAML domain as a Cartopy extent, or None for an unbounded domain."""
    data = raw.get("data") or {}
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping or null")

    names = ("lon_min", "lon_max", "lat_min", "lat_max")
    raw_values = [data.get(name) for name in names]

    def is_missing(value: Any) -> bool:
        return value is None or (
            isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
        )

    missing = [is_missing(value) for value in raw_values]
    if all(missing):
        return None
    if any(missing):
        absent = [name for name, value_missing in zip(names, missing) if value_missing]
        raise ValueError(
            "A regional map requires all data boundaries; missing "
            + ", ".join(absent)
            + "."
        )

    try:
        lon_min, lon_max, lat_min, lat_max = map(float, raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("data map boundaries must be numeric or null") from exc
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(
            f"Invalid latitude boundaries: lat_min={lat_min:g}, lat_max={lat_max:g}."
        )
    if not lon_min < lon_max:
        raise ValueError(
            f"Invalid longitude boundaries: lon_min={lon_min:g}, lon_max={lon_max:g}."
        )

    normalized_lon_min = ((lon_min + 180.0) % 360.0) - 180.0
    normalized_lon_max = ((lon_max + 180.0) % 360.0) - 180.0
    if normalized_lon_min >= normalized_lon_max:
        raise ValueError(
            "Longitude boundaries crossing the antimeridian are not supported by "
            "the current map layout."
        )
    return normalized_lon_min, normalized_lon_max, lat_min, lat_max


def load_settings(config_path: Path) -> dict[str, Any]:
    """Load evaluation settings using the notebook's path-derived defaults."""
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    case_name = str(raw.get("case_name") or "").strip()
    if not case_name:
        raise ValueError(f"YAML must define case_name: {config_path}")
    root = repository_root(config_path.parent)
    paths = raw.get("paths") or {}
    if not isinstance(paths, dict):
        raise TypeError("paths must be a mapping or null")
    evaluation = raw.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a mapping or null")

    project_dir = Path(paths.get("project_root", config_path.parent)).expanduser()
    if not project_dir.is_absolute():
        project_dir = (config_path.parent / project_dir).resolve()
    else:
        project_dir = project_dir.resolve()

    data_dir = resolve_from_base(paths.get("data_dir", project_dir / "data"), project_dir)
    case_data_dir = data_dir if data_dir.name == case_name else data_dir / case_name
    output_base = resolve_from_base(
        paths.get("output_dir", project_dir / "outputs"), project_dir
    )
    case_output_dir = (
        output_base if output_base.name == case_name else output_base / case_name
    )
    baseline_candidates = (
        root / "outputs" / "cams_rollouts",
        root / "examples" / "outputs" / "cams_rollouts",
    )
    baseline_default = next(
        (candidate for candidate in baseline_candidates if candidate.exists()),
        baseline_candidates[0],
    )

    def configured_path(key: str, default: Path) -> Path:
        value = evaluation.get(key)
        return default.resolve() if value in (None, "") else resolve_path(value, root)

    lead_hours = evaluation.get("lead_times_hours")
    if lead_hours is None:
        lead_indices = (raw.get("data") or {}).get("target_lead_times")
        step_hours = (raw.get("rollout") or {}).get("rollout_step_hours")
        lead_hours = (
            [float(value) * float(step_hours) for value in lead_indices]
            if lead_indices and step_hours is not None
            else [12, 24, 36, 48, 60, 72]
        )

    return {
        "raw": raw,
        "case_name": case_name,
        "root": root,
        "truth_path": configured_path("ground_truth_path", case_data_dir / "test.nc"),
        "map_extent": configured_map_extent(raw),
        "baseline_dir": configured_path("baseline_rollout_path", baseline_default),
        "finetuned_dir": configured_path("finetuned_rollout_path", case_output_dir),
        "output_dir": configured_path("output_dir", case_output_dir / "evaluation"),
        "lead_hours": [float(value) for value in lead_hours],
        "coordinate_tolerance": float(evaluation.get("coordinate_tolerance", 2.0e-5)),
        "level_tolerance": float(evaluation.get("level_tolerance", 1.0e-3)),
    }


def target_selections(raw: Mapping[str, Any]) -> list[Selection]:
    """Read targets and requested levels from the YAML."""
    evaluation = raw.get("evaluation") or {}
    items = evaluation.get("target_variables")
    if items is None:
        items = (raw.get("data") or {}).get("target_variables")
    if not items:
        raise ValueError("No target_variables are configured.")
    configured_levels = evaluation.get("levels")
    atmos_levels = (raw.get("data") or {}).get("atmos_levels")
    selections: list[Selection] = []
    for item in items:
        if isinstance(item, str):
            name, kind, loss_levels = item, None, None
        else:
            name = str(item.get("dataset_name") or item.get("name") or "").strip()
            kind = None if item.get("kind") is None else str(item.get("kind")).lower()
            loss_levels = item.get("loss_levels")
        if not name:
            raise ValueError(f"Invalid target variable: {item!r}")
        if kind in {"surf", "surface"}:
            levels: Iterable[float | None] = [None]
        else:
            if isinstance(configured_levels, Mapping):
                levels = configured_levels.get(name) or loss_levels or atmos_levels
            else:
                levels = configured_levels or loss_levels or atmos_levels
            if not levels:
                raise ValueError(f"No levels could be determined for atmospheric target {name}.")
            levels = [float(value) for value in levels]
        selections.extend(Selection(name, level, kind) for level in levels)
    return selections


def timestamp_from_name(path: Path) -> np.datetime64 | None:
    """Parse known rollout initialization timestamps from a filename."""
    patterns = (
        (r"init_(\d{8}T\d{6})", "%Y%m%dT%H%M%S"),
        (r"rollout_(\d{8}_\d{6})", "%Y%m%d_%H%M%S"),
    )
    for pattern, date_format in patterns:
        match = re.search(pattern, path.stem)
        if match:
            import datetime as dt

            return np.datetime64(dt.datetime.strptime(match.group(1), date_format), "ns")
    return None


def rollout_files_by_initialization(
    directory: Path,
    pattern: str,
) -> dict[int, Path]:
    """Index rollout paths by filename initialization timestamp."""
    result: dict[int, Path] = {}
    for path in sorted(directory.glob(pattern)):
        initialization = timestamp_from_name(path)
        if initialization is None:
            continue
        key = int(initialization.astype(np.int64))
        if key in result:
            raise ValueError(
                f"Duplicate rollout initialization {initialization}: {result[key]} and {path}"
            )
        result[key] = path
    return result


def match_coordinate_indices(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance: float,
    *,
    name: str,
) -> np.ndarray:
    """Match every candidate coordinate to one reference coordinate."""
    indices: list[int] = []
    for value in candidate:
        matches = np.flatnonzero(np.isclose(reference, value, atol=tolerance, rtol=0))
        if len(matches) != 1:
            raise ValueError(
                f"{name}={value:g} matched {len(matches)} truth coordinates within "
                f"tolerance {tolerance:g}; no interpolation was attempted."
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def select_level_index(values: np.ndarray, level: float, tolerance: float) -> int:
    """Find exactly one level coordinate within tolerance."""
    matches = np.flatnonzero(np.isclose(values, level, atol=tolerance, rtol=0))
    if len(matches) != 1:
        raise ValueError(f"Level {level:g} matched {len(matches)} values in {values.tolist()}.")
    return int(matches[0])


def robust_limits(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    """Return shared 1st-to-99th-percentile limits."""
    finite_parts = [values[np.isfinite(values)] for values in arrays]
    finite_parts = [values for values in finite_parts if values.size]
    if not finite_parts:
        return 0.0, 1.0
    values = np.concatenate(finite_parts)
    low, high = np.nanpercentile(values, [1, 99])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def plot_overall(
    fields: Mapping[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
    selection: Selection,
    units: str,
    forecast_count: int,
    extent: tuple[float, float, float, float] | None,
    path: Path,
) -> None:
    """Plot globally or within the configured regional YAML domain."""
    field_limits = robust_limits(
        [fields["mean_ground_truth"], fields["mean_baseline"], fields["mean_finetuned"]]
    )
    error_limit = max(
        abs(value)
        for value in robust_limits(
            [fields["baseline_mean_error"], fields["finetuned_mean_error"]]
        )
    )
    improvement_limit = max(
        abs(value) for value in robust_limits([fields["mae_improvement"]])
    )
    error_limit = error_limit or 1.0
    improvement_limit = improvement_limit or 1.0
    panels = [
        ("Mean ground truth", fields["mean_ground_truth"], "viridis", field_limits),
        ("Mean baseline", fields["mean_baseline"], "viridis", field_limits),
        ("Mean fine-tuned", fields["mean_finetuned"], "viridis", field_limits),
        (
            "Baseline mean error",
            fields["baseline_mean_error"],
            "RdBu_r",
            (-error_limit, error_limit),
        ),
        (
            "Fine-tuned mean error",
            fields["finetuned_mean_error"],
            "RdBu_r",
            (-error_limit, error_limit),
        ),
        (
            "MAE improvement: baseline − fine-tuned",
            fields["mae_improvement"],
            "RdBu",
            (-improvement_limit, improvement_limit),
        ),
    ]
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 8.5),
        constrained_layout=True,
        subplot_kw={"projection": data_crs},
    )
    for panel_index, (axis, (title, values, cmap, limits)) in enumerate(
        zip(axes.flat, panels)
    ):
        # These features remain visible wherever the evaluated field is missing,
        # while coastlines and borders are drawn above the data for orientation.
        axis.add_feature(
            cfeature.OCEAN.with_scale("110m"),
            facecolor="#eaf3f8",
            edgecolor="none",
            zorder=0,
        )
        axis.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="#f2f2f2",
            edgecolor="none",
            zorder=0,
        )
        mesh = axis.pcolormesh(
            longitude,
            latitude,
            values,
            transform=data_crs,
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            shading="auto",
            rasterized=True,
            zorder=1,
        )
        axis.coastlines(
            resolution="110m", color="#202020", linewidth=0.55, zorder=2
        )
        axis.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            edgecolor="#404040",
            linewidth=0.35,
            zorder=2,
        )
        gridlines = axis.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.35,
            color="#505050",
            alpha=0.35,
            linestyle=":",
            x_inline=False,
            y_inline=False,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.bottom_labels = panel_index >= 3
        gridlines.left_labels = panel_index % 3 == 0
        fig.colorbar(mesh, ax=axis, shrink=0.8, label=units or None)
        axis.set_title(title)
        if extent is None:
            axis.set_global()
        else:
            axis.set_extent(extent, crs=data_crs)
    level = "surface" if selection.level is None else f"{selection.level:g} hPa"
    fig.suptitle(
        f"Overall CAMS differences: {selection.variable} ({level})\n"
        f"{forecast_count} matched forecast cases across configured leads; "
        "positive MAE improvement is better"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_overall_maps(config_path: Path) -> list[Path]:
    """Stream all matched cases and write aggregate NetCDF and PNG maps."""
    settings = load_settings(config_path)
    selections = target_selections(settings["raw"])
    output_dir = settings["output_dir"]
    figure_dir = output_dir / "figures" / "overall"
    figure_dir.mkdir(parents=True, exist_ok=True)

    baseline_files = rollout_files_by_initialization(
        settings["baseline_dir"], "rollout_*.nc"
    )
    finetuned_files = rollout_files_by_initialization(
        settings["finetuned_dir"], "rollout_predictions_init_*.nc"
    )
    common_initializations = sorted(set(baseline_files) & set(finetuned_files))
    if not common_initializations:
        raise FileNotFoundError(
            "No common baseline/fine-tuned rollout initializations. "
            f"Found {len(baseline_files)} timestamped baseline files under "
            f"{settings['baseline_dir']} and {len(finetuned_files)} timestamped "
            f"fine-tuned files under {settings['finetuned_dir']}."
        )

    truth = xr.open_dataset(settings["truth_path"], decode_times=True)
    try:
        truth_times = np.asarray(truth["time"].values).astype("datetime64[ns]")
        truth_lookup = {
            int(value.astype(np.int64)): index for index, value in enumerate(truth_times)
        }
        first_baseline = xr.open_dataset(baseline_files[common_initializations[0]])
        first_finetuned = xr.open_dataset(finetuned_files[common_initializations[0]])
        try:
            tolerance = settings["coordinate_tolerance"]
            latitude = np.asarray(first_finetuned["latitude"].values, dtype=float)
            longitude = np.asarray(first_finetuned["longitude"].values, dtype=float)
            truth_latitude_indices = match_coordinate_indices(
                np.asarray(truth["latitude"].values, dtype=float),
                latitude,
                tolerance,
                name="latitude",
            )
            truth_longitude_indices = match_coordinate_indices(
                np.asarray(truth["longitude"].values, dtype=float),
                longitude,
                tolerance,
                name="longitude",
            )
            baseline_latitude_indices = match_coordinate_indices(
                np.asarray(first_baseline["latitude"].values, dtype=float),
                latitude,
                tolerance,
                name="latitude",
            )
            baseline_longitude_indices = match_coordinate_indices(
                np.asarray(first_baseline["longitude"].values, dtype=float),
                longitude,
                tolerance,
                name="longitude",
            )
        finally:
            first_baseline.close()
            first_finetuned.close()

        accumulators = {
            selection: Accumulator.create((len(latitude), len(longitude)))
            for selection in selections
        }
        lead_tolerance = 1.0e-6
        requested_leads = settings["lead_hours"]
        for initialization_ns in tqdm(
            common_initializations,
            desc="Aggregate overall maps",
            unit="initialization",
        ):
            baseline = xr.open_dataset(
                baseline_files[initialization_ns], decode_times=True
            )
            finetuned = xr.open_dataset(
                finetuned_files[initialization_ns], decode_times=True
            )
            try:
                valid_times = np.asarray(finetuned["time"].values).astype("datetime64[ns]")
                initialization = np.datetime64(initialization_ns, "ns")
                for time_index, valid_time in enumerate(valid_times):
                    lead = float((valid_time - initialization) / np.timedelta64(1, "h"))
                    if not any(
                        np.isclose(lead, requested, atol=lead_tolerance, rtol=0)
                        for requested in requested_leads
                    ):
                        continue
                    truth_index = truth_lookup.get(int(valid_time.astype(np.int64)))
                    if truth_index is None:
                        continue
                    for selection in selections:
                        variable = selection.variable
                        truth_field = truth[variable].isel(
                            time=truth_index,
                            latitude=truth_latitude_indices,
                            longitude=truth_longitude_indices,
                        )
                        baseline_field = baseline[variable].isel(
                            time=time_index,
                            latitude=baseline_latitude_indices,
                            longitude=baseline_longitude_indices,
                        )
                        finetuned_field = finetuned[variable].isel(time=time_index)
                        if selection.level is not None:
                            truth_level = select_level_index(
                                np.asarray(truth["level"].values, dtype=float),
                                selection.level,
                                settings["level_tolerance"],
                            )
                            baseline_level = select_level_index(
                                np.asarray(baseline["level"].values, dtype=float),
                                selection.level,
                                settings["level_tolerance"],
                            )
                            finetuned_level = select_level_index(
                                np.asarray(finetuned["level"].values, dtype=float),
                                selection.level,
                                settings["level_tolerance"],
                            )
                            truth_field = truth_field.isel(level=truth_level)
                            baseline_field = baseline_field.isel(level=baseline_level)
                            finetuned_field = finetuned_field.isel(level=finetuned_level)
                        # Select time/level before reducing the ensemble so the
                        # backend reads only the requested 2-D member slices.
                        if "member" in finetuned_field.dims:
                            finetuned_field = finetuned_field.mean("member", skipna=True)
                        accumulators[selection].update(
                            np.asarray(truth_field.values, dtype=float),
                            np.asarray(baseline_field.values, dtype=float),
                            np.asarray(finetuned_field.values, dtype=float),
                        )
            finally:
                baseline.close()
                finetuned.close()

        normalized_longitude = ((longitude + 180.0) % 360.0) - 180.0
        longitude_order = np.argsort(normalized_longitude)
        normalized_longitude = normalized_longitude[longitude_order]
        generated: list[Path] = []
        datasets: list[xr.Dataset] = []
        for selection in selections:
            accumulator = accumulators[selection]
            fields = {
                name: values[:, longitude_order]
                for name, values in accumulator.means().items()
            }
            units = str(truth[selection.variable].attrs.get("units") or "")
            plot_path = figure_dir / f"{selection.label}_overall_differences.png"
            plot_overall(
                fields,
                latitude,
                normalized_longitude,
                selection,
                units,
                accumulator.forecast_count,
                settings["map_extent"],
                plot_path,
            )
            generated.append(plot_path)
            ds = xr.Dataset(
                {
                    name: (("latitude", "longitude"), values.astype(np.float32))
                    for name, values in fields.items()
                },
                coords={
                    "latitude": latitude,
                    "longitude": normalized_longitude,
                },
                attrs={
                    "mae_improvement_definition": (
                        "baseline spatial MAE minus fine-tuned spatial MAE; positive is better"
                    ),
                },
            )
            ds["valid_count"] = (
                ("latitude", "longitude"),
                accumulator.valid_count[:, longitude_order].astype(np.int32),
            )
            ds = ds.expand_dims(selection=[selection.label])
            ds = ds.assign_coords(
                variable=("selection", [selection.variable]),
                level=(
                    "selection",
                    ["surface" if selection.level is None else f"{selection.level:g}"],
                ),
                units=("selection", [units]),
                number_of_forecasts=(
                    "selection",
                    [accumulator.forecast_count],
                ),
            )
            datasets.append(ds)

        aggregate = xr.concat(datasets, dim="selection", join="exact")
        aggregate_path = output_dir / "overall_spatial_differences.nc"
        aggregate.to_netcdf(aggregate_path)
        aggregate.close()
        for ds in datasets:
            ds.close()
        print(f"Saved aggregate spatial data: {aggregate_path}")
        for generated_path in generated:
            print(f"Saved overall map: {generated_path}")
        return generated
    finally:
        truth.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Evaluation YAML path.")
    args = parser.parse_args()
    generate_overall_maps(args.config.expanduser().resolve())


if __name__ == "__main__":
    main()
