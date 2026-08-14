"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Generate overall spatial-difference maps for CAMS rollout evaluation.

The aggregation streams matched forecast cases and requested lead times. For
each configured target variable and level it writes:

* mean CAMS, Aurora, and fine-tuned fields;
* Aurora and fine-tuned bias (mean signed forecast-minus-CAMS error);
* spatial MAE improvement, defined as baseline MAE minus fine-tuned MAE.

Positive MAE improvement therefore means that fine-tuning reduced local
absolute error. No interpolation is performed.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.refinement.io import resolve_forecast_variable


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
            "aurora_bias": divide(self.baseline_error_sum),
            "finetuned_bias": divide(self.finetuned_error_sum),
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
    data_case_name = str(paths.get("data_case_name") or case_name).strip()
    case_data_dir = (
        data_dir if data_dir.name == data_case_name else data_dir / data_case_name
    )
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
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError(f"{name} coordinates must be finite.")
    if np.unique(reference).size != reference.size:
        raise ValueError(f"Reference {name} coordinates contain duplicate values.")
    if np.unique(candidate).size != candidate.size:
        raise ValueError(f"Candidate {name} coordinates contain duplicate values.")
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


def assert_same_coordinate(
    expected: np.ndarray,
    observed: np.ndarray,
    tolerance: float,
    *,
    name: str,
    context: str,
) -> None:
    """Require a coordinate to have the same values in the same order."""
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise ValueError(f"{context} {name} coordinates must be finite.")
    if np.unique(expected).size != expected.size or np.unique(observed).size != observed.size:
        raise ValueError(f"{context} {name} coordinates contain duplicate values.")
    if expected.shape != observed.shape or not np.allclose(
        expected, observed, atol=tolerance, rtol=0
    ):
        raise ValueError(
            f"{context} {name} coordinates do not match the evaluation grid in "
            f"value and order: expected shape {expected.shape}, observed "
            f"{observed.shape}."
        )


def rollout_valid_times(dataset: xr.Dataset, *, context: str) -> tuple[np.ndarray, str]:
    """Return a unique one-dimensional valid-time coordinate and its step dim."""
    if "time" not in dataset:
        raise KeyError(f"{context} has no forecast valid-time coordinate 'time'.")
    coordinate = dataset["time"]
    if coordinate.ndim != 1 or len(coordinate.dims) != 1:
        raise ValueError(
            f"{context} time must be one-dimensional, got dims {coordinate.dims}."
        )
    try:
        values = np.asarray(coordinate.values).astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} time contains invalid datetimes.") from exc
    if np.isnat(values).any():
        raise ValueError(f"{context} time contains NaT.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{context} time contains duplicate valid times.")
    return values, str(coordinate.dims[0])


def _datetime_candidates(dataset: xr.Dataset) -> list[tuple[str, np.datetime64]]:
    candidates: list[tuple[str, np.datetime64]] = []
    for name in ("forecast_reference_time", "init_time"):
        if name not in dataset:
            continue
        raw = np.asarray(dataset[name].values).reshape(-1)
        try:
            parsed = raw.astype("datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains invalid datetimes.") from exc
        if np.isnat(parsed).any() or np.unique(parsed).size != 1:
            raise ValueError(f"{name} must contain one non-NaT initialization time.")
        candidates.append((name, parsed[0]))
    if dataset.attrs.get("initialization_time") not in (None, ""):
        try:
            value = np.datetime64(dataset.attrs["initialization_time"], "ns")
        except (TypeError, ValueError) as exc:
            raise ValueError("initialization_time attribute is not a datetime.") from exc
        if np.isnat(value):
            raise ValueError("initialization_time attribute is NaT.")
        candidates.append(("initialization_time attribute", value))
    return candidates


def validate_rollout_time_metadata(
    dataset: xr.Dataset,
    expected_initialization: np.datetime64,
    *,
    context: str,
) -> tuple[np.ndarray, str]:
    """Validate init + lead = valid time and return valid times / step dim."""
    expected_initialization = np.datetime64(expected_initialization, "ns")
    candidates = _datetime_candidates(dataset)
    if not candidates:
        raise ValueError(
            f"{context} has no forecast_reference_time, init_time, or "
            "initialization_time metadata."
        )
    for name, value in candidates:
        if value != expected_initialization:
            raise ValueError(
                f"{context} {name}={value} disagrees with filename initialization "
                f"{expected_initialization}."
            )
    valid_times, step_dim = rollout_valid_times(dataset, context=context)
    derived_hours = (
        valid_times - expected_initialization
    ) / np.timedelta64(1, "h")
    for lead_name in ("lead_time_hours", "lead_time"):
        if lead_name not in dataset:
            continue
        raw = np.asarray(dataset[lead_name].values).reshape(-1)
        if np.issubdtype(raw.dtype, np.timedelta64):
            stored_hours = raw / np.timedelta64(1, "h")
        else:
            stored_hours = raw.astype(float)
        if stored_hours.size != valid_times.size or not np.allclose(
            stored_hours, derived_hours, atol=1.0e-6, rtol=0
        ):
            raise ValueError(
                f"{context} violates valid_time = initialization_time + forecast_lead: "
                f"stored {lead_name}={stored_hours.tolist()}, derived="
                f"{derived_hours.tolist()}."
            )
    return valid_times, step_dim


def match_valid_time_index(
    valid_times: np.ndarray,
    requested: np.datetime64,
    *,
    context: str,
) -> int:
    """Return the unique exact index of ``requested`` in a rollout."""
    requested = np.datetime64(requested, "ns")
    matches = np.flatnonzero(valid_times == requested)
    if len(matches) != 1:
        raise ValueError(
            f"{context} valid time {requested} matched {len(matches)} entries; "
            "forecast steps must be matched by valid time, never by array position."
        )
    return int(matches[0])


def aligned_2d_values(data_array: xr.DataArray, *, context: str) -> np.ndarray:
    """Require an already selected field to be exactly latitude x longitude."""
    required = {"latitude", "longitude"}
    if set(data_array.dims) != required:
        raise ValueError(
            f"{context} must have only latitude/longitude dimensions after time, "
            f"level, and member selection; got {data_array.dims}."
        )
    return np.asarray(data_array.transpose("latitude", "longitude").values, dtype=float)


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
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError(
            "Matplotlib and Cartopy are required only to render overall map PNGs. "
            "Install the plotting extras or use the metric/NetCDF evaluation helpers "
            "without plotting."
        ) from exc
    field_limits = robust_limits(
        [fields["mean_ground_truth"], fields["mean_baseline"], fields["mean_finetuned"]]
    )
    error_limit = max(
        abs(value)
        for value in robust_limits(
            [fields["aurora_bias"], fields["finetuned_bias"]]
        )
    )
    error_limit = error_limit or 1.0
    panels = [
        ("Mean CAMS", fields["mean_ground_truth"], "viridis", field_limits),
        ("Mean Aurora", fields["mean_baseline"], "viridis", field_limits),
        ("Mean fine-tuned", fields["mean_finetuned"], "viridis", field_limits),
        (
            "Aurora bias (Aurora − CAMS)",
            fields["aurora_bias"],
            "RdBu_r",
            (-error_limit, error_limit),
        ),
        (
            "Fine-tuned bias (fine-tuned − CAMS)",
            fields["finetuned_bias"],
            "RdBu_r",
            (-error_limit, error_limit),
        ),
    ]
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        6,
        height_ratios=(1.0, 0.06, 1.0, 0.06),
    )
    # The two bias maps are centered in the gaps between the three field maps.
    axes = [
        fig.add_subplot(grid[0, 0:2], projection=data_crs),
        fig.add_subplot(grid[0, 2:4], projection=data_crs),
        fig.add_subplot(grid[0, 4:6], projection=data_crs),
        fig.add_subplot(grid[2, 1:3], projection=data_crs),
        fig.add_subplot(grid[2, 3:5], projection=data_crs),
    ]
    top_colorbar_axis = fig.add_subplot(grid[1, 1:5])
    bottom_colorbar_axis = fig.add_subplot(grid[3, 1:5])
    meshes = []
    for panel_index, (axis, (title, values, cmap, limits)) in enumerate(
        zip(axes, panels)
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
        meshes.append(mesh)
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
        gridlines.left_labels = panel_index in {0, 3}
        axis.set_title(title, fontsize=16)
        if extent is None:
            axis.set_global()
        else:
            axis.set_extent(extent, crs=data_crs)
    fig.colorbar(
        meshes[0],
        cax=top_colorbar_axis,
        orientation="horizontal",
        label=units or None,
    )
    fig.colorbar(
        meshes[3],
        cax=bottom_colorbar_axis,
        orientation="horizontal",
        label=units or None,
    )
    level = "surface" if selection.level is None else f"{selection.level:g} hPa"
    fig.suptitle(
        f"Overall CAMS differences: {selection.variable} ({level})\n"
        f"{forecast_count} matched forecast cases across configured leads",
        fontsize=18,
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
        if np.isnat(truth_times).any() or np.unique(truth_times).size != truth_times.size:
            raise ValueError("CAMS truth time must contain unique, non-NaT valid times.")
        truth_lookup = {
            int(value.astype(np.int64)): index for index, value in enumerate(truth_times)
        }
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
        finally:
            first_finetuned.close()

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

        accumulators = {
            selection: Accumulator.create((len(latitude), len(longitude)))
            for selection in selections
        }
        lead_tolerance = 1.0e-6
        requested_leads = settings["lead_hours"]
        skipped_without_truth = 0
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
                initialization = np.datetime64(initialization_ns, "ns")
                baseline_context = str(baseline_files[initialization_ns])
                finetuned_context = str(finetuned_files[initialization_ns])
                valid_times, finetuned_step_dim = validate_rollout_time_metadata(
                    finetuned,
                    initialization,
                    context=finetuned_context,
                )
                baseline_valid_times, baseline_step_dim = validate_rollout_time_metadata(
                    baseline,
                    initialization,
                    context=baseline_context,
                )
                assert_same_coordinate(
                    latitude,
                    np.asarray(finetuned["latitude"].values, dtype=float),
                    tolerance,
                    name="latitude",
                    context=finetuned_context,
                )
                assert_same_coordinate(
                    longitude,
                    np.asarray(finetuned["longitude"].values, dtype=float),
                    tolerance,
                    name="longitude",
                    context=finetuned_context,
                )
                baseline_latitude_indices = match_coordinate_indices(
                    np.asarray(baseline["latitude"].values, dtype=float),
                    latitude,
                    tolerance,
                    name=f"{baseline_context} latitude",
                )
                baseline_longitude_indices = match_coordinate_indices(
                    np.asarray(baseline["longitude"].values, dtype=float),
                    longitude,
                    tolerance,
                    name=f"{baseline_context} longitude",
                )
                for finetuned_time_index, valid_time in enumerate(valid_times):
                    lead = float((valid_time - initialization) / np.timedelta64(1, "h"))
                    if not any(
                        np.isclose(lead, requested, atol=lead_tolerance, rtol=0)
                        for requested in requested_leads
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
                        variable = selection.variable
                        if variable not in baseline.data_vars:
                            raise KeyError(
                                f"{baseline_context} is missing configured variable "
                                f"{variable!r}."
                            )
                        finetuned_variable = resolve_forecast_variable(
                            finetuned, variable
                        )
                        if finetuned_step_dim not in finetuned_variable.dims:
                            raise ValueError(
                                f"{finetuned_context} forecast variable "
                                f"{finetuned_variable.name!r} does not use time step "
                                f"dimension {finetuned_step_dim!r}; dims are "
                                f"{finetuned_variable.dims}."
                            )
                        truth_field = truth[variable].isel(
                            time=truth_index,
                            latitude=truth_latitude_indices,
                            longitude=truth_longitude_indices,
                        )
                        baseline_field = baseline[variable].isel(
                            {
                                baseline_step_dim: baseline_time_index,
                                "latitude": baseline_latitude_indices,
                                "longitude": baseline_longitude_indices,
                            }
                        )
                        finetuned_field = finetuned_variable.isel(
                            {finetuned_step_dim: finetuned_time_index}
                        )
                        if selection.level is not None:
                            truth_level = select_level_index(
                                np.asarray(truth_field["level"].values, dtype=float),
                                selection.level,
                                settings["level_tolerance"],
                            )
                            baseline_level = select_level_index(
                                np.asarray(baseline_field["level"].values, dtype=float),
                                selection.level,
                                settings["level_tolerance"],
                            )
                            finetuned_level = select_level_index(
                                np.asarray(finetuned_field["level"].values, dtype=float),
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
                        truth_values = aligned_2d_values(
                            truth_field,
                            context=f"CAMS {variable!r} at {valid_time}",
                        )
                        baseline_values = aligned_2d_values(
                            baseline_field,
                            context=f"Aurora {variable!r} at {valid_time}",
                        )
                        finetuned_values = aligned_2d_values(
                            finetuned_field,
                            context=f"refined {variable!r} at {valid_time}",
                        )
                        expected_shape = (len(latitude), len(longitude))
                        for field_name, values in (
                            ("CAMS", truth_values),
                            ("Aurora", baseline_values),
                            ("refined", finetuned_values),
                        ):
                            if values.shape != expected_shape:
                                raise ValueError(
                                    f"{field_name} {variable!r} at {valid_time} has "
                                    f"shape {values.shape}; expected {expected_shape}."
                                )
                        accumulators[selection].update(
                            truth_values,
                            baseline_values,
                            finetuned_values,
                        )
            finally:
                baseline.close()
                finetuned.close()

        if skipped_without_truth:
            print(
                "Skipped "
                f"{skipped_without_truth} requested forecast steps outside the CAMS "
                "truth valid-time range; no nearest-time matching was attempted."
            )

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
                    "bias_definition": (
                        "mean signed forecast minus CAMS error; negative values are "
                        "low biases and positive values are high biases"
                    ),
                    "mae_improvement_definition": (
                        "Aurora spatial MAE minus fine-tuned spatial MAE; positive is better"
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
