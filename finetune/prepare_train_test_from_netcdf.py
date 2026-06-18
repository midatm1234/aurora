#!/usr/bin/env python3
"""Prepare Aurora fine-tuning train/validation/test NetCDF files from CAMS-style inputs.

This script:
1. Reads `predictor_variables` and `target_variables` from a rollout config YAML.
2. Loads and merges multiple surface/atmospheric NetCDF files.
3. Converts CAMS forecast dims to Aurora-style dims (`time`, `level`, `latitude`, `longitude`).
4. Splits by date ranges into train, validation (optional), and test sets.

Example:
    # Run with defaults configured below:
    python finetune/prepare_train_test_from_netcdf.py

    # Or override specific values:
    python finetune/prepare_train_test_from_netcdf.py \
      --val-start-time 2026-03-14T00:00:00 \
      --test-start-time 2026-03-16T00:00:00
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import threading
import time
from collections import OrderedDict
from glob import glob as _glob
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


# ============================================================================
# CONFIGURATION: Edit these defaults to match your setup
# ============================================================================

CONFIG = {
    # Path to finetune config YAML
    "config": "aurora_O3_finetune_US-WEST_3day_lead_config_v4.yaml",

    # Folder containing input NetCDF files (searched with glob patterns)
    "data_folder": "/data/cams",

    # Output NetCDF files. These are derived from paths.data_dir/case_name
    # in the YAML config unless explicitly overridden to the same case folder.
    "train_out": None,
    "val_out": None,  # Set to path for validation set, or None to skip
    "test_out": None,

    # Date-based split (ISO format: YYYY-MM-DDTHH:MM:SS)
    "train_start_time": "2023-07-01T00:00:00",
    "train_end_time": "2024-06-30T12:00:00",
    "test_start_time": "2024-07-01T00:00:00",
    "test_end_time": "2024-09-30T12:00:00",

    # Spatial domain subset (set to None to disable)
    "lat_min": 31.0,
    "lat_max": 52.0,
    "lon_min": -128.0,
    "lon_max": -100.0,

    # NetCDF compression level (0-9)
    "compression_level": 1,

    # Dry run: print info without writing files
    "dry_run": False,

    # Stream each input file pair directly into output NetCDFs. This avoids
    # materialising the full multi-file dataset in memory.
    "streaming_write": True,
}

# ============================================================================

SURFACE_KINDS = {"surf", "surface"}
ATMOS_KINDS = {"atmos", "atmospheric"}


def _log(message: str) -> None:
    print(message, flush=True)


def _progress(iterable, *, desc: str, unit: str = "item", total: int | None = None):
    if tqdm is None:
        _log(f"{desc}...")
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        dynamic_ncols=True,
    )


@contextlib.contextmanager
def _elapsed_status(label: str):
    """Print elapsed time while a long blocking xarray/netCDF call runs."""
    if tqdm is not None:
        start = time.monotonic()
        bar = tqdm(
            total=0,
            bar_format="{desc}",
            dynamic_ncols=True,
            leave=True,
        )
        stop = threading.Event()

        def update() -> None:
            for tick in itertools.cycle("|/-\\"):
                if stop.wait(1.0):
                    return
                elapsed = time.monotonic() - start
                bar.set_description_str(f"{label} {tick} elapsed {elapsed:,.0f}s")

        thread = threading.Thread(target=update, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2.0)
            elapsed = time.monotonic() - start
            bar.set_description_str(f"{label} done in {elapsed:,.1f}s")
            bar.close()
        return

    start = time.monotonic()
    _log(f"{label}...")
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        _log(f"{label} done in {elapsed:,.1f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test NetCDF files from multiple surface/atmospheric inputs. "
                    "Edit the CONFIG dict at the top of this file to set defaults.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG["config"],
        help=f"Path to finetune config YAML (default: {CONFIG['config']}).",
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=CONFIG["data_folder"],
        help=f"Folder to search for surface/atmospheric NetCDF files (default: {CONFIG['data_folder']}).",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=CONFIG["train_out"],
        help="Optional output train.nc path; must be inside paths.data_dir/case_name.",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=CONFIG["val_out"],
        help="Output validation.nc path (default: None, skips validation set).",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=CONFIG["test_out"],
        help="Optional output test.nc path; must be inside paths.data_dir/case_name.",
    )
    parser.add_argument(
        "--train-start-time",
        type=str,
        default=CONFIG["train_start_time"],
        help="ISO timestamp; train set starts at this time.",
    )
    parser.add_argument(
        "--train-end-time",
        type=str,
        default=CONFIG["train_end_time"],
        help="ISO timestamp; train set ends at this time (inclusive).",
    )
    parser.add_argument(
        "--test-start-time",
        type=str,
        default=CONFIG["test_start_time"],
        help="ISO timestamp; test set starts at this time.",
    )
    parser.add_argument(
        "--test-end-time",
        type=str,
        default=CONFIG["test_end_time"],
        help="ISO timestamp; test set ends at this time (inclusive).",
    )
    parser.add_argument(
        "--lat-min", type=float, default=CONFIG["lat_min"],
        help=f"Min latitude for spatial subset (default: {CONFIG['lat_min']}). Set to None to disable.",
    )
    parser.add_argument(
        "--lat-max", type=float, default=CONFIG["lat_max"],
        help=f"Max latitude for spatial subset (default: {CONFIG['lat_max']}).",
    )
    parser.add_argument(
        "--lon-min", type=float, default=CONFIG["lon_min"],
        help=f"Min longitude for spatial subset (default: {CONFIG['lon_min']}).",
    )
    parser.add_argument(
        "--lon-max", type=float, default=CONFIG["lon_max"],
        help=f"Max longitude for spatial subset (default: {CONFIG['lon_max']}).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=CONFIG["compression_level"],
        help=f"NetCDF zlib compression level 0-9 (default: {CONFIG['compression_level']}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=CONFIG["dry_run"],
        help="Print selected variables and split sizes without writing outputs.",
    )
    parser.add_argument(
        "--streaming-write",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["streaming_write"],
        help="Write train/val/test files incrementally by input time chunk.",
    )
    return parser.parse_args()


def _read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_case_data_paths(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    train_out: Path | None,
    val_out: Path | None,
    test_out: Path | None,
) -> tuple[Path, Path | None, Path, Path]:
    case_name = str(cfg.get("case_name", "")).strip()
    if not case_name:
        raise ValueError(
            f"Config at {config_path} must define a non-empty top-level `case_name`. "
            "Prepared train/test data are written under paths.data_dir/case_name."
        )

    paths_cfg = cfg.get("paths", {})
    project_root = Path(paths_cfg.get("project_root", config_path.parent)).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    data_root = _resolve_path(paths_cfg.get("data_dir", project_root / "data"), project_root)
    case_data_dir = data_root if data_root.name == case_name else data_root / case_name
    case_data_dir = case_data_dir.resolve()

    resolved_train = (case_data_dir / "train.nc").resolve()
    resolved_test = (case_data_dir / "test.nc").resolve()
    resolved_val = None if val_out is None else _resolve_path(val_out, project_root)

    if train_out is not None:
        resolved_train = _resolve_path(train_out, project_root)
    if test_out is not None:
        resolved_test = _resolve_path(test_out, project_root)

    # Outputs must live in the case-specific data folder so prepared data stays
    # co-located with checkpoints/outputs for the same case_name. If a stale or
    # mismatched --{train,test,val}-out is supplied (e.g. re-running an old
    # command from a previous case), redirect it into case_data_dir keeping the
    # given filename rather than hard-failing.
    def _coerce_into_case_dir(label: str, path: Path | None) -> Path | None:
        if path is None:
            return None
        if path.parent.resolve() != case_data_dir:
            redirected = (case_data_dir / path.name).resolve()
            print(
                f"[prepare] WARNING: {label} output {path} is not inside the "
                f"case folder for case_name; redirecting to {redirected}.",
                flush=True,
            )
            return redirected
        return path

    resolved_train = _coerce_into_case_dir("train", resolved_train)
    resolved_test = _coerce_into_case_dir("test", resolved_test)
    resolved_val = _coerce_into_case_dir("validation", resolved_val)

    return resolved_train, resolved_val, resolved_test, case_data_dir


def _normalize_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    kind_norm = str(kind).strip().lower()
    if kind_norm in SURFACE_KINDS:
        return "surf"
    if kind_norm in ATMOS_KINDS:
        return "atmos"
    if kind_norm == "static":
        return "static"
    raise ValueError(f"Unsupported variable kind: {kind!r}")


def _extract_requested_variables(cfg: dict[str, Any]) -> OrderedDict[str, str | None]:
    data_cfg = cfg.get("data", {})
    requested: OrderedDict[str, str | None] = OrderedDict()

    def add_items(items: Any) -> None:
        if not items:
            return
        for item in items:
            if isinstance(item, str):
                name = item
                kind = None
            elif isinstance(item, dict):
                name = str(item.get("dataset_name") or item.get("name") or "").strip()
                if not name:
                    raise ValueError(f"Invalid variable item: {item!r}")
                kind = _normalize_kind(item.get("kind"))
            else:
                raise TypeError(f"Unsupported variable spec type: {type(item)}")

            existing = requested.get(name)
            if existing is None:
                requested[name] = kind
            elif kind is not None and existing is not None and kind != existing:
                raise ValueError(
                    f"Variable `{name}` has conflicting kinds in config: {existing!r} vs {kind!r}."
                )
            elif existing is None and kind is not None:
                requested[name] = kind

    add_items(data_cfg.get("predictor_variables"))
    add_items(data_cfg.get("target_variables"))
    return requested


def _first_present_name(options: list[str], present: set[str]) -> str | None:
    for name in options:
        if name in present:
            return name
    return None


def _collapse_forecast_dims(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    dims = set(ds.dims)
    frt_name = _first_present_name(["forecast_reference_time", "time"], dims)
    fp_name = _first_present_name(["forecast_period", "step", "leadtime", "lead_time"], dims)

    if frt_name is None:
        if time_dim in ds.dims:
            return ds
        raise ValueError(
            "Could not find a forecast reference time dimension. Expected one of "
            "`forecast_reference_time` or `time`."
        )

    if fp_name is not None:
        stacked_dim = "__stacked_time__"
        ds_stacked = ds.stack({stacked_dim: (frt_name, fp_name)})

        # Prefer valid_time if provided.
        if "valid_time" in ds and set(ds["valid_time"].dims) == {frt_name, fp_name}:
            vt = ds["valid_time"].stack({stacked_dim: (frt_name, fp_name)})
            time_values = np.asarray(vt.values)
        else:
            frt_vals = np.asarray(ds[frt_name].values).astype("datetime64[ns]")
            fp_vals = np.asarray(ds[fp_name].values).astype("timedelta64[ns]")
            time_values = (frt_vals[:, None] + fp_vals[None, :]).reshape(-1)

        ds_stacked = ds_stacked.reset_index(stacked_dim, drop=True)
        ds_stacked = ds_stacked.rename({stacked_dim: time_dim})
        ds_stacked = ds_stacked.assign_coords({time_dim: time_values})
        ds = ds_stacked.sortby(time_dim)
    elif frt_name != time_dim:
        ds = ds.rename({frt_name: time_dim})

    if "valid_time" in ds.coords and time_dim in ds["valid_time"].dims:
        ds = ds.assign_coords({time_dim: ds["valid_time"]})

    if "valid_time" in ds:
        ds = ds.drop_vars("valid_time")
    return ds


def _normalize_dims(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))
    lat_dim = str(data_cfg.get("lat_dim", "latitude"))
    lon_dim = str(data_cfg.get("lon_dim", "longitude"))
    level_dim = str(data_cfg.get("level_dim", "level"))

    ds = _collapse_forecast_dims(ds, time_dim=time_dim)

    dim_names = set(ds.dims)
    rename_map: dict[str, str] = {}

    src_lat = _first_present_name([lat_dim, "latitude", "lat"], dim_names)
    src_lon = _first_present_name([lon_dim, "longitude", "lon"], dim_names)
    src_level = _first_present_name([level_dim, "pressure_level", "isobaricInhPa"], dim_names)

    if src_lat is not None and src_lat != lat_dim:
        rename_map[src_lat] = lat_dim
    if src_lon is not None and src_lon != lon_dim:
        rename_map[src_lon] = lon_dim
    if src_level is not None and src_level != level_dim:
        rename_map[src_level] = level_dim

    if rename_map:
        ds = ds.rename(rename_map)

    if level_dim in ds.dims and "atmos_levels" in data_cfg and data_cfg["atmos_levels"]:
        requested_levels = [float(x) for x in data_cfg["atmos_levels"]]
        ds = ds.sel({level_dim: requested_levels})

    return ds


def _split_surface_and_atmos_vars(
    requested: OrderedDict[str, str | None],
    surf_ds: xr.Dataset,
    atmos_ds: xr.Dataset,
) -> tuple[list[str], list[str]]:
    surf_vars: list[str] = []
    atmos_vars: list[str] = []
    missing: list[str] = []

    for name, kind in requested.items():
        in_surf = name in surf_ds.data_vars
        in_atmos = name in atmos_ds.data_vars

        if kind == "surf":
            if not in_surf:
                missing.append(name)
            else:
                surf_vars.append(name)
            continue

        if kind == "atmos":
            if not in_atmos:
                missing.append(name)
            else:
                atmos_vars.append(name)
            continue

        # If kind unspecified, infer by where it exists.
        if in_surf and in_atmos:
            raise ValueError(
                f"Variable `{name}` exists in both files but has no explicit kind in config."
            )
        if in_surf:
            surf_vars.append(name)
        elif in_atmos:
            atmos_vars.append(name)
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            "The following requested variables were not found in input files: "
            + ", ".join(sorted(missing))
        )

    return surf_vars, atmos_vars


def _split_requested_vars_by_config_kind(
    requested: OrderedDict[str, str | None],
) -> tuple[list[str] | None, list[str] | None]:
    """Return surface/atmos variable lists when config kinds are explicit.

    If any variable kind is omitted, return (None, None) so the older
    dataset-inspection path can infer variable locations safely.
    """
    if any(kind is None for kind in requested.values()):
        return None, None
    surf_vars = [name for name, kind in requested.items() if kind == "surf"]
    atmos_vars = [name for name, kind in requested.items() if kind == "atmos"]
    return surf_vars, atmos_vars


def _select_vars(ds: xr.Dataset, var_names: list[str] | None, *, label: str, path: Path) -> xr.Dataset:
    if var_names is None:
        return ds
    missing = [name for name in var_names if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"{label} file {path} is missing requested variable(s): {', '.join(missing)}"
        )
    return ds[var_names]


def _drop_non_spatiotemporal_extras(
    da: xr.DataArray,
    required_dims: list[str],
) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in required_dims:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
            continue
        raise ValueError(
            f"Variable `{da.name}` has unsupported extra dimension `{dim}` with size "
            f"{da.sizes[dim]}."
        )
    missing = [d for d in required_dims if d not in da.dims]
    if missing:
        raise ValueError(f"Variable `{da.name}` is missing required dims: {missing}")
    return da.transpose(*required_dims)


def _load_and_merge_files(
    file_paths: list[Path | str],
    cfg: dict[str, Any],
    *,
    label: str,
    var_names: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> xr.Dataset:
    """Load multiple NetCDF files and merge them along the time dimension."""
    if not file_paths:
        raise ValueError("No file paths provided")

    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))

    datasets = []
    for path in _progress(file_paths, desc=f"Opening {label} files", unit="file"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ds = xr.open_dataset(path)
        ds = _select_vars(ds, var_names, label=label, path=path)
        ds = _normalize_dims(ds, cfg)
        ds = _apply_domain_subset(
            ds,
            cfg,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            verbose=False,
        )
        ds = _apply_time_subset(
            ds,
            time_dim=time_dim,
            start_time=time_start,
            end_time=time_end,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    # Merge along time dimension
    with _elapsed_status(f"Concatenating {label} files"):
        merged = xr.concat(datasets, dim=time_dim)
    with _elapsed_status(f"Sorting {label} by {time_dim}"):
        merged = merged.sortby(time_dim)

    # Remove duplicate timestamps if any
    with _elapsed_status(f"Checking {label} duplicate timestamps"):
        _, unique_indices = np.unique(merged[time_dim].values, return_index=True)
    if len(unique_indices) < len(merged[time_dim]):
        _log(f"Warning: Removed {len(merged[time_dim]) - len(unique_indices)} duplicate timestamps")
        merged = merged.isel({time_dim: sorted(unique_indices)})

    return merged


def _build_merged_dataset(
    cfg: dict[str, Any],
    surface_files: list[Path | str],
    atmos_files: list[Path | str],
    requested: OrderedDict[str, str | None],
    *,
    time_start: str | None = None,
    time_end: str | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> xr.Dataset:
    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))
    lat_dim = str(data_cfg.get("lat_dim", "latitude"))
    lon_dim = str(data_cfg.get("lon_dim", "longitude"))
    level_dim = str(data_cfg.get("level_dim", "level"))

    surf_vars, atmos_vars = _split_requested_vars_by_config_kind(requested)

    surface_ds = _load_and_merge_files(
        surface_files,
        cfg,
        label="surface",
        var_names=surf_vars,
        time_start=time_start,
        time_end=time_end,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    atmos_ds = _load_and_merge_files(
        atmos_files,
        cfg,
        label="atmospheric",
        var_names=atmos_vars,
        time_start=time_start,
        time_end=time_end,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    if surf_vars is None or atmos_vars is None:
        surf_vars, atmos_vars = _split_surface_and_atmos_vars(requested, surface_ds, atmos_ds)
    parts: list[xr.Dataset] = []
    if surf_vars:
        parts.append(surface_ds[surf_vars])
    if atmos_vars:
        parts.append(atmos_ds[atmos_vars])
    if not parts:
        raise ValueError("No variables selected from input datasets.")

    if len(parts) > 1:
        with _elapsed_status("Aligning surface and atmospheric datasets"):
            aligned = xr.align(*parts, join="inner", copy=False)
            parts = list(aligned)

    with _elapsed_status("Merging selected variables"):
        merged = xr.merge(parts, compat="override", combine_attrs="drop_conflicts")
    with _elapsed_status(f"Sorting merged dataset by {time_dim}"):
        merged = merged.sortby(time_dim)

    # Force expected dim ordering and remove singleton extras.
    for var_name, kind in _progress(
        list(requested.items()),
        desc="Normalizing variable dimensions",
        unit="var",
    ):
        if var_name not in merged:
            continue
        if kind == "atmos":
            required = [time_dim, level_dim, lat_dim, lon_dim]
        elif kind == "surf":
            required = [time_dim, lat_dim, lon_dim]
        else:
            # Inferred kind from merged dims.
            if level_dim in merged[var_name].dims:
                required = [time_dim, level_dim, lat_dim, lon_dim]
            else:
                required = [time_dim, lat_dim, lon_dim]
        merged[var_name] = _drop_non_spatiotemporal_extras(merged[var_name], required_dims=required)

    # Keep only essential coords.
    coord_keep = [c for c in [time_dim, level_dim, lat_dim, lon_dim] if c in merged.coords]
    merged = merged.set_coords(coord_keep)

    return merged


def _split_train_val_test(
    ds: xr.Dataset,
    *,
    time_dim: str,
    train_start_time: str,
    train_end_time: str,
    test_start_time: str,
    test_end_time: str,
    create_val: bool,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset]:
    """Split dataset into train, validation (optional), and test sets.

    Returns: (train_ds, val_ds, test_ds) where val_ds may be None.
    """
    n_time = int(ds.sizes[time_dim])
    if n_time < 2:
        raise ValueError(f"Need at least 2 timestamps, found {n_time}.")

    time_vals = np.asarray(ds[time_dim].values)

    train_start = np.datetime64(train_start_time)
    train_end = np.datetime64(train_end_time)
    test_start = np.datetime64(test_start_time)
    test_end = np.datetime64(test_end_time)

    is_train = (time_vals >= train_start) & (time_vals <= train_end)
    is_test = (time_vals >= test_start) & (time_vals <= test_end)

    n_train, n_test = int(is_train.sum()), int(is_test.sum())
    if n_train == 0 or n_test == 0:
        raise ValueError(
            f"Invalid date split: train={n_train}, test={n_test}. "
            f"Check train/test start/end times."
        )

    train_ds = ds.isel({time_dim: np.where(is_train)[0]})
    test_ds = ds.isel({time_dim: np.where(is_test)[0]})

    if create_val:
        # Validation = timestamps between train_end and test_start
        is_val = (time_vals > train_end) & (time_vals < test_start)
        n_val = int(is_val.sum())
        if n_val == 0:
            raise ValueError(
                "No validation timestamps found between train_end_time and test_start_time."
            )
        val_ds = ds.isel({time_dim: np.where(is_val)[0]})
        return train_ds, val_ds, test_ds

    return train_ds, None, test_ds


def _split_chunk_train_val_test(
    ds: xr.Dataset,
    *,
    time_dim: str,
    train_start_time: str,
    train_end_time: str,
    test_start_time: str,
    test_end_time: str,
    create_val: bool,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset]:
    """Split one already-subset chunk without requiring every split to be non-empty."""
    time_vals = np.asarray(ds[time_dim].values)
    train_start = np.datetime64(train_start_time)
    train_end = np.datetime64(train_end_time)
    test_start = np.datetime64(test_start_time)
    test_end = np.datetime64(test_end_time)

    is_train = (time_vals >= train_start) & (time_vals <= train_end)
    is_test = (time_vals >= test_start) & (time_vals <= test_end)

    train_ds = ds.isel({time_dim: np.where(is_train)[0]})
    test_ds = ds.isel({time_dim: np.where(is_test)[0]})

    if create_val:
        is_val = (time_vals > train_end) & (time_vals < test_start)
        val_ds = ds.isel({time_dim: np.where(is_val)[0]})
        return train_ds, val_ds, test_ds

    return train_ds, None, test_ds


def _apply_time_subset(
    ds: xr.Dataset,
    *,
    time_dim: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> xr.Dataset:
    if start_time is None and end_time is None:
        return ds
    if time_dim not in ds.coords and time_dim not in ds.dims:
        return ds

    time_vals = np.asarray(ds[time_dim].values)
    mask = np.ones(time_vals.shape, dtype=bool)
    if start_time is not None:
        mask &= time_vals >= np.datetime64(start_time)
    if end_time is not None:
        mask &= time_vals <= np.datetime64(end_time)
    return ds.isel({time_dim: np.where(mask)[0]})


def _apply_domain_subset(
    ds: xr.Dataset,
    cfg: dict[str, Any],
    *,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    verbose: bool = True,
) -> xr.Dataset:
    """Subset dataset to a regional domain.

    Explicit lat/lon bounds take priority over values in the YAML config.
    If no bounds are provided (neither explicit nor in config), returns ds unchanged.
    """
    data_cfg = cfg.get("data", {})
    lat_dim = str(data_cfg.get("lat_dim", "latitude"))
    lon_dim = str(data_cfg.get("lon_dim", "longitude"))

    # Resolve bounds: explicit args > YAML config > None (no subset)
    lat_min = lat_min if lat_min is not None else data_cfg.get("lat_min")
    lat_max = lat_max if lat_max is not None else data_cfg.get("lat_max")
    lon_min = lon_min if lon_min is not None else data_cfg.get("lon_min")
    lon_max = lon_max if lon_max is not None else data_cfg.get("lon_max")

    if None in {lat_min, lat_max, lon_min, lon_max}:
        return ds

    lat_hi = float(max(lat_min, lat_max))
    lat_lo = float(min(lat_min, lat_max))

    # Convert longitude bounds and data to [0, 360) to match Aurora convention.
    lon_min_mod = float(lon_min) % 360.0
    lon_max_mod = float(lon_max) % 360.0
    if lon_min_mod > lon_max_mod:
        raise ValueError(
            "Regional longitude range crosses the prime meridian/date line after conversion to "
            "[0, 360). Please choose a non-wrapping regional longitude interval."
        )

    lon_values = np.asarray(ds[lon_dim].values, dtype=np.float64) % 360.0
    order = np.argsort(lon_values)
    ds = ds.isel({lon_dim: order}).assign_coords({lon_dim: lon_values[order]})

    # Subset: lat slice works whether latitudes are ascending or descending.
    lat_values = np.asarray(ds[lat_dim].values)
    if lat_values[0] > lat_values[-1]:
        # Descending latitudes (standard ECMWF order)
        ds = ds.sel({lat_dim: slice(lat_hi, lat_lo)})
    else:
        ds = ds.sel({lat_dim: slice(lat_lo, lat_hi)})

    ds = ds.sel({lon_dim: slice(lon_min_mod, lon_max_mod)})

    if ds.sizes.get(lat_dim, 0) == 0 or ds.sizes.get(lon_dim, 0) == 0:
        raise ValueError(
            f"Regional domain subset is empty. Check lat/lon bounds in config: "
            f"lat=[{lat_lo}, {lat_hi}], lon=[{lon_min_mod}, {lon_max_mod}]."
        )

    if verbose:
        _log(f"Subsetted to regional domain: lat=[{lat_lo}, {lat_hi}], lon=[{lon_min_mod}, {lon_max_mod}]")
        _log(f"  Grid: {ds.sizes[lat_dim]} lat x {ds.sizes[lon_dim]} lon")
    return ds


def _write_netcdf(ds: xr.Dataset, path: Path, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": int(compression_level)} for var in ds.data_vars
    }
    with _elapsed_status(f"Writing {path}"):
        ds.to_netcdf(path=path, mode="w", format="NETCDF4", encoding=encoding)


def _encoding_for(ds: xr.Dataset, compression_level: int) -> dict[str, dict[str, Any]]:
    return {
        var: {"zlib": True, "complevel": int(compression_level)} for var in ds.data_vars
    }


def _append_netcdf(ds: xr.Dataset, path: Path, *, time_dim: str, compression_level: int) -> int:
    """Append a time-sorted dataset to a NetCDF file along *time_dim*."""
    n_time = int(ds.sizes.get(time_dim, 0))
    if n_time == 0:
        return 0

    ds = ds.sortby(time_dim)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        ds.to_netcdf(
            path=path,
            mode="w",
            format="NETCDF4",
            encoding=_encoding_for(ds, compression_level),
            unlimited_dims=[time_dim],
        )
        return n_time

    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError(
            "`netCDF4` is required for streaming append after the first chunk. "
            "Install netCDF4 or rerun with --no-streaming-write."
        ) from exc

    with netCDF4.Dataset(path, mode="a") as root:
        if time_dim not in root.dimensions:
            raise ValueError(f"Output file {path} has no `{time_dim}` dimension.")
        start = len(root.dimensions[time_dim])
        end = start + n_time

        time_var = root.variables[time_dim]
        time_values = ds[time_dim].values
        if np.issubdtype(time_values.dtype, np.datetime64):
            units = getattr(time_var, "units", None)
            calendar = getattr(time_var, "calendar", "standard")
            if units is None:
                raise ValueError(f"Output time variable in {path} has no units attribute.")
            py_datetimes = time_values.astype("datetime64[us]").astype(object).tolist()
            time_var[start:end] = netCDF4.date2num(
                py_datetimes,
                units=units,
                calendar=calendar,
            )
        else:
            time_var[start:end] = time_values

        for name, da in ds.data_vars.items():
            if name not in root.variables:
                raise ValueError(f"Output file {path} is missing variable `{name}`.")
            var = root.variables[name]
            if time_dim not in da.dims:
                continue
            axis = da.dims.index(time_dim)
            values = da.values
            if axis != 0:
                values = np.moveaxis(values, axis, 0)
            var[start:end, ...] = values

    return n_time


def _pair_surface_atmos_files(
    surface_files: list[Path | str],
    atmos_files: list[Path | str],
) -> list[tuple[Path, Path]]:
    if len(surface_files) != len(atmos_files):
        raise ValueError(
            f"Expected the same number of surface and atmospheric files, got "
            f"{len(surface_files)} and {len(atmos_files)}."
        )
    return [(Path(s), Path(a)) for s, a in zip(surface_files, atmos_files)]


def _stream_write_outputs(
    *,
    cfg: dict[str, Any],
    surface_files: list[Path | str],
    atmos_files: list[Path | str],
    requested: OrderedDict[str, str | None],
    args: argparse.Namespace,
    time_dim: str,
) -> tuple[int, int, int]:
    train_path = Path(args.train_out)
    val_path = Path(args.val_out) if args.val_out is not None else None
    test_path = Path(args.test_out)

    for path in [train_path, val_path, test_path]:
        if path is not None and path.exists():
            path.unlink()

    total_train = 0
    total_val = 0
    total_test = 0

    pairs = _pair_surface_atmos_files(surface_files, atmos_files)
    iterator = _progress(pairs, desc="Streaming file pairs", unit="pair")
    for surface_path, atmos_path in iterator:
        _log(f"\nProcessing pair: {surface_path.name} + {atmos_path.name}")
        chunk = _build_merged_dataset(
            cfg=cfg,
            surface_files=[surface_path],
            atmos_files=[atmos_path],
            requested=requested,
            time_start=args.train_start_time,
            time_end=args.test_end_time,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
        )

        if int(chunk.sizes.get(time_dim, 0)) == 0:
            _log("  No timestamps in requested range; skipping.")
            continue

        train_ds, val_ds, test_ds = _split_chunk_train_val_test(
            chunk,
            time_dim=time_dim,
            train_start_time=args.train_start_time,
            train_end_time=args.train_end_time,
            test_start_time=args.test_start_time,
            test_end_time=args.test_end_time,
            create_val=(val_path is not None),
        )

        n_train = _append_netcdf(
            train_ds,
            train_path,
            time_dim=time_dim,
            compression_level=args.compression_level,
        )
        n_test = _append_netcdf(
            test_ds,
            test_path,
            time_dim=time_dim,
            compression_level=args.compression_level,
        )
        n_val = 0
        if val_path is not None and val_ds is not None:
            n_val = _append_netcdf(
                val_ds,
                val_path,
                time_dim=time_dim,
                compression_level=args.compression_level,
            )

        total_train += n_train
        total_val += n_val
        total_test += n_test
        _log(f"  Appended times: train={n_train}, val={n_val}, test={n_test}")

    if total_train == 0 or total_test == 0:
        raise ValueError(
            f"Streaming write produced empty required split(s): "
            f"train={total_train}, test={total_test}. Check date ranges."
        )
    if val_path is not None and total_val == 0:
        raise ValueError("Streaming write produced an empty validation split.")

    return total_train, total_val, total_test


def main() -> None:
    args = _parse_args()
    _log(f"Reading config: {args.config}")
    cfg = _read_config(args.config)
    train_out, val_out, test_out, case_data_dir = _resolve_case_data_paths(
        cfg,
        Path(args.config).expanduser().resolve(),
        train_out=args.train_out,
        val_out=args.val_out,
        test_out=args.test_out,
    )
    args.train_out = train_out
    args.val_out = val_out
    args.test_out = test_out
    _log(f"Case data folder: {case_data_dir}")
    _log(f"Train output: {args.train_out}")
    _log(f"Test output: {args.test_out}")
    if args.val_out is not None:
        _log(f"Validation output: {args.val_out}")

    requested = _extract_requested_variables(cfg)
    if not requested:
        raise ValueError("No variables found in data.predictor_variables/target_variables.")

    # Discover files via glob in data_folder
    data_folder = str(args.data_folder)
    _log(f"Discovering input files in: {data_folder}")
    surface_files = sorted(_glob(f"{data_folder}/*lead0-surface-level.nc"))
    atmos_files = sorted(_glob(f"{data_folder}/*lead0-atmospheric.nc"))

    if not surface_files:
        raise FileNotFoundError(f"No *lead0-surface-level.nc files found in {data_folder}")
    if not atmos_files:
        raise FileNotFoundError(f"No *lead0-atmospheric.nc files found in {data_folder}")

    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))

    _log(f"Found {len(surface_files)} surface file(s) and {len(atmos_files)} atmospheric file(s).")
    _log(
        "Subsetting each file before concat: "
        f"time=[{args.train_start_time}, {args.test_end_time}], "
        f"lat=[{args.lat_min}, {args.lat_max}], lon=[{args.lon_min}, {args.lon_max}]"
    )

    if args.streaming_write:
        _log("Streaming write enabled: output files will be created first and extended by time chunk.")
        if args.dry_run:
            _log("Dry run enabled; no files were written.")
            return
        train_count, val_count, test_count = _stream_write_outputs(
            cfg=cfg,
            surface_files=surface_files,
            atmos_files=atmos_files,
            requested=requested,
            args=args,
            time_dim=time_dim,
        )
        print("\n" + "=" * 70)
        print("Selected variables:", ", ".join(requested.keys()))
        print(f"Wrote train times: {train_count} -> {args.train_out}")
        if args.val_out is not None:
            print(f"Wrote validation times: {val_count} -> {args.val_out}")
        print(f"Wrote test times: {test_count} -> {args.test_out}")
        print("=" * 70)
        print("\nDone!", flush=True)
        return

    _log("Building merged dataset...")
    merged = _build_merged_dataset(
        cfg=cfg,
        surface_files=surface_files,
        atmos_files=atmos_files,
        requested=requested,
        time_start=args.train_start_time,
        time_end=args.test_end_time,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
    )

    with _elapsed_status("Applying regional subset"):
        merged = _apply_domain_subset(
            merged, cfg,
            lat_min=args.lat_min, lat_max=args.lat_max,
            lon_min=args.lon_min, lon_max=args.lon_max,
        )

    create_val = args.val_out is not None
    with _elapsed_status("Splitting train/validation/test datasets"):
        train_ds, val_ds, test_ds = _split_train_val_test(
            merged,
            time_dim=time_dim,
            train_start_time=args.train_start_time,
            train_end_time=args.train_end_time,
            test_start_time=args.test_start_time,
            test_end_time=args.test_end_time,
            create_val=create_val,
        )

    print("\n" + "=" * 70)
    print("Selected variables:", ", ".join(requested.keys()))
    print("Merged dataset sizes:", dict(merged.sizes))
    print("-" * 70)
    print("Train dataset sizes:", dict(train_ds.sizes))
    print("Train time range:", str(train_ds[time_dim].values[0]), "->", str(train_ds[time_dim].values[-1]))

    if val_ds is not None:
        print("-" * 70)
        print("Validation dataset sizes:", dict(val_ds.sizes))
        print("Val time range:", str(val_ds[time_dim].values[0]), "->", str(val_ds[time_dim].values[-1]))

    print("-" * 70)
    print("Test dataset sizes:", dict(test_ds.sizes))
    print("Test time range:", str(test_ds[time_dim].values[0]), "->", str(test_ds[time_dim].values[-1]))
    print("=" * 70)

    if args.dry_run:
        print("\nDry run enabled; no files were written.")
        return

    print("\nWriting output files...", flush=True)
    _write_netcdf(train_ds, args.train_out, compression_level=args.compression_level)
    print(f"✓ Wrote train dataset: {args.train_out}", flush=True)

    if val_ds is not None:
        _write_netcdf(val_ds, args.val_out, compression_level=args.compression_level)
        print(f"✓ Wrote validation dataset: {args.val_out}", flush=True)

    _write_netcdf(test_ds, args.test_out, compression_level=args.compression_level)
    print(f"✓ Wrote test dataset: {args.test_out}", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
