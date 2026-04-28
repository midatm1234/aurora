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
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml


# ============================================================================
# CONFIGURATION: Edit these defaults to match your setup
# ============================================================================

CONFIG = {
    # Path to finetune config YAML
    "config": "aurora_finetune_rollout_config.yaml",

    # Input NetCDF files (can be lists of paths)
    "surface_files": [
        "/mnt/data3/cams/2026-02-01_to_2026-02-14-cams-range-lead0-surface-level.nc",
        "/mnt/data3/cams/2026-02-15_to_2026-02-28-cams-range-lead0-surface-level.nc",
        "/mnt/data3/cams/2026-03-01_to_2026-03-15-cams-range-lead0-surface-level.nc",
        "/mnt/data3/cams/2026-03-16_to_2026-03-16-cams-range-lead0-surface-level.nc",
        "/mnt/data3/cams/2026-03-17_to_2026-03-31-cams-range-lead0-surface-level.nc",
    ],
    "atmos_files": [
        "/mnt/data3/cams/2026-02-01_to_2026-02-14-cams-range-lead0-atmospheric.nc",
        "/mnt/data3/cams/2026-02-15_to_2026-02-28-cams-range-lead0-atmospheric.nc",
        "/mnt/data3/cams/2026-03-01_to_2026-03-15-cams-range-lead0-atmospheric.nc",
        "/mnt/data3/cams/2026-03-16_to_2026-03-16-cams-range-lead0-atmospheric.nc",
        "/mnt/data3/cams/2026-03-17_to_2026-03-31-cams-range-lead0-atmospheric.nc",
    ],

    # Output NetCDF files
    "train_out": "/mnt/data3/aurora/data/train.nc",
    "val_out": None,  # Set to path for validation set, or None to skip
    "test_out": "/mnt/data3/aurora/data/test.nc",

    # Date-based split (ISO format: YYYY-MM-DDTHH:MM:SS)
    # Times < val_start_time -> train
    # val_start_time <= times < test_start_time -> validation
    # times >= test_start_time -> test
    "val_start_time": None,  # e.g., "2026-03-14T00:00:00"
    "test_start_time": "2026-03-29T00:00:00",  # Required if using date-based split

    # Fallback: fraction-based split (used only if test_start_time is None)
    "test_fraction": 0.2,
    "val_fraction": 0.1,  # Taken from remaining data after test split

    # NetCDF compression level (0-9)
    "compression_level": 1,

    # Dry run: print info without writing files
    "dry_run": False,
}

# ============================================================================

SURFACE_KINDS = {"surf", "surface"}
ATMOS_KINDS = {"atmos", "atmospheric"}


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
        "--surface-files",
        type=str,
        nargs="+",
        default=None,
        help="Surface NetCDF file paths (space-separated). Overrides CONFIG['surface_files'].",
    )
    parser.add_argument(
        "--atmos-files",
        type=str,
        nargs="+",
        default=None,
        help="Atmospheric NetCDF file paths (space-separated). Overrides CONFIG['atmos_files'].",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=CONFIG["train_out"],
        help=f"Output train.nc path (default: {CONFIG['train_out']}).",
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
        help=f"Output test.nc path (default: {CONFIG['test_out']}).",
    )
    parser.add_argument(
        "--val-start-time",
        type=str,
        default=CONFIG["val_start_time"],
        help="ISO timestamp; times >= this go to validation (if val_out is set).",
    )
    parser.add_argument(
        "--test-start-time",
        type=str,
        default=CONFIG["test_start_time"],
        help="ISO timestamp; times >= this go to test set.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=CONFIG["test_fraction"],
        help=f"Test fraction if not using date-based split (default: {CONFIG['test_fraction']}).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=CONFIG["val_fraction"],
        help=f"Validation fraction if not using date-based split (default: {CONFIG['val_fraction']}).",
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
    return parser.parse_args()


def _read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


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


def _load_and_merge_files(file_paths: list[Path | str], cfg: dict[str, Any]) -> xr.Dataset:
    """Load multiple NetCDF files and merge them along the time dimension."""
    if not file_paths:
        raise ValueError("No file paths provided")

    datasets = []
    for path in file_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ds = xr.open_dataset(path)
        ds = _normalize_dims(ds, cfg)
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    # Merge along time dimension
    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))
    merged = xr.concat(datasets, dim=time_dim)
    merged = merged.sortby(time_dim)

    # Remove duplicate timestamps if any
    _, unique_indices = np.unique(merged[time_dim].values, return_index=True)
    if len(unique_indices) < len(merged[time_dim]):
        print(f"Warning: Removed {len(merged[time_dim]) - len(unique_indices)} duplicate timestamps")
        merged = merged.isel({time_dim: sorted(unique_indices)})

    return merged


def _build_merged_dataset(
    cfg: dict[str, Any],
    surface_files: list[Path | str],
    atmos_files: list[Path | str],
    requested: OrderedDict[str, str | None],
) -> xr.Dataset:
    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))
    lat_dim = str(data_cfg.get("lat_dim", "latitude"))
    lon_dim = str(data_cfg.get("lon_dim", "longitude"))
    level_dim = str(data_cfg.get("level_dim", "level"))

    surface_ds = _load_and_merge_files(surface_files, cfg)
    atmos_ds = _load_and_merge_files(atmos_files, cfg)

    surf_vars, atmos_vars = _split_surface_and_atmos_vars(requested, surface_ds, atmos_ds)
    parts: list[xr.Dataset] = []
    if surf_vars:
        parts.append(surface_ds[surf_vars])
    if atmos_vars:
        parts.append(atmos_ds[atmos_vars])
    if not parts:
        raise ValueError("No variables selected from input datasets.")

    if len(parts) > 1:
        aligned = xr.align(*parts, join="inner", copy=False)
        parts = list(aligned)

    merged = xr.merge(parts, compat="override", combine_attrs="drop_conflicts")
    merged = merged.sortby(time_dim)

    # Force expected dim ordering and remove singleton extras.
    for var_name, kind in requested.items():
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
    val_start_time: str | None,
    test_start_time: str | None,
    test_fraction: float,
    val_fraction: float,
    create_val: bool,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset]:
    """Split dataset into train, validation (optional), and test sets.

    Returns: (train_ds, val_ds, test_ds) where val_ds may be None.
    """
    n_time = int(ds.sizes[time_dim])
    if n_time < 2:
        raise ValueError(f"Need at least 2 timestamps, found {n_time}.")

    time_vals = np.asarray(ds[time_dim].values)

    # Date-based split
    if test_start_time:
        test_time = np.datetime64(test_start_time)
        is_test = time_vals >= test_time

        if create_val and val_start_time:
            val_time = np.datetime64(val_start_time)
            is_val = (time_vals >= val_time) & (time_vals < test_time)
            is_train = time_vals < val_time

            n_train, n_val, n_test = int(is_train.sum()), int(is_val.sum()), int(is_test.sum())
            if n_train == 0 or n_val == 0 or n_test == 0:
                raise ValueError(
                    f"Invalid date split: train={n_train}, val={n_val}, test={n_test}. "
                    f"Check --val-start-time and --test-start-time."
                )

            train_ds = ds.isel({time_dim: np.where(is_train)[0]})
            val_ds = ds.isel({time_dim: np.where(is_val)[0]})
            test_ds = ds.isel({time_dim: np.where(is_test)[0]})
            return train_ds, val_ds, test_ds
        else:
            is_train = ~is_test
            n_train, n_test = int(is_train.sum()), int(is_test.sum())
            if n_train == 0 or n_test == 0:
                raise ValueError(
                    f"Invalid test split: train={n_train}, test={n_test}. "
                    f"Check --test-start-time."
                )

            train_ds = ds.isel({time_dim: np.where(is_train)[0]})
            test_ds = ds.isel({time_dim: np.where(is_test)[0]})
            return train_ds, None, test_ds

    # Fraction-based split
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")

    n_test = max(1, int(round(n_time * test_fraction)))
    n_test = min(n_test, n_time - 1)
    test_start_idx = n_time - n_test

    if create_val:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1) when creating validation set.")

        n_remaining = test_start_idx
        n_val = max(1, int(round(n_remaining * val_fraction)))
        n_val = min(n_val, n_remaining - 1)
        val_start_idx = n_remaining - n_val

        train_ds = ds.isel({time_dim: slice(0, val_start_idx)})
        val_ds = ds.isel({time_dim: slice(val_start_idx, test_start_idx)})
        test_ds = ds.isel({time_dim: slice(test_start_idx, None)})
        return train_ds, val_ds, test_ds
    else:
        train_ds = ds.isel({time_dim: slice(0, test_start_idx)})
        test_ds = ds.isel({time_dim: slice(test_start_idx, None)})
        return train_ds, None, test_ds


def _apply_domain_subset(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    """Subset dataset to the regional domain defined in config, if domain_type is 'regional'."""
    data_cfg = cfg.get("data", {})
    domain_type = str(data_cfg.get("domain_type", "global")).lower()
    if domain_type != "regional":
        return ds

    lat_dim = str(data_cfg.get("lat_dim", "latitude"))
    lon_dim = str(data_cfg.get("lon_dim", "longitude"))

    lat_min = data_cfg.get("lat_min")
    lat_max = data_cfg.get("lat_max")
    lon_min = data_cfg.get("lon_min")
    lon_max = data_cfg.get("lon_max")
    if None in {lat_min, lat_max, lon_min, lon_max}:
        raise ValueError(
            "domain_type is 'regional' but one or more of lat_min, lat_max, lon_min, lon_max "
            "are not set in the config."
        )

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

    print(f"Subsetted to regional domain: lat=[{lat_lo}, {lat_hi}], lon=[{lon_min_mod}, {lon_max_mod}]")
    print(f"  Grid: {ds.sizes[lat_dim]} lat x {ds.sizes[lon_dim]} lon")
    return ds


def _write_netcdf(ds: xr.Dataset, path: Path, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": int(compression_level)} for var in ds.data_vars
    }
    ds.to_netcdf(path=path, mode="w", format="NETCDF4", encoding=encoding)


def main() -> None:
    args = _parse_args()
    cfg = _read_config(args.config)
    requested = _extract_requested_variables(cfg)
    if not requested:
        raise ValueError("No variables found in data.predictor_variables/target_variables.")

    # Get file lists from args or CONFIG
    surface_files = args.surface_files if args.surface_files else CONFIG["surface_files"]
    atmos_files = args.atmos_files if args.atmos_files else CONFIG["atmos_files"]

    if isinstance(surface_files, str):
        surface_files = [surface_files]
    if isinstance(atmos_files, str):
        atmos_files = [atmos_files]

    data_cfg = cfg.get("data", {})
    time_dim = str(data_cfg.get("time_dim", "time"))

    print(f"Loading {len(surface_files)} surface file(s) and {len(atmos_files)} atmospheric file(s)...")
    merged = _build_merged_dataset(
        cfg=cfg,
        surface_files=surface_files,
        atmos_files=atmos_files,
        requested=requested,
    )

    merged = _apply_domain_subset(merged, cfg)

    create_val = args.val_out is not None
    train_ds, val_ds, test_ds = _split_train_val_test(
        merged,
        time_dim=time_dim,
        val_start_time=args.val_start_time,
        test_start_time=args.test_start_time,
        test_fraction=float(args.test_fraction),
        val_fraction=float(args.val_fraction),
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

    print("\nWriting output files...")
    _write_netcdf(train_ds, args.train_out, compression_level=args.compression_level)
    print(f"✓ Wrote train dataset: {args.train_out}")

    if val_ds is not None:
        _write_netcdf(val_ds, args.val_out, compression_level=args.compression_level)
        print(f"✓ Wrote validation dataset: {args.val_out}")

    _write_netcdf(test_ds, args.test_out, compression_level=args.compression_level)
    print(f"✓ Wrote test dataset: {args.test_out}")
    print("\nDone!")


if __name__ == "__main__":
    main()
