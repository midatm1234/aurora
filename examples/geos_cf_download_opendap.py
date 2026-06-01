#!/usr/bin/env python3
"""Download GEOS-CF v1 `assim` variables over OPeNDAP for a user-defined region.

Outputs:
- Daily combined files (default): one NetCDF per day with both 2D+3D variables.
- Optional range files: CAMS-style surface/atmospheric pair over the full date range.

Data source base (default):
    https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/

Collections used:
- met_tavg_1hr_g1440x721_x1     -> t2m, u10m, v10m, slp, zpbl
- xgc_tavg_1hr_g1440x721_x1     -> totcol_no2, tropcol_no2
- met_inst_1hr_g1440x721_p23    -> q, t, u, v
- chm_inst_1hr_g1440x721_p23    -> no2

Time handling:
- `*_inst_*` collections are expected at :00.
- `*_tavg_*` collections are expected at :30 and are shifted by -30 minutes.
- All timestamps are rounded to exact UTC hour boundaries to remove sub-second jitter
  (e.g. 59.999997s vs 00.000003s) before merges.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


DEFAULT_BASE_URL = "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf"
DEFAULT_STREAM = "assim"
DEFAULT_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

SURFACE_COLLECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("met_tavg_1hr_g1440x721_x1", ("t2m", "u10m", "v10m", "slp", "zpbl")),
    ("xgc_tavg_1hr_g1440x721_x1", ("totcol_no2", "tropcol_no2")),
)

ATMOS_COLLECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("met_inst_1hr_g1440x721_p23", ("q", "t", "u", "v")),
    ("chm_inst_1hr_g1440x721_p23", ("no2",)),
)

# Default output names are CAMS-compatible where needed.
CAMS_COMPAT_RENAME = {
    "u10m": "u10",
    "v10m": "v10",
    "slp": "msl",
    "totcol_no2": "tcno2",
    "zpbl": "z",  # CAMS z replacement requested: use zpbl data for z.
}

# ============================================================================
# CONFIGURATION: edit these defaults to avoid long CLI argument lists
# ============================================================================
CONFIG = {
    # Time range (inclusive). Accepts YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.
    "start": "2023-08-01",
    "end": "2024-07-31",
    # Regional bounds
    "lat_min": 31.0,
    "lat_max": 52.0,
    "lon_min": -128.0,
    "lon_max": -100.0,
    # p23 pressure levels (hPa)
    "levels": DEFAULT_LEVELS,
    # Output controls
    "output_dir": "/mnt/data3/geos-cf",
    "file_prefix": "",
    "overwrite": True,
    "compression_level": 1,
    # Data source
    "stream": DEFAULT_STREAM,
    "base_url": DEFAULT_BASE_URL,
    "open_retries": 4,
    "retry_wait_seconds": 8.0,
    "operation_timeout_seconds": 900.0,
    # Naming/output variants
    "keep_geos_names": False,
    "add_zpbl_alias": False,
    "write_original_surface_copy": True,
    # Output structure
    "write_range_files": False,
    "write_daily_combined_files": True,
    "daily_require_24h": True,
    # Cleanup behavior
    "cleanup_interrupted_outputs": True,
    "interrupted_size_threshold_bytes": 1024,
    # Safety
    "dry_run": False,
}

# ============================================================================


def _levels_default_string() -> str:
    levels = CONFIG.get("levels", DEFAULT_LEVELS)
    if isinstance(levels, (list, tuple)):
        return ",".join(str(v) for v in levels)
    return str(levels)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GEOS-CF data over OPeNDAP for a date range and regional bbox, "
            "and write daily combined and/or range NetCDF files."
        )
    )
    parser.add_argument(
        "--start",
        default=CONFIG["start"],
        help="Start time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )
    parser.add_argument(
        "--end",
        default=CONFIG["end"],
        help="End time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )

    parser.add_argument("--lat-min", type=float, default=CONFIG["lat_min"], help="Minimum latitude.")
    parser.add_argument("--lat-max", type=float, default=CONFIG["lat_max"], help="Maximum latitude.")
    parser.add_argument(
        "--lon-min",
        type=float,
        default=CONFIG["lon_min"],
        help="Minimum longitude in degrees east.",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=CONFIG["lon_max"],
        help="Maximum longitude in degrees east.",
    )

    parser.add_argument(
        "--levels",
        type=str,
        default=_levels_default_string(),
        help=(
            "Pressure levels in hPa for p23 collections (comma-separated). "
            f"Default: {_levels_default_string()}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(CONFIG["output_dir"]),
        help="Directory to write NetCDF outputs.",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=str(CONFIG["file_prefix"]),
        help="Optional output prefix. Default: <start>_to_<end>-geos-cf-<stream>",
    )

    parser.add_argument(
        "--stream",
        type=str,
        default=str(CONFIG["stream"]),
        help=f"GEOS-CF stream (default: {CONFIG['stream']}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=str(CONFIG["base_url"]),
        help="OPeNDAP base URL.",
    )

    parser.add_argument(
        "--keep-geos-names",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["keep_geos_names"]),
        help="Keep GEOS variable names instead of CAMS-compatible names.",
    )
    parser.add_argument(
        "--add-zpbl-alias",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["add_zpbl_alias"]),
        help="When using CAMS-compatible names, also keep zpbl as a duplicate variable.",
    )
    parser.add_argument(
        "--write-original-surface-copy",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["write_original_surface_copy"]),
        help=(
            "Also write a second surface file with original GEOS variable names. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--write-range-files",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["write_range_files"]),
        help="Write full-range CAMS-style surface/atmospheric files.",
    )
    parser.add_argument(
        "--write-daily-combined-files",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["write_daily_combined_files"]),
        help="Write one combined 2D+3D file per day.",
    )
    parser.add_argument(
        "--daily-require-24h",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["daily_require_24h"]),
        help="Require exactly 24 hourly timestamps in each daily file.",
    )

    parser.add_argument(
        "--open-retries",
        type=int,
        default=int(CONFIG["open_retries"]),
        help="Retries for opening each OPeNDAP collection.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=float(CONFIG["retry_wait_seconds"]),
        help="Retry sleep seconds.",
    )
    parser.add_argument(
        "--operation-timeout-seconds",
        type=float,
        default=float(CONFIG["operation_timeout_seconds"]),
        help=(
            "Per-operation timeout in seconds for remote fetch/write steps. "
            "Set <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--cleanup-interrupted-outputs",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["cleanup_interrupted_outputs"]),
        help="Delete obviously incomplete remnants (small .nc files, *.part) before download.",
    )
    parser.add_argument(
        "--interrupted-size-threshold-bytes",
        type=int,
        default=int(CONFIG["interrupted_size_threshold_bytes"]),
        help="Size threshold to consider an existing output file as interrupted/incomplete.",
    )

    parser.add_argument(
        "--compression-level",
        type=int,
        default=int(CONFIG["compression_level"]),
        help="NetCDF compression level (0-9).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["overwrite"]),
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG["dry_run"]),
        help="Print planned outputs and variable lists only.",
    )

    return parser.parse_args()


def _parse_datetime(value: str, *, is_end: bool) -> np.datetime64:
    raw = value.strip()
    if "T" not in raw and len(raw) == 10:
        raw = f"{raw}T23:59:59" if is_end else f"{raw}T00:00:00"
    try:
        return np.datetime64(raw, "ns")
    except ValueError as exc:
        raise ValueError(f"Invalid datetime value: {value!r}") from exc


def _parse_levels(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--levels must contain at least one value.")
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError as exc:
            raise ValueError(f"Invalid pressure level {p!r} in --levels.") from exc
    return out


def _hour_round(times: np.ndarray) -> np.ndarray:
    """Round datetime64[ns] array to nearest hour."""
    t_ns = times.astype("datetime64[ns]").astype(np.int64)
    one_hour_ns = int(np.timedelta64(1, "h") / np.timedelta64(1, "ns"))
    rounded = ((t_ns + one_hour_ns // 2) // one_hour_ns) * one_hour_ns
    return rounded.astype("datetime64[ns]")


class OperationTimeoutError(RuntimeError):
    """Raised when a guarded operation exceeds the configured timeout."""


@contextlib.contextmanager
def _timeout_guard(seconds: float, label: str):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum, _frame):
        raise OperationTimeoutError(f"Timed out after {seconds:.1f}s during {label}.")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _ensure_unique_time(ds: xr.Dataset, *, label: str) -> xr.Dataset:
    """Ensure time coordinate is sorted and unique for safe xarray alignment."""
    if "time" not in ds.coords:
        return ds

    time_values = np.asarray(ds["time"].values).astype("datetime64[ns]")
    if time_values.size == 0:
        return ds

    order = np.argsort(time_values.astype(np.int64), kind="stable")
    if not np.array_equal(order, np.arange(time_values.size)):
        ds = ds.isel(time=order)
        time_values = np.asarray(ds["time"].values).astype("datetime64[ns]")

    _, unique_idx = np.unique(time_values, return_index=True)
    if unique_idx.size != time_values.size:
        dropped = int(time_values.size - unique_idx.size)
        print(f"  dropped {dropped} duplicate time step(s) in {label}")
        ds = ds.isel(time=np.sort(unique_idx))

    return ds


def _lon_to_360(values: np.ndarray) -> np.ndarray:
    return np.mod(values.astype(np.float64), 360.0)


def _open_with_retries(url: str, retries: int, retry_wait_seconds: float) -> xr.Dataset:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return xr.open_dataset(url, engine="netcdf4", decode_times=True)
        except OperationTimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            print(f"  open failed ({attempt}/{retries}) for {url}: {exc}")
            time.sleep(retry_wait_seconds)
    raise RuntimeError(f"Failed to open {url} after {retries} attempts: {last_error}")


def _cleanup_interrupted_output_files(
    paths: list[Path],
    *,
    size_threshold_bytes: int,
) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            if path.stat().st_size <= int(size_threshold_bytes):
                path.unlink(missing_ok=True)
                print(f"Removed incomplete file: {path}")
        except OSError as exc:
            print(f"Warning: failed to inspect/remove {path}: {exc}")

    for path in paths:
        part_path = path.with_name(path.name + ".part")
        if part_path.exists():
            try:
                part_path.unlink(missing_ok=True)
                print(f"Removed stale temp file: {part_path}")
            except OSError as exc:
                print(f"Warning: failed to remove temp file {part_path}: {exc}")


def _subset_region_and_time(
    ds: xr.Dataset,
    *,
    start_time: np.datetime64,
    end_time: np.datetime64,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    if not {"time", "lat", "lon"}.issubset(ds.coords):
        missing = {"time", "lat", "lon"} - set(ds.coords)
        raise ValueError(f"Dataset is missing expected coords: {sorted(missing)}")

    time_values = np.asarray(ds["time"].values).astype("datetime64[ns]")
    if time_values.size == 0:
        raise ValueError("Empty time coordinate in source dataset.")

    # Some GEOS-CF collections expose tiny timestamp jitter and non-monotonic
    # ordering via OPeNDAP. Sort first, then apply a boolean mask.
    if np.any(np.diff(time_values.astype(np.int64)) < 0):
        order = np.argsort(time_values)
        ds = ds.isel(time=order)
        time_values = np.asarray(ds["time"].values).astype("datetime64[ns]")

    mask = (time_values >= start_time) & (time_values <= end_time)
    idx = np.where(mask)[0]
    ds = ds.isel(time=idx)

    lat_lo = float(min(lat_min, lat_max))
    lat_hi = float(max(lat_min, lat_max))
    lat_values = np.asarray(ds["lat"].values)
    if lat_values.size == 0:
        raise ValueError("Empty latitude coordinate in source dataset.")
    if lat_values[0] > lat_values[-1]:
        ds = ds.sel(lat=slice(lat_hi, lat_lo))
    else:
        ds = ds.sel(lat=slice(lat_lo, lat_hi))

    lon_vals = _lon_to_360(np.asarray(ds["lon"].values))
    order = np.argsort(lon_vals)
    ds = ds.isel(lon=order).assign_coords(lon=lon_vals[order])

    lon_min_mod = float(lon_min) % 360.0
    lon_max_mod = float(lon_max) % 360.0
    if lon_min_mod <= lon_max_mod:
        ds = ds.sel(lon=slice(lon_min_mod, lon_max_mod))
    else:
        left = ds.sel(lon=slice(lon_min_mod, 360.0))
        right = ds.sel(lon=slice(0.0, lon_max_mod))
        ds = xr.concat([left, right], dim="lon")

    # Remove any duplicate longitudes that may appear from wrap handling.
    lon_now = np.asarray(ds["lon"].values)
    rounded = np.round(lon_now, 8)
    _, unique_idx = np.unique(rounded, return_index=True)
    if len(unique_idx) != lon_now.size:
        ds = ds.isel(lon=np.sort(unique_idx))

    if ds.sizes.get("time", 0) == 0:
        raise ValueError("No timesteps left after subsetting. Expand --start/--end.")
    if ds.sizes.get("lat", 0) == 0 or ds.sizes.get("lon", 0) == 0:
        raise ValueError("No grid cells left after region subsetting. Check lat/lon bounds.")
    return ds


def _fetch_collection(
    *,
    base_url: str,
    stream: str,
    collection: str,
    variables: Iterable[str],
    start_time: np.datetime64,
    end_time: np.datetime64,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    levels: list[float] | None,
    shift_tavg: bool,
    open_retries: int,
    retry_wait_seconds: float,
    operation_timeout_seconds: float,
) -> xr.Dataset:
    url = f"{base_url.rstrip('/')}/{stream}/{collection}"
    print(f"\nOpening: {url}")
    src: xr.Dataset | None = None
    try:
        with _timeout_guard(operation_timeout_seconds, f"fetch collection {collection}"):
            src = _open_with_retries(url=url, retries=open_retries, retry_wait_seconds=retry_wait_seconds)

            missing_vars = [name for name in variables if name not in src.data_vars]
            if missing_vars:
                raise ValueError(f"Collection {collection} missing requested variables: {missing_vars}")

            ds = src[list(variables)]

            # tavg collections are stamped at :30 and then shifted by -30 minutes to
            # align with :00 inst collections. Expand selection window accordingly so
            # requested [start, end] is respected after the shift.
            subset_start = start_time + (np.timedelta64(30, "m") if shift_tavg else np.timedelta64(0, "m"))
            subset_end = end_time + (np.timedelta64(30, "m") if shift_tavg else np.timedelta64(0, "m"))
            # Account for tiny OPeNDAP time jitter (e.g. :29:59.999997 or :00:00.000003).
            subset_start = subset_start - np.timedelta64(2, "m")
            subset_end = subset_end + np.timedelta64(2, "m")

            ds = _subset_region_and_time(
                ds,
                start_time=subset_start,
                end_time=subset_end,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )

            if levels is not None and "lev" in ds.coords:
                available = set(np.asarray(ds["lev"].values, dtype=np.float64).tolist())
                missing_levels = [lev for lev in levels if float(lev) not in available]
                if missing_levels:
                    raise ValueError(f"Collection {collection} missing requested pressure levels: {missing_levels}")
                ds = ds.sel(lev=levels)

            times = np.asarray(ds["time"].values)
            if shift_tavg:
                times = times.astype("datetime64[ns]") - np.timedelta64(30, "m")

            ds = ds.assign_coords(time=_hour_round(times))
            ds = _ensure_unique_time(ds, label=collection)
            return ds
    finally:
        if src is not None:
            with contextlib.suppress(Exception):
                src.close()


def _merge_collections(datasets: list[xr.Dataset]) -> xr.Dataset:
    if not datasets:
        raise ValueError("No datasets were provided for merging.")

    cleaned = [
        _ensure_unique_time(ds, label=f"merge_input_{i}")
        for i, ds in enumerate(datasets, start=1)
    ]
    aligned = xr.align(*cleaned, join="inner", copy=False)
    merged = xr.merge(aligned, compat="override", combine_attrs="drop_conflicts")
    merged = merged.sortby("time")
    merged = _ensure_unique_time(merged, label="merged_collections")

    if merged.sizes.get("time", 0) == 0:
        raise ValueError(
            "Merged time axis is empty after alignment. Check time range and collection availability."
        )

    # Remove duplicate timestamps if present.
    time_values = np.asarray(merged["time"].values)
    _, unique_idx = np.unique(time_values, return_index=True)
    if len(unique_idx) < len(time_values):
        merged = merged.isel(time=np.sort(unique_idx))

    return merged


def _finalize_dims(ds: xr.Dataset) -> xr.Dataset:
    rename_dims = {"lat": "latitude", "lon": "longitude", "lev": "level"}
    usable = {k: v for k, v in rename_dims.items() if k in ds.dims or k in ds.coords}
    if usable:
        ds = ds.rename(usable)

    # Aurora helpers expect descending latitude and increasing [0, 360) longitude.
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude", ascending=False)
    if "longitude" in ds.coords:
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        order = np.argsort(lon)
        ds = ds.isel(longitude=order).assign_coords(longitude=lon[order])

    for var in list(ds.data_vars):
        da = ds[var]
        if "level" in da.dims:
            ds[var] = da.transpose("time", "level", "latitude", "longitude")
        else:
            ds[var] = da.transpose("time", "latitude", "longitude")

    return ds


def _rename_surface_vars(ds: xr.Dataset, keep_geos_names: bool, add_zpbl_alias: bool) -> xr.Dataset:
    if keep_geos_names:
        return ds

    rename = {old: new for old, new in CAMS_COMPAT_RENAME.items() if old in ds.data_vars}
    out = ds.rename(rename)

    if add_zpbl_alias and "z" in out.data_vars and "zpbl" not in out.data_vars:
        out["zpbl"] = out["z"].copy(deep=False)
        out["zpbl"].attrs = dict(out["z"].attrs)
        out["zpbl"].attrs["note"] = "Alias of z (derived from GEOS-CF zpbl)."

    return out


def _write_netcdf(
    ds: xr.Dataset,
    path: Path,
    compression_level: int,
    *,
    operation_timeout_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": int(compression_level)} for var in ds.data_vars
    }
    tmp_path = path.with_name(path.name + ".part")
    tmp_path.unlink(missing_ok=True)
    try:
        with _timeout_guard(operation_timeout_seconds, f"write {path.name}"):
            ds.to_netcdf(path=tmp_path, mode="w", format="NETCDF4", encoding=encoding)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _iter_days(start_time: np.datetime64, end_time: np.datetime64) -> list[np.datetime64]:
    start_day = start_time.astype("datetime64[D]")
    end_day = end_time.astype("datetime64[D]")
    n_days = int((end_day - start_day) / np.timedelta64(1, "D")) + 1
    return [start_day + np.timedelta64(i, "D") for i in range(n_days)]


def _daily_base_name(day: np.datetime64, *, stream: str, file_prefix: str) -> str:
    day_tag = str(day.astype("datetime64[D]"))
    _ = file_prefix  # Daily naming is fixed by convention.
    return f"geos-cf-{stream}-{day_tag}"


def _daily_output_path(
    output_dir: Path,
    day: np.datetime64,
    *,
    stream: str,
    file_prefix: str,
    geos_original: bool,
) -> Path:
    base = _daily_base_name(day, stream=stream, file_prefix=file_prefix)
    suffix = "-geos-original.nc" if geos_original else ".nc"
    return output_dir / f"{base}{suffix}"


def _select_daily_window(
    ds: xr.Dataset,
    day: np.datetime64,
    *,
    require_24h: bool,
    label: str,
) -> xr.Dataset:
    day_h = day.astype("datetime64[h]")
    expected_h = day_h + np.arange(24).astype("timedelta64[h]")
    expected_ns = expected_h.astype("datetime64[ns]")
    day_end_ns = expected_ns[-1] + np.timedelta64(59, "m") + np.timedelta64(59, "s")

    out = ds.sel(time=slice(expected_ns[0], day_end_ns))
    out = _ensure_unique_time(out, label=f"{label}_{str(day.astype('datetime64[D]'))}")

    actual_h = np.asarray(out["time"].values).astype("datetime64[h]")
    actual_h = np.unique(actual_h)
    in_day = np.isin(actual_h, expected_h)
    actual_h = actual_h[in_day]

    if require_24h:
        missing = expected_h[~np.isin(expected_h, actual_h)]
        if missing.size > 0 or actual_h.size != 24:
            missing_preview = ", ".join(str(v) for v in missing[:6]) if missing.size > 0 else "none"
            raise ValueError(
                f"Day {str(day.astype('datetime64[D]'))} does not contain 24 hourly steps. "
                f"Found {actual_h.size}/24. Missing (first up to 6): {missing_preview}."
            )
        out = out.sel(time=expected_ns)
    else:
        keep = np.isin(np.asarray(out["time"].values).astype("datetime64[h]"), expected_h)
        out = out.isel(time=np.where(keep)[0])
        out = _ensure_unique_time(out, label=f"{label}_subset")

    return out


def main() -> None:
    args = _parse_args()

    if not (0 <= args.compression_level <= 9):
        raise ValueError("--compression-level must be between 0 and 9.")

    start_time = _parse_datetime(args.start, is_end=False)
    end_time = _parse_datetime(args.end, is_end=True)
    if start_time > end_time:
        raise ValueError("--start must be <= --end.")

    levels = _parse_levels(args.levels)

    prefix = args.file_prefix.strip()
    days = _iter_days(start_time, end_time)

    if not args.write_range_files and not args.write_daily_combined_files:
        raise ValueError("Enable at least one output mode: --write-range-files or --write-daily-combined-files.")

    if args.write_daily_combined_files and args.daily_require_24h:
        day_start_ns = start_time.astype("datetime64[D]").astype("datetime64[ns]")
        day_end_ns = (
            end_time.astype("datetime64[D]").astype("datetime64[ns]")
            + np.timedelta64(23, "h")
            + np.timedelta64(59, "m")
            + np.timedelta64(59, "s")
        )
        if start_time != day_start_ns or end_time != day_end_ns:
            raise ValueError(
                "daily_require_24h=true requires full-day bounds. Use start at 00:00:00 and end at 23:59:59 "
                "(or pass date-only values like YYYY-MM-DD)."
            )

    range_surface_path: Path | None = None
    range_atmos_path: Path | None = None
    range_surface_original_path: Path | None = None
    if args.write_range_files:
        range_prefix = prefix
        if not range_prefix:
            start_tag = str(start_time.astype("datetime64[D]"))
            end_tag = str(end_time.astype("datetime64[D]"))
            range_prefix = f"{start_tag}_to_{end_tag}-geos-cf-{args.stream}"
        range_surface_path = args.output_dir / f"{range_prefix}-surface-level.nc"
        range_atmos_path = args.output_dir / f"{range_prefix}-atmospheric.nc"
        range_surface_original_path = args.output_dir / f"{range_prefix}-surface-level-geos-original.nc"

    daily_paths: list[Path] = []
    daily_original_paths: list[Path] = []
    if args.write_daily_combined_files:
        daily_paths = [
            _daily_output_path(
                args.output_dir,
                day,
                stream=args.stream,
                file_prefix=prefix,
                geos_original=False,
            )
            for day in days
        ]
        if not args.keep_geos_names and args.write_original_surface_copy:
            daily_original_paths = [
                _daily_output_path(
                    args.output_dir,
                    day,
                    stream=args.stream,
                    file_prefix=prefix,
                    geos_original=True,
                )
                for day in days
            ]

    print("=" * 88)
    print("GEOS-CF OPeNDAP download")
    print("=" * 88)
    print(f"Time range          : {start_time} -> {end_time}")
    print(f"Region (lat)        : {args.lat_min} .. {args.lat_max}")
    print(f"Region (lon, input) : {args.lon_min} .. {args.lon_max}")
    print(f"Region (lon, 0-360) : {args.lon_min % 360.0} .. {args.lon_max % 360.0}")
    print(f"Pressure levels hPa : {levels}")
    print(f"Stream              : {args.stream}")
    print(f"Base URL            : {args.base_url}")
    if args.write_range_files:
        print(f"Range surface output: {range_surface_path}")
        print(f"Range atmos output  : {range_atmos_path}")
        if not args.keep_geos_names and args.write_original_surface_copy:
            print(f"Range surface orig  : {range_surface_original_path}")
    if args.write_daily_combined_files:
        print(f"Daily files         : {len(days)}")
        if daily_paths:
            print(f"Daily first         : {daily_paths[0]}")
            print(f"Daily last          : {daily_paths[-1]}")
        if daily_original_paths:
            print(f"Daily orig first    : {daily_original_paths[0]}")
            print(f"Daily orig last     : {daily_original_paths[-1]}")
    print(f"Keep GEOS names     : {args.keep_geos_names}")
    if not args.keep_geos_names:
        print(f"Add zpbl alias      : {args.add_zpbl_alias}")
    print("=" * 88)

    if args.dry_run:
        surf_vars = [v for _, vs in SURFACE_COLLECTIONS for v in vs]
        atm_vars = [v for _, vs in ATMOS_COLLECTIONS for v in vs]
        print("Dry run: no downloads/write performed.")
        print(f"Surface source vars : {surf_vars}")
        print(f"Atmos source vars   : {atm_vars}")
        if not args.keep_geos_names:
            print(f"Surface output vars : {[CAMS_COMPAT_RENAME.get(v, v) for v in surf_vars]}")
        if args.write_daily_combined_files:
            print(f"Daily combined count: {len(days)}")
        return

    if not args.overwrite:
        candidates: list[Path] = []
        if args.write_range_files:
            candidates.extend([range_surface_path, range_atmos_path])  # type: ignore[arg-type]
            if not args.keep_geos_names and args.write_original_surface_copy:
                candidates.append(range_surface_original_path)  # type: ignore[arg-type]
        if args.write_daily_combined_files:
            candidates.extend(daily_paths)
            candidates.extend(daily_original_paths)
        existing = [p for p in candidates if p.exists()]
        if existing:
            joined = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {joined}. Use --overwrite to replace."
            )
    else:
        if args.cleanup_interrupted_outputs:
            cleanup_targets: list[Path] = []
            if args.write_range_files:
                cleanup_targets.extend([range_surface_path, range_atmos_path])  # type: ignore[arg-type]
                if not args.keep_geos_names and args.write_original_surface_copy:
                    cleanup_targets.append(range_surface_original_path)  # type: ignore[arg-type]
            if args.write_daily_combined_files:
                cleanup_targets.extend(daily_paths)
                cleanup_targets.extend(daily_original_paths)
            _cleanup_interrupted_output_files(
                cleanup_targets,
                size_threshold_bytes=args.interrupted_size_threshold_bytes,
            )

    range_surface_chunks: list[xr.Dataset] = []
    range_surface_original_chunks: list[xr.Dataset] = []
    range_atmos_chunks: list[xr.Dataset] = []
    daily_written = 0
    last_surface_ds: xr.Dataset | None = None
    last_atmos_ds: xr.Dataset | None = None

    for i, day in enumerate(days, start=1):
        day_ns = day.astype("datetime64[ns]")
        day_start = day_ns
        day_end = day_ns + np.timedelta64(23, "h") + np.timedelta64(59, "m") + np.timedelta64(59, "s")
        day_tag = str(day.astype("datetime64[D]"))
        print(f"\n[{i}/{len(days)}] Fetching day {day_tag} ...")
        try:
            surface_parts: list[xr.Dataset] = []
            for collection, variables in SURFACE_COLLECTIONS:
                part = _fetch_collection(
                    base_url=args.base_url,
                    stream=args.stream,
                    collection=collection,
                    variables=variables,
                    start_time=day_start,
                    end_time=day_end,
                    lat_min=args.lat_min,
                    lat_max=args.lat_max,
                    lon_min=args.lon_min,
                    lon_max=args.lon_max,
                    levels=None,
                    shift_tavg=True,
                    open_retries=args.open_retries,
                    retry_wait_seconds=args.retry_wait_seconds,
                    operation_timeout_seconds=args.operation_timeout_seconds,
                )
                surface_parts.append(part)

            atmos_parts: list[xr.Dataset] = []
            for collection, variables in ATMOS_COLLECTIONS:
                part = _fetch_collection(
                    base_url=args.base_url,
                    stream=args.stream,
                    collection=collection,
                    variables=variables,
                    start_time=day_start,
                    end_time=day_end,
                    lat_min=args.lat_min,
                    lat_max=args.lat_max,
                    lon_min=args.lon_min,
                    lon_max=args.lon_max,
                    levels=levels,
                    shift_tavg=False,
                    open_retries=args.open_retries,
                    retry_wait_seconds=args.retry_wait_seconds,
                    operation_timeout_seconds=args.operation_timeout_seconds,
                )
                atmos_parts.append(part)

            surface_ds_day = _merge_collections(surface_parts)
            atmos_ds_day = _merge_collections(atmos_parts)

            # Ensure surface and atmos share the same timestamps after collection alignment.
            surface_ds_day = _ensure_unique_time(surface_ds_day, label=f"surface_merged_{day_tag}")
            atmos_ds_day = _ensure_unique_time(atmos_ds_day, label=f"atmos_merged_{day_tag}")
            surface_ds_day, atmos_ds_day = xr.align(surface_ds_day, atmos_ds_day, join="inner", copy=False)
            if surface_ds_day.sizes.get("time", 0) == 0:
                raise ValueError(
                    f"No common timestamps remain between surface and atmospheric datasets after alignment for {day_tag}."
                )

            surface_ds_day = _finalize_dims(surface_ds_day)
            atmos_ds_day = _finalize_dims(atmos_ds_day)
            surface_original_ds_day = surface_ds_day
            surface_ds_day = _rename_surface_vars(
                surface_ds_day,
                keep_geos_names=args.keep_geos_names,
                add_zpbl_alias=args.add_zpbl_alias,
            )

            if args.write_daily_combined_files:
                combined_ds_day = xr.merge(
                    [surface_ds_day, atmos_ds_day],
                    compat="override",
                    combine_attrs="drop_conflicts",
                )
                combined_ds_day = _ensure_unique_time(combined_ds_day.sortby("time"), label=f"daily_combined_{day_tag}")
                daily_ds = _select_daily_window(
                    combined_ds_day,
                    day,
                    require_24h=args.daily_require_24h,
                    label="daily_combined",
                )
                _write_netcdf(
                    daily_ds,
                    daily_paths[i - 1],
                    compression_level=args.compression_level,
                    operation_timeout_seconds=args.operation_timeout_seconds,
                )

                if not args.keep_geos_names and args.write_original_surface_copy:
                    combined_original_ds_day = xr.merge(
                        [surface_original_ds_day, atmos_ds_day],
                        compat="override",
                        combine_attrs="drop_conflicts",
                    )
                    combined_original_ds_day = _ensure_unique_time(
                        combined_original_ds_day.sortby("time"),
                        label=f"daily_combined_original_{day_tag}",
                    )
                    daily_original_ds = _select_daily_window(
                        combined_original_ds_day,
                        day,
                        require_24h=args.daily_require_24h,
                        label="daily_combined_original",
                    )
                    _write_netcdf(
                        daily_original_ds,
                        daily_original_paths[i - 1],
                        compression_level=args.compression_level,
                        operation_timeout_seconds=args.operation_timeout_seconds,
                    )

                print(f"[{i}/{len(days)}] Wrote daily file(s) for {day_tag}")
                daily_written += 1

            if args.write_range_files:
                surface_day_for_range = _select_daily_window(
                    surface_ds_day,
                    day,
                    require_24h=args.daily_require_24h,
                    label="range_surface",
                )
                atmos_day_for_range = _select_daily_window(
                    atmos_ds_day,
                    day,
                    require_24h=args.daily_require_24h,
                    label="range_atmos",
                )
                range_surface_chunks.append(surface_day_for_range)
                range_atmos_chunks.append(atmos_day_for_range)

                if not args.keep_geos_names and args.write_original_surface_copy:
                    surface_original_day_for_range = _select_daily_window(
                        surface_original_ds_day,
                        day,
                        require_24h=args.daily_require_24h,
                        label="range_surface_original",
                    )
                    range_surface_original_chunks.append(surface_original_day_for_range)

            last_surface_ds = surface_ds_day
            last_atmos_ds = atmos_ds_day
        except Exception as exc:
            if args.write_daily_combined_files:
                with contextlib.suppress(Exception):
                    daily_paths[i - 1].unlink(missing_ok=True)
                if not args.keep_geos_names and args.write_original_surface_copy:
                    with contextlib.suppress(Exception):
                        daily_original_paths[i - 1].unlink(missing_ok=True)
            print(f"ERROR: failed on day {day_tag}: {exc}")
            raise SystemExit(1) from exc

    if args.write_range_files:
        if not range_surface_chunks or not range_atmos_chunks:
            raise ValueError("Range output requested but no daily slices were available.")

        range_surface_ds = xr.concat(range_surface_chunks, dim="time").sortby("time")
        range_atmos_ds = xr.concat(range_atmos_chunks, dim="time").sortby("time")
        range_surface_ds = _ensure_unique_time(range_surface_ds, label="range_surface_final")
        range_atmos_ds = _ensure_unique_time(range_atmos_ds, label="range_atmos_final")

        if not args.keep_geos_names and args.write_original_surface_copy:
            if not range_surface_original_chunks:
                raise ValueError("Range original surface output requested but no daily slices were available.")
            range_surface_original_ds = xr.concat(range_surface_original_chunks, dim="time").sortby("time")
            range_surface_original_ds = _ensure_unique_time(
                range_surface_original_ds,
                label="range_surface_original_final",
            )
            _write_netcdf(
                range_surface_original_ds,
                range_surface_original_path,  # type: ignore[arg-type]
                compression_level=args.compression_level,
                operation_timeout_seconds=args.operation_timeout_seconds,
            )

        _write_netcdf(
            range_surface_ds,
            range_surface_path,  # type: ignore[arg-type]
            compression_level=args.compression_level,
            operation_timeout_seconds=args.operation_timeout_seconds,
        )
        _write_netcdf(
            range_atmos_ds,
            range_atmos_path,  # type: ignore[arg-type]
            compression_level=args.compression_level,
            operation_timeout_seconds=args.operation_timeout_seconds,
        )

    print("\nDone.")
    if args.write_range_files:
        print(f"Wrote: {range_surface_path}")
        print(f"Wrote: {range_atmos_path}")
        if not args.keep_geos_names and args.write_original_surface_copy:
            print(f"Wrote: {range_surface_original_path}")
    if args.write_daily_combined_files:
        print(f"Wrote daily combined files: {daily_written}")
        if daily_original_paths:
            print(f"Wrote daily original files: {daily_written}")
    if last_surface_ds is not None and last_atmos_ds is not None:
        print(f"Last day surface dims: {dict(last_surface_ds.sizes)}")
        print(f"Last day atmos dims  : {dict(last_atmos_ds.sizes)}")
        print("Surface vars:", sorted(last_surface_ds.data_vars))
        print("Atmos vars  :", sorted(last_atmos_ds.data_vars))


if __name__ == "__main__":
    main()
