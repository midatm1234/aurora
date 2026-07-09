"""
Sequential CAMS downloader in half-month chunks.
Takes a start_date and end_date and generates date ranges automatically.
Skips ranges whose output files already exist and pass a basic completeness check.
"""

import argparse
import calendar
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

OUTPUT_DIR = Path("/data/cams")
TIMES_UTC = ["00:00", "12:00"]
OVERWRITE = False
REMOVE_ZIP_AFTER_EXTRACTION = True

START_DATE = "2014-07-01"
END_DATE = "2023-06-30"

CAMS_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "specific_humidity",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
]

CAMS_PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000",
]


def _generate_half_month_ranges(start: date, end: date) -> list[tuple[str, str]]:
    """Split [start, end] into chunks of at most half a month (1st–15th / 16th–last day)."""
    ranges = []
    cursor = start
    while cursor <= end:
        year, month = cursor.year, cursor.month
        last_day = calendar.monthrange(year, month)[1]
        if cursor.day <= 15:
            chunk_end = min(date(year, month, 15), end)
        else:
            chunk_end = min(date(year, month, last_day), end)
        ranges.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + timedelta(days=1)
    return ranges


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _expected_time_bounds(start: str, end: str) -> tuple[np.datetime64, np.datetime64, int]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    times = [time.fromisoformat(t) for t in TIMES_UTC]
    first = datetime.combine(start_date, min(times))
    last = datetime.combine(end_date, max(times))
    count = ((end_date - start_date).days + 1) * len(times)
    return np.datetime64(first), np.datetime64(last), count


def _time_values(ds: xr.Dataset) -> np.ndarray:
    for coord_name in ("forecast_reference_time", "time", "valid_time"):
        if coord_name in ds.coords or coord_name in ds.variables:
            values = np.asarray(ds[coord_name].values).reshape(-1)
            if values.size:
                return values
    raise ValueError("no usable time coordinate found")


def _is_complete_netcdf(path: Path, start: str, end: str, required_vars: set[str]) -> tuple[bool, str]:
    if not _exists_nonempty(path):
        return False, "missing or empty"

    expected_first, expected_last, expected_count = _expected_time_bounds(start, end)
    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            missing_vars = sorted(required_vars - set(ds.data_vars))
            if missing_vars:
                return False, f"missing variables: {', '.join(missing_vars[:5])}"

            values = _time_values(ds).astype("datetime64[ns]")
            unique_values = np.unique(values)
            actual_first = unique_values[0]
            actual_last = unique_values[-1]
            if unique_values.size != expected_count:
                return False, f"expected {expected_count} times, found {unique_values.size}"
            if actual_first != expected_first.astype("datetime64[ns]"):
                return False, f"first time is {actual_first}, expected {expected_first}"
            if actual_last != expected_last.astype("datetime64[ns]"):
                return False, f"last time is {actual_last}, expected {expected_last}"
    except Exception as exc:
        return False, f"cannot read NetCDF: {exc}"

    return True, "complete"


def _outputs_are_complete(surface_path: Path, atmos_path: Path, start: str, end: str) -> tuple[bool, str]:
    surface_vars = {"u10", "v10", "t2m", "msl", "pm1", "pm2p5", "pm10"}
    atmos_vars = {"t", "z", "q", "co", "no2", "no", "go3", "so2"}
    surface_ok, surface_reason = _is_complete_netcdf(surface_path, start, end, surface_vars)
    atmos_ok, atmos_reason = _is_complete_netcdf(atmos_path, start, end, atmos_vars)
    if surface_ok and atmos_ok:
        return True, "surface and atmospheric files are complete"
    return False, f"surface: {surface_reason}; atmos: {atmos_reason}"


def download_range(start: str, end: str) -> bool:
    """Returns True on success, False on failure."""
    date_range = f"{start}/{end}"
    base_name = f"{start}_to_{end}-cams-range-lead0"
    zip_path = OUTPUT_DIR / f"{base_name}.nc.zip"
    surface_path = OUTPUT_DIR / f"{base_name}-surface-level.nc"
    atmos_path = OUTPUT_DIR / f"{base_name}-atmospheric.nc"

    print(f"\n{'='*60}")
    print(f"Processing: {date_range}")
    print(f"  Surface : {surface_path.name}")
    print(f"  Atmos   : {atmos_path.name}")

    outputs_complete, completeness_reason = _outputs_are_complete(surface_path, atmos_path, start, end)
    if outputs_complete and not OVERWRITE:
        print(f"  -> Skipping: {completeness_reason}.")
        return True
    if not outputs_complete:
        print(f"  -> Existing output incomplete or absent: {completeness_reason}")

    request = {
        "type": "forecast",
        "leadtime_hour": "0",
        "variable": CAMS_VARIABLES,
        "pressure_level": CAMS_PRESSURE_LEVELS,
        "date": date_range,
        "time": TIMES_UTC,
        "format": "netcdf_zip",
    }

    try:
        if _exists_nonempty(zip_path) and not OVERWRITE:
            print(f"  -> Zip already exists, skipping download: {zip_path.name}")
        else:
            import cdsapi

            client = cdsapi.Client()
            client.retrieve(
                "cams-global-atmospheric-composition-forecasts",
                request,
                str(zip_path),
            )
            print(f"  -> Downloaded: {zip_path.name}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with open(surface_path, "wb") as f:
                f.write(zf.read("data_sfc.nc"))
            with open(atmos_path, "wb") as f:
                f.write(zf.read("data_plev.nc"))
        print(f"  -> Extracted: {surface_path.name}")
        print(f"  -> Extracted: {atmos_path.name}")

        outputs_complete, completeness_reason = _outputs_are_complete(surface_path, atmos_path, start, end)
        if not outputs_complete:
            raise ValueError(f"extracted files failed completeness check: {completeness_reason}")

        if REMOVE_ZIP_AFTER_EXTRACTION and zip_path.exists():
            zip_path.unlink()
            print(f"  -> Removed zip: {zip_path.name}")
        return True

    except Exception as e:
        print(f"  -> FAILED: {e}")
        # Clean up partial zip to avoid issues on retry
        outputs_complete, _ = _outputs_are_complete(surface_path, atmos_path, start, end)
        if zip_path.exists() and not outputs_complete:
            zip_path.unlink(missing_ok=True)
            print(f"  -> Cleaned up partial zip: {zip_path.name}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CAMS data in half-month chunks.")
    parser.add_argument("--start-date", default=START_DATE, help=f"Start date YYYY-MM-DD (default: {START_DATE})")
    parser.add_argument("--end-date", default=END_DATE, help=f"End date YYYY-MM-DD (default: {END_DATE})")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--overwrite", action="store_true", help="Download and extract even when complete files exist.")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OVERWRITE = args.overwrite
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")

    date_ranges = _generate_half_month_ranges(start, end)

    total = len(date_ranges)
    failed = []
    for i, (rng_start, rng_end) in enumerate(date_ranges, 1):
        print(f"\n[{i}/{total}] Range: {rng_start} to {rng_end}")
        success = download_range(rng_start, rng_end)
        if not success:
            failed.append((rng_start, rng_end))

    print("\nAll downloads attempted.")
    if failed:
        print(f"\nFAILED ranges ({len(failed)}):")
        for rng_start, rng_end in failed:
            print(f"  {rng_start} to {rng_end}")
    else:
        print("All ranges downloaded successfully.")
