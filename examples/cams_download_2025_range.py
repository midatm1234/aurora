"""
Sequential CAMS downloader in half-month chunks.
Takes a start_date and end_date and generates date ranges automatically.
Skips ranges whose output files already exist.
"""

import argparse
import calendar
import zipfile
from datetime import date, timedelta
from pathlib import Path

import cdsapi

OUTPUT_DIR = Path("/mnt/data3/cams")
TIMES_UTC = ["00:00", "12:00"]
OVERWRITE = False
REMOVE_ZIP_AFTER_EXTRACTION = True

START_DATE = "2023-07-01"
END_DATE = "2024-09-30"

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

    if _exists_nonempty(surface_path) and _exists_nonempty(atmos_path) and not OVERWRITE:
        print("  -> Skipping: output files already exist.")
        return True

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

        if REMOVE_ZIP_AFTER_EXTRACTION and zip_path.exists() and _exists_nonempty(surface_path) and _exists_nonempty(atmos_path):
            zip_path.unlink()
            print(f"  -> Removed zip: {zip_path.name}")
        return True

    except Exception as e:
        print(f"  -> FAILED: {e}")
        # Clean up partial zip to avoid issues on retry
        if zip_path.exists() and not (_exists_nonempty(surface_path) and _exists_nonempty(atmos_path)):
            zip_path.unlink(missing_ok=True)
            print(f"  -> Cleaned up partial zip: {zip_path.name}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CAMS data in half-month chunks.")
    parser.add_argument("--start-date", default=START_DATE, help=f"Start date YYYY-MM-DD (default: {START_DATE})")
    parser.add_argument("--end-date", default=END_DATE, help=f"End date YYYY-MM-DD (default: {END_DATE})")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
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
