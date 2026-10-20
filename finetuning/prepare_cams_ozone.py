"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Download a few days of CAMS global analyses (00/12 UTC, `leadtime_hour=0`) plus the static
variables needed by `AuroraAirPollution`, for use with `finetuning/finetune_ozone.py`.

Requires an Atmosphere Data Store (ADS) account that has accepted the ADS terms of use and the
CAMS licence. The API key is read from a `.cdsapirc`-style file; the ADS URL is always used, so a
file pointing at the CDS works too.

Example:

    python finetuning/prepare_cams_ozone.py --start 2022-06-01 --end 2022-06-08 \\
        --out-dir ~/cams_ozone --rc-file ~/.cdsapirc
"""

import argparse
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
DATASET = "cams-global-atmospheric-composition-forecasts"
LEVELS = ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
VARIABLES = [
    # Meteorological surface-level variables:
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    # Pollution surface-level variables:
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
    # Meteorological atmospheric variables:
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "specific_humidity",
    # Pollution atmospheric variables:
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
]


def read_key(rc_file: Path) -> str:
    match = re.search(r"^key:\s*(\S+)", rc_file.expanduser().read_text(), re.M)
    if not match:
        raise ValueError(f"No `key:` line found in {rc_file}.")
    return match.group(1)


def download_day(day: date, out_dir: Path, key: str) -> None:
    import cdsapi

    sfc = out_dir / f"{day}-cams-surface-level.nc"
    plev = out_dir / f"{day}-cams-atmospheric.nc"
    if sfc.exists() and plev.exists():
        print(f"{day}: already present")
        return
    zip_path = out_dir / f"{day}-cams.nc.zip"
    if not zip_path.exists():
        client = cdsapi.Client(url=ADS_URL, key=key, quiet=True)
        tmp = zip_path.with_suffix(".part")
        client.retrieve(
            DATASET,
            {
                "type": "forecast",
                "leadtime_hour": "0",
                "variable": VARIABLES,
                "pressure_level": LEVELS,
                "date": str(day),
                "time": ["00:00", "12:00"],
                "data_format": "netcdf_zip",
            },
            str(tmp),
        )
        tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for member, target in (("data_sfc.nc", sfc), ("data_plev.nc", plev)):
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()
    print(f"{day}: done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 6, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2022, 6, 8))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rc-file", type=Path, default=Path("~/.cdsapirc"))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ADS requests.")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.4-air-pollution-static.pickle",
        local_dir=out_dir,
    )
    print("Static variables downloaded.")

    key = read_key(args.rc_file)
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(download_day, d, out_dir, key) for d in days]:
            future.result()


if __name__ == "__main__":
    main()
