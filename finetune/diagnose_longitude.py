#!/usr/bin/env python3
"""Report dateline continuity for gridded NetCDF predictions or residuals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.longitude import dateline_discontinuity_from_edges, longitude_is_periodic


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the last-to-first longitude jump with interior and nearby gradients."
        )
    )
    parser.add_argument("path", type=Path, help="Prediction/target NetCDF file.")
    parser.add_argument(
        "--variable",
        action="append",
        default=[],
        help="Variable to inspect (repeatable; default: every longitude-dependent variable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with xr.open_dataset(args.path) as ds:
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        if lon_name not in ds.coords:
            raise ValueError("Dataset has no longitude coordinate.")
        lon = ds[lon_name].values
        if not longitude_is_periodic(lon):
            raise ValueError("Longitude coordinate is not a complete periodic global grid.")
        if lon.size > 1 and np.isclose(
            lon[0] % 360.0, lon[-1] % 360.0, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(
                "Longitude contains a duplicated cyclic endpoint; canonicalize the file "
                "before diagnosing its stored boundary."
            )

        names = args.variable or [
            name for name, da in ds.data_vars.items() if lon_name in da.dims
        ]
        print("variable,seam_jump,left_grad,right_grad,local_grad,local_ratio")
        for name in names:
            if name not in ds:
                raise KeyError(f"Variable {name!r} is not present in {args.path}.")
            da = ds[name]
            edges = [da.isel({lon_name: index}).values for index in (0, 1, -2, -1)]
            metric = dateline_discontinuity_from_edges(*edges)
            print(
                f"{name},{metric['seam_jump']:.8g},{metric['left_grad']:.8g},"
                f"{metric['right_grad']:.8g},{metric['local_grad']:.8g},"
                f"{metric['local_ratio']:.6g}"
            )


if __name__ == "__main__":
    main()
