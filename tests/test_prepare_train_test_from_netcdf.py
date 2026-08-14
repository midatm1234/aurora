"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Fail-fast alignment tests for CAMS training-data preparation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from finetune.prepare_train_test_from_netcdf import (
    _align_surface_and_atmospheric_parts,
)


def _part(name: str, times, *, latitude=(52.0, 51.6), longitude=(232.0, 232.4)):
    values = np.zeros((len(times), len(latitude), len(longitude)), dtype=np.float32)
    return xr.Dataset(
        {name: (("time", "latitude", "longitude"), values)},
        coords={
            "time": np.asarray(times, dtype="datetime64[h]"),
            "latitude": np.asarray(latitude),
            "longitude": np.asarray(longitude),
        },
    )


def test_surface_and_atmospheric_parts_require_exact_valid_times() -> None:
    surface = _part("tcno2", ["2024-01-01T00", "2024-01-01T12"])
    atmospheric = _part("no2", ["2024-01-01T00", "2024-01-02T00"])

    with pytest.raises(ValueError, match="exactly matching 'time' coordinates"):
        _align_surface_and_atmospheric_parts([surface, atmospheric], {"data": {}})


def test_surface_and_atmospheric_parts_do_not_inner_join_missing_time() -> None:
    surface = _part("tcno2", ["2024-01-01T00", "2024-01-01T12"])
    atmospheric = _part("no2", ["2024-01-01T00"])

    with pytest.raises(ValueError, match="exactly matching 'time' coordinates"):
        _align_surface_and_atmospheric_parts([surface, atmospheric], {"data": {}})


def test_equivalent_spatial_roundoff_is_snapped_without_changing_time() -> None:
    times = ["2024-01-01T00", "2024-01-01T12"]
    surface = _part("tcno2", times)
    atmospheric = _part(
        "no2",
        times,
        latitude=(52.0 + 1.0e-8, 51.6 - 1.0e-8),
        longitude=(232.0 + 1.0e-8, 232.4 - 1.0e-8),
    )

    aligned = _align_surface_and_atmospheric_parts(
        [surface, atmospheric], {"data": {}}
    )

    np.testing.assert_array_equal(aligned[0].time, aligned[1].time)
    np.testing.assert_array_equal(aligned[0].latitude, aligned[1].latitude)
    np.testing.assert_array_equal(aligned[0].longitude, aligned[1].longitude)


def test_reversed_latitude_order_is_rejected() -> None:
    times = ["2024-01-01T00"]
    surface = _part("tcno2", times)
    atmospheric = _part("no2", times, latitude=(51.6, 52.0))

    with pytest.raises(ValueError, match="latitude grid differs"):
        _align_surface_and_atmospheric_parts([surface, atmospheric], {"data": {}})
