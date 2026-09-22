"""Synthetic CAMS contract tests; no ADS credentials or network access."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aurora_workflow.data import (
    LEVELS, VARIABLES, audit_history, canonicalize_cams, fit_training_statistics,
    plan_cams, prepare, resolve_data_spec, safe_extract, split_cases,
)


def cams_fixture():
    cycles = np.array(["2024-07-01T00", "2024-07-01T12", "2024-07-02T00"], dtype="datetime64[ns]")
    leads = np.array([0, 12], dtype="timedelta64[h]")
    coords = {"forecast_reference_time": cycles, "step": leads, "latitude": [40., 39.6, 39.2],
              "longitude": [-120., -119.6, -119.2], "pressure_level": LEVELS}
    variables = {}
    for _, name, _, units, kind in VARIABLES:
        dims = ("forecast_reference_time", "step") + (("pressure_level",) if kind == "atmos" else ()) + ("latitude", "longitude")
        shape = tuple(len(coords[v]) for v in dims)
        variables[name] = (dims, np.ones(shape, dtype=np.float32), {"units": units})
    ds = xr.Dataset(variables, coords=coords)
    ds = ds.assign_coords(valid_time=(("forecast_reference_time", "step"), cycles[:, None] + leads[None, :]))
    ds.attrs["model_cycle"] = "synthetic-not-an-operational-model-cycle"
    return ds


def test_plan_includes_previous_history_and_current_ads_field():
    plan = plan_cams({"start_date": "2024-07-01", "end_date": "2024-07-01"})
    assert plan["history_start"] == "2024-06-30T12:00:00"
    assert plan["request_count"] == 3
    assert plan["cycles"] == ["2024-07-01T00:00:00", "2024-07-01T12:00:00"]
    for item in plan["requests"]:
        assert item["request"]["data_format"] == "netcdf_zip"
        assert "format" not in item["request"]
        assert len(item["request"]["pressure_level"]) == 13
    assert plan["estimated_uncompressed_bytes"] > 0


@pytest.mark.parametrize("extra", [
    {"pressure_levels": [1000, 925, 850]}, {"reference_kind": "reanalysis"},
    {"lead_hours": [12, 12]}, {"area": [52, -128, 31, -100]},
    {"variables": ["nitrogen_dioxide"]}, {"chunk_days": 0},
])
def test_plan_rejects_incomplete_or_different_product(extra):
    with pytest.raises(ValueError):
        plan_cams({"start_date": "2024-07-01", "end_date": "2024-07-01", **extra})


def test_unavailable_old_date_and_bounds():
    with pytest.raises(ValueError, match="2015"):
        plan_cams({"start_date": "2014-07-01", "end_date": "2014-07-01"})
    with pytest.raises(ValueError, match="request"):
        plan_cams({"start_date": "2024-07-01", "end_date": "2024-07-01"}, {"max_requests": 1})


def test_sparse_real_planning_and_versioned_validation_dates():
    spec = {"cycles": ["2024-07-01T12", "2024-07-05T12", "2024-07-09T12"],
            "target_variables": ["no2", "tcno2"], "target_levels": [1000, 925, 850],
            "domain": {"north": 52, "west": -128, "south": 31, "east": -100}}
    plan = plan_cams(spec)
    assert plan["request_count"] == 9
    assert plan["estimated_uncompressed_bytes"] < 2_000_000_000
    refs = [v for v in plan["requests"] if v["purpose"] == "reference"]
    assert len(refs[0]["request"]["variable"]) == 2
    assert refs[0]["request"]["area"] == [52, -128, 31, -100]
    resolved = resolve_data_spec({"historical_train": ["2023-07-01", "2024-06-30T12"],
                                 "historical_test": ["2024-07-01", "2024-09-30T12"],
                                 "validation": {"method": "train_tail", "fraction": .1}}, "reproduce")
    assert set(resolved["splits"]) == {"train", "val", "test"}
    assert "historical exact dates unavailable" in resolved["validation_derivation"]


def test_preserves_different_forecasts_at_same_valid_time():
    ds = canonicalize_cams(cams_fixture())
    assert ds.sizes["forecast_reference_time"] == 3
    assert ds.sizes["lead_time"] == 2
    assert ds.valid_time.values[0, 1] == ds.valid_time.values[1, 0]
    assert np.all(np.diff(ds.longitude) > 0)
    assert list(ds.longitude.values) == pytest.approx([240, 240.4, 240.8])


def test_units_provenance_and_missing_backbone_rejected():
    ds = cams_fixture()
    ds["no2"].attrs["units"] = "ppb"
    with pytest.raises(ValueError, match="units"):
        canonicalize_cams(ds)
    with pytest.raises(ValueError, match="backbone"):
        canonicalize_cams(cams_fixture().drop_vars("co"))
    ds = cams_fixture()
    ds["valid_time"] = ds["valid_time"] + np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="valid_time"):
        canonicalize_cams(ds)


def test_split_windows_purge_overlapping_forecast_support():
    splits = {"train": ["2024-01-01", "2024-01-10T12"], "test": ["2024-01-11", "2024-01-20"]}
    kept, excluded = split_cases(["2024-01-08T00", "2024-01-11T00", "2024-01-12T00"], splits)
    assert len(excluded) == 2
    assert kept[0]["split"] == "test"
    with pytest.raises(ValueError, match="Overlapping"):
        split_cases([], {"train": ["2024-01-01", "2024-01-12"], "test": ["2024-01-11", "2024-01-20"]})


def test_future_inference_inputs_and_training_only_statistics():
    audit_history("2024-07-01T12", ["2024-07-01T00", "2024-07-01T12"])
    with pytest.raises(ValueError, match="future"):
        audit_history("2024-07-01T12", ["2024-07-01T12", "2024-07-02T00"])
    stats = fit_training_statistics(np.array([[1., 3.], [3., 5.], [900., 900.]]), ["train", "train", "test"])
    assert stats["mean"] == [2., 4.]


@pytest.mark.parametrize("name", ["../evil.nc", "/tmp/evil.nc", "data.py", "folder/data.nc"])
def test_archive_traversal_and_non_netcdf_rejected(tmp_path, name):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, b"fake fixture")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "out", 1000)


def test_archive_size_bound(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data.nc", b"123456")
    with pytest.raises(ValueError, match="bounds"):
        safe_extract(archive, tmp_path / "out", 5)


def test_preparation_wrapper_preserves_reference_and_audits_history(tmp_path):
    raw = tmp_path / "raw.nc"
    cams_fixture().to_netcdf(raw)
    result = prepare({"raw_paths": [str(raw)], "cycles": ["2024-07-01T12"], "lead_hours": [12],
                      "splits": {"test": ["2024-07-01T00", "2024-07-03T00"]}}, tmp_path / "out")
    assert result["cases"][0]["history_times"] == ["2024-07-01T00:00:00", "2024-07-01T12:00:00"]
    with xr.open_dataset(result["reference_path"]) as ds:
        assert ds.sizes["forecast_reference_time"] == 1
        assert ds.sizes["lead_time"] == 1
    with xr.open_dataset(result["data_path"]) as ds:
        assert "time" in ds.dims
        assert "lead_time" not in ds.dims
    receipt = json.loads((tmp_path / "out" / "prepared.json").read_text())
    assert len(receipt["sha256"]["data_path"]) == 64


def test_missing_initialization_is_fatal_reference_nan_is_allowed(tmp_path):
    ds = cams_fixture()
    ds["no2"].values[0, 0, 0, 0, 0] = np.nan
    raw = tmp_path / "raw.nc"
    ds.to_netcdf(raw)
    with pytest.raises(ValueError, match="Nonfinite initialization"):
        prepare({"raw_paths": [str(raw)], "cycles": ["2024-07-01T12"], "lead_hours": [12],
                 "splits": {"test": ["2024-07-01T00", "2024-07-03T00"]}}, tmp_path / "out")


def test_global_history_and_small_regional_reference_can_have_distinct_grids(tmp_path):
    raw = cams_fixture()
    history = raw.sel(step=[np.timedelta64(0, "h")])
    reference = raw[["no2", "tcno2"]].sel(step=[np.timedelta64(12, "h")], pressure_level=[1000, 925, 850])
    reference = reference.isel(latitude=slice(0, 2), longitude=slice(0, 2))
    hp, rp = tmp_path / "history.nc", tmp_path / "reference.nc"
    history.to_netcdf(hp)
    reference.to_netcdf(rp)
    result = prepare({"raw_paths": [str(hp), str(rp)], "cycles": ["2024-07-01T12"], "lead_hours": [12],
                      "splits": {"test": ["2024-07-01T00", "2024-07-03T00"]}}, tmp_path / "out")
    with xr.open_dataset(result["reference_path"]) as prepared:
        assert set(prepared.data_vars) == {"no2", "tcno2"}
        assert prepared.sizes["latitude"] == 2
    with xr.open_dataset(result["data_path"]) as prepared:
        assert prepared.sizes["latitude"] == 3
