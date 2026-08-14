"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune import aurora_finetune_utils as ft
from finetune import prepare_train_test_from_netcdf as prep


def _config_text() -> str:
    return """
case_name: smoke_case
paths:
  project_root: .
  data_dir: data
  output_dir: outputs
  checkpoint_dir: outputs/checkpoints
data:
  time_dim: time
  lat_dim: latitude
  lon_dim: longitude
"""


def _tiny_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"sample": (("time", "latitude", "longitude"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2024-01-01T00:00:00"], dtype="datetime64[ns]"),
            "latitude": np.array([1.0, 0.0], dtype=np.float32),
            "longitude": np.array([10.0, 11.0], dtype=np.float32),
        },
    )


def _check_case_specific_prepare_finetune_and_inference_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text())
    cfg_raw = prep._read_config(config_path)

    train_path, val_path, test_path, case_data_dir = prep._resolve_case_data_paths(
        cfg_raw,
        config_path,
        train_out=None,
        val_out=None,
        test_out=None,
    )

    assert val_path is None
    assert case_data_dir == tmp_path / "data" / "smoke_case"
    assert train_path == case_data_dir / "train.nc"
    assert test_path == case_data_dir / "test.nc"

    ds = _tiny_dataset()
    prep._write_netcdf(ds, train_path, compression_level=0)
    prep._write_netcdf(ds, test_path, compression_level=0)

    cfg = ft.load_config(config_path)
    assert Path(cfg["paths"]["train_data_path"]) == train_path
    # Configured validation resolves only to val.nc; it must never silently use
    # the held-out test split when that file is absent.
    assert Path(cfg["paths"]["val_data_path"]) == case_data_dir / "val.nc"
    assert Path(cfg["paths"]["val_data_path"]) != test_path
    assert Path(cfg["paths"]["test_data_path"]) == test_path

    train_ds = ft.open_dataset(cfg["paths"]["train_data_path"], cfg)
    inference_ds = ft.open_dataset(cfg["paths"]["test_data_path"], cfg)
    try:
        assert train_ds.sizes["time"] == 1
        assert inference_ds.sizes["time"] == 1
    finally:
        train_ds.close()
        inference_ds.close()


def _check_mismatched_out_path_is_redirected_into_case_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text())
    cfg_raw = prep._read_config(config_path)

    stale_out = tmp_path / "data" / "old_case" / "train.nc"

    train_path, _val_path, test_path, case_data_dir = prep._resolve_case_data_paths(
        cfg_raw,
        config_path,
        train_out=stale_out,
        val_out=None,
        test_out=None,
    )

    assert case_data_dir == tmp_path / "data" / "smoke_case"
    assert train_path == case_data_dir / "train.nc"
    assert test_path == case_data_dir / "test.nc"


def _check_missing_case_name_and_missing_prepared_files_are_clear(tmp_path: Path) -> None:
    missing_case_config = tmp_path / "missing_case.yaml"
    missing_case_config.write_text(
        """
paths:
  project_root: .
  data_dir: data
"""
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "case_name"):
        ft.load_config(missing_case_config)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_text())

    with unittest.TestCase().assertRaisesRegex(FileNotFoundError, "prepare_train_test_from_netcdf.py"):
        ft.load_config(config_path)


def _check_shared_data_case_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _config_text().replace(
            "  data_dir: data\n",
            "  data_dir: data\n  data_case_name: shared_prepared_data\n",
        )
    )
    cfg_raw = prep._read_config(config_path)
    train_path, _val_path, test_path, case_data_dir = prep._resolve_case_data_paths(
        cfg_raw,
        config_path,
        train_out=None,
        val_out=None,
        test_out=None,
    )
    assert case_data_dir == tmp_path / "data" / "shared_prepared_data"
    ds = _tiny_dataset()
    prep._write_netcdf(ds, train_path, compression_level=0)
    prep._write_netcdf(ds, test_path, compression_level=0)

    cfg = ft.load_config(config_path)
    assert cfg["case_name"] == "smoke_case"
    assert cfg["paths"]["data_case_name"] == "shared_prepared_data"
    assert Path(cfg["paths"]["train_data_path"]) == train_path


class CaseDataPathSmokeTest(unittest.TestCase):
    def test_case_specific_prepare_finetune_and_inference_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            _check_case_specific_prepare_finetune_and_inference_paths(Path(tmp))

    def test_missing_case_name_and_missing_prepared_files_are_clear(self) -> None:
        with TemporaryDirectory() as tmp:
            _check_missing_case_name_and_missing_prepared_files_are_clear(Path(tmp))

    def test_mismatched_out_path_is_redirected_into_case_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            _check_mismatched_out_path_is_redirected_into_case_dir(Path(tmp))

    def test_shared_data_case_name(self) -> None:
        with TemporaryDirectory() as tmp:
            _check_shared_data_case_name(Path(tmp))


def test_case_specific_prepare_finetune_and_inference_paths(tmp_path: Path) -> None:
    _check_case_specific_prepare_finetune_and_inference_paths(tmp_path)


def test_missing_case_name_and_missing_prepared_files_are_clear(tmp_path: Path) -> None:
    _check_missing_case_name_and_missing_prepared_files_are_clear(tmp_path)


def test_mismatched_out_path_is_redirected_into_case_dir(tmp_path: Path) -> None:
    _check_mismatched_out_path_is_redirected_into_case_dir(tmp_path)


def test_shared_data_case_name(tmp_path: Path) -> None:
    _check_shared_data_case_name(tmp_path)


def test_train_tail_validation_uses_train_never_test(
    tmp_path: Path, monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _config_text()
        + "\ntraining:\n  validation_source: train_tail\n"
    )
    cfg_raw = prep._read_config(config_path)
    train_path, _val_path, test_path, _case_data_dir = prep._resolve_case_data_paths(
        cfg_raw,
        config_path,
        train_out=None,
        val_out=None,
        test_out=None,
    )
    dataset = _tiny_dataset()
    prep._write_netcdf(dataset, train_path, compression_level=0)
    prep._write_netcdf(dataset, test_path, compression_level=0)

    # This fixture intentionally exercises only load_config path resolution; the
    # minimal preparation recipe omits the full model/training schema.
    monkeypatch.setattr(ft, "validate_config", lambda *_args, **_kwargs: None)
    cfg = ft.load_config(config_path)
    assert Path(cfg["paths"]["val_data_path"]) == train_path
    assert Path(cfg["paths"]["val_data_path"]) != test_path


def test_atmospheric_missing_mask_is_matched_by_pressure_value() -> None:
    dataset = xr.Dataset(
        {
            "no2_valid": (
                ("time", "level", "latitude", "longitude"),
                np.asarray([[[[0]], [[0]], [[1]]]], dtype=np.int8),
            )
        },
        coords={
            "time": np.asarray(["2024-01-01T12"], dtype="datetime64[h]"),
            # Deliberately opposite to the configured Aurora level order.
            "level": np.asarray([850.0, 925.0, 1000.0]),
            "latitude": np.asarray([40.0]),
            "longitude": np.asarray([240.0]),
        },
    )
    config = {
        "data": {
            "time_dim": "time",
            "lat_dim": "latitude",
            "lon_dim": "longitude",
            "level_dim": "level",
            "atmos_levels": [1000.0, 925.0, 850.0],
            "extra_dim_indexers": {},
            "optional_masks_for_missing_values": {"no2": "no2_valid"},
        }
    }
    result = ft._target_missing_mask(
        dataset,
        ft.VariableSpec("no2", "no2", "atmos"),
        {"target_indices": {1: 0}},
        lead=1,
        config=config,
    )

    assert result is not None
    np.testing.assert_array_equal(result[:, 0, 0].numpy(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
