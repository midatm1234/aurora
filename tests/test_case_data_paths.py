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
    assert Path(cfg["paths"]["val_data_path"]) == test_path
    assert Path(cfg["paths"]["test_data_path"]) == test_path

    train_ds = ft.open_dataset(cfg["paths"]["train_data_path"], cfg)
    inference_ds = ft.open_dataset(cfg["paths"]["test_data_path"], cfg)
    try:
        assert train_ds.sizes["time"] == 1
        assert inference_ds.sizes["time"] == 1
    finally:
        train_ds.close()
        inference_ds.close()


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


class CaseDataPathSmokeTest(unittest.TestCase):
    def test_case_specific_prepare_finetune_and_inference_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            _check_case_specific_prepare_finetune_and_inference_paths(Path(tmp))

    def test_missing_case_name_and_missing_prepared_files_are_clear(self) -> None:
        with TemporaryDirectory() as tmp:
            _check_missing_case_name_and_missing_prepared_files_are_clear(Path(tmp))


def test_case_specific_prepare_finetune_and_inference_paths(tmp_path: Path) -> None:
    _check_case_specific_prepare_finetune_and_inference_paths(tmp_path)


def test_missing_case_name_and_missing_prepared_files_are_clear(tmp_path: Path) -> None:
    _check_missing_case_name_and_missing_prepared_files_are_clear(tmp_path)


if __name__ == "__main__":
    unittest.main()
