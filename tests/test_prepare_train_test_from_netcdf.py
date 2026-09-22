"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Fail-fast alignment tests for CAMS training-data preparation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from finetune.prepare_train_test_from_netcdf import (
    _align_surface_and_atmospheric_parts,
    _resolve_split_times,
)


@pytest.mark.parametrize('use_environment', [False, True])
def test_portable_data_default_and_explicit_cli_override(tmp_path, monkeypatch, use_environment):
    import runpy
    import sys
    from finetune import prepare_train_test_from_netcdf as prep

    monkeypatch.setattr(sys, 'path', list(sys.path))
    if use_environment:
        expected = tmp_path / 'recipient-cams'
        monkeypatch.setenv('CAMS_DATA_DIR', str(expected))
    else:
        monkeypatch.delenv('CAMS_DATA_DIR', raising=False)
        expected = Path(prep.__file__).resolve().parents[1] / 'data' / 'cams'
    namespace = runpy.run_path(prep.__file__, run_name='preparation_cli_test')
    monkeypatch.setattr(sys, 'argv', ['prepare'])
    assert namespace['_parse_args']().data_folder == expected
    override = tmp_path / 'explicit-raw-data'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--data-folder', str(override)])
    assert namespace['_parse_args']().data_folder == override


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


def test_split_times_are_loaded_from_yaml_with_cli_overrides() -> None:
    args = Namespace(
        train_start_time=None,
        train_end_time=None,
        test_start_time="2024-08-01T00:00:00",
        test_end_time=None,
    )
    _resolve_split_times(
        {
            "data": {
                "split_times": {
                    "train": {
                        "start": "2023-01-01T00:00:00",
                        "end": "2023-12-31T12:00:00",
                    },
                    "test": {
                        "start": "2024-01-01T00:00:00",
                        "end": "2024-06-30T12:00:00",
                    },
                }
            }
        },
        args,
    )
    assert args.train_start_time == "2023-01-01T00:00:00"
    assert args.train_end_time == "2023-12-31T12:00:00"
    assert args.test_start_time == "2024-08-01T00:00:00"
    assert args.test_end_time == "2024-06-30T12:00:00"


def test_split_times_are_required_in_yaml() -> None:
    args = Namespace(
        train_start_time=None,
        train_end_time=None,
        test_start_time=None,
        test_end_time=None,
    )
    with pytest.raises(ValueError, match="data.split_times"):
        _resolve_split_times({"data": {}}, args)


def _output_args(tmp_path):
    return Namespace(train_out=tmp_path / 'train.nc', val_out=None, test_out=tmp_path / 'test.nc')


def test_staged_outputs_back_up_existing_files_only_after_success(tmp_path):
    from finetune.prepare_train_test_from_netcdf import _staged_output_paths

    args = _output_args(tmp_path)
    args.train_out.write_bytes(b'old train')
    args.test_out.write_bytes(b'old test')
    with _staged_output_paths(args) as staged:
        staged.train_out.write_bytes(b'new train')
        staged.test_out.write_bytes(b'new test')
        assert args.train_out.read_bytes() == b'old train'
        assert args.test_out.read_bytes() == b'old test'
    assert args.train_out.read_bytes() == b'new train'
    assert args.test_out.read_bytes() == b'new test'
    backup, = tmp_path.glob('backup_*')
    assert (backup / 'train.nc').read_bytes() == b'old train'
    assert (backup / 'test.nc').read_bytes() == b'old test'
    # A repeated run preserves the earlier backup and makes another one.
    with _staged_output_paths(args) as staged:
        staged.train_out.write_bytes(b'next train')
        staged.test_out.write_bytes(b'next test')
    assert len(list(tmp_path.glob('backup_*'))) == 2
    assert (backup / 'train.nc').read_bytes() == b'old train'


def test_failed_preparation_leaves_current_datasets_intact(tmp_path):
    from finetune.prepare_train_test_from_netcdf import _staged_output_paths

    args = _output_args(tmp_path)
    args.train_out.write_bytes(b'old train')
    args.test_out.write_bytes(b'old test')
    with pytest.raises(RuntimeError, match='interrupted'):
        with _staged_output_paths(args) as staged:
            staged.train_out.write_bytes(b'partial train')
            raise RuntimeError('interrupted')
    assert args.train_out.read_bytes() == b'old train'
    assert args.test_out.read_bytes() == b'old test'
    assert not list(tmp_path.glob('.prepare-*'))
    assert not list(tmp_path.glob('backup_*'))


def test_failed_backup_does_not_replace_any_output(tmp_path, monkeypatch):
    from finetune import prepare_train_test_from_netcdf as prep

    args = _output_args(tmp_path)
    args.train_out.write_bytes(b'old train')
    args.test_out.write_bytes(b'old test')

    def fail_copy(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(prep.shutil, 'copy2', fail_copy)
    with pytest.raises(OSError, match='disk full'):
        with prep._staged_output_paths(args) as staged:
            staged.train_out.write_bytes(b'new train')
            staged.test_out.write_bytes(b'new test')
    assert args.train_out.read_bytes() == b'old train'
    assert args.test_out.read_bytes() == b'old test'


def test_missing_staged_split_does_not_replace_any_output(tmp_path):
    from finetune.prepare_train_test_from_netcdf import _staged_output_paths

    args = _output_args(tmp_path)
    args.train_out.write_bytes(b'old train')
    args.test_out.write_bytes(b'old test')
    with pytest.raises(ValueError, match='required output'):
        with _staged_output_paths(args) as staged:
            staged.train_out.write_bytes(b'new train')
    assert args.train_out.read_bytes() == b'old train'
    assert args.test_out.read_bytes() == b'old test'


@pytest.mark.parametrize('streaming', [True, False])
def test_main_prepares_yaml_splits_and_backs_up_existing_outputs(tmp_path, monkeypatch, streaming):
    from finetune import prepare_train_test_from_netcdf as prep
    import yaml

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({
        'case_name': 'test_case',
        'paths': {'data_dir': str(tmp_path)},
        'data': {
            'predictor_variables': [{'dataset_name': 'tcno2', 'kind': 'surf'}],
            'split_times': {
                'train': {'start': '2023-12-31T00', 'end': '2023-12-31T12'},
                'test': {'start': '2024-01-01T00', 'end': '2024-01-01T12'},
            },
        },
    }))
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'sample-lead0-surface-level.nc').touch()
    (inputs / 'sample-lead0-atmospheric.nc').touch()
    case_dir = tmp_path / 'test_case'
    case_dir.mkdir()
    (case_dir / 'train.nc').write_bytes(b'old train')
    (case_dir / 'test.nc').write_bytes(b'old test')
    args = Namespace(
        config=config_path, data_folder=inputs,
        train_out=None, val_out=None, test_out=None,
        train_start_time=None, train_end_time=None,
        test_start_time=None, test_end_time=None,
        lat_min=None, lat_max=None, lon_min=None, lon_max=None,
        compression_level=0, dry_run=False, streaming_write=streaming,
    )
    times = ['2023-12-31T00', '2023-12-31T12', '2024-01-01T00', '2024-01-01T12']
    monkeypatch.setattr(prep, '_parse_args', lambda: args)
    monkeypatch.setattr(prep, '_build_merged_dataset', lambda **kwargs: _part('tcno2', times))
    prep.main()
    with xr.open_dataset(case_dir / 'train.nc') as ds:
        np.testing.assert_array_equal(ds.time.values, np.asarray(times[:2], dtype='datetime64[ns]'))
    with xr.open_dataset(case_dir / 'test.nc') as ds:
        np.testing.assert_array_equal(ds.time.values, np.asarray(times[2:], dtype='datetime64[ns]'))
    backup, = case_dir.glob('backup_*')
    assert (backup / 'train.nc').read_bytes() == b'old train'
    assert (backup / 'test.nc').read_bytes() == b'old test'
