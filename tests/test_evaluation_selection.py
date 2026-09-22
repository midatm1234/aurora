"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from finetune.evaluation_selection import InferenceSelection


@pytest.fixture
def selected(tmp_path):
    case = tmp_path / 'outputs' / 'checkpoints' / 'case'
    case.mkdir(parents=True)
    (case / 'last.ckpt').write_bytes(b'latest weights')
    (case / 'last.ckpt.metadata.json').write_text(json.dumps({
        'training_run_id': 'current', 'epoch': 99, 'best_val_loss': float('inf'),
        'validated_for_inference': False,
    }))
    (case / 'training_run.metadata.json').write_text(json.dumps({'training_run_id': 'current'}))
    raw = {'case_name': 'case', 'paths': {'project_root': '.', 'output_dir': 'outputs'},
           'data': {'split_times': {'test': {'start': '2024-01-01T00', 'end': '2024-01-02T00'}}}}
    return InferenceSelection.from_config(raw, tmp_path / 'config.yaml')


def test_selection_uses_latest_weights_and_inclusive_yaml_period(selected):
    assert selected.checkpoint.name == 'last.ckpt'
    assert selected.epoch == 99
    assert not selected.validated
    assert selected.contains('2024-01-01T00')
    assert selected.contains('2024-01-02T00')
    assert not selected.contains('2023-12-31T12')
    assert not selected.contains('2024-01-02T12')
    assert selected.accepts({'checkpoint_sha256': selected.sha256})
    assert not selected.accepts({'checkpoint_sha256': 'older-epoch', 'checkpoint_training_run_id': 'current'})
    assert not selected.accepts({'checkpoint_sha256': selected.sha256, 'checkpoint_training_run_id': 'stale'})
    assert not selected.accepts({})


def test_expected_coverage_retains_history_and_rejects_short_truth(selected):
    times = np.array(['2023-12-31T12', '2024-01-01T00', '2024-01-01T12', '2024-01-02T00'], dtype='datetime64[ns]')
    assert len(selected.expected_initializations(times, 2)) == 3
    assert len(selected.expected_initializations(times[1:], 2)) == 2
    with pytest.raises(ValueError, match='does not cover'):
        selected.expected_initializations(times[:-1], 2)


def test_selection_rejects_provenance_changes_during_hashing(selected, tmp_path, monkeypatch):
    from finetune import evaluation_selection as selection_module

    original_sha256 = selection_module.hashlib.sha256
    sidecar = selected.checkpoint.with_suffix('.ckpt.metadata.json')

    class RacingDigest:
        def __init__(self):
            self.digest = original_sha256()

        def update(self, chunk):
            self.digest.update(chunk)
            metadata = json.loads(sidecar.read_text())
            metadata['epoch'] += 1
            sidecar.write_text(json.dumps(metadata))

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(selection_module.hashlib, 'sha256', RacingDigest)
    raw = {'case_name': 'case', 'paths': {'project_root': '.', 'output_dir': 'outputs'},
           'data': {'split_times': {'test': {'start': '2024-01-01', 'end': '2024-01-02'}}}}
    with pytest.raises(ValueError, match='provenance changed'):
        InferenceSelection.from_config(raw, tmp_path / 'config.yaml')


def test_selection_rejects_missing_epoch_with_actionable_error(selected, tmp_path):
    sidecar = selected.checkpoint.with_suffix('.ckpt.metadata.json')
    metadata = json.loads(sidecar.read_text())
    metadata.pop('epoch')
    sidecar.write_text(json.dumps(metadata))
    raw = {'case_name': 'case', 'paths': {'project_root': '.', 'output_dir': 'outputs'},
           'data': {'split_times': {'test': {'start': '2024-01-01', 'end': '2024-01-02'}}}}
    with pytest.raises(ValueError, match='nonnegative integer epoch'):
        InferenceSelection.from_config(raw, tmp_path / 'config.yaml')


def test_notebook_catalog_excludes_stale_and_out_of_period_forecasts(selected, tmp_path):
    notebook = json.loads((Path(__file__).parents[1] / 'evaluate_finetuned_cams_rollouts_flow_matching.ipynb').read_text())
    module = types.ModuleType('evaluation_notebook_selection_test')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    try:
        imports = ''.join(notebook['cells'][3]['source']).split('# Main notebook input.')[0]
        exec(imports, ns)
        definitions = ''.join(notebook['cells'][4]['source']).split('RAW_CONFIG, EVAL =')[0]
        exec(definitions, ns)
        ns.update(RAW_CONFIG={'data': {}}, EVAL={
            'recursive_file_search': False, 'file_pattern': '*.nc',
            'lead_time_tolerance_hours': 1e-6, 'lead_time_numeric_units': 'hours',
        }, INFERENCE_SELECTION=selected)
        exec(''.join(notebook['cells'][6]['source']), ns)
        forecasts = tmp_path / 'forecasts'
        forecasts.mkdir()
        for name, init, checksum in (
            ('latest', '2024-01-01T00', selected.sha256),
            ('stale_duplicate', '2024-01-01T00', 'stale'),
            ('outside', '2024-01-03T00', selected.sha256),
        ):
            ds = xr.Dataset({'tcno2': (('time', 'latitude', 'longitude'), np.ones((1, 2, 2)))},
                coords={'time': [np.datetime64(init) + np.timedelta64(12, 'h')],
                        'latitude': [1., 0.], 'longitude': [1., 2.]},
                attrs={'initialization_time': init, 'checkpoint_sha256': checksum})
            ds.to_netcdf(forecasts / f'{name}.nc')
        catalog = ns['build_catalog']('fine-tuned', forecasts)
        assert [path.name for path in catalog.files] == ['latest.nc']
        assert len(catalog.records) == 1
        assert catalog.records[0].initialization_time == np.datetime64('2024-01-01T00')
    finally:
        del sys.modules[module.__name__]
