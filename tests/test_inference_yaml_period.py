"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Exercise the inference notebook's date selection without loading a model.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


NOTEBOOK = Path(__file__).resolve().parents[1] / 'finetune/aurora_inference_rollout.ipynb'


def run_selection(start='2024-01-01T00', end='2024-01-02T00', times=None, bounds=True):
    if times is None:
        times = np.arange(
            np.datetime64('2023-12-31T12'),
            np.datetime64('2024-01-03T00'),
            np.timedelta64(12, 'h'),
        )
    ds = xr.Dataset(coords={'time': times})
    data = {'input_time_steps': 2}
    if bounds:
        data['split_times'] = {'test': {'start': start, 'end': end}}
    scope = {'cfg': {'data': data}, 'test_ds': ds, 'np': np}
    notebook = json.loads(NOTEBOOK.read_text())
    source = next(
        ''.join(cell['source']) for cell in notebook['cells']
        if cell['cell_type'] == 'code'
        and 'rollout_start_samples = [' in ''.join(cell['source'])
    )
    exec(compile(source, str(NOTEBOOK), 'exec'), scope)
    return scope


def test_inclusive_period_retains_history_and_later_predictors():
    scope = run_selection()
    samples = scope['rollout_start_samples']
    assert [s['anchor_index'] for s in samples] == [1, 2, 3]
    assert samples[0]['history_indices'] == [0, 1]
    assert scope['test_ds'].sizes['time'] == 5
    assert scope['last_anchor_time'] == np.datetime64('2024-01-02T00')


def test_changing_yaml_changes_initializations():
    scope = run_selection(start='2024-01-01T12', end='2024-01-01T12')
    assert [s['anchor_index'] for s in scope['rollout_start_samples']] == [2]


def test_missing_initial_history_is_reported(capsys):
    times = np.array(['2024-01-01T00', '2024-01-01T12', '2024-01-02T00'], dtype='datetime64[h]')
    scope = run_selection(times=times)
    assert [s['anchor_index'] for s in scope['rollout_start_samples']] == [1, 2]
    assert 'Skipped 1 initialization(s)' in capsys.readouterr().out


@pytest.mark.parametrize('kwargs, message', [
    ({'bounds': False}, 'must define valid'),
    ({'start': 'not-a-date'}, 'must define valid'),
    ({'start': None}, 'finite timestamps'),
    ({'start': '2024-01-03'}, 'start <= end'),
    ({'start': '2023-12-30'}, 'Regenerate'),
    ({'end': '2024-01-04'}, 'Regenerate'),
    ({'start': '2024-01-01T01', 'end': '2024-01-01T02'}, 'No initialization'),
    ({'times': np.array([], dtype='datetime64[h]')}, 'valid timestamps'),
    ({'times': np.array(['2024-01-01', '2024-01-01'], dtype='datetime64[h]')}, 'strictly increasing'),
])
def test_invalid_or_uncovered_period_fails(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_selection(**kwargs)
