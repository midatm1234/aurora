"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the standalone overall evaluation-map helper."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from finetune.generate_overall_evaluation_maps import (
    Accumulator,
    assert_same_coordinate,
    configured_map_extent,
    load_settings,
    match_coordinate_indices,
    match_valid_time_index,
    validate_rollout_time_metadata,
)


def test_load_settings_matches_notebook_path_defaults(tmp_path: Path) -> None:
    """Training paths and the example-baseline fallback match the notebook."""
    root = tmp_path / "repository"
    config_dir = root / "finetune"
    (root / ".git").mkdir(parents=True)
    (root / "examples" / "outputs" / "cams_rollouts").mkdir(parents=True)
    config_dir.mkdir()
    config_path = config_dir / "case.yaml"
    config_path.write_text(
        """\
case_name: example_case
paths:
  project_root: .
  data_dir: ../data
  output_dir: ./outputs
data:
  target_lead_times: [1, 2]
  lat_min: 31
  lat_max: 52
  lon_min: -128
  lon_max: -100
rollout:
  rollout_step_hours: 12
"""
    )

    settings = load_settings(config_path)

    assert settings["truth_path"] == root / "data" / "example_case" / "test.nc"
    assert settings["baseline_dir"] == root / "examples" / "outputs" / "cams_rollouts"
    assert settings["finetuned_dir"] == config_dir / "outputs" / "example_case"
    assert (
        settings["output_dir"]
        == config_dir / "outputs" / "example_case" / "evaluation"
    )
    assert settings["lead_hours"] == [12.0, 24.0]
    assert settings["map_extent"] == (-128.0, -100.0, 31.0, 52.0)
    assert settings["coordinate_tolerance"] == 2.0e-5


def test_load_settings_honors_separate_data_case_name(tmp_path: Path) -> None:
    """Diffusion experiments may reuse data prepared under another case name."""
    root = tmp_path / "repository"
    config_dir = root / "finetune"
    (root / ".git").mkdir(parents=True)
    config_dir.mkdir()
    config_path = config_dir / "diffusion_transformer.yaml"
    config_path.write_text(
        """\
case_name: diffusion_transformer_experiment
paths:
  project_root: .
  data_dir: ../data
  data_case_name: prepared_no2_data
  output_dir: ./outputs
data:
  target_lead_times: [1, 2, 3]
rollout:
  rollout_step_hours: 12
"""
    )

    settings = load_settings(config_path)

    assert settings["truth_path"] == root / "data" / "prepared_no2_data" / "test.nc"
    assert (
        settings["finetuned_dir"]
        == config_dir / "outputs" / "diffusion_transformer_experiment"
    )
    assert settings["lead_hours"] == [12.0, 24.0, 36.0]


def test_accumulator_reports_signed_bias_and_nonnegative_mae() -> None:
    """Signed residual means are biases; MAE fields remain nonnegative."""
    accumulator = Accumulator.create((1, 2))
    accumulator.update(
        truth=np.array([[2.0, 2.0]]),
        baseline=np.array([[1.0, 4.0]]),
        finetuned=np.array([[1.5, 3.0]]),
    )

    fields = accumulator.means()

    np.testing.assert_allclose(fields["aurora_bias"], [[-1.0, 2.0]])
    np.testing.assert_allclose(fields["finetuned_bias"], [[-0.5, 1.0]])
    assert np.all(fields["baseline_mae"] >= 0)
    assert np.all(fields["finetuned_mae"] >= 0)
    assert "baseline_mean_error" not in fields
    assert "finetuned_mean_error" not in fields


def test_none_boundaries_select_global_map() -> None:
    """O3-style None boundaries select Cartopy's true global extent."""
    raw = {
        "data": {
            "domain_type": "global",
            "lat_min": "None",
            "lat_max": "None",
            "lon_min": "None",
            "lon_max": "None",
        }
    }

    assert configured_map_extent(raw) is None


def _rollout_metadata_dataset(*, times, leads, initialization="2024-01-01T00"):
    return xr.Dataset(
        coords={
            "time": np.asarray(times, dtype="datetime64[h]"),
            "lead_time": ("time", np.asarray(leads, dtype=float)),
            "forecast_reference_time": np.datetime64(initialization, "h"),
        },
        attrs={"initialization_time": initialization},
    )


def test_rollout_time_metadata_enforces_init_plus_lead_equals_valid() -> None:
    dataset = _rollout_metadata_dataset(
        times=["2024-01-01T12", "2024-01-02T00"],
        leads=[12.0, 24.0],
    )

    times, step_dim = validate_rollout_time_metadata(
        dataset,
        np.datetime64("2024-01-01T00"),
        context="refined.nc",
    )

    assert step_dim == "time"
    np.testing.assert_array_equal(
        times,
        np.asarray(["2024-01-01T12", "2024-01-02T00"], dtype="datetime64[ns]"),
    )


def test_rollout_time_metadata_rejects_shifted_lead() -> None:
    dataset = _rollout_metadata_dataset(
        times=["2024-01-01T12", "2024-01-02T00"],
        leads=[12.0, 36.0],
    )

    with pytest.raises(ValueError, match="valid_time = initialization_time"):
        validate_rollout_time_metadata(
            dataset,
            np.datetime64("2024-01-01T00"),
            context="refined.nc",
        )


def test_baseline_step_is_matched_by_valid_time_not_position() -> None:
    reversed_times = np.asarray(
        ["2024-01-02T00", "2024-01-01T12"], dtype="datetime64[ns]"
    )

    assert (
        match_valid_time_index(
            reversed_times,
            np.datetime64("2024-01-01T12"),
            context="baseline.nc",
        )
        == 1
    )


def test_coordinate_order_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="value and order"):
        assert_same_coordinate(
            np.array([52.0, 51.6]),
            np.array([51.6, 52.0]),
            1.0e-6,
            name="latitude",
            context="refined.nc",
        )


def test_duplicate_coordinate_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        match_coordinate_indices(
            np.array([51.6, 52.0]),
            np.array([52.0, 52.0]),
            1.0e-6,
            name="latitude",
        )


@pytest.mark.parametrize('use_inference_selection', [True, False])
def test_aggregate_write_preserves_inference_provenance_with_multiple_targets(
    tmp_path, monkeypatch, use_inference_selection,
):
    """Exercise the full aggregate writer past both variable-selection loops."""
    import yaml
    from finetune import generate_overall_evaluation_maps as maps
    from finetune.evaluation_selection import InferenceSelection

    baseline_dir = tmp_path / 'baseline'
    finetuned_dir = tmp_path / 'finetuned'
    baseline_dir.mkdir()
    finetuned_dir.mkdir()
    coords = {'latitude': [52., 51.], 'longitude': [232., 233.]}
    times = np.array(['2024-01-01T12', '2024-01-02T12'], dtype='datetime64[ns]')
    variables = ('tcno2', 'gtco3')

    def dataset(value, valid_times):
        return xr.Dataset(
            {name: (('time', 'latitude', 'longitude'),
                    np.full((len(valid_times), 2, 2), value)) for name in variables},
            coords={**coords, 'time': valid_times},
        )

    truth_path = tmp_path / 'truth.nc'
    dataset(5., times).to_netcdf(truth_path)
    for index, day in enumerate(('20240101', '20240102')):
        initialization = str(times[index] - np.timedelta64(12, 'h'))
        baseline = dataset(3., times[index:index + 1])
        baseline.attrs['initialization_time'] = initialization
        baseline.to_netcdf(baseline_dir / f'rollout_{day}_000000.nc')
        fine = dataset(4., times[index:index + 1])
        fine.attrs.update(initialization_time=initialization,
                          checkpoint_sha256='latest' if index == 0 else 'stale',
                          checkpoint_training_run_id='current')
        fine.to_netcdf(finetuned_dir / f'rollout_predictions_init_{day}T000000.nc')

    output_dir = tmp_path / 'evaluation'
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({
        'case_name': 'case',
        'data': {'target_variables': [
            {'dataset_name': name, 'kind': 'surf'} for name in variables
        ]},
        'evaluation': {
            'ground_truth_path': str(truth_path),
            'baseline_rollout_path': str(baseline_dir),
            'finetuned_rollout_path': str(finetuned_dir),
            'output_dir': str(output_dir),
            'lead_times_hours': [12],
        },
    }))
    inference_selection = InferenceSelection(
        np.datetime64('2024-01-01', 'ns'), np.datetime64('2024-01-01T12', 'ns'),
        tmp_path / 'last.ckpt', 'latest', 'current', 99, False,
    ) if use_inference_selection else None
    # Plot rendering is unrelated to the writer bug and requires Cartopy data.
    monkeypatch.setattr(maps, 'plot_overall', lambda *args: args[-1].write_bytes(b'plot'))
    generated = maps.generate_overall_maps(config_path, selection=inference_selection)
    assert len(generated) == 2
    assert all(path.is_file() for path in generated)
    with xr.open_dataset(output_dir / 'overall_spatial_differences.nc') as aggregate:
        assert aggregate.sizes['selection'] == 2
        np.testing.assert_allclose(aggregate.aurora_bias, -2.)
        np.testing.assert_allclose(aggregate.finetuned_bias, -1.)
        np.testing.assert_array_equal(aggregate.number_of_forecasts, [1, 1] if use_inference_selection else [2, 2])
        if use_inference_selection:
            for key, value in inference_selection.provenance().items():
                assert aggregate.attrs[key] == value
        else:
            assert 'checkpoint_sha256' not in aggregate.attrs
