"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

NetCDF output contract for refined Aurora rollouts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")

from finetune.refinement.io import (  # noqa: E402
    build_refined_dataset,
    resolve_forecast_variable,
    write_refined_netcdf,
)

from tests.refinement_fixtures import build_packing  # noqa: E402


def make_dataset(members: int = 3, steps: int = 3, height: int = 5, width: int = 6):
    packing = build_packing(height, width)
    channels = packing.num_channels
    torch.manual_seed(0)
    deterministic = torch.randn(steps, channels, height, width)
    residual = 0.1 * torch.randn(steps, channels, height, width)
    member_fields = deterministic.unsqueeze(1) + 0.1 * torch.randn(
        steps, members, channels, height, width
    )
    init = np.array(["2024-01-01T00"] * steps, dtype="datetime64[h]")
    valid = np.array([f"2024-01-01T{6 * (i + 1):02d}" for i in range(steps)], dtype="datetime64[h]")
    leads = [6.0 * (i + 1) for i in range(steps)]
    dataset = build_refined_dataset(
        packing=packing,
        deterministic=deterministic,
        init_time=init,
        valid_time=valid,
        lead_time_hours=leads,
        refined=member_fields.mean(dim=1),
        residual=residual,
        members=member_fields,
        ensemble_mean=member_fields.mean(dim=1),
        ensemble_spread=member_fields.std(dim=1, unbiased=True),
        truth=deterministic + 0.2,
        attrs={"experiment": "unit-test"},
    )
    return packing, dataset, deterministic, member_fields, leads, valid


def test_dataset_preserves_variables_levels_and_coordinates() -> None:
    packing, dataset, deterministic, members, leads, valid = make_dataset()

    assert "gtco3" in dataset and "go3" in dataset
    assert dataset["gtco3"].dims == ("rollout_step", "latitude", "longitude")
    assert dataset["go3"].dims == ("rollout_step", "level", "latitude", "longitude")
    assert dataset["go3_members"].dims == (
        "rollout_step",
        "member",
        "level",
        "latitude",
        "longitude",
    )
    assert list(dataset["level"].values) == [500.0, 850.0]
    assert list(dataset["latitude"].values) == list(packing.lat)
    assert list(dataset["longitude"].values) == list(packing.lon)
    assert dataset["gtco3"].attrs["units"] == "kg m-2"
    assert dataset["go3"].attrs["units"] == "kg kg-1"
    assert dataset["gtco3_residual"].attrs["units"].startswith("1 (normalized")
    assert (
        dataset["gtco3_residual"].attrs["aurora_refinement_role"]
        == "predicted_correction_normalized"
    )
    assert dataset["gtco3_refined"].attrs["aurora_refinement_role"] == "refined_forecast"
    assert dataset.attrs["correction_target_convention"] == "CAMS - Aurora"
    assert (
        dataset.attrs["refined_forecast_convention"]
        == "Aurora + predicted_correction_physical"
    )
    assert dataset.attrs["experiment"] == "unit-test"
    assert "refinement_note" in dataset.attrs


def test_forecast_times_and_rollout_order_are_preserved() -> None:
    _, dataset, _, _, leads, valid = make_dataset()
    assert list(dataset["rollout_step"].values) == [1, 2, 3]
    assert list(dataset["lead_time_hours"].values) == leads
    assert list(dataset["time"].values) == list(valid)
    assert len(set(dataset["init_time"].values.tolist())) == 1
    assert dataset["lead_time_hours"].attrs["units"] == "hours"


def test_values_and_member_order_survive_the_round_trip() -> None:
    packing, dataset, deterministic, members, _, _ = make_dataset()
    surf_index = packing.index_of("gtco3", None)
    assert np.allclose(dataset["gtco3"].values, deterministic[:, surf_index].numpy())
    level_indices = [spec.index for spec in packing.channels_for("go3")]
    assert np.allclose(dataset["go3"].values, deterministic[:, level_indices].numpy())
    assert np.allclose(dataset["go3_members"].values, members[:, :, level_indices].numpy())
    assert list(dataset["member"].values) == [0, 1, 2]


def test_default_forecast_resolves_refined_physical_field_not_plain_aurora() -> None:
    packing, dataset, deterministic, members, _, _ = make_dataset()
    surface_index = packing.index_of("gtco3", None)
    expected_refined = members.mean(dim=1)[:, surface_index].numpy()

    selected = resolve_forecast_variable(dataset, "gtco3")

    assert selected.name == "gtco3_refined"
    np.testing.assert_allclose(selected.values, expected_refined)
    assert not np.allclose(selected.values, deterministic[:, surface_index].numpy())
    np.testing.assert_allclose(
        dataset["gtco3_predicted_correction_physical"].values,
        expected_refined - deterministic[:, surface_index].numpy(),
    )


def test_legacy_plain_forecast_remains_supported() -> None:
    dataset = xr.Dataset({"tcno2": (("time", "latitude", "longitude"), np.ones((1, 2, 3)))})

    selected = resolve_forecast_variable(dataset, "tcno2")

    assert selected.name == "tcno2"


def test_two_phase_dataset_without_refinement_resolves_plain_aurora() -> None:
    packing = build_packing(4, 4)
    deterministic = torch.zeros(1, packing.num_channels, 4, 4)
    dataset = build_refined_dataset(
        packing=packing,
        deterministic=deterministic,
        init_time=np.array(["2024-01-01T00"], dtype="datetime64[h]"),
        valid_time=np.array(["2024-01-01T06"], dtype="datetime64[h]"),
        lead_time_hours=[6.0],
    )

    selected = resolve_forecast_variable(dataset, "gtco3")

    assert selected.name == "gtco3"
    assert selected.attrs["aurora_refinement_role"] == "aurora_forecast"


def test_conventional_refined_suffix_wins_for_legacy_two_phase_file() -> None:
    dataset = xr.Dataset(
        {
            "tcno2": (("time", "latitude", "longitude"), np.zeros((1, 2, 3))),
            "tcno2_refined": (("time", "latitude", "longitude"), np.ones((1, 2, 3))),
        }
    )

    selected = resolve_forecast_variable(dataset, "tcno2")

    assert selected.name == "tcno2_refined"


def test_no_unexpected_nan_or_infinite_values() -> None:
    _, dataset, _, _, _, _ = make_dataset()
    for name, variable in dataset.data_vars.items():
        assert np.isfinite(variable.values).all(), name


def test_write_and_reread(tmp_path) -> None:
    _, dataset, deterministic, members, _, _ = make_dataset()
    target = tmp_path / "refined.nc"
    write_refined_netcdf(dataset, target, compression=True, compression_level=1)
    assert target.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["refined.nc"]

    reopened = xr.open_dataset(target)
    try:
        assert set(reopened.data_vars) == set(dataset.data_vars)
        assert np.allclose(reopened["gtco3"].values, dataset["gtco3"].values)
        assert np.allclose(reopened["go3_members"].values, dataset["go3_members"].values)
        assert resolve_forecast_variable(reopened, "gtco3").name == "gtco3_refined"
        assert list(reopened["level"].values) == [500.0, 850.0]
        assert list(reopened["rollout_step"].values) == [1, 2, 3]
    finally:
        reopened.close()


def test_channel_count_mismatch_is_rejected() -> None:
    packing = build_packing(4, 4)
    with pytest.raises(ValueError, match="channels"):
        build_refined_dataset(
            packing=packing,
            deterministic=torch.zeros(2, packing.num_channels + 1, 4, 4),
            init_time=["a", "b"],
            valid_time=["c", "d"],
            lead_time_hours=[6.0, 12.0],
        )


def test_time_length_mismatch_is_rejected() -> None:
    packing = build_packing(4, 4)
    with pytest.raises(ValueError, match="rollout steps"):
        build_refined_dataset(
            packing=packing,
            deterministic=torch.zeros(2, packing.num_channels, 4, 4),
            init_time=["a"],
            valid_time=["c", "d"],
            lead_time_hours=[6.0, 12.0],
        )


def test_valid_time_must_equal_initialization_plus_lead() -> None:
    packing = build_packing(4, 4)
    with pytest.raises(ValueError, match="Forecast-time mismatch"):
        build_refined_dataset(
            packing=packing,
            deterministic=torch.zeros(2, packing.num_channels, 4, 4),
            init_time=np.array(["2024-01-01T00", "2024-01-01T00"], dtype="datetime64[h]"),
            valid_time=np.array(["2024-01-01T06", "2024-01-01T13"], dtype="datetime64[h]"),
            lead_time_hours=[6.0, 12.0],
        )


def test_optional_field_shape_mismatch_is_rejected() -> None:
    packing = build_packing(4, 4)
    with pytest.raises(ValueError, match="refined has non-member shape"):
        build_refined_dataset(
            packing=packing,
            deterministic=torch.zeros(2, packing.num_channels, 4, 4),
            refined=torch.zeros(1, packing.num_channels, 4, 4),
            init_time=np.array(["2024-01-01T00", "2024-01-01T00"], dtype="datetime64[h]"),
            valid_time=np.array(["2024-01-01T06", "2024-01-01T12"], dtype="datetime64[h]"),
            lead_time_hours=[6.0, 12.0],
        )
