"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

End-to-end oracle contract for physical stochastic refinement outputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")

from finetune.refinement.io import (
    build_refined_dataset,
    resolve_forecast_variable,
    write_refined_netcdf,
)
from finetune.refinement.residual_scaling import ResidualScaler
from finetune.refinement.target_space import NormalizedTargetSpace
from tests.refinement_fixtures import build_packing


def _physical_fields(steps: int, height: int, width: int):
    rows = torch.linspace(-1.0, 1.0, height).view(1, height, 1)
    cols = torch.linspace(-1.0, 1.0, width).view(1, 1, width)
    time = torch.arange(steps, dtype=torch.float32).view(steps, 1, 1)
    aurora = {
        "gtco3": 1.0e-3 + 2.0e-5 * rows + 1.0e-5 * cols + 1.0e-6 * time,
        "go3": torch.stack(
            (
                4.0e-7 + 2.0e-8 * rows + 1.0e-8 * cols + 1.0e-9 * time,
                7.0e-7 - 1.0e-8 * rows + 3.0e-8 * cols + 2.0e-9 * time,
            ),
            dim=1,
        ),
    }
    correction = {
        "gtco3": 3.0e-5 + 4.0e-6 * rows - 2.0e-6 * cols,
        "go3": torch.stack(
            (
                -2.0e-8 + 3.0e-9 * rows + 1.0e-9 * cols,
                5.0e-8 - 2.0e-9 * rows + 4.0e-9 * cols,
            ),
            dim=1,
        ),
    }
    correction = {
        name: value.expand_as(aurora[name]).clone()
        for name, value in correction.items()
    }
    cams = {name: aurora[name] + correction[name] for name in aurora}
    return aurora, cams


def test_oracle_correction_survives_complete_serialization_path(tmp_path) -> None:
    """CAMS - Aurora must reconstruct CAMS after every transform and I/O boundary."""
    steps, height, width = 2, 5, 6
    packing = build_packing(height=height, width=width)
    target_space = NormalizedTargetSpace(packing)
    aurora_fields, cams_fields = _physical_fields(steps, height, width)
    aurora_physical = packing.pack(aurora_fields)
    cams_physical = packing.pack(cams_fields)

    aurora_normalized = target_space.encode(aurora_physical)
    cams_normalized = target_space.encode(cams_physical)
    correction_target_normalized, valid = (
        target_space.correction_target_from_normalized(
            cams_normalized,
            aurora_normalized,
        )
    )
    assert bool(valid.all())

    # Exercise the checkpointed per-channel correction normalizer and its exact
    # inverse. This is the one and only correction denormalization boundary.
    correction_scaler = ResidualScaler(
        packing.num_channels,
        mode="per_channel",
        center=True,
        warmup_batches=1,
    )
    correction_scaler.train()
    correction_scaler.fit(correction_target_normalized, valid)
    predicted_correction_normalized = correction_scaler.decode(
        correction_scaler.encode(correction_target_normalized)
    )
    predicted_correction_physical = target_space.correction_to_physical(
        predicted_correction_normalized
    )
    refined_normalized, refined_physical = target_space.apply_correction(
        aurora_normalized,
        predicted_correction_normalized,
        apply_constraints=False,
    )

    torch.testing.assert_close(refined_normalized, cams_normalized)
    torch.testing.assert_close(refined_physical, cams_physical, rtol=1e-5, atol=1e-12)
    torch.testing.assert_close(
        predicted_correction_physical,
        cams_physical - aurora_physical,
        rtol=1e-5,
        atol=1e-12,
    )
    unpacked = packing.unpack(refined_physical)
    for name, expected in cams_fields.items():
        torch.testing.assert_close(unpacked[name], expected, rtol=1e-5, atol=1e-12)

    initialization = np.array(["2024-01-01T00"] * steps, dtype="datetime64[h]")
    leads = np.array([6.0, 12.0])
    valid_times = initialization + leads.astype("timedelta64[h]")
    dataset = build_refined_dataset(
        packing=packing,
        deterministic=aurora_physical,
        refined=refined_physical,
        residual=predicted_correction_normalized,
        truth=cams_physical,
        init_time=initialization,
        valid_time=valid_times,
        lead_time_hours=leads,
    )
    output_path = tmp_path / "oracle_refined.nc"
    write_refined_netcdf(dataset, output_path, compression=False)

    with xr.open_dataset(output_path) as reopened:
        for variable in ("gtco3", "go3"):
            selected = resolve_forecast_variable(reopened, variable)
            assert selected.attrs["aurora_refinement_role"] == "refined_forecast"
            np.testing.assert_allclose(
                selected.values,
                cams_fields[variable].numpy(),
                rtol=1e-5,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                reopened[f"{variable}_predicted_correction_physical"].values,
                (cams_fields[variable] - aurora_fields[variable]).numpy(),
                rtol=1e-5,
                atol=1e-12,
            )


def test_zero_and_opposite_sign_corrections_have_explicit_semantics() -> None:
    packing = build_packing(height=4, width=5)
    target_space = NormalizedTargetSpace(packing)
    aurora_fields, cams_fields = _physical_fields(1, 4, 5)
    aurora_physical = packing.pack(aurora_fields)
    cams_physical = packing.pack(cams_fields)
    aurora_normalized = target_space.encode(aurora_physical)
    cams_normalized = target_space.encode(cams_physical)

    _, unchanged = target_space.apply_correction(
        aurora_normalized,
        torch.zeros_like(aurora_normalized),
        apply_constraints=False,
    )
    torch.testing.assert_close(unchanged, aurora_physical, rtol=1e-5, atol=1e-12)

    cams_minus_aurora = cams_normalized - aurora_normalized
    _, added = target_space.apply_correction(
        aurora_normalized,
        cams_minus_aurora,
        apply_constraints=False,
    )
    torch.testing.assert_close(added, cams_physical, rtol=1e-5, atol=1e-12)

    aurora_minus_cams = aurora_normalized - cams_normalized
    _, subtracted = target_space.apply_correction(
        aurora_normalized,
        -aurora_minus_cams,
        apply_constraints=False,
    )
    torch.testing.assert_close(subtracted, cams_physical, rtol=1e-5, atol=1e-12)
