"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from finetune.mamba_temporal import PackedMambaTemporalAdapter
from finetune.refinement.benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    _channel_specs,
)
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.run_mamba_ablation_study import (
    _as_sequences,
    _three_way_split,
    _validation_skill_guard,
    _write_rollout_directory,
)


def _dataset(initializations: int = 12) -> BenchmarkDataset:
    case = BenchmarkCase(
        name="synthetic_temporal",
        rollout_dir="unused",
        truth_path="unused",
        surf_variables=("surface",),
        atmos_variables=("profile",),
        pressure_levels=(1000.0,),
        lon_periodic=False,
        expected_lead_hours=(12.0, 24.0, 36.0),
    )
    packing = FieldPacking(
        channels=(
            ChannelSpec(0, "surface", "surface", "surf", None, None),
            ChannelSpec(1, "profile", "profile", "atmos", 1000.0, 0),
        ),
        lat=(40.0, 41.0),
        lon=(-120.0, -119.0, -118.0),
        lead_times_hours=case.expected_lead_hours,
        lead_time_scale_hours=36.0,
        lon_periodic=False,
    )
    samples = initializations * 3
    rollout = torch.arange(samples * 2 * 2 * 3, dtype=torch.float32).reshape(samples, 2, 2, 3)
    rollout = rollout / 100.0
    target = rollout + 0.1
    valid = torch.ones_like(rollout, dtype=torch.bool)
    initialization_ids = []
    initialization_times = []
    valid_times = []
    lead_hours = []
    lead_index = []
    start = np.datetime64("2024-01-01T00:00:00", "ns")
    for initialization in range(initializations):
        init = start + np.timedelta64(initialization * 10, "D")
        name = np.datetime_as_string(init, unit="s")
        for step, lead in enumerate(case.expected_lead_hours):
            initialization_ids.append(name)
            initialization_times.append(init)
            valid_times.append(init + np.timedelta64(int(lead), "h"))
            lead_hours.append(lead)
            lead_index.append(step)
    return BenchmarkDataset(
        packing=packing,
        rollout=rollout,
        target=target,
        valid=valid,
        rollout_valid=valid.clone(),
        lead_hours=torch.tensor(lead_hours),
        lead_index=torch.tensor(lead_index),
        lat=np.asarray(packing.lat),
        lon=np.asarray(packing.lon),
        case=case,
        initialization_ids=tuple(initialization_ids),
        initialization_times=np.asarray(initialization_times, dtype="datetime64[ns]"),
        valid_times=np.asarray(valid_times, dtype="datetime64[ns]"),
    )


def test_o3_sparse_source_levels_use_contiguous_packed_indices() -> None:
    case = BenchmarkCase(
        name="o3",
        rollout_dir="unused",
        truth_path="unused",
        surf_variables=("gtco3",),
        atmos_variables=("go3",),
        pressure_levels=(1000.0, 500.0, 100.0, 50.0),
        lon_periodic=True,
    )
    full_levels = (
        1000.0,
        925.0,
        850.0,
        700.0,
        600.0,
        500.0,
        400.0,
        300.0,
        250.0,
        200.0,
        150.0,
        100.0,
        50.0,
    )
    specs, _ = _channel_specs(case, full_levels)
    assert [spec.level_index for spec in specs[1:]] == [0, 1, 2, 3]
    FieldPacking(channels=tuple(specs))


def test_sequence_builder_keeps_initializations_and_leads_distinct() -> None:
    data = _dataset(4)
    permutation = []
    for initialization in range(4):
        start = initialization * 3
        permutation.extend((start + 2, start, start + 1))
    shuffled = data.subset(permutation)
    base = shuffled.rollout + 0.25
    sequence = _as_sequences(shuffled, base)
    assert sequence.base.shape == (4, 3, 2, 2, 3)
    assert torch.equal(sequence.lead_hours[0], torch.tensor([12.0, 24.0, 36.0]))
    for row, initialization_id in enumerate(sequence.initialization_ids):
        assert all(
            shuffled.initialization_ids[index] == initialization_id
            for index in sequence.flat_indices[row * 3 : (row + 1) * 3]
        )


def test_three_way_split_is_grouped_chronological_and_purged() -> None:
    data = _dataset(12)
    train, validation, test, metadata = _three_way_split(
        data,
        validation_fraction=0.2,
        test_fraction=0.2,
        purge_hours=72.0,
    )
    groups = [
        set(train.initialization_ids),
        set(validation.initialization_ids),
        set(test.initialization_ids),
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert max(train.initialization_times) < min(validation.initialization_times)
    assert max(validation.initialization_times) < min(test.initialization_times)
    assert metadata["method"].startswith("nested_chronological")


def test_identity_temporal_model_fails_benefit_guard_and_reconstructs(tmp_path) -> None:
    data = _dataset(4)
    sequence = _as_sequences(data, data.rollout)
    model = PackedMambaTemporalAdapter(
        data.packing,
        channels=4,
        d_state=2,
        n_layers=1,
        d_conv=2,
        expand=1,
        mode="packed_joint",
        gated_fusion=True,
        gate_init=0.0,
        lead_time_conditioning=True,
        mask_conditioning=True,
    )
    corrected = model.corrected_sequence(
        sequence.base,
        lead_hours=sequence.lead_hours,
        valid_cell_mask=sequence.input_valid,
    )
    assert torch.equal(corrected, sequence.base)
    guard = _validation_skill_guard(model, sequence, device=torch.device("cpu"))
    assert guard["passed"] is False

    true_residual = sequence.target - sequence.rollout
    assert torch.allclose(sequence.rollout + true_residual, sequence.target)

    manifest = {
        "mode": "off",
        "spatial_head": "diffusion_unet",
        "seed": 3,
        "phase1_checkpoint_sha256": "a" * 64,
        "spatial_checkpoint_sha256": "b" * 64,
        "temporal_config": {"enabled": False, "semantic_version": 2},
        "sampling_configuration": {"deterministic_inference": True},
        "trajectory_policy": {
            "source": "cached_raw_rollout",
            "aurora_autoregressive_feedback": False,
        },
    }
    output = tmp_path / "rollouts"
    _write_rollout_directory(
        output,
        corrected,
        sequence,
        packing=data.packing,
        manifest=manifest,
    )
    files = sorted(output.glob("rollout_predictions_init_*.nc"))
    assert len(files) == 4
    with xr.open_dataset(files[0]) as dataset:
        assert tuple(dataset["surface"].dims) == ("time", "latitude", "longitude")
        assert tuple(dataset["profile"].dims) == (
            "time",
            "level",
            "latitude",
            "longitude",
        )
        assert dataset.attrs["mamba_ablation_mode"] == "off"
