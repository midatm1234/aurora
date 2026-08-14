"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Focused contracts for distributed residual calibration and validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr

import finetune.aurora_finetune_distributed as trainer
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.residual_scaling import ResidualScaler


def _packing(*, leads=(6.0, 12.0)) -> FieldPacking:
    return FieldPacking(
        channels=(
            ChannelSpec(
                index=0,
                aurora_name="tcno2",
                dataset_name="tcno2",
                kind="surf",
                level=None,
                level_index=None,
                units="kg m-2",
                std=1.0e-6,
            ),
            ChannelSpec(
                index=1,
                aurora_name="no2",
                dataset_name="no2",
                kind="atmos",
                level=1000.0,
                level_index=0,
                units="kg kg-1",
                std=1.0e-9,
            ),
        ),
        lat=(51.0,),
        lon=(-125.0,),
        lead_times_hours=tuple(leads),
    )


def _config() -> dict:
    return {
        "paths": {"train_data_path": "/tmp/example-train.nc"},
        "data": {
            "time_dim": "time",
            "lat_dim": "latitude",
            "lon_dim": "longitude",
            "level_dim": "level",
            "target_lead_times": [1, 2],
        },
        "rollout": {"rollout_step_hours": 6.0},
        "training": {"batch_size": 2},
    }


def _dataset() -> xr.Dataset:
    return xr.Dataset(
        coords={
            "time": np.arange(9),
            "latitude": [51.0],
            "longitude": [-125.0],
            "level": [1000.0],
        }
    )


def _samples() -> list[dict]:
    return [
        {
            "anchor_index": index,
            "history_indices": [index],
            "target_indices": {1: index + 1, 2: index + 2},
        }
        for index in range(5)
    ]


class _PackedScalerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packing = _packing()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.residual_scaler = ResidualScaler(
            self.packing.num_channels,
            mode="per_channel",
            center=True,
        )


def test_calibration_shards_are_disjoint_unpadded_and_complete() -> None:
    samples = _samples()
    shards = [
        trainer._disjoint_rank_samples(samples, rank, 3)
        for rank in range(3)
    ]
    anchors = [sample["anchor_index"] for shard in shards for sample in shard]
    assert sorted(anchors) == list(range(5))
    assert len(anchors) == len(set(anchors)) == len(samples)
    assert [len(shard) for shard in shards] == [2, 2, 1]


def test_training_split_fingerprint_covers_samples_coordinates_and_packing() -> None:
    cfg = _config()
    ds = _dataset()
    samples = _samples()
    packing = _packing()
    expected = trainer._training_split_calibration_fingerprint(
        ds, samples, cfg, packing
    )
    assert len(expected) == 32
    assert expected == trainer._training_split_calibration_fingerprint(
        ds, list(samples), cfg, packing
    )
    assert expected != trainer._training_split_calibration_fingerprint(
        ds, list(reversed(samples)), cfg, packing
    )
    shifted = ds.assign_coords(time=ds.time.values + 1)
    assert expected != trainer._training_split_calibration_fingerprint(
        shifted, samples, cfg, packing
    )


def test_production_prefit_observes_every_packed_example_once(monkeypatch) -> None:
    model = _PackedScalerModel()
    samples = _samples()
    seen_anchors = []

    def fake_compute_supervised_loss(*, model, samples, **kwargs):
        del kwargs
        assert model.training
        assert not model.dropout.training
        assert model.residual_scaler.training
        batch = list(samples)
        seen_anchors.extend(sample["anchor_index"] for sample in batch)
        values = []
        for lead in (1, 2):
            for sample in batch:
                anchor = float(sample["anchor_index"])
                values.append([anchor + lead, 10.0 * anchor + lead])
        residual = torch.tensor(values, dtype=torch.float32).view(-1, 2, 1, 1)
        model.residual_scaler.observe(
            residual,
            torch.ones_like(residual, dtype=torch.bool),
        )
        return torch.zeros((), dtype=torch.float32), {}

    monkeypatch.setattr(trainer.ft, "compute_supervised_loss", fake_compute_supervised_loss)
    report = trainer._calibrate_residual_scalers_from_training_split(
        model=model,
        train_ds=_dataset(),
        train_samples=samples,
        cfg=_config(),
        resolved_specs=SimpleNamespace(targets=()),
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        norm_stats=None,
        global_step=0,
    )

    assert seen_anchors == list(range(5))
    assert model.residual_scaler.has_exact_training_split_calibration
    assert int(model.residual_scaler.calibration_examples.item()) == 10
    torch.testing.assert_close(
        model.residual_scaler.calibration_count.flatten(),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
    )
    assert report is not None
    assert report["complete"] is True
    assert report["logical_samples"] == 5
    assert report["packed_examples"] == 10
    assert report["correction_convention"] == "CAMS_minus_Aurora"
    channels = report["scalers"][0]["channels"]
    assert channels[0]["physical_correction_std"] > channels[1]["physical_correction_std"]


def test_trained_resume_rejects_non_exact_scaler_without_recalibrating(monkeypatch) -> None:
    model = _PackedScalerModel()
    model.residual_scaler.fit(torch.randn(10, 2, 1, 1))
    called = False

    def forbidden_compute(**kwargs):
        del kwargs
        nonlocal called
        called = True
        raise AssertionError("resume must not refit a learned coordinate system")

    monkeypatch.setattr(trainer.ft, "compute_supervised_loss", forbidden_compute)
    try:
        trainer._calibrate_residual_scalers_from_training_split(
            model=model,
            train_ds=_dataset(),
            train_samples=_samples(),
            cfg=_config(),
            resolved_specs=SimpleNamespace(targets=()),
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            norm_stats=None,
            global_step=1,
        )
    except RuntimeError as exc:
        assert "Cannot resume trained stochastic-refinement weights" in str(exc)
    else:
        raise AssertionError("non-exact trained scaler resume was accepted")
    assert called is False


def test_expected_validation_channels_cover_variable_level_and_lead_product() -> None:
    channels = (
        ChannelSpec(0, "tcno2", "tcno2", "surf", None, None),
        ChannelSpec(1, "no2", "no2", "atmos", 1000.0, 0),
        ChannelSpec(2, "no2", "no2", "atmos", 925.0, 1),
        ChannelSpec(3, "no2", "no2", "atmos", 850.0, 2),
    )
    packing = FieldPacking(channels=channels, lead_times_hours=(24.0, 48.0, 72.0))
    model = SimpleNamespace(packing=packing)
    actual = trainer._expected_physical_validation_channels(
        model,
        _config(),
        SimpleNamespace(targets=()),
    )
    expected = {
        f"{variable}@{level}@lead{lead}h"
        for variable, level in (
            ("tcno2", "surface"),
            ("no2", "1000hPa"),
            ("no2", "925hPa"),
            ("no2", "850hPa"),
        )
        for lead in (24, 48, 72)
    }
    assert set(actual) == expected


def test_validation_coverage_rejects_missing_extra_and_zero_count_groups() -> None:
    expected = {"tcno2@surface@lead24h", "no2@1000hPa@lead24h"}
    physical = {
        "tcno2@surface@lead24h": {
            "refined": {"count": 100.0},
            "baseline": {"count": 100.0},
        },
        "no2@1000hPa@lead24h": {
            "refined": {"count": 0.0},
            "baseline": {"count": 100.0},
        },
    }
    result = trainer._summarize_physical_channel_coverage(
        expected_channels=expected,
        observed_refined={"tcno2@surface@lead24h", "unexpected"},
        observed_baseline=expected,
        physical_channels=physical,
    )
    assert result["complete"] is False
    assert result["missing_refined"] == ["no2@1000hPa@lead24h"]
    assert result["extra_refined"] == ["unexpected"]
    assert result["zero_count_refined"] == ["no2@1000hPa@lead24h"]
    assert not trainer._checkpoint_candidate_improves(
        should_validate=True,
        checkpoint_metric_value=0.5,
        non_degrading=True,
        physical_channel_coverage_complete=bool(result["complete"]),
        best_value=1.0,
        min_delta=0.0,
    )

    complete = trainer._summarize_physical_channel_coverage(
        expected_channels=expected,
        observed_refined=expected,
        observed_baseline=expected,
        physical_channels={
            name: {
                "refined": {"count": 1.0},
                "baseline": {"count": 1.0},
            }
            for name in expected
        },
    )
    assert complete["complete"] is True
    assert trainer._checkpoint_candidate_improves(
        should_validate=True,
        checkpoint_metric_value=0.5,
        non_degrading=True,
        physical_channel_coverage_complete=bool(complete["complete"]),
        best_value=1.0,
        min_delta=0.0,
    )
