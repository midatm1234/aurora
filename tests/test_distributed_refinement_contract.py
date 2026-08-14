"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Focused contracts for distributed residual calibration and validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import finetune.aurora_finetune_distributed as trainer
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.residual_scaling import ResidualScaler


ROOT = Path(__file__).resolve().parents[1]
NO2_CONFIGS = (
    "aurora_NO2_finetune_US-WEST_3day_lead_config.yaml",
    "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_config.yaml",
    "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml",
    "aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml",
)


def _raw_no2_config(name: str = NO2_CONFIGS[2]) -> dict:
    config = yaml.safe_load((ROOT / "finetune" / name).read_text())
    assert isinstance(config, dict)
    return config


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


def test_spatial_pattern_correlation_is_per_sample_masked_and_scale_independent() -> None:
    truth = 1.0e-10 * torch.tensor(
        [
            [[[0.0, 1.0, 2.0, 3.0]]],
            [[[0.0, 1.0, 2.0, 3.0]]],
            [[[1.0, 1.0, 1.0, 1.0]]],
        ]
    )
    prediction = 1.0e-10 * torch.tensor(
        [
            [[[0.0, 2.0, 4.0, 6.0]]],
            [[[3.0, 2.0, 1.0, 0.0]]],
            [[[2.0, 2.0, 2.0, 2.0]]],
        ]
    )
    valid = torch.ones_like(truth, dtype=torch.bool)
    prediction[0, ..., -1] = float("inf")
    valid[0, ..., -1] = False

    correlation_sum, correlation_count = (
        trainer.ft._spatial_pattern_correlation_sum_count(
            prediction,
            truth,
            valid,
        )
    )

    assert correlation_count == 2.0
    assert correlation_sum == pytest.approx(0.0, abs=1.0e-12)


def test_physical_validation_merge_and_finalize_include_spatial_correlation() -> None:
    sums: dict[str, dict[str, float]] = {}
    trainer._merge_physical_validation_sums(
        sums,
        {
            "physical_error_sums": {
                "tcno2@surface@lead24h": {
                    "count": 2.0,
                    "error_sum": 2.0,
                    "abs_error_sum": 2.0,
                    "sq_error_sum": 2.0,
                    "spatial_correlation_sum": 1.5,
                    "spatial_correlation_count": 2.0,
                }
            }
        },
    )
    # Old batch/checkpoint producers have no correlation keys; merging them
    # remains valid and does not dilute the sample-correlation mean.
    trainer._merge_physical_validation_sums(
        sums,
        {
            "physical_error_sums": {
                "tcno2@surface@lead24h": {
                    "count": 2.0,
                    "error_sum": 0.0,
                    "abs_error_sum": 2.0,
                    "sq_error_sum": 2.0,
                }
            }
        },
    )

    result = trainer._finalize_physical_validation_channel(
        sums["tcno2@surface@lead24h"]
    )

    assert result["count"] == 4.0
    assert result["bias"] == pytest.approx(0.5)
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(1.0)
    assert result["pattern_correlation_count"] == 2.0
    assert result["pattern_correlation"] == pytest.approx(0.75)


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


def _guard_channels(**candidate_overrides) -> dict:
    candidate = {
        "count": 100.0,
        "bias": 0.1,
        "mae": 1.01,
        "rmse": 1.8,
        "pattern_correlation": 0.79,
    }
    candidate.update(candidate_overrides)
    return {
        "tcno2@surface@lead24h": {
            "baseline": {
                "count": 100.0,
                "bias": 0.001,
                "mae": 1.0,
                "rmse": 2.0,
                "pattern_correlation": 0.8,
            },
            "refined": candidate,
        }
    }


def _evaluate_guards(channels: dict) -> dict[str, object]:
    return trainer._evaluate_checkpoint_guards(
        physical_channels=channels,
        expected_channels={"tcno2@surface@lead24h"},
        metrics=("mae", "absolute_bias", "pattern_correlation"),
        relative_tolerance=0.01,
        correlation_tolerance=0.01,
        bias_rmse_floor_fraction=0.05,
    )


def test_checkpoint_guards_apply_tolerances_and_bias_rmse_floor() -> None:
    passing = _evaluate_guards(_guard_channels())
    assert passing["status"] == "passed"
    assert passing["passed"] is True
    bias_detail = passing["channels"]["tcno2@surface@lead24h"][
        "absolute_bias"
    ]
    assert bias_detail["relative_threshold"] == pytest.approx(0.00101)
    assert bias_detail["rmse_floor_threshold"] == pytest.approx(0.1)
    assert bias_detail["threshold"] == pytest.approx(0.1)

    failures = {
        "mae": _guard_channels(mae=1.011),
        "absolute_bias": _guard_channels(bias=0.101),
        "pattern_correlation": _guard_channels(pattern_correlation=0.789),
    }
    for expected_metric, channels in failures.items():
        report = _evaluate_guards(channels)
        assert report["status"] == "failed"
        assert report["passed"] is False
        assert report["failed_channels"] == ["tcno2@surface@lead24h"]
        failed_metrics = {item["metric"] for item in report["failures"]}
        assert expected_metric in failed_metrics


def test_checkpoint_guards_are_opt_in_and_veto_promotion_when_enabled() -> None:
    disabled = trainer._evaluate_checkpoint_guards(
        physical_channels=_guard_channels(mae=100.0),
        expected_channels={"tcno2@surface@lead24h"},
        metrics=(),
        relative_tolerance=0.0,
        correlation_tolerance=0.0,
        bias_rmse_floor_fraction=0.0,
    )
    assert disabled["status"] == "disabled"
    assert disabled["passed"] is True
    assert not trainer._checkpoint_candidate_improves(
        should_validate=True,
        checkpoint_metric_value=0.5,
        non_degrading=True,
        physical_channel_coverage_complete=True,
        checkpoint_guards_passed=False,
        best_value=1.0,
        min_delta=0.0,
    )


def test_checkpoint_guard_detail_is_structured_in_metadata_and_history(
    tmp_path: Path,
) -> None:
    report = _evaluate_guards(_guard_channels(mae=1.011))
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "best.ckpt"
    trainer.ft.save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        None,
        epoch=1,
        global_step=2,
        best_val_loss=0.9,
        config={
            "model": {},
            "training": {
                "checkpoint_guard_metrics": [
                    "mae",
                    "absolute_bias",
                    "pattern_correlation",
                ]
            },
            "runtime": {"training_run_id": "guard-test"},
        },
        validation={
            "status": "checkpoint_guard_failed",
            "checkpoint_guards": report,
        },
        validated_for_inference=False,
    )
    metadata = json.loads(
        checkpoint_path.with_suffix(".ckpt.metadata.json").read_text()
    )
    stored_report = metadata["validation"]["checkpoint_guards"]
    assert stored_report["status"] == "failed"
    assert stored_report["failures"][0]["channel"] == (
        "tcno2@surface@lead24h"
    )

    trainer.ft.write_training_history(
        [{"epoch": 1, "checkpoint_guards": report}],
        tmp_path / "history",
    )
    history = json.loads(
        (tmp_path / "history" / "training_history.json").read_text()
    )
    assert history[0]["checkpoint_guards"]["status"] == "failed"


@pytest.mark.parametrize("name", NO2_CONFIGS)
def test_no2_configs_enable_all_scientific_checkpoint_guards(name: str) -> None:
    config = _raw_no2_config(name)
    trainer.ft.validate_config(config, ROOT / "finetune" / name)
    training = config["training"]
    assert training["checkpoint_guard_metrics"] == [
        "mae",
        "absolute_bias",
        "pattern_correlation",
    ]
    assert training["checkpoint_guard_relative_tolerance"] == pytest.approx(0.0)
    assert training["checkpoint_guard_correlation_tolerance"] == pytest.approx(0.0)
    assert training["checkpoint_guard_bias_rmse_floor_fraction"] == pytest.approx(0.05)


def test_checkpoint_guard_config_defaults_are_backward_compatible() -> None:
    config = _raw_no2_config()
    training = config["training"]
    for key in (
        "checkpoint_guard_metrics",
        "checkpoint_guard_relative_tolerance",
        "checkpoint_guard_correlation_tolerance",
        "checkpoint_guard_bias_rmse_floor_fraction",
    ):
        training.pop(key, None)

    trainer.ft.validate_config(config)

    assert training["checkpoint_guard_metrics"] == []
    assert training["checkpoint_guard_relative_tolerance"] == 0.0
    assert training["checkpoint_guard_correlation_tolerance"] == 0.0
    assert training["checkpoint_guard_bias_rmse_floor_fraction"] == 0.0


@pytest.mark.parametrize(
    "value",
    ["mae", ["rmse"], ["mae", "mae"], [1]],
)
def test_checkpoint_guard_metric_list_is_strict(value) -> None:
    config = _raw_no2_config()
    config["training"]["checkpoint_guard_metrics"] = value
    with pytest.raises(ValueError, match="checkpoint_guard_metrics"):
        trainer.ft.validate_config(config)


@pytest.mark.parametrize(
    "key",
    [
        "checkpoint_guard_relative_tolerance",
        "checkpoint_guard_correlation_tolerance",
        "checkpoint_guard_bias_rmse_floor_fraction",
    ],
)
def test_checkpoint_guard_tolerances_must_be_nonnegative(key: str) -> None:
    config = _raw_no2_config()
    config["training"][key] = -0.001
    with pytest.raises(ValueError, match=key):
        trainer.ft.validate_config(config)


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
