"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from finetune.refinement import benchmark as bench
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace


def _surface_packing() -> FieldPacking:
    return FieldPacking(
        channels=(
            ChannelSpec(
                index=0,
                aurora_name="tcno2",
                dataset_name="tcno2",
                kind="surf",
                level=None,
                level_index=None,
                mean=0.0,
                std=1.0,
            ),
        ),
        lat=(45.0, 35.0),
        lon=(240.0, 241.0),
        lon_periodic=False,
    )


def _synthetic_dataset(*, initializations: int = 8, leads=(12.0, 24.0)):
    packing = _surface_packing()
    case = replace(
        bench.CASES["no2_uswest"],
        surf_variables=("tcno2",),
        atmos_variables=(),
        pressure_levels=(),
    )
    init_ids: list[str] = []
    init_times: list[np.datetime64] = []
    valid_times: list[np.datetime64] = []
    lead_values: list[float] = []
    lead_index: list[int] = []
    base = np.datetime64("2024-07-01T00:00:00", "ns")
    for group in range(initializations):
        init = base + np.timedelta64(12 * group, "h")
        init_id = np.datetime_as_string(init, unit="s")
        for index, lead in enumerate(leads):
            init_ids.append(init_id)
            init_times.append(init)
            valid_times.append(init + np.timedelta64(int(lead), "h"))
            lead_values.append(float(lead))
            lead_index.append(index)
    samples = len(init_ids)
    rollout = torch.zeros(samples, 1, 2, 2)
    target = rollout + 0.1
    return bench.BenchmarkDataset(
        packing=packing,
        rollout=rollout,
        target=target,
        valid=torch.ones_like(rollout, dtype=torch.bool),
        lead_hours=torch.tensor(lead_values),
        lead_index=torch.tensor(lead_index),
        lat=np.asarray(packing.lat),
        lon=np.asarray(packing.lon),
        case=case,
        initialization_ids=tuple(init_ids),
        initialization_times=np.asarray(init_times, dtype="datetime64[ns]"),
        valid_times=np.asarray(valid_times, dtype="datetime64[ns]"),
        source_validation={
            "declared_source_kind": "raw_aurora",
            "source_model_stage": "pretrained",
            "baseline_label": "raw_pretrained_aurora",
        },
    )


def test_cases_use_raw_pretrained_source_and_configured_loss_levels() -> None:
    assert bench.CASES["o3_global"].rollout_dir == "examples/outputs/cams_rollouts"
    assert bench.CASES["o3_global"].pressure_levels == (1000.0, 500.0, 100.0, 50.0)
    assert bench.CASES["no2_uswest"].rollout_dir == "examples/outputs/cams_rollouts"
    assert bench.CASES["no2_uswest"].pressure_levels == (1000.0, 925.0, 850.0)
    assert bench.CASES["no2_uswest"].source_model_stage == "pretrained"


def test_override_merge_does_not_mutate_shared_preset() -> None:
    before = copy.deepcopy(bench.LEGACY_PRESET)
    merged: dict[str, object] = {}
    bench._deep_update(merged, bench.LEGACY_PRESET)
    bench._deep_update(merged, {"loss": {"bias_weight": 99.0}})
    assert bench.LEGACY_PRESET == before


def test_refined_manifest_is_rejected_as_raw_baseline(tmp_path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    (rollout_dir / "run_manifest.json").write_text(
        json.dumps({"config": {"model": {"refinement": {"enabled": True}}}})
    )
    case = replace(
        bench.CASES["no2_uswest"],
        rollout_dir=str(rollout_dir),
        truth_path=str(tmp_path / "unused.nc"),
    )
    with pytest.raises(ValueError, match="refined-model product.*raw_aurora"):
        bench.load_case(case, verbose=False)


def test_load_case_preserves_init_valid_time_and_level_identity(tmp_path) -> None:
    xr = pytest.importorskip("xarray")
    rollout_dir = tmp_path / "raw"
    rollout_dir.mkdir()
    init = np.datetime64("2024-07-01T00:00:00", "ns")
    times = init + np.arange(1, 7) * np.timedelta64(12, "h")
    lat = np.asarray([40.0, 39.6])
    lon = np.asarray([240.0, 240.4])
    roll_levels = np.asarray([1000.0, 925.0, 850.0])
    truth_levels = roll_levels[::-1]
    shape = (len(times), len(lat), len(lon))
    rollout = xr.Dataset(
        {
            "tcno2": (("time", "latitude", "longitude"), np.full(shape, 1e-5)),
            "no2": (
                ("time", "level", "latitude", "longitude"),
                np.stack([np.full(shape, level * 1e-9) for level in roll_levels], axis=1),
            ),
        },
        coords={"time": times, "level": roll_levels, "latitude": lat, "longitude": lon},
        attrs={"initialization_time": str(init), "step_hours": 12},
    )
    rollout_path = rollout_dir / "rollout_20240701_000000.nc"
    rollout.to_netcdf(rollout_path)

    truth = xr.Dataset(
        {
            "tcno2": (("time", "latitude", "longitude"), np.full(shape, 2e-5)),
            "no2": (
                ("time", "level", "latitude", "longitude"),
                np.stack([np.full(shape, level * 2e-9) for level in truth_levels], axis=1),
            ),
        },
        coords={"time": times, "level": truth_levels, "latitude": lat, "longitude": lon},
    )
    truth_path = tmp_path / "truth.nc"
    truth.to_netcdf(truth_path)
    case = replace(
        bench.CASES["no2_uswest"],
        rollout_dir=str(rollout_dir),
        truth_path=str(truth_path),
    )

    data = bench.load_case(case, max_initializations=1, verbose=False)
    assert len(data) == 6
    assert set(data.initialization_ids) == {"2024-07-01T00:00:00"}
    np.testing.assert_array_equal(data.valid_times, times)
    assert [spec.level for spec in data.packing.channels] == [None, 1000.0, 925.0, 850.0]
    decoded = NormalizedTargetSpace(data.packing).decode(data.target)
    for index, level in enumerate(case.pressure_levels, start=1):
        torch.testing.assert_close(
            decoded[:, index], torch.full_like(decoded[:, index], level * 2e-9)
        )
    assert data.source_files == (str(rollout_path.resolve()),)
    assert data.source_validation["baseline_label"] == "raw_pretrained_aurora"


def test_grouped_purged_split_has_no_shared_init_or_valid_time() -> None:
    data = _synthetic_dataset()
    train, test, info = bench.grouped_purged_split(data, test_fraction=0.25)
    assert set(train.initialization_ids).isdisjoint(test.initialization_ids)
    assert set(train.valid_times).isdisjoint(test.valid_times)
    assert info["purge_hours"] == 24.0
    assert info["purged_initialization_ids"]
    assert info["shared_initialization_ids"] == []
    assert info["shared_valid_times"] == []


def test_weighted_correlation_uses_one_area_weight() -> None:
    pred = torch.tensor([[[[0.0, 2.0], [4.0, 9.0]]]], dtype=torch.float64)
    obs = torch.tensor([[[[1.0, 3.0], [8.0, 2.0]]]], dtype=torch.float64)
    valid = torch.ones_like(pred, dtype=torch.bool)
    area = torch.tensor([[[[1.0], [0.25]]]], dtype=torch.float64)
    row = bench.evaluate_fields(pred, obs, valid, area_weight=area, label="one-channel")

    weights = np.asarray([1.0, 1.0, 0.25, 0.25])
    p = pred.numpy().reshape(-1)
    o = obs.numpy().reshape(-1)
    p_mean = np.average(p, weights=weights)
    o_mean = np.average(o, weights=weights)
    expected = np.sum(weights * (p - p_mean) * (o - o_mean)) / np.sqrt(
        np.sum(weights * (p - p_mean) ** 2) * np.sum(weights * (o - o_mean) ** 2)
    )
    assert row.correlation == pytest.approx(expected)


def test_training_correction_statistics_are_physical_cams_minus_aurora() -> None:
    data = _synthetic_dataset(initializations=3, leads=(12.0, 24.0))
    rows = bench.correction_statistics(data)

    assert len(rows) == 1
    row = rows[0]
    assert row["variable"] == "tcno2"
    assert row["level_hpa"] is None
    assert row["count"] == 3 * 2 * 2 * 2
    assert row["mean"] == pytest.approx(0.1)
    assert row["std"] == pytest.approx(0.0)
    assert row["q01"] == pytest.approx(0.1)
    assert row["q50"] == pytest.approx(0.1)
    assert row["q99"] == pytest.approx(0.1)
    assert row["normalization_space"] == "physical_cams_minus_aurora_correction"
    formatted = bench.format_correction_statistics(rows)
    assert "tcno2" in formatted and "surface" in formatted


class _RecordingScaleFitRefiner:
    def __init__(self) -> None:
        self.correction: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.frozen = False

    def fit_residual_scale(self, correction: torch.Tensor, mask: torch.Tensor) -> None:
        self.correction = correction.detach().clone()
        self.mask = mask.detach().clone()

    def freeze_residual_scale(self) -> None:
        self.frozen = True


def test_scaler_fit_uses_every_grouped_training_sample() -> None:
    data = _synthetic_dataset(initializations=40)
    sample_codes = torch.arange(len(data), dtype=data.target.dtype).view(-1, 1, 1, 1)
    data.target.copy_(data.rollout + sample_codes)
    data.valid[7, :, 0, 0] = False
    refiner = _RecordingScaleFitRefiner()

    bench._fit_training_correction_scaler(refiner, data)

    assert refiner.correction is not None and refiner.mask is not None
    assert refiner.correction.shape[0] == 80
    expected = torch.where(data.valid, data.target - data.rollout, torch.zeros_like(data.target))
    torch.testing.assert_close(refiner.correction, expected)
    torch.testing.assert_close(
        refiner.correction[-1], torch.full_like(refiner.correction[-1], 79.0)
    )
    torch.testing.assert_close(refiner.mask, data.valid)
    assert refiner.frozen


@pytest.mark.parametrize("name", sorted(bench.LOSS_ABLATIONS))
def test_each_loss_ablation_starts_with_every_auxiliary_weight_zero(name: str) -> None:
    override = bench.isolated_ablation_overrides(name)
    explicitly_enabled = set(bench.LOSS_ABLATIONS[name].get("loss", {}))
    for key in bench.AUXILIARY_LOSS_WEIGHT_KEYS:
        if key not in explicitly_enabled:
            assert override["loss"][key] == 0.0


class _DirectRegressionNet(torch.nn.Module):
    def forward(self, noisy, conditioning, process_time, forecast_lead_time):
        del conditioning, process_time, forecast_lead_time
        return torch.full_like(noisy, 0.1)


class _IdentityResidualScaler:
    def decode(self, value):
        return value


class _DirectRegressionRefiner:
    def __init__(self) -> None:
        self.net = _DirectRegressionNet()
        self.residual_scaler = _IdentityResidualScaler()


class _RecordingRefiner:
    def __init__(self) -> None:
        self.deterministic_calls = 0
        self.generator_ids: list[int] = []

    def deterministic_residual(self, conditioning, **kwargs):
        self.deterministic_calls += 1
        return torch.zeros_like(conditioning[:, :1])

    def sample_residual(self, conditioning, *, generator, **kwargs):
        self.generator_ids.append(id(generator))
        return torch.randn(
            conditioning.shape[0],
            1,
            *conditioning.shape[-2:],
            generator=generator,
            device=conditioning.device,
        )


def test_direct_regression_transformer_predicts_a_physical_correction() -> None:
    data = _synthetic_dataset(initializations=2, leads=(12.0,))
    space = NormalizedTargetSpace(data.packing)
    config = bench.build_config(
        "diffusion_transformer",
        lon_periodic=False,
        overrides={"deterministic_inference": True},
    )
    refined, diagnostics = bench._predict_refined(
        _DirectRegressionRefiner(),
        config,
        data,
        space,
        batch_size=1,
        device=torch.device("cpu"),
        seed=7,
        ensemble_size=1,
        direct_regression=True,
    )

    torch.testing.assert_close(refined, space.decode(data.target))
    assert diagnostics["inference_mode"] == "direct_regression"
    assert diagnostics["evaluated_ensemble_size"] == 0


def test_prediction_honors_deterministic_mode_and_persists_member_rng_streams() -> None:
    data = _synthetic_dataset(initializations=2, leads=(12.0,))
    space = NormalizedTargetSpace(data.packing)
    deterministic = bench.build_config(
        "diffusion_unet",
        lon_periodic=False,
        overrides={"deterministic_inference": True},
    )
    refiner = _RecordingRefiner()
    _, diagnostics = bench._predict_refined(
        refiner,
        deterministic,
        data,
        space,
        batch_size=1,
        device=torch.device("cpu"),
        seed=7,
        ensemble_size=2,
    )
    assert refiner.deterministic_calls == 2
    assert refiner.generator_ids == []
    assert diagnostics["inference_mode"] == "deterministic_correction"

    stochastic = bench.build_config(
        "diffusion_unet",
        lon_periodic=False,
        overrides={"deterministic_inference": False, "ensemble_size": 2},
    )
    refiner = _RecordingRefiner()
    _, diagnostics = bench._predict_refined(
        refiner,
        stochastic,
        data,
        space,
        batch_size=1,
        device=torch.device("cpu"),
        seed=7,
        ensemble_size=None,
    )
    assert refiner.generator_ids[0] == refiner.generator_ids[2]
    assert refiner.generator_ids[1] == refiner.generator_ids[3]
    assert refiner.generator_ids[0] != refiner.generator_ids[1]
    assert diagnostics["rng_stream_lifetime"] == "whole_test_evaluation"


def test_report_is_per_channel_and_contains_complete_provenance(monkeypatch) -> None:
    data = _synthetic_dataset()

    def fake_train(head, train, test, **kwargs):
        row = bench.MetricRow(
            label=head,
            mae=0.1,
            rmse=0.1,
            bias=0.0,
            correlation=1.0,
            std_ratio=1.0,
            p90_bias=0.0,
            p95_bias=0.0,
            p99_bias=0.0,
            tail_mae=0.1,
        )
        return bench.HeadResult(
            head=head,
            channel_metrics={"tcno2": row},
            train_seconds=0.0,
            parameters=1,
            final_loss=0.1,
            per_lead={"tcno2": {12.0: row, 24.0: row}},
            resolved_config={"type": head},
        )

    monkeypatch.setattr(bench, "train_and_evaluate", fake_train)
    report = bench.run_benchmark(
        "no2_uswest",
        heads=("diffusion_unet",),
        data=data,
        epochs=1,
        device="cpu",
        seed=19,
        verbose=False,
    )
    assert report["metric_scope"] == "physical_units_per_variable_and_pressure_level_only"
    assert set(report["baseline"]["channels"]) == {"tcno2"}
    assert "mae" not in report["baseline"]
    provenance = report["provenance"]
    assert set(provenance) == {
        "args",
        "seed",
        "source",
        "split",
        "resolved_config",
        "git",
        "environment",
    }
    assert provenance["seed"] == 19
    assert provenance["resolved_config"]["diffusion_unet"]["type"] == "diffusion_unet"
    assert provenance["source"]["variables_and_levels"] == [
        {
            "channel_index": 0,
            "variable": "tcno2",
            "kind": "surf",
            "pressure_level_hpa": None,
        }
    ]
    json.dumps(report, allow_nan=False)
