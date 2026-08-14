"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import dataclasses
import math
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from torch import nn

from aurora import Batch, Metadata
from finetune.aurora_finetune_distributed import _optimizer_updates_per_epoch
from finetune.calibrate_refinement_gate import least_squares_scale
from finetune.aurora_finetune_utils import (
    ResolvedVariableSpecs,
    VariableSpec,
    _dataset_frames_for_predictor,
    _forecast_lead_hours_for_samples,
    maybe_wrap_flow_refine,
    run_rollout,
    select_refinement_checkpoint,
    validate_checkpoint_refinement_contract,
)
from finetune.flow_refine import (
    DEFAULT_AUX_LOSS_CONFIG,
    AuroraFlowRefine,
    _extreme_weighted_mse,
    _peak_loss,
)


class _RecordingHead(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.lead_calls: list[torch.Tensor | None] = []

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        doy: torch.Tensor | None = None,
        lead_hours: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cond, doy, coords
        self.calls.append((x_t.detach().clone(), t.detach().clone()))
        self.lead_calls.append(
            None if lead_hours is None else lead_hours.detach().clone()
        )
        return self.value.expand_as(x_t)


class _EchoHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        doy: torch.Tensor | None = None,
        lead_hours: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cond, doy, lead_hours, coords
        self.calls.append((x_t.detach().clone(), t.detach().clone()))
        return x_t


def _wrapper(
    *,
    sampling_steps: int = 1,
    lead_time_cond: bool = False,
    contract_version: int = 2,
) -> AuroraFlowRefine:
    model = AuroraFlowRefine(
        nn.Identity(),
        target_surf_vars=("x",),
        hidden=8,
        sampling_steps=sampling_steps,
        flow_refine_contract_version=contract_version,
        lead_time_cond=lead_time_cond,
        lead_time_scale_hours=72.0,
        lon_periodic=False,
    )
    return model


@pytest.mark.parametrize(
    ("contract_version", "expected_t"),
    [(1, 1.0), (2, 0.0), (3, 0.0)],
)
def test_single_step_sampler_uses_versioned_endpoint(
    contract_version: int,
    expected_t: float,
) -> None:
    model = _wrapper(sampling_steps=1, contract_version=contract_version)
    head = _RecordingHead(value=2.0)
    cond = torch.randn(2, 1, 4, 4)

    result = model._sample_residual(cond, head)

    assert torch.equal(result, torch.full_like(result, 2.0))
    assert len(head.calls) == 1
    x_t, t = head.calls[0]
    assert torch.count_nonzero(x_t) == 0
    assert torch.equal(t, torch.full_like(t, expected_t))


def test_legacy_builder_defaults_missing_contract_to_v1() -> None:
    specs = ResolvedVariableSpecs(
        predictors=(),
        targets=(VariableSpec("x", "x", "surf"),),
        static=(),
    )
    config = {
        "model": {
            "flow_refine_enabled": True,
            "flow_refine_hidden": 8,
            "flow_refine_time_dim": 128,
            "flow_refine_sampling_steps": 1,
            "lon_periodic": False,
        },
        "data": {},
    }

    model = maybe_wrap_flow_refine(nn.Identity(), config, specs)

    assert isinstance(model, AuroraFlowRefine)
    assert model.flow_refine_contract_version == 1
    head = _RecordingHead(value=0.0)
    model._sample_residual(torch.zeros(1, 1, 4, 4), head)
    _, t = head.calls[0]
    assert torch.equal(t, torch.ones_like(t))


def test_legacy_extreme_tail_defaults_to_both_and_can_focus_upper() -> None:
    target = torch.tensor([[[[-10.0, 0.0, 1.0, 2.0]]]])
    refined = target.clone()
    refined[..., 0] += 1.0

    both = _extreme_weighted_mse(
        refined, target, quantile=0.75, intensity=5.0,
    )
    upper = _extreme_weighted_mse(
        refined, target, quantile=0.75, intensity=5.0, tail="upper",
    )
    peak_both = _peak_loss(refined, target)
    peak_upper = _peak_loss(refined, target, tail="upper")

    assert DEFAULT_AUX_LOSS_CONFIG["extreme_tail"] == "both"
    assert float(both) > float(upper)
    assert float(peak_both) > 0.0
    assert float(peak_upper) == 0.0


def test_multistep_sampler_honors_generator() -> None:
    model = _wrapper(sampling_steps=2)
    cond = torch.zeros(1, 1, 4, 4)
    head_a = _EchoHead()
    head_b = _EchoHead()
    generator_a = torch.Generator().manual_seed(17)
    generator_b = torch.Generator().manual_seed(17)

    result_a = model._sample_residual(cond, head_a, generator=generator_a)
    result_b = model._sample_residual(cond, head_b, generator=generator_b)

    torch.testing.assert_close(result_a, result_b)
    assert len(head_a.calls) == len(head_b.calls) == 2
    torch.testing.assert_close(head_a.calls[0][0], head_b.calls[0][0])
    assert torch.equal(generator_a.get_state(), generator_b.get_state())


def test_flow_loss_trains_deterministic_query_and_degradation_hinge() -> None:
    model = _wrapper()
    head = _RecordingHead(value=3.0)
    model.surf_flow["x"] = head
    model.set_aux_loss_config(
        {
            "enabled": True,
            "deterministic_reconstruction_weight": 1.0,
            "degradation_weight": 1.0,
            "aux_on_deterministic": True,
        }
    )
    pred = torch.zeros(1, 4, 4)
    target = torch.ones_like(pred)

    loss = model.flow_loss(pred, target, "x", "surf")

    # Random-t x1 loss: (3-1)^2 = 4.
    # Deterministic reconstruction: (3-1)^2 = 4.
    # Degradation hinge: relu((3-1)^2 - 1^2) = 3.
    assert float(loss.detach()) == pytest.approx(11.0)
    assert any(
        torch.count_nonzero(x_t) == 0 and torch.count_nonzero(t) == 0
        for x_t, t in head.calls
    )


def test_flow_loss_respects_explicit_valid_mask() -> None:
    model = _wrapper()
    model.surf_flow["x"] = _RecordingHead(value=0.0)
    model.set_aux_loss_config({"enabled": False})
    pred = torch.zeros(1, 4, 4)
    target_a = torch.ones_like(pred)
    target_b = target_a.clone()
    target_b[0, 0, 0] = 1000.0
    valid = torch.ones_like(pred, dtype=torch.bool)
    valid[0, 0, 0] = False

    torch.manual_seed(7)
    loss_a = model.flow_loss(pred, target_a, "x", "surf", valid_mask=valid)
    torch.manual_seed(7)
    loss_b = model.flow_loss(pred, target_b, "x", "surf", valid_mask=valid)

    assert float(loss_a.detach()) == pytest.approx(float(loss_b.detach()))


def test_atmospheric_residual_zscore_is_per_level() -> None:
    model = AuroraFlowRefine(
        nn.Identity(),
        target_atmos_vars=("x",),
        hidden=8,
        sampling_steps=1,
        atmos_loss_levels={"x": [0, 1]},
        residual_zscore=True,
        lon_periodic=False,
    )
    model.atmos_flow["x"] = _RecordingHead(value=0.0)
    model.set_aux_loss_config({"enabled": False})
    pattern = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.stack([pattern, 10 * pattern], dim=1)

    model.flow_loss(pred, target, "x", "atmos")

    scale = model._residual_std(
        "atmos", "x", target - pred, update=False,
    )
    assert tuple(scale.shape) == (2,)
    assert float(scale[1] / scale[0]) == pytest.approx(10.0, rel=1.0e-5)


def test_scheduler_horizon_counts_optimizer_updates() -> None:
    assert _optimizer_updates_per_epoch(
        725,
        world_size=1,
        batch_size=1,
        accumulation_steps=32,
    ) == 23


def test_validation_gate_can_abstain_from_anticorrelated_correction() -> None:
    raw, selected = least_squares_scale(-2.0, 4.0)

    assert raw == pytest.approx(-0.5)
    assert selected == 0.0
    assert _optimizer_updates_per_epoch(
        725,
        world_size=4,
        batch_size=1,
        accumulation_steps=8,
    ) == 23


def test_checkpoint_contract_rejects_unvalidated_checkpoint() -> None:
    model = _wrapper()
    specs = ResolvedVariableSpecs(
        predictors=(),
        targets=(VariableSpec("x", "x", "surf"),),
        static=(),
    )
    config = {
        "data": {"target_variables": [{"dataset_name": "x", "kind": "surf"}]},
        "model": {
            "flow_refine_enabled": True,
            "flow_refine_hidden": 8,
            "flow_refine_time_dim": 128,
        },
        "inference": {"require_validated_checkpoint": True},
    }
    checkpoint = {
        "best_val_loss": math.inf,
        "config": {
            "data": {
                "target_variables": [
                    {"dataset_name": "x", "aurora_name": "x", "kind": "surf"}
                ]
            },
            "model": {
                "flow_refine_enabled": True,
                "flow_refine_hidden": 8,
                "flow_refine_time_dim": 128,
            },
        },
        "norm_stats": {
            "x": {"mean": torch.tensor([0.0]), "std": torch.tensor([1.0])}
        },
    }

    with pytest.raises(ValueError, match="no finite validation score"):
        validate_checkpoint_refinement_contract(
            model, checkpoint, config, specs,
        )


def _write_checkpoint_metadata(
    path: Path,
    *,
    run_id: str,
    best_val_loss: float,
) -> None:
    path.touch()
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "training_run_id": run_id,
                "best_val_loss": best_val_loss,
            }
        )
    )


def test_checkpoint_selection_rejects_stale_best(tmp_path: Path) -> None:
    _write_checkpoint_metadata(
        tmp_path / "best.ckpt", run_id="old", best_val_loss=0.5,
    )
    _write_checkpoint_metadata(
        tmp_path / "last.ckpt", run_id="current", best_val_loss=math.inf,
    )

    with pytest.raises(ValueError, match="no matching validated best"):
        select_refinement_checkpoint(tmp_path, require_validated=True)


def test_checkpoint_selection_uses_same_run_validated_best(
    tmp_path: Path,
) -> None:
    _write_checkpoint_metadata(
        tmp_path / "best.ckpt", run_id="current", best_val_loss=0.4,
    )
    _write_checkpoint_metadata(
        tmp_path / "last.ckpt", run_id="current", best_val_loss=0.4,
    )

    selected = select_refinement_checkpoint(tmp_path, require_validated=True)

    assert selected == tmp_path / "best.ckpt"



def test_lead_conditioning_is_required_and_repeated_by_atmospheric_level() -> None:
    model = AuroraFlowRefine(
        nn.Identity(),
        target_atmos_vars=("x",),
        hidden=8,
        sampling_steps=1,
        lead_time_cond=True,
        lead_time_scale_hours=72.0,
        lon_periodic=False,
    )
    head = _RecordingHead(value=0.0)
    model.atmos_flow["x"] = head
    model.set_aux_loss_config({"enabled": False})
    pred = torch.zeros(2, 3, 4, 4)
    target = torch.ones_like(pred)

    with pytest.raises(ValueError, match="requires forecast_lead_time_hours"):
        model.flow_loss(pred, target, "x", "atmos")
    with pytest.raises(ValueError, match="trained support"):
        model.flow_loss(
            pred,
            target,
            "x",
            "atmos",
            lead_time_hours=torch.tensor([12.0, 84.0]),
        )

    model.flow_loss(
        pred,
        target,
        "x",
        "atmos",
        lead_time_hours=torch.tensor([12.0, 48.0]),
    )

    assert len(head.lead_calls) == 1
    assert torch.equal(
        head.lead_calls[0],
        torch.tensor([12.0, 12.0, 12.0, 48.0, 48.0, 48.0]),
    )


def test_deterministic_refinement_receives_forecast_lead_hours() -> None:
    model = _wrapper(lead_time_cond=True)
    head = _RecordingHead(value=0.0)
    model.surf_flow["x"] = head

    model.refine_norm_deterministic(
        torch.zeros(2, 4, 4),
        "x",
        "surf",
        lead_time_hours=torch.tensor([24.0, 72.0]),
    )

    assert len(head.lead_calls) == 1
    assert torch.equal(head.lead_calls[0], torch.tensor([24.0, 72.0]))


def test_forecast_lead_hours_use_exact_dataset_valid_times() -> None:
    times = np.datetime64("2024-01-01T00") + np.arange(6) * np.timedelta64(12, "h")
    ds = xr.Dataset(coords={"time": times})
    config = {
        "data": {"time_dim": "time"},
        "rollout": {"rollout_step_hours": 12},
    }
    samples = [
        {"anchor_index": 0, "target_indices": {2: 2}},
        {"anchor_index": 2, "target_indices": {2: 4}},
    ]

    leads = _forecast_lead_hours_for_samples(ds, samples, 2, config)

    assert torch.equal(leads, torch.tensor([24.0, 24.0]))


def test_forecast_lead_hours_reject_irregular_or_mismatched_cadence() -> None:
    times = np.asarray(
        [
            np.datetime64("2024-01-01T00"),
            np.datetime64("2024-01-01T12"),
            np.datetime64("2024-01-02T06"),
        ]
    )
    ds = xr.Dataset(coords={"time": times})
    config = {
        "data": {"time_dim": "time"},
        "rollout": {"rollout_step_hours": 12},
    }

    with pytest.raises(ValueError, match="time cadence does not match"):
        _forecast_lead_hours_for_samples(
            ds,
            [{"anchor_index": 0, "target_indices": {1: 1}}],
            1,
            config,
        )


def test_batched_future_predictors_use_each_samples_own_time() -> None:
    values = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
    ds = xr.Dataset(
        {"exo": (("time", "latitude", "longitude"), values)},
        coords={
            "time": np.datetime64("2024-01-01T00")
            + np.arange(4) * np.timedelta64(12, "h"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    )
    spec = VariableSpec("exo", "exo", "surf")
    config = {
        "data": {
            "time_dim": "time",
            "lat_dim": "latitude",
            "lon_dim": "longitude",
        }
    }

    frames = _dataset_frames_for_predictor(
        ds,
        spec,
        time_indices=[1, 3],
        config=config,
        device=torch.device("cpu"),
    )

    assert tuple(frames.shape) == (2, 1, 1, 1)
    assert torch.equal(frames[:, 0, 0, 0], torch.tensor([1.0, 3.0]))



def test_checkpoint_contract_rejects_missing_lead_embedding() -> None:
    model = _wrapper(lead_time_cond=True)
    specs = ResolvedVariableSpecs(
        predictors=(),
        targets=(VariableSpec("x", "x", "surf"),),
        static=(),
    )
    target_config = [
        {"dataset_name": "x", "aurora_name": "x", "kind": "surf"}
    ]
    config = {
        "data": {"target_variables": target_config},
        "model": {
            "flow_refine_enabled": True,
            "flow_refine_contract_version": 3,
            "flow_refine_hidden": 8,
            "flow_refine_time_dim": 128,
            "flow_refine_lead_time_cond": True,
            "flow_refine_lead_time_scale_hours": 72.0,
        },
    }
    checkpoint = {
        "config": {
            "data": {"target_variables": target_config},
            "model": {
                "flow_refine_enabled": True,
                "flow_refine_contract_version": 2,
                "flow_refine_hidden": 8,
                "flow_refine_time_dim": 128,
                "flow_refine_lead_time_cond": False,
                "flow_refine_lead_time_scale_hours": 72.0,
            },
        },
        "norm_stats": {
            "x": {"mean": torch.tensor([0.0]), "std": torch.tensor([1.0])}
        },
    }

    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_checkpoint_refinement_contract(model, checkpoint, config, specs)

    checkpoint["config"]["model"].update(
        {
            "flow_refine_contract_version": 3,
            "flow_refine_lead_time_cond": True,
        }
    )
    checkpoint["config"]["data"]["target_lead_times"] = [1, 2, 3, 4, 5, 6]
    checkpoint["config"]["rollout"] = {"rollout_step_hours": 6}
    config["data"]["target_lead_times"] = [1, 2, 3, 4, 5, 6]
    config["rollout"] = {"rollout_step_hours": 12}
    with pytest.raises(ValueError, match="forecast-lead cadence mismatch"):
        validate_checkpoint_refinement_contract(model, checkpoint, config, specs)



class _OneStepBatchModel(nn.Module):
    def forward(self, batch: Batch) -> Batch:
        return dataclasses.replace(
            batch,
            surf_vars={"x": batch.surf_vars["x"][:, -1:]},
        )


class _RecordingIncrementBatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, batch: Batch) -> Batch:
        latest = batch.surf_vars["x"][:, -1:]
        self.inputs.append(latest.detach().clone())
        return dataclasses.replace(
            batch,
            surf_vars={"x": latest + 1.0},
        )


@pytest.mark.parametrize(
    (
        "contract_version",
        "feedback_enabled",
        "temporal_offset",
        "expected_outputs",
        "expected_base_inputs",
    ),
    [
        (2, False, 0.0, [11.0, 12.0], [0.0, 1.0]),
        (2, True, 0.0, [11.0, 22.0], [0.0, 11.0]),
        (2, False, 100.0, [111.0, 112.0], [0.0, 1.0]),
        # Missing/legacy v1 semantics refine and feed back unconditionally.
        (1, False, 0.0, [11.0, 22.0], [0.0, 11.0]),
    ],
)
def test_flow_rollout_two_step_refinement_provenance(
    monkeypatch: pytest.MonkeyPatch,
    contract_version: int,
    feedback_enabled: bool,
    temporal_offset: float,
    expected_outputs: list[float],
    expected_base_inputs: list[float],
) -> None:
    batch = Batch(
        surf_vars={"x": torch.zeros(1, 2, 1, 1)},
        static_vars={},
        atmos_vars={},
        metadata=Metadata(
            lat=torch.tensor([0.0]),
            lon=torch.tensor([0.0]),
            time=(np.datetime64("2024-01-01T12").tolist(),),
            atmos_levels=(),
        ),
    )
    base = _RecordingIncrementBatchModel()
    model = AuroraFlowRefine(
        base,
        target_surf_vars=("x",),
        hidden=8,
        # V2+ must still use one deterministic query even when the configured
        # sampler has multiple integration steps.
        sampling_steps=3 if contract_version >= 2 else 1,
        flow_refine_contract_version=contract_version,
        lon_periodic=False,
    )
    head = _RecordingHead(value=10.0)
    model.surf_flow["x"] = head
    if temporal_offset:
        model.temporal_enabled = True
        model.temporal = nn.Identity()

        def add_temporal_offset(
            pred: Batch,
            history: dict[tuple[str, str], list[torch.Tensor]],
        ) -> Batch:
            del history
            return dataclasses.replace(
                pred,
                surf_vars={"x": pred.surf_vars["x"] + temporal_offset},
            )

        monkeypatch.setattr(model, "apply_temporal_rollout", add_temporal_offset)

    times = np.datetime64("2024-01-01T00") + np.arange(6) * np.timedelta64(12, "h")
    ds = xr.Dataset(coords={"time": times})
    spec = VariableSpec("x", "x", "surf")
    specs = ResolvedVariableSpecs(
        predictors=(spec,),
        targets=(spec,),
        static=(),
    )
    config = {
        "data": {
            "time_dim": "time",
            "input_time_steps": 2,
            "target_lead_times": [1, 2],
        },
        "training": {
            "flow_refine_autoregressive_feedback": feedback_enabled,
        },
        "rollout": {
            "rollout_num_steps": 2,
            "rollout_step_hours": 12,
            "autoregressive_inputs": True,
            "keep_exogenous_predictors": "fixed",
            "verbose_provenance": False,
        },
    }
    monkeypatch.setattr(
        "finetune.aurora_finetune_utils.build_aurora_batch",
        lambda *args, **kwargs: batch,
    )

    predictions = run_rollout(
        model,
        ds,
        {"anchor_index": 1, "history_indices": [0, 1], "target_indices": {}},
        config,
        specs,
        device="cpu",
    )

    outputs = [float(pred.surf_vars["x"].item()) for pred in predictions]
    base_inputs = [float(value.item()) for value in base.inputs]
    assert outputs == expected_outputs
    assert base_inputs == expected_base_inputs
    assert len(head.calls) == 2
    expected_t = 1.0 if contract_version == 1 else 0.0
    assert all(
        torch.equal(t, torch.full_like(t, expected_t))
        and torch.count_nonzero(x_t) == 0
        for x_t, t in head.calls
    )


def test_v2_rollout_stochastic_sampling_is_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(
        surf_vars={"x": torch.zeros(1, 2, 1, 1)},
        static_vars={},
        atmos_vars={},
        metadata=Metadata(
            lat=torch.tensor([0.0]),
            lon=torch.tensor([0.0]),
            time=(np.datetime64("2024-01-01T12").tolist(),),
            atmos_levels=(),
        ),
    )
    model = AuroraFlowRefine(
        _RecordingIncrementBatchModel(),
        target_surf_vars=("x",),
        hidden=8,
        sampling_steps=2,
        flow_refine_contract_version=2,
        lon_periodic=False,
    )
    head = _EchoHead()
    model.surf_flow["x"] = head
    times = np.datetime64("2024-01-01T00") + np.arange(4) * np.timedelta64(12, "h")
    ds = xr.Dataset(coords={"time": times})
    spec = VariableSpec("x", "x", "surf")
    specs = ResolvedVariableSpecs(
        predictors=(spec,),
        targets=(spec,),
        static=(),
    )
    config = {
        "data": {
            "time_dim": "time",
            "input_time_steps": 2,
            "target_lead_times": [1],
        },
        "rollout": {
            "rollout_num_steps": 1,
            "rollout_step_hours": 12,
            "autoregressive_inputs": True,
            "keep_exogenous_predictors": "fixed",
            "verbose_provenance": False,
            "flow_refine_stochastic_sampling": True,
        },
    }
    monkeypatch.setattr(
        "finetune.aurora_finetune_utils.build_aurora_batch",
        lambda *args, **kwargs: batch,
    )
    first = run_rollout(
        model,
        ds,
        {"anchor_index": 1, "history_indices": [0, 1], "target_indices": {}},
        config,
        specs,
        device="cpu",
        refinement_seed=29,
    )
    second = run_rollout(
        model,
        ds,
        {"anchor_index": 1, "history_indices": [0, 1], "target_indices": {}},
        config,
        specs,
        device="cpu",
        refinement_seed=29,
    )
    third = run_rollout(
        model,
        ds,
        {"anchor_index": 1, "history_indices": [0, 1], "target_indices": {}},
        config,
        specs,
        device="cpu",
        refinement_seed=30,
    )

    assert len(head.calls) == 6
    torch.testing.assert_close(
        first[0].surf_vars["x"], second[0].surf_vars["x"],
    )
    assert not torch.equal(
        first[0].surf_vars["x"], third[0].surf_vars["x"],
    )


def test_run_rollout_passes_cumulative_physical_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(
        surf_vars={"x": torch.zeros(1, 2, 1, 1)},
        static_vars={},
        atmos_vars={},
        metadata=Metadata(
            lat=torch.tensor([0.0]),
            lon=torch.tensor([0.0]),
            time=(np.datetime64("2024-01-01T12").tolist(),),
            atmos_levels=(),
        ),
    )
    model = AuroraFlowRefine(
        _OneStepBatchModel(),
        target_surf_vars=("x",),
        hidden=8,
        sampling_steps=1,
        lead_time_cond=True,
        lead_time_scale_hours=72.0,
        lon_periodic=False,
    )
    head = _RecordingHead(value=0.0)
    model.surf_flow["x"] = head
    times = np.datetime64("2024-01-01T00") + np.arange(8) * np.timedelta64(12, "h")
    ds = xr.Dataset(coords={"time": times})
    spec = VariableSpec("x", "x", "surf")
    specs = ResolvedVariableSpecs(
        predictors=(spec,),
        targets=(spec,),
        static=(),
    )
    config = {
        "data": {
            "time_dim": "time",
            "input_time_steps": 2,
            "target_lead_times": [1, 2, 3],
        },
        "rollout": {
            "rollout_num_steps": 3,
            "rollout_step_hours": 12,
            "autoregressive_inputs": True,
            "keep_exogenous_predictors": "fixed",
            "verbose_provenance": False,
        },
    }
    monkeypatch.setattr(
        "finetune.aurora_finetune_utils.build_aurora_batch",
        lambda *args, **kwargs: batch,
    )

    predictions = run_rollout(
        model,
        ds,
        {"anchor_index": 1, "history_indices": [0, 1], "target_indices": {}},
        config,
        specs,
        device="cpu",
    )

    assert len(predictions) == 3
    assert [float(call.item()) for call in head.lead_calls if call is not None] == [
        12.0,
        24.0,
        36.0,
    ]
