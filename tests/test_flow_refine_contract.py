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
    run_rollout,
    select_refinement_checkpoint,
    validate_checkpoint_refinement_contract,
)
from finetune.flow_refine import AuroraFlowRefine


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


def _wrapper(
    *,
    sampling_steps: int = 1,
    lead_time_cond: bool = False,
) -> AuroraFlowRefine:
    model = AuroraFlowRefine(
        nn.Identity(),
        target_surf_vars=("x",),
        hidden=8,
        sampling_steps=sampling_steps,
        lead_time_cond=lead_time_cond,
        lead_time_scale_hours=72.0,
        lon_periodic=False,
    )
    return model


def test_single_step_sampler_queries_source_mean() -> None:
    model = _wrapper(sampling_steps=1)
    head = _RecordingHead(value=2.0)
    cond = torch.randn(2, 1, 4, 4)

    result = model._sample_residual(cond, head)

    assert torch.equal(result, torch.full_like(result, 2.0))
    assert len(head.calls) == 1
    x_t, t = head.calls[0]
    assert torch.count_nonzero(x_t) == 0
    assert torch.count_nonzero(t) == 0


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
