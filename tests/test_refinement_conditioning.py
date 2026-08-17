"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared input-state/static conditioning contracts for unified refinement."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest
import torch
from finetune.refinement.integration import (
    maybe_build_stochastic_refiner,
    pack_refinement_conditioning,
    refine_batch_prediction,
)
from finetune.refinement.two_phase import build_two_phase_refiner

from tests.refinement_fixtures import DummySpec, build_packing, refinement_config

from aurora import Batch, Metadata


def _conditioning_config(*, temporal: bool = False) -> dict:
    config = refinement_config(
        "diffusion_transformer",
        conditioning={
            "aurora_rollout": True,
            "aurora_input_state": True,
            "static_fields": True,
            "masks": True,
            "forecast_lead_time": True,
        },
        diffusion={"training_timesteps": 8, "inference_steps": 1},
        transformer={
            "patch_size": [4, 4],
            "embedding_dim": 16,
            "num_heads": 4,
            "num_blocks": 1,
            "mlp_ratio": 2.0,
            "max_tokens_lat": 8,
            "max_tokens_lon": 8,
            "optimized_attention": "math",
        },
    )
    config["model"].update(
        {
            "mamba_temporal_enabled": temporal,
            "mamba_temporal_channels": 4,
            "mamba_temporal_state": 2,
            "mamba_temporal_layers": 1,
            "mamba_temporal_conv": 2,
            "mamba_temporal_expand": 1,
        }
    )
    return config


def _batch(batch_size: int = 2, height: int = 8, width: int = 8) -> Batch:
    surf = torch.empty(batch_size, 2, height, width)
    atmos = torch.empty(batch_size, 2, 3, height, width)
    for sample in range(batch_size):
        surf[sample, 0].fill_(-10.0 - sample)
        surf[sample, 1].fill_(1.0e-3 + sample * 2.0e-4)
        atmos[sample, 0].fill_(-20.0 - sample)
        # Metadata order is deliberately different from FieldPacking order.
        atmos[sample, 1, 0].fill_(10.0 + sample)  # 1000 hPa, not selected
        atmos[sample, 1, 1].fill_(2.0e-7 + sample * 2.0e-7)  # 850 hPa
        atmos[sample, 1, 2].fill_(3.0e-7 + sample * 1.0e-7)  # 500 hPa
    return Batch(
        surf_vars={"gtco3": surf},
        # Mapping insertion order is the reverse of the configured order.
        static_vars={
            "z": torch.full((height, width), 5.0),
            "lsm": torch.full((height, width), 4.0),
        },
        atmos_vars={"go3": atmos},
        metadata=Metadata(
            lat=torch.linspace(50.0, 43.0, height),
            lon=torch.linspace(230.0, 237.0, width),
            time=tuple(datetime(2024, 1, 1) for _ in range(batch_size)),
            atmos_levels=(1000.0, 850.0, 500.0),
        ),
    )


def _model(*, temporal: bool = False):
    model = build_two_phase_refiner(
        None,
        build_packing(height=8, width=8),
        _conditioning_config(temporal=temporal),
        conditioning_static_names=("lsm", "z"),
    )
    model.initialize_refiner(model.conditioning_channels())
    return model


def test_eval_forward_rejects_enabled_temporal_single_step() -> None:
    model = _model(temporal=True).eval()

    with pytest.raises(
        RuntimeError,
        match=(
            "cannot apply enabled temporal Mamba to a single forecast step.*"
            "sequence-aware rollout path.*chronological temporal_history"
        ),
    ):
        model(_batch(batch_size=1), forecast_lead_time_hours=24.0)


def test_eval_forward_without_temporal_preserves_refinement_path(monkeypatch) -> None:
    model = _model(temporal=False).eval()
    batch = _batch(batch_size=1)
    sentinel = object()
    seen = {}

    def _refine(wrapper, pred, **kwargs):
        seen.update(wrapper=wrapper, pred=pred, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(
        "finetune.refinement.integration.refine_batch_prediction", _refine
    )

    result = model(batch, forecast_lead_time_hours=24.0)

    assert result is sentinel
    assert seen["wrapper"] is model
    assert seen["pred"] is batch
    assert seen["kwargs"] == {
        "aurora_input_batch": batch,
        "forecast_lead_time_hours": 24.0,
        "ensemble_size": 1,
        "seed": model.refinement_config.seed,
    }


def _coordinate_model(packing=None):
    config = refinement_config(
        "diffusion_transformer",
        conditioning={
            "aurora_rollout": True,
            "aurora_input_state": False,
            "static_fields": False,
            "masks": False,
            "forecast_lead_time": True,
            "latitude": True,
            "longitude": True,
        },
    )
    return build_two_phase_refiner(
        None,
        build_packing(height=3, width=4, lon_periodic=True)
        if packing is None
        else packing,
        config,
    )


def test_coordinate_conditioning_has_exact_normalized_periodic_channels() -> None:
    packing = replace(
        build_packing(height=3, width=4, lon_periodic=True),
        lon=(0.0, 90.0, 180.0, 360.0),
    )
    model = _coordinate_model(packing)
    rollout = torch.arange(2 * 3 * 3 * 4, dtype=torch.float32).reshape(
        2, 3, 3, 4
    )

    conditioning = model.build_conditioning(rollout)

    assert model.conditioning_channels() == 6
    assert conditioning.shape == (2, 6, 3, 4)
    assert torch.equal(conditioning[:, :3], rollout)
    expected_latitude = torch.tensor([1.0, 0.0, -1.0]).view(1, 1, 3, 1)
    expected_latitude = expected_latitude.expand(2, 1, 3, 4)
    expected_sine = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(1, 1, 1, 4)
    expected_sine = expected_sine.expand(2, 1, 3, 4)
    expected_cosine = torch.tensor([1.0, 0.0, -1.0, 1.0]).view(1, 1, 1, 4)
    expected_cosine = expected_cosine.expand(2, 1, 3, 4)
    torch.testing.assert_close(conditioning[:, 3:4], expected_latitude)
    torch.testing.assert_close(
        conditioning[:, 4:5], expected_sine, rtol=0.0, atol=2.0e-7
    )
    torch.testing.assert_close(
        conditioning[:, 5:6], expected_cosine, rtol=0.0, atol=2.0e-7
    )
    # Equivalent 0- and 360-degree longitudes have identical periodic features.
    torch.testing.assert_close(
        conditioning[:, 4:6, :, 0],
        conditioning[:, 4:6, :, -1],
        rtol=0.0,
        atol=2.0e-7,
    )


@pytest.mark.parametrize(
    "axis,bad_values,message",
    [
        ("lat", (90.0, 0.0), "latitude.*expected 3, got 2"),
        (
            "lon",
            (0.0, 90.0, float("nan"), 270.0),
            "longitude coordinates must be.*finite",
        ),
    ],
)
def test_coordinate_conditioning_validates_metadata_shape_and_finiteness(
    axis: str,
    bad_values: tuple[float, ...],
    message: str,
) -> None:
    model = _coordinate_model()
    # Simulate malformed restored metadata, bypassing FieldPacking's constructor
    # guard so build_conditioning's defensive checks are exercised directly.
    object.__setattr__(model.packing, axis, bad_values)

    with pytest.raises(ValueError, match=message):
        model.build_conditioning(torch.zeros(1, 3, 3, 4))


def test_conditioning_packer_uses_latest_state_selected_levels_and_static_order() -> None:
    model = _model()
    batch = _batch()
    reference = torch.zeros(2, model.packing.num_channels, 8, 8)
    packed = pack_refinement_conditioning(
        model,
        reference,
        aurora_input_batch=batch,
    )

    expected_physical = model.packing.pack(
        {
            "gtco3": batch.surf_vars["gtco3"][:, -1],
            # Packing declares (500, 850), while the Batch declares
            # (1000, 850, 500).
            "go3": batch.atmos_vars["go3"][:, -1, [2, 1]],
        }
    )
    assert torch.equal(
        packed.input_state_normalized,
        model.target_space.encode(expected_physical),
    )
    assert packed.static_fields is not None
    assert packed.static_fields.shape == (2, 2, 8, 8)
    assert torch.all(packed.static_fields[:, 0] == 4.0)  # lsm
    assert torch.all(packed.static_fields[:, 1] == 5.0)  # z
    assert model.refiner is not None
    assert model.refiner.cond_channels == 9  # rollout 3 + input 3 + statics 2 + mask


def test_diffusion_transformer_temporal_loss_reuses_exact_full_conditioning(
    monkeypatch,
) -> None:
    model = _model(temporal=True)
    batch = _batch(batch_size=1)
    rollout = torch.zeros(2, model.packing.num_channels, 8, 8)
    target = rollout + 0.25
    packed = pack_refinement_conditioning(
        model,
        rollout,
        aurora_input_batch=batch,
    )
    out = model.training_step(
        rollout,
        target,
        input_state_normalized=packed.input_state_normalized,
        static_fields=packed.static_fields,
        valid_mask=torch.ones_like(rollout, dtype=torch.bool),
        forecast_lead_time=torch.tensor([24.0, 48.0]),
        lead_index=torch.tensor([0, 1]),
        generator=torch.Generator().manual_seed(7),
    )
    assert out.conditioning is not None
    assert out.conditioning.shape == (2, 9, 8, 8)
    assert torch.equal(out.conditioning[:, :3], rollout)
    assert torch.equal(out.conditioning[:, 3:6], packed.input_state_normalized)
    assert torch.equal(out.conditioning[:, 6:8], packed.static_fields)
    assert torch.all(out.conditioning[:, 8:] == 1)

    seen: list[torch.Tensor] = []

    def _sample(conditioning, **_):
        seen.append(conditioning.detach().clone())
        return torch.zeros(
            conditioning.shape[0],
            model.packing.num_channels,
            *conditioning.shape[-2:],
            device=conditioning.device,
            dtype=conditioning.dtype,
        )

    assert model.refiner is not None
    monkeypatch.setattr(model.refiner, "sample_residual", _sample)
    temporal_loss, metrics = model.temporal_training_loss(
        rollout,
        target,
        valid_mask=torch.ones_like(rollout, dtype=torch.bool),
        forecast_lead_time=torch.tensor([24.0, 48.0]),
        lead_index=torch.tensor([0, 1]),
        conditioning=out.conditioning,
    )
    assert seen and torch.equal(seen[0], out.conditioning)
    assert torch.isfinite(temporal_loss)
    assert set(metrics) == {
        "temporal_loss/gtco3",
        "temporal_loss/go3",
        "temporal_base_loss",
        "temporal_total_loss",
    }
    temporal_loss.backward()
    assert model.temporal is not None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in model.temporal.named_parameters()
        if ".decoder." in name
    )


def test_refine_batch_prediction_passes_preforecast_conditioning(monkeypatch) -> None:
    model = _model(temporal=True)
    input_batch = _batch(batch_size=1)
    pred = Batch(
        surf_vars={"gtco3": input_batch.surf_vars["gtco3"][:, -1:] + 0.1},
        static_vars=input_batch.static_vars,
        atmos_vars={"go3": input_batch.atmos_vars["go3"][:, -1:] + 0.1},
        metadata=input_batch.metadata,
    )
    seen: list[torch.Tensor] = []

    def _sample(conditioning, **_):
        seen.append(conditioning.detach().clone())
        return torch.zeros(
            conditioning.shape[0],
            model.packing.num_channels,
            *conditioning.shape[-2:],
            device=conditioning.device,
            dtype=conditioning.dtype,
        )

    assert model.refiner is not None
    monkeypatch.setattr(model.refiner, "sample_innovation_normalized", _sample)
    refined = refine_batch_prediction(
        model,
        pred,
        aurora_input_batch=input_batch,
        forecast_lead_time_hours=24.0,
        ensemble_size=1,
        temporal_history=[],
    )
    assert isinstance(refined, Batch)
    assert seen and seen[0].shape == (1, 9, 8, 8)
    with pytest.raises(ValueError, match="aurora_input_batch=None"):
        refine_batch_prediction(model, pred, ensemble_size=1)


def test_shared_factory_builds_real_conditioning_width_and_validates_inputs() -> None:
    targets = (
        DummySpec("gtco3", "gtco3", "surf", units="kg m-2"),
        DummySpec("go3", "go3", "atmos", [500.0, 850.0]),
    )
    statics = (
        DummySpec("lsm", "lsm", "static"),
        DummySpec("z", "z_static", "static"),
    )
    resolved = SimpleNamespace(targets=targets, predictors=targets, static=statics)
    config = _conditioning_config()
    config["model"]["patch_size"] = 4
    config["data"] = {
        "atmos_levels": [500.0, 850.0],
        "target_lead_times": [1, 2],
    }
    config["rollout"] = {"rollout_step_hours": 24}
    wrapper = maybe_build_stochastic_refiner(
        None,
        config,
        resolved,
        lat=list(range(8, 0, -1)),
        lon=list(range(8)),
    )
    assert wrapper is not None and wrapper.refiner is not None
    assert wrapper.conditioning_static_names == ("lsm", "z")
    assert wrapper.refiner.cond_channels == 9

    missing = SimpleNamespace(targets=targets, predictors=targets[:1], static=statics)
    with pytest.raises(ValueError, match="Missing target predictors.*go3"):
        maybe_build_stochastic_refiner(None, config, missing)
