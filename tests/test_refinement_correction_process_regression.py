"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Regression tests for correction mean/innovation and diffusion semantics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import MethodType

import pytest
import torch
import yaml

from finetune.refinement.base import RefinerOutput
from finetune.refinement.checkpoint import (
    build_refinement_checkpoint,
    load_refinement_state_dict,
)
from finetune.refinement.config import (
    CORRECTION_CONVENTION,
    ConfigValidationError,
    resolve_refinement_config,
)
from finetune.refinement.schedules import DiffusionSchedule
from tests.refinement_fixtures import build_refiner_model


@pytest.mark.parametrize("prediction_type", ("epsilon", "velocity", "sample"))
def test_parameterizations_recover_x0_and_epsilon_in_float32(prediction_type):
    schedule = DiffusionSchedule(num_train_timesteps=64, schedule="cosine")
    clean = torch.randn(3, 2, 4, 5, dtype=torch.bfloat16)
    epsilon = torch.randn_like(clean)
    timesteps = torch.tensor([2, 23, 61])
    noisy = schedule.add_noise(clean, epsilon, timesteps)
    target = schedule.training_target(prediction_type, clean, epsilon, timesteps)

    recovered_clean = schedule.to_clean(
        prediction_type, target, noisy, timesteps
    )
    recovered_epsilon = schedule.to_epsilon(
        prediction_type, target, noisy, timesteps
    )

    assert noisy.dtype == target.dtype == torch.float32
    assert recovered_clean.dtype == recovered_epsilon.dtype == torch.float32
    assert torch.allclose(recovered_clean, clean.float(), atol=2.0e-4)
    assert torch.allclose(recovered_epsilon, epsilon.float(), atol=2.0e-4)


@pytest.mark.parametrize("kind", ("diffusion_unet", "diffusion_transformer"))
def test_zero_initialized_epsilon_refiner_is_exact_identity(kind):
    model = build_refiner_model(
        kind,
        height=8,
        width=8,
        diffusion={"prediction_type": "epsilon", "inference_steps": 3},
    )
    rollout = torch.randn(2, model.packing.num_channels, 8, 8)
    conditioning = model.build_conditioning(rollout)
    lead = torch.tensor([24.0, 48.0])

    correction = model.refiner.sample_correction_normalized(
        conditioning,
        forecast_lead_time=lead,
        generator=torch.Generator().manual_seed(7),
        num_steps=3,
    )
    result = model.refine(
        rollout, forecast_lead_time=lead, ensemble_size=2, seed=17
    )

    assert correction.dtype == torch.float32
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(result.refined_normalized, rollout.float())
    assert torch.count_nonzero(result.member_corrections_normalized) == 0
    assert torch.count_nonzero(result.member_innovations_normalized) == 0


def _set_output_bias(network, value):
    with torch.no_grad():
        network.out_proj.weight.zero_()
        network.out_proj.bias.fill_(value)


def test_training_process_target_is_correction_minus_detached_mean():
    model = build_refiner_model("diffusion_unet", height=8, width=8)
    refiner = model.refiner
    _set_output_bias(refiner.mean_net, 0.25)
    captured = {}

    def fake_training_loss(
        self,
        process_target,
        conditioning,
        *,
        forecast_lead_time,
        mask,
        generator,
    ):
        captured["target"] = process_target.detach().clone()
        return (
            RefinerOutput(generative_loss=process_target.square().mean()),
            torch.zeros_like(process_target),
        )

    refiner._training_loss = MethodType(fake_training_loss, refiner)
    conditioning = torch.randn(2, model.conditioning_channels(), 8, 8)
    correction = torch.full((2, model.packing.num_channels, 8, 8), 0.75)
    output = refiner.compute_training_loss(
        correction,
        conditioning,
        forecast_lead_time=torch.tensor([24.0, 48.0]),
    )

    assert torch.allclose(
        captured["target"], torch.full_like(captured["target"], 0.5)
    )
    assert float(output.deterministic_loss) == pytest.approx(0.125)
    assert float(output.total_loss) == pytest.approx(0.375)
    output.total_loss.backward()
    assert torch.count_nonzero(refiner.mean_net.out_proj.bias.grad) > 0


def test_sampling_composes_mean_and_shift_free_innovation_once():
    model = build_refiner_model(
        "diffusion_unet",
        height=8,
        width=8,
        target_space={
            "residual_scaling": "per_channel",
            "residual_scaling_center": True,
            "residual_scaling_warmup_batches": 1,
        },
    )
    refiner = model.refiner
    calibration = torch.full((2, model.packing.num_channels, 8, 8), 4.0)
    calibration[:, :, 0, 0] = 6.0
    refiner.fit_residual_scale(calibration)
    _set_output_bias(refiner.mean_net, 0.2)
    with torch.no_grad():
        refiner.net.out_proj.bias.fill_(1.0e-6)

    def fake_sample(
        self,
        conditioning,
        *,
        forecast_lead_time,
        generator,
        num_steps,
    ):
        return torch.full(self.residual_shape(conditioning), 0.3)

    refiner._sample = MethodType(fake_sample, refiner)
    conditioning = torch.randn(2, model.conditioning_channels(), 8, 8)
    lead = torch.tensor([24.0, 48.0])
    mean = refiner.deterministic_mean_correction_normalized(
        conditioning, forecast_lead_time=lead
    )
    innovation = refiner.sample_innovation_normalized(
        conditioning,
        forecast_lead_time=lead,
        num_steps=2,
        mean_correction_normalized=mean,
    )
    correction = refiner.sample_correction_normalized(
        conditioning, forecast_lead_time=lead, num_steps=2
    )

    expected = refiner.residual_scaler.decode_difference(
        torch.full_like(innovation, 0.3)
    )
    assert torch.allclose(innovation, expected)
    assert torch.allclose(correction, mean + innovation)
    assert torch.allclose(
        correction - innovation,
        refiner.residual_scaler.decode(torch.full_like(mean, 0.2)),
    )


def test_pre_innovation_checkpoint_loads_strictly_as_legacy():
    source = build_refiner_model("diffusion_unet", height=8, width=8).refiner
    legacy_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("mean_net.")
        and key != "innovation_parameterization_version"
    }
    target = build_refiner_model("diffusion_unet", height=8, width=8).refiner

    incompatible = target.load_state_dict(legacy_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert not target.uses_innovation_parameterization

def test_two_phase_exposes_mean_plus_sampled_innovation():
    model = build_refiner_model("diffusion_unet", height=8, width=8)
    refiner = model.refiner
    _set_output_bias(refiner.mean_net, 0.2)
    with torch.no_grad():
        refiner.net.out_proj.bias.fill_(1.0e-6)

    def fake_sample(
        self,
        conditioning,
        *,
        forecast_lead_time,
        generator,
        num_steps,
    ):
        return torch.full(self.residual_shape(conditioning), 0.3)

    refiner._sample = MethodType(fake_sample, refiner)
    rollout = torch.zeros(1, model.packing.num_channels, 8, 8)
    result = model.refine(
        rollout,
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=2,
        seed=9,
    )

    assert torch.allclose(
        result.conditional_mean_correction_normalized,
        torch.full_like(rollout, 0.2),
    )
    assert torch.allclose(
        result.member_innovations_normalized,
        torch.full_like(result.member_innovations_normalized, 0.3),
    )
    assert torch.allclose(
        result.member_corrections_normalized,
        torch.full_like(result.member_corrections_normalized, 0.5),
    )
    assert torch.allclose(
        result.deterministic_refined_normalized,
        torch.full_like(rollout, 0.2),
    )
    assert torch.allclose(
        result.refined_normalized, torch.full_like(rollout, 0.5)
    )

@pytest.mark.parametrize("prediction_type", ("epsilon", "velocity", "sample"))
def test_oracle_reverse_trajectory_recovers_clean_sample(prediction_type):
    model = build_refiner_model(
        "diffusion_unet",
        height=8,
        width=8,
        diffusion={
            "training_timesteps": 64,
            "inference_steps": 6,
            "prediction_type": prediction_type,
        },
    )
    refiner = model.refiner
    conditioning = torch.zeros(2, model.conditioning_channels(), 8, 8)
    clean = torch.randn(2, model.packing.num_channels, 8, 8)
    epsilon = torch.randn_like(clean)
    first_timestep = refiner.schedule.inference_timesteps(
        6, torch.device("cpu")
    )[0].expand(clean.shape[0])
    initial = refiner.schedule.add_noise(clean, epsilon, first_timestep)

    def oracle_prediction(
        self, state, conditioning, process_time, lead
    ):
        timesteps = process_time.long()
        sqrt_alpha, sqrt_one_minus_alpha = self.schedule._coefficients(
            state, timesteps
        )
        current_epsilon = (
            state.float() - sqrt_alpha * clean
        ) / sqrt_one_minus_alpha.clamp(min=1.0e-12)
        if prediction_type == "sample":
            return clean
        if prediction_type == "epsilon":
            return current_epsilon
        return (
            sqrt_alpha * current_epsilon
            - sqrt_one_minus_alpha * clean
        )

    refiner._network_prediction = MethodType(oracle_prediction, refiner)
    recovered = refiner._ddim_loop(
        initial,
        conditioning,
        lead=None,
        steps=6,
        eta=0.0,
        generator=None,
        stochastic=False,
    )
    assert torch.allclose(recovered, clean, atol=2.0e-4)


def test_stochastic_diffusion_requires_two_steps_but_mean_does_not():
    model = build_refiner_model(
        "diffusion_unet",
        height=8,
        width=8,
        diffusion={"training_timesteps": 8, "inference_steps": 1},
    )
    refiner = model.refiner
    conditioning = torch.zeros(1, model.conditioning_channels(), 8, 8)

    mean = refiner.deterministic_mean_correction_normalized(conditioning)
    assert torch.count_nonzero(mean) == 0
    assert int(refiner.schedule.inference_timesteps(1, torch.device("cpu"))[0]) == 7
    with pytest.raises(ValueError, match="at least 2 reverse steps"):
        refiner._sample(
            conditioning,
            forecast_lead_time=None,
            generator=torch.Generator().manual_seed(3),
            num_steps=1,
        )


def test_config_correction_contract_and_all_unified_no2_switches():
    path = Path(
        "finetune/"
        "aurora_NO2_finetune_US-WEST_3day_lead_"
        "diffusion_transformer_config.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    assert raw["case_name"].endswith("_corrected")
    for kind in (
        "flow_matching_conv_unet",
        "flow_matching_transformer",
        "diffusion_unet",
        "diffusion_transformer",
    ):
        candidate = deepcopy(raw)
        candidate["model"]["refinement"]["type"] = kind
        resolved = resolve_refinement_config(candidate)
        assert resolved.type == kind
        assert resolved.correction_convention == CORRECTION_CONVENTION
        assert resolved.train_on_residual

    with pytest.raises(ConfigValidationError, match="correction_convention"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "enabled": True,
                        "type": "diffusion_unet",
                        "correction_convention": "aurora_minus_cams_subtract",
                    }
                }
            }
        )
    with pytest.raises(ConfigValidationError, match="train_on_residual=true"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "enabled": True,
                        "type": "diffusion_unet",
                        "train_on_residual": False,
                    }
                }
            }
        )


def test_active_unified_yaml_examples_declare_correction_contract():
    paths = (
        "finetune/examples/stochastic_refinement/"
        "aurora_O3_global_flow_matching_conv_unet.yaml",
        "finetune/examples/stochastic_refinement/"
        "aurora_O3_global_flow_matching_transformer.yaml",
        "finetune/examples/stochastic_refinement/"
        "aurora_O3_global_diffusion_unet.yaml",
        "finetune/examples/stochastic_refinement/"
        "aurora_O3_global_diffusion_transformer.yaml",
    )
    for raw_path in paths:
        raw = yaml.safe_load(Path(raw_path).read_text())
        refinement = raw["model"]["refinement"]
        assert refinement["correction_convention"] == CORRECTION_CONVENTION


def test_checkpoint_records_contract_and_high_level_loader_migrates_v1():
    source_model = build_refiner_model("diffusion_unet", height=8, width=8)
    payload = build_refinement_checkpoint(
        source_model,
        resolved_config=source_model.refinement_config.to_dict(),
        refinement_type="diffusion_unet",
    )
    assert payload["correction_convention"] == CORRECTION_CONVENTION
    assert payload["scientific_contract"] == {
        "correction_convention": CORRECTION_CONVENTION,
        "correction_target": "cams_target - aurora_forecast",
        "application": "aurora_forecast + predicted_correction_physical",
    }

    legacy_payload = dict(payload)
    legacy_payload["model_state_dict"] = {
        key: value.clone()
        for key, value in payload["model_state_dict"].items()
        if not key.startswith("refiner.mean_net.")
        and key != "refiner.innovation_parameterization_version"
    }
    target_model = build_refiner_model(
        "diffusion_unet", height=8, width=8
    )
    report = load_refinement_state_dict(target_model, legacy_payload)

    assert report.missing == []
    assert report.unexpected == []
    assert not target_model.refiner.uses_innovation_parameterization
