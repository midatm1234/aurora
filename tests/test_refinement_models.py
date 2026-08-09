"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Generative-process contracts: flow matching and diffusion.
"""

from __future__ import annotations

import pytest
import torch
from finetune.refinement.schedules import (
    DiffusionSchedule,
    LeadTimeEmbedding,
    ProcessTimeEmbedding,
)

from tests.refinement_fixtures import build_refiner_model

# --------------------------------------------------------------------------
# Flow matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", ["flow_matching_unet", "flow_matching_transformer"])
def test_one_flow_training_step(refiner_type: str) -> None:
    model = build_refiner_model(refiner_type, height=16, width=16)
    rollout = torch.randn(2, model.packing.num_channels, 16, 16)
    target = rollout + 0.05 * torch.randn_like(rollout)
    generator = torch.Generator().manual_seed(0)
    out = model.training_step(
        rollout,
        target,
        forecast_lead_time=torch.tensor([24.0, 72.0]),
        lead_index=torch.tensor([0, 1]),
        generator=generator,
    )
    loss = out.losses["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.refiner.parameters())


def test_flow_time_sampling_stays_inside_the_open_unit_interval() -> None:
    model = build_refiner_model("flow_matching_transformer")
    refiner = model.refiner
    generator = torch.Generator().manual_seed(11)
    t = refiner._sample_flow_time(512, torch.device("cpu"), generator)
    assert t.shape == (512,)
    assert float(t.min()) >= refiner.sigma_min
    assert float(t.max()) <= 1.0 - refiner.sigma_min


def test_logit_normal_sampling_matches_the_existing_aurora_distribution() -> None:
    """t = sigmoid(N(mean, std)) is the distribution the Aurora head uses."""
    model = build_refiner_model("flow_matching_transformer")
    refiner = model.refiner
    assert refiner.time_sampling == "logit_normal"
    assert refiner.logit_normal_mean == pytest.approx(-0.5)
    assert refiner.logit_normal_std == pytest.approx(1.2)

    torch.manual_seed(3)
    reference = torch.sigmoid(
        torch.randn(64) * refiner.logit_normal_std + refiner.logit_normal_mean
    ).clamp(refiner.sigma_min, 1.0 - refiner.sigma_min)
    generator = torch.Generator().manual_seed(3)
    drawn = refiner._sample_flow_time(64, torch.device("cpu"), generator)
    assert torch.allclose(drawn, reference)


def test_existing_aurora_path_regresses_the_clean_residual() -> None:
    """The default path predicts x1 (the residual), not the velocity."""
    model = build_refiner_model("flow_matching_transformer")
    assert model.refiner.interpolation_path == "existing_aurora"
    assert model.refiner.predicts_velocity is False


def test_rectified_flow_path_regresses_the_velocity() -> None:
    model = build_refiner_model(
        "flow_matching_transformer",
        flow_matching={"interpolation_path": "rectified_flow", "time_sampling": "uniform"},
    )
    refiner = model.refiner
    assert refiner.predicts_velocity is True

    # With a zero-output network the Euler integration keeps the initial state,
    # so the residual equals the drawn source sample exactly.
    conditioning = torch.zeros(1, model.conditioning_channels(), 16, 16)
    generator = torch.Generator().manual_seed(5)
    residual = refiner.sample_residual(
        conditioning,
        forecast_lead_time=torch.tensor([24.0]),
        generator=generator,
        num_steps=4,
    )
    expected = torch.randn(
        (1, model.packing.num_channels, 16, 16),
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.allclose(residual, expected)


@pytest.mark.parametrize("solver", ["euler", "midpoint", "heun"])
def test_flow_solvers_run(solver: str) -> None:
    model = build_refiner_model(
        "flow_matching_transformer",
        flow_matching={"interpolation_path": "rectified_flow", "solver": solver},
    )
    conditioning = torch.zeros(1, model.conditioning_channels(), 16, 16)
    residual = model.refiner.sample_residual(
        conditioning, forecast_lead_time=torch.tensor([24.0]), num_steps=2
    )
    assert residual.shape == (1, model.packing.num_channels, 16, 16)
    assert torch.isfinite(residual).all()


def test_single_step_data_parameterisation_is_the_source_mean_query() -> None:
    """``integration_steps == 1`` reproduces the existing deterministic query."""
    model = build_refiner_model(
        "flow_matching_transformer",
        flow_matching={"integration_steps": 1},
        transformer={"zero_init_output": False},
    )
    conditioning = torch.randn(2, model.conditioning_channels(), 16, 16)
    lead = torch.tensor([24.0, 48.0])
    sampled = model.refiner.sample_residual(conditioning, forecast_lead_time=lead)
    deterministic = model.refiner.deterministic_residual(conditioning, forecast_lead_time=lead)
    assert torch.equal(sampled, deterministic)


# --------------------------------------------------------------------------
# Diffusion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["cosine", "linear", "scaled_linear"])
def test_schedule_construction(schedule: str) -> None:
    built = DiffusionSchedule(num_train_timesteps=100, schedule=schedule)
    assert built.betas.shape == (100,)
    assert bool((built.betas > 0).all())
    assert bool((built.alphas_cumprod > 0).all())
    # alpha_bar must decrease monotonically.
    assert bool((built.alphas_cumprod.diff() <= 0).all())


def test_unsupported_schedule_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported diffusion schedule"):
        DiffusionSchedule(schedule="quadratic")


def test_forward_noising_matches_the_closed_form() -> None:
    schedule = DiffusionSchedule(num_train_timesteps=50, schedule="cosine")
    clean = torch.randn(4, 2, 3, 3)
    noise = torch.randn_like(clean)
    steps = torch.tensor([0, 7, 23, 49])
    noisy = schedule.add_noise(clean, noise, steps)
    a = schedule.sqrt_alphas_cumprod[steps].view(-1, 1, 1, 1)
    b = schedule.sqrt_one_minus_alphas_cumprod[steps].view(-1, 1, 1, 1)
    assert torch.allclose(noisy, a * clean + b * noise, atol=1e-6)


@pytest.mark.parametrize("prediction_type", ["epsilon", "velocity", "sample"])
def test_prediction_type_conversion_round_trip(prediction_type: str) -> None:
    schedule = DiffusionSchedule(num_train_timesteps=60, schedule="cosine")
    clean = torch.randn(3, 2, 4, 4)
    noise = torch.randn_like(clean)
    steps = torch.tensor([5, 20, 45])
    noisy = schedule.add_noise(clean, noise, steps)
    target = schedule.training_target(prediction_type, clean, noise, steps)

    recovered_clean = schedule.to_clean(prediction_type, target, noisy, steps)
    recovered_noise = schedule.to_epsilon(prediction_type, target, noisy, steps)
    assert torch.allclose(recovered_clean, clean, atol=1e-4)
    assert torch.allclose(recovered_noise, noise, atol=1e-4)


def test_inference_timesteps_are_descending_and_bounded() -> None:
    schedule = DiffusionSchedule(num_train_timesteps=100)
    steps = schedule.inference_timesteps(10, torch.device("cpu"))
    assert steps.shape == (10,)
    assert bool((steps.diff() < 0).all())
    assert int(steps.max()) < 100 and int(steps.min()) >= 0
    with pytest.raises(ValueError, match="exceeds"):
        schedule.inference_timesteps(500, torch.device("cpu"))


@pytest.mark.parametrize("refiner_type", ["diffusion_unet", "diffusion_transformer"])
def test_one_diffusion_training_step_and_sampling_run(refiner_type: str) -> None:
    model = build_refiner_model(refiner_type, height=16, width=16)
    rollout = torch.randn(2, model.packing.num_channels, 16, 16)
    target = rollout + 0.05 * torch.randn_like(rollout)
    generator = torch.Generator().manual_seed(0)
    out = model.training_step(
        rollout,
        target,
        forecast_lead_time=torch.tensor([24.0, 72.0]),
        lead_index=torch.tensor([0, 1]),
        generator=generator,
    )
    assert out.process_time.shape == (2,)
    assert bool((out.process_time < model.refiner.num_train_timesteps).all())
    out.losses["total_loss"].backward()

    sampled = model.refiner.sample_residual(
        model.build_conditioning(rollout),
        forecast_lead_time=torch.tensor([24.0, 72.0]),
        generator=torch.Generator().manual_seed(2),
        num_steps=3,
    )
    assert sampled.shape == rollout.shape
    assert torch.isfinite(sampled).all()


def test_ddim_eta_zero_is_deterministic_for_a_fixed_initial_state() -> None:
    model = build_refiner_model("diffusion_unet", height=16, width=16)
    conditioning = torch.randn(1, model.conditioning_channels(), 16, 16)
    lead = torch.tensor([24.0])
    first = model.refiner.sample_residual(
        conditioning,
        forecast_lead_time=lead,
        generator=torch.Generator().manual_seed(9),
        num_steps=4,
    )
    second = model.refiner.sample_residual(
        conditioning,
        forecast_lead_time=lead,
        generator=torch.Generator().manual_seed(9),
        num_steps=4,
    )
    assert torch.equal(first, second)


def test_ddim_eta_one_injects_noise() -> None:
    deterministic = build_refiner_model(
        "diffusion_unet", height=16, width=16, diffusion={"eta": 0.0}
    )
    stochastic = build_refiner_model(
        "diffusion_unet", height=16, width=16, diffusion={"eta": 1.0, "sampler": "ddpm"}
    )
    stochastic.refiner.load_state_dict(deterministic.refiner.state_dict())
    conditioning = torch.randn(1, deterministic.conditioning_channels(), 16, 16)
    lead = torch.tensor([24.0])
    a = deterministic.refiner.sample_residual(
        conditioning,
        forecast_lead_time=lead,
        generator=torch.Generator().manual_seed(1),
        num_steps=4,
    )
    b = stochastic.refiner.sample_residual(
        conditioning,
        forecast_lead_time=lead,
        generator=torch.Generator().manual_seed(1),
        num_steps=4,
    )
    assert not torch.equal(a, b)


# --------------------------------------------------------------------------
# Ensembles and process-time separation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refiner_type",
    ["flow_matching_transformer", "diffusion_unet", "diffusion_transformer"],
)
def test_ensembles_are_reproducible_and_ordered(refiner_type: str) -> None:
    model = build_refiner_model(refiner_type, height=16, width=16)
    rollout = torch.randn(2, model.packing.num_channels, 16, 16)
    lead = torch.tensor([24.0, 72.0])
    first = model.refine(rollout, forecast_lead_time=lead, ensemble_size=3, seed=42)
    second = model.refine(rollout, forecast_lead_time=lead, ensemble_size=3, seed=42)
    assert first.members.shape == (2, 3, model.packing.num_channels, 16, 16)
    assert torch.equal(first.members, second.members)
    assert torch.equal(first.member_residuals, second.member_residuals)
    assert first.ensemble_spread.shape == rollout.shape
    assert torch.isfinite(first.ensemble_mean).all()


def test_process_time_and_forecast_lead_time_use_separate_embeddings() -> None:
    model = build_refiner_model("diffusion_transformer", height=16, width=16)
    embedding = model.refiner.net.cond_embed
    assert isinstance(embedding.process_time_embed, ProcessTimeEmbedding)
    assert isinstance(embedding.lead_time_embed, LeadTimeEmbedding)
    assert embedding.process_time_embed is not embedding.lead_time_embed
    process_params = {id(p) for p in embedding.process_time_embed.parameters()}
    lead_params = {id(p) for p in embedding.lead_time_embed.parameters()}
    assert process_params.isdisjoint(lead_params)


def test_lead_time_embedding_is_identity_at_construction() -> None:
    embedding = LeadTimeEmbedding(16, scale_hours=72.0)
    out = embedding(torch.tensor([0.0, 24.0, 72.0]))
    assert torch.count_nonzero(out) == 0


def test_lead_time_embedding_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        LeadTimeEmbedding(16, scale_hours=0.0)


def test_changing_only_the_lead_time_changes_the_output_once_trained() -> None:
    model = build_refiner_model(
        "diffusion_transformer", height=16, width=16, transformer={"zero_init_output": False}
    )
    # Conditioning paths (adaLN, lead-time MLP, output projection) are
    # zero-initialised so a fresh refiner is the identity; perturb them to
    # emulate a trained model.
    torch.manual_seed(0)
    with torch.no_grad():
        for param in model.refiner.parameters():
            param.add_(0.05 * torch.randn_like(param))
    conditioning = torch.randn(1, model.conditioning_channels(), 16, 16)
    state = torch.randn(1, model.packing.num_channels, 16, 16)
    process = torch.tensor([10.0])
    short = model.refiner.net(state, conditioning, process, torch.tensor([6.0]))
    long = model.refiner.net(state, conditioning, process, torch.tensor([72.0]))
    assert not torch.equal(short, long)
