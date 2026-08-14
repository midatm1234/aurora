"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Regression coverage shared by legacy and unified temporal Mamba paths."""

from __future__ import annotations

import pytest
import torch
from finetune.flow_refine import AuroraFlowRefine
from finetune.mamba_temporal import SelectiveSSM
from finetune.refinement.two_phase import build_two_phase_refiner

from tests.refinement_fixtures import build_packing, refinement_config


def _legacy_wrapper() -> AuroraFlowRefine:
    return AuroraFlowRefine(
        base=torch.nn.Identity(),
        target_surf_vars=("tcno2",),
        target_atmos_vars=("no2",),
        hidden=4,
        time_dim=8,
        lon_periodic=False,
        temporal_enabled=True,
        temporal_channels=4,
        temporal_state=2,
        temporal_layers=1,
        temporal_conv=2,
        temporal_expand=1,
    )


def _unified_config(*, temporal: bool | None) -> dict:
    config = refinement_config("diffusion_transformer")
    config["model"]["refinement"]["diffusion"].update(
        {"training_timesteps": 8, "inference_steps": 1}
    )
    if temporal is not None:
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


def _unified_model(*, temporal: bool | None):
    model = build_two_phase_refiner(
        None,
        build_packing(
            8,
            8,
            lead_times_hours=(12.0, 24.0, 36.0, 48.0, 60.0, 72.0),
        ),
        _unified_config(temporal=temporal),
    )
    model.initialize_refiner(model.conditioning_channels())
    return model


def test_legacy_flow_temporal_forward_backward_and_checkpoint_round_trip() -> None:
    model = _legacy_wrapper()
    assert model.has_temporal
    assert model.temporal is not None
    with torch.no_grad():
        model.temporal.surf_heads["tcno2"].decoder.weight.fill_(0.1)
        model.temporal.atmos_heads["no2"].decoder.weight.fill_(0.1)

    surface = torch.randn(2, 6, 5, 7)
    atmosphere = torch.randn(2, 6, 3, 5, 7)
    surface_correction = model.temporal_residual(surface, "tcno2", "surf")
    atmosphere_correction = model.temporal_residual(atmosphere, "no2", "atmos")
    loss = surface_correction.square().mean() + atmosphere_correction.square().mean()
    loss.backward()

    assert surface_correction.shape == surface.shape
    assert atmosphere_correction.shape == atmosphere.shape
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.temporal.parameters()
    )

    state = model.state_dict()
    restored = _legacy_wrapper()
    restored.load_state_dict(state, strict=True)
    for key, value in state.items():
        assert torch.equal(restored.state_dict()[key], value)


def test_unified_temporal_omission_preserves_disabled_state_and_numerics() -> None:
    implicit = _unified_model(temporal=None)
    explicit = _unified_model(temporal=False)

    assert not implicit.has_temporal
    assert implicit.temporal is None
    assert explicit.temporal is None
    assert not any(key.startswith("temporal.") for key in implicit.state_dict())
    assert set(implicit.state_dict()) == set(explicit.state_dict())
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    enabled = _unified_model(temporal=True)
    assert enabled.refiner is not None and implicit.refiner is not None
    enabled.refiner.load_state_dict(implicit.refiner.state_dict(), strict=True)

    rollout = torch.randn(1, implicit.packing.num_channels, 8, 8)
    implicit.eval()
    explicit.eval()
    implicit_out = implicit.refine(
        rollout,
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=2,
        seed=19,
        num_steps=1,
    )
    explicit_out = explicit.refine(
        rollout,
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=2,
        seed=19,
        num_steps=1,
    )
    enabled.eval()
    enabled_out = enabled.refine(
        rollout,
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=2,
        seed=19,
        num_steps=1,
    )
    for field in (
        "member_residuals",
        "members",
        "ensemble_mean",
        "ensemble_spread",
        "refined_normalized",
        "refined_physical",
    ):
        assert torch.equal(getattr(implicit_out, field), getattr(explicit_out, field))
        assert torch.equal(getattr(implicit_out, field), getattr(enabled_out, field))


def test_unified_temporal_objective_is_target_independent_and_detached() -> None:
    model = _unified_model(temporal=True)
    assert model.temporal is not None
    assert model.refiner is not None
    model.train()

    batch, steps = 2, 6
    folded = batch * steps
    rollout = torch.randn(
        folded,
        model.packing.num_channels,
        8,
        8,
        requires_grad=True,
    )
    target_a = rollout.detach() + 0.5
    target_b = rollout.detach() - 0.75
    valid = torch.ones_like(rollout, dtype=torch.bool)
    lead_index = torch.arange(steps).repeat_interleave(batch)
    lead_hours = (torch.arange(1, steps + 1, dtype=torch.float32) * 12.0).repeat_interleave(batch)
    conditioning = model.build_conditioning(rollout.detach())

    seen_sequences = []
    hook = model.temporal.core.surf_heads["gtco3"].register_forward_pre_hook(
        lambda _module, args: seen_sequences.append(args[0].detach().clone())
    )
    generator_a = torch.Generator(device=rollout.device).manual_seed(29)
    generator_b = torch.Generator(device=rollout.device).manual_seed(29)
    loss_a, _ = model.temporal_training_loss(
        rollout,
        target_a,
        valid_mask=valid,
        forecast_lead_time=lead_hours,
        lead_index=lead_index,
        conditioning=conditioning,
        generator=generator_a,
    )
    loss_b, _ = model.temporal_training_loss(
        rollout,
        target_b,
        valid_mask=valid,
        forecast_lead_time=lead_hours,
        lead_index=lead_index,
        conditioning=conditioning,
        generator=generator_b,
    )
    hook.remove()

    assert len(seen_sequences) == 2
    assert seen_sequences[0].shape == (batch, steps, 8, 8)
    assert torch.equal(seen_sequences[0], seen_sequences[1])
    assert not torch.equal(loss_a.detach(), loss_b.detach())

    optimizer = torch.optim.Adam(model.temporal.parameters(), lr=1.0e-2)
    loss_a.backward()
    assert rollout.grad is None
    assert all(parameter.grad is None for parameter in model.refiner.parameters())
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.temporal.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    non_decoder_before = {
        name: parameter.detach().clone()
        for name, parameter in model.temporal.named_parameters()
        if ".decoder." not in name
    }
    generator_c = torch.Generator(device=rollout.device).manual_seed(31)
    loss_c, _ = model.temporal_training_loss(
        rollout,
        target_a,
        valid_mask=valid,
        forecast_lead_time=lead_hours,
        lead_index=lead_index,
        conditioning=conditioning,
        generator=generator_c,
    )
    loss_c.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in model.temporal.named_parameters()
        if ".decoder." not in name
    )
    optimizer.step()
    assert any(
        not torch.equal(non_decoder_before[name], parameter.detach())
        for name, parameter in model.temporal.named_parameters()
        if name in non_decoder_before
    )


def test_unified_temporal_loss_weights_variables_equally(monkeypatch) -> None:
    model = _unified_model(temporal=True)
    assert model.temporal is not None and model.refiner is not None
    rollout = torch.zeros(2, model.packing.num_channels, 8, 8)
    target = torch.empty_like(rollout)
    target[:, 0] = 1.0
    target[:, 1:] = 3.0
    valid = torch.ones_like(rollout, dtype=torch.bool)
    lead_index = torch.tensor([0, 1])
    lead_hours = torch.tensor([12.0, 24.0])
    conditioning = model.build_conditioning(rollout)
    seen = {}

    def sample_residual(conditioning_arg, **kwargs):
        seen["conditioning_ptr"] = conditioning_arg.data_ptr()
        seen["num_steps"] = kwargs["num_steps"]
        return torch.zeros_like(rollout)

    monkeypatch.setattr(model.refiner, "sample_residual", sample_residual)
    loss, diagnostics = model.temporal_training_loss(
        rollout,
        target,
        valid_mask=valid,
        forecast_lead_time=lead_hours,
        lead_index=lead_index,
        conditioning=conditioning,
    )

    assert seen == {"conditioning_ptr": conditioning.data_ptr(), "num_steps": None}
    torch.testing.assert_close(diagnostics["temporal_loss/gtco3"], torch.tensor(1.0))
    torch.testing.assert_close(diagnostics["temporal_loss/go3"], torch.tensor(9.0))
    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_obsolete_lazy_mamba_state_loads_strictly_with_warning() -> None:
    model = _legacy_wrapper()
    state = model.state_dict()
    ssm_prefix = "temporal.surf_heads.tcno2.blocks.0.ssm."
    assert f"{ssm_prefix}in_proj.weight" in state
    obsolete_key = f"{ssm_prefix}_mamba_impl.in_proj.weight"
    state[obsolete_key] = torch.randn(1)

    restored = _legacy_wrapper()
    with pytest.warns(RuntimeWarning, match="obsolete lazy mamba_ssm"):
        restored.load_state_dict(state, strict=True)

    restored_state = restored.state_dict()
    assert obsolete_key not in restored_state
    for key, value in model.state_dict().items():
        assert torch.equal(restored_state[key], value)


def test_unified_temporal_member_histories_preserve_chunking_parity() -> None:
    model = _unified_model(temporal=True)
    assert model.temporal is not None
    with torch.no_grad():
        for heads in (
            model.temporal.core.surf_heads,
            model.temporal.core.atmos_heads,
        ):
            for head in heads.values():
                head.decoder.weight.zero_()
                head.decoder.bias.fill_(0.25)
        # The spatial refiner is identity-at-initialization, so every ensemble
        # member would otherwise draw exactly the same (zero) residual and the
        # member-separation assertion below would be vacuous. Perturbing the
        # output projection restores genuine member-to-member spread without
        # touching the chunking logic under test.
        refiner_net = model.refiner.net
        refiner_net.out_proj.weight.add_(torch.randn_like(refiner_net.out_proj.weight) * 0.1)
        refiner_net.out_proj.bias.add_(torch.randn_like(refiner_net.out_proj.bias) * 0.1)
    model.eval()

    rollout = torch.zeros(1, model.packing.num_channels, 8, 8)
    serial_history: list[torch.Tensor] = []
    batched_history: list[torch.Tensor] = []
    seen_shapes = []
    hook = model.temporal.core.surf_heads["gtco3"].register_forward_pre_hook(
        lambda _module, args: seen_shapes.append(tuple(args[0].shape))
    )
    for step in range(2):
        kwargs = {
            "forecast_lead_time": torch.tensor([12.0 * (step + 1)]),
            "ensemble_size": 3,
            "seed": 101 + step,
            # Stochastic diffusion sampling requires a genuine reverse
            # trajectory. One step is reserved for the deterministic mean
            # product and is rejected by the sampler contract.
            "num_steps": 2,
        }
        serial = model.refine(
            rollout,
            chunk_size=1,
            temporal_history=serial_history,
            **kwargs,
        )
        batched = model.refine(
            rollout,
            chunk_size=3,
            temporal_history=batched_history,
            **kwargs,
        )
        for field in (
            "member_residuals",
            "members",
            "ensemble_mean",
            "ensemble_spread",
            "refined_normalized",
            "refined_physical",
        ):
            assert torch.equal(getattr(serial, field), getattr(batched, field))
        assert len(serial_history) == len(batched_history) == step + 1
        for serial_frame, batched_frame in zip(serial_history, batched_history, strict=True):
            assert torch.equal(serial_frame, batched_frame)
    hook.remove()

    assert seen_shapes == [
        (3, 1, 8, 8),
        (3, 1, 8, 8),
        (3, 2, 8, 8),
        (3, 2, 8, 8),
    ]
    assert not torch.equal(serial_history[0][:, 0], serial_history[0][:, 1])


def test_selective_scan_bfloat16_forward_backward_is_finite() -> None:
    module = SelectiveSSM(d_model=4, d_state=2, d_conv=2, expand=1).to(dtype=torch.bfloat16)
    sequence = torch.randn(6, 6, 4, dtype=torch.bfloat16, requires_grad=True)
    output = module(sequence)
    loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad.float()).all()
        for parameter in module.parameters()
    )
