"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Target space, variable/pressure-level packing and forecast-timing contracts.
"""

from __future__ import annotations

import math

import pytest
import torch
from finetune.refinement.integration import LeadStepBuffer
from finetune.refinement.packing import FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace

from tests.refinement_fixtures import DummySpec, build_packing, build_refiner_model

# --------------------------------------------------------------------------
# Variable / pressure-level packing
# --------------------------------------------------------------------------


def test_channel_layout_is_surface_then_atmosphere_by_level() -> None:
    packing = build_packing()
    assert [(c.aurora_name, c.level) for c in packing.channels] == [
        ("gtco3", None),
        ("go3", 500.0),
        ("go3", 850.0),
    ]
    assert packing.num_channels == 3
    assert packing.variables == ("gtco3", "go3")
    assert packing.levels_for("go3") == (500.0, 850.0)
    assert packing.index_of("go3", 850.0) == 2


def test_pack_unpack_round_trip_preserves_everything() -> None:
    packing = build_packing(height=7, width=11)
    fields = {
        "gtco3": torch.randn(3, 7, 11),
        "go3": torch.randn(3, 2, 7, 11),
    }
    packed = packing.pack(fields)
    assert packed.shape == (3, 3, 7, 11)
    restored = packing.unpack(packed)
    assert set(restored) == set(fields)
    for name, tensor in fields.items():
        assert torch.equal(restored[name], tensor)
        assert restored[name].shape == tensor.shape
    # Metadata survives untouched.
    assert [c.units for c in packing.channels] == ["kg m-2", "kg kg-1", "kg kg-1"]
    assert len(packing.lat) == 7 and len(packing.lon) == 11


def test_pack_rejects_wrong_level_count() -> None:
    packing = build_packing()
    with pytest.raises(ValueError, match="levels"):
        packing.pack({"gtco3": torch.randn(2, 20, 30), "go3": torch.randn(2, 3, 20, 30)})


def test_pack_rejects_missing_variable() -> None:
    packing = build_packing()
    with pytest.raises(KeyError, match="missing"):
        packing.pack({"gtco3": torch.randn(2, 20, 30)})


def test_channel_stats_follow_the_declared_levels() -> None:
    packing = build_packing()
    stds = packing.stds().reshape(-1).tolist()
    assert stds == pytest.approx([2.0e-4, 1.0e-7, 2.0e-7])
    means = packing.means().reshape(-1).tolist()
    assert means == pytest.approx([1.0e-3, 0.0, 0.0])


def test_atmospheric_specs_without_levels_use_the_configured_axis() -> None:
    packing = FieldPacking.from_specs(
        [DummySpec("go3", "go3", "atmos")], atmos_levels=[100.0, 200.0, 300.0]
    )
    assert packing.levels_for("go3") == (100.0, 200.0, 300.0)
    assert packing.num_channels == 3


# --------------------------------------------------------------------------
# Normalized target space
# --------------------------------------------------------------------------


def test_encode_decode_round_trip() -> None:
    packing = build_packing(height=5, width=6)
    space = NormalizedTargetSpace(packing)
    physical = torch.stack(
        [
            torch.full((5, 6), 1.2e-3),
            torch.full((5, 6), 3.0e-7),
            torch.full((5, 6), -1.0e-7),
        ]
    ).unsqueeze(0)
    assert torch.allclose(space.decode(space.encode(physical)), physical, atol=0, rtol=1e-6)


def test_decode_of_rollout_norm_recovers_the_rollout() -> None:
    packing = build_packing(height=4, width=4)
    space = NormalizedTargetSpace(packing)
    rollout = torch.randn(2, 3, 4, 4)
    assert torch.allclose(space.encode(space.decode(rollout)), rollout, atol=1e-6)


def test_residual_is_built_added_and_decoded_in_one_space() -> None:
    packing = build_packing(height=4, width=5)
    space = NormalizedTargetSpace(packing)
    torch.manual_seed(0)
    rollout_physical = torch.randn(2, 3, 4, 5).abs() * 1e-4
    target_physical = rollout_physical * 1.1
    rollout_norm = space.encode(rollout_physical)

    residual, mask = space.residual_target(target_physical, rollout_norm)
    assert bool(mask.all())
    # residual + rollout in normalized space decodes back to the target.
    refined_norm, refined_physical = space.reconstruct(rollout_norm, residual)
    # Compared with a relative tolerance: normalized values span several orders
    # of magnitude because the per-level scales do.
    assert torch.allclose(refined_norm, space.encode(target_physical), rtol=1e-5, atol=1e-3)
    assert torch.allclose(refined_physical, target_physical, rtol=1e-5, atol=1e-12)


def test_residual_from_already_normalized_target_matches_the_physical_path() -> None:
    packing = build_packing(height=3, width=3)
    space = NormalizedTargetSpace(packing)
    torch.manual_seed(0)
    rollout_physical = torch.randn(2, 3, 3, 3) * 1e-4
    target_physical = torch.randn(2, 3, 3, 3) * 1e-4
    rollout_norm = space.encode(rollout_physical)
    target_norm = space.encode(target_physical)

    from_physical, _ = space.residual_target(target_physical, rollout_norm)
    from_normalized, _ = space.residual_target_from_normalized(target_norm, rollout_norm)
    assert torch.allclose(from_physical, from_normalized, rtol=1e-5, atol=1e-3)


def test_masked_cells_get_zero_residual_and_stay_nan() -> None:
    packing = build_packing(height=3, width=3)
    space = NormalizedTargetSpace(packing)
    rollout_norm = torch.randn(1, 3, 3, 3)
    target = torch.randn(1, 3, 3, 3)
    target[0, 1, 0, 0] = float("nan")

    residual, mask = space.residual_target(target, rollout_norm)
    assert mask[0, 1, 0, 0].item() is False
    assert float(residual[0, 1, 0, 0]) == 0.0
    assert torch.isfinite(residual).all()

    _, refined = space.reconstruct(rollout_norm, residual, target_mask=mask)
    assert math.isnan(float(refined[0, 1, 0, 0]))
    assert torch.isfinite(refined[mask]).all()


def test_constraints_are_applied_once_and_are_idempotent() -> None:
    packing = build_packing(height=3, width=3)
    space = NormalizedTargetSpace(packing, nonnegative_variables=["go3"])
    physical = torch.full((1, 3, 3, 3), -5.0)
    once = space.apply_physical_constraints(physical)
    twice = space.apply_physical_constraints(once)
    assert torch.equal(once, twice)
    # Only the configured variable is clamped; the surface channel is untouched.
    assert float(once[0, 0, 0, 0]) == -5.0
    assert float(once[0, 1, 0, 0]) == 0.0


def test_reconstruct_rejects_shape_mismatch() -> None:
    packing = build_packing(height=3, width=3)
    space = NormalizedTargetSpace(packing)
    with pytest.raises(ValueError, match="does not match"):
        space.reconstruct(torch.zeros(1, 3, 3, 3), torch.zeros(1, 3, 4, 4))


def test_target_space_rejects_wrong_channel_count() -> None:
    packing = build_packing(height=3, width=3)
    space = NormalizedTargetSpace(packing)
    with pytest.raises(ValueError, match="channels"):
        space.encode(torch.zeros(1, 5, 3, 3))


# --------------------------------------------------------------------------
# Forecast timing / rollout-step alignment
# --------------------------------------------------------------------------


def test_lead_step_buffer_preserves_rollout_order() -> None:
    packing = build_packing(height=4, width=4)
    buffer = LeadStepBuffer(packing)
    batch = 2
    per_lead = {}
    for lead, hours in ((1, 24.0), (2, 48.0), (3, 72.0)):
        surf = torch.full((batch, 4, 4), float(lead))
        atmos = torch.full((batch, 2, 4, 4), float(lead))
        per_lead[lead] = (surf, atmos)
        buffer.add(
            lead,
            "gtco3",
            rollout_normalized=surf,
            target_normalized=surf + 1,
            lead_hours=hours,
        )
        buffer.add(
            lead,
            "go3",
            rollout_normalized=atmos,
            target_normalized=atmos + 1,
            lead_hours=hours,
        )

    assert buffer.leads == [1, 2, 3]
    assert buffer.is_complete()
    rollout, target, mask, hours, index = buffer.pack()
    assert rollout.shape == (3 * batch, 3, 4, 4)
    # Ascending rollout-step order, each block carrying its own lead time.
    assert hours.tolist() == [24.0, 24.0, 48.0, 48.0, 72.0, 72.0]
    assert index.tolist() == [0, 0, 1, 1, 2, 2]
    for position, lead in enumerate([1, 2, 3]):
        block = rollout[position * batch : (position + 1) * batch]
        assert torch.equal(block[:, 0], per_lead[lead][0])
        assert torch.equal(block[:, 1:], per_lead[lead][1])
    # The target of step n is the target for the SAME valid time (no off-by-one).
    assert torch.equal(target, rollout + 1)
    assert bool(mask.all())


def test_incomplete_buffer_is_not_used() -> None:
    packing = build_packing(height=4, width=4)
    buffer = LeadStepBuffer(packing)
    buffer.add(
        1,
        "gtco3",
        rollout_normalized=torch.zeros(1, 4, 4),
        target_normalized=torch.zeros(1, 4, 4),
        lead_hours=24.0,
    )
    assert buffer.is_complete() is False


def test_lead_times_do_not_interact_through_the_refiner() -> None:
    """Changing one lead-time entry must not change another's residual."""
    model = build_refiner_model(
        "flow_matching_transformer",
        height=16,
        width=16,
        transformer={"zero_init_output": False},
    )
    packing = model.packing
    rollout = torch.randn(4, packing.num_channels, 16, 16)
    lead = torch.tensor([24.0, 24.0, 72.0, 72.0])

    baseline = model.refine(rollout, forecast_lead_time=lead, ensemble_size=1, seed=3)
    perturbed_rollout = rollout.clone()
    perturbed_rollout[2] += 5.0
    perturbed = model.refine(perturbed_rollout, forecast_lead_time=lead, ensemble_size=1, seed=3)
    untouched = [0, 1, 3]
    assert torch.equal(baseline.member_residuals[untouched], perturbed.member_residuals[untouched])
    assert not torch.equal(baseline.member_residuals[2], perturbed.member_residuals[2])


def test_conditioning_never_contains_the_target() -> None:
    model = build_refiner_model("diffusion_unet", height=8, width=8)
    rollout = torch.randn(2, model.packing.num_channels, 8, 8)
    conditioning = model.build_conditioning(rollout)
    # rollout channels + one mask channel; nothing target-derived.
    assert conditioning.shape[1] == model.conditioning_channels()
    assert torch.equal(conditioning[:, : rollout.shape[1]], rollout)


def test_refinement_does_not_mutate_the_deterministic_rollout() -> None:
    model = build_refiner_model("diffusion_unet", height=8, width=8)
    rollout = torch.randn(2, model.packing.num_channels, 8, 8)
    reference = rollout.clone()
    out = model.refine(
        rollout,
        forecast_lead_time=torch.tensor([24.0, 48.0]),
        ensemble_size=2,
        seed=5,
    )
    assert torch.equal(rollout, reference)
    assert torch.equal(out.deterministic_normalized, reference)
