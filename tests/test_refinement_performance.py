"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Performance-parity contracts.

Every optimization in the refinement package must be numerically neutral.
These tests compare the reference path with the optimized path using identical
initial noise, seeds, generator states, ensemble ordering, diffusion schedule
and timesteps, DDIM ``eta``, flow time grid and flow solver, and they compare
individual ensemble members rather than only the ensemble mean.
"""

from __future__ import annotations

import pytest
import torch
from finetune.refinement.backbones import patchify_2d, unpatchify_2d
from finetune.refinement.cache import RolloutCache, build_cache_key

from tests.refinement_fixtures import build_refiner_model

REFINERS = [
    "flow_matching_unet",
    "flow_matching_transformer",
    "diffusion_unet",
    "diffusion_transformer",
]


def build_model(refiner_type: str, **kwargs):
    """A refiner whose output is not identically zero.

    Fresh refiners are zero-initialised (identity at construction), which would
    make every parity comparison trivially true.
    """
    overrides = {
        "unet": {"zero_init_output": False},
        "transformer": {"zero_init_output": False},
    }
    for key, value in kwargs.items():
        if key in overrides and isinstance(value, dict):
            overrides[key] = {**overrides[key], **value}
        else:
            overrides[key] = value
    model = build_refiner_model(refiner_type, **overrides)
    if refiner_type == "flow_matching_unet":
        # The legacy head has its own mandatory identity-at-init projection,
        # independent of UNetRefinerConfig.zero_init_output. Make it nonzero so
        # these parity checks exercise source-noise generator propagation.
        assert model.refiner is not None
        legacy = model.refiner.legacy
        with torch.no_grad():
            for head in (*legacy.surf_flow.values(), *legacy.atmos_flow.values()):
                head.out.weight.fill_(0.01)
    return model


# --------------------------------------------------------------------------
# Ensemble batching
# --------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_serial_and_batched_ensemble_generation_are_bitwise_identical(
    refiner_type: str,
) -> None:
    model = build_model(refiner_type, height=16, width=16)
    rollout = torch.randn(2, model.packing.num_channels, 16, 16)
    lead = torch.tensor([24.0, 72.0])

    serial = model.refine(rollout, forecast_lead_time=lead, ensemble_size=4, seed=7, chunk_size=1)
    batched = model.refine(rollout, forecast_lead_time=lead, ensemble_size=4, seed=7, chunk_size=4)
    partial = model.refine(rollout, forecast_lead_time=lead, ensemble_size=4, seed=7, chunk_size=3)
    # Compare individual members, not only the ensemble mean.
    for member in range(4):
        assert torch.equal(serial.member_residuals[:, member], batched.member_residuals[:, member])
        assert torch.equal(serial.member_residuals[:, member], partial.member_residuals[:, member])
    assert torch.equal(serial.ensemble_mean, batched.ensemble_mean)
    assert torch.equal(serial.ensemble_spread, batched.ensemble_spread)


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_member_order_is_stable(refiner_type: str) -> None:
    model = build_model(refiner_type, height=16, width=16)
    rollout = torch.randn(1, model.packing.num_channels, 16, 16)
    lead = torch.tensor([24.0])
    two = model.refine(rollout, forecast_lead_time=lead, ensemble_size=2, seed=3)
    four = model.refine(rollout, forecast_lead_time=lead, ensemble_size=4, seed=3)
    # The first two members of a 4-member draw are the 2-member draw.
    assert torch.equal(two.member_residuals, four.member_residuals[:, :2])


def test_ensemble_statistics_ignore_masked_cells() -> None:
    model = build_model("diffusion_unet", height=8, width=8)
    rollout = torch.randn(1, model.packing.num_channels, 8, 8)
    mask = torch.ones_like(rollout, dtype=torch.bool)
    mask[0, 0, 0, 0] = False
    out = model.refine(
        rollout,
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=3,
        seed=1,
        target_mask=mask,
    )
    assert torch.isnan(out.ensemble_mean[0, 0, 0, 0])
    assert torch.isfinite(out.ensemble_mean[mask]).all()
    assert torch.isfinite(out.ensemble_spread[mask]).all()


def test_ensemble_statistics_use_float32_accumulation() -> None:
    model = build_model("diffusion_unet", height=8, width=8)
    rollout = torch.randn(1, model.packing.num_channels, 8, 8)
    out = model.refine(rollout, forecast_lead_time=torch.tensor([24.0]), ensemble_size=4, seed=2)
    reference = out.members.float().mean(dim=1)
    assert torch.allclose(out.ensemble_mean.float(), reference, atol=0)


# --------------------------------------------------------------------------
# Deterministic sampling paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("refiner_type", REFINERS)
def test_identical_seeds_reproduce_identical_members(refiner_type: str) -> None:
    model = build_model(refiner_type, height=16, width=16)
    rollout = torch.randn(2, model.packing.num_channels, 16, 16)
    lead = torch.tensor([24.0, 72.0])
    a = model.refine(rollout, forecast_lead_time=lead, ensemble_size=3, seed=11)
    b = model.refine(rollout, forecast_lead_time=lead, ensemble_size=3, seed=11)
    c = model.refine(rollout, forecast_lead_time=lead, ensemble_size=3, seed=12)
    assert torch.equal(a.members, b.members)
    assert not torch.equal(a.members, c.members)


def test_step_counts_are_never_silently_reduced() -> None:
    model = build_model("diffusion_unet", height=8, width=8, diffusion={"inference_steps": 4})
    assert model.refinement_config.diffusion.inference_steps == 4
    calls = {"count": 0}
    original = model.refiner.net.forward

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    model.refiner.net.forward = counting  # type: ignore[method-assign]
    model.refine(
        rollout_normalized := torch.randn(1, model.packing.num_channels, 8, 8),
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=1,
        seed=1,
    )
    assert rollout_normalized is not None
    assert calls["count"] == 4


# --------------------------------------------------------------------------
# Vectorised helpers
# --------------------------------------------------------------------------


def test_vectorised_patchify_matches_an_explicit_loop() -> None:
    x = torch.randn(2, 3, 8, 12)
    tokens, grid_h, grid_w = patchify_2d(x, 4, 4)
    for batch in range(2):
        for index in range(grid_h * grid_w):
            row, col = index // grid_w, index % grid_w
            block = x[batch, :, row * 4 : row * 4 + 4, col * 4 : col * 4 + 4].reshape(-1)
            assert torch.equal(tokens[batch, index], block)
    assert torch.equal(unpatchify_2d(tokens, 3, grid_h, grid_w, 4, 4), x)


def test_diffusion_schedule_buffers_are_not_persisted() -> None:
    """Schedules are a deterministic function of the config, so they stay out
    of the checkpoint (smaller payloads, no stale coefficients)."""
    model = build_model("diffusion_unet", height=8, width=8)
    keys = list(model.refiner.state_dict())
    assert not any("schedule.betas" in key for key in keys)
    assert not any("alphas_cumprod" in key for key in keys)


def test_positional_encoding_cache_is_reused() -> None:
    model = build_model(
        "diffusion_transformer",
        height=16,
        width=16,
        transformer={"positional_encoding": "sincos_2d"},
    )
    net = model.refiner.net
    first = net._positional(4, 4, torch.device("cpu"), torch.float32)
    second = net._positional(4, 4, torch.device("cpu"), torch.float32)
    assert first is second


# --------------------------------------------------------------------------
# Rollout cache
# --------------------------------------------------------------------------


def _key(**overrides):
    params = dict(
        aurora_fingerprint="abc123",
        dataset="cams",
        split="train",
        init_time="2024-01-01T00",
        valid_time="2024-01-02T00",
        lead_time_hours=24.0,
        rollout_interval_hours=6.0,
        input_history_steps=2,
        variables=["gtco3", "go3"],
        levels=[500.0, 850.0],
        domain=[-90.0, 90.0, 0.0, 360.0],
        normalization="aurora_location_scale",
    )
    params.update(overrides)
    return build_cache_key(**params)


def test_cache_round_trip(tmp_path) -> None:
    cache = RolloutCache(tmp_path)
    key = _key()
    tensors = {"rollout": torch.randn(2, 3, 4, 4)}
    cache.store(key, tensors)
    loaded = cache.load(key)
    assert loaded is not None
    assert torch.equal(loaded["rollout"], tensors["rollout"])


@pytest.mark.parametrize(
    "override",
    [
        {"aurora_fingerprint": "different"},
        {"dataset": "other"},
        {"split": "val"},
        {"init_time": "2024-06-01T00"},
        {"valid_time": "2024-06-02T00"},
        {"lead_time_hours": 48.0},
        {"rollout_interval_hours": 12.0},
        {"input_history_steps": 3},
        {"variables": ["gtco3"]},
        {"levels": [500.0]},
        {"domain": [0.0, 60.0, 0.0, 360.0]},
        {"normalization": "other"},
    ],
)
def test_every_semantic_change_misses_the_cache(tmp_path, override) -> None:
    cache = RolloutCache(tmp_path)
    cache.store(_key(), {"rollout": torch.zeros(1)})
    assert cache.load(_key(**override)) is None


def test_cache_validation_rejects_stale_entries() -> None:
    online = {"rollout": torch.zeros(2, 2)}
    stale = {"rollout": torch.full((2, 2), 0.5)}
    RolloutCache.validate({"rollout": torch.zeros(2, 2)}, online, tolerance=1e-6)
    with pytest.raises(RuntimeError, match="stale"):
        RolloutCache.validate(stale, online, tolerance=1e-6)
    with pytest.raises(RuntimeError, match="missing tensor"):
        RolloutCache.validate({}, online)
    with pytest.raises(RuntimeError, match="shape"):
        RolloutCache.validate({"rollout": torch.zeros(3, 3)}, online)


def test_disabled_cache_is_a_no_op(tmp_path) -> None:
    cache = RolloutCache(tmp_path / "unused", enabled=False)
    assert cache.store(_key(), {"a": torch.zeros(1)}) is None
    assert cache.load(_key()) is None
