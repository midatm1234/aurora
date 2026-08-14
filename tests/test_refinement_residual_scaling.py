"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Contract tests for the residual-scaling and forecast-skill objectives added to
the Phase-2 stochastic refiners.

The central test here is :func:`test_synthetic_known_residual_is_learned`: it
builds a problem whose true residual is a *known* function of the conditioning,
with the same small magnitude the real Aurora residual has, and asserts that
every head recovers it with the right sign and scale. That is the property no
training-loss curve can demonstrate.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from finetune.refinement import config as refinement_config
from finetune.refinement.base import build_refiner
from finetune.refinement.backbones import sincos_2d_positional_encoding
from finetune.refinement.config import ConfigValidationError, resolve_refinement_config
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.residual_scaling import ResidualScaler
from finetune.refinement.target_space import NormalizedTargetSpace

#: Every packed head, i.e. everything except the legacy Aurora flow wrapper.
PACKED_HEADS = (
    "diffusion_unet",
    "diffusion_transformer",
    "flow_matching_conv_unet",
    "flow_matching_transformer",
)

HEIGHT, WIDTH = 16, 24
CHANNELS = 3


def _packing(*, lon_periodic: bool = False) -> FieldPacking:
    channels = tuple(
        ChannelSpec(
            index=i,
            aurora_name="go3" if i else "gtco3",
            dataset_name="go3" if i else "gtco3",
            kind="atmos" if i else "surf",
            level=float(500 * i) if i else None,
            level_index=(i - 1) if i else None,
            mean=0.0,
            std=1.0e-7 * (i + 1),
        )
        for i in range(CHANNELS)
    )
    return FieldPacking(
        channels=channels,
        lat=tuple(90.0 - i * (180.0 / (HEIGHT - 1)) for i in range(HEIGHT)),
        lon=tuple(i * (360.0 / WIDTH) for i in range(WIDTH)),
        lon_periodic=lon_periodic,
    )


def _config(head: str, **overrides) -> object:
    block: dict = {
        "enabled": True,
        "type": head,
        "ensemble_size": 1,
        "deterministic_inference": True,
        "unet": {"hidden_channels": 8, "num_levels": 2, "num_residual_blocks": 1},
        "transformer": {
            "patch_size": [4, 4],
            "embedding_dim": 64,
            "num_heads": 4,
            "num_blocks": 2,
        },
        "target_space": {"residual_scaling": "per_channel"},
    }
    if head.startswith("diffusion_"):
        block["diffusion"] = {
            "training_timesteps": 200,
            "inference_steps": 10,
            "prediction_type": "sample",
            "snr_weighting": "auto",
            "timestep_distribution": "auto",
        }
    block.update(overrides)
    return resolve_refinement_config({"model": {"refinement": block}})


def _build(head: str, *, lon_periodic: bool = False, **overrides):
    packing = _packing(lon_periodic=lon_periodic)
    refiner = build_refiner(
        _config(head, **overrides),
        residual_channels=CHANNELS,
        cond_channels=CHANNELS + 1,
        metadata=packing,
    )
    assert refiner is not None
    return refiner, packing


def _make_refiner_eval_ready(refiner) -> None:
    """Fit test-only statistics before exercising production eval guards."""
    if refiner.residual_scaler.is_active and not refiner.residual_scaler.is_ready:
        values = torch.linspace(
            -0.05,
            0.05,
            2 * CHANNELS * HEIGHT * WIDTH,
        ).reshape(2, CHANNELS, HEIGHT, WIDTH)
        refiner.fit_residual_scale(values, torch.ones_like(values, dtype=torch.bool))
    refiner.eval()


# ---------------------------------------------------------------------------
# Residual scaling
# ---------------------------------------------------------------------------


def test_residual_scaler_round_trip_is_exact() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    residual = torch.randn(8, CHANNELS, HEIGHT, WIDTH) * torch.tensor(
        [0.01, 0.05, 0.3]
    ).view(1, -1, 1, 1)
    scaler.fit(residual)
    encoded = scaler.encode(residual)
    torch.testing.assert_close(scaler.decode(encoded), residual, rtol=1e-5, atol=1e-7)


def test_residual_scaler_recovers_per_channel_std() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    stds = torch.tensor([0.01, 0.05, 0.3])
    residual = torch.randn(64, CHANNELS, HEIGHT, WIDTH) * stds.view(1, -1, 1, 1)
    scaler.fit(residual)
    torch.testing.assert_close(scaler.scale.flatten(), stds, rtol=0.1, atol=1e-3)
    torch.testing.assert_close(
        scaler.encode(residual).std(dim=(0, 2, 3)),
        torch.ones(CHANNELS),
        rtol=0.1,
        atol=0.05,
    )


def test_residual_scaler_mask_excludes_invalid_cells() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    residual = torch.randn(16, CHANNELS, HEIGHT, WIDTH) * 0.02
    mask = torch.ones_like(residual, dtype=torch.bool)
    mask[:, :, :, WIDTH // 2 :] = False
    # Poison the masked half; a correct implementation must ignore it entirely.
    residual = torch.where(mask, residual, torch.full_like(residual, 1000.0))
    scaler.fit(residual, mask)
    assert float(scaler.scale.max()) < 0.1


def test_residual_scaler_frozen_in_eval() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    scaler.fit(torch.randn(8, CHANNELS, HEIGHT, WIDTH) * 0.02)
    before = scaler.scale.clone()
    scaler.eval()
    scaler.observe(torch.randn(8, CHANNELS, HEIGHT, WIDTH) * 5.0)
    torch.testing.assert_close(scaler.scale, before)


def test_residual_scaler_none_mode_is_identity() -> None:
    scaler = ResidualScaler(CHANNELS, mode="none")
    scaler.train()
    residual = torch.randn(4, CHANNELS, HEIGHT, WIDTH)
    scaler.fit(residual)
    torch.testing.assert_close(scaler.encode(residual), residual)
    torch.testing.assert_close(scaler.decode(residual), residual)


def test_residual_scaler_none_mode_has_no_persistent_state() -> None:
    scaler = ResidualScaler(CHANNELS, mode="none")
    assert dict(scaler.state_dict()) == {}


def test_residual_scaler_all_masked_batch_does_not_update_state() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    before = {name: value.clone() for name, value in scaler.state_dict().items()}
    scaler.observe(
        torch.randn(2, CHANNELS, 3, 4),
        torch.zeros(2, CHANNELS, 3, 4, dtype=torch.bool),
    )
    after = scaler.state_dict()
    assert after.keys() == before.keys()
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected)


def test_residual_scaler_zero_count_channel_stays_unchanged() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    residual = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0]], [[100.0, 100.0, 100.0, 100.0]], [[2.0, 4.0, 6.0, 8.0]]]]
    )
    mask = torch.ones_like(residual, dtype=torch.bool)
    mask[:, 1] = False
    scaler.fit(residual, mask)

    assert float(scaler.scale[0, 1, 0, 0]) == 1.0
    assert float(scaler.shift[0, 1, 0, 0]) == 0.0
    assert int(scaler.observed_channels[0, 1, 0, 0]) == 0
    assert float(scaler.calibration_count[0, 1, 0, 0]) == 0.0

    previous = scaler.scale.clone()
    second = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0]], [[10.0, 12.0, 14.0, 16.0]], [[0.0, 0.0, 0.0, 0.0]]]]
    )
    second_mask = torch.zeros_like(second, dtype=torch.bool)
    second_mask[:, 1] = True
    scaler.observe(second, second_mask)

    torch.testing.assert_close(scaler.scale[:, (0, 2)], previous[:, (0, 2)])
    torch.testing.assert_close(
        scaler.scale[0, 1, 0, 0], torch.tensor(math.sqrt(5.0)), rtol=1e-6, atol=1e-6
    )
    assert int(scaler.observed_channels[0, 1, 0, 0]) == 1


def _mock_two_rank_sum(monkeypatch, reduce_callback) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_callback)


def test_distributed_residual_scaler_uses_global_sufficient_statistics(
    monkeypatch,
) -> None:
    remote = torch.tensor([2.0, 2.0, 10.0, 40.0, 52.0, 808.0], dtype=torch.float64)

    def add_remote(packed, op=None):
        del op
        packed.add_(remote)

    _mock_two_rank_sum(monkeypatch, add_remote)
    scaler = ResidualScaler(2, mode="per_channel", center=True, warmup_batches=1)
    scaler.train()
    scaler.enable_distributed_calibration()
    local = torch.tensor([[[[0.0, 2.0]], [[10.0, 14.0]]]])
    scaler.observe(local)

    torch.testing.assert_close(
        scaler.shift.flatten(), torch.tensor([3.0, 16.0]), rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        scaler.scale.flatten(),
        torch.tensor([math.sqrt(5.0), math.sqrt(20.0)]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        scaler.calibration_count.flatten(),
        torch.tensor([4.0, 4.0], dtype=torch.float64),
    )
    assert bool(scaler.frozen.item())


def test_distributed_residual_scaler_resume_preserves_partial_calibration(
    monkeypatch,
) -> None:
    def duplicate_rank(packed, op=None):
        del op
        packed.mul_(2.0)

    _mock_two_rank_sum(monkeypatch, duplicate_rank)
    scaler = ResidualScaler(1, mode="per_channel", warmup_batches=2)
    scaler.train()
    scaler.enable_distributed_calibration()
    scaler.observe(torch.tensor([[[[0.0, 2.0]]]]))
    assert not bool(scaler.frozen.item())

    state = {name: value.clone() for name, value in scaler.state_dict().items()}
    restored = ResidualScaler(1, mode="per_channel", warmup_batches=2)
    restored.load_state_dict(state)
    restored.train()
    restored.enable_distributed_calibration()
    for name, expected in state.items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

    restored.observe(torch.tensor([[[[2.0, 4.0]]]]))
    torch.testing.assert_close(
        restored.scale.flatten(), torch.tensor([math.sqrt(2.0)]), rtol=1e-6, atol=1e-6
    )
    assert int(restored.observed_batches.item()) == 2
    assert bool(restored.frozen.item())


def test_residual_scaler_statistics_survive_state_dict() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.train()
    scaler.fit(torch.randn(32, CHANNELS, HEIGHT, WIDTH) * 0.04)
    restored = ResidualScaler(CHANNELS, mode="per_channel")
    restored.load_state_dict(scaler.state_dict())
    torch.testing.assert_close(restored.scale, scaler.scale)
    torch.testing.assert_close(restored.shift, scaler.shift)

def test_active_unfitted_or_unfrozen_scaler_is_rejected_in_eval() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    with pytest.raises(RuntimeError, match="cannot enter eval/inference mode"):
        scaler.eval()

    scaler.fit(
        torch.randn(4, CHANNELS, 3, 4),
        freeze_after_fit=False,
    )
    assert scaler.is_fitted and not scaler.is_ready
    with pytest.raises(RuntimeError, match="cannot enter eval/inference mode"):
        scaler.eval()


def test_manual_fit_can_restore_eval_mode_atomically() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel", center=True)
    # Regression: fit() used to restore eval before marking calibration complete,
    # which recursively triggered the readiness guard.
    scaler.train()
    scaler.fit(torch.randn(5, CHANNELS, 3, 4))
    scaler.eval()
    scaler.fit(torch.randn(5, CHANNELS, 3, 4))

    assert not scaler.training
    assert scaler.is_ready
    assert not scaler.has_exact_training_split_calibration
    encoded = scaler.encode(torch.zeros(1, CHANNELS, 3, 4))
    assert bool(torch.isfinite(encoded).all())


def _fit_exact_in_order(order: tuple[int, ...]) -> ResidualScaler:
    batches = (
        (
            torch.tensor(
                [
                    [
                        [[1.0, 2.0]],
                        [[10.0, float("nan")]],
                        [[-2.0, 4.0]],
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [[True, True]],
                        [[True, True]],
                        [[True, False]],
                    ]
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    [
                        [[3.0, 4.0]],
                        [[14.0, 18.0]],
                        [[6.0, 8.0]],
                    ],
                    [
                        [[5.0, 6.0]],
                        [[22.0, 26.0]],
                        [[10.0, 12.0]],
                    ],
                ]
            ),
            torch.ones(2, CHANNELS, 1, 2, dtype=torch.bool),
        ),
    )
    scaler = ResidualScaler(CHANNELS, mode="per_channel", center=True)
    scaler.train()
    scaler.begin_exact_training_split_calibration()
    for index in order:
        residual, mask = batches[index]
        scaler.observe(residual, mask)
    scaler.finalize_exact_training_split_calibration(
        logical_samples=3,
        packed_examples=3,
        fingerprint=b"x" * 32,
    )
    return scaler


def test_exact_training_split_calibration_is_masked_finite_and_order_independent() -> None:
    forward = _fit_exact_in_order((0, 1))
    reverse = _fit_exact_in_order((1, 0))

    for name in (
        "calibration_count",
        "calibration_sum",
        "calibration_sum_sq",
        "scale",
        "shift",
    ):
        torch.testing.assert_close(getattr(forward, name), getattr(reverse, name))
    torch.testing.assert_close(
        forward.calibration_count.flatten(),
        torch.tensor([6.0, 5.0, 5.0], dtype=torch.float64),
    )
    assert forward.is_ready
    assert forward.has_exact_training_split_calibration
    assert int(forward.calibration_examples.item()) == 3
    assert int(forward.calibration_logical_samples.item()) == 3
    forward.eval()


def test_exact_calibration_metadata_round_trips_and_is_strictly_validated() -> None:
    scaler = _fit_exact_in_order((0, 1))
    restored = ResidualScaler(CHANNELS, mode="per_channel", center=True)
    restored.load_state_dict(scaler.state_dict(), strict=True)
    restored.validate_exact_training_split_calibration(
        logical_samples=3,
        packed_examples=3,
        fingerprint=b"x" * 32,
    )
    restored.eval()
    with pytest.raises(RuntimeError, match="sample count mismatch"):
        restored.validate_exact_training_split_calibration(
            logical_samples=4,
            packed_examples=3,
            fingerprint=b"x" * 32,
        )
    with pytest.raises(RuntimeError, match="packed-example count mismatch"):
        restored.validate_exact_training_split_calibration(
            logical_samples=3,
            packed_examples=4,
            fingerprint=b"x" * 32,
        )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        restored.validate_exact_training_split_calibration(
            logical_samples=3,
            packed_examples=3,
            fingerprint=b"y" * 32,
        )


def test_legacy_frozen_scaler_loads_without_false_exact_provenance() -> None:
    scaler = ResidualScaler(CHANNELS, mode="per_channel")
    scaler.fit(torch.randn(5, CHANNELS, 3, 4))
    provenance = {
        "calibration_complete",
        "calibration_method",
        "calibration_examples",
        "calibration_logical_samples",
        "calibration_fingerprint",
    }
    legacy_state = {
        name: value.clone()
        for name, value in scaler.state_dict().items()
        if name not in provenance
    }
    restored = ResidualScaler(CHANNELS, mode="per_channel")
    restored.load_state_dict(legacy_state, strict=True)
    assert restored.is_ready
    assert not restored.has_exact_training_split_calibration
    restored.eval()


def test_exact_distributed_calibration_reduces_only_once_at_finalize(monkeypatch) -> None:
    calls = []

    def duplicate_rank(packed, op=None):
        del op
        calls.append(packed.clone())
        packed.mul_(2.0)

    _mock_two_rank_sum(monkeypatch, duplicate_rank)
    scaler = ResidualScaler(1, mode="per_channel", center=True)
    scaler.train()
    scaler.begin_exact_training_split_calibration()
    scaler.observe(torch.tensor([[[[0.0, 2.0]]]]))
    scaler.observe(torch.tensor([[[[2.0, 4.0]]]]))
    assert calls == []
    scaler.finalize_exact_training_split_calibration(
        logical_samples=4,
        packed_examples=4,
        fingerprint=b"z" * 32,
        distributed=True,
    )
    assert len(calls) == 1
    assert scaler.has_exact_training_split_calibration



# ---------------------------------------------------------------------------
# Identity at initialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", PACKED_HEADS)
def test_untrained_head_returns_the_unchanged_rollout(head: str) -> None:
    """A fresh head must satisfy ``refined == Aurora`` exactly.

    This is what ``zero_init_output`` is supposed to buy, and it did *not* hold
    for the diffusion heads under epsilon-prediction: a zero prediction there
    telescopes to ``x_0 = x_T / sqrt(abar_T)``, returning a residual tens of
    times larger than the correction being learned.
    """
    refiner, _ = _build(head)
    _make_refiner_eval_ready(refiner)
    cond = torch.randn(2, CHANNELS + 1, HEIGHT, WIDTH)
    lead = torch.tensor([24.0, 48.0])

    deterministic = refiner.deterministic_residual(cond, forecast_lead_time=lead)
    assert torch.count_nonzero(deterministic) == 0

    generator = torch.Generator().manual_seed(0)
    sampled = refiner.sample_residual(cond, forecast_lead_time=lead, generator=generator)
    assert torch.count_nonzero(sampled) == 0


def test_epsilon_prediction_warning_is_emitted_once(monkeypatch) -> None:
    monkeypatch.setattr(refinement_config, "_EPSILON_RESIDUAL_WARNING_EMITTED", False)
    explicit_epsilon = {
        "model": {
            "refinement": {
                "enabled": True,
                "type": "diffusion_unet",
                "diffusion": {"prediction_type": "epsilon"},
            }
        }
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # The legacy-safe omitted default is not an explicit scientific choice
        # and must not consume the one-shot warning.
        resolve_refinement_config(
            {"model": {"refinement": {"enabled": True, "type": "diffusion_unet"}}}
        )
        resolve_refinement_config(explicit_epsilon)
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "enabled": True,
                        "type": "diffusion_transformer",
                        "diffusion": {"prediction_type": "epsilon"},
                    }
                }
            }
        )
    epsilon_warnings = [
        item
        for item in caught
        if issubclass(item.category, RuntimeWarning)
        and "poor fit for conditional" in str(item.message)
    ]
    assert len(epsilon_warnings) == 1


# ---------------------------------------------------------------------------
# Shapes, leads and levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", PACKED_HEADS)
def test_shapes_round_trip_for_multi_level_multi_lead(head: str) -> None:
    refiner, _ = _build(head)
    refiner.train()
    batch = 6  # 2 initializations x 3 forecast leads folded into the batch
    residual = torch.randn(batch, CHANNELS, HEIGHT, WIDTH) * 0.03
    cond = torch.randn(batch, CHANNELS + 1, HEIGHT, WIDTH)
    lead = torch.tensor([12.0, 36.0, 72.0, 12.0, 36.0, 72.0])
    lead_index = torch.tensor([0, 1, 2, 0, 1, 2])
    mask = torch.ones_like(residual, dtype=torch.bool)

    out = refiner.compute_training_loss(
        residual,
        cond,
        forecast_lead_time=lead,
        mask=mask,
        lead_index=lead_index,
        rollout_normalized=torch.randn_like(residual),
    )
    assert out.total_loss is not None and torch.isfinite(out.total_loss)

    _make_refiner_eval_ready(refiner)
    sample = refiner.sample_residual(cond, forecast_lead_time=lead)
    assert sample.shape == (batch, CHANNELS, HEIGHT, WIDTH)


@pytest.mark.parametrize("head", PACKED_HEADS)
def test_lead_time_conditioning_changes_the_output(head: str) -> None:
    """Forecast lead must actually reach the network output.

    Identity-at-init is implemented by zero-initialising the output projection
    *and* every conditioning modulation (FiLM / adaLN), so a fresh network is
    deliberately blind to both process time and lead time. All of those have to
    be perturbed before lead-time sensitivity can be observed at all.
    """
    refiner, _ = _build(head)
    with torch.no_grad():
        for name, param in refiner.mean_net.named_parameters():
            if any(key in name for key in ("out_proj", "head", "lead", "film", "ada_ln")):
                param.add_(torch.randn_like(param) * 0.1)
    refiner.fit_residual_scale(torch.randn(4, CHANNELS, HEIGHT, WIDTH))
    _make_refiner_eval_ready(refiner)
    cond = torch.randn(2, CHANNELS + 1, HEIGHT, WIDTH)
    short = refiner.deterministic_residual(cond, forecast_lead_time=torch.tensor([12.0, 12.0]))
    long = refiner.deterministic_residual(cond, forecast_lead_time=torch.tensor([72.0, 72.0]))
    assert not torch.allclose(short, long)


# ---------------------------------------------------------------------------
# Residual construction and reconstruction
# ---------------------------------------------------------------------------


def test_residual_sign_convention_and_reconstruction() -> None:
    packing = _packing()
    space = NormalizedTargetSpace(packing)
    rollout_norm = torch.randn(4, CHANNELS, HEIGHT, WIDTH)
    target_norm = rollout_norm + 0.05

    residual, valid = space.residual_target_from_normalized(target_norm, rollout_norm)
    torch.testing.assert_close(residual, torch.full_like(residual, 0.05))
    assert bool(valid.all())

    refined_norm, refined_physical = space.reconstruct(
        rollout_norm, residual, apply_constraints=False
    )
    torch.testing.assert_close(refined_norm, target_norm)
    torch.testing.assert_close(refined_physical, space.decode(target_norm))


def test_normalization_round_trip_is_exact() -> None:
    space = NormalizedTargetSpace(_packing())
    physical = torch.randn(3, CHANNELS, HEIGHT, WIDTH) * 1e-7
    torch.testing.assert_close(
        space.decode(space.encode(physical)), physical, rtol=1e-5, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Longitude periodicity
# ---------------------------------------------------------------------------


def test_periodic_longitude_positional_encoding_has_no_seam() -> None:
    """The 0/360 seam must look like any other pair of adjacent columns."""
    dim, grid_h, grid_w = 32, 3, 8
    encoding = sincos_2d_positional_encoding(
        dim, grid_h, grid_w, periodic_lon=True
    ).reshape(grid_h, grid_w, dim)[0]
    interior = torch.stack(
        [(encoding[j + 1] - encoding[j]).norm() for j in range(grid_w - 1)]
    )
    seam = (encoding[0] - encoding[-1]).norm()
    torch.testing.assert_close(seam, interior.mean(), rtol=1e-4, atol=1e-5)


def test_nonperiodic_longitude_positional_encoding_has_a_seam() -> None:
    dim, grid_h, grid_w = 32, 3, 8
    encoding = sincos_2d_positional_encoding(
        dim, grid_h, grid_w, periodic_lon=False
    ).reshape(grid_h, grid_w, dim)[0]
    interior = torch.stack(
        [(encoding[j + 1] - encoding[j]).norm() for j in range(grid_w - 1)]
    )
    seam = (encoding[0] - encoding[-1]).norm()
    assert float(seam) > 2.0 * float(interior.mean())


@pytest.mark.parametrize("head", PACKED_HEADS)
def test_no_artificial_discontinuity_at_the_longitude_seam(head: str) -> None:
    """A longitudinally smooth global input must not produce a seam artefact.

    The correction's finite difference across the 0/360 boundary is compared
    with the largest interior finite difference. On a periodic grid the seam is
    an ordinary pair of neighbouring columns, so it must not stand out.
    """
    refiner, _ = _build(head, lon_periodic=True)
    with torch.no_grad():
        for name, param in refiner.net.named_parameters():
            if "out_proj" in name or name.endswith("head.weight"):
                param.add_(torch.randn_like(param) * 0.05)
    _make_refiner_eval_ready(refiner)

    lon = torch.linspace(0.0, 2.0 * math.pi, WIDTH + 1)[:WIDTH].view(1, 1, 1, -1)
    lat = torch.linspace(-1.0, 1.0, HEIGHT).view(1, 1, -1, 1)
    smooth = torch.sin(2.0 * lon) * torch.cos(math.pi * lat)
    cond = smooth.expand(1, CHANNELS + 1, HEIGHT, WIDTH).contiguous()

    out = refiner.deterministic_residual(cond, forecast_lead_time=torch.tensor([24.0]))
    interior = (out[..., 1:] - out[..., :-1]).abs().max()
    seam = (out[..., 0] - out[..., -1]).abs().max()
    assert float(seam) <= 1.5 * float(interior) + 1e-8, (
        f"{head}: seam jump {float(seam):.4g} vs interior {float(interior):.4g}"
    )


@pytest.mark.parametrize("head", ("diffusion_unet", "flow_matching_conv_unet"))
def test_regional_grid_does_not_wrap(head: str) -> None:
    """A regional domain must keep edge padding rather than joining its edges."""
    refiner, packing = _build(head, lon_periodic=False)
    assert not refiner.lon_periodic
    for module in refiner.net.modules():
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3):
            assert module.padding_mode == "replicate"


# ---------------------------------------------------------------------------
# The decisive test: a known residual must be recovered
# ---------------------------------------------------------------------------


def _synthetic_problem(samples: int, *, seed: int = 0):
    """Rollout/target pairs whose residual is a known function of the rollout.

    The residual amplitude (0.03 in normalized target space) matches what was
    measured on the real global O3 rollout, so the test exercises exactly the
    signal-to-noise regime that broke the original implementation.
    """
    generator = torch.Generator().manual_seed(seed)
    y = torch.linspace(-1.0, 1.0, HEIGHT).view(1, 1, -1, 1)
    x = torch.linspace(-1.0, 1.0, WIDTH).view(1, 1, 1, -1)
    rollout = torch.randn(samples, CHANNELS, HEIGHT, WIDTH, generator=generator) * 0.5
    rollout = rollout + torch.sin(3.0 * math.pi * x) * torch.cos(2.0 * math.pi * y)
    # Known, deterministic, conditioning-driven residual plus a small
    # irreducible part, exactly as in the real problem.
    residual = 0.03 * torch.tanh(rollout)
    residual = residual + 0.005 * torch.randn(
        samples, CHANNELS, HEIGHT, WIDTH, generator=generator
    )
    return rollout, rollout + residual, residual


@pytest.mark.parametrize("head", PACKED_HEADS)
def test_synthetic_known_residual_is_learned(head: str) -> None:
    """End-to-end: sign, scale and reconstruction must all be right.

    A head that learned the residual with the wrong sign, or at the wrong
    magnitude, would still produce a decreasing training loss. Only scoring the
    reconstruction against the truth catches it, so that is what is asserted:
    the refined field must beat the unrefined rollout by a wide margin.
    """
    torch.manual_seed(0)
    backend_overrides = (
        {"diffusion": {"training_timesteps": 200, "inference_steps": 10}}
        if head.startswith("diffusion_")
        else {}
    )

    refiner, _ = _build(
        head,
        loss={"deterministic_weight": 1.0},
        **backend_overrides,
    )
    rollout, target, residual = _synthetic_problem(48)
    mask = torch.ones_like(residual, dtype=torch.bool)
    cond = torch.cat([rollout, mask[:, :1].float()], dim=1)

    refiner.fit_residual_scale(residual, mask)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(1)
    refiner.train()
    for _ in range(150):
        index = torch.randint(0, rollout.shape[0], (8,), generator=generator)
        out = refiner.compute_training_loss(
            residual[index],
            cond[index],
            forecast_lead_time=torch.full((8,), 24.0),
            mask=mask[index],
            generator=generator,
            rollout_normalized=rollout[index],
        )
        optimizer.zero_grad(set_to_none=True)
        out.total_loss.backward()
        optimizer.step()

    _make_refiner_eval_ready(refiner)
    test_rollout, test_target, test_residual = _synthetic_problem(16, seed=7)
    test_cond = torch.cat([test_rollout, torch.ones(16, 1, HEIGHT, WIDTH)], dim=1)
    predicted = refiner.deterministic_residual(
        test_cond, forecast_lead_time=torch.full((16,), 24.0)
    )

    # 1. Correct sign: the prediction must be positively correlated with truth.
    correlation = torch.corrcoef(
        torch.stack([predicted.flatten(), test_residual.flatten()])
    )[0, 1]
    assert float(correlation) > 0.5, f"{head}: residual correlation {float(correlation):.3f}"

    # 2. Correct magnitude: within a factor of two of the truth.
    ratio = float(predicted.std() / test_residual.std())
    assert 0.5 < ratio < 2.0, f"{head}: residual magnitude ratio {ratio:.3f}"

    # 3. The reconstruction must actually be better than doing nothing.
    before = (test_rollout - test_target).pow(2).mean()
    after = (test_rollout + predicted - test_target).pow(2).mean()
    assert float(after) < 0.7 * float(before), f"{head}: MSE {float(before):.5f} -> {float(after):.5f}"


@pytest.mark.parametrize("head", PACKED_HEADS)
def test_checkpoint_round_trip_preserves_predictions(head: str) -> None:
    torch.manual_seed(3)
    refiner, _ = _build(head)
    rollout, _, residual = _synthetic_problem(8, seed=4)
    mask = torch.ones_like(residual, dtype=torch.bool)
    refiner.fit_residual_scale(residual, mask)
    with torch.no_grad():
        for name, param in refiner.net.named_parameters():
            if "out_proj" in name or name.endswith("head.weight"):
                param.add_(torch.randn_like(param) * 0.05)
    _make_refiner_eval_ready(refiner)

    cond = torch.cat([rollout, mask[:, :1].float()], dim=1)
    lead = torch.full((8,), 24.0)
    expected = refiner.deterministic_residual(cond, forecast_lead_time=lead)

    restored, _ = _build(head)
    restored.load_state_dict(refiner.state_dict())
    restored.eval()
    torch.testing.assert_close(
        restored.deterministic_residual(cond, forecast_lead_time=lead), expected
    )
    torch.testing.assert_close(restored.residual_scaler.scale, refiner.residual_scaler.scale)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_flow_matching_alias_is_unchanged() -> None:
    config = resolve_refinement_config(
        {"model": {"refinement": {"enabled": True, "type": "flow_matching"}}}
    )
    assert config.type == "flow_matching_unet"
    assert config.backend == "legacy"


def test_new_conv_flow_head_is_a_unified_head() -> None:
    config = resolve_refinement_config(
        {"model": {"refinement": {"enabled": True, "type": "flow_matching_conv_unet"}}}
    )
    assert config.backend == "unified"
    assert config.is_flow_matching and not config.uses_transformer


def test_configuration_without_new_keys_still_resolves() -> None:
    """An existing YAML that predates every new option must keep working."""
    config = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "enabled": True,
                    "type": "diffusion_unet",
                    "target_space": {
                        "use_existing_normalization": True,
                        "residual_space": "normalized",
                    },
                    "loss": {
                        "generative": "mse",
                        "reconstruction_weight": 0.0,
                        "bias_weight": 0.0,
                        "gradient_weight": 0.0,
                        "pattern_correlation_weight": 0.0,
                        "area_weighted": True,
                        "separate_by_variable": True,
                        "separate_by_level": True,
                        "separate_by_lead_time": True,
                    },
                }
            }
        }
    )
    assert config.is_active
    assert config.loss.deterministic_weight == 0.0
    assert not config.loss.has_auxiliary_terms


def test_residual_scaling_can_be_disabled_for_exact_legacy_numerics() -> None:
    config = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "enabled": True,
                    "type": "diffusion_unet",
                    "target_space": {"residual_scaling": "none"},
                }
            }
        }
    )
    assert config.target_space.resolved_residual_scaling() == "none"
    refiner = build_refiner(
        config, residual_channels=CHANNELS, cond_channels=CHANNELS + 1, metadata=_packing()
    )
    assert refiner is not None
    assert not refiner.residual_scaler.is_active


def test_unknown_loss_key_is_rejected() -> None:
    with pytest.raises(ConfigValidationError):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "enabled": True,
                        "type": "diffusion_unet",
                        "loss": {"not_a_real_weight": 1.0},
                    }
                }
            }
        )


# ---------------------------------------------------------------------------
# Auxiliary loss terms
# ---------------------------------------------------------------------------


def test_every_auxiliary_term_is_finite_and_contributes() -> None:
    refiner, _ = _build(
        "flow_matching_conv_unet",
        loss={
            "deterministic_weight": 1.0,
            "reconstruction_weight": 1.0,
            "bias_weight": 1.0,
            "gradient_weight": 1.0,
            "pattern_correlation_weight": 1.0,
            "mae_weight": 1.0,
            "extreme_weight": 1.0,
            "peak_weight": 1.0,
            "quantile_weight": 1.0,
            "variance_weight": 1.0,
            "spectral_weight": 1.0,
            "degradation_weight": 1.0,
            "magnitude_weight": 1.0,
        },
    )
    refiner.train()
    residual = torch.randn(4, CHANNELS, HEIGHT, WIDTH) * 0.03
    cond = torch.randn(4, CHANNELS + 1, HEIGHT, WIDTH)
    out = refiner.compute_training_loss(
        residual,
        cond,
        forecast_lead_time=torch.full((4,), 24.0),
        mask=torch.ones_like(residual, dtype=torch.bool),
        lead_index=torch.tensor([0, 1, 2, 3]),
        rollout_normalized=torch.randn_like(residual),
    )
    assert torch.isfinite(out.total_loss)
    expected = {
        "reconstruction",
        "bias",
        "gradient",
        "pattern_correlation",
        "deterministic",
        "mae",
        "extreme",
        "peak",
        "quantile",
        "variance",
        "spectral",
        "degradation",
        "magnitude",
    }
    assert expected.issubset(out.diagnostics.keys())
    assert all(math.isfinite(v) for v in out.diagnostics.values())
    out.total_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in refiner.parameters())


def test_degradation_term_is_zero_for_a_perfect_correction() -> None:
    from finetune.refinement.config import LossConfig
    from finetune.refinement.losses import compute_auxiliary_losses

    residual = torch.randn(2, CHANNELS, HEIGHT, WIDTH) * 0.03
    terms = compute_auxiliary_losses(
        residual,
        residual,
        config=LossConfig(degradation_weight=1.0),
        mask=torch.ones_like(residual, dtype=torch.bool),
    )
    assert float(terms.degradation) == pytest.approx(0.0, abs=1e-12)


def test_masked_cells_never_enter_any_loss() -> None:
    from finetune.refinement.config import LossConfig
    from finetune.refinement.losses import compute_auxiliary_losses

    residual = torch.randn(2, CHANNELS, HEIGHT, WIDTH) * 0.03
    estimate = residual.clone()
    mask = torch.ones_like(residual, dtype=torch.bool)
    mask[..., WIDTH // 2 :] = False
    estimate = torch.where(mask, estimate, torch.full_like(estimate, 1e6))
    config = LossConfig(reconstruction_weight=1.0, mae_weight=1.0, bias_weight=1.0)
    terms = compute_auxiliary_losses(estimate, residual, config=config, mask=mask)
    assert float(terms.reconstruction) == pytest.approx(0.0, abs=1e-10)
    assert float(terms.mae) == pytest.approx(0.0, abs=1e-10)
