"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Focused regressions for longitude-boundary operators and masked spatial losses.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from finetune.longitude import PeriodicConv2d
from finetune.refinement.backbones import (
    ConditionalResidualUNet,
    SpatialResidualTransformer,
    learned_periodic_longitude_encoding,
)
from finetune.refinement.config import LossConfig
from finetune.refinement.losses import (
    BiasLossTerms,
    _extreme_loss,
    _pattern_correlation_loss,
    _peak_loss,
    _quantile_loss,
    compute_auxiliary_losses,
)


def _transformer(*, lon_periodic: bool) -> SpatialResidualTransformer:
    return SpatialResidualTransformer(
        in_channels=1,
        cond_channels=1,
        out_channels=1,
        patch_size=(2, 2),
        embedding_dim=16,
        num_heads=4,
        num_blocks=1,
        positional_encoding="latlon_2d",
        max_tokens_lat=8,
        max_tokens_lon=16,
        lon_periodic=lon_periodic,
        local_refinement=True,
    )


def test_periodic_convolution_wraps_longitude_but_not_latitude() -> None:
    field = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]]
    )
    conv = PeriodicConv2d(1, 1, 3, bias=False, lon_periodic=True)

    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 0] = 1.0
    # The left neighbour of the first longitude column is the last column.
    assert float(conv(field)[0, 0, 1, 0].detach()) == 8.0

    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 0, 1] = 1.0
    # The neighbour above the north edge is the replicated north-edge value,
    # not a value wrapped from the south edge.
    assert float(conv(field)[0, 0, 0, 1].detach()) == 2.0


def test_refinement_backbones_use_longitude_only_convolutions() -> None:
    unet = ConditionalResidualUNet(
        1,
        1,
        1,
        hidden_channels=4,
        num_levels=2,
        num_residual_blocks=1,
        time_embedding_dim=8,
        bottleneck_attention=False,
        lon_periodic=True,
    )
    blocks = [block for stage in (*unet.down_blocks, *unet.up_blocks) for block in stage]
    assert blocks
    assert all(isinstance(block.conv1, PeriodicConv2d) for block in blocks)
    assert all(isinstance(block.conv2, PeriodicConv2d) for block in blocks)
    assert "down_blocks.0.0.conv1.weight" in unet.state_dict()
    assert not any("conv1.conv." in key for key in unet.state_dict())

    transformer = _transformer(lon_periodic=True)
    assert isinstance(transformer.stem, PeriodicConv2d)
    assert isinstance(transformer.head, PeriodicConv2d)
    assert "stem.weight" in transformer.state_dict()
    assert "head.weight" in transformer.state_dict()

    regional = _transformer(lon_periodic=False)
    assert type(regional.stem) is nn.Conv2d
    assert type(regional.head) is nn.Conv2d
    assert regional.stem.padding_mode == "replicate"
    assert regional.head.padding_mode == "replicate"


def test_periodic_learned_encoding_closes_exactly_and_keeps_checkpoint_shape() -> None:
    torch.manual_seed(4)
    coefficients = torch.randn(17, 16, requires_grad=True)
    positions = torch.tensor([0.0, 1.0, 7.0, 8.0])
    encoded = learned_periodic_longitude_encoding(coefficients, positions, period=8)
    torch.testing.assert_close(encoded[0], encoded[-1], rtol=1e-5, atol=1e-5)
    encoded.sum().backward()
    assert coefficients.grad is not None
    assert float(coefficients.grad.abs().sum()) > 0.0

    # Switching a historical learned-position model to periodic longitude does
    # not add, remove, or rename parameters, so strict checkpoint loading still
    # works even though the periodic forward interpretation is now seam-safe.
    old_layout = _transformer(lon_periodic=False)
    periodic = _transformer(lon_periodic=True)
    periodic.load_state_dict(old_layout.state_dict(), strict=True)
    assert periodic.pos_lon.shape == old_layout.pos_lon.shape


def test_pattern_correlation_applies_spatial_weight_once() -> None:
    refined = torch.tensor([[[[0.0, 1.0, 4.0, 2.0]]]])
    truth = torch.tensor([[[[0.0, 3.0, 1.0, 2.0]]]])
    weight = torch.tensor([[[[1.0, 2.0, 4.0, 8.0]]]])

    norm = weight.sum(dim=(-2, -1))
    refined_anom = refined - (refined * weight).sum(dim=(-2, -1))[..., None, None] / norm[
        ..., None, None
    ]
    truth_anom = truth - (truth * weight).sum(dim=(-2, -1))[..., None, None] / norm[
        ..., None, None
    ]
    covariance = (weight * refined_anom * truth_anom).sum(dim=(-2, -1))
    refined_variance = (weight * refined_anom.square()).sum(dim=(-2, -1))
    truth_variance = (weight * truth_anom.square()).sum(dim=(-2, -1))
    expected = 1.0 - covariance / (refined_variance.sqrt() * truth_variance.sqrt())

    actual = _pattern_correlation_loss(refined, truth, weight)
    torch.testing.assert_close(actual, expected.mean())


def test_pattern_correlation_config_weight_is_applied_once() -> None:
    terms = BiasLossTerms(pattern_correlation=torch.tensor(2.0))
    config = LossConfig(pattern_correlation_weight=3.0)
    assert float(terms.weighted_total(config, torch.tensor(0.0))) == 6.0


def test_extreme_thresholds_and_extrema_ignore_masked_cells() -> None:
    truth = torch.tensor([[[[0.0, 1.0, 2.0, -1_000.0, 1_000.0]]]])
    refined = torch.tensor([[[[1.0, 1.5, 5.0, 9_000.0, -9_000.0]]]])
    mask = torch.tensor([[[[1.0, 1.0, 1.0, 0.0, 0.0]]]])

    masked = _extreme_loss(refined, truth, mask, quantile=0.75, intensity=4.0)
    compact = _extreme_loss(
        refined[..., :3],
        truth[..., :3],
        torch.ones_like(mask[..., :3]),
        quantile=0.75,
        intensity=4.0,
    )
    torch.testing.assert_close(masked, compact)

    masked_peak = _peak_loss(refined, truth, mask)
    compact_peak = _peak_loss(
        refined[..., :3], truth[..., :3], torch.ones_like(mask[..., :3])
    )
    torch.testing.assert_close(masked_peak, compact_peak)


def test_extreme_tail_can_focus_on_high_concentrations_only() -> None:
    truth = torch.tensor([[[[0.0, 1.0, 2.0, 3.0, 4.0]]]])
    # Put all error at the observed lower extreme. A symmetric objective should
    # emphasise it, while an upper-only NO2-tail objective should not.
    refined = truth.clone()
    refined[..., 0] = 10.0
    weight = torch.ones_like(truth)

    legacy_default = _extreme_loss(
        refined, truth, weight, quantile=0.8, intensity=4.0
    )
    both = _extreme_loss(
        refined, truth, weight, quantile=0.8, intensity=4.0, tail="both"
    )
    upper = _extreme_loss(
        refined, truth, weight, quantile=0.8, intensity=4.0, tail="upper"
    )

    torch.testing.assert_close(legacy_default, both)
    assert float(both) > float(upper)


def test_quantile_loss_exactly_excludes_masked_cells() -> None:
    refined = torch.tensor([[[[0.0, 10.0, -999.0, 999.0]]]])
    truth = torch.tensor([[[[2.0, 4.0, 999.0, -999.0]]]])
    mask = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])

    # Valid sorted pairs are (0, 2) and (10, 4): (4 + 36) / 2 = 20.
    torch.testing.assert_close(_quantile_loss(refined, truth, mask), torch.tensor(20.0))


def test_gradient_loss_uses_packing_periodicity_for_the_wrap_edge() -> None:
    estimate = torch.tensor([[[[0.0, 0.0, 0.0, 4.0]]]])
    target = torch.zeros_like(estimate)
    mask = torch.ones_like(estimate)
    config = LossConfig(gradient_weight=1.0)

    regional = compute_auxiliary_losses(estimate, target, config=config, mask=mask)
    global_terms = compute_auxiliary_losses(
        estimate,
        target,
        config=config,
        mask=mask,
        packing=SimpleNamespace(lon_periodic=True),
    )

    torch.testing.assert_close(regional.gradient, torch.tensor(16.0 / 3.0))
    torch.testing.assert_close(global_terms.gradient, torch.tensor(8.0))
    assert float(global_terms.gradient) > float(regional.gradient)
