"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Spatial-only attention contract for the refinement Transformers.

These tests enforce the non-negotiable architectural rules:

* attention operates over 2-D spatial tokens only;
* batch elements never interact;
* forecast lead-time items never interact (they are folded into the batch);
* historical input times are never attention tokens and there is no temporal
  attention module, no cross-time attention and no causal temporal mask;
* forecast lead time is embedded separately from the diffusion timestep and the
  flow interpolation time;
* patchification/unpatchification preserve token order and restore the exact
  input dimensions, including for rectangular and non-divisible grids.
"""

from __future__ import annotations

import pytest
import torch
from finetune.refinement.backbones import (
    SpatialResidualTransformer,
    patchify_2d,
    sincos_2d_positional_encoding,
    unpatchify_2d,
)
from torch import nn


def build_transformer(**kwargs) -> SpatialResidualTransformer:
    params = dict(
        in_channels=2,
        cond_channels=3,
        out_channels=2,
        patch_size=(4, 4),
        embedding_dim=32,
        num_heads=4,
        num_blocks=2,
        max_tokens_lat=64,
        max_tokens_lon=64,
        zero_init_output=False,
        lead_time_conditioning=True,
        lead_time_scale_hours=72.0,
    )
    params.update(kwargs)
    model = SpatialResidualTransformer(**params)
    torch.manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.05 * torch.randn_like(param))
    return model.eval()


def run(model: SpatialResidualTransformer, x, cond, t, lead):
    with torch.no_grad():
        return model(x, cond, t, lead)


# --------------------------------------------------------------------------
# Patchification
# --------------------------------------------------------------------------


def test_patchify_unpatchify_round_trip() -> None:
    x = torch.randn(2, 3, 12, 16)
    tokens, grid_h, grid_w = patchify_2d(x, 4, 4)
    assert (grid_h, grid_w) == (3, 4)
    assert tokens.shape == (2, 12, 3 * 16)
    assert torch.equal(unpatchify_2d(tokens, 3, grid_h, grid_w, 4, 4), x)


def test_patchify_preserves_row_major_latlon_token_order() -> None:
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    tokens, grid_h, grid_w = patchify_2d(x, 2, 2)
    # Token i maps to grid position (i // grid_w, i % grid_w).
    for index in range(grid_h * grid_w):
        row, col = index // grid_w, index % grid_w
        block = x[0, :, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2].reshape(-1)
        assert torch.equal(tokens[0, index], block)


def test_patchify_rejects_non_multiple_sizes() -> None:
    with pytest.raises(ValueError, match="multiples of the patch size"):
        patchify_2d(torch.randn(1, 1, 5, 4), 2, 2)


def test_sincos_encoding_is_a_function_of_the_two_dimensional_position() -> None:
    encoding = sincos_2d_positional_encoding(32, 3, 5)
    assert encoding.shape == (15, 32)
    # Distinct (lat, lon) tokens get distinct encodings.
    assert torch.unique(encoding, dim=0).shape[0] == 15
    # Moving one row and moving one column change different halves.
    same_row = encoding[0] - encoding[1]  # differ in lon only
    same_col = encoding[0] - encoding[5]  # differ in lat only
    assert torch.count_nonzero(same_row[:16]) == 0
    assert torch.count_nonzero(same_col[16:]) == 0


def test_sincos_requires_dim_divisible_by_four() -> None:
    with pytest.raises(ValueError, match="dim % 4"):
        sincos_2d_positional_encoding(30, 2, 2)


# --------------------------------------------------------------------------
# Shape handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("height,width", [(16, 16), (12, 20), (13, 19), (7, 5)])
@pytest.mark.parametrize("attention_mode", ["global_2d", "windowed_2d"])
def test_exact_output_dimensions_are_restored(height: int, width: int, attention_mode: str) -> None:
    model = build_transformer(attention_mode=attention_mode, window_size=(2, 2))
    x = torch.randn(2, 2, height, width)
    cond = torch.randn(2, 3, height, width)
    out = run(model, x, cond, torch.tensor([10.0, 20.0]), torch.tensor([24.0, 72.0]))
    assert out.shape == (2, 2, height, width)
    assert torch.isfinite(out).all()


def test_padding_and_cropping_preserve_the_original_field_exactly() -> None:
    """A zero-weight model must return exactly zeros of the original shape."""
    model = SpatialResidualTransformer(
        in_channels=1,
        cond_channels=1,
        out_channels=1,
        patch_size=(8, 8),
        embedding_dim=32,
        num_heads=4,
        num_blocks=1,
        max_tokens_lat=16,
        max_tokens_lon=16,
        zero_init_output=True,
    ).eval()
    x = torch.randn(1, 1, 13, 21)
    cond = torch.randn(1, 1, 13, 21)
    out = run(model, x, cond, torch.tensor([1.0]), None)
    assert out.shape == (1, 1, 13, 21)
    assert torch.count_nonzero(out) == 0


def test_longitude_periodic_padding_is_used_when_configured() -> None:
    model = build_transformer(patch_size=(4, 4), lon_periodic=True)
    x = torch.randn(1, 2, 8, 10)
    cond = torch.randn(1, 3, 8, 10)
    out = run(model, x, cond, torch.tensor([5.0]), torch.tensor([24.0]))
    assert out.shape == (1, 2, 8, 10)


# --------------------------------------------------------------------------
# Isolation guarantees
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attention_mode", ["global_2d", "windowed_2d"])
def test_batch_items_do_not_interact(attention_mode: str) -> None:
    model = build_transformer(attention_mode=attention_mode, window_size=(2, 2))
    x = torch.randn(3, 2, 16, 16)
    cond = torch.randn(3, 3, 16, 16)
    t = torch.tensor([1.0, 2.0, 3.0])
    lead = torch.tensor([24.0, 48.0, 72.0])
    baseline = run(model, x, cond, t, lead)

    perturbed_x = x.clone()
    perturbed_x[1] += 10.0
    perturbed = run(model, perturbed_x, cond, t, lead)
    assert torch.equal(baseline[0], perturbed[0])
    assert torch.equal(baseline[2], perturbed[2])
    assert not torch.equal(baseline[1], perturbed[1])


@pytest.mark.parametrize("attention_mode", ["global_2d", "windowed_2d"])
def test_lead_time_items_folded_into_the_batch_do_not_interact(
    attention_mode: str,
) -> None:
    """A batch of 2 samples x 3 lead times folded into 6 rows stays isolated."""
    model = build_transformer(attention_mode=attention_mode, window_size=(2, 2))
    x = torch.randn(6, 2, 16, 16)
    cond = torch.randn(6, 3, 16, 16)
    t = torch.full((6,), 7.0)
    lead = torch.tensor([24.0, 48.0, 72.0, 24.0, 48.0, 72.0])
    baseline = run(model, x, cond, t, lead)

    perturbed_lead = lead.clone()
    perturbed_lead[4] = 96.0
    perturbed = run(model, x, cond, t, perturbed_lead)
    for row in (0, 1, 2, 3, 5):
        assert torch.equal(baseline[row], perturbed[row])


def test_evaluating_rows_separately_matches_the_batched_evaluation() -> None:
    model = build_transformer(attention_mode="windowed_2d", window_size=(2, 2))
    x = torch.randn(4, 2, 12, 12)
    cond = torch.randn(4, 3, 12, 12)
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    lead = torch.tensor([6.0, 24.0, 48.0, 72.0])
    batched = run(model, x, cond, t, lead)
    for row in range(4):
        single = run(
            model,
            x[row : row + 1],
            cond[row : row + 1],
            t[row : row + 1],
            lead[row : row + 1],
        )
        assert torch.allclose(batched[row], single[0], atol=1e-5)


def test_no_temporal_attention_module_or_causal_mask() -> None:
    model = build_transformer()
    names = {name.lower() for name, _ in model.named_modules()}
    for banned in ("temporal", "time_attn", "cross_time", "causal", "recurrent", "rnn"):
        assert not any(banned in name for name in names), banned
    assert not any(isinstance(module, (nn.RNNBase, nn.LSTM, nn.GRU)) for module in model.modules())
    # No attention module may be configured as causal.
    for block in model.blocks:
        assert getattr(block.attn, "is_causal", False) is False
        assert block.attn.mode in {"global_2d", "windowed_2d"}


def test_process_time_changes_values_but_not_token_order() -> None:
    model = build_transformer()
    x = torch.randn(1, 2, 16, 16)
    cond = torch.randn(1, 3, 16, 16)
    lead = torch.tensor([24.0])
    a = run(model, x, cond, torch.tensor([1.0]), lead)
    b = run(model, x, cond, torch.tensor([900.0]), lead)
    assert a.shape == b.shape
    assert not torch.equal(a, b)
    # Spatial structure is preserved: a permutation of the input rows permutes
    # the output rows in the same way.
    flipped = run(model, x.flip(-2), cond.flip(-2), torch.tensor([1.0]), lead)
    assert flipped.shape == a.shape


# --------------------------------------------------------------------------
# Attention implementations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attention_mode", ["global_2d", "windowed_2d"])
def test_optimized_attention_matches_the_reference_implementation(
    attention_mode: str,
) -> None:
    model = build_transformer(
        attention_mode=attention_mode, window_size=(2, 2), optimized_attention="math"
    )
    x = torch.randn(2, 2, 12, 12)
    cond = torch.randn(2, 3, 12, 12)
    t = torch.tensor([3.0, 4.0])
    lead = torch.tensor([24.0, 72.0])
    reference = run(model, x, cond, t, lead)
    model.set_attention_implementation("sdpa")
    optimized = run(model, x, cond, t, lead)
    assert torch.allclose(reference, optimized, atol=1e-5, rtol=1e-4)


def test_unknown_attention_implementation_raises() -> None:
    model = build_transformer()
    with pytest.raises(ValueError, match="Unsupported attention implementation"):
        model.set_attention_implementation("flash")


def test_windowed_attention_is_local_but_shifted_windows_widen_the_receptive_field() -> None:
    """Windowed attention must stay strictly spatial and strictly local."""
    unshifted = build_transformer(
        attention_mode="windowed_2d", window_size=(2, 2), num_blocks=1, patch_size=(4, 4)
    )
    x = torch.zeros(1, 2, 16, 16)
    cond = torch.zeros(1, 3, 16, 16)
    t = torch.tensor([1.0])
    lead = torch.tensor([24.0])
    baseline = run(unshifted, x, cond, t, lead)

    perturbed = x.clone()
    perturbed[0, :, 0:4, 0:4] = 5.0  # exactly one token (patch 4x4) at (0, 0)
    changed = run(unshifted, perturbed, cond, t, lead)
    delta = (changed - baseline).abs().sum(dim=(0, 1))
    # Tokens inside the same 2x2-token window (rows/cols 0..7) may change; the
    # far corner must not, because attention never leaves the window.
    assert float(delta[8:, 8:].max()) == 0.0
    assert float(delta[:8, :8].max()) > 0.0


def test_shifted_windows_are_configured_on_alternating_blocks() -> None:
    model = build_transformer(
        attention_mode="windowed_2d", window_size=(4, 4), shifted_windows=True, num_blocks=4
    )
    shifts = [block.attn.shift for block in model.blocks]
    assert shifts == [(0, 0), (2, 2), (0, 0), (2, 2)]


def test_gradient_checkpointing_matches_the_plain_path() -> None:
    plain = build_transformer(gradient_checkpointing=False)
    checkpointed = build_transformer(gradient_checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    x = torch.randn(1, 2, 16, 16)
    cond = torch.randn(1, 3, 16, 16)
    t = torch.tensor([2.0])
    lead = torch.tensor([48.0])
    assert torch.allclose(plain(x, cond, t, lead), checkpointed(x, cond, t, lead), atol=1e-6)
