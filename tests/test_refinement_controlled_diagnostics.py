"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the bounded stochastic-refinement diagnostic experiments."""

from __future__ import annotations

import json

import pytest
import torch
from finetune.run_refinement_diagnostics import (
    fit_tiny_dataset,
    make_smooth_synthetic_dataset,
    spatial_texture_statistics,
    tensor_statistics,
)


def test_synthetic_correction_has_required_multilevel_smooth_structure() -> None:
    data = make_smooth_synthetic_dataset(samples=4, height=16, width=20)
    correction = data.target - data.rollout

    assert correction.shape == (4, 4, 16, 20)
    assert data.lead_hours.tolist() == [24.0] * 4
    assert [spec.level for spec in data.packing.channels] == [None, 1000.0, 925.0, 850.0]
    assert [spec.std for spec in data.packing.channels] == pytest.approx(
        [6.141e-5, 1.931e-7, 1.379e-7, 9.422e-8]
    )
    assert data.lat[0] > data.lat[-1]
    assert data.lon[0] < data.lon[-1]
    assert float(correction.min()) < 0.0 < float(correction.max())
    assert torch.isfinite(correction).all()
    # Every pressure level has its own field; no accidental broadcasting.
    assert not torch.equal(correction[:, 1], correction[:, 2])
    assert not torch.equal(correction[:, 2], correction[:, 3])
    # The moving plume makes correction fields vary with the conditioning.
    assert not torch.equal(correction[0], correction[1])
    texture = spatial_texture_statistics(correction)
    assert texture["high_frequency_power_fraction"] < 0.1
    assert data.source_validation["correction_convention"] == "CAMS - Aurora"


def test_tensor_statistics_reports_nonfinite_counts_and_is_json_safe() -> None:
    value = torch.tensor([0.0, 2.0, float("nan"), float("inf"), -float("inf")])
    stats = tensor_statistics(value)
    assert stats["shape"] == [5]
    assert stats["finite_count"] == 2
    assert stats["nan_count"] == 1
    assert stats["positive_inf_count"] == 1
    assert stats["negative_inf_count"] == 1
    assert stats["mean"] == pytest.approx(1.0)
    json.dumps(stats, allow_nan=False)


def test_smooth_field_has_less_high_frequency_energy_than_grid_noise() -> None:
    data = make_smooth_synthetic_dataset(samples=2, height=16, width=20)
    smooth = spatial_texture_statistics(data.target - data.rollout)
    generator = torch.Generator().manual_seed(9)
    noise = torch.randn(data.rollout.shape, generator=generator)
    noisy = spatial_texture_statistics(noise)
    assert smooth["high_frequency_power_fraction"] < noisy["high_frequency_power_fraction"]


def test_direct_transformer_tiny_fit_uses_exact_correction_interface() -> None:
    data = make_smooth_synthetic_dataset(samples=2, height=12, width=15)
    result = fit_tiny_dataset(
        "direct_regression_transformer",
        data,
        train_steps=2,
        batch_size=2,
        learning_rate=1e-3,
        device="cpu",
        seed=3,
        overrides={
            "transformer": {
                "patch_size": [3, 5],
                "embedding_dim": 32,
                "num_heads": 4,
                "num_blocks": 1,
                "mlp_ratio": 1.0,
                "local_refinement": False,
            }
        },
        verbose=False,
    )
    assert result.predicted_correction.shape == data.rollout.shape
    assert result.refined.shape == data.rollout.shape
    assert result.history[0]["training_loss"] is None
    assert result.history[-1]["training_loss"] is not None
    assert torch.isfinite(result.predicted_correction).all()
    # The reported refined tensor is a physical field produced by adding the
    # predicted CAMS-minus-Aurora correction exactly once.
    assert result.refined.dtype == torch.float32
    json.dumps(result.history, allow_nan=False)


@pytest.mark.parametrize("samples", [0, 9])
def test_synthetic_tiny_sample_limit_is_explicit(samples: int) -> None:
    with pytest.raises(ValueError, match=r"samples must be in \[1, 8\]"):
        make_smooth_synthetic_dataset(samples=samples)
