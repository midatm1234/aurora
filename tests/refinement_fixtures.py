"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared fixtures for the stochastic residual-refinement tests.

Everything here is synthetic and CPU-only: the tests exercise the refinement
package's contracts (configuration, target space, packing, forecast timing,
flow matching, diffusion, spatial Transformers, checkpoints, performance
parity) without touching datasets, checkpoints or outputs on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from finetune.refinement.packing import FieldPacking
from finetune.refinement.two_phase import AuroraTwoPhaseRefiner

__all__ = [
    "DummySpec",
    "build_packing",
    "build_refiner_model",
    "refinement_config",
]


@dataclass
class DummySpec:
    """Stand-in for ``ResolvedVariableSpecs.targets`` entries."""

    aurora_name: str
    dataset_name: str
    kind: str
    loss_levels: list[float] | None = None
    units: str = "kg kg-1"


def build_packing(
    height: int = 20,
    width: int = 30,
    *,
    lon_periodic: bool = False,
    lead_times_hours: tuple[float, ...] = (24.0, 48.0, 72.0),
) -> FieldPacking:
    specs = [
        DummySpec("gtco3", "gtco3", "surf", units="kg m-2"),
        DummySpec("go3", "go3", "atmos", [500.0, 850.0], units="kg kg-1"),
    ]
    norm_stats = {
        "gtco3": {"mean": torch.tensor([1.0e-3]), "std": torch.tensor([2.0e-4])},
        "go3": {
            "mean": torch.tensor([0.0, 0.0]),
            "std": torch.tensor([1.0e-7, 2.0e-7]),
        },
    }
    return FieldPacking.from_specs(
        specs,
        norm_stats=norm_stats,
        atmos_levels=[500.0, 850.0],
        lat=[90.0 - i * (180.0 / max(height - 1, 1)) for i in range(height)],
        lon=[i * (360.0 / width) for i in range(width)],
        lead_times_hours=lead_times_hours,
        lon_periodic=lon_periodic,
    )


def refinement_config(refinement_type: str, **overrides) -> dict:
    """Small, fast configuration for the given refiner type."""
    refinement = {
        "type": refinement_type,
        "enabled": refinement_type != "none",
        "ensemble_size": 2,
        "seed": 1234,
        "conditioning": {
            "aurora_rollout": True,
            "static_fields": False,
            "masks": True,
            "forecast_lead_time": True,
        },
        "unet": {
            "hidden_channels": 8,
            "num_levels": 2,
            "num_residual_blocks": 1,
            "time_embedding_dim": 16,
            "attention_heads": 4,
        },
        "transformer": {
            "patch_size": [4, 4],
            "embedding_dim": 32,
            "num_heads": 4,
            "num_blocks": 2,
            "max_tokens_lat": 32,
            "max_tokens_lon": 32,
        },
    }
    # Only attach the section that belongs to the selected process, so resolving
    # the configuration never warns about ignored settings.
    if refinement_type.startswith("diffusion"):
        refinement["diffusion"] = {"training_timesteps": 40, "inference_steps": 4}
    elif refinement_type.startswith("flow_matching"):
        refinement["flow_matching"] = {"integration_steps": 3}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(refinement.get(key), dict):
            refinement[key] = {**refinement[key], **value}
        else:
            refinement[key] = value
    return {"model": {"refinement": refinement}}


def build_refiner_model(
    refinement_type: str,
    *,
    height: int = 20,
    width: int = 30,
    lon_periodic: bool = False,
    **overrides,
) -> AuroraTwoPhaseRefiner:
    from finetune.refinement.two_phase import build_two_phase_refiner

    packing = build_packing(height, width, lon_periodic=lon_periodic)
    model = build_two_phase_refiner(None, packing, refinement_config(refinement_type, **overrides))
    if model.refinement_config.is_active:
        model.initialize_refiner(model.conditioning_channels())
    return model
