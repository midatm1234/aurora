"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from finetune import refinement_parity_check as parity
from finetune.flow_refine import AuroraFlowRefine


def _legacy_flow_payload(*, with_temporal: bool) -> dict:
    config = {
        "model": {
            "flow_refine_enabled": True,
            "flow_refine_hidden": 4,
            "flow_refine_time_dim": 8,
            "flow_refine_sampling_steps": 1,
            "mamba_temporal_enabled": with_temporal,
            "mamba_temporal_channels": 4,
            "mamba_temporal_state": 2,
            "mamba_temporal_layers": 1,
            "mamba_temporal_conv": 2,
            "mamba_temporal_expand": 1,
        }
    }
    model = AuroraFlowRefine(
        nn.Identity(),
        target_surf_vars=("gtco3",),
        hidden=4,
        time_dim=8,
        sampling_steps=1,
    )
    state = {key: value.clone() for key, value in model.state_dict().items()}
    if with_temporal:
        # A representative temporal key is sufficient here: the parity utility
        # intentionally compares only the spatial legacy-flow module.
        state["temporal.surf_heads.gtco3.decoder.weight"] = torch.randn(1, 1)
    return {"model_state_dict": state, "config": config}


@pytest.mark.parametrize("with_temporal", [False, True])
def test_flow_checkpoint_runs_spatial_parity(
    tmp_path: Path, with_temporal: bool
) -> None:
    checkpoint = tmp_path / f"flow_temporal_{with_temporal}.ckpt"
    torch.save(_legacy_flow_payload(with_temporal=with_temporal), checkpoint)

    loaded = parity._load_payload(str(checkpoint))
    assert any(key.startswith("surf_flow.") for key in loaded["heads"])
    assert any(key.startswith("temporal.") for key in loaded["heads"]) is with_temporal

    report = parity.run_parity(
        str(checkpoint), height=8, width=8, batch=1, seed=17, device="cpu"
    )
    assert report["parity"] is True
    assert report["variables"] == ["gtco3"]


@pytest.mark.parametrize("head_only", [False, True])
def test_temporal_only_checkpoint_is_rejected_early(
    tmp_path: Path, head_only: bool
) -> None:
    checkpoint = tmp_path / f"temporal_only_{head_only}.ckpt"
    temporal = {"temporal.surf_heads.gtco3.decoder.weight": torch.randn(1, 1)}
    torch.save(
        {"heads": temporal} if head_only else {"model_state_dict": temporal}, checkpoint
    )

    with pytest.raises(SystemExit, match=r"does not contain flow-matching.*surf_flow"):
        parity._load_payload(str(checkpoint))
