"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Checkpoint compatibility contract for the two-phase Aurora wrapper.
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch
from finetune import aurora_finetune_utils as ft
from finetune.model_factory import validate_unified_checkpoint_contract
from finetune.refinement.checkpoint import (
    CHECKPOINT_KIND_AURORA,
    CHECKPOINT_KIND_COMBINED,
    CHECKPOINT_KIND_REFINEMENT,
    aurora_state_fingerprint,
    build_refinement_checkpoint,
    load_aurora_state_dict,
    load_refinement_state_dict,
    migrate_aurora_state_dict,
    migrate_legacy_flow_state_dict,
    save_checkpoint_atomic,
    strip_wrapper_prefixes,
    validate_aurora_reference,
)
from finetune.refinement.two_phase import AuroraTwoPhaseRefiner, build_two_phase_refiner
from torch import nn

from tests.refinement_fixtures import build_packing, refinement_config


class DummyAurora(nn.Module):
    """Minimal stand-in for the Aurora model with a stable state dict."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Linear(width, width)
        self.decoder = nn.Linear(width, width)
        self.register_buffer("scale", torch.ones(width))

    def forward(self, x):  # pragma: no cover - not exercised
        return self.decoder(self.encoder(x))


def build_model(refinement_type: str = "diffusion_unet") -> AuroraTwoPhaseRefiner:
    packing = build_packing(8, 8)
    model = build_two_phase_refiner(DummyAurora(), packing, refinement_config(refinement_type))
    model.initialize_refiner(model.conditioning_channels())
    return model


def aurora_only_state(model: AuroraTwoPhaseRefiner) -> dict[str, torch.Tensor]:
    return {
        key[len("aurora.") :]: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("aurora.")
    }


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------


def test_wrapper_prefixes_are_stripped() -> None:
    state = {"module._orig_mod.encoder.weight": torch.zeros(2)}
    assert list(strip_wrapper_prefixes(state)) == ["encoder.weight"]


def test_aurora_migration_is_explicit_and_auditable() -> None:
    state = {"encoder.weight": torch.zeros(2), "decoder.bias": torch.zeros(2)}
    migrated, renames = migrate_aurora_state_dict(state)
    assert set(migrated) == {"aurora.encoder.weight", "aurora.decoder.bias"}
    assert renames == {
        "encoder.weight": "aurora.encoder.weight",
        "decoder.bias": "aurora.decoder.bias",
    }


def test_legacy_flow_migration_splits_aurora_and_refinement_keys() -> None:
    state = {
        "base.encoder.weight": torch.zeros(2),
        "surf_flow.gtco3.out.weight": torch.zeros(2),
        "atmos_flow.go3.out.bias": torch.zeros(2),
        "_res_std__surf__gtco3": torch.zeros(()),
    }
    migrated, renames = migrate_legacy_flow_state_dict(state)
    assert migrated["aurora.encoder.weight"] is state["base.encoder.weight"]
    assert "refiner.legacy.surf_flow.gtco3.out.weight" in migrated
    assert "refiner.legacy.atmos_flow.go3.out.bias" in migrated
    assert "refiner.legacy._res_std__surf__gtco3" in migrated
    assert len(renames) == 4


def test_fingerprint_is_order_and_prefix_independent() -> None:
    a = {"b.weight": torch.arange(4.0), "a.weight": torch.ones(2)}
    b = {"module.a.weight": torch.ones(2), "module.b.weight": torch.arange(4.0)}
    assert aurora_state_fingerprint(a) == aurora_state_fingerprint(b)


def test_fingerprint_detects_a_changed_weight() -> None:
    a = {"a.weight": torch.ones(2)}
    b = {"a.weight": torch.ones(2) * 2}
    assert aurora_state_fingerprint(a) != aurora_state_fingerprint(b)


# --------------------------------------------------------------------------
# Strict loading
# --------------------------------------------------------------------------


def test_aurora_checkpoint_loads_strictly_into_the_wrapper() -> None:
    source = DummyAurora()
    model = build_model()
    report = load_aurora_state_dict(model, {"model_state_dict": source.state_dict()})
    assert report.unexpected == []
    assert report.shape_mismatched == []
    # Only the newly introduced refiner keys may be missing.
    assert all(key.startswith("refiner.") for key in report.missing)
    assert report.loaded == len(source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[f"aurora.{key}"], value)


def test_missing_aurora_key_raises() -> None:
    model = build_model()
    state = DummyAurora().state_dict()
    state.pop("encoder.weight")
    with pytest.raises(RuntimeError, match="missing Aurora key"):
        load_aurora_state_dict(model, {"model_state_dict": state})


def test_unexpected_aurora_key_raises() -> None:
    model = build_model()
    state = DummyAurora().state_dict()
    state["encoder.does_not_exist"] = torch.zeros(2)
    with pytest.raises(RuntimeError, match="unexpected key"):
        load_aurora_state_dict(model, {"model_state_dict": state})


def test_shape_mismatch_raises() -> None:
    model = build_model()
    state = DummyAurora().state_dict()
    state["encoder.weight"] = torch.zeros(9, 9)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_aurora_state_dict(model, {"model_state_dict": state})


def test_refinement_only_checkpoint_never_touches_aurora() -> None:
    model = build_model()
    payload = build_refinement_checkpoint(
        model, kind=CHECKPOINT_KIND_REFINEMENT, refinement_type="diffusion_unet"
    )
    assert all(key.startswith("refiner.") for key in payload["model_state_dict"])

    target = build_model()
    before = aurora_only_state(target)
    report = load_refinement_state_dict(target, payload)
    assert report.missing == []
    assert report.unexpected == []
    for key, value in before.items():
        assert torch.equal(target.state_dict()[f"aurora.{key}"], value)
    # Refiner weights actually transferred.
    for key, value in payload["model_state_dict"].items():
        assert torch.equal(target.state_dict()[key], value)


def test_refinement_only_checkpoint_includes_optional_temporal_state() -> None:
    config = refinement_config("diffusion_transformer")
    config["model"].update(
        {
            "mamba_temporal_enabled": True,
            "mamba_temporal_channels": 4,
            "mamba_temporal_state": 2,
            "mamba_temporal_layers": 1,
            "mamba_temporal_conv": 2,
            "mamba_temporal_expand": 1,
        }
    )
    model = build_two_phase_refiner(DummyAurora(), build_packing(8, 8), config)
    model.initialize_refiner(model.conditioning_channels())
    payload = build_refinement_checkpoint(
        model,
        kind=CHECKPOINT_KIND_REFINEMENT,
        refinement_type="diffusion_transformer",
    )
    assert any(key.startswith("refiner.") for key in payload["model_state_dict"])
    assert any(key.startswith("temporal.") for key in payload["model_state_dict"])
    assert all(
        key.startswith(("refiner.", "temporal."))
        for key in payload["model_state_dict"]
    )

    target = build_two_phase_refiner(DummyAurora(), build_packing(8, 8), config)
    target.initialize_refiner(target.conditioning_channels())
    report = load_refinement_state_dict(target, payload)
    assert report.missing == []
    assert report.unexpected == []
    for key, value in payload["model_state_dict"].items():
        assert torch.equal(target.state_dict()[key], value)


def test_refinement_checkpoint_with_missing_keys_raises() -> None:
    model = build_model()
    payload = build_refinement_checkpoint(model, kind=CHECKPOINT_KIND_REFINEMENT)
    keys = list(payload["model_state_dict"])
    payload["model_state_dict"].pop(keys[0])
    with pytest.raises(RuntimeError, match="missing key"):
        load_refinement_state_dict(build_model(), payload)


def test_combined_checkpoint_contains_both_phases() -> None:
    model = build_model()
    payload = build_refinement_checkpoint(model, kind=CHECKPOINT_KIND_COMBINED)
    assert any(key.startswith("aurora.") for key in payload["model_state_dict"])
    assert any(key.startswith("refiner.") for key in payload["model_state_dict"])


def test_aurora_only_checkpoint_contains_no_refiner_keys() -> None:
    model = build_model()
    payload = build_refinement_checkpoint(model, kind=CHECKPOINT_KIND_AURORA)
    assert all(key.startswith("aurora.") for key in payload["model_state_dict"])


def test_unknown_checkpoint_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint kind"):
        build_refinement_checkpoint(build_model(), kind="something")


# --------------------------------------------------------------------------
# Resume metadata
# --------------------------------------------------------------------------


def test_refinement_checkpoint_records_everything_needed_for_resume() -> None:
    model = build_model()
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    payload = build_refinement_checkpoint(
        model,
        kind=CHECKPOINT_KIND_REFINEMENT,
        aurora_checkpoint="/path/to/aurora.ckpt",
        aurora_fingerprint=aurora_state_fingerprint(aurora_only_state(model)),
        refinement_type=model.refinement_config.type,
        resolved_config=model.refinement_config.to_dict(),
        packing=model.packing,
        precision={"mode": "fp32", "allow_tf32": False},
        epoch=3,
        global_step=120,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    for key in (
        "checkpoint_kind",
        "checkpoint_schema_version",
        "model_state_dict",
        "refinement_type",
        "epoch",
        "global_step",
        "aurora_checkpoint",
        "aurora_fingerprint",
        "resolved_config",
        "field_packing",
        "precision",
        "rng_state",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        assert key in payload, key
    packing_meta = payload["field_packing"]
    assert [c["aurora_name"] for c in packing_meta["channels"]] == ["gtco3", "go3", "go3"]
    assert [c["level"] for c in packing_meta["channels"]] == [None, 500.0, 850.0]
    assert packing_meta["lead_time_scale_hours"] == 72.0
    assert payload["epoch"] == 3 and payload["global_step"] == 120


def test_checkpoint_rng_restore_matches_uninterrupted_stream() -> None:
    torch.manual_seed(314159)
    torch.randn(7)
    checkpoint = {
        "rng_state": {
            "cpu": torch.get_rng_state().clone(),
            "cuda": None,
            "cuda_current": None,
        }
    }
    uninterrupted = torch.randn(11)

    # Simulate stochastic calibration work in a newly launched resume process.
    torch.manual_seed(271828)
    torch.randn(23)
    assert ft.restore_checkpoint_rng_state(checkpoint, device="cpu") is True
    resumed = torch.randn(11)
    assert torch.equal(resumed, uninterrupted)


def test_checkpoint_rng_restore_accepts_cuda_device_ordinal(monkeypatch) -> None:
    checkpoint = {
        "rng_state": {
            "cpu": torch.get_rng_state().clone(),
            "cuda_current": torch.arange(8, dtype=torch.uint8),
        }
    }
    restored: dict[str, object] = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def capture_cuda_state(state: torch.Tensor, device=None) -> None:
        restored["state"] = state
        restored["device"] = device

    monkeypatch.setattr(torch.cuda, "set_rng_state", capture_cuda_state)

    assert ft.restore_checkpoint_rng_state(checkpoint, device=3) is True
    assert restored["device"] == torch.device("cuda", 3)
    assert torch.equal(restored["state"], checkpoint["rng_state"]["cuda_current"])


def test_checkpoint_rng_restore_rejects_boolean_device_ordinal() -> None:
    checkpoint = {"rng_state": {"cpu": torch.get_rng_state().clone()}}
    with pytest.raises(TypeError, match="CUDA ordinal"):
        ft.restore_checkpoint_rng_state(checkpoint, device=False)


@pytest.mark.parametrize(
    ("saved_head", "current_head"),
    [
        ("shared_process", "separate_mean"),
        ("separate_mean", "shared_process"),
    ],
)
def test_checkpoint_rejects_deterministic_head_migration_in_both_directions(
    saved_head: str,
    current_head: str,
) -> None:
    saved_config = refinement_config(
        "diffusion_unet", deterministic_head=saved_head
    )
    current_config = refinement_config(
        "diffusion_unet", deterministic_head=current_head
    )
    packing = build_packing(8, 8)
    saved_model = build_two_phase_refiner(
        DummyAurora(), packing, saved_config
    )
    current_model = build_two_phase_refiner(
        DummyAurora(), packing, current_config
    )
    checkpoint = {
        "config": saved_config,
        "resolved_refinement_config": saved_model.refinement_config.to_dict(),
        "field_packing": packing.to_dict(),
    }

    with pytest.raises(
        ValueError,
        match="Checkpoint unified refinement mismatch: deterministic_head",
    ):
        validate_unified_checkpoint_contract(
            current_model,
            checkpoint,
            current_config,
        )


@pytest.mark.parametrize(
    ("refinement_type", "inactive_section", "key", "value"),
    [
        ("diffusion_unet", "flow_matching", "integration_steps", 7),
        ("diffusion_unet", "transformer", "embedding_dim", 48),
        ("flow_matching_transformer", "diffusion", "inference_steps", 7),
        ("flow_matching_transformer", "unet", "hidden_channels", 12),
    ],
)
def test_checkpoint_ignores_inactive_process_and_backbone_sections(
    refinement_type: str,
    inactive_section: str,
    key: str,
    value: object,
) -> None:
    saved_config = refinement_config(refinement_type)
    current_config = copy.deepcopy(saved_config)
    inactive = current_config["model"]["refinement"].setdefault(
        inactive_section, {}
    )
    inactive[key] = value
    packing = build_packing(8, 8)
    saved_model = build_two_phase_refiner(DummyAurora(), packing, saved_config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        current_model = build_two_phase_refiner(
            DummyAurora(), packing, current_config
        )
        validate_unified_checkpoint_contract(
            current_model,
            {
                "config": saved_config,
                "resolved_refinement_config": (
                    saved_model.refinement_config.to_dict()
                ),
                "field_packing": packing.to_dict(),
            },
            current_config,
        )


@pytest.mark.parametrize(
    ("refinement_type", "active_section", "key", "value"),
    [
        ("diffusion_unet", "diffusion", "inference_steps", 7),
        ("diffusion_unet", "unet", "hidden_channels", 12),
        ("flow_matching_transformer", "flow_matching", "integration_steps", 7),
        ("flow_matching_transformer", "transformer", "embedding_dim", 48),
    ],
)
def test_checkpoint_rejects_active_process_and_backbone_sections(
    refinement_type: str,
    active_section: str,
    key: str,
    value: object,
) -> None:
    saved_config = refinement_config(refinement_type)
    current_config = copy.deepcopy(saved_config)
    current_config["model"]["refinement"][active_section][key] = value
    packing = build_packing(8, 8)
    saved_model = build_two_phase_refiner(DummyAurora(), packing, saved_config)
    current_model = build_two_phase_refiner(DummyAurora(), packing, current_config)
    checkpoint = {
        "config": saved_config,
        "resolved_refinement_config": saved_model.refinement_config.to_dict(),
        "field_packing": packing.to_dict(),
    }

    with pytest.raises(
        ValueError,
        match=rf"Checkpoint unified refinement mismatch: {active_section}",
    ):
        validate_unified_checkpoint_contract(
            current_model,
            checkpoint,
            current_config,
        )


def test_checkpoint_round_trips_residual_amplitude_safeguard_and_statistics() -> None:
    config = refinement_config(
        "diffusion_unet",
        deterministic_head="shared_process",
        target_space={
            "residual_scaling": "per_channel",
            "residual_scaling_center": True,
            "residual_clip_standard_deviations": 4.0,
        },
    )
    model = build_two_phase_refiner(DummyAurora(), build_packing(8, 8), config)
    model.initialize_refiner(model.conditioning_channels())
    assert model.refiner is not None
    calibration = torch.tensor(
        [[
            [[-2.0, 0.0], [2.0, 4.0]],
            [[-1.0, 1.0], [3.0, 5.0]],
            [[-4.0, -2.0], [0.0, 2.0]],
        ]]
    )
    model.refiner.fit_residual_scale(calibration)
    payload = build_refinement_checkpoint(
        model,
        kind=CHECKPOINT_KIND_REFINEMENT,
        refinement_type=model.refinement_config.type,
        resolved_config=model.refinement_config.to_dict(),
    )

    assert (
        payload["resolved_config"]["target_space"][
            "residual_clip_standard_deviations"
        ]
        == 4.0
    )

    restored = build_two_phase_refiner(DummyAurora(), build_packing(8, 8), config)
    restored.initialize_refiner(restored.conditioning_channels())
    report = load_refinement_state_dict(restored, payload)
    assert report.missing == []
    assert report.unexpected == []
    assert restored.refiner is not None
    assert restored.refiner.residual_clip_standard_deviations == 4.0
    torch.testing.assert_close(
        restored.refiner.residual_scaler.scale, model.refiner.residual_scaler.scale
    )
    torch.testing.assert_close(
        restored.refiner.residual_scaler.shift, model.refiner.residual_scaler.shift
    )


def test_aurora_reference_validation() -> None:
    model = build_model()
    state = aurora_only_state(model)
    payload = build_refinement_checkpoint(
        model,
        kind=CHECKPOINT_KIND_REFINEMENT,
        aurora_fingerprint=aurora_state_fingerprint(state),
    )
    assert validate_aurora_reference(payload, state) is True

    other = aurora_only_state(build_model())
    with pytest.raises(RuntimeError, match="Aurora identity mismatch"):
        validate_aurora_reference(payload, other)


def test_missing_fingerprint_is_refused_in_strict_mode() -> None:
    payload = build_refinement_checkpoint(build_model(), kind=CHECKPOINT_KIND_REFINEMENT)
    payload["aurora_fingerprint"] = None
    with pytest.raises(RuntimeError, match="does not record an Aurora fingerprint"):
        validate_aurora_reference(payload, {})
    assert validate_aurora_reference(payload, {}, strict=False) is False


def test_atomic_save_leaves_no_temporary_files(tmp_path) -> None:
    target = tmp_path / "nested" / "refinement.ckpt"
    save_checkpoint_atomic({"a": torch.ones(2)}, target)
    assert target.exists()
    assert [p.name for p in target.parent.iterdir()] == ["refinement.ckpt"]
    restored = torch.load(target, weights_only=False)
    assert torch.equal(restored["a"], torch.ones(2))
