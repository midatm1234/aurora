"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Checkpoint provenance tests without Aurora or plotting dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from finetune.checkpoint_selection import (
    refinement_checkpoint_status,
    select_refinement_checkpoint,
    validate_checkpoint_validation_provenance,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=True))


def _checkpoint(
    directory: Path,
    role: str,
    *,
    run_id: str,
    best_val_loss: float,
    validated: bool | None = None,
    validation: dict[str, object] | None = None,
) -> Path:
    path = directory / f"{role}.ckpt"
    path.touch()
    metadata: dict[str, object] = {
        "training_run_id": run_id,
        "epoch": 14,
        "global_step": 9690,
        "best_val_loss": best_val_loss,
    }
    if validated is not None:
        metadata["validated_for_inference"] = validated
    if validation is not None:
        metadata["validation"] = validation
    _write_json(path.with_suffix(".ckpt.metadata.json"), metadata)
    return path


def test_current_failed_run_is_strictly_rejected_with_actionable_context(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "training_run.metadata.json",
        {"training_run_id": "current"},
    )
    _checkpoint(
        tmp_path,
        "best",
        run_id="stale",
        best_val_loss=0.5,
        validated=True,
    )
    last = _checkpoint(
        tmp_path,
        "last",
        run_id="current",
        best_val_loss=math.inf,
        validated=False,
        validation={
            "status": "degrades_deterministic_baseline",
            "epoch": 14,
            "val_loss": 0.034391511,
            "baseline_val_loss": 0.0012777303,
            "val_improvement_percent": -2591.6095,
        },
    )

    with pytest.raises(ValueError) as caught:
        select_refinement_checkpoint(tmp_path, require_validated=True)

    message = str(caught.value)
    assert "no validated best checkpoint" in message
    assert "best.ckpt=stale-run" in message
    assert "last.ckpt=current-run" in message
    assert "degrades_deterministic_baseline" in message
    assert "validation_val_loss=0.034391511" in message
    assert "validation_baseline_val_loss=0.0012777303" in message
    assert select_refinement_checkpoint(
        tmp_path, require_validated=False
    ) == last

    status = refinement_checkpoint_status(tmp_path)
    assert status["training_run_id"] == "current"
    assert status["best"]["belongs_to_current_run"] is False
    assert status["last"]["belongs_to_current_run"] is True
    assert status["last"]["validated_for_inference"] is False


def test_current_explicitly_validated_best_is_selected(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "training_run.metadata.json",
        {"training_run_id": "current"},
    )
    best = _checkpoint(
        tmp_path,
        "best",
        run_id="current",
        best_val_loss=0.4,
        validated=True,
    )
    _checkpoint(
        tmp_path,
        "last",
        run_id="current",
        best_val_loss=0.4,
        validated=False,
    )

    assert select_refinement_checkpoint(
        tmp_path, require_validated=True
    ) == best


def test_finite_loss_cannot_override_explicit_rejection(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "training_run.metadata.json",
        {"training_run_id": "current"},
    )
    _checkpoint(
        tmp_path,
        "best",
        run_id="current",
        best_val_loss=0.4,
        validated=False,
    )
    last = _checkpoint(
        tmp_path,
        "last",
        run_id="current",
        best_val_loss=0.4,
        validated=False,
    )

    with pytest.raises(ValueError, match="best.ckpt=not-accepted"):
        select_refinement_checkpoint(tmp_path, require_validated=True)
    assert select_refinement_checkpoint(tmp_path) == last


def test_pre_explicit_flag_sidecars_remain_compatible(tmp_path: Path) -> None:
    best = _checkpoint(
        tmp_path,
        "best",
        run_id="legacy-sidecar-run",
        best_val_loss=0.4,
    )
    _checkpoint(
        tmp_path,
        "last",
        run_id="legacy-sidecar-run",
        best_val_loss=0.4,
    )

    assert select_refinement_checkpoint(
        tmp_path, require_validated=True
    ) == best


def test_malformed_metadata_has_field_specific_error(tmp_path: Path) -> None:
    (tmp_path / "last.ckpt").touch()
    (tmp_path / "last.ckpt.metadata.json").write_text("[")

    with pytest.raises(
        ValueError,
        match=r"Invalid last-checkpoint metadata JSON.*last\.ckpt\.metadata\.json",
    ):
        refinement_checkpoint_status(tmp_path)


def test_payload_rejects_non_best_weights_with_an_earlier_finite_score() -> None:
    checkpoint = {
        "best_val_loss": 0.4,
        "validated_for_inference": False,
        "validation": {"status": "degrades_deterministic_baseline"},
    }

    with pytest.raises(ValueError, match="validated_for_inference is false"):
        validate_checkpoint_validation_provenance(
            checkpoint,
            require_validated=True,
        )
    # Training resume and explicitly labelled notebook diagnostics can inspect
    # these weights without changing the production-inference default.
    validate_checkpoint_validation_provenance(
        checkpoint,
        require_validated=False,
    )


def test_payload_legacy_finite_score_fallback_is_preserved() -> None:
    validate_checkpoint_validation_provenance(
        {"best_val_loss": 0.4},
        require_validated=True,
    )
    with pytest.raises(ValueError, match="no finite validation score"):
        validate_checkpoint_validation_provenance(
            {"best_val_loss": math.inf},
            require_validated=True,
        )


def test_finetune_notebook_preserves_distributed_checkpoint_gate_manifest() -> None:
    notebook_path = (
        Path(__file__).parents[1] / "finetune" / "aurora_finetune_rollout.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code_cells):
        compile(source, f"aurora_finetune_rollout.ipynb:cell-{index}", "exec")
    source = "\n".join(code_cells)

    assert "existing_manifest.get('extras', {})" in source
    assert "manifest_extras.update(existing_extras)" in source
    assert "extras=manifest_extras" in source
    assert "'diagnostic_checkpoint': str(Path(ckpt_to_load).resolve())" in source
    assert "'diagnostic_checkpoint_validated_for_inference'" in source
    # Skipped production rollout artifacts are left untouched; provenance in
    # the merged manifest, rather than destructive cleanup, identifies them.
    assert "unlink(" not in source
