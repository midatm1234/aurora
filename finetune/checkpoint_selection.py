"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Lightweight, provenance-aware refinement checkpoint selection.

This module intentionally has no Aurora, PyTorch, plotting, or NetCDF imports.
Both notebooks and the distributed trainer can therefore inspect checkpoint
sidecars before allocating a model or importing optional visualization stacks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "refinement_checkpoint_status",
    "select_refinement_checkpoint",
    "validate_checkpoint_validation_provenance",
]


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"Invalid {label} at {path}: expected a JSON object, "
            f"got {type(value).__name__}."
        )
    return value


def _finite_best_loss(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    try:
        return math.isfinite(float(metadata.get("best_val_loss", math.inf)))
    except (TypeError, ValueError):
        return False


def _validated(metadata: dict[str, Any] | None) -> bool:
    """Return whether sidecar metadata explicitly represents accepted weights.

    Old sidecars predate ``validated_for_inference``. Their established
    contract was a finite ``best_val_loss``, which remains the compatibility
    fallback. New sidecars additionally require the explicit acceptance flag,
    preventing a non-best ``last.ckpt`` from being promoted merely because an
    earlier epoch in the same run had a finite best score.
    """
    if not _finite_best_loss(metadata):
        return False
    assert metadata is not None
    explicit = metadata.get("validated_for_inference")
    return explicit is True if explicit is not None else True


def validate_checkpoint_validation_provenance(
    checkpoint: dict[str, Any],
    *,
    require_validated: bool,
) -> None:
    """Reject checkpoint payloads not accepted for production inference.

    A ``last.ckpt`` can contain the finite best score from an earlier epoch
    while its own, later weights failed validation. New checkpoints therefore
    carry an explicit per-payload flag. Checkpoints written before that flag
    retain the historical finite-score fallback.
    """
    if not require_validated:
        return
    try:
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint field best_val_loss must be a finite number for "
            f"validated inference; got {checkpoint.get('best_val_loss')!r}."
        ) from exc
    explicit = checkpoint.get("validated_for_inference")
    if explicit is not None and not isinstance(explicit, bool):
        raise ValueError(
            "Checkpoint field validated_for_inference must be boolean when "
            f"present; got {explicit!r}."
        )
    if explicit is False:
        validation = checkpoint.get("validation")
        status = (
            validation.get("status")
            if isinstance(validation, dict)
            else None
        )
        raise ValueError(
            "Checkpoint field validated_for_inference is false: these exact "
            "weights were not accepted for production inference"
            + (f" (validation status={status!r})" if status is not None else "")
            + ". Select the matching accepted best.ckpt or explicitly use "
            "last.ckpt only for diagnostics."
        )
    if not math.isfinite(best_val_loss):
        raise ValueError(
            "Checkpoint has no finite validation score "
            f"(best_val_loss={best_val_loss!r}). Enable validation, select a "
            "non-degrading best checkpoint, and rerun training."
        )


def _checkpoint_entry(
    path: Path,
    metadata: dict[str, Any] | None,
    *,
    current_run_id: str | None,
) -> dict[str, Any]:
    run_id = metadata.get("training_run_id") if metadata is not None else None
    belongs_to_current_run = bool(
        current_run_id and run_id and str(run_id) == current_run_id
    )
    return {
        "path": str(path),
        "exists": path.exists(),
        "metadata_present": metadata is not None,
        "training_run_id": run_id,
        "belongs_to_current_run": belongs_to_current_run,
        "finite_best_val_loss": _finite_best_loss(metadata),
        "validated_for_inference": _validated(metadata),
        "metadata": metadata,
    }


def refinement_checkpoint_status(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Return JSON-serializable best/last/current-run selection provenance.

    The status is intentionally descriptive rather than policy-bearing: callers
    still choose whether validated weights are mandatory. It is useful for
    notebook diagnostics and for actionable errors without loading multi-GB
    checkpoint payloads.
    """
    checkpoint_dir = Path(checkpoint_dir)
    best = checkpoint_dir / "best.ckpt"
    last = checkpoint_dir / "last.ckpt"
    marker_path = checkpoint_dir / "training_run.metadata.json"
    best_meta = _read_json_mapping(
        best.with_suffix(best.suffix + ".metadata.json"),
        label="best-checkpoint metadata",
    )
    last_meta = _read_json_mapping(
        last.with_suffix(last.suffix + ".metadata.json"),
        label="last-checkpoint metadata",
    )
    marker = _read_json_mapping(marker_path, label="training-run metadata")

    marker_run_id = marker.get("training_run_id") if marker is not None else None
    fallback_run_id = (
        last_meta.get("training_run_id")
        if last_meta is not None
        else best_meta.get("training_run_id") if best_meta is not None else None
    )
    raw_run_id = marker_run_id or fallback_run_id
    current_run_id = str(raw_run_id) if raw_run_id else None

    best_entry = _checkpoint_entry(
        best, best_meta, current_run_id=current_run_id
    )
    last_entry = _checkpoint_entry(
        last, last_meta, current_run_id=current_run_id
    )
    latest_validation = None
    for source in (last_meta, marker):
        candidate = source.get("validation") if isinstance(source, dict) else None
        if isinstance(candidate, dict):
            latest_validation = candidate
            break

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "training_run_marker_present": marker is not None,
        "training_run_id": current_run_id,
        "run_marker": marker,
        "latest_validation": latest_validation,
        "best": best_entry,
        "last": last_entry,
    }


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    return f"{number:.8g}" if math.isfinite(number) else str(number)


def _status_detail(status: dict[str, Any]) -> str:
    best = status["best"]
    last = status["last"]
    details = [
        f"checkpoint_dir={status['checkpoint_dir']}",
        f"training_run_id={status['training_run_id']!r}",
        (
            "best.ckpt="
            + (
                "missing"
                if not best["exists"]
                else "metadata-missing"
                if not best["metadata_present"]
                else "stale-run"
                if not best["belongs_to_current_run"]
                else "accepted"
                if best["validated_for_inference"]
                else "not-accepted"
            )
        ),
        (
            "last.ckpt="
            + (
                "missing"
                if not last["exists"]
                else "metadata-missing"
                if not last["metadata_present"]
                else "current-run"
                if last["belongs_to_current_run"]
                else "stale-run"
            )
        ),
    ]
    last_meta = last.get("metadata")
    if isinstance(last_meta, dict):
        if last_meta.get("epoch") is not None:
            details.append(f"last_epoch={last_meta['epoch']}")
        if last_meta.get("global_step") is not None:
            details.append(f"last_global_step={last_meta['global_step']}")
        details.append(
            "recorded_best_val_loss="
            + _format_number(last_meta.get("best_val_loss", math.inf))
        )
    validation = status.get("latest_validation")
    if isinstance(validation, dict):
        for key in (
            "status",
            "epoch",
            "val_loss",
            "baseline_val_loss",
            "val_improvement_percent",
        ):
            if key in validation:
                value = validation[key]
                details.append(
                    f"validation_{key}="
                    + (repr(value) if key == "status" else _format_number(value))
                )
    return "; ".join(details) + "."


def select_refinement_checkpoint(
    checkpoint_dir: str | Path,
    *,
    require_validated: bool = False,
) -> Path:
    """Select a checkpoint from the latest logical training run.

    A validated best checkpoint is preferred. With ``require_validated=False``,
    the current run's last checkpoint is an explicit diagnostic fallback.
    Stale best checkpoints from previous runs are never selected implicitly.
    """
    checkpoint_dir = Path(checkpoint_dir)
    status = refinement_checkpoint_status(checkpoint_dir)
    best = checkpoint_dir / "best.ckpt"
    last = checkpoint_dir / "last.ckpt"
    best_status = status["best"]
    last_status = status["last"]
    has_marker = bool(status["training_run_marker_present"])
    current_run_id = status["training_run_id"]

    if has_marker and current_run_id:
        if (
            best_status["exists"]
            and best_status["belongs_to_current_run"]
            and best_status["validated_for_inference"]
        ):
            return best
        if require_validated:
            raise ValueError(
                "The latest training run has no validated best checkpoint. "
                "Its refinement did not pass the configured validation acceptance "
                "criterion, so production inference cannot fall back to last.ckpt "
                "or a stale best.ckpt. "
                + _status_detail(status)
            )
        if last_status["exists"] and last_status["belongs_to_current_run"]:
            return last
        raise FileNotFoundError(
            "The current training-run marker has no matching checkpoint. "
            + _status_detail(status)
        )

    # No marker: retain sidecar-based compatibility for runs produced by the
    # first provenance-aware pipeline version.
    if last_status["exists"] and last_status["metadata_present"]:
        run_id = last_status["training_run_id"]
        same_run_best = bool(
            best_status["exists"]
            and best_status["metadata_present"]
            and run_id
            and best_status["training_run_id"] == run_id
            and best_status["validated_for_inference"]
        )
        if same_run_best:
            return best
        if require_validated:
            raise ValueError(
                "The latest training run has no matching validated best "
                "checkpoint. Its refinement did not pass validation, or its "
                "checkpoint metadata are incomplete. Do not fall back to a "
                "stale best.ckpt. "
                + _status_detail(status)
            )
        return last

    if require_validated:
        raise ValueError(
            "Checkpoint metadata sidecars are missing. These are legacy "
            "checkpoints whose run provenance cannot be verified. Retrain with "
            "the current pipeline before production inference. "
            + _status_detail(status)
        )
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(
        f"No best.ckpt or last.ckpt under {checkpoint_dir}. "
        + _status_detail(status)
    )
