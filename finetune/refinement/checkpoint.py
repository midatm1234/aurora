"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Checkpoint compatibility for Aurora stochastic residual refinement.

Design rules
------------
* Existing Aurora pretrained, fine-tuned and flow-matching checkpoints are
  **never** modified, renamed or converted in place. They are read-only inputs.
* Aurora module names are preserved. The only change is the ``aurora.`` prefix
  introduced by :class:`~finetune.refinement.two_phase.AuroraTwoPhaseRefiner`,
  performed explicitly by :func:`migrate_aurora_state_dict` together with the
  historical ``module.`` / ``_orig_mod.`` wrapper prefixes.
* Aurora weights are loaded **strictly** after migration. ``strict=False`` is
  never used as a blanket escape hatch: :func:`load_aurora_state_dict` proves
  that every missing key belongs to the newly introduced refinement module and
  that no Aurora key is unexpected or shape-mismatched.
* Refinement-only checkpoints record the identity (SHA-256 over the Aurora
  tensors) of the Aurora checkpoint they were trained against, the refinement
  type, the variable/level mapping, the target-space normalization, the
  forecast lead-time configuration, the fully resolved YAML configuration, the
  precision settings, optimizer/scheduler/scaler state, epoch, global step and
  RNG states.

Adapted from ``granitewxc.refinement.checkpoint`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

__all__ = [
    "CHECKPOINT_KIND_AURORA",
    "CHECKPOINT_KIND_COMBINED",
    "CHECKPOINT_KIND_REFINEMENT",
    "CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_REFINEMENT_KEY_PREFIXES",
    "StateDictReport",
    "aurora_state_fingerprint",
    "build_refinement_checkpoint",
    "extract_model_state",
    "load_aurora_state_dict",
    "load_refinement_state_dict",
    "migrate_aurora_state_dict",
    "migrate_legacy_flow_state_dict",
    "save_checkpoint_atomic",
    "strip_wrapper_prefixes",
    "validate_aurora_reference",
]

CHECKPOINT_KIND_AURORA = "aurora"
CHECKPOINT_KIND_REFINEMENT = "refinement"
CHECKPOINT_KIND_COMBINED = "combined"
CHECKPOINT_SCHEMA_VERSION = 1

#: Wrapper prefixes historically produced by DDP / FSDP / ``torch.compile``.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")

#: Top-level keys of a legacy ``AuroraFlowRefine`` checkpoint that belong to the
#: refinement heads rather than to Aurora. They are mapped **explicitly** onto
#: the unified wrapper's ``refiner.legacy.*`` namespace; nothing is ever moved
#: by a blanket ``strict=False``.
LEGACY_REFINEMENT_KEY_PREFIXES: tuple[str, ...] = (
    "surf_flow.",
    "atmos_flow.",
    "temporal.",
    "_res_std__",
)

#: Explicit legacy -> current Aurora key renames. Empty today: the Aurora
#: architecture on this branch is key-compatible with the source branch. Any
#: future rename must be added here, never handled by ``strict=False``.
LEGACY_AURORA_KEY_RENAMES: dict[str, str] = {}


@dataclass
class StateDictReport:
    """Outcome of a controlled state-dict load."""

    loaded: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )
    renamed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"loaded={self.loaded} missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} "
            f"shape_mismatched={len(self.shape_mismatched)} renamed={len(self.renamed)}"
        )


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the model tensors from any of the checkpoint layouts in use."""
    if checkpoint is None:
        raise ValueError("Checkpoint is empty.")
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "model", "state_dict", "aurora"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                state = checkpoint[key]
                break
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, Mapping):
        raise ValueError(
            f"Could not locate a state dict in checkpoint of type {type(checkpoint)!r}."
        )
    return {str(k): v for k, v in state.items()}


def strip_wrapper_prefixes(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove ``module.`` / ``_orig_mod.`` wrapper prefixes (possibly nested)."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def _is_legacy_refinement_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in LEGACY_REFINEMENT_KEY_PREFIXES)


def migrate_legacy_flow_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    refiner_prefix: str = "refiner.legacy.",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Map a legacy ``AuroraFlowRefine`` state dict onto the unified wrapper.

    Legacy checkpoints store the Aurora backbone under ``base.*`` and the
    refinement heads at the top level (``surf_flow.*``, ``atmos_flow.*``,
    ``_res_std__*``). Every rename is explicit and reported.
    """
    stripped = strip_wrapper_prefixes(state)
    migrated: dict[str, torch.Tensor] = {}
    renames: dict[str, str] = {}
    for key, value in stripped.items():
        if key.startswith("base."):
            new_key = "aurora." + key[len("base.") :]
        elif _is_legacy_refinement_key(key):
            new_key = refiner_prefix + key
        else:
            new_key = key
        migrated[new_key] = value
        if new_key != key:
            renames[key] = new_key
    return migrated, renames


def migrate_aurora_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    target_prefix: str = "aurora.",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Map a deterministic Aurora state dict onto the two-phase wrapper.

    Returns ``(migrated_state, renames)`` where ``renames`` documents every key
    whose name changed, so the migration is auditable.
    """
    stripped = strip_wrapper_prefixes(state)
    migrated: dict[str, torch.Tensor] = {}
    renames: dict[str, str] = {}
    for key, value in stripped.items():
        new_key = LEGACY_AURORA_KEY_RENAMES.get(key, key)
        if new_key.startswith("refiner."):
            migrated[new_key] = value
            continue
        if _is_legacy_refinement_key(new_key):
            new_key = "refiner.legacy." + new_key
            migrated[new_key] = value
            renames[key] = new_key
            continue
        if new_key.startswith("base."):
            new_key = target_prefix + new_key[len("base.") :]
        elif not new_key.startswith(target_prefix):
            new_key = target_prefix + new_key
        migrated[new_key] = value
        if new_key != key:
            renames[key] = new_key
    return migrated, renames


def aurora_state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Stable SHA-256 identity of an Aurora tensor collection.

    Hashes ``(key, dtype, shape, raw bytes)`` for every tensor in sorted key
    order, so it is independent of dict ordering and of the wrapper prefix.
    """
    digest = hashlib.sha256()
    normalized = strip_wrapper_prefixes(state)
    for key in sorted(normalized):
        value = normalized[key]
        if not torch.is_tensor(value):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_aurora_state_dict(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    target_prefix: str = "aurora.",
    allow_missing_refinement: bool = True,
    skip_keys: Iterable[str] = (),
) -> StateDictReport:
    """Load an Aurora checkpoint into a two-phase wrapper, strictly.

    Missing keys are tolerated only when they belong to the newly introduced
    refinement module (``refiner.*``) and ``allow_missing_refinement`` is set.
    Any other missing key, any unexpected key and any shape mismatch raises.

    ``skip_keys`` may name Aurora buffers that are intentionally rebuilt from
    the current configuration. Skipped keys are reported, never silently
    dropped.
    """
    raw = extract_model_state(checkpoint)
    migrated, renames = migrate_aurora_state_dict(raw, target_prefix=target_prefix)

    skip = {str(k) for k in skip_keys}
    if skip:
        migrated = {k: v for k, v in migrated.items() if not any(part in k for part in skip)}

    model_state = model.state_dict()
    report = StateDictReport(renamed=renames)

    for key, value in migrated.items():
        target = model_state.get(key)
        if target is None:
            report.unexpected.append(key)
            continue
        if torch.is_tensor(value) and tuple(target.shape) != tuple(value.shape):
            report.shape_mismatched.append((key, tuple(target.shape), tuple(value.shape)))
            continue
        model_state[key] = value
        report.loaded += 1

    provided = set(migrated)
    report.missing = [key for key in model_state if key not in provided]

    if report.shape_mismatched:
        details = ", ".join(f"{k}: model{m} vs ckpt{c}" for k, m, c in report.shape_mismatched[:8])
        raise RuntimeError(
            f"Refusing to load checkpoint: {len(report.shape_mismatched)} shape "
            f"mismatch(es). {details}"
        )
    if report.unexpected:
        raise RuntimeError(
            f"Refusing to load checkpoint: {len(report.unexpected)} unexpected key(s), "
            f"e.g. {sorted(report.unexpected)[:8]}"
        )

    unexplained = [k for k in report.missing if not k.startswith("refiner.")]
    unexplained = [k for k in unexplained if not any(part in k for part in skip)]
    if unexplained:
        raise RuntimeError(
            f"Refusing to load checkpoint: {len(unexplained)} missing Aurora key(s), "
            f"e.g. {sorted(unexplained)[:8]}"
        )
    if report.missing and not allow_missing_refinement:
        raise RuntimeError(
            f"Checkpoint is missing {len(report.missing)} key(s) and "
            "allow_missing_refinement is disabled."
        )

    # ``strict`` is only relaxed *because* every missing key was proven above to
    # belong to the newly introduced refinement module.
    model.load_state_dict(model_state, strict=not report.missing)
    return report


def load_refinement_state_dict(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    prefix: str = "refiner.",
    aurora_prefix: str = "aurora.",
) -> StateDictReport:
    """Load Phase-2 weights without touching Aurora weights.

    Only keys under ``prefix`` are applied; the Aurora sub-module is left
    exactly as it was, which is verified after the load.
    """
    raw = extract_model_state(checkpoint)
    stripped = strip_wrapper_prefixes(raw)
    incoming = {k: v for k, v in stripped.items() if k.startswith(prefix)}
    if not incoming:
        legacy = {k: v for k, v in stripped.items() if _is_legacy_refinement_key(k)}
        if legacy:
            incoming = {prefix + "legacy." + k: v for k, v in legacy.items()}
        else:
            # Bare refiner state dict (no wrapper prefix).
            incoming = {prefix + k: v for k, v in stripped.items()}

    model_state = model.state_dict()
    report = StateDictReport()
    aurora_before = {k: v for k, v in model_state.items() if k.startswith(aurora_prefix)}

    for key, value in incoming.items():
        target = model_state.get(key)
        if target is None:
            report.unexpected.append(key)
            continue
        if torch.is_tensor(value) and tuple(target.shape) != tuple(value.shape):
            report.shape_mismatched.append((key, tuple(target.shape), tuple(value.shape)))
            continue
        model_state[key] = value
        report.loaded += 1

    refiner_keys = {key for key in model_state if key.startswith(prefix)}
    report.missing = sorted(refiner_keys - set(incoming))

    if report.shape_mismatched:
        raise RuntimeError(
            f"Refusing to load refinement checkpoint: shape mismatch(es) "
            f"{report.shape_mismatched[:8]}"
        )
    if report.unexpected:
        raise RuntimeError(
            f"Refusing to load refinement checkpoint: unexpected key(s) "
            f"{sorted(report.unexpected)[:8]}"
        )
    if report.missing:
        raise RuntimeError(
            f"Refusing to load refinement checkpoint: missing key(s) {report.missing[:8]}"
        )

    model.load_state_dict(model_state, strict=True)
    after = {k: v for k, v in model.state_dict().items() if k.startswith(aurora_prefix)}
    for key, before in aurora_before.items():
        if torch.is_tensor(before) and not torch.equal(before, after[key]):
            raise RuntimeError(
                f"Loading a refinement checkpoint changed Aurora weight {key!r}. "
                "This must never happen."
            )
    return report


def save_checkpoint_atomic(
    payload: Mapping[str, Any], path: str | os.PathLike, *, atomic: bool = True
) -> str:
    """Persist a checkpoint, optionally via a same-directory temporary file.

    Atomic writes prevent a crash mid-save from leaving a truncated checkpoint
    that would break resume.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    if not atomic:
        torch.save(payload, path)
        return path
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def build_refinement_checkpoint(
    model: torch.nn.Module,
    *,
    kind: str = CHECKPOINT_KIND_REFINEMENT,
    aurora_checkpoint: str | None = None,
    aurora_fingerprint: str | None = None,
    refinement_type: str = "none",
    resolved_config: Mapping[str, Any] | None = None,
    packing: Any = None,
    precision: Mapping[str, Any] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    extra: Mapping[str, Any] | None = None,
    refiner_prefix: str = "refiner.",
    aurora_prefix: str = "aurora.",
) -> dict[str, Any]:
    """Assemble a checkpoint payload recording everything needed for resume.

    ``kind`` distinguishes ``aurora`` / ``refinement`` / ``combined`` payloads. A
    ``refinement`` checkpoint deliberately excludes the (unchanged, frozen)
    Aurora weights and instead records the Aurora checkpoint path and
    fingerprint, so it stays small while remaining verifiable.
    """
    if kind not in {CHECKPOINT_KIND_AURORA, CHECKPOINT_KIND_REFINEMENT, CHECKPOINT_KIND_COMBINED}:
        raise ValueError(f"Unsupported checkpoint kind {kind!r}")

    full_state = model.state_dict()
    if kind == CHECKPOINT_KIND_REFINEMENT:
        model_state = {k: v for k, v in full_state.items() if k.startswith(refiner_prefix)}
    elif kind == CHECKPOINT_KIND_AURORA:
        model_state = {k: v for k, v in full_state.items() if k.startswith(aurora_prefix)}
    else:
        model_state = dict(full_state)

    payload: dict[str, Any] = {
        "checkpoint_kind": kind,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model_state,
        "refinement_type": str(refinement_type),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "aurora_checkpoint": aurora_checkpoint,
        "aurora_fingerprint": aurora_fingerprint,
        "resolved_config": dict(resolved_config or {}),
        "field_packing": packing.to_dict() if hasattr(packing, "to_dict") else packing,
        "precision": dict(precision or {}),
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        payload["scheduler_state_dict"] = {
            k: v for k, v in sched_state.items() if k != "anneal_func"
        }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(dict(extra))
    return payload


def validate_aurora_reference(
    checkpoint: Mapping[str, Any],
    aurora_state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> bool:
    """Check that a refinement checkpoint matches the loaded Aurora weights."""
    expected = checkpoint.get("aurora_fingerprint")
    if not expected:
        if strict:
            raise RuntimeError(
                "Refinement checkpoint does not record an Aurora fingerprint; refusing "
                "to assume compatibility. Re-save it with build_refinement_checkpoint()."
            )
        return False
    actual = aurora_state_fingerprint(aurora_state)
    if actual != expected:
        message = (
            "Aurora identity mismatch: the refinement checkpoint was trained against "
            f"Aurora fingerprint {expected[:16]}... but the loaded Aurora is "
            f"{actual[:16]}...  (referenced checkpoint: "
            f"{checkpoint.get('aurora_checkpoint')!r})"
        )
        if strict:
            raise RuntimeError(message)
        return False
    return True
