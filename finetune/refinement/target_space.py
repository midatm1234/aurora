"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Normalized target space used for Aurora residual refinement.

Phase 2 always operates in the **normalized target space** that the existing
Aurora fine-tuning loss already uses (Aurora's own per-variable / per-pressure
level location and scale constants from :mod:`aurora.normalisation`, resolved by
``compute_target_normalization_stats``). Refinement never happens in physical
units.

The contract is::

    rollout_norm = encode(aurora_rollout)  # physical -> normalized
    target_norm = encode(ground_truth)  # physical -> normalized
    residual_target = target_norm - rollout_norm  # ONE space

    predicted_residual = refiner(...)  # same space
    refined_norm = rollout_norm + predicted_residual
    refined_physical = decode(refined_norm)  # inverse-normalize ONCE
    refined_physical = apply_constraints(...)  # constraints ONCE
    refined_physical = apply_mask(...)  # mask ONCE

A normalized residual is never added to a physical field, physical and
normalized fields are never subtracted from each other, and the inverse
transform, the physical constraints and the mask are each applied exactly once.

Adapted from ``granitewxc.refinement.target_space.NormalizedTargetSpace`` in the
Prithvi stochastic residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0). The Prithvi version derives
its scalers from a downscaling model's output head; this version derives them
from Aurora's variable/pressure-level normalization metadata carried by
:class:`~finetune.refinement.packing.FieldPacking`.
"""

from __future__ import annotations

from typing import Sequence

import torch

from finetune.refinement.packing import FieldPacking

__all__ = ["NormalizedTargetSpace"]

_MIN_SCALE = 1.0e-12


class NormalizedTargetSpace:
    """Invertible mapping between physical Aurora/CAMS units and normalized space.

    Args:
        packing: the channel layout, which carries the per-channel location and
            scale of the target normalization.
        nonnegative_variables: Aurora variable names whose physical values must
            stay ``>= 0`` (e.g. mixing ratios). The constraint is applied once,
            after :meth:`decode`.
    """

    def __init__(
        self,
        packing: FieldPacking,
        *,
        nonnegative_variables: Sequence[str] = (),
    ) -> None:
        self.packing = packing
        self.nonnegative_variables = tuple(dict.fromkeys(nonnegative_variables))
        self._nonneg_mask = torch.tensor(
            [spec.aurora_name in set(self.nonnegative_variables) for spec in packing.channels],
            dtype=torch.bool,
        ).view(1, -1, 1, 1)

    # -- scalers ---------------------------------------------------------
    @property
    def num_channels(self) -> int:
        return self.packing.num_channels

    def _scalers(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return validated channel scalers in float32 on the reference device."""
        mean = self.packing.means(device=reference.device, dtype=torch.float32)
        std = self.packing.stds(device=reference.device, dtype=torch.float32)
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("Target-space normalization means contain non-finite values.")
        if not bool(torch.isfinite(std).all()):
            raise ValueError("Target-space normalization scales contain non-finite values.")
        if bool((std < _MIN_SCALE).any()):
            bad = torch.nonzero(std.reshape(-1) < _MIN_SCALE).reshape(-1).tolist()
            raise ValueError(
                "Target-space normalization scales must be at least "
                f"{_MIN_SCALE:g}; invalid packed channel indices={bad}."
            )
        return mean, std

    # -- transforms ------------------------------------------------------
    def encode(self, physical: torch.Tensor) -> torch.Tensor:
        """Map a packed physical field into normalized space in float32."""
        self._check_channels(physical)
        mean, std = self._scalers(physical)
        return (physical.float() - mean) / std

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse-normalize exactly once, with arithmetic kept in float32."""
        self._check_channels(normalized)
        mean, std = self._scalers(normalized)
        return normalized.float() * std + mean

    def correction_to_normalized(
        self, correction_physical: torch.Tensor
    ) -> torch.Tensor:
        """Convert a physical CAMS-minus-Aurora correction to normalized units.

        A correction is a difference, so the field-normalization location is
        intentionally not subtracted.
        """
        self._check_channels(correction_physical)
        _, std = self._scalers(correction_physical)
        return correction_physical.float() / std

    def correction_to_physical(
        self, correction_normalized: torch.Tensor
    ) -> torch.Tensor:
        """Convert a normalized CAMS-minus-Aurora correction to physical units."""
        self._check_channels(correction_normalized)
        _, std = self._scalers(correction_normalized)
        return correction_normalized.float() * std

    # -- constraints -----------------------------------------------------
    def physical_constraint_mask(self, physical: torch.Tensor) -> torch.Tensor:
        """Cells changed by configured non-negativity constraints."""
        self._check_channels(physical)
        enabled = self._nonneg_mask.to(device=physical.device)
        return enabled & torch.isfinite(physical) & (physical < 0)

    def apply_physical_constraints(self, physical: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints after constructing the corrected field."""
        self._check_channels(physical)
        if not bool(self._nonneg_mask.any()):
            return physical
        enabled = self._nonneg_mask.to(physical.device)
        return torch.where(enabled, physical.clamp(min=0.0), physical)

    # -- correction construction ----------------------------------------
    def correction_target(
        self,
        cams_target_physical: torch.Tensor,
        aurora_forecast_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build normalized CAMS-minus-Aurora and its finite validity mask."""
        self._check_channels(cams_target_physical)
        self._check_channels(aurora_forecast_normalized)
        if cams_target_physical.shape != aurora_forecast_normalized.shape:
            raise ValueError(
                "CAMS target and Aurora forecast shapes must match before "
                f"subtraction; got {tuple(cams_target_physical.shape)} and "
                f"{tuple(aurora_forecast_normalized.shape)}."
            )
        target32 = cams_target_physical.float()
        rollout32 = aurora_forecast_normalized.float()
        valid = torch.isfinite(target32) & torch.isfinite(rollout32)
        if valid_mask is not None:
            try:
                supplied = torch.broadcast_to(
                    valid_mask.to(device=valid.device, dtype=torch.bool), valid.shape
                )
            except RuntimeError as exc:
                raise ValueError(
                    "valid_mask must be broadcastable to correction shape "
                    f"{tuple(valid.shape)}; got {tuple(valid_mask.shape)}."
                ) from exc
            valid = valid & supplied
        filled = torch.where(valid, target32, torch.zeros_like(target32))
        target_normalized = self.encode(filled)
        correction = target_normalized - rollout32
        correction = torch.where(valid, correction, torch.zeros_like(correction))
        return correction, valid

    def residual_target(
        self,
        target_physical: torch.Tensor,
        rollout_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility alias for correction_target."""
        return self.correction_target(
            target_physical, rollout_normalized, valid_mask=valid_mask
        )

    def correction_target_from_normalized(
        self,
        cams_target_normalized: torch.Tensor,
        aurora_forecast_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build normalized CAMS-minus-Aurora from two normalized fields."""
        self._check_channels(cams_target_normalized)
        self._check_channels(aurora_forecast_normalized)
        if cams_target_normalized.shape != aurora_forecast_normalized.shape:
            raise ValueError(
                "Normalized CAMS target and Aurora forecast shapes must match "
                f"before subtraction; got {tuple(cams_target_normalized.shape)} "
                f"and {tuple(aurora_forecast_normalized.shape)}."
            )
        target32 = cams_target_normalized.float()
        rollout32 = aurora_forecast_normalized.float()
        valid = torch.isfinite(target32) & torch.isfinite(rollout32)
        if valid_mask is not None:
            try:
                supplied = torch.broadcast_to(
                    valid_mask.to(device=valid.device, dtype=torch.bool), valid.shape
                )
            except RuntimeError as exc:
                raise ValueError(
                    "valid_mask must be broadcastable to correction shape "
                    f"{tuple(valid.shape)}; got {tuple(valid_mask.shape)}."
                ) from exc
            valid = valid & supplied
        correction = target32 - rollout32
        correction = torch.where(valid, correction, torch.zeros_like(correction))
        return correction, valid

    def residual_target_from_normalized(
        self,
        target_normalized: torch.Tensor,
        rollout_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility alias for correction_target_from_normalized."""
        return self.correction_target_from_normalized(
            target_normalized, rollout_normalized, valid_mask=valid_mask
        )

    def apply_correction(
        self,
        aurora_forecast_normalized: torch.Tensor,
        predicted_correction_normalized: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        apply_constraints: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Aurora plus predicted correction in normalized/physical units."""
        self._check_channels(aurora_forecast_normalized)
        self._check_channels(predicted_correction_normalized)
        if aurora_forecast_normalized.shape != predicted_correction_normalized.shape:
            raise ValueError(
                "Predicted correction shape "
                f"{tuple(predicted_correction_normalized.shape)} does not match "
                f"Aurora forecast shape {tuple(aurora_forecast_normalized.shape)}."
            )
        refined_normalized = (
            aurora_forecast_normalized.float()
            + predicted_correction_normalized.float()
        )
        refined_physical = self.decode(refined_normalized)
        if apply_constraints:
            refined_physical = self.apply_physical_constraints(refined_physical)
        if target_mask is not None:
            try:
                mask = torch.broadcast_to(
                    target_mask.to(
                        device=refined_physical.device, dtype=torch.bool
                    ),
                    refined_physical.shape,
                )
            except RuntimeError as exc:
                raise ValueError(
                    "target_mask must be broadcastable to refined field shape "
                    f"{tuple(refined_physical.shape)}; got {tuple(target_mask.shape)}."
                ) from exc
            refined_physical = torch.where(
                mask,
                refined_physical,
                torch.full_like(refined_physical, float("nan")),
            )
        return refined_normalized, refined_physical

    def reconstruct(
        self,
        rollout_normalized: torch.Tensor,
        predicted_residual: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        apply_constraints: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility alias for apply_correction."""
        return self.apply_correction(
            rollout_normalized,
            predicted_residual,
            target_mask=target_mask,
            apply_constraints=apply_constraints,
        )

    # -- helpers ---------------------------------------------------------
    def _check_channels(self, tensor: torch.Tensor) -> None:
        if tensor.dim() != 4:
            raise ValueError(
                f"Target-space tensors must be [B, C, H, W], got {tuple(tensor.shape)}."
            )
        if tensor.shape[1] != self.num_channels:
            raise ValueError(
                f"Target-space tensor has {tensor.shape[1]} channels but the packing "
                f"declares {self.num_channels}."
            )
        if not tensor.is_floating_point():
            raise TypeError(
                "Target-space tensors must have a floating dtype, got "
                f"{tensor.dtype}."
            )
