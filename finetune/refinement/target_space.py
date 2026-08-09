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
        mean = self.packing.means(device=reference.device, dtype=reference.dtype)
        std = self.packing.stds(device=reference.device, dtype=reference.dtype)
        return mean, std.clamp(min=_MIN_SCALE)

    # -- transforms ------------------------------------------------------
    def encode(self, physical: torch.Tensor) -> torch.Tensor:
        """Map a packed physical field ``[B, C, H, W]`` into normalized space."""
        self._check_channels(physical)
        mean, std = self._scalers(physical)
        return (physical - mean) / std

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        """Exact inverse of :meth:`encode` (applied exactly once per field)."""
        self._check_channels(normalized)
        mean, std = self._scalers(normalized)
        return normalized * std + mean

    # -- constraints -----------------------------------------------------
    def apply_physical_constraints(self, physical: torch.Tensor) -> torch.Tensor:
        """Clamp configured non-negative variables to ``>= 0``.

        Idempotent, and applied only to physical-unit tensors.
        """
        if not bool(self._nonneg_mask.any()):
            return physical
        enabled = self._nonneg_mask.to(physical.device)
        return torch.where(enabled, physical.clamp(min=0.0), physical)

    # -- residual construction -------------------------------------------
    def residual_target(
        self,
        target_physical: torch.Tensor,
        rollout_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(residual_target, valid_mask)`` in normalized space.

        Invalid cells (non-finite targets, or cells excluded by ``valid_mask``)
        get exactly zero residual so they contribute nothing to a masked loss and
        never leak NaNs into the network.
        """
        valid = torch.isfinite(target_physical) & torch.isfinite(rollout_normalized)
        if valid_mask is not None:
            valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)
        filled = torch.where(valid, target_physical, torch.zeros_like(target_physical))
        target_norm = self.encode(filled)
        residual = target_norm - rollout_normalized
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return residual, valid

    def residual_target_from_normalized(
        self,
        target_normalized: torch.Tensor,
        rollout_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual target when the ground truth is *already* normalized.

        This is the path used by the Aurora training loop, whose supervised loss
        normalizes both the prediction and the target with the same statistics
        before any refinement happens. Using it avoids a redundant
        decode/encode round trip (and therefore any risk of double
        normalization).
        """
        valid = torch.isfinite(target_normalized) & torch.isfinite(rollout_normalized)
        if valid_mask is not None:
            valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)
        residual = target_normalized - rollout_normalized
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return residual, valid

    def reconstruct(
        self,
        rollout_normalized: torch.Tensor,
        predicted_residual: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        apply_constraints: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a normalized residual; return ``(refined_norm, refined_physical)``.

        The residual is added in normalized space, the field is inverse
        normalized exactly once, and constraints/masks are then applied exactly
        once.
        """
        if rollout_normalized.shape != predicted_residual.shape:
            raise ValueError(
                f"Residual shape {tuple(predicted_residual.shape)} does not match the "
                f"deterministic normalized rollout {tuple(rollout_normalized.shape)}."
            )
        refined_norm = rollout_normalized + predicted_residual
        refined_physical = self.decode(refined_norm)
        if apply_constraints:
            refined_physical = self.apply_physical_constraints(refined_physical)
        if target_mask is not None:
            mask = target_mask.to(device=refined_physical.device, dtype=torch.bool)
            refined_physical = torch.where(
                mask, refined_physical, torch.full_like(refined_physical, float("nan"))
            )
        return refined_norm, refined_physical

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
