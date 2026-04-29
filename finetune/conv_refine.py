"""Convolutional refinement heads for Aurora fine-tuning.

The patch-based ViT decoder (``unpatchify``) produces outputs that are
piecewise-constant within each P×P patch, creating visible blocky artifacts.
This module adds a U-Net-style residual CNN on top that operates at full
pixel resolution, smoothing patch boundaries while preserving learned structure.

Inspired by the Prithvi WxC / Granite-WxC CORDEX fine-tuning approach
(``ClimateDownscaleFinetuneUNETModel``) which uses:
  - Conv layers with ``padding_mode='replicate'`` (no edge artifacts)
  - PReLU / LeakyReLU activations
  - Skip connections from static covariates

Architecture per variable (residual refinement, no spatial rescaling):

    input (H, W) ──► Conv7 ──► PReLU ──► Conv5 ──► PReLU ──► Conv3 ──► PReLU ──► Conv1 ──► out
         │                                                                              │
         └──────────────────────────── residual skip ──────────────────────────────────┘

* ``padding_mode='replicate'`` on all convolutions to avoid edge artifacts.
* Kernel sizes 7→5→3→1 ensure the first layer spans 2+ patch boundaries
  (for patch_size=3).
* The final conv is zero-initialized so the wrapper is identity at init
  (the refinement starts as zero → output = input).  Training learns to smooth.
* For atmospheric variables, levels are treated as extra spatial slices
  through the same shared head.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn

from aurora.batch import Batch


class ConvRefineBlock(nn.Module):
    """Residual convolutional refinement for a single 2-D field.

    Follows the Prithvi WxC conv patterns: replicate padding, PReLU activations.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=7, padding=3, padding_mode="replicate"),
            nn.PReLU(num_parameters=hidden),
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.PReLU(num_parameters=hidden),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.PReLU(num_parameters=hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # Zero-init the last layer so the residual starts as identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a 2-D field.

        Args:
            x: (*, H, W) — any number of leading dimensions.

        Returns:
            Tensor with same shape, refined.
        """
        leading = x.shape[:-2]
        h, w = x.shape[-2], x.shape[-1]
        # Flatten leading dims → (N, 1, H, W) for Conv2d
        flat = x.reshape(-1, 1, h, w)
        refined = flat + self.net(flat)
        return refined.reshape(*leading, h, w)


class AuroraConvRefine(nn.Module):
    """Wrapper that adds convolutional refinement heads to an Aurora model.

    Only the specified *target* variables get refinement heads; all other
    variables pass through unchanged.  The base Aurora model can be frozen
    while only the refinement heads are trained, or both can be trained
    jointly.

    Usage::

        base_model = AuroraAirPollution(...)
        base_model.load_checkpoint(strict=False)

        wrapper = AuroraConvRefine(
            base_model,
            target_surf_vars=("tcno2",),
            target_atmos_vars=("no2",),
        )
        # Freeze base, train only conv heads:
        wrapper.freeze_base()
    """

    def __init__(
        self,
        base: nn.Module,
        target_surf_vars: tuple[str, ...] = (),
        target_atmos_vars: tuple[str, ...] = (),
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.base = base
        self.target_surf_vars = set(target_surf_vars)
        self.target_atmos_vars = set(target_atmos_vars)

        self.surf_refine = nn.ModuleDict(
            {name: ConvRefineBlock(hidden=hidden) for name in target_surf_vars}
        )
        self.atmos_refine = nn.ModuleDict(
            {name: ConvRefineBlock(hidden=hidden) for name in target_atmos_vars}
        )

    # --- Delegate common Aurora attributes to the base model ---

    @property
    def patch_size(self) -> int:
        return self.base.patch_size

    @property
    def surf_stats(self) -> dict:
        return self.base.surf_stats

    def batch_transform_hook(self, batch: Batch) -> Batch:
        return self.base.batch_transform_hook(batch)

    def configure_activation_checkpointing(self) -> None:
        if hasattr(self.base, "configure_activation_checkpointing"):
            self.base.configure_activation_checkpointing()

    def load_checkpoint(self, *args, **kwargs) -> None:
        self.base.load_checkpoint(*args, **kwargs)

    def load_checkpoint_local(self, *args, **kwargs) -> None:
        if hasattr(self.base, "load_checkpoint_local"):
            self.base.load_checkpoint_local(*args, **kwargs)

    # --- Forward ---

    def forward(self, batch: Batch) -> Batch:
        pred = self.base(batch)

        # Refine target surface variables.
        new_surf: dict[str, torch.Tensor] = {}
        for name, tensor in pred.surf_vars.items():
            if name in self.surf_refine:
                new_surf[name] = self.surf_refine[name](tensor)
            else:
                new_surf[name] = tensor

        # Refine target atmospheric variables.
        new_atmos: dict[str, torch.Tensor] = {}
        for name, tensor in pred.atmos_vars.items():
            if name in self.atmos_refine:
                new_atmos[name] = self.atmos_refine[name](tensor)
            else:
                new_atmos[name] = tensor

        return dataclasses.replace(pred, surf_vars=new_surf, atmos_vars=new_atmos)

    # --- Convenience ---

    def freeze_base(self) -> None:
        """Freeze all base model parameters; only conv heads remain trainable."""
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self) -> None:
        """Unfreeze the base decoder (useful for joint training)."""
        if hasattr(self.base, "decoder"):
            for param in self.base.decoder.parameters():
                param.requires_grad = True

    def refine_parameter_count(self) -> int:
        """Count trainable parameters in refinement heads only."""
        return sum(
            p.numel()
            for heads in (self.surf_refine, self.atmos_refine)
            for head in heads.values()
            for p in head.parameters()
        )
