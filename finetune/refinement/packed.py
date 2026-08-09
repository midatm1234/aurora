"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared construction helpers for the packed (unified) Phase-2 refiners.

``diffusion_unet``, ``diffusion_transformer`` and ``flow_matching_transformer``
all consume the same packed representation

    residual state   [N, C_res,  H, W]   (normalized target space)
    conditioning     [N, C_cond, H, W]   (packed rollout / statics / masks)
    process time     [N]                 (diffusion timestep or flow time)
    forecast lead    [N]                 (physical hours, separate embedding)

where ``N`` is ``batch * rollout_lead_times`` because the caller folds the
lead-time axis into the effective batch dimension. Everything specific to a
generative process lives in :mod:`finetune.refinement.diffusion` or
:mod:`finetune.refinement.flow_matching`.
"""

from __future__ import annotations

import torch

from finetune.refinement.backbones import ConditionalResidualUNet, SpatialResidualTransformer
from finetune.refinement.base import RefinerOutput, ResidualRefiner
from finetune.refinement.config import RefinementConfig
from finetune.refinement.losses import compute_auxiliary_losses

__all__ = ["PackedRefiner"]


class PackedRefiner(ResidualRefiner):
    """Base class for the refiners that operate on packed multi-channel fields."""

    def __init__(
        self,
        config: RefinementConfig,
        *,
        residual_channels: int,
        cond_channels: int,
        metadata=None,
    ) -> None:
        super().__init__(
            config,
            residual_channels=residual_channels,
            cond_channels=cond_channels,
            metadata=metadata,
        )
        self.lead_time_conditioning = bool(config.conditioning.forecast_lead_time)
        self.lead_time_scale_hours = float(getattr(metadata, "lead_time_scale_hours", 72.0) or 72.0)
        self.lon_periodic = bool(getattr(metadata, "lon_periodic", False))
        self.net = self._build_net(config)

    # -- construction ----------------------------------------------------
    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:
        if config.uses_transformer:
            t = config.transformer
            return SpatialResidualTransformer(
                in_channels=self.residual_channels,
                cond_channels=self.cond_channels,
                out_channels=self.residual_channels,
                patch_size=t.patch_size,
                embedding_dim=t.embedding_dim,
                num_heads=t.num_heads,
                num_blocks=t.num_blocks,
                mlp_ratio=t.mlp_ratio,
                dropout=t.dropout,
                positional_encoding=t.positional_encoding,
                max_tokens_lat=t.max_tokens_lat,
                max_tokens_lon=t.max_tokens_lon,
                attention_mode=t.attention_mode,
                window_size=t.window_size,
                shifted_windows=t.shifted_windows,
                gradient_checkpointing=t.gradient_checkpointing,
                optimized_attention=t.optimized_attention,
                zero_init_output=t.zero_init_output,
                lead_time_conditioning=self.lead_time_conditioning,
                lead_time_scale_hours=self.lead_time_scale_hours,
                lon_periodic=self.lon_periodic,
            )
        u = config.unet
        return ConditionalResidualUNet(
            in_channels=self.residual_channels,
            cond_channels=self.cond_channels,
            out_channels=self.residual_channels,
            hidden_channels=u.hidden_channels,
            num_levels=u.num_levels,
            num_residual_blocks=u.num_residual_blocks,
            time_embedding_dim=u.time_embedding_dim,
            dropout=u.dropout,
            bottleneck_attention=u.bottleneck_attention,
            attention_heads=u.attention_heads,
            zero_init_output=u.zero_init_output,
            lead_time_conditioning=self.lead_time_conditioning,
            lead_time_scale_hours=self.lead_time_scale_hours,
            lon_periodic=self.lon_periodic,
        )

    def set_attention_implementation(self, implementation: str) -> None:
        setter = getattr(self.net, "set_attention_implementation", None)
        if setter is None:
            raise AttributeError(
                f"{type(self.net).__name__} has no configurable attention implementation."
            )
        setter(implementation)

    # -- shared helpers --------------------------------------------------
    def _lead_for(self, forecast_lead_time: torch.Tensor | None) -> torch.Tensor | None:
        return forecast_lead_time if self.lead_time_conditioning else None

    def _finish(
        self,
        output: RefinerOutput,
        *,
        residual_estimate: torch.Tensor,
        residual_target: torch.Tensor,
        mask: torch.Tensor | None,
        lead_index: torch.Tensor | None,
        area_weight: torch.Tensor | None,
        rollout_normalized: torch.Tensor | None,
    ) -> RefinerOutput:
        """Attach the enabled bias-aware auxiliary terms and the total loss."""
        loss_cfg = self.config.loss
        terms = compute_auxiliary_losses(
            residual_estimate,
            residual_target,
            config=loss_cfg,
            mask=mask,
            packing=self.metadata if hasattr(self.metadata, "channels") else None,
            area_weight=area_weight,
            lead_index=lead_index,
            rollout_normalized=rollout_normalized,
        )
        output.reconstruction_loss = terms.reconstruction
        output.bias_loss = terms.bias
        output.gradient_loss = terms.gradient
        output.pattern_correlation_loss = terms.pattern_correlation
        generative = output.generative_loss
        assert generative is not None
        total = generative
        if loss_cfg.has_auxiliary_terms:
            total = total + terms.weighted_total(loss_cfg, generative).to(generative.dtype)
        output.total_loss = total
        return output
