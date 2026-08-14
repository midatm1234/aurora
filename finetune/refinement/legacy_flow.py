"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Registry adapter for the **existing** Aurora flow-matching UNet refiner.

``refinement.type: flow_matching_unet`` (alias ``flow_matching``) *is* the
implementation that already lives on the ``aurora_finetune_flow_matching``
branch: :class:`finetune.flow_refine.AuroraFlowRefine` with its per-variable
:class:`finetune.flow_refine.ResidualFlowUNet` heads.

This module does **not** re-implement that formulation. It holds a real
``AuroraFlowRefine`` instance (constructed with a placeholder Phase-1 module,
because only the refinement heads are used here) and delegates every numerical
operation to it, so:

* the source distribution, interpolation path, flow-time sampling, flow-time
  embedding, residual z-scoring, masked loss, initial state, integration
  direction/interval, solver, step count, conditioning and lead-time
  conditioning are byte-for-byte the existing ones;
* existing flow-matching checkpoints load into ``refiner.legacy.*`` through the
  explicit key migration in :mod:`finetune.refinement.checkpoint`;
* the production training/inference path
  (:func:`finetune.aurora_finetune_utils.maybe_wrap_flow_refine`) is untouched,
  so legacy runs keep their exact numerical behaviour.

Differences from the Prithvi reference flow matching are intentional and
documented in ``docs/STOCHASTIC_REFINEMENT.md``: Aurora regresses the clean
residual (``x1``/data parameterisation) while Prithvi regresses the velocity
``u_t = r - x0``. The Aurora formulation is preserved for legacy configurations
and checkpoints; the velocity formulation is available only by explicitly
selecting ``flow_matching.interpolation_path: rectified_flow`` on the new
``flow_matching_transformer`` refiner.
"""

from __future__ import annotations

import torch
from torch import nn

from finetune.refinement.base import RefinerOutput, ResidualRefiner, register_refiner
from finetune.refinement.config import ConfigValidationError, RefinementConfig
from finetune.refinement.packing import FieldPacking

__all__ = ["LegacyFlowMatchingUNetRefiner"]


@register_refiner("flow_matching_unet")
class LegacyFlowMatchingUNetRefiner(ResidualRefiner):
    """Common-interface adapter around :class:`AuroraFlowRefine`.

    Args:
        config: resolved refinement configuration.
        residual_channels: number of packed target channels.
        cond_channels: number of packed conditioning channels. The legacy heads
            only consume the deterministic rollout, so anything beyond it is
            ignored (and a mismatch is reported rather than silently dropped).
        metadata: the :class:`~finetune.refinement.packing.FieldPacking` that
            defines the variable / pressure-level layout.
    """

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
        if not isinstance(metadata, FieldPacking):
            raise ConfigValidationError(
                "refinement.type='flow_matching_unet' needs the FieldPacking metadata "
                "(variable and pressure-level layout) to build its per-variable heads."
            )
        from finetune.flow_refine import AuroraFlowRefine

        packing: FieldPacking = metadata
        surf_vars = tuple(spec.aurora_name for spec in packing.channels if spec.kind == "surf")
        atmos_vars = tuple(
            dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "atmos")
        )
        atmos_loss_levels = {
            name: list(range(len(packing.levels_for(name)))) for name in atmos_vars
        }
        flow = config.flow_matching
        self.packing = packing
        # ``nn.Identity`` stands in for Phase 1: only the refinement heads of
        # ``AuroraFlowRefine`` are used through this adapter.
        self.legacy = AuroraFlowRefine(
            base=nn.Identity(),
            target_surf_vars=surf_vars,
            target_atmos_vars=atmos_vars,
            hidden=config.unet.hidden_channels,
            time_dim=config.unet.time_embedding_dim,
            sampling_steps=flow.integration_steps,
            sigma_min=flow.sigma_min,
            # This adapter was introduced with the corrected source-endpoint
            # contract; pin it explicitly so the legacy wrapper's missing-key
            # v1 fallback does not change unified-refiner semantics.
            flow_refine_contract_version=2,
            atmos_loss_levels=atmos_loss_levels or None,
            lead_time_cond=config.conditioning.forecast_lead_time,
            lead_time_scale_hours=packing.lead_time_scale_hours,
            residual_zscore=flow.residual_zscore,
            res_std_momentum=flow.res_std_momentum,
            lon_periodic=packing.lon_periodic,
        )

    # -- helpers ---------------------------------------------------------
    def _split(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.packing.unpack(packed)

    @staticmethod
    def _lead_for(forecast_lead_time: torch.Tensor | None, enabled: bool) -> torch.Tensor | None:
        return forecast_lead_time if enabled else None

    def _require_rollout(self, rollout_normalized: torch.Tensor | None) -> torch.Tensor:
        if rollout_normalized is None:
            raise ValueError(
                "The legacy flow-matching refiner conditions on the deterministic "
                "Aurora rollout; pass rollout_normalized explicitly."
            )
        return rollout_normalized

    # -- common interface -------------------------------------------------
    def compute_training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        lead_index: torch.Tensor | None = None,
        area_weight: torch.Tensor | None = None,
        rollout_normalized: torch.Tensor | None = None,
    ) -> RefinerOutput:
        """Sum of the existing per-variable ``AuroraFlowRefine.flow_loss`` terms."""
        rollout = self._require_rollout(rollout_normalized)
        rollout_fields = self._split(rollout)
        target_fields = self._split(rollout + residual_target)
        mask_fields = self._split(mask.to(rollout.dtype)) if mask is not None else None
        lead = self._lead_for(forecast_lead_time, self.legacy.lead_time_cond)

        total = torch.zeros((), device=rollout.device, dtype=torch.float32)
        count = 0
        for name, pred in rollout_fields.items():
            kind = "surf" if pred.dim() == 3 else "atmos"
            valid = None
            if mask_fields is not None:
                valid = mask_fields[name].bool()
            loss = self.legacy.flow_loss(
                pred_norm=pred,
                target_norm=target_fields[name],
                var_name=name,
                kind=kind,
                valid_mask=valid,
                lead_time_hours=lead,
                generator=generator,
            )
            total = total + loss.float()
            count += 1
        generative = total / max(count, 1)
        return RefinerOutput(generative_loss=generative, total_loss=generative)

    @torch.no_grad()
    def sample_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        rollout_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample the normalized residual with the existing sampler."""
        rollout = self._require_rollout(rollout_normalized)
        return self._per_variable_residual(
            rollout,
            forecast_lead_time,
            deterministic=False,
            num_steps=num_steps,
            generator=generator,
        )

    @torch.no_grad()
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        rollout_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rollout = self._require_rollout(rollout_normalized)
        return self._per_variable_residual(
            rollout,
            forecast_lead_time,
            deterministic=True,
            num_steps=None,
            generator=None,
        )

    def _per_variable_residual(
        self,
        rollout_normalized: torch.Tensor,
        forecast_lead_time: torch.Tensor | None,
        *,
        deterministic: bool,
        num_steps: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        lead = self._lead_for(forecast_lead_time, self.legacy.lead_time_cond)
        fields = self._split(rollout_normalized)
        saved_steps = self.legacy.sampling_steps
        if num_steps is not None:
            self.legacy.sampling_steps = int(num_steps)
        try:
            residuals: dict[str, torch.Tensor] = {}
            for name, pred in fields.items():
                kind = "surf" if pred.dim() == 3 else "atmos"
                if deterministic:
                    residuals[name] = (
                        self.legacy.refine_norm_deterministic(
                            pred, name, kind, lead_time_hours=lead
                        )
                        - pred
                    )
                else:
                    residuals[name] = self._stochastic_residual(
                        pred, name, kind, lead, generator
                    )
        finally:
            self.legacy.sampling_steps = saved_steps
        return self.packing.pack(residuals)

    def _stochastic_residual(
        self,
        pred_norm: torch.Tensor,
        var_name: str,
        kind: str,
        lead: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Un-standardised residual sample, exactly as ``refine_prediction`` does."""
        heads = self.legacy.surf_flow if kind == "surf" else self.legacy.atmos_flow
        if var_name not in heads:
            return torch.zeros_like(pred_norm)
        head = heads[var_name]
        if kind == "atmos":
            b, levels, h, w = pred_norm.shape
            cond = pred_norm.reshape(b * levels, 1, h, w)
        else:
            b, h, w = pred_norm.shape
            levels = 1
            cond = pred_norm.reshape(b, 1, h, w)
        lead_n = None
        if lead is not None:
            lead_b = self.legacy._lead_hours_for_batch(lead, b, cond.device)
            lead_n = (
                lead_b.repeat_interleave(levels)
                if lead_b is not None and kind == "atmos"
                else lead_b
            )
        residual = self.legacy._sample_residual(
            cond, head, lead_hours=lead_n, generator=generator,
        )
        sigma = self.legacy._residual_std(kind, var_name, pred_norm, update=False)
        if kind == "atmos":
            if sigma.ndim == 1:
                sigma = sigma.view(1, levels, 1, 1)
            residual = residual.reshape(b, levels, h, w)
        else:
            residual = residual.reshape(b, h, w)
            sigma = sigma.reshape(-1)[0]
        return residual * sigma

    # -- pass-through -----------------------------------------------------
    def set_norm_stats(self, stats) -> None:
        self.legacy.set_norm_stats(stats)

    def set_aux_loss_config(self, cfg) -> None:
        self.legacy.set_aux_loss_config(cfg)
