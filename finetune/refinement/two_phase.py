"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Unified two-phase Aurora model wrapper.

Phase 1 (deterministic)
    The existing Aurora forecast / autoregressive rollout. Architecture,
    checkpoint behaviour, input history, variable and level ordering, forecast
    timing, normalization and postprocessing are unchanged.

Phase 2 (optional, stochastic)
    One of ``flow_matching_unet``, ``flow_matching_transformer``,
    ``diffusion_unet`` or ``diffusion_transformer``, predicting a **residual**
    in Aurora's normalized target space for one forecast valid time. The four
    options are mutually exclusive alternatives; flow matching never requires
    the diffusion head to run first and the two are never stacked.

Rollout-semantics contract
--------------------------
Refinement is **postprocessing of a deterministic rollout step**::

    Aurora state at step n
      -> deterministic Aurora prediction for step n+1
      -> optional stochastic residual correction for step n+1

The refined field is returned alongside — never in place of — the deterministic
prediction, so the deterministic state used to generate later rollout steps is
untouched. Feeding the refined field back into the rollout is an explicit,
disabled-by-default, experimental option
(``refinement.feedback_to_rollout: true``) handled by a separate code path in
the training/inference driver.

This wrapper never shifts, re-indexes or re-orders the time axis, never changes
the rollout length or interval or the input history, and never uses a future
target value as conditioning.

Adapted from ``granitewxc.refinement.two_phase`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), restructured around
Aurora's per-rollout-step packed fields and forecast lead times.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from finetune.refinement.base import ChunkNoiseSource, ResidualRefiner, build_refiner
from finetune.refinement.config import (
    PerformanceConfig,
    RefinementConfig,
    resolve_performance_config,
    resolve_refinement_config,
)
from finetune.refinement.losses import area_weights_from_latitudes
from finetune.refinement.packing import FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace

__all__ = ["AuroraTwoPhaseRefiner", "TwoPhaseStepOutput", "build_two_phase_refiner"]


@dataclass
class TwoPhaseStepOutput:
    """Everything the training loop, the evaluator and the writer may need.

    All tensors refer to **one forecast valid time per leading batch entry**;
    when several rollout lead times were folded into the batch dimension the
    caller unfolds them with the same ``(batch, lead)`` order.
    """

    deterministic_normalized: torch.Tensor
    """Deterministic Aurora rollout in normalized target space, ``[N, C, H, W]``."""

    deterministic_physical: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    """Predicted residual in normalized target space (ensemble mean if N > 1)."""

    refined_normalized: torch.Tensor | None = None
    refined_physical: torch.Tensor | None = None
    members: torch.Tensor | None = None
    """Per-member refined physical fields, ``[N, M, C, H, W]`` in draw order."""

    member_residuals: torch.Tensor | None = None
    ensemble_mean: torch.Tensor | None = None
    ensemble_spread: torch.Tensor | None = None
    residual_target: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    process_time: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != {}}


def _align_to(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Crop or resample a conditioning field onto the target grid."""
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    if h >= size[0] and w >= size[1]:
        return tensor[..., : size[0], : size[1]]
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class AuroraTwoPhaseRefiner(nn.Module):
    """Deterministic Aurora plus an optional stochastic residual refiner.

    Module names are chosen so that Aurora state-dict keys are simply prefixed
    with ``aurora.`` and refinement keys with ``refiner.``; see
    :mod:`finetune.refinement.checkpoint` for the explicit migration helpers.

    Args:
        aurora: the deterministic Aurora model (Phase 1). May be ``None`` when
            the caller drives the rollout itself and only needs Phase 2.
        packing: the variable / pressure-level channel layout.
        refinement: resolved :class:`RefinementConfig`.
        performance: resolved :class:`PerformanceConfig`.
        nonnegative_variables: variables constrained to ``>= 0`` in physical
            units.
    """

    def __init__(
        self,
        aurora: nn.Module | None,
        packing: FieldPacking,
        refinement: RefinementConfig | None = None,
        performance: PerformanceConfig | None = None,
        *,
        nonnegative_variables: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.aurora = aurora if aurora is not None else nn.Identity()
        self.packing = packing
        self.refinement_config = refinement or RefinementConfig()
        self.performance_config = performance or PerformanceConfig()
        self.target_space = NormalizedTargetSpace(
            packing, nonnegative_variables=nonnegative_variables
        )
        self.refiner: ResidualRefiner | None = None
        self._area_weight_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._apply_aurora_freeze()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _apply_aurora_freeze(self) -> None:
        """Freeze Aurora when Phase 2 is trained on top of it."""
        self.aurora_frozen = bool(
            self.refinement_config.is_active and self.refinement_config.freeze_aurora
        )
        if self.aurora_frozen:
            for param in self.aurora.parameters():
                param.requires_grad_(False)
            self.aurora.eval()

    def train(self, mode: bool = True):  # noqa: D102 - torch API
        super().train(mode)
        if self.aurora_frozen:
            # A frozen Aurora must stay in eval() so dropout / normalization
            # layers cannot perturb the deterministic conditioning.
            self.aurora.eval()
        return self

    def conditioning_channels(
        self,
        *,
        static_fields: int = 0,
        input_state_channels: int = 0,
    ) -> int:
        """Number of packed spatial conditioning channels for this configuration."""
        cond = self.refinement_config.conditioning
        total = 0
        if cond.aurora_rollout:
            total += self.packing.num_channels
        if cond.aurora_input_state:
            total += int(input_state_channels or self.packing.num_channels)
        if cond.static_fields:
            total += int(static_fields)
        if cond.masks:
            total += 1
        return total

    def initialize_refiner(self, cond_channels: int) -> None:
        """Construct the Phase-2 network for a known conditioning width.

        Idempotent for a matching width; raises if called again with a different
        width after parameters were created, because that would silently discard
        trained weights.
        """
        if not self.refinement_config.is_active:
            return
        if self.refiner is not None:
            if self.refiner.cond_channels == int(cond_channels):
                return
            raise RuntimeError(
                f"Refiner was built for {self.refiner.cond_channels} conditioning "
                f"channels but the batch provides {cond_channels}. The dataset or the "
                "conditioning configuration changed after construction."
            )
        self.refiner = build_refiner(
            self.refinement_config,
            residual_channels=self.packing.num_channels,
            cond_channels=int(cond_channels),
            metadata=self.packing,
        )

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def build_conditioning(
        self,
        rollout_normalized: torch.Tensor,
        *,
        input_state_normalized: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        input_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate the configured spatial conditioning fields.

        Only fields available *at the forecast valid time without looking at the
        target* are used. The ground-truth mask is never conditioning: that
        would leak target information.
        """
        cond_cfg = self.refinement_config.conditioning
        size = (int(rollout_normalized.shape[-2]), int(rollout_normalized.shape[-1]))
        dtype = rollout_normalized.dtype
        parts: list[torch.Tensor] = []

        if cond_cfg.aurora_rollout:
            parts.append(rollout_normalized)

        if cond_cfg.aurora_input_state:
            if input_state_normalized is None:
                raise RuntimeError(
                    "refinement.conditioning.aurora_input_state is enabled but no "
                    "input state was supplied. It must be the Aurora *input* state of "
                    "the rollout step (never a future target)."
                )
            state = torch.nan_to_num(input_state_normalized.to(dtype), nan=0.0)
            parts.append(_align_to(state, size))

        if cond_cfg.static_fields:
            if static_fields is None:
                raise RuntimeError(
                    "refinement.conditioning.static_fields is enabled but no static "
                    "fields were supplied."
                )
            statics = torch.nan_to_num(static_fields.to(dtype), nan=0.0)
            parts.append(_align_to(statics, size))

        if cond_cfg.masks:
            if input_valid_mask is None:
                mask = torch.isfinite(rollout_normalized).all(dim=1, keepdim=True).to(dtype)
            else:
                mask = input_valid_mask.to(dtype)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                if mask.shape[1] != 1:
                    mask = mask.all(dim=1, keepdim=True).to(dtype)
            parts.append(_align_to(mask, size))

        conditioning = torch.cat(parts, dim=1)
        return torch.nan_to_num(conditioning, nan=0.0, posinf=0.0, neginf=0.0)

    def area_weight(self, height: int, device, dtype) -> torch.Tensor | None:
        key = (int(height), torch.device(device), dtype)
        if key not in self._area_weight_cache:
            weights = area_weights_from_latitudes(
                self.packing.lat, height, device=device, dtype=dtype
            )
            if weights is None:
                return None
            self._area_weight_cache[key] = weights
        return self._area_weight_cache[key]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(
        self,
        rollout_normalized: torch.Tensor,
        target_normalized: torch.Tensor,
        *,
        conditioning: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        input_state_normalized: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        forecast_lead_time: torch.Tensor | None = None,
        lead_index: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TwoPhaseStepOutput:
        """One Phase-2 training-objective evaluation for a set of rollout steps.

        ``rollout_normalized`` and ``target_normalized`` are packed
        ``[N, C, H, W]`` tensors that refer to the **same forecast valid time**
        per leading entry. ``forecast_lead_time`` carries the matching physical
        lead time in hours; ``lead_index`` the rollout-step index used only to
        group the bias-aware auxiliary losses.
        """
        if not self.refinement_config.is_active:
            raise RuntimeError(
                "training_step() requires an active refinement configuration; use the "
                "deterministic training path for Aurora-only runs."
            )
        rollout = rollout_normalized.detach()
        if conditioning is None:
            conditioning = self.build_conditioning(
                rollout,
                input_state_normalized=input_state_normalized,
                static_fields=static_fields,
            )
        self.initialize_refiner(conditioning.shape[1])
        assert self.refiner is not None

        if self.refinement_config.train_on_residual:
            residual_target, valid = self.target_space.residual_target_from_normalized(
                target_normalized, rollout, valid_mask=valid_mask
            )
        else:
            valid = torch.isfinite(target_normalized)
            if valid_mask is not None:
                valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)
            residual_target = torch.where(
                valid, target_normalized, torch.zeros_like(target_normalized)
            )

        result = self.refiner.compute_training_loss(
            residual_target.to(conditioning.dtype),
            conditioning,
            forecast_lead_time=forecast_lead_time,
            mask=valid,
            generator=generator,
            lead_index=lead_index,
            area_weight=self.area_weight(rollout.shape[-2], rollout.device, torch.float32),
            rollout_normalized=rollout,
        )
        losses = {
            name: value
            for name, value in (
                ("generative_loss", result.generative_loss),
                ("reconstruction_loss", result.reconstruction_loss),
                ("bias_loss", result.bias_loss),
                ("gradient_loss", result.gradient_loss),
                ("pattern_correlation_loss", result.pattern_correlation_loss),
                ("total_loss", result.total_loss),
            )
            if value is not None
        }
        return TwoPhaseStepOutput(
            deterministic_normalized=rollout,
            residual_target=residual_target,
            valid_mask=valid,
            process_time=result.process_time,
            losses=losses,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refine(
        self,
        rollout_normalized: torch.Tensor,
        *,
        conditioning: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        input_state_normalized: torch.Tensor | None = None,
        forecast_lead_time: torch.Tensor | None = None,
        ensemble_size: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        target_mask: torch.Tensor | None = None,
        return_members: bool = True,
        chunk_size: int | None = None,
        num_steps: int | None = None,
    ) -> TwoPhaseStepOutput:
        """Deterministic plus (optionally) refined ensemble inference.

        The deterministic rollout is always returned unchanged; refinement only
        adds fields.
        """
        rollout = rollout_normalized
        out = TwoPhaseStepOutput(
            deterministic_normalized=rollout,
            deterministic_physical=self.target_space.decode(rollout),
        )
        if not self.refinement_config.is_active:
            return out
        if conditioning is None:
            conditioning = self.build_conditioning(
                rollout,
                input_state_normalized=input_state_normalized,
                static_fields=static_fields,
            )
        self.initialize_refiner(conditioning.shape[1])
        assert self.refiner is not None

        n = int(
            ensemble_size if ensemble_size is not None else self.refinement_config.ensemble_size
        )
        if n < 1:
            return out

        if generator is None:
            seed_value = seed if seed is not None else self.refinement_config.seed
            if seed_value is not None:
                generator = torch.Generator(device=conditioning.device)
                generator.manual_seed(int(seed_value))

        perf = self.performance_config.ensemble
        if chunk_size is None:
            chunk_size = (
                perf.chunk_size if perf.chunk_size is not None else (n if perf.batch_members else 1)
            )
        chunk_size = max(1, min(int(chunk_size), n))

        # Each member gets its own generator seeded from the caller's generator,
        # so member m sees the same noise sequence regardless of chunking. This
        # makes serial and batched ensemble generation bitwise identical.
        gen_device = generator.device if generator is not None else conditioning.device
        if generator is not None:
            member_seeds = torch.randint(
                0, 2**62, (n,), generator=generator, device=gen_device, dtype=torch.int64
            ).tolist()
        else:
            member_seeds = torch.randint(0, 2**62, (n,), dtype=torch.int64).tolist()
        member_generators = []
        for member_seed in member_seeds:
            g = torch.Generator(device=gen_device)
            g.manual_seed(int(member_seed))
            member_generators.append(g)

        batch = conditioning.shape[0]
        residuals: list[torch.Tensor] = []
        drawn = 0
        while drawn < n:
            members = min(chunk_size, n - drawn)
            source = ChunkNoiseSource(member_generators[drawn : drawn + members], batch)
            if members == 1:
                with self.refiner.use_noise_source(source):
                    res = self._sample(conditioning, rollout, forecast_lead_time, num_steps)
                residuals.append(res.unsqueeze(1))
            else:
                cond_rep = conditioning.repeat_interleave(members, dim=0)
                rollout_rep = rollout.repeat_interleave(members, dim=0)
                lead_rep = (
                    forecast_lead_time.repeat_interleave(members, dim=0)
                    if forecast_lead_time is not None
                    else None
                )
                with self.refiner.use_noise_source(source):
                    res = self._sample(cond_rep, rollout_rep, lead_rep, num_steps)
                res = res.reshape(batch, members, *res.shape[1:])
                residuals.append(res)
            drawn += members

        member_residuals = torch.cat(residuals, dim=1)  # [N, M, C, H, W]
        member_fields = []
        for idx in range(n):
            _, refined_physical = self.target_space.reconstruct(
                rollout,
                member_residuals[:, idx].to(rollout.dtype),
                target_mask=target_mask,
            )
            member_fields.append(refined_physical.unsqueeze(1))
        members_physical = torch.cat(member_fields, dim=1)

        # Ensemble statistics are accumulated in float32 regardless of the
        # compute dtype so mixed precision cannot bias the mean or the spread.
        # Masked cells (NaN) are excluded rather than poisoning every member.
        stack32 = members_physical.float()
        valid = torch.isfinite(stack32)
        counts = valid.sum(dim=1)
        filled = torch.where(valid, stack32, torch.zeros_like(stack32))
        mean32 = filled.sum(dim=1) / counts.clamp(min=1)
        mean32 = torch.where(counts > 0, mean32, torch.full_like(mean32, float("nan")))
        ensemble_mean = mean32.to(members_physical.dtype)
        spread = None
        if n > 1:
            deviations = torch.where(
                valid, stack32 - mean32.unsqueeze(1), torch.zeros_like(stack32)
            )
            var = deviations.pow(2).sum(dim=1) / (counts - 1).clamp(min=1)
            var = torch.where(counts > 1, var, torch.full_like(var, float("nan")))
            spread = var.sqrt().to(members_physical.dtype)

        mean_residual = member_residuals.float().mean(dim=1).to(rollout.dtype)
        refined_norm, refined_physical = self.target_space.reconstruct(
            rollout, mean_residual, target_mask=target_mask
        )

        out.members = members_physical if return_members else None
        out.member_residuals = member_residuals if return_members else None
        out.ensemble_mean = ensemble_mean
        out.ensemble_spread = spread
        out.residual = mean_residual
        out.refined_normalized = refined_norm
        out.refined_physical = ensemble_mean if n > 1 else members_physical[:, 0]
        return out

    def _sample(
        self,
        conditioning: torch.Tensor,
        rollout: torch.Tensor,
        forecast_lead_time: torch.Tensor | None,
        num_steps: int | None,
    ) -> torch.Tensor:
        assert self.refiner is not None
        kwargs: dict[str, Any] = {
            "forecast_lead_time": forecast_lead_time,
            "num_steps": num_steps,
        }
        if self.refinement_config.is_legacy_flow_matching:
            kwargs["rollout_normalized"] = rollout
        return self.refiner.sample_residual(conditioning, **kwargs)

    # ------------------------------------------------------------------
    # Aurora (Phase-1) delegation
    # ------------------------------------------------------------------
    @property
    def patch_size(self) -> int:
        return int(self.aurora.patch_size)

    @property
    def surf_stats(self) -> dict:
        return dict(self.aurora.surf_stats)

    def batch_transform_hook(self, batch):
        return self.aurora.batch_transform_hook(batch)

    def configure_activation_checkpointing(self) -> None:
        hook = getattr(self.aurora, "configure_activation_checkpointing", None)
        if callable(hook):
            hook()

    def load_checkpoint(self, *args, **kwargs) -> None:
        self.aurora.load_checkpoint(*args, **kwargs)

    def load_checkpoint_local(self, *args, **kwargs) -> None:
        hook = getattr(self.aurora, "load_checkpoint_local", None)
        if callable(hook):
            hook(*args, **kwargs)

    def freeze_base(self) -> None:
        for param in self.aurora.parameters():
            param.requires_grad_(False)
        self.aurora.eval()
        self.aurora_frozen = True

    def refine_parameter_count(self) -> int:
        return 0 if self.refiner is None else sum(p.numel() for p in self.refiner.parameters())

    def forward(self, batch, forecast_lead_time_hours: float | torch.Tensor | None = None):
        """Deterministic Aurora prediction, refined at inference time only.

        In training mode the unrefined Aurora prediction is returned: Phase 2 is
        exercised through :meth:`training_step` from the supervised-loss code
        path, which keeps the (frozen) Aurora graph decoupled from the refiner's
        gradient. At inference the deterministic prediction is refined as
        postprocessing; the caller decides whether the deterministic or the
        refined state continues the rollout.
        """
        pred = self.aurora(batch)
        if self.training or not self.refinement_config.is_active:
            return pred
        from finetune.refinement.integration import refine_batch_prediction

        return refine_batch_prediction(
            self,
            pred,
            forecast_lead_time_hours=forecast_lead_time_hours,
            ensemble_size=1,
            seed=self.refinement_config.seed,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.refiner is None or not self.refinement_config.freeze_aurora:
            return [p for p in self.parameters() if p.requires_grad]
        return [p for p in self.refiner.parameters() if p.requires_grad]

    def describe(self) -> dict[str, Any]:
        return {
            "refinement": self.refinement_config.to_dict(),
            "performance": self.performance_config.to_dict(),
            "aurora_frozen": self.aurora_frozen,
            "aurora_parameters": sum(p.numel() for p in self.aurora.parameters()),
            "refiner_parameters": (
                sum(p.numel() for p in self.refiner.parameters()) if self.refiner is not None else 0
            ),
            "field_packing": self.packing.to_dict(),
        }


def build_two_phase_refiner(
    aurora: nn.Module | None,
    packing: FieldPacking,
    config: Mapping[str, Any] | None,
    *,
    nonnegative_variables: Sequence[str] = (),
) -> AuroraTwoPhaseRefiner:
    """Build the wrapper from a raw experiment configuration."""
    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    if refinement.feedback_to_rollout:
        warnings.warn(
            "refinement.feedback_to_rollout is enabled: this experimental path changes "
            "the deterministic Aurora rollout trajectory and must be evaluated "
            "independently.",
            RuntimeWarning,
            stacklevel=2,
        )
    return AuroraTwoPhaseRefiner(
        aurora,
        packing,
        refinement=refinement,
        performance=performance,
        nonnegative_variables=nonnegative_variables,
    )
