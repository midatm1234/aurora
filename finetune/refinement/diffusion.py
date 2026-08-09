"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Diffusion-based Phase-2 residual refiners for Aurora rollouts.

Formulation
-----------
Let ``r`` be the residual between the CAMS/ground-truth target and the
deterministic Aurora rollout **for the same forecast valid time**, both in
Aurora's normalized target space::

    r = target_norm - rollout_norm

Training draws a noise-process index ``k ~ U{0, ..., K-1}`` and Gaussian noise
``eps ~ N(0, I)``, forms the forward-noised residual

    r_k = sqrt(abar_k) * r + sqrt(1 - abar_k) * eps

and regresses the configured parameterisation (``epsilon``, ``velocity`` or
``sample``) with a masked loss. ``k`` indexes the *noise* process only: it is
not a forecast lead time, not a valid time and not a rollout-step index. The
forecast lead time is a **separate** conditioning input with its own embedding.

Inference runs a DDIM reverse pass over ``inference_steps`` timesteps taken from
the same schedule used in training. ``eta = 0`` gives the deterministic
probability-flow path and ``eta = 1`` recovers the ancestral DDPM update, which
is what ``sampler: ddpm`` selects.

Adapted from ``granitewxc.refinement.diffusion`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0).
"""

from __future__ import annotations

import torch

from finetune.refinement.base import RefinerOutput, masked_loss, register_refiner
from finetune.refinement.config import RefinementConfig
from finetune.refinement.packed import PackedRefiner
from finetune.refinement.schedules import DiffusionSchedule

__all__ = ["DiffusionTransformerRefiner", "DiffusionUNetRefiner"]


class _BaseDiffusionRefiner(PackedRefiner):
    def __init__(
        self,
        config: RefinementConfig,
        *,
        residual_channels: int,
        cond_channels: int,
        metadata=None,
    ) -> None:
        d = config.diffusion
        self.prediction_type = d.prediction_type
        self.num_train_timesteps = d.training_timesteps
        self.num_inference_steps = d.inference_steps
        self.eta = d.eta
        self.clip_sample = d.clip_sample
        self.clip_sample_range = d.clip_sample_range
        super().__init__(
            config,
            residual_channels=residual_channels,
            cond_channels=cond_channels,
            metadata=metadata,
        )
        self.schedule = DiffusionSchedule(
            num_train_timesteps=d.training_timesteps,
            schedule=d.schedule,
            beta_start=d.beta_start,
            beta_end=d.beta_end,
            cosine_s=d.cosine_s,
        )

    # -- process-time scaling -------------------------------------------
    @staticmethod
    def _embed_time(timesteps: torch.Tensor) -> torch.Tensor:
        """Feed the raw integer noise index to the embedding (float cast only)."""
        return timesteps.float()

    # -- training --------------------------------------------------------
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
        batch = residual_target.shape[0]
        device = residual_target.device
        if generator is not None and generator.device != torch.device(device):
            timesteps = torch.randint(
                0,
                self.num_train_timesteps,
                (batch,),
                generator=generator,
                device=generator.device,
            ).to(device)
        else:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (batch,), generator=generator, device=device
            )
        noise = self._randn(tuple(residual_target.shape), device, residual_target.dtype, generator)

        noisy = self.schedule.add_noise(residual_target, noise, timesteps)
        target = self.schedule.training_target(
            self.prediction_type, residual_target, noise, timesteps
        )
        prediction = self.net(
            noisy, conditioning, self._embed_time(timesteps), self._lead_for(forecast_lead_time)
        )
        generative = masked_loss(prediction, target, mask, self.loss_kind)

        output = RefinerOutput(generative_loss=generative, process_time=timesteps)
        if not self.config.loss.has_auxiliary_terms:
            output.total_loss = generative
            return output
        # Auxiliary terms are always computed from the *clean residual*
        # reconstructed with the same conversion helper the sampler uses.
        clean = self.schedule.to_clean(self.prediction_type, prediction, noisy, timesteps)
        return self._finish(
            output,
            residual_estimate=clean,
            residual_target=residual_target,
            mask=mask,
            lead_index=lead_index,
            area_weight=area_weight,
            rollout_normalized=rollout_normalized,
        )

    # -- sampling --------------------------------------------------------
    @torch.no_grad()
    def sample_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(num_steps if num_steps is not None else self.num_inference_steps)
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        lead = self._lead_for(forecast_lead_time)

        x = self._randn(shape, device, dtype, generator)
        timesteps = self.schedule.inference_timesteps(steps, device)
        alphas_cumprod = self.schedule.alphas_cumprod.to(device=device, dtype=torch.float32)

        for idx in range(steps):
            t = timesteps[idx]
            t_batch = t.expand(shape[0])
            model_out = self.net(x, conditioning, self._embed_time(t_batch), lead)

            x0 = self.schedule.to_clean(self.prediction_type, model_out, x, t_batch)
            if self.clip_sample:
                x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
            eps = self.schedule.to_epsilon(self.prediction_type, model_out, x, t_batch)

            a_t = alphas_cumprod[t]
            prev_index = timesteps[idx + 1] if idx + 1 < steps else None
            a_prev = (
                alphas_cumprod[prev_index]
                if prev_index is not None
                else torch.ones((), device=device)
            )

            # DDIM update: eta=0 is the deterministic probability-flow path and
            # eta=1 recovers the ancestral DDPM update.
            sigma = self.eta * torch.sqrt(
                ((1 - a_prev) / (1 - a_t).clamp(min=1e-12)) * (1 - a_t / a_prev.clamp(min=1e-12))
            )
            sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
            dir_coeff = torch.sqrt((1 - a_prev - sigma.pow(2)).clamp(min=0.0))
            x = a_prev.sqrt().to(dtype) * x0 + dir_coeff.to(dtype) * eps
            if float(sigma) > 0 and prev_index is not None:
                x = x + sigma.to(dtype) * self._randn(shape, device, dtype, generator)
        return x

    @torch.no_grad()
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-step posterior mean from a zero state at the largest timestep.

        A cheap diagnostic, not a substitute for the configured multi-step
        sampler.
        """
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        x = torch.zeros(shape, device=device, dtype=dtype)
        t = torch.full((shape[0],), self.num_train_timesteps - 1, device=device, dtype=torch.long)
        model_out = self.net(
            x, conditioning, self._embed_time(t), self._lead_for(forecast_lead_time)
        )
        return self.schedule.to_clean(self.prediction_type, model_out, x, t)


@register_refiner("diffusion_unet")
class DiffusionUNetRefiner(_BaseDiffusionRefiner):
    """Convolutional UNet DDPM/DDIM residual refiner."""


@register_refiner("diffusion_transformer")
class DiffusionTransformerRefiner(_BaseDiffusionRefiner):
    """Spatial-token Transformer DDPM/DDIM residual refiner.

    Identical residual target, schedule, forward process, prediction
    parameterisation, loss, DDIM sampler and random-generator handling as
    :class:`DiffusionUNetRefiner`; only the denoiser architecture differs.
    """
