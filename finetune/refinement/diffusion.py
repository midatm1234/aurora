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
        self.sampler = d.sampler
        self.eta = d.eta
        self.clip_sample = d.clip_sample
        self.clip_sample_range = d.clip_sample_range
        self.snr_weighting = d.resolved_snr_weighting()
        self.snr_gamma = d.snr_gamma
        self.timestep_bias = d.timestep_bias
        self.timestep_distribution = d.resolved_timestep_distribution()
        self.deterministic_estimator = d.resolved_deterministic_estimator()
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

    def _draw_timesteps(
        self, batch: int, device: torch.device, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Sample training timesteps, optionally biased along the noise axis.

        With ``sample`` (x0) prediction the low-noise timesteps are close to the
        trivial identity ``x_0 ~= x_t`` and teach the network a shortcut, while
        the high-noise end - which is where the deterministic ``posterior_mean``
        query lives - is what forces it to use the conditioning. A negative
        ``timestep_bias`` concentrates training there without ever excluding the
        rest of the schedule.
        """
        gen_device = generator.device if generator is not None else device
        if self.timestep_distribution == "uniform" or self.timestep_bias == 0.0:
            steps = torch.randint(
                0, self.num_train_timesteps, (batch,), generator=generator, device=gen_device
            )
        else:
            u = torch.rand(batch, generator=generator, device=gen_device, dtype=torch.float32)
            shaped = u.pow(1.0 + abs(self.timestep_bias))
            if self.timestep_distribution == "high_noise":
                shaped = 1.0 - shaped
            steps = (shaped * self.num_train_timesteps).long()
            steps = steps.clamp(0, self.num_train_timesteps - 1)
        return steps.to(device)

    def _loss_weight(self, timesteps: torch.Tensor) -> torch.Tensor | None:
        """Per-sample weighting of the generative loss across noise levels.

        With ``prediction_type='epsilon'`` the plain DDPM objective implicitly
        weights every timestep equally in *epsilon* space, which corresponds to
        a ``1/SNR`` weighting in residual space: exactly the wrong emphasis for
        a corrector that is judged on the residual it reconstructs. Min-SNR-gamma
        (Hang et al., 2023) caps that imbalance.
        """
        if self.snr_weighting == "none":
            return None
        alphas = self.schedule.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
        abar = alphas[timesteps].clamp(min=1e-12, max=1.0 - 1e-12)
        snr = abar / (1.0 - abar)
        if self.snr_weighting == "snr":
            weight = snr
        elif self.snr_weighting == "truncated_snr":
            weight = snr.clamp(min=1.0)
        else:  # min_snr
            weight = torch.minimum(snr, torch.full_like(snr, max(self.snr_gamma, 1e-6)))
        if self.prediction_type == "epsilon":
            weight = weight / snr
        elif self.prediction_type == "velocity":
            weight = weight / (snr + 1.0)
        # 'sample' already predicts the clean residual: the weight applies as is.
        return weight.reshape(-1, *([1] * 3))
    def _network_prediction(
        self,
        state_float32: torch.Tensor,
        conditioning: torch.Tensor,
        process_time: torch.Tensor,
        lead: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate the denoiser at model precision, then return float32."""
        model_state = state_float32.to(dtype=conditioning.dtype)
        prediction = self.net(model_state, conditioning, process_time, lead)
        if prediction.shape != state_float32.shape:
            raise RuntimeError(
                "Diffusion denoiser output shape must match its state; got "
                f"{tuple(prediction.shape)} and {tuple(state_float32.shape)}."
            )
        prediction32 = prediction.float()
        if not bool(torch.isfinite(prediction32).all()):
            raise FloatingPointError(
                "Diffusion denoiser produced non-finite prediction values."
            )
        return prediction32


    # -- training --------------------------------------------------------
    def _training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        mask: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[RefinerOutput, torch.Tensor]:
        batch = residual_target.shape[0]
        device = residual_target.device
        correction32 = residual_target.float()
        timesteps = self._draw_timesteps(batch, device, generator)
        noise32 = self._randn(
            tuple(correction32.shape), device, torch.float32, generator
        )

        noisy32 = self.schedule.add_noise(correction32, noise32, timesteps)
        target32 = self.schedule.training_target(
            self.prediction_type, correction32, noise32, timesteps
        )
        prediction32 = self._network_prediction(
            noisy32,
            conditioning,
            self._embed_time(timesteps),
            self._lead_for(forecast_lead_time),
        )

        sample_weight = self._loss_weight(timesteps)
        generative = masked_loss(
            prediction32, target32, mask, self.loss_kind, sample_weight=sample_weight
        )

        output = RefinerOutput(generative_loss=generative, process_time=timesteps)
        clean32 = self.schedule.to_clean(
            self.prediction_type, prediction32, noisy32, timesteps
        )
        return output, clean32

    # -- sampling --------------------------------------------------------
    def _ddim_loop(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        lead: torch.Tensor | None,
        steps: int,
        *,
        eta: float,
        generator: torch.Generator | None,
        stochastic: bool,
    ) -> torch.Tensor:
        device = conditioning.device
        x = x.float()
        shape = tuple(x.shape)
        timesteps = self.schedule.inference_timesteps(steps, device)
        alphas_cumprod = self.schedule.alphas_cumprod.to(
            device=device, dtype=torch.float32
        )

        for idx in range(steps):
            t = timesteps[idx]
            t_batch = t.expand(shape[0])
            model_out32 = self._network_prediction(
                x, conditioning, self._embed_time(t_batch), lead
            )

            x0 = self.schedule.to_clean(
                self.prediction_type, model_out32, x, t_batch
            )
            if self.clip_sample:
                x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
                # Once x0 changes, epsilon must be recomputed from that same x0;
                # mixing clipped x0 with the original epsilon is not one coherent
                # reverse-process state.
                eps = self.schedule.to_epsilon("sample", x0, x, t_batch)
            else:
                eps = self.schedule.to_epsilon(
                    self.prediction_type, model_out32, x, t_batch
                )

            a_t = alphas_cumprod[t]
            prev_index = timesteps[idx + 1] if idx + 1 < steps else None
            a_prev = (
                alphas_cumprod[prev_index]
                if prev_index is not None
                else torch.ones((), device=device, dtype=torch.float32)
            )

            # Generalized DDIM over the configured (possibly strided) grid.
            variance = ((1.0 - a_prev) / (1.0 - a_t).clamp(min=1e-12)) * (
                1.0 - a_t / a_prev.clamp(min=1e-12)
            )
            sigma = float(eta) * variance.clamp(min=0.0).sqrt()
            if not bool(torch.isfinite(sigma)):
                raise FloatingPointError("DDIM variance produced a non-finite sigma.")
            direction = (1.0 - a_prev - sigma.square()).clamp(min=0.0).sqrt()
            x = a_prev.sqrt() * x0 + direction * eps
            if stochastic and eta > 0.0 and prev_index is not None:
                noise32 = self._randn(shape, device, torch.float32, generator)
                x = x + sigma * noise32
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError(
                    f"DDIM state became non-finite at inference step {idx}."
                )
        return x

    def _sample(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        generator: torch.Generator | None,
        num_steps: int | None,
    ) -> torch.Tensor:
        steps = int(num_steps if num_steps is not None else self.num_inference_steps)
        if steps < 2:
            raise ValueError(
                "Stochastic diffusion sampling requires at least 2 reverse "
                "steps. Use deterministic_mean_correction_normalized() for the "
                "zero-step supervised mean product."
            )
        shape = self.residual_shape(conditioning)
        device = conditioning.device
        x = self._randn(shape, device, torch.float32, generator)
        sampling_eta = 1.0 if self.sampler == "ddpm" else float(self.eta)
        return self._ddim_loop(
            x,
            conditioning,
            self._lead_for(forecast_lead_time),
            steps,
            eta=sampling_eta,
            generator=generator,
            stochastic=True,
        )

    def _deterministic(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """Deterministic (mean-like) residual estimate.

        Two estimators are available; see
        ``RefinementConfig.diffusion.deterministic_estimator``.

        ``posterior_mean`` is the direct analogue of the query that makes the
        existing Aurora flow-matching head work: at the largest timestep the
        noisy state is independent of the residual, so the squared-error optimum
        is ``E[r | conditioning]`` regardless of the state, and evaluating at the
        prior mean ``x = 0`` returns it in a single forward pass. Under
        ``sample`` prediction the conversion to the clean residual is the
        identity, so nothing is divided by ``sqrt(abar_T)``.

        The previous implementation used this query with ``epsilon`` prediction,
        where ``x_0 = (0 - sqrt(1 - abar_T) * eps) / sqrt(abar_T)`` amplifies the
        prediction by ``~1 / sqrt(abar_T)`` and returns a field tens of times
        larger than the residual it estimates. That combination is now rejected
        by :meth:`resolved_deterministic_estimator`, which falls back to the
        integrated ODE.
        """
        shape = self.residual_shape(conditioning)
        device = conditioning.device
        lead = self._lead_for(forecast_lead_time)
        context = torch.enable_grad() if differentiable else torch.no_grad()

        if self.deterministic_estimator == "posterior_mean":
            x = torch.zeros(shape, device=device, dtype=torch.float32)
            t = torch.full(
                (shape[0],), self.num_train_timesteps - 1, device=device, dtype=torch.long
            )
            with context:
                model_out32 = self._network_prediction(
                    x, conditioning, self._embed_time(t), lead
                )
                return self.schedule.to_clean(
                    self.prediction_type, model_out32, x, t
                )

        steps = (
            min(self.config.diffusion.deterministic_training_steps, self.num_inference_steps)
            if differentiable
            else self.num_inference_steps
        )
        x = torch.zeros(shape, device=device, dtype=torch.float32)
        with context:
            return self._ddim_loop(
                x,
                conditioning,
                lead,
                max(1, int(steps)),
                eta=0.0,
                generator=None,
                stochastic=False,
            )


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
