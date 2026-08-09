"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Conditional flow-matching Phase-2 residual refiner (spatial Transformer).

Formulation
-----------
The residual is defined per rollout step, in Aurora's normalized target space::

    r = target_norm - rollout_norm

Two documented interpolation paths are supported. Both use the straight
(optimal-transport) probability path ``x_t = (1 - t) * x0 + t * r`` with
``x0 ~ N(0, I)``; they differ only in what the network regresses:

``existing_aurora`` (default)
    The ``x1`` / **data** parameterisation used by the existing Aurora
    flow-matching implementation (:mod:`finetune.flow_refine`): the network
    predicts the clean residual ``r`` directly from ``(x_t, t, conditioning)``
    and the loss is a masked MSE against ``r``. Flow time is sampled from the
    same logit-normal distribution ``sigmoid(N(mean, std))`` and inference uses
    the same DDIM-style straight-line update. Keeping this as the default means
    the Transformer variant is a pure *architecture* change relative to the
    branch it extends.

``rectified_flow``
    The **velocity** parameterisation used by the Prithvi reference: the network
    regresses ``u_t = r - x0`` and inference integrates ``dx/dt = v_theta`` from
    ``t = 0`` to ``t = 1`` with the configured solver. Selecting it is an
    explicit, documented scientific choice; it is never applied silently.

``t`` is the integration coordinate of a probability path. It is **not** a
forecast lead time and **not** a diffusion timestep. Forecast lead time is a
separate conditioning input with its own embedding.

Reproducibility
---------------
The initial state ``x0`` is drawn from a caller-supplied
:class:`torch.Generator`, so ensembles are reproducible and individual members
can be compared one-to-one. ``stochastic_initialization: false`` starts from
``x0 = 0``, making the refiner a deterministic (mean-path) corrector.

Adapted from ``granitewxc.refinement.flow_matching`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), with the Aurora
``existing_aurora`` path and the lead-time conditioning added.
"""

from __future__ import annotations

import torch

from finetune.refinement.base import RefinerOutput, masked_loss, register_refiner
from finetune.refinement.config import RefinementConfig
from finetune.refinement.packed import PackedRefiner

__all__ = ["FlowMatchingTransformerRefiner"]

#: The flow time is scaled to roughly the same numeric range as a diffusion
#: timestep index so a single sinusoidal embedding implementation serves both.
#: It remains a distinct module instance with distinct parameters.
_TIME_SCALE = 1000.0


class _BaseFlowMatchingRefiner(PackedRefiner):
    def __init__(
        self,
        config: RefinementConfig,
        *,
        residual_channels: int,
        cond_channels: int,
        metadata=None,
    ) -> None:
        f = config.flow_matching
        self.interpolation_path = f.interpolation_path
        self.integration_steps = f.integration_steps
        self.solver = f.solver
        self.sigma_min = f.sigma_min
        self.stochastic_initialization = f.stochastic_initialization
        self.time_sampling = f.time_sampling
        self.logit_normal_mean = f.logit_normal_mean
        self.logit_normal_std = f.logit_normal_std
        super().__init__(
            config,
            residual_channels=residual_channels,
            cond_channels=cond_channels,
            metadata=metadata,
        )

    @property
    def predicts_velocity(self) -> bool:
        return self.interpolation_path == "rectified_flow"

    @staticmethod
    def _embed_time(t: torch.Tensor) -> torch.Tensor:
        return t.float() * _TIME_SCALE

    def _sample_flow_time(
        self, batch: int, device: torch.device, generator: torch.Generator | None
    ) -> torch.Tensor:
        gen_device = generator.device if generator is not None else device
        if self.time_sampling == "uniform":
            t = torch.rand(batch, generator=generator, device=gen_device, dtype=torch.float32)
        else:  # logit_normal, matching the existing Aurora sampler
            z = torch.randn(batch, generator=generator, device=gen_device, dtype=torch.float32)
            t = torch.sigmoid(z * self.logit_normal_std + self.logit_normal_mean)
        return t.to(device).clamp(self.sigma_min, 1.0 - self.sigma_min)

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
        device, dtype = residual_target.device, residual_target.dtype

        x0 = self._randn(tuple(residual_target.shape), device, dtype, generator)
        t = self._sample_flow_time(batch, device, generator)
        t_b = t.reshape(-1, *([1] * (residual_target.ndim - 1))).to(dtype)

        x_t = (1.0 - t_b) * x0 + t_b * residual_target
        prediction = self.net(
            x_t, conditioning, self._embed_time(t), self._lead_for(forecast_lead_time)
        )

        if self.predicts_velocity:
            objective = residual_target - x0
            # Endpoint estimate consistent with the straight path:
            # x_1 = x_t + (1 - t) * u_t.
            residual_estimate = x_t + (1.0 - t_b) * prediction
        else:
            objective = residual_target
            residual_estimate = prediction

        generative = masked_loss(prediction, objective, mask, self.loss_kind)
        output = RefinerOutput(generative_loss=generative, process_time=t)
        if not self.config.loss.has_auxiliary_terms:
            output.total_loss = generative
            return output
        return self._finish(
            output,
            residual_estimate=residual_estimate,
            residual_target=residual_target,
            mask=mask,
            lead_index=lead_index,
            area_weight=area_weight,
            rollout_normalized=rollout_normalized,
        )

    # -- integration -----------------------------------------------------
    @torch.no_grad()
    def sample_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(num_steps if num_steps is not None else self.integration_steps)
        if steps < 1:
            raise ValueError("integration_steps must be >= 1")
        shape = self.residual_shape(conditioning)
        device, dtype = conditioning.device, conditioning.dtype
        lead = self._lead_for(forecast_lead_time)
        batch = shape[0]

        if self.stochastic_initialization:
            x = self._randn(shape, device, dtype, generator)
        else:
            x = torch.zeros(shape, device=device, dtype=dtype)

        def evaluate(state: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
            return self.net(state, conditioning, self._embed_time(t_value.expand(batch)), lead)

        if not self.predicts_velocity:
            return self._sample_data_parameterisation(x, evaluate, steps, device, dtype)

        # The integration grid is a fixed function of ``steps`` and is built once
        # per call, on the target device, so the loop performs no host sync.
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
        for idx in range(steps):
            t0, t1 = grid[idx], grid[idx + 1]
            dt = (t1 - t0).to(dtype)
            if self.solver == "euler":
                x = x + dt * evaluate(x, t0)
            elif self.solver == "midpoint":
                k1 = evaluate(x, t0)
                mid = x + 0.5 * dt * k1
                x = x + dt * evaluate(mid, (t0 + t1) * 0.5)
            elif self.solver == "heun":
                k1 = evaluate(x, t0)
                k2 = evaluate(x + dt * k1, t1)
                x = x + 0.5 * dt * (k1 + k2)
            else:  # pragma: no cover - guarded by config validation
                raise ValueError(f"Unsupported flow solver {self.solver!r}")
        return x

    def _sample_data_parameterisation(
        self,
        x: torch.Tensor,
        evaluate,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Existing-Aurora sampler for the ``x1`` (data) parameterisation.

        ``steps == 1`` reduces to a single deterministic regression at the source
        mean, exactly as the existing implementation does; more steps run the
        DDIM-style straight-line update.
        """
        if steps == 1:
            zero = torch.zeros_like(x)
            t_zero = torch.zeros((), device=device, dtype=torch.float32)
            return evaluate(zero, t_zero)

        schedule = torch.linspace(0.0, 1.0 - self.sigma_min, steps + 1, device=device)
        r_hat = torch.zeros_like(x)
        for idx in range(steps):
            t_curr = schedule[idx]
            t_next = schedule[idx + 1]
            r_hat = evaluate(x, t_curr)
            curr = t_curr.to(dtype)
            nxt = t_next.to(dtype)
            if float(t_curr) > 0.0:
                x0_est = (x - curr * r_hat) / (1.0 - curr)
                x = (1.0 - nxt) * x0_est + nxt * r_hat
            else:
                x = (1.0 - nxt) * x + nxt * r_hat
        return r_hat

    @torch.no_grad()
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-path residual estimate (no random draw)."""
        if not self.predicts_velocity:
            shape = self.residual_shape(conditioning)
            device, dtype = conditioning.device, conditioning.dtype
            zero = torch.zeros(shape, device=device, dtype=dtype)
            t_zero = torch.zeros(shape[0], device=device, dtype=torch.float32)
            return self.net(
                zero,
                conditioning,
                self._embed_time(t_zero),
                self._lead_for(forecast_lead_time),
            )
        saved = self.stochastic_initialization
        try:
            self.stochastic_initialization = False
            return self.sample_residual(
                conditioning, forecast_lead_time=forecast_lead_time, generator=None
            )
        finally:
            self.stochastic_initialization = saved


@register_refiner("flow_matching_transformer")
class FlowMatchingTransformerRefiner(_BaseFlowMatchingRefiner):
    """Spatial-token Transformer conditional flow-matching residual refiner."""
