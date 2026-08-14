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
        self.deterministic_training_steps = f.deterministic_training_steps
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
    def _network_prediction(
        self,
        state_float32: torch.Tensor,
        conditioning: torch.Tensor,
        process_time: torch.Tensor,
        lead: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate the flow network at model precision and return float32."""
        prediction = self.net(
            state_float32.to(dtype=conditioning.dtype),
            conditioning,
            process_time,
            lead,
        )
        if prediction.shape != state_float32.shape:
            raise RuntimeError(
                "Flow network output shape must match its state; got "
                f"{tuple(prediction.shape)} and {tuple(state_float32.shape)}."
            )
        prediction32 = prediction.float()
        if not bool(torch.isfinite(prediction32).all()):
            raise FloatingPointError("Flow network produced non-finite values.")
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

        source32 = self._randn(
            tuple(correction32.shape), device, torch.float32, generator
        )
        t = self._sample_flow_time(batch, device, generator)
        t_b = t.reshape(-1, *([1] * (correction32.ndim - 1)))

        state32 = (1.0 - t_b) * source32 + t_b * correction32
        prediction32 = self._network_prediction(
            state32,
            conditioning,
            self._embed_time(t),
            self._lead_for(forecast_lead_time),
        )

        if self.predicts_velocity:
            objective32 = correction32 - source32
            # Endpoint estimate for the straight path: x1 = xt + (1-t) u_t.
            correction_estimate32 = state32 + (1.0 - t_b) * prediction32
        else:
            objective32 = correction32
            correction_estimate32 = prediction32

        generative = masked_loss(
            prediction32, objective32, mask, self.loss_kind
        )
        return (
            RefinerOutput(generative_loss=generative, process_time=t),
            correction_estimate32,
        )

    # -- integration -----------------------------------------------------
    def _sample(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        generator: torch.Generator | None,
        num_steps: int | None,
    ) -> torch.Tensor:
        steps = int(num_steps if num_steps is not None else self.integration_steps)
        if steps < 1:
            raise ValueError("integration_steps must be >= 1")
        shape = self.residual_shape(conditioning)
        device = conditioning.device
        lead = self._lead_for(forecast_lead_time)
        batch = shape[0]

        if self.stochastic_initialization:
            state32 = self._randn(shape, device, torch.float32, generator)
        else:
            state32 = torch.zeros(shape, device=device, dtype=torch.float32)

        def evaluate(state: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
            return self._network_prediction(
                state,
                conditioning,
                self._embed_time(t_value.expand(batch)),
                lead,
            )

        if not self.predicts_velocity:
            return self._sample_data_parameterisation(
                state32, evaluate, steps, device
            )

        grid = torch.linspace(
            0.0, 1.0, steps + 1, device=device, dtype=torch.float32
        )
        for idx in range(steps):
            t0, t1 = grid[idx], grid[idx + 1]
            dt = t1 - t0
            if self.solver == "euler":
                state32 = state32 + dt * evaluate(state32, t0)
            elif self.solver == "midpoint":
                k1 = evaluate(state32, t0)
                midpoint = state32 + 0.5 * dt * k1
                state32 = state32 + dt * evaluate(midpoint, (t0 + t1) * 0.5)
            elif self.solver == "heun":
                k1 = evaluate(state32, t0)
                k2 = evaluate(state32 + dt * k1, t1)
                state32 = state32 + 0.5 * dt * (k1 + k2)
            else:  # pragma: no cover - guarded by config validation
                raise ValueError(f"Unsupported flow solver {self.solver!r}")
            if not bool(torch.isfinite(state32).all()):
                raise FloatingPointError(
                    f"Flow state became non-finite at integration step {idx}."
                )
        return state32

    def _sample_data_parameterisation(
        self,
        state32: torch.Tensor,
        evaluate,
        steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Existing-Aurora x1 sampler with float32 straight-line updates."""
        if steps == 1:
            source_mean = torch.zeros_like(state32)
            t_zero = torch.zeros((), device=device, dtype=torch.float32)
            return evaluate(source_mean, t_zero)

        schedule = torch.linspace(
            0.0,
            1.0 - self.sigma_min,
            steps + 1,
            device=device,
            dtype=torch.float32,
        )
        correction_estimate32 = torch.zeros_like(state32)
        for idx in range(steps):
            current = schedule[idx]
            following = schedule[idx + 1]
            correction_estimate32 = evaluate(state32, current)
            if idx > 0:
                source_estimate32 = (
                    state32 - current * correction_estimate32
                ) / (1.0 - current).clamp(min=1e-12)
                state32 = (
                    (1.0 - following) * source_estimate32
                    + following * correction_estimate32
                )
            else:
                state32 = (
                    (1.0 - following) * state32
                    + following * correction_estimate32
                )
            if not bool(torch.isfinite(state32).all()):
                raise FloatingPointError(
                    f"Flow data state became non-finite at integration step {idx}."
                )
        return correction_estimate32

    def _deterministic(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """Mean-path residual estimate (no random draw).

        For the ``x1`` (data) parameterisation this is a single query at
        ``t = 0, x = 0``. Because the source noise is independent of the
        residual, ``E[r | x_0, t=0, cond] = E[r | cond]`` for *any* ``x_0``, so
        evaluating at the source mean returns the squared-error optimum. This is
        also exactly what ``integration_steps: 1`` computes at inference, which
        is what makes ``loss.deterministic_weight`` supervise the deployed
        estimator directly.
        """
        context = torch.enable_grad() if differentiable else torch.no_grad()
        if not self.predicts_velocity:
            shape = self.residual_shape(conditioning)
            device = conditioning.device
            source_mean = torch.zeros(
                shape, device=device, dtype=torch.float32
            )
            t_zero = torch.zeros(shape[0], device=device, dtype=torch.float32)
            with context:
                return self._network_prediction(
                    source_mean,
                    conditioning,
                    self._embed_time(t_zero),
                    self._lead_for(forecast_lead_time),
                )
        saved = self.stochastic_initialization
        try:
            self.stochastic_initialization = False
            steps = (
                min(self.deterministic_training_steps, self.integration_steps)
                if differentiable
                else None
            )
            with context:
                return self._sample(
                    conditioning,
                    forecast_lead_time=forecast_lead_time,
                    generator=None,
                    num_steps=steps,
                )
        finally:
            self.stochastic_initialization = saved


@register_refiner("flow_matching_transformer")
class FlowMatchingTransformerRefiner(_BaseFlowMatchingRefiner):
    """Spatial-token Transformer conditional flow-matching residual refiner."""


@register_refiner("flow_matching_conv_unet")
class FlowMatchingConvUNetRefiner(_BaseFlowMatchingRefiner):
    """Conditional flow matching on the convolutional UNet backbone.

    Identical objective, interpolation path, time sampling, sampler and residual
    scaling as :class:`FlowMatchingTransformerRefiner`; only the network differs.
    It exists because the Transformer is *not* automatically the better choice
    for this task: patch tokenization discards sub-patch locality, which a fully
    convolutional backbone keeps for free. Having both under one objective makes
    "UNet versus Transformer" a controlled architecture comparison rather than a
    confounded one.

    This is distinct from ``flow_matching_unet``, which remains routed to the
    original Aurora wrapper (:mod:`finetune.flow_refine`) so that existing
    configurations and checkpoints are untouched.
    """
