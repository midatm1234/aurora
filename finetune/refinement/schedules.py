"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Process-time embeddings, forecast lead-time embedding and diffusion schedules.

Three quantities are kept strictly separate:

``diffusion timestep``
    a discrete index ``k`` in ``[0, training_timesteps)`` of a Gaussian noising
    process. Embedded by :class:`ProcessTimeEmbedding`.
``flow interpolation time``
    a continuous coordinate ``t`` in ``[0, 1]`` along a probability path. Also
    embedded by :class:`ProcessTimeEmbedding`, but with its own instance and its
    own configuration fields.
``forecast lead time``
    the *physical* time in hours between forecast initialization and valid time.
    Embedded by :class:`LeadTimeEmbedding`, which is a separate module with
    separate parameters. It is never reused as a process-time embedding and it
    never becomes an attention token.

Schedules are constructed once per configuration and cached as buffers on the
owning module so no host/device synchronisation happens inside sampling loops.

Adapted from ``granitewxc.refinement.schedules`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0). :class:`LeadTimeEmbedding`
is Aurora-specific and follows the lead-time feature construction already used
by :mod:`finetune.flow_refine`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = [
    "DiffusionSchedule",
    "GaussianFourierProjection",
    "LeadTimeEmbedding",
    "ProcessTimeEmbedding",
    "sinusoidal_embedding",
]


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of a scalar process time.

    Args:
        t: ``[N]`` tensor of process-time values (any real scale).
        dim: even embedding width.
        max_period: largest sinusoid period.

    Returns:
        ``[N, dim]`` embedding.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_embedding requires an even dim, got {dim}")
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GaussianFourierProjection(nn.Module):
    """Random Fourier features of a scalar process time."""

    def __init__(self, embed_dim: int, scale: float = 16.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                f"GaussianFourierProjection requires an even embed_dim, got {embed_dim}"
            )
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        proj = t.reshape(-1)[:, None].float() * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ProcessTimeEmbedding(nn.Module):
    """MLP-projected embedding of the diffusion timestep **or** flow time.

    One instance embeds exactly one process variable. Diffusion and flow
    refiners each own their own instance; the two are never shared.
    """

    def __init__(self, dim: int, kind: str = "sinusoidal", fourier_scale: float = 16.0) -> None:
        super().__init__()
        kind = str(kind).lower()
        if kind not in {"sinusoidal", "fourier"}:
            raise ValueError(f"Unsupported process-time embedding {kind!r}")
        self.kind = kind
        self.dim = int(dim)
        if self.dim % 2 != 0:
            raise ValueError(f"ProcessTimeEmbedding dim must be even, got {dim}")
        self.fourier = (
            GaussianFourierProjection(self.dim, scale=fourier_scale) if kind == "fourier" else None
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = self.fourier(t) if self.fourier is not None else sinusoidal_embedding(t, self.dim)
        return self.mlp(base.to(self.mlp[0].weight.dtype))


class LeadTimeEmbedding(nn.Module):
    """Embedding of the **physical forecast lead time** (hours).

    Distinct module, distinct parameters, distinct configuration from the
    process-time embedding. The three-feature construction
    ``[s, log1p(s), sqrt(s)]`` with ``s = lead_hours / scale_hours`` matches the
    existing Aurora flow-matching lead conditioning in
    :mod:`finetune.flow_refine`, so the two representations stay comparable.

    The output is *added* to the process-time embedding before modulation, which
    is feature-wise conditioning, not attention: no token is created for lead
    time and lead-time items never interact with each other.
    """

    def __init__(self, dim: int, scale_hours: float = 72.0, zero_init: bool = True) -> None:
        super().__init__()
        scale = float(scale_hours)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"LeadTimeEmbedding scale_hours must be positive, got {scale_hours!r}")
        self.dim = int(dim)
        self.scale_hours = scale
        self.mlp = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        if zero_init:
            # Zero-initialised so enabling lead conditioning is the identity at
            # construction and never perturbs a freshly built refiner.
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, lead_hours: torch.Tensor) -> torch.Tensor:
        scaled = lead_hours.reshape(-1).float() / self.scale_hours
        features = torch.stack(
            [scaled, torch.log1p(scaled.clamp(min=0.0)), scaled.clamp(min=0.0).sqrt()], dim=-1
        )
        return self.mlp(features.to(self.mlp[0].weight.dtype))


class DiffusionSchedule(nn.Module):
    """Discrete DDPM noise schedule with DDIM-compatible reverse sampling.

    All coefficient tensors are non-persistent buffers: they live on the
    module's device, are never rebuilt inside a sampling loop, and are excluded
    from the state dict because they are a deterministic function of the
    configuration.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 8e-3,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule = str(schedule).lower()
        if self.num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be at least one.")
        if self.schedule in {"linear", "scaled_linear"}:
            start = float(beta_start)
            end = float(beta_end)
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and 0.0 < start <= end < 1.0
            ):
                raise ValueError(
                    "Linear diffusion betas require 0 < beta_start <= beta_end < 1; "
                    f"got beta_start={beta_start!r}, beta_end={beta_end!r}."
                )
        if not math.isfinite(float(cosine_s)) or float(cosine_s) < 0.0:
            raise ValueError(f"cosine_s must be finite and non-negative, got {cosine_s!r}.")

        betas = self._build_betas(
            self.num_train_timesteps, self.schedule, beta_start, beta_end, cosine_s
        ).float()
        if not bool(torch.isfinite(betas).all()) or bool(
            ((betas <= 0.0) | (betas >= 1.0)).any()
        ):
            raise ValueError("Diffusion beta schedule contains values outside (0, 1).")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        if not bool(torch.isfinite(alphas_cumprod).all()) or bool(
            (alphas_cumprod <= 0.0).any()
        ):
            raise ValueError("Diffusion cumulative alphas are non-finite or non-positive.")

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False
        )

    @staticmethod
    def _build_betas(
        num_steps: int, schedule: str, beta_start: float, beta_end: float, cosine_s: float
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64).float()
        if schedule == "scaled_linear":
            return (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, dtype=torch.float64) ** 2
            ).float()
        if schedule == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            f = torch.cos((steps + cosine_s) / (1.0 + cosine_s) * math.pi * 0.5) ** 2
            alphas_cumprod = f / f[0]
            betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return betas.clamp(1e-8, 0.999).float()
        raise ValueError(f"Unsupported diffusion schedule {schedule!r}")

    # -- forward (noising) process --------------------------------------
    def _coefficients(
        self, reference: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference.ndim < 1 or not reference.is_floating_point():
            raise TypeError(
                "Diffusion state must be a floating tensor with a batch dimension."
            )
        indices = timesteps.reshape(-1).to(device=reference.device, dtype=torch.long)
        if indices.numel() != reference.shape[0]:
            raise ValueError(
                "Diffusion timestep count must match state batch size; "
                f"got {indices.numel()} and {reference.shape[0]}."
            )
        if bool(((indices < 0) | (indices >= self.num_train_timesteps)).any()):
            raise ValueError(
                f"Diffusion timesteps must lie in [0, {self.num_train_timesteps - 1}]."
            )
        shape = (-1,) + (1,) * (reference.ndim - 1)
        s_a = self.sqrt_alphas_cumprod.to(
            device=reference.device, dtype=torch.float32
        )[indices].reshape(shape)
        s_1ma = self.sqrt_one_minus_alphas_cumprod.to(
            device=reference.device, dtype=torch.float32
        )[indices].reshape(shape)
        return s_a, s_1ma

    @staticmethod
    def _matching_float32(
        first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first.shape != second.shape:
            raise ValueError(
                f"{first_name} and {second_name} shapes must match; got "
                f"{tuple(first.shape)} and {tuple(second.shape)}."
            )
        if first.device != second.device:
            raise ValueError(
                f"{first_name} and {second_name} devices must match; got "
                f"{first.device} and {second.device}."
            )
        return first.float(), second.float()

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Apply the forward noising equation entirely in float32."""
        clean32, noise32 = self._matching_float32(clean, noise, "clean", "noise")
        s_a, s_1ma = self._coefficients(clean32, timesteps)
        return s_a * clean32 + s_1ma * noise32

    def velocity_target(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Return the Salimans-Ho velocity target in float32."""
        clean32, noise32 = self._matching_float32(clean, noise, "clean", "noise")
        s_a, s_1ma = self._coefficients(clean32, timesteps)
        return s_a * noise32 - s_1ma * clean32

    def training_target(
        self,
        prediction_type: str,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        clean32, noise32 = self._matching_float32(clean, noise, "clean", "noise")
        if prediction_type == "epsilon":
            return noise32
        if prediction_type == "sample":
            return clean32
        if prediction_type == "velocity":
            return self.velocity_target(clean32, noise32, timesteps)
        raise ValueError(f"Unsupported prediction_type {prediction_type!r}")

    # -- reverse (denoising) process ------------------------------------
    def to_clean(
        self,
        prediction_type: str,
        model_output: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a model parameterisation to clean x0 in float32."""
        output32, noisy32 = self._matching_float32(
            model_output, noisy, "model_output", "noisy"
        )
        s_a, s_1ma = self._coefficients(noisy32, timesteps)
        if prediction_type == "sample":
            return output32
        if prediction_type == "epsilon":
            return (noisy32 - s_1ma * output32) / s_a.clamp(min=1e-12)
        if prediction_type == "velocity":
            return s_a * noisy32 - s_1ma * output32
        raise ValueError(f"Unsupported prediction_type {prediction_type!r}")

    def to_epsilon(
        self,
        prediction_type: str,
        model_output: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a model parameterisation to epsilon in float32."""
        output32, noisy32 = self._matching_float32(
            model_output, noisy, "model_output", "noisy"
        )
        s_a, s_1ma = self._coefficients(noisy32, timesteps)
        if prediction_type == "epsilon":
            return output32
        if prediction_type == "sample":
            return (noisy32 - s_a * output32) / s_1ma.clamp(min=1e-12)
        if prediction_type == "velocity":
            return s_a * output32 + s_1ma * noisy32
        raise ValueError(f"Unsupported prediction_type {prediction_type!r}")

    def inference_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """Descending DDIM timestep grid, built once per sampling call."""
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) exceeds "
                f"num_train_timesteps ({self.num_train_timesteps})"
            )
        # Every reverse process starts at the noisiest state the network saw in
        # training and terminates at t=0. The old arange*stride grid started at
        # T-stride (980 for T=1000, S=50), while S=1 incorrectly started at
        # t=0 despite drawing its state from a Gaussian prior.
        return (
            torch.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.float64,
            )
            .round()
            .long()
            .to(device)
        )
