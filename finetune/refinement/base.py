"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Common Phase-2 residual-refiner interface and registry.

Every refiner implements the same three operations so training, validation,
inference, evaluation, checkpoint loading and output writing never need a
per-type ``if/elif`` chain:

* ``compute_training_loss(residual_target, conditioning, *, forecast_lead_time,
  mask, generator, lead_index, area_weight, rollout_normalized)``
* ``sample_residual(conditioning, *, forecast_lead_time, generator, num_steps)``
* ``deterministic_residual(conditioning, *, forecast_lead_time)``

Ensemble generation is driven by the caller (see
:meth:`finetune.refinement.two_phase.AuroraTwoPhaseRefiner.refine`), which draws
``ensemble_size`` members through per-member generators.

All tensors are in the **normalized target space**
(:mod:`finetune.refinement.target_space`).

``forecast_lead_time`` is the physical time between forecast initialization and
valid time. It is *never* the diffusion timestep or the flow interpolation
coordinate: those are generated internally by the refiner and use separate
embeddings.

Adapted from ``granitewxc.refinement.base`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), extended with an explicit
forecast lead-time argument and a structured :class:`RefinerOutput`.
"""

from __future__ import annotations

import abc
import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import nn

from finetune.refinement.config import ConfigValidationError, RefinementConfig

__all__ = [
    "ChunkNoiseSource",
    "RefinerOutput",
    "ResidualRefiner",
    "available_refiners",
    "build_refiner",
    "masked_loss",
    "register_refiner",
]


_REGISTRY: dict[str, Callable[..., ResidualRefiner]] = {}


@dataclass
class RefinerOutput:
    """Structured result shared by every refiner.

    Fields that do not apply to a given refiner or call stay ``None``, so a
    caller can consume any refiner through the same accessors.
    """

    generative_loss: torch.Tensor | None = None
    reconstruction_loss: torch.Tensor | None = None
    bias_loss: torch.Tensor | None = None
    gradient_loss: torch.Tensor | None = None
    pattern_correlation_loss: torch.Tensor | None = None
    total_loss: torch.Tensor | None = None
    predicted_residual: torch.Tensor | None = None
    refined_normalized: torch.Tensor | None = None
    refined_physical: torch.Tensor | None = None
    members: torch.Tensor | None = None
    member_residuals: torch.Tensor | None = None
    ensemble_mean: torch.Tensor | None = None
    ensemble_spread: torch.Tensor | None = None
    #: diffusion timesteps / flow interpolation times drawn for this call.
    process_time: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != {}}

    def scalar_losses(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in (
            "generative_loss",
            "reconstruction_loss",
            "bias_loss",
            "gradient_loss",
            "pattern_correlation_loss",
            "total_loss",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value.detach())
        return out


class ChunkNoiseSource:
    """Per-ensemble-member noise source.

    Batched ensemble generation replicates the (identical) conditioning across
    members with ``repeat_interleave``, giving the flat layout
    ``[b0m0, b0m1, ..., b0m(M-1), b1m0, ...]``. Drawing one big ``randn`` for
    that flat batch would make the result depend on how many members happen to
    share a chunk. Instead, every member owns a dedicated
    :class:`torch.Generator`, so member ``m`` receives exactly the same noise
    sequence whether it was evaluated alone or inside a chunk of any size.

    This is what makes "serial versus batched ensemble generation" a bitwise
    parity test rather than a statistical one.
    """

    def __init__(self, generators: Sequence[torch.Generator], batch_size: int) -> None:
        self.generators = list(generators)
        self.batch_size = int(batch_size)

    def randn(self, shape: tuple[int, ...], device, dtype) -> torch.Tensor:
        members = len(self.generators)
        total, *rest = shape
        if total != self.batch_size * members:
            raise RuntimeError(
                f"ChunkNoiseSource expected a leading dimension of "
                f"{self.batch_size * members}, got {total}."
            )
        per_member = [
            torch.randn(
                (self.batch_size, *rest),
                generator=g,
                device=g.device,
                dtype=torch.float32,
            ).to(device=device, dtype=dtype)
            for g in self.generators
        ]
        return torch.stack(per_member, dim=1).reshape(total, *rest)


def register_refiner(name: str):
    """Class decorator registering a refiner under ``name``."""

    def _decorator(cls):
        key = str(name).lower()
        if key in _REGISTRY:
            raise ValueError(f"Refiner {key!r} is already registered.")
        _REGISTRY[key] = cls
        cls.refiner_type = key
        return cls

    return _decorator


def available_refiners() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_refiner(
    config: RefinementConfig,
    *,
    residual_channels: int,
    cond_channels: int,
    metadata: Any = None,
) -> ResidualRefiner | None:
    """Factory for Phase-2 refiners.

    Returns ``None`` when refinement is disabled so callers keep a single code
    path for the deterministic configuration.

    Args:
        config: resolved :class:`RefinementConfig`.
        residual_channels: number of packed target channels.
        cond_channels: number of packed spatial conditioning channels.
        metadata: optional :class:`~finetune.refinement.packing.FieldPacking`
            (or any object) forwarded to refiners that need variable/level
            metadata.
    """
    if not config.is_active:
        return None
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise ConfigValidationError(
            f"Unknown refinement.type {config.type!r}. Registered refiners: "
            f"{list(available_refiners())}."
        )
    return cls(
        config,
        residual_channels=residual_channels,
        cond_channels=cond_channels,
        metadata=metadata,
    )


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    kind: str = "mse",
) -> torch.Tensor:
    """Masked reduction shared by every refiner.

    Invalid cells contribute exactly zero and are excluded from the denominator,
    so masks never bias the loss magnitude. The reduction is always performed in
    float32 for numerical stability under automatic mixed precision.
    """
    diff = (prediction - target).float()
    if kind == "mse":
        elementwise = diff.pow(2)
    elif kind == "l1":
        elementwise = diff.abs()
    elif kind == "huber":
        elementwise = torch.nn.functional.huber_loss(
            prediction.float(), target.float(), reduction="none", delta=1.0
        )
    else:
        raise ConfigValidationError(f"Unsupported refinement.loss.generative {kind!r}")

    if valid_mask is None:
        return elementwise.mean()
    mask = valid_mask.to(elementwise.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (elementwise * mask).sum() / denom


class ResidualRefiner(nn.Module, abc.ABC):
    """Base class for all Phase-2 stochastic residual refiners."""

    #: set by :func:`register_refiner`
    refiner_type: str = "abstract"

    def __init__(
        self,
        config: RefinementConfig,
        *,
        residual_channels: int,
        cond_channels: int,
        metadata: Any = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.residual_channels = int(residual_channels)
        self.cond_channels = int(cond_channels)
        self.loss_kind = config.loss.generative
        self._noise_source: ChunkNoiseSource | None = None

    @contextlib.contextmanager
    def use_noise_source(self, source: ChunkNoiseSource | None):
        """Temporarily route every ``randn`` draw through ``source``."""
        previous = self._noise_source
        self._noise_source = source
        try:
            yield
        finally:
            self._noise_source = previous

    # -- required API ----------------------------------------------------
    @abc.abstractmethod
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
        """One stochastic training-objective evaluation.

        Args:
            residual_target: ``[N, C, H, W]`` normalized residual target.
            conditioning: ``[N, C_cond, H, W]`` spatial conditioning stack.
            forecast_lead_time: ``[N]`` physical forecast lead time in hours, or
                ``None`` when lead conditioning is disabled.
            mask: ``[N, C, H, W]`` boolean validity mask.
            generator: reproducible source of randomness.
            lead_index: ``[N]`` rollout-step index used only to group the
                bias-aware auxiliary losses by forecast lead time.
            area_weight: ``[1, 1, H, 1]`` cosine-latitude weights for the
                auxiliary losses.
            rollout_normalized: deterministic rollout in normalized space,
                required by the pattern-correlation auxiliary term.
        """

    @abc.abstractmethod
    def sample_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Draw one residual sample per batch element."""

    @abc.abstractmethod
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cheapest deterministic (mean-like) residual estimate."""

    # -- shared helpers --------------------------------------------------
    def residual_shape(self, conditioning: torch.Tensor) -> tuple[int, ...]:
        return (conditioning.shape[0], self.residual_channels, *conditioning.shape[-2:])

    def _randn(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self._noise_source is not None:
            return self._noise_source.randn(shape, device, dtype)
        if generator is not None and generator.device != torch.device(device):
            # ``torch.randn`` requires the generator and the output tensor to be
            # on the same device. Draw on the generator's device and move.
            drawn = torch.randn(
                shape, generator=generator, device=generator.device, dtype=torch.float32
            )
            return drawn.to(device=device, dtype=dtype)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        return (
            f"type={self.refiner_type}, residual_channels={self.residual_channels}, "
            f"cond_channels={self.cond_channels}"
        )
