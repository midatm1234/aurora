"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared causal temporal-conditioning interface for all four unified refiners.

Why this module exists
----------------------
The unified refiners are *spatial* networks: the caller folds the lead axis into
the batch, so ``flow_matching_conv_unet``, ``flow_matching_transformer``,
``diffusion_unet`` and ``diffusion_transformer`` each see one frame at a time
and cannot represent forecast evolution. Restoring the sequence for a spatial
operation is bookkeeping, not temporal modelling.

:class:`TemporalContextEncoder` closes that gap. It consumes the *trajectory*

    ``[batch, lead_time, packed_variable_level, latitude, longitude]``

and produces a per-frame **context field**

    ``[batch, lead_time, context_channels, latitude, longitude]``

which is concatenated onto the spatial conditioning of the generative network.
The context therefore enters the denoiser / velocity field at **every** process
evaluation, not as a smoothing filter applied to already-sampled frames.

Separation of axes
------------------
Physical time (lead) and generative-process time (``tau`` / ``k``) are different
axes and this module only ever touches the first one. Its inputs are the raw
Aurora forecast trajectory, previously *refined* frames and calendar metadata --
never the current noised residual. That is a structural guarantee, not a
convention: :meth:`TemporalContextEncoder.forward` has no argument through which
a noised state could be passed. Consequently the context is invariant across
solver evaluations of a single frame and may be computed once per frame and
reused for all ``num_steps`` denoiser calls, which is why
:class:`TemporalContextCache` is sound.

Causality
---------
Every backend is strictly causal: the context at lead ``j`` depends only on
frames ``<= j``. There is no normalization, pooling or padding that mixes
future steps into the past:

* ``causal_conv`` pads only on the left;
* ``conv_gru`` is a forward-only recurrence;
* ``attention`` applies an upper-triangular mask;
* ``mamba`` reuses the reference selective scan, which is a forward scan.

Normalization is :class:`torch.nn.GroupNorm` applied per frame over the channel
axis, so no statistic is ever pooled across the lead axis.
:meth:`TemporalContextEncoder.forward` is prefix invariant, which the test suite
asserts numerically rather than by inspection.

Backends are independent of the spatial backbone
------------------------------------------------
``temporal.backend`` selects the temporal mixer; ``refinement.type`` selects the
spatial backbone; ``transformer.attention_mode`` selects spatial attention. The
three are orthogonal. In particular ``backend="causal_conv"`` adds no attention
anywhere, so an attention-free temporal control really is attention-free.

Memory bound
------------
Temporal mixing runs on a spatially strided grid (``spatial_stride``, default
``4``) and the context is interpolated back to the full grid. Cost is
``O(B * S * D * H * W / stride^2)`` rather than attention over every global
space-time-level token.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from finetune.longitude import PeriodicConv2d, periodic_bilinear_interpolate

__all__ = [
    "TEMPORAL_BACKENDS",
    "TemporalContextCache",
    "TemporalContextEncoder",
    "build_temporal_encoder",
]

#: ``none`` is the explicit spatial-only control and builds no module at all.
TEMPORAL_BACKENDS: tuple[str, ...] = (
    "none",
    "causal_conv",
    "conv_gru",
    "attention",
    "mamba",
)


def _num_groups(channels: int, maximum: int = 8) -> int:
    for group in range(min(maximum, channels), 0, -1):
        if channels % group == 0:
            return group
    return 1


class _StepNorm(nn.Module):
    """Channel normalization applied **independently at each lead time**.

    A plain ``GroupNorm`` over a ``[N, D, S]`` tensor reduces over the trailing
    axis, which here is the lead axis: the statistics of a frame would then
    depend on future frames and the module would silently stop being causal.
    Reshaping so that every step is its own sample keeps the normalization
    strictly within a frame.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(dim), dim)
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``[N, S, D]`` over ``D`` for each ``(N, S)`` pair."""
        count, steps, dim = x.shape
        flat = x.reshape(count * steps, dim, 1)
        return self.norm(flat).reshape(count, steps, dim)


# ---------------------------------------------------------------------------
# Temporal mixers. Each consumes [B, S, D, h, w] and returns the same shape.
# ---------------------------------------------------------------------------


class _CausalTemporalConv(nn.Module):
    """Dilated causal 1-D convolution along the lead axis.

    Left padding only, so output ``j`` sees inputs ``<= j``. Dilations grow
    geometrically to cover the whole rollout with few parameters.
    """

    def __init__(self, dim: int, *, layers: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        kernel = int(kernel_size)
        if kernel < 2:
            raise ValueError(f"causal_conv kernel must be >= 2, got {kernel_size!r}.")
        self.kernel_size = kernel
        self.dilations = [2**index for index in range(max(1, int(layers)))]
        self.convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel, dilation=dilation, groups=_num_groups(dim))
            for dilation in self.dilations
        )
        self.mixers = nn.ModuleList(nn.Conv1d(dim, dim, 1) for _ in self.dilations)
        self.norms = nn.ModuleList(_StepNorm(dim) for _ in self.dilations)

    def forward(self, x: torch.Tensor, valid_steps: torch.Tensor) -> torch.Tensor:
        # x: [N, S, D] -> conv1d wants [N, D, S]
        h = x.transpose(1, 2)
        mask = valid_steps.to(h.dtype)[:, None, :]
        for conv, mixer, norm in zip(self.convs, self.mixers, self.norms):
            residual = h
            padded = F.pad(h * mask, (conv.dilation[0] * (self.kernel_size - 1), 0))
            convolved = conv(padded).transpose(1, 2)
            out = mixer(F.silu(norm(convolved)).transpose(1, 2))
            h = residual + out * mask
        return h.transpose(1, 2)


class _CausalTemporalGRU(nn.Module):
    """Forward-only gated recurrence along the lead axis.

    Gates are computed from ``[x_t, h_{t-1}]`` with a linear map. Invalid steps
    carry the previous state forward unchanged instead of injecting zeros, so a
    missing frame does not silently reset the memory.
    """

    def __init__(self, dim: int, *, layers: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(2 * dim, 3 * dim) for _ in range(max(1, int(layers)))
        )
        self.norms = nn.ModuleList(_StepNorm(dim) for _ in range(len(self.layers)))
        self.dim = int(dim)

    def forward(self, x: torch.Tensor, valid_steps: torch.Tensor) -> torch.Tensor:
        h_seq = x
        for layer, norm in zip(self.layers, self.norms):
            batch, steps, dim = h_seq.shape
            state = h_seq.new_zeros(batch, dim)
            outputs = []
            for step in range(steps):
                current = h_seq[:, step]
                gates = layer(torch.cat([current, state], dim=-1))
                reset, update, candidate = gates.chunk(3, dim=-1)
                reset = torch.sigmoid(reset)
                update = torch.sigmoid(update)
                candidate = torch.tanh(candidate * reset)
                new_state = (1.0 - update) * state + update * candidate
                keep = valid_steps[:, step, None].to(new_state.dtype)
                state = keep * new_state + (1.0 - keep) * state
                outputs.append(state)
            h_seq = h_seq + norm(torch.stack(outputs, dim=1))
        return h_seq


class _CausalTemporalAttention(nn.Module):
    """Masked self-attention whose tokens are the lead times of one trajectory.

    Tokens are frames, never grid cells, so this is temporal attention and is
    unrelated to the spatial-attention setting of the backbone.
    """

    def __init__(self, dim: int, *, layers: int = 1, num_heads: int = 4) -> None:
        super().__init__()
        heads = int(num_heads)
        while heads > 1 and dim % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.ModuleList(nn.Linear(dim, 3 * dim) for _ in range(max(1, int(layers))))
        self.proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(len(self.qkv)))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(len(self.qkv)))

    def forward(self, x: torch.Tensor, valid_steps: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        causal = torch.ones(steps, steps, dtype=torch.bool, device=x.device).tril()
        keep = valid_steps.to(torch.bool)[:, None, None, :]
        attn_mask = causal[None, None] & keep
        # A fully masked row would produce NaNs; always let a step see itself.
        eye = torch.eye(steps, dtype=torch.bool, device=x.device)[None, None]
        attn_mask = attn_mask | eye
        h = x
        for qkv, proj, norm in zip(self.qkv, self.proj, self.norms):
            normed = norm(h)
            q, k, v = qkv(normed).chunk(3, dim=-1)

            def _heads(t: torch.Tensor) -> torch.Tensor:
                # Contiguous so SDPA can use its memory-efficient kernels
                # rather than materializing an [N, heads, S, S] score matrix.
                return (
                    t.view(batch, steps, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )

            out = F.scaled_dot_product_attention(
                _heads(q), _heads(k), _heads(v), attn_mask=attn_mask
            )
            out = out.transpose(1, 2).reshape(batch, steps, dim)
            h = h + proj(out) * valid_steps.to(out.dtype)[:, :, None]
        return h


class _CausalTemporalMamba(nn.Module):
    """Selective state-space mixing reusing the repository's reference scan.

    The scan itself is unchanged; what differs from the historical
    ``mamba_temporal`` head is the surrounding contract: the sequence arrives
    *after* spatial encoding and is consumed as generative conditioning rather
    than as a post-sampling additive correction.
    """

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 2,
        state_dim: int = 8,
        conv_kernel: int = 3,
        expand: int = 2,
    ) -> None:
        super().__init__()
        from finetune.mamba_temporal import MambaBlock

        self.blocks = nn.ModuleList(
            MambaBlock(dim, d_state=state_dim, d_conv=conv_kernel, expand=expand)
            for _ in range(max(1, int(layers)))
        )

    def forward(self, x: torch.Tensor, valid_steps: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, valid_steps=valid_steps.to(torch.bool))
        return h


def _build_mixer(backend: str, dim: int, **kwargs) -> nn.Module:
    if backend == "causal_conv":
        return _CausalTemporalConv(
            dim, layers=kwargs.get("layers", 2), kernel_size=kwargs.get("conv_kernel", 3)
        )
    if backend == "conv_gru":
        return _CausalTemporalGRU(dim, layers=kwargs.get("layers", 1))
    if backend == "attention":
        return _CausalTemporalAttention(
            dim, layers=kwargs.get("layers", 1), num_heads=kwargs.get("num_heads", 4)
        )
    if backend == "mamba":
        return _CausalTemporalMamba(
            dim,
            layers=kwargs.get("layers", 2),
            state_dim=kwargs.get("state_dim", 8),
            conv_kernel=kwargs.get("conv_kernel", 3),
            expand=kwargs.get("expansion_factor", 2),
        )
    raise ValueError(
        f"Unknown temporal backend {backend!r}; supported: {list(TEMPORAL_BACKENDS)}."
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TemporalContextEncoder(nn.Module):
    """Turn a causal forecast trajectory into per-frame generative conditioning.

    Args:
        input_channels: packed channels of one trajectory frame.
        context_channels: width of the emitted conditioning field.
        backend: one of :data:`TEMPORAL_BACKENDS` except ``none``.
        hidden_channels: width of the internal spatial/temporal representation.
        layers: depth of the temporal mixer.
        spatial_stride: spatial decimation used for temporal mixing.
        metadata_features: per-frame scalar features (calendar, lead, elapsed
            hours) broadcast onto the coarse grid before temporal mixing.
        lon_periodic: wrap longitude in the spatial stem.
        max_sequence_length: refuse longer trajectories rather than silently
            exhausting memory.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        context_channels: int = 8,
        backend: str = "causal_conv",
        hidden_channels: int = 32,
        layers: int = 2,
        spatial_stride: int = 4,
        metadata_features: int = 0,
        lon_periodic: bool = False,
        max_sequence_length: int = 64,
        state_dim: int = 8,
        conv_kernel: int = 3,
        expansion_factor: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        backend = str(backend).lower()
        if backend not in TEMPORAL_BACKENDS or backend == "none":
            raise ValueError(
                f"TemporalContextEncoder needs a real backend, got {backend!r}; "
                f"supported: {[b for b in TEMPORAL_BACKENDS if b != 'none']}."
            )
        stride = int(spatial_stride)
        if stride < 1:
            raise ValueError(f"spatial_stride must be >= 1, got {spatial_stride!r}.")

        self.backend = backend
        self.input_channels = int(input_channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.spatial_stride = stride
        self.metadata_features = int(metadata_features)
        self.lon_periodic = bool(lon_periodic)
        self.max_sequence_length = int(max_sequence_length)

        # Spatial stem: the frame is spatially contextual *before* temporal
        # mixing, so the recurrence is never an isolated per-pixel process.
        stem_in = self.input_channels * 2  # field + per-cell validity mask
        self.stem = nn.Sequential(
            PeriodicConv2d(stem_in, self.hidden_channels, 3, lon_periodic=self.lon_periodic),
            nn.GroupNorm(_num_groups(self.hidden_channels), self.hidden_channels),
            nn.SiLU(),
            PeriodicConv2d(
                self.hidden_channels, self.hidden_channels, 3, lon_periodic=self.lon_periodic
            ),
            nn.GroupNorm(_num_groups(self.hidden_channels), self.hidden_channels),
            nn.SiLU(),
        )
        self.metadata_proj = (
            nn.Linear(self.metadata_features, self.hidden_channels)
            if self.metadata_features > 0
            else None
        )
        self.mixer = _build_mixer(
            backend,
            self.hidden_channels,
            layers=layers,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expansion_factor=expansion_factor,
            num_heads=num_heads,
        )
        # A second spatial stage after temporal mixing lets information move
        # between locations *and* times rather than only along time.
        self.spatial_post = nn.Sequential(
            PeriodicConv2d(
                self.hidden_channels, self.hidden_channels, 3, lon_periodic=self.lon_periodic
            ),
            nn.GroupNorm(_num_groups(self.hidden_channels), self.hidden_channels),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Conv2d(self.hidden_channels, self.context_channels, 1)
        # Single zero-initialized projection: the emitted context starts at
        # exactly zero (so enabling temporal conditioning does not perturb a
        # freshly built refiner) while its input activation is non-zero, so the
        # very first backward pass produces a non-zero gradient here and the
        # branch trains immediately. Never pair this with a zero gate.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # -- helpers ---------------------------------------------------------
    def _coarse_size(self, height: int, width: int) -> tuple[int, int]:
        stride = self.spatial_stride
        return max(1, height // stride), max(1, width // stride)

    def describe(self) -> dict[str, object]:
        return {
            "temporal_backend": self.backend,
            "context_channels": self.context_channels,
            "hidden_channels": self.hidden_channels,
            "spatial_stride": self.spatial_stride,
            "metadata_features": self.metadata_features,
            "lon_periodic": self.lon_periodic,
            "max_sequence_length": self.max_sequence_length,
            "causal": True,
        }

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        sequence: torch.Tensor,
        *,
        valid_cell_mask: torch.Tensor | None = None,
        valid_step_mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B, S, context_channels, H, W]`` causal context.

        Args:
            sequence: ``[B, S, C, H, W]`` trajectory in normalized target space.
                Frame ``j`` must be information available at lead ``j``: raw
                Aurora forecasts through ``j`` and previously refined states.
                **Never** the current noised residual and never future truth.
            valid_cell_mask: ``[B, S, C, H, W]`` finite-data mask.
            valid_step_mask: ``[B, S]`` mask marking frames that exist.
            metadata: ``[B, S, metadata_features]`` calendar / lead / elapsed
                features for each frame.
        """
        if sequence.ndim != 5:
            raise ValueError(
                "TemporalContextEncoder expects [batch, lead, channel, latitude, "
                f"longitude], got {tuple(sequence.shape)}."
            )
        batch, steps, channels, height, width = sequence.shape
        if channels != self.input_channels:
            raise ValueError(
                f"TemporalContextEncoder was built for {self.input_channels} packed "
                f"channels, got {channels}."
            )
        if steps > self.max_sequence_length:
            raise ValueError(
                f"Trajectory has {steps} leads but max_sequence_length is "
                f"{self.max_sequence_length}. Raise it deliberately or shorten the "
                "rollout; silently truncating would break causality bookkeeping."
            )

        if valid_cell_mask is None:
            valid_cell_mask = torch.isfinite(sequence)
        if valid_cell_mask.shape != sequence.shape:
            raise ValueError(
                f"valid_cell_mask must match the sequence shape {tuple(sequence.shape)}, "
                f"got {tuple(valid_cell_mask.shape)}."
            )
        if valid_step_mask is None:
            valid_step_mask = valid_cell_mask.flatten(2).any(dim=2)
        if valid_step_mask.shape != (batch, steps):
            raise ValueError(
                f"valid_step_mask must be [{batch}, {steps}], got "
                f"{tuple(valid_step_mask.shape)}."
            )

        clean = torch.where(valid_cell_mask, sequence, torch.zeros_like(sequence))
        stem_input = torch.cat(
            [clean, valid_cell_mask.to(clean.dtype)], dim=2
        ).reshape(batch * steps, 2 * channels, height, width)
        features = self.stem(stem_input)

        coarse_h, coarse_w = self._coarse_size(height, width)
        if (coarse_h, coarse_w) != (height, width):
            features = F.adaptive_avg_pool2d(features, (coarse_h, coarse_w))

        if self.metadata_proj is not None:
            if metadata is None:
                raise ValueError(
                    "TemporalContextEncoder was built with metadata_features="
                    f"{self.metadata_features} but no metadata tensor was supplied."
                )
            if metadata.shape != (batch, steps, self.metadata_features):
                raise ValueError(
                    f"metadata must be [{batch}, {steps}, {self.metadata_features}], "
                    f"got {tuple(metadata.shape)}."
                )
            projected = self.metadata_proj(metadata.to(features.dtype))
            features = features + projected.reshape(
                batch * steps, self.hidden_channels, 1, 1
            )

        # [B*S, D, h, w] -> [B*h*w, S, D] so the mixer sees one sequence per
        # coarse cell. The lead axis is restored before any temporal operation.
        tokens = (
            features.reshape(batch, steps, self.hidden_channels, coarse_h, coarse_w)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch * coarse_h * coarse_w, steps, self.hidden_channels)
        )
        step_mask = (
            valid_step_mask[:, None, None, :]
            .expand(batch, coarse_h, coarse_w, steps)
            .reshape(batch * coarse_h * coarse_w, steps)
        )
        mixed = self.mixer(tokens, step_mask)
        mixed = (
            mixed.reshape(batch, coarse_h, coarse_w, steps, self.hidden_channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch * steps, self.hidden_channels, coarse_h, coarse_w)
        )

        mixed = self.spatial_post(mixed)
        mixed = self.dropout(mixed)
        context = self.out_proj(mixed)
        if (coarse_h, coarse_w) != (height, width):
            context = periodic_bilinear_interpolate(
                context, (height, width), self.lon_periodic
            )
        context = context.reshape(batch, steps, self.context_channels, height, width)
        return context * valid_step_mask.to(context.dtype)[:, :, None, None, None]


def build_temporal_encoder(
    backend: str,
    *,
    input_channels: int,
    lon_periodic: bool,
    metadata_features: int,
    **kwargs,
) -> TemporalContextEncoder | None:
    """Build a temporal encoder, or ``None`` for the ``none`` spatial-only control."""
    name = str(backend).lower()
    if name == "none":
        return None
    return TemporalContextEncoder(
        backend=name,
        input_channels=int(input_channels),
        lon_periodic=bool(lon_periodic),
        metadata_features=int(metadata_features),
        **kwargs,
    )


class TemporalContextCache:
    """Per-frame context reused across solver evaluations of the same frame.

    Physical-time recurrence and diffusion / flow integration are different
    axes. A denoiser is queried ``num_steps`` times for a *single* forecast
    frame; those queries are not successive forecast hours and the temporal
    state must not advance between them.

    This cache stores exactly one context tensor, keyed by
    ``(initialization id, ensemble member, lead index)``. Because the context is
    a function of the causal trajectory only -- never of the noised residual --
    reusing it across solver evaluations is exact rather than approximate. A key
    is required to include the member and the initialization so recurrence state
    can never leak between ensemble members or between initializations.
    """

    def __init__(self) -> None:
        self._store: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _check_key(key: Sequence[object]) -> tuple:
        key = tuple(key)
        if len(key) != 3:
            raise ValueError(
                "TemporalContextCache keys must be (initialization, member, lead_index) "
                f"triples, got {key!r}."
            )
        return key

    def get(self, key: Sequence[object]) -> torch.Tensor | None:
        return self._store.get(self._check_key(key))

    def put(self, key: Sequence[object], value: torch.Tensor) -> None:
        self._store[self._check_key(key)] = value

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


def temporal_metadata_matrix(
    scalar_features: torch.Tensor, batch: int, steps: int
) -> torch.Tensor:
    """Reshape lead-major ``[B*S, F]`` calendar features to ``[B, S, F]``.

    The packed refiners flatten ``(batch, lead)`` in lead-major order. Restoring
    the sequence with the wrong order would pair a frame with another sample's
    calendar, so the reshape lives here and is covered by tests.
    """
    if scalar_features.ndim != 2:
        raise ValueError(
            f"scalar_features must be [batch*steps, features], got "
            f"{tuple(scalar_features.shape)}."
        )
    expected = batch * steps
    if scalar_features.shape[0] != expected:
        raise ValueError(
            f"scalar_features has {scalar_features.shape[0]} rows but batch*steps is "
            f"{expected}."
        )
    return scalar_features.reshape(batch, steps, scalar_features.shape[1])


def elapsed_hours_from_leads(
    lead_hours: torch.Tensor, valid_step_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Actual hours between adjacent frames of ``[B, S]`` cumulative leads.

    The first frame reports its own lead. A gap is therefore visible as a larger
    elapsed time rather than being silently compressed to one cadence step.
    """
    if lead_hours.ndim != 2:
        raise ValueError(f"lead_hours must be [batch, steps], got {tuple(lead_hours.shape)}.")
    elapsed = torch.empty_like(lead_hours)
    elapsed[:, 0] = lead_hours[:, 0]
    if lead_hours.shape[1] > 1:
        elapsed[:, 1:] = lead_hours[:, 1:] - lead_hours[:, :-1]
    if valid_step_mask is not None:
        elapsed = torch.where(
            valid_step_mask.to(torch.bool), elapsed, torch.zeros_like(elapsed)
        )
    return elapsed


def assert_increasing_leads(lead_hours: torch.Tensor) -> None:
    """Fail loudly if a trajectory's leads are not strictly increasing.

    Merging two initializations whose valid times interleave, or feeding a
    shuffled sequence to a causal model, both show up here.
    """
    if lead_hours.ndim != 2:
        raise ValueError(f"lead_hours must be [batch, steps], got {tuple(lead_hours.shape)}.")
    if lead_hours.shape[1] < 2:
        return
    deltas = lead_hours[:, 1:] - lead_hours[:, :-1]
    if not bool((deltas > 0).all()):
        raise ValueError(
            "Forecast leads within one trajectory must be strictly increasing; got "
            f"{lead_hours.detach().cpu().tolist()}. Separate initializations must not "
            "be merged into one sequence."
        )


def sinusoidal_elapsed_features(elapsed_hours: torch.Tensor, scale_hours: float) -> torch.Tensor:
    """``[s, log1p(s)]`` for ``s = elapsed_hours / scale_hours``."""
    scale = float(scale_hours)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale_hours must be positive, got {scale_hours!r}.")
    scaled = elapsed_hours.float() / scale
    return torch.stack([scaled, torch.log1p(scaled.clamp(min=0.0))], dim=-1)
