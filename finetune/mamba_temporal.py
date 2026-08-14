"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Mamba-based temporal module for Aurora long-lead bias correction.

Aurora's spatial refinement heads correct the *spatial* structure and per-step
bias of a prediction, but they treat
each rollout step independently — it has no memory of how the atmospheric
state (and Aurora's error in it) evolves through time. Empirically this is
exactly where long rollouts degrade: a 6 h correction is easy, but by 2–3 days
the accumulated, temporally-correlated drift is something a per-step spatial
model cannot see.

This module adds a small **temporal corrector** that runs *on top of* a spatial
refinement output. For each refined variable it:

  1. encodes every (spatially refined, normalised) frame of a short rollout
     sequence into a per-pixel feature vector,
  2. runs a selective state-space model (Mamba / S6) over the **time axis**
     independently at every pixel (parameters shared across space and, for
     atmospheric variables, across pressure levels),
  3. decodes the temporal state back to a single-channel *temporal correction*
     that is added to the spatially refined field.

The decoder output projection is zero-initialised, so at construction the
temporal correction is identically zero: enabling the module changes nothing
until it is trained (``identity-at-init``), which keeps existing
spatial-only checkpoints numerically unchanged.

Selective-scan backend
----------------------
The authoritative implementation is a self-contained pure-PyTorch selective
scan. It runs on both CPU and GPU and, critically, uses the parameters that are
registered before optimizer/DDP construction. A previous optional fast path
created an unrelated ``mamba_ssm.Mamba`` module lazily during ``forward``;
those new weights were neither the trained fallback weights nor guaranteed to
be in the optimizer/checkpoint, so that path is intentionally not used.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.longitude import PeriodicConv2d

if TYPE_CHECKING:
    from finetune.refinement.packing import FieldPacking

__all__ = [
    "MambaTemporalModule",
    "PackedMambaTemporalAdapter",
    "SelectiveSSM",
    "MambaBlock",
]


# ---------------------------------------------------------------------------
# Selective state-space core (Mamba / S6)
# ---------------------------------------------------------------------------


def _selective_scan_ref(
    u: torch.Tensor,  # (B, S, D)  — gated input sequence
    delta: torch.Tensor,  # (B, S, D)  — per-step, per-channel timestep
    A: torch.Tensor,  # (D, N)     — diagonal state matrix (negative)
    B_mat: torch.Tensor,  # (B, S, N)  — input projection
    C_mat: torch.Tensor,  # (B, S, N)  — output projection
    D_skip: torch.Tensor,  # (D,)       — skip connection
) -> torch.Tensor:
    """Pure-PyTorch selective scan (sequential over the time axis).

    Implements the discretised, input-dependent SSM recurrence

        h_s = exp(delta_s · A) · h_{s-1}  +  (delta_s · B_s) · u_s
        y_s = C_s · h_s  +  D · u_s

    with a diagonal ``A``. The scan is sequential in ``S`` (typically 2–6
    rollout steps), so the Python loop is cheap; all other axes are vectorised.

    Returns ``y`` of shape (B, S, D).
    """
    output_dtype = u.dtype
    # Keep the recurrence in fp32. A is deliberately parameterised in fp32;
    # allowing bf16/fp16 inputs to meet it directly in einsum/multiplication
    # either raises a dtype error or loses too much precision over long scans.
    u = u.float()
    delta = delta.float()
    A = A.float()
    B_mat = B_mat.float()
    C_mat = C_mat.float()
    D_skip = D_skip.float()

    Bsz, S, Dn = u.shape
    N = A.shape[1]

    # Discretise: dA (B,S,D,N) = exp(delta · A); dB·u (B,S,D,N).
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,S,D,N)
    dBu = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * u.unsqueeze(-1)  # (B,S,D,N)

    h = torch.zeros(Bsz, Dn, N, device=u.device, dtype=u.dtype)
    ys = []
    for s in range(S):
        h = dA[:, s] * h + dBu[:, s]  # (B,D,N)
        y_s = torch.einsum("bdn,bn->bd", h, C_mat[:, s])  # (B,D)
        ys.append(y_s)
    y = torch.stack(ys, dim=1)  # (B,S,D)
    y = y + u * D_skip.view(1, 1, -1)
    return y.to(output_dtype)


class SelectiveSSM(nn.Module):
    """A single Mamba selective state-space layer over a (B, S, d_model) seq.

    Faithful to the reference S6 block: input expansion, a causal depthwise
    temporal conv, input-dependent (Δ, B, C), diagonal state matrix ``A`` and a
    SiLU gate. Uses :func:`_selective_scan_ref` on both CPU and GPU so the same
    eagerly registered parameters are trained, saved and restored everywhere.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.d_inner = int(expand) * self.d_model
        self.dt_rank = int(dt_rank) if dt_rank else max(1, self.d_model // 16)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is parameterised in log-space and kept negative: A = -exp(A_log).
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, S, d_model). Returns (B, S, d_model)."""
        B, S, _ = x.shape
        xz = self.in_proj(x)  # (B,S,2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B,S,d_inner)

        # Causal depthwise temporal conv (truncate the right padding to S).
        xc = x_in.transpose(1, 2)  # (B,d_inner,S)
        xc = self.conv1d(xc)[..., :S]
        x_in = F.silu(xc.transpose(1, 2))  # (B,S,d_inner)

        params = self.x_proj(x_in)  # (B,S,dt_rank+2*d_state)
        dt, B_mat, C_mat = torch.split(
            params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt))  # (B,S,d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner,d_state)

        y = _selective_scan_ref(x_in, delta, A, B_mat, C_mat, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Discard weights from the removed, unrelated lazy CUDA module.

        Older GPU runs could instantiate ``_mamba_impl`` during ``forward``.
        That module did not share parameters with this S6 implementation, but
        PyTorch nevertheless serialized it below each ``SelectiveSSM``. The
        registered S6 weights are also present and remain the authoritative
        state; accepting only this exact obsolete prefix lets such checkpoints
        load strictly without weakening any other architecture check.
        """
        obsolete_prefix = f"{prefix}_mamba_impl."
        obsolete = [key for key in state_dict if key.startswith(obsolete_prefix)]
        for key in obsolete:
            state_dict.pop(key)
        if obsolete:
            warnings.warn(
                "Ignoring obsolete lazy mamba_ssm checkpoint state under "
                f"{obsolete_prefix!r}; the eagerly registered SelectiveSSM "
                "weights are authoritative.",
                RuntimeWarning,
                stacklevel=2,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper around a :class:`SelectiveSSM` layer."""

    def __init__(self, d_model: int, **ssm_kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, **ssm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# Per-variable temporal corrector
# ---------------------------------------------------------------------------


class _VarTemporalHead(nn.Module):
    """Encode→Mamba(time)→decode temporal corrector for one variable.

    Operates on a sequence of single-channel normalised frames of shape
    ``(Bf, S, H, W)`` where ``Bf`` is the effective batch (for atmospheric
    variables the pressure-level axis is folded into ``Bf``). The Mamba layers
    run over the ``S`` (time / rollout-step) axis independently at every pixel,
    so the temporal dynamics are shared across space and levels but evolve from
    each pixel's own history.
    """

    def __init__(
        self,
        channels: int = 16,
        d_state: int = 8,
        n_layers: int = 2,
        d_conv: int = 3,
        expand: int = 2,
        lon_periodic: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.lon_periodic = bool(lon_periodic)
        # Spatial encoder: 1 frame channel -> C features (3x3 context). Longitude
        # is padded circularly on a periodic (global) domain so the temporal
        # correction does not inherit a seam at the 0°/360° dateline.
        # GroupNorm requires ``channels % groups == 0``. Choose the largest
        # useful divisor rather than rejecting otherwise valid widths such as
        # 10 at construction time.
        groups = max(
            group for group in range(1, min(8, self.channels) + 1) if self.channels % group == 0
        )
        self.encoder = nn.Sequential(
            PeriodicConv2d(1, self.channels, 3, lon_periodic=lon_periodic),
            nn.GroupNorm(groups, self.channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            MambaBlock(
                self.channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(max(1, int(n_layers)))
        )
        # Decoder: C features -> 1-channel correction. Zero-init => identity.
        self.decoder = nn.Conv2d(self.channels, 1, 1)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Args: seq (Bf, S, H, W). Returns correction (Bf, S, H, W)."""
        Bf, S, H, W = seq.shape
        frames = seq.reshape(Bf * S, 1, H, W)
        feat = self.encoder(frames)  # (Bf*S, C, H, W)
        C = feat.shape[1]
        # (Bf*S,C,H,W) -> (Bf,H,W,S,C) -> (Bf*H*W, S, C): per-pixel time series.
        feat = feat.reshape(Bf, S, C, H, W).permute(0, 3, 4, 1, 2).reshape(Bf * H * W, S, C)
        for blk in self.blocks:
            feat = blk(feat)
        # back to (Bf*S, C, H, W)
        feat = feat.reshape(Bf, H, W, S, C).permute(0, 3, 4, 1, 2).reshape(Bf * S, C, H, W)
        corr = self.decoder(feat).reshape(Bf, S, H, W)
        return corr


class MambaTemporalModule(nn.Module):
    """Per-variable Mamba temporal correctors for spatial refinement wrappers.

    Holds one :class:`_VarTemporalHead` per refined surface and atmospheric
    variable. The public API mirrors the flow head: callers pass a sequence of
    spatially refined *normalised* frames and receive a temporal correction (or
    the corrected sequence) in the same normalised space.

    Shapes:
      * surf  sequence: ``(B, S, H, W)``
      * atmos sequence: ``(B, S, L, H, W)`` (levels folded into batch internally)
    """

    def __init__(
        self,
        surf_vars: Sequence[str] = (),
        atmos_vars: Sequence[str] = (),
        channels: int = 16,
        d_state: int = 8,
        n_layers: int = 2,
        d_conv: int = 3,
        expand: int = 2,
        lon_periodic: bool = True,
    ) -> None:
        super().__init__()
        self.lon_periodic = bool(lon_periodic)
        head_kwargs = dict(
            channels=channels,
            d_state=d_state,
            n_layers=n_layers,
            d_conv=d_conv,
            expand=expand,
            lon_periodic=lon_periodic,
        )
        self.surf_heads = nn.ModuleDict({n: _VarTemporalHead(**head_kwargs) for n in surf_vars})
        self.atmos_heads = nn.ModuleDict({n: _VarTemporalHead(**head_kwargs) for n in atmos_vars})

    def has_var(self, var_name: str, kind: str) -> bool:
        heads = self.surf_heads if kind == "surf" else self.atmos_heads
        return var_name in heads

    def temporal_residual(
        self,
        seq_norm: torch.Tensor,
        var_name: str,
        kind: str,
    ) -> torch.Tensor:
        """Temporal correction for a normalised spatially refined sequence.

        Args:
          seq_norm: ``(B, S, H, W)`` (surf) or ``(B, S, L, H, W)`` (atmos).
          var_name: aurora variable name (selects the per-variable head).
          kind: ``"surf"`` or ``"atmos"``.

        Returns:
          Correction tensor, same shape as ``seq_norm``. Zeros (no-op) when the
          variable has no head.
        """
        heads = self.surf_heads if kind == "surf" else self.atmos_heads
        if var_name not in heads:
            return torch.zeros_like(seq_norm)
        head = heads[var_name]

        if kind == "atmos":
            assert seq_norm.dim() == 5, f"atmos seq must be (B,S,L,H,W), got {seq_norm.shape}"
            B, S, L, H, W = seq_norm.shape
            # Fold levels into the effective batch: (B,S,L,H,W) -> (B*L,S,H,W).
            folded = seq_norm.permute(0, 2, 1, 3, 4).reshape(B * L, S, H, W)
            corr = head(folded)
            return corr.reshape(B, L, S, H, W).permute(0, 2, 1, 3, 4)
        else:
            assert seq_norm.dim() == 4, f"surf seq must be (B,S,H,W), got {seq_norm.shape}"
            return head(seq_norm)


class PackedMambaTemporalAdapter(nn.Module):
    """Apply :class:`MambaTemporalModule` to a canonical packed field sequence.

    The unified refinement stack represents targets as ``[B, C, H, W]`` using
    :class:`~finetune.refinement.packing.FieldPacking`, whereas the legacy
    temporal module operates per variable. This adapter is the single mapping
    between those contracts. It preserves surface-first channel ordering and
    the configured atmospheric ``loss_levels`` exactly.
    """

    def __init__(
        self,
        packing: FieldPacking,
        *,
        channels: int = 16,
        d_state: int = 8,
        n_layers: int = 2,
        d_conv: int = 3,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.packing = packing
        surf_vars = list(
            dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "surf")
        )
        atmos_vars = list(
            dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "atmos")
        )
        self.core = MambaTemporalModule(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            channels=channels,
            d_state=d_state,
            n_layers=n_layers,
            d_conv=d_conv,
            expand=expand,
            lon_periodic=packing.lon_periodic,
        )

    def _validate_sequence(self, sequence: torch.Tensor) -> tuple[int, int, int, int, int]:
        if sequence.ndim != 5:
            raise ValueError(
                "Packed temporal sequence must have shape [batch, lead, channel, "
                f"latitude, longitude], got {tuple(sequence.shape)}."
            )
        batch, steps, channels, height, width = sequence.shape
        if steps < 1:
            raise ValueError("Packed temporal sequence lead dimension must be >= 1, got 0.")
        if channels != self.packing.num_channels:
            raise ValueError(
                "Packed temporal sequence channel dimension mismatch: expected "
                f"FieldPacking.num_channels={self.packing.num_channels}, got {channels}."
            )
        return batch, steps, channels, height, width

    def temporal_residual(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return a correction for ``[B, S, C, H, W]`` normalized fields."""
        batch, steps, channels, height, width = self._validate_sequence(sequence)
        flat = sequence.reshape(batch * steps, channels, height, width)
        fields = self.packing.unpack(flat)
        corrections: dict[str, torch.Tensor] = {}
        for name in self.packing.variables:
            channel_specs = self.packing.channels_for(name)
            kind = channel_specs[0].kind
            values = fields[name]
            if kind == "surf":
                variable_sequence = values.reshape(batch, steps, height, width)
            else:
                levels = len(channel_specs)
                variable_sequence = values.reshape(batch, steps, levels, height, width)
            correction = self.core.temporal_residual(variable_sequence, name, kind)
            corrections[name] = correction.reshape(batch * steps, *correction.shape[2:])
        packed = self.packing.pack(corrections)
        return packed.reshape(batch, steps, channels, height, width)

    def corrected_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return the causally corrected normalized sequence."""
        clean = torch.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        corrected = clean + self.temporal_residual(clean)
        # Missing cells stay missing for downstream masks/serialization rather
        # than being silently replaced by the normalization mean.
        return torch.where(torch.isfinite(sequence), corrected, sequence)

    def correct_causal(self, history: torch.Tensor) -> torch.Tensor:
        """Correct and return only the latest frame from a causal history."""
        return self.corrected_sequence(history)[:, -1]

    def causal_residual(self, history: torch.Tensor) -> torch.Tensor:
        """Return only the temporal correction for the latest history frame.

        Keeping the correction separate lets residual refiners add it directly
        to their sampled residual. In particular, a zero-initialized Mamba then
        preserves the spatial refiner output without an avoidable
        ``(rollout + residual) - rollout`` round trip.
        """
        self._validate_sequence(history)
        clean = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
        correction = self.temporal_residual(clean)[:, -1]
        latest_is_finite = torch.isfinite(history[:, -1])
        return torch.where(latest_is_finite, correction, torch.zeros_like(correction))
