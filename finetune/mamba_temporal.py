"""Mamba-based temporal module for Aurora long-lead bias correction.

The flow-matching refine head (:mod:`finetune.flow_refine`) corrects the
*spatial* structure and per-step bias of Aurora's prediction, but it treats
each rollout step independently — it has no memory of how the atmospheric
state (and Aurora's error in it) evolves through time. Empirically this is
exactly where long rollouts degrade: a 6 h correction is easy, but by 2–3 days
the accumulated, temporally-correlated drift is something a per-step spatial
model cannot see.

This module adds a small **temporal corrector** that runs *on top of* the
flow-matching output. For each refined variable it:

  1. encodes every (flow-corrected, normalised) frame of a short rollout
     sequence into a per-pixel feature vector,
  2. runs a selective state-space model (Mamba / S6) over the **time axis**
     independently at every pixel (parameters shared across space and, for
     atmospheric variables, across pressure levels),
  3. decodes the temporal state back to a single-channel *temporal correction*
     that is added to the flow-corrected field.

The decoder output projection is zero-initialised, so at construction the
temporal correction is identically zero: enabling the module changes nothing
until it is trained (``identity-at-init``), which keeps existing
flow-matching-only checkpoints numerically unchanged.

Selective-scan backend
----------------------
The official ``mamba_ssm`` CUDA kernels are used automatically when available
(GPU training). Everything also runs through a self-contained, pure-PyTorch
selective scan so the module trains and tests on CPU with no extra
dependencies. The two paths are mathematically equivalent (same parameterised
S6 recurrence); only the kernel differs.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.longitude import PeriodicConv2d

__all__ = ["MambaTemporalModule", "SelectiveSSM", "MambaBlock"]


# ---------------------------------------------------------------------------
# Selective state-space core (Mamba / S6)
# ---------------------------------------------------------------------------


def _selective_scan_ref(
    u: torch.Tensor,      # (B, S, D)  — gated input sequence
    delta: torch.Tensor,  # (B, S, D)  — per-step, per-channel timestep
    A: torch.Tensor,      # (D, N)     — diagonal state matrix (negative)
    B_mat: torch.Tensor,  # (B, S, N)  — input projection
    C_mat: torch.Tensor,  # (B, S, N)  — output projection
    D_skip: torch.Tensor, # (D,)       — skip connection
) -> torch.Tensor:
    """Pure-PyTorch selective scan (sequential over the time axis).

    Implements the discretised, input-dependent SSM recurrence

        h_s = exp(delta_s · A) · h_{s-1}  +  (delta_s · B_s) · u_s
        y_s = C_s · h_s  +  D · u_s

    with a diagonal ``A``. The scan is sequential in ``S`` (typically 2–6
    rollout steps), so the Python loop is cheap; all other axes are vectorised.

    Returns ``y`` of shape (B, S, D).
    """
    Bsz, S, Dn = u.shape
    N = A.shape[1]

    # Discretise: dA (B,S,D,N) = exp(delta · A); dB·u (B,S,D,N).
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,S,D,N)
    dBu = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * u.unsqueeze(-1)   # (B,S,D,N)

    h = torch.zeros(Bsz, Dn, N, device=u.device, dtype=u.dtype)
    ys = []
    for s in range(S):
        h = dA[:, s] * h + dBu[:, s]            # (B,D,N)
        y_s = torch.einsum("bdn,bn->bd", h, C_mat[:, s])  # (B,D)
        ys.append(y_s)
    y = torch.stack(ys, dim=1)                  # (B,S,D)
    y = y + u * D_skip.view(1, 1, -1)
    return y


class SelectiveSSM(nn.Module):
    """A single Mamba selective state-space layer over a (B, S, d_model) seq.

    Faithful to the reference S6 block: input expansion, a causal depthwise
    temporal conv, input-dependent (Δ, B, C), diagonal state matrix ``A`` and a
    SiLU gate. Uses the ``mamba_ssm`` CUDA kernel when installed and running on
    GPU; otherwise falls back to :func:`_selective_scan_ref`.
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
            self.d_inner, self.d_inner, kernel_size=self.d_conv,
            groups=self.d_inner, padding=self.d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is parameterised in log-space and kept negative: A = -exp(A_log).
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Optional fast path (GPU only).
        self._mamba_impl = None

    def _maybe_build_fast(self, device: torch.device) -> None:
        if self._mamba_impl is not None or device.type != "cuda":
            return
        try:  # pragma: no cover - exercised only on GPU boxes with mamba_ssm
            from mamba_ssm import Mamba  # type: ignore

            m = Mamba(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.d_inner // self.d_model,
            ).to(device)
            self._mamba_impl = m
        except Exception:
            self._mamba_impl = False  # mark "unavailable" so we don't retry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, S, d_model). Returns (B, S, d_model)."""
        self._maybe_build_fast(x.device)
        if self._mamba_impl not in (None, False):  # pragma: no cover
            return self._mamba_impl(x)

        B, S, _ = x.shape
        xz = self.in_proj(x)                       # (B,S,2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)              # each (B,S,d_inner)

        # Causal depthwise temporal conv (truncate the right padding to S).
        xc = x_in.transpose(1, 2)                  # (B,d_inner,S)
        xc = self.conv1d(xc)[..., :S]
        x_in = F.silu(xc.transpose(1, 2))          # (B,S,d_inner)

        params = self.x_proj(x_in)                 # (B,S,dt_rank+2*d_state)
        dt, B_mat, C_mat = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt))       # (B,S,d_inner)
        A = -torch.exp(self.A_log.float())         # (d_inner,d_state)

        y = _selective_scan_ref(x_in, delta, A, B_mat, C_mat, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


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
        self.encoder = nn.Sequential(
            PeriodicConv2d(1, self.channels, 3, lon_periodic=lon_periodic),
            nn.GroupNorm(min(8, self.channels), self.channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            MambaBlock(
                self.channels, d_state=d_state, d_conv=d_conv, expand=expand,
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
        feat = self.encoder(frames)                     # (Bf*S, C, H, W)
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
    """Per-variable Mamba temporal correctors for the flow-refine wrapper.

    Holds one :class:`_VarTemporalHead` per refined surface and atmospheric
    variable. The public API mirrors the flow head: callers pass a sequence of
    flow-corrected *normalised* frames and receive a temporal correction (or
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
            channels=channels, d_state=d_state, n_layers=n_layers,
            d_conv=d_conv, expand=expand, lon_periodic=lon_periodic,
        )
        self.surf_heads = nn.ModuleDict(
            {n: _VarTemporalHead(**head_kwargs) for n in surf_vars}
        )
        self.atmos_heads = nn.ModuleDict(
            {n: _VarTemporalHead(**head_kwargs) for n in atmos_vars}
        )

    def has_var(self, var_name: str, kind: str) -> bool:
        heads = self.surf_heads if kind == "surf" else self.atmos_heads
        return var_name in heads

    def temporal_residual(
        self, seq_norm: torch.Tensor, var_name: str, kind: str,
    ) -> torch.Tensor:
        """Temporal correction for a normalised flow-corrected sequence.

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
