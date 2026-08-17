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
    "PackedJointMambaTemporalHead",
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
    valid_steps: torch.Tensor | None = None,  # (B, S) — recurrence updates
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
    if valid_steps is not None:
        if valid_steps.shape != (Bsz, S):
            raise ValueError(
                "valid_steps must have shape [batch, lead] matching the "
                f"selective scan; expected {(Bsz, S)}, got "
                f"{tuple(valid_steps.shape)}."
            )
        if valid_steps.device != u.device:
            raise ValueError(
                "valid_steps must be on the same device as the sequence."
            )
    N = A.shape[1]

    # Discretise: dA (B,S,D,N) = exp(delta · A); dB·u (B,S,D,N).
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,S,D,N)
    dBu = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * u.unsqueeze(-1)  # (B,S,D,N)

    h = torch.zeros(Bsz, Dn, N, device=u.device, dtype=u.dtype)
    ys = []
    for s in range(S):
        candidate = dA[:, s] * h + dBu[:, s]  # (B,D,N)
        if valid_steps is None:
            h = candidate
        else:
            h = torch.where(valid_steps[:, s, None, None], candidate, h)
        y_s = torch.einsum("bdn,bn->bd", h, C_mat[:, s])  # (B,D)
        if valid_steps is not None:
            y_s = torch.where(valid_steps[:, s, None], y_s, torch.zeros_like(y_s))
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a causal scan over [batch, lead, feature] tokens."""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                "SelectiveSSM expects [batch, lead, feature] with feature="
                f"{self.d_model}, got {tuple(x.shape)}."
            )
        batch, steps, _ = x.shape
        if valid_steps is not None:
            if valid_steps.shape != (batch, steps):
                raise ValueError(
                    "valid_steps must match the SelectiveSSM batch/lead axes; "
                    f"expected {(batch, steps)}, got {tuple(valid_steps.shape)}."
                )
            if valid_steps.device != x.device:
                raise ValueError(
                    "valid_steps must be on the same device as the sequence."
                )
            valid_steps = valid_steps.to(dtype=torch.bool)
            x = torch.where(valid_steps.unsqueeze(-1), x, torch.zeros_like(x))

        xz = self.in_proj(x)  # (B,S,2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B,S,d_inner)

        # Causal depthwise temporal conv (truncate the right padding to S).
        xc = x_in.transpose(1, 2)  # (B,d_inner,S)
        xc = self.conv1d(xc)[..., :steps]
        x_in = F.silu(xc.transpose(1, 2))  # (B,S,d_inner)

        params = self.x_proj(x_in)  # (B,S,dt_rank+2*d_state)
        dt, B_mat, C_mat = torch.split(
            params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt))  # (B,S,d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner,d_state)

        y = _selective_scan_ref(
            x_in,
            delta,
            A,
            B_mat,
            C_mat,
            self.D,
            valid_steps=valid_steps,
        )
        y = y * F.silu(z)
        output = self.out_proj(y)
        if valid_steps is not None:
            output = torch.where(
                valid_steps.unsqueeze(-1), output, torch.zeros_like(output)
            )
        return output

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x + self.ssm(self.norm(x), valid_steps=valid_steps)


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


class PackedJointMambaTemporalHead(nn.Module):
    """Joint packed-channel temporal corrector with conservative gated fusion.

    Every frame is spatially encoded with all target channels and configured
    level fields present. Mamba then runs over the lead axis independently at
    each grid cell. A channel-wise ReZero gate is initialized at (usually) zero,
    while the decoder remains nonzero, so the first forward is an exact identity
    and the first backward can immediately train the fusion strength.
    """

    def __init__(
        self,
        packed_channels: int,
        *,
        channels: int = 16,
        d_state: int = 8,
        n_layers: int = 2,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        lon_periodic: bool = True,
        gated_fusion: bool = True,
        gate_init: float = 0.0,
        lead_time_conditioning: bool = True,
        mask_conditioning: bool = True,
        coordinate_conditioning: bool = False,
        latitude: Sequence[float] = (),
        longitude: Sequence[float] = (),
        lead_time_scale_hours: float = 72.0,
    ) -> None:
        super().__init__()
        self.packed_channels = int(packed_channels)
        self.channels = int(channels)
        self.gated_fusion = bool(gated_fusion)
        self.dropout = nn.Dropout(float(dropout))
        self.lead_time_conditioning = bool(lead_time_conditioning)
        self.mask_conditioning = bool(mask_conditioning)
        self.coordinate_conditioning = bool(coordinate_conditioning)
        self.lead_time_scale_hours = float(lead_time_scale_hours)
        input_channels = self.packed_channels
        if self.mask_conditioning:
            input_channels += self.packed_channels
        if self.lead_time_conditioning:
            # Absolute physical lead and the spacing since the previous lead.
            input_channels += 2
        if self.coordinate_conditioning:
            if not latitude or not longitude:
                raise ValueError(
                    "coordinate-conditioned packed temporal refinement requires "
                    "non-empty FieldPacking latitude and longitude coordinates."
                )
            latitude_tensor = torch.as_tensor(latitude, dtype=torch.float32)
            longitude_tensor = torch.as_tensor(longitude, dtype=torch.float32)
            if not bool(torch.isfinite(latitude_tensor).all()) or not bool(
                torch.isfinite(longitude_tensor).all()
            ):
                raise ValueError(
                    "Packed temporal latitude/longitude coordinates must be finite."
                )
            if bool((latitude_tensor.abs() > 90.0).any()):
                raise ValueError(
                    "Packed temporal latitude coordinates must lie in [-90, 90]."
                )
            latitude_grid = (latitude_tensor / 90.0)[:, None].expand(
                latitude_tensor.numel(), longitude_tensor.numel()
            )
            longitude_radians = torch.deg2rad(longitude_tensor)
            sin_longitude = torch.sin(longitude_radians)[None, :].expand_as(
                latitude_grid
            )
            cos_longitude = torch.cos(longitude_radians)[None, :].expand_as(
                latitude_grid
            )
            coordinate_features = torch.stack(
                (latitude_grid, sin_longitude, cos_longitude), dim=0
            )
            input_channels += 3
        else:
            coordinate_features = torch.empty(0, dtype=torch.float32)
        self.register_buffer(
            "coordinate_features", coordinate_features, persistent=False
        )
        groups = max(
            group
            for group in range(1, min(8, self.channels) + 1)
            if self.channels % group == 0
        )
        self.encoder = nn.Sequential(
            PeriodicConv2d(
                input_channels,
                self.channels,
                3,
                lon_periodic=lon_periodic,
            ),
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
        # Keep this projection nonzero: with an exact-zero fusion gate the
        # first backward trains the gate, and subsequent steps reach the core.
        self.decoder = nn.Conv2d(self.channels, self.packed_channels, 1)
        if self.gated_fusion:
            self.fusion_gate = nn.Parameter(
                torch.full((self.packed_channels,), float(gate_init))
            )
        else:
            self.register_parameter("fusion_gate", None)
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    @property
    def fusion_strength(self) -> torch.Tensor:
        """Bounded channel-wise correction strength used for deployment."""
        if self.fusion_gate is None:
            return torch.ones_like(self.decoder.bias)
        return torch.tanh(self.fusion_gate)

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        lead_hours: torch.Tensor | None,
        valid_cell_mask: torch.Tensor,
        valid_step_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a joint correction for validated [B,S,C,H,W] fields."""
        batch, steps, channels, height, width = sequence.shape
        clean = torch.where(
            valid_cell_mask,
            sequence,
            torch.zeros_like(sequence),
        )
        encoder_parts = [clean]
        if self.mask_conditioning:
            encoder_parts.append(valid_cell_mask.to(dtype=sequence.dtype))
        if self.lead_time_conditioning:
            if lead_hours is None:
                raise ValueError(
                    "packed_joint temporal lead conditioning requires lead_hours."
                )
            spacing = torch.empty_like(lead_hours)
            spacing[:, 0] = lead_hours[:, 0]
            if steps > 1:
                spacing[:, 1:] = lead_hours[:, 1:] - lead_hours[:, :-1]
            scale = self.lead_time_scale_hours
            lead_features = torch.stack(
                (lead_hours / scale, spacing / scale),
                dim=2,
            )
            lead_features = lead_features.to(dtype=sequence.dtype)
            lead_features = lead_features[:, :, :, None, None].expand(
                batch, steps, 2, height, width
            )
            encoder_parts.append(lead_features)

        if self.coordinate_conditioning:
            if tuple(self.coordinate_features.shape[1:]) != (height, width):
                raise ValueError(
                    "Packed temporal coordinate shape mismatch: FieldPacking "
                    f"coordinates describe {tuple(self.coordinate_features.shape[1:])}, "
                    f"but sequence uses {(height, width)}."
                )
            coordinates = self.coordinate_features.to(
                device=sequence.device, dtype=sequence.dtype
            ).view(1, 1, 3, height, width)
            encoder_parts.append(
                coordinates.expand(batch, steps, 3, height, width)
            )

        frames = torch.cat(encoder_parts, dim=2).reshape(
            batch * steps, -1, height, width
        )
        features = self.encoder(frames)
        latent = features.shape[1]
        tokens = (
            self.dropout(features).reshape(batch, steps, latent, height, width)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch * height * width, steps, latent)
        )
        pixel_valid = (
            valid_cell_mask.any(dim=2)
            & valid_step_mask[:, :, None, None]
        )
        token_valid = (
            pixel_valid.permute(0, 2, 3, 1)
            .reshape(batch * height * width, steps)
        )
        for block in self.blocks:
            tokens = block(tokens, valid_steps=token_valid)
        features = (
            tokens.reshape(batch, height, width, steps, latent)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch * steps, latent, height, width)
        )
        raw = self.decoder(features).reshape(
            batch, steps, channels, height, width
        )
        correction = raw * self.fusion_strength.view(1, 1, channels, 1, 1)
        output_valid = (
            valid_cell_mask
            & valid_step_mask[:, :, None, None, None]
        )
        return torch.where(
            output_valid,
            correction,
            torch.zeros_like(correction),
        )


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
        dropout: float = 0.0,
        mode: str = "per_variable",
        gated_fusion: bool = False,
        gate_init: float = 0.0,
        lead_time_conditioning: bool = False,
        mask_conditioning: bool = False,
        coordinate_conditioning: bool = False,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.packing = packing
        self.mode = str(mode).strip().lower()
        self.lead_time_conditioning = bool(lead_time_conditioning)
        self.mask_conditioning = bool(mask_conditioning)
        self.coordinate_conditioning = bool(coordinate_conditioning)
        self.causal = bool(causal)
        if self.mode not in {"per_variable", "packed_joint"}:
            raise ValueError(
                "PackedMambaTemporalAdapter mode must be 'per_variable' or "
                f"'packed_joint', got {mode!r}."
            )
        if not self.causal:
            raise ValueError(
                "Non-causal temporal refinement is unsupported for rollout inference."
            )
        if self.mode == "per_variable":
            surf_vars = list(
                dict.fromkeys(
                    spec.aurora_name
                    for spec in packing.channels
                    if spec.kind == "surf"
                )
            )
            atmos_vars = list(
                dict.fromkeys(
                    spec.aurora_name
                    for spec in packing.channels
                    if spec.kind == "atmos"
                )
            )
            self.core: MambaTemporalModule | PackedJointMambaTemporalHead = (
                MambaTemporalModule(
                    surf_vars=surf_vars,
                    atmos_vars=atmos_vars,
                    channels=channels,
                    d_state=d_state,
                    n_layers=n_layers,
                    d_conv=d_conv,
                    expand=expand,
                    lon_periodic=packing.lon_periodic,
                )
            )
        else:
            self.core = PackedJointMambaTemporalHead(
                packing.num_channels,
                channels=channels,
                d_state=d_state,
                n_layers=n_layers,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                lon_periodic=packing.lon_periodic,
                gated_fusion=gated_fusion,
                gate_init=gate_init,
                lead_time_conditioning=self.lead_time_conditioning,
                mask_conditioning=self.mask_conditioning,
                coordinate_conditioning=self.coordinate_conditioning,
                latitude=packing.lat,
                longitude=packing.lon,
                lead_time_scale_hours=packing.lead_time_scale_hours,
            )

    def _validate_sequence(
        self,
        sequence: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if sequence.ndim != 5:
            raise ValueError(
                "Packed temporal sequence must have shape [batch, lead, channel, "
                f"latitude, longitude], got {tuple(sequence.shape)}."
            )
        if not sequence.is_floating_point():
            raise ValueError(
                "Packed temporal sequence must use a floating dtype, "
                f"got {sequence.dtype}."
            )
        batch, steps, channels, height, width = sequence.shape
        if batch < 1 or steps < 1 or height < 1 or width < 1:
            raise ValueError(
                "Packed temporal sequence batch, lead, latitude and longitude "
                f"dimensions must be positive, got {tuple(sequence.shape)}."
            )
        if channels != self.packing.num_channels:
            raise ValueError(
                "Packed temporal sequence channel dimension mismatch: expected "
                f"FieldPacking.num_channels={self.packing.num_channels}, got "
                f"{channels}."
            )
        return batch, steps, channels, height, width

    def _resolve_masks(
        self,
        sequence: torch.Tensor,
        *,
        valid_cell_mask: torch.Tensor | None,
        valid_step_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = self._validate_sequence(sequence)
        finite = torch.isfinite(sequence)
        if valid_cell_mask is None:
            cell_mask = finite
        else:
            if valid_cell_mask.device != sequence.device:
                raise ValueError(
                    "valid_cell_mask must be on the same device as sequence."
                )
            if valid_cell_mask.dtype != torch.bool:
                raise ValueError("valid_cell_mask must have boolean dtype.")
            allowed = {
                (batch, steps, channels, height, width),
                (batch, steps, 1, height, width),
            }
            if tuple(valid_cell_mask.shape) not in allowed:
                raise ValueError(
                    "valid_cell_mask must have shape [B,S,C,H,W] or "
                    f"[B,S,1,H,W]; expected one of {sorted(allowed)}, got "
                    f"{tuple(valid_cell_mask.shape)}."
                )
            cell_mask = valid_cell_mask.expand_as(sequence) & finite

        inferred_steps = cell_mask.flatten(2).any(dim=2)
        if valid_step_mask is None:
            step_mask = inferred_steps
        else:
            if valid_step_mask.shape != (batch, steps):
                raise ValueError(
                    "valid_step_mask must have shape [batch, lead]; expected "
                    f"{(batch, steps)}, got {tuple(valid_step_mask.shape)}."
                )
            if valid_step_mask.device != sequence.device:
                raise ValueError(
                    "valid_step_mask must be on the same device as sequence."
                )
            if valid_step_mask.dtype != torch.bool:
                raise ValueError("valid_step_mask must have boolean dtype.")
            step_mask = valid_step_mask & inferred_steps
        if not bool(step_mask.any(dim=1).all()):
            raise ValueError(
                "Every packed temporal sequence must contain at least one valid lead."
            )
        if steps > 1 and bool((~step_mask[:, :-1] & step_mask[:, 1:]).any()):
            raise ValueError(
                "valid_step_mask must be a contiguous valid prefix; a padded or "
                "missing step cannot be followed by a valid future step."
            )
        return cell_mask, step_mask

    def _resolve_lead_hours(
        self,
        lead_hours: torch.Tensor | None,
        *,
        batch: int,
        steps: int,
        device: torch.device,
        valid_step_mask: torch.Tensor,
        validate_order: bool,
    ) -> torch.Tensor | None:
        if lead_hours is None:
            if self.mode == "packed_joint" and self.lead_time_conditioning:
                raise ValueError(
                    "packed_joint temporal lead conditioning requires lead_hours "
                    "with shape [batch, lead]."
                )
            return None
        lead = torch.as_tensor(
            lead_hours,
            device=device,
            dtype=torch.float32,
        )
        if lead.ndim == 1 and lead.numel() == steps:
            lead = lead.view(1, steps).expand(batch, steps)
        elif lead.ndim == 1 and steps == 1 and lead.numel() == batch:
            lead = lead.view(batch, 1)
        elif lead.shape != (batch, steps):
            raise ValueError(
                "lead_hours must have shape [lead] or [batch, lead]; expected "
                f"{(steps,)} or {(batch, steps)}, got {tuple(lead.shape)}."
            )
        selected = lead[valid_step_mask]
        if not bool(torch.isfinite(selected).all()) or bool((selected <= 0).any()):
            raise ValueError(
                "Every valid temporal lead hour must be finite and positive."
            )
        if validate_order and steps > 1:
            adjacent = valid_step_mask[:, 1:] & valid_step_mask[:, :-1]
            differences = lead[:, 1:] - lead[:, :-1]
            if bool((differences[adjacent] <= 0).any()):
                raise ValueError(
                    "lead_hours must be strictly increasing over valid temporal "
                    "steps for causal rollout refinement."
                )
        return torch.where(valid_step_mask, lead, torch.zeros_like(lead))

    def temporal_residual(
        self,
        sequence: torch.Tensor,
        *,
        lead_hours: torch.Tensor | None = None,
        valid_cell_mask: torch.Tensor | None = None,
        valid_step_mask: torch.Tensor | None = None,
        validate_lead_order: bool = True,
    ) -> torch.Tensor:
        """Return a correction for canonical [B,S,C,H,W] normalized fields."""
        batch, steps, channels, height, width = self._validate_sequence(sequence)
        cell_mask, step_mask = self._resolve_masks(
            sequence,
            valid_cell_mask=valid_cell_mask,
            valid_step_mask=valid_step_mask,
        )
        leads = self._resolve_lead_hours(
            lead_hours,
            batch=batch,
            steps=steps,
            device=sequence.device,
            valid_step_mask=step_mask,
            validate_order=validate_lead_order,
        )
        clean = torch.where(cell_mask, sequence, torch.zeros_like(sequence))
        if self.mode == "packed_joint":
            assert isinstance(self.core, PackedJointMambaTemporalHead)
            return self.core(
                clean,
                lead_hours=leads,
                valid_cell_mask=cell_mask,
                valid_step_mask=step_mask,
            )

        assert isinstance(self.core, MambaTemporalModule)
        flat = clean.reshape(batch * steps, channels, height, width)
        fields = self.packing.unpack(flat)
        corrections: dict[str, torch.Tensor] = {}
        for name in self.packing.variables:
            channel_specs = self.packing.channels_for(name)
            kind = channel_specs[0].kind
            values = fields[name]
            if kind == "surf":
                variable_sequence = values.reshape(
                    batch, steps, height, width
                )
            else:
                levels = len(channel_specs)
                variable_sequence = values.reshape(
                    batch, steps, levels, height, width
                )
            correction = self.core.temporal_residual(
                variable_sequence,
                name,
                kind,
            )
            corrections[name] = correction.reshape(
                batch * steps, *correction.shape[2:]
            )
        packed = self.packing.pack(corrections).reshape(
            batch, steps, channels, height, width
        )
        output_valid = cell_mask & step_mask[:, :, None, None, None]
        return torch.where(output_valid, packed, torch.zeros_like(packed))

    def corrected_sequence(
        self,
        sequence: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return the causally corrected normalized sequence."""
        correction = self.temporal_residual(sequence, **kwargs)
        clean = torch.nan_to_num(
            sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        corrected = clean + correction
        return torch.where(torch.isfinite(sequence), corrected, sequence)

    def correct_causal(
        self,
        history: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Correct and return only the latest frame from a causal history."""
        return self.corrected_sequence(history, **kwargs)[:, -1]

    def causal_residual(
        self,
        history: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return only the temporal correction for the latest causal frame."""
        self._validate_sequence(history)
        correction = self.temporal_residual(history, **kwargs)[:, -1]
        latest_is_finite = torch.isfinite(history[:, -1])
        return torch.where(
            latest_is_finite,
            correction,
            torch.zeros_like(correction),
        )
