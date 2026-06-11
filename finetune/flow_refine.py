"""Flow-matching residual decoder for Aurora bias correction.

This module adds a small **conditional residual regressor** on top of a
frozen Aurora backbone.  The decoder learns

    p(r | ŷ, x)  where  r = y_true - ŷ

in the per-variable normalised space (so the per-level NO2 std-floor
problem disappears: the loss is a unit-normal-scale MSE, no 1/std
amplification).

We use the **x₁ (data) parameterisation** of rectified flow, which is
mathematically equivalent to velocity-prediction but has two
practically important advantages:

* **Identity-at-init.** The output layer is zero-init so r̂ ≡ 0 at
  step 0; the wrapper output equals Aurora's prediction unchanged
  before any training happens. (With v-prediction, zero-init produces
  pure-noise residuals at eval, which is the wrong baseline.)
* **Single-step deterministic eval is principled.** At ``sampling_steps
  = 1`` the head reduces to a regression of ``E[r | ŷ]`` — exactly
  what a deterministic conv-refine learns — but trained at *all* noise
  levels, which acts as a strong stochastic regulariser. Multi-step
  refinement is a strict generalisation.

Why not DDPM:

* DDPM's variance-preserving schedule and SNR weighting are tuned for
  natural images; for unit-variance bias residuals the simpler
  rectified-flow / data-prediction objective is more stable on small
  data and converges in fewer epochs.

The wrapper ``AuroraFlowRefine`` plugs in the same way as
``AuroraConvRefine``:

    base = AuroraAirPollution(...)
    base.load_checkpoint(strict=False)
    model = AuroraFlowRefine(
        base,
        target_surf_vars=("tcno2",),
        target_atmos_vars=("no2",),
        hidden=64,
        sampling_steps=1,   # 1 = deterministic regression; >1 = MCMC-style
    )
    model.set_norm_stats(norm_stats)  # required for de-norm at inference

Training:

* ``model.train()`` then ``forward(batch)`` returns Aurora's prediction
  unchanged (the FM head sees its conditioning via a separate code path).
* The supervised loss in :mod:`finetune.aurora_finetune_utils` detects
  ``AuroraFlowRefine`` and replaces the standard MSE with
  :meth:`AuroraFlowRefine.flow_loss` (residual-MSE at sampled noise level).

Inference / validation:

* ``model.eval()`` then ``forward(batch)`` runs Aurora once, then for
  each target variable iteratively refines a residual estimate via the
  noise-conditioned regressor, and returns ``ŷ + r̂`` in physical
  space.

For atmospheric variables, each pressure level is processed
independently by the same shared UNet (level dim is collapsed into the
batch axis).  Cross-level coupling can be added later by switching the
UNet to consume L levels as channels; this v1 keeps the parameter
count small.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.batch import Batch


# ---------------------------------------------------------------------------
# Structural auxiliary-loss configuration
# ---------------------------------------------------------------------------

# Default weights/parameters for the structural auxiliary losses layered on
# top of the rectified-flow residual MSE. All weights default to 0.0 so that,
# unless a config opts in, ``flow_loss`` is exactly the original residual-MSE
# objective (backward compatible). The ``training.flow_aux_loss`` config block
# overrides these via :meth:`AuroraFlowRefine.set_aux_loss_config`.
DEFAULT_AUX_LOSS_CONFIG: dict = {
    # Master on/off switch. When False, every structural term (including
    # coherence) is skipped and ``flow_loss`` reduces to the pure residual MSE,
    # regardless of the individual weights below. Lets a YAML toggle the whole
    # feature with a single line.
    "enabled": True,
    # Emphasise tail / extreme-event errors. ``extreme_quantile`` is the
    # per-sample |z|-quantile above which a cell is "extreme"; cells above it
    # get ``extreme_weight``× the squared error. ``peak`` matches the spatial
    # max/min magnitude so high-ozone plumes are reproduced.
    "extreme_weight": 0.0,
    "extreme_quantile": 0.95,
    "extreme_intensity": 4.0,
    "peak_weight": 0.0,
    # Spatial-pattern agreement: gradient-field MSE + anomaly correlation.
    "spatial_grad_weight": 0.0,
    "spatial_acc_weight": 0.0,
    # Distributional errors: variance matching + sorted-value Wasserstein.
    "dist_var_weight": 0.0,
    "dist_wasserstein_weight": 0.0,
    # Vertical-profile consistency (atmospheric vars only): level-to-level
    # finite-difference MSE.
    "vertical_weight": 0.0,
    # Column / profile coherence (cross-variable; applied in the supervised
    # loop, not inside flow_loss).
    "coherence_weight": 0.0,
    "coherence_column_var": "",
    "coherence_profile_var": "",
}


# ---------------------------------------------------------------------------
# Structural loss primitives
#
# All operate on normalised fields of shape (N, 1, H, W) — the same layout the
# flow head consumes — except :func:`_vertical_profile_loss` which takes
# (B, L, H, W). They are pure functions (no learnable state) so they can be
# unit-tested in isolation and reused by both the surf and atmos paths.
# ---------------------------------------------------------------------------


def _extreme_weighted_mse(
    refined: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
    intensity: float,
) -> torch.Tensor:
    """Squared error up-weighted on extreme (tail) cells.

    A cell is "extreme" when its normalised target magnitude exceeds the
    per-sample ``quantile`` of |target|. Extreme cells contribute
    ``intensity``× their squared error, so high-ozone plumes (the tails the
    plain MSE averages away) are corrected harder.
    """
    se = (refined - target) ** 2
    z = target.abs()
    N = z.shape[0]
    z_flat = z.reshape(N, -1)
    q = torch.clamp(torch.tensor(quantile, device=z.device, dtype=torch.float32), 0.0, 1.0)
    thr = torch.quantile(z_flat.float(), q, dim=1).to(z.dtype)
    thr = thr.view(N, 1, 1, 1)
    weight = torch.where(z >= thr, torch.as_tensor(intensity, dtype=se.dtype, device=se.device),
                         torch.ones((), dtype=se.dtype, device=se.device))
    return (weight * se).sum() / weight.sum().clamp_min(1.0)


def _peak_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match the per-sample spatial maximum and minimum magnitudes."""
    r_max = refined.amax(dim=(-1, -2))
    t_max = target.amax(dim=(-1, -2))
    r_min = refined.amin(dim=(-1, -2))
    t_min = target.amin(dim=(-1, -2))
    return 0.5 * (F.mse_loss(r_max, t_max) + F.mse_loss(r_min, t_min))


def _spatial_gradient_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Finite-difference spatial-gradient MSE (penalises pattern/edge errors)."""
    dx_r = refined[..., :, 1:] - refined[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_r = refined[..., 1:, :] - refined[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (F.mse_loss(dx_r, dx_t) + F.mse_loss(dy_r, dy_t))


def _anomaly_correlation_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − spatial anomaly correlation coefficient (ACC), averaged per sample.

    ACC is the standard meteorological spatial-pattern skill score: the
    Pearson correlation of the spatial anomalies (field minus its spatial
    mean). Maximising ACC forces the *shape* of the corrected field to match
    truth even where absolute magnitudes are already close.
    """
    N = refined.shape[0]
    r = refined.reshape(N, -1).float()
    t = target.reshape(N, -1).float()
    r = r - r.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (r * t).sum(dim=1)
    den = torch.sqrt((r * r).sum(dim=1) * (t * t).sum(dim=1)).clamp_min(1e-12)
    corr = (num / den).clamp(-1.0, 1.0)
    return (1.0 - corr).mean().to(refined.dtype)


def _variance_match_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match per-sample spatial standard deviation (distributional spread)."""
    sr = refined.reshape(refined.shape[0], -1).float().std(dim=1, unbiased=False)
    st = target.reshape(target.shape[0], -1).float().std(dim=1, unbiased=False)
    return F.mse_loss(sr, st).to(refined.dtype)


def _sorted_wasserstein_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared 1-D Wasserstein-2 distance via sorted values, per sample.

    Sorting the flattened spatial values and comparing them position-by-position
    is the closed-form 1-D optimal-transport cost. It penalises mismatch of the
    whole value *distribution* (CDF), independent of where errors sit spatially.
    """
    N = refined.shape[0]
    r = refined.reshape(N, -1).float().sort(dim=1).values
    t = target.reshape(N, -1).float().sort(dim=1).values
    return F.mse_loss(r, t).to(refined.dtype)


def _vertical_profile_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Level-to-level finite-difference MSE → vertical-profile consistency.

    Args:
      refined / target: (B, L, H, W) normalised fields.

    Matching the differences between adjacent pressure levels enforces a
    physically coherent vertical shape, which the per-level-independent head
    cannot otherwise guarantee.
    """
    dz_r = refined[:, 1:] - refined[:, :-1]
    dz_t = target[:, 1:] - target[:, :-1]
    return F.mse_loss(dz_r, dz_t)


def _pressure_thickness_weights(
    levels: Sequence[float], device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Normalised mass (Δp) weights for a vertical integral over ``levels``.

    Returns weights aligned to the *input* order of ``levels`` (which may be
    unsorted), summing to 1. Used to turn a profile into a column-like integral
    for the coherence loss.
    """
    p = torch.tensor([float(x) for x in levels], dtype=torch.float32)
    if p.numel() == 1:
        return torch.ones(1, device=device, dtype=dtype)
    order = torch.argsort(p)
    ps = p[order]
    # Layer thickness via centred differences on sorted pressures.
    thick_sorted = torch.gradient(ps)[0].abs()
    weights = torch.empty_like(p)
    weights[order] = thick_sorted
    weights = weights / weights.sum().clamp_min(1e-12)
    return weights.to(device=device, dtype=dtype)


def _sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding of t∈[0,1] → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim → pad
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _FiLM(nn.Module):
    """FiLM modulation: scale & shift conv features by a time embedding."""

    def __init__(self, time_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(time_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # h: (N, C, H, W), t_emb: (N, time_dim)
        scale_shift = self.proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + scale) + shift


class _SelfAttention2d(nn.Module):
    """Spatial self-attention block for the UNet bottleneck.

    Gives the model a global receptive field at the coarsest scale, which
    is essential for capturing large-scale systematic biases (e.g.
    continent-wide ozone over-prediction) that local conv operations miss.

    The output projection is zero-initialised so the block is an exact
    identity at construction — no change to initial predictions.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False,
            bias=True,
        )
        # Zero-init so this block starts as an identity transform.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(1, 2, 0).reshape(B, C, H, W)
        return x + h  # residual connection


class _ConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU → FiLM(t) → Conv → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film = _FiLM(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.film(h, t_emb)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ResidualFlowUNet(nn.Module):
    """Tiny conditional UNet predicting the rectified-flow velocity.

    Inputs (concatenated along the channel axis):
      * ``cond``  — Aurora's normalised prediction ŷ_norm   (1 ch)
      * ``x_t``   — interpolated noisy residual              (1 ch)
    Time ``t`` is fed via FiLM modulation in every conv block.

    The output (1 ch) is the predicted velocity ``v_θ(x_t, t, ŷ)``.

    The architecture is intentionally small (~250k params at hidden=64,
    3 levels) — the conditioning carries most of the structure, the UNet
    only learns *how the residual deviates from zero given ŷ*.
    """

    def __init__(self, hidden: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # 3-level UNet operating at full / 1/2 / 1/4 resolution.
        self.down1 = _ConvBlock(2, hidden, time_dim)
        self.down2 = _ConvBlock(hidden, hidden * 2, time_dim)
        self.down3 = _ConvBlock(hidden * 2, hidden * 4, time_dim)
        # Self-attention at the bottleneck: global context for large-scale
        # systematic biases (e.g. continent-wide ozone over-prediction).
        self.bottleneck_attn = _SelfAttention2d(hidden * 4)
        self.up2 = _ConvBlock(hidden * 4 + hidden * 2, hidden * 2, time_dim)
        self.up1 = _ConvBlock(hidden * 2 + hidden, hidden, time_dim)
        self.out = nn.Conv2d(hidden, 1, 1)
        # Zero-init final projection so v(x_t, t, ŷ) ≡ 0 at init.
        # Then x_1 = x_0 (pure noise) → r̂ = 0 → output = ŷ at init,
        # exactly like the conv-refine "identity-at-init" property.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the FM velocity field.

        Args:
          x_t:  (N, 1, H, W)  — interpolated noisy residual.
          t:    (N,)          — flow time in [0, 1].
          cond: (N, 1, H, W)  — Aurora's normalised prediction.

        Returns:
          (N, 1, H, W) velocity prediction.
        """
        t_emb = self.time_mlp(_sinusoidal_time_embed(t, self.time_dim))

        h0 = torch.cat([cond, x_t], dim=1)
        d1 = self.down1(h0, t_emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb)
        d3 = self.down3(F.avg_pool2d(d2, 2), t_emb)
        d3 = self.bottleneck_attn(d3)  # global context at coarsest scale

        u2 = F.interpolate(d3, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        return self.out(u1)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class AuroraFlowRefine(nn.Module):
    """Wraps an Aurora model with per-variable rectified-flow refine heads.

    Training:
      * ``forward(batch)`` simply returns the (frozen) Aurora prediction.
        The flow loss is computed externally via :meth:`flow_loss`, which
        the supervised-loss code in ``aurora_finetune_utils`` calls in
        place of the standard MSE.

    Eval / inference:
      * ``forward(batch)`` runs Aurora, then for each target variable
        integrates the FM ODE (Euler, ``sampling_steps`` steps) starting
        from a Gaussian noise sample, and returns ``ŷ + r̂`` in physical
        space.

    The wrapper requires the per-variable normalisation stats (mean/std
    in physical space) so it can convert between ŷ_norm (where the FM
    head operates) and the physical-space prediction expected by the
    Aurora ``Batch`` interface.  Stats are injected via
    :meth:`set_norm_stats` after construction (during training setup).
    """

    def __init__(
        self,
        base: nn.Module,
        target_surf_vars: tuple[str, ...] = (),
        target_atmos_vars: tuple[str, ...] = (),
        hidden: int = 64,
        time_dim: int = 128,
        sampling_steps: int = 8,
        sigma_min: float = 1e-3,
        atmos_loss_levels: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.target_surf_vars = tuple(target_surf_vars)
        self.target_atmos_vars = tuple(target_atmos_vars)
        self.hidden = hidden
        self.sampling_steps = int(sampling_steps)
        self.sigma_min = float(sigma_min)

        # Per-variable mapping from atmos var name → list of level indices
        # (into the full L-dim) that should receive bias correction. Levels
        # NOT in this list are passed through from the backbone unchanged.
        self._atmos_loss_level_indices: dict[str, list[int]] = (
            {k: list(v) for k, v in atmos_loss_levels.items()}
            if atmos_loss_levels is not None
            else {}
        )

        # Structural auxiliary-loss configuration (weights + params). Default
        # weights are all 0 → behaviour is identical to the pure residual-MSE
        # flow loss. Populate via :meth:`set_aux_loss_config` (driven by the
        # ``training.flow_aux_loss`` config block).
        self.aux_loss_cfg = dict(DEFAULT_AUX_LOSS_CONFIG)

        self.surf_flow = nn.ModuleDict(
            {n: ResidualFlowUNet(hidden, time_dim) for n in target_surf_vars}
        )
        self.atmos_flow = nn.ModuleDict(
            {n: ResidualFlowUNet(hidden, time_dim) for n in target_atmos_vars}
        )

        # Norm stats (physical-space mean/std per variable). For atmos the
        # std/mean are per-level tensors of shape (L,); for surf they're
        # scalars (or 1-tensors).  Set via :meth:`set_norm_stats`.
        self._norm_stats: dict[str, dict[str, torch.Tensor]] = {}

    # --- Delegate Aurora interface -----------------------------------------

    @property
    def patch_size(self) -> int:
        return self.base.patch_size

    @property
    def surf_stats(self) -> dict:
        return self.base.surf_stats

    def batch_transform_hook(self, batch: Batch) -> Batch:
        return self.base.batch_transform_hook(batch)

    def configure_activation_checkpointing(self) -> None:
        if hasattr(self.base, "configure_activation_checkpointing"):
            self.base.configure_activation_checkpointing()

    def load_checkpoint(self, *args, **kwargs) -> None:
        self.base.load_checkpoint(*args, **kwargs)

    def load_checkpoint_local(self, *args, **kwargs) -> None:
        if hasattr(self.base, "load_checkpoint_local"):
            self.base.load_checkpoint_local(*args, **kwargs)

    # --- Norm stats --------------------------------------------------------

    def set_norm_stats(
        self, stats: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        """Register physical-space mean/std for each refined variable."""
        self._norm_stats = {
            k: {kk: vv.detach().clone() for kk, vv in v.items()}
            for k, v in stats.items()
        }

    def set_aux_loss_config(self, cfg: Mapping[str, Any] | None) -> None:
        """Configure the structural auxiliary losses.

        ``cfg`` keys override :data:`DEFAULT_AUX_LOSS_CONFIG`. Unknown keys are
        ignored. Pass ``None`` / empty to keep defaults (all weights 0 → pure
        residual-MSE flow loss).
        """
        merged = dict(DEFAULT_AUX_LOSS_CONFIG)
        if cfg:
            for k, v in cfg.items():
                if k in merged:
                    merged[k] = v
        self.aux_loss_cfg = merged

    def _norm_for(
        self, var: str, ref: torch.Tensor, kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ns = self._norm_stats.get(var)
        if ns is None:
            mean = torch.zeros((), device=ref.device, dtype=ref.dtype)
            std = torch.ones((), device=ref.device, dtype=ref.dtype)
            return mean, std
        mean = ns["mean"].to(device=ref.device, dtype=ref.dtype)
        std = ns["std"].to(device=ref.device, dtype=ref.dtype)
        if kind == "atmos" and mean.numel() > 1:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        return mean, std

    # --- Flow-matching loss (called by supervised-loss code) ---------------

    def flow_loss(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        var_name: str,
        kind: str,
    ) -> torch.Tensor:
        """Conditional residual regression at a sampled noise level.

        We use the **x₁ (data) parameterisation** of rectified flow: the
        UNet predicts the clean residual r = target − pred directly, but
        from a noise-perturbed input ``x_t = (1-t)·x₀ + t·r`` with random
        ``t ∈ [0,1]`` and ``x₀ ∼ N(0,I)``. The loss is MSE(r̂, r).

        Why this parameterisation:
          * **Identity-at-init.** The output layer is zero-initialised so
            r̂ = 0 → fine-tune output = Aurora pred at init.
          * **Stable.** No 1/(1-t) division; loss is bounded for all t.
          * **Single-step deterministic eval.** At ``t = 1`` and ``x₀ = 0``
            the model is exactly E[r | ŷ] (ridge regression of the
            residual on the conditioning), the same target a deterministic
            conv-refine learns. Multi-step refinement is a strict
            generalisation and can be enabled later.
          * **Mathematically equivalent to v-pred.** The FM velocity
            v* = x₁ − x₀ is recoverable as r̂ − (x_t − t·r̂)/(1-t).

        Args:
          pred_norm:   Aurora prediction in normalised space, (B, [L,] H, W).
          target_norm: Ground-truth target in normalised space, same shape.
          var_name:    aurora-name of the variable (selects the UNet head).
          kind:        ``"surf"`` or ``"atmos"``.

        Returns:
          Scalar tensor — mean residual MSE.
        """
        heads = self.surf_flow if kind == "surf" else self.atmos_flow
        if var_name not in heads:
            return torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
        head = heads[var_name]

        if kind == "atmos":
            assert pred_norm.dim() == 4, f"atmos pred must be (B,L,H,W), got {pred_norm.shape}"
            B, L, H, W = pred_norm.shape
            cond = pred_norm.reshape(B * L, 1, H, W)
            r = (target_norm - pred_norm).reshape(B * L, 1, H, W)
        else:
            assert pred_norm.dim() == 3, f"surf pred must be (B,H,W), got {pred_norm.shape}"
            B, H, W = pred_norm.shape
            cond = pred_norm.reshape(B, 1, H, W)
            r = (target_norm - pred_norm).reshape(B, 1, H, W)

        cond = cond.detach()
        r = r.detach()

        N = cond.shape[0]
        device = cond.device
        dtype = cond.dtype

        x0 = torch.randn(N, 1, H, W, device=device, dtype=dtype)
        # Log-normal t sampling: concentrates training on informative
        # mid-noise levels (t ≈ 0.3–0.7) rather than near-pure-noise (t ≈ 0,
        # uninformative) or near-clean (t ≈ 1, tiny gradients). This
        # distribution (sigmoid of N(-0.5, 1.2)) has median ≈ 0.38 and is
        # empirically known to accelerate flow matching convergence.
        log_t = torch.randn(N, device=device, dtype=torch.float32) * 1.2 - 0.5
        t = torch.sigmoid(log_t).clamp(self.sigma_min, 1.0 - self.sigma_min)
        t_b = t.view(N, 1, 1, 1).to(dtype)

        x_t = (1.0 - t_b) * x0 + t_b * r
        r_pred = head(x_t, t, cond)
        base_loss = F.mse_loss(r_pred, r)

        # --- Structural auxiliary losses ------------------------------------
        # The head's x₁-prediction r_pred is its clean-residual estimate at the
        # sampled noise level. The corresponding bias-corrected field is
        #   refined = cond + r_pred,   target = cond + r
        # so the structural terms below (spatial pattern, extremes,
        # distribution, vertical shape) penalise the *structured* error
        # (r_pred − r) rather than only its per-pixel magnitude.
        aux = self._aux_structural_loss(
            refined_flat=cond + r_pred,
            target_flat=cond + r,
            kind=kind,
            group=(B, L) if kind == "atmos" else (B, 1),
        )
        return base_loss + aux

    # --- Structural auxiliary losses ---------------------------------------

    def _aux_structural_loss(
        self,
        refined_flat: torch.Tensor,
        target_flat: torch.Tensor,
        kind: str,
        group: tuple[int, int],
    ) -> torch.Tensor:
        """Weighted sum of the structural losses for one variable.

        Args:
          refined_flat: (N, 1, H, W) normalised refined field (cond + r̂).
          target_flat:  (N, 1, H, W) normalised target field (cond + r).
          kind: ``"surf"`` or ``"atmos"``.
          group: (B, C) such that ``N == B * C``. For atmos ``C`` is the number
            of pressure levels (used by the vertical term); for surf ``C == 1``.

        Returns:
          Scalar tensor (0 when all aux weights are 0).
        """
        cfg = self.aux_loss_cfg
        device = refined_flat.device
        dtype = refined_flat.dtype
        total = torch.zeros((), device=device, dtype=dtype)

        if not bool(cfg.get("enabled", True)):
            return total

        w_extreme = float(cfg.get("extreme_weight", 0.0))
        w_peak = float(cfg.get("peak_weight", 0.0))
        w_grad = float(cfg.get("spatial_grad_weight", 0.0))
        w_acc = float(cfg.get("spatial_acc_weight", 0.0))
        w_var = float(cfg.get("dist_var_weight", 0.0))
        w_wass = float(cfg.get("dist_wasserstein_weight", 0.0))
        w_vert = float(cfg.get("vertical_weight", 0.0))

        if w_extreme > 0.0:
            total = total + w_extreme * _extreme_weighted_mse(
                refined_flat, target_flat,
                quantile=float(cfg.get("extreme_quantile", 0.95)),
                intensity=float(cfg.get("extreme_intensity", 4.0)),
            )
        if w_peak > 0.0:
            total = total + w_peak * _peak_loss(refined_flat, target_flat)
        if w_grad > 0.0:
            total = total + w_grad * _spatial_gradient_loss(refined_flat, target_flat)
        if w_acc > 0.0:
            total = total + w_acc * _anomaly_correlation_loss(refined_flat, target_flat)
        if w_var > 0.0:
            total = total + w_var * _variance_match_loss(refined_flat, target_flat)
        if w_wass > 0.0:
            total = total + w_wass * _sorted_wasserstein_loss(refined_flat, target_flat)
        if w_vert > 0.0 and kind == "atmos":
            B, C = group
            if C > 1:
                _, _, H, W = refined_flat.shape
                refined_lc = refined_flat.reshape(B, C, H, W)
                target_lc = target_flat.reshape(B, C, H, W)
                total = total + w_vert * _vertical_profile_loss(refined_lc, target_lc)

        return total

    # --- Column / profile coherence ----------------------------------------

    def refine_norm_deterministic(
        self, pred_norm: torch.Tensor, var_name: str, kind: str,
    ) -> torch.Tensor:
        """Deterministic refined estimate E[r|ŷ] added to ``pred_norm``.

        Runs the head once at the clean limit (``x_t = 0``, ``t = 1``) — the
        same single-step regression :meth:`_sample_residual` uses — but **with
        gradients enabled** so it can drive the coherence loss. Returns the
        refined field in normalised space, same shape as ``pred_norm``.
        """
        heads = self.surf_flow if kind == "surf" else self.atmos_flow
        if var_name not in heads:
            return pred_norm
        head = heads[var_name]

        if kind == "atmos":
            B, L, H, W = pred_norm.shape
            cond = pred_norm.reshape(B * L, 1, H, W)
        else:
            B, H, W = pred_norm.shape
            cond = pred_norm.reshape(B, 1, H, W)

        N = cond.shape[0]
        x_t = torch.zeros(N, 1, H, W, device=cond.device, dtype=cond.dtype)
        t = torch.ones(N, device=cond.device, dtype=torch.float32)
        r_hat = head(x_t, t, cond)
        refined = (cond + r_hat).reshape(pred_norm.shape)
        return refined

    def coherence_loss(
        self,
        profile_pred_norm: torch.Tensor,
        profile_tgt_norm: torch.Tensor,
        column_pred_norm: torch.Tensor,
        column_tgt_norm: torch.Tensor,
        level_pressures: Sequence[float],
        profile_var: str,
        column_var: str,
    ) -> torch.Tensor:
        """Column ↔ profile coherence penalty.

        Defines the (mass-weighted) coherence discrepancy
            D(col, prof) = col − Σ_l w_l · prof_l
        and matches the model's refined discrepancy to the ground-truth one:
            MSE( D(refined_col, refined_prof), D(tgt_col, tgt_prof) ).

        Because ``D`` is linear and the same weights apply to both sides, the
        per-variable normalisation constants cancel and the term simply forces
        the bias correction applied to the column to stay consistent with the
        correction applied to the vertically-integrated profile — exactly the
        coupling the per-variable heads otherwise lose.

        Shapes:
          profile_*: (B, L, H, W)   column_*: (B, H, W)
        """
        refined_prof = self.refine_norm_deterministic(profile_pred_norm, profile_var, "atmos")
        refined_col = self.refine_norm_deterministic(column_pred_norm, column_var, "surf")

        w = _pressure_thickness_weights(
            level_pressures, device=refined_prof.device, dtype=refined_prof.dtype,
        ).view(1, -1, 1, 1)

        integ_refined = (refined_prof * w).sum(dim=1)        # (B, H, W)
        integ_target = (profile_tgt_norm * w).sum(dim=1)     # (B, H, W)

        d_refined = refined_col - integ_refined
        d_target = column_tgt_norm - integ_target
        return F.mse_loss(d_refined, d_target)

    # --- Sampling ----------------------------------------------------------

    @torch.no_grad()
    def _sample_residual(
        self,
        cond_norm: torch.Tensor,
        head: ResidualFlowUNet,
    ) -> torch.Tensor:
        """Iterative refinement of the residual prediction.

        Uses the **data (x₁) parameterisation**: at each step the head
        predicts r̂ from a noise-perturbed input ``x_t`` (with decreasing
        noise level), and the result is re-noised at the next lower level
        and re-predicted. With ``sampling_steps == 1`` and a clean input
        (``x_t = 0, t = 1``) this reduces to a single deterministic
        regression; with more steps the model can express multi-modal
        residual distributions.

        Args:
          cond_norm: (N, 1, H, W) — normalised Aurora prediction (cond).
          head: per-variable :class:`ResidualFlowUNet`.

        Returns:
          (N, 1, H, W) sampled residual in normalised space.
        """
        N = cond_norm.shape[0]
        device = cond_norm.device
        dtype = cond_norm.dtype
        H, W = cond_norm.shape[-2:]

        steps = max(1, self.sampling_steps)

        if steps == 1:
            # Single-step deterministic regression E[r | ŷ]:
            # x_t = 0, t = 1 → head predicts residual from conditioning alone.
            x_t = torch.zeros(N, 1, H, W, device=device, dtype=dtype)
            t = torch.ones(N, device=device, dtype=torch.float32)
            return head(x_t, t, cond_norm)

        # Multi-step DDIM-style Euler ODE from t=0 (pure noise) toward t=1
        # (clean residual) using the x₁-prediction parameterisation.
        #
        # At each step we have x_t and the current r̂ = head(x_t, t, cond).
        # We estimate x₀ = (x_t − t·r̂) / (1 − t), then advance along the
        # straight-line trajectory:
        #   x_{t'} = (1 − t') · x₀_est  +  t' · r̂
        # This is the DDIM deterministic update, which avoids the compounding
        # noise injection of the previous Markov-chain approach and gives
        # cleaner, more reproducible multi-step predictions.
        t_schedule = torch.linspace(
            0.0, 1.0 - self.sigma_min, steps + 1, device=device
        )
        x = torch.randn(N, 1, H, W, device=device, dtype=dtype)
        r_hat = torch.zeros_like(x)
        for i in range(steps):
            t_curr = float(t_schedule[i].item())
            t_next = float(t_schedule[i + 1].item())
            t = torch.full((N,), t_curr, device=device, dtype=torch.float32)
            r_hat = head(x, t, cond_norm)
            if t_curr > 0.0:
                # Estimate x₀ from x_t and r̂, then interpolate to t_next.
                x0_est = (x - t_curr * r_hat) / (1.0 - t_curr)
                x = (1.0 - t_next) * x0_est + t_next * r_hat
            else:
                # At t=0: x_t IS x₀, best next point is along straight path.
                x = (1.0 - t_next) * x + t_next * r_hat
        return r_hat

    # --- Forward -----------------------------------------------------------

    def forward(self, batch: Batch) -> Batch:
        pred = self.base(batch)

        # Training: return Aurora's prediction unchanged. The FM head is
        # only exercised via :meth:`flow_loss` from the supervised loss
        # code path. This keeps the gradient through the (frozen) backbone
        # decoupled from the FM head's gradient, which is desirable.
        if self.training:
            return pred

        # Eval / inference: integrate the FM ODE and add the sampled
        # residual to Aurora's prediction in physical space.
        new_surf: dict[str, torch.Tensor] = dict(pred.surf_vars)
        for name, head in self.surf_flow.items():
            if name not in pred.surf_vars:
                continue
            tensor = pred.surf_vars[name]  # (B, T, H, W) or (B, H, W)
            # Aurora surf vars are (B, T, H, W). We refine each time slice.
            orig_shape = tensor.shape
            t_dim = 1 if tensor.dim() == 4 else 0
            if tensor.dim() == 4:
                B, T, H, W = tensor.shape
                flat = tensor.reshape(B * T, H, W)
            else:
                flat = tensor  # (B, H, W)
                B, H, W = flat.shape
                T = 1
            mean, std = self._norm_for(name, flat, kind="surf")
            cond = ((flat - mean) / std).unsqueeze(1)  # (N, 1, H, W)
            r = self._sample_residual(cond, head).squeeze(1)
            refined = flat + r * std  # de-normalise residual
            new_surf[name] = refined.reshape(orig_shape)

        new_atmos: dict[str, torch.Tensor] = dict(pred.atmos_vars)
        for name, head in self.atmos_flow.items():
            if name not in pred.atmos_vars:
                continue
            tensor = pred.atmos_vars[name]  # (B, T, L, H, W)
            orig_shape = tensor.shape
            if tensor.dim() == 5:
                B, T, L, H, W = tensor.shape
                flat = tensor.reshape(B * T, L, H, W)
            else:
                B, L, H, W = tensor.shape
                T = 1
                flat = tensor

            # Only refine loss_levels (if specified); other levels pass through.
            level_idx = self._atmos_loss_level_indices.get(name)
            if level_idx is not None and len(level_idx) < L:
                idx_t = torch.tensor(level_idx, dtype=torch.long, device=flat.device)
                flat_subset = flat.index_select(1, idx_t)  # (N, Lsub, H, W)
                mean, std = self._norm_for(name, flat_subset, kind="atmos")
                cond_norm = (flat_subset - mean) / std
                N, Lsub = cond_norm.shape[0], cond_norm.shape[1]
                cond_in = cond_norm.reshape(N * Lsub, 1, H, W)
                r = self._sample_residual(cond_in, head).reshape(N, Lsub, H, W)
                refined_subset = flat_subset + r * std
                # Scatter refined levels back; unrefined levels stay as-is.
                result = flat.clone()
                result[:, level_idx] = refined_subset
                new_atmos[name] = result.reshape(orig_shape)
            else:
                mean, std = self._norm_for(name, flat, kind="atmos")
                cond_norm = (flat - mean) / std  # (N, L, H, W)
                N = cond_norm.shape[0]
                cond_in = cond_norm.reshape(N * L, 1, H, W)
                r = self._sample_residual(cond_in, head).reshape(N, L, H, W)
                refined = flat + r * std
                new_atmos[name] = refined.reshape(orig_shape)

        return dataclasses.replace(pred, surf_vars=new_surf, atmos_vars=new_atmos)

    # --- Convenience --------------------------------------------------------

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = False

    def refine_parameter_count(self) -> int:
        return sum(
            p.numel()
            for heads in (self.surf_flow, self.atmos_flow)
            for head in heads.values()
            for p in head.parameters()
        )
