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
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.batch import Batch


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


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
    ) -> None:
        super().__init__()
        self.base = base
        self.target_surf_vars = tuple(target_surf_vars)
        self.target_atmos_vars = tuple(target_atmos_vars)
        self.hidden = hidden
        self.sampling_steps = int(sampling_steps)
        self.sigma_min = float(sigma_min)

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
        t = torch.rand(N, device=device, dtype=torch.float32)
        t = t.clamp(self.sigma_min, 1.0 - self.sigma_min)
        t_b = t.view(N, 1, 1, 1).to(dtype)

        x_t = (1.0 - t_b) * x0 + t_b * r
        r_pred = head(x_t, t, cond)
        return F.mse_loss(r_pred, r)

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

        steps = max(1, self.sampling_steps)
        # t schedule: from low (noisy) to high (clean), evenly spaced.
        # Single-step (steps==1) → t=1 deterministic regression.
        if steps == 1:
            t_vals = [1.0]
        else:
            t_vals = [(k + 1) / steps for k in range(steps)]

        x = torch.randn(N, 1, *cond_norm.shape[-2:], device=device, dtype=dtype)
        r_hat = torch.zeros_like(x)
        for t_val in t_vals:
            t = torch.full((N,), float(t_val), device=device, dtype=torch.float32)
            # Build x_t at this noise level using current best r̂ estimate.
            # On the first iteration r_hat=0 so x_t is pure noise scaled by
            # (1-t); the head fills in r_hat which feeds the next iter.
            t_b = torch.tensor(float(t_val), device=device, dtype=dtype)
            x_t = (1.0 - t_b) * x + t_b * r_hat
            r_hat = head(x_t, t, cond_norm)
            # Resample noise for next iteration (Markov-chain style).
            x = torch.randn_like(x)
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
            mean, std = self._norm_for(name, flat, kind="atmos")
            cond_norm = (flat - mean) / std  # (N, L, H, W)
            # Per-level independent sampling: collapse L into batch.
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
