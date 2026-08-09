"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared spatial backbones for the Phase-2 residual refiners.

Both backbones map

    (noisy / interpolated residual field, spatial conditioning, process time,
     forecast lead time)  ->  residual-shaped output field

on the target latitude/longitude grid of **one sample at one forecast lead
time**. When a tensor carries several rollout lead times, the caller folds the
lead-time axis into the effective batch dimension before calling these modules,
so lead-time items are processed completely independently.

Explicitly absent, by design
----------------------------
* temporal self-attention, cross-time attention, cross-lead-time attention
* causal temporal masks and autoregressive decoding
* recurrent temporal hidden state
* a temporal token sequence

The only scalar "process time" input is the diffusion timestep or the flow
interpolation coordinate. Forecast lead time is injected through a **separate**
:class:`~finetune.refinement.schedules.LeadTimeEmbedding` and is combined with
the process-time embedding by addition before feature-wise modulation
(FiLM for the UNet, adaptive layer norm for the Transformer). That is
conditioning, not attention: no token is ever created for lead time.

Adapted from ``granitewxc.refinement.backbones`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0). Aurora-specific additions:
forecast lead-time conditioning, the ``windowed_2d`` spatial attention mode with
longitude periodicity, and configurable residual-block depth.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from finetune.refinement.schedules import LeadTimeEmbedding, ProcessTimeEmbedding

__all__ = [
    "ConditionalResidualUNet",
    "SpatialResidualTransformer",
    "patchify_2d",
    "sincos_2d_positional_encoding",
    "unpatchify_2d",
]


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _ConditioningEmbedding(nn.Module):
    """Process time (+ optionally forecast lead time) -> one conditioning vector."""

    def __init__(
        self,
        dim: int,
        *,
        lead_time_conditioning: bool = False,
        lead_time_scale_hours: float = 72.0,
        time_embedding_kind: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.process_time_embed = ProcessTimeEmbedding(dim, kind=time_embedding_kind)
        self.lead_time_embed = (
            LeadTimeEmbedding(dim, scale_hours=lead_time_scale_hours)
            if lead_time_conditioning
            else None
        )

    @property
    def uses_lead_time(self) -> bool:
        return self.lead_time_embed is not None

    def forward(self, process_time: torch.Tensor, lead_hours: torch.Tensor | None) -> torch.Tensor:
        emb = self.process_time_embed(process_time)
        if self.lead_time_embed is None:
            return emb
        if lead_hours is None:
            raise ValueError(
                "Forecast lead-time conditioning is enabled but no forecast_lead_time "
                "tensor was supplied."
            )
        lead = lead_hours.reshape(-1)
        if lead.numel() == 1 and process_time.reshape(-1).numel() > 1:
            lead = lead.expand(process_time.reshape(-1).numel())
        if lead.numel() != process_time.reshape(-1).numel():
            raise ValueError(
                f"forecast_lead_time has {lead.numel()} entries but the batch has "
                f"{process_time.reshape(-1).numel()}."
            )
        return emb + self.lead_time_embed(lead.to(emb.device))


# ---------------------------------------------------------------------------
# Convolutional backbone
# ---------------------------------------------------------------------------


class _FiLM(nn.Module):
    """Feature-wise linear modulation of a conv activation by the conditioning."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond_emb.to(self.proj.weight.dtype)).to(h.dtype).chunk(2, dim=-1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class _SpatialSelfAttention2d(nn.Module):
    """Multi-head self-attention over the flattened spatial grid of one sample.

    Tokens are grid cells ``(lat, lon)`` of a single sample at a single lead
    time. Batch elements are never mixed and there is no time axis.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"Bottleneck attention channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.num_heads = int(num_heads)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads

        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, self.num_heads, head_dim, h * w).transpose(-2, -1)

        out = F.scaled_dot_product_attention(_heads(q), _heads(k), _heads(v))
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class _ConvBlock(nn.Module):
    """Residual conv block with FiLM conditioning."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        dropout: float = 0.0,
        lon_periodic: bool = False,
    ) -> None:
        super().__init__()
        padding_mode = "circular" if lon_periodic else "replicate"
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode)
        self.norm1 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.film = _FiLM(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=padding_mode)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond_emb)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class ConditionalResidualUNet(nn.Module):
    """Conditional convolutional UNet predicting a residual-shaped field.

    The network is fully convolutional, so rectangular and non-power-of-two
    grids work. Feature maps are pooled with ``ceil_mode`` and resampled back to
    the exact skip-connection size on the way up, so the output height and width
    always equal the input height and width exactly.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 64,
        num_levels: int = 3,
        num_residual_blocks: int = 2,
        time_embedding_dim: int = 128,
        dropout: float = 0.0,
        bottleneck_attention: bool = True,
        attention_heads: int = 4,
        zero_init_output: bool = True,
        lead_time_conditioning: bool = False,
        lead_time_scale_hours: float = 72.0,
        lon_periodic: bool = False,
        time_embedding_kind: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if num_residual_blocks < 1:
            raise ValueError(f"num_residual_blocks must be >= 1, got {num_residual_blocks}")
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.out_channels = int(out_channels)
        self.num_levels = int(num_levels)
        self.lon_periodic = bool(lon_periodic)

        self.cond_embed = _ConditioningEmbedding(
            time_embedding_dim,
            lead_time_conditioning=lead_time_conditioning,
            lead_time_scale_hours=lead_time_scale_hours,
            time_embedding_kind=time_embedding_kind,
        )

        widths = [hidden_channels * (2**i) for i in range(num_levels)]
        self.down_blocks = nn.ModuleList()
        prev = in_channels + cond_channels
        for width in widths:
            stage = nn.ModuleList()
            for block_index in range(num_residual_blocks):
                stage.append(
                    _ConvBlock(
                        prev if block_index == 0 else width,
                        width,
                        time_embedding_dim,
                        dropout,
                        lon_periodic=self.lon_periodic,
                    )
                )
            self.down_blocks.append(stage)
            prev = width

        self.bottleneck_attn = (
            _SpatialSelfAttention2d(widths[-1], attention_heads) if bottleneck_attention else None
        )

        self.up_blocks = nn.ModuleList()
        for level in range(num_levels - 1, 0, -1):
            stage = nn.ModuleList()
            in_width = widths[level] + widths[level - 1]
            out_width = widths[level - 1]
            for block_index in range(num_residual_blocks):
                stage.append(
                    _ConvBlock(
                        in_width if block_index == 0 else out_width,
                        out_width,
                        time_embedding_dim,
                        dropout,
                        lon_periodic=self.lon_periodic,
                    )
                )
            self.up_blocks.append(stage)

        self.out_proj = nn.Conv2d(widths[0], out_channels, 1)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        process_time: torch.Tensor,
        lead_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"Residual field {tuple(x.shape[-2:])} and conditioning "
                f"{tuple(cond.shape[-2:])} must share the spatial grid."
            )
        cond_emb = self.cond_embed(process_time, lead_hours)
        h = torch.cat([x, cond], dim=1)

        skips: list[torch.Tensor] = []
        for level, stage in enumerate(self.down_blocks):
            for block in stage:
                h = block(h, cond_emb)
            if level < self.num_levels - 1:
                skips.append(h)
                h = F.avg_pool2d(h, kernel_size=2, ceil_mode=True)

        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)

        for stage in self.up_blocks:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h, cond_emb)

        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Spatial Transformer backbone
# ---------------------------------------------------------------------------


def patchify_2d(x: torch.Tensor, patch_h: int, patch_w: int) -> tuple[torch.Tensor, int, int]:
    """Vectorised 2-D patchification.

    Args:
        x: ``[B, C, H, W]`` where ``H`` and ``W`` are exact multiples of the
            patch size.

    Returns:
        ``(tokens, grid_h, grid_w)`` where ``tokens`` is
        ``[B, grid_h * grid_w, C * patch_h * patch_w]`` in **row-major**
        ``(lat, lon)`` order, so token index ``i`` maps to grid position
        ``(i // grid_w, i % grid_w)``.
    """
    b, c, h, w = x.shape
    if h % patch_h or w % patch_w:
        raise ValueError(
            f"patchify_2d requires H and W to be multiples of the patch size; got "
            f"({h}, {w}) with patch ({patch_h}, {patch_w})."
        )
    grid_h, grid_w = h // patch_h, w // patch_w
    tokens = (
        x.reshape(b, c, grid_h, patch_h, grid_w, patch_w)
        .permute(0, 2, 4, 1, 3, 5)  # B, gh, gw, C, ph, pw
        .reshape(b, grid_h * grid_w, c * patch_h * patch_w)
    )
    return tokens, grid_h, grid_w


def unpatchify_2d(
    tokens: torch.Tensor, channels: int, grid_h: int, grid_w: int, patch_h: int, patch_w: int
) -> torch.Tensor:
    """Exact inverse of :func:`patchify_2d`."""
    b, n, d = tokens.shape
    if n != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} tokens, got {n}.")
    if d != channels * patch_h * patch_w:
        raise ValueError(f"Expected token width {channels * patch_h * patch_w}, got {d}.")
    return (
        tokens.reshape(b, grid_h, grid_w, channels, patch_h, patch_w)
        .permute(0, 3, 1, 4, 2, 5)  # B, C, gh, ph, gw, pw
        .reshape(b, channels, grid_h * patch_h, grid_w * patch_w)
    )


def sincos_2d_positional_encoding(
    dim: int, grid_h: int, grid_w: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Fixed 2-D sin/cos positional encoding, ``[grid_h * grid_w, dim]``.

    Half of the embedding encodes the latitude token index and half the
    longitude token index, so the encoding is a genuine function of the
    two-dimensional grid location, not of a flattened sequence position.
    """
    if dim % 4 != 0:
        raise ValueError(f"sincos_2d_positional_encoding requires dim % 4 == 0, got {dim}")
    half = dim // 2

    def _axis(positions: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(half // 2, device=device, dtype=torch.float32)
        omega = 1.0 / (10_000 ** (omega / (half / 2)))
        out = positions.float().reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    rows = torch.arange(grid_h, device=device)
    cols = torch.arange(grid_w, device=device)
    lat_grid = rows.reshape(-1, 1).expand(grid_h, grid_w).reshape(-1)
    lon_grid = cols.reshape(1, -1).expand(grid_h, grid_w).reshape(-1)
    return torch.cat([_axis(lat_grid), _axis(lon_grid)], dim=1).to(dtype)


class _SpatialAttention(nn.Module):
    """Exact multi-head self-attention across spatial tokens.

    ``global_2d`` attends over every spatial token of the sample; ``windowed_2d``
    attends only within (optionally shifted) 2-D windows of the token grid. Both
    are strictly spatial: batch elements never interact and there is no time
    axis. ``implementation`` selects between PyTorch's fused
    ``scaled_dot_product_attention`` and an explicit reference implementation;
    both compute the *same* mathematical operation, and no sparse, local or
    linearised approximation is ever substituted for a requested exact mode.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        implementation: str = "auto",
        *,
        mode: str = "global_2d",
        window_size: tuple[int, int] = (8, 8),
        shift: tuple[int, int] = (0, 0),
        lon_periodic: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({dim}) must be divisible by num_heads ({num_heads}).")
        if mode not in {"global_2d", "windowed_2d"}:
            raise ValueError(f"Unsupported spatial attention mode {mode!r}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.mode = mode
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.shift = (int(shift[0]), int(shift[1]))
        self.lon_periodic = bool(lon_periodic)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.set_attention_implementation(implementation)

    def set_attention_implementation(self, implementation: str) -> None:
        impl = str(implementation).lower()
        if impl not in {"auto", "sdpa", "math"}:
            raise ValueError(f"Unsupported attention implementation {implementation!r}")
        if impl == "auto":
            impl = "sdpa" if hasattr(F, "scaled_dot_product_attention") else "math"
        if impl == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "optimized_attention='sdpa' requested but this PyTorch build has no "
                "scaled_dot_product_attention; use 'math' or 'auto'."
            )
        self.implementation = impl

    # -- core attention --------------------------------------------------
    def _attend(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        p = self.dropout if self.training else 0.0
        if self.implementation == "sdpa":
            # is_causal=False: attention is bidirectional across spatial tokens.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=p, is_causal=False
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            if p > 0:
                weights = F.dropout(weights, p=p, training=True)
            out = weights @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.mode == "global_2d":
            return self._attend(x, None)
        return self._windowed(x, grid_h, grid_w)

    # -- windowed attention ----------------------------------------------
    def _windowed(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        b, n, d = x.shape
        if n != grid_h * grid_w:
            raise ValueError(f"Expected {grid_h * grid_w} tokens, got {n}.")
        win_h = min(self.window_size[0], grid_h)
        win_w = min(self.window_size[1], grid_w)
        shift_h = self.shift[0] % win_h if win_h > 1 else 0
        shift_w = self.shift[1] % win_w if win_w > 1 else 0

        grid = x.reshape(b, grid_h, grid_w, d)

        # Roll first (circular), then record which rows/columns hold wrapped
        # content so they are not allowed to attend across the seam unless the
        # axis really is periodic.
        wrapped_lat = torch.zeros(grid_h, dtype=torch.bool, device=x.device)
        wrapped_lon = torch.zeros(grid_w, dtype=torch.bool, device=x.device)
        if shift_h:
            grid = torch.roll(grid, shifts=-shift_h, dims=1)
            wrapped_lat[grid_h - shift_h :] = True
        if shift_w:
            grid = torch.roll(grid, shifts=-shift_w, dims=2)
            if not self.lon_periodic:
                wrapped_lon[grid_w - shift_w :] = True

        pad_h = (-grid_h) % win_h
        pad_w = (-grid_w) % win_w
        valid_lat = torch.ones(grid_h + pad_h, dtype=torch.bool, device=x.device)
        valid_lon = torch.ones(grid_w + pad_w, dtype=torch.bool, device=x.device)
        region_lat = F.pad(wrapped_lat, (0, pad_h), value=False).long()
        region_lon = F.pad(wrapped_lon, (0, pad_w), value=False).long()
        if pad_h:
            # Latitude is never periodic: pad with zeros and exclude the padded
            # rows from every softmax.
            grid = F.pad(grid.permute(0, 3, 1, 2), (0, 0, 0, pad_h)).permute(0, 2, 3, 1)
            valid_lat[grid_h:] = False
        if pad_w:
            if self.lon_periodic:
                # Circular padding keeps genuine wrap-around adjacency in the
                # final partial window. The duplicated columns act as keys only;
                # their outputs are cropped away below.
                grid = torch.cat([grid, grid[:, :, :pad_w]], dim=2)
                region_lon[grid_w:] = region_lon[:pad_w]
            else:
                grid = F.pad(grid.permute(0, 3, 1, 2), (0, pad_w)).permute(0, 2, 3, 1)
                valid_lon[grid_w:] = False

        padded_h, padded_w = grid_h + pad_h, grid_w + pad_w
        n_win_h, n_win_w = padded_h // win_h, padded_w // win_w

        windows = (
            grid.reshape(b, n_win_h, win_h, n_win_w, win_w, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b * n_win_h * n_win_w, win_h * win_w, d)
        )

        mask = self._window_mask(
            valid_lat, valid_lon, region_lat, region_lon, n_win_h, n_win_w, win_h, win_w
        )
        mask = mask.repeat(b, 1, 1).unsqueeze(1)  # [B*nW, 1, S, S]

        out = self._attend(windows, mask)

        out = (
            out.reshape(b, n_win_h, n_win_w, win_h, win_w, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, padded_h, padded_w, d)
        )
        out = out[:, :grid_h, :grid_w]
        if shift_w:
            out = torch.roll(out, shifts=shift_w, dims=2)
        if shift_h:
            out = torch.roll(out, shifts=shift_h, dims=1)
        return out.reshape(b, n, d)

    @staticmethod
    def _window_mask(
        valid_lat: torch.Tensor,
        valid_lon: torch.Tensor,
        region_lat: torch.Tensor,
        region_lon: torch.Tensor,
        n_win_h: int,
        n_win_w: int,
        win_h: int,
        win_w: int,
    ) -> torch.Tensor:
        """Boolean ``[nW, S, S]`` mask (``True`` == attend)."""
        valid = valid_lat[:, None] & valid_lon[None, :]
        region = region_lat[:, None] * 2 + region_lon[None, :]

        def _partition(grid: torch.Tensor) -> torch.Tensor:
            return (
                grid.reshape(n_win_h, win_h, n_win_w, win_w)
                .permute(0, 2, 1, 3)
                .reshape(n_win_h * n_win_w, win_h * win_w)
            )

        valid_w = _partition(valid)
        region_w = _partition(region)
        same_region = region_w[:, :, None] == region_w[:, None, :]
        key_valid = valid_w[:, None, :]
        mask = same_region & key_valid
        # A fully-masked query row would produce NaNs; always allow self
        # attention. Padded queries are cropped away afterwards.
        eye = torch.eye(win_h * win_w, dtype=torch.bool, device=valid.device)
        return mask | eye.unsqueeze(0)


class _DiTBlock(nn.Module):
    """Pre-norm Transformer block with adaptive-layer-norm conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        implementation: str,
        *,
        mode: str = "global_2d",
        window_size: tuple[int, int] = (8, 8),
        shift: tuple[int, int] = (0, 0),
        lon_periodic: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _SpatialAttention(
            dim,
            num_heads,
            dropout,
            implementation,
            mode=mode,
            window_size=window_size,
            shift=shift,
            lon_periodic=lon_periodic,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

    def forward(
        self, x: torch.Tensor, cond_emb: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        params = self.ada_ln(cond_emb.to(x.dtype)).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (p.unsqueeze(1) for p in params)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h, grid_h, grid_w)
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


class SpatialResidualTransformer(nn.Module):
    """DiT-style Transformer refiner operating on 2-D spatial tokens only.

    Pipeline for each sample at each forecast lead time::

        conditioning + residual state  [B, C, H, W]
          -> pad to a multiple of the patch size
          -> patchify into (grid_h x grid_w) 2-D tokens, row-major (lat, lon)
          -> linear token projection + 2-D positional encoding
          -> N x (spatial self-attention + MLP) with adaLN conditioning on
             (process time  +  forecast lead time), injected separately
          -> linear projection back to patch pixels
          -> unpatchify
          -> crop to the exact original (H, W)

    Padding is applied at the trailing edges and removed by an exact crop, so
    the returned field has the original latitude/longitude dimensions and token
    order is preserved throughout.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        *,
        patch_size: tuple[int, int] = (8, 8),
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        positional_encoding: str = "latlon_2d",
        max_tokens_lat: int = 256,
        max_tokens_lon: int = 512,
        attention_mode: str = "global_2d",
        window_size: tuple[int, int] = (8, 8),
        shifted_windows: bool = False,
        gradient_checkpointing: bool = False,
        optimized_attention: str = "auto",
        zero_init_output: bool = True,
        lead_time_conditioning: bool = False,
        lead_time_scale_hours: float = 72.0,
        lon_periodic: bool = False,
        time_embedding_kind: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.patch_h, self.patch_w = int(patch_size[0]), int(patch_size[1])
        if self.patch_h < 1 or self.patch_w < 1:
            raise ValueError(f"patch size must be positive, got {patch_size!r}")
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.out_channels = int(out_channels)
        self.embedding_dim = int(embedding_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.positional_encoding = str(positional_encoding).lower()
        self.max_tokens_lat = int(max_tokens_lat)
        self.max_tokens_lon = int(max_tokens_lon)
        self.attention_mode = str(attention_mode).lower()
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.shifted_windows = bool(shifted_windows)
        self.lon_periodic = bool(lon_periodic)

        token_in = (self.in_channels + self.cond_channels) * self.patch_h * self.patch_w
        token_out = self.out_channels * self.patch_h * self.patch_w
        self.token_proj = nn.Linear(token_in, self.embedding_dim)
        self.cond_embed = _ConditioningEmbedding(
            self.embedding_dim,
            lead_time_conditioning=lead_time_conditioning,
            lead_time_scale_hours=lead_time_scale_hours,
            time_embedding_kind=time_embedding_kind,
        )

        self.pos_lat: nn.Parameter | None = None
        self.pos_lon: nn.Parameter | None = None
        if self.positional_encoding in {"latlon_2d", "learned_2d"}:
            # Separable learned 2-D encoding: one table per axis, summed. The
            # parameter count stays linear in the grid extent while every
            # (lat, lon) token still receives a distinct embedding.
            self.pos_lat = nn.Parameter(torch.zeros(self.max_tokens_lat, self.embedding_dim))
            self.pos_lon = nn.Parameter(torch.zeros(self.max_tokens_lon, self.embedding_dim))
            nn.init.trunc_normal_(self.pos_lat, std=0.02)
            nn.init.trunc_normal_(self.pos_lon, std=0.02)
        elif self.positional_encoding == "sincos_2d":
            self._sincos_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
        else:
            raise ValueError(f"Unsupported positional_encoding {positional_encoding!r}")

        blocks = []
        for index in range(int(num_blocks)):
            shift = (0, 0)
            if self.attention_mode == "windowed_2d" and self.shifted_windows and index % 2 == 1:
                shift = (self.window_size[0] // 2, self.window_size[1] // 2)
            blocks.append(
                _DiTBlock(
                    self.embedding_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    optimized_attention,
                    mode=self.attention_mode,
                    window_size=self.window_size,
                    shift=shift,
                    lon_periodic=self.lon_periodic,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(self.embedding_dim, elementwise_affine=False, eps=1e-6)
        self.final_ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(self.embedding_dim, 2 * self.embedding_dim)
        )
        nn.init.zeros_(self.final_ada_ln[1].weight)
        nn.init.zeros_(self.final_ada_ln[1].bias)
        self.out_proj = nn.Linear(self.embedding_dim, token_out)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        self._warned_seam = False

    # -- helpers ---------------------------------------------------------
    def set_attention_implementation(self, implementation: str) -> None:
        for block in self.blocks:
            block.attn.set_attention_implementation(implementation)

    def _positional(self, grid_h: int, grid_w: int, device, dtype) -> torch.Tensor:
        if self.positional_encoding in {"latlon_2d", "learned_2d"}:
            if grid_h > self.max_tokens_lat or grid_w > self.max_tokens_lon:
                raise ValueError(
                    f"Token grid ({grid_h}, {grid_w}) exceeds the configured "
                    f"max_tokens_lat/max_tokens_lon ({self.max_tokens_lat}, "
                    f"{self.max_tokens_lon}). Increase them or the patch size."
                )
            lat = self.pos_lat[:grid_h].unsqueeze(1)  # [gh, 1, D]
            lon = self.pos_lon[:grid_w].unsqueeze(0)  # [1, gw, D]
            return (lat + lon).reshape(grid_h * grid_w, self.embedding_dim).to(dtype)
        key = (grid_h, grid_w, device, dtype)
        cached = self._sincos_cache.get(key)
        if cached is None:
            cached = sincos_2d_positional_encoding(
                self.embedding_dim, grid_h, grid_w, device=device, dtype=dtype
            )
            self._sincos_cache[key] = cached
        return cached

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        process_time: torch.Tensor,
        lead_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"Residual field {tuple(x.shape[-2:])} and conditioning "
                f"{tuple(cond.shape[-2:])} must share the spatial grid."
            )
        b, _, h, w = x.shape
        stacked = torch.cat([x, cond], dim=1)

        pad_h = (-h) % self.patch_h
        pad_w = (-w) % self.patch_w
        if pad_h or pad_w:
            # Replicate (or circular in longitude) padding avoids inventing
            # boundary gradients; the padded rows/columns are cropped away
            # exactly after unpatchify.
            if pad_w and self.lon_periodic:
                stacked = torch.cat([stacked, stacked[..., :pad_w]], dim=-1)
                pad_lon_applied = True
            else:
                pad_lon_applied = False
            trailing = (0, 0 if pad_lon_applied else pad_w, 0, pad_h)
            if any(trailing):
                stacked = F.pad(stacked, trailing, mode="replicate")

        tokens, grid_h, grid_w = patchify_2d(stacked, self.patch_h, self.patch_w)
        if (
            self.attention_mode == "windowed_2d"
            and self.lon_periodic
            and grid_w % min(self.window_size[1], grid_w)
            and not self._warned_seam
        ):
            warnings.warn(
                f"windowed_2d attention with a periodic longitude and a token grid of "
                f"{grid_w} columns that is not a multiple of window_size[1]="
                f"{self.window_size[1]}: wrap-around adjacency in the final partial "
                "window is handled by circular padding. Choose a window size dividing "
                "the token grid for an exactly uniform partition.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_seam = True

        tokens = self.token_proj(tokens)
        tokens = tokens + self._positional(grid_h, grid_w, tokens.device, tokens.dtype).unsqueeze(0)

        cond_emb = self.cond_embed(process_time, lead_hours).to(tokens.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, cond_emb, grid_h, grid_w, use_reentrant=False)
            else:
                tokens = block(tokens, cond_emb, grid_h, grid_w)

        shift, scale = self.final_ada_ln(cond_emb).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tokens = self.out_proj(tokens)

        out = unpatchify_2d(tokens, self.out_channels, grid_h, grid_w, self.patch_h, self.patch_w)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        if out.shape != (b, self.out_channels, h, w):
            raise RuntimeError(
                f"Transformer output shape {tuple(out.shape)} does not match the expected "
                f"{(b, self.out_channels, h, w)}."
            )
        return out
