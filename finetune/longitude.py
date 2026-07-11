"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Longitude periodicity helpers for the Aurora post-processing pipeline.

The bias-correction heads (conv-refine, flow-matching, Mamba temporal) treat a
field as a flat 2-D image ``(H=lat, W=lon)``. On a *global* grid the longitude
axis is periodic: the column at lon ≈ 0° is physically adjacent to the column
at lon ≈ 360°. Plain ``padding_mode="replicate"`` convolutions instead treat the
0°/360° boundary as a hard edge, which injects a spurious discontinuity (a
"seam") into the correction field near the dateline.

This module centralises the primitives used to make the whole pipeline
longitude-aware:

* :func:`longitude_is_periodic` / :func:`resolve_lon_periodic` — decide whether a
  domain wraps the globe (global) or not (regional). Regional domains must keep
  the old non-periodic behaviour, so the fix is *opt-in per domain*.
* :class:`PeriodicConv2d` — a drop-in ``nn.Conv2d`` that pads longitude (``W``)
  circularly and latitude (``H``) by replication. With ``lon_periodic=False`` it
  is numerically identical to ``nn.Conv2d(padding_mode="replicate")`` and it
  keeps the exact same ``state_dict`` keys, so existing checkpoints load
  unchanged.
* :func:`lon_cyclic_features` — ``sin``/``cos`` longitude channels, so longitude
  is never fed to a network as a raw discontinuous value.
* :func:`add_cyclic_column` / :func:`dateline_discontinuity` — evaluation and
  plotting helpers that measure / close the seam across the dateline.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "to_360",
    "canonical_longitudes",
    "longitude_is_periodic",
    "longitude_grid_signature",
    "resolve_lon_periodic",
    "pad_lonlat",
    "PeriodicConv2d",
    "periodic_avg_pool2d",
    "periodic_bilinear_interpolate",
    "lon_cyclic_features",
    "add_cyclic_column",
    "dateline_discontinuity_from_edges",
    "dateline_discontinuity",
]


# ---------------------------------------------------------------------------
# Longitude convention + periodicity detection
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    """Convert array-likes (incl. grad-requiring torch tensors) to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_360(lon: Any) -> np.ndarray:
    """Wrap longitudes to the ``[0, 360)`` convention.

    Works for both the ``[-180, 180)`` and ``[0, 360)`` source conventions so the
    rest of the pipeline can reason about a single convention.
    """
    arr = _to_numpy(lon).astype(np.float64)
    wrapped = np.mod(arr, 360.0)
    # Floating-point values just below/above a multiple of 360 should map to
    # the canonical zero rather than produce a spurious near-360 coordinate.
    return np.asarray(
        np.where(np.isclose(wrapped, 360.0, rtol=0.0, atol=1e-8), 0.0, wrapped),
        dtype=np.float64,
    )


def canonical_longitudes(
    lon: Any,
    *,
    duplicate_atol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique ``[0, 360)`` longitudes and source indices.

    The returned indices can be used to reorder every field on the longitude
    axis in exactly the same way as the coordinate.  This is the central
    convention used by preprocessing, training, inference, and NetCDF output.
    It accepts either common source convention and deterministically removes a
    duplicated cyclic endpoint (for example both 0 and 360, or both -180 and
    180).

    Args:
        lon: One-dimensional longitude coordinate in degrees.
        duplicate_atol: Absolute tolerance for coordinates that become equal
            after wrapping.

    Returns:
        ``(longitude, source_indices)`` where ``longitude`` is strictly
        increasing in ``[0, 360)`` and ``source_indices`` selects the matching
        columns from the input.
    """
    raw = _to_numpy(lon).astype(np.float64)
    wrapped = to_360(raw)
    if wrapped.ndim != 1:
        raise ValueError(f"Longitude coordinate must be one-dimensional, got {wrapped.shape}.")
    if wrapped.size == 0:
        raise ValueError("Longitude coordinate must not be empty.")
    if not np.all(np.isfinite(wrapped)):
        raise ValueError("Longitude coordinate contains non-finite values.")

    # Stable sorting makes duplicate endpoint selection deterministic: for a
    # conventional [0, ..., 360] input, the real 0-degree column is kept.
    order = np.argsort(wrapped, kind="stable")
    sorted_lon = wrapped[order]
    keep = np.ones(sorted_lon.size, dtype=bool)
    if sorted_lon.size > 1:
        keep[1:] = np.diff(sorted_lon) > float(duplicate_atol)
        for index in np.where(~keep)[0]:
            delta_turns = abs(float(raw[order[index]] - raw[order[index - 1]])) / 360.0
            if delta_turns < 0.5 or not np.isclose(
                delta_turns, round(delta_turns), rtol=0.0,
                atol=float(duplicate_atol) / 360.0,
            ):
                raise ValueError(
                    "Longitude contains a duplicate coordinate that is not a cyclic "
                    "endpoint representation."
                )

    canonical = sorted_lon[keep]
    source_indices = order[keep]
    if canonical.size > 1 and not np.all(np.diff(canonical) > 0):
        raise ValueError("Longitudes are not strictly increasing after canonicalisation.")
    return canonical, source_indices


def longitude_is_periodic(lon: Any, rtol: float = 0.5, atol_deg: float = 1e-3) -> bool:
    """Return ``True`` when a 1-D longitude coordinate wraps the whole globe.

    A grid is treated as periodic when the gap from the last point back to the
    first (across the 360° wrap) matches the typical interior spacing, i.e. the
    grid "closes up" into a full circle. This is convention-agnostic (it first
    maps to ``[0, 360)``) and robustly separates a *global* grid (wrap gap ≈
    grid spacing) from a *regional* subset (wrap gap ≫ grid spacing).

    Args:
        lon: 1-D longitude values in degrees (any convention).
        rtol: Relative tolerance on the wrap gap vs. the median grid spacing.
        atol_deg: Absolute tolerance in degrees to absorb floating-point jitter.

    Returns:
        Whether the longitude axis should be treated as periodic.
    """
    s, _ = canonical_longitudes(lon)
    if s.size < 3:
        # Too few points to reliably infer a global wrap; be conservative.
        return False
    diffs = np.diff(s)
    med = float(np.median(diffs))
    if med <= 0:
        return False
    wrap = float((s[0] + 360.0) - s[-1])
    tol = max(rtol * med, atol_deg)
    # Periodic iff the wrap gap looks like one more grid step (globe closes) and
    # the interior spacing is roughly uniform (no large internal gaps).
    interior_uniform = float(np.max(diffs) - np.min(diffs)) <= max(rtol * med, atol_deg)
    return abs(wrap - med) <= tol and interior_uniform


def longitude_grid_signature(lon: Any) -> str:
    """Stable fingerprint for checkpoint/training/inference grid validation.

    Coordinates are quantised to ``1e-4`` degrees before hashing so harmless
    float32-vs-float64 representation differences do not change the signature.
    """
    canonical, _ = canonical_longitudes(lon)
    quantised = np.rint(canonical * 1e4).astype("<i8", copy=False)
    digest = hashlib.sha256(quantised.tobytes()).hexdigest()
    return f"{canonical.size}:{digest}"


def resolve_lon_periodic(config: Mapping[str, Any], lon: Any | None = None) -> bool:
    """Resolve whether longitude should be treated as periodic for this run.

    Resolution order:

    1. Explicit ``model.lon_periodic`` (``true`` / ``false``) always wins, so a
       user can override detection.
    2. ``auto`` (the default): detect from the actual ``lon`` grid when provided
       (the most robust signal, convention-agnostic).
    3. Otherwise fall back to ``data.domain_type`` (``global`` ⇒ periodic).

    Using the same config/grid for both training and inference guarantees the
    two stages agree on the periodic-boundary logic.
    """
    model_cfg = config.get("model", {}) if isinstance(config, Mapping) else {}
    val = model_cfg.get("lon_periodic", "auto")
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "on", "periodic"}:
        return True
    if s in {"false", "0", "no", "off", "regional"}:
        return False
    if s != "auto":
        raise ValueError(
            "model.lon_periodic must be true, false, or 'auto'; "
            f"got {val!r}."
        )
    # auto
    if lon is not None:
        return longitude_is_periodic(lon)
    resolved = model_cfg.get("lon_periodic_resolved")
    if isinstance(resolved, bool):
        return resolved
    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    domain = str(data_cfg.get("domain_type", "global")).strip().lower()
    return domain in {"global", "periodic"}


# ---------------------------------------------------------------------------
# Circular-in-longitude convolution
# ---------------------------------------------------------------------------


def pad_lonlat(
    x: torch.Tensor,
    pad_h: int,
    pad_w: int,
    lon_periodic: bool,
    lat_mode: str = "replicate",
) -> torch.Tensor:
    """Pad the last two dims ``(H=lat, W=lon)`` for a spatial convolution.

    Longitude (``W``) is padded circularly when ``lon_periodic`` so information
    crosses the dateline; otherwise it uses ``lat_mode`` (replicate). Latitude
    (``H``) always uses ``lat_mode`` because the poles are *not* periodic.

    With ``lon_periodic=False`` this reproduces ``padding_mode="replicate"``
    exactly (replicate on both axes).
    """
    if pad_w > 0:
        mode_w = "circular" if lon_periodic else lat_mode
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode=mode_w)
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode=lat_mode)
    return x


class PeriodicConv2d(nn.Conv2d):
    """``Conv2d`` with circular longitude padding and replicate latitude padding.

    This is a drop-in replacement for ``nn.Conv2d(..., padding_mode="replicate")``
    that makes the longitude axis periodic. It performs the padding manually and
    calls the convolution with zero internal padding, so:

    * ``lon_periodic=True`` → longitude wraps across the dateline (the fix);
    * ``lon_periodic=False`` → identical to the previous replicate behaviour;
    * the learnable parameters remain ``weight``/``bias`` (no submodules), so the
      ``state_dict`` is unchanged and existing checkpoints load as-is.

    Only "same"-size odd-kernel convolutions are used in this project; the pad
    width is derived from the kernel size and dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        lon_periodic: bool = True,
        **kwargs: Any,
    ) -> None:
        # We handle padding ourselves; force the underlying conv to pad with
        # nothing and keep the default (zeros) padding_mode so `_conv_forward`
        # uses `self.padding` (which is 0) directly.
        kwargs.pop("padding", None)
        kwargs.pop("padding_mode", None)
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        kh, kw = self.kernel_size
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("PeriodicConv2d requires odd spatial kernel sizes for same padding.")
        dh, dw = self.dilation
        self._pad_h = dh * (kh // 2)
        self._pad_w = dw * (kw // 2)
        self.lon_periodic = bool(lon_periodic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad_lonlat(x, self._pad_h, self._pad_w, self.lon_periodic)
        return self._conv_forward(x, self.weight, self.bias)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic only
        return f"{super().extra_repr()}, lon_periodic={self.lon_periodic}"


def periodic_avg_pool2d(x: torch.Tensor, lon_periodic: bool) -> torch.Tensor:
    """Downsample by two without discarding an unmatched longitude column.

    A regular ``avg_pool2d(..., 2)`` silently drops the last column when the
    width is odd.  On a periodic grid that creates a scale-dependent hard edge.
    We instead append the first longitude column as a one-cell cyclic halo,
    then pool.  Latitude retains the existing floor behaviour because it is
    not periodic.

    Regional/non-periodic inputs intentionally keep PyTorch's original pooling
    behaviour for checkpoint compatibility.
    """
    if not lon_periodic or x.shape[-1] % 2 == 0:
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    x = torch.cat((x, x[..., :1]), dim=-1)
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def periodic_bilinear_interpolate(
    x: torch.Tensor,
    size: tuple[int, int],
    lon_periodic: bool,
) -> torch.Tensor:
    """Bilinearly resize ``(H, W)`` while wrapping longitude.

    ``torch.nn.functional.interpolate`` clamps samples outside the first and
    last columns.  Consequently its two output edges are independent even if
    all surrounding convolutions use circular padding.  This function keeps
    PyTorch's ordinary latitude interpolation but performs the longitude
    interpolation explicitly with modulo indices, so the first and last
    columns share their correct cyclic neighbours.

    The coordinate mapping matches ``align_corners=False``.
    """
    target_h, target_w = (int(size[0]), int(size[1]))
    if not lon_periodic:
        return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Interpolation size must be positive, got {size}.")

    source_w = int(x.shape[-1])
    if source_w == 0:
        raise ValueError("Cannot interpolate an empty longitude axis.")

    # Resize latitude only. Since the width is unchanged, this introduces no
    # longitude boundary condition.
    if int(x.shape[-2]) != target_h:
        x = F.interpolate(
            x,
            size=(target_h, source_w),
            mode="bilinear",
            align_corners=False,
        )
    if source_w == target_w:
        return x

    # Source positions used by align_corners=False, wrapped into [0, W).
    pos = (
        (torch.arange(target_w, device=x.device, dtype=torch.float64) + 0.5)
        * (source_w / target_w)
        - 0.5
    )
    left_unwrapped = torch.floor(pos)
    weight_right = (pos - left_unwrapped).to(dtype=x.dtype)
    left = left_unwrapped.to(dtype=torch.long).remainder(source_w)
    right = (left + 1).remainder(source_w)

    left_values = x.index_select(-1, left)
    right_values = x.index_select(-1, right)
    shape = [1] * x.ndim
    shape[-1] = target_w
    weight_right = weight_right.reshape(shape)
    return left_values + (right_values - left_values) * weight_right


# ---------------------------------------------------------------------------
# Cyclic longitude features (never feed raw longitude to a network)
# ---------------------------------------------------------------------------


def lon_cyclic_features(
    lon: torch.Tensor,
    height: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build ``[sin(lon), cos(lon)]`` feature maps of shape ``(2, H, W)``.

    Longitude is a periodic coordinate, so a raw value has a discontinuity at
    the 0°/360° wrap. The ``sin``/``cos`` pair is smooth across the dateline and
    is the recommended encoding when longitude is used as a network input.
    """
    lon_t = torch.as_tensor(lon, device=device, dtype=torch.float64)
    ang = torch.deg2rad(torch.remainder(lon_t, 360.0))
    feats = torch.stack([torch.sin(ang), torch.cos(ang)], dim=0)  # (2, W)
    feats = feats.unsqueeze(1).expand(2, int(height), lon_t.shape[-1])  # (2, H, W)
    out_dtype = dtype if dtype is not None else torch.float32
    return feats.to(dtype=out_dtype)


# ---------------------------------------------------------------------------
# Evaluation / plotting helpers
# ---------------------------------------------------------------------------


def add_cyclic_column(
    lon: Any,
    field: np.ndarray,
    axis: int = -1,
    *,
    only_if_periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a wrap-around column so plots don't show a false seam at 0°/360°.

    Adds a longitude at ``lon[0] + 360`` whose data equals the first column, so
    a global field plotted from lon[0]..lon[-1] visually closes the gap between
    the last grid point and the wrap. This is a *cosmetic* helper for plotting /
    evaluation only — it does not alter the model or stored outputs.
    """
    lon_arr = np.asarray(lon, dtype=np.float64)
    field_arr = np.asarray(field)
    if lon_arr.ndim != 1 or lon_arr.size < 2:
        return lon_arr, field_arr
    if only_if_periodic and not longitude_is_periodic(lon_arr):
        return lon_arr, field_arr
    # Do not add another wrap column to plotting data that already contains a
    # duplicated endpoint. Stored model outputs should never have one, but this
    # makes the visualisation helper safe for third-party files.
    if np.isclose(lon_arr[0] % 360.0, lon_arr[-1] % 360.0, rtol=0.0, atol=1e-7):
        return lon_arr, field_arr
    lon_ext = np.concatenate([lon_arr, [lon_arr[0] + 360.0]])
    first = np.take(field_arr, indices=[0], axis=axis)
    field_ext = np.concatenate([field_arr, first], axis=axis)
    return lon_ext, field_ext


def _safe_ratio(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def dateline_discontinuity_from_edges(
    first: Any,
    second: Any,
    penultimate: Any,
    last: Any,
    lon_periodic: bool = True,
) -> dict[str, float]:
    """Compute a low-memory seam diagnostic from four longitude slices.

    This is suitable for lazy xarray/NetCDF data: callers load only columns
    ``0``, ``1``, ``-2``, and ``-1`` instead of materialising a global
    time/member/level field.
    """
    first_arr = _to_numpy(first).astype(np.float64)
    second_arr = _to_numpy(second).astype(np.float64)
    penultimate_arr = _to_numpy(penultimate).astype(np.float64)
    last_arr = _to_numpy(last).astype(np.float64)
    seam = float(np.nanmean(np.abs(first_arr - last_arr)))
    left_grad = float(np.nanmean(np.abs(second_arr - first_arr)))
    right_grad = float(np.nanmean(np.abs(last_arr - penultimate_arr)))
    local_grad = 0.5 * (left_grad + right_grad)
    return {
        "seam_jump": seam,
        "left_grad": left_grad,
        "right_grad": right_grad,
        "local_grad": local_grad,
        "local_ratio": _safe_ratio(seam, local_grad),
        "lon_periodic": bool(lon_periodic),
    }


def dateline_discontinuity(field: Any, lon_periodic: bool = True) -> dict[str, float]:
    """Diagnose a longitude seam by comparing the wrap jump to interior gradients.

    For a smooth *periodic* field, the difference between the first and last
    longitude columns (which are physically one grid step apart) should be
    comparable to nearby longitudinal steps. Large global or local ratios flag
    a spurious seam.

    Args:
        field: Array with longitude as the last axis, shape ``(..., H, W)``.
        lon_periodic: Whether the field is expected to wrap (informational).

    Returns:
        Dict with ``seam_jump``, ``interior_grad`` and their ``ratio``.
    """
    f = _to_numpy(field)
    if f.shape[-1] < 2:
        return {
            "seam_jump": 0.0,
            "interior_grad": 0.0,
            "ratio": 0.0,
            "left_grad": 0.0,
            "right_grad": 0.0,
            "local_grad": 0.0,
            "local_ratio": 0.0,
            "lon_periodic": bool(lon_periodic),
        }
    edge_metrics = dateline_discontinuity_from_edges(
        f[..., 0], f[..., 1], f[..., -2], f[..., -1], lon_periodic=lon_periodic,
    )
    # Accumulate one longitude pair at a time so this diagnostic never creates
    # another full-width array. For lazy NetCDF/xarray inputs, use the edge-only
    # helper directly to avoid loading the original full field as well.
    interior_sum = 0.0
    interior_count = 0
    for index in range(f.shape[-1] - 1):
        delta = np.abs(
            np.asarray(f[..., index + 1], dtype=np.float64)
            - np.asarray(f[..., index], dtype=np.float64)
        )
        finite = np.isfinite(delta)
        interior_sum += float(np.sum(delta[finite]))
        interior_count += int(np.sum(finite))
    interior_mean = interior_sum / interior_count if interior_count else float("nan")
    return {
        **edge_metrics,
        "interior_grad": interior_mean,
        "ratio": _safe_ratio(edge_metrics["seam_jump"], interior_mean),
    }
