"""
Create a GIF animation from Aurora hourly rollout netCDF files.

Layout: 3 columns × 2 rows (5 ensemble members + ensemble mean).
Each frame is one hourly lead time.  Output is saved as a GIF.

Usage:
    python examples/animate_rollout.py

All configuration lives in the CONFIG block below.
"""

# ============================================================
#  CONFIG — edit here
# ============================================================
ROLLOUT_DIR   = "./outputs/ecmwf_hourly_rollouts"
INIT_DAY      = "2026-06-01"       # must match a rollout_<INIT_DAY>_d*.nc set
FORECAST_DAYS = [1, 2]             # which day-files to animate (1-based); [] = all

VARIABLE      = "scaled_tp_1h"     # netCDF variable name to plot
# Inverse transform applied to raw values before plotting.
# The rollout already applies the inverse log-transform, so outputs are in
# physical units (m/h for precipitation).  Use "identity" for all rollout
# variables.  Use "log_untransform" only if you are reading raw model-space
# values that have not yet been un-transformed.
TRANSFORM     = "identity"
SCALE_FACTOR  = 1000.0             # multiply after transform (e.g. m/h -> mm/h)
UNITS_LABEL   = "mm / h"

# Colourmap and fixed colour limits (None = per-frame auto-scale)
CMAP           = "YlGnBu"
VMIN           = 0.0
VMAX           = None              # None = 99th-percentile across all frames (auto)
# Values at or below ZERO_THRESHOLD are rendered white (no precipitation).
# Set to 0.0 to colour every non-negative value.
ZERO_THRESHOLD = 0.01              # mm / h

# Map extent [lon_min, lon_max, lat_min, lat_max] or None for global
MAP_EXTENT    = None               # e.g. [-130, -60, 20, 55] for CONUS

# Animation
FRAME_DURATION_MS = 200            # ms per frame
OUTPUT_GIF    = f"./outputs/{INIT_DAY}_{VARIABLE}.gif"

# Figure layout
FIG_WIDTH     = 18   # inches
FIG_HEIGHT    = 9    # inches
DPI           = 100
# ============================================================

import io
import os
import sys
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import netCDF4 as nc
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ── Inverse transforms ────────────────────────────────────────────────────────
_LOG_EPS = 1e-3

def _log_untransform(x: np.ndarray) -> np.ndarray:
    return _LOG_EPS * (np.exp(x.astype(np.float64)) - 1.0)

def _identity(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64)

TRANSFORMS = {
    "log_untransform": _log_untransform,
    "identity":        _identity,
}

if TRANSFORM not in TRANSFORMS:
    sys.exit(f"Unknown TRANSFORM {TRANSFORM!r}. Choose from: {list(TRANSFORMS)}")
apply_transform = TRANSFORMS[TRANSFORM]


# ── Locate files ─────────────────────────────────────────────────────────────
rollout_dir = Path(ROLLOUT_DIR)
if FORECAST_DAYS:
    nc_files = sorted([rollout_dir / f"rollout_{INIT_DAY}_d{d:02d}.nc"
                       for d in FORECAST_DAYS])
else:
    nc_files = sorted(rollout_dir.glob(f"rollout_{INIT_DAY}_d*.nc"))

missing = [f for f in nc_files if not f.exists()]
if missing:
    sys.exit(f"Files not found:\n" + "\n".join(str(f) for f in missing))

print(f"Animating {VARIABLE}  |  init day {INIT_DAY}  |  {len(nc_files)} day-file(s)")


# ── Build per-frame data list ─────────────────────────────────────────────────
# Each entry: (valid_time_str, field_per_panel)
# field_per_panel: list of 6 arrays [ens0..ens4, mean], each (lat, lon)

frames = []   # list of (label, list_of_6_2d_arrays)
lat = lon = None

for nc_path in nc_files:
    with nc.Dataset(str(nc_path)) as ds:
        if VARIABLE not in ds.variables:
            sys.exit(f"{VARIABLE!r} not found in {nc_path.name}.\n"
                     f"Available variables: {list(ds.variables)}")

        if lat is None:
            lat = ds.variables["latitude"][:]
            lon = ds.variables["longitude"][:]

        n_members_file  = ds.dimensions["ensemble"].size
        n_leads         = ds.dimensions["lead_minutes"].size
        lead_mins_arr   = ds.variables["lead_minutes"][:]
        time_arr        = ds.variables["time"][:]          # seconds since 1970-01-01
        init_time_sec   = int(ds.variables["init_time"][0])

        # Load all ensemble data for this file at once: (ensemble, init_time, lead, lat, lon)
        raw = ds.variables[VARIABLE][:, 0, :, :, :]  # -> (n_ens, n_leads, lat, lon)

    for h_idx in range(n_leads):
        raw_h   = raw[:, h_idx, :, :]                 # (n_ens, lat, lon)
        phys    = apply_transform(raw_h) * SCALE_FACTOR

        panels = [phys[m] for m in range(min(5, n_members_file))]
        # Pad to 5 if fewer members
        while len(panels) < 5:
            panels.append(np.full_like(panels[0], np.nan))
        panels.append(np.nanmean(phys, axis=0))        # ensemble mean (6th panel)

        valid_sec = int(time_arr[h_idx])
        lead_h    = (valid_sec - init_time_sec) / 3600.0
        epoch     = np.datetime64("1970-01-01", "s")
        valid_dt  = str(epoch + np.timedelta64(valid_sec, "s"))
        label     = f"Init: {str(epoch + np.timedelta64(init_time_sec, 's'))[:13]}Z  |  " \
                    f"Valid: {valid_dt[:16]}Z  |  Lead: +{lead_h:.0f} h"
        frames.append((label, panels))

print(f"Total frames: {len(frames)}")

# ── Compute consistent colour limits across all frames ────────────────────────
# If VMIN/VMAX are None, derive them from all frames so the colourbar is stable.
_vmin = VMIN
_vmax = VMAX
if _vmin is None or _vmax is None:
    all_vals = np.concatenate([
        p[np.isfinite(p)].ravel()
        for _, panels in frames
        for p in panels
    ])
    all_vals = all_vals[all_vals > 0]   # ignore exact zeros for scaling
    if _vmin is None:
        _vmin = 0.0
    if _vmax is None:
        _vmax = float(np.percentile(all_vals, 99)) if all_vals.size else 1.0
    print(f"Auto colour limits: vmin={_vmin:.4f}  vmax={_vmax:.4f}  [{UNITS_LABEL}]")


# ── Plotting helpers ──────────────────────────────────────────────────────────
PANEL_TITLES = ["Member 1", "Member 2", "Member 3",
                "Member 4", "Member 5", "Ensemble Mean"]
PROJ         = ccrs.PlateCarree()

# Build colourmap: white below threshold, darkest colour above vmax
import matplotlib.colors as mcolors
_cmap = plt.get_cmap(CMAP).copy()
_cmap.set_under("white")
_cmap.set_over(_cmap(1.0))   # darkest colour of the map for values above vmax

def make_frame(label, panels) -> Image.Image:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.12, wspace=0.05,
                            left=0.04, right=0.88,
                            top=0.90, bottom=0.04)

    vmin = max(_vmin, ZERO_THRESHOLD)   # colourmap starts at threshold
    vmax = _vmax

    axes = []
    for idx in range(6):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col], projection=PROJ)
        axes.append(ax)

        if MAP_EXTENT:
            ax.set_extent(MAP_EXTENT, crs=PROJ)
        else:
            ax.set_global()

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")

        im = ax.pcolormesh(lon, lat, panels[idx],
                           transform=PROJ,
                           cmap=_cmap, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)

        ax.set_title(PANEL_TITLES[idx], fontsize=9, pad=3)

        # Highlight ensemble-mean panel
        if idx == 5:
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

    # Shared colourbar on the right
    cbar_ax = fig.add_axes([0.90, 0.10, 0.015, 0.75])
    sm = plt.cm.ScalarMappable(cmap=_cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, extend="max")
    cbar.set_label(f"{VARIABLE}  [{UNITS_LABEL}]", fontsize=9)

    fig.suptitle(label, fontsize=10, fontweight="bold", y=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Render frames & save GIF ──────────────────────────────────────────────────
out_path = Path(OUTPUT_GIF)
out_path.parent.mkdir(parents=True, exist_ok=True)

print("Rendering frames ...", end="", flush=True)
images = []
for i, (label, panels) in enumerate(frames):
    images.append(make_frame(label, panels))
    if (i + 1) % 10 == 0:
        print(f" {i+1}/{len(frames)}", end="", flush=True)
print()

print(f"Saving GIF -> {out_path}")
images[0].save(
    str(out_path),
    save_all=True,
    append_images=images[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
    optimize=False,
)
print(f"Done. {len(images)} frames, {out_path.stat().st_size / 1e6:.1f} MB")
