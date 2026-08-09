"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

NetCDF output for two-phase (deterministic + refined ensemble) Aurora rollouts.

Everything the deterministic workflow already preserves is preserved here —
initialization time, valid time, forecast lead time, rollout-step order,
variable names, pressure levels, latitude, longitude, units, calendar, masks,
fill values and attributes — and the Phase-2 products are added:

===========================  ==============================================
``<var>``                    deterministic Aurora rollout
``<var>_residual``           predicted residual (normalized target space)
``<var>_refined``            refined prediction (ensemble mean when M > 1)
``<var>_members``            every ensemble member, explicit ``member`` dim
``<var>_ensemble_mean``      ensemble mean
``<var>_ensemble_spread``    unbiased ensemble standard deviation
``<var>_truth``              ground truth, when provided
===========================  ==============================================

I/O settings only affect *how* the data is stored: chunking and compression are
configurable, values and sample order are never changed.

Adapted from ``granitewxc.refinement.io`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), extended with Aurora's
forecast-time coordinates and pressure-level dimension.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from finetune.refinement.packing import FieldPacking

__all__ = ["build_refined_dataset", "write_refined_netcdf"]

_RESIDUAL_UNITS = "1 (normalized target space)"

_ROLLOUT_NOTE = (
    "Phase-2 stochastic residual refinement of a deterministic Aurora rollout. "
    "The residual is constructed, predicted and added in Aurora's normalized "
    "target space; inverse normalization and physical constraints are applied "
    "exactly once, afterwards. Refinement is postprocessing of each rollout "
    "step and does not alter the deterministic state used for later steps."
)


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def build_refined_dataset(
    *,
    packing: FieldPacking,
    deterministic,
    init_time,
    valid_time,
    lead_time_hours,
    refined=None,
    residual=None,
    members=None,
    ensemble_mean=None,
    ensemble_spread=None,
    truth=None,
    attrs: Mapping[str, Any] | None = None,
    step_dim: str = "rollout_step",
    level_dim: str = "level",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    member_dim: str = "member",
):
    """Assemble an :class:`xarray.Dataset` from packed two-phase outputs.

    Array layout expectations (rollout-step order is preserved verbatim):

    * ``deterministic`` / ``refined`` / ``residual`` / ``ensemble_*`` / ``truth``:
      ``[rollout_step, channel, lat, lon]``
    * ``members``: ``[rollout_step, member, channel, lat, lon]`` in draw order
    """
    import xarray as xr

    det = _as_numpy(deterministic)
    if det.ndim != 4:
        raise ValueError(
            f"deterministic must be [rollout_step, channel, lat, lon], got {det.shape}"
        )
    steps, channels, height, width = det.shape
    if channels != packing.num_channels:
        raise ValueError(
            f"deterministic has {channels} channels but the packing declares "
            f"{packing.num_channels}."
        )
    for name, value in (
        ("init_time", init_time),
        ("valid_time", valid_time),
        ("lead_time_hours", lead_time_hours),
    ):
        if len(value) != steps:
            raise ValueError(
                f"{name} has {len(value)} entries but there are {steps} rollout steps."
            )

    surface_vars = [spec.aurora_name for spec in packing.channels if spec.kind == "surf"]
    atmos_vars = list(
        dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "atmos")
    )

    def _emit(array, suffix: str, extra: Mapping[str, Any] | None, with_members: bool = False):
        data = _as_numpy(array)
        if data is None:
            return {}
        channel_axis = 2 if with_members else 1
        if data.shape[channel_axis] != packing.num_channels:
            raise ValueError(
                f"{suffix or 'field'} has {data.shape[channel_axis]} channels but "
                f"{packing.num_channels} were declared."
            )
        out: dict[str, Any] = {}
        dims: tuple[str, ...]
        for name in surface_vars:
            index = packing.index_of(name, None)
            spec = packing.channels[index]
            slab = data[:, :, index] if with_members else data[:, index]
            dims = (
                (step_dim, member_dim, lat_dim, lon_dim)
                if with_members
                else (step_dim, lat_dim, lon_dim)
            )
            out[f"{spec.dataset_name}{suffix}"] = xr.DataArray(
                slab, dims=dims, attrs=_attrs(spec.units, extra)
            )
        for name in atmos_vars:
            specs = packing.channels_for(name)
            indices = [spec.index for spec in specs]
            slab = data[:, :, indices] if with_members else data[:, indices]
            dims = (
                (step_dim, member_dim, level_dim, lat_dim, lon_dim)
                if with_members
                else (step_dim, level_dim, lat_dim, lon_dim)
            )
            out[f"{specs[0].dataset_name}{suffix}"] = xr.DataArray(
                slab, dims=dims, attrs=_attrs(specs[0].units, extra)
            )
        return out

    data_vars: dict[str, Any] = {}
    data_vars.update(_emit(det, "", {"long_name": "deterministic Aurora rollout"}))
    if residual is not None:
        data_vars.update(
            _emit(
                residual,
                "_residual",
                {
                    "long_name": "predicted Phase-2 residual",
                    "units": _RESIDUAL_UNITS,
                    "comment": _ROLLOUT_NOTE,
                },
            )
        )
    if refined is not None:
        data_vars.update(_emit(refined, "_refined", {"long_name": "refined prediction"}))
    if ensemble_mean is not None:
        data_vars.update(_emit(ensemble_mean, "_ensemble_mean", {"long_name": "ensemble mean"}))
    if ensemble_spread is not None:
        data_vars.update(
            _emit(
                ensemble_spread,
                "_ensemble_spread",
                {"long_name": "ensemble standard deviation (unbiased)"},
            )
        )
    if truth is not None:
        data_vars.update(_emit(truth, "_truth", {"long_name": "ground truth"}))
    if members is not None:
        member_data = _as_numpy(members)
        if member_data.ndim != 5:
            raise ValueError(
                "members must be [rollout_step, member, channel, lat, lon], got "
                f"shape {member_data.shape}"
            )
        data_vars.update(
            _emit(
                member_data,
                "_members",
                {"long_name": "refined ensemble members in draw order"},
                with_members=True,
            )
        )

    coords: dict[str, Any] = {
        step_dim: np.arange(1, steps + 1, dtype="int32"),
        "init_time": (step_dim, np.asarray(init_time)),
        "time": (step_dim, np.asarray(valid_time)),
        "lead_time_hours": (step_dim, np.asarray(lead_time_hours, dtype="float64")),
    }
    if packing.lat:
        coords[lat_dim] = np.asarray(packing.lat, dtype="float64")
    else:
        coords[lat_dim] = np.arange(height, dtype="float64")
    if packing.lon:
        coords[lon_dim] = np.asarray(packing.lon, dtype="float64")
    else:
        coords[lon_dim] = np.arange(width, dtype="float64")
    if atmos_vars:
        levels = packing.levels_for(atmos_vars[0])
        coords[level_dim] = np.asarray(levels, dtype="float64")
    if members is not None:
        coords[member_dim] = np.arange(_as_numpy(members).shape[1], dtype="int32")

    dataset = xr.Dataset(data_vars, coords=coords)
    dataset[step_dim].attrs.update({"long_name": "autoregressive rollout step index (1-based)"})
    dataset["init_time"].attrs.update({"long_name": "forecast initialization time"})
    dataset["time"].attrs.update({"long_name": "forecast valid time"})
    dataset["lead_time_hours"].attrs.update({"long_name": "forecast lead time", "units": "hours"})
    if lat_dim in dataset.coords:
        dataset[lat_dim].attrs.update({"units": "degrees_north"})
    if lon_dim in dataset.coords:
        dataset[lon_dim].attrs.update({"units": "degrees_east"})
    if level_dim in dataset.coords:
        dataset[level_dim].attrs.update({"units": "hPa", "positive": "down"})
    dataset.attrs.update(dict(attrs or {}))
    dataset.attrs.setdefault("refinement_note", _ROLLOUT_NOTE)
    return dataset


def _attrs(units: str, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    attributes: dict[str, Any] = {"units": units or ""}
    if extra:
        attributes.update(extra)
    return attributes


def write_refined_netcdf(
    dataset,
    path: str | os.PathLike,
    *,
    compression: bool = True,
    compression_level: int = 4,
    chunk_sizes: Mapping[str, int] | None = None,
    atomic: bool = True,
    unlimited_dims: Sequence[str] | None = None,
) -> str:
    """Write ``dataset`` once, with configurable chunking and compression.

    The file is produced in a single pass (never opened and rewritten per
    variable or per ensemble member), optionally through a temporary file so a
    crash cannot leave a partial output in place.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in dataset.data_vars.items():
        spec: dict[str, Any] = {}
        if compression:
            spec.update({"zlib": True, "complevel": int(compression_level)})
        if chunk_sizes:
            sizes = [int(chunk_sizes.get(dim, dataset.sizes[dim])) for dim in variable.dims]
            spec["chunksizes"] = tuple(
                max(1, min(s, dataset.sizes[d])) for s, d in zip(sizes, variable.dims)
            )
        if np.issubdtype(variable.dtype, np.floating):
            spec.setdefault("_FillValue", np.nan)
        if spec:
            encoding[name] = spec

    kwargs: dict[str, Any] = {"encoding": encoding, "format": "NETCDF4"}
    if unlimited_dims:
        kwargs["unlimited_dims"] = list(unlimited_dims)

    if not atomic:
        dataset.to_netcdf(path, **kwargs)
        return path

    fd, tmp = tempfile.mkstemp(prefix=".nc-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        dataset.to_netcdf(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
