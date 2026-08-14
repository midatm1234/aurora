"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

NetCDF output for two-phase (deterministic + refined ensemble) Aurora rollouts.

Everything the deterministic workflow already preserves is preserved here —
initialization time, valid time, forecast lead time, rollout-step order,
variable names, pressure levels, latitude, longitude, units, calendar, masks,
fill values and attributes — and the Phase-2 products are added:

* ``<var>``: deterministic Aurora rollout.
* ``<var>_residual``: predicted correction in normalized target space (the
  legacy suffix is retained for compatibility).
* ``<var>_refined``: refined prediction (ensemble mean when M > 1).
* ``<var>_predicted_correction_physical``: physical correction, refined minus
  Aurora.
* ``<var>_members``: every ensemble member, with an explicit ``member`` dim.
* ``<var>_ensemble_mean``: ensemble mean.
* ``<var>_ensemble_spread``: unbiased ensemble standard deviation.
* ``<var>_truth``: ground truth, when provided.

The dataset-level ``aurora_default_forecast_variable_map`` identifies the
physical field evaluators must use. This resolves to ``<var>_refined`` when
present, while old rollout files containing only ``<var>`` remain readable.

I/O settings only affect *how* the data is stored: chunking and compression are
configurable, values and sample order are never changed.

Adapted from ``granitewxc.refinement.io`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), extended with Aurora's
forecast-time coordinates and pressure-level dimension.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from finetune.refinement.packing import FieldPacking

__all__ = [
    "build_refined_dataset",
    "resolve_forecast_variable",
    "write_refined_netcdf",
]

_RESIDUAL_UNITS = "1 (normalized target space)"
_ROLE_ATTR = "aurora_refinement_role"
_SOURCE_VARIABLE_ATTR = "aurora_source_variable"
_DEFAULT_VARIABLE_MAP_ATTR = "aurora_default_forecast_variable_map"
_SCHEMA_VERSION = "2"

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


def resolve_forecast_variable(dataset, variable: str):
    """Return the physical forecast product for ``variable``.

    Version-2 refinement files explicitly identify the default physical field
    in dataset metadata. Older two-phase files are recognized by their
    ``<variable>_refined`` / ``<variable>_ensemble_mean`` suffixes, while the
    plain variable remains the final backwards-compatible fallback used by
    legacy rollout files. A normalized residual or ensemble spread is never
    accepted as a forecast field.
    """
    variable = str(variable)
    declared = dataset.attrs.get(_DEFAULT_VARIABLE_MAP_ATTR)
    if declared:
        try:
            mapping = json.loads(str(declared))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset attribute {_DEFAULT_VARIABLE_MAP_ATTR!r} is not valid JSON."
            ) from exc
        name = mapping.get(variable)
        if name is not None:
            if name not in dataset.data_vars:
                raise KeyError(
                    f"Dataset declares {name!r} as the default physical forecast for "
                    f"{variable!r}, but that data variable is absent."
                )
            role = str(dataset[name].attrs.get(_ROLE_ATTR, ""))
            if role not in {
                "refined_forecast",
                "ensemble_mean",
                "legacy_forecast",
                "aurora_forecast",
            }:
                raise ValueError(
                    f"Dataset declares {name!r} as the forecast for {variable!r}, "
                    f"but its {_ROLE_ATTR!r} is {role!r}."
                )
            return dataset[name]

    # Attribute-based lookup also supports files that retained per-variable
    # metadata but lost the dataset-level map during concatenation.
    by_role: dict[str, list[str]] = {}
    for name, data_array in dataset.data_vars.items():
        source = str(data_array.attrs.get(_SOURCE_VARIABLE_ATTR, ""))
        if source != variable:
            continue
        role = str(data_array.attrs.get(_ROLE_ATTR, ""))
        by_role.setdefault(role, []).append(str(name))
    for role in (
        "refined_forecast",
        "ensemble_mean",
        "legacy_forecast",
        "aurora_forecast",
    ):
        names = by_role.get(role, [])
        if len(names) > 1:
            raise ValueError(
                f"Multiple {role!r} variables describe {variable!r}: {sorted(names)}."
            )
        if names:
            return dataset[names[0]]

    # Conventional names predate the explicit schema metadata.
    for name in (f"{variable}_refined", f"{variable}_ensemble_mean", variable):
        if name in dataset.data_vars:
            role = str(dataset[name].attrs.get(_ROLE_ATTR, ""))
            if role in {
                "predicted_correction_normalized",
                "predicted_correction_physical",
                "ensemble_spread",
                "truth",
            }:
                raise ValueError(
                    f"{name!r} is labelled as {role!r}, not a physical forecast field."
                )
            return dataset[name]
    raise KeyError(
        f"No physical refined forecast was found for {variable!r}. Available data "
        f"variables: {sorted(map(str, dataset.data_vars))}."
    )


def _validate_field_shape(
    name: str,
    value,
    *,
    expected: tuple[int, int, int, int],
    with_members: bool = False,
) -> np.ndarray | None:
    data = _as_numpy(value)
    if data is None:
        return None
    if with_members:
        if data.ndim != 5:
            raise ValueError(
                f"{name} must be [rollout_step, member, channel, lat, lon], "
                f"got {data.shape}."
            )
        if data.shape[1] < 1:
            raise ValueError(f"{name} must contain at least one ensemble member.")
        observed = (data.shape[0], data.shape[2], data.shape[3], data.shape[4])
    else:
        if data.ndim != 4:
            raise ValueError(
                f"{name} must be [rollout_step, channel, lat, lon], got {data.shape}."
            )
        observed = tuple(data.shape)
    if observed != expected:
        raise ValueError(
            f"{name} has non-member shape {observed}, expected {expected} "
            "([rollout_step, channel, lat, lon])."
        )
    return data


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
    expected_shape = (steps, channels, height, width)
    refined_data = _validate_field_shape("refined", refined, expected=expected_shape)
    residual_data = _validate_field_shape("residual", residual, expected=expected_shape)
    ensemble_mean_data = _validate_field_shape(
        "ensemble_mean", ensemble_mean, expected=expected_shape
    )
    ensemble_spread_data = _validate_field_shape(
        "ensemble_spread", ensemble_spread, expected=expected_shape
    )
    truth_data = _validate_field_shape("truth", truth, expected=expected_shape)
    member_data = _validate_field_shape(
        "members", members, expected=expected_shape, with_members=True
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

    if packing.lat and len(packing.lat) != height:
        raise ValueError(
            f"Packing latitude has {len(packing.lat)} values but fields have height {height}."
        )
    if packing.lon and len(packing.lon) != width:
        raise ValueError(
            f"Packing longitude has {len(packing.lon)} values but fields have width {width}."
        )

    init_values = np.asarray(init_time)
    valid_values = np.asarray(valid_time)
    lead_values = np.asarray(lead_time_hours, dtype="float64")
    if not np.isfinite(lead_values).all() or np.any(lead_values < 0):
        raise ValueError("lead_time_hours must contain finite, non-negative values.")
    try:
        init_ns = init_values.astype("datetime64[ns]")
        valid_ns = valid_values.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError("init_time and valid_time must contain valid datetimes.") from exc
    if np.isnat(init_ns).any() or np.isnat(valid_ns).any():
        raise ValueError("init_time and valid_time must not contain NaT.")
    actual_lead_hours = (valid_ns - init_ns) / np.timedelta64(1, "h")
    if not np.allclose(actual_lead_hours, lead_values, atol=1.0e-6, rtol=0):
        mismatch = int(np.flatnonzero(~np.isclose(
            actual_lead_hours, lead_values, atol=1.0e-6, rtol=0
        ))[0])
        raise ValueError(
            "Forecast-time mismatch at rollout step "
            f"{mismatch + 1}: init_time={init_ns[mismatch]}, "
            f"valid_time={valid_ns[mismatch]}, derived lead="
            f"{actual_lead_hours[mismatch]:g} h, stored lead={lead_values[mismatch]:g} h."
        )

    surface_vars = [spec.aurora_name for spec in packing.channels if spec.kind == "surf"]
    atmos_vars = list(
        dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "atmos")
    )

    dataset_names: dict[str, str] = {}
    for spec in packing.channels:
        existing = dataset_names.setdefault(spec.dataset_name, spec.aurora_name)
        if existing != spec.aurora_name:
            raise ValueError(
                f"Dataset variable name {spec.dataset_name!r} maps to both "
                f"{existing!r} and {spec.aurora_name!r}."
            )
    if atmos_vars:
        shared_levels = packing.levels_for(atmos_vars[0])
        for name in atmos_vars[1:]:
            levels = packing.levels_for(name)
            if levels != shared_levels:
                raise ValueError(
                    "All atmospheric output variables currently share one NetCDF "
                    f"{level_dim!r} coordinate, but {atmos_vars[0]!r} has levels "
                    f"{list(shared_levels)} and {name!r} has {list(levels)}."
                )

    def _emit(
        array,
        suffix: str,
        extra: Mapping[str, Any] | None,
        *,
        role: str,
        with_members: bool = False,
    ):
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
            attributes = {
                **dict(extra or {}),
                _ROLE_ATTR: role,
                _SOURCE_VARIABLE_ATTR: spec.dataset_name,
            }
            out[f"{spec.dataset_name}{suffix}"] = xr.DataArray(
                slab, dims=dims, attrs=_attrs(spec.units, attributes)
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
            attributes = {
                **dict(extra or {}),
                _ROLE_ATTR: role,
                _SOURCE_VARIABLE_ATTR: specs[0].dataset_name,
            }
            out[f"{specs[0].dataset_name}{suffix}"] = xr.DataArray(
                slab, dims=dims, attrs=_attrs(specs[0].units, attributes)
            )
        return out

    data_vars: dict[str, Any] = {}
    data_vars.update(
        _emit(
            det,
            "",
            {"long_name": "deterministic Aurora rollout"},
            role="aurora_forecast",
        )
    )
    if residual_data is not None:
        data_vars.update(
            _emit(
                residual_data,
                "_residual",
                {
                    "long_name": "predicted normalized correction (legacy residual name)",
                    "units": _RESIDUAL_UNITS,
                    "comment": _ROLLOUT_NOTE,
                },
                role="predicted_correction_normalized",
            )
        )
    if refined_data is not None:
        data_vars.update(
            _emit(
                refined_data,
                "_refined",
                {"long_name": "refined physical forecast"},
                role="refined_forecast",
            )
        )
    if ensemble_mean_data is not None:
        data_vars.update(
            _emit(
                ensemble_mean_data,
                "_ensemble_mean",
                {"long_name": "ensemble-mean refined physical forecast"},
                role="ensemble_mean",
            )
        )
    default_refined = refined_data if refined_data is not None else ensemble_mean_data
    if default_refined is not None:
        data_vars.update(
            _emit(
                default_refined - det,
                "_predicted_correction_physical",
                {
                    "long_name": "predicted physical-space correction applied to Aurora",
                    "comment": "refined_forecast = Aurora + predicted_correction_physical",
                },
                role="predicted_correction_physical",
            )
        )
    if ensemble_spread_data is not None:
        data_vars.update(
            _emit(
                ensemble_spread_data,
                "_ensemble_spread",
                {"long_name": "ensemble standard deviation (unbiased)"},
                role="ensemble_spread",
            )
        )
    if truth_data is not None:
        data_vars.update(
            _emit(
                truth_data,
                "_truth",
                {"long_name": "CAMS ground truth"},
                role="truth",
            )
        )
    if member_data is not None:
        data_vars.update(
            _emit(
                member_data,
                "_members",
                {"long_name": "refined ensemble members in draw order"},
                role="ensemble_members",
                with_members=True,
            )
        )

    coords: dict[str, Any] = {
        step_dim: np.arange(1, steps + 1, dtype="int32"),
        "init_time": (step_dim, init_ns),
        "time": (step_dim, valid_ns),
        "lead_time_hours": (step_dim, lead_values),
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
    if member_data is not None:
        coords[member_dim] = np.arange(member_data.shape[1], dtype="int32")

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
    dataset.attrs.setdefault("aurora_refinement_schema_version", _SCHEMA_VERSION)
    dataset.attrs.setdefault("correction_target_convention", "CAMS - Aurora")
    dataset.attrs.setdefault(
        "refined_forecast_convention",
        "Aurora + predicted_correction_physical",
    )
    default_suffix = (
        "_refined"
        if refined_data is not None
        else "_ensemble_mean" if ensemble_mean_data is not None else ""
    )
    default_map = {
        dataset_name: f"{dataset_name}{default_suffix}"
        for dataset_name in dataset_names
    }
    dataset.attrs.setdefault(
        _DEFAULT_VARIABLE_MAP_ATTR,
        json.dumps(default_map, sort_keys=True),
    )
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
