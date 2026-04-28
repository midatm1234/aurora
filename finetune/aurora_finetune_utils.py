"""Utilities for config-driven Aurora fine-tuning and rollout workflows."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import math
import pickle
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import xarray as xr

from aurora import Batch, Metadata

try:
    import yaml
except ImportError:  # pragma: no cover - dependency check is runtime-facing.
    yaml = None


__all__ = [
    "VariableSpec",
    "ResolvedVariableSpecs",
    "load_config",
    "set_seed",
    "open_dataset",
    "merge_external_static_vars",
    "resolve_variable_specs",
    "derive_model_variable_config",
    "build_training_samples",
    "build_aurora_batch",
    "build_targets",
    "configure_trainable_parameters",
    "create_optimizer",
    "create_scheduler",
    "compute_supervised_loss",
    "compute_target_normalization_stats",
    "run_validation",
    "run_rollout",
    "save_checkpoint",
    "load_checkpoint_if_available",
    "write_training_history",
    "write_run_manifest",
    "save_predictions",
]


@dataclass(frozen=True)
class VariableSpec:
    """Resolved variable metadata for notebook fine-tuning pipelines."""

    dataset_name: str
    aurora_name: str
    kind: str  # surf | atmos | static


@dataclass(frozen=True)
class ResolvedVariableSpecs:
    """Collection of resolved variable specs used to build batches and targets."""

    predictors: tuple[VariableSpec, ...]
    targets: tuple[VariableSpec, ...]
    static: tuple[VariableSpec, ...]

    @property
    def predictor_by_aurora(self) -> dict[str, VariableSpec]:
        return {spec.aurora_name: spec for spec in self.predictors}

    @property
    def target_by_aurora(self) -> dict[str, VariableSpec]:
        return {spec.aurora_name: spec for spec in self.targets}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dim_names(config: dict[str, Any]) -> tuple[str, str, str, str]:
    data_cfg = config.get("data", {})
    time_dim = data_cfg.get("time_dim", "time")
    lat_dim = data_cfg.get("lat_dim", "latitude")
    lon_dim = data_cfg.get("lon_dim", "longitude")
    level_dim = data_cfg.get("level_dim", "level")
    return time_dim, lat_dim, lon_dim, level_dim


def _to_python_datetime(value: Any) -> datetime:
    # `tolist()` on numpy datetime64 may already produce datetime.
    if isinstance(value, datetime):
        return value
    return np.datetime64(value, "s").tolist()


def _ensure_monotonic_lat_lon(ds: xr.Dataset, lat_dim: str, lon_dim: str) -> xr.Dataset:
    if ds[lat_dim].ndim != 1 or ds[lon_dim].ndim != 1:
        raise ValueError(
            "Regional fine-tuning notebook helper currently expects 1D latitude/longitude "
            f"coordinates, got dims {ds[lat_dim].dims} and {ds[lon_dim].dims}."
        )

    # Aurora metadata requires descending latitude.
    lat_values = np.asarray(ds[lat_dim].values)
    if lat_values.size < 2:
        raise ValueError("Latitude coordinate must contain at least two points.")
    if not np.all(np.diff(lat_values) < 0):
        ds = ds.sortby(lat_dim, ascending=False)

    # Aurora metadata requires longitudes in [0, 360) and strictly increasing.
    lon_values = np.asarray(ds[lon_dim].values, dtype=np.float64)
    lon_values = np.mod(lon_values, 360.0)
    order = np.argsort(lon_values)
    lon_sorted = lon_values[order]
    ds = ds.isel({lon_dim: order}).assign_coords({lon_dim: lon_sorted})

    # Drop any duplicate longitudes that may arise after modulo conversion.
    rounded = np.round(lon_sorted, 8)
    unique_values, unique_indices = np.unique(rounded, return_index=True)
    if unique_values.size != lon_sorted.size:
        ds = ds.isel({lon_dim: np.sort(unique_indices)})

    lat_values = np.asarray(ds[lat_dim].values)
    lon_values = np.asarray(ds[lon_dim].values)
    if not np.all(np.diff(lat_values) < 0):
        raise ValueError("Latitudes must be strictly decreasing after preprocessing.")
    if not np.all(np.diff(lon_values) > 0):
        raise ValueError("Longitudes must be strictly increasing after preprocessing.")
    if lon_values.min() < 0 or lon_values.max() >= 360:
        raise ValueError("Longitudes must lie in [0, 360) after preprocessing.")

    return ds


def _subset_domain(
    ds: xr.Dataset,
    config: dict[str, Any],
    lat_dim: str,
    lon_dim: str,
) -> xr.Dataset:
    data_cfg = config.get("data", {})
    domain_type = str(data_cfg.get("domain_type", "global")).lower()

    if domain_type not in {"global", "regional"}:
        raise ValueError(f"Unsupported data.domain_type: {domain_type!r}.")

    if domain_type == "global":
        return ds

    lat_min = data_cfg.get("lat_min")
    lat_max = data_cfg.get("lat_max")
    lon_min = data_cfg.get("lon_min")
    lon_max = data_cfg.get("lon_max")
    if None in {lat_min, lat_max, lon_min, lon_max}:
        raise ValueError(
            "Regional mode requires data.lat_min, data.lat_max, data.lon_min, and data.lon_max."
        )

    lat_hi = float(max(lat_min, lat_max))
    lat_lo = float(min(lat_min, lat_max))
    ds = ds.sel({lat_dim: slice(lat_hi, lat_lo)})

    lon_min_mod = float(lon_min) % 360.0
    lon_max_mod = float(lon_max) % 360.0
    if lon_min_mod > lon_max_mod:
        raise ValueError(
            "Regional longitude range crosses the prime meridian/date line after conversion to "
            "[0, 360). Please choose a non-wrapping regional longitude interval."
        )

    ds = ds.sel({lon_dim: slice(lon_min_mod, lon_max_mod)})
    if ds.sizes.get(lat_dim, 0) == 0 or ds.sizes.get(lon_dim, 0) == 0:
        raise ValueError(
            "Regional domain subset is empty. Please review latitude/longitude bounds."
        )

    return ds


def _resolve_path(path_str: str, project_root: Path) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load and merge YAML config with optional in-notebook overrides."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for this workflow. Install with `pip install pyyaml` and retry."
        )

    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")

    if overrides:
        config = _deep_merge(config, overrides)

    paths_cfg = config.setdefault("paths", {})
    project_root = Path(paths_cfg.get("project_root", config_path.parent)).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    paths_cfg["project_root"] = str(project_root)

    path_keys = [
        "data_dir",
        "train_data_path",
        "val_data_path",
        "test_data_path",
        "output_dir",
        "checkpoint_dir",
        "pretrained_checkpoint",
        "optional_resume_checkpoint",
    ]
    for key in path_keys:
        value = paths_cfg.get(key)
        if not value:
            continue
        paths_cfg[key] = _resolve_path(str(value), project_root)

    output_dir = Path(paths_cfg.get("output_dir", project_root / "outputs"))
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", output_dir / "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    paths_cfg["output_dir"] = str(output_dir.resolve())
    paths_cfg["checkpoint_dir"] = str(checkpoint_dir.resolve())

    return config


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def open_dataset(path: str | Path, config: dict[str, Any]) -> xr.Dataset:
    """Open an xarray dataset and apply Aurora-compatible coordinate/domain handling."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data_cfg = config.get("data", {})
    backend = str(data_cfg.get("backend") or data_cfg.get("data_format") or "auto").lower()

    if backend == "zarr" or path.suffix == ".zarr":
        ds = xr.open_zarr(str(path), consolidated=False)
    else:
        # Default to NetCDF-compatible open.
        ds = xr.open_dataset(str(path), engine=data_cfg.get("xarray_engine", None))

    time_dim, lat_dim, lon_dim, _ = _dim_names(config)
    for required_dim in (time_dim, lat_dim, lon_dim):
        if required_dim not in ds.coords and required_dim not in ds.dims:
            raise ValueError(
                f"Dataset {path} is missing required coordinate/dimension `{required_dim}`."
            )

    ds = _ensure_monotonic_lat_lon(ds, lat_dim=lat_dim, lon_dim=lon_dim)
    ds = _subset_domain(ds, config, lat_dim=lat_dim, lon_dim=lon_dim)
    ds = _ensure_monotonic_lat_lon(ds, lat_dim=lat_dim, lon_dim=lon_dim)

    return ds


def merge_external_static_vars(
    ds: xr.Dataset,
    static_path: str | Path,
    config: dict[str, Any],
) -> xr.Dataset:
    """Load static variables from an external pickle file and merge into *ds*.

    Follows the pattern from ``cams_prediction_local.ipynb`` where static
    variables (land-sea mask, orography, soil type, etc.) are stored in a
    separate pickle downloaded from HuggingFace rather than being embedded in
    the training NetCDF/Zarr files.

    For each entry in ``config["data"]["static_variables"]`` that is **not**
    already present in *ds*, the function looks up the array in the pickle
    (first by ``dataset_name``, then by ``aurora_name``) and injects it as a
    new variable on the dataset's lat/lon grid.  When the pickle covers a
    larger domain than *ds* (e.g. global vs. regional), the array is subset
    to match using nearest-neighbour coordinate selection.

    Returns a (shallow-copy) dataset with the additional variables.
    """
    static_path = Path(static_path).expanduser().resolve()
    if not static_path.exists():
        raise FileNotFoundError(f"Static data file not found: {static_path}")

    with open(static_path, "rb") as f:
        static_data = pickle.load(f)
    if not isinstance(static_data, dict):
        raise TypeError(f"Expected dict from static pickle, got {type(static_data)}")

    _, lat_dim, lon_dim, _ = _dim_names(config)
    static_vars_cfg = config.get("data", {}).get("static_variables", [])

    ds_lat = ds[lat_dim].values
    ds_lon = ds[lon_dim].values

    new_vars: dict[str, xr.DataArray] = {}
    for item in static_vars_cfg:
        if isinstance(item, str):
            dataset_name = item
            aurora_name = item
        elif isinstance(item, dict):
            dataset_name = str(item.get("dataset_name") or item.get("name") or "")
            aurora_name = str(item.get("aurora_name") or dataset_name)
        else:
            continue

        if dataset_name in ds:
            continue

        if dataset_name in static_data:
            arr = static_data[dataset_name]
        elif aurora_name in static_data:
            arr = static_data[aurora_name]
        else:
            raise KeyError(
                f"Static variable '{dataset_name}' (aurora_name='{aurora_name}') "
                f"not found in pickle. Available keys: {sorted(static_data.keys())}"
            )

        arr = np.asarray(arr, dtype=np.float32)

        if arr.shape == (len(ds_lat), len(ds_lon)):
            # Grids already match — assign directly.
            new_vars[dataset_name] = xr.DataArray(
                arr,
                dims=[lat_dim, lon_dim],
                coords={lat_dim: ds_lat, lon_dim: ds_lon},
            )
        else:
            # Pickle covers a different (typically global) grid.  Build a
            # full-resolution DataArray with a regular lat/lon grid inferred
            # from the array shape and select the subset that matches *ds*.
            n_lat, n_lon = arr.shape
            full_lat = np.linspace(90, -90, n_lat, dtype=np.float64)
            full_lon = np.linspace(0, 360 - 360 / n_lon, n_lon, dtype=np.float64)
            full_da = xr.DataArray(
                arr,
                dims=[lat_dim, lon_dim],
                coords={lat_dim: full_lat, lon_dim: full_lon},
            )
            # Use .values so the result carries *ds* coordinates exactly,
            # avoiding NaN from floating-point coordinate misalignment.
            subset_vals = full_da.sel(
                {lat_dim: ds_lat, lon_dim: ds_lon},
                method="nearest",
            ).values
            new_vars[dataset_name] = xr.DataArray(
                subset_vals,
                dims=[lat_dim, lon_dim],
                coords={lat_dim: ds_lat, lon_dim: ds_lon},
            )

    if new_vars:
        ds = ds.assign(new_vars)

    return ds


def _normalise_mapping(mapping: dict[str, str] | None) -> dict[str, str]:
    if not mapping:
        return {}
    return {str(k): str(v) for k, v in mapping.items()}


def _guess_kind(var: xr.DataArray, time_dim: str, level_dim: str) -> str:
    dims = set(var.dims)
    if time_dim in dims and level_dim in dims:
        return "atmos"
    if time_dim in dims:
        return "surf"
    if level_dim in dims:
        return "atmos"
    return "static"


def _resolve_aurora_name(dataset_name: str, mapping: dict[str, str]) -> str:
    if dataset_name in mapping:
        return mapping[dataset_name]
    # Support reversed mapping for convenience.
    for aurora_name, mapped_dataset_name in mapping.items():
        if mapped_dataset_name == dataset_name:
            return aurora_name
    return dataset_name


def _parse_variable_specs(
    ds: xr.Dataset,
    config_vars: Sequence[Any],
    mapping: dict[str, str],
    *,
    default_kind: str | None,
    time_dim: str,
    level_dim: str,
) -> tuple[VariableSpec, ...]:
    specs: list[VariableSpec] = []
    for item in config_vars:
        if isinstance(item, str):
            dataset_name = item
            aurora_name = _resolve_aurora_name(dataset_name, mapping)
            if dataset_name not in ds:
                raise ValueError(f"Variable `{dataset_name}` not found in dataset.")
            kind = default_kind or _guess_kind(
                ds[dataset_name],
                time_dim=time_dim,
                level_dim=level_dim,
            )
        elif isinstance(item, dict):
            dataset_name = str(item.get("dataset_name") or item.get("name") or "").strip()
            if not dataset_name:
                raise ValueError(f"Invalid variable spec: {item!r}")
            if dataset_name not in ds:
                raise ValueError(f"Variable `{dataset_name}` not found in dataset.")
            aurora_name = str(
                item.get("aurora_name")
                or _resolve_aurora_name(dataset_name, mapping)
            )
            kind = str(
                item.get("kind")
                or default_kind
                or _guess_kind(ds[dataset_name], time_dim, level_dim)
            )
        else:
            raise TypeError(f"Unsupported variable spec type: {type(item)}")

        kind = kind.lower()
        if kind not in {"surf", "atmos", "static"}:
            raise ValueError(f"Variable `{dataset_name}` has unsupported kind `{kind}`.")

        specs.append(VariableSpec(dataset_name=dataset_name, aurora_name=aurora_name, kind=kind))

    # Preserve order but ensure unique Aurora names.
    seen: set[str] = set()
    unique_specs: list[VariableSpec] = []
    for spec in specs:
        if spec.aurora_name in seen:
            continue
        seen.add(spec.aurora_name)
        unique_specs.append(spec)

    return tuple(unique_specs)


def resolve_variable_specs(ds: xr.Dataset, config: dict[str, Any]) -> ResolvedVariableSpecs:
    """Resolve predictor/target/static variables from YAML config and dataset metadata."""
    data_cfg = config.get("data", {})
    time_dim, _, _, level_dim = _dim_names(config)

    predictor_mapping = _normalise_mapping(data_cfg.get("predictor_mapping_to_aurora_names"))
    target_mapping = _normalise_mapping(data_cfg.get("target_mapping_from_aurora_outputs"))
    static_mapping = _normalise_mapping(data_cfg.get("static_mapping_to_aurora_names"))

    predictors = _parse_variable_specs(
        ds,
        config_vars=tuple(data_cfg.get("predictor_variables", [])),
        mapping=predictor_mapping,
        default_kind=None,
        time_dim=time_dim,
        level_dim=level_dim,
    )
    targets = _parse_variable_specs(
        ds,
        config_vars=tuple(data_cfg.get("target_variables", [])),
        mapping=target_mapping,
        default_kind=None,
        time_dim=time_dim,
        level_dim=level_dim,
    )
    static = _parse_variable_specs(
        ds,
        config_vars=tuple(data_cfg.get("static_variables", [])),
        mapping=static_mapping,
        default_kind="static",
        time_dim=time_dim,
        level_dim=level_dim,
    )

    if not predictors:
        raise ValueError("Config must define at least one predictor variable.")
    if not targets:
        raise ValueError("Config must define at least one target variable.")

    # Validate that targets can be produced by the configured model inputs.
    include_targets = bool(data_cfg.get("include_target_variables_as_predictors", False))
    predictor_aurora_names = {spec.aurora_name for spec in predictors}
    target_aurora_names = {spec.aurora_name for spec in targets}
    # Warn (don't error) when targets are not a subset of predictors.  This allows
    # fine-tuning workflows that predict variables not present in the model input,
    # at the cost of requiring architectural support (e.g. a separate output head).
    if not include_targets and not target_aurora_names.issubset(predictor_aurora_names):
        missing = sorted(target_aurora_names - predictor_aurora_names)
        warnings.warn(
            "Some target Aurora variables are not present in predictors and will "
            "need dedicated model output heads: " + ", ".join(missing),
            stacklevel=2,
        )

    return ResolvedVariableSpecs(predictors=predictors, targets=targets, static=static)


def derive_model_variable_config(
    resolved_specs: ResolvedVariableSpecs,
    config: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Derive model variable tuples for Aurora constructor from resolved specs."""
    include_targets = bool(
        config.get("data", {}).get("include_target_variables_as_predictors", False)
    )

    predictor_specs = list(resolved_specs.predictors)
    if include_targets:
        existing = {spec.aurora_name for spec in predictor_specs}
        for target_spec in resolved_specs.targets:
            if target_spec.aurora_name not in existing:
                predictor_specs.append(target_spec)
                existing.add(target_spec.aurora_name)

    surf_vars = tuple(spec.aurora_name for spec in predictor_specs if spec.kind == "surf")
    atmos_vars = tuple(spec.aurora_name for spec in predictor_specs if spec.kind == "atmos")
    static_vars = tuple(spec.aurora_name for spec in resolved_specs.static)

    if not surf_vars:
        raise ValueError("At least one surface predictor variable is required.")
    if not atmos_vars:
        raise ValueError("At least one atmospheric predictor variable is required.")

    return {
        "surf_vars": surf_vars,
        "atmos_vars": atmos_vars,
        "static_vars": static_vars,
    }


def _prepare_data_array(
    da: xr.DataArray,
    *,
    allowed_dims: set[str],
    extra_dim_indexers: dict[str, int],
) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in allowed_dims:
            continue
        if dim in extra_dim_indexers:
            da = da.isel({dim: int(extra_dim_indexers[dim])})
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0})
            continue
        raise ValueError(
            f"Variable `{da.name}` has unsupported non-singleton dimension `{dim}`. "
            "Use data.extra_dim_indexers in config to select a specific index."
        )
    return da


def _select_levels_if_needed(
    da: xr.DataArray,
    config: dict[str, Any],
    level_dim: str,
) -> xr.DataArray:
    data_cfg = config.get("data", {})
    levels = data_cfg.get("atmos_levels")
    if not levels or level_dim not in da.dims:
        return da

    # Keep requested level order.
    selected = da.sel({level_dim: levels})
    return selected


def _extract_predictor_tensor(
    ds: xr.Dataset,
    spec: VariableSpec,
    history_indices: Sequence[int],
    config: dict[str, Any],
) -> torch.Tensor:
    time_dim, lat_dim, lon_dim, level_dim = _dim_names(config)
    extra_dim_indexers = config.get("data", {}).get("extra_dim_indexers", {})

    da = ds[spec.dataset_name]
    if spec.kind == "surf":
        da = _prepare_data_array(
            da,
            allowed_dims={time_dim, lat_dim, lon_dim},
            extra_dim_indexers=extra_dim_indexers,
        )
        da = da.isel({time_dim: list(history_indices)}).transpose(time_dim, lat_dim, lon_dim)
        x = torch.from_numpy(np.asarray(da.values, dtype=np.float32))
        return x  # (T, H, W)

    if spec.kind == "atmos":
        da = _select_levels_if_needed(da, config=config, level_dim=level_dim)
        da = _prepare_data_array(
            da,
            allowed_dims={time_dim, level_dim, lat_dim, lon_dim},
            extra_dim_indexers=extra_dim_indexers,
        )
        da = da.isel({time_dim: list(history_indices)}).transpose(
            time_dim,
            level_dim,
            lat_dim,
            lon_dim,
        )
        x = torch.from_numpy(np.asarray(da.values, dtype=np.float32))
        return x  # (T, C, H, W)

    raise ValueError(f"Predictor variable `{spec.dataset_name}` cannot be static.")


def _extract_static_tensor(
    ds: xr.Dataset,
    spec: VariableSpec,
    anchor_index: int,
    config: dict[str, Any],
) -> torch.Tensor:
    time_dim, lat_dim, lon_dim, _ = _dim_names(config)
    extra_dim_indexers = config.get("data", {}).get("extra_dim_indexers", {})

    da = ds[spec.dataset_name]
    if time_dim in da.dims:
        da = da.isel({time_dim: int(anchor_index)})

    da = _prepare_data_array(
        da,
        allowed_dims={lat_dim, lon_dim},
        extra_dim_indexers=extra_dim_indexers,
    )
    da = da.transpose(lat_dim, lon_dim)
    x = torch.from_numpy(np.asarray(da.values, dtype=np.float32))
    return x  # (H, W)


def _extract_target_tensor(
    ds: xr.Dataset,
    spec: VariableSpec,
    time_index: int,
    config: dict[str, Any],
) -> torch.Tensor:
    time_dim, lat_dim, lon_dim, level_dim = _dim_names(config)
    extra_dim_indexers = config.get("data", {}).get("extra_dim_indexers", {})

    da = ds[spec.dataset_name]
    if spec.kind == "surf":
        da = _prepare_data_array(
            da,
            allowed_dims={time_dim, lat_dim, lon_dim},
            extra_dim_indexers=extra_dim_indexers,
        )
        da = da.isel({time_dim: int(time_index)}).transpose(lat_dim, lon_dim)
        return torch.from_numpy(np.asarray(da.values, dtype=np.float32))

    if spec.kind == "atmos":
        da = _select_levels_if_needed(da, config=config, level_dim=level_dim)
        da = _prepare_data_array(
            da,
            allowed_dims={time_dim, level_dim, lat_dim, lon_dim},
            extra_dim_indexers=extra_dim_indexers,
        )
        da = da.isel({time_dim: int(time_index)}).transpose(level_dim, lat_dim, lon_dim)
        return torch.from_numpy(np.asarray(da.values, dtype=np.float32))

    raise ValueError(f"Target variable `{spec.dataset_name}` cannot be static.")


def _align_spatial_dims_for_patch(
    surf_vars: dict[str, torch.Tensor],
    static_vars: dict[str, torch.Tensor],
    atmos_vars: dict[str, torch.Tensor],
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_size: int,
    *,
    strategy: str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    if not surf_vars:
        raise ValueError("No surface variables available for spatial alignment.")

    h, w = next(iter(surf_vars.values())).shape[-2:]
    target_h = h - (h % patch_size)
    target_w = w - (w % patch_size)

    if target_h == 0 or target_w == 0:
        raise ValueError(
            f"Spatial shape {(h, w)} is smaller than patch_size={patch_size} after alignment."
        )

    if (target_h, target_w) == (h, w):
        return surf_vars, static_vars, atmos_vars, lat, lon, (h, w)

    if strategy != "crop":
        raise ValueError(
            f"Unsupported data.patch_alignment_strategy={strategy!r}. Currently only `crop` is "
            "implemented."
        )

    surf_vars = {k: v[..., :target_h, :target_w] for k, v in surf_vars.items()}
    static_vars = {k: v[..., :target_h, :target_w] for k, v in static_vars.items()}
    atmos_vars = {k: v[..., :target_h, :target_w] for k, v in atmos_vars.items()}

    lat = lat[:target_h]
    lon = lon[:target_w]

    return surf_vars, static_vars, atmos_vars, lat, lon, (h, w)


def _validate_batch_spatial_shapes(batch: Batch) -> None:
    h, w = batch.spatial_shape

    for name, x in batch.surf_vars.items():
        if x.shape[-2:] != (h, w):
            raise ValueError(
                f"Surface variable `{name}` has shape {x.shape}, expected (*, *, {h}, {w})."
            )

    for name, x in batch.static_vars.items():
        if x.shape[-2:] != (h, w):
            raise ValueError(f"Static variable `{name}` has shape {x.shape}, expected ({h}, {w}).")

    for name, x in batch.atmos_vars.items():
        if x.shape[-2:] != (h, w):
            raise ValueError(
                f"Atmospheric variable `{name}` has shape {x.shape}, expected (*, *, *, {h}, {w})."
            )


def _sample_list(samples: dict[str, Any] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(samples, dict):
        return [samples]
    if isinstance(samples, Sequence):
        return [dict(sample) for sample in samples]
    raise TypeError("`samples` must be a sample dict or a sequence of sample dicts.")


def build_training_samples(
    ds: xr.Dataset,
    config: dict[str, Any],
    split_name: str = "train",
) -> list[dict[str, Any]]:
    """Create temporal training/validation/test samples from dataset time axis."""
    data_cfg = config.get("data", {})
    time_dim, _, _, _ = _dim_names(config)

    if time_dim not in ds.dims:
        raise ValueError(f"Dataset is missing time dimension `{time_dim}`.")

    input_steps = int(data_cfg.get("input_time_steps", 2))
    lead_times = tuple(int(x) for x in data_cfg.get("target_lead_times", [1]))
    if input_steps < 1:
        raise ValueError("data.input_time_steps must be >= 1.")
    if not lead_times:
        raise ValueError("data.target_lead_times must contain at least one positive integer.")
    if min(lead_times) < 1:
        raise ValueError("data.target_lead_times must be positive integers (model steps).")

    max_lead = max(lead_times)
    n_time = int(ds.sizes[time_dim])
    start = input_steps - 1
    end_exclusive = n_time - max_lead

    if end_exclusive <= start:
        raise ValueError(
            f"Not enough timestamps for input_time_steps={input_steps} and "
            f"target_lead_times={list(lead_times)}."
        )

    samples: list[dict[str, Any]] = []
    for anchor_idx in range(start, end_exclusive):
        history_indices = list(range(anchor_idx - input_steps + 1, anchor_idx + 1))
        target_indices = {lead: anchor_idx + lead for lead in lead_times}
        samples.append(
            {
                "anchor_index": anchor_idx,
                "history_indices": history_indices,
                "target_indices": target_indices,
            }
        )

    split_controls = data_cfg.get("split_controls", {})
    split_control = split_controls.get(split_name, {}) if isinstance(split_controls, dict) else {}

    if split_control:
        start_offset_raw = split_control.get("start_offset", 0)
        end_offset_raw = split_control.get("end_offset", len(samples))
        stride_raw = split_control.get("stride", 1)
        start_offset = 0 if start_offset_raw is None else int(start_offset_raw)
        end_offset = len(samples) if end_offset_raw is None else int(end_offset_raw)
        stride = 1 if stride_raw is None else int(stride_raw)
        max_samples = split_control.get("max_samples")

        samples = samples[start_offset:end_offset:stride]
        if max_samples is not None:
            samples = samples[: int(max_samples)]

    return samples


def build_aurora_batch(
    ds: xr.Dataset,
    samples: dict[str, Any] | Sequence[dict[str, Any]],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
) -> Batch:
    """Build an `aurora.Batch` from one or more temporal sample definitions."""
    sample_list = _sample_list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required to build a batch.")

    time_dim, lat_dim, lon_dim, level_dim = _dim_names(config)
    model_cfg = config.get("model", {})
    patch_size = int(model_cfg.get("patch_size", 4))
    alignment_strategy = str(config.get("data", {}).get("patch_alignment_strategy", "crop"))

    surf_vars_stacked: dict[str, list[torch.Tensor]] = {}
    atmos_vars_stacked: dict[str, list[torch.Tensor]] = {}
    static_vars_ref: dict[str, torch.Tensor] | None = None
    times: list[datetime] = []

    for sample in sample_list:
        history_indices = sample["history_indices"]
        anchor_index = sample["anchor_index"]

        surf_single: dict[str, torch.Tensor] = {}
        atmos_single: dict[str, torch.Tensor] = {}
        for spec in resolved_specs.predictors:
            x = _extract_predictor_tensor(ds, spec, history_indices=history_indices, config=config)
            if spec.kind == "surf":
                surf_single[spec.aurora_name] = x
            elif spec.kind == "atmos":
                atmos_single[spec.aurora_name] = x

        static_single: dict[str, torch.Tensor] = {}
        for spec in resolved_specs.static:
            static_single[spec.aurora_name] = _extract_static_tensor(
                ds,
                spec,
                anchor_index=anchor_index,
                config=config,
            )

        lat = torch.from_numpy(np.asarray(ds[lat_dim].values, dtype=np.float32))
        lon = torch.from_numpy(np.asarray(ds[lon_dim].values, dtype=np.float32))

        surf_single, static_single, atmos_single, lat, lon, _ = _align_spatial_dims_for_patch(
            surf_vars=surf_single,
            static_vars=static_single,
            atmos_vars=atmos_single,
            lat=lat,
            lon=lon,
            patch_size=patch_size,
            strategy=alignment_strategy,
        )

        if static_vars_ref is None:
            static_vars_ref = static_single
        else:
            # Static values are expected to be consistent across samples. Keep first and validate.
            for name, value in static_single.items():
                if not torch.allclose(static_vars_ref[name], value, atol=0.0, rtol=0.0):
                    raise ValueError(
                        f"Static variable `{name}` changes across selected samples. "
                        "This workflow expects static variables for the domain to be constant."
                    )

        for name, value in surf_single.items():
            surf_vars_stacked.setdefault(name, []).append(value)
        for name, value in atmos_single.items():
            atmos_vars_stacked.setdefault(name, []).append(value)

        current_time = _to_python_datetime(ds[time_dim].values[int(anchor_index)])
        times.append(current_time)

    if static_vars_ref is None:
        static_vars_ref = {}

    surf_vars = {k: torch.stack(v, dim=0) for k, v in surf_vars_stacked.items()}  # (B, T, H, W)
    atmos_vars = {
        k: torch.stack(v, dim=0) for k, v in atmos_vars_stacked.items()
    }  # (B, T, C, H, W)

    levels_cfg = config.get("data", {}).get("atmos_levels")
    if levels_cfg:
        atmos_levels = tuple(float(level) for level in levels_cfg)
    else:
        if level_dim not in ds.coords and level_dim not in ds.dims:
            raise ValueError(
                "Provide data.atmos_levels in config or include a level coordinate in the dataset."
            )
        atmos_levels = tuple(float(level) for level in ds[level_dim].values.tolist())

    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars_ref,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=tuple(times),
            atmos_levels=atmos_levels,
        ),
    )
    _validate_batch_spatial_shapes(batch)

    return batch


def build_targets(
    ds: xr.Dataset,
    samples: dict[str, Any] | Sequence[dict[str, Any]],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    spatial_shape: tuple[int, int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Build target tensors keyed by lead time and Aurora variable name."""
    sample_list = _sample_list(samples)
    patch_size = int(config.get("model", {}).get("patch_size", 4))
    alignment_strategy = str(config.get("data", {}).get("patch_alignment_strategy", "crop"))

    target_by_lead: dict[int, dict[str, list[torch.Tensor]]] = {}

    for sample in sample_list:
        for lead, target_index in sample["target_indices"].items():
            lead = int(lead)
            lead_targets = target_by_lead.setdefault(lead, {})
            for spec in resolved_specs.targets:
                tensor = _extract_target_tensor(ds, spec, time_index=target_index, config=config)

                # Align with predictor batch spatial dimensions.
                if spec.kind == "surf":
                    tensor4 = tensor[None, None]  # (1, 1, H, W)
                    surf, _, _, _, _, _ = _align_spatial_dims_for_patch(
                        surf_vars={"tmp": tensor4},
                        static_vars={"tmp_static": torch.zeros_like(tensor)},
                        atmos_vars={
                            "tmp_atmos": torch.zeros(
                                1,
                                1,
                                1,
                                *tensor.shape,
                                dtype=tensor.dtype,
                            )
                        },
                        lat=torch.arange(tensor.shape[-2], dtype=tensor.dtype),
                        lon=torch.arange(tensor.shape[-1], dtype=tensor.dtype),
                        patch_size=patch_size,
                        strategy=alignment_strategy,
                    )
                    tensor = surf["tmp"][0, 0]
                elif spec.kind == "atmos":
                    tensor5 = tensor[None, None]  # (1, 1, C, H, W)
                    _, _, atmos, _, _, _ = _align_spatial_dims_for_patch(
                        surf_vars={
                            "tmp": torch.zeros(
                                1,
                                1,
                                tensor.shape[-2],
                                tensor.shape[-1],
                                dtype=tensor.dtype,
                            )
                        },
                        static_vars={
                            "tmp_static": torch.zeros(
                                tensor.shape[-2],
                                tensor.shape[-1],
                                dtype=tensor.dtype,
                            )
                        },
                        atmos_vars={"tmp_atmos": tensor5},
                        lat=torch.arange(tensor.shape[-2], dtype=tensor.dtype),
                        lon=torch.arange(tensor.shape[-1], dtype=tensor.dtype),
                        patch_size=patch_size,
                        strategy=alignment_strategy,
                    )
                    tensor = atmos["tmp_atmos"][0, 0]

                # Final safety crop to exactly match batch spatial shape.
                h, w = spatial_shape
                tensor = tensor[..., :h, :w]

                lead_targets.setdefault(spec.aurora_name, []).append(tensor)

    stacked_targets: dict[int, dict[str, torch.Tensor]] = {}
    for lead, var_map in target_by_lead.items():
        stacked_targets[lead] = {
            name: torch.stack(values, dim=0) for name, values in var_map.items()
        }

    return stacked_targets


def _build_land_ocean_mask(batch: Batch, mode: str, threshold: float) -> torch.Tensor | None:
    if "lsm" not in batch.static_vars:
        return None

    lsm = batch.static_vars["lsm"]
    if mode == "land":
        return lsm >= threshold
    if mode == "ocean":
        return lsm < threshold
    return None


def _target_missing_mask(
    ds: xr.Dataset,
    spec: VariableSpec,
    sample: dict[str, Any],
    lead: int,
    config: dict[str, Any],
) -> torch.Tensor | None:
    missing_cfg = config.get("data", {}).get("optional_masks_for_missing_values", {})
    if not isinstance(missing_cfg, dict):
        return None

    mask_name = missing_cfg.get(spec.dataset_name) or missing_cfg.get(spec.aurora_name)
    if not mask_name:
        return None
    if mask_name not in ds:
        raise ValueError(f"Missing-value mask variable `{mask_name}` not present in dataset.")

    time_dim, lat_dim, lon_dim, level_dim = _dim_names(config)
    da = ds[mask_name]
    target_index = int(sample["target_indices"][lead])

    if spec.kind == "surf":
        da = _prepare_data_array(
            da,
            allowed_dims={time_dim, lat_dim, lon_dim},
            extra_dim_indexers=config.get("data", {}).get("extra_dim_indexers", {}),
        )
        arr = da.isel({time_dim: target_index}).transpose(lat_dim, lon_dim).values
    else:
        da = _prepare_data_array(
            da,
            allowed_dims={time_dim, level_dim, lat_dim, lon_dim},
            extra_dim_indexers=config.get("data", {}).get("extra_dim_indexers", {}),
        )
        arr = da.isel({time_dim: target_index}).transpose(level_dim, lat_dim, lon_dim).values

    return torch.from_numpy(np.asarray(arr) > 0)


def _apply_loss_mask(
    loss_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    spatial_mask: torch.Tensor | None,
    missing_mask: torch.Tensor | None,
) -> torch.Tensor:
    combined_mask = torch.isfinite(pred_tensor) & torch.isfinite(target_tensor)

    if spatial_mask is not None:
        while spatial_mask.dim() < combined_mask.dim():
            spatial_mask = spatial_mask.unsqueeze(0)
        combined_mask = combined_mask & spatial_mask

    if missing_mask is not None:
        while missing_mask.dim() < combined_mask.dim():
            missing_mask = missing_mask.unsqueeze(0)
        combined_mask = combined_mask & missing_mask

    if not torch.any(combined_mask):
        return torch.zeros((), device=loss_tensor.device, dtype=loss_tensor.dtype)

    return loss_tensor[combined_mask].mean()


def compute_target_normalization_stats(
    ds: xr.Dataset,
    resolved_specs: ResolvedVariableSpecs,
    config: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute per-variable, per-level mean and std from the training dataset.

    Returns a dict keyed by Aurora variable name::

        {
            "no2": {"mean": Tensor(n_levels,), "std": Tensor(n_levels,)},
            "tcno2": {"mean": Tensor(),        "std": Tensor()},
            ...
        }

    For atmospheric variables the stats have shape ``(n_levels,)`` so they
    can broadcast against ``(batch, n_levels, H, W)`` tensors when reshaped
    to ``(1, n_levels, 1, 1)``.  For surface variables the stats are scalar.
    """
    time_dim, lat_dim, lon_dim, level_dim = _dim_names(config)
    stats: dict[str, dict[str, torch.Tensor]] = {}

    for spec in resolved_specs.targets:
        da = ds[spec.dataset_name]

        if spec.kind == "atmos" and level_dim in da.dims:
            # Per-level mean and std — reduce over time + spatial dims.
            reduce_dims = [d for d in da.dims if d != level_dim]
            mean_np = da.mean(dim=reduce_dims).values.astype(np.float32)
            std_np = da.std(dim=reduce_dims).values.astype(np.float32)
        else:
            # Scalar stats for surface variables.
            mean_np = np.float32(float(da.mean()))
            std_np = np.float32(float(da.std()))

        # Clamp std to avoid division by zero for near-constant fields.
        std_np = np.maximum(std_np, np.float32(1e-30))

        stats[spec.aurora_name] = {
            "mean": torch.from_numpy(np.atleast_1d(mean_np)),
            "std": torch.from_numpy(np.atleast_1d(std_np)),
        }

    return stats


def _loss_tensor(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name in {"mse", "l2"}:
        return (pred - target) ** 2
    if loss_name in {"mae", "l1"}:
        return torch.abs(pred - target)
    if loss_name in {"smooth_l1", "huber"}:
        return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unsupported loss function: {loss_name}")


def _autocast_context(device: torch.device, mixed_precision: str):
    mode = mixed_precision.lower()
    if mode in {"none", "off", "false"}:
        return contextlib.nullcontext()

    if device.type == "cuda":
        dtype = torch.bfloat16 if mode in {"bf16", "bfloat16"} else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    if device.type == "xpu":
        dtype = torch.bfloat16 if mode in {"bf16", "bfloat16"} else torch.float16
        return torch.autocast(device_type="xpu", dtype=dtype)

    return contextlib.nullcontext()


def _advance_batch_with_prediction(batch: Batch, pred: Batch) -> Batch:
    """Build next autoregressive input from the current batch and prediction.

    Handles mismatched keys between predictors (batch) and model outputs (pred):
      * Variables in both: shift history and append the new prediction.
      * Variables only in batch (exogenous predictors): carry forward as-is
        by shifting history and repeating the last available step.
      * Variables only in pred (e.g. modulation heads): dropped.
    """

    def _merge(
        batch_vars: dict[str, torch.Tensor],
        pred_vars: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        for name in batch_vars:
            history_tail = batch_vars[name][:, 1:]  # drop oldest step
            if name in pred_vars:
                merged[name] = torch.cat([history_tail, pred_vars[name]], dim=1)
            else:
                # Exogenous predictor not predicted by the model: repeat last step.
                merged[name] = torch.cat([history_tail, batch_vars[name][:, -1:]], dim=1)
        return merged

    return dataclasses.replace(
        pred,
        surf_vars=_merge(batch.surf_vars, pred.surf_vars),
        atmos_vars=_merge(batch.atmos_vars, pred.atmos_vars),
    )


def compute_supervised_loss(
    model: torch.nn.Module,
    ds: xr.Dataset,
    samples: dict[str, Any] | Sequence[dict[str, Any]],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    device: str | torch.device,
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run autoregressive forward passes and compute supervised multi-lead loss."""
    sample_list = _sample_list(samples)
    device = torch.device(device)
    training_cfg = config.get("training", {})
    loss_name = str(training_cfg.get("loss_function", "mse")).lower()
    weights_cfg = training_cfg.get("multi_target_loss_weights", {})

    batch = build_aurora_batch(ds, sample_list, config, resolved_specs=resolved_specs)
    batch = batch.to(device)

    targets = build_targets(
        ds,
        sample_list,
        config,
        resolved_specs=resolved_specs,
        spatial_shape=batch.spatial_shape,
    )

    lead_times = sorted(int(x) for x in targets)
    max_lead = max(lead_times)

    data_cfg = config.get("data", {})
    loss_mask_cfg = data_cfg.get("loss_mask_options", {})
    mask_mode = str(loss_mask_cfg.get("mode", "none")).lower()
    mask_threshold = float(loss_mask_cfg.get("threshold", 0.5))
    spatial_mask = _build_land_ocean_mask(batch, mode=mask_mode, threshold=mask_threshold)
    if spatial_mask is not None:
        spatial_mask = spatial_mask.to(device)

    mixed_precision = str(config.get("model", {}).get("mixed_precision", "none"))

    preds_by_lead: dict[int, Batch] = {}
    current_batch = batch
    with _autocast_context(device=device, mixed_precision=mixed_precision):
        for lead in range(1, max_lead + 1):
            pred = model(current_batch)
            preds_by_lead[lead] = pred
            current_batch = _advance_batch_with_prediction(current_batch, pred)

    total_loss = torch.zeros((), device=device)
    total_weight = 0.0

    for lead in lead_times:
        pred = preds_by_lead[lead]
        target_map = targets[lead]

        for target_spec in resolved_specs.targets:
            aurora_name = target_spec.aurora_name
            if aurora_name not in target_map:
                continue

            if target_spec.kind == "surf":
                pred_tensor = pred.surf_vars[aurora_name][:, 0]
            else:
                pred_tensor = pred.atmos_vars[aurora_name][:, 0]

            target_tensor = target_map[aurora_name].to(device=device, dtype=pred_tensor.dtype)

            # Normalize pred and target to ~O(1) so that MSE gradients are
            # meaningful even for variables with tiny physical magnitudes
            # (e.g. NO2 ~1e-10 kg/kg).
            if norm_stats is not None and aurora_name in norm_stats:
                _ns = norm_stats[aurora_name]
                _mean = _ns["mean"].to(device=device, dtype=pred_tensor.dtype)
                _std = _ns["std"].to(device=device, dtype=pred_tensor.dtype)
                if target_spec.kind == "atmos" and _mean.numel() > 1:
                    # Reshape (n_levels,) → (1, n_levels, 1, 1) for broadcasting.
                    _mean = _mean.view(1, -1, 1, 1)
                    _std = _std.view(1, -1, 1, 1)
                pred_tensor = (pred_tensor - _mean) / _std
                target_tensor = (target_tensor - _mean) / _std

            missing_masks: list[torch.Tensor] = []
            for sample in sample_list:
                maybe_mask = _target_missing_mask(ds, target_spec, sample, lead=lead, config=config)
                if maybe_mask is not None:
                    missing_masks.append(maybe_mask)
            missing_mask = torch.stack(missing_masks, dim=0).to(device) if missing_masks else None

            elementwise = _loss_tensor(pred_tensor, target_tensor, loss_name=loss_name)
            masked = _apply_loss_mask(
                loss_tensor=elementwise,
                pred_tensor=pred_tensor,
                target_tensor=target_tensor,
                spatial_mask=spatial_mask,
                missing_mask=missing_mask,
            )

            key_step = f"{aurora_name}@{lead}"
            weight = float(
                weights_cfg.get(
                    key_step,
                    weights_cfg.get(
                        target_spec.dataset_name,
                        weights_cfg.get(aurora_name, 1.0),
                    ),
                )
            )
            total_loss = total_loss + weight * masked
            total_weight += weight

    if total_weight <= 0:
        raise ValueError("Total loss weight evaluated to <= 0. Check multi_target_loss_weights.")

    total_loss = total_loss / total_weight

    metrics = {
        "batch_size": len(sample_list),
        "lead_times": lead_times,
        "spatial_shape": batch.spatial_shape,
    }
    return total_loss, metrics


def configure_trainable_parameters(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, int]:
    """Apply freeze/unfreeze config and return parameter summary."""
    model_cfg = config.get("model", {})

    for param in model.parameters():
        param.requires_grad = True

    backbone_freeze = bool(model_cfg.get("backbone_freeze", False))
    freeze_embeddings = bool(model_cfg.get("freeze_embeddings", False))
    freeze_encoder = bool(model_cfg.get("freeze_encoder", False))
    freeze_decoder = bool(model_cfg.get("freeze_decoder", False))
    trainable_head_only = bool(model_cfg.get("trainable_head_only", False))

    if backbone_freeze and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False

    if freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False

    if freeze_decoder and hasattr(model, "decoder"):
        for param in model.decoder.parameters():
            param.requires_grad = False

    if freeze_embeddings:
        for name, param in model.named_parameters():
            if "token_embeds" in name or "levels_embed" in name or "patch_embedding" in name:
                param.requires_grad = False

    if trainable_head_only:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if (
                "decoder.surf_heads" in name
                or "decoder.atmos_heads" in name
                or "decoder.modulation_heads" in name
                or "surf_feature_combiner" in name
                or "atmos_feature_combiner" in name
            ):
                param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
    }


def create_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config.training section."""
    training_cfg = config.get("training", {})
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters were found after freeze/unfreeze configuration.")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(training_cfg.get("sgd_momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    num_training_steps: int,
):
    """Create LR scheduler from config.training.scheduler."""
    training_cfg = config.get("training", {})
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()

    if scheduler_name in {"none", "off", "false"}:
        return None

    if scheduler_name == "cosine":
        t_max = int(training_cfg.get("scheduler_t_max", max(1, num_training_steps)))
        eta_min = float(training_cfg.get("scheduler_eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if scheduler_name == "step":
        step_size = int(training_cfg.get("scheduler_step_size", max(1, num_training_steps // 5)))
        gamma = float(training_cfg.get("scheduler_gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def run_validation(
    model: torch.nn.Module,
    ds_val: xr.Dataset,
    val_samples: Sequence[dict[str, Any]],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    device: str | torch.device,
    *,
    max_batches: int | None = None,
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, float]:
    """Run validation loop and return aggregate metrics."""
    if not val_samples:
        return {"val_loss": math.nan, "num_val_batches": 0}

    batch_size = int(config.get("training", {}).get("batch_size", 1))
    model.eval()

    losses: list[float] = []
    with torch.inference_mode():
        batch_count = 0
        for i in range(0, len(val_samples), batch_size):
            sample_batch = val_samples[i : i + batch_size]
            loss, _ = compute_supervised_loss(
                model=model,
                ds=ds_val,
                samples=sample_batch,
                config=config,
                resolved_specs=resolved_specs,
                device=device,
                norm_stats=norm_stats,
            )
            losses.append(float(loss.detach().cpu().item()))
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

    val_loss = float(np.mean(losses)) if losses else math.nan
    return {
        "val_loss": val_loss,
        "num_val_batches": float(len(losses)),
    }


def _dataset_frame_for_predictor(
    ds: xr.Dataset,
    spec: VariableSpec,
    time_index: int,
    batch_size: int,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    tensor = _extract_target_tensor(ds, spec, time_index=time_index, config=config)
    if spec.kind == "surf":
        frame = tensor[None, None].repeat(batch_size, 1, 1, 1)
    else:
        frame = tensor[None, None].repeat(batch_size, 1, 1, 1, 1)
    return frame.to(device=device)


def run_rollout(
    model: torch.nn.Module,
    ds: xr.Dataset,
    start_sample: dict[str, Any],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    device: str | torch.device,
) -> list[Batch]:
    """Run rollout from a starting sample using a fine-tuned model."""
    rollout_cfg = config.get("rollout", {})
    steps = int(rollout_cfg.get("rollout_num_steps", 0))
    if steps <= 0:
        return []

    autoregressive = bool(rollout_cfg.get("autoregressive_inputs", True))
    feedback_fields_cfg = rollout_cfg.get("predicted_fields_get_fed_back") or rollout_cfg.get(
        "predicted_fields_feedback"
    )
    feedback_fields = set(str(x) for x in feedback_fields_cfg) if feedback_fields_cfg else None

    keep_exogenous_mode = str(
        rollout_cfg.get("keep_exogenous_predictors", "fixed")
    ).lower()
    refresh_exogenous = keep_exogenous_mode in {"refresh", "refresh_from_dataset", "dataset"}

    device = torch.device(device)
    model.eval()

    current = build_aurora_batch(
        ds,
        start_sample,
        config=config,
        resolved_specs=resolved_specs,
    ).to(device)

    predictor_by_aurora = resolved_specs.predictor_by_aurora
    anchor_idx = int(start_sample["anchor_index"])

    predictions: list[Batch] = []
    with torch.inference_mode():
        for step in range(1, steps + 1):
            pred = model(current)
            predictions.append(pred.to("cpu"))

            if not autoregressive:
                # Teacher-forced rollout based on dataset history.
                input_steps = int(config.get("data", {}).get("input_time_steps", 2))
                next_anchor = anchor_idx + step
                history_indices = list(range(next_anchor - input_steps + 1, next_anchor + 1))
                forced_sample = {
                    "anchor_index": next_anchor,
                    "history_indices": history_indices,
                    "target_indices": {},
                }
                current = build_aurora_batch(
                    ds,
                    forced_sample,
                    config=config,
                    resolved_specs=resolved_specs,
                ).to(device)
                continue

            next_time_index = anchor_idx + step

            surf_next: dict[str, torch.Tensor] = {}
            for name, old in current.surf_vars.items():
                use_prediction = (feedback_fields is None and name in pred.surf_vars) or (
                    feedback_fields is not None and name in feedback_fields
                )
                if use_prediction:
                    new_frame = pred.surf_vars[name]
                elif refresh_exogenous and next_time_index < ds.sizes[_dim_names(config)[0]]:
                    predictor_spec = predictor_by_aurora.get(name)
                    if predictor_spec is not None and predictor_spec.kind == "surf":
                        new_frame = _dataset_frame_for_predictor(
                            ds,
                            predictor_spec,
                            time_index=next_time_index,
                            batch_size=old.shape[0],
                            config=config,
                            device=device,
                        )
                    else:
                        new_frame = old[:, -1:]
                else:
                    new_frame = old[:, -1:]

                surf_next[name] = torch.cat([old[:, 1:], new_frame], dim=1)

            atmos_next: dict[str, torch.Tensor] = {}
            for name, old in current.atmos_vars.items():
                use_prediction = (feedback_fields is None and name in pred.atmos_vars) or (
                    feedback_fields is not None and name in feedback_fields
                )
                if use_prediction:
                    new_frame = pred.atmos_vars[name]
                elif refresh_exogenous and next_time_index < ds.sizes[_dim_names(config)[0]]:
                    predictor_spec = predictor_by_aurora.get(name)
                    if predictor_spec is not None and predictor_spec.kind == "atmos":
                        new_frame = _dataset_frame_for_predictor(
                            ds,
                            predictor_spec,
                            time_index=next_time_index,
                            batch_size=old.shape[0],
                            config=config,
                            device=device,
                        )
                    else:
                        new_frame = old[:, -1:]
                else:
                    new_frame = old[:, -1:]

                atmos_next[name] = torch.cat([old[:, 1:], new_frame], dim=1)

            current = Batch(
                surf_vars=surf_next,
                static_vars=current.static_vars,
                atmos_vars=atmos_next,
                metadata=pred.metadata,
            )

    return predictions


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    torch.save(payload, str(path))


def load_checkpoint_if_available(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str | Path | None,
    device: str | torch.device,
) -> dict[str, Any] | None:
    """Load a training checkpoint if provided and available."""
    if not checkpoint_path:
        return None

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=torch.device(device),
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def write_training_history(
    history: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write training history to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "training_history.json"
    csv_path = output_dir / "training_history.csv"

    json_path.write_text(json.dumps(list(history), indent=2, default=str))

    if history:
        fieldnames: list[str] = []
        for row in history:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

    return {
        "training_history_json": str(json_path.resolve()),
        "training_history_csv": str(csv_path.resolve()),
    }


def write_run_manifest(
    config: dict[str, Any],
    output_dir: str | Path,
    extras: dict[str, Any] | None = None,
) -> str:
    """Write a run manifest JSON with effective config and key artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": config,
        "extras": extras or {},
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return str(path.resolve())


def save_predictions(
    predictions: Sequence[Batch],
    output_path: str | Path,
    *,
    save_netcdf: bool = True,
    resolved_specs: "ResolvedVariableSpecs | None" = None,
) -> xr.Dataset:
    """Save rollout predictions to NetCDF using original dataset variable names.

    The output structure mirrors the input data files (e.g. ``train.nc``):

    * Dimensions: ``(time, latitude, longitude)`` for surface variables,
      ``(time, level, latitude, longitude)`` for atmospheric variables.
    * ``time`` is a datetime64 coordinate built from the prediction metadata.
    * ``latitude`` / ``longitude`` / ``level`` use float64 to match CF conventions.
    * No ``step``, ``batch``, or ``valid_time`` dimensions are created.

    Variable names match the dataset names from *resolved_specs*
    (e.g. ``t2m``, ``no2``).  When *resolved_specs* is not provided the Aurora
    internal names are used as a fallback (e.g. ``2t``, ``no2``).
    """
    if not predictions:
        raise ValueError("No predictions were provided to save_predictions.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build aurora_name -> dataset_name reverse mappings from resolved_specs.
    surf_name_map: dict[str, str] = {}
    atmos_name_map: dict[str, str] = {}
    if resolved_specs is not None:
        for spec in resolved_specs.predictors:
            if spec.kind == "surf":
                surf_name_map[spec.aurora_name] = spec.dataset_name
            elif spec.kind == "atmos":
                atmos_name_map[spec.aurora_name] = spec.dataset_name
        for spec in resolved_specs.targets:
            if spec.kind == "surf":
                surf_name_map[spec.aurora_name] = spec.dataset_name
            elif spec.kind == "atmos":
                atmos_name_map[spec.aurora_name] = spec.dataset_name

    first = predictions[0]
    # Use float64 for spatial coords to match CF / input data conventions.
    lat = np.asarray(first.metadata.lat.detach().cpu().numpy(), dtype=np.float64)
    lon = np.asarray(first.metadata.lon.detach().cpu().numpy(), dtype=np.float64)
    levels = np.asarray(first.metadata.atmos_levels, dtype=np.float64)

    # Build a 1-D time coordinate from prediction metadata (first batch element).
    times = np.array(
        [np.datetime64(pred.metadata.time[0], "ns") for pred in predictions],
        dtype="datetime64[ns]",
    )

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    for var_name in first.surf_vars:
        # pred.surf_vars[name] shape: (batch, 1, H, W) — take batch=0, history=0
        arr = np.stack(
            [pred.surf_vars[var_name][0, 0].detach().cpu().numpy() for pred in predictions],
            axis=0,
        )  # (time, lat, lon)
        out_name = surf_name_map.get(var_name, var_name)
        data_vars[out_name] = (("time", "latitude", "longitude"), arr)

    for var_name in first.atmos_vars:
        arr = np.stack(
            [pred.atmos_vars[var_name][0, 0].detach().cpu().numpy() for pred in predictions],
            axis=0,
        )  # (time, level, lat, lon)
        out_name = atmos_name_map.get(var_name, var_name)
        data_vars[out_name] = (("time", "level", "latitude", "longitude"), arr)

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "latitude": lat,
            "longitude": lon,
            "level": levels,
        },
    )

    if save_netcdf:
        ds_out.to_netcdf(str(output_path))

    return ds_out
