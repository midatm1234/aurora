"""Utilities for config-driven Aurora fine-tuning and rollout workflows."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import logging
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


logger = logging.getLogger("aurora.finetune")


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
    "maybe_wrap_conv_refine",
    "maybe_wrap_flow_refine",
]


@dataclass(frozen=True)
class VariableSpec:
    """Resolved variable metadata for notebook fine-tuning pipelines."""

    dataset_name: str
    aurora_name: str
    kind: str  # surf | atmos | static
    loss_levels: tuple[float, ...] | None = None


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

    case_name = str(config.get("case_name", "")).strip()
    if not case_name:
        raise ValueError(
            f"Config at {config_path} must define a non-empty top-level `case_name`. "
            "This value is used to locate case-specific train/test data."
        )

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

    data_dir = Path(paths_cfg.get("data_dir", project_root / "data"))
    case_data_dir = data_dir if data_dir.name == case_name else data_dir / case_name
    paths_cfg["data_dir"] = str(data_dir.resolve())
    paths_cfg["case_data_dir"] = str(case_data_dir.resolve())
    paths_cfg["train_data_path"] = str((case_data_dir / "train.nc").resolve())
    paths_cfg["val_data_path"] = str((case_data_dir / "test.nc").resolve())
    paths_cfg["test_data_path"] = str((case_data_dir / "test.nc").resolve())

    required_data_paths = [
        Path(paths_cfg["train_data_path"]),
        Path(paths_cfg["test_data_path"]),
    ]
    missing_data_paths = [path for path in required_data_paths if not path.exists()]
    if missing_data_paths:
        missing = ", ".join(str(path) for path in missing_data_paths)
        raise FileNotFoundError(
            f"Required case-specific prepared dataset file(s) are missing: {missing}. "
            "Run finetune/prepare_train_test_from_netcdf.py with the same YAML config first."
        )

    output_dir = Path(paths_cfg.get("output_dir", project_root / "outputs"))
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", output_dir / "checkpoints"))

    # Optional `case_name` (top-level YAML key) groups all artifacts of a
    # single experiment under <output_dir>/<case_name>/ and
    # <checkpoint_dir>/<case_name>/. Idempotent: re-resolving an already
    # case-suffixed path is a no-op, so calling resolve_paths twice on the
    # same dict (e.g., notebook + script) doesn't double-nest.
    if output_dir.name != case_name:
        output_dir = output_dir / case_name
    if checkpoint_dir.name != case_name:
        checkpoint_dir = checkpoint_dir / case_name

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
        expected = {
            Path(config.get("paths", {}).get("train_data_path", "")).expanduser().resolve(),
            Path(config.get("paths", {}).get("val_data_path", "")).expanduser().resolve(),
            Path(config.get("paths", {}).get("test_data_path", "")).expanduser().resolve(),
        }
        if path in expected:
            raise FileNotFoundError(
                f"Required prepared dataset is missing: {path}. "
                "Run finetune/prepare_train_test_from_netcdf.py with the same YAML config first."
            )
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
            loss_levels_raw = item.get("loss_levels")
            loss_levels = (
                tuple(float(x) for x in loss_levels_raw)
                if loss_levels_raw is not None
                else None
            )
        else:
            raise TypeError(f"Unsupported variable spec type: {type(item)}")

        kind = kind.lower()
        if kind not in {"surf", "atmos", "static"}:
            raise ValueError(f"Variable `{dataset_name}` has unsupported kind `{kind}`.")

        if not isinstance(item, dict):
            loss_levels = None

        specs.append(
            VariableSpec(
                dataset_name=dataset_name,
                aurora_name=aurora_name,
                kind=kind,
                loss_levels=loss_levels,
            )
        )

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
    """Return per-variable normalisation stats for the loss function.

    Returns Aurora's own internal normalisation constants (location & scale
    from ``aurora.normalisation``), which for pollutants follow the paper's
    Eq. B8 (centre=0, scale = ½·mean_t(spatial_max), per pressure level).

    Why the model's internal stats and not data-derived ones?  The
    ``AuroraAirPollution`` model already calls ``batch.normalise(...)`` on
    inputs and ``pred.unnormalise(...)`` on outputs using these exact
    constants (see ``aurora/model/aurora.py``).  Predictions are therefore
    returned in **physical units**, having round-tripped through the
    paper's centre/scale.  Using these same stats to renormalise pred and
    target before the MSE puts the loss in O(1) space — matching the
    space the model was pretrained in — without any double-scaling.

    Per-level scales are preserved (no flattening) so the natural vertical
    structure of the variable is respected.
    """
    from aurora.normalisation import level_to_str, locations, scales

    level_values = config.get("data", {}).get(
        "pressure_levels",
        config.get("data", {}).get(
            "atmos_levels",
            [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        ),
    )
    level_values = [float(lv) for lv in level_values]

    # Default 0.0 = no clamp (use Aurora's exact pretrained per-level scales).
    # Setting min_norm_scale > 0 was previously used as a band-aid for the
    # runaway-gradient issue at top-of-atmosphere NO2 levels (scales ~1e-9),
    # but it decouples the loss from Aurora's pretrained loss surface and is
    # not a correct fix. The proper fix lives in the optimizer (per-parameter
    # LR scaling by weight magnitude). Leave this knob exposed only for
    # diagnostic experiments; default keeps Aurora's exact normalisation.
    min_scale = float(config.get("training", {}).get("min_norm_scale", 0.0))

    stats: dict[str, dict[str, torch.Tensor]] = {}

    for spec in resolved_specs.targets:
        aurora_name = spec.aurora_name

        if spec.kind == "atmos":
            # Compute stats only for loss_levels if specified; the flow-refine
            # head only operates on these levels at inference.
            if spec.loss_levels is not None:
                target_levels = [float(lv) for lv in spec.loss_levels]
            else:
                target_levels = level_values
            mean_list: list[float] = []
            std_list: list[float] = []
            for lvl in target_levels:
                key = f"{aurora_name}_{level_to_str(lvl)}"
                mean_list.append(float(locations.get(key, 0.0)))
                std_list.append(float(scales.get(key, 1.0)))
            std_tensor = torch.tensor(std_list, dtype=torch.float32)
            if min_scale > 0:
                std_tensor = torch.clamp(std_tensor, min=min_scale)
            stats[aurora_name] = {
                "mean": torch.tensor(mean_list, dtype=torch.float32),
                "std": std_tensor,
            }
        else:
            loc = float(locations.get(aurora_name, 0.0))
            sc = float(scales.get(aurora_name, 1.0))
            std_tensor = torch.tensor([sc], dtype=torch.float32)
            if min_scale > 0:
                std_tensor = torch.clamp(std_tensor, min=min_scale)
            stats[aurora_name] = {
                "mean": torch.tensor([loc], dtype=torch.float32),
                "std": std_tensor,
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


def _advance_batch_with_prediction(
    batch: Batch,
    pred: Batch,
    feedback_vars: set[str] | None = None,
) -> Batch:
    """Build next autoregressive input from the current batch and prediction.

    Handles mismatched keys between predictors (batch) and model outputs (pred):
      * Variables in both: shift history and append the new prediction.
      * Variables only in batch (exogenous predictors): carry forward as-is
        by shifting history and repeating the last available step.
      * Variables only in pred (e.g. modulation heads): dropped.

    When *feedback_vars* is given, only those variables use the model
    prediction; all others carry forward the last available history step.
    This prevents unsupervised (exogenous) predictions from corrupting
    subsequent autoregressive steps during fine-tuned rollouts.
    """

    def _merge(
        batch_vars: dict[str, torch.Tensor],
        pred_vars: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        for name in batch_vars:
            history_tail = batch_vars[name][:, 1:]  # drop oldest step
            use_pred = name in pred_vars and (
                feedback_vars is None or name in feedback_vars
            )
            if use_pred:
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


# Module-level diagnostic state.  When ``active`` is set to True, the loss
# function appends per-variable breakdown rows to ``rows``.  Caller is
# responsible for resetting and consuming these.
_DIAG_LOSS_BREAKDOWN: dict[str, Any] = {"active": False}


def _resolve_coherence_pair(
    coh_cache: dict[str, dict[str, Any]],
    column_var: str,
    profile_var: str,
) -> tuple[str | None, str | None]:
    """Pick the (column=surf, profile=atmos) variable pair for coherence.

    Honours explicit ``column_var``/``profile_var`` from config when both are
    present in the cache; otherwise auto-detects the first cached surf var as
    the column and the first cached atmos var as the profile.
    """
    if column_var and profile_var:
        if column_var in coh_cache and profile_var in coh_cache:
            return column_var, profile_var
        return None, None

    col = next((n for n, v in coh_cache.items() if v["kind"] == "surf"), None)
    prof = next((n for n, v in coh_cache.items() if v["kind"] == "atmos"), None)
    return col, prof


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

    # Column/profile coherence (cross-variable structural term). Active only
    # for a flow-refine wrapper in training mode; see end of the lead loop.
    aux_cfg = training_cfg.get("flow_aux_loss", {})
    aux_enabled = bool(aux_cfg.get("enabled", True))
    coherence_weight = (
        float(aux_cfg.get("coherence_weight", 0.0)) if aux_enabled else 0.0
    )
    coherence_col_var = str(aux_cfg.get("coherence_column_var", "") or "")
    coherence_prof_var = str(aux_cfg.get("coherence_profile_var", "") or "")

    # Mamba temporal loss weight. Active only when the flow-refine wrapper has
    # the temporal module enabled (see end of this function). 0 → disabled.
    temporal_weight = float(training_cfg.get("mamba_temporal_weight", 1.0))

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

    # Only feed back target variables' predictions; exogenous predictors use ground truth.
    target_feedback_vars = {spec.aurora_name for spec in resolved_specs.targets}
    predictor_by_aurora = resolved_specs.predictor_by_aurora
    spatial_h, spatial_w = batch.spatial_shape

    preds_by_lead: dict[int, Batch] = {}
    current_batch = batch
    # Per-variable normalised sequences (ordered by lead) for the Mamba
    # temporal loss. Populated inside the lead loop only when a flow-refine
    # wrapper with the temporal module enabled is in training mode.
    temporal_seq: dict[str, dict[str, Any]] = {}
    with _autocast_context(device=device, mixed_precision=mixed_precision):
        for lead in range(1, max_lead + 1):
            pred = model(current_batch)
            preds_by_lead[lead] = pred

            # Build next autoregressive input:
            # - Target variables: use model prediction (autoregressive)
            # - Exogenous predictors: use ground truth from dataset
            anchor_idx = sample_list[0]["anchor_index"]
            next_time_index = anchor_idx + lead

            def _advance_with_gt(
                batch_vars: dict[str, torch.Tensor],
                pred_vars: dict[str, torch.Tensor],
                var_kind: str,
            ) -> dict[str, torch.Tensor]:
                merged: dict[str, torch.Tensor] = {}
                for name in batch_vars:
                    history_tail = batch_vars[name][:, 1:]
                    if name in target_feedback_vars and name in pred_vars:
                        merged[name] = torch.cat([history_tail, pred_vars[name]], dim=1)
                    else:
                        # Use ground truth from dataset for exogenous predictors.
                        spec = predictor_by_aurora.get(name)
                        if (
                            spec is not None
                            and spec.kind == var_kind
                            and next_time_index < ds.sizes[_dim_names(config)[0]]
                        ):
                            gt_frame = _dataset_frame_for_predictor(
                                ds, spec,
                                time_index=next_time_index,
                                batch_size=batch_vars[name].shape[0],
                                config=config,
                                device=device,
                            )
                            # Crop to match batch spatial shape.
                            gt_frame = gt_frame[..., :spatial_h, :spatial_w]
                            merged[name] = torch.cat([history_tail, gt_frame], dim=1)
                        else:
                            merged[name] = torch.cat(
                                [history_tail, batch_vars[name][:, -1:]], dim=1,
                            )
                return merged

            current_batch = dataclasses.replace(
                pred,
                surf_vars=_advance_with_gt(
                    current_batch.surf_vars, pred.surf_vars, "surf",
                ),
                atmos_vars=_advance_with_gt(
                    current_batch.atmos_vars, pred.atmos_vars, "atmos",
                ),
            )

    total_loss = torch.zeros((), device=device)
    total_weight = 0.0

    for lead in lead_times:
        pred = preds_by_lead[lead]
        target_map = targets[lead]

        # Per-lead cache of normalised (pred, target) tensors keyed by aurora
        # name, used to compute the cross-variable column/profile coherence
        # term after all per-variable losses for this lead are accumulated.
        coh_cache: dict[str, dict[str, Any]] = {}

        for target_spec in resolved_specs.targets:
            aurora_name = target_spec.aurora_name
            if aurora_name not in target_map:
                continue

            if target_spec.kind == "surf":
                pred_tensor = pred.surf_vars[aurora_name][:, 0]
            else:
                pred_tensor = pred.atmos_vars[aurora_name][:, 0]

            target_tensor = target_map[aurora_name].to(device=device, dtype=pred_tensor.dtype)

            # Pressure levels aligned to ``pred_tensor``'s level axis (used by
            # the column/profile coherence term). ``None`` for surface vars.
            var_levels: list[float] | None = None

            # Optionally restrict atmospheric loss to a subset of pressure
            # levels (e.g. drop levels where Aurora's internal scale is
            # numerically degenerate, causing 1/std to blow up).
            if target_spec.kind == "atmos" and target_spec.loss_levels is not None:
                full_levels = config.get("data", {}).get(
                    "atmos_levels",
                    config.get("data", {}).get("pressure_levels", []),
                )
                full_levels = [float(lv) for lv in full_levels]
                wanted = [float(lv) for lv in target_spec.loss_levels]
                level_idx = [full_levels.index(lv) for lv in wanted]
                idx_tensor = torch.tensor(level_idx, dtype=torch.long, device=pred_tensor.device)
                pred_tensor = pred_tensor.index_select(1, idx_tensor)
                if target_tensor.shape[1] == len(full_levels):
                    target_tensor = target_tensor.index_select(
                        1, idx_tensor.to(target_tensor.device)
                    )
                var_levels = wanted
            elif target_spec.kind == "atmos":
                var_levels = [
                    float(lv)
                    for lv in config.get("data", {}).get(
                        "atmos_levels",
                        config.get("data", {}).get("pressure_levels", []),
                    )
                ]

            # Upcast to fp32 for numerically stable loss computation.
            # bf16 gradients through 1/std (up to ~1e10) cause overflow.
            pred_tensor = pred_tensor.float()
            target_tensor = target_tensor.float()

            # Normalize pred and target to ~O(1) so that MSE gradients are
            # meaningful even for variables with tiny physical magnitudes
            # (e.g. NO2 ~1e-10 kg/kg).
            if norm_stats is not None and aurora_name in norm_stats:
                _ns = norm_stats[aurora_name]
                _mean = _ns["mean"].to(device=device, dtype=torch.float32)
                _std = _ns["std"].to(device=device, dtype=torch.float32)
                if target_spec.kind == "atmos" and _mean.numel() > 1:
                    # norm_stats are computed for loss_levels only (when
                    # specified), so they already align with pred_tensor after
                    # the loss_levels index_select above.
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

            # If the model is a flow-matching refine wrapper AND we're in
            # training mode, replace the deterministic MSE with the
            # rectified-flow velocity-MSE. This gives a unit-normal-scale
            # loss surface (no 1/std blowup) and lets a small UNet learn
            # the conditional residual distribution p(r | ŷ).
            base_for_fm = model.module if hasattr(model, "module") else model
            try:
                from finetune.flow_refine import AuroraFlowRefine as _AFR
            except Exception:
                _AFR = None
            use_flow_loss = (
                _AFR is not None
                and isinstance(base_for_fm, _AFR)
                and base_for_fm.training
            )
            if use_flow_loss:
                flow_doy = None
                if getattr(base_for_fm, "doy_cond", False):
                    flow_doy = _AFR._doy_from_metadata(pred)
                    if flow_doy is not None:
                        flow_doy = flow_doy.to(device)
                masked = base_for_fm.flow_loss(
                    pred_norm=pred_tensor,
                    target_norm=target_tensor,
                    var_name=aurora_name,
                    kind=target_spec.kind,
                    doy=flow_doy,
                )
                # Cache normalised tensors for the coherence term (computed
                # once per lead after this inner loop).
                if coherence_weight > 0.0:
                    coh_cache[aurora_name] = {
                        "pred": pred_tensor,
                        "target": target_tensor,
                        "kind": target_spec.kind,
                        "levels": var_levels,
                    }
                # Collect the per-lead normalised (base) prediction and target
                # so the Mamba temporal module can be trained on the ordered
                # rollout sequence after the lead loop.
                if temporal_weight > 0.0 and getattr(base_for_fm, "has_temporal", False):
                    slot = temporal_seq.setdefault(
                        aurora_name,
                        {"kind": target_spec.kind, "preds": [], "targets": []},
                    )
                    slot["preds"].append(pred_tensor)
                    slot["targets"].append(target_tensor)
            else:
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

            # ---- DIAGNOSTIC (one-shot, rank 0) ----
            if _DIAG_LOSS_BREAKDOWN.get("active", False):
                with torch.no_grad():
                    _DIAG_LOSS_BREAKDOWN.setdefault("rows", []).append({
                        "var": aurora_name,
                        "lead": lead,
                        "kind": target_spec.kind,
                        "pred_abs_max": float(pred_tensor.detach().abs().max().item()),
                        "tgt_abs_max": float(target_tensor.detach().abs().max().item()),
                        "diff_abs_max": float((pred_tensor - target_tensor).detach().abs().max().item()),
                        "loss_var": float(masked.detach().item()),
                        "weight": weight,
                    })
            # ---- end diagnostic ----

        # ---- Column / profile coherence (cross-variable, per lead) ----
        if coherence_weight > 0.0 and len(coh_cache) >= 2:
            col_name, prof_name = _resolve_coherence_pair(
                coh_cache, coherence_col_var, coherence_prof_var,
            )
            if col_name is not None and prof_name is not None:
                col = coh_cache[col_name]
                prof = coh_cache[prof_name]
                coh = base_for_fm.coherence_loss(
                    profile_pred_norm=prof["pred"],
                    profile_tgt_norm=prof["target"],
                    column_pred_norm=col["pred"],
                    column_tgt_norm=col["target"],
                    level_pressures=prof["levels"] or [],
                    profile_var=prof_name,
                    column_var=col_name,
                )
                total_loss = total_loss + coherence_weight * coh
                total_weight += coherence_weight

    # ---- Mamba temporal loss (sequential, cross-lead) ----
    # Trains the temporal module on the ordered rollout sequence so it learns
    # how the flow-corrected field and Aurora's residual evolve through time,
    # rather than treating each lead independently. The flow-corrected frames
    # are detached, so this term updates *only* the Mamba parameters and leaves
    # the trained flow head / backbone untouched.
    temporal_metrics: dict[str, float] = {}
    if temporal_weight > 0.0 and temporal_seq:
        base_for_fm = model.module if hasattr(model, "module") else model
        if getattr(base_for_fm, "has_temporal", False):
            for aurora_name, slot in temporal_seq.items():
                preds = slot["preds"]
                tgts = slot["targets"]
                kind = slot["kind"]
                if len(preds) < 2:
                    # A temporal model needs at least two ordered steps.
                    continue
                with torch.no_grad():
                    flow_frames = [
                        base_for_fm.refine_norm_deterministic(p, aurora_name, kind).detach()
                        for p in preds
                    ]
                seq = torch.stack(flow_frames, dim=1)             # (B,S,[L,]H,W)
                tgt_seq = torch.stack([t.detach() for t in tgts], dim=1)
                corr = base_for_fm.temporal_residual(seq, aurora_name, kind)
                temporal_pred = seq + corr
                t_loss = torch.nn.functional.mse_loss(temporal_pred, tgt_seq)
                total_loss = total_loss + temporal_weight * t_loss
                total_weight += temporal_weight
                temporal_metrics[f"temporal_loss/{aurora_name}"] = float(t_loss.detach())

    if total_weight <= 0:
        raise ValueError("Total loss weight evaluated to <= 0. Check multi_target_loss_weights.")

    total_loss = total_loss / total_weight

    metrics = {
        "batch_size": len(sample_list),
        "lead_times": lead_times,
        "spatial_shape": batch.spatial_shape,
    }
    if temporal_metrics:
        metrics["temporal"] = temporal_metrics
    return total_loss, metrics


def maybe_wrap_conv_refine(
    model: torch.nn.Module,
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
) -> torch.nn.Module:
    """Optionally wrap *model* with convolutional refinement heads.

    Returns the original model unchanged if ``model.conv_refine_enabled`` is
    false in the config. If ``flow_refine.enabled`` is true, this helper
    skips conv-refine in favor of :func:`maybe_wrap_flow_refine` (the two
    wrappers are mutually exclusive).
    """
    model_cfg = config.get("model", {})
    if bool(model_cfg.get("flow_refine_enabled", False)):
        return model  # flow refine takes precedence
    if not bool(model_cfg.get("conv_refine_enabled", False)):
        return model

    from finetune.conv_refine import AuroraConvRefine

    target_surf = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "surf"
    )
    target_atmos = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "atmos"
    )
    hidden = int(model_cfg.get("conv_refine_hidden", 32))

    wrapper = AuroraConvRefine(
        base=model,
        target_surf_vars=target_surf,
        target_atmos_vars=target_atmos,
        hidden=hidden,
    )
    return wrapper


def maybe_wrap_flow_refine(
    model: torch.nn.Module,
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
) -> torch.nn.Module:
    """Optionally wrap *model* with rectified-flow residual refine heads.

    Enabled via ``model.flow_refine_enabled = true`` in config. Mutually
    exclusive with :func:`maybe_wrap_conv_refine`. Norm stats must be
    attached after construction by the training driver (call
    ``model.set_norm_stats(...)`` once stats have been computed) so the
    wrapper can de-normalise sampled residuals at inference time.
    """
    model_cfg = config.get("model", {})
    if not bool(model_cfg.get("flow_refine_enabled", False)):
        return model

    from finetune.flow_refine import AuroraFlowRefine

    target_surf = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "surf"
    )
    target_atmos = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "atmos"
    )
    hidden = int(model_cfg.get("flow_refine_hidden", 64))
    time_dim = int(model_cfg.get("flow_refine_time_dim", 128))
    sampling_steps = int(model_cfg.get("flow_refine_sampling_steps", 8))
    doy_cond = bool(model_cfg.get("flow_refine_doy_cond", False))
    residual_zscore = bool(model_cfg.get("flow_refine_residual_zscore", False))
    res_std_momentum = float(model_cfg.get("flow_refine_res_std_momentum", 0.99))

    # Mamba temporal module (optional; default off → backward compatible).
    temporal_enabled = bool(model_cfg.get("mamba_temporal_enabled", False))
    temporal_channels = int(model_cfg.get("mamba_temporal_channels", 16))
    temporal_state = int(model_cfg.get("mamba_temporal_state", 8))
    temporal_layers = int(model_cfg.get("mamba_temporal_layers", 2))
    temporal_conv = int(model_cfg.get("mamba_temporal_conv", 3))
    temporal_expand = int(model_cfg.get("mamba_temporal_expand", 2))

    # Build per-variable loss_levels → level-index mapping so the wrapper
    # only applies bias correction to the configured levels at inference.
    data_cfg = config.get("data", {})
    full_levels = [
        float(lv)
        for lv in data_cfg.get("atmos_levels", data_cfg.get("pressure_levels", []))
    ]
    atmos_loss_levels: dict[str, list[int]] = {}
    for spec in resolved_specs.targets:
        if spec.kind == "atmos" and spec.loss_levels is not None:
            idx = [full_levels.index(float(lv)) for lv in spec.loss_levels]
            atmos_loss_levels[spec.aurora_name] = idx

    wrapper = AuroraFlowRefine(
        base=model,
        target_surf_vars=target_surf,
        target_atmos_vars=target_atmos,
        hidden=hidden,
        time_dim=time_dim,
        sampling_steps=sampling_steps,
        atmos_loss_levels=atmos_loss_levels if atmos_loss_levels else None,
        doy_cond=doy_cond,
        residual_zscore=residual_zscore,
        res_std_momentum=res_std_momentum,
        temporal_enabled=temporal_enabled,
        temporal_channels=temporal_channels,
        temporal_state=temporal_state,
        temporal_layers=temporal_layers,
        temporal_conv=temporal_conv,
        temporal_expand=temporal_expand,
    )

    # Structural auxiliary-loss weights (extreme-event, spatial-pattern,
    # distributional, vertical-profile, column/profile coherence). Read from
    # the training.flow_aux_loss block; absent → all zero → pure residual MSE.
    aux_cfg = config.get("training", {}).get("flow_aux_loss", {})
    wrapper.set_aux_loss_config(aux_cfg)
    return wrapper


def configure_trainable_parameters(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, int]:
    """Apply freeze/unfreeze config and return parameter summary."""
    model_cfg = config.get("model", {})

    # When using a refine wrapper, apply freeze logic to the base model.
    from finetune.conv_refine import AuroraConvRefine
    try:
        from finetune.flow_refine import AuroraFlowRefine
    except Exception:
        AuroraFlowRefine = None  # type: ignore[assignment]

    is_conv_refine = isinstance(model, AuroraConvRefine)
    is_flow_refine = (
        AuroraFlowRefine is not None and isinstance(model, AuroraFlowRefine)
    )
    base = model.base if (is_conv_refine or is_flow_refine) else model

    for param in base.parameters():
        param.requires_grad = True

    backbone_freeze = bool(model_cfg.get("backbone_freeze", False))
    freeze_embeddings = bool(model_cfg.get("freeze_embeddings", False))
    freeze_encoder = bool(model_cfg.get("freeze_encoder", False))
    freeze_decoder = bool(model_cfg.get("freeze_decoder", False))
    trainable_head_only = bool(model_cfg.get("trainable_head_only", False))

    if backbone_freeze and hasattr(base, "backbone"):
        for param in base.backbone.parameters():
            param.requires_grad = False

    if freeze_encoder and hasattr(base, "encoder"):
        for param in base.encoder.parameters():
            param.requires_grad = False

    if freeze_decoder and hasattr(base, "decoder"):
        for param in base.decoder.parameters():
            param.requires_grad = False

    if freeze_embeddings:
        for name, param in base.named_parameters():
            if "token_embeds" in name or "levels_embed" in name or "patch_embedding" in name:
                # Don't freeze decoder embeddings when the decoder is being trained.
                if not freeze_decoder and name.startswith("decoder."):
                    continue
                param.requires_grad = False

    if trainable_head_only:
        for param in base.parameters():
            param.requires_grad = False

        for name, param in base.named_parameters():
            if (
                "decoder.surf_heads" in name
                or "decoder.atmos_heads" in name
                or "decoder.modulation_heads" in name
                or "surf_feature_combiner" in name
                or "atmos_feature_combiner" in name
            ):
                param.requires_grad = True

    # Conv refinement heads are always trainable.
    if is_conv_refine:
        for param in model.surf_refine.parameters():
            param.requires_grad = True
        for param in model.atmos_refine.parameters():
            param.requires_grad = True

    # Flow-matching refinement heads are always trainable.
    if is_flow_refine:
        for param in model.surf_flow.parameters():
            param.requires_grad = True
        for param in model.atmos_flow.parameters():
            param.requires_grad = True
        # Mamba temporal module (when enabled) is always trainable.
        if getattr(model, "temporal", None) is not None:
            for param in model.temporal.parameters():
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

    if scheduler_name == "cosine_warmup":
        warmup_steps = int(training_cfg.get("scheduler_warmup_steps", 10))
        t_max = int(training_cfg.get("scheduler_t_max", max(1, num_training_steps)))
        eta_min = float(training_cfg.get("scheduler_eta_min", 0.0))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, t_max - warmup_steps)
            return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

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
        # Fall back to max(target_lead_times) from the data config.
        lead_times = config.get("data", {}).get("target_lead_times", [])
        if lead_times:
            steps = max(int(x) for x in lead_times)
    if steps <= 0:
        return []

    autoregressive = bool(rollout_cfg.get("autoregressive_inputs", True))
    feedback_fields_cfg = rollout_cfg.get("predicted_fields_get_fed_back") or rollout_cfg.get(
        "predicted_fields_feedback"
    )
    if feedback_fields_cfg:
        feedback_fields = set(str(x) for x in feedback_fields_cfg)
    else:
        # Default: only feed back target variable predictions to prevent
        # unsupervised variables from corrupting the autoregressive rollout.
        feedback_fields = {spec.aurora_name for spec in resolved_specs.targets}

    keep_exogenous_mode = str(
        rollout_cfg.get("keep_exogenous_predictors", "fixed")
    ).lower()
    refresh_exogenous = keep_exogenous_mode in {"refresh", "refresh_from_dataset", "dataset"}

    device = torch.device(device)
    model.eval()

    # Target variables that are bias-corrected. These must be advanced by the
    # MODEL's prediction during rollout and must NEVER be overwritten with
    # future CAMS truth (that would leak the answer). Everything else is
    # context/exogenous and is refreshed from CAMS when available.
    target_var_names = {spec.aurora_name for spec in resolved_specs.targets}

    # Safeguard: the feedback set (model-advanced vars) must contain every
    # bias-corrected target, otherwise a target would be pulled from CAMS.
    missing_fb = target_var_names - feedback_fields
    assert not missing_fb, (
        "Rollout misconfiguration: bias-corrected target variable(s) "
        f"{sorted(missing_fb)} are not in the feedback set {sorted(feedback_fields)}; "
        "they would be overwritten by CAMS truth during rollout."
    )

    # Optional Mamba temporal correction during rollout.
    base_for_fm = model.module if hasattr(model, "module") else model
    try:
        from finetune.flow_refine import AuroraFlowRefine as _AFR
    except Exception:
        _AFR = None
    temporal_active = (
        _AFR is not None
        and isinstance(base_for_fm, _AFR)
        and getattr(base_for_fm, "has_temporal", False)
    )
    temporal_history: dict[tuple[str, str], list[torch.Tensor]] = {}
    verbose_provenance = bool(rollout_cfg.get("verbose_provenance", True))

    current = build_aurora_batch(
        ds,
        start_sample,
        config=config,
        resolved_specs=resolved_specs,
    ).to(device)

    predictor_by_aurora = resolved_specs.predictor_by_aurora
    anchor_idx = int(start_sample["anchor_index"])
    spatial_h, spatial_w = current.spatial_shape

    predictions: list[Batch] = []
    with torch.inference_mode():
        for step in range(1, steps + 1):
            pred = model(current)

            # Mamba temporal correction (causal): refine the flow-corrected
            # target fields using their evolution across the rollout so far.
            if temporal_active:
                pred = base_for_fm.apply_temporal_rollout(pred, temporal_history)
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

            # Provenance tracking (items: verify CAMS vs model per variable).
            prov_model: list[str] = []   # advanced by model prediction
            prov_cams: list[str] = []    # refreshed from CAMS dataset
            prov_carry: list[str] = []   # carried forward (last step repeated)

            surf_next: dict[str, torch.Tensor] = {}
            for name, old in current.surf_vars.items():
                use_prediction = (feedback_fields is None and name in pred.surf_vars) or (
                    feedback_fields is not None and name in feedback_fields
                )
                if use_prediction:
                    new_frame = pred.surf_vars[name]
                    prov_model.append(name)
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
                        new_frame = new_frame[..., :spatial_h, :spatial_w]
                        prov_cams.append(name)
                    else:
                        new_frame = old[:, -1:]
                        prov_carry.append(name)
                else:
                    new_frame = old[:, -1:]
                    prov_carry.append(name)

                surf_next[name] = torch.cat([old[:, 1:], new_frame], dim=1)

            atmos_next: dict[str, torch.Tensor] = {}
            for name, old in current.atmos_vars.items():
                use_prediction = (feedback_fields is None and name in pred.atmos_vars) or (
                    feedback_fields is not None and name in feedback_fields
                )
                if use_prediction:
                    new_frame = pred.atmos_vars[name]
                    prov_model.append(name)
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
                        new_frame = new_frame[..., :spatial_h, :spatial_w]
                        prov_cams.append(name)
                    else:
                        new_frame = old[:, -1:]
                        prov_carry.append(name)
                else:
                    new_frame = old[:, -1:]
                    prov_carry.append(name)

                atmos_next[name] = torch.cat([old[:, 1:], new_frame], dim=1)

            # ---- Safeguards (items 6–8): verify the CAMS/model split ----
            # 1) No bias-corrected target may be sourced from CAMS truth.
            leaked = target_var_names & set(prov_cams)
            assert not leaked, (
                f"[rollout step {step}] target variable(s) {sorted(leaked)} were "
                "refreshed from CAMS truth — they must use model predictions only."
            )
            # 2) Every bias-corrected target must be advanced by the model.
            not_modeled = target_var_names - set(prov_model)
            assert not not_modeled, (
                f"[rollout step {step}] target variable(s) {sorted(not_modeled)} were "
                "not advanced by the model prediction during rollout."
            )
            if verbose_provenance:
                logger.info(
                    "[rollout step %d → t=%d] model/bias-corrected=%s | CAMS=%s | carried=%s",
                    step, next_time_index,
                    sorted(prov_model), sorted(prov_cams), sorted(prov_carry),
                )

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
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Persist flow-refine sampling_steps so inference uses the same phase
    # (1-step deterministic vs multi-step stochastic) that training settled on.
    _inner = model.module if hasattr(model, "module") else model
    flow_sampling_steps: int | None = None
    try:
        from finetune.flow_refine import AuroraFlowRefine as _AFR  # noqa: PLC0415
        if isinstance(_inner, _AFR):
            flow_sampling_steps = int(_inner.sampling_steps)
    except Exception:
        pass

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    if flow_sampling_steps is not None:
        payload["flow_sampling_steps"] = flow_sampling_steps
    if norm_stats is not None:
        payload["norm_stats"] = {
            k: {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
            for k, v in norm_stats.items()
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


def _smooth_patch_artifacts(arr: np.ndarray, sigma: float, patch_size: int) -> np.ndarray:
    """Apply Gaussian smoothing to remove patch-boundary artifacts.

    Works on 2-D (H, W) or higher-dimensional arrays by smoothing the last
    two spatial dimensions independently per slice.  Uses
    ``scipy.ndimage.gaussian_filter`` when available, otherwise falls back to a
    simple uniform box filter that is almost as effective for small sigma.

    The smoothing preserves the overall magnitude and spatial structure while
    blending the sharp patch-boundary discontinuities produced by the ViT
    decoder's ``unpatchify`` operation.
    """
    if sigma <= 0:
        return arr

    try:
        from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

        # Smooth only the last two (lat, lon) dimensions.
        axes = tuple(range(arr.ndim - 2, arr.ndim))  # e.g. (-2, -1)
        return gaussian_filter(arr.astype(np.float64), sigma=sigma, axes=axes).astype(arr.dtype)
    except ImportError:
        pass

    # Fallback: uniform box filter with kernel_size ~ 2*sigma+1
    kernel_size = max(3, int(2 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    def _smooth_2d(img: np.ndarray) -> np.ndarray:
        padded = np.pad(img, pad, mode="reflect")
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size**2)
        from numpy.lib.stride_tricks import sliding_window_view  # type: ignore[attr-defined]

        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        return (windows * kernel).sum(axis=(-2, -1)).astype(img.dtype)

    result = np.empty_like(arr)
    it = np.nditer(arr[..., 0, 0], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        result[idx] = _smooth_2d(arr[idx])
        it.iternext()
    return result


def save_predictions(
    predictions: Sequence[Batch],
    output_path: str | Path,
    *,
    save_netcdf: bool = True,
    resolved_specs: "ResolvedVariableSpecs | None" = None,
    smooth_sigma: float = 0.0,
    patch_size: int = 3,
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

    When *smooth_sigma* > 0, a Gaussian filter with this sigma (in grid cells)
    is applied to the spatial dimensions to remove patch-boundary artifacts
    from the ViT decoder.
    """
    if not predictions:
        raise ValueError("No predictions were provided to save_predictions.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build aurora_name -> dataset_name reverse mappings from resolved_specs.
    # Only save target variables — exogenous predictor outputs are unsupervised
    # and produce meaningless blocky artifacts.
    surf_name_map: dict[str, str] = {}
    atmos_name_map: dict[str, str] = {}
    target_aurora_names: set[str] | None = None
    if resolved_specs is not None:
        target_aurora_names = set()
        for spec in resolved_specs.targets:
            target_aurora_names.add(spec.aurora_name)
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
        if target_aurora_names is not None and var_name not in target_aurora_names:
            continue
        # pred.surf_vars[name] shape: (batch, 1, H, W) — take batch=0, history=0
        arr = np.stack(
            [pred.surf_vars[var_name][0, 0].detach().cpu().float().numpy() for pred in predictions],
            axis=0,
        )  # (time, lat, lon)
        if smooth_sigma > 0:
            arr = _smooth_patch_artifacts(arr, sigma=smooth_sigma, patch_size=patch_size)
        out_name = surf_name_map.get(var_name, var_name)
        data_vars[out_name] = (("time", "latitude", "longitude"), arr)

    for var_name in first.atmos_vars:
        if target_aurora_names is not None and var_name not in target_aurora_names:
            continue
        arr = np.stack(
            [pred.atmos_vars[var_name][0, 0].detach().cpu().float().numpy() for pred in predictions],
            axis=0,
        )  # (time, level, lat, lon)
        if smooth_sigma > 0:
            arr = _smooth_patch_artifacts(arr, sigma=smooth_sigma, patch_size=patch_size)
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
