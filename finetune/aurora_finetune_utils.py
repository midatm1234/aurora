"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Utilities for config-driven Aurora fine-tuning and rollout workflows."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import logging
import math
import pickle
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from finetune.checkpoint_selection import (
    refinement_checkpoint_status,
    select_refinement_checkpoint,
    validate_checkpoint_validation_provenance,
)

from aurora import Batch, Metadata

try:
    import yaml
except ImportError:  # pragma: no cover - dependency check is runtime-facing.
    yaml = None


logger = logging.getLogger("aurora.finetune")


__all__ = [
    "VariableSpec",
    "ResolvedVariableSpecs",
    "validate_config",
    "validate_dataset_contract",
    "load_config",
    "set_seed",
    "restore_checkpoint_rng_state",
    "derive_refinement_seed",
    "open_dataset",
    "validate_longitude_consistency",
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
    "refinement_checkpoint_status",
    "select_refinement_checkpoint",
    "validate_checkpoint_validation_provenance",
    "load_checkpoint_if_available",
    "validate_checkpoint_longitude",
    "validate_checkpoint_refinement_contract",
    "write_training_history",
    "write_run_manifest",
    "save_predictions",
    "build_finetune_model",
    "load_model_from_checkpoint",
    "maybe_wrap_conv_refine",
    "maybe_wrap_flow_refine",
    "maybe_wrap_stochastic_refine",
]


@dataclass(frozen=True)
class VariableSpec:
    """Resolved variable metadata for notebook fine-tuning pipelines."""

    dataset_name: str
    aurora_name: str
    kind: str  # surf | atmos | static
    loss_levels: tuple[float, ...] | None = None
    units: str = ""


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


def validate_config(
    config: dict[str, Any],
    source: str | Path | None = None,
) -> None:
    """Validate and normalize the shared fine-tuning/inference configuration.

    The notebooks, distributed trainer, and inference workflow all call
    :func:`load_config`, so this is the single schema boundary for both regional
    NO2 and global O3 cases. Validation intentionally rejects settings that are
    present in YAML but not implemented by the workflow instead of silently
    ignoring them.
    """
    location = f" at {Path(source).resolve()}" if source is not None else ""
    prefix = f"Configuration{location}"
    if not isinstance(config, dict):
        raise TypeError(f"{prefix} must be a mapping.")

    case_name = str(config.get("case_name", "")).strip()
    if not case_name:
        raise ValueError(f"{prefix} must define a non-empty top-level `case_name`.")

    # Preparation/path-only callers historically use a deliberately minimal
    # config without model/training/rollout sections. Preserve that narrow API;
    # any config that declares workflow variables is validated as a complete
    # fine-tuning/inference recipe below.
    minimal_data = config.get("data", {})
    workflow_declared = (
        any(name in config for name in ("model", "training", "rollout"))
        or (
            isinstance(minimal_data, dict)
            and any(
                key in minimal_data
                for key in ("predictor_variables", "target_variables", "atmos_levels")
            )
        )
    )
    if not workflow_declared:
        if not isinstance(config.get("paths"), dict):
            raise ValueError(f"{prefix} must define `paths` as a mapping.")
        if not isinstance(minimal_data, dict):
            raise ValueError(f"{prefix} must define `data` as a mapping.")
        return

    sections: dict[str, dict[str, Any]] = {}
    for name in ("paths", "data", "model", "training", "rollout"):
        value = config.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{prefix} must define `{name}` as a mapping.")
        sections[name] = value
    paths_cfg = sections["paths"]
    data_cfg = sections["data"]
    model_cfg = sections["model"]
    training_cfg = sections["training"]
    rollout_cfg = sections["rollout"]

    def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be numeric, got {value!r}.")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric, got {value!r}.") from exc
        if not np.isfinite(result) or (positive and result <= 0):
            qualifier = "finite and positive" if positive else "finite"
            raise ValueError(f"{label} must be {qualifier}, got {value!r}.")
        return result

    def positive_integer(value: Any, label: str) -> int:
        number = finite_number(value, label, positive=True)
        integer = int(number)
        if number != integer:
            raise ValueError(f"{label} must be an integer, got {value!r}.")
        return integer

    def variable_entries(
        key: str,
        *,
        allowed_kinds: set[str],
        required: bool,
    ) -> list[dict[str, Any]]:
        raw = data_cfg.get(key, [])
        if not isinstance(raw, (list, tuple)) or (required and not raw):
            requirement = "a non-empty list" if required else "a list"
            raise ValueError(f"data.{key} must be {requirement}.")
        entries: list[dict[str, Any]] = []
        seen_dataset: set[str] = set()
        seen_aurora: set[str] = set()
        for index, item in enumerate(raw):
            label = f"data.{key}[{index}]"
            if isinstance(item, str):
                dataset_name = item.strip()
                aurora_name = dataset_name
                kind = None
                loss_levels = None
            elif isinstance(item, dict):
                dataset_name = str(item.get("dataset_name") or item.get("name") or "").strip()
                aurora_name = str(item.get("aurora_name") or dataset_name).strip()
                kind_value = item.get("kind")
                kind = None if kind_value is None else str(kind_value).strip().lower()
                loss_levels = item.get("loss_levels")
                units_value = item.get("units", "")
                if not isinstance(units_value, str):
                    raise ValueError(
                        f"{label}.units must be a string, got {units_value!r}."
                    )
            else:
                raise TypeError(f"{label} must be a string or mapping, got {type(item).__name__}.")
            if not dataset_name or not aurora_name:
                raise ValueError(f"{label} must define non-empty dataset and Aurora names.")
            if kind is not None and kind not in allowed_kinds:
                raise ValueError(
                    f"{label}.kind must be one of {sorted(allowed_kinds)}, got {kind!r}."
                )
            if dataset_name in seen_dataset:
                raise ValueError(f"data.{key} repeats dataset variable {dataset_name!r}.")
            if aurora_name in seen_aurora:
                raise ValueError(f"data.{key} repeats Aurora variable {aurora_name!r}.")
            seen_dataset.add(dataset_name)
            seen_aurora.add(aurora_name)
            entries.append(
                {
                    "dataset_name": dataset_name,
                    "aurora_name": aurora_name,
                    "kind": kind,
                    "loss_levels": loss_levels,
                    "label": label,
                }
            )
        return entries

    predictors = variable_entries(
        "predictor_variables", allowed_kinds={"surf", "atmos"}, required=True
    )
    targets = variable_entries(
        "target_variables", allowed_kinds={"surf", "atmos"}, required=True
    )
    variable_entries(
        "static_variables", allowed_kinds={"static"}, required=False
    )

    raw_levels = data_cfg.get("atmos_levels")
    if not isinstance(raw_levels, (list, tuple)) or not raw_levels:
        raise ValueError("data.atmos_levels must be a non-empty list of pressure levels.")
    levels = [
        finite_number(value, f"data.atmos_levels[{index}]", positive=True)
        for index, value in enumerate(raw_levels)
    ]
    if len(set(levels)) != len(levels):
        raise ValueError("data.atmos_levels must not contain duplicate pressure levels.")

    raw_leads = data_cfg.get("target_lead_times")
    if not isinstance(raw_leads, (list, tuple)) or not raw_leads:
        raise ValueError("data.target_lead_times must contain positive model-step indices.")
    target_leads = [
        positive_integer(value, f"data.target_lead_times[{index}]")
        for index, value in enumerate(raw_leads)
    ]
    if len(set(target_leads)) != len(target_leads):
        raise ValueError("data.target_lead_times must not contain duplicates.")
    positive_integer(data_cfg.get("input_time_steps", 2), "data.input_time_steps")
    if target_leads != sorted(target_leads):
        raise ValueError(
            f"data.target_lead_times must be strictly increasing, got {target_leads}."
        )
    target_type = str(data_cfg.get("target_type", "rollout")).strip().lower()
    if target_type not in {"next-step", "multi-step", "rollout"}:
        raise ValueError(
            "data.target_type must be next-step, multi-step, or rollout; "
            f"got {data_cfg.get('target_type')!r}."
        )
    nonnegative = data_cfg.get("nonnegative_target_variables", []) or []
    if not isinstance(nonnegative, (list, tuple)):
        raise ValueError("data.nonnegative_target_variables must be a list of target names.")
    target_names = {entry["dataset_name"] for entry in targets} | {
        entry["aurora_name"] for entry in targets
    }
    unknown_nonnegative = sorted({str(name) for name in nonnegative} - target_names)
    if unknown_nonnegative:
        raise ValueError(
            "data.nonnegative_target_variables contains names that are not configured "
            f"targets: {unknown_nonnegative}; expected a subset of {sorted(target_names)}."
        )

    for entry in targets:
        loss_levels_raw = entry["loss_levels"]
        if loss_levels_raw is None:
            continue
        if entry["kind"] != "atmos":
            raise ValueError(
                f"{entry['label']}.loss_levels is valid only for kind='atmos'."
            )
        if not isinstance(loss_levels_raw, (list, tuple)) or not loss_levels_raw:
            raise ValueError(f"{entry['label']}.loss_levels must be a non-empty list.")
        loss_levels = [
            finite_number(value, f"{entry['label']}.loss_levels[{index}]", positive=True)
            for index, value in enumerate(loss_levels_raw)
        ]
        if len(set(loss_levels)) != len(loss_levels):
            raise ValueError(f"{entry['label']}.loss_levels contains duplicates.")
        missing = [value for value in loss_levels if value not in levels]
        if missing:
            raise ValueError(
                f"{entry['label']}.loss_levels {missing} are absent from data.atmos_levels."
            )

    dim_names = [str(data_cfg.get(key, default)).strip() for key, default in (
        ("time_dim", "time"),
        ("lat_dim", "latitude"),
        ("lon_dim", "longitude"),
        ("level_dim", "level"),
    )]
    if any(not name for name in dim_names) or len(set(dim_names)) != len(dim_names):
        raise ValueError(
            "data.time_dim, lat_dim, lon_dim, and level_dim must be distinct non-empty names."
        )
    extra_indexers = data_cfg.get("extra_dim_indexers", {})
    if not isinstance(extra_indexers, dict):
        raise ValueError("data.extra_dim_indexers must be a mapping of dimension names to indices.")

    domain = str(data_cfg.get("domain_type", "")).strip().lower()
    if domain not in {"regional", "global"}:
        raise ValueError("data.domain_type must be `regional` or `global`.")
    data_cfg["domain_type"] = domain
    bound_keys = ("lat_min", "lat_max", "lon_min", "lon_max")
    for key in bound_keys:
        value = data_cfg.get(key)
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "~"}:
            data_cfg[key] = None
    if domain == "regional":
        missing_bounds = [key for key in bound_keys if data_cfg.get(key) is None]
        if missing_bounds:
            raise ValueError(
                "Regional data.domain_type requires finite bounds: " + ", ".join(missing_bounds)
            )
        lat_min = finite_number(data_cfg["lat_min"], "data.lat_min")
        lat_max = finite_number(data_cfg["lat_max"], "data.lat_max")
        lon_min = finite_number(data_cfg["lon_min"], "data.lon_min")
        lon_max = finite_number(data_cfg["lon_max"], "data.lon_max")
        if not -90 <= lat_min < lat_max <= 90:
            raise ValueError("Regional latitude bounds must satisfy -90 <= lat_min < lat_max <= 90.")
        lon_min_mod = lon_min % 360.0
        lon_max_mod = lon_max % 360.0
        if lon_min_mod >= lon_max_mod:
            raise ValueError(
                "Regional longitude bounds must define a non-wrapping interval after "
                "conversion to [0, 360)."
            )
    elif any(data_cfg.get(key) is not None for key in bound_keys):
        raise ValueError(
            "Global data.domain_type requires lat_min/lat_max/lon_min/lon_max to be null."
        )

    alignment = str(data_cfg.get("patch_alignment_strategy", "crop")).strip().lower()
    if alignment != "crop":
        raise ValueError("data.patch_alignment_strategy currently supports only `crop`.")
    normalization = data_cfg.get("normalization", {})
    if not isinstance(normalization, dict):
        raise ValueError("data.normalization must be a mapping.")
    normalization_enabled = normalization.get("enabled", False)
    if not isinstance(normalization_enabled, bool):
        raise ValueError("data.normalization.enabled must be true or false.")
    normalization_mode = str(normalization.get("mode", "none")).strip().lower()
    if normalization_enabled or normalization_mode != "none":
        raise ValueError(
            "Custom data.normalization is not implemented by this workflow. Set "
            "data.normalization.enabled=false and mode=none; Aurora target scales are "
            "derived consistently by compute_target_normalization_stats."
        )
    for stats_key in ("input_stats", "target_stats"):
        if not isinstance(normalization.get(stats_key, {}), dict):
            raise ValueError(f"data.normalization.{stats_key} must be a mapping.")

    if not isinstance(model_cfg.get("model_kwargs", {}), dict):
        raise ValueError("model.model_kwargs must be a mapping.")

    if not str(model_cfg.get("model_variant", "")).strip():
        raise ValueError("model.model_variant must be a non-empty model registry name.")
    positive_integer(model_cfg.get("patch_size", 0), "model.patch_size")
    mixed_precision = str(model_cfg.get("mixed_precision", "none")).strip().lower()
    if mixed_precision not in {"none", "off", "false", "bf16", "bfloat16", "fp16"}:
        raise ValueError("model.mixed_precision must be none, bf16, or fp16.")
    lon_periodic = model_cfg.get("lon_periodic", "auto")
    if not isinstance(lon_periodic, bool) and str(lon_periodic).strip().lower() not in {
        "auto", "true", "false", "1", "0", "yes", "no", "on", "off", "periodic", "regional",
    }:
        raise ValueError("model.lon_periodic must be true, false, or 'auto'.")

    flow_enabled = bool(model_cfg.get("flow_refine_enabled", False))
    conv_enabled = bool(model_cfg.get("conv_refine_enabled", False))
    if flow_enabled and conv_enabled:
        raise ValueError(
            "model.flow_refine_enabled and model.conv_refine_enabled are mutually exclusive."
        )

    # Unified stochastic residual refinement (`model.refinement` / `performance`).
    # Resolving the sections here surfaces invalid refiner names, Transformer
    # geometry, patch/window settings, diffusion schedules, prediction
    # parameterizations, flow solvers and incompatible cache settings as clear
    # configuration errors before any model is built. A configuration without
    # these sections resolves to the existing behaviour.
    from finetune.refinement.config import (
        ConfigValidationError,
        resolve_performance_config,
        resolve_refinement_config,
    )

    try:
        refinement_cfg = resolve_refinement_config(config)
        resolve_performance_config(config)
    except ConfigValidationError as exc:
        raise ValueError(str(exc)) from exc
    legacy_flow_selected = flow_enabled or refinement_cfg.backend == "legacy"
    flow_contract_version = (
        positive_integer(
            model_cfg.get("flow_refine_contract_version", 1),
            "model.flow_refine_contract_version",
        )
        if legacy_flow_selected
        else 1
    )
    if refinement_cfg.backend == "unified" and conv_enabled:
        raise ValueError(
            "model.conv_refine_enabled is mutually exclusive with a stochastic "
            f"model.refinement.type of {refinement_cfg.type!r}."
        )
    if refinement_cfg.backend == "unified" and flow_enabled:
        raise ValueError(
            "model.flow_refine_enabled (the legacy flow-matching UNet) is mutually "
            f"exclusive with model.refinement.type={refinement_cfg.type!r}. Use "
            "model.refinement.type: flow_matching_unet for the existing head."
        )
    if flow_enabled:
        positive_integer(
            model_cfg.get("flow_refine_contract_version", 1),
            "model.flow_refine_contract_version",
        )
        positive_integer(model_cfg.get("flow_refine_hidden", 64), "model.flow_refine_hidden")
        positive_integer(model_cfg.get("flow_refine_time_dim", 128), "model.flow_refine_time_dim")
        positive_integer(
            model_cfg.get("flow_refine_sampling_steps", 1),
            "model.flow_refine_sampling_steps",
        )
        positive_integer(
            model_cfg.get(
                "flow_refine_sampling_steps_late",
                model_cfg.get("flow_refine_sampling_steps", 1),
            ),
            "model.flow_refine_sampling_steps_late",
        )
        phase = finite_number(
            model_cfg.get("flow_refine_phase_fraction", 1.0 / 3.0),
            "model.flow_refine_phase_fraction",
            positive=True,
        )
        if phase > 1:
            raise ValueError("model.flow_refine_phase_fraction must be in (0, 1].")
        momentum = finite_number(
            model_cfg.get("flow_refine_res_std_momentum", 0.99),
            "model.flow_refine_res_std_momentum",
        )
        if not 0 <= momentum < 1:
            raise ValueError("model.flow_refine_res_std_momentum must be in [0, 1).")

    step_value = rollout_cfg.get("rollout_step_hours")
    step_hours = None if step_value is None else finite_number(
        step_value, "rollout.rollout_step_hours", positive=True
    )
    rollout_steps = int(rollout_cfg.get("rollout_num_steps", 0))
    if rollout_steps < 0:
        raise ValueError("rollout.rollout_num_steps must be >= 0.")
    keep_exogenous = str(
        rollout_cfg.get("keep_exogenous_predictors", "fixed")
    ).strip().lower()
    if keep_exogenous not in {
        "fixed", "carry", "carry_forward", "refresh", "refresh_from_dataset", "dataset"
    }:
        raise ValueError(
            "rollout.keep_exogenous_predictors must be fixed or refresh_from_dataset."
        )
    flow_stochastic_sampling = rollout_cfg.get(
        "flow_refine_stochastic_sampling", False
    )
    if not isinstance(flow_stochastic_sampling, bool):
        raise ValueError(
            "rollout.flow_refine_stochastic_sampling must be a boolean."
        )
    flow_autoregressive_feedback = training_cfg.get(
        "flow_refine_autoregressive_feedback", False
    )
    if not isinstance(flow_autoregressive_feedback, bool):
        raise ValueError(
            "training.flow_refine_autoregressive_feedback must be a boolean."
        )

    if flow_autoregressive_feedback and (
        not legacy_flow_selected or flow_contract_version < 2
    ):
        raise ValueError(
            "training.flow_refine_autoregressive_feedback=true is supported "
            "only by legacy flow_refine_contract_version >= 2."
        )

    if bool(model_cfg.get("flow_refine_lead_time_cond", False)):
        if not legacy_flow_selected:
            raise ValueError(
                "model.flow_refine_lead_time_cond requires flow_refine_enabled=true or "
                "model.refinement.type: flow_matching_unet."
            )
        if step_hours is None:
            raise ValueError(
                "model.flow_refine_lead_time_cond requires rollout.rollout_step_hours."
            )
        expected_scale = step_hours * max(target_leads)
        configured_scale = finite_number(
            model_cfg.get("flow_refine_lead_time_scale_hours", expected_scale),
            "model.flow_refine_lead_time_scale_hours",
            positive=True,
        )
        if not np.isclose(configured_scale, expected_scale, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "model.flow_refine_lead_time_scale_hours must equal the largest "
                f"supervised lead ({expected_scale:g} hours)."
            )

    from finetune.refinement.two_phase import resolve_temporal_config

    temporal_config = resolve_temporal_config(config)
    temporal_enabled = bool(temporal_config["enabled"])
    temporal_only = training_cfg.get("mamba_temporal_only", False)
    if not isinstance(temporal_only, bool):
        raise ValueError("training.mamba_temporal_only must be true or false.")
    if "mamba_temporal_weight" in training_cfg:
        configured_temporal_weight = finite_number(
            training_cfg["mamba_temporal_weight"],
            "training.mamba_temporal_weight",
        )
        if configured_temporal_weight < 0:
            raise ValueError(
                "training.mamba_temporal_weight must be >= 0, got "
                f"{training_cfg['mamba_temporal_weight']!r}."
            )
    if temporal_enabled:
        supported_temporal_backend = (
            legacy_flow_selected or refinement_cfg.backend == "unified"
        )
        if not supported_temporal_backend:
            raise ValueError(
                "model.mamba_temporal_enabled requires an active refinement backend "
                "(legacy flow_matching_unet or a unified flow/diffusion refiner); "
                f"actual model.refinement.type={refinement_cfg.type!r}."
            )
        if legacy_flow_selected and temporal_config["mode"] != "per_variable":
            raise ValueError(
                "model.mamba_temporal.mode=packed_joint is supported only by the "
                "unified refinement wrapper; legacy flow_matching_unet must use "
                "per_variable."
            )
        if legacy_flow_selected:
            unsupported_legacy_temporal = [
                key
                for key, default in (
                    ("dropout", 0.0),
                    ("gated_fusion", False),
                    ("gate_init", 0.0),
                    ("lead_time_conditioning", False),
                    ("mask_conditioning", False),
                    ("coordinate_conditioning", False),
                )
                if temporal_config[key] != default
            ]
            legacy_objective_defaults = {
                "base": "mse",
                "huber_delta": 1.0,
                "charbonnier_epsilon": 1.0e-3,
                "tendency_weight": 0.0,
                "structure_weight": 0.0,
                "extreme_weight": 0.0,
                "extreme_quantile": 0.95,
            }
            if temporal_config["objective"] != legacy_objective_defaults:
                unsupported_legacy_temporal.append("objective")
            if unsupported_legacy_temporal:
                raise ValueError(
                    "Legacy flow_matching_unet temporal Mamba does not implement "
                    "these model.mamba_temporal option(s): "
                    f"{unsupported_legacy_temporal}. Use their legacy-safe defaults "
                    "or a unified packed_joint head."
                )
            warnings.warn(
                "Legacy flow Mamba corrections are not included in validation "
                "checkpoint selection and its historical temporal training input "
                "does not match multi-step stochastic deployment. Use the unified "
                "packed_joint deterministic two-stage path for new studies.",
                RuntimeWarning,
                stacklevel=2,
            )
        if len(target_leads) < 2:
            raise ValueError(
                "model.mamba_temporal.enabled requires at least two "
                f"data.target_lead_times; actual value is {target_leads!r}."
            )
        expected_temporal_leads = list(range(1, max(target_leads) + 1))
        if target_leads != expected_temporal_leads:
            raise ValueError(
                "data.target_lead_times must be consecutive rollout indices "
                "[1, ..., max] when temporal Mamba is enabled, because it is "
                "applied at every inference step; expected "
                f"{expected_temporal_leads!r}, got {target_leads!r}."
            )
        if rollout_steps > max(target_leads):
            raise ValueError(
                "rollout.rollout_num_steps exceeds the trained temporal-Mamba "
                "horizon from data.target_lead_times; expected 0 or a value <= "
                f"{max(target_leads)}, got {rollout_steps}."
            )
        if step_hours is None:
            raise ValueError(
                "Temporal Mamba requires a positive rollout.rollout_step_hours; "
                f"actual value is {step_value!r}."
            )
        temporal_weight = (
            configured_temporal_weight
            if "mamba_temporal_weight" in training_cfg
            else 1.0
        )
        if temporal_weight <= 0:
            raise ValueError(
                "A training.mamba_temporal_weight > 0 is required when the "
                "temporal module is enabled."
            )
    elif temporal_only:
        raise ValueError(
            "training.mamba_temporal_only=true requires temporal Mamba to be enabled."
        )

    if (
        refinement_cfg.is_active
        and refinement_cfg.conditioning.forecast_lead_time
        and step_hours is None
    ):
        raise ValueError(
            "refinement.conditioning.forecast_lead_time requires "
            "rollout.rollout_step_hours so the physical forecast lead time is defined."
        )

    positive_integer(training_cfg.get("batch_size", 0), "training.batch_size")
    positive_integer(training_cfg.get("num_epochs", 0), "training.num_epochs")
    finite_number(training_cfg.get("learning_rate", 0), "training.learning_rate", positive=True)
    positive_integer(training_cfg.get("accumulation_steps", 1), "training.accumulation_steps")
    positive_integer(training_cfg.get("validation_frequency", 1), "training.validation_frequency")
    positive_integer(
        training_cfg.get("validation_refinement_ensemble_size", 1),
        "training.validation_refinement_ensemble_size",
    )
    checkpoint_metric = str(
        training_cfg.get("checkpoint_metric", "validation_loss")
    ).strip().lower()
    if checkpoint_metric not in {
        "validation_loss",
        "val_loss",
        "mean_physical_rmse_ratio",
    }:
        raise ValueError(
            "training.checkpoint_metric must be validation_loss or "
            "mean_physical_rmse_ratio."
        )
    raw_guard_metrics = training_cfg.get("checkpoint_guard_metrics", [])
    if not isinstance(raw_guard_metrics, (list, tuple)):
        raise ValueError(
            "training.checkpoint_guard_metrics must be a list containing only "
            "mae, absolute_bias, and/or pattern_correlation."
        )
    allowed_guard_metrics = {"mae", "absolute_bias", "pattern_correlation"}
    checkpoint_guard_metrics: list[str] = []
    for index, value in enumerate(raw_guard_metrics):
        if not isinstance(value, str):
            raise ValueError(
                f"training.checkpoint_guard_metrics[{index}] must be one of "
                f"{sorted(allowed_guard_metrics)}, got {value!r}."
            )
        metric = value.strip().lower()
        if metric not in allowed_guard_metrics:
            raise ValueError(
                f"training.checkpoint_guard_metrics[{index}] must be one of "
                f"{sorted(allowed_guard_metrics)}, got {value!r}."
            )
        if metric in checkpoint_guard_metrics:
            raise ValueError(
                "training.checkpoint_guard_metrics must not contain duplicate "
                f"metric {metric!r}."
            )
        checkpoint_guard_metrics.append(metric)
    training_cfg["checkpoint_guard_metrics"] = checkpoint_guard_metrics

    for key in (
        "checkpoint_guard_relative_tolerance",
        "checkpoint_guard_correlation_tolerance",
        "checkpoint_guard_bias_rmse_floor_fraction",
    ):
        value = finite_number(training_cfg.get(key, 0.0), f"training.{key}")
        if value < 0:
            raise ValueError(f"training.{key} must be >= 0, got {value!r}.")
        training_cfg[key] = value

    for boolean_key in (
        "require_refinement_improvement",
        "require_all_physical_channels_improve",
    ):
        if boolean_key in training_cfg and not isinstance(training_cfg[boolean_key], bool):
            raise ValueError(f"training.{boolean_key} must be true or false.")

    validation_source = str(
        training_cfg.get("validation_source", "configured")
    ).strip().lower()
    if validation_source not in {"configured", "train_tail"}:
        raise ValueError("training.validation_source must be `configured` or `train_tail`.")
    if validation_source == "train_tail":
        validation_fraction = finite_number(
            training_cfg.get("validation_fraction", 0.1),
            "training.validation_fraction",
            positive=True,
        )
        if validation_fraction >= 0.5:
            raise ValueError("training.validation_fraction must be between 0 and 0.5.")

    target_lookup: dict[str, dict[str, Any]] = {}
    for entry in targets:
        target_lookup[entry["dataset_name"]] = entry
        target_lookup[entry["aurora_name"]] = entry
    aux_cfg = training_cfg.get("flow_aux_loss", {}) or {}
    if not isinstance(aux_cfg, dict):
        raise ValueError("training.flow_aux_loss must be a mapping.")
    extreme_tail = str(aux_cfg.get("extreme_tail", "both")).strip().lower()
    if extreme_tail not in {"both", "upper"}:
        raise ValueError(
            "training.flow_aux_loss.extreme_tail must be 'both' or 'upper'."
        )
    coherence_weight = finite_number(
        aux_cfg.get("coherence_weight", 0.0),
        "training.flow_aux_loss.coherence_weight",
    )
    if coherence_weight < 0:
        raise ValueError("training.flow_aux_loss.coherence_weight must be >= 0.")
    column_name = str(aux_cfg.get("coherence_column_var", "") or "").strip()
    profile_name = str(aux_cfg.get("coherence_profile_var", "") or "").strip()
    # Explicit variable names are schema, even when the term currently has
    # zero weight or the auxiliary block is disabled. Validating them eagerly
    # prevents a dormant typo from becoming a delayed training failure when
    # the weight is later enabled.
    if column_name or profile_name:
        if bool(column_name) != bool(profile_name):
            raise ValueError(
                "Set both coherence_column_var and coherence_profile_var, or leave both empty."
            )
        if column_name not in target_lookup:
            raise ValueError(
                "training.flow_aux_loss.coherence_column_var "
                f"{column_name!r} is not a configured target."
            )
        if profile_name not in target_lookup:
            raise ValueError(
                "training.flow_aux_loss.coherence_profile_var "
                f"{profile_name!r} is not a configured target."
            )
        if target_lookup[column_name]["kind"] != "surf":
            raise ValueError("coherence_column_var must identify a surface target.")
        if target_lookup[profile_name]["kind"] != "atmos":
            raise ValueError("coherence_profile_var must identify an atmospheric target.")
    elif bool(aux_cfg.get("enabled", True)) and coherence_weight > 0:
        if not any(entry["kind"] == "surf" for entry in targets) or not any(
            entry["kind"] == "atmos" for entry in targets
        ):
            raise ValueError(
                "Automatic coherence pairing requires one surface and one atmospheric target."
            )

    feedback = rollout_cfg.get("predicted_fields_get_fed_back") or rollout_cfg.get(
        "predicted_fields_feedback"
    )
    if feedback:
        if not isinstance(feedback, (list, tuple)):
            raise ValueError("rollout.predicted_fields_get_fed_back must be a list.")
        feedback_names = {str(value) for value in feedback}
        target_names = {entry["aurora_name"] for entry in targets}
        missing_feedback = target_names - feedback_names
        if missing_feedback:
            raise ValueError(
                "rollout.predicted_fields_get_fed_back must include every target Aurora "
                f"name; missing {sorted(missing_feedback)}."
            )
        predictor_names = {entry["aurora_name"] for entry in predictors}
        unknown_feedback = feedback_names - predictor_names
        if unknown_feedback:
            raise ValueError(
                "rollout.predicted_fields_get_fed_back contains unknown predictor names: "
                f"{sorted(unknown_feedback)}."
            )

    notebook_cfg = config.get("notebook", {}) or {}
    if not isinstance(notebook_cfg, dict):
        raise ValueError("notebook must be a mapping when provided.")
    plot_variables = notebook_cfg.get("plot_variables", []) or []
    if not isinstance(plot_variables, (list, tuple)):
        raise ValueError("notebook.plot_variables must be a list of target names.")
    allowed_plot_names = {
        name for entry in targets for name in (entry["dataset_name"], entry["aurora_name"])
    }
    unknown_plot_names = [
        str(name) for name in plot_variables if str(name) not in allowed_plot_names
    ]
    if unknown_plot_names:
        raise ValueError(
            "notebook.plot_variables contains names that are not saved target outputs: "
            f"{unknown_plot_names}. Allowed names: {sorted(allowed_plot_names)}."
        )

    if not isinstance(paths_cfg.get("project_root", "."), (str, Path)):
        raise ValueError("paths.project_root must be a filesystem path.")


def validate_dataset_contract(
    ds: xr.Dataset,
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
) -> None:
    """Validate variable dimensions, pressure levels, cadence, and patch geometry."""
    time_dim, lat_dim, lon_dim, level_dim = _dim_names(config)
    extra_indexers = config.get("data", {}).get("extra_dim_indexers", {}) or {}
    allowed_extra = set(extra_indexers)

    for group_name, specs in (
        ("predictor", resolved_specs.predictors),
        ("target", resolved_specs.targets),
        ("static", resolved_specs.static),
    ):
        for spec in specs:
            da = ds[spec.dataset_name]
            dims = set(da.dims)
            required = {lat_dim, lon_dim}
            if group_name != "static":
                required.add(time_dim)
            if spec.kind == "atmos":
                required.add(level_dim)
            missing = sorted(required - dims)
            if missing:
                raise ValueError(
                    f"Configured {group_name} variable {spec.dataset_name!r} has kind "
                    f"{spec.kind!r} but is missing dimension(s) {missing}; got {da.dims}."
                )
            if spec.kind == "surf" and level_dim in dims:
                raise ValueError(
                    f"Configured surface variable {spec.dataset_name!r} unexpectedly has "
                    f"pressure-level dimension {level_dim!r}; use kind='atmos'."
                )
            expected = required | allowed_extra
            if group_name == "static":
                expected.add(time_dim)
            unsupported = [
                dim for dim in da.dims
                if dim not in expected and da.sizes[dim] > 1
            ]
            if unsupported:
                raise ValueError(
                    f"Variable {spec.dataset_name!r} has unsupported non-singleton "
                    f"dimension(s) {unsupported}. Add data.extra_dim_indexers entries."
                )

    atmos_specs = [
        spec
        for spec in (
            *resolved_specs.predictors,
            *resolved_specs.targets,
        )
        if spec.kind == "atmos"
    ]
    if atmos_specs:
        if level_dim not in ds.coords:
            raise ValueError(
                f"Atmospheric variables require coordinate {level_dim!r}."
            )
        observed = np.asarray(ds[level_dim].values, dtype=np.float64)
        configured = [
            float(value) for value in config.get("data", {}).get("atmos_levels", ())
        ]
        missing_levels = [
            level
            for level in configured
            if not np.any(np.isclose(observed, level, rtol=0.0, atol=1.0e-6))
        ]
        if missing_levels:
            raise ValueError(
                f"Dataset pressure coordinate {level_dim!r} is missing configured "
                f"data.atmos_levels {missing_levels}; observed={observed.tolist()}."
            )

    rollout_step = config.get("rollout", {}).get("rollout_step_hours")
    if rollout_step is not None:
        if int(ds.sizes[time_dim]) < 2:
            raise ValueError(
                f"Dataset needs at least two {time_dim!r} values to validate "
                "rollout.rollout_step_hours."
            )
        _resolve_rollout_step_hours(ds, config)

    patch_size = int(config.get("model", {}).get("patch_size", 0))
    height = int(ds.sizes[lat_dim])
    width = int(ds.sizes[lon_dim])
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Spatial grid {(height, width)} is smaller than model.patch_size={patch_size}."
        )
    from finetune.longitude import longitude_is_periodic

    if longitude_is_periodic(ds[lon_dim].values) and width % patch_size:
        raise ValueError(
            f"Global longitude width {width} is not divisible by model.patch_size={patch_size}; "
            "periodic longitude cannot be cropped safely."
        )



    from finetune.refinement.config import resolve_refinement_config

    refinement = resolve_refinement_config(config)
    if refinement.is_active and refinement.uses_transformer:
        patch_h, patch_w = refinement.transformer.patch_size
        tokens_h = math.ceil(height / patch_h)
        tokens_w = math.ceil(width / patch_w)
        max_h = refinement.transformer.max_tokens_lat
        max_w = refinement.transformer.max_tokens_lon
        if tokens_h > max_h or tokens_w > max_w:
            raise ValueError(
                "Dataset/refinement Transformer token geometry is incompatible: "
                f"spatial grid={(height, width)}, "
                f"refinement.transformer.patch_size={(patch_h, patch_w)} gives "
                f"token grid={(tokens_h, tokens_w)}, but configured maxima are "
                f"max_tokens_lat={max_h}, max_tokens_lon={max_w}."
            )


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


def _resolve_rollout_step_hours(
    ds: xr.Dataset,
    config: dict[str, Any],
) -> float:
    # Resolve one model step in physical hours and reject index/time mismatch.
    time_dim = _dim_names(config)[0]
    if time_dim not in ds.coords:
        raise ValueError(
            f"Dataset is missing time coordinate {time_dim!r}; cumulative forecast "
            "lead hours cannot be determined."
        )
    time_values = np.asarray(ds[time_dim].values)
    if time_values.size < 2:
        raise ValueError("At least two dataset times are required to infer forecast cadence.")
    try:
        datetime_values = np.asarray(time_values, dtype="datetime64[us]")
        if np.any(np.isnat(datetime_values)):
            raise ValueError("time coordinate contains NaT")
        diffs = np.asarray(
            np.diff(datetime_values) / np.timedelta64(1, "h"),
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        # Fallback for datetime-like coordinate types that NumPy cannot cast.
        diffs = np.asarray(
            [
                (
                    _to_python_datetime(time_values[i + 1])
                    - _to_python_datetime(time_values[i])
                ).total_seconds()
                / 3600.0
                for i in range(time_values.size - 1)
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
        raise ValueError(
            f"Time coordinate {time_dim!r} must be strictly increasing with finite gaps."
        )
    cadence = float(np.median(diffs))
    configured = config.get("rollout", {}).get("rollout_step_hours")
    step_hours = cadence if configured is None else float(configured)
    if not np.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("rollout.rollout_step_hours must be finite and positive.")
    if not np.allclose(diffs, step_hours, rtol=0.0, atol=1.0e-6):
        unique = sorted({round(float(value), 6) for value in diffs})
        raise ValueError(
            "Dataset time cadence does not match rollout.rollout_step_hours: "
            f"configured={step_hours:g} h, observed gaps={unique} h. Lead indices "
            "cannot be converted safely; regularize times explicitly before training."
        )
    return step_hours


def _forecast_lead_hours_for_samples(
    ds: xr.Dataset,
    samples: Sequence[dict[str, Any]],
    lead: int,
    config: dict[str, Any],
    *,
    step_hours: float | None = None,
) -> torch.Tensor:
    # Return exact cumulative hours from each initialization to this lead.
    if int(lead) <= 0:
        raise ValueError(f"Forecast lead index must be positive, got {lead}.")
    time_dim = _dim_names(config)[0]
    time_values = np.asarray(ds[time_dim].values)
    if step_hours is None:
        step_hours = _resolve_rollout_step_hours(ds, config)
    expected = float(lead) * float(step_hours)
    result: list[float] = []
    for sample_idx, sample in enumerate(samples):
        anchor = int(sample["anchor_index"])
        target = anchor + int(lead)
        mapped_targets = sample.get("target_indices", {})
        if int(lead) in mapped_targets and int(mapped_targets[int(lead)]) != target:
            raise ValueError(
                f"Sample {sample_idx} maps lead {lead} to time index "
                f"{mapped_targets[int(lead)]}, expected {target}."
            )
        if anchor < 0 or target >= time_values.size:
            raise ValueError(
                f"Sample {sample_idx} lead {lead} requires time index {target}, but "
                f"dataset {time_dim!r} has length {time_values.size}."
            )
        hours = (
            _to_python_datetime(time_values[target])
            - _to_python_datetime(time_values[anchor])
        ).total_seconds() / 3600.0
        if not np.isclose(hours, expected, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"Sample {sample_idx} lead {lead} spans {hours:g} h, expected "
                f"{expected:g} h from rollout.rollout_step_hours."
            )
        result.append(float(hours))
    return torch.tensor(result, dtype=torch.float32)


def _canonicalize_longitude_coordinate(ds: xr.Dataset, lon_dim: str) -> xr.Dataset:
    """Wrap, sort, and de-duplicate a 1-D longitude coordinate and its data.

    All fine-tuning stages use ``[0, 360)`` internally because
    :class:`aurora.batch.Metadata` requires that convention. The source may use
    either common Earth convention; reindexing the dataset here keeps every
    predictor/target/static field aligned with the canonical coordinate.
    """
    from finetune.longitude import canonical_longitudes

    coord = ds[lon_dim]
    if coord.ndim != 1 or coord.dims != (lon_dim,):
        raise ValueError(
            f"Longitude `{lon_dim}` must be a 1-D coordinate on its own dimension; "
            f"got dims {coord.dims}."
        )
    canonical, source_indices = canonical_longitudes(coord.values)
    attrs = dict(coord.attrs)
    attrs.update(
        {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
    )
    ds = ds.isel({lon_dim: source_indices})
    return ds.assign_coords(
        {lon_dim: xr.DataArray(canonical, dims=(lon_dim,), attrs=attrs)}
    )


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
    ds = _canonicalize_longitude_coordinate(ds, lon_dim)

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

    validate_config(config, source=config_path)

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
        "static_data_path",
        "optional_resume_checkpoint",
    ]
    for key in path_keys:
        value = paths_cfg.get(key)
        if not value:
            continue
        paths_cfg[key] = _resolve_path(str(value), project_root)

    data_dir = Path(paths_cfg.get("data_dir", project_root / "data"))
    data_case_name = str(paths_cfg.get("data_case_name", case_name)).strip()
    if not data_case_name:
        raise ValueError(
            "paths.data_case_name must be a non-empty prepared-data folder name when set."
        )
    paths_cfg["data_case_name"] = data_case_name
    case_data_dir = data_dir if data_dir.name == data_case_name else data_dir / data_case_name
    paths_cfg["data_dir"] = str(data_dir.resolve())
    paths_cfg["case_data_dir"] = str(case_data_dir.resolve())
    paths_cfg["train_data_path"] = str((case_data_dir / "train.nc").resolve())
    paths_cfg["test_data_path"] = str((case_data_dir / "test.nc").resolve())

    # Never use the held-out test split for checkpoint selection. Older
    # versions silently assigned test.nc to both roles, which made a finite
    # validation loss look like held-out evidence. A real val.nc is used when
    # present; train-tail validation deliberately points at train.nc because
    # the trainer constructs a grouped, purged tail split in memory. Keeping
    # the (possibly absent) val.nc path for ``validation_source=configured``
    # makes an unsafe/missing split fail when training attempts to open it.
    training_cfg = config.setdefault("training", {})
    validation_source = str(
        training_cfg.get("validation_source", "configured")
    ).strip().lower()
    skip_validation = bool(training_cfg.get("skip_validation", False))
    prepared_val_path = (case_data_dir / "val.nc").resolve()
    if prepared_val_path.exists():
        paths_cfg["val_data_path"] = str(prepared_val_path)
    elif validation_source == "train_tail" or skip_validation:
        paths_cfg["val_data_path"] = paths_cfg["train_data_path"]
    else:
        paths_cfg["val_data_path"] = str(prepared_val_path)
        logger.warning(
            "Configured validation requires %s, which does not exist. The held-out "
            "test split will not be substituted. Prepare val.nc or set "
            "training.validation_source=train_tail.",
            prepared_val_path,
        )

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

    static_path = paths_cfg.get("static_data_path")
    if static_path and not Path(static_path).exists():
        raise FileNotFoundError(
            f"Configured paths.static_data_path does not exist: {static_path}."
        )
    pretrained_path = paths_cfg.get("pretrained_checkpoint")
    if pretrained_path and not Path(pretrained_path).exists():
        raise FileNotFoundError(
            f"Configured paths.pretrained_checkpoint does not exist: {pretrained_path}."
        )

    return config


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_checkpoint_rng_state(
    checkpoint: dict[str, Any],
    *,
    device: torch.device | str | int | None = None,
) -> bool:
    """Restore a serialized PyTorch RNG stream for exact training resume.

    Returns ``False`` for checkpoints written before RNG metadata was added so
    callers can fall back to their configured seed. CUDA checkpoints record the
    active rank-zero device explicitly; that stream is restored onto each
    worker's active CUDA device rather than assigning rank-zero device ordinals
    to the local process.
    """
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        return False
    cpu_state = rng_state.get("cpu")
    if not torch.is_tensor(cpu_state):
        return False

    target_cuda: torch.device | None = None
    if device is None:
        if torch.cuda.is_available():
            target_cuda = torch.device("cuda", torch.cuda.current_device())
    else:
        if isinstance(device, bool):
            raise TypeError(
                "device must be a torch device, string, CUDA ordinal, or None."
            )
        target_device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if target_device.type == "cuda" and torch.cuda.is_available():
            target_cuda = target_device

    cuda_state = None
    if target_cuda is not None:
        cuda_state = rng_state.get("cuda_current")
        if not torch.is_tensor(cuda_state):
            legacy_states = rng_state.get("cuda")
            if isinstance(legacy_states, (list, tuple)) and legacy_states:
                cuda_state = legacy_states[0]
        if not torch.is_tensor(cuda_state):
            return False

    torch.set_rng_state(cpu_state.detach().to(device="cpu", dtype=torch.uint8))
    if target_cuda is not None:
        assert torch.is_tensor(cuda_state)
        torch.cuda.set_rng_state(
            cuda_state.detach().to(device="cpu", dtype=torch.uint8),
            device=target_cuda,
        )
    return True


def derive_refinement_seed(
    base_seed: int,
    initialization_time: Any,
    member_index: int = 0,
) -> int:
    """Derive a stable RNG seed for one forecast initialization and member.

    Reusing ``base_seed + member_index`` for every initialization replays the
    same spatial diffusion latent in every forecast. Fixed grid artifacts then
    survive aggregation over hundreds of cases. This SHA-256 based mapping is
    stable across Python processes (unlike :func:`hash`) while retaining exact
    reproducibility for a given ``(base seed, initialization, member)`` tuple.
    """
    if int(member_index) < 0:
        raise ValueError(
            f"member_index must be non-negative, got {member_index!r}."
        )
    initialization_ns = np.datetime64(initialization_time, "ns").astype("int64")
    payload = f"{int(base_seed)}|{int(initialization_ns)}|{int(member_index)}".encode()
    # torch.Generator.manual_seed accepts a signed 64-bit-compatible integer.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


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
        if required_dim not in ds.coords:
            raise ValueError(
                f"Dataset {path} is missing required coordinate `{required_dim}`. "
                "Implicit integer dimensions cannot be interpreted as Earth coordinates."
            )

    ds = _ensure_monotonic_lat_lon(ds, lat_dim=lat_dim, lon_dim=lon_dim)
    ds = _subset_domain(ds, config, lat_dim=lat_dim, lon_dim=lon_dim)
    ds = _ensure_monotonic_lat_lon(ds, lat_dim=lat_dim, lon_dim=lon_dim)

    return ds


def validate_longitude_consistency(
    datasets: Sequence[xr.Dataset],
    config: dict[str, Any],
) -> bool:
    """Validate one canonical longitude grid across train/validation/inference.

    Returns the single resolved periodicity value that must be used to build
    refinement heads and write their outputs.
    """
    if not datasets:
        raise ValueError("At least one dataset is required for longitude validation.")
    from finetune.longitude import (
        longitude_grid_signature,
        longitude_is_periodic,
        resolve_lon_periodic,
    )

    _, _, lon_dim, _ = _dim_names(config)
    grids = [np.asarray(ds[lon_dim].values, dtype=np.float64) for ds in datasets]
    reference = grids[0]
    reference_signature = longitude_grid_signature(reference)
    for index, grid in enumerate(grids[1:], start=1):
        if grid.shape != reference.shape or longitude_grid_signature(
            grid
        ) != reference_signature:
            raise ValueError(
                "Training, validation, and inference datasets must use the same canonical "
                f"longitude grid; dataset 0 has shape {reference.shape}, dataset {index} "
                f"has shape {grid.shape}."
            )

    detected = longitude_is_periodic(reference)
    resolved = resolve_lon_periodic(config, reference)
    if resolved and not detected:
        raise ValueError(
            "Periodic longitude was requested, but the canonical coordinate does not cover "
            "a complete, uniformly spaced 360-degree grid. Refusing to join unrelated edges."
        )
    domain = str(config.get("data", {}).get("domain_type", "global")).strip().lower()
    requested = str(config.get("model", {}).get("lon_periodic", "auto")).strip().lower()
    if domain == "global" and requested == "auto" and not detected:
        raise ValueError(
            "data.domain_type is global, but the longitude grid is incomplete. Do not crop "
            "longitude from a global training grid."
        )
    model_cfg = config.setdefault("model", {})
    model_cfg["lon_periodic_resolved"] = bool(resolved)
    model_cfg["longitude_grid_signature"] = reference_signature
    return bool(resolved)


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
            units = str(item.get("units") or ds[dataset_name].attrs.get("units") or "")
        else:
            raise TypeError(f"Unsupported variable spec type: {type(item)}")

        kind = kind.lower()
        if kind not in {"surf", "atmos", "static"}:
            raise ValueError(f"Variable `{dataset_name}` has unsupported kind `{kind}`.")

        if not isinstance(item, dict):
            loss_levels = None
            units = str(ds[dataset_name].attrs.get("units") or "")

        specs.append(
            VariableSpec(
                dataset_name=dataset_name,
                aurora_name=aurora_name,
                kind=kind,
                loss_levels=loss_levels,
                units=units,
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

    resolved = ResolvedVariableSpecs(
        predictors=predictors,
        targets=targets,
        static=static,
    )
    validate_dataset_contract(ds, config, resolved)

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

    return resolved


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

    observed = np.asarray(da[level_dim].values, dtype=np.float64)
    missing = [
        float(level)
        for level in levels
        if not np.any(np.isclose(observed, float(level), rtol=0.0, atol=1.0e-6))
    ]
    if missing:
        raise ValueError(
            f"Atmospheric variable {da.name!r} is missing configured pressure "
            f"level(s) {missing}; observed={observed.tolist()}."
        )

    # Keep requested level order after the explicit compatibility check.
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
    lon_periodic: bool = False,
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

    if lon_periodic and target_w != w:
        raise ValueError(
            f"Global longitude width {w} is not divisible by patch_size={patch_size}. "
            "Cropping longitude would remove part of the periodic Earth grid; regrid the "
            "source to a compatible global width instead."
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
    from finetune.longitude import longitude_is_periodic

    # Preserve the actual global grid even when circular padding is explicitly
    # disabled for an ablation; model semantics must not permit data loss.
    lon_periodic = longitude_is_periodic(ds[lon_dim].values)

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

        # Keep coordinate precision in metadata / NetCDF output. Aurora's
        # encoder performs its own float32 cast for positional calculations.
        lat = torch.from_numpy(
            np.asarray(ds[lat_dim].values, dtype=np.float64).copy()
        )
        lon = torch.from_numpy(
            np.asarray(ds[lon_dim].values, dtype=np.float64).copy()
        )

        surf_single, static_single, atmos_single, lat, lon, _ = _align_spatial_dims_for_patch(
            surf_vars=surf_single,
            static_vars=static_single,
            atmos_vars=atmos_single,
            lat=lat,
            lon=lon,
            patch_size=patch_size,
            strategy=alignment_strategy,
            lon_periodic=lon_periodic,
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
    _, _, lon_dim, _ = _dim_names(config)
    from finetune.longitude import longitude_is_periodic

    lon_periodic = longitude_is_periodic(ds[lon_dim].values)

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
                        lon_periodic=lon_periodic,
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
                        lon_periodic=lon_periodic,
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
        da = _select_levels_if_needed(da, config=config, level_dim=level_dim)
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
    combined_mask = _combined_loss_mask(
        pred_tensor,
        target_tensor,
        spatial_mask=spatial_mask,
        missing_mask=missing_mask,
    )

    if not torch.any(combined_mask):
        return torch.zeros((), device=loss_tensor.device, dtype=loss_tensor.dtype)

    return loss_tensor[combined_mask].mean()


def _combined_loss_mask(
    pred_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    *,
    spatial_mask: torch.Tensor | None,
    missing_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return the common finite/configured mask used by every loss path."""
    combined_mask = torch.isfinite(pred_tensor) & torch.isfinite(target_tensor)

    if spatial_mask is not None:
        while spatial_mask.dim() < combined_mask.dim():
            spatial_mask = spatial_mask.unsqueeze(0)
        combined_mask = combined_mask & spatial_mask

    if missing_mask is not None:
        while missing_mask.dim() < combined_mask.dim():
            missing_mask = missing_mask.unsqueeze(0)
        combined_mask = combined_mask & missing_mask

    return combined_mask


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


def _make_refinement_generator(
    device: torch.device | str,
    seed: int | None,
) -> torch.Generator | None:
    """Create one persistent refinement RNG for an entire causal rollout."""
    if seed is None:
        return None
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def _validation_refinement_generator(
    model: torch.nn.Module,
    device: torch.device | str,
    *,
    stream: int = 0,
) -> torch.Generator | None:
    """Create one reproducible RNG stream for a unified validation loop.

    The generator is intentionally owned by the validation loop rather than by
    each batch loss call. This keeps repeated validation runs reproducible while
    ensuring consecutive batches do not all reuse the first stochastic draw.
    Distributed ranks use distinct deterministic streams.
    """
    from finetune.refinement.two_phase import AuroraTwoPhaseRefiner

    inner = model.module if hasattr(model, "module") else model
    if not isinstance(inner, AuroraTwoPhaseRefiner):
        return None
    if not inner.refinement_config.is_active:
        return None
    seed = inner.refinement_config.seed
    if seed is None:
        return None
    return _make_refinement_generator(device, int(seed) + int(stream))


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


def _spatial_pattern_correlation_sum_count(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[float, float]:
    """Sum valid per-sample spatial correlations without a physical-scale floor.

    Correlation is computed independently for each forecast sample and only
    finite, jointly valid cells participate. Constant/one-cell fields have no
    defined spatial pattern and therefore do not increment the count.
    """
    if prediction.shape != truth.shape or prediction.shape != valid_mask.shape:
        raise ValueError(
            "prediction, truth, and valid_mask must have identical shapes for "
            "spatial pattern correlation."
        )
    if prediction.ndim < 2:
        raise ValueError(
            "Spatial pattern correlation expects a leading sample dimension "
            "and at least one spatial dimension."
        )

    correlation_sum = 0.0
    correlation_count = 0.0
    for sample_index in range(prediction.shape[0]):
        selected = (
            valid_mask[sample_index].to(dtype=torch.bool)
            & torch.isfinite(prediction[sample_index])
            & torch.isfinite(truth[sample_index])
        )
        pred_values = prediction[sample_index][selected].double()
        truth_values = truth[sample_index][selected].double()
        if pred_values.numel() < 2:
            continue
        pred_anomaly = pred_values - pred_values.mean()
        truth_anomaly = truth_values - truth_values.mean()
        pred_variance_sum = pred_anomaly.square().sum()
        truth_variance_sum = truth_anomaly.square().sum()
        denominator = (pred_variance_sum * truth_variance_sum).sqrt()
        if not bool(torch.isfinite(denominator) & (denominator > 0)):
            continue
        correlation = (pred_anomaly * truth_anomaly).sum() / denominator
        if not bool(torch.isfinite(correlation)):
            continue
        correlation_sum += float(correlation.clamp(-1.0, 1.0).detach().cpu())
        correlation_count += 1.0
    return correlation_sum, correlation_count


def compute_supervised_loss(
    model: torch.nn.Module,
    ds: xr.Dataset,
    samples: dict[str, Any] | Sequence[dict[str, Any]],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    device: str | torch.device,
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    refinement_generator: torch.Generator | None = None,
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

    # Mamba temporal loss weight. Active when either refinement wrapper has the
    # optional temporal module enabled (see end of this function). 0 disables
    # the loss for legacy compatibility.
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
    base_for_fm = model.module if hasattr(model, "module") else model
    from finetune.flow_refine import AuroraFlowRefine as _AFR
    from finetune.refinement.integration import (
        LeadStepBuffer,
        pack_refinement_conditioning,
        refine_batch_prediction,
    )
    from finetune.refinement.two_phase import AuroraTwoPhaseRefiner as _ATP

    is_flow_refine = isinstance(base_for_fm, _AFR)
    unified_refiner = base_for_fm if isinstance(base_for_fm, _ATP) else None
    unified_active = (
        unified_refiner is not None
        and unified_refiner.refinement_config.is_active
        and unified_refiner.training
    )
    unified_eval_active = (
        unified_refiner is not None
        and unified_refiner.refinement_config.is_active
        and not unified_refiner.training
    )
    lead_buffer = (
        LeadStepBuffer(unified_refiner.packing) if unified_active else None
    )
    unified_input_batches: dict[int, Batch] = {}
    lead_conditioned_flow = (
        is_flow_refine and getattr(base_for_fm, "lead_time_cond", False)
    )
    needs_lead_hours = (
        lead_conditioned_flow or unified_active or unified_eval_active or not model.training
    )
    lead_hours_by_step: dict[int, torch.Tensor] = {}
    if needs_lead_hours:
        forecast_step_hours = _resolve_rollout_step_hours(ds, config)
        lead_hours_by_step = {
            lead: _forecast_lead_hours_for_samples(
                ds,
                sample_list,
                lead,
                config,
                step_hours=forecast_step_hours,
            ).to(device)
            for lead in range(1, max_lead + 1)
        }
    flow_feedback_enabled = (
        is_flow_refine
        and base_for_fm.training
        and int(base_for_fm.flow_refine_contract_version) >= 2
        and training_cfg.get("flow_refine_autoregressive_feedback", False) is True
    )
    # Per-variable normalised sequences (ordered by lead) for the Mamba
    # temporal loss. Populated inside the lead loop only when a flow-refine
    # wrapper with the temporal module enabled is in training mode.
    temporal_seq: dict[str, dict[str, Any]] = {}
    unified_eval_temporal_history: list[torch.Tensor] | None = (
        []
        if unified_eval_active and getattr(unified_refiner, "has_temporal", False)
        else None
    )
    unified_eval_temporal_lead_history: list[torch.Tensor] | None = (
        [] if unified_eval_temporal_history is not None else None
    )
    unified_eval_generator = None
    if unified_eval_active:
        # Validation loops pass one persistent generator so stochastic draws
        # advance across batches instead of restarting from the same seed for
        # every batch. Direct callers retain the historical deterministic
        # per-call behavior when they do not supply a generator.
        unified_eval_generator = (
            refinement_generator
            if refinement_generator is not None
            else _make_refinement_generator(
                device,
                unified_refiner.refinement_config.seed,
            )
        )
    validation_refinement_ensemble_size = int(
        training_cfg.get("validation_refinement_ensemble_size", 1)
    )
    with _autocast_context(device=device, mixed_precision=mixed_precision):
        for lead in range(1, max_lead + 1):
            lead_hours = lead_hours_by_step.get(lead)
            if unified_active:
                # Preserve the exact pre-forecast state for optional input-state
                # and static conditioning. Later autoregressive updates must not
                # overwrite which state conditioned this forecast valid time.
                unified_input_batches[lead] = current_batch
            if unified_eval_active:
                assert unified_refiner is not None
                deterministic_pred = unified_refiner.aurora(current_batch)
                pred = refine_batch_prediction(
                    unified_refiner,
                    deterministic_pred,
                    aurora_input_batch=current_batch,
                    forecast_lead_time_hours=lead_hours,
                    ensemble_size=validation_refinement_ensemble_size,
                    generator=unified_eval_generator,
                    temporal_history=unified_eval_temporal_history,
                    temporal_lead_history=unified_eval_temporal_lead_history,
                )
                feedback_pred = (
                    pred
                    if unified_refiner.refinement_config.feedback_to_rollout
                    else deterministic_pred
                )
            else:
                pred = (
                    model(
                        current_batch,
                        forecast_lead_time_hours=lead_hours,
                    )
                    if (is_flow_refine or unified_refiner is not None)
                    else model(current_batch)
                )
                feedback_pred = pred
                if flow_feedback_enabled:
                    # Match inference-state provenance without backpropagating
                    # through the full rollout. The residual loss below still uses
                    # `pred` as its unrefined baseline, while the next Aurora step
                    # consumes the deterministic corrected target fields.
                    with torch.no_grad():
                        feedback_pred = base_for_fm.refine_prediction(
                            pred,
                            deterministic=True,
                            forecast_lead_time_hours=lead_hours,
                        )
            preds_by_lead[lead] = pred

            # Build next autoregressive input:
            # - Target variables: use model prediction (autoregressive)
            # - Exogenous predictors: use ground truth from dataset
            next_time_indices = [
                int(sample["anchor_index"]) + lead for sample in sample_list
            ]

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
                            and min(next_time_indices) >= 0
                            and max(next_time_indices) < ds.sizes[_dim_names(config)[0]]
                        ):
                            gt_frame = _dataset_frames_for_predictor(
                                ds,
                                spec,
                                time_indices=next_time_indices,
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
                feedback_pred,
                surf_vars=_advance_with_gt(
                    current_batch.surf_vars, feedback_pred.surf_vars, "surf",
                ),
                atmos_vars=_advance_with_gt(
                    current_batch.atmos_vars, feedback_pred.atmos_vars, "atmos",
                ),
            )

    total_loss = torch.zeros((), device=device)
    total_weight = 0.0

    physical_error_sums: dict[str, dict[str, float]] = {}
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
            level_idx: list[int] | None = None
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
            pred_physical_tensor = pred_tensor
            target_physical_tensor = target_tensor
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
                    if level_idx is not None and maybe_mask.shape[0] == len(full_levels):
                        maybe_mask = maybe_mask.index_select(
                            0, torch.tensor(level_idx, dtype=torch.long),
                        )
                    missing_masks.append(maybe_mask)
            missing_mask = torch.stack(missing_masks, dim=0).to(device) if missing_masks else None
            valid_mask = _combined_loss_mask(
                pred_tensor,
                target_tensor,
                spatial_mask=spatial_mask,
                missing_mask=missing_mask,
            )

            if not model.training:
                metric_lead_hours = lead_hours_by_step.get(lead)
                if metric_lead_hours is None or metric_lead_hours.numel() == 0:
                    raise RuntimeError("Physical validation metrics require forecast lead hours.")
                if not torch.allclose(
                    metric_lead_hours, metric_lead_hours[:1].expand_as(metric_lead_hours)
                ):
                    raise ValueError("A validation batch mixes physical hours for one lead index.")
                lead_suffix = f"@lead{float(metric_lead_hours[0]):g}h"
                if target_spec.kind == "surf":
                    channel_views = [
                        (
                            f"{target_spec.dataset_name}@surface{lead_suffix}",
                            pred_physical_tensor,
                            target_physical_tensor,
                            valid_mask,
                        )
                    ]
                else:
                    levels_for_metrics = list(var_levels or ())
                    if len(levels_for_metrics) != pred_physical_tensor.shape[1]:
                        levels_for_metrics = list(range(pred_physical_tensor.shape[1]))
                    channel_views = [
                        (
                            f"{target_spec.dataset_name}@{float(level):g}hPa{lead_suffix}",
                            pred_physical_tensor[:, index],
                            target_physical_tensor[:, index],
                            valid_mask[:, index],
                        )
                        for index, level in enumerate(levels_for_metrics)
                    ]
                for channel_name, physical_pred, physical_target, channel_mask in channel_views:
                    selected = (
                        channel_mask
                        & torch.isfinite(physical_pred)
                        & torch.isfinite(physical_target)
                    )
                    if not bool(selected.any().item()):
                        continue
                    error = (physical_pred[selected] - physical_target[selected]).double()
                    correlation_sum, correlation_count = (
                        _spatial_pattern_correlation_sum_count(
                            physical_pred,
                            physical_target,
                            selected,
                        )
                    )
                    slot = physical_error_sums.setdefault(
                        channel_name,
                        {
                            "count": 0.0,
                            "error_sum": 0.0,
                            "abs_error_sum": 0.0,
                            "sq_error_sum": 0.0,
                            "spatial_correlation_sum": 0.0,
                            "spatial_correlation_count": 0.0,
                        },
                    )
                    slot["count"] += float(error.numel())
                    slot["error_sum"] += float(error.sum().detach().cpu())
                    slot["abs_error_sum"] += float(error.abs().sum().detach().cpu())
                    slot["sq_error_sum"] += float(error.square().sum().detach().cpu())
                    slot["spatial_correlation_sum"] += correlation_sum
                    slot["spatial_correlation_count"] += correlation_count

            # If the model is a flow-matching refine wrapper AND we're in
            # training mode, replace the deterministic MSE with the
            # rectified-flow clean-endpoint residual MSE. With residual
            # z-scoring this gives a unit-scale loss surface and lets a small
            # UNet learn the conditional residual distribution p(r | ŷ).
            base_for_fm = model.module if hasattr(model, "module") else model
            use_flow_loss = (
                isinstance(base_for_fm, _AFR)
                and base_for_fm.training
            )
            if use_flow_loss:
                flow_doy = None
                if getattr(base_for_fm, "doy_cond", False):
                    flow_doy = _AFR._doy_from_metadata(pred)
                    if flow_doy is not None:
                        flow_doy = flow_doy.to(device)
                # Longitude vector for the cyclic ``lon_encoding`` input. Built
                # only when the feature is on so training matches inference.
                flow_lon = None
                if getattr(base_for_fm, "lon_encoding", False):
                    flow_lon = _AFR._lon_from_metadata(pred)
                    if flow_lon is not None:
                        flow_lon = flow_lon.to(device)
                masked = base_for_fm.flow_loss(
                    pred_norm=pred_tensor,
                    target_norm=target_tensor,
                    var_name=aurora_name,
                    kind=target_spec.kind,
                    doy=flow_doy,
                    lon=flow_lon,
                    valid_mask=valid_mask,
                    lead_time_hours=lead_hours_by_step.get(lead),
                )
                # Cache normalised tensors for the coherence term (computed
                # once per lead after this inner loop).
                if coherence_weight > 0.0:
                    coh_cache[aurora_name] = {
                        "pred": pred_tensor,
                        "target": target_tensor,
                        "kind": target_spec.kind,
                        "levels": var_levels,
                        "lon": flow_lon,
                    }
                # Collect the per-lead normalised (base) prediction and target
                # so the Mamba temporal module can be trained on the ordered
                # rollout sequence after the lead loop.
                if temporal_weight > 0.0 and getattr(base_for_fm, "has_temporal", False):
                    slot = temporal_seq.setdefault(
                        aurora_name,
                        {
                            "kind": target_spec.kind,
                            "preds": [],
                            "targets": [],
                            "lead_hours": [],
                            "lon": flow_lon,
                        },
                    )
                    slot["preds"].append(pred_tensor)
                    slot["targets"].append(target_tensor)
                    slot["lead_hours"].append(lead_hours_by_step.get(lead))
            else:
                elementwise = _loss_tensor(pred_tensor, target_tensor, loss_name=loss_name)
                masked = _apply_loss_mask(
                    loss_tensor=elementwise,
                    pred_tensor=pred_tensor,
                    target_tensor=target_tensor,
                    spatial_mask=spatial_mask,
                    missing_mask=missing_mask,
                )

            # Collect the normalized (rollout, target) pair of THIS rollout step
            # for the unified stochastic refiner. The deterministic supervised
            # loss above is untouched; Phase 2 adds its own term after the loop.
            # ``pred_tensor``/``target_tensor`` refer to the same forecast valid
            # time, so no rollout-step shift can be introduced here.
            if lead_buffer is not None:
                lead_buffer.add(
                    lead,
                    aurora_name,
                    rollout_normalized=pred_tensor.detach(),
                    target_normalized=target_tensor.detach(),
                    valid_mask=valid_mask,
                    lead_hours=lead_hours_by_step.get(lead),
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
                    lon=prof.get("lon"),
                    lead_time_hours=lead_hours_by_step.get(lead),
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
                flow_lon = slot.get("lon")
                sequence_leads = slot["lead_hours"]
                if len(preds) < 2:
                    # A temporal model needs at least two ordered steps.
                    continue
                with torch.no_grad():
                    flow_frames = [
                        base_for_fm.refine_norm_deterministic(
                            p,
                            aurora_name,
                            kind,
                            lon=flow_lon,
                            lead_time_hours=lead_hours,
                        ).detach()
                        for p, lead_hours in zip(preds, sequence_leads, strict=True)
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

    # ---- Unified stochastic residual refinement (Phase 2) ----
    # The deterministic Aurora rollout above is frozen input; the generative
    # term trains the spatial refiner on folded steps. When enabled, Mamba then
    # unfolds those steps into [batch, lead, channel, lat, lon] and adds a
    # separate cross-lead loss without leaking target-conditioned diffusion
    # states into its input.
    refinement_metrics: dict[str, float] = {}
    if lead_buffer is not None and lead_buffer.is_complete():
        assert unified_refiner is not None
        rollout_n, target_n, mask_n, lead_hours_n, lead_index_n = lead_buffer.pack(device)
        for position, lead in enumerate(lead_buffer.leads):
            selected_rollout = rollout_n[lead_index_n == position]
            packed_inputs = pack_refinement_conditioning(
                unified_refiner,
                selected_rollout,
                aurora_input_batch=unified_input_batches[lead],
            )
            lead_buffer.set_conditioning(
                lead,
                input_state_normalized=packed_inputs.input_state_normalized,
                static_fields=packed_inputs.static_fields,
            )
        packed_inputs = lead_buffer.pack_conditioning(device)
        refinement_weight = float(training_cfg.get("refinement_loss_weight", 1.0))
        step_out = unified_refiner.training_step(
            rollout_n,
            target_n,
            input_state_normalized=packed_inputs.input_state_normalized,
            static_fields=packed_inputs.static_fields,
            valid_mask=mask_n,
            forecast_lead_time=lead_hours_n,
            lead_index=lead_index_n,
        )
        refinement_metrics = {
            name: float(value.detach()) for name, value in step_out.losses.items()
        }
        total_loss = total_loss + refinement_weight * step_out.losses["total_loss"]
        if temporal_weight > 0.0 and unified_refiner.has_temporal:
            temporal_loss, unified_temporal_metrics = (
                unified_refiner.temporal_training_loss(
                    rollout_n,
                    target_n,
                    valid_mask=mask_n,
                    forecast_lead_time=lead_hours_n,
                    lead_index=lead_index_n,
                    conditioning=step_out.conditioning,
                )
            )
            total_loss = total_loss + temporal_weight * temporal_loss
            temporal_metrics.update(
                {
                    name: float(value.detach())
                    for name, value in unified_temporal_metrics.items()
                }
            )

    metrics = {
        "batch_size": len(sample_list),
        "lead_times": lead_times,
        "spatial_shape": batch.spatial_shape,
    }
    if physical_error_sums:
        metrics["physical_error_sums"] = physical_error_sums
    if temporal_metrics:
        metrics["temporal"] = temporal_metrics
    if refinement_metrics:
        metrics["refinement"] = refinement_metrics
    return total_loss, metrics


def maybe_wrap_conv_refine(
    model: torch.nn.Module,
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    lon: Any | None = None,
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
    from finetune.longitude import longitude_grid_signature, resolve_lon_periodic

    target_surf = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "surf"
    )
    target_atmos = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "atmos"
    )
    hidden = int(model_cfg.get("conv_refine_hidden", 32))
    # Treat longitude as periodic on global domains so the conv correction has
    # no seam at the 0°/360° dateline (regional domains keep replicate padding).
    lon_periodic = resolve_lon_periodic(config, lon)
    model_cfg["lon_periodic_resolved"] = lon_periodic
    if lon is not None:
        model_cfg["longitude_grid_signature"] = longitude_grid_signature(lon)

    wrapper = AuroraConvRefine(
        base=model,
        target_surf_vars=target_surf,
        target_atmos_vars=target_atmos,
        hidden=hidden,
        lon_periodic=lon_periodic,
    )
    wrapper.longitude_grid_signature = model_cfg.get("longitude_grid_signature")
    return wrapper


def maybe_wrap_flow_refine(
    model: torch.nn.Module,
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    lon: Any | None = None,
) -> torch.nn.Module:
    """Optionally wrap *model* with rectified-flow residual refine heads.

    Enabled via ``model.flow_refine_enabled = true`` in config. Mutually
    exclusive with :func:`maybe_wrap_conv_refine`. Norm stats must be
    attached after construction by the training driver (call
    ``model.set_norm_stats(...)`` once stats have been computed) so the
    wrapper can de-normalise sampled residuals at inference time.
    """
    model_cfg = config.get("model", {})
    from finetune.refinement.config import resolve_refinement_config

    refinement_cfg = resolve_refinement_config(config)
    if not bool(model_cfg.get("flow_refine_enabled", False)):
        if refinement_cfg.backend != "legacy":
            return model
        # ``refinement.type: flow_matching_unet`` (or its ``flow_matching``
        # alias) selects exactly this wrapper. Mirror the resolved settings onto
        # the historical keys so there is a single construction path and both
        # spellings build an identical head.
        model_cfg = config.setdefault("model", {})
        model_cfg["flow_refine_enabled"] = True
        model_cfg.setdefault("flow_refine_hidden", refinement_cfg.unet.hidden_channels)
        model_cfg.setdefault(
            "flow_refine_time_dim", refinement_cfg.unet.time_embedding_dim
        )
        model_cfg.setdefault(
            "flow_refine_sampling_steps", refinement_cfg.flow_matching.integration_steps
        )
        model_cfg.setdefault(
            "flow_refine_residual_zscore", refinement_cfg.flow_matching.residual_zscore
        )
        model_cfg.setdefault(
            "flow_refine_res_std_momentum", refinement_cfg.flow_matching.res_std_momentum
        )
        model_cfg.setdefault(
            "flow_refine_lead_time_cond", refinement_cfg.conditioning.forecast_lead_time
        )

    from finetune.flow_refine import AuroraFlowRefine
    from finetune.longitude import longitude_grid_signature, resolve_lon_periodic

    target_surf = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "surf"
    )
    target_atmos = tuple(
        spec.aurora_name for spec in resolved_specs.targets if spec.kind == "atmos"
    )
    hidden = int(model_cfg.get("flow_refine_hidden", 64))
    time_dim = int(model_cfg.get("flow_refine_time_dim", 128))
    # A single source-endpoint query is the safe deterministic correction.
    # Multi-step stochastic sampling must be requested explicitly.
    sampling_steps = int(model_cfg.get("flow_refine_sampling_steps", 1))
    # Missing contract metadata means a historical checkpoint/configuration.
    # Keep that endpoint and rollout behaviour until the user opts into v2+.
    flow_refine_contract_version = int(
        model_cfg.get("flow_refine_contract_version", 1)
    )
    doy_cond = bool(model_cfg.get("flow_refine_doy_cond", False))
    lead_time_cond = bool(model_cfg.get("flow_refine_lead_time_cond", False))
    lead_scale_value = model_cfg.get("flow_refine_lead_time_scale_hours")
    trained_max_lead_hours: float | None = None
    if lead_time_cond:
        step_value = config.get("rollout", {}).get("rollout_step_hours")
        target_values = config.get("data", {}).get("target_lead_times", ())
        if step_value is None or not target_values:
            raise ValueError(
                "model.flow_refine_lead_time_cond=true requires positive "
                "rollout.rollout_step_hours and data.target_lead_times."
            )
        step_hours = float(step_value)
        target_steps = [int(value) for value in target_values]
        if (
            not np.isfinite(step_hours)
            or step_hours <= 0
            or any(value <= 0 for value in target_steps)
        ):
            raise ValueError(
                "Lead-conditioned refinement requires positive rollout cadence "
                "and target lead indices."
            )
        trained_max_lead_hours = step_hours * max(target_steps)
        if lead_scale_value is None:
            lead_scale_value = trained_max_lead_hours
            model_cfg["flow_refine_lead_time_scale_hours"] = lead_scale_value
    lead_time_scale_hours = float(
        72.0 if lead_scale_value is None else lead_scale_value
    )
    if not np.isfinite(lead_time_scale_hours) or lead_time_scale_hours <= 0:
        raise ValueError("flow_refine_lead_time_scale_hours must be finite and positive.")
    if (
        trained_max_lead_hours is not None
        and not np.isclose(
            lead_time_scale_hours,
            trained_max_lead_hours,
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ValueError(
            "flow_refine_lead_time_scale_hours must equal the largest supervised "
            f"physical lead ({trained_max_lead_hours:g} hours)."
        )
    residual_zscore = bool(model_cfg.get("flow_refine_residual_zscore", False))
    res_std_momentum = float(model_cfg.get("flow_refine_res_std_momentum", 0.99))
    # Longitude periodicity: circular conv padding on global domains (no seam at
    # the dateline). ``flow_refine_lon_encoding`` additionally feeds smooth
    # sin/cos-longitude channels as UNet inputs (opt-in; default off to keep the
    # input-channel count and existing checkpoints unchanged).
    lon_periodic = resolve_lon_periodic(config, lon)
    model_cfg["lon_periodic_resolved"] = lon_periodic
    if lon is not None:
        model_cfg["longitude_grid_signature"] = longitude_grid_signature(lon)
    lon_encoding = bool(model_cfg.get("flow_refine_lon_encoding", False))

    # Mamba temporal module (optional; nested and flat schemas are equivalent).
    from finetune.refinement.two_phase import resolve_temporal_config

    temporal_config = resolve_temporal_config(config)
    temporal_enabled = temporal_config["enabled"]
    temporal_channels = temporal_config["channels"]
    temporal_state = temporal_config["state"]
    temporal_layers = temporal_config["layers"]
    temporal_conv = temporal_config["conv"]
    temporal_expand = temporal_config["expand"]

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
        flow_refine_contract_version=flow_refine_contract_version,
        atmos_loss_levels=atmos_loss_levels if atmos_loss_levels else None,
        doy_cond=doy_cond,
        lead_time_cond=lead_time_cond,
        lead_time_scale_hours=lead_time_scale_hours,
        residual_zscore=residual_zscore,
        res_std_momentum=res_std_momentum,
        lon_periodic=lon_periodic,
        lon_encoding=lon_encoding,
        temporal_enabled=temporal_enabled,
        temporal_channels=temporal_channels,
        temporal_state=temporal_state,
        temporal_layers=temporal_layers,
        temporal_conv=temporal_conv,
        temporal_expand=temporal_expand,
    )
    wrapper.longitude_grid_signature = model_cfg.get("longitude_grid_signature")

    # Structural auxiliary-loss weights (extreme-event, spatial-pattern,
    # distributional, vertical-profile, column/profile coherence). Read from
    # the training.flow_aux_loss block; absent → all zero → pure residual MSE.
    aux_cfg = config.get("training", {}).get("flow_aux_loss", {})
    wrapper.set_aux_loss_config(aux_cfg)
    return wrapper


def maybe_wrap_stochastic_refine(
    model: torch.nn.Module,
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    lon: Any | None = None,
    lat: Any | None = None,
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    nonnegative_variables: Sequence[str] = (),
) -> torch.nn.Module:
    """Optionally wrap *model* with the unified stochastic residual refiner.

    Handles ``model.refinement.type`` values ``flow_matching_transformer``,
    ``diffusion_unet`` and ``diffusion_transformer``. ``none`` and the legacy
    ``flow_matching_unet`` / ``flow_matching`` alias return *model* unchanged so
    :func:`maybe_wrap_flow_refine` keeps owning the existing implementation.

    Aurora is frozen and put in ``eval()`` by default; only the refiner is
    trained. Refinement is postprocessing of each deterministic rollout step and
    never replaces the state used to produce later steps.
    """
    from finetune.longitude import longitude_grid_signature, resolve_lon_periodic
    from finetune.refinement.integration import maybe_build_stochastic_refiner

    lon_periodic = resolve_lon_periodic(config, lon)
    model_cfg = config.setdefault("model", {})
    model_cfg["lon_periodic_resolved"] = lon_periodic
    if lon is not None:
        model_cfg["longitude_grid_signature"] = longitude_grid_signature(lon)

    lat_values = None if lat is None else [float(v) for v in np.asarray(lat).ravel()]
    lon_values = None if lon is None else [float(v) for v in np.asarray(lon).ravel()]

    wrapper = maybe_build_stochastic_refiner(
        model,
        config,
        resolved_specs,
        norm_stats=norm_stats,
        lat=lat_values,
        lon=lon_values,
        lon_periodic=lon_periodic,
        nonnegative_variables=nonnegative_variables,
    )
    if wrapper is None:
        return model
    wrapper.longitude_grid_signature = model_cfg.get("longitude_grid_signature")
    return wrapper


def build_finetune_model(
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    *,
    lon: Any | None = None,
    lat: Any | None = None,
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    load_pretrained: bool = False,
    autocast: bool | None = None,
) -> torch.nn.Module:
    """Build Aurora and its configured refinement through the shared factory."""
    from finetune.model_factory import build_finetune_model as _build

    return _build(
        config,
        resolved_specs,
        lon=lon,
        lat=lat,
        norm_stats=norm_stats,
        load_pretrained=load_pretrained,
        autocast=autocast,
    )


def load_model_from_checkpoint(
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Strictly reconstruct and load a combined training checkpoint."""
    from finetune.model_factory import load_model_from_checkpoint as _load

    return _load(
        config,
        resolved_specs,
        checkpoint_path,
        **kwargs,
    )


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

    from finetune.refinement.two_phase import AuroraTwoPhaseRefiner

    is_conv_refine = isinstance(model, AuroraConvRefine)
    is_flow_refine = (
        AuroraFlowRefine is not None and isinstance(model, AuroraFlowRefine)
    )
    is_stochastic_refine = isinstance(model, AuroraTwoPhaseRefiner)
    base = (
        model.aurora if is_stochastic_refine
        else model.base if (is_conv_refine or is_flow_refine)
        else model
    )

    if bool(config.get("training", {}).get("mamba_temporal_only", False)):
        temporal_module = getattr(model, "temporal", None)
        if temporal_module is None:
            raise ValueError(
                "mamba_temporal_only requires a constructed temporal module."
            )
        for param in model.parameters():
            param.requires_grad = False
        for param in temporal_module.parameters():
            param.requires_grad = True
        if is_stochastic_refine:
            model.aurora_frozen = True
            model.aurora.eval()
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "frozen_parameters": int(total - trainable),
        }

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

    if is_stochastic_refine:
        if model.refinement_config.freeze_aurora:
            for param in model.aurora.parameters():
                param.requires_grad = False
            model.aurora_frozen = True
            model.aurora.eval()
        if model.refiner is not None:
            for param in model.refiner.parameters():
                param.requires_grad = True
        if model.temporal is not None:
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
        raise ValueError(
            "No trainable parameters were found after freeze/unfreeze configuration."
        )

    from finetune.refinement.two_phase import resolve_temporal_config

    temporal_multiplier = resolve_temporal_config(config)[
        "learning_rate_multiplier"
    ]
    optimizer_params: Any = params
    if temporal_multiplier != 1.0:
        inner = model.module if hasattr(model, "module") else model
        temporal_module = getattr(inner, "temporal", None)
        if temporal_module is not None:
            temporal_ids = {
                id(param)
                for param in temporal_module.parameters()
                if param.requires_grad
            }
            temporal_params = [
                param for param in params if id(param) in temporal_ids
            ]
            spatial_params = [
                param for param in params if id(param) not in temporal_ids
            ]
            optimizer_params = []
            if spatial_params:
                optimizer_params.append({"params": spatial_params, "lr": lr})
            if temporal_params:
                optimizer_params.append(
                    {
                        "params": temporal_params,
                        "lr": lr * temporal_multiplier,
                    }
                )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            optimizer_params, lr=lr, weight_decay=weight_decay
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            optimizer_params, lr=lr, weight_decay=weight_decay
        )
    if optimizer_name == "sgd":
        momentum = float(training_cfg.get("sgd_momentum", 0.9))
        return torch.optim.SGD(
            optimizer_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

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
    refinement_generator = _validation_refinement_generator(model, device)

    losses: list[float] = []
    with torch.inference_mode():
        for batch_count, i in enumerate(
            range(0, len(val_samples), batch_size), start=1
        ):
            sample_batch = val_samples[i : i + batch_size]
            loss, _ = compute_supervised_loss(
                model=model,
                ds=ds_val,
                samples=sample_batch,
                config=config,
                resolved_specs=resolved_specs,
                device=device,
                norm_stats=norm_stats,
                refinement_generator=refinement_generator,
            )
            losses.append(float(loss.detach().cpu().item()))
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


def _dataset_frames_for_predictor(
    ds: xr.Dataset,
    spec: VariableSpec,
    time_indices: Sequence[int],
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    # Stack each sample from its own future time. Repeating one sample leaks the
    # first batch item into every other autoregressive trajectory.
    if not time_indices:
        raise ValueError("time_indices must contain at least one dataset index.")
    frames = [
        _dataset_frame_for_predictor(
            ds,
            spec,
            time_index=int(time_index),
            batch_size=1,
            config=config,
            device=device,
        )
        for time_index in time_indices
    ]
    return torch.cat(frames, dim=0)


def run_rollout(
    model: torch.nn.Module,
    ds: xr.Dataset,
    start_sample: dict[str, Any],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    device: str | torch.device,
    *,
    refinement_seed: int | None = None,
    refinement_ensemble_size: int | None = None,
    temporal_control: str = "on",
) -> list[Batch]:
    """Run rollout from a starting sample using a fine-tuned model."""
    temporal_control = str(temporal_control).strip().lower()
    if temporal_control not in {"on", "off", "shuffled"}:
        raise ValueError(
            "temporal_control must be on, off, or shuffled; "
            f"got {temporal_control!r}."
        )
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
    flow_refine_active = _AFR is not None and isinstance(base_for_fm, _AFR)
    flow_refine_contract_version = (
        int(getattr(base_for_fm, "flow_refine_contract_version", 1))
        if flow_refine_active
        else 1
    )
    versioned_flow_postprocessing = (
        flow_refine_active and flow_refine_contract_version >= 2
    )
    flow_refine_feedback = (
        versioned_flow_postprocessing
        and bool(
            config.get("training", {}).get(
                "flow_refine_autoregressive_feedback", False
            )
        )
    )
    flow_refine_stochastic_sampling = (
        versioned_flow_postprocessing
        and bool(rollout_cfg.get("flow_refine_stochastic_sampling", False))
    )
    try:
        from finetune.refinement.two_phase import AuroraTwoPhaseRefiner as _ATP
    except Exception:
        _ATP = None
    unified_refiner = (
        base_for_fm
        if _ATP is not None
        and isinstance(base_for_fm, _ATP)
        and base_for_fm.refinement_config.is_active
        else None
    )
    # Postprocessing semantics by default: the refined field is recorded but the
    # DETERMINISTIC prediction continues the rollout. Feeding the refined field
    # back is an explicit, experimental opt-in.
    refinement_feedback = (
        unified_refiner is not None
        and unified_refiner.refinement_config.feedback_to_rollout
    )
    lead_conditioned_flow = (
        flow_refine_active and getattr(base_for_fm, "lead_time_cond", False)
    )
    temporal_active = (
        flow_refine_active
        and getattr(base_for_fm, "has_temporal", False)
    )
    unified_temporal_active = (
        unified_refiner is not None
        and getattr(unified_refiner, "has_temporal", False)
    )
    trained_leads = sorted(
        int(value)
        for value in config.get("data", {}).get("target_lead_times", ())
    )
    if (temporal_active or unified_temporal_active) and (
        not trained_leads or steps > max(trained_leads)
    ):
        trained_horizon = max(trained_leads) if trained_leads else None
        raise ValueError(
            "rollout.rollout_num_steps exceeds the trained temporal-Mamba "
            "horizon from data.target_lead_times; expected a value <= "
            f"{trained_horizon!r}, got {steps}."
        )
    step_hours = (
        _resolve_rollout_step_hours(ds, config)
        if (lead_conditioned_flow or unified_refiner is not None)
        else float(rollout_cfg.get("rollout_step_hours", 1.0))
    )
    if (
        lead_conditioned_flow
        and steps * step_hours
        > float(getattr(base_for_fm, "lead_time_scale_hours")) + 1.0e-6
    ):
        raise ValueError(
            f"Requested rollout reaches {steps * step_hours:g} h, beyond the "
            f"lead-conditioned checkpoint support of "
            f"{float(getattr(base_for_fm, "lead_time_scale_hours")):g} h. "
            "Retrain with the longer lead range instead of extrapolating silently."
        )
    anchor_idx = int(start_sample["anchor_index"])
    resolved_refinement_seed: int | None = None
    resolved_legacy_flow_seed: int | None = None
    resolved_refinement_ensemble_size = 1
    if flow_refine_stochastic_sampling:
        if refinement_seed is not None:
            resolved_legacy_flow_seed = int(refinement_seed)
        else:
            # Legacy recipes already define training.seed (default 42). Derive
            # a stable initialization-specific stream just like the unified
            # refiners so repeated rollout calls are reproducible without
            # producing identical draws for every forecast initialization.
            time_dim = _dim_names(config)[0]
            resolved_legacy_flow_seed = derive_refinement_seed(
                int(config.get("training", {}).get("seed", 42)),
                ds[time_dim].values[anchor_idx],
            )
    if unified_refiner is not None:
        configured_seed = unified_refiner.refinement_config.seed
        if refinement_seed is not None:
            resolved_refinement_seed = int(refinement_seed)
        elif configured_seed is not None:
            time_dim = _dim_names(config)[0]
            resolved_refinement_seed = derive_refinement_seed(
                int(configured_seed), ds[time_dim].values[anchor_idx]
            )
        resolved_refinement_ensemble_size = int(
            refinement_ensemble_size
            if refinement_ensemble_size is not None
            else (
                1
                if unified_refiner.refinement_config.deterministic_inference
                else unified_refiner.refinement_config.ensemble_size
            )
        )
        if resolved_refinement_ensemble_size < 1:
            raise ValueError("refinement_ensemble_size must be at least one.")

    legacy_temporal_history: dict[tuple[str, str], list[torch.Tensor]] = {}
    unified_temporal_history: list[torch.Tensor] | None = (
        [] if unified_temporal_active else None
    )
    unified_temporal_lead_history: list[torch.Tensor] | None = (
        [] if unified_temporal_history is not None else None
    )
    unified_refinement_generator = (
        _make_refinement_generator(
            device,
            resolved_refinement_seed,
        )
        if unified_refiner is not None
        else None
    )
    legacy_flow_generator = (
        _make_refinement_generator(device, resolved_legacy_flow_seed)
        if flow_refine_stochastic_sampling
        else None
    )
    verbose_provenance = bool(rollout_cfg.get("verbose_provenance", True))

    current = build_aurora_batch(
        ds,
        start_sample,
        config=config,
        resolved_specs=resolved_specs,
    ).to(device)

    predictor_by_aurora = resolved_specs.predictor_by_aurora
    spatial_h, spatial_w = current.spatial_shape

    predictions: list[Batch] = []
    with torch.inference_mode():
        for step in range(1, steps + 1):
            forecast_lead_time_hours = float(step) * step_hours
            if unified_refiner is not None:
                # Phase 1: deterministic Aurora prediction for this rollout
                # step. Phase 2: stochastic residual correction of that same
                # forecast valid time, applied as postprocessing.
                from finetune.refinement.integration import refine_batch_prediction

                deterministic_pred = unified_refiner.aurora(current)
                pred = refine_batch_prediction(
                    unified_refiner,
                    deterministic_pred,
                    aurora_input_batch=current,
                    forecast_lead_time_hours=forecast_lead_time_hours,
                    ensemble_size=resolved_refinement_ensemble_size,
                    generator=unified_refinement_generator,
                    temporal_history=unified_temporal_history,
                    temporal_lead_history=unified_temporal_lead_history,
                    temporal_control=temporal_control,
                )
                feedback_pred = pred if refinement_feedback else deterministic_pred
            else:
                if versioned_flow_postprocessing:
                    # V2+ treats flow correction as postprocessing by default:
                    # Aurora's raw state advances the physical trajectory while
                    # the corrected field is emitted to the caller. Feedback is
                    # an explicit training-config opt-in.
                    deterministic_pred = base_for_fm.base(current)
                    pred = base_for_fm.refine_prediction(
                        deterministic_pred,
                        deterministic=not flow_refine_stochastic_sampling,
                        forecast_lead_time_hours=forecast_lead_time_hours,
                        generator=legacy_flow_generator,
                    )
                    feedback_pred = (
                        pred if flow_refine_feedback else deterministic_pred
                    )
                else:
                    # Contract v1 intentionally retains the historical inline
                    # refinement and autoregressive feedback behaviour exactly.
                    pred = (
                        model(
                            current,
                            forecast_lead_time_hours=forecast_lead_time_hours,
                        )
                        if flow_refine_active
                        else model(current)
                    )
                    feedback_pred = pred

            # Mamba temporal correction (causal): refine the flow-corrected
            # target fields using their evolution across the rollout so far.
            if temporal_active:
                if temporal_control == "on":
                    pred = base_for_fm.apply_temporal_rollout(
                        pred, legacy_temporal_history
                    )
                elif temporal_control == "shuffled":
                    # Update the canonical raw-frame history once, then evaluate
                    # a deterministic negative control with previous leads reversed
                    # while the current lead remains last.
                    base_for_fm.apply_temporal_rollout(
                        pred, legacy_temporal_history
                    )
                    shuffled_history = {
                        key: list(reversed(frames[:-1]))
                        for key, frames in legacy_temporal_history.items()
                    }
                    pred = base_for_fm.apply_temporal_rollout(
                        pred, shuffled_history
                    )
                # temporal_control="off" intentionally keeps the spatial frame.
                # V1 historically fed the temporal output back unconditionally.
                # For v2+, temporal correction is part of the emitted
                # postprocessing and must not replace raw-Aurora provenance
                # unless refinement feedback was explicitly enabled.
                if temporal_control != "off" and (
                    not versioned_flow_postprocessing or flow_refine_feedback
                ):
                    feedback_pred = pred
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
                use_prediction = (
                    feedback_fields is None and name in feedback_pred.surf_vars
                ) or (feedback_fields is not None and name in feedback_fields)
                if use_prediction:
                    new_frame = feedback_pred.surf_vars[name]
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
                use_prediction = (
                    feedback_fields is None and name in feedback_pred.atmos_vars
                ) or (feedback_fields is not None and name in feedback_fields)
                if use_prediction:
                    new_frame = feedback_pred.atmos_vars[name]
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
                metadata=feedback_pred.metadata,
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
    validation: dict[str, Any] | None = None,
    validated_for_inference: bool | None = None,
) -> None:
    """Save training checkpoint and lightweight validation provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Persist flow-refine sampling_steps so inference uses the same phase
    # (1-step deterministic vs multi-step stochastic) that training settled on.
    _inner = model.module if hasattr(model, "module") else model
    flow_sampling_steps: int | None = None
    from finetune.flow_refine import AuroraFlowRefine as _AFR  # noqa: PLC0415
    if isinstance(_inner, _AFR):
        flow_sampling_steps = int(_inner.sampling_steps)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    if validation is not None:
        payload["validation"] = dict(validation)
    if validated_for_inference is None:
        validated_for_inference = bool(
            path.stem == "best" and np.isfinite(float(best_val_loss))
        )
    payload["validated_for_inference"] = bool(validated_for_inference)
    if flow_sampling_steps is not None:
        payload["flow_sampling_steps"] = flow_sampling_steps

    # Record the fully resolved refinement / performance configuration and, for
    # the unified backend, the variable-and-level packing and Aurora identity, so
    # a checkpoint is self-describing and resume is verifiable.
    from finetune.refinement.checkpoint import (  # noqa: PLC0415
        CHECKPOINT_SCHEMA_VERSION,
        aurora_state_fingerprint,
    )
    from finetune.refinement.integration import describe_refinement  # noqa: PLC0415
    from finetune.refinement.two_phase import (  # noqa: PLC0415
        AuroraTwoPhaseRefiner as _ATP,
    )

    resolved = describe_refinement(config)
    payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload["resolved_refinement_config"] = resolved["refinement"]
    payload["resolved_temporal_config"] = resolved["temporal"]
    payload["resolved_performance_config"] = resolved["performance"]
    payload["refinement_backend"] = resolved["backend"]
    payload["refinement_type"] = resolved["refinement"]["type"]
    payload["rng_state"] = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cuda_current": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if isinstance(_inner, _ATP):
        payload["field_packing"] = _inner.packing.to_dict()
        payload["aurora_fingerprint"] = aurora_state_fingerprint(
            {
                key[len("aurora.") :]: value
                for key, value in _inner.state_dict().items()
                if key.startswith("aurora.")
            }
        )
    if norm_stats is not None:
        payload["norm_stats"] = {
            k: {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
            for k, v in norm_stats.items()
        }
    torch.save(payload, str(path))
    # A small sidecar lets inference compare best/last run provenance without
    # loading two multi-gigabyte checkpoint payloads.
    runtime = config.get("runtime", {})
    metadata = {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_role": path.stem,
        "training_run_id": runtime.get("training_run_id"),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "require_refinement_improvement": bool(
            config.get("training", {}).get(
                "require_refinement_improvement", False
            )
        ),
        "validated_for_inference": bool(validated_for_inference),
    }
    if validation is not None:
        def _json_safe_validation_value(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if torch.is_tensor(value):
                tensor = value.detach().cpu()
                return tensor.item() if tensor.numel() == 1 else tensor.tolist()
            if isinstance(value, dict):
                return {
                    str(key): _json_safe_validation_value(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [_json_safe_validation_value(item) for item in value]
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            return str(value)

        metadata["validation"] = _json_safe_validation_value(validation)
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True) + "\n"
    )


def validate_checkpoint_longitude(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
) -> None:
    """Refuse to silently change longitude semantics when loading weights.

    Circular padding is architectural behavior but has no learnable tensor, so
    it is not represented by ordinary convolution state dictionaries. The
    resolved training value is therefore persisted in the checkpoint config
    and compared with the constructed inference/resume model here.
    """
    inner = model.module if hasattr(model, "module") else model
    if not hasattr(inner, "lon_periodic"):
        return

    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(
            "Checkpoint has no saved config, so longitude padding semantics cannot be "
            "verified. Use a checkpoint produced by the periodic-longitude pipeline."
        )
    checkpoint_model_cfg = checkpoint_cfg.get("model", {})
    current_periodic = bool(getattr(inner, "lon_periodic"))
    current_encoding = bool(getattr(inner, "lon_encoding", False))
    saved_periodic = checkpoint_model_cfg.get("lon_periodic_resolved")
    if not isinstance(saved_periodic, bool):
        # Before this field was persisted, every refinement head used replicate
        # longitude padding and no cyclic coordinate channels. That is exactly
        # the current non-periodic architecture (including state-dict keys), so
        # regional legacy checkpoints remain safe. They must never be promoted
        # to a periodic/global model merely because the tensor shapes happen to
        # load successfully.
        if current_periodic or current_encoding:
            raise ValueError(
                "Legacy checkpoint has no model.lon_periodic_resolved value. Its weights "
                "were trained with non-periodic longitude edges and cannot be loaded into "
                "a periodic or longitude-encoded model. Retrain/fine-tune with this "
                "pipeline to create a longitude-aware checkpoint."
            )
        saved_periodic = False
    if bool(saved_periodic) != current_periodic:
        raise ValueError(
            "Checkpoint longitude mismatch: training used "
            f"lon_periodic={bool(saved_periodic)}, but the current model uses "
            f"lon_periodic={current_periodic}. Use the checkpoint's data/model convention."
        )

    saved_encoding = bool(checkpoint_model_cfg.get("flow_refine_lon_encoding", False))
    if saved_encoding != current_encoding:
        raise ValueError(
            "Checkpoint longitude-feature mismatch: training used "
            f"flow_refine_lon_encoding={saved_encoding}, but the current model uses "
            f"{current_encoding}."
        )

    saved_signature = checkpoint_model_cfg.get("longitude_grid_signature")
    current_signature = getattr(inner, "longitude_grid_signature", None)
    if (
        isinstance(saved_signature, str)
        and isinstance(current_signature, str)
        and saved_signature != current_signature
    ):
        raise ValueError(
            "Checkpoint longitude-grid mismatch: inference coordinates differ from the "
            "canonical grid used for training."
        )


def validate_checkpoint_refinement_contract(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    resolved_specs: ResolvedVariableSpecs,
    *,
    allow_temporal_migration: bool = False,
    require_validated: bool | None = None,
) -> None:
    """Validate target, architecture, statistics, and validation provenance.

    Tensor-shape compatibility alone cannot detect reordered predictors or
    pressure levels, changed target semantics, optional temporal modules, or an
    unvalidated ``last.ckpt``. This check makes those non-tensor contracts
    explicit before any refinement weights are used. Training may explicitly
    allow a temporal-only warm-start migration; inference remains strict.
    """
    if require_validated is None:
        require_validated = bool(
            config.get("inference", {}).get(
                "require_validated_checkpoint", False
            )
        )
    validate_checkpoint_validation_provenance(
        checkpoint,
        require_validated=bool(require_validated),
    )

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, Mapping):
        obsolete_lazy_keys = [
            key for key in state_dict if "._mamba_impl." in str(key)
        ]
        if obsolete_lazy_keys:
            raise ValueError(
                "Checkpoint contains obsolete lazy CUDA Mamba state under "
                "'._mamba_impl.'. That backend was instantiated after optimizer "
                "construction and used weights independent from the trained "
                "SelectiveSSM path, so the checkpoint cannot be represented as "
                "numerically compatible. Retrain the temporal phase with the "
                "eager selective_scan_ref backend."
            )

    inner = model.module if hasattr(model, "module") else model
    from finetune.flow_refine import AuroraFlowRefine
    from finetune.refinement.two_phase import AuroraTwoPhaseRefiner

    if isinstance(inner, AuroraTwoPhaseRefiner):
        from finetune.model_factory import validate_unified_checkpoint_contract

        validate_unified_checkpoint_contract(
            inner,
            checkpoint,
            config,
            allow_temporal_migration=allow_temporal_migration,
        )
        return

    if not isinstance(inner, AuroraFlowRefine):
        return

    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(
            "Flow-refine checkpoint has no saved configuration; target ordering "
            "and normalization semantics cannot be verified."
        )

    def variable_signature(
        items: Sequence[Any],
        *,
        default_kind: str | None = None,
        include_loss_levels: bool = False,
    ) -> tuple[tuple[Any, ...], ...]:
        signature: list[tuple[Any, ...]] = []
        for item in items:
            if isinstance(item, VariableSpec):
                dataset_name = item.dataset_name
                aurora_name = item.aurora_name
                kind = item.kind
                loss_levels = tuple(item.loss_levels or ())
            elif isinstance(item, str):
                dataset_name = item
                aurora_name = item
                kind = default_kind or ""
                loss_levels = ()
            elif isinstance(item, dict):
                dataset_name = str(item.get("dataset_name") or item.get("name") or "")
                aurora_name = str(item.get("aurora_name") or dataset_name)
                kind = str(item.get("kind") or default_kind or "")
                loss_levels = tuple(
                    float(x) for x in (item.get("loss_levels") or ())
                )
            else:
                continue
            fields: tuple[Any, ...] = (dataset_name, aurora_name, kind)
            if include_loss_levels:
                fields += (loss_levels,)
            signature.append(fields)
        return tuple(signature)

    saved_data = checkpoint_cfg.get("data", {})
    saved_targets = variable_signature(
        saved_data.get("target_variables", ()),
        include_loss_levels=True,
    )
    current_targets = variable_signature(
        resolved_specs.targets,
        include_loss_levels=True,
    )
    if saved_targets != current_targets:
        raise ValueError(
            "Checkpoint target contract mismatch. "
            f"saved={saved_targets}, current={current_targets}. Dataset/Aurora names, "
            "kinds, and pressure-level ordering must be identical."
        )

    for label, saved_items, current_items, default_kind in (
        (
            "predictor",
            saved_data.get("predictor_variables", ()),
            resolved_specs.predictors,
            None,
        ),
        (
            "static-variable",
            saved_data.get("static_variables", ()),
            resolved_specs.static,
            "static",
        ),
    ):
        saved_signature = variable_signature(
            saved_items,
            default_kind=default_kind,
        )
        current_signature = variable_signature(
            current_items,
            default_kind=default_kind,
        )
        if saved_signature != current_signature:
            raise ValueError(
                f"Checkpoint {label} contract mismatch. "
                f"saved={saved_signature}, current={current_signature}."
            )

    requires_atmospheric_levels = any(
        spec.kind == "atmos"
        for spec in (*resolved_specs.predictors, *resolved_specs.targets)
    )
    saved_levels = tuple(float(value) for value in saved_data.get("atmos_levels", ()))
    current_levels = tuple(
        float(value) for value in config.get("data", {}).get("atmos_levels", ())
    )
    if requires_atmospheric_levels and (
        not saved_levels or saved_levels != current_levels
    ):
        raise ValueError(
            "Checkpoint atmospheric-level contract mismatch: "
            f"saved={saved_levels}, current={current_levels}."
        )

    saved_model = checkpoint_cfg.get("model", {})
    current_model = config.get("model", {})
    architecture_keys = (
        "model_variant",
        "patch_size",
        "flow_refine_enabled",
        "flow_refine_contract_version",
        "flow_refine_hidden",
        "flow_refine_time_dim",
        "flow_refine_doy_cond",
        "flow_refine_lead_time_cond",
        "flow_refine_lead_time_scale_hours",
        "flow_refine_lon_encoding",
        "flow_refine_residual_zscore",
        "flow_refine_res_std_momentum",
    )
    defaults: dict[str, Any] = {
        "model_variant": "aurora_pretrained",
        "patch_size": 4,
        "flow_refine_enabled": False,
        "flow_refine_contract_version": 1,
        "flow_refine_hidden": 64,
        "flow_refine_time_dim": 128,
        "flow_refine_doy_cond": False,
        "flow_refine_lead_time_cond": False,
        "flow_refine_lead_time_scale_hours": 72.0,
        "flow_refine_lon_encoding": False,
        "flow_refine_residual_zscore": False,
        "flow_refine_res_std_momentum": 0.99,
    }
    mismatches = [
        (
            key,
            saved_model.get(key, defaults[key]),
            current_model.get(key, defaults[key]),
        )
        for key in architecture_keys
        if saved_model.get(key, defaults[key])
        != current_model.get(key, defaults[key])
    ]
    if mismatches:
        detail = ", ".join(
            f"{key}: saved={saved!r}, current={current!r}"
            for key, saved, current in mismatches
        )
        raise ValueError(
            "Checkpoint refinement configuration mismatch: " + detail
        )

    from finetune.refinement.two_phase import resolve_temporal_config

    saved_temporal = resolve_temporal_config(checkpoint_cfg)
    serialized_temporal = checkpoint.get("resolved_temporal_config")
    if isinstance(serialized_temporal, dict):
        saved_temporal.update(serialized_temporal)
    current_temporal = resolve_temporal_config(config)
    temporal_fields = {
        "enabled": "mamba_temporal_enabled",
        "mode": "mamba_temporal.mode",
        "channels": "mamba_temporal_channels",
        "state": "mamba_temporal_state",
        "layers": "mamba_temporal_layers",
        "conv": "mamba_temporal_conv",
        "expand": "mamba_temporal_expand",
        "dropout": "mamba_temporal.dropout",
        "gated_fusion": "mamba_temporal.gated_fusion",
        "gate_init": "mamba_temporal.gate_init",
        "lead_time_conditioning": "mamba_temporal.lead_time_conditioning",
        "mask_conditioning": "mamba_temporal.mask_conditioning",
        "coordinate_conditioning": "mamba_temporal.coordinate_conditioning",
        "causal": "mamba_temporal.causal",
        "semantic_version": "mamba_temporal.semantic_version",
        "scan_backend": "mamba_temporal.scan_backend",
    }
    temporal_both_enabled = bool(saved_temporal["enabled"]) and bool(
        current_temporal["enabled"]
    )
    compared_temporal_fields = (
        tuple(temporal_fields)
        if temporal_both_enabled
        else ("enabled",)
    )
    temporal_mismatches = [
        (
            temporal_fields[key],
            saved_temporal.get(key),
            current_temporal.get(key),
        )
        for key in compared_temporal_fields
        if saved_temporal.get(key) != current_temporal.get(key)
    ]
    if temporal_mismatches and not allow_temporal_migration:
        detail = ", ".join(
            f"{key}: saved={saved!r}, current={current!r}"
            for key, saved, current in temporal_mismatches
        )
        raise ValueError(
            "Checkpoint Mamba temporal configuration mismatch: "
            + detail
            + ". Use a checkpoint trained with the current YAML, or explicitly "
            "run a training warm-start migration before inference."
        )

    if (
        bool(current_model.get("flow_refine_lead_time_cond", False))
        or temporal_both_enabled
    ):
        saved_step = checkpoint_cfg.get("rollout", {}).get("rollout_step_hours")
        current_step = config.get("rollout", {}).get("rollout_step_hours")
        if saved_step is None or current_step is None or not np.isclose(
            float(saved_step), float(current_step), rtol=0.0, atol=1.0e-6,
        ):
            raise ValueError(
                "Checkpoint forecast-lead cadence mismatch: "
                f"saved rollout_step_hours={saved_step!r}, "
                f"current={current_step!r}."
            )
        saved_steps = tuple(
            int(value)
            for value in checkpoint_cfg.get("data", {}).get(
                "target_lead_times", ()
            )
        )
        current_steps = tuple(
            int(value)
            for value in config.get("data", {}).get("target_lead_times", ())
        )
        if not saved_steps or saved_steps != current_steps:
            raise ValueError(
                "Checkpoint forecast-lead supervision mismatch: "
                f"saved target_lead_times={saved_steps}, current={current_steps}."
            )

    stats = checkpoint.get("norm_stats")
    if not isinstance(stats, dict):
        raise ValueError(
            "Flow-refine checkpoint has no norm_stats. Recreate the checkpoint "
            "with normalization metadata instead of silently assuming unit scale."
        )
    for spec in resolved_specs.targets:
        if spec.aurora_name not in stats:
            raise ValueError(
                f"Checkpoint norm_stats are missing target {spec.aurora_name!r}."
            )
        expected = (
            len(spec.loss_levels)
            if spec.kind == "atmos" and spec.loss_levels is not None
            else 1 if spec.kind == "surf" else len(
                config.get("data", {}).get("atmos_levels", ())
            )
        )
        for key in ("mean", "std"):
            value = stats[spec.aurora_name].get(key)
            if value is None or int(torch.as_tensor(value).numel()) != expected:
                raise ValueError(
                    f"Checkpoint norm_stats[{spec.aurora_name!r}][{key!r}] "
                    f"must contain {expected} value(s)."
                )
        if torch.any(torch.as_tensor(stats[spec.aurora_name]["std"]) <= 0):
            raise ValueError(
                f"Checkpoint norm_stats[{spec.aurora_name!r}]['std'] "
                "contains a non-positive scale."
            )


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
    validate_checkpoint_longitude(model, checkpoint)
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
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    resolved_path = output_dir / "resolved_config.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to record resolved_config.yaml.")
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    )
    return str(path.resolve())


def _smooth_patch_artifacts(
    arr: np.ndarray, sigma: float, patch_size: int, lon_periodic: bool = False,
) -> np.ndarray:
    """Apply Gaussian smoothing to remove patch-boundary artifacts.

    Works on 2-D (H, W) or higher-dimensional arrays by smoothing the last
    two spatial dimensions independently per slice.  Uses
    ``scipy.ndimage.gaussian_filter`` when available, otherwise falls back to a
    simple uniform box filter that is almost as effective for small sigma.

    The smoothing preserves the overall magnitude and spatial structure while
    blending the sharp patch-boundary discontinuities produced by the ViT
    decoder's ``unpatchify`` operation.

    When ``lon_periodic`` the longitude axis (last) is smoothed with a *wrap*
    boundary so no seam is introduced at the 0°/360° dateline; latitude (the
    poles) always uses the non-periodic boundary. With ``lon_periodic=False``
    the behaviour is identical to the previous replicate/reflect smoothing.
    """
    if sigma <= 0:
        return arr

    try:
        from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

        # Smooth only the last two (lat, lon) dimensions.
        axes = tuple(range(arr.ndim - 2, arr.ndim))  # e.g. (-2, -1)
        # Per-axis boundary mode: latitude keeps the default 'reflect'; the
        # longitude axis wraps when the domain is periodic.
        mode = ["reflect", "wrap"] if lon_periodic else "reflect"
        return gaussian_filter(
            arr.astype(np.float64), sigma=sigma, axes=axes, mode=mode,
        ).astype(arr.dtype)
    except ImportError:
        pass

    # Fallback: uniform box filter with kernel_size ~ 2*sigma+1
    kernel_size = max(3, int(2 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    def _smooth_2d(img: np.ndarray) -> np.ndarray:
        if lon_periodic:
            # Latitude reflect, longitude wrap (periodic across the dateline).
            padded = np.pad(img, ((pad, pad), (0, 0)), mode="reflect")
            padded = np.pad(padded, ((0, 0), (pad, pad)), mode="wrap")
        else:
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
    lon_periodic: bool | None = None,
    initialization_time: Any | None = None,
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

    Longitude is treated as periodic (wrap smoothing, no seam at the 0°/360°
    dateline) when *lon_periodic* is true. ``None`` (default) auto-detects it
    from the longitude grid, so global outputs are handled correctly without
    extra configuration. A duplicated wrap column (both 0° and 360° present) is
    dropped so the CF longitude coordinate is not degenerate.
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
    levels = np.asarray(first.metadata.atmos_levels, dtype=np.float64)

    # Canonicalise every step independently and retain its reindexer. This both
    # removes a duplicated 0/360 endpoint and prevents values from being written
    # under the wrong coordinate if one step arrives in a different convention.
    from finetune.longitude import canonical_longitudes, longitude_is_periodic

    lon_indices: list[np.ndarray] = []
    lon: np.ndarray | None = None
    for step, pred in enumerate(predictions):
        step_lat = np.asarray(pred.metadata.lat.detach().cpu().numpy(), dtype=np.float64)
        step_levels = np.asarray(pred.metadata.atmos_levels, dtype=np.float64)
        if step_levels.shape != levels.shape or not np.allclose(
            step_levels, levels, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(f"Prediction step {step} uses different atmospheric levels.")
        raw_step_lon = np.asarray(
            pred.metadata.lon.detach().cpu().numpy(), dtype=np.float64,
        )
        for kind, variables in (
            ("surface", pred.surf_vars),
            ("atmospheric", pred.atmos_vars),
        ):
            for name, value in variables.items():
                if value.shape[-2:] != (step_lat.size, raw_step_lon.size):
                    raise ValueError(
                        f"Prediction step {step} {kind} variable {name!r} has spatial "
                        f"shape {tuple(value.shape[-2:])}, expected "
                        f"{(step_lat.size, raw_step_lon.size)} from metadata."
                    )
                if kind == "atmospheric" and value.shape[-3] != step_levels.size:
                    raise ValueError(
                        f"Prediction step {step} atmospheric variable {name!r} has "
                        f"{value.shape[-3]} levels, expected {step_levels.size}."
                    )
        step_lon, step_indices = canonical_longitudes(raw_step_lon)
        if step_lat.shape != lat.shape or not np.allclose(
            step_lat, lat, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(f"Prediction step {step} uses a different latitude grid.")
        if lon is None:
            lon = step_lon
        elif step_lon.shape != lon.shape or not np.allclose(
            step_lon, lon, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(f"Prediction step {step} uses a different longitude grid.")
        lon_indices.append(step_indices)
    assert lon is not None

    coordinate_is_periodic = bool(longitude_is_periodic(lon))
    if lon_periodic is None:
        lon_periodic = coordinate_is_periodic
    if lon_periodic and not coordinate_is_periodic:
        raise ValueError(
            "Cannot apply periodic output processing to an incomplete longitude coordinate."
        )

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
            [
                pred.surf_vars[var_name][0, 0].detach().cpu().float().numpy()[
                    ..., lon_indices[step]
                ]
                for step, pred in enumerate(predictions)
            ],
            axis=0,
        )  # (time, lat, lon)
        if smooth_sigma > 0:
            arr = _smooth_patch_artifacts(
                arr, sigma=smooth_sigma, patch_size=patch_size, lon_periodic=lon_periodic,
            )
        out_name = surf_name_map.get(var_name, var_name)
        data_vars[out_name] = (("time", "latitude", "longitude"), arr)

    for var_name in first.atmos_vars:
        if target_aurora_names is not None and var_name not in target_aurora_names:
            continue
        arr = np.stack(
            [
                pred.atmos_vars[var_name][0, 0].detach().cpu().float().numpy()[
                    ..., lon_indices[step]
                ]
                for step, pred in enumerate(predictions)
            ],
            axis=0,
        )  # (time, level, lat, lon)
        if smooth_sigma > 0:
            arr = _smooth_patch_artifacts(
                arr, sigma=smooth_sigma, patch_size=patch_size, lon_periodic=lon_periodic,
            )
        out_name = atmos_name_map.get(var_name, var_name)
        data_vars[out_name] = (("time", "level", "latitude", "longitude"), arr)

    coordinates: dict[str, Any] = {
        "time": times,
        "latitude": lat,
        "longitude": lon,
        "level": levels,
    }
    if initialization_time is not None:
        init_time = np.datetime64(initialization_time, "ns")
        coordinates["forecast_reference_time"] = init_time
        coordinates["lead_time"] = (
            "time", np.asarray((times - init_time) / np.timedelta64(1, "h"), dtype=np.float64)
        )
    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coordinates,
    )
    if resolved_specs is not None:
        for spec in resolved_specs.targets:
            if spec.dataset_name in ds_out and spec.units:
                ds_out[spec.dataset_name].attrs["units"] = spec.units
                ds_out[spec.dataset_name].attrs["long_name"] = spec.dataset_name


    # CF-compliant coordinate metadata so downstream tools recognise the axes
    # and (for longitude) the periodic wrap convention.
    ds_out["latitude"].attrs.update(
        {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
    )
    lon_attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }
    if coordinate_is_periodic:
        # Helpful circular-axis extensions. CF recognition itself comes from
        # standard_name + degrees_east; no duplicate endpoint is stored.
        lon_attrs["modulo"] = 360.0
        lon_attrs["topology"] = "circular"
    ds_out["longitude"].attrs.update(lon_attrs)
    if "level" in ds_out.coords:
        ds_out["level"].attrs.update(
            {
                "standard_name": "air_pressure",
                "units": "hPa",
                "axis": "Z",
                "positive": "down",
            }
        )
    if "forecast_reference_time" in ds_out.coords:
        ds_out["forecast_reference_time"].attrs.update(
            {"standard_name": "forecast_reference_time"}
        )
    if "lead_time" in ds_out.coords:
        ds_out["lead_time"].attrs.update(
            {"standard_name": "forecast_period", "long_name": "forecast lead time", "units": "hours"}
        )
    ds_out["time"].attrs.update({"standard_name": "time", "axis": "T"})
    ds_out.attrs["Conventions"] = "CF-1.10"
    for coord_name in ("time", "latitude", "longitude", "level"):
        ds_out[coord_name].encoding["_FillValue"] = None

    if save_netcdf:
        ds_out.to_netcdf(str(output_path))

    return ds_out
