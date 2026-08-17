"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Single model-construction path for Aurora training and rollout inference."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _model_registry() -> dict[str, type[torch.nn.Module]]:
    from aurora import (
        Aurora,
        Aurora12hPretrained,
        AuroraAirPollution,
        AuroraHighRes,
        AuroraPretrained,
        AuroraSmallPretrained,
        AuroraWave,
    )

    return {
        "aurora": Aurora,
        "aurora_pretrained": AuroraPretrained,
        "aurora_small_pretrained": AuroraSmallPretrained,
        "aurora_12h_pretrained": Aurora12hPretrained,
        "aurora_highres": AuroraHighRes,
        "aurora_air_pollution": AuroraAirPollution,
        "aurora_wave": AuroraWave,
    }


def build_finetune_model(
    config: dict[str, Any],
    resolved_specs: Any,
    *,
    lon: Any | None = None,
    lat: Any | None = None,
    norm_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    load_pretrained: bool = False,
    autocast: bool | None = None,
) -> torch.nn.Module:
    """Construct Aurora and every configured refinement wrapper exactly once."""
    from finetune import aurora_finetune_utils as ft

    model_cfg = config.get("model", {})
    variant = str(model_cfg.get("model_variant", "aurora_pretrained")).strip().lower()
    registry = _model_registry()
    if variant not in registry:
        raise ValueError(
            f"model.model_variant={variant!r} is unsupported; expected one of "
            f"{sorted(registry)}."
        )

    variable_config = ft.derive_model_variable_config(resolved_specs, config)
    model_kwargs = model_cfg.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError(
            f"model.model_kwargs must be a mapping, got {type(model_kwargs).__name__}."
        )
    model_kwargs = dict(model_kwargs)
    if "patch_size" in model_cfg and "patch_size" not in model_kwargs:
        model_kwargs["patch_size"] = int(model_cfg["patch_size"])
    if autocast is None:
        precision = str(model_cfg.get("mixed_precision", "none")).lower()
        autocast = precision in {"bf16", "bfloat16", "fp16"}
    model_kwargs["autocast"] = bool(autocast)

    model = registry[variant](
        surf_vars=variable_config["surf_vars"],
        static_vars=variable_config["static_vars"],
        atmos_vars=variable_config["atmos_vars"],
        **model_kwargs,
    )

    if load_pretrained and bool(model_cfg.get("use_pretrained_weights", True)):
        checkpoint = config.get("paths", {}).get("pretrained_checkpoint")
        if checkpoint:
            model.load_checkpoint_local(str(checkpoint), strict=False)
        else:
            model.load_checkpoint(strict=False)

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.configure_activation_checkpointing()

    model = ft.maybe_wrap_conv_refine(model, config, resolved_specs, lon=lon)
    model = ft.maybe_wrap_flow_refine(model, config, resolved_specs, lon=lon)

    from finetune.flow_refine import AuroraFlowRefine

    if isinstance(model, AuroraFlowRefine) and norm_stats is not None:
        model.set_norm_stats(norm_stats)

    configured_nonnegative = {
        str(name)
        for name in config.get("data", {}).get("nonnegative_target_variables", ())
    }
    nonnegative = tuple(
        spec.aurora_name
        for spec in resolved_specs.targets
        if spec.aurora_name in configured_nonnegative
        or spec.dataset_name in configured_nonnegative
    )
    model = ft.maybe_wrap_stochastic_refine(
        model,
        config,
        resolved_specs,
        lon=lon,
        lat=lat,
        norm_stats=norm_stats,
        nonnegative_variables=nonnegative,
    )
    return model


def validate_unified_checkpoint_contract(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    *,
    allow_temporal_migration: bool = False,
) -> None:
    """Validate the complete unified refinement architecture and packing contract."""
    from finetune.refinement.integration import describe_refinement
    from finetune.refinement.two_phase import resolve_temporal_config

    saved = checkpoint.get("resolved_refinement_config")
    if not isinstance(saved, dict):
        raise ValueError(
            "Unified refinement checkpoint is missing resolved_refinement_config; "
            "architecture compatibility cannot be verified."
        )
    # Re-resolve the saved raw YAML with today's schema so newly added optional
    # fields receive their legacy-safe defaults. Comparing the old serialized
    # partial dict directly would reject an otherwise identical checkpoint.
    saved_config = checkpoint.get("config")
    current = describe_refinement(config)["refinement"]
    if isinstance(saved_config, dict):
        saved = describe_refinement(saved_config)["refinement"]
    common_scientific_keys = (
        "enabled",
        "type",
        "correction_convention",
        "freeze_aurora",
        "joint_finetuning",
        "train_on_residual",
        "feedback_to_rollout",
        "deterministic_head",
        "target_space",
        "conditioning",
        "loss",
    )
    # A resolved config serializes both stochastic-process and both backbone
    # sections, including legacy-safe defaults for the inactive alternatives.
    # Those inactive values do not describe the checkpoint architecture and
    # therefore must not prevent a compatible load. The canonical type still
    # remains a common field, so switching process or backbone is rejected.
    current_type = str(current.get("type", "none"))
    active_scientific_keys: list[str] = []
    if current_type.startswith("diffusion_"):
        active_scientific_keys.append("diffusion")
    elif current_type.startswith("flow_matching_"):
        active_scientific_keys.append("flow_matching")
    if current_type != "none":
        active_scientific_keys.append(
            "transformer" if current_type.endswith("_transformer") else "unet"
        )
    scientific_keys = (*common_scientific_keys, *active_scientific_keys)
    mismatches = [
        key for key in scientific_keys if saved.get(key) != current.get(key)
    ]
    if mismatches:
        details = "; ".join(
            f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
            for key in mismatches
        )
        raise ValueError("Checkpoint unified refinement mismatch: " + details)

    saved_packing = checkpoint.get("field_packing")
    if not isinstance(saved_packing, dict):
        raise ValueError(
            "Unified refinement checkpoint is missing field_packing metadata."
    )
    current_packing = model.packing.to_dict()
    patch_size = int(config.get("model", {}).get("patch_size", 4))

    def _packing_value_matches(key: str) -> bool:
        saved_value = saved_packing.get(key)
        current_value = current_packing.get(key)
        if saved_value == current_value:
            return True
        # Checkpoints written before packing coordinates followed Aurora's
        # patch crop stored the full source axis. Accept only the exact legacy
        # prefix pattern (fewer than one patch of trailing coordinates).
        if key in {"lat", "lon"} and isinstance(saved_value, list) and isinstance(
            current_value, list
        ):
            trailing = len(saved_value) - len(current_value)
            if (
                0 < trailing < patch_size
                and saved_value[: len(current_value)] == current_value
            ):
                warnings.warn(
                    f"Checkpoint field_packing.{key} contains {trailing} legacy "
                    "trailing coordinate(s) removed by Aurora patch alignment; "
                    "accepting the compatible cropped prefix.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return True
        return False

    for key in (
        "channels",
        "lat",
        "lon",
        "lead_times_hours",
        "lead_time_scale_hours",
        "lon_periodic",
    ):
        if not _packing_value_matches(key):
            raise ValueError(
                f"Checkpoint field_packing.{key} mismatch: expected "
                f"{saved_packing.get(key)!r}, got {current_packing.get(key)!r}."
            )

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Unified refinement checkpoint is missing its training config.")
    saved_step = saved_config.get("rollout", {}).get("rollout_step_hours")
    saved_model = saved_config.get("model", {})
    current_model = config.get("model", {})
    for key, default in (
        ("model_variant", "aurora_pretrained"),
        ("patch_size", 4),
        ("model_kwargs", {}),
    ):
        saved_value = saved_model.get(key, default)
        current_value = current_model.get(key, default)
        if saved_value != current_value:
            raise ValueError(
                f"Checkpoint model.{key} mismatch: expected {saved_value!r}, "
                f"got {current_value!r}."
            )

    saved_temporal = resolve_temporal_config(saved_config)
    serialized_temporal = checkpoint.get("resolved_temporal_config")
    if isinstance(serialized_temporal, dict):
        # Hydrate newly added fields from legacy-safe raw-config defaults while
        # retaining every field explicitly serialized by newer checkpoints.
        saved_temporal.update(serialized_temporal)
    current_temporal = resolve_temporal_config(config)
    temporal_fields = {
        "enabled": "model.mamba_temporal_enabled",
        "mode": "model.mamba_temporal.mode",
        "channels": "model.mamba_temporal_channels",
        "state": "model.mamba_temporal_state",
        "layers": "model.mamba_temporal_layers",
        "conv": "model.mamba_temporal_conv",
        "expand": "model.mamba_temporal_expand",
        "dropout": "model.mamba_temporal.dropout",
        "gated_fusion": "model.mamba_temporal.gated_fusion",
        "gate_init": "model.mamba_temporal.gate_init",
        "lead_time_conditioning": "model.mamba_temporal.lead_time_conditioning",
        "mask_conditioning": "model.mamba_temporal.mask_conditioning",
        "coordinate_conditioning": "model.mamba_temporal.coordinate_conditioning",
        "causal": "model.mamba_temporal.causal",
        "learning_rate_multiplier": (
            "model.mamba_temporal.learning_rate_multiplier"
        ),
        "objective": "model.mamba_temporal.objective",
        "semantic_version": "model.mamba_temporal.semantic_version",
        "scan_backend": "model.mamba_temporal.scan_backend",
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
        key
        for key in compared_temporal_fields
        if saved_temporal.get(key) != current_temporal.get(key)
    ]
    if temporal_mismatches and not allow_temporal_migration:
        details = "; ".join(
            f"{temporal_fields[key]}: saved={saved_temporal.get(key)!r}, "
            f"current={current_temporal.get(key)!r}"
            for key in temporal_mismatches
        )
        raise ValueError(
            "Checkpoint Mamba temporal configuration mismatch: "
            + details
            + ". Inference requires the exact training architecture."
        )

    saved_data = saved_config.get("data", {})
    current_data = config.get("data", {})

    def _variable_signature(items: Any) -> tuple[tuple[Any, ...], ...]:
        result = []
        for item in items or ():
            if isinstance(item, str):
                result.append((item, item, None))
            elif isinstance(item, dict):
                dataset_name = str(item.get("dataset_name") or item.get("name") or "")
                result.append(
                    (
                        dataset_name,
                        str(item.get("aurora_name") or dataset_name),
                        item.get("kind"),
                    )
                )
        return tuple(result)

    for key in ("predictor_variables", "static_variables"):
        saved_value = _variable_signature(saved_data.get(key))
        current_value = _variable_signature(current_data.get(key))
        if saved_value != current_value:
            raise ValueError(
                f"Checkpoint data.{key} mismatch: expected {saved_value!r}, "
                f"got {current_value!r}."
            )
    current_step = config.get("rollout", {}).get("rollout_step_hours")
    if saved_step != current_step:
        raise ValueError(
            "Checkpoint rollout.rollout_step_hours mismatch: "
            f"expected {saved_step!r}, got {current_step!r}."
        )
    saved_leads = list(saved_config.get("data", {}).get("target_lead_times", ()))
    current_leads = list(config.get("data", {}).get("target_lead_times", ()))
    if saved_leads != current_leads:
        raise ValueError(
            "Checkpoint data.target_lead_times mismatch: "
            f"expected {saved_leads!r}, got {current_leads!r}."
        )


def validate_loaded_residual_scaler_contract(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
) -> None:
    """Validate exact training-split correction scaling after state restoration.

    This check intentionally runs after load_state_dict. Tensor-shape and config
    checks cannot distinguish exact training-split statistics from legacy online
    warm-up statistics because both have the same buffers and can both be frozen.
    Production inference verifies loaded buffers against checkpoint provenance.
    """
    from finetune.refinement.residual_scaling import ResidualScaler

    inner = model.module if hasattr(model, "module") else model
    active_scalers = {
        name.removeprefix("module."): module
        for name, module in inner.named_modules()
        if isinstance(module, ResidualScaler) and module.is_active
    }
    # Legacy flow refinement and unified refiners configured with scaling=none
    # have no learned correction transform and remain fully compatible.
    if not active_scalers:
        return

    packing = getattr(inner, "packing", None)
    if packing is None:
        raise ValueError(
            "Production inference with an active ResidualScaler requires the "
            "runtime model's canonical field-packing metadata."
        )

    saved_config = checkpoint.get("config")
    runtime = saved_config.get("runtime") if isinstance(saved_config, Mapping) else None
    report = runtime.get("residual_calibration") if isinstance(runtime, Mapping) else None
    if not isinstance(report, Mapping):
        raise ValueError(
            "Production inference with active residual scaling requires checkpoint "
            "config.runtime.residual_calibration from an exact training-split "
            "calibration pre-pass."
        )

    if report.get("method") != "exact_training_split" or report.get("complete") is not True:
        raise ValueError(
            "Checkpoint residual calibration is not marked as a complete exact "
            "training-split calibration."
        )
    if int(report.get("version", -1)) != 1:
        raise ValueError(
            "Unsupported checkpoint residual-calibration report version "
            f"{report.get('version')!r}; expected 1."
        )
    if report.get("correction_convention") not in {
        "CAMS_minus_Aurora",
        "cams_minus_aurora_add",
    }:
        raise ValueError(
            "Checkpoint residual calibration has an incompatible correction "
            f"convention: {report.get('correction_convention')!r}."
        )

    try:
        logical_samples = int(report["logical_samples"])
        packed_examples = int(report["packed_examples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint residual calibration is missing integer sample counts."
        ) from exc
    if logical_samples <= 0 or packed_examples <= 0:
        raise ValueError(
            "Checkpoint residual-calibration sample counts must be positive."
        )

    fingerprint_hex = report.get("training_split_fingerprint_sha256")
    try:
        fingerprint = bytes.fromhex(str(fingerprint_hex))
    except ValueError as exc:
        raise ValueError(
            "Checkpoint residual calibration has an invalid training-split "
            "fingerprint."
        ) from exc
    if len(fingerprint) != 32:
        raise ValueError(
            "Checkpoint residual-calibration fingerprint must contain 32 bytes."
        )

    runtime_leads = tuple(float(value) for value in packing.lead_times_hours)
    try:
        report_leads = tuple(float(value) for value in report["lead_times_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint residual calibration is missing valid lead-time metadata."
        ) from exc
    if report_leads != runtime_leads:
        raise ValueError(
            "Checkpoint residual-calibration lead times do not match runtime "
            f"field packing: checkpoint={report_leads}, runtime={runtime_leads}."
        )
    if not runtime_leads or packed_examples != logical_samples * len(runtime_leads):
        raise ValueError(
            "Checkpoint residual-calibration packed-example count is inconsistent "
            "with logical samples and the runtime lead-time packing."
        )

    raw_scaler_reports = report.get("scalers")
    if not isinstance(raw_scaler_reports, list):
        raise ValueError(
            "Checkpoint residual calibration must contain a scaler report list."
        )
    scaler_reports: dict[str, Mapping[str, Any]] = {}
    for raw in raw_scaler_reports:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("module"), str):
            raise ValueError(
                "Checkpoint residual calibration contains an invalid scaler record."
            )
        name = str(raw["module"]).removeprefix("module.")
        if name in scaler_reports:
            raise ValueError(
                f"Checkpoint residual calibration duplicates scaler module {name!r}."
            )
        scaler_reports[name] = raw
    if set(scaler_reports) != set(active_scalers):
        raise ValueError(
            "Checkpoint residual-calibration scaler modules do not match the "
            f"runtime model: checkpoint={sorted(scaler_reports)}, "
            f"runtime={sorted(active_scalers)}."
        )

    channel_specs = tuple(packing.channels)
    for name, scaler in active_scalers.items():
        saved = scaler_reports[name]
        if (
            saved.get("method") != "exact_training_split"
            or saved.get("complete") is not True
            or saved.get("frozen") is not True
        ):
            raise ValueError(
                f"Checkpoint scaler {name!r} is not recorded as complete, exact, "
                "and frozen."
            )
        if saved.get("mode") != scaler.mode or bool(saved.get("center")) != scaler.center:
            raise ValueError(
                f"Checkpoint scaler {name!r} mode/centering metadata does not "
                "match the runtime scaler."
            )
        if (
            int(saved.get("logical_samples", -1)) != logical_samples
            or int(saved.get("packed_examples", -1)) != packed_examples
        ):
            raise ValueError(
                f"Checkpoint scaler {name!r} sample counts disagree with the "
                "top-level calibration report."
            )
        try:
            scaler.validate_exact_training_split_calibration(
                logical_samples=logical_samples,
                packed_examples=packed_examples,
                fingerprint=fingerprint,
            )
        except Exception as exc:
            raise ValueError(
                f"Loaded scaler {name!r} failed exact training-split provenance "
                "validation."
            ) from exc

        saved_channels = saved.get("channels")
        if not isinstance(saved_channels, list) or len(saved_channels) != len(channel_specs):
            raise ValueError(
                f"Checkpoint scaler {name!r} channel report does not match runtime "
                "field packing."
            )
        counts = scaler.calibration_count.detach().double().reshape(-1).cpu().tolist()
        for index, (record, spec, count) in enumerate(
            zip(saved_channels, channel_specs, counts)
        ):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Checkpoint scaler {name!r} channel {index} is not a mapping."
                )
            expected = {
                "index": int(spec.index),
                "variable": str(spec.dataset_name),
                "aurora_variable": str(spec.aurora_name),
                "level_hpa": None if spec.level is None else float(spec.level),
                "units": str(spec.units),
                "valid_cell_count": int(count),
            }
            mismatches = {
                key: (record.get(key), value)
                for key, value in expected.items()
                if record.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"Checkpoint scaler {name!r} channel {index} metadata/count "
                    f"does not match runtime packing/state: {mismatches}."
                )


def load_model_from_checkpoint(
    config: dict[str, Any],
    resolved_specs: Any,
    checkpoint_path: str | Path,
    *,
    lon: Any | None = None,
    lat: Any | None = None,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
    autocast: bool | None = None,
    require_validated: bool | None = None,
    flow_sampling_steps_override: int | None = None,
    allow_unsafe_temporal_sampling_override: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild and strictly load a combined checkpoint.

    ``require_validated=None`` follows
    ``inference.require_validated_checkpoint``. Training-only diagnostics may
    pass ``False`` to inspect the current run's ``last.ckpt`` without weakening
    the default inference policy.
    """
    from finetune import aurora_finetune_utils as ft

    production_inference = bool(
        config.get("inference", {}).get(
            "require_validated_checkpoint", False
        )
        if require_validated is None else require_validated
    )

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(
        str(path), map_location=map_location, weights_only=False, mmap=mmap
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint {path} must contain a mapping, got {type(checkpoint).__name__}."
        )
    norm_stats = checkpoint.get("norm_stats")
    if checkpoint.get("refinement_backend") == "unified" and not isinstance(
        norm_stats, dict
    ):
        raise ValueError(
            f"Unified refinement checkpoint {path} is missing norm_stats."
        )
    model = build_finetune_model(
        config,
        resolved_specs,
        lon=lon,
        lat=lat,
        norm_stats=norm_stats,
        load_pretrained=False,
        autocast=autocast,
    )
    ft.validate_checkpoint_longitude(model, checkpoint)
    ft.validate_checkpoint_refinement_contract(
        model,
        checkpoint,
        config,
        resolved_specs,
        require_validated=require_validated,
    )
    from finetune.flow_refine import AuroraFlowRefine

    if not isinstance(allow_unsafe_temporal_sampling_override, bool):
        raise ValueError(
            "allow_unsafe_temporal_sampling_override must be true or false."
        )
    if isinstance(model, AuroraFlowRefine):
        def _positive_sampling_steps(value: Any, source: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{source} must be a positive integer, got {value!r}."
                )
            return int(value)

        saved_steps_raw = checkpoint.get("flow_sampling_steps")
        if saved_steps_raw is None:
            if bool(getattr(model, "has_temporal", False)):
                raise ValueError(
                    "Legacy AuroraFlowRefine checkpoint has active temporal Mamba "
                    "but is missing top-level flow_sampling_steps; the exact "
                    "training-time flow integration trajectory cannot be proven. "
                    "Refusing to fall back to runtime configuration because that "
                    "would change the spatial-frame distribution seen by temporal "
                    "Mamba. Use a checkpoint that persisted flow_sampling_steps."
                )
            resolved_sampling_steps = _positive_sampling_steps(
                model.sampling_steps, "configured flow sampling steps"
            )
            sampling_steps_source = "config"
        else:
            resolved_sampling_steps = _positive_sampling_steps(
                saved_steps_raw, "checkpoint flow_sampling_steps"
            )
            sampling_steps_source = "checkpoint"

        if flow_sampling_steps_override is not None:
            override_steps = _positive_sampling_steps(
                flow_sampling_steps_override, "flow_sampling_steps_override"
            )
            if (
                bool(getattr(model, "has_temporal", False))
                and override_steps != resolved_sampling_steps
                and not allow_unsafe_temporal_sampling_override
            ):
                raise ValueError(
                    "A legacy temporal-Mamba checkpoint must use its persisted "
                    f"flow sampling steps ({resolved_sampling_steps}); requested "
                    f"override={override_steps} changes the spatial-frame "
                    "distribution seen by temporal Mamba. Set "
                    "allow_unsafe_temporal_sampling_override=true only for an "
                    "explicit diagnostic ablation."
                )
            if override_steps != resolved_sampling_steps and bool(
                getattr(model, "has_temporal", False)
            ):
                warnings.warn(
                    "UNSAFE diagnostic: overriding temporal-Mamba checkpoint flow "
                    f"sampling steps {resolved_sampling_steps} -> {override_steps}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            resolved_sampling_steps = override_steps
            sampling_steps_source = "explicit_override"
        model.sampling_steps = resolved_sampling_steps
        model.flow_sampling_steps_source = sampling_steps_source
        checkpoint["resolved_flow_sampling_steps"] = resolved_sampling_steps
        checkpoint["flow_sampling_steps_source"] = sampling_steps_source
    elif flow_sampling_steps_override is not None:
        raise ValueError(
            "flow_sampling_steps_override applies only to legacy AuroraFlowRefine "
            "checkpoints."
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} is missing model_state_dict.")
    model.load_state_dict(state, strict=True)
    if production_inference:
        validate_loaded_residual_scaler_contract(model, checkpoint)
    return model, checkpoint
