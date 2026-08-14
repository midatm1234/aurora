"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Configuration contract for the unified stochastic residual refinement.
"""

from __future__ import annotations

import warnings

import pytest
import yaml
from finetune.refinement.config import (
    REFINEMENT_TYPES,
    ConfigValidationError,
    resolve_performance_config,
    resolve_refinement_config,
)

EXAMPLE_DIR = "finetune/examples/stochastic_refinement"


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_missing_section_preserves_deterministic_behaviour() -> None:
    cfg = resolve_refinement_config({"model": {}})
    assert cfg.type == "none"
    assert cfg.enabled is False
    assert cfg.is_active is False
    assert cfg.backend == "none"


def test_none_config_object() -> None:
    assert resolve_refinement_config(None).backend == "none"


def test_legacy_flow_refine_keys_resolve_to_flow_matching_unet() -> None:
    cfg = resolve_refinement_config(
        {
            "model": {
                "flow_refine_enabled": True,
                "flow_refine_hidden": 16,
                "flow_refine_time_dim": 128,
                "flow_refine_sampling_steps": 1,
                "flow_refine_residual_zscore": True,
                "flow_refine_lead_time_cond": True,
            },
            "training": {"flow_refine_autoregressive_feedback": True},
        }
    )
    assert cfg.type == "flow_matching_unet"
    assert cfg.backend == "legacy"
    assert cfg.from_legacy_keys is True
    assert cfg.unet.hidden_channels == 16
    assert cfg.unet.time_embedding_dim == 128
    assert cfg.flow_matching.integration_steps == 1
    assert cfg.flow_matching.residual_zscore is True
    assert cfg.conditioning.forecast_lead_time is True
    # The legacy inference path applies refinement inline; record it faithfully.
    assert cfg.feedback_to_rollout is True


def test_legacy_autoregressive_feedback_requires_a_boolean() -> None:
    with pytest.raises(
        ConfigValidationError,
        match="training.flow_refine_autoregressive_feedback must be a boolean",
    ):
        resolve_refinement_config(
            {
                "model": {"flow_refine_enabled": True},
                "training": {"flow_refine_autoregressive_feedback": "false"},
            }
        )


def test_flow_matching_alias() -> None:
    aliased = resolve_refinement_config({"model": {"refinement": {"type": "flow_matching"}}})
    explicit = resolve_refinement_config({"model": {"refinement": {"type": "flow_matching_unet"}}})
    assert aliased.type == "flow_matching_unet" == explicit.type
    assert aliased.to_dict() == explicit.to_dict()


@pytest.mark.parametrize("refinement_type", REFINEMENT_TYPES)
def test_every_supported_type_resolves(refinement_type: str) -> None:
    cfg = resolve_refinement_config({"model": {"refinement": {"type": refinement_type}}})
    assert cfg.type == refinement_type
    assert cfg.is_active == (refinement_type != "none")


def test_disabled_flag_wins() -> None:
    cfg = resolve_refinement_config(
        {"model": {"refinement": {"type": "diffusion_unet", "enabled": False}}}
    )
    assert cfg.is_active is False
    assert cfg.type == "none"


def test_type_none_disables() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = resolve_refinement_config(
            {"model": {"refinement": {"type": "none", "enabled": True}}}
        )
    assert cfg.is_active is False


def test_default_is_postprocessing_not_feedback() -> None:
    cfg = resolve_refinement_config({"model": {"refinement": {"type": "diffusion_unet"}}})
    assert cfg.feedback_to_rollout is False
    assert cfg.freeze_aurora is True
    assert cfg.joint_finetuning is False


def test_feedback_to_rollout_warns() -> None:
    with pytest.warns(RuntimeWarning, match="EXPERIMENTAL"):
        cfg = resolve_refinement_config(
            {"model": {"refinement": {"type": "diffusion_unet", "feedback_to_rollout": True}}}
        )
    assert cfg.feedback_to_rollout is True


# --------------------------------------------------------------------------
# Validation errors
# --------------------------------------------------------------------------


def test_invalid_refiner_type() -> None:
    with pytest.raises(ConfigValidationError, match="Unknown refinement.type"):
        resolve_refinement_config({"model": {"refinement": {"type": "gan"}}})


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="Unknown key"):
        resolve_refinement_config({"model": {"refinement": {"typo": 1}}})


@pytest.mark.parametrize(
    "diffusion,message",
    [
        ({"schedule": "quadratic"}, "schedule"),
        ({"prediction_type": "score"}, "prediction_type"),
        ({"sampler": "pndm"}, "sampler"),
        ({"eta": 2.0}, "eta"),
        ({"inference_steps": 5000}, "cannot exceed"),
        ({"schedule": "linear", "beta_start": 0.5, "beta_end": 0.1}, "beta_end"),
        ({"sampler": "ddpm", "eta": 0.0}, "requires eta=1.0"),
    ],
)
def test_invalid_diffusion_settings(diffusion, message) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        resolve_refinement_config(
            {"model": {"refinement": {"type": "diffusion_unet", "diffusion": diffusion}}}
        )


@pytest.mark.parametrize(
    "flow,message",
    [
        ({"solver": "rk4"}, "solver"),
        ({"source_distribution": "uniform"}, "source_distribution"),
        ({"time_sampling": "beta"}, "time_sampling"),
        ({"integration_steps": 0}, "integration_steps"),
        ({"interpolation_path": "probability"}, "interpolation_path"),
    ],
)
def test_invalid_flow_settings(flow, message) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        resolve_refinement_config(
            {"model": {"refinement": {"type": "flow_matching_transformer", "flow_matching": flow}}}
        )


@pytest.mark.parametrize(
    "transformer,message",
    [
        ({"embedding_dim": 30, "num_heads": 8}, "divisible"),
        ({"embedding_dim": 12, "num_heads": 4}, "head dimension"),
        ({"patch_size": [1, 2, 3]}, "patch_size"),
        ({"patch_size": 0}, "patch_size"),
        ({"patch_size": [4, 4], "patch_height": 4}, "not both"),
        ({"positional_encoding": "rope"}, "positional_encoding"),
        ({"attention_mode": "linear"}, "attention_mode"),
        ({"attention_mode": "windowed_2d", "window_size": [0, 4]}, "window_size"),
        (
            {"attention_mode": "windowed_2d", "window_size": [1, 1], "shifted_windows": True},
            "shifted_windows",
        ),
        ({"optimized_attention": "flash3"}, "optimized_attention"),
        (
            {"embedding_dim": 18, "num_heads": 1, "positional_encoding": "sincos_2d"},
            "divisible by 4",
        ),
    ],
)
def test_invalid_transformer_geometry(transformer, message) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_transformer",
                        "transformer": transformer,
                    }
                }
            }
        )


def test_invalid_unet_geometry() -> None:
    with pytest.raises(ConfigValidationError, match="attention_heads"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "unet": {
                            "hidden_channels": 9,
                            "num_levels": 1,
                            "attention_heads": 4,
                        },
                    }
                }
            }
        )


def test_joint_finetuning_conflicts_with_freeze() -> None:
    with pytest.raises(ConfigValidationError, match="joint_finetuning"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "joint_finetuning": True,
                        "freeze_aurora": True,
                    }
                }
            }
        )


def test_conditioning_cannot_be_empty() -> None:
    with pytest.raises(ConfigValidationError, match="at least one"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "conditioning": {
                            "aurora_rollout": False,
                            "aurora_input_state": False,
                            "aurora_features": False,
                            "static_fields": False,
                            "masks": False,
                        },
                    }
                }
            }
        )


def test_coordinate_conditioning_flags_default_off_and_round_trip() -> None:
    default = resolve_refinement_config({"model": {"refinement": {"type": "diffusion_unet"}}})
    configured = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "type": "diffusion_unet",
                    "conditioning": {"latitude": True, "longitude": True},
                }
            }
        }
    )
    round_tripped = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "type": "diffusion_unet",
                    "conditioning": configured.conditioning.to_dict(),
                }
            }
        }
    )

    assert default.conditioning.latitude is False
    assert default.conditioning.longitude is False
    assert configured.conditioning.latitude is True
    assert configured.conditioning.longitude is True
    assert round_tripped.conditioning == configured.conditioning


def test_residual_amplitude_safeguard_defaults_off_and_round_trips() -> None:
    default = resolve_refinement_config({"model": {"refinement": {"type": "diffusion_unet"}}})
    configured = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "type": "diffusion_unet",
                    "target_space": {
                        "residual_scaling": "per_channel",
                        "residual_scaling_center": True,
                        "residual_clip_standard_deviations": 4.0,
                    },
                }
            }
        }
    )
    round_tripped = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "type": "diffusion_unet",
                    "target_space": configured.target_space.to_dict(),
                }
            }
        }
    )

    assert default.target_space.residual_clip_standard_deviations == 0.0
    assert configured.target_space.residual_clip_standard_deviations == 4.0
    assert round_tripped.target_space == configured.target_space


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_residual_amplitude_safeguard_is_rejected(value: float) -> None:
    with pytest.raises(ConfigValidationError, match="residual_clip_standard_deviations"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "target_space": {
                            "residual_scaling": "per_channel",
                            "residual_clip_standard_deviations": value,
                        },
                    }
                }
            }
        )


def test_residual_amplitude_safeguard_requires_active_scaling() -> None:
    with pytest.raises(ConfigValidationError, match="requires active residual_scaling"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "target_space": {
                            "residual_scaling": "none",
                            "residual_clip_standard_deviations": 4.0,
                        },
                    }
                }
            }
        )


def test_legacy_head_rejects_unimplemented_residual_amplitude_safeguard() -> None:
    with pytest.raises(ConfigValidationError, match="only by unified"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "flow_matching_unet",
                        "target_space": {
                            "residual_scaling": "per_channel",
                            "residual_clip_standard_deviations": 4.0,
                        },
                    }
                }
            }
        )


def test_coordinates_supplement_instead_of_replace_state_conditioning() -> None:
    with pytest.raises(ConfigValidationError, match="at least one"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "conditioning": {
                            "aurora_rollout": False,
                            "aurora_input_state": False,
                            "aurora_features": False,
                            "static_fields": False,
                            "masks": False,
                            "latitude": True,
                            "longitude": True,
                        },
                    }
                }
            }
        )


def test_unknown_coordinate_conditioning_key_is_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="latitude_degrees"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "conditioning": {"latitude_degrees": True},
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "name",
    [
        "aurora_NO2_finetune_US-WEST_3day_lead_config.yaml",
        "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_config.yaml",
        "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml",
        "aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml",
    ],
)
def test_unified_no2_configs_enable_geophysical_coordinates(name: str) -> None:
    with open(f"finetune/{name}") as handle:
        raw = yaml.safe_load(handle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        config = resolve_refinement_config(raw)

    assert config.backend == "unified"
    assert config.conditioning.latitude is True
    assert config.conditioning.longitude is True
    assert config.target_space.residual_clip_standard_deviations == 4.0
    assert config.loss.extreme_tail == "upper"


def test_legacy_type_rejects_new_auxiliary_losses() -> None:
    with pytest.raises(ConfigValidationError, match="auxiliary weights"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "flow_matching_unet",
                        "loss": {"bias_weight": 0.5},
                    }
                }
            }
        )


def test_legacy_type_rejects_other_interpolation_paths() -> None:
    with pytest.raises(ConfigValidationError, match="existing_aurora"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "flow_matching_unet",
                        "flow_matching": {"interpolation_path": "rectified_flow"},
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "flow_override",
    [
        {"solver": "heun"},
        {"stochastic_initialization": False},
        {"time_sampling": "uniform"},
        {"logit_normal_mean": 0.0},
        {"logit_normal_std": 0.8},
        {"deterministic_training_steps": 8},
    ],
)
def test_legacy_type_rejects_silently_unsupported_flow_options(
    flow_override: dict[str, object],
) -> None:
    with pytest.raises(ConfigValidationError, match="does not implement"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "flow_matching_unet",
                        "flow_matching": flow_override,
                    }
                }
            }
        )


def test_auxiliary_losses_default_to_zero() -> None:
    cfg = resolve_refinement_config({"model": {"refinement": {"type": "diffusion_unet"}}})
    assert cfg.loss.has_auxiliary_terms is False
    assert cfg.loss.reconstruction_weight == 0.0
    assert cfg.loss.bias_weight == 0.0
    assert cfg.loss.gradient_weight == 0.0


def test_extreme_tail_defaults_to_legacy_both_and_accepts_upper() -> None:
    default = resolve_refinement_config({"model": {"refinement": {"type": "diffusion_unet"}}})
    upper = resolve_refinement_config(
        {
            "model": {
                "refinement": {
                    "type": "diffusion_unet",
                    "loss": {"extreme_tail": "upper"},
                }
            }
        }
    )

    assert default.loss.extreme_tail == "both"
    assert default.loss.to_dict()["extreme_tail"] == "both"
    assert upper.loss.extreme_tail == "upper"


def test_invalid_extreme_tail_is_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="extreme_tail"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "loss": {"extreme_tail": "lower"},
                    }
                }
            }
        )


# --------------------------------------------------------------------------
# Performance section
# --------------------------------------------------------------------------


def test_performance_defaults_are_neutral() -> None:
    perf = resolve_performance_config({})
    assert perf.profile is False
    assert perf.precision.mode == "fp32"
    assert perf.precision.allow_tf32 is False
    assert perf.compile.enabled is False
    assert perf.rollout_cache.enabled is False


def test_cache_requires_frozen_aurora() -> None:
    with pytest.raises(ConfigValidationError, match="freeze_aurora"):
        resolve_performance_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "freeze_aurora": False,
                    }
                },
                "performance": {"rollout_cache": {"enabled": True, "path": "/tmp/cache"}},
            }
        )


def test_cache_incompatible_with_joint_finetuning() -> None:
    with pytest.raises(ConfigValidationError, match="joint_finetuning"):
        resolve_performance_config(
            {
                "model": {
                    "refinement": {
                        "type": "diffusion_unet",
                        "joint_finetuning": True,
                    }
                },
                "performance": {"conditioning_cache": {"enabled": True, "path": "/tmp/cache"}},
            }
        )


def test_cache_requires_path() -> None:
    with pytest.raises(ConfigValidationError, match="no 'path'"):
        resolve_performance_config({"performance": {"rollout_cache": {"enabled": True}}})


def test_precision_change_warns() -> None:
    with pytest.warns(RuntimeWarning, match="parity"):
        resolve_performance_config({"performance": {"precision": {"mode": "bf16"}}})


# --------------------------------------------------------------------------
# Example configurations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("aurora_O3_global_rollout_no_refinement.yaml", "none"),
        ("aurora_O3_global_flow_matching_unet.yaml", "flow_matching_unet"),
        ("aurora_O3_global_flow_matching_conv_unet.yaml", "flow_matching_conv_unet"),
        ("aurora_O3_global_flow_matching_transformer.yaml", "flow_matching_transformer"),
        ("aurora_O3_global_diffusion_unet.yaml", "diffusion_unet"),
        ("aurora_O3_global_diffusion_transformer.yaml", "diffusion_transformer"),
    ],
)
def test_example_configurations(name: str, expected: str) -> None:
    with open(f"{EXAMPLE_DIR}/{name}") as handle:
        raw = yaml.safe_load(handle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = resolve_refinement_config(raw)
        resolve_performance_config(raw)
    assert cfg.type == expected
    # Refinement is postprocessing by default in every shipped example.
    assert cfg.feedback_to_rollout is False


_SAFE_UNIFIED_EXAMPLES = (
    "aurora_O3_global_flow_matching_conv_unet.yaml",
    "aurora_O3_global_flow_matching_transformer.yaml",
    "aurora_O3_global_diffusion_unet.yaml",
    "aurora_O3_global_diffusion_transformer.yaml",
)


@pytest.mark.parametrize("name", _SAFE_UNIFIED_EXAMPLES)
def test_unified_examples_ship_a_safe_correction_product(name: str) -> None:
    path = f"{EXAMPLE_DIR}/{name}"
    with open(path) as handle:
        text = handle.read()
    raw = yaml.safe_load(text)
    cfg = resolve_refinement_config(raw)

    # The equations are intentionally present beside the machine-readable flag:
    # this prevents an ambiguous `residual` label from reversing the correction.
    assert "correction_target = CAMS - Aurora" in text
    assert "refined_forecast = Aurora + predicted_correction" in text
    assert cfg.train_on_residual is True
    assert cfg.feedback_to_rollout is False

    # A deterministic conditional mean is the default forecast product. A
    # stochastic product must be an actual ensemble, never one arbitrary draw.
    assert cfg.deterministic_inference is True
    assert cfg.ensemble_size == 1
    assert cfg.deterministic_inference or cfg.ensemble_size >= 2

    assert cfg.target_space.residual_scaling == "per_channel"
    assert cfg.target_space.residual_scaling_center is True
    assert cfg.loss.deterministic_weight > 0.0
    assert cfg.loss.aux_on_deterministic is True
    assert cfg.loss.bias_weight > 0.0
    assert cfg.loss.degradation_weight > 0.0

    model = raw["model"]
    training = raw["training"]
    assert model["mamba_temporal_enabled"] is False
    assert training["mamba_temporal_weight"] == 0.0
    assert training["validation_refinement_ensemble_size"] == 1
    assert training["validation_source"] == "train_tail"
    assert training["checkpoint_metric"] == "mean_physical_rmse_ratio"
    assert training["require_all_physical_channels_improve"] is True
    assert training["require_refinement_improvement"] is True
    assert training["resume_training"] is False
    assert training["resume_from"] == ""

    if cfg.type.startswith("diffusion_"):
        backbone_name = "transformer" if cfg.type.endswith("transformer") else "unet"
        zero_initialized = bool(raw["model"]["refinement"][backbone_name]["zero_init_output"])
        # Zero model output means x0=0 only for clean-sample prediction. With
        # epsilon prediction, it is amplified Gaussian latent, not identity.
        assert zero_initialized is True
        assert cfg.diffusion.prediction_type == "sample"
        assert not (zero_initialized and cfg.diffusion.prediction_type == "epsilon")


def test_unified_head_warns_when_legacy_auxiliary_block_is_enabled() -> None:
    raw = {
        "model": {"refinement": {"enabled": True, "type": "diffusion_unet"}},
        "training": {"flow_aux_loss": {"enabled": True, "bias_weight": 1.0}},
    }
    with pytest.warns(RuntimeWarning, match="training.flow_aux_loss.*ignored"):
        config = resolve_refinement_config(raw)
    assert config.backend == "unified"


def test_existing_aurora_flow_warns_when_ode_solver_would_be_ignored() -> None:
    raw = {
        "model": {
            "refinement": {
                "enabled": True,
                "type": "flow_matching_conv_unet",
                "flow_matching": {
                    "interpolation_path": "existing_aurora",
                    "solver": "heun",
                },
            }
        }
    }
    with pytest.warns(RuntimeWarning, match="solver is ignored"):
        config = resolve_refinement_config(raw)
    assert config.flow_matching.solver == "heun"
