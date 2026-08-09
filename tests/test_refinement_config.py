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


def test_auxiliary_losses_default_to_zero() -> None:
    cfg = resolve_refinement_config({"model": {"refinement": {"type": "diffusion_unet"}}})
    assert cfg.loss.has_auxiliary_terms is False
    assert cfg.loss.reconstruction_weight == 0.0
    assert cfg.loss.bias_weight == 0.0
    assert cfg.loss.gradient_weight == 0.0


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
