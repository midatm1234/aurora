"""Aurora fine-tuning utilities package.

Utilities are loaded lazily so lightweight coordinate/preparation helpers can
be imported without constructing the full Aurora/timm model stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "VariableSpec",
    "ResolvedVariableSpecs",
    "load_config",
    "set_seed",
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
    "load_checkpoint_if_available",
    "validate_checkpoint_longitude",
    "write_training_history",
    "write_run_manifest",
    "save_predictions",
    "maybe_wrap_conv_refine",
    "maybe_wrap_flow_refine",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from finetune import aurora_finetune_utils

    return getattr(aurora_finetune_utils, name)
