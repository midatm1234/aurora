"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Unified stochastic residual-refinement package for Aurora rollouts.

Phase 1 is the existing deterministic Aurora forecast/rollout. Phase 2 is an
optional *stochastic residual* correction of each deterministic rollout step,
selected through ``model.refinement.type``:

``none``
    Refinement disabled (deterministic Aurora only).
``flow_matching_unet`` (alias ``flow_matching``)
    The **existing** Aurora x1/data-prediction flow UNet
    (:mod:`finetune.flow_refine`). Its formulation, checkpoints and numerical
    behaviour are preserved exactly.
``flow_matching_conv_unet`` (alias ``flow_matching_conv``)
    Unified packed-field conditional flow matching with the convolutional UNet
    backbone. This is distinct from the legacy ``flow_matching_unet`` wrapper
    and does not share its checkpoint format.
``flow_matching_transformer``
    Same flow-matching target/path, spatial-token Transformer velocity network.
``diffusion_unet``
    Conditional convolutional UNet, DDPM training / DDIM sampling.
``diffusion_transformer``
    Spatial-token Transformer denoiser, DDPM training / DDIM sampling.

The five stochastic refiners are **alternatives**; diffusion and flow matching
are never stacked and flow matching never requires diffusion to run first.

Attribution
-----------
The unified two-phase design, the refiner registry, the normalized target-space
abstraction, the diffusion/flow-matching helpers, the spatial-only Transformer
backbone and the checkpoint-migration approach are adapted from the
``Prithvi-UNet-stochastic_refinement`` reference implementation
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0 licensed
``granitewxc.refinement`` package). That project is a *spatial downscaling*
implementation in which predictors and targets share a timestamp and no
forecast lead time exists. Everything here has been adapted to Aurora's
forecast/rollout structures: forecast lead time is an explicit, separately
embedded conditioning input, residuals are built per rollout step in Aurora's
normalized variable/pressure-level space, and refinement is postprocessing of a
deterministic rollout step by default.

Scientific-time contract
------------------------
Three distinct quantities are kept strictly separate and never share an
embedding or a configuration field:

===========================  ================================================
forecast lead time           physical hours between initialization and valid
                             time (``refinement.conditioning.forecast_lead_time``)
diffusion timestep           index of a Gaussian noising process
                             (``refinement.diffusion.*``)
flow interpolation time      coordinate along the flow path in ``[0, 1]``
                             (``refinement.flow_matching.*``)
===========================  ================================================
"""

from __future__ import annotations

# Importing the concrete refiners populates the registry.
from finetune.refinement import diffusion as _diffusion  # noqa: F401  (side effect)
from finetune.refinement import flow_matching as _flow_matching  # noqa: F401  (side effect)
from finetune.refinement import legacy_flow as _legacy_flow  # noqa: F401  (side effect)
from finetune.refinement.base import (
    ChunkNoiseSource,
    RefinerOutput,
    ResidualRefiner,
    available_refiners,
    build_refiner,
    masked_loss,
    register_refiner,
)
from finetune.refinement.checkpoint import (
    CHECKPOINT_KIND_AURORA,
    CHECKPOINT_KIND_COMBINED,
    CHECKPOINT_KIND_REFINEMENT,
    StateDictReport,
    aurora_state_fingerprint,
    build_refinement_checkpoint,
    load_aurora_state_dict,
    load_refinement_state_dict,
    save_checkpoint_atomic,
    validate_aurora_reference,
)
from finetune.refinement.config import (
    REFINEMENT_TYPES,
    ConfigValidationError,
    PerformanceConfig,
    RefinementConfig,
    config_fingerprint,
    resolve_performance_config,
    resolve_refinement_config,
)
from finetune.refinement.evaluation import (
    compare_raw_and_refined,
    evaluate_packed,
    summarize,
)
from finetune.refinement.losses import BiasLossTerms, compute_auxiliary_losses
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace
from finetune.refinement.two_phase import (
    AuroraTwoPhaseRefiner,
    TwoPhaseStepOutput,
    build_two_phase_refiner,
)

__all__ = [
    "CHECKPOINT_KIND_AURORA",
    "CHECKPOINT_KIND_COMBINED",
    "CHECKPOINT_KIND_REFINEMENT",
    "REFINEMENT_TYPES",
    "AuroraTwoPhaseRefiner",
    "BiasLossTerms",
    "ChannelSpec",
    "ChunkNoiseSource",
    "ConfigValidationError",
    "FieldPacking",
    "NormalizedTargetSpace",
    "PerformanceConfig",
    "RefinementConfig",
    "RefinerOutput",
    "ResidualRefiner",
    "StateDictReport",
    "TwoPhaseStepOutput",
    "aurora_state_fingerprint",
    "available_refiners",
    "build_refinement_checkpoint",
    "build_refiner",
    "build_two_phase_refiner",
    "compare_raw_and_refined",
    "compute_auxiliary_losses",
    "config_fingerprint",
    "evaluate_packed",
    "load_aurora_state_dict",
    "load_refinement_state_dict",
    "masked_loss",
    "register_refiner",
    "resolve_performance_config",
    "resolve_refinement_config",
    "save_checkpoint_atomic",
    "summarize",
    "validate_aurora_reference",
]
