"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Validated configuration schema for Aurora stochastic residual refinement.

Backward-compatibility contract
-------------------------------
* A configuration **without** a ``model.refinement`` section keeps the existing
  behaviour exactly. When the legacy keys ``model.flow_refine_enabled`` /
  ``model.conv_refine_enabled`` are present they are translated into the new
  schema, so every existing YAML file keeps working unchanged.
* ``refinement.type: flow_matching`` is a permanent alias for
  ``flow_matching_unet``, which *is* the existing
  :class:`finetune.flow_refine.AuroraFlowRefine` architecture.
* ``refinement.enabled: false`` and ``refinement.type: none`` both disable
  Phase 2.
* Unknown refiner names, invalid Transformer geometry, invalid patch/window
  settings, invalid diffusion schedules, invalid prediction parameterizations,
  invalid flow solvers and incompatible option pairs raise
  :class:`ConfigValidationError` instead of being silently coerced.

Scientific settings (``model.refinement``) and workflow settings
(``performance``) are kept in separate sections. Nothing in ``performance`` may
change architecture, loss, ensemble size, diffusion steps, flow steps, sampler,
solver or validation data.

Adapted from ``granitewxc.refinement.config`` in the Prithvi stochastic
residual-refinement reference implementation
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), extended with Aurora's
forecast lead-time conditioning, rollout-feedback switch and bias-aware loss
section.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "CORRECTION_CONVENTION",
    "REFINEMENT_TYPES",
    "BiasLossConfig",
    "ConditioningConfig",
    "ConfigValidationError",
    "DiffusionConfig",
    "FlowMatchingConfig",
    "PerformanceConfig",
    "RefinementConfig",
    "TargetSpaceConfig",
    "TransformerConfig",
    "UNetRefinerConfig",
    "config_fingerprint",
    "resolve_performance_config",
    "resolve_refinement_config",
]


class ConfigValidationError(ValueError):
    """Raised for any invalid or mutually incompatible refinement setting."""


_EPSILON_RESIDUAL_WARNING_LOCK = threading.Lock()
_EPSILON_RESIDUAL_WARNING_EMITTED = False


def _warn_epsilon_residual_prediction_once() -> None:
    """Emit the explicit-epsilon scientific warning once per Python process."""
    global _EPSILON_RESIDUAL_WARNING_EMITTED
    with _EPSILON_RESIDUAL_WARNING_LOCK:
        if _EPSILON_RESIDUAL_WARNING_EMITTED:
            return
        warnings.warn(
            "refinement.diffusion.prediction_type='epsilon' is a poor fit for "
            "conditional residual correction because clean-sample conversion can "
            "amplify prediction errors. Prefer prediction_type='sample' (or "
            "'velocity') for newly tuned residual refiners.",
            RuntimeWarning,
            stacklevel=3,
        )
        _EPSILON_RESIDUAL_WARNING_EMITTED = True


#: Machine-readable sign/application contract used in configs and checkpoints.
CORRECTION_CONVENTION = "cams_minus_aurora_add"
_CORRECTION_CONVENTIONS = (CORRECTION_CONVENTION,)


#: Every supported Phase-2 refinement type. ``none`` disables Phase 2.
REFINEMENT_TYPES: tuple[str, ...] = (
    "none",
    "flow_matching_unet",
    "flow_matching_transformer",
    "flow_matching_conv_unet",
    "diffusion_unet",
    "diffusion_transformer",
)

#: ``flow_matching`` is a *permanent* backward-compatible alias for the existing
#: Aurora rectified-flow UNet.
_REFINEMENT_ALIASES: dict[str, str] = {
    "": "none",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "deterministic": "none",
    "flow_matching": "flow_matching_unet",
    "flow": "flow_matching_unet",
    "flow_matching_unet": "flow_matching_unet",
    "flow_matching_transformer": "flow_matching_transformer",
    # The unified (Phase-2) flow-matching objective on the convolutional
    # backbone. Distinct from ``flow_matching_unet``, which keeps routing to the
    # original Aurora wrapper so existing checkpoints and configurations are
    # untouched.
    "flow_matching_conv": "flow_matching_conv_unet",
    "flow_matching_conv_unet": "flow_matching_conv_unet",
    "diffusion": "diffusion_unet",
    "diffusion_unet": "diffusion_unet",
    "diffusion_transformer": "diffusion_transformer",
}

_LEGACY_TYPES = frozenset({"flow_matching_unet"})
_DIFFUSION_TYPES = frozenset({"diffusion_unet", "diffusion_transformer"})
_FLOW_TYPES = frozenset(
    {"flow_matching_unet", "flow_matching_transformer", "flow_matching_conv_unet"}
)
_TRANSFORMER_TYPES = frozenset({"diffusion_transformer", "flow_matching_transformer"})

_PREDICTION_TYPES = ("epsilon", "velocity", "sample")
_SCHEDULES = ("cosine", "linear", "scaled_linear")
_SAMPLERS = ("ddim", "ddpm")
_SOLVERS = ("euler", "heun", "midpoint")
_SOURCE_DISTRIBUTIONS = ("gaussian",)
_INTERPOLATION_PATHS = ("existing_aurora", "rectified_flow")
_TIME_SAMPLINGS = ("uniform", "logit_normal")
_POSITIONAL_ENCODINGS = ("latlon_2d", "learned_2d", "sincos_2d")
_ATTENTION_MODES = ("global_2d", "windowed_2d")
_ATTENTION_IMPLEMENTATIONS = ("auto", "sdpa", "math")
_PRECISION_MODES = ("fp32", "bf16", "fp16")
_GENERATIVE_LOSSES = ("mse", "l1", "huber")
_RESIDUAL_SPACES = ("normalized",)
_RESIDUAL_SCALINGS = ("auto", "none", "global", "per_channel")
_SNR_WEIGHTINGS = ("auto", "none", "min_snr", "snr", "truncated_snr")
_DETERMINISTIC_ESTIMATORS = ("auto", "ode", "posterior_mean")
_TIMESTEP_DISTRIBUTIONS = ("auto", "uniform", "low_noise", "high_noise")


# ---------------------------------------------------------------------------
# Primitive coercion helpers
# ---------------------------------------------------------------------------


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce YAML/namespace-ish objects into a plain ``dict``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("__")}
    raise ConfigValidationError(
        f"Expected a mapping for a configuration section, got {type(value).__name__}"
    )


def _reject_unknown(section: str, data: Mapping[str, Any], known: Sequence[str]) -> None:
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigValidationError(
            f"Unknown key(s) {unknown} in '{section}'. Supported keys: {sorted(known)}."
        )


def _as_bool(section: str, key: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "on", "1"}:
            return True
        if text in {"false", "no", "off", "0"}:
            return False
    raise ConfigValidationError(f"{section}.{key} must be a boolean, got {value!r}")


def _as_int(section: str, key: str, value: Any, default: int, *, minimum: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigValidationError(f"{section}.{key} must be an integer, got {value!r}")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{section}.{key} must be an integer, got {value!r}") from exc
    if out < minimum:
        raise ConfigValidationError(f"{section}.{key} must be >= {minimum}, got {out}")
    return out


def _as_float(
    section: str,
    key: str,
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{section}.{key} must be a number, got {value!r}") from exc
    if out != out:  # NaN
        raise ConfigValidationError(f"{section}.{key} must be finite, got {value!r}")
    if minimum is not None and out < minimum:
        raise ConfigValidationError(f"{section}.{key} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise ConfigValidationError(f"{section}.{key} must be <= {maximum}, got {out}")
    return out


def _as_choice(section: str, key: str, value: Any, default: str, choices: Sequence[str]) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in choices:
        raise ConfigValidationError(
            f"{section}.{key} must be one of {list(choices)}, got {value!r}"
        )
    return text


def _as_pair(section: str, key: str, value: Any, default: tuple[int, int]) -> tuple[int, int]:
    """Accept ``n`` or ``[h, w]`` and return a validated ``(h, w)`` pair."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ConfigValidationError(
                f"{section}.{key} must be an int or a [height, width] pair, got {value!r}"
            )
        return (
            _as_int(section, f"{key}[0]", value[0], default[0]),
            _as_int(section, f"{key}[1]", value[1], default[1]),
        )
    shared = _as_int(section, key, value, default[0])
    return (shared, shared)


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpaceConfig:
    """How the residual target space is derived.

    ``use_existing_normalization`` reuses the Aurora/CAMS normalization already
    used by the supervised loss (``compute_target_normalization_stats``), which
    is what makes the residual construction consistent with the existing
    flow-matching implementation.
    """

    use_existing_normalization: bool = True
    residual_space: str = "normalized"
    #: Standardization of the residual *before* the generative process. The
    #: residual lives in Aurora's field-normalized space, but its scale can still
    #: differ substantially by variable and level. The backward-compatible
    #: default remains ``none``; new configurations opt in explicitly with
    #: ``per_channel`` or ``auto``. The legacy flow wrapper remains untouched.
    residual_scaling: str = "none"
    residual_scaling_center: bool = False
    residual_scaling_momentum: float = 0.05
    residual_scaling_warmup_batches: int = 32
    residual_scaling_target_std: float = 1.0

    _KEYS = (
        "use_existing_normalization",
        "residual_space",
        "residual_scaling",
        "residual_scaling_center",
        "residual_scaling_momentum",
        "residual_scaling_warmup_batches",
        "residual_scaling_target_std",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> TargetSpaceConfig:
        raw = _as_mapping(data)
        sec = "refinement.target_space"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        use_existing = _as_bool(
            sec,
            "use_existing_normalization",
            raw.get("use_existing_normalization"),
            d.use_existing_normalization,
        )
        if not use_existing:
            raise ConfigValidationError(
                f"{sec}.use_existing_normalization=false is not supported: the residual "
                "must be built in the same normalized target space that Aurora's "
                "supervised loss and the existing flow-matching refiner use."
            )
        return cls(
            use_existing_normalization=use_existing,
            residual_space=_as_choice(
                sec,
                "residual_space",
                raw.get("residual_space"),
                d.residual_space,
                _RESIDUAL_SPACES,
            ),
            residual_scaling=_as_choice(
                sec,
                "residual_scaling",
                raw.get("residual_scaling"),
                d.residual_scaling,
                _RESIDUAL_SCALINGS,
            ),
            residual_scaling_center=_as_bool(
                sec,
                "residual_scaling_center",
                raw.get("residual_scaling_center"),
                d.residual_scaling_center,
            ),
            residual_scaling_momentum=_as_float(
                sec,
                "residual_scaling_momentum",
                raw.get("residual_scaling_momentum"),
                d.residual_scaling_momentum,
                minimum=0.0,
                maximum=1.0,
            ),
            residual_scaling_warmup_batches=_as_int(
                sec,
                "residual_scaling_warmup_batches",
                raw.get("residual_scaling_warmup_batches"),
                d.residual_scaling_warmup_batches,
                minimum=0,
            ),
            residual_scaling_target_std=_as_float(
                sec,
                "residual_scaling_target_std",
                raw.get("residual_scaling_target_std"),
                d.residual_scaling_target_std,
                minimum=1.0e-6,
            ),
        )

    def resolved_residual_scaling(self) -> str:
        """``auto`` resolved to a concrete mode."""
        return "per_channel" if self.residual_scaling == "auto" else self.residual_scaling

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class ConditioningConfig:
    """Which fields are concatenated into the Phase-2 spatial conditioning."""

    aurora_rollout: bool = True
    aurora_input_state: bool = False
    aurora_features: bool = False
    static_fields: bool = True
    masks: bool = True
    forecast_lead_time: bool = True

    _KEYS = (
        "aurora_rollout",
        "aurora_input_state",
        "aurora_features",
        "static_fields",
        "masks",
        "forecast_lead_time",
    )
    _SPATIAL_KEYS = (
        "aurora_rollout",
        "aurora_input_state",
        "aurora_features",
        "static_fields",
        "masks",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> ConditioningConfig:
        raw = _as_mapping(data)
        sec = "refinement.conditioning"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        out = cls(**{key: _as_bool(sec, key, raw.get(key), getattr(d, key)) for key in cls._KEYS})
        if not any(getattr(out, key) for key in cls._SPATIAL_KEYS):
            raise ConfigValidationError(
                f"{sec} disables every spatial conditioning input; the refiner would "
                "have nothing to condition on. Enable at least one of "
                f"{list(cls._SPATIAL_KEYS)}."
            )
        if out.aurora_features:
            raise ConfigValidationError(
                f"{sec}.aurora_features is not implemented: Aurora does not expose a "
                "stable frozen spatial feature map through the rollout API. Use "
                "aurora_rollout / aurora_input_state instead."
            )
        return out

    def to_dict(self) -> dict[str, bool]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class LossConfig:
    """Generative loss plus optional bias-aware auxiliary terms.

    All auxiliary weights default to ``0.0`` so enabling the section never
    changes an existing run's objective.
    """

    generative: str = "mse"
    reconstruction_weight: float = 0.0
    bias_weight: float = 0.0
    gradient_weight: float = 0.0
    pattern_correlation_weight: float = 0.0
    #: Direct supervision of the *exact quantity inference computes*. The
    #: generative objective trains a denoiser/velocity field; it does not train
    #: the deterministic estimate the forecast actually uses. This optional
    #: term closes that objective/deployment gap and must be tuned by case.
    deterministic_weight: float = 0.0
    #: L1 on the residual error. Targets MAE directly (MSE targets RMSE).
    mae_weight: float = 0.0
    #: Quantile-weighted squared error emphasising the tails of the refined
    #: field, plus a spatial max/min magnitude match.
    extreme_weight: float = 0.0
    extreme_quantile: float = 0.95
    extreme_intensity: float = 4.0
    peak_weight: float = 0.0
    #: Sorted-value (1-D Wasserstein-2) distance between the refined and true
    #: field distributions. Matches the full PDF, not just the tails.
    quantile_weight: float = 0.0
    #: Spatial standard-deviation match. Counteracts oversmoothing.
    variance_weight: float = 0.0
    #: Radially averaged log power-spectrum match. Preserves the scale
    #: distribution of the refined field.
    spectral_weight: float = 0.0
    #: Hinge that is zero while the correction is no worse than leaving the
    #: rollout unchanged, and positive where it degrades a point.
    degradation_weight: float = 0.0
    #: L2 shrinkage on the predicted residual magnitude.
    magnitude_weight: float = 0.0
    #: Evaluate the structural terms (pattern correlation, extreme, peak,
    #: quantile, variance, spectral) on the **deterministic** residual estimate
    #: rather than on the generative estimate drawn at a random noise level.
    #: The deterministic estimate is what inference emits, so this is the field
    #: whose spread, tails and spectrum actually reach the forecast; scoring the
    #: noisy training estimate instead optimises a quantity nobody consumes.
    #: Requires ``deterministic_weight > 0`` (which is what computes the
    #: estimate); ignored otherwise.
    aux_on_deterministic: bool = True
    area_weighted: bool = True
    separate_by_variable: bool = True
    separate_by_level: bool = True
    separate_by_lead_time: bool = True

    _KEYS = (
        "generative",
        "reconstruction_weight",
        "bias_weight",
        "gradient_weight",
        "pattern_correlation_weight",
        "deterministic_weight",
        "mae_weight",
        "extreme_weight",
        "extreme_quantile",
        "extreme_intensity",
        "peak_weight",
        "quantile_weight",
        "variance_weight",
        "spectral_weight",
        "degradation_weight",
        "magnitude_weight",
        "aux_on_deterministic",
        "area_weighted",
        "separate_by_variable",
        "separate_by_level",
        "separate_by_lead_time",
    )

    #: weights that are plain non-negative scalars multiplying a loss term.
    _WEIGHT_KEYS = (
        "reconstruction_weight",
        "bias_weight",
        "gradient_weight",
        "pattern_correlation_weight",
        "deterministic_weight",
        "mae_weight",
        "extreme_weight",
        "peak_weight",
        "quantile_weight",
        "variance_weight",
        "spectral_weight",
        "degradation_weight",
        "magnitude_weight",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> LossConfig:
        raw = _as_mapping(data)
        sec = "refinement.loss"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        weights = {
            key: _as_float(sec, key, raw.get(key), getattr(d, key), minimum=0.0)
            for key in cls._WEIGHT_KEYS
        }
        out = cls(
            generative=_as_choice(
                sec, "generative", raw.get("generative"), d.generative, _GENERATIVE_LOSSES
            ),
            extreme_quantile=_as_float(
                sec,
                "extreme_quantile",
                raw.get("extreme_quantile"),
                d.extreme_quantile,
                minimum=0.0,
                maximum=1.0,
            ),
            extreme_intensity=_as_float(
                sec,
                "extreme_intensity",
                raw.get("extreme_intensity"),
                d.extreme_intensity,
                minimum=0.0,
            ),
            area_weighted=_as_bool(sec, "area_weighted", raw.get("area_weighted"), d.area_weighted),
            aux_on_deterministic=_as_bool(
                sec,
                "aux_on_deterministic",
                raw.get("aux_on_deterministic"),
                d.aux_on_deterministic,
            ),
            separate_by_variable=_as_bool(
                sec,
                "separate_by_variable",
                raw.get("separate_by_variable"),
                d.separate_by_variable,
            ),
            separate_by_level=_as_bool(
                sec, "separate_by_level", raw.get("separate_by_level"), d.separate_by_level
            ),
            separate_by_lead_time=_as_bool(
                sec,
                "separate_by_lead_time",
                raw.get("separate_by_lead_time"),
                d.separate_by_lead_time,
            ),
            **weights,
        )
        if out.extreme_weight > 0.0 and not 0.0 < out.extreme_quantile < 1.0:
            raise ConfigValidationError(
                f"{sec}.extreme_quantile must be strictly between 0 and 1 when "
                "extreme_weight > 0."
            )
        return out

    @property
    def has_auxiliary_terms(self) -> bool:
        """Whether any term beyond the pure generative objective is enabled."""
        return any(getattr(self, key) > 0.0 for key in self._WEIGHT_KEYS)

    @property
    def needs_rollout(self) -> bool:
        """Terms that must be evaluated on the refined *field*, not the residual."""
        return (
            self.pattern_correlation_weight > 0.0
            or self.extreme_weight > 0.0
            or self.peak_weight > 0.0
            or self.quantile_weight > 0.0
            or self.variance_weight > 0.0
            or self.spectral_weight > 0.0
        )

    @property
    def needs_deterministic_estimate(self) -> bool:
        """Whether an extra deterministic forward pass is required."""
        return self.deterministic_weight > 0.0 or (
            self.aux_on_deterministic and self.needs_rollout
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Conditional flow-matching settings.

    ``interpolation_path``:

    ``existing_aurora``
        The formulation implemented on ``aurora_finetune_flow_matching``: the
        network predicts the **clean standardised residual** ``r`` (the ``x1``
        / data parameterisation) from ``x_t = (1-t)*x0 + t*r``, trained with a
        masked MSE against ``r`` and sampled with the existing DDIM-style
        update. This is the default so legacy configurations never change.
    ``rectified_flow``
        The velocity parameterisation used by the Prithvi reference: the
        network regresses ``u_t = r - x0`` and inference integrates
        ``dx/dt = v_theta`` from ``t=0`` to ``t=1``.

    ``t`` is the interpolation coordinate of a probability path. It is neither a
    forecast lead time nor a diffusion timestep.
    """

    source_distribution: str = "gaussian"
    interpolation_path: str = "existing_aurora"
    integration_steps: int = 1
    solver: str = "euler"
    stochastic_initialization: bool = True
    time_sampling: str = "logit_normal"
    sigma_min: float = 1.0e-3
    logit_normal_mean: float = -0.5
    logit_normal_std: float = 1.2
    residual_zscore: bool = False
    res_std_momentum: float = 0.99
    #: Trajectory length used by ``loss.deterministic_weight`` for the
    #: ``rectified_flow`` (velocity) path. The ``existing_aurora`` path needs a
    #: single query and ignores it.
    deterministic_training_steps: int = 4

    _KEYS = (
        "source_distribution",
        "interpolation_path",
        "integration_steps",
        "solver",
        "stochastic_initialization",
        "time_sampling",
        "sigma_min",
        "logit_normal_mean",
        "logit_normal_std",
        "residual_zscore",
        "res_std_momentum",
        "deterministic_training_steps",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> FlowMatchingConfig:
        raw = _as_mapping(data)
        sec = "refinement.flow_matching"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        out = cls(
            source_distribution=_as_choice(
                sec,
                "source_distribution",
                raw.get("source_distribution"),
                d.source_distribution,
                _SOURCE_DISTRIBUTIONS,
            ),
            interpolation_path=_as_choice(
                sec,
                "interpolation_path",
                raw.get("interpolation_path"),
                d.interpolation_path,
                _INTERPOLATION_PATHS,
            ),
            integration_steps=_as_int(
                sec, "integration_steps", raw.get("integration_steps"), d.integration_steps
            ),
            solver=_as_choice(sec, "solver", raw.get("solver"), d.solver, _SOLVERS),
            stochastic_initialization=_as_bool(
                sec,
                "stochastic_initialization",
                raw.get("stochastic_initialization"),
                d.stochastic_initialization,
            ),
            time_sampling=_as_choice(
                sec,
                "time_sampling",
                raw.get("time_sampling"),
                d.time_sampling,
                _TIME_SAMPLINGS,
            ),
            sigma_min=_as_float(
                sec, "sigma_min", raw.get("sigma_min"), d.sigma_min, minimum=0.0, maximum=0.5
            ),
            logit_normal_mean=_as_float(
                sec, "logit_normal_mean", raw.get("logit_normal_mean"), d.logit_normal_mean
            ),
            logit_normal_std=_as_float(
                sec,
                "logit_normal_std",
                raw.get("logit_normal_std"),
                d.logit_normal_std,
                minimum=1.0e-6,
            ),
            residual_zscore=_as_bool(
                sec, "residual_zscore", raw.get("residual_zscore"), d.residual_zscore
            ),
            res_std_momentum=_as_float(
                sec,
                "res_std_momentum",
                raw.get("res_std_momentum"),
                d.res_std_momentum,
                minimum=0.0,
                maximum=0.999999,
            ),
            deterministic_training_steps=_as_int(
                sec,
                "deterministic_training_steps",
                raw.get("deterministic_training_steps"),
                d.deterministic_training_steps,
                minimum=1,
            ),
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class DiffusionConfig:
    """Discrete-time DDPM settings for the ``diffusion_*`` refiners.

    ``training_timesteps`` and ``inference_steps`` index a *noise* process. They
    carry no forecast semantics and are unrelated to rollout lead times.
    """

    training_timesteps: int = 1000
    inference_steps: int = 50
    #: ``epsilon`` remains the legacy-safe default so omitted YAML keys preserve
    #: existing behavior. New residual-correction configurations should select
    #: ``sample`` (x0) explicitly: it makes a zero-initialised DDIM head return
    #: zero residual and avoids dividing prediction error by ``sqrt(abar)``.
    #: This mirrors the ``x1`` parameterisation used by the working flow head.
    prediction_type: str = "epsilon"
    schedule: str = "cosine"
    sampler: str = "ddim"
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    cosine_s: float = 8.0e-3
    eta: float = 0.0
    clip_sample: bool = False
    clip_sample_range: float = 10.0
    #: Loss weighting across noise levels. ``none`` preserves legacy behavior;
    #: improved configurations can explicitly choose ``auto`` (Min-SNR-gamma
    #: for epsilon/velocity and no extra weighting for sample prediction).
    snr_weighting: str = "none"
    snr_gamma: float = 5.0
    #: Where along the noise axis training timesteps are concentrated.
    #:
    #: ``low_noise``
    #:     the informative end for an ``epsilon`` network.
    #: ``high_noise``
    #:     the end where a ``sample`` (x0) network is forced to use the
    #:     conditioning instead of copying its input, and where the
    #:     deterministic ``posterior_mean`` query lives. Uniform sampling visits
    #:     that query point once per ``training_timesteps`` batches. Biasing
    #:     toward high noise gives the conditional signal more training weight,
    #:     but the resulting trade-off must be validated separately for each
    #:     dataset and architecture.
    #: ``uniform``
    #:     standard DDPM sampling.
    #: ``auto``
    #:     ``high_noise`` for ``sample``, ``low_noise`` for ``epsilon``, and
    #:     ``uniform`` for ``velocity``.
    timestep_distribution: str = "uniform"
    #: Strength of ``timestep_distribution``; ``0`` degenerates to ``uniform``.
    timestep_bias: float = 3.0
    #: Trajectory length used by ``loss.deterministic_weight``. Kept short so the
    #: extra differentiable pass stays affordable.
    deterministic_training_steps: int = 4
    #: How the deterministic (mean-like) residual is produced.
    #:
    #: ``posterior_mean``
    #:     one query at the largest timestep from the prior mean ``x = 0``. At
    #:     that timestep the noisy state is independent of the residual, so
    #:     ``E[r | x_T, cond] = E[r | cond]`` for *any* ``x_T`` - including the
    #:     mean - and that conditional mean is exactly the squared-error optimum.
    #:     This is the direct analogue of the ``t = 0, x = 0`` query that makes
    #:     the existing Aurora flow-matching head work, and it costs one forward
    #:     pass instead of ``inference_steps``. Requires ``sample`` or
    #:     ``velocity`` prediction, because under ``epsilon`` the conversion
    #:     divides by ``sqrt(abar_T)``.
    #: ``ode``
    #:     integrate the full ``eta = 0`` probability-flow trajectory from
    #:     ``x_T = 0``.
    #: ``auto``
    #:     ``posterior_mean`` when the parameterisation supports it, else ``ode``.
    deterministic_estimator: str = "auto"

    _KEYS = (
        "training_timesteps",
        "inference_steps",
        "prediction_type",
        "schedule",
        "sampler",
        "beta_start",
        "beta_end",
        "cosine_s",
        "eta",
        "clip_sample",
        "clip_sample_range",
        "snr_weighting",
        "snr_gamma",
        "timestep_distribution",
        "timestep_bias",
        "deterministic_training_steps",
        "deterministic_estimator",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> DiffusionConfig:
        raw = _as_mapping(data)
        sec = "refinement.diffusion"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        out = cls(
            training_timesteps=_as_int(
                sec, "training_timesteps", raw.get("training_timesteps"), d.training_timesteps
            ),
            inference_steps=_as_int(
                sec, "inference_steps", raw.get("inference_steps"), d.inference_steps
            ),
            prediction_type=_as_choice(
                sec,
                "prediction_type",
                raw.get("prediction_type"),
                d.prediction_type,
                _PREDICTION_TYPES,
            ),
            schedule=_as_choice(sec, "schedule", raw.get("schedule"), d.schedule, _SCHEDULES),
            sampler=_as_choice(sec, "sampler", raw.get("sampler"), d.sampler, _SAMPLERS),
            beta_start=_as_float(
                sec, "beta_start", raw.get("beta_start"), d.beta_start, minimum=0.0
            ),
            beta_end=_as_float(sec, "beta_end", raw.get("beta_end"), d.beta_end, minimum=0.0),
            cosine_s=_as_float(sec, "cosine_s", raw.get("cosine_s"), d.cosine_s, minimum=0.0),
            eta=_as_float(sec, "eta", raw.get("eta"), d.eta, minimum=0.0, maximum=1.0),
            clip_sample=_as_bool(sec, "clip_sample", raw.get("clip_sample"), d.clip_sample),
            clip_sample_range=_as_float(
                sec,
                "clip_sample_range",
                raw.get("clip_sample_range"),
                d.clip_sample_range,
                minimum=0.0,
            ),
            snr_weighting=_as_choice(
                sec,
                "snr_weighting",
                raw.get("snr_weighting"),
                d.snr_weighting,
                _SNR_WEIGHTINGS,
            ),
            snr_gamma=_as_float(
                sec, "snr_gamma", raw.get("snr_gamma"), d.snr_gamma, minimum=0.0
            ),
            timestep_distribution=_as_choice(
                sec,
                "timestep_distribution",
                raw.get("timestep_distribution"),
                d.timestep_distribution,
                _TIMESTEP_DISTRIBUTIONS,
            ),
            timestep_bias=_as_float(
                sec,
                "timestep_bias",
                raw.get("timestep_bias"),
                d.timestep_bias,
                minimum=0.0,
                maximum=8.0,
            ),
            deterministic_training_steps=_as_int(
                sec,
                "deterministic_training_steps",
                raw.get("deterministic_training_steps"),
                d.deterministic_training_steps,
                minimum=1,
            ),
            deterministic_estimator=_as_choice(
                sec,
                "deterministic_estimator",
                raw.get("deterministic_estimator"),
                d.deterministic_estimator,
                _DETERMINISTIC_ESTIMATORS,
            ),
        )
        if out.inference_steps > out.training_timesteps:
            raise ConfigValidationError(
                f"{sec}.inference_steps ({out.inference_steps}) cannot exceed "
                f"training_timesteps ({out.training_timesteps})."
            )
        if out.schedule in {"linear", "scaled_linear"} and out.beta_end <= out.beta_start:
            raise ConfigValidationError(
                f"{sec}.beta_end must be greater than beta_start for the "
                f"'{out.schedule}' schedule."
            )
        if out.sampler == "ddpm" and out.eta != 1.0:
            raise ConfigValidationError(
                f"{sec}.sampler='ddpm' requires eta=1.0 (ancestral sampling); got "
                f"eta={out.eta}. Use sampler='ddim' for eta<1."
            )
        if out.clip_sample and out.clip_sample_range <= 0.0:
            raise ConfigValidationError(
                f"{sec}.clip_sample_range must be > 0 when clip_sample is enabled."
            )
        if (
            out.prediction_type == "epsilon"
            and out.resolved_deterministic_estimator() == "posterior_mean"
        ):
            raise ConfigValidationError(
                f"{sec}.deterministic_estimator='posterior_mean' is incompatible "
                "with prediction_type='epsilon'; expected 'ode' or prediction_type='sample'."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}

    def resolved_snr_weighting(self) -> str:
        """``auto`` resolved from the prediction parameterisation.

        The native loss of each parameterisation already carries an implicit
        weighting across noise levels:

        * ``epsilon`` weights every timestep equally *in epsilon space*, which is
          a ``1/SNR`` weighting of the residual error - it under-trains exactly
          the low-noise timesteps that carry the signal, so it needs
          ``min_snr``;
        * ``sample`` measures the error directly on the clean residual, so it is
          already correctly balanced. Applying ``min_snr`` on top would
          up-weight the high-SNR timesteps where predicting ``x_0`` from
          ``x_t`` is the trivial identity, and down-weight the high-noise
          timesteps where the network must actually use the conditioning. Leave
          it unweighted.
        """
        if self.snr_weighting != "auto":
            return self.snr_weighting
        return "none" if self.prediction_type == "sample" else "min_snr"

    def resolved_deterministic_estimator(self) -> str:
        if self.deterministic_estimator != "auto":
            return self.deterministic_estimator
        return "posterior_mean" if self.prediction_type in {"sample", "velocity"} else "ode"

    def resolved_timestep_distribution(self) -> str:
        if self.timestep_distribution != "auto":
            return self.timestep_distribution
        if self.prediction_type == "sample":
            return "high_noise"
        if self.prediction_type == "epsilon":
            return "low_noise"
        return "uniform"


@dataclass(frozen=True)
class UNetRefinerConfig:
    """Geometry of the convolutional residual refiners."""

    hidden_channels: int = 64
    num_levels: int = 3
    num_residual_blocks: int = 2
    time_embedding_dim: int = 128
    dropout: float = 0.0
    bottleneck_attention: bool = True
    attention_heads: int = 4
    zero_init_output: bool = True

    _KEYS = (
        "hidden_channels",
        "num_levels",
        "num_residual_blocks",
        "time_embedding_dim",
        "dropout",
        "bottleneck_attention",
        "attention_heads",
        "zero_init_output",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> UNetRefinerConfig:
        raw = _as_mapping(data)
        sec = "refinement.unet"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        out = cls(
            hidden_channels=_as_int(
                sec, "hidden_channels", raw.get("hidden_channels"), d.hidden_channels
            ),
            num_levels=_as_int(sec, "num_levels", raw.get("num_levels"), d.num_levels),
            num_residual_blocks=_as_int(
                sec,
                "num_residual_blocks",
                raw.get("num_residual_blocks"),
                d.num_residual_blocks,
            ),
            time_embedding_dim=_as_int(
                sec,
                "time_embedding_dim",
                raw.get("time_embedding_dim"),
                d.time_embedding_dim,
                minimum=2,
            ),
            dropout=_as_float(
                sec, "dropout", raw.get("dropout"), d.dropout, minimum=0.0, maximum=1.0
            ),
            bottleneck_attention=_as_bool(
                sec,
                "bottleneck_attention",
                raw.get("bottleneck_attention"),
                d.bottleneck_attention,
            ),
            attention_heads=_as_int(
                sec, "attention_heads", raw.get("attention_heads"), d.attention_heads
            ),
            zero_init_output=_as_bool(
                sec, "zero_init_output", raw.get("zero_init_output"), d.zero_init_output
            ),
        )
        if out.time_embedding_dim % 2 != 0:
            raise ConfigValidationError(
                f"{sec}.time_embedding_dim must be even (sin/cos pairs), got "
                f"{out.time_embedding_dim}."
            )
        bottleneck_channels = out.hidden_channels * (2 ** (out.num_levels - 1))
        if out.bottleneck_attention and bottleneck_channels % out.attention_heads != 0:
            raise ConfigValidationError(
                f"{sec}: bottleneck channels ({bottleneck_channels}) must be divisible "
                f"by attention_heads ({out.attention_heads})."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class TransformerConfig:
    """Spatial-only Transformer geometry.

    Attention is applied across **2-D spatial tokens** of one sample at one
    forecast lead time. There is deliberately no temporal axis, no cross-lead
    attention and no causal mask. Forecast lead time enters only through a
    separate scalar embedding.
    """

    patch_size: tuple[int, int] = (8, 8)
    embedding_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    positional_encoding: str = "latlon_2d"
    max_tokens_lat: int = 256
    max_tokens_lon: int = 512
    attention_mode: str = "global_2d"
    window_size: tuple[int, int] = (8, 8)
    shifted_windows: bool = False
    gradient_checkpointing: bool = False
    optimized_attention: str = "auto"
    zero_init_output: bool = True
    #: Optionally add a 3x3 convolutional stem before patchification and a
    #: zero-initialised 3x3 residual head after reconstruction. Patch tokenization
    #: compresses each patch to one vector; these convolutions restore overlapping
    #: local support while preserving identity-at-init. Disabled by default so
    #: older YAMLs and checkpoints retain their architecture.
    local_refinement: bool = False

    _KEYS = (
        "patch_size",
        "patch_height",
        "patch_width",
        "embedding_dim",
        "num_heads",
        "num_blocks",
        "mlp_ratio",
        "dropout",
        "positional_encoding",
        "max_tokens_lat",
        "max_tokens_lon",
        "attention_mode",
        "window_size",
        "shifted_windows",
        "gradient_checkpointing",
        "optimized_attention",
        "zero_init_output",
        "local_refinement",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> TransformerConfig:
        raw = _as_mapping(data)
        sec = "refinement.transformer"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()

        patch = raw.get("patch_size")
        if patch is None:
            patch_size = (
                _as_int(sec, "patch_height", raw.get("patch_height"), d.patch_size[0]),
                _as_int(sec, "patch_width", raw.get("patch_width"), d.patch_size[1]),
            )
        else:
            if raw.get("patch_height") is not None or raw.get("patch_width") is not None:
                raise ConfigValidationError(
                    f"{sec}: set either patch_size or patch_height/patch_width, not both."
                )
            patch_size = _as_pair(sec, "patch_size", patch, d.patch_size)

        embedding_dim = _as_int(sec, "embedding_dim", raw.get("embedding_dim"), d.embedding_dim)
        num_heads = _as_int(sec, "num_heads", raw.get("num_heads"), d.num_heads)
        if embedding_dim % num_heads != 0:
            raise ConfigValidationError(
                f"{sec}.embedding_dim ({embedding_dim}) must be divisible by num_heads "
                f"({num_heads}); head_dim would be {embedding_dim / num_heads:.3f}."
            )
        head_dim = embedding_dim // num_heads
        if head_dim % 2 != 0:
            raise ConfigValidationError(f"{sec}: head dimension ({head_dim}) must be even.")

        positional_encoding = _as_choice(
            sec,
            "positional_encoding",
            raw.get("positional_encoding"),
            d.positional_encoding,
            _POSITIONAL_ENCODINGS,
        )
        if positional_encoding == "latlon_2d":
            # Canonical Aurora name for the separable learned 2-D lat/lon table.
            positional_encoding = "latlon_2d"
        if positional_encoding == "sincos_2d" and embedding_dim % 4 != 0:
            raise ConfigValidationError(
                f"{sec}.embedding_dim must be divisible by 4 for the 'sincos_2d' "
                f"positional encoding, got {embedding_dim}."
            )

        attention_mode = _as_choice(
            sec, "attention_mode", raw.get("attention_mode"), d.attention_mode, _ATTENTION_MODES
        )
        window_size = _as_pair(sec, "window_size", raw.get("window_size"), d.window_size)
        shifted_windows = _as_bool(
            sec, "shifted_windows", raw.get("shifted_windows"), d.shifted_windows
        )
        if attention_mode == "global_2d" and (
            raw.get("window_size") is not None or raw.get("shifted_windows") is not None
        ):
            warnings.warn(
                "refinement.transformer.window_size / shifted_windows are ignored for "
                "attention_mode='global_2d'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if attention_mode == "windowed_2d":
            if window_size[0] < 1 or window_size[1] < 1:
                raise ConfigValidationError(
                    f"{sec}.window_size entries must be >= 1, got {list(window_size)}."
                )
            if shifted_windows and (window_size[0] < 2 or window_size[1] < 2):
                raise ConfigValidationError(
                    f"{sec}.shifted_windows requires a window of at least 2x2 tokens, "
                    f"got {list(window_size)}."
                )

        out = cls(
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=_as_int(sec, "num_blocks", raw.get("num_blocks"), d.num_blocks),
            mlp_ratio=_as_float(sec, "mlp_ratio", raw.get("mlp_ratio"), d.mlp_ratio, minimum=0.5),
            dropout=_as_float(
                sec, "dropout", raw.get("dropout"), d.dropout, minimum=0.0, maximum=1.0
            ),
            positional_encoding=positional_encoding,
            max_tokens_lat=_as_int(
                sec, "max_tokens_lat", raw.get("max_tokens_lat"), d.max_tokens_lat
            ),
            max_tokens_lon=_as_int(
                sec, "max_tokens_lon", raw.get("max_tokens_lon"), d.max_tokens_lon
            ),
            attention_mode=attention_mode,
            window_size=window_size,
            shifted_windows=shifted_windows,
            gradient_checkpointing=_as_bool(
                sec,
                "gradient_checkpointing",
                raw.get("gradient_checkpointing"),
                d.gradient_checkpointing,
            ),
            optimized_attention=_as_choice(
                sec,
                "optimized_attention",
                raw.get("optimized_attention"),
                d.optimized_attention,
                _ATTENTION_IMPLEMENTATIONS,
            ),
            zero_init_output=_as_bool(
                sec, "zero_init_output", raw.get("zero_init_output"), d.zero_init_output
            ),
            local_refinement=_as_bool(
                sec, "local_refinement", raw.get("local_refinement"), d.local_refinement
            ),
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_size": list(self.patch_size),
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "positional_encoding": self.positional_encoding,
            "max_tokens_lat": self.max_tokens_lat,
            "max_tokens_lon": self.max_tokens_lon,
            "attention_mode": self.attention_mode,
            "window_size": list(self.window_size),
            "shifted_windows": self.shifted_windows,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optimized_attention": self.optimized_attention,
            "zero_init_output": self.zero_init_output,
            "local_refinement": self.local_refinement,
        }


@dataclass(frozen=True)
class BiasLossConfig:
    """Alias kept for readability in downstream imports."""

    #: :class:`LossConfig` carries the actual fields; this exists so callers can
    #: refer to the bias-aware loss section by an explicit name.
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass(frozen=True)
class RefinementConfig:
    """Fully resolved Phase-2 configuration."""

    enabled: bool = False
    type: str = "none"
    correction_convention: str = CORRECTION_CONVENTION
    checkpoint: str | None = None
    freeze_aurora: bool = True
    joint_finetuning: bool = False
    train_on_residual: bool = True
    feedback_to_rollout: bool = False
    ensemble_size: int = 1
    seed: int | None = None
    #: Use the deterministic (mean-path / probability-flow) estimate at inference
    #: instead of averaging stochastic draws. Pointwise scores (MAE, RMSE, bias)
    #: are minimised by the conditional *mean*, which the deterministic path
    #: targets directly; stochastic draws only approach it as the ensemble grows.
    #: Enable this when the product is a single deterministic forecast, and leave
    #: it off when calibrated ensemble spread is the objective.
    deterministic_inference: bool = False
    target_space: TargetSpaceConfig = field(default_factory=TargetSpaceConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    unet: UNetRefinerConfig = field(default_factory=UNetRefinerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    #: ``True`` when the configuration was derived from the legacy
    #: ``model.flow_refine_*`` keys rather than an explicit ``refinement`` block.
    from_legacy_keys: bool = False

    _KEYS = (
        "enabled",
        "type",
        "correction_convention",
        "checkpoint",
        "freeze_aurora",
        "joint_finetuning",
        "train_on_residual",
        "feedback_to_rollout",
        "ensemble_size",
        "seed",
        "deterministic_inference",
        "target_space",
        "conditioning",
        "loss",
        "flow_matching",
        "diffusion",
        "unet",
        "transformer",
    )

    # -- derived helpers -------------------------------------------------
    @property
    def is_active(self) -> bool:
        return bool(self.enabled) and self.type != "none"

    @property
    def is_legacy_flow_matching(self) -> bool:
        """``True`` for the existing Aurora rectified-flow UNet."""
        return self.is_active and self.type in _LEGACY_TYPES

    @property
    def is_diffusion(self) -> bool:
        return self.type in _DIFFUSION_TYPES

    @property
    def is_flow_matching(self) -> bool:
        return self.type in _FLOW_TYPES

    @property
    def uses_transformer(self) -> bool:
        return self.type in _TRANSFORMER_TYPES

    @property
    def backend(self) -> str:
        """``"none"``, ``"legacy"`` (existing wrapper) or ``"unified"``."""
        if not self.is_active:
            return "none"
        return "legacy" if self.type in _LEGACY_TYPES else "unified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "type": self.type,
            "correction_convention": self.correction_convention,
            "checkpoint": self.checkpoint,
            "freeze_aurora": self.freeze_aurora,
            "joint_finetuning": self.joint_finetuning,
            "train_on_residual": self.train_on_residual,
            "feedback_to_rollout": self.feedback_to_rollout,
            "ensemble_size": self.ensemble_size,
            "seed": self.seed,
            "deterministic_inference": self.deterministic_inference,
            "target_space": self.target_space.to_dict(),
            "conditioning": self.conditioning.to_dict(),
            "loss": self.loss.to_dict(),
            "flow_matching": self.flow_matching.to_dict(),
            "diffusion": self.diffusion.to_dict(),
            "unet": self.unet.to_dict(),
            "transformer": self.transformer.to_dict(),
            "from_legacy_keys": self.from_legacy_keys,
        }


# ---------------------------------------------------------------------------
# Performance (workflow-only) settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataloaderPerfConfig:
    num_workers: int | None = None  # ``None`` == "auto"
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking_transfer: bool = True

    _KEYS = (
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "non_blocking_transfer",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> DataloaderPerfConfig:
        raw = _as_mapping(data)
        sec = "performance.dataloader"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        workers_raw = raw.get("num_workers")
        if workers_raw is None or str(workers_raw).strip().lower() == "auto":
            workers = None
        else:
            workers = _as_int(sec, "num_workers", workers_raw, 0, minimum=0)
        return cls(
            num_workers=workers,
            pin_memory=_as_bool(sec, "pin_memory", raw.get("pin_memory"), d.pin_memory),
            persistent_workers=_as_bool(
                sec, "persistent_workers", raw.get("persistent_workers"), d.persistent_workers
            ),
            prefetch_factor=_as_int(
                sec, "prefetch_factor", raw.get("prefetch_factor"), d.prefetch_factor
            ),
            non_blocking_transfer=_as_bool(
                sec,
                "non_blocking_transfer",
                raw.get("non_blocking_transfer"),
                d.non_blocking_transfer,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class PrecisionPerfConfig:
    mode: str = "fp32"
    allow_tf32: bool = False

    _KEYS = ("mode", "allow_tf32")

    @classmethod
    def from_mapping(cls, data: Any) -> PrecisionPerfConfig:
        raw = _as_mapping(data)
        sec = "performance.precision"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        return cls(
            mode=_as_choice(sec, "mode", raw.get("mode"), d.mode, _PRECISION_MODES),
            allow_tf32=_as_bool(sec, "allow_tf32", raw.get("allow_tf32"), d.allow_tf32),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class CompilePerfConfig:
    enabled: bool = False
    aurora: bool = False
    refinement: bool = False
    mode: str = "default"

    _KEYS = ("enabled", "aurora", "refinement", "mode")

    @classmethod
    def from_mapping(cls, data: Any) -> CompilePerfConfig:
        raw = _as_mapping(data)
        sec = "performance.compile"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        return cls(
            enabled=_as_bool(sec, "enabled", raw.get("enabled"), d.enabled),
            aurora=_as_bool(sec, "aurora", raw.get("aurora"), d.aurora),
            refinement=_as_bool(sec, "refinement", raw.get("refinement"), d.refinement),
            mode=_as_choice(
                sec,
                "mode",
                raw.get("mode"),
                d.mode,
                ("default", "reduce-overhead", "max-autotune"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class CachePerfConfig:
    """Deterministic Aurora rollout / conditioning cache."""

    section: str = "rollout_cache"
    enabled: bool = False
    path: str | None = None
    validate_cache: bool = True
    validate_samples: int = 4
    validate_tolerance: float = 1.0e-5

    _KEYS = ("enabled", "path", "validate_cache", "validate_samples", "validate_tolerance")

    @classmethod
    def from_mapping(cls, data: Any, *, section: str) -> CachePerfConfig:
        raw = _as_mapping(data)
        sec = f"performance.{section}"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        path = raw.get("path")
        out = cls(
            section=section,
            enabled=_as_bool(sec, "enabled", raw.get("enabled"), d.enabled),
            path=None if path in (None, "", "null") else str(path),
            validate_cache=_as_bool(
                sec, "validate_cache", raw.get("validate_cache"), d.validate_cache
            ),
            validate_samples=_as_int(
                sec,
                "validate_samples",
                raw.get("validate_samples"),
                d.validate_samples,
                minimum=0,
            ),
            validate_tolerance=_as_float(
                sec,
                "validate_tolerance",
                raw.get("validate_tolerance"),
                d.validate_tolerance,
                minimum=0.0,
            ),
        )
        if out.enabled and not out.path:
            raise ConfigValidationError(f"{sec}.enabled is true but no 'path' was provided.")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class EnsemblePerfConfig:
    batch_members: bool = True
    chunk_size: int | None = None  # ``None`` == "auto"

    _KEYS = ("batch_members", "chunk_size")

    @classmethod
    def from_mapping(cls, data: Any) -> EnsemblePerfConfig:
        raw = _as_mapping(data)
        sec = "performance.ensemble"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        chunk_raw = raw.get("chunk_size")
        if chunk_raw is None or str(chunk_raw).strip().lower() == "auto":
            chunk = None
        else:
            chunk = _as_int(sec, "chunk_size", chunk_raw, 1)
        return cls(
            batch_members=_as_bool(sec, "batch_members", raw.get("batch_members"), d.batch_members),
            chunk_size=chunk,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"batch_members": self.batch_members, "chunk_size": self.chunk_size}


@dataclass(frozen=True)
class IoPerfConfig:
    atomic_checkpoints: bool = True
    netcdf_compression: bool = True
    netcdf_compression_level: int = 4

    _KEYS = ("atomic_checkpoints", "netcdf_compression", "netcdf_compression_level")

    @classmethod
    def from_mapping(cls, data: Any) -> IoPerfConfig:
        raw = _as_mapping(data)
        sec = "performance.io"
        _reject_unknown(sec, raw, cls._KEYS)
        d = cls()
        level = _as_int(
            sec,
            "netcdf_compression_level",
            raw.get("netcdf_compression_level"),
            d.netcdf_compression_level,
            minimum=1,
        )
        if level > 9:
            raise ConfigValidationError(
                f"{sec}.netcdf_compression_level must be in [1, 9], got {level}."
            )
        return cls(
            atomic_checkpoints=_as_bool(
                sec, "atomic_checkpoints", raw.get("atomic_checkpoints"), d.atomic_checkpoints
            ),
            netcdf_compression=_as_bool(
                sec, "netcdf_compression", raw.get("netcdf_compression"), d.netcdf_compression
            ),
            netcdf_compression_level=level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class PerformanceConfig:
    """Workflow-only settings.

    Nothing here may change model architecture, loss, sampler, solver, ensemble
    size, diffusion/flow step counts or evaluation data. Unsupported
    combinations raise instead of silently degrading numerical behaviour.
    """

    profile: bool = False
    dataloader: DataloaderPerfConfig = field(default_factory=DataloaderPerfConfig)
    precision: PrecisionPerfConfig = field(default_factory=PrecisionPerfConfig)
    compile: CompilePerfConfig = field(default_factory=CompilePerfConfig)
    rollout_cache: CachePerfConfig = field(
        default_factory=lambda: CachePerfConfig(section="rollout_cache")
    )
    conditioning_cache: CachePerfConfig = field(
        default_factory=lambda: CachePerfConfig(section="conditioning_cache")
    )
    ensemble: EnsemblePerfConfig = field(default_factory=EnsemblePerfConfig)
    io: IoPerfConfig = field(default_factory=IoPerfConfig)

    _KEYS = (
        "profile",
        "dataloader",
        "precision",
        "compile",
        "rollout_cache",
        "conditioning_cache",
        "ensemble",
        "io",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "dataloader": self.dataloader.to_dict(),
            "precision": self.precision.to_dict(),
            "compile": self.compile.to_dict(),
            "rollout_cache": self.rollout_cache.to_dict(),
            "conditioning_cache": self.conditioning_cache.to_dict(),
            "ensemble": self.ensemble.to_dict(),
            "io": self.io.to_dict(),
        }


# ---------------------------------------------------------------------------
# Resolution entry points
# ---------------------------------------------------------------------------


def _locate_section(config: Any, name: str) -> Any:
    """Find ``name`` on ``config`` or on ``config['model']``."""
    if config is None:
        return None
    holders: list[Any] = [config]
    if isinstance(config, Mapping):
        holders.append(config.get("model"))
    else:
        holders.append(getattr(config, "model", None))
    for holder in holders:
        if holder is None:
            continue
        if isinstance(holder, Mapping):
            if name in holder:
                return holder[name]
        elif hasattr(holder, name):
            value = getattr(holder, name)
            if value is not None:
                return value
    return None


def _legacy_refinement_config(config: Any) -> RefinementConfig:
    """Translate the legacy ``model.flow_refine_*`` keys into the new schema.

    This keeps every existing YAML file working unchanged: a config that only
    sets ``model.flow_refine_enabled: true`` resolves to
    ``type='flow_matching_unet'`` with the legacy backend, which runs the
    unmodified :class:`finetune.flow_refine.AuroraFlowRefine` code path.
    """
    model_cfg = _as_mapping(_locate_section(config, "model") or {})
    training_cfg = _as_mapping(_locate_section(config, "training") or {})
    if not bool(model_cfg.get("flow_refine_enabled", False)):
        return RefinementConfig()

    sampling_steps = _as_int(
        "model", "flow_refine_sampling_steps", model_cfg.get("flow_refine_sampling_steps"), 1
    )
    flow = FlowMatchingConfig(
        interpolation_path="existing_aurora",
        integration_steps=sampling_steps,
        residual_zscore=bool(model_cfg.get("flow_refine_residual_zscore", False)),
        res_std_momentum=float(model_cfg.get("flow_refine_res_std_momentum", 0.99)),
    )
    # The legacy heads take their width and time-embedding width from the
    # historical ``model.flow_refine_*`` keys; mirror them into the new section
    # so both spellings describe the same network.
    unet = UNetRefinerConfig(
        hidden_channels=_as_int(
            "model", "flow_refine_hidden", model_cfg.get("flow_refine_hidden"), 64
        ),
        time_embedding_dim=_as_int(
            "model",
            "flow_refine_time_dim",
            model_cfg.get("flow_refine_time_dim"),
            128,
            minimum=2,
        ),
    )
    return RefinementConfig(
        enabled=True,
        type="flow_matching_unet",
        # The legacy inference path applies refinement inline in the rollout
        # loop; record that faithfully instead of silently changing behaviour.
        feedback_to_rollout=bool(training_cfg.get("flow_refine_autoregressive_feedback", False)),
        flow_matching=flow,
        unet=unet,
        conditioning=ConditioningConfig(
            aurora_rollout=True,
            static_fields=False,
            masks=False,
            forecast_lead_time=bool(model_cfg.get("flow_refine_lead_time_cond", False)),
        ),
        from_legacy_keys=True,
    )


def resolve_refinement_config(config: Any) -> RefinementConfig:
    """Normalize the ``model.refinement`` section of an experiment configuration.

    Accepts a raw YAML mapping (the format used by
    :func:`finetune.aurora_finetune_utils.load_config`), any object exposing the
    same attributes, or ``None``. A missing section falls back to the legacy
    ``model.flow_refine_*`` keys and otherwise yields the deterministic default.
    """
    raw_section = _locate_section(config, "refinement")
    if raw_section is None:
        return _legacy_refinement_config(config)

    raw = _as_mapping(raw_section)
    _reject_unknown("refinement", raw, RefinementConfig._KEYS)

    raw_type = raw.get("type")
    if raw_type is None:
        resolved_type = "none"
    else:
        key = str(raw_type).strip().lower()
        if key not in _REFINEMENT_ALIASES:
            raise ConfigValidationError(
                f"Unknown refinement.type {raw_type!r}. Supported values: "
                f"{list(REFINEMENT_TYPES)} (plus the aliases "
                f"{sorted(set(_REFINEMENT_ALIASES) - set(REFINEMENT_TYPES))})."
            )
        resolved_type = _REFINEMENT_ALIASES[key]

    enabled_raw = raw.get("enabled")
    enabled = (
        resolved_type != "none"
        if enabled_raw is None
        else _as_bool("refinement", "enabled", enabled_raw, True)
    )
    if enabled and resolved_type == "none":
        warnings.warn(
            "refinement.enabled is true but refinement.type is 'none'; running the "
            "deterministic Aurora rollout without stochastic refinement.",
            RuntimeWarning,
            stacklevel=2,
        )
        enabled = False

    correction_convention = _as_choice(
        "refinement",
        "correction_convention",
        raw.get("correction_convention"),
        CORRECTION_CONVENTION,
        _CORRECTION_CONVENTIONS,
    )

    checkpoint = raw.get("checkpoint")
    checkpoint = None if checkpoint in (None, "", "null") else str(checkpoint)

    freeze_aurora = _as_bool("refinement", "freeze_aurora", raw.get("freeze_aurora"), True)
    joint = _as_bool("refinement", "joint_finetuning", raw.get("joint_finetuning"), False)
    if joint and raw.get("freeze_aurora") is not None and freeze_aurora:
        raise ConfigValidationError(
            "refinement.joint_finetuning=true is incompatible with "
            "refinement.freeze_aurora=true. Joint Aurora/refiner training must be "
            "requested explicitly and requires an unfrozen Aurora."
        )
    if joint:
        freeze_aurora = False

    train_on_residual = _as_bool(
        "refinement", "train_on_residual", raw.get("train_on_residual"), True
    )
    if enabled and not train_on_residual:
        raise ConfigValidationError(
            "Active refinement requires train_on_residual=true under "
            "correction_convention='cams_minus_aurora_add': Phase 2 must predict "
            "CAMS - Aurora and the forecast must add that correction to Aurora. "
            "Whole-field prediction is not a supported refinement contract."
        )

    feedback = _as_bool("refinement", "feedback_to_rollout", raw.get("feedback_to_rollout"), False)
    if feedback:
        warnings.warn(
            "refinement.feedback_to_rollout=true is EXPERIMENTAL: the refined field is "
            "fed back into later Aurora rollout steps, which changes the deterministic "
            "rollout trajectory and can accumulate error. Evaluate it independently.",
            RuntimeWarning,
            stacklevel=2,
        )

    ensemble_size = _as_int("refinement", "ensemble_size", raw.get("ensemble_size"), 1)
    seed_raw = raw.get("seed")
    seed = None if seed_raw is None else _as_int("refinement", "seed", seed_raw, 0, minimum=0)
    deterministic_inference = _as_bool(
        "refinement", "deterministic_inference", raw.get("deterministic_inference"), False
    )

    flow_matching = FlowMatchingConfig.from_mapping(raw.get("flow_matching"))
    diffusion = DiffusionConfig.from_mapping(raw.get("diffusion"))
    transformer = TransformerConfig.from_mapping(raw.get("transformer"))
    unet = UNetRefinerConfig.from_mapping(raw.get("unet"))
    loss = LossConfig.from_mapping(raw.get("loss"))

    cfg = RefinementConfig(
        enabled=enabled,
        type=resolved_type if enabled else "none",
        correction_convention=correction_convention,
        checkpoint=checkpoint,
        freeze_aurora=freeze_aurora,
        joint_finetuning=joint,
        train_on_residual=train_on_residual,
        feedback_to_rollout=feedback,
        ensemble_size=ensemble_size,
        seed=seed,
        deterministic_inference=deterministic_inference,
        target_space=TargetSpaceConfig.from_mapping(raw.get("target_space")),
        conditioning=ConditioningConfig.from_mapping(raw.get("conditioning")),
        loss=loss,
        flow_matching=flow_matching,
        diffusion=diffusion,
        unet=unet,
        transformer=transformer,
    )

    raw_diffusion = _as_mapping(raw.get("diffusion"))
    if (
        cfg.is_active
        and cfg.is_diffusion
        and cfg.diffusion.prediction_type == "epsilon"
        and raw_diffusion.get("prediction_type") is not None
    ):
        _warn_epsilon_residual_prediction_once()

    if cfg.is_active and cfg.is_diffusion and raw.get("flow_matching"):
        warnings.warn(
            f"refinement.flow_matching settings are ignored for type={cfg.type!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg.is_active and cfg.is_flow_matching and raw.get("diffusion"):
        warnings.warn(
            f"refinement.diffusion settings are ignored for type={cfg.type!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg.is_legacy_flow_matching:
        if cfg.flow_matching.interpolation_path != "existing_aurora":
            raise ConfigValidationError(
                "refinement.type='flow_matching_unet' is the existing Aurora "
                "rectified-flow implementation and only supports "
                "flow_matching.interpolation_path='existing_aurora'. Use "
                "'flow_matching_transformer' to select another documented path."
            )
        if cfg.loss.has_auxiliary_terms:
            raise ConfigValidationError(
                "refinement.loss.* auxiliary weights are not applied by the legacy "
                "'flow_matching_unet' refiner; it keeps its existing "
                "training.flow_aux_loss objective. Set them to 0.0 or choose one of "
                "the new refiners."
            )
    return cfg


def resolve_performance_config(config: Any) -> PerformanceConfig:
    """Normalize the ``performance`` section of an experiment configuration."""
    raw_section = _locate_section(config, "performance")
    if raw_section is None:
        return PerformanceConfig()

    raw = _as_mapping(raw_section)
    _reject_unknown("performance", raw, PerformanceConfig._KEYS)

    cfg = PerformanceConfig(
        profile=_as_bool("performance", "profile", raw.get("profile"), False),
        dataloader=DataloaderPerfConfig.from_mapping(raw.get("dataloader")),
        precision=PrecisionPerfConfig.from_mapping(raw.get("precision")),
        compile=CompilePerfConfig.from_mapping(raw.get("compile")),
        rollout_cache=CachePerfConfig.from_mapping(
            raw.get("rollout_cache"), section="rollout_cache"
        ),
        conditioning_cache=CachePerfConfig.from_mapping(
            raw.get("conditioning_cache"), section="conditioning_cache"
        ),
        ensemble=EnsemblePerfConfig.from_mapping(raw.get("ensemble")),
        io=IoPerfConfig.from_mapping(raw.get("io")),
    )

    refinement = resolve_refinement_config(config)
    for cache in (cfg.rollout_cache, cfg.conditioning_cache):
        if not cache.enabled:
            continue
        if refinement.joint_finetuning:
            raise ConfigValidationError(
                f"performance.{cache.section}.enabled is incompatible with "
                "refinement.joint_finetuning: cached Aurora rollouts would be stale as "
                "soon as Aurora weights are updated."
            )
        if not refinement.freeze_aurora:
            raise ConfigValidationError(
                f"performance.{cache.section}.enabled requires refinement.freeze_aurora=true."
            )
    if cfg.compile.enabled and not (cfg.compile.aurora or cfg.compile.refinement):
        warnings.warn(
            "performance.compile.enabled is true but neither compile.aurora nor "
            "compile.refinement is enabled; nothing will be compiled.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg.precision.mode != "fp32" or cfg.precision.allow_tf32:
        warnings.warn(
            "performance.precision requests a reduced-precision path; validate "
            "numerical parity against the fp32 reference before trusting results.",
            RuntimeWarning,
            stacklevel=2,
        )
    return cfg


def config_fingerprint(payload: Any) -> str:
    """Stable SHA-256 fingerprint of a JSON-serialisable configuration payload."""

    def _default(obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    blob = json.dumps(copy.deepcopy(payload), sort_keys=True, default=_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
