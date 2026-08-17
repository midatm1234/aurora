"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Unified two-phase Aurora model wrapper.

Phase 1 (deterministic)
    The existing Aurora forecast / autoregressive rollout. Architecture,
    checkpoint behaviour, input history, variable and level ordering, forecast
    timing, normalization and postprocessing are unchanged.

Phase 2 (optional, stochastic)
    One of ``flow_matching_unet``, ``flow_matching_transformer``,
    ``diffusion_unet`` or ``diffusion_transformer``, predicting a **residual**
    in Aurora's normalized target space for one forecast valid time. The four
    options are mutually exclusive alternatives; flow matching never requires
    the diffusion head to run first and the two are never stacked.

Rollout-semantics contract
--------------------------
Refinement is **postprocessing of a deterministic rollout step**::

    Aurora state at step n
      -> deterministic Aurora prediction for step n+1
      -> optional stochastic residual correction for step n+1

The refined field is returned alongside — never in place of — the deterministic
prediction, so the deterministic state used to generate later rollout steps is
untouched. Feeding the refined field back into the rollout is an explicit,
disabled-by-default, experimental option
(``refinement.feedback_to_rollout: true``) handled by a separate code path in
the training/inference driver.

This wrapper never shifts, re-indexes or re-orders the time axis, never changes
the rollout length or interval or the input history, and never uses a future
target value as conditioning.

Adapted from ``granitewxc.refinement.two_phase`` in the Prithvi stochastic
residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), restructured around
Aurora's per-rollout-step packed fields and forecast lead times.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from finetune.mamba_temporal import PackedMambaTemporalAdapter
from finetune.refinement.base import ChunkNoiseSource, ResidualRefiner, build_refiner
from finetune.refinement.config import (
    PerformanceConfig,
    RefinementConfig,
    resolve_performance_config,
    resolve_refinement_config,
)
from finetune.refinement.losses import area_weights_from_latitudes
from finetune.refinement.packing import FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace

__all__ = [
    "AuroraTwoPhaseRefiner",
    "TwoPhaseStepOutput",
    "build_two_phase_refiner",
    "resolve_temporal_config",
]


_TEMPORAL_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mode": "per_variable",
    "channels": 16,
    "state": 8,
    "layers": 2,
    "conv": 3,
    "expand": 2,
    "dropout": 0.0,
    "gated_fusion": False,
    "gate_init": 0.0,
    "lead_time_conditioning": False,
    "mask_conditioning": False,
    "coordinate_conditioning": False,
    "causal": True,
    "learning_rate_multiplier": 1.0,
}
_TEMPORAL_OBJECTIVE_DEFAULTS: dict[str, Any] = {
    "base": "mse",
    "huber_delta": 1.0,
    "charbonnier_epsilon": 1.0e-3,
    "tendency_weight": 0.0,
    "structure_weight": 0.0,
    "extreme_weight": 0.0,
    "extreme_quantile": 0.95,
}
_TEMPORAL_LEGACY_KEYS = {
    "enabled": "mamba_temporal_enabled",
    "channels": "mamba_temporal_channels",
    "state": "mamba_temporal_state",
    "layers": "mamba_temporal_layers",
    "conv": "mamba_temporal_conv",
    "expand": "mamba_temporal_expand",
}
_TEMPORAL_READABLE_ALIASES = {
    "state_dim": "state",
    "num_layers": "layers",
    "conv_kernel": "conv",
    "expansion_factor": "expand",
}
_TEMPORAL_NESTED_KEYS = frozenset(
    {
        *_TEMPORAL_DEFAULTS,
        *_TEMPORAL_READABLE_ALIASES,
        "objective",
    }
)
_TEMPORAL_OBJECTIVE_KEYS = frozenset(_TEMPORAL_OBJECTIVE_DEFAULTS)


def _finite_temporal_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {number}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}, got {number}.")
    return number


def _positive_temporal_int(value: Any, field_name: str) -> int:
    """Resolve an integer architecture width without silently truncating."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if integer < 1 or isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"{field_name} must be a positive integer, got {value!r}."
        )
    return integer


def resolve_temporal_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the nested temporal schema and translate legacy flat keys."""
    model = (config or {}).get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping for temporal configuration.")
    nested = model.get("mamba_temporal", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ValueError(
            "model.mamba_temporal must be a mapping, "
            f"got {type(nested).__name__}."
        )
    unknown = sorted(set(nested) - _TEMPORAL_NESTED_KEYS)
    if unknown:
        raise ValueError(
            "Unknown model.mamba_temporal key(s) "
            f"{unknown}; supported keys are {sorted(_TEMPORAL_NESTED_KEYS)}."
        )

    readable_by_short = {
        short: readable for readable, short in _TEMPORAL_READABLE_ALIASES.items()
    }
    for short, legacy_key in _TEMPORAL_LEGACY_KEYS.items():
        nested_keys = (short, readable_by_short.get(short))
        for nested_key in nested_keys:
            if nested_key is None:
                continue
            if (
                legacy_key in model
                and nested_key in nested
                and model[legacy_key] != nested[nested_key]
            ):
                raise ValueError(
                    f"model.{legacy_key} conflicts with "
                    f"model.mamba_temporal.{nested_key}."
                )

    resolved = dict(_TEMPORAL_DEFAULTS)
    for name, legacy_key in _TEMPORAL_LEGACY_KEYS.items():
        if legacy_key in model:
            resolved[name] = model[legacy_key]
    for key in _TEMPORAL_DEFAULTS:
        if key in nested:
            resolved[key] = nested[key]
    for readable, short in _TEMPORAL_READABLE_ALIASES.items():
        if readable in nested and short in nested and nested[readable] != nested[short]:
            raise ValueError(
                f"model.mamba_temporal.{readable} conflicts with legacy alias "
                f"model.mamba_temporal.{short}."
            )
        if readable in nested:
            resolved[short] = nested[readable]

    enabled = resolved["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(
            "model.mamba_temporal.enabled must be true or false, "
            f"got {enabled!r}."
        )
    mode = str(resolved["mode"]).strip().lower()
    if mode not in {"per_variable", "packed_joint"}:
        raise ValueError(
            "model.mamba_temporal.mode must be 'per_variable' or "
            f"'packed_joint', got {resolved['mode']!r}."
        )
    for key in (
        "lead_time_conditioning",
        "mask_conditioning",
        "coordinate_conditioning",
        "causal",
        "gated_fusion",
    ):
        if key not in nested and mode == "packed_joint" and key in {
            "lead_time_conditioning",
            "mask_conditioning",
            "gated_fusion",
        }:
            resolved[key] = True
        if not isinstance(resolved[key], bool):
            raise ValueError(
                f"model.mamba_temporal.{key} must be true or false, "
                f"got {resolved[key]!r}."
            )
    if not resolved["causal"]:
        raise ValueError(
            "model.mamba_temporal.causal=false is unsupported; temporal "
            "refinement must not use future forecast leads."
        )

    gate_init = _finite_temporal_number(
        resolved["gate_init"],
        "model.mamba_temporal.gate_init",
        minimum=0.0,
    )
    if gate_init >= 1.0:
        raise ValueError(
            "model.mamba_temporal.gate_init must be less than 1."
        )
    dropout = _finite_temporal_number(
        resolved["dropout"],
        "model.mamba_temporal.dropout",
        minimum=0.0,
    )
    if dropout >= 1.0:
        raise ValueError("model.mamba_temporal.dropout must be less than 1.")
    learning_rate_multiplier = _finite_temporal_number(
        resolved["learning_rate_multiplier"],
        "model.mamba_temporal.learning_rate_multiplier",
        minimum=0.0,
    )
    if learning_rate_multiplier <= 0.0:
        raise ValueError(
            "model.mamba_temporal.learning_rate_multiplier must be greater than 0."
        )

    objective_raw = nested.get("objective", {})
    if objective_raw is None:
        objective_raw = {}
    if not isinstance(objective_raw, Mapping):
        raise ValueError("model.mamba_temporal.objective must be a mapping.")
    objective_unknown = sorted(set(objective_raw) - _TEMPORAL_OBJECTIVE_KEYS)
    if objective_unknown:
        raise ValueError(
            "Unknown model.mamba_temporal.objective key(s) "
            f"{objective_unknown}; supported keys are "
            f"{sorted(_TEMPORAL_OBJECTIVE_KEYS)}."
        )
    objective = dict(_TEMPORAL_OBJECTIVE_DEFAULTS)
    objective.update(objective_raw)
    objective["base"] = str(objective["base"]).strip().lower()
    if objective["base"] not in {"mse", "huber", "charbonnier"}:
        raise ValueError(
            "model.mamba_temporal.objective.base must be mse, huber, or "
            f"charbonnier, got {objective['base']!r}."
        )
    for key in (
        "huber_delta",
        "charbonnier_epsilon",
        "tendency_weight",
        "structure_weight",
        "extreme_weight",
    ):
        minimum = 1.0e-12 if key in {"huber_delta", "charbonnier_epsilon"} else 0.0
        objective[key] = _finite_temporal_number(
            objective[key],
            f"model.mamba_temporal.objective.{key}",
            minimum=minimum,
        )
    objective["extreme_quantile"] = _finite_temporal_number(
        objective["extreme_quantile"],
        "model.mamba_temporal.objective.extreme_quantile",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < objective["extreme_quantile"] < 1.0:
        raise ValueError(
            "model.mamba_temporal.objective.extreme_quantile must be in (0, 1)."
        )

    if objective["extreme_weight"] > 0.0:
        raise ValueError(
            "model.mamba_temporal.objective.extreme_weight requires persisted "
            "training-split tail thresholds, which are not yet available; keep "
            "it at 0.0 to avoid deriving thresholds from evaluation targets."
        )

    def _architecture_label(short: str, readable: str) -> str:
        legacy = _TEMPORAL_LEGACY_KEYS[short]
        if legacy in model and short not in nested and readable not in nested:
            return f"model.{legacy}"
        return f"model.mamba_temporal.{readable}"

    channels = _positive_temporal_int(
        resolved["channels"], _architecture_label("channels", "channels")
    )
    state = _positive_temporal_int(
        resolved["state"], _architecture_label("state", "state_dim")
    )
    layers = _positive_temporal_int(
        resolved["layers"], _architecture_label("layers", "num_layers")
    )
    conv = _positive_temporal_int(
        resolved["conv"], _architecture_label("conv", "conv_kernel")
    )
    expand = _positive_temporal_int(
        resolved["expand"], _architecture_label("expand", "expansion_factor")
    )
    return {
        "enabled": enabled,
        "mode": mode,
        "channels": channels,
        "state": state,
        "state_dim": state,
        "layers": layers,
        "num_layers": layers,
        "conv": conv,
        "conv_kernel": conv,
        "expand": expand,
        "expansion_factor": expand,
        "dropout": dropout,
        "gated_fusion": bool(resolved["gated_fusion"]),
        "gate_init": gate_init,
        "lead_time_conditioning": bool(resolved["lead_time_conditioning"]),
        "mask_conditioning": bool(resolved["mask_conditioning"]),
        "coordinate_conditioning": bool(resolved["coordinate_conditioning"]),
        "causal": True,
        "learning_rate_multiplier": learning_rate_multiplier,
        "objective": objective,
        "semantic_version": 1 if mode == "per_variable" else 2,
        "scan_backend": "selective_scan_ref_v1",
    }


@dataclass
class TwoPhaseStepOutput:
    """Everything the training loop, the evaluator and the writer may need.

    All tensors refer to **one forecast valid time per leading batch entry**;
    when several rollout lead times were folded into the batch dimension the
    caller unfolds them with the same ``(batch, lead)`` order.
    """

    deterministic_normalized: torch.Tensor
    """Deterministic Aurora rollout in normalized target space."""
    conditioning: torch.Tensor | None = None
    deterministic_physical: torch.Tensor | None = None

    correction_target_normalized: torch.Tensor | None = None
    correction_target_physical: torch.Tensor | None = None
    # Historical alias for correction_target_normalized.
    residual_target: torch.Tensor | None = None

    predicted_correction_normalized: torch.Tensor | None = None
    predicted_correction_physical: torch.Tensor | None = None
    # Historical alias for predicted_correction_normalized.
    residual: torch.Tensor | None = None

    conditional_mean_correction_normalized: torch.Tensor | None = None
    conditional_mean_correction_physical: torch.Tensor | None = None
    deterministic_refined_normalized: torch.Tensor | None = None
    deterministic_refined_physical: torch.Tensor | None = None

    refined_normalized: torch.Tensor | None = None
    refined_physical: torch.Tensor | None = None
    members: torch.Tensor | None = None
    member_corrections_normalized: torch.Tensor | None = None
    member_corrections_physical: torch.Tensor | None = None
    member_innovations_normalized: torch.Tensor | None = None
    member_innovations_physical: torch.Tensor | None = None
    # Historical alias for member_corrections_normalized.
    member_residuals: torch.Tensor | None = None
    ensemble_mean: torch.Tensor | None = None
    ensemble_spread: torch.Tensor | None = None
    constraint_activation_fraction: torch.Tensor | None = None

    valid_mask: torch.Tensor | None = None
    process_time: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and not (isinstance(value, dict) and not value)
        }


def _align_to(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Crop or resample a conditioning field onto the target grid."""
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    if h >= size[0] and w >= size[1]:
        return tensor[..., : size[0], : size[1]]
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class AuroraTwoPhaseRefiner(nn.Module):
    """Deterministic Aurora plus an optional stochastic residual refiner.

    Module names are chosen so that Aurora state-dict keys are simply prefixed
    with ``aurora.`` and refinement keys with ``refiner.``; see
    :mod:`finetune.refinement.checkpoint` for the explicit migration helpers.

    Args:
        aurora: the deterministic Aurora model (Phase 1). May be ``None`` when
            the caller drives the rollout itself and only needs Phase 2.
        packing: the variable / pressure-level channel layout.
        refinement: resolved :class:`RefinementConfig`.
        performance: resolved :class:`PerformanceConfig`.
        nonnegative_variables: variables constrained to ``>= 0`` in physical
            units.
    """

    def __init__(
        self,
        aurora: nn.Module | None,
        packing: FieldPacking,
        refinement: RefinementConfig | None = None,
        performance: PerformanceConfig | None = None,
        *,
        nonnegative_variables: Sequence[str] = (),
        conditioning_static_names: Sequence[str] = (),
        temporal_enabled: bool = False,
        temporal_channels: int = 16,
        temporal_state: int = 8,
        temporal_layers: int = 2,
        temporal_conv: int = 3,
        temporal_expand: int = 2,
        temporal_mode: str = "per_variable",
        temporal_dropout: float = 0.0,
        temporal_gated_fusion: bool = False,
        temporal_gate_init: float = 0.0,
        temporal_lead_time_conditioning: bool = False,
        temporal_mask_conditioning: bool = False,
        temporal_coordinate_conditioning: bool = False,
        temporal_causal: bool = True,
        temporal_learning_rate_multiplier: float = 1.0,
        temporal_objective: Mapping[str, Any] | None = None,
        temporal_semantic_version: int = 1,
        temporal_scan_backend: str = "selective_scan_ref_v1",
    ) -> None:
        super().__init__()
        self.aurora = aurora if aurora is not None else nn.Identity()
        self.packing = packing
        self.refinement_config = refinement or RefinementConfig()
        self.performance_config = performance or PerformanceConfig()
        self.target_space = NormalizedTargetSpace(
            packing, nonnegative_variables=nonnegative_variables
        )
        self.conditioning_static_names = tuple(
            dict.fromkeys(str(name) for name in conditioning_static_names)
        )
        self.refiner: ResidualRefiner | None = None
        objective = dict(_TEMPORAL_OBJECTIVE_DEFAULTS)
        if temporal_objective is not None:
            objective.update(temporal_objective)
        self.temporal_config = {
            "enabled": bool(temporal_enabled),
            "mode": str(temporal_mode).strip().lower(),
            "channels": int(temporal_channels),
            "state": int(temporal_state),
            "state_dim": int(temporal_state),
            "layers": int(temporal_layers),
            "num_layers": int(temporal_layers),
            "conv": int(temporal_conv),
            "conv_kernel": int(temporal_conv),
            "expand": int(temporal_expand),
            "expansion_factor": int(temporal_expand),
            "dropout": float(temporal_dropout),
            "gated_fusion": bool(temporal_gated_fusion),
            "gate_init": float(temporal_gate_init),
            "lead_time_conditioning": bool(temporal_lead_time_conditioning),
            "mask_conditioning": bool(temporal_mask_conditioning),
            "coordinate_conditioning": bool(
                temporal_coordinate_conditioning
            ),
            "causal": bool(temporal_causal),
            "learning_rate_multiplier": float(
                temporal_learning_rate_multiplier
            ),
            "objective": objective,
            "semantic_version": int(temporal_semantic_version),
            "scan_backend": str(temporal_scan_backend),
        }
        if self.temporal_config["enabled"] and not self.refinement_config.is_active:
            raise ValueError(
                "model.mamba_temporal_enabled=true requires an active "
                "model.refinement.type; actual refinement type is 'none'."
            )
        for temporal_field in ("channels", "state", "layers", "conv", "expand"):
            value = self.temporal_config[temporal_field]
            if value < 1:
                raise ValueError(
                    f"model.mamba_temporal_{temporal_field} must be a positive integer, "
                    f"got {value!r}."
                )
        if self.temporal_config["mode"] not in {"per_variable", "packed_joint"}:
            raise ValueError(
                "model.mamba_temporal.mode must be per_variable or packed_joint."
            )
        if not self.temporal_config["causal"]:
            raise ValueError(
                "model.mamba_temporal.causal=false is unsupported for rollout inference."
            )
        dropout = self.temporal_config["dropout"]
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("model.mamba_temporal.dropout must be in [0, 1).")
        gate_init = self.temporal_config["gate_init"]
        if not math.isfinite(gate_init) or not 0.0 <= gate_init < 1.0:
            raise ValueError("model.mamba_temporal.gate_init must be in [0, 1).")
        lr_multiplier = self.temporal_config["learning_rate_multiplier"]
        if not math.isfinite(lr_multiplier) or lr_multiplier <= 0.0:
            raise ValueError(
                "model.mamba_temporal.learning_rate_multiplier must be positive."
            )
        self.temporal: PackedMambaTemporalAdapter | None = None
        if self.temporal_config["enabled"]:
            self.temporal = PackedMambaTemporalAdapter(
                packing,
                channels=self.temporal_config["channels"],
                d_state=self.temporal_config["state"],
                n_layers=self.temporal_config["layers"],
                d_conv=self.temporal_config["conv"],
                expand=self.temporal_config["expand"],
                dropout=self.temporal_config["dropout"],
                mode=self.temporal_config["mode"],
                gated_fusion=self.temporal_config["gated_fusion"],
                gate_init=self.temporal_config["gate_init"],
                lead_time_conditioning=self.temporal_config["lead_time_conditioning"],
                mask_conditioning=self.temporal_config["mask_conditioning"],
                coordinate_conditioning=self.temporal_config[
                    "coordinate_conditioning"
                ],
                causal=self.temporal_config["causal"],
            )
        self._area_weight_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._apply_aurora_freeze()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _apply_aurora_freeze(self) -> None:
        """Freeze Aurora when Phase 2 is trained on top of it."""
        self.aurora_frozen = bool(
            self.refinement_config.is_active and self.refinement_config.freeze_aurora
        )
        if self.aurora_frozen:
            for param in self.aurora.parameters():
                param.requires_grad_(False)
            self.aurora.eval()

    def train(self, mode: bool = True):  # noqa: D102 - torch API
        super().train(mode)
        if self.aurora_frozen:
            # A frozen Aurora must stay in eval() so dropout / normalization
            # layers cannot perturb the deterministic conditioning.
            self.aurora.eval()
        return self

    def conditioning_channels(
        self,
        *,
        static_fields: int | None = None,
        input_state_channels: int | None = None,
    ) -> int:
        """Number of packed spatial conditioning channels for this configuration."""
        cond = self.refinement_config.conditioning
        total = 0
        if cond.aurora_rollout:
            total += self.packing.num_channels
        if cond.aurora_input_state:
            total += int(
                self.packing.num_channels
                if input_state_channels is None
                else input_state_channels
            )
        if cond.static_fields:
            total += int(
                len(self.conditioning_static_names)
                if static_fields is None
                else static_fields
            )
        if cond.latitude:
            total += 1
        if cond.longitude:
            total += 2
        if cond.masks:
            total += 1
        return total

    def initialize_refiner(self, cond_channels: int) -> None:
        """Construct the Phase-2 network for a known conditioning width.

        Idempotent for a matching width; raises if called again with a different
        width after parameters were created, because that would silently discard
        trained weights.
        """
        if not self.refinement_config.is_active:
            return
        if self.refiner is not None:
            if self.refiner.cond_channels == int(cond_channels):
                return
            raise RuntimeError(
                f"Refiner was built for {self.refiner.cond_channels} conditioning "
                f"channels but the batch provides {cond_channels}. The dataset or the "
                "conditioning configuration changed after construction."
            )
        self.refiner = build_refiner(
            self.refinement_config,
            residual_channels=self.packing.num_channels,
            cond_channels=int(cond_channels),
            metadata=self.packing,
        )

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    @staticmethod
    def _coordinate_vector(
        name: str,
        values: Sequence[float],
        expected_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Validate one FieldPacking coordinate axis and move it to the field device."""
        if len(values) != int(expected_size):
            raise ValueError(
                f"refinement.conditioning.{name} requires one FieldPacking {name} "
                f"coordinate per grid cell; expected {expected_size}, got {len(values)}."
            )
        coordinate = torch.as_tensor(
            tuple(values),
            device=reference.device,
            dtype=torch.float32,
        )
        if coordinate.ndim != 1 or not bool(torch.isfinite(coordinate).all()):
            raise ValueError(
                f"FieldPacking {name} coordinates must be a finite one-dimensional "
                "sequence."
            )
        return coordinate

    def build_conditioning(
        self,
        rollout_normalized: torch.Tensor,
        *,
        input_state_normalized: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        input_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate the configured spatial conditioning fields.

        Only fields available *at the forecast valid time without looking at the
        target* are used. The ground-truth mask is never conditioning: that
        would leak target information. Enabled coordinate channels are appended
        after dynamic/static fields in latitude, sin(longitude), cos(longitude)
        order, followed by the validity mask.
        """
        cond_cfg = self.refinement_config.conditioning
        if (cond_cfg.latitude or cond_cfg.longitude) and rollout_normalized.ndim != 4:
            raise ValueError(
                "Coordinate conditioning requires rollout_normalized shaped "
                f"[B, C, H, W], got {tuple(rollout_normalized.shape)}."
            )
        size = (int(rollout_normalized.shape[-2]), int(rollout_normalized.shape[-1]))
        dtype = rollout_normalized.dtype
        parts: list[torch.Tensor] = []

        if cond_cfg.aurora_rollout:
            parts.append(rollout_normalized)

        if cond_cfg.aurora_input_state:
            if input_state_normalized is None:
                raise RuntimeError(
                    "refinement.conditioning.aurora_input_state is enabled but no "
                    "input state was supplied. It must be the Aurora *input* state of "
                    "the rollout step (never a future target)."
                )
            state = torch.nan_to_num(input_state_normalized.to(dtype), nan=0.0)
            parts.append(_align_to(state, size))

        if cond_cfg.static_fields:
            if static_fields is None:
                raise RuntimeError(
                    "refinement.conditioning.static_fields is enabled but no static "
                    "fields were supplied."
                )
            statics = torch.nan_to_num(static_fields.to(dtype), nan=0.0)
            parts.append(_align_to(statics, size))

        if cond_cfg.latitude:
            latitude = self._coordinate_vector(
                "latitude",
                self.packing.lat,
                size[0],
                rollout_normalized,
            )
            if bool((latitude.abs() > 90.0).any()):
                raise ValueError(
                    "FieldPacking latitude coordinates must lie within "
                    "[-90, 90] degrees."
                )
            latitude = (latitude / 90.0).to(dtype=dtype)
            parts.append(
                latitude.view(1, 1, size[0], 1).expand(
                    rollout_normalized.shape[0], 1, size[0], size[1]
                )
            )

        if cond_cfg.longitude:
            longitude = self._coordinate_vector(
                "longitude",
                self.packing.lon,
                size[1],
                rollout_normalized,
            )
            longitude_radians = torch.deg2rad(longitude)
            for periodic_coordinate in (
                torch.sin(longitude_radians),
                torch.cos(longitude_radians),
            ):
                parts.append(
                    periodic_coordinate.to(dtype=dtype)
                    .view(1, 1, 1, size[1])
                    .expand(rollout_normalized.shape[0], 1, size[0], size[1])
                )

        if cond_cfg.masks:
            if input_valid_mask is None:
                mask = torch.isfinite(rollout_normalized).all(dim=1, keepdim=True).to(dtype)
            else:
                mask = input_valid_mask.to(dtype)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                if mask.shape[1] != 1:
                    mask = mask.all(dim=1, keepdim=True).to(dtype)
            parts.append(_align_to(mask, size))

        conditioning = torch.cat(parts, dim=1)
        return torch.nan_to_num(conditioning, nan=0.0, posinf=0.0, neginf=0.0)

    def area_weight(self, height: int, device, dtype) -> torch.Tensor | None:
        key = (int(height), torch.device(device), dtype)
        if key not in self._area_weight_cache:
            weights = area_weights_from_latitudes(
                self.packing.lat, height, device=device, dtype=dtype
            )
            if weights is None:
                return None
            self._area_weight_cache[key] = weights
        return self._area_weight_cache[key]

    # ------------------------------------------------------------------
    @staticmethod
    def _forecast_lead_vector(
        forecast_lead_time: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if forecast_lead_time is None:
            return None
        lead = forecast_lead_time.detach().reshape(-1).to(
            device=device, dtype=torch.float32
        )
        if lead.numel() == 1:
            lead = lead.expand(batch_size)
        elif lead.numel() != batch_size:
            raise ValueError(
                "forecast_lead_time must contain one value or one value per "
                f"packed sample; got {lead.numel()} for batch {batch_size}."
            )
        if not bool(torch.isfinite(lead).all()) or bool((lead < 0.0).any()):
            raise ValueError(
                "forecast_lead_time values must be finite and non-negative."
            )
        return lead

    # Training
    # ------------------------------------------------------------------
    def training_step(
        self,
        rollout_normalized: torch.Tensor,
        target_normalized: torch.Tensor,
        *,
        conditioning: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        input_state_normalized: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        forecast_lead_time: torch.Tensor | None = None,
        lead_index: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TwoPhaseStepOutput:
        """One Phase-2 training-objective evaluation for a set of rollout steps.

        ``rollout_normalized`` and ``target_normalized`` are packed
        ``[N, C, H, W]`` tensors that refer to the **same forecast valid time**
        per leading entry. ``forecast_lead_time`` carries the matching physical
        lead time in hours; ``lead_index`` the rollout-step index used only to
        group the bias-aware auxiliary losses.
        """
        if not self.refinement_config.is_active:
            raise RuntimeError(
                "training_step() requires an active refinement configuration; use the "
                "deterministic training path for Aurora-only runs."
            )
        rollout = rollout_normalized.detach().float()
        target = target_normalized.float()
        if conditioning is None:
            conditioning = self.build_conditioning(
                rollout,
                input_state_normalized=input_state_normalized,
                static_fields=static_fields,
            )
        forecast_lead_time = self._forecast_lead_vector(
            forecast_lead_time,
            batch_size=conditioning.shape[0],
            device=conditioning.device,
        )
        self.initialize_refiner(conditioning.shape[1])
        assert self.refiner is not None

        correction_target, valid = (
            self.target_space.correction_target_from_normalized(
                target, rollout, valid_mask=valid_mask
            )
        )
        if self.refinement_config.train_on_residual:
            process_target = correction_target
        else:
            # Deprecated whole-field checkpoints keep their historical training
            # target, while all public semantics still expose CAMS - Aurora.
            process_target = torch.where(
                valid, target, torch.zeros_like(target)
            )

        result = self.refiner.compute_training_loss(
            process_target,
            conditioning,
            forecast_lead_time=forecast_lead_time,
            mask=valid,
            generator=generator,
            lead_index=lead_index,
            area_weight=self.area_weight(
                rollout.shape[-2], rollout.device, torch.float32
            ),
            rollout_normalized=rollout,
        )
        estimated_process_target = result.predicted_correction_normalized
        estimated_correction = None
        estimated_correction_physical = None
        if estimated_process_target is not None:
            estimated_correction = (
                estimated_process_target
                if self.refinement_config.train_on_residual
                else estimated_process_target - rollout
            )
            estimated_correction_physical = self.target_space.correction_to_physical(
                estimated_correction
            )

        losses = {
            name: value
            for name, value in (
                ("generative_loss", result.generative_loss),
                ("reconstruction_loss", result.reconstruction_loss),
                ("bias_loss", result.bias_loss),
                ("gradient_loss", result.gradient_loss),
                ("pattern_correlation_loss", result.pattern_correlation_loss),
                ("deterministic_loss", result.deterministic_loss),
                ("mae_loss", result.mae_loss),
                ("extreme_loss", result.extreme_loss),
                ("peak_loss", result.peak_loss),
                ("quantile_loss", result.quantile_loss),
                ("variance_loss", result.variance_loss),
                ("spectral_loss", result.spectral_loss),
                ("degradation_loss", result.degradation_loss),
                ("magnitude_loss", result.magnitude_loss),
                ("total_loss", result.total_loss),
            )
            if value is not None
        }
        return TwoPhaseStepOutput(
            deterministic_normalized=rollout,
            deterministic_physical=self.target_space.decode(rollout),
            correction_target_normalized=correction_target,
            correction_target_physical=self.target_space.correction_to_physical(
                correction_target
            ),
            residual_target=correction_target,
            predicted_correction_normalized=estimated_correction,
            predicted_correction_physical=estimated_correction_physical,
            residual=estimated_correction,
            valid_mask=valid,
            conditioning=conditioning,
            process_time=result.process_time,
            losses=losses,
        )

    @property
    def has_temporal(self) -> bool:
        """Whether an eagerly registered packed temporal corrector is active."""
        return self.temporal is not None

    @staticmethod
    def lead_major_to_sequence(
        tensor: torch.Tensor,
        lead_index: torch.Tensor,
        *,
        field_name: str = "tensor",
    ) -> torch.Tensor:
        """Unfold lead-major folded batches into [batch, lead, ...]."""
        if tensor.ndim < 1:
            raise ValueError(
                f"{field_name} must have a leading folded batch dimension, "
                f"got shape {tuple(tensor.shape)}."
            )
        index = lead_index.reshape(-1).to(device=tensor.device, dtype=torch.long)
        if index.numel() != tensor.shape[0]:
            raise ValueError(
                f"{field_name} leading dimension is {tensor.shape[0]}, but "
                f"lead_index contains {index.numel()} values."
            )
        unique, counts = torch.unique_consecutive(index, return_counts=True)
        expected = torch.arange(unique.numel(), device=index.device)
        if unique.numel() < 2:
            raise ValueError(
                f"{field_name} temporal sequence requires at least 2 ordered "
                f"lead indices, got {unique.tolist()}."
            )
        if not torch.equal(unique, expected):
            raise ValueError(
                f"{field_name} lead_index must be lead-major contiguous indices "
                f"[0, ..., S-1], got {unique.tolist()}."
            )
        if not torch.all(counts == counts[0]):
            raise ValueError(
                f"{field_name} must contain the same batch size for every lead; "
                f"got per-lead counts {counts.tolist()}."
            )
        steps = int(unique.numel())
        batch = int(counts[0])
        return (
            tensor.reshape(steps, batch, *tensor.shape[1:])
            .transpose(0, 1)
            .contiguous()
        )

    def temporal_training_loss(
        self,
        rollout_normalized: torch.Tensor,
        target_normalized: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        forecast_lead_time: torch.Tensor | None,
        lead_index: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Masked cross-lead loss on target-independent spatial refinements.

        The spatial refiner is evaluated without gradients and detached, so
        this loss updates only Mamba. It never reuses the diffusion training
        denoised estimate, whose noised input contains target data.
        """
        if self.temporal is None:
            raise RuntimeError(
                "temporal_training_loss() requires "
                "model.mamba_temporal_enabled=true."
            )
        if target_normalized.shape != rollout_normalized.shape:
            raise ValueError(
                "target_normalized shape must match rollout_normalized for "
                f"temporal training; expected {tuple(rollout_normalized.shape)}, "
                f"got {tuple(target_normalized.shape)}."
            )
        if valid_mask.shape != rollout_normalized.shape:
            raise ValueError(
                "valid_mask shape must match rollout_normalized for temporal "
                f"training; expected {tuple(rollout_normalized.shape)}, got "
                f"{tuple(valid_mask.shape)}."
            )
        rollout = rollout_normalized.detach()
        with torch.no_grad():
            if conditioning is None:
                conditioning = self.build_conditioning(rollout)
            else:
                conditioning = conditioning.detach()
                if conditioning.ndim != 4:
                    raise ValueError(
                        "conditioning must have shape [folded_batch, channel, "
                        f"latitude, longitude], got {tuple(conditioning.shape)}."
                    )
                expected = (
                    rollout.shape[0],
                    rollout.shape[-2],
                    rollout.shape[-1],
                )
                actual = (
                    conditioning.shape[0],
                    conditioning.shape[-2],
                    conditioning.shape[-1],
                )
                if actual != expected:
                    raise ValueError(
                        "conditioning batch/spatial dimensions must match "
                        f"rollout_normalized; expected {expected}, got {actual}."
                    )
                if conditioning.device != rollout.device:
                    raise ValueError(
                        "conditioning must be on the same device as "
                        f"rollout_normalized; expected {rollout.device}, got "
                        f"{conditioning.device}."
                    )
            forecast_lead_time = self._forecast_lead_vector(
                forecast_lead_time,
                batch_size=conditioning.shape[0],
                device=conditioning.device,
            )
            self.initialize_refiner(conditioning.shape[1])
            assert self.refiner is not None
            # Disable stochastic layers only on the spatial networks. Calling
            # refiner.eval() recursively also switches ResidualScaler to
            # inference mode, where an intentionally uncalibrated training
            # smoke batch must fail. Production validation still requires the
            # scaler to have been fitted and frozen.
            inference_modules = [
                module
                for module in (
                    getattr(self.refiner, "net", None),
                    getattr(self.refiner, "mean_net", None),
                    getattr(self.refiner, "legacy", None),
                )
                if isinstance(module, nn.Module)
            ]
            training_modes = [module.training for module in inference_modules]
            try:
                for module in inference_modules:
                    module.eval()
                # Train temporal Mamba on the same spatial-refiner distribution
                # it receives during rollout inference.
                if self.refinement_config.deterministic_inference:
                    model_output = self.refiner.deterministic_residual(
                        conditioning, forecast_lead_time=forecast_lead_time
                    )
                else:
                    model_output = self.refiner.sample_residual(
                        conditioning,
                        forecast_lead_time=forecast_lead_time,
                        generator=generator,
                        num_steps=num_steps,
                    )
            finally:
                for module, was_training in zip(
                    inference_modules, training_modes
                ):
                    module.train(was_training)
            base_refined = (
                rollout + model_output
                if self.refinement_config.train_on_residual
                else model_output
            ).detach()

        sequence = self.lead_major_to_sequence(
            base_refined, lead_index, field_name="base_refined"
        )
        target_sequence = self.lead_major_to_sequence(
            target_normalized.detach(), lead_index, field_name="target_normalized"
        )
        mask_sequence = self.lead_major_to_sequence(
            valid_mask.to(dtype=torch.bool), lead_index, field_name="valid_mask"
        )
        finite_base = torch.isfinite(sequence)
        input_step_mask = finite_base.flatten(2).any(dim=2)
        lead_sequence = (
            self.lead_major_to_sequence(
                forecast_lead_time, lead_index, field_name="forecast_lead_time"
            )
            if forecast_lead_time is not None
            else None
        )
        corrected = self.temporal.corrected_sequence(
            sequence,
            lead_hours=lead_sequence,
            valid_cell_mask=finite_base,
            valid_step_mask=input_step_mask,
        )
        valid = (
            mask_sequence
            & finite_base
            & torch.isfinite(target_sequence)
            & torch.isfinite(corrected)
        )
        if not bool(valid.any()):
            raise ValueError(
                "valid_mask contains zero finite cells across the temporal "
                f"sequence with shape {tuple(valid.shape)}."
            )
        safe_corrected = torch.where(
            valid, corrected.float(), torch.zeros_like(corrected, dtype=torch.float32)
        )
        safe_target = torch.where(
            valid,
            target_sequence.float(),
            torch.zeros_like(target_sequence, dtype=torch.float32),
        )
        objective = self.temporal_config["objective"]

        def _point_error(
            prediction: torch.Tensor, truth: torch.Tensor
        ) -> torch.Tensor:
            difference = prediction - truth
            base_name = objective["base"]
            if base_name == "mse":
                return difference.square()
            if base_name == "huber":
                return F.huber_loss(
                    prediction,
                    truth,
                    reduction="none",
                    delta=objective["huber_delta"],
                )
            epsilon = objective["charbonnier_epsilon"]
            return torch.sqrt(difference.square() + epsilon**2) - epsilon

        point_error = _point_error(safe_corrected, safe_target)

        diagnostics: dict[str, torch.Tensor] = {}
        variable_losses: list[torch.Tensor] = []
        for name in self.packing.variables:
            indices = [
                spec.index
                for spec in self.packing.channels
                if spec.aurora_name == name
            ]
            var_valid = valid[:, :, indices]
            var_squared = point_error[:, :, indices]
            var_count = var_valid.sum()
            if int(var_count.detach()) == 0:
                continue
            var_denom = var_count.to(var_squared.dtype)
            var_loss = (
                var_squared * var_valid.to(var_squared.dtype)
            ).sum() / var_denom
            diagnostics[f"temporal_loss/{name}"] = var_loss
            variable_losses.append(var_loss)
        if not variable_losses:
            raise ValueError(
                "No configured target variable has a finite valid cell for the "
                "temporal loss."
            )
        # Match the legacy flow-Mamba contract: each target variable contributes
        # equally, regardless of how many atmospheric levels it packs.
        base_loss = torch.stack(variable_losses).mean()
        diagnostics["temporal_base_loss"] = base_loss

        def _variable_equal_mean(
            error: torch.Tensor, error_mask: torch.Tensor
        ) -> torch.Tensor:
            terms: list[torch.Tensor] = []
            for variable_name in self.packing.variables:
                channel_indices = [
                    spec.index
                    for spec in self.packing.channels
                    if spec.aurora_name == variable_name
                ]
                selected_mask = error_mask[:, :, channel_indices]
                count = selected_mask.sum()
                if int(count.detach()) == 0:
                    continue
                selected_error = error[:, :, channel_indices]
                terms.append(
                    (
                        selected_error
                        * selected_mask.to(dtype=selected_error.dtype)
                    ).sum()
                    / count.to(dtype=selected_error.dtype)
                )
            if not terms:
                return error.sum() * 0.0
            return torch.stack(terms).mean()

        total_loss = base_loss
        tendency_weight = float(objective["tendency_weight"])
        if tendency_weight > 0.0:
            tendency_mask = valid[:, 1:] & valid[:, :-1]
            predicted_tendency = safe_corrected[:, 1:] - safe_corrected[:, :-1]
            target_tendency = safe_target[:, 1:] - safe_target[:, :-1]
            tendency_loss = _variable_equal_mean(
                _point_error(predicted_tendency, target_tendency),
                tendency_mask,
            )
            diagnostics["temporal_tendency_loss"] = tendency_loss
            total_loss = total_loss + tendency_weight * tendency_loss

        structure_weight = float(objective["structure_weight"])
        if structure_weight > 0.0:
            structure_terms: list[torch.Tensor] = []
            if safe_corrected.shape[-2] > 1:
                latitude_mask = valid[..., 1:, :] & valid[..., :-1, :]
                structure_terms.append(
                    _variable_equal_mean(
                        _point_error(
                            safe_corrected[..., 1:, :] - safe_corrected[..., :-1, :],
                            safe_target[..., 1:, :] - safe_target[..., :-1, :],
                        ),
                        latitude_mask,
                    )
                )
            if safe_corrected.shape[-1] > 1:
                longitude_mask = valid[..., 1:] & valid[..., :-1]
                structure_terms.append(
                    _variable_equal_mean(
                        _point_error(
                            safe_corrected[..., 1:] - safe_corrected[..., :-1],
                            safe_target[..., 1:] - safe_target[..., :-1],
                        ),
                        longitude_mask,
                    )
                )
                if self.packing.lon_periodic:
                    longitude_seam_mask = valid[..., :1] & valid[..., -1:]
                    structure_terms.append(
                        _variable_equal_mean(
                            _point_error(
                                safe_corrected[..., :1] - safe_corrected[..., -1:],
                                safe_target[..., :1] - safe_target[..., -1:],
                            ),
                            longitude_seam_mask,
                        )
                    )
            structure_loss = (
                torch.stack(structure_terms).mean()
                if structure_terms
                else base_loss * 0.0
            )
            diagnostics["temporal_structure_loss"] = structure_loss
            total_loss = total_loss + structure_weight * structure_loss

        extreme_weight = float(objective["extreme_weight"])
        if extreme_weight > 0.0:
            extreme_terms: list[torch.Tensor] = []
            quantile = float(objective["extreme_quantile"])
            for variable_name in self.packing.variables:
                channel_indices = [
                    spec.index
                    for spec in self.packing.channels
                    if spec.aurora_name == variable_name
                ]
                variable_valid = valid[:, :, channel_indices]
                if not bool(variable_valid.any()):
                    continue
                variable_target = safe_target[:, :, channel_indices]
                threshold = torch.quantile(
                    variable_target[variable_valid], quantile
                )
                tail_mask = variable_valid & (variable_target >= threshold)
                variable_error = point_error[:, :, channel_indices]
                extreme_terms.append(
                    (variable_error * tail_mask.to(variable_error.dtype)).sum()
                    / tail_mask.sum().to(variable_error.dtype)
                )
            extreme_loss = (
                torch.stack(extreme_terms).mean()
                if extreme_terms
                else base_loss * 0.0
            )
            diagnostics["temporal_extreme_loss"] = extreme_loss
            total_loss = total_loss + extreme_weight * extreme_loss

        diagnostics["temporal_total_loss"] = total_loss
        return total_loss, diagnostics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refine(
        self,
        rollout_normalized: torch.Tensor,
        *,
        conditioning: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        input_state_normalized: torch.Tensor | None = None,
        forecast_lead_time: torch.Tensor | None = None,
        ensemble_size: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        target_mask: torch.Tensor | None = None,
        return_members: bool = True,
        chunk_size: int | None = None,
        num_steps: int | None = None,
        temporal_history: list[torch.Tensor] | None = None,
        temporal_lead_history: list[torch.Tensor] | None = None,
        temporal_control: str = "on",
    ) -> TwoPhaseStepOutput:
        """Deterministic plus (optionally) refined ensemble inference.

        The deterministic rollout is always returned unchanged; refinement only
        adds fields.
        """
        temporal_control = str(temporal_control).strip().lower()
        if temporal_control not in {"on", "off", "shuffled"}:
            raise ValueError(
                "temporal_control must be on, off, or shuffled; "
                f"got {temporal_control!r}."
            )
        rollout = rollout_normalized.float()
        out = TwoPhaseStepOutput(
            deterministic_normalized=rollout,
            deterministic_physical=self.target_space.decode(rollout),
        )
        if not self.refinement_config.is_active:
            return out
        if conditioning is None:
            conditioning = self.build_conditioning(
                rollout,
                input_state_normalized=input_state_normalized,
                static_fields=static_fields,
            )
        forecast_lead_time = self._forecast_lead_vector(
            forecast_lead_time,
            batch_size=conditioning.shape[0],
            device=conditioning.device,
        )
        self.initialize_refiner(conditioning.shape[1])
        assert self.refiner is not None

        # The systematic correction is always produced by the separately
        # supervised conditional-mean head. Stochastic members add innovations
        # around this stable field; no arbitrary draw is used as bias correction.
        mean_process_output = self._deterministic_mean_process(
            conditioning, rollout, forecast_lead_time
        ).float()
        conditional_mean_correction = (
            mean_process_output
            if self.refinement_config.train_on_residual
            else mean_process_output - rollout
        ).float()
        (
            deterministic_refined_normalized,
            deterministic_refined_physical,
        ) = self.target_space.apply_correction(
            rollout,
            conditional_mean_correction,
            target_mask=target_mask,
        )

        n = int(
            ensemble_size
            if ensemble_size is not None
            else self.refinement_config.ensemble_size
        )
        if n < 1:
            return out

        batch = conditioning.shape[0]
        if self.refinement_config.deterministic_inference:
            member_innovations = torch.zeros_like(mean_process_output).unsqueeze(1).expand(
                -1, n, -1, -1, -1
            ).clone()
        else:
            if generator is None:
                seed_value = (
                    seed if seed is not None else self.refinement_config.seed
                )
                if seed_value is not None:
                    generator = torch.Generator(device=conditioning.device)
                    generator.manual_seed(int(seed_value))

            perf = self.performance_config.ensemble
            if chunk_size is None:
                chunk_size = (
                    perf.chunk_size
                    if perf.chunk_size is not None
                    else (n if perf.batch_members else 1)
                )
            chunk_size = max(1, min(int(chunk_size), n))
            # The legacy adapter owns a single-generator sampler rather than
            # the common noise-source interface. Serialize its members so each
            # member-specific generator is forwarded without changing member
            # ordering when callers request batched chunks.
            if self.refinement_config.is_legacy_flow_matching:
                chunk_size = 1

            # Dedicated member generators keep serial and chunked draws identical.
            gen_device = (
                generator.device
                if generator is not None
                else conditioning.device
            )
            if generator is not None:
                member_seeds = torch.randint(
                    0,
                    2**62,
                    (n,),
                    generator=generator,
                    device=gen_device,
                    dtype=torch.int64,
                ).tolist()
            else:
                member_seeds = torch.randint(
                    0, 2**62, (n,), dtype=torch.int64
                ).tolist()
            member_generators = []
            for member_seed in member_seeds:
                member_generator = torch.Generator(device=gen_device)
                member_generator.manual_seed(int(member_seed))
                member_generators.append(member_generator)

            sampled_innovations: list[torch.Tensor] = []
            drawn = 0
            while drawn < n:
                member_count = min(chunk_size, n - drawn)
                source = ChunkNoiseSource(
                    member_generators[drawn : drawn + member_count], batch
                )
                if member_count == 1:
                    with self.refiner.use_noise_source(source):
                        innovation = self._sample_innovation(
                            conditioning,
                            rollout,
                            forecast_lead_time,
                            num_steps,
                            mean_process_output,
                            generator=member_generators[drawn],
                        )
                    sampled_innovations.append(innovation.float().unsqueeze(1))
                else:
                    cond_rep = conditioning.repeat_interleave(
                        member_count, dim=0
                    )
                    rollout_rep = rollout.repeat_interleave(
                        member_count, dim=0
                    )
                    lead_rep = (
                        forecast_lead_time.repeat_interleave(
                            member_count, dim=0
                        )
                        if forecast_lead_time is not None
                        else None
                    )
                    mean_rep = mean_process_output.repeat_interleave(
                        member_count, dim=0
                    )
                    with self.refiner.use_noise_source(source):
                        innovation = self._sample_innovation(
                            cond_rep,
                            rollout_rep,
                            lead_rep,
                            num_steps,
                            mean_rep,
                            generator=None,
                        )
                    innovation = innovation.float().reshape(
                        batch, member_count, *innovation.shape[1:]
                    )
                    sampled_innovations.append(innovation)
                drawn += member_count
            member_innovations = torch.cat(sampled_innovations, dim=1)

        member_corrections = (
            conditional_mean_correction.unsqueeze(1) + member_innovations
        )
        if (
            self.refinement_config.target_space.residual_clip_standard_deviations
            > 0.0
        ):
            member_corrections = self._guard_deployed_member_corrections(
                member_corrections
            )
            member_innovations = (
                member_corrections
                - conditional_mean_correction.unsqueeze(1)
            )
        member_normalized = rollout.unsqueeze(1) + member_corrections
        if self.temporal is not None:
            history = temporal_history if temporal_history is not None else []
            lead_history = (
                temporal_lead_history
                if temporal_lead_history is not None
                else []
            )
            expected_shape = tuple(member_normalized.shape)
            for history_index, frame in enumerate(history):
                if tuple(frame.shape) != expected_shape:
                    raise ValueError(
                        "Temporal ensemble history shape mismatch at index "
                        f"{history_index}: expected {expected_shape}, got "
                        f"{tuple(frame.shape)}."
                    )
                if frame.device != member_normalized.device:
                    raise ValueError(
                        "Temporal ensemble history device mismatch at index "
                        f"{history_index}: expected {member_normalized.device}, "
                        f"got {frame.device}."
                    )
                if frame.dtype != member_normalized.dtype:
                    raise ValueError(
                        "Temporal ensemble history dtype mismatch at index "
                        f"{history_index}: expected {member_normalized.dtype}, "
                        f"got {frame.dtype}."
                    )
            if temporal_lead_history is not None and len(lead_history) != len(history):
                raise ValueError(
                    "temporal_lead_history length must match temporal_history; "
                    f"got {len(lead_history)} and {len(history)}."
                )
            if (
                history
                and self.temporal.lead_time_conditioning
                and temporal_lead_history is None
            ):
                raise ValueError(
                    "Packed temporal lead conditioning requires one physical lead "
                    "entry for every temporal history frame."
                )
            current_raw = member_normalized.detach()
            temporal_sequence = torch.stack([*history, current_raw], dim=2)
            batch_size, members, steps, channels, height, width = (
                temporal_sequence.shape
            )
            temporal_leads: torch.Tensor | None = None
            if forecast_lead_time is not None and (
                temporal_lead_history is not None
                or self.temporal.lead_time_conditioning
            ):
                resolved_leads = [
                    self._forecast_lead_vector(
                        entry,
                        batch_size=batch_size,
                        device=member_normalized.device,
                    )
                    for entry in lead_history
                ]
                if any(entry is None for entry in resolved_leads):
                    raise ValueError("Temporal lead history contains a missing lead.")
                lead_frames = [
                    entry for entry in resolved_leads if entry is not None
                ]
                lead_frames.append(forecast_lead_time)
                lead_sequence = torch.stack(lead_frames, dim=1)
                temporal_leads = (
                    lead_sequence[:, None, :]
                    .expand(batch_size, members, steps)
                    .reshape(batch_size * members, steps)
                )
            elif lead_history or self.temporal.lead_time_conditioning:
                raise ValueError(
                    "Physical forecast leads are required by temporal lead "
                    "conditioning and its history contract."
                )

            validate_lead_order = True
            if temporal_control == "shuffled":
                # Deterministic negative control: permute only the available past;
                # the current lead remains last, so no future state is introduced.
                order = torch.cat(
                    (
                        torch.arange(
                            steps - 2,
                            -1,
                            -1,
                            device=temporal_sequence.device,
                        ),
                        torch.tensor(
                            [steps - 1], device=temporal_sequence.device
                        ),
                    )
                )
                temporal_sequence = temporal_sequence.index_select(2, order)
                if temporal_leads is not None:
                    temporal_leads = (
                        temporal_leads.reshape(batch_size, members, steps)
                        .index_select(2, order)
                        .reshape(batch_size * members, steps)
                    )
                validate_lead_order = False

            flat_sequence = temporal_sequence.reshape(
                batch_size * members, steps, channels, height, width
            )
            temporal_input_mask = torch.isfinite(flat_sequence)
            temporal_step_mask = temporal_input_mask.flatten(2).any(dim=2)
            if temporal_control == "off":
                temporal_correction = torch.zeros_like(member_normalized)
            else:
                temporal_correction = self.temporal.causal_residual(
                    flat_sequence,
                    lead_hours=temporal_leads,
                    valid_cell_mask=temporal_input_mask,
                    valid_step_mask=temporal_step_mask,
                    validate_lead_order=validate_lead_order,
                ).reshape(
                    batch_size, members, channels, height, width
                ).float()
            member_normalized = member_normalized.float() + temporal_correction
            member_corrections = member_normalized - rollout.unsqueeze(1)
            # Temporal post-processing is itself a residual correction. Guard
            # the complete deployed amplitude again after it has been added so
            # Mamba cannot bypass the training-residual range safeguard.
            if (
                self.refinement_config.target_space.residual_clip_standard_deviations
                > 0.0
            ):
                member_corrections = self._guard_deployed_member_corrections(
                    member_corrections
                )
                member_normalized = rollout.unsqueeze(1) + member_corrections
            member_innovations = (
                member_corrections
                - conditional_mean_correction.unsqueeze(1)
            )
            history.append(current_raw)
            if temporal_lead_history is not None:
                if forecast_lead_time is None:
                    raise ValueError(
                        "temporal_lead_history requires forecast_lead_time."
                    )
                lead_history.append(forecast_lead_time.detach())
            if self.refinement_config.deterministic_inference:
                conditional_mean_correction = member_corrections[:, 0]
                member_innovations = (
                    member_corrections
                    - conditional_mean_correction.unsqueeze(1)
                )
                (
                    deterministic_refined_normalized,
                    deterministic_refined_physical,
                ) = self.target_space.apply_correction(
                    rollout,
                    conditional_mean_correction,
                    target_mask=target_mask,
                )
        conditional_mean_source = "supervised_mean_head"
        flat_shape = (
            batch * n,
            member_corrections.shape[2],
            member_corrections.shape[3],
            member_corrections.shape[4],
        )
        member_corrections_physical = (
            self.target_space.correction_to_physical(
                member_corrections.reshape(flat_shape)
            ).reshape_as(member_corrections)
        )
        member_innovations_physical = (
            self.target_space.correction_to_physical(
                member_innovations.reshape(flat_shape)
            ).reshape_as(member_innovations)
        )

        member_fields: list[torch.Tensor] = []
        constraint_masks: list[torch.Tensor] = []
        for member_index in range(n):
            _, unconstrained = self.target_space.apply_correction(
                rollout,
                member_corrections[:, member_index],
                target_mask=None,
                apply_constraints=False,
            )
            constraint_masks.append(
                self.target_space.physical_constraint_mask(
                    unconstrained
                ).unsqueeze(1)
            )
            _, constrained = self.target_space.apply_correction(
                rollout,
                member_corrections[:, member_index],
                target_mask=target_mask,
                apply_constraints=True,
            )
            member_fields.append(constrained.unsqueeze(1))
        members_physical = torch.cat(member_fields, dim=1)
        changed = torch.cat(constraint_masks, dim=1)
        changed_per_channel = changed.sum(dim=(0, 1, 3, 4)).float()
        total_per_channel = torch.full_like(
            changed_per_channel,
            float(batch * n * rollout.shape[-2] * rollout.shape[-1]),
        )
        constraint_fraction = changed_per_channel / total_per_channel.clamp(min=1)

        stack32 = members_physical.float()
        valid = torch.isfinite(stack32)
        counts = valid.sum(dim=1)
        filled = torch.where(valid, stack32, torch.zeros_like(stack32))
        ensemble_mean = filled.sum(dim=1) / counts.clamp(min=1)
        ensemble_mean = torch.where(
            counts > 0,
            ensemble_mean,
            torch.full_like(ensemble_mean, float("nan")),
        )
        spread = None
        if n > 1:
            deviations = torch.where(
                valid,
                stack32 - ensemble_mean.unsqueeze(1),
                torch.zeros_like(stack32),
            )
            variance = deviations.square().sum(dim=1) / (counts - 1).clamp(min=1)
            variance = torch.where(
                counts > 1,
                variance,
                torch.full_like(variance, float("nan")),
            )
            spread = variance.sqrt()

        predicted_correction = member_corrections.mean(dim=1)
        predicted_correction_physical = (
            self.target_space.correction_to_physical(predicted_correction)
        )
        refined_normalized, refined_physical = (
            self.target_space.apply_correction(
                rollout,
                predicted_correction,
                target_mask=target_mask,
            )
        )

        out.members = members_physical if return_members else None
        out.member_corrections_normalized = (
            member_corrections if return_members else None
        )
        out.member_corrections_physical = (
            member_corrections_physical if return_members else None
        )
        out.member_innovations_normalized = (
            member_innovations if return_members else None
        )
        out.member_innovations_physical = (
            member_innovations_physical if return_members else None
        )
        out.member_residuals = member_corrections if return_members else None
        out.ensemble_mean = ensemble_mean
        out.ensemble_spread = spread
        out.predicted_correction_normalized = predicted_correction
        out.predicted_correction_physical = predicted_correction_physical
        out.residual = predicted_correction
        out.refined_normalized = refined_normalized
        # The primary field now corresponds exactly to the reported mean
        # correction; constraints are applied once after ensemble averaging.
        out.refined_physical = refined_physical
        out.conditional_mean_correction_normalized = conditional_mean_correction
        out.conditional_mean_correction_physical = (
            self.target_space.correction_to_physical(
                conditional_mean_correction
            )
        )
        out.deterministic_refined_normalized = deterministic_refined_normalized
        out.deterministic_refined_physical = deterministic_refined_physical
        out.constraint_activation_fraction = constraint_fraction
        out.diagnostics["constraint_activation_fraction"] = changed.float().mean()
        out.diagnostics["conditional_mean_source"] = conditional_mean_source
        out.diagnostics["constraint_activation_fraction_per_channel"] = (
            constraint_fraction
        )
        return out

    def _guard_deployed_member_corrections(
        self, corrections: torch.Tensor
    ) -> torch.Tensor:
        """Apply the unified scaled-space safeguard to complete ensemble members."""
        if self.refinement_config.backend != "unified":
            return corrections
        assert self.refiner is not None
        guard = getattr(self.refiner, "guard_correction_normalized", None)
        if guard is None:
            raise RuntimeError(
                "Unified refiner does not expose its residual amplitude safeguard."
            )
        if corrections.ndim != 5:
            raise ValueError(
                "Member corrections must have shape [B, member, C, H, W], "
                f"got {tuple(corrections.shape)}."
            )
        batch, members, channels, height, width = corrections.shape
        guarded = guard(
            corrections.reshape(batch * members, channels, height, width)
        )
        return guarded.reshape(batch, members, channels, height, width)

    def _deterministic_mean_process(
        self,
        conditioning: torch.Tensor,
        rollout: torch.Tensor,
        forecast_lead_time: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.refiner is not None
        if self.refinement_config.is_legacy_flow_matching:
            return self.refiner.deterministic_residual(
                conditioning,
                forecast_lead_time=forecast_lead_time,
                rollout_normalized=rollout,
            )
        return self.refiner.deterministic_mean_correction_normalized(
            conditioning, forecast_lead_time=forecast_lead_time
        )

    def _sample_innovation(
        self,
        conditioning: torch.Tensor,
        rollout: torch.Tensor,
        forecast_lead_time: torch.Tensor | None,
        num_steps: int | None,
        mean_process_output: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        assert self.refiner is not None
        if self.refinement_config.is_legacy_flow_matching:
            sample = self.refiner.sample_residual(
                conditioning,
                forecast_lead_time=forecast_lead_time,
                generator=generator,
                num_steps=num_steps,
                rollout_normalized=rollout,
            )
            if sample.shape != mean_process_output.shape:
                raise RuntimeError(
                    "Legacy flow correction sample and deterministic mean have "
                    f"different shapes: {tuple(sample.shape)} and "
                    f"{tuple(mean_process_output.shape)}."
                )
            return sample.float() - mean_process_output.float()
        kwargs: dict[str, Any] = {
            "forecast_lead_time": forecast_lead_time,
            "num_steps": num_steps,
            "mean_correction_normalized": mean_process_output,
        }
        return self.refiner.sample_innovation_normalized(
            conditioning, **kwargs
        )

    # ------------------------------------------------------------------
    # Aurora (Phase-1) delegation
    # ------------------------------------------------------------------
    @property
    def patch_size(self) -> int:
        return int(self.aurora.patch_size)

    @property
    def surf_stats(self) -> dict:
        return dict(self.aurora.surf_stats)

    def batch_transform_hook(self, batch):
        return self.aurora.batch_transform_hook(batch)

    def configure_activation_checkpointing(self) -> None:
        hook = getattr(self.aurora, "configure_activation_checkpointing", None)
        if callable(hook):
            hook()

    def load_checkpoint(self, *args, **kwargs) -> None:
        self.aurora.load_checkpoint(*args, **kwargs)

    def load_checkpoint_local(self, *args, **kwargs) -> None:
        hook = getattr(self.aurora, "load_checkpoint_local", None)
        if callable(hook):
            hook(*args, **kwargs)

    def freeze_base(self) -> None:
        for param in self.aurora.parameters():
            param.requires_grad_(False)
        self.aurora.eval()
        self.aurora_frozen = True

    def refine_parameter_count(self) -> int:
        return 0 if self.refiner is None else sum(p.numel() for p in self.refiner.parameters())

    def temporal_parameter_count(self) -> int:
        return (
            0
            if self.temporal is None
            else sum(p.numel() for p in self.temporal.parameters())
        )

    def forward(self, batch, forecast_lead_time_hours: float | torch.Tensor | None = None):
        """Deterministic Aurora prediction, refined at inference time only.

        In training mode the unrefined Aurora prediction is returned: Phase 2 is
        exercised through :meth:`training_step` from the supervised-loss code
        path, which keeps the (frozen) Aurora graph decoupled from the refiner's
        gradient. At inference the deterministic prediction is refined as
        postprocessing; the caller decides whether the deterministic or the
        refined state continues the rollout.
        """
        if not self.training and self.temporal is not None:
            raise RuntimeError(
                "AuroraTwoPhaseRefiner.forward cannot apply enabled temporal Mamba "
                "to a single forecast step. Use the sequence-aware rollout path "
                "with chronological temporal_history instead."
            )
        pred = self.aurora(batch)
        if self.training or not self.refinement_config.is_active:
            return pred
        from finetune.refinement.integration import refine_batch_prediction

        return refine_batch_prediction(
            self,
            pred,
            aurora_input_batch=batch,
            forecast_lead_time_hours=forecast_lead_time_hours,
            ensemble_size=1,
            seed=self.refinement_config.seed,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        if not self.refinement_config.freeze_aurora:
            return [p for p in self.parameters() if p.requires_grad]
        phase_two: list[nn.Parameter] = []
        if self.refiner is not None:
            phase_two.extend(p for p in self.refiner.parameters() if p.requires_grad)
        if self.temporal is not None:
            phase_two.extend(p for p in self.temporal.parameters() if p.requires_grad)
        return phase_two

    def describe(self) -> dict[str, Any]:
        return {
            "refinement": self.refinement_config.to_dict(),
            "temporal": dict(self.temporal_config),
            "performance": self.performance_config.to_dict(),
            "aurora_frozen": self.aurora_frozen,
            "aurora_parameters": sum(p.numel() for p in self.aurora.parameters()),
            "refiner_parameters": (
                sum(p.numel() for p in self.refiner.parameters()) if self.refiner is not None else 0
            ),
            "temporal_parameters": self.temporal_parameter_count(),
            "conditioning_static_names": list(self.conditioning_static_names),
            "field_packing": self.packing.to_dict(),
        }


def build_two_phase_refiner(
    aurora: nn.Module | None,
    packing: FieldPacking,
    config: Mapping[str, Any] | None,
    *,
    nonnegative_variables: Sequence[str] = (),
    conditioning_static_names: Sequence[str] = (),
) -> AuroraTwoPhaseRefiner:
    """Build the wrapper from a raw experiment configuration."""
    refinement = resolve_refinement_config(config)
    performance = resolve_performance_config(config)
    temporal = resolve_temporal_config(config)
    if refinement.feedback_to_rollout:
        warnings.warn(
            "refinement.feedback_to_rollout is enabled: this experimental path changes "
            "the deterministic Aurora rollout trajectory and must be evaluated "
            "independently.",
            RuntimeWarning,
            stacklevel=2,
        )
    return AuroraTwoPhaseRefiner(
        aurora,
        packing,
        refinement=refinement,
        performance=performance,
        nonnegative_variables=nonnegative_variables,
        conditioning_static_names=conditioning_static_names,
        temporal_enabled=temporal["enabled"],
        temporal_channels=temporal["channels"],
        temporal_state=temporal["state"],
        temporal_layers=temporal["layers"],
        temporal_conv=temporal["conv"],
        temporal_expand=temporal["expand"],
        temporal_mode=temporal["mode"],
        temporal_dropout=temporal["dropout"],
        temporal_gated_fusion=temporal["gated_fusion"],
        temporal_gate_init=temporal["gate_init"],
        temporal_lead_time_conditioning=temporal["lead_time_conditioning"],
        temporal_mask_conditioning=temporal["mask_conditioning"],
        temporal_coordinate_conditioning=temporal[
            "coordinate_conditioning"
        ],
        temporal_causal=temporal["causal"],
        temporal_learning_rate_multiplier=temporal["learning_rate_multiplier"],
        temporal_objective=temporal["objective"],
        temporal_semantic_version=temporal["semantic_version"],
        temporal_scan_backend=temporal["scan_backend"],
    )
