"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Glue between the existing Aurora fine-tuning workflow and the unified
stochastic residual-refinement package.

This module is the **single dispatch point** between the three refinement
backends, so the training loop, the rollout driver and the checkpoint helpers
contain no per-type ``if/elif`` chains:

``none``
    Refinement disabled; the deterministic Aurora path runs unchanged.
``legacy``
    ``refinement.type: flow_matching_unet`` (alias ``flow_matching``) and every
    configuration that only sets the historical ``model.flow_refine_*`` keys.
    These keep using :func:`finetune.aurora_finetune_utils.maybe_wrap_flow_refine`
    and therefore keep their exact numerical behaviour and checkpoints.
``unified``
    ``flow_matching_transformer``, ``diffusion_unet`` and
    ``diffusion_transformer``, driven by
    :class:`finetune.refinement.two_phase.AuroraTwoPhaseRefiner`.

Rollout semantics are preserved in every backend: the refinement of rollout step
``n`` uses only the deterministic Aurora prediction for that step (and fields
available at its valid time), it is matched to the target at the **same**
forecast valid time, and it does not replace the deterministic state used to
produce later steps unless the experimental
``refinement.feedback_to_rollout`` flag is explicitly enabled.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Sequence

import torch

from finetune.refinement.config import (
    RefinementConfig,
    resolve_performance_config,
    resolve_refinement_config,
)
from finetune.refinement.packing import FieldPacking
from finetune.refinement.two_phase import (
    AuroraTwoPhaseRefiner,
    build_two_phase_refiner,
    resolve_temporal_config,
)

__all__ = [
    "LeadStepBuffer",
    "PackedConditioningInputs",
    "build_field_packing",
    "maybe_build_stochastic_refiner",
    "pack_refinement_conditioning",
    "refine_batch_prediction",
    "refinement_backend",
]


def refinement_backend(config: Mapping[str, Any] | None) -> str:
    """``"none"``, ``"legacy"`` or ``"unified"`` for an experiment configuration."""
    return resolve_refinement_config(config).backend


def _atmos_levels(config: Mapping[str, Any]) -> list[float]:
    data_cfg = config.get("data", {}) if config else {}
    values = data_cfg.get("atmos_levels", data_cfg.get("pressure_levels", []))
    return [float(v) for v in values]


def _lead_times_hours(config: Mapping[str, Any]) -> list[float]:
    data_cfg = config.get("data", {}) if config else {}
    rollout_cfg = config.get("rollout", {}) if config else {}
    step_hours = rollout_cfg.get("rollout_step_hours")
    leads = [int(v) for v in data_cfg.get("target_lead_times", ()) or ()]
    if step_hours is None or not leads:
        return []
    return [float(step_hours) * lead for lead in sorted(leads)]


def build_field_packing(
    config: Mapping[str, Any] | None,
    resolved_specs: Any,
    *,
    norm_stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    lat: Sequence[float] | None = None,
    lon: Sequence[float] | None = None,
    lon_periodic: bool = False,
) -> FieldPacking:
    """Build the canonical variable / pressure-level channel layout for a run.

    Exactly this layout is used by training, validation, inference, evaluation,
    NetCDF output and checkpoint metadata.
    """
    cfg = dict(config or {})
    leads = _lead_times_hours(cfg)
    patch_size = int(cfg.get("model", {}).get("patch_size", 4))
    if patch_size < 1:
        raise ValueError(
            f"model.patch_size must be a positive integer, got {patch_size!r}."
        )

    def _aligned_coordinates(
        values: Sequence[float] | None,
        *,
        axis: str,
    ) -> list[float] | None:
        if values is None:
            return None
        resolved = [float(value) for value in values]
        aligned = len(resolved) - (len(resolved) % patch_size)
        if aligned < 1:
            raise ValueError(
                f"{axis} coordinate length {len(resolved)} is smaller than "
                f"model.patch_size={patch_size}."
            )
        if axis == "longitude" and lon_periodic and aligned != len(resolved):
            raise ValueError(
                f"Periodic longitude length {len(resolved)} is not divisible by "
                f"model.patch_size={patch_size}; regrid instead of cropping a "
                "global longitude axis."
            )
        return resolved[:aligned]

    aligned_lat = _aligned_coordinates(lat, axis="latitude")
    aligned_lon = _aligned_coordinates(lon, axis="longitude")
    return FieldPacking.from_specs(
        list(resolved_specs.targets),
        norm_stats=norm_stats,
        atmos_levels=_atmos_levels(cfg),
        lat=aligned_lat,
        lon=aligned_lon,
        lead_times_hours=leads,
        lead_time_scale_hours=max(leads) if leads else None,
        lon_periodic=lon_periodic,
    )


@dataclasses.dataclass(frozen=True)
class PackedConditioningInputs:
    """Optional non-rollout fields consumed by the unified spatial refiner.

    Both tensors use the same effective batch and spatial grid as the packed
    deterministic rollout. ``input_state_normalized`` follows
    :class:`FieldPacking` exactly; ``static_fields`` follows the configured
    ``data.static_variables`` order recorded on the wrapper.
    """

    input_state_normalized: torch.Tensor | None = None
    static_fields: torch.Tensor | None = None


def _repeat_to_effective_batch(
    tensor: torch.Tensor,
    effective_batch: int,
    *,
    field_name: str,
) -> torch.Tensor:
    source_batch = int(tensor.shape[0])
    if source_batch == effective_batch:
        return tensor
    if source_batch < 1 or effective_batch % source_batch:
        raise ValueError(
            f"{field_name} has batch size {source_batch}, which cannot be aligned "
            f"with the refinement effective batch size {effective_batch}. Expected "
            "an equal batch size or an exact divisor."
        )
    return tensor.repeat_interleave(effective_batch // source_batch, dim=0)


def _pack_latest_target_state(
    refiner: AuroraTwoPhaseRefiner,
    batch: Any,
) -> torch.Tensor:
    """Pack the latest Aurora input frame in target channel/level order."""
    packing = refiner.packing
    fields: dict[str, torch.Tensor] = {}
    metadata_levels = tuple(
        float(value)
        for value in getattr(getattr(batch, "metadata", None), "atmos_levels", ())
    )

    for name in packing.variables:
        specs = packing.channels_for(name)
        kind = specs[0].kind
        source = batch.surf_vars if kind == "surf" else batch.atmos_vars
        if name not in source:
            raise ValueError(
                "refinement.conditioning.aurora_input_state requires target "
                f"variable {name!r} in the Aurora input batch {kind}_vars; "
                f"available variables are {sorted(source)}."
            )
        tensor = source[name]
        if kind == "surf":
            if tensor.ndim == 4:
                tensor = tensor[:, -1]
            elif tensor.ndim != 3:
                raise ValueError(
                    "refinement.conditioning.aurora_input_state surface variable "
                    f"{name!r} must have shape [batch, time, latitude, longitude] "
                    f"or [batch, latitude, longitude], got {tuple(tensor.shape)}."
                )
        else:
            if tensor.ndim == 5:
                tensor = tensor[:, -1]
            elif tensor.ndim != 4:
                raise ValueError(
                    "refinement.conditioning.aurora_input_state atmospheric "
                    f"variable {name!r} must have shape [batch, time, level, "
                    "latitude, longitude] or [batch, level, latitude, longitude], "
                    f"got {tuple(tensor.shape)}."
                )
            selected_levels = tuple(float(spec.level) for spec in specs)
            if not metadata_levels:
                if tensor.shape[1] != len(selected_levels):
                    raise ValueError(
                        "refinement.conditioning.aurora_input_state cannot select "
                        f"levels {selected_levels} for {name!r}: batch.metadata."
                        "atmos_levels is empty and the tensor contains "
                        f"{tensor.shape[1]} levels."
                    )
            else:
                try:
                    indices = [metadata_levels.index(level) for level in selected_levels]
                except ValueError as exc:
                    missing = [level for level in selected_levels if level not in metadata_levels]
                    raise ValueError(
                        "refinement.conditioning.aurora_input_state requires "
                        f"atmospheric levels {selected_levels} for {name!r}, but "
                        f"batch.metadata.atmos_levels={metadata_levels}; missing "
                        f"levels are {missing}."
                    ) from exc
                tensor = tensor.index_select(
                    1,
                    torch.tensor(indices, dtype=torch.long, device=tensor.device),
                )
        fields[name] = tensor

    return refiner.target_space.encode(packing.pack(fields))


def _pack_static_fields(
    refiner: AuroraTwoPhaseRefiner,
    batch: Any,
    *,
    effective_batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    names = tuple(refiner.conditioning_static_names)
    if not names:
        raise ValueError(
            "refinement.conditioning.static_fields=true requires at least one "
            "resolved data.static_variables entry; actual resolved list is empty."
        )
    source = getattr(batch, "static_vars", None)
    if not isinstance(source, Mapping):
        raise ValueError(
            "refinement.conditioning.static_fields=true requires an Aurora batch "
            "with a static_vars mapping."
        )

    parts: list[torch.Tensor] = []
    for name in names:
        if name not in source:
            raise ValueError(
                "refinement.conditioning.static_fields requires configured static "
                f"variable {name!r}, but batch.static_vars contains "
                f"{sorted(source)}."
            )
        tensor = source[name].to(device=device, dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        elif tensor.ndim != 3:
            raise ValueError(
                "refinement.conditioning.static_fields variable "
                f"{name!r} must have shape [latitude, longitude], [batch, "
                "latitude, longitude], or [batch, 1, latitude, longitude], "
                f"got {tuple(tensor.shape)}."
            )
        tensor = _repeat_to_effective_batch(
            tensor,
            effective_batch,
            field_name=f"static field {name!r}",
        )
        parts.append(tensor.unsqueeze(1))
    return torch.cat(parts, dim=1)


def pack_refinement_conditioning(
    refiner: AuroraTwoPhaseRefiner,
    reference: torch.Tensor,
    *,
    aurora_input_batch: Any | None = None,
    static_batch: Any | None = None,
) -> PackedConditioningInputs:
    """Pack configured input-state/statics for shared training and inference.

    ``reference`` is the normalized deterministic rollout ``[N,C,H,W]`` and
    defines the effective batch, device and dtype. The input state is always
    taken from the latest frame of the *pre-forecast* Aurora batch, never from
    the forecast target or the deterministic prediction being corrected.
    """
    if reference.ndim != 4:
        raise ValueError(
            "reference must have shape [effective_batch, channel, latitude, "
            f"longitude], got {tuple(reference.shape)}."
        )
    cond = refiner.refinement_config.conditioning
    effective_batch = int(reference.shape[0])
    input_state: torch.Tensor | None = None
    statics: torch.Tensor | None = None

    if cond.aurora_input_state:
        if aurora_input_batch is None:
            raise ValueError(
                "refinement.conditioning.aurora_input_state=true requires the "
                "pre-forecast Aurora input batch, but aurora_input_batch=None."
            )
        input_state = _pack_latest_target_state(refiner, aurora_input_batch)
        input_state = _repeat_to_effective_batch(
            input_state,
            effective_batch,
            field_name="packed Aurora input state",
        ).to(device=reference.device, dtype=reference.dtype)

    if cond.static_fields:
        source = static_batch if static_batch is not None else aurora_input_batch
        if source is None:
            raise ValueError(
                "refinement.conditioning.static_fields=true requires a batch "
                "containing configured static_vars, but no static_batch or "
                "aurora_input_batch was supplied."
            )
        statics = _pack_static_fields(
            refiner,
            source,
            effective_batch=effective_batch,
            device=reference.device,
            dtype=reference.dtype,
        )

    return PackedConditioningInputs(
        input_state_normalized=None if input_state is None else input_state.detach(),
        static_fields=None if statics is None else statics.detach(),
    )


def maybe_build_stochastic_refiner(
    aurora: torch.nn.Module | None,
    config: Mapping[str, Any] | None,
    resolved_specs: Any,
    *,
    norm_stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    lat: Sequence[float] | None = None,
    lon: Sequence[float] | None = None,
    lon_periodic: bool = False,
    nonnegative_variables: Iterable[str] = (),
) -> AuroraTwoPhaseRefiner | None:
    """Build the two-phase wrapper for the ``unified`` backend only.

    Returns ``None`` for ``none`` and for the ``legacy`` backend, which keeps
    using the untouched :class:`finetune.flow_refine.AuroraFlowRefine` path.
    """
    refinement = resolve_refinement_config(config)
    if refinement.backend != "unified":
        return None
    static_names = tuple(spec.aurora_name for spec in resolved_specs.static)
    if refinement.conditioning.static_fields and not static_names:
        raise ValueError(
            "refinement.conditioning.static_fields=true requires at least one "
            "resolved data.static_variables entry; actual resolved list is empty."
        )
    if refinement.conditioning.aurora_input_state:
        predictor_names = {spec.aurora_name for spec in resolved_specs.predictors}
        target_names = {spec.aurora_name for spec in resolved_specs.targets}
        missing_inputs = sorted(target_names - predictor_names)
        if missing_inputs:
            raise ValueError(
                "refinement.conditioning.aurora_input_state=true requires every "
                "refinement target to be present in the Aurora input state. Missing "
                f"target predictors are {missing_inputs}; add them to "
                "data.predictor_variables or enable "
                "data.include_target_variables_as_predictors."
            )
    packing = build_field_packing(
        config,
        resolved_specs,
        norm_stats=norm_stats,
        lat=lat,
        lon=lon,
        lon_periodic=lon_periodic,
    )
    wrapper = build_two_phase_refiner(
        aurora,
        packing,
        config,
        nonnegative_variables=tuple(nonnegative_variables),
        conditioning_static_names=static_names,
    )
    wrapper.initialize_refiner(
        wrapper.conditioning_channels(
            static_fields=len(wrapper.conditioning_static_names),
            input_state_channels=packing.num_channels,
        )
    )
    return wrapper


class LeadStepBuffer:
    """Collects per-rollout-step normalized fields and packs them once.

    The buffer preserves rollout-step ordering: entries are stored per lead
    step and folded into the effective batch dimension in ascending lead order,
    together with the matching physical forecast lead time. Nothing is shifted,
    re-indexed or interleaved, so the target of rollout step ``n`` always stays
    matched to the Aurora prediction for the same forecast valid time.
    """

    def __init__(self, packing: FieldPacking) -> None:
        self.packing = packing
        self._rollout: dict[int, dict[str, torch.Tensor]] = {}
        self._target: dict[int, dict[str, torch.Tensor]] = {}
        self._mask: dict[int, dict[str, torch.Tensor]] = {}
        self._lead_hours: dict[int, torch.Tensor | float | None] = {}
        self._input_state: dict[int, torch.Tensor] = {}
        self._static_fields: dict[int, torch.Tensor] = {}

    def add(
        self,
        lead: int,
        aurora_name: str,
        *,
        rollout_normalized: torch.Tensor,
        target_normalized: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lead_hours: torch.Tensor | float | None = None,
    ) -> None:
        self._rollout.setdefault(lead, {})[aurora_name] = rollout_normalized
        self._target.setdefault(lead, {})[aurora_name] = target_normalized
        if valid_mask is not None:
            self._mask.setdefault(lead, {})[aurora_name] = valid_mask
        self._lead_hours.setdefault(lead, lead_hours)

    @property
    def leads(self) -> list[int]:
        return sorted(self._rollout)

    def is_complete(self) -> bool:
        """``True`` when every configured channel is present for every lead."""
        wanted = set(self.packing.variables)
        return bool(self._rollout) and all(
            set(self._rollout[lead]) == wanted for lead in self._rollout
        )

    def set_conditioning(
        self,
        lead: int,
        *,
        input_state_normalized: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
    ) -> None:
        """Record target-independent conditioning for one rollout step."""
        if input_state_normalized is not None:
            self._input_state[lead] = input_state_normalized.detach()
        if static_fields is not None:
            self._static_fields[lead] = static_fields.detach()

    def pack_conditioning(
        self,
        device: torch.device | str | None = None,
    ) -> PackedConditioningInputs:
        """Fold optional conditioning into the same lead-major layout as ``pack``."""

        def _pack_optional(
            values: Mapping[int, torch.Tensor], field_name: str
        ) -> torch.Tensor | None:
            if not values:
                return None
            missing = [lead for lead in self.leads if lead not in values]
            extra = sorted(set(values) - set(self.leads))
            if missing or extra:
                raise ValueError(
                    f"{field_name} lead coverage must match rollout leads "
                    f"{self.leads}; missing={missing}, extra={extra}."
                )
            packed = torch.cat([values[lead] for lead in self.leads], dim=0)
            return packed.to(device) if device is not None else packed

        return PackedConditioningInputs(
            input_state_normalized=_pack_optional(
                self._input_state, "Aurora input-state conditioning"
            ),
            static_fields=_pack_optional(
                self._static_fields, "static-field conditioning"
            ),
        )

    def pack(
        self, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rollout, target, mask, lead_hours, lead_index)``.

        All tensors have the effective batch layout ``[lead0 batch, lead1
        batch, ...]`` in ascending rollout-step order.
        """
        if not self._rollout:
            raise RuntimeError("LeadStepBuffer is empty.")
        rollouts, targets, masks, hours, indices = [], [], [], [], []
        for position, lead in enumerate(self.leads):
            rollout = self.packing.pack(self._rollout[lead])
            target = self.packing.pack(self._target[lead])
            if lead in self._mask and set(self._mask[lead]) == set(self.packing.variables):
                mask = self.packing.pack(
                    {k: v.to(rollout.dtype) for k, v in self._mask[lead].items()}
                ).bool()
            else:
                mask = torch.isfinite(rollout) & torch.isfinite(target)
            batch = rollout.shape[0]
            lead_hours = self._lead_hours.get(lead)
            if lead_hours is None:
                hours_tensor = torch.zeros(batch, dtype=torch.float32)
            elif torch.is_tensor(lead_hours):
                hours_tensor = lead_hours.reshape(-1).float()
                if hours_tensor.numel() == 1:
                    hours_tensor = hours_tensor.expand(batch)
            else:
                hours_tensor = torch.full((batch,), float(lead_hours), dtype=torch.float32)
            rollouts.append(rollout)
            targets.append(target)
            masks.append(mask)
            hours.append(hours_tensor.to(rollout.device))
            indices.append(torch.full((batch,), position, dtype=torch.long, device=rollout.device))
        stacked = (
            torch.cat(rollouts, dim=0),
            torch.cat(targets, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(hours, dim=0),
            torch.cat(indices, dim=0),
        )
        if device is not None:
            stacked = tuple(t.to(device) for t in stacked)  # type: ignore[assignment]
        return stacked  # type: ignore[return-value]


def refine_batch_prediction(
    refiner: AuroraTwoPhaseRefiner,
    pred: Any,
    *,
    forecast_lead_time_hours: float | torch.Tensor | None = None,
    ensemble_size: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    aurora_input_batch: Any | None = None,
    return_details: bool = False,
    temporal_history: list[torch.Tensor] | None = None,
) -> Any:
    """Refine one deterministic Aurora prediction ``Batch`` as postprocessing.

    The input ``pred`` is **not** modified: a new ``Batch`` carrying the refined
    physical fields is returned, so the caller decides whether the deterministic
    or the refined state continues the rollout. Only the variables and pressure
    levels declared by the packing are touched; every other field passes through
    unchanged.
    """
    packing = refiner.packing
    surf_names = [spec.aurora_name for spec in packing.channels if spec.kind == "surf"]
    atmos_names = list(
        dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "atmos")
    )

    fields: dict[str, torch.Tensor] = {}
    layout: dict[str, tuple[Any, ...]] = {}
    for name in surf_names:
        tensor = pred.surf_vars[name]
        layout[name] = tuple(tensor.shape)
        # Aurora surface tensors are (B, T, H, W); refine every time slice.
        flat = tensor.reshape(-1, *tensor.shape[-2:]) if tensor.dim() == 4 else tensor
        fields[name] = flat
    level_index: dict[str, list[int]] = {}
    for name in atmos_names:
        tensor = pred.atmos_vars[name]
        layout[name] = tuple(tensor.shape)
        flat = tensor.reshape(-1, *tensor.shape[-3:]) if tensor.dim() == 5 else tensor
        levels = list(packing.levels_for(name))
        full_levels = [float(v) for v in getattr(pred.metadata, "atmos_levels", levels)]
        try:
            indices = [full_levels.index(float(level)) for level in levels]
        except ValueError as exc:
            raise ValueError(
                f"Refinement level(s) {levels} for {name!r} are not present in the "
                f"prediction levels {full_levels}."
            ) from exc
        level_index[name] = indices
        fields[name] = flat.index_select(
            1, torch.tensor(indices, dtype=torch.long, device=flat.device)
        )

    reference = next(iter(fields.values()))
    batch = reference.shape[0]
    physical = packing.pack(fields)
    rollout_norm = refiner.target_space.encode(physical)
    packed_conditioning = pack_refinement_conditioning(
        refiner,
        rollout_norm,
        aurora_input_batch=aurora_input_batch,
        static_batch=aurora_input_batch if aurora_input_batch is not None else pred,
    )

    lead = _lead_tensor(forecast_lead_time_hours, batch, physical.device)
    result = refiner.refine(
        rollout_norm,
        input_state_normalized=packed_conditioning.input_state_normalized,
        static_fields=packed_conditioning.static_fields,
        forecast_lead_time=lead,
        ensemble_size=ensemble_size,
        seed=seed,
        generator=generator,
        return_members=return_details,
        temporal_history=temporal_history,
    )
    refined_physical = result.refined_physical
    if refined_physical is None:
        return (pred, result) if return_details else pred

    refined_fields = packing.unpack(refined_physical)
    new_surf = dict(pred.surf_vars)
    for name in surf_names:
        new_surf[name] = refined_fields[name].reshape(layout[name])
    new_atmos = dict(pred.atmos_vars)
    for name in atmos_names:
        original = pred.atmos_vars[name]
        flat = original.reshape(-1, *original.shape[-3:]) if original.dim() == 5 else original
        updated = flat.clone()
        updated[:, level_index[name]] = refined_fields[name].to(updated.dtype)
        new_atmos[name] = updated.reshape(layout[name])

    refined_batch = dataclasses.replace(pred, surf_vars=new_surf, atmos_vars=new_atmos)
    return (refined_batch, result) if return_details else refined_batch


def _lead_tensor(
    value: float | torch.Tensor | None, batch: int, device: torch.device
) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.reshape(-1).float().to(device)
        if flat.numel() == 1:
            return flat.expand(batch)
        if flat.numel() == batch:
            return flat
        if batch % flat.numel() == 0:
            return flat.repeat_interleave(batch // flat.numel())
        raise ValueError(
            f"forecast_lead_time_hours has {flat.numel()} entries, which does not "
            f"divide the effective batch {batch}."
        )
    return torch.full((batch,), float(value), dtype=torch.float32, device=device)


def describe_refinement(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fully resolved refinement + performance configuration for run metadata."""
    refinement: RefinementConfig = resolve_refinement_config(config)
    return {
        "refinement": refinement.to_dict(),
        "temporal": resolve_temporal_config(config),
        "performance": resolve_performance_config(config).to_dict(),
        "backend": refinement.backend,
    }
