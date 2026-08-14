"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Centralized variable / pressure-level packing for stochastic refinement.

Every refiner, the training loop, validation, inference, evaluation, the NetCDF
writer and the checkpoint metadata use **one** channel mapping, described by an
ordered tuple of :class:`ChannelSpec`. Nothing here hardcodes a CAMS species, a
pressure level, a rollout length or a regional domain: the layout is derived
from the resolved variable specification of the run.

Channel order is deterministic and stable:

1. surface targets, in the order they appear in ``resolved_specs.targets``
2. atmospheric targets, in the same order, each expanded over its configured
   (loss-)levels in the order given by the configuration

so that channel ``i`` always maps to the same ``(variable, level)`` pair for a
given configuration and can be recorded in checkpoints and output metadata.

Adapted from the Prithvi stochastic residual-refinement reference
(https://github.com/midatm1234/Prithvi-UNet-stocahstic, branch
``Prithvi-UNet-stochastic_refinement``, Apache-2.0), which packs predictands
into a single ``[B, C, H, W]`` tensor. Aurora additionally needs pressure-level
expansion and forecast lead-time bookkeeping, both added here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
__all__ = ["ChannelSpec", "FieldPacking"]


@dataclass(frozen=True)
class ChannelSpec:
    """Metadata for one packed channel.

    Attributes:
        index: position in the packed channel axis.
        aurora_name: Aurora variable name (e.g. ``"go3"``).
        dataset_name: name used in the source dataset / NetCDF output.
        kind: ``"surf"`` or ``"atmos"``.
        level: pressure level in hPa, or ``None`` for surface variables.
        level_index: index of ``level`` inside the variable's own level axis.
        units: physical units of the variable.
        normalization: name of the normalization used to build the target space.
        mean: normalization location for this channel.
        std: normalization scale for this channel.
    """

    index: int
    aurora_name: str
    dataset_name: str
    kind: str
    level: float | None
    level_index: int | None
    units: str = ""
    normalization: str = "aurora_location_scale"
    mean: float = 0.0
    std: float = 1.0

    @property
    def key(self) -> tuple[str, float | None]:
        return (self.aurora_name, self.level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "aurora_name": self.aurora_name,
            "dataset_name": self.dataset_name,
            "kind": self.kind,
            "level": self.level,
            "level_index": self.level_index,
            "units": self.units,
            "normalization": self.normalization,
            "mean": self.mean,
            "std": self.std,
        }


@dataclass(frozen=True)
class FieldPacking:
    """Ordered channel layout plus pack/unpack helpers.

    ``pack`` maps a mapping of *normalized* per-variable tensors

    * surface: ``[B, H, W]``
    * atmospheric: ``[B, L, H, W]``

    onto a single ``[B, C, H, W]`` tensor, and ``unpack`` is its exact inverse.
    Values, variable names, pressure levels, channel order, spatial dimensions,
    units and masks all round-trip unchanged.
    """

    channels: tuple[ChannelSpec, ...]
    lat: tuple[float, ...] = ()
    lon: tuple[float, ...] = ()
    #: physical forecast lead times (hours) this packing is used with, in
    #: rollout-step order. Purely descriptive metadata.
    lead_times_hours: tuple[float, ...] = ()
    #: scale used to normalize the forecast lead time before embedding it. It is
    #: a conditioning scale only and never changes the rollout itself.
    lead_time_scale_hours: float = 72.0
    #: whether the longitude axis wraps (global domains). Regional domains keep
    #: replicate padding.
    lon_periodic: bool = False
    #: variable -> ordered levels, cached for fast unpacking.
    _levels_by_var: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)
    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("FieldPacking requires at least one target channel.")

        expected_indices = tuple(range(len(self.channels)))
        actual_indices = tuple(int(spec.index) for spec in self.channels)
        if actual_indices != expected_indices:
            raise ValueError(
                "Packed channel indices must be contiguous and match tuple order; "
                f"expected {expected_indices}, got {actual_indices}."
            )

        seen_keys: set[tuple[str, float | None]] = set()
        kinds_by_variable: dict[str, str] = {}
        derived_levels: dict[str, list[float]] = {}
        level_indices: dict[str, list[int]] = {}
        for spec in self.channels:
            if not spec.aurora_name or not spec.dataset_name:
                raise ValueError(
                    f"Packed channel {spec.index} has an empty variable name."
                )
            if spec.kind not in {"surf", "atmos"}:
                raise ValueError(
                    f"Packed channel {spec.index} has invalid kind {spec.kind!r}."
                )
            previous_kind = kinds_by_variable.setdefault(spec.aurora_name, spec.kind)
            if previous_kind != spec.kind:
                raise ValueError(
                    f"Variable {spec.aurora_name!r} mixes surface and atmospheric channels."
                )
            if spec.key in seen_keys:
                raise ValueError(f"Duplicate packed channel key {spec.key!r}.")
            seen_keys.add(spec.key)
            if not math.isfinite(float(spec.mean)):
                raise ValueError(
                    f"Packed channel {spec.index} has non-finite normalization mean."
                )
            if not math.isfinite(float(spec.std)) or float(spec.std) <= 0.0:
                raise ValueError(
                    f"Packed channel {spec.index} must have a finite positive "
                    f"normalization std, got {spec.std!r}."
                )

            if spec.kind == "surf":
                if spec.level is not None or spec.level_index is not None:
                    raise ValueError(
                        f"Surface channel {spec.aurora_name!r} must not carry a "
                        "pressure level or level index."
                    )
            else:
                if spec.level is None or spec.level_index is None:
                    raise ValueError(
                        f"Atmospheric channel {spec.aurora_name!r} requires both "
                        "a pressure-level value and a level index."
                    )
                if not math.isfinite(float(spec.level)):
                    raise ValueError(
                        f"Atmospheric channel {spec.aurora_name!r} has a non-finite level."
                    )
                derived_levels.setdefault(spec.aurora_name, []).append(float(spec.level))
                level_indices.setdefault(spec.aurora_name, []).append(int(spec.level_index))

        for name, indices in level_indices.items():
            expected = list(range(len(indices)))
            if indices != expected:
                raise ValueError(
                    f"Atmospheric channel level indices for {name!r} must be "
                    f"contiguous in packed order; expected {expected}, got {indices}."
                )

        normalized_levels = {
            name: tuple(values) for name, values in derived_levels.items()
        }
        if self._levels_by_var:
            provided = {
                str(name): tuple(float(value) for value in values)
                for name, values in self._levels_by_var.items()
            }
            if provided != normalized_levels:
                raise ValueError(
                    "FieldPacking _levels_by_var disagrees with channel metadata: "
                    f"channels={normalized_levels}, provided={provided}."
                )
        object.__setattr__(self, "_levels_by_var", normalized_levels)

        self._validate_coordinate("latitude", self.lat)
        self._validate_coordinate("longitude", self.lon)
        if not math.isfinite(float(self.lead_time_scale_hours)) or (
            float(self.lead_time_scale_hours) <= 0.0
        ):
            raise ValueError(
                "lead_time_scale_hours must be finite and positive, got "
                f"{self.lead_time_scale_hours!r}."
            )
        if any(not math.isfinite(float(value)) or float(value) < 0.0
               for value in self.lead_times_hours):
            raise ValueError("Forecast lead times must be finite and non-negative.")
        if len(set(self.lead_times_hours)) != len(self.lead_times_hours):
            raise ValueError("Forecast lead times must be unique in rollout-step order.")

    @staticmethod
    def _validate_coordinate(name: str, values: Sequence[float]) -> None:
        if not values:
            return
        floats = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in floats):
            raise ValueError(f"{name.capitalize()} coordinates contain non-finite values.")
        if len(set(floats)) != len(floats):
            raise ValueError(f"{name.capitalize()} coordinates must be unique.")
        if len(floats) > 1:
            diffs = [right - left for left, right in zip(floats[:-1], floats[1:])]
            if not (all(diff > 0.0 for diff in diffs) or all(diff < 0.0 for diff in diffs)):
                raise ValueError(
                    f"{name.capitalize()} coordinates must be strictly monotonic."
                )


    # -- construction ----------------------------------------------------
    @classmethod
    def from_specs(
        cls,
        target_specs: Iterable[Any],
        *,
        norm_stats: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        atmos_levels: Sequence[float] | None = None,
        lat: Sequence[float] | None = None,
        lon: Sequence[float] | None = None,
        lead_times_hours: Sequence[float] | None = None,
        lead_time_scale_hours: float | None = None,
        lon_periodic: bool = False,
    ) -> FieldPacking:
        """Build a packing from ``ResolvedVariableSpecs.targets``.

        Args:
            target_specs: the resolved target variable specifications. Each must
                expose ``aurora_name``, ``dataset_name``, ``kind`` and (for
                atmospheric variables) ``loss_levels``.
            norm_stats: ``{aurora_name: {"mean": tensor, "std": tensor}}`` as
                produced by
                :func:`finetune.aurora_finetune_utils.compute_target_normalization_stats`.
                Per-level entries are consumed in the same order the stats were
                computed in, which is the order of ``loss_levels`` when set.
            atmos_levels: full pressure-level axis used when a spec does not
                restrict its levels.
            lat / lon: coordinate vectors recorded for output metadata.
            lead_times_hours: descriptive rollout lead times.
        """
        channels: list[ChannelSpec] = []
        levels_by_var: dict[str, tuple[float, ...]] = {}
        full_levels = [
            float(v) for v in (() if atmos_levels is None else atmos_levels)
        ]

        surf_specs = [s for s in target_specs if getattr(s, "kind", "surf") == "surf"]
        atmos_specs = [s for s in target_specs if getattr(s, "kind", "surf") == "atmos"]

        index = 0
        for spec in surf_specs:
            name = str(spec.aurora_name)
            mean, std = _stat_for(norm_stats, name, 0)
            channels.append(
                ChannelSpec(
                    index=index,
                    aurora_name=name,
                    dataset_name=str(getattr(spec, "dataset_name", name)),
                    kind="surf",
                    level=None,
                    level_index=None,
                    units=str(getattr(spec, "units", "") or ""),
                    mean=mean,
                    std=std,
                )
            )
            index += 1

        for spec in atmos_specs:
            name = str(spec.aurora_name)
            loss_levels = getattr(spec, "loss_levels", None)
            levels = [float(v) for v in (loss_levels if loss_levels is not None else full_levels)]
            if not levels:
                raise ValueError(
                    f"Atmospheric target {name!r} has no pressure levels; set "
                    "data.atmos_levels or the variable's loss_levels."
                )
            levels_by_var[name] = tuple(levels)
            for level_index, level in enumerate(levels):
                mean, std = _stat_for(norm_stats, name, level_index)
                channels.append(
                    ChannelSpec(
                        index=index,
                        aurora_name=name,
                        dataset_name=str(getattr(spec, "dataset_name", name)),
                        kind="atmos",
                        level=level,
                        level_index=level_index,
                        units=str(getattr(spec, "units", "") or ""),
                        mean=mean,
                        std=std,
                    )
                )
                index += 1

        leads = tuple(
            float(v)
            for v in (() if lead_times_hours is None else lead_times_hours)
        )
        if lead_time_scale_hours is not None:
            scale = float(lead_time_scale_hours)
        elif leads:
            scale = float(max(leads))
        else:
            scale = 72.0
        if not (scale > 0):
            raise ValueError(
                f"lead_time_scale_hours must be positive, got {lead_time_scale_hours!r}."
            )
        return cls(
            channels=tuple(channels),
            lat=tuple(float(v) for v in (() if lat is None else lat)),
            lon=tuple(float(v) for v in (() if lon is None else lon)),
            lead_times_hours=leads,
            lead_time_scale_hours=scale,
            lon_periodic=bool(lon_periodic),
            _levels_by_var=levels_by_var,
        )

    # -- introspection ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.channels)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def variables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for spec in self.channels:
            if spec.aurora_name not in seen:
                seen.append(spec.aurora_name)
        return tuple(seen)

    def levels_for(self, aurora_name: str) -> tuple[float, ...]:
        return self._levels_by_var.get(aurora_name, ())

    def channels_for(self, aurora_name: str) -> tuple[ChannelSpec, ...]:
        return tuple(spec for spec in self.channels if spec.aurora_name == aurora_name)

    def index_of(self, aurora_name: str, level: float | None = None) -> int:
        for spec in self.channels:
            if spec.aurora_name == aurora_name and spec.level == level:
                return spec.index
        raise KeyError(f"No packed channel for variable {aurora_name!r} at level {level!r}.")

    def means(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """``[1, C, 1, 1]`` normalization locations in channel order."""
        values = [spec.mean for spec in self.channels]
        return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)

    def stds(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """``[1, C, 1, 1]`` normalization scales in channel order."""
        values = [spec.std for spec in self.channels]
        return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": [spec.to_dict() for spec in self.channels],
            "lat": list(self.lat),
            "lon": list(self.lon),
            "lead_times_hours": list(self.lead_times_hours),
            "lead_time_scale_hours": self.lead_time_scale_hours,
            "lon_periodic": self.lon_periodic,
        }

    # -- packing ---------------------------------------------------------
    def pack(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Pack per-variable tensors into ``[B, C, H, W]`` in channel order.

        Surface tensors are ``[B, H, W]``, atmospheric tensors ``[B, L, H, W]``
        with ``L`` matching the configured levels for that variable.
        """
        parts: list[torch.Tensor] = []
        reference: tuple[int, tuple[int, int], torch.device, torch.dtype] | None = None
        for name in self.variables:
            if name not in fields:
                raise KeyError(
                    f"Cannot pack: variable {name!r} is missing from the provided fields "
                    f"({sorted(fields)})."
                )
            tensor = fields[name]
            specs = self.channels_for(name)
            if specs[0].kind == "surf":
                if tensor.dim() != 3:
                    raise ValueError(
                        f"Surface variable {name!r} must be [B, H, W], got "
                        f"{tuple(tensor.shape)}."
                    )
                block = tensor.unsqueeze(1)
            else:
                if tensor.dim() != 4:
                    raise ValueError(
                        f"Atmospheric variable {name!r} must be [B, L, H, W], got "
                        f"{tuple(tensor.shape)}."
                    )
                if tensor.shape[1] != len(specs):
                    raise ValueError(
                        f"Atmospheric variable {name!r} has {tensor.shape[1]} levels but "
                        f"the packing declares {len(specs)} ({[s.level for s in specs]})."
                    )
                block = tensor

            signature = (
                int(block.shape[0]),
                (int(block.shape[-2]), int(block.shape[-1])),
                block.device,
                block.dtype,
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(
                    "All packed variables must share batch size, latitude/longitude "
                    "shape, device, and dtype. "
                    f"Variable {name!r} has {signature}; expected {reference}."
                )
            parts.append(block)

        if not parts or reference is None:
            raise ValueError("Cannot pack an empty variable set.")
        _, (height, width), _, _ = reference
        if self.lat and len(self.lat) != height:
            raise ValueError(
                f"Packed latitude size {height} does not match metadata size {len(self.lat)}."
            )
        if self.lon and len(self.lon) != width:
            raise ValueError(
                f"Packed longitude size {width} does not match metadata size {len(self.lon)}."
            )

        packed = torch.cat(parts, dim=1)
        if packed.shape[1] != self.num_channels:
            raise RuntimeError(
                f"Packed {packed.shape[1]} channels but the packing declares "
                f"{self.num_channels}."
            )
        return packed

    def unpack(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exact inverse of :meth:`pack`."""
        if packed.dim() != 4:
            raise ValueError(f"Packed tensor must be [B, C, H, W], got {tuple(packed.shape)}.")
        if packed.shape[1] != self.num_channels:
            raise ValueError(
                f"Packed tensor has {packed.shape[1]} channels but the packing declares "
                f"{self.num_channels}."
            )
        if self.lat and packed.shape[-2] != len(self.lat):
            raise ValueError(
                f"Packed latitude size {packed.shape[-2]} does not match metadata "
                f"size {len(self.lat)}."
            )
        if self.lon and packed.shape[-1] != len(self.lon):
            raise ValueError(
                f"Packed longitude size {packed.shape[-1]} does not match metadata "
                f"size {len(self.lon)}."
            )
        out: dict[str, torch.Tensor] = {}
        cursor = 0
        for name in self.variables:
            specs = self.channels_for(name)
            width = len(specs)
            block = packed[:, cursor : cursor + width]
            out[name] = block.squeeze(1) if specs[0].kind == "surf" else block
            cursor += width
        if cursor != self.num_channels:
            raise RuntimeError(
                f"Unpacked {cursor} channels but metadata declares {self.num_channels}."
            )
        return out


def _stat_for(
    norm_stats: Mapping[str, Mapping[str, torch.Tensor]] | None,
    name: str,
    level_index: int,
) -> tuple[float, float]:
    """Return ``(mean, std)`` for one channel, defaulting to ``(0, 1)``."""
    if not norm_stats or name not in norm_stats:
        return 0.0, 1.0
    stats = norm_stats[name]
    mean = stats.get("mean")
    std = stats.get("std")
    return (
        _scalar(mean, level_index, 0.0),
        _scalar(std, level_index, 1.0),
    )


def _scalar(value: Any, index: int, default: float) -> float:
    if value is None:
        return default
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return default
    if flat.numel() == 1:
        selected = flat[0]
    elif index < flat.numel():
        selected = flat[index]
    else:
        raise ValueError(
            f"Normalization statistics contain {flat.numel()} entries but "
            f"packed level index {index} was requested. Supply one scalar or "
            "one value for every configured level."
        )
    return float(selected.item())
