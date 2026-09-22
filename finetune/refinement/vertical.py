"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Continuous vertical / variable-identity encoding for packed refinement channels.

The packed representation is a single ``[B, C, H, W]`` tensor whose channel axis
interleaves surface, column and per-pressure-level fields. A network that only
sees the channel *index* has no way to know that channel 3 is 500 hPa ozone and
channel 4 is 100 hPa ozone, and no way to generalize to a level it did not see
at that index. This module attaches the missing physics to every channel:

* a **continuous** pressure coordinate ``log(p / p_ref)`` rather than a
  positional index, so the representation interpolates between levels;
* an explicit **pressure-applicable mask**, because a total-column quantity
  such as ``gtco3`` has no pressure at all. It is *not* given a fabricated
  surface or zero-pressure level;
* a **type embedding** separating ``atmos`` profile levels, ``surf`` fields and
  vertically integrated ``column`` quantities;
* a learned **variable identity** embedding, so ``go3`` at 500 hPa and a
  hypothetical ``no2`` at 500 hPa are not forced to share a correction.

Nothing here imposes a relationship between the column quantity and the
supervised profile levels. Four levels do not integrate to a total column, so
the column is represented jointly and left statistically coupled by the
network, never tied by a fabricated conservation constraint.

Units
-----
Pressure is taken from :class:`~finetune.refinement.packing.ChannelSpec.level`,
which Aurora defines in **hPa** (see :class:`aurora.batch.Metadata`). Values are
validated to a physically plausible hPa range so a Pa-valued packing fails
loudly instead of producing a silently wrong ``log(p)``.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch
from torch import nn

from finetune.refinement.packing import FieldPacking

__all__ = [
    "COLUMN_UNIT_PATTERNS",
    "VERTICAL_FEATURE_NAMES",
    "VerticalChannelEncoder",
    "is_column_channel",
]

#: Reference pressure for the continuous vertical coordinate, in hPa.
PRESSURE_REFERENCE_HPA = 1000.0

#: ``log(p / p_ref)`` divisor. ``log(1000 / 10)`` maps the 10-1000 hPa range
#: onto roughly ``[-1, 0]``. Fixed and documented so a checkpoint's vertical
#: features keep their meaning.
LOG_PRESSURE_SCALE = math.log(100.0)

#: Plausible hPa bounds. A packing in Pa would violate the upper bound.
_MIN_LEVEL_HPA = 1.0e-3
_MAX_LEVEL_HPA = 1.1e3

#: Units that denote a vertically integrated (per-area) amount rather than a
#: local concentration or mixing ratio. Matched case-insensitively after
#: normalizing ``**`` and whitespace.
COLUMN_UNIT_PATTERNS: tuple[str, ...] = (
    r"^kg\s*m-2$",
    r"^g\s*m-2$",
    r"^mol\s*m-2$",
    r"^molecules?\s*cm-2$",
    r"^du$",
    r"^dobson(\s*units?)?$",
)

VERTICAL_FEATURE_NAMES: tuple[str, ...] = (
    "log_pressure_normalized",
    "pressure_applicable",
    "is_atmos_level",
    "is_surface_field",
    "is_column_field",
)


def _normalize_units(units: str) -> str:
    text = str(units or "").strip().lower()
    text = text.replace("**", "")
    text = text.replace("^", "")
    return re.sub(r"\s+", " ", text)


def is_column_channel(
    aurora_name: str,
    kind: str,
    level: float | None,
    units: str,
    *,
    column_variables: Sequence[str] = (),
) -> bool:
    """Whether a packed channel is a vertically integrated column quantity.

    A channel is a column when it carries no pressure level *and* either its
    units are a per-area amount (see :data:`COLUMN_UNIT_PATTERNS`) or its
    Aurora name was declared in ``column_variables``.

    The explicit declaration exists because unit strings in NetCDF products are
    not standardized; it is never inferred from the variable name alone.
    """
    if level is not None or str(kind).lower() == "atmos":
        return False
    if str(aurora_name) in {str(name) for name in column_variables}:
        return True
    normalized = _normalize_units(units)
    return any(re.match(pattern, normalized) for pattern in COLUMN_UNIT_PATTERNS)


class VerticalChannelEncoder(nn.Module):
    """Per-packed-channel vertical / identity embedding.

    The static part is a ``[C, F]`` buffer built once from the packing. The
    learned part is a small MLP plus a variable-identity table. Two products are
    exposed:

    ``embedding``
        ``[C, dim]`` per-channel vectors, mean-pooled into the shared
        conditioning vector so the scalar conditioning knows which fields are
        present.
    ``channel_modulation``
        ``(scale[C], shift[C])`` zero-initialized channel-wise affine applied to
        the packed residual state. Zero initialization makes the encoder an
        exact identity at construction, and the gradient still reaches it on the
        first backward pass because the *input* to the affine is non-zero.

    Args:
        packing: the authoritative channel layout.
        dim: embedding width.
        column_variables: Aurora names to force-classify as column quantities.
    """

    def __init__(
        self,
        packing: FieldPacking,
        dim: int,
        *,
        column_variables: Sequence[str] = (),
    ) -> None:
        super().__init__()
        width = int(dim)
        if width <= 0:
            raise ValueError(f"VerticalChannelEncoder dim must be positive, got {dim!r}.")
        self.dim = width
        self.num_channels = int(packing.num_channels)

        variables = list(packing.variables)
        variable_index = {name: index for index, name in enumerate(variables)}
        self.variables = tuple(variables)

        features: list[list[float]] = []
        identity: list[int] = []
        column_flags: list[bool] = []
        for spec in packing.channels:
            level = spec.level
            is_column = is_column_channel(
                spec.aurora_name,
                spec.kind,
                level,
                spec.units,
                column_variables=column_variables,
            )
            column_flags.append(is_column)
            if level is None:
                log_pressure = 0.0
                applicable = 0.0
            else:
                pressure = float(level)
                if not math.isfinite(pressure) or pressure <= 0.0:
                    raise ValueError(
                        f"Channel {spec.index} ({spec.aurora_name}) has a non-physical "
                        f"pressure level {level!r}."
                    )
                if not (_MIN_LEVEL_HPA <= pressure <= _MAX_LEVEL_HPA):
                    raise ValueError(
                        f"Channel {spec.index} ({spec.aurora_name}) has pressure "
                        f"{pressure} outside the plausible hPa range "
                        f"[{_MIN_LEVEL_HPA}, {_MAX_LEVEL_HPA}]. Aurora pressure levels "
                        "are in hPa; a Pa-valued packing must be converted first."
                    )
                log_pressure = math.log(pressure / PRESSURE_REFERENCE_HPA) / LOG_PRESSURE_SCALE
                applicable = 1.0
            is_atmos = float(str(spec.kind).lower() == "atmos")
            is_surface = float(not is_column and not is_atmos)
            features.append(
                [
                    log_pressure,
                    applicable,
                    is_atmos,
                    is_surface,
                    float(is_column),
                ]
            )
            identity.append(variable_index[spec.aurora_name])

        static = torch.tensor(features, dtype=torch.float32)
        if static.shape != (self.num_channels, len(VERTICAL_FEATURE_NAMES)):
            raise RuntimeError(
                "Vertical feature matrix drifted from VERTICAL_FEATURE_NAMES: "
                f"{tuple(static.shape)}."
            )
        self.register_buffer("static_features", static, persistent=False)
        self.register_buffer(
            "variable_identity", torch.tensor(identity, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "column_mask", torch.tensor(column_flags, dtype=torch.bool), persistent=False
        )

        self.variable_embedding = nn.Embedding(max(len(variables), 1), width)
        nn.init.normal_(self.variable_embedding.weight, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(len(VERTICAL_FEATURE_NAMES) + width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        # Channel-wise affine on the packed residual state. Zero-initialized so
        # construction is an exact identity; the branch is still trainable
        # because its input activation is non-zero.
        self.modulation = nn.Linear(width, 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    @property
    def column_channels(self) -> tuple[int, ...]:
        """Indices of packed channels that are vertically integrated columns."""
        return tuple(int(i) for i in torch.nonzero(self.column_mask).reshape(-1).tolist())

    @property
    def pressure_channels(self) -> tuple[int, ...]:
        """Indices of packed channels that carry a real pressure level."""
        applicable = self.static_features[:, 1] > 0.5
        return tuple(int(i) for i in torch.nonzero(applicable).reshape(-1).tolist())

    def describe(self) -> dict[str, object]:
        """Machine-readable vertical contract persisted with the checkpoint."""
        return {
            "vertical_feature_names": list(VERTICAL_FEATURE_NAMES),
            "pressure_reference_hpa": PRESSURE_REFERENCE_HPA,
            "log_pressure_scale": LOG_PRESSURE_SCALE,
            "variables": list(self.variables),
            "column_channels": list(self.column_channels),
            "pressure_channels": list(self.pressure_channels),
            "num_channels": self.num_channels,
        }

    def embedding(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """``[C, dim]`` per-channel embedding."""
        identity = self.variable_embedding(self.variable_identity)
        static = self.static_features.to(identity.dtype)
        out = self.mlp(torch.cat([static, identity], dim=-1))
        return out if dtype is None else out.to(dtype)

    def pooled(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """``[dim]`` summary of the packed field set for the conditioning vector."""
        return self.embedding(dtype=dtype).mean(dim=0)

    def channel_modulation(
        self, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(scale[C], shift[C])`` zero-initialized channel-wise affine."""
        params = self.modulation(self.embedding())
        scale, shift = params[:, 0], params[:, 1]
        if dtype is not None:
            scale, shift = scale.to(dtype), shift.to(dtype)
        return scale, shift

    def apply_channel_modulation(self, state: torch.Tensor) -> torch.Tensor:
        """Apply ``state * (1 + scale_c) + shift_c`` over the packed channel axis."""
        if state.shape[-3] != self.num_channels:
            raise ValueError(
                f"VerticalChannelEncoder was built for {self.num_channels} packed "
                f"channels but received {state.shape[-3]}."
            )
        scale, shift = self.channel_modulation(dtype=state.dtype)
        view = (1,) * (state.ndim - 3) + (self.num_channels, 1, 1)
        return state * (1.0 + scale.view(view)) + shift.view(view)
