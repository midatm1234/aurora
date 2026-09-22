"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared calendar / solar-geometry feature builder for Phase-2 refinement.

This module is the **single** place where physical calendar information is
turned into network inputs. Training, validation, inference, caching and
evaluation all call it, so a feature can never mean one thing in the trainer
and something else in the evaluator.

Four time coordinates
---------------------
The repository keeps four coordinates strictly separate. They never share an
embedding, a configuration key or a tensor:

======================  ====================================================
``t0``                  forecast initialization time (a real UTC timestamp)
``ell``                 forecast lead, cumulative physical hours since ``t0``
``t_valid = t0 + ell``  the valid time a frame verifies against
``tau`` / ``k``         flow interpolation coordinate / diffusion noise index
======================  ====================================================

Only the first three are physical. ``tau``/``k`` are generative-process
coordinates owned by :mod:`finetune.refinement.schedules` and are deliberately
absent from this module: nothing here may ever be indexed by them.

Every calendar feature below is derived from the **actual valid timestamp of
the frame it describes**. Reusing the initialization day or hour for a whole
trajectory is precisely the defect this module exists to prevent, so
:func:`valid_times` is the only supported way to obtain the timestamps and it
always adds the per-frame lead.

Calendar convention
-------------------
* Timestamps are interpreted in **UTC**. Aware datetimes are converted with
  :meth:`datetime.datetime.astimezone`; naive datetimes are *assumed* to be UTC
  (Aurora's :class:`~aurora.batch.Metadata` stores naive UTC timestamps). The
  machine's local time zone is never consulted: :func:`_to_utc` never calls
  ``astimezone()`` without an explicit ``tz`` argument and never uses
  ``datetime.now``/``utcnow``/``fromtimestamp``.
* Only the **proleptic Gregorian** calendar is supported, which is what
  :class:`datetime.datetime` implements and what CAMS/ERA5 NetCDF products use.
  ``cftime`` objects carrying ``360_day``, ``365_day``/``noleap``, ``366_day``/
  ``all_leap`` or ``julian`` calendars are rejected loudly by
  :func:`require_gregorian` rather than silently mis-encoded.
* Leap years follow the Gregorian rule, so ``days_in_year`` is 366 in 2024 and
  365 in 1900 and 2100. Sub-hour precision (minutes, seconds, microseconds) is
  carried into the fractional UTC hour when present.

Feature groups
--------------
``scalar`` features describe one (sample, lead) pair and are fed to the
conditioning **vector** (FiLM / adaptive normalization). ``spatial`` features
additionally depend on the grid cell and are fed as conditioning **channels**.

The scalar order is fixed by :data:`SCALAR_FEATURE_NAMES` and persisted in the
checkpoint signature; the spatial order is fixed by
:data:`SPATIAL_FEATURE_NAMES`. Changing either requires a version bump, because
a trained network's first layer is indexed by position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import torch

__all__ = [
    "CALENDAR_CONVENTION",
    "SCALAR_FEATURE_NAMES",
    "SPATIAL_FEATURE_NAMES",
    "CalendarFeatureBuilder",
    "CalendarSpec",
    "cos_solar_zenith_angle",
    "days_in_year",
    "fractional_day_of_year",
    "fractional_utc_hour",
    "local_mean_solar_hour",
    "require_gregorian",
    "valid_times",
    "year_phase",
]


#: Identifier persisted alongside checkpoints. Bump when the feature order, the
#: calendar interpretation or the normalization of any feature changes.
CALENDAR_CONVENTION = "gregorian_utc_v1"

#: Reference pressure used by the vertical encoder, in hPa.
PRESSURE_REFERENCE_HPA = 1000.0

SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    # Seasonal position of the *valid* time.
    "valid_season_sin",
    "valid_season_cos",
    # UTC time of day of the *valid* time.
    "valid_utc_sin",
    "valid_utc_cos",
    # Initialization UTC cycle (00/06/12/18 ...), kept separately identifiable
    # so the network can distinguish "which cycle produced this" from "what
    # time of day does this frame verify at".
    "init_cycle_sin",
    "init_cycle_cos",
    # Seasonal position of the initialization, which differs from the valid
    # season across a year boundary or a long rollout.
    "init_season_sin",
    "init_season_cos",
    # Cumulative forecast lead, three monotone scalings (see LeadTimeEmbedding).
    "lead_hours_scaled",
    "lead_hours_log1p",
    "lead_hours_sqrt",
    # Elapsed hours since the previous frame of this trajectory. Equals the
    # lead for the first frame. Makes an irregular or gappy cadence explicit
    # instead of silently compressing it.
    "elapsed_hours_scaled",
    # 1.0 when this frame follows a real predecessor, 0.0 at the trajectory
    # start or after a missing frame.
    "has_predecessor",
)

SPATIAL_FEATURE_NAMES: tuple[str, ...] = (
    # Seam-safe spherical position. Longitude enters only periodically.
    "geo_x",
    "geo_y",
    "geo_z",
    # Latitude is *not* periodic and is kept as an explicit monotone channel.
    "latitude_normalized",
    # UTC x longitude interaction: cyclic local mean solar hour.
    "local_solar_sin",
    "local_solar_cos",
    # Solar geometry from valid date, UTC, latitude and longitude.
    "cos_solar_zenith",
)


_NON_GREGORIAN_CALENDARS = frozenset(
    {
        "360_day",
        "365_day",
        "366_day",
        "all_leap",
        "noleap",
        "julian",
        "proleptic_julian",
    }
)


def require_gregorian(value: object) -> None:
    """Reject timestamps that do not use the proleptic Gregorian calendar.

    ``cftime`` datetimes expose a ``calendar`` attribute. A ``360_day`` or
    ``noleap`` timestamp would silently produce wrong day-of-year phases, so it
    is refused here instead of being encoded incorrectly.
    """
    calendar = getattr(value, "calendar", None)
    if calendar is None:
        return
    name = str(calendar).strip().lower()
    if name in {"", "standard", "gregorian", "proleptic_gregorian"}:
        return
    if name in _NON_GREGORIAN_CALENDARS:
        raise ValueError(
            f"Unsupported calendar {calendar!r}. The refinement calendar features "
            "are defined for the proleptic Gregorian calendar only "
            f"({CALENDAR_CONVENTION}); a {calendar!r} timestamp would give a wrong "
            "day-of-year phase. Convert the dataset to a standard calendar first."
        )
    raise ValueError(
        f"Unrecognized calendar {calendar!r}; refusing to guess. Supported: "
        "standard / gregorian / proleptic_gregorian."
    )


def _to_utc(value: datetime) -> datetime:
    """Return ``value`` as a naive UTC timestamp.

    Aware inputs are converted to UTC explicitly. Naive inputs are assumed to
    be UTC already, matching Aurora's :class:`~aurora.batch.Metadata`. The local
    time zone of the machine is never consulted.
    """
    require_gregorian(value)
    if not isinstance(value, datetime):
        # ``cftime`` and ``numpy.datetime64`` expose the same attributes we
        # need; accept anything that quacks like a datetime.
        for attribute in ("year", "month", "day", "hour"):
            if not hasattr(value, attribute):
                raise TypeError(
                    "Calendar features require datetime-like timestamps with "
                    f"year/month/day/hour attributes, got {type(value).__name__}."
                )
        return value  # type: ignore[return-value]
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def days_in_year(year: int) -> int:
    """Number of days in ``year`` under the Gregorian leap rule."""
    year = int(year)
    is_leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    return 366 if is_leap else 365


def fractional_utc_hour(value: datetime) -> float:
    """Hour of day in ``[0, 24)`` including minutes, seconds and microseconds."""
    value = _to_utc(value)
    return (
        float(value.hour)
        + float(value.minute) / 60.0
        + float(value.second) / 3600.0
        + float(getattr(value, "microsecond", 0)) / 3.6e9
    )


def _day_of_year(value: datetime) -> int:
    """1-based day of year, leap-year aware."""
    value = _to_utc(value)
    if hasattr(value, "timetuple"):
        try:
            return int(value.timetuple().tm_yday)
        except (AttributeError, ValueError):  # pragma: no cover - cftime fallback
            pass
    return int(getattr(value, "dayofyr", 0)) or 1


def fractional_day_of_year(value: datetime) -> float:
    """``day_of_year - 1 + fractional_UTC_hour / 24``, in ``[0, days_in_year)``."""
    return float(_day_of_year(value) - 1) + fractional_utc_hour(value) / 24.0


def year_phase(value: datetime) -> float:
    """Seasonal phase in ``[0, 1)``.

    ``(day_of_year - 1 + fractional_UTC_hour / 24) / days_in_year``. Dividing by
    the *actual* length of the year keeps 1 March at the same phase in leap and
    common years and makes 31 December 23:00 adjacent to 1 January 00:00.
    """
    value = _to_utc(value)
    return fractional_day_of_year(value) / float(days_in_year(value.year))


def local_mean_solar_hour(
    utc_hour: torch.Tensor, longitude_east_degrees: torch.Tensor
) -> torch.Tensor:
    """Local **mean solar** hour in ``[0, 24)``.

    ``(UTC_hour + longitude_east_degrees / 15) mod 24``.

    This is mean solar time, which is neither civil local time (no time zones,
    no daylight saving) nor apparent solar time (no equation of time). It exists
    to give the network the UTC x longitude interaction explicitly, because a
    diurnal ozone cycle is organised by solar position rather than by UTC.
    """
    return torch.remainder(utc_hour + longitude_east_degrees / 15.0, 24.0)


def valid_times(
    init_time: Sequence[datetime], lead_hours: Sequence[float] | torch.Tensor
) -> tuple[datetime, ...]:
    """Valid timestamps ``t0 + ell`` for one initialization per entry.

    ``init_time`` and ``lead_hours`` must have the same length: every frame
    carries its own lead, so a rollout never reuses the initialization hour.
    """
    if isinstance(lead_hours, torch.Tensor):
        leads = [float(v) for v in lead_hours.detach().reshape(-1).tolist()]
    else:
        leads = [float(v) for v in lead_hours]
    if len(init_time) != len(leads):
        raise ValueError(
            f"valid_times needs one lead per initialization; got {len(init_time)} "
            f"timestamps and {len(leads)} leads."
        )
    out: list[datetime] = []
    for stamp, lead in zip(init_time, leads):
        if not math.isfinite(lead):
            raise ValueError(f"Forecast lead must be finite, got {lead!r}.")
        out.append(_to_utc(stamp) + timedelta(hours=lead))
    return tuple(out)


def cos_solar_zenith_angle(
    valid_time: Sequence[datetime],
    latitude_degrees: torch.Tensor,
    longitude_degrees: torch.Tensor,
) -> torch.Tensor:
    """Cosine of the solar zenith angle on a lat/lon grid.

    Returns ``[N, H, W]`` for ``N = len(valid_time)``, ``H = latitude.numel()``
    and ``W = longitude.numel()``.

    The orbital approximation (1995 elements, first-order equation of centre)
    is the one already used by :func:`aurora.insolation.insolation`, reproduced
    in torch so the feature can be built on the training device without a
    NumPy round trip. It is an *approximation*: no atmospheric refraction, no
    equation of time beyond first order, and no topographic shading. Values are
    negative at night and are deliberately **not** clipped, because the sign
    carries the day/night transition the network needs.

    :func:`aurora.insolation.insolation` additionally multiplies by the
    Earth-Sun distance factor ``rho**-2``; that factor is a function of the day
    of year alone and is already representable from the seasonal features, so
    it is omitted here to keep this channel a pure geometry term.
    """
    stamps = [_to_utc(stamp) for stamp in valid_time]
    lat = latitude_degrees.detach().reshape(-1).to(torch.float64)
    lon = longitude_degrees.detach().reshape(-1).to(device=lat.device, dtype=torch.float64)
    if lat.numel() == 0 or lon.numel() == 0:
        raise ValueError("cos_solar_zenith_angle requires non-empty lat/lon axes.")

    # Fractional days since the start of the frame's own year, matching
    # ``aurora.insolation``'s ``(dates - start_of_year) / 1 day``.
    days = torch.tensor(
        [fractional_day_of_year(stamp) for stamp in stamps],
        dtype=torch.float64,
        device=lat.device,
    )

    obliquity = math.radians(23.4441)
    eccentricity = 0.016715
    perihelion = math.radians(282.7)
    beta = math.sqrt(1.0 - eccentricity**2)

    lambda_m0 = eccentricity * (1.0 + beta) * math.sin(perihelion)
    lambda_m = lambda_m0 + 2.0 * math.pi * (days - 80.5) / 365.0
    lambda_true = lambda_m + 2.0 * eccentricity * torch.sin(lambda_m - perihelion)
    declination = torch.asin(math.sin(obliquity) * torch.sin(lambda_true))

    # Hour angle: 2*pi*(fractional day + lon/360). ``cos`` is even and
    # 2*pi-periodic, so the longitude wrap at 0/360 degrees is exact.
    hour_angle = 2.0 * math.pi * (days[:, None] + lon[None, :] / 360.0)

    lat_radians = torch.deg2rad(lat)
    sin_term = torch.sin(lat_radians)[None, :, None] * torch.sin(declination)[:, None, None]
    cos_term = (
        torch.cos(lat_radians)[None, :, None]
        * torch.cos(declination)[:, None, None]
        * torch.cos(hour_angle)[:, None, :]
    )
    return (sin_term - cos_term).to(torch.float32)


@dataclass(frozen=True)
class CalendarSpec:
    """Everything the builder needs about one batch of trajectory frames.

    Attributes:
        init_time: one initialization timestamp per flattened entry. Entries
            belonging to the same trajectory repeat the same ``t0``.
        lead_hours: cumulative forecast lead in hours, one per entry.
        elapsed_hours: hours since the previous frame of the same trajectory.
            ``None`` means "same as ``lead_hours``", i.e. the first frame.
        has_predecessor: ``False`` at a trajectory start or after a gap.
    """

    init_time: tuple[datetime, ...]
    lead_hours: torch.Tensor
    elapsed_hours: torch.Tensor | None = None
    has_predecessor: torch.Tensor | None = None

    def __post_init__(self) -> None:
        count = len(self.init_time)
        if count == 0:
            raise ValueError("CalendarSpec requires at least one initialization time.")
        for name in ("lead_hours", "elapsed_hours", "has_predecessor"):
            value = getattr(self, name)
            if value is None:
                continue
            if value.reshape(-1).numel() != count:
                raise ValueError(
                    f"CalendarSpec.{name} must have one entry per initialization "
                    f"timestamp; got {value.reshape(-1).numel()} for {count} timestamps."
                )
        if not bool(torch.isfinite(self.lead_hours.reshape(-1)).all()):
            raise ValueError("CalendarSpec.lead_hours must be finite.")
        if bool((self.lead_hours.reshape(-1) < 0).any()):
            raise ValueError("CalendarSpec.lead_hours must be non-negative.")


class CalendarFeatureBuilder:
    """Builds the scalar and spatial calendar features for a packed batch.

    One instance is created per :class:`~finetune.refinement.packing.FieldPacking`
    and reused everywhere, so the trainer, the sampler and the evaluator cannot
    drift apart.

    Args:
        latitude: grid latitudes in degrees, north to south or south to north.
        longitude: grid longitudes in degrees east.
        lead_time_scale_hours: divisor applied to lead / elapsed hours before
            embedding. Conditioning scale only; it never changes the rollout.
        init_cycle_hours: period of the initialization cycle, in hours. ``24``
            encodes 00/06/12/18 UTC cycles on a daily circle.
    """

    def __init__(
        self,
        latitude: Sequence[float],
        longitude: Sequence[float],
        *,
        lead_time_scale_hours: float = 72.0,
        init_cycle_hours: float = 24.0,
    ) -> None:
        lat = torch.as_tensor(tuple(float(v) for v in latitude), dtype=torch.float32)
        lon = torch.as_tensor(tuple(float(v) for v in longitude), dtype=torch.float32)
        if lat.numel() == 0 or lon.numel() == 0:
            raise ValueError("CalendarFeatureBuilder requires non-empty lat/lon axes.")
        if not bool(torch.isfinite(lat).all()) or not bool(torch.isfinite(lon).all()):
            raise ValueError("CalendarFeatureBuilder lat/lon coordinates must be finite.")
        if bool((lat.abs() > 90.0).any()):
            raise ValueError(
                "CalendarFeatureBuilder latitudes must lie within [-90, 90] degrees."
            )
        scale = float(lead_time_scale_hours)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"lead_time_scale_hours must be positive, got {lead_time_scale_hours!r}."
            )
        cycle = float(init_cycle_hours)
        if not math.isfinite(cycle) or cycle <= 0.0:
            raise ValueError(f"init_cycle_hours must be positive, got {init_cycle_hours!r}.")

        self.latitude = lat
        self.longitude = lon
        self.lead_time_scale_hours = scale
        self.init_cycle_hours = cycle
        self._geometry_cache: dict[tuple, torch.Tensor] = {}

    # -- sizes -----------------------------------------------------------
    @property
    def num_scalar_features(self) -> int:
        return len(SCALAR_FEATURE_NAMES)

    @property
    def num_spatial_features(self) -> int:
        return len(SPATIAL_FEATURE_NAMES)

    @property
    def height(self) -> int:
        return int(self.latitude.numel())

    @property
    def width(self) -> int:
        return int(self.longitude.numel())

    def convention(self) -> dict[str, object]:
        """Machine-readable record persisted in the checkpoint signature."""
        return {
            "calendar_convention": CALENDAR_CONVENTION,
            "scalar_feature_names": list(SCALAR_FEATURE_NAMES),
            "spatial_feature_names": list(SPATIAL_FEATURE_NAMES),
            "lead_time_scale_hours": self.lead_time_scale_hours,
            "init_cycle_hours": self.init_cycle_hours,
            "height": self.height,
            "width": self.width,
        }

    # -- scalar features -------------------------------------------------
    def scalar_features(
        self,
        spec: CalendarSpec,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return ``[N, num_scalar_features]`` in :data:`SCALAR_FEATURE_NAMES` order."""
        # All calendar arithmetic is done in float64 on the CPU: the inputs are
        # Python datetimes, and mixing a CUDA ``lead_hours`` tensor with the
        # CPU-built phase tensors would fail in ``torch.stack``. The finished
        # matrix is moved to the requested device once, at the end.
        leads = spec.lead_hours.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        stamps = valid_times(spec.init_time, leads)
        inits = [_to_utc(stamp) for stamp in spec.init_time]

        valid_phase = torch.tensor(
            [year_phase(stamp) for stamp in stamps], dtype=torch.float64
        )
        valid_hour = torch.tensor(
            [fractional_utc_hour(stamp) for stamp in stamps], dtype=torch.float64
        )
        init_phase = torch.tensor(
            [year_phase(stamp) for stamp in inits], dtype=torch.float64
        )
        init_hour = torch.tensor(
            [fractional_utc_hour(stamp) for stamp in inits], dtype=torch.float64
        )

        two_pi = 2.0 * math.pi
        scaled_lead = leads / self.lead_time_scale_hours
        if spec.elapsed_hours is None:
            elapsed = leads.clone()
        else:
            elapsed = (
                spec.elapsed_hours.detach()
                .reshape(-1)
                .to(device="cpu", dtype=torch.float64)
            )
            if not bool(torch.isfinite(elapsed).all()):
                raise ValueError("CalendarSpec.elapsed_hours must be finite.")
        if spec.has_predecessor is None:
            predecessor = torch.zeros_like(leads)
        else:
            predecessor = (
                spec.has_predecessor.detach()
                .reshape(-1)
                .to(device="cpu", dtype=torch.float64)
                > 0.5
            ).to(torch.float64)

        features = torch.stack(
            [
                torch.sin(two_pi * valid_phase),
                torch.cos(two_pi * valid_phase),
                torch.sin(two_pi * valid_hour / 24.0),
                torch.cos(two_pi * valid_hour / 24.0),
                torch.sin(two_pi * init_hour / self.init_cycle_hours),
                torch.cos(two_pi * init_hour / self.init_cycle_hours),
                torch.sin(two_pi * init_phase),
                torch.cos(two_pi * init_phase),
                scaled_lead,
                torch.log1p(scaled_lead.clamp(min=0.0)),
                scaled_lead.clamp(min=0.0).sqrt(),
                elapsed / self.lead_time_scale_hours,
                predecessor,
            ],
            dim=-1,
        )
        if features.shape[-1] != self.num_scalar_features:
            raise RuntimeError(
                "Calendar scalar feature count drifted from SCALAR_FEATURE_NAMES: "
                f"built {features.shape[-1]}, expected {self.num_scalar_features}."
            )
        return features.to(device=device, dtype=dtype)

    # -- spatial features ------------------------------------------------
    def _static_geometry(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """``[4, H, W]``: spherical x/y/z plus normalized latitude."""
        key = (torch.device(device), dtype)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        lat = torch.deg2rad(self.latitude.to(torch.float64))
        lon = torch.deg2rad(self.longitude.to(torch.float64))
        cos_lat = torch.cos(lat)[:, None]
        sin_lat = torch.sin(lat)[:, None]
        cos_lon = torch.cos(lon)[None, :]
        sin_lon = torch.sin(lon)[None, :]
        height, width = self.height, self.width
        geometry = torch.stack(
            [
                (cos_lat * cos_lon).expand(height, width),
                (cos_lat * sin_lon).expand(height, width),
                sin_lat.expand(height, width),
                (self.latitude.to(torch.float64) / 90.0)[:, None].expand(height, width),
            ],
            dim=0,
        ).to(device=device, dtype=dtype)
        self._geometry_cache[key] = geometry
        return geometry

    def spatial_features(
        self,
        spec: CalendarSpec,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return ``[N, num_spatial_features, H, W]`` in :data:`SPATIAL_FEATURE_NAMES` order."""
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        stamps = valid_times(spec.init_time, spec.lead_hours.detach().reshape(-1))
        count = len(stamps)

        geometry = self._static_geometry(target_device, dtype)
        geometry = geometry[None].expand(count, geometry.shape[0], self.height, self.width)

        valid_hour = torch.tensor(
            [fractional_utc_hour(stamp) for stamp in stamps], dtype=torch.float32
        )
        solar_hour = local_mean_solar_hour(
            valid_hour[:, None], self.longitude[None, :]
        )  # [N, W]
        two_pi = 2.0 * math.pi
        solar_sin = torch.sin(two_pi * solar_hour / 24.0)
        solar_cos = torch.cos(two_pi * solar_hour / 24.0)
        solar = torch.stack([solar_sin, solar_cos], dim=1)  # [N, 2, W]
        solar = solar[:, :, None, :].expand(count, 2, self.height, self.width)
        solar = solar.to(device=target_device, dtype=dtype)

        zenith = cos_solar_zenith_angle(stamps, self.latitude, self.longitude)
        zenith = zenith[:, None].to(device=target_device, dtype=dtype)

        features = torch.cat([geometry, solar, zenith], dim=1)
        if features.shape[1] != self.num_spatial_features:
            raise RuntimeError(
                "Calendar spatial feature count drifted from SPATIAL_FEATURE_NAMES: "
                f"built {features.shape[1]}, expected {self.num_spatial_features}."
            )
        return features

    # -- convenience -----------------------------------------------------
    @staticmethod
    def expand_initializations(
        init_time: Iterable[datetime], repeats: int
    ) -> tuple[datetime, ...]:
        """Repeat each ``t0`` ``repeats`` times in **lead-major** order.

        Produces the order ``[b0l0, b0l1, ..., b1l0, ...]``, which is what
        :meth:`AuroraTwoPhaseRefiner.build_temporal_context` emits when a
        ``[B, S, ...]`` trajectory is reshaped to ``[B*S, ...]``.

        .. warning::
           This is **not** the layout produced by
           :meth:`finetune.refinement.integration.LeadStepBuffer.pack`, which
           concatenates whole lead blocks and therefore uses
           ``[l0b0, l0b1, ..., l1b0, ...]``. Use
           :meth:`expand_initializations_lead_blocked` for that path. Pairing
           the two silently attaches each frame's calendar to the wrong sample,
           so the two orders are separate named methods rather than a flag.
        """
        repeats = int(repeats)
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}.")
        return tuple(stamp for stamp in init_time for _ in range(repeats))

    @staticmethod
    def expand_initializations_lead_blocked(
        init_time: Iterable[datetime], repeats: int
    ) -> tuple[datetime, ...]:
        """Repeat each ``t0`` in **lead-blocked** order ``[l0b0, l0b1, ...]``.

        Matches :meth:`finetune.refinement.integration.LeadStepBuffer.pack`,
        which concatenates one complete batch per rollout step.
        """
        repeats = int(repeats)
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}.")
        stamps = tuple(init_time)
        return tuple(stamp for _ in range(repeats) for stamp in stamps)

    @staticmethod
    def lead_blocked_to_sequence(
        value: torch.Tensor, batch: int, steps: int
    ) -> torch.Tensor:
        """Restore ``[B, S, ...]`` from a lead-blocked ``[S*B, ...]`` tensor.

        ``LeadStepBuffer.pack`` emits rows as ``position * batch + sample``.
        Reshaping such a tensor directly to ``[B, S, ...]`` would transpose the
        two axes and build a "trajectory" out of unrelated initializations, so
        the conversion is done explicitly here and covered by tests.
        """
        if value.shape[0] != batch * steps:
            raise ValueError(
                f"Expected {batch * steps} lead-blocked rows for batch={batch}, "
                f"steps={steps}; got {value.shape[0]}."
            )
        return value.reshape(steps, batch, *value.shape[1:]).transpose(0, 1)

    @staticmethod
    def sequence_to_lead_blocked(value: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`lead_blocked_to_sequence` for a ``[B, S, ...]`` tensor."""
        if value.ndim < 2:
            raise ValueError(
                f"Expected a [batch, steps, ...] tensor, got {tuple(value.shape)}."
            )
        batch, steps = value.shape[0], value.shape[1]
        return value.transpose(0, 1).reshape(steps * batch, *value.shape[2:])
