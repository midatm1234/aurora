"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as RGI

from aurora.normalisation import (
    normalise_atmos_var,
    normalise_surf_var,
    unnormalise_atmos_var,
    unnormalise_surf_var,
)

__all__ = ["Metadata", "Batch"]


@dataclasses.dataclass
class Metadata:
    """Metadata in a batch.

    Args:
        lat (:class:`torch.Tensor`): Latitudes.
        lon (:class:`torch.Tensor`): Longitudes.
        time (tuple[datetime, ...]): For every batch element, the time.
        atmos_levels (tuple[int | float, ...]): Pressure levels for the atmospheric variables in
            hPa.
        rollout_step (int, optional): How many roll-out steps were used to produce this prediction.
            If equal to `0`, which is the default, then this means that this is not a prediction,
            but actual data. This field is automatically populated by the model and used to use a
            separate LoRA for every roll-out step. Generally, you are safe to ignore this field.
    """

    lat: torch.Tensor
    lon: torch.Tensor
    time: tuple[datetime, ...]
    atmos_levels: tuple[int | float, ...]
    rollout_step: int = 0

    def __post_init__(self):
        if not (torch.all(self.lat <= 90) and torch.all(self.lat >= -90)):
            raise ValueError("Latitudes must be in the range [-90, 90].")
        if not (torch.all(self.lon >= 0) and torch.all(self.lon < 360)):
            raise ValueError("Longitudes must be in the range [0, 360).")

        # Validate vector-valued latitudes and longitudes:
        if self.lat.dim() == self.lon.dim() == 1:
            if not torch.all(self.lat[1:] - self.lat[:-1] < 0):
                raise ValueError("Latitudes must be strictly decreasing.")
            if not torch.all(self.lon[1:] - self.lon[:-1] > 0):
                raise ValueError("Longitudes must be strictly increasing.")

        # Validate matrix-valued latitudes and longitudes:
        elif self.lat.dim() == self.lon.dim() == 2:
            if not torch.all(self.lat[1:, :] - self.lat[:-1, :]):
                raise ValueError("Latitudes must be strictly decreasing along every column.")
            if not torch.all(self.lon[:, 1:] - self.lon[:, :-1] > 0):
                raise ValueError("Longitudes must be strictly increasing along every row.")

        else:
            raise ValueError(
                "The latitudes and longitudes must either both be vectors or both be matrices."
            )


@dataclasses.dataclass
class Batch:
    """A batch of data.

    Args:
        surf_vars (dict[str, :class:`torch.Tensor`]): Surface-level variables with shape
            `(b, t, h, w)`.
        static_vars (dict[str, :class:`torch.Tensor`]): Static variables with shape `(h, w)`.
        atmos_vars (dict[str, :class:`torch.Tensor`]): Atmospheric variables with shape
            `(b, t, c, h, w)`.
        metadata (:class:`Metadata`): Metadata associated to this batch.
    """

    surf_vars: dict[str, torch.Tensor]
    static_vars: dict[str, torch.Tensor]
    atmos_vars: dict[str, torch.Tensor]
    metadata: Metadata

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Get the spatial shape from an arbitrary surface-level variable."""
        return next(iter(self.surf_vars.values())).shape[-2:]

    def normalise(
        self,
        surf_stats: dict[str, tuple[float, float]],
    ) -> "Batch":
        """Normalise all variables in the batch.

        Args:
            surf_stats (dict[str, tuple[float, float]]): For these surface-level variables, adjust
                the normalisation to the given tuple consisting of a new location and scale.

        Returns:
            :class:`.Batch`: Normalised batch.
        """
        return Batch(
            surf_vars={
                k: normalise_surf_var(v, k, stats=surf_stats) for k, v in self.surf_vars.items()
            },
            static_vars={
                k: normalise_surf_var(v, k, stats=surf_stats) for k, v in self.static_vars.items()
            },
            atmos_vars={
                k: normalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def unnormalise(
        self,
        surf_stats: dict[str, tuple[float, float]],
    ) -> "Batch":
        """Unnormalise all variables in the batch.

        Args:
            surf_stats (dict[str, tuple[float, float]]): For these surface-level variables, adjust
                the normalisation to the given tuple consisting of a new location and scale.

        Returns:
            :class:`.Batch`: Unnormalised batch.
        """
        return Batch(
            surf_vars={
                k: unnormalise_surf_var(v, k, stats=surf_stats) for k, v in self.surf_vars.items()
            },
            static_vars={
                k: unnormalise_surf_var(v, k, stats=surf_stats) for k, v in self.static_vars.items()
            },
            atmos_vars={
                k: unnormalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def crop(self, patch_size: int) -> "Batch":
        """Crop the variables in the batch to patch size `patch_size`."""
        h, w = self.spatial_shape

        if w % patch_size != 0:
            raise ValueError("Width of the data must be a multiple of the patch size.")

        if h % patch_size == 0:
            return self
        elif h % patch_size == 1:
            return Batch(
                surf_vars={k: v[..., :-1, :] for k, v in self.surf_vars.items()},
                static_vars={k: v[..., :-1, :] for k, v in self.static_vars.items()},
                atmos_vars={k: v[..., :-1, :] for k, v in self.atmos_vars.items()},
                metadata=Metadata(
                    lat=self.metadata.lat[:-1],
                    lon=self.metadata.lon,
                    atmos_levels=self.metadata.atmos_levels,
                    time=self.metadata.time,
                    rollout_step=self.metadata.rollout_step,
                ),
            )
        else:
            raise ValueError(
                f"There can at most be one latitude too many, "
                f"but there are {h % patch_size} too many."
            )

    def _fmap(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Batch":
        return Batch(
            surf_vars={k: f(v) for k, v in self.surf_vars.items()},
            static_vars={k: f(v) for k, v in self.static_vars.items()},
            atmos_vars={k: f(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=f(self.metadata.lat),
                lon=f(self.metadata.lon),
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to(self, device: str | torch.device) -> "Batch":
        """Move the batch to another device."""
        return self._fmap(lambda x: x.to(device))

    def type(self, t: type) -> "Batch":
        """Convert everything to type `t`.

        Note: Metadata lat/lon are kept in their original dtype to preserve
        coordinate precision for validation checks and downstream use.
        """
        return Batch(
            surf_vars={k: v.type(t) for k, v in self.surf_vars.items()},
            static_vars={k: v.type(t) for k, v in self.static_vars.items()},
            atmos_vars={k: v.type(t) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=self.metadata.lat,
                lon=self.metadata.lon,
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def regrid(self, res: float) -> "Batch":
        """Regrid the batch to a `res` degrees resolution.

        This results in `float32` data on the CPU.

        This function is not optimised for either speed or accuracy. Use at your own risk.
        """

        if self.metadata.lon.dim() != 1 or not _longitude_is_periodic(
            _np(self.metadata.lon)
        ):
            raise ValueError(
                "Batch.regrid creates a global grid and therefore requires a complete "
                "periodic source longitude axis."
            )

        shape = (round(180 / res) + 1, round(360 / res))
        lat_new = torch.from_numpy(np.linspace(90, -90, shape[0]))
        lon_new = torch.from_numpy(np.linspace(0, 360, shape[1], endpoint=False))
        interpolate_res = partial(
            interpolate,
            lat=self.metadata.lat,
            lon=self.metadata.lon,
            lat_new=lat_new,
            lon_new=lon_new,
            periodic=True,
        )

        return Batch(
            surf_vars={k: interpolate_res(v) for k, v in self.surf_vars.items()},
            static_vars={k: interpolate_res(v) for k, v in self.static_vars.items()},
            atmos_vars={k: interpolate_res(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=lat_new,
                lon=lon_new,
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to_netcdf(self, path: str | Path) -> None:
        """Write the batch to a file.

        This requires `xarray` and `netcdf4` to be installed.
        """
        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e

        ds = xr.Dataset(
            {
                **{
                    f"surf_{k}": (("batch", "history", "latitude", "longitude"), _np(v))
                    for k, v in self.surf_vars.items()
                },
                **{
                    f"static_{k}": (("latitude", "longitude"), _np(v))
                    for k, v in self.static_vars.items()
                },
                **{
                    f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"), _np(v))
                    for k, v in self.atmos_vars.items()
                },
            },
            coords={
                "latitude": _np(self.metadata.lat),
                "longitude": _np(self.metadata.lon),
                "time": list(self.metadata.time),
                "level": list(self.metadata.atmos_levels),
                "rollout_step": self.metadata.rollout_step,
            },
        )
        ds["latitude"].attrs.update(
            {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "axis": "Y",
            }
        )
        ds["longitude"].attrs.update(
            {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
                "axis": "X",
            }
        )
        if self.metadata.lon.dim() == 1 and _longitude_is_periodic(_np(self.metadata.lon)):
            ds["longitude"].attrs.update({"modulo": 360.0, "topology": "circular"})
        ds["time"].attrs.update({"standard_name": "time", "axis": "T"})
        ds.attrs["Conventions"] = "CF-1.10"
        for coord_name in ("latitude", "longitude", "time", "level"):
            ds[coord_name].encoding["_FillValue"] = None
        # Enable compression and chunking to reduce file size.
        encoding = {}
        for name, var in ds.data_vars.items():
            enc: dict = {"zlib": True, "complevel": 4}
            dims = var.dims
            chunks: dict = {}
            if "history" in dims:
                chunks["history"] = 1
            if "level" in dims:
                chunks["level"] = 1
            if chunks:
                enc["chunksizes"] = tuple(chunks.get(d, s) for d, s in zip(dims, var.shape))
            encoding[name] = enc
        ds.to_netcdf(path, encoding=encoding)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> "Batch":
        """Load a batch from a file."""
        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e

        ds = xr.load_dataset(path, engine="netcdf4")

        surf_vars: List[str] = []
        static_vars: List[str] = []
        atmos_vars: List[str] = []

        for k in ds:
            if k.startswith("surf_"):
                surf_vars.append(k.removeprefix("surf_"))
            elif k.startswith("static_"):
                static_vars.append(k.removeprefix("static_"))
            elif k.startswith("atmos_"):
                atmos_vars.append(k.removeprefix("atmos_"))

        lon, lon_indices = _canonical_longitudes(ds.longitude.values)

        return Batch(
            surf_vars={
                k: torch.from_numpy(ds[f"surf_{k}"].values[..., lon_indices])
                for k in surf_vars
            },
            static_vars={
                k: torch.from_numpy(ds[f"static_{k}"].values[..., lon_indices])
                for k in static_vars
            },
            atmos_vars={
                k: torch.from_numpy(ds[f"atmos_{k}"].values[..., lon_indices])
                for k in atmos_vars
            },
            metadata=Metadata(
                lat=torch.from_numpy(np.asarray(ds.latitude.values).copy()),
                lon=torch.from_numpy(lon.copy()),
                time=tuple(ds.time.values.astype("datetime64[s]").tolist()),
                atmos_levels=tuple(ds.level.values),
                rollout_step=int(ds.rollout_step.values),
            ),
        )


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _canonical_longitudes(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core, dependency-free longitude canonicalisation for Batch I/O."""
    raw = np.asarray(lon, dtype=np.float64)
    wrapped = np.mod(raw, 360.0)
    wrapped[np.isclose(wrapped, 360.0, rtol=0.0, atol=1e-8)] = 0.0
    if wrapped.ndim != 1 or not np.all(np.isfinite(wrapped)):
        raise ValueError("Longitude must be a finite one-dimensional coordinate.")
    order = np.argsort(wrapped, kind="stable")
    sorted_lon = wrapped[order]
    keep = np.ones(sorted_lon.size, dtype=bool)
    if sorted_lon.size > 1:
        keep[1:] = np.diff(sorted_lon) > 1e-7
        for index in np.where(~keep)[0]:
            turns = abs(float(raw[order[index]] - raw[order[index - 1]])) / 360.0
            if turns < 0.5 or not np.isclose(
                turns, round(turns), rtol=0.0, atol=1e-7 / 360.0,
            ):
                raise ValueError(
                    "Longitude contains a duplicate that is not a cyclic endpoint."
                )
    return sorted_lon[keep], order[keep]


def _longitude_is_periodic(lon: np.ndarray) -> bool:
    lon, _ = _canonical_longitudes(lon)
    if lon.size < 3:
        return False
    spacing = np.diff(lon)
    median = float(np.median(spacing))
    wrap = float(lon[0] + 360.0 - lon[-1])
    tolerance = max(0.5 * median, 1e-3)
    return (
        median > 0
        and abs(wrap - median) <= tolerance
        and float(np.max(spacing) - np.min(spacing)) <= tolerance
    )


def interpolate(
    v: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat_new: torch.Tensor,
    lon_new: torch.Tensor,
    periodic: bool | None = None,
) -> torch.Tensor:
    """Interpolate a variable `v` with latitudes `lat` and longitudes `lon` to new latitudes
    `lat_new` and new longitudes `lon_new`."""
    # Perform the interpolation in double precision.
    return torch.from_numpy(
        interpolate_numpy(
            v.double().numpy(),
            lat.double().numpy(),
            lon.double().numpy(),
            lat_new.double().numpy(),
            lon_new.double().numpy(),
            periodic=periodic,
        )
    ).float()


def interpolate_numpy(
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_new: np.ndarray,
    lon_new: np.ndarray,
    periodic: bool | None = None,
) -> np.ndarray:
    """Like :func:`.interpolate`, but for NumPy tensors."""

    # Use one internal convention for either common source convention and move
    # every data column with its coordinate.
    lon_canonical, lon_indices = _canonical_longitudes(lon)
    v = np.asarray(v)[..., lon_indices]
    lon_new_canonical = np.mod(np.asarray(lon_new, dtype=np.float64), 360.0)
    detected_periodic = _longitude_is_periodic(lon_canonical)
    if periodic is None:
        periodic = detected_periodic
    elif periodic and not detected_periodic:
        raise ValueError(
            "periodic=True requires a complete, uniformly spaced global longitude grid."
        )
    if periodic:
        lon_interp = np.concatenate(
            (lon_canonical[-1:] - 360, lon_canonical, lon_canonical[:1] + 360)
        )
    else:
        spacing = np.diff(lon_canonical)
        if spacing.size > 1 and np.max(spacing) > 3.0 * np.median(spacing):
            raise ValueError(
                "A non-periodic longitude grid cannot cross the 0/360 wrap in canonical "
                "coordinates. Regrid or select a non-wrapping regional interval."
            )
        tolerance = max(1e-7, float(np.median(spacing)) * 1e-4) if spacing.size else 1e-7
        if np.any(lon_new_canonical < lon_canonical[0] - tolerance) or np.any(
            lon_new_canonical > lon_canonical[-1] + tolerance
        ):
            raise ValueError(
                "Target longitude lies outside the non-periodic source domain; refusing "
                "linear extrapolation across unrelated Earth longitudes."
            )
        lon_interp = lon_canonical

    # Merge all batch dimensions into one.
    batch_shape = v.shape[:-2]
    v = v.reshape(-1, *v.shape[-2:])

    # Loop over all batch elements.
    vs_regridded = []
    for vi in v:
        if periodic:
            # Cyclic halo makes the first and last grid columns neighbours.
            vi = np.concatenate((vi[:, -1:], vi, vi[:, :1]), axis=1)

        rgi = RGI(
            (lat, lon_interp),
            vi,
            method="linear",
            bounds_error=False,  # Allow out of bounds, for the latitudes.
            fill_value=None,  # Extrapolate latitudes if they are out of bounds.
        )
        lat_new_grid, lon_new_grid = np.meshgrid(
            lat_new,
            lon_new_canonical,
            indexing="ij",
            sparse=True,
        )
        vs_regridded.append(rgi((lat_new_grid, lon_new_grid)))

    # Recreate the batch dimensions.
    v_regridded = np.stack(vs_regridded, axis=0)
    v_regridded = v_regridded.reshape(*batch_shape, lat_new.shape[0], lon_new.shape[0])

    return v_regridded
