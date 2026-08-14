"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Real-data benchmark harness for the Phase-2 stochastic residual refiners.

The harness answers one question that neither code inspection nor a training
loss can answer: **does a refinement head actually improve the Aurora rollout
against ground truth?**

It reuses *unrefined* Aurora rollout NetCDF files, pairs every forecast valid
time with the matching CAMS ground truth,
and builds exactly the packed, normalized residual-refinement problem that
:mod:`finetune.refinement.two_phase` solves during training::

    rollout_norm = encode(aurora_rollout)
    target_norm  = encode(cams_truth)
    residual     = target_norm - rollout_norm

Because Aurora itself is frozen in Phase 2 and its raw rollout is already on disk,
the benchmark trains and scores refiner heads without re-running Aurora. That
makes a full four-head comparison cheap enough to run as a regression test.

Metrics are computed in **physical units** on the decoded fields, per variable,
per pressure level and per forecast lead, against the same ground truth:

* MAE, RMSE, mean bias
* area-weighted spatial (pattern) correlation
* spatial standard deviation ratio
* P90 / P95 / P99 quantile bias and tail MAE

Usage::

    python -m finetune.refinement.benchmark --case o3_global --epochs 8
    python -m finetune.refinement.benchmark --case no2_uswest --heads flow_matching_transformer
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from finetune.refinement.base import build_refiner, masked_loss
from finetune.refinement.config import RefinementConfig, resolve_refinement_config
from finetune.refinement.losses import area_weights_from_latitudes
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace

__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "evaluate_fields",
    "evaluate_channels",
    "grouped_purged_split",
    "load_case",
    "run_benchmark",
]


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """A raw Aurora rollout collection plus its ground-truth dataset."""

    name: str
    rollout_dir: str
    truth_path: str
    surf_variables: tuple[str, ...]
    atmos_variables: tuple[str, ...]
    pressure_levels: tuple[float, ...]
    lon_periodic: bool
    lead_step_hours: float = 12.0
    expected_lead_hours: tuple[float, ...] = (12.0, 24.0, 36.0, 48.0, 60.0, 72.0)
    rollout_glob: str = "rollout_*.nc"
    source_kind: str = "raw_aurora"
    source_model_stage: str = "pretrained"


CASES: dict[str, BenchmarkCase] = {
    "o3_global": BenchmarkCase(
        name="o3_global",
        rollout_dir="examples/outputs/cams_rollouts",
        truth_path="data/O3_global_3day_lead/test.nc",
        surf_variables=("gtco3",),
        atmos_variables=("go3",),
        pressure_levels=(1000.0, 500.0, 100.0, 50.0),
        lon_periodic=True,
    ),
    "no2_uswest": BenchmarkCase(
        name="no2_uswest",
        rollout_dir="examples/outputs/cams_rollouts",
        truth_path="data/NO2_US-WEST_3day_lead/test.nc",
        surf_variables=("tcno2",),
        atmos_variables=("no2",),
        pressure_levels=(1000.0, 925.0, 850.0),
        lon_periodic=False,
    ),
}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkDataset:
    """Packed rollout/truth pairs in normalized target space.

    ``rollout`` and ``target`` are ``[S, C, H, W]`` with ``S`` samples; every
    sample is one (initialization, forecast lead) pair, which is exactly the
    effective batch layout the packed refiners consume.
    """

    packing: FieldPacking
    rollout: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    lead_hours: torch.Tensor
    lead_index: torch.Tensor
    lat: np.ndarray
    lon: np.ndarray
    case: BenchmarkCase
    static: torch.Tensor | None = None
    initialization_ids: tuple[str, ...] = ()
    initialization_times: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype="datetime64[ns]")
    )
    valid_times: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype="datetime64[ns]")
    )
    source_files: tuple[str, ...] = ()
    source_validation: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.rollout.shape[0])

    def subset(self, indices: Sequence[int]) -> BenchmarkDataset:
        positions = list(indices)
        idx = torch.as_tensor(positions, dtype=torch.long)
        init_ids = (
            tuple(self.initialization_ids[i] for i in positions)
            if self.initialization_ids
            else ()
        )
        return BenchmarkDataset(
            packing=self.packing,
            rollout=self.rollout[idx],
            target=self.target[idx],
            valid=self.valid[idx],
            lead_hours=self.lead_hours[idx],
            lead_index=self.lead_index[idx],
            lat=self.lat,
            lon=self.lon,
            case=self.case,
            static=self.static,
            initialization_ids=init_ids,
            initialization_times=(
                self.initialization_times[positions]
                if self.initialization_times.size
                else self.initialization_times.copy()
            ),
            valid_times=(
                self.valid_times[positions]
                if self.valid_times.size
                else self.valid_times.copy()
            ),
            source_files=self.source_files,
            source_validation=dict(self.source_validation),
        )


def _channel_specs(
    case: BenchmarkCase, levels: Sequence[float]
) -> tuple[list[ChannelSpec], list[tuple[str, float | None]]]:
    """Packed channel layout plus the (variable, pressure-level) source of each."""
    from aurora.normalisation import level_to_str, locations, scales

    specs: list[ChannelSpec] = []
    sources: list[tuple[str, float | None]] = []
    for name in case.surf_variables:
        specs.append(
            ChannelSpec(
                index=len(specs),
                aurora_name=name,
                dataset_name=name,
                kind="surf",
                level=None,
                level_index=None,
                mean=float(locations[name]),
                std=float(scales[name]),
            )
        )
        sources.append((name, None))
    available = np.asarray(levels, dtype=np.float64)
    missing_levels = [
        level
        for level in case.pressure_levels
        if not bool(np.any(np.isclose(available, level, rtol=0.0, atol=1e-6)))
    ]
    if missing_levels:
        raise ValueError(
            f"case.pressure_levels contains unavailable levels {missing_levels}; "
            f"actual rollout levels={available.tolist()}"
        )
    for name in case.atmos_variables:
        for level in case.pressure_levels:
            index = int(np.flatnonzero(np.isclose(available, level, rtol=0.0, atol=1e-6))[0])
            key = f"{name}_{level_to_str(float(level))}"
            specs.append(
                ChannelSpec(
                    index=len(specs),
                    aurora_name=name,
                    dataset_name=name,
                    kind="atmos",
                    level=float(level),
                    level_index=index,
                    mean=float(locations[key]),
                    std=float(scales[key]),
                )
            )
            sources.append((name, float(level)))
    return specs, sources


def _parse_initialization_from_name(path: Path) -> np.datetime64 | None:
    for pattern in ("rollout_%Y%m%d_%H%M%S.nc", "rollout_%Y%m%d_%H%M.nc"):
        try:
            return np.datetime64(datetime.strptime(path.name, pattern), "ns")
        except ValueError:
            pass
    return None


def _manifest_enables_refinement(value: Any) -> bool:
    """Conservatively detect a refinement product in a nearby manifest."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"flow_refine_enabled", "conv_refine_enabled"} and item is True:
                return True
            if key_text == "refinement" and isinstance(item, Mapping):
                if item.get("enabled") is True or str(item.get("type", "none")).lower() not in {
                    "", "none", "off", "disabled",
                }:
                    return True
            if _manifest_enables_refinement(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_manifest_enables_refinement(item) for item in value)
    return False


def _validate_raw_source_directory(root: Path, case: BenchmarkCase) -> dict[str, Any]:
    if case.source_kind != "raw_aurora":
        raise ValueError(
            f"Benchmark case {case.name!r} declares source_kind={case.source_kind!r}; "
            "expected 'raw_aurora'. Refined forecasts cannot be relabelled as the baseline."
        )
    manifests: list[str] = []
    for name in ("run_manifest.json", "manifest.json"):
        path = root / name
        if not path.exists():
            continue
        manifests.append(str(path.resolve()))
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot validate rollout provenance from {path}: {exc}") from exc
        if _manifest_enables_refinement(payload):
            raise ValueError(
                f"Rollout source {root} is a refined-model product according to {path}; "
                "the raw_aurora baseline requires unrefined Aurora outputs."
            )
    return {
        "declared_source_kind": case.source_kind,
        "source_model_stage": case.source_model_stage,
        "baseline_label": f"raw_{case.source_model_stage}_aurora",
        "limitation": (
            "This is the repository's raw pretrained-Aurora rollout collection; "
            "a stage-1 fine-tuned, explicitly no-refinement rollout was not substituted."
        ),
        "validation": "per-file Aurora initialization_time/step_hours attributes; no member dimension",
        "manifests_checked": manifests,
    }


def _crop_rollout_to_truth_domain(roll, truth):
    """Crop a global raw rollout to the truth domain without changing its grid."""
    lat_min = float(np.nanmin(truth.latitude.values))
    lat_max = float(np.nanmax(truth.latitude.values))
    lon_min = float(np.nanmin(truth.longitude.values))
    lon_max = float(np.nanmax(truth.longitude.values))
    lat = roll.latitude.values
    lon = roll.longitude.values
    return roll.isel(
        latitude=np.flatnonzero((lat >= lat_min - 1e-6) & (lat <= lat_max + 1e-6)),
        longitude=np.flatnonzero((lon >= lon_min - 1e-6) & (lon <= lon_max + 1e-6)),
    )


def load_case(
    case: BenchmarkCase,
    *,
    max_initializations: int = 24,
    stride: int = 1,
    lat_stride: int = 1,
    lon_stride: int = 1,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> BenchmarkDataset:
    """Build packed pairs from provenance-validated *raw* Aurora rollouts."""
    import xarray as xr

    root = Path(case.rollout_dir)
    source_validation = _validate_raw_source_directory(root, case)
    all_files = sorted(root.glob(case.rollout_glob))
    if not all_files:
        raise FileNotFoundError(f"No {case.rollout_glob!r} files under {root}")
    if any(path.name.startswith("rollout_predictions_init_") for path in all_files):
        raise ValueError(
            f"Rollout source {root} contains fine-tuning outputs named "
            "'rollout_predictions_init_*'. Supply genuine unrefined Aurora rollouts."
        )
    if max_initializations < 1:
        raise ValueError(
            f"max_initializations must be >= 1, actual {max_initializations}."
        )
    if stride < 1 or lat_stride < 1 or lon_stride < 1:
        raise ValueError(
            "stride, lat_stride, and lon_stride must each be >= 1; actual "
            f"{stride}, {lat_stride}, {lon_stride}."
        )

    truth = xr.open_dataset(case.truth_path)
    available_truth = sorted(truth.data_vars)
    missing_truth = [
        name
        for name in (*case.surf_variables, *case.atmos_variables)
        if name not in truth.data_vars
    ]
    if missing_truth:
        truth.close()
        raise ValueError(
            f"truth_path={case.truth_path!r} is missing required variables {missing_truth}; "
            f"available={available_truth}"
        )
    truth_times = truth.time.values.astype("datetime64[ns]")
    truth_start, truth_end = truth_times.min(), truth_times.max()
    expected = np.asarray(case.expected_lead_hours, dtype=np.float64)
    if expected.size == 0 or np.any(expected <= 0) or np.any(np.diff(expected) <= 0):
        truth.close()
        raise ValueError(
            f"case.expected_lead_hours must be positive and strictly increasing; "
            f"actual {case.expected_lead_hours}."
        )
    candidate_files: list[Path] = []
    for path in all_files:
        init = _parse_initialization_from_name(path)
        if init is None:
            continue
        first = init + np.timedelta64(int(round(float(expected.min()) * 3600)), "s")
        last = init + np.timedelta64(int(round(float(expected.max()) * 3600)), "s")
        if first >= truth_start and last <= truth_end:
            candidate_files.append(path)
    files = candidate_files[:: max(1, int(stride))][: int(max_initializations)]
    if not files:
        truth.close()
        raise ValueError(
            f"No raw Aurora initialization under {root} has the complete expected lead range "
            f"{case.expected_lead_hours} inside truth time coverage [{truth_start}, {truth_end}]."
        )

    rollouts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    leads: list[float] = []
    lead_ids: list[int] = []
    initialization_ids: list[str] = []
    initialization_times: list[np.datetime64] = []
    valid_times: list[np.datetime64] = []
    lat_values: np.ndarray | None = None
    lon_values: np.ndarray | None = None
    specs: list[ChannelSpec] | None = None
    sources: list[tuple[str, float | None]] | None = None
    used_files: list[str] = []

    for path in files:
        roll = xr.open_dataset(path)
        if "member" in roll.dims:
            member_size = int(roll.sizes["member"])
            roll.close()
            truth.close()
            raise ValueError(
                f"Raw Aurora source {path} has member dimension size={member_size}; "
                "select an explicitly documented raw member upstream rather than silently "
                "using member=0."
            )
        if "initialization_time" not in roll.attrs or "step_hours" not in roll.attrs:
            actual_attrs = sorted(roll.attrs)
            roll.close()
            truth.close()
            raise ValueError(
                f"Raw Aurora source {path} must carry initialization_time and step_hours "
                f"attributes; actual attributes={actual_attrs}. This file's baseline "
                "provenance cannot be proven."
            )
        initialization = np.datetime64(str(roll.attrs["initialization_time"]), "ns")
        from_name = _parse_initialization_from_name(path)
        if from_name is None or initialization != from_name:
            roll.close()
            truth.close()
            raise ValueError(
                f"Raw Aurora source {path} initialization mismatch: filename={from_name}, "
                f"attribute={initialization}."
            )
        actual_step = float(roll.attrs["step_hours"])
        if not math.isclose(actual_step, case.lead_step_hours, rel_tol=0.0, abs_tol=1e-6):
            roll.close()
            truth.close()
            raise ValueError(
                f"Raw Aurora source {path} step_hours expected {case.lead_step_hours}, "
                f"actual {actual_step}."
            )
        roll = _crop_rollout_to_truth_domain(roll, truth)
        roll = roll.isel(
            latitude=slice(None, None, lat_stride), longitude=slice(None, None, lon_stride)
        )
        missing_roll = [
            name
            for name in (*case.surf_variables, *case.atmos_variables)
            if name not in roll.data_vars
        ]
        if missing_roll:
            available_roll = sorted(roll.data_vars)
            roll.close()
            truth.close()
            raise ValueError(
                f"Raw Aurora source {path} is missing required variables {missing_roll}; "
                f"available={available_roll}"
            )
        levels = [float(v) for v in roll.level.values] if "level" in roll.dims else []
        if specs is None:
            specs, sources = _channel_specs(case, levels)
            lat_values = roll.latitude.values.astype(np.float64)
            lon_values = roll.longitude.values.astype(np.float64)
        elif not np.array_equal(lat_values, roll.latitude.values) or not np.array_equal(
            lon_values, roll.longitude.values
        ):
            roll.close()
            truth.close()
            raise ValueError(f"Raw Aurora grid changed between source files; offending file={path}.")

        times = roll.time.values.astype("datetime64[ns]")
        lead_values = ((times - initialization) / np.timedelta64(1, "h")).astype(
            np.float64
        )
        if len(lead_values) != len(expected) or not np.allclose(
            lead_values, expected, rtol=0.0, atol=1e-6
        ):
            actual_leads = lead_values.tolist()
            roll.close()
            truth.close()
            raise ValueError(
                f"Raw Aurora source {path} expected lead hours {expected.tolist()}, "
                f"actual {actual_leads}."
            )
        if "lead_time" in roll.coords:
            coordinate_leads = np.asarray(roll.lead_time.values)
            if np.issubdtype(coordinate_leads.dtype, np.timedelta64):
                coordinate_leads = (
                    coordinate_leads / np.timedelta64(1, "h")
                ).astype(np.float64)
            else:
                coordinate_leads = coordinate_leads.astype(np.float64)
            if not np.allclose(coordinate_leads, lead_values, rtol=0.0, atol=1e-6):
                actual_coordinate = coordinate_leads.tolist()
                roll.close()
                truth.close()
                raise ValueError(
                    f"Raw Aurora source {path} lead_time coordinate is inconsistent with "
                    f"initialization_time and time: coordinate={actual_coordinate}, "
                    f"derived={lead_values.tolist()}."
                )
        try:
            truth_slice = truth.sel(time=times)
        except (KeyError, ValueError) as exc:
            roll.close()
            truth.close()
            raise ValueError(
                f"truth_path={case.truth_path!r} lacks exact valid times for {path}: "
                f"{[str(v) for v in times]}. Nearest-time matching is intentionally disabled."
            ) from exc
        truth_slice = truth_slice.reindex(
            latitude=roll.latitude,
            longitude=roll.longitude,
            method="nearest",
            tolerance=1e-4,
        )
        probe_name = case.surf_variables[0] if case.surf_variables else case.atmos_variables[0]
        if bool(truth_slice[probe_name].isnull().all().item()):
            roll.close()
            truth.close()
            raise ValueError(
                f"Could not align truth coordinates to raw Aurora grid for {path}; "
                "coordinate tolerance is 1e-4 degrees."
            )

        n_time = len(times)
        per_channel_roll: list[np.ndarray] = []
        per_channel_truth: list[np.ndarray] = []
        assert sources is not None
        for name, level in sources:
            if level is None:
                per_channel_roll.append(roll[name].values.astype(np.float32))
                per_channel_truth.append(truth_slice[name].values.astype(np.float32))
            else:
                if not bool(np.any(np.isclose(truth_slice.level.values, level))):
                    actual_levels = truth_slice.level.values.tolist()
                    roll.close()
                    truth.close()
                    raise ValueError(
                        f"truth_path={case.truth_path!r} lacks pressure level {level:g} hPa "
                        f"for variable {name!r}; actual levels={actual_levels}"
                    )
                per_channel_roll.append(roll[name].sel(level=level).values.astype(np.float32))
                per_channel_truth.append(
                    truth_slice[name].sel(level=level).values.astype(np.float32)
                )

        rollouts.append(np.stack(per_channel_roll, axis=1))  # [T, C, H, W]
        targets.append(np.stack(per_channel_truth, axis=1))
        leads.extend(lead_values.tolist())
        lead_ids.extend(list(range(n_time)))
        init_id = np.datetime_as_string(initialization, unit="s")
        initialization_ids.extend([init_id] * n_time)
        initialization_times.extend([initialization] * n_time)
        valid_times.extend(list(times))
        used_files.append(str(path.resolve()))
        roll.close()

        if verbose and len(rollouts) % 8 == 0:
            print(f"  loaded {len(rollouts)}/{len(files)} initializations", flush=True)

    assert specs is not None and lat_values is not None and lon_values is not None
    packing = FieldPacking(
        channels=tuple(specs),
        lat=tuple(lat_values.tolist()),
        lon=tuple(lon_values.tolist()),
        lon_periodic=case.lon_periodic,
    )
    space = NormalizedTargetSpace(packing)

    rollout_physical = torch.from_numpy(np.concatenate(rollouts, axis=0))
    target_physical = torch.from_numpy(np.concatenate(targets, axis=0))
    truth.close()

    rollout_norm = space.encode(rollout_physical)
    target_norm = space.encode(target_physical)
    valid = torch.isfinite(rollout_norm) & torch.isfinite(target_norm)
    rollout_norm = torch.nan_to_num(rollout_norm, nan=0.0, posinf=0.0, neginf=0.0)
    target_norm = torch.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return BenchmarkDataset(
        packing=packing,
        rollout=rollout_norm.to(device),
        target=target_norm.to(device),
        valid=valid.to(device),
        lead_hours=torch.tensor(leads, dtype=torch.float32, device=device),
        lead_index=torch.tensor(lead_ids, dtype=torch.long, device=device),
        lat=lat_values,
        lon=lon_values,
        case=case,
        initialization_ids=tuple(initialization_ids),
        initialization_times=np.asarray(initialization_times, dtype="datetime64[ns]"),
        valid_times=np.asarray(valid_times, dtype="datetime64[ns]"),
        source_files=tuple(used_files),
        source_validation=source_validation,
    )


def _iso_time(value: np.datetime64) -> str:
    return np.datetime_as_string(np.datetime64(value, "ns"), unit="s")


def grouped_purged_split(
    data: BenchmarkDataset,
    *,
    test_fraction: float = 0.25,
    purge_hours: float | None = None,
) -> tuple[BenchmarkDataset, BenchmarkDataset, dict[str, Any]]:
    """Chronological initialization-group split with a forecast-horizon purge.

    Every sample from one initialization stays in one arm.  Initializations
    within one maximum forecast horizon of the test boundary are discarded,
    and an explicit valid-time intersection check guards against unusual lead
    layouts.
    """
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1); actual {test_fraction}.")
    n = len(data)
    if (
        len(data.initialization_ids) != n
        or data.initialization_times.size != n
        or data.valid_times.size != n
    ):
        raise ValueError(
            "BenchmarkDataset must preserve initialization_ids, initialization_times, and "
            f"valid_times for all {n} samples; actual lengths are "
            f"{len(data.initialization_ids)}, {data.initialization_times.size}, "
            f"{data.valid_times.size}."
        )

    groups: dict[str, list[int]] = {}
    group_init: dict[str, np.datetime64] = {}
    for index, init_id in enumerate(data.initialization_ids):
        groups.setdefault(init_id, []).append(index)
        value = np.datetime64(data.initialization_times[index], "ns")
        previous = group_init.setdefault(init_id, value)
        if previous != value:
            raise ValueError(
                f"initialization_id={init_id!r} maps to multiple initialization times: "
                f"{previous} and {value}."
            )
    ordered = sorted(groups, key=lambda item: (group_init[item], item))
    if len(ordered) < 2:
        raise ValueError(
            f"Grouped benchmark split needs at least 2 initializations; actual {len(ordered)}."
        )
    n_test_groups = max(1, int(math.ceil(len(ordered) * float(test_fraction))))
    n_test_groups = min(n_test_groups, len(ordered) - 1)
    test_ids = ordered[-n_test_groups:]
    candidate_train_ids = ordered[:-n_test_groups]
    first_test_init = min(group_init[item] for item in test_ids)
    if purge_hours is None:
        purge_hours = float(data.lead_hours.max().detach().cpu())
    if not math.isfinite(float(purge_hours)) or float(purge_hours) < 0.0:
        raise ValueError(f"purge_hours must be finite and >= 0; actual {purge_hours}.")
    purge_delta = np.timedelta64(int(round(float(purge_hours) * 3600)), "s")
    cutoff = first_test_init - purge_delta
    train_ids = [item for item in candidate_train_ids if group_init[item] <= cutoff]
    purged_ids = [item for item in candidate_train_ids if item not in train_ids]

    test_indices = [index for item in test_ids for index in groups[item]]
    test_valid = {np.datetime64(data.valid_times[index], "ns") for index in test_indices}
    overlap_ids = [
        item
        for item in train_ids
        if any(
            np.datetime64(data.valid_times[index], "ns") in test_valid
            for index in groups[item]
        )
    ]
    if overlap_ids:
        train_ids = [item for item in train_ids if item not in overlap_ids]
        purged_ids.extend(overlap_ids)
    if not train_ids:
        raise ValueError(
            "Grouped/purged split left no training initializations. Increase "
            f"max_initializations or reduce test_fraction/purge_hours; groups={len(ordered)}, "
            f"test_groups={len(test_ids)}, purge_hours={purge_hours:g}."
        )
    train_indices = [index for item in train_ids for index in groups[item]]
    train_valid = {np.datetime64(data.valid_times[index], "ns") for index in train_indices}
    overlap = train_valid.intersection(test_valid)
    if overlap:  # Defensive invariant; overlap groups above should have removed it.
        raise RuntimeError(
            "Grouped/purged split invariant failed: training and test share valid times "
            f"{sorted(_iso_time(value) for value in overlap)}."
        )

    info = {
        "method": "chronological_initialization_groups_with_forecast_horizon_purge",
        "test_fraction": float(test_fraction),
        "purge_hours": float(purge_hours),
        "train_initialization_ids": train_ids,
        "test_initialization_ids": test_ids,
        "purged_initialization_ids": sorted(set(purged_ids)),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_valid_time_range": [
            _iso_time(min(train_valid)),
            _iso_time(max(train_valid)),
        ],
        "test_valid_time_range": [
            _iso_time(min(test_valid)),
            _iso_time(max(test_valid)),
        ],
        "shared_initialization_ids": [],
        "shared_valid_times": [],
    }
    return data.subset(train_indices), data.subset(test_indices), info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor, dims) -> torch.Tensor:
    return (values * weight).sum(dim=dims) / weight.sum(dim=dims).clamp(min=1e-12)


@dataclass
class MetricRow:
    label: str
    mae: float
    rmse: float
    bias: float
    correlation: float
    std_ratio: float
    p90_bias: float
    p95_bias: float
    p99_bias: float
    tail_mae: float

    def as_dict(self) -> dict[str, float | str | None]:
        def finite(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "label": self.label,
            "mae": finite(self.mae),
            "rmse": finite(self.rmse),
            "bias": finite(self.bias),
            "correlation": finite(self.correlation),
            "std_ratio": finite(self.std_ratio),
            "p90_bias": finite(self.p90_bias),
            "p95_bias": finite(self.p95_bias),
            "p99_bias": finite(self.p99_bias),
            "tail_mae": finite(self.tail_mae),
        }


def evaluate_fields(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    area_weight: torch.Tensor | None,
    label: str,
) -> MetricRow:
    """Physical-unit metrics with area-weighted means and spatial moments.

    ``prediction``/``truth`` are ``[S, C, H, W]`` physical fields.
    Quantile diagnostics pool valid grid cells without latitude weighting.
    """
    pred = prediction.double()
    obs = truth.double()
    mask = valid.double()
    if area_weight is not None:
        mask = mask * area_weight.to(device=mask.device, dtype=mask.dtype)

    error = pred - obs
    total = mask.sum().clamp(min=1.0)
    mae = float((error.abs() * mask).sum() / total)
    rmse = float(((error.pow(2) * mask).sum() / total).sqrt())
    bias = float((error * mask).sum() / total)

    dims = (-2, -1)
    pred_mean = _weighted_mean(pred, mask, dims)
    obs_mean = _weighted_mean(obs, mask, dims)
    pred_anom = pred - pred_mean[..., None, None]
    obs_anom = obs - obs_mean[..., None, None]
    # Apply the latitude/missing-data weight exactly once. Multiplying both
    # anomalies by it would silently square cosine-latitude weights.
    cov = (mask * pred_anom * obs_anom).sum(dim=dims)
    pred_var = (mask * pred_anom.pow(2)).sum(dim=dims)
    obs_var = (mask * obs_anom.pow(2)).sum(dim=dims)
    ok = (pred_var > 1e-24) & (obs_var > 1e-24)
    correlation = (
        float((cov[ok] / (pred_var[ok].sqrt() * obs_var[ok].sqrt())).mean()) if bool(ok.any()) else float("nan")
    )
    std_ratio = (
        float((pred_var[ok].sqrt() / obs_var[ok].sqrt()).mean()) if bool(ok.any()) else float("nan")
    )

    selected = mask > 0
    flat_pred = pred[selected]
    flat_obs = obs[selected]
    quantiles = torch.tensor([0.90, 0.95, 0.99], dtype=torch.float64, device=flat_pred.device)
    if flat_pred.numel() > 0:
        pq = torch.quantile(flat_pred.float(), quantiles.float()).double()
        oq = torch.quantile(flat_obs.float(), quantiles.float()).double()
        p90_bias, p95_bias, p99_bias = (float(v) for v in (pq - oq))
        threshold = oq[1]
        tail = flat_obs >= threshold
        tail_mae = (
            float((flat_pred[tail] - flat_obs[tail]).abs().mean()) if bool(tail.any()) else float("nan")
        )
    else:  # pragma: no cover - empty mask
        p90_bias = p95_bias = p99_bias = tail_mae = float("nan")

    return MetricRow(
        label=label,
        mae=mae,
        rmse=rmse,
        bias=bias,
        correlation=correlation,
        std_ratio=std_ratio,
        p90_bias=p90_bias,
        p95_bias=p95_bias,
        p99_bias=p99_bias,
        tail_mae=tail_mae,
    )


def _channel_label(spec: ChannelSpec) -> str:
    if spec.level is None:
        return spec.dataset_name
    return f"{spec.dataset_name}@{float(spec.level):g}hPa"


def evaluate_channels(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    packing: FieldPacking,
    *,
    area_weight: torch.Tensor | None,
    label: str,
) -> dict[str, MetricRow]:
    """Physical-unit metrics per variable/level; never aggregate mixed units."""
    if prediction.shape[1] != packing.num_channels:
        raise ValueError(
            f"prediction channel dimension expected {packing.num_channels}, "
            f"actual {prediction.shape[1]}."
        )
    rows: dict[str, MetricRow] = {}
    for spec in packing.channels:
        channel = _channel_label(spec)
        rows[channel] = evaluate_fields(
            prediction[:, spec.index : spec.index + 1],
            truth[:, spec.index : spec.index + 1],
            valid[:, spec.index : spec.index + 1],
            area_weight=area_weight,
            label=f"{label}:{channel}",
        )
    return rows


def evaluate_channels_by_lead(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    lead_hours: torch.Tensor,
    packing: FieldPacking,
    *,
    area_weight: torch.Tensor | None,
    label: str,
) -> dict[str, dict[float, MetricRow]]:
    out: dict[str, dict[float, MetricRow]] = {}
    for spec in packing.channels:
        channel = _channel_label(spec)
        out[channel] = evaluate_by_lead(
            prediction[:, spec.index : spec.index + 1],
            truth[:, spec.index : spec.index + 1],
            valid[:, spec.index : spec.index + 1],
            lead_hours,
            area_weight=area_weight,
            label=f"{label}:{channel}",
        )
    return out


def evaluate_by_lead(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    lead_hours: torch.Tensor,
    *,
    area_weight: torch.Tensor | None,
    label: str,
) -> dict[float, MetricRow]:
    """Per-forecast-lead metrics.

    A refinement head that improves 12 h while degrading 72 h is not an
    improvement, and an aggregate score hides exactly that.
    """
    out: dict[float, MetricRow] = {}
    for lead in sorted({float(v) for v in lead_hours.tolist()}):
        sel = lead_hours == lead
        if not bool(sel.any()):
            continue
        out[lead] = evaluate_fields(
            prediction[sel],
            truth[sel],
            valid[sel],
            area_weight=area_weight,
            label=f"{label}@{lead:g}h",
        )
    return out


def format_lead_table(
    per_lead: Mapping[str, Mapping[float, MetricRow]],
    baseline_key: str = "raw_pretrained_aurora",
) -> str:
    """Lead-time error curve for one variable/level, relative to its baseline."""
    baseline = per_lead[baseline_key]
    leads = sorted(baseline)
    header = f"{'head':<28}" + "".join(f"{f'{lead:g}h':>16}" for lead in leads)
    lines = [header, "-" * len(header), f"MAE change vs {baseline_key} (%)"]
    for name, rows in per_lead.items():
        if name == baseline_key:
            continue
        cells = []
        for lead in leads:
            base = baseline[lead].mae
            cells.append(f"{100.0 * (rows[lead].mae - base) / base:>16.2f}" if base else f"{'nan':>16}")
        lines.append(f"{name:<28}" + "".join(cells))
    lines.append(f"RMSE change vs {baseline_key} (%)")
    for name, rows in per_lead.items():
        if name == baseline_key:
            continue
        cells = []
        for lead in leads:
            base = baseline[lead].rmse
            cells.append(
                f"{100.0 * (rows[lead].rmse - base) / base:>16.2f}" if base else f"{'nan':>16}"
            )
        lines.append(f"{name:<28}" + "".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Head configuration presets
# ---------------------------------------------------------------------------


def _base_refinement_block(head: str, *, lon_periodic: bool) -> dict[str, Any]:
    """Recommended settings for a residual-correction refiner.

    These are the *improved* defaults: ``sample`` (x0) prediction so that a
    zero-initialised head is the exact identity and no prediction error is
    divided by ``sqrt(abar)``, Min-SNR-gamma weighting, deterministic inference,
    and a deterministic-anchor loss term that supervises the estimate the
    forecast actually uses.
    """
    block: dict[str, Any] = {
        "enabled": True,
        "type": head,
        "train_on_residual": True,
        "seed": 1234,
        "ensemble_size": 4,
        "deterministic_inference": True,
        "target_space": {
            "residual_scaling": "per_channel",
            "residual_scaling_center": True,
        },
        "conditioning": {
            "aurora_rollout": True,
            "aurora_input_state": False,
            "static_fields": False,
            "masks": True,
            "forecast_lead_time": True,
        },
        "loss": {
            "generative": "mse",
            # Candidate multi-objective weights for an executable comparison;
            # they are not production recommendations until validated over a
            # representative holdout period for the selected case.
            # LOSS_ABLATIONS provides isolated term-by-term experiments.
            "deterministic_weight": 1.0,
            "bias_weight": 1.0,
            "variance_weight": 2.0,
            "extreme_weight": 1.0,
            "quantile_weight": 1.0,
            "gradient_weight": 0.25,
            "pattern_correlation_weight": 0.5,
            "aux_on_deterministic": True,
        },
        "unet": {
            "hidden_channels": 32,
            "num_levels": 3,
            "num_residual_blocks": 2,
            "time_embedding_dim": 128,
            "bottleneck_attention": True,
            "zero_init_output": True,
        },
        "transformer": {
            "patch_size": [5, 5] if not lon_periodic else [6, 6],
            "embedding_dim": 192,
            "num_heads": 6,
            "num_blocks": 4,
            "mlp_ratio": 4.0,
            "positional_encoding": "sincos_2d",
            "attention_mode": "global_2d",
            "zero_init_output": True,
            "local_refinement": True,
        },
        "diffusion": {
            "training_timesteps": 1000,
            "inference_steps": 50,
            "prediction_type": "sample",
            "schedule": "cosine",
            "sampler": "ddim",
            "eta": 0.0,
            "snr_weighting": "auto",
            "snr_gamma": 5.0,
            "deterministic_estimator": "auto",
            "timestep_distribution": "auto",
            "timestep_bias": 3.0,
        },
        "flow_matching": {
            "interpolation_path": "existing_aurora",
            "integration_steps": 1,
            "time_sampling": "logit_normal",
        },
    }
    return block


#: Settings that reproduce the pre-audit behaviour, used as the "before" arm of
#: the benchmark: no residual standardization, epsilon-prediction, no SNR
#: weighting, no deterministic anchor and ensemble-mean inference.
LEGACY_PRESET: dict[str, Any] = {
    "deterministic_inference": False,
    "target_space": {"residual_scaling": "none"},
    "loss": {
        "generative": "mse",
        "deterministic_weight": 0.0,
        "bias_weight": 0.0,
        "variance_weight": 0.0,
        "extreme_weight": 0.0,
        "quantile_weight": 0.0,
        "gradient_weight": 0.0,
        "pattern_correlation_weight": 0.0,
    },
    "diffusion": {
        "prediction_type": "epsilon",
        "snr_weighting": "none",
        "deterministic_estimator": "ode",
        "timestep_distribution": "uniform",
    },
    "transformer": {"local_refinement": False},
}

PRESETS: dict[str, Mapping[str, Any]] = {"improved": {}, "legacy": LEGACY_PRESET}

# Keep this list explicit and synchronized with LossConfig._WEIGHT_KEYS.  An
# ablation is meaningful only when every term not named by the arm is zero.
AUXILIARY_LOSS_WEIGHT_KEYS: tuple[str, ...] = (
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

#: Loss-term ablation used to decide, experimentally rather than by assertion,
#: which auxiliary objectives actually help forecast skill.
LOSS_ABLATIONS: dict[str, Mapping[str, Any]] = {
    "generative_only": {"loss": {}},
    "deterministic": {"loss": {"deterministic_weight": 1.0}},
    "det+bias": {"loss": {"deterministic_weight": 1.0, "bias_weight": 1.0}},
    "det+mae": {"loss": {"deterministic_weight": 1.0, "mae_weight": 0.5}},
    "det+variance": {"loss": {"deterministic_weight": 1.0, "variance_weight": 5.0}},
    "det+extreme": {
        "loss": {
            "deterministic_weight": 1.0,
            "extreme_weight": 2.0,
            "extreme_quantile": 0.95,
            "extreme_intensity": 4.0,
        }
    },
    "det+quantile": {"loss": {"deterministic_weight": 1.0, "quantile_weight": 2.0}},
    "det+spectral": {"loss": {"deterministic_weight": 1.0, "spectral_weight": 0.5}},
    "det+gradient": {"loss": {"deterministic_weight": 1.0, "gradient_weight": 0.5}},
    "det+corr": {"loss": {"deterministic_weight": 1.0, "pattern_correlation_weight": 1.0}},
    "det+degradation": {"loss": {"deterministic_weight": 1.0, "degradation_weight": 0.5}},
    "balanced": {
        "loss": {
            "deterministic_weight": 1.0,
            "bias_weight": 1.0,
            "variance_weight": 2.0,
            "quantile_weight": 1.0,
            "gradient_weight": 0.25,
        }
    },
    "skill_plus_structure": {
        "loss": {
            "deterministic_weight": 1.0,
            "bias_weight": 1.0,
            "variance_weight": 2.0,
            "extreme_weight": 1.0,
            "quantile_weight": 1.0,
            "gradient_weight": 0.25,
            "pattern_correlation_weight": 0.5,
        }
    },
}


def isolated_ablation_overrides(name: str) -> dict[str, Any]:
    if name not in LOSS_ABLATIONS:
        raise ValueError(
            f"Unknown loss ablation {name!r}; expected one of {sorted(LOSS_ABLATIONS)}."
        )
    loss: dict[str, Any] = {
        "generative": "mse",
        "aux_on_deterministic": True,
        **{key: 0.0 for key in AUXILIARY_LOSS_WEIGHT_KEYS},
    }
    out: dict[str, Any] = {"loss": loss}
    _deep_update(out, LOSS_ABLATIONS[name])
    return out


def build_config(head: str, *, lon_periodic: bool, overrides: Mapping[str, Any] | None = None) -> RefinementConfig:
    block = _base_refinement_block(head, lon_periodic=lon_periodic)
    if overrides:
        _deep_update(block, overrides)
    if head.startswith("diffusion_"):
        block.pop("flow_matching", None)
    else:
        block.pop("diffusion", None)
    return resolve_refinement_config({"model": {"refinement": block}})


def _deep_update(base: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


# ---------------------------------------------------------------------------
# Train / evaluate one head
# ---------------------------------------------------------------------------


def _conditioning(rollout: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.all(dim=1, keepdim=True).to(rollout.dtype)
    return torch.cat([rollout, mask], dim=1)


@dataclass
class HeadResult:
    head: str
    channel_metrics: dict[str, MetricRow]
    train_seconds: float
    parameters: int
    final_loss: float
    per_lead: dict[str, dict[float, MetricRow]] = field(default_factory=dict)
    refined: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)


def _predict_refined(
    refiner,
    config: RefinementConfig,
    test: BenchmarkDataset,
    space: NormalizedTargetSpace,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    ensemble_size: int | None,
    direct_regression: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate with the configured deterministic path or persistent RNG streams."""
    members = int(ensemble_size if ensemble_size is not None else config.ensemble_size)
    if members < 1:
        raise ValueError(f"ensemble_size must be >= 1; actual {members}.")
    generators: list[torch.Generator] = []
    if not config.deterministic_inference:
        for member in range(members):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 1000 * member + 3)
            generators.append(generator)

    refined_chunks: list[torch.Tensor] = []
    eval_batch = max(1, int(batch_size))
    with torch.no_grad():
        for begin in range(0, len(test), eval_batch):
            end = min(begin + eval_batch, len(test))
            rollout = test.rollout[begin:end].to(device)
            valid = test.valid[begin:end].to(device)
            lead = test.lead_hours[begin:end].to(device)
            cond = _conditioning(rollout, valid)
            common: dict[str, Any] = {"forecast_lead_time": lead}
            if config.is_legacy_flow_matching:
                common["rollout_normalized"] = rollout
            if direct_regression:
                process_time = torch.zeros(
                    rollout.shape[0], device=device, dtype=torch.float32
                )
                predicted_scaled = refiner.net(
                    torch.zeros_like(rollout), cond, process_time, lead
                )
                residual = refiner.residual_scaler.decode(predicted_scaled)
            elif config.deterministic_inference:
                residual = refiner.deterministic_residual(cond, **common)
            else:
                accumulator = torch.zeros_like(rollout)
                for generator in generators:
                    residual_member = refiner.sample_residual(
                        cond, generator=generator, **common
                    )
                    accumulator = accumulator + residual_member
                residual = accumulator / members
            refined_chunks.append(space.decode(rollout + residual).cpu())
    return torch.cat(refined_chunks, dim=0), {
        "inference_mode": (
            "direct_regression"
            if direct_regression
            else (
                "deterministic_correction" if config.deterministic_inference else "ensemble_mean"
            )
        ),
        "configured_ensemble_size": int(config.ensemble_size),
        "evaluated_ensemble_size": 0 if config.deterministic_inference else members,
        "rng_streams": (
            []
            if config.deterministic_inference
            else [seed + 1000 * member + 3 for member in range(members)]
        ),
        "rng_stream_lifetime": "whole_test_evaluation",
    }


def _fit_training_correction_scaler(refiner: Any, train: BenchmarkDataset) -> None:
    """Fit fixed correction statistics from every grouped training sample.

    Calibration is intentionally performed before shuffling and uses only the
    already selected training arm. Keeping the source tensors on their current
    device avoids copying an entire global field collection to the accelerator;
    ResidualScaler transfers only its tiny sufficient statistics to its buffers.
    """
    fit = getattr(refiner, "fit_residual_scale", None)
    if not callable(fit):
        return
    with torch.no_grad():
        correction = torch.where(
            train.valid,
            train.target - train.rollout,
            torch.zeros_like(train.target),
        )
        fit(correction, train.valid)
        refiner.freeze_residual_scale()


def train_and_evaluate(
    head: str,
    train: BenchmarkDataset,
    test: BenchmarkDataset,
    *,
    epochs: int = 8,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    device: torch.device | str = "cuda",
    overrides: Mapping[str, Any] | None = None,
    seed: int = 0,
    ensemble_size: int | None = None,
    verbose: bool = True,
) -> HeadResult:
    device = torch.device(device)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1; actual {epochs}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; actual {batch_size}.")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be finite and > 0; actual {learning_rate}."
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    direct_regression = head == "direct_regression_transformer"
    configured_head = "diffusion_transformer" if direct_regression else head
    config = build_config(
        configured_head, lon_periodic=train.case.lon_periodic, overrides=overrides
    )
    packing = train.packing
    cond_channels = packing.num_channels + 1

    refiner = build_refiner(
        config,
        residual_channels=packing.num_channels,
        cond_channels=cond_channels,
        metadata=packing,
    )
    assert refiner is not None
    refiner = refiner.to(device)
    parameters = sum(p.numel() for p in refiner.parameters() if p.requires_grad)

    space = NormalizedTargetSpace(packing)
    area = area_weights_from_latitudes(packing.lat, len(packing.lat), device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=learning_rate, weight_decay=0.0)
    n = len(train)
    steps_per_epoch = max(1, math.ceil(n / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps_per_epoch)
    )

    process_generator = torch.Generator(device=device)
    process_generator.manual_seed(seed + 17)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed + 29)
    # Fixed training-only statistics must include every grouped training case.
    _fit_training_correction_scaler(refiner, train)


    start = time.time()
    final_loss = float("nan")
    refiner.train()
    for epoch in range(epochs):
        order = torch.randperm(n, generator=order_generator)
        running = 0.0
        count = 0
        for step in range(steps_per_epoch):
            idx = order[step * batch_size : (step + 1) * batch_size]
            if idx.numel() == 0:
                continue
            rollout = train.rollout[idx].to(device)
            target = train.target[idx].to(device)
            valid = train.valid[idx].to(device)
            lead = train.lead_hours[idx].to(device)
            lead_id = train.lead_index[idx].to(device)
            cond = _conditioning(rollout, valid)
            residual = torch.where(valid, target - rollout, torch.zeros_like(target))

            if direct_regression:
                scaled_target = refiner.residual_scaler.encode(residual)
                process_time = torch.zeros(
                    rollout.shape[0], device=device, dtype=torch.float32
                )
                predicted_scaled = refiner.net(
                    torch.zeros_like(scaled_target), cond, process_time, lead
                )
                predicted_correction = refiner.residual_scaler.decode(predicted_scaled)
                loss = masked_loss(
                    predicted_correction, residual, valid, "mse"
                )
            else:
                out = refiner.compute_training_loss(
                    residual,
                    cond,
                    forecast_lead_time=lead,
                    mask=valid,
                    generator=process_generator,
                    lead_index=lead_id,
                    area_weight=area,
                    rollout_normalized=rollout,
                )
                loss = out.total_loss
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            count += 1
        final_loss = running / max(1, count)
        if not math.isfinite(final_loss):
            raise RuntimeError(
                f"[{head}] training produced non-finite loss at epoch {epoch + 1}: "
                f"{final_loss}."
            )
        if verbose:
            print(f"    [{head}] epoch {epoch + 1}/{epochs} loss={final_loss:.6f}", flush=True)
    train_seconds = time.time() - start

    # -- evaluation ---------------------------------------------------
    refiner.eval()
    refined, inference_diagnostics = _predict_refined(
        refiner,
        config,
        test,
        space,
        batch_size=batch_size,
        device=device,
        seed=seed,
        ensemble_size=ensemble_size,
        direct_regression=direct_regression,
    )

    truth_physical = space.decode(test.target)
    area_cpu = area_weights_from_latitudes(packing.lat, len(packing.lat), dtype=torch.float64)
    channel_metrics = evaluate_channels(
        refined,
        truth_physical,
        test.valid,
        packing,
        area_weight=area_cpu,
        label=head,
    )
    per_lead = evaluate_channels_by_lead(
        refined,
        truth_physical,
        test.valid,
        test.lead_hours.cpu(),
        packing,
        area_weight=area_cpu,
        label=head,
    )
    return HeadResult(
        head=head,
        channel_metrics=channel_metrics,
        train_seconds=train_seconds,
        parameters=parameters,
        final_loss=final_loss,
        per_lead=per_lead,
        refined=refined,
        diagnostics=inference_diagnostics,
        resolved_config=config.to_dict(),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DEFAULT_HEADS = (
    "diffusion_unet",
    "diffusion_transformer",
    "flow_matching_conv_unet",
    "direct_regression_transformer",
    "flow_matching_transformer",
)


def correction_statistics(data: BenchmarkDataset) -> list[dict[str, Any]]:
    """Fixed training-split CAMS-minus-Aurora statistics by packed channel."""
    correction_normalized = (data.target - data.rollout).float()
    rows: list[dict[str, Any]] = []
    for spec in data.packing.channels:
        valid = data.valid[:, spec.index]
        values = correction_normalized[:, spec.index][valid].double()
        if values.numel() == 0:
            raise ValueError(
                f"Training correction channel {spec.key!r} has no valid cells."
            )
        physical = values * float(spec.std)
        rows.append(
            {
                "channel_index": int(spec.index),
                "variable": spec.dataset_name,
                "level_hpa": None if spec.level is None else float(spec.level),
                "count": int(physical.numel()),
                "mean": float(physical.mean()),
                "std": float(physical.std(unbiased=False)),
                "q01": float(torch.quantile(physical, 0.01)),
                "q50": float(torch.quantile(physical, 0.50)),
                "q99": float(torch.quantile(physical, 0.99)),
                "normalization_space": "physical_cams_minus_aurora_correction",
            }
        )
    return rows


def format_correction_statistics(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        f"{'channel':<20}{'mean':>14}{'std':>14}"
        f"{'q01':>14}{'median':>14}{'q99':>14}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        level = "surface" if row["level_hpa"] is None else f"{row['level_hpa']:g}hPa"
        name = f"{row['variable']}@{level}"
        lines.append(
            f"{name:<20}{row['mean']:>14.5e}{row['std']:>14.5e}"
            f"{row['q01']:>14.5e}{row['q50']:>14.5e}{row['q99']:>14.5e}"
        )
    return "\n".join(lines)


def _file_identity(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_provenance(data: BenchmarkDataset) -> dict[str, Any]:
    files = [_file_identity(path) for path in data.source_files]
    truth = _file_identity(data.case.truth_path)
    identity_payload = json.dumps(
        {"rollouts": files, "truth": truth}, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        **dict(data.source_validation),
        "rollout_directory": str(Path(data.case.rollout_dir).resolve()),
        "rollout_glob": data.case.rollout_glob,
        "rollout_files": files,
        "truth_file": truth,
        "file_identity_method": "absolute_path+size_bytes+mtime_ns",
        "source_collection_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "variables_and_levels": [
            {
                "channel_index": int(spec.index),
                "variable": spec.dataset_name,
                "kind": spec.kind,
                "pressure_level_hpa": (
                    None if spec.level is None else float(spec.level)
                ),
            }
            for spec in data.packing.channels
        ],
        "latitude_count": len(data.packing.lat),
        "longitude_count": len(data.packing.lon),
        "lead_hours": sorted({float(value) for value in data.lead_hours.tolist()}),
    }


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain=v1")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "repository_root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "changed_paths": [] if not status else status.splitlines(),
        "benchmark_source_sha256": source_hash,
    }


def _environment_provenance(device: str) -> dict[str, Any]:
    try:
        import xarray as xr

        xarray_version: str | None = xr.__version__
    except ImportError:  # pragma: no cover - load_case already requires xarray
        xarray_version = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "xarray_version": xarray_version,
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "selected_environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
        },
    }


def _metric_mapping(rows: Mapping[str, MetricRow]) -> dict[str, Any]:
    return {name: row.as_dict() for name, row in rows.items()}


def _lead_metric_mapping(
    rows: Mapping[str, Mapping[float, MetricRow]],
) -> dict[str, Any]:
    return {
        channel: {str(lead): row.as_dict() for lead, row in lead_rows.items()}
        for channel, lead_rows in rows.items()
    }


def _relabel(row: MetricRow, label: str) -> MetricRow:
    return MetricRow(
        label=label,
        mae=row.mae,
        rmse=row.rmse,
        bias=row.bias,
        correlation=row.correlation,
        std_ratio=row.std_ratio,
        p90_bias=row.p90_bias,
        p95_bias=row.p95_bias,
        p99_bias=row.p99_bias,
        tail_mae=row.tail_mae,
    )


def format_channel_tables(
    baseline: Mapping[str, MetricRow],
    comparisons: Mapping[str, Mapping[str, MetricRow]],
    *,
    baseline_label: str,
) -> str:
    sections: list[str] = []
    for channel, base in baseline.items():
        rows = [_relabel(base, baseline_label)]
        rows.extend(
            _relabel(metrics[channel], name)
            for name, metrics in comparisons.items()
        )
        sections.append(f"channel: {channel}\n{format_table(rows, rows[0])}")
    return "\n\n".join(sections)


def _ensure_raw_dataset_provenance(data: BenchmarkDataset) -> None:
    declared = data.source_validation.get("declared_source_kind")
    if data.case.source_kind != "raw_aurora" or declared != "raw_aurora":
        raise ValueError(
            "BenchmarkDataset baseline provenance is not validated as raw Aurora: "
            f"case.source_kind={data.case.source_kind!r}, "
            f"source_validation.declared_source_kind={declared!r}."
        )


def run_loss_ablation(
    case_name: str,
    head: str,
    data: BenchmarkDataset,
    *,
    epochs: int = 8,
    batch_size: int = 8,
    device: str = "cuda",
    test_fraction: float = 0.25,
    purge_hours: float | None = None,
    learning_rate: float = 3e-4,
    ensemble_size: int | None = None,
    seed: int = 0,
    max_initializations: int | None = None,
    lat_stride: int | None = None,
    lon_stride: int | None = None,
    output: str | None = None,
    ablations: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one head under several loss configurations and score them all."""
    if data.case.name != case_name:
        raise ValueError(
            f"BenchmarkDataset case expected {case_name!r}, actual {data.case.name!r}."
        )
    _ensure_raw_dataset_provenance(data)
    train, test, split = grouped_purged_split(
        data, test_fraction=test_fraction, purge_hours=purge_hours
    )

    space = NormalizedTargetSpace(data.packing)
    area_cpu = area_weights_from_latitudes(
        data.packing.lat, len(data.packing.lat), dtype=torch.float64
    )
    baseline_label = str(
        data.source_validation.get("baseline_label", "raw_pretrained_aurora")
    )
    baseline = evaluate_channels(
        space.decode(test.rollout),
        space.decode(test.target),
        test.valid,
        data.packing,
        area_weight=area_cpu,
        label=baseline_label,
    )
    baseline_per_lead = evaluate_channels_by_lead(
        space.decode(test.rollout),
        space.decode(test.target),
        test.valid,
        test.lead_hours.cpu(),
        data.packing,
        area_weight=area_cpu,
        label=baseline_label,
    )
    selected_ablations = list(ablations or LOSS_ABLATIONS)
    comparison_rows: dict[str, dict[str, MetricRow]] = {}
    results: dict[str, Any] = {
        "case": case_name,
        "head": head,
        "metric_scope": "physical_units_per_variable_and_pressure_level_only",
        "baseline": {
            "label": baseline_label,
            "channels": _metric_mapping(baseline),
            "per_lead": _lead_metric_mapping(baseline_per_lead),
        },
        "ablations": {},
        "provenance": {
            "args": {
                "case": case_name,
                "head": head,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "device": str(device),
                "test_fraction": float(test_fraction),
                "purge_hours": purge_hours,
                "ensemble_size": ensemble_size,
                "max_initializations": max_initializations,
                "lat_stride": lat_stride,
                "lon_stride": lon_stride,
                "output": output,
                "ablations": selected_ablations,
            },
            "seed": int(seed),
            "source": _source_provenance(data),
            "split": split,
            "resolved_config": {},
            "git": _git_provenance(),
            "environment": _environment_provenance(device),
        },
    }
    for name in selected_ablations:
        if verbose:
            print(f"  [{head}] ablation {name} ...", flush=True)
        arm_overrides = isolated_ablation_overrides(name)
        result = train_and_evaluate(
            head,
            train,
            test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            overrides=arm_overrides,
            seed=seed,
            ensemble_size=ensemble_size,
            verbose=False,
        )
        comparison_rows[name] = result.channel_metrics
        results["ablations"][name] = {
            "channels": _metric_mapping(result.channel_metrics),
            "per_lead": _lead_metric_mapping(result.per_lead),
            "train_seconds": result.train_seconds,
            "parameters": result.parameters,
            "final_loss": result.final_loss,
            "diagnostics": result.diagnostics,
            "isolated_overrides": arm_overrides,
        }
        results["provenance"]["resolved_config"][name] = result.resolved_config
    table = format_channel_tables(
        baseline, comparison_rows, baseline_label=baseline_label
    )
    results["table"] = table
    if verbose:
        print(table)
    if output:
        Path(output).write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def run_benchmark(
    case_name: str = "o3_global",
    *,
    heads: Sequence[str] = DEFAULT_HEADS,
    epochs: int = 8,
    max_initializations: int = 24,
    lat_stride: int = 1,
    lon_stride: int = 1,
    batch_size: int = 4,
    device: str = "cuda",
    test_fraction: float = 0.25,
    purge_hours: float | None = None,
    learning_rate: float = 3e-4,
    ensemble_size: int | None = None,
    seed: int = 0,
    preset: str = "improved",
    overrides: Mapping[str, Any] | None = None,
    output: str | None = None,
    artifact_dir: str | None = None,
    data: BenchmarkDataset | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if case_name not in CASES:
        raise ValueError(f"Unknown case {case_name!r}; expected one of {sorted(CASES)}.")
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(PRESETS)}.")
    case = CASES[case_name]
    if data is None:
        if verbose:
            print(f"Loading case {case_name} ...", flush=True)
        data = load_case(
            case,
            max_initializations=max_initializations,
            lat_stride=lat_stride,
            lon_stride=lon_stride,
            verbose=verbose,
        )
    if data.case.name != case_name:
        raise ValueError(
            f"Injected BenchmarkDataset case expected {case_name!r}, actual {data.case.name!r}."
        )
    _ensure_raw_dataset_provenance(data)
    combined: dict[str, Any] = {}
    _deep_update(combined, PRESETS[preset])
    if overrides:
        _deep_update(combined, overrides)

    train, test, split = grouped_purged_split(
        data, test_fraction=test_fraction, purge_hours=purge_hours
    )
    if verbose:
        print(f"  samples: train={len(train)} test={len(test)} channels={data.packing.num_channels}")
    training_correction_stats = correction_statistics(train)
    if verbose:
        print("\nTraining correction statistics (CAMS - Aurora):\n" + format_correction_statistics(training_correction_stats))


    space = NormalizedTargetSpace(data.packing)
    area_cpu = area_weights_from_latitudes(
        data.packing.lat, len(data.packing.lat), dtype=torch.float64
    )
    baseline_label = str(
        data.source_validation.get("baseline_label", "raw_pretrained_aurora")
    )
    baseline = evaluate_channels(
        space.decode(test.rollout),
        space.decode(test.target),
        test.valid,
        data.packing,
        area_weight=area_cpu,
        label=baseline_label,
    )
    baseline_per_lead = evaluate_channels_by_lead(
        space.decode(test.rollout),
        space.decode(test.target),
        test.valid,
        test.lead_hours.cpu(),
        data.packing,
        area_weight=area_cpu,
        label=baseline_label,
    )
    comparison_rows: dict[str, dict[str, MetricRow]] = {}
    results: dict[str, Any] = {
        "case": case_name,
        "preset": preset,
        "metric_scope": "physical_units_per_variable_and_pressure_level_only",
        "samples": {"train": len(train), "test": len(test)},
        "training_correction_statistics": training_correction_stats,
        "baseline": {
            "label": baseline_label,
            "channels": _metric_mapping(baseline),
            "per_lead": _lead_metric_mapping(baseline_per_lead),
        },
        "heads": {},
        "provenance": {
            "args": {
                "case": case_name,
                "heads": list(heads),
                "epochs": int(epochs),
                "max_initializations": int(max_initializations),
                "lat_stride": int(lat_stride),
                "lon_stride": int(lon_stride),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "device": str(device),
                "test_fraction": float(test_fraction),
                "purge_hours": purge_hours,
                "ensemble_size": ensemble_size,
                "preset": preset,
                "overrides": dict(overrides or {}),
                "output": output,
                "artifact_dir": artifact_dir,
            },
            "seed": int(seed),
            "source": _source_provenance(data),
            "split": split,
            "resolved_config": {},
            "git": _git_provenance(),
            "environment": _environment_provenance(device),
        },
    }
    for head in heads:
        if verbose:
            print(f"  training head {head} [{preset}] ...", flush=True)
        result = train_and_evaluate(
            head,
            train,
            test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            overrides=combined,
            seed=seed,
            ensemble_size=ensemble_size,
            verbose=verbose,
        )
        comparison_rows[head] = result.channel_metrics
        artifacts = None
        if artifact_dir is not None:
            if result.refined is None:
                raise RuntimeError(
                    f"Head {head!r} did not return held-out physical fields for plotting."
                )
            from finetune.run_refinement_diagnostics import (
                save_heldout_diagnostic_artifacts,
            )

            artifacts = save_heldout_diagnostic_artifacts(
                Path(artifact_dir) / preset / head,
                data=test,
                head=head,
                refined_physical=result.refined,
            )
        results["heads"][head] = {
            "channels": _metric_mapping(result.channel_metrics),
            "train_seconds": result.train_seconds,
            "parameters": result.parameters,
            "final_loss": result.final_loss,
            "per_lead": _lead_metric_mapping(result.per_lead),
            "diagnostics": result.diagnostics,
        }
        if artifacts is not None:
            results["heads"][head]["artifacts"] = artifacts
        results["provenance"]["resolved_config"][head] = result.resolved_config

    table = format_channel_tables(
        baseline, comparison_rows, baseline_label=baseline_label
    )
    results["table"] = table
    if verbose:
        print(table)
    if output:
        Path(output).write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def format_table(rows: Iterable[MetricRow], baseline: MetricRow) -> str:
    header = (
        f"{'head':<28}{'MAE':>12}{'d%':>11}{'RMSE':>12}{'d%':>11}"
        f"{'bias':>12}{'corr':>8}{'sd_rat':>8}{'P99bias':>12}{'tailMAE':>12}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        d_mae = 100.0 * (row.mae - baseline.mae) / baseline.mae if baseline.mae else float("nan")
        d_rmse = (
            100.0 * (row.rmse - baseline.rmse) / baseline.rmse if baseline.rmse else float("nan")
        )
        lines.append(
            f"{row.label:<28}{row.mae:>12.5g}{d_mae:>11.3g}{row.rmse:>12.5g}{d_rmse:>11.3g}"
            f"{row.bias:>12.4g}{row.correlation:>8.4f}{row.std_ratio:>8.4f}"
            f"{row.p99_bias:>12.4g}{row.tail_mae:>12.5g}"
        )
    lines.append("")
    lines.append(
        f"d% is relative to {baseline.label}; negative is better for MAE/RMSE."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="o3_global", choices=sorted(CASES))
    parser.add_argument("--heads", nargs="*", default=list(DEFAULT_HEADS))
    parser.add_argument("--preset", default="improved", choices=sorted(PRESETS))
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the legacy and improved presets back to back on identical data.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-initializations", type=int, default=24)
    parser.add_argument("--lat-stride", type=int, default=1)
    parser.add_argument("--lon-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--purge-hours",
        type=float,
        default=None,
        help="Initialization-boundary purge; default is the maximum forecast lead.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory for held-out physical maps and distribution diagnostics.",
    )
    parser.add_argument(
        "--loss-ablation",
        metavar="HEAD",
        default=None,
        help="Run the loss-term ablation for one head instead of the head comparison.",
    )
    args = parser.parse_args(argv)

    case = CASES[args.case]
    print(f"Loading case {args.case} ...", flush=True)
    data = load_case(
        case,
        max_initializations=args.max_initializations,
        lat_stride=args.lat_stride,
        lon_stride=args.lon_stride,
    )

    if args.loss_ablation:
        collected = run_loss_ablation(
            args.case,
            args.loss_ablation,
            data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            test_fraction=args.test_fraction,
            purge_hours=args.purge_hours,
            ensemble_size=args.ensemble_size,
            seed=args.seed,
            max_initializations=args.max_initializations,
            lat_stride=args.lat_stride,
            lon_stride=args.lon_stride,
            output=args.output,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(collected, indent=2, allow_nan=False))
        return 0

    presets = ["legacy", "improved"] if args.compare else [args.preset]
    collected: dict[str, Any] = {}
    for preset in presets:
        print(f"\n=== preset: {preset} ===", flush=True)
        collected[preset] = run_benchmark(
            args.case,
            heads=args.heads,
            epochs=args.epochs,
            max_initializations=args.max_initializations,
            lat_stride=args.lat_stride,
            lon_stride=args.lon_stride,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            test_fraction=args.test_fraction,
            purge_hours=args.purge_hours,
            ensemble_size=args.ensemble_size,
            seed=args.seed,
            preset=preset,
            data=data,
            artifact_dir=args.artifact_dir,
        )
        collected[preset]["provenance"]["args"]["output"] = args.output
    if args.output:
        Path(args.output).write_text(json.dumps(collected, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
