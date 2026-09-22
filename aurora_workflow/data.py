"""Bounded ADS acquisition and cycle-preserving CAMS preparation.

No credentials or scientific libraries are imported by planning/metadata calls.
Execution is called only by the approved workflow worker, never by the planner.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from contextlib import ExitStack
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PRODUCT = "cams-global-atmospheric-composition-forecasts"
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
FORM_SHA256 = "ff578604ee4b3615add68329e0630fe5c463621ce4904fef6ba5fe0d533694e5"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEADS = [12, 24, 36, 48, 60, 72]
# ADS spelling, NetCDF spelling, Aurora spelling, physical units, kind.
VARIABLES = [
    ("2m_temperature", "t2m", "2t", "K", "surf"),
    ("10m_u_component_of_wind", "u10", "10u", "m s-1", "surf"),
    ("10m_v_component_of_wind", "v10", "10v", "m s-1", "surf"),
    ("mean_sea_level_pressure", "msl", "msl", "Pa", "surf"),
    ("particulate_matter_1um", "pm1", "pm1", "kg m-3", "surf"),
    ("particulate_matter_2.5um", "pm2p5", "pm2p5", "kg m-3", "surf"),
    ("particulate_matter_10um", "pm10", "pm10", "kg m-3", "surf"),
    ("total_column_carbon_monoxide", "tcco", "tcco", "kg m-2", "surf"),
    ("total_column_nitrogen_monoxide", "tc_no", "tc_no", "kg m-2", "surf"),
    ("total_column_nitrogen_dioxide", "tcno2", "tcno2", "kg m-2", "surf"),
    ("total_column_ozone", "gtco3", "gtco3", "kg m-2", "surf"),
    ("total_column_sulphur_dioxide", "tcso2", "tcso2", "kg m-2", "surf"),
    ("geopotential", "z", "z", "m2 s-2", "atmos"),
    ("u_component_of_wind", "u", "u", "m s-1", "atmos"),
    ("v_component_of_wind", "v", "v", "m s-1", "atmos"),
    ("temperature", "t", "t", "K", "atmos"),
    ("specific_humidity", "q", "q", "kg kg-1", "atmos"),
    ("carbon_monoxide", "co", "co", "kg kg-1", "atmos"),
    ("nitrogen_monoxide", "no", "no", "kg kg-1", "atmos"),
    ("nitrogen_dioxide", "no2", "no2", "kg kg-1", "atmos"),
    ("ozone", "go3", "go3", "kg kg-1", "atmos"),
    ("sulphur_dioxide", "so2", "so2", "kg kg-1", "atmos"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo:
        if parsed.utcoffset() != timedelta(0):
            raise ValueError("Times must be UTC.")
        parsed = parsed.replace(tzinfo=None)
    return parsed


def resolve_data_spec(spec: dict, mode: str = "replay") -> dict:
    """Resolve recipe date metadata into explicit executable windows, without inventing historical validation."""
    spec = dict(spec)
    requested = spec.get("requested_dates") or {}
    if "start_date" not in spec:
        if requested:
            spec["start_date"], spec["end_date"] = requested["start"], requested["end"]
        elif spec.get("cycles"):
            spec["start_date"], spec["end_date"] = min(spec["cycles"]), max(spec["cycles"])
        elif spec.get("historical_test"):
            spec["start_date"], spec["end_date"] = spec["historical_test"]
            if mode in ("reproduce", "resume", "warm-start") and spec.get("historical_train"):
                spec["start_date"] = spec["historical_train"][0]
        else:
            raise ValueError("Specify start_date/end_date, requested_dates, or explicit cycles.")
    if not spec.get("splits"):
        splits = {}
        if mode in ("reproduce", "resume", "warm-start") and spec.get("historical_train"):
            first, last = map(_datetime, spec["historical_train"])
            validation = spec.get("validation", {})
            if validation.get("method") == "train_tail":
                fraction = float(validation.get("fraction", .1))
                if not 0 < fraction < .5:
                    raise ValueError("train_tail validation fraction must be in (0,.5).")
                steps = int((last - first).total_seconds() // 43200) + 1
                val_start = first + timedelta(hours=12 * (steps - max(1, int(steps * fraction))))
                splits["train"] = [first.isoformat(), (val_start - timedelta(hours=12)).isoformat()]
                splits["val"] = [val_start.isoformat(), last.isoformat()]
                spec["validation_derivation"] = "New deterministic chronological train-tail split; historical exact dates unavailable; purge full case support."
            else:
                splits["train"] = [first.isoformat(), last.isoformat()]
        if spec.get("historical_test"):
            splits["test"] = spec["historical_test"]
        if not splits:
            # Explicit requested integration dates alone cannot claim a training/validation split.
            splits["test"] = [(_datetime(spec["start_date"]) - timedelta(hours=12)).isoformat(),
                              (_datetime(spec["end_date"]) + timedelta(hours=72)).isoformat()]
        spec["splits"] = splits
    return spec


def plan_cams(spec: dict, limits: dict | None = None) -> dict:
    """Build exact requests including the preceding 12-hour initialization.

    Storage estimates are uncompressed float32 values plus a 25% allowance;
    they are estimates, while byte/request limits are execution constraints.
    """
    spec = resolve_data_spec(spec, spec.get("mode", "replay"))
    limits = limits or spec.get("limits", {})
    if spec.get("product", PRODUCT) != PRODUCT:
        raise ValueError("Only the named CAMS operational product is supported.")
    if spec.get("cadence_hours", 12) != 12 or spec.get("history_hours", [-12, 0]) != [-12, 0]:
        raise ValueError("Backbone cadence/history must be 12 hours and [-12,0].")
    start = _datetime(spec["start_date"])
    end = _datetime(spec["end_date"])
    if len(str(spec["end_date"])) == 10:
        end += timedelta(hours=12)
    if start > end or start.hour not in (0, 12) or end.hour not in (0, 12):
        raise ValueError("CAMS requires ordered dates and 00/12 UTC cycles.")
    if any((v.minute or v.second or v.microsecond) for v in (start, end)):
        raise ValueError("CAMS cycles must be exact 00/12 UTC.")
    history_start = start - timedelta(hours=12)
    if history_start.date() < date(2015, 1, 1) or end.date() > datetime.now(timezone.utc).date():
        raise ValueError("Operational CAMS archive starts 2015-01-01; include available history.")
    kind = spec.get("reference_kind", "forecast")
    if kind not in ("forecast", "analysis"):
        raise ValueError("reference_kind must be forecast or analysis; reanalysis is a different product.")
    leads = spec.get("lead_hours", LEADS)
    if not leads or any(type(v) is not int or v < 12 or v > 72 or v % 12 for v in leads):
        raise ValueError("Supported lead_hours are unique multiples of 12 through 72.")
    if leads != sorted(set(leads)):
        raise ValueError("lead_hours must be sorted and unique.")
    if spec.get("pressure_levels", LEVELS) != LEVELS:
        raise ValueError("All thirteen backbone pressure levels are required; target levels are separate.")
    if spec.get("variables", [v[0] for v in VARIABLES]) != [v[0] for v in VARIABLES]:
        raise ValueError("Acquisition must include every pretrained backbone input variable.")
    # Global backbone history is necessary even when the refinement domain is regional.
    area = spec.get("area", [90, -180, -90, 180])
    if area != [90, -180, -90, 180]:
        raise ValueError("Canonical backbone acquisition must be global; crop refinement after rollout.")
    chunk_days = spec.get("chunk_days", 1)
    if type(chunk_days) is not int or not 1 <= chunk_days <= 15:
        raise ValueError("chunk_days must be an integer in [1,15].")
    requests = []
    cycles = []
    cursor = start
    while cursor <= end:
        cycles.append(cursor.isoformat())
        cursor += timedelta(hours=12)
    if spec.get("cycles"):
        cycles = [_datetime(v).isoformat() for v in spec["cycles"]]
        if len(cycles) != len(set(cycles)) or any(_datetime(v).hour not in (0,12) or
                _datetime(v).minute or _datetime(v).second or not start <= _datetime(v) <= end for v in cycles):
            raise ValueError("Explicit cycles must be unique 00/12 UTC initializations.")
    last_history = end + timedelta(hours=max(leads) if kind == "analysis" else 0)
    # Separate lead0 history from forecasts to avoid multiplying expensive global inputs.
    for purpose, first, last, selected_leads in (
        ("initialization", history_start.date(), last_history.date(), [0]),
        ("reference", start.date(), end.date(), leads if kind == "forecast" else []),
    ):
        if not selected_leads:
            continue
        cursor_date = first
        while cursor_date <= last:
            chunk_end = min(cursor_date + timedelta(days=chunk_days - 1), last)
            request = {"type": "forecast", "date": f"{cursor_date}/{chunk_end}",
                       "time": ["00:00", "12:00"], "leadtime_hour": [str(v) for v in selected_leads],
                       "variable": [v[0] for v in VARIABLES], "pressure_level": [str(v) for v in LEVELS],
                       "data_format": "netcdf_zip"}
            key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            fields = 12 + 10 * len(LEVELS)
            size = int(((chunk_end - cursor_date).days + 1) * 2 * len(selected_leads) * 451 * 900 * fields * 4 * 1.25)
            requests.append({"request_id": key, "purpose": purpose, "request": request,
                             "estimated_uncompressed_bytes": size})
            cursor_date = chunk_end + timedelta(days=1)
    if spec.get("cycles"):
        # Sparse integration: only six global initialization snapshots for three
        # cases, and separate selected-cycle target references.
        requests = []
        time_keys = sorted({(_datetime(c) + timedelta(hours=offset)).isoformat()
                            for c in cycles for offset in (-12, 0)})
        if kind == "analysis":
            time_keys = sorted(set(time_keys) | {(_datetime(c) + timedelta(hours=h)).isoformat() for c in cycles for h in leads})
        selections = [("initialization", value, [0]) for value in time_keys]
        if kind == "forecast":
            selections += [("reference", value, leads) for value in cycles]
        for purpose, value, selected_leads in selections:
            when = _datetime(value)
            request = {"type": "forecast", "date": str(when.date()), "time": [when.strftime("%H:%M")],
                       "leadtime_hour": [str(v) for v in selected_leads], "variable": [v[0] for v in VARIABLES],
                       "pressure_level": [str(v) for v in LEVELS], "data_format": "netcdf_zip"}
            fields, cells = 142, 451 * 900
            key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            requests.append({"request_id": key, "purpose": purpose, "request": request,
                             "estimated_uncompressed_bytes": int(len(selected_leads) * cells * fields * 4 * 1.25)})
    for item in requests:
        if item["purpose"] != "reference" or not spec.get("target_variables"):
            continue
        request = item["request"]
        target_names = spec["target_variables"]
        if any(v not in {x[1] for x in VARIABLES} for v in target_names):
            raise ValueError("Unknown target reference variable.")
        request["variable"] = [v[0] for v in VARIABLES if v[1] in target_names]
        selected_levels = spec.get("target_levels", LEVELS)
        if not selected_levels or not set(selected_levels) <= set(LEVELS):
            raise ValueError("Unsupported refined reference levels.")
        request["pressure_level"] = [str(v) for v in selected_levels]
        fields = sum(len(selected_levels) if v[4] == "atmos" else 1 for v in VARIABLES if v[1] in target_names)
        cells = 451 * 900
        domain = spec.get("domain", {})
        if domain:
            north, west, south, east = [float(domain[v]) for v in ("north", "west", "south", "east")]
            if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
                raise ValueError("Invalid target domain.")
            if [north, west, south, east] != [90, -180, -90, 180]:
                request["area"] = [north, west, south, east]
                cells = (int((north - south) / .4) + 1) * (int((east - west) / .4) + 1)
        bounds = request["date"].split("/")
        days = (date.fromisoformat(bounds[-1]) - date.fromisoformat(bounds[0])).days + 1
        item["estimated_uncompressed_bytes"] = int(days * len(request["time"]) * len(request["leadtime_hour"]) * cells * fields * 4 * 1.25)
        item["request_id"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    total = sum(v["estimated_uncompressed_bytes"] for v in requests)
    max_requests = int(limits.get("max_requests", spec.get("max_requests", 32)))
    max_bytes = int(limits.get("max_download_bytes", spec.get("max_download_bytes", 50 * 1024**3)))
    if max_requests <= 0 or max_bytes <= 0 or len(requests) > max_requests:
        raise ValueError("CAMS plan exceeds request count or has invalid limits.")
    return {"product": PRODUCT, "reference_kind": kind, "reference_interpretation":
            "specified-cycle operational model forecast" if kind == "forecast" else "operational lead-zero initial state (analysis)",
            "history_start": history_start.isoformat(), "cycles": cycles, "lead_hours": leads,
            "requests": requests, "request_count": len(requests), "estimated_uncompressed_bytes": total,
            "max_download_bytes": max_bytes, "form_sha256": FORM_SHA256,
            "warnings": ["Historical data may require slow tape retrieval; storage is estimated, service cost is unknown.",
                         "ADS personal access token and manual dataset terms acceptance are required."]}


def safe_extract(archive: Path, destination: Path, max_bytes: int) -> list[Path]:
    """Extract only plain NetCDF members; no paths, symlinks, or ZIP bombs."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if not members or len(members) > 64 or sum(v.file_size for v in members) > max_bytes:
            raise ValueError("CAMS archive exceeds extraction bounds.")
        for member in members:
            name = Path(member.filename)
            if (name.name != member.filename or name.suffix != ".nc" or member.is_dir()
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive member; expected flat NetCDF files.")
        outputs = []
        for member in members:
            target = destination / member.filename
            tmp = target.with_suffix(".nc.partial")
            with zipped.open(member) as source, tmp.open("wb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            os.replace(tmp, target)
            outputs.append(target)
        return outputs


def acquire(spec: dict, cache_root: Path, limits: dict) -> dict:
    """Retrieve using the official client; downloaded bytes use bounded streaming.

    cdsapi handles authenticated job creation/polling. Its result URL is accepted
    only for the ADS service's ECMWF object-store hosts, never from a tool argument.
    """
    acquisition = plan_cams(spec, limits)
    import cdsapi
    import requests
    from urllib.parse import urlparse
    client = cdsapi.Client(url=ADS_URL, quiet=True, debug=False, retry_max=1, timeout=60)
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    budget_key = hashlib.sha256(json.dumps(acquisition, sort_keys=True).encode()).hexdigest()
    budget_path = cache_root / f"budget-{budget_key}.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    used = budget.get("used_bytes", 0)
    submitted = budget.get("submitted_requests", 0)
    results = []
    for item in acquisition["requests"]:
        folder = cache_root / item["request_id"]
        receipt = folder / "receipt.json"
        if receipt.exists():
            cached = json.loads(receipt.read_text())
            if cached.get("files") and all(Path(v["name"]).name == v["name"] and (folder / v["name"]).is_file() and sha256(folder / v["name"]) == v["sha256"]
                   for v in cached["files"]):
                results.append(cached)
                continue
        folder.mkdir(exist_ok=True)
        archive = folder / "data.zip"
        tmp = folder / "data.zip.partial"
        attempts = min(int(limits.get("retries", 1)), 3) + 1
        for attempt in range(attempts):
            try:
                if submitted >= int(limits.get("max_requests", acquisition["request_count"])):
                    raise ValueError("Approved ADS submission count exhausted, including retries.")
                submitted += 1
                atomic_json(budget_path, {"used_bytes": used, "submitted_requests": submitted})
                result = client.retrieve(PRODUCT, item["request"])
                location = result.location
                if int(result.content_length) > acquisition["max_download_bytes"] - used:
                    raise ValueError("ADS result exceeds remaining approved download byte limit.")
                parsed = urlparse(location)
                if (parsed.scheme != "https" or parsed.username or parsed.password or
                        not (parsed.hostname == "ads.atmosphere.copernicus.eu" or
                             str(parsed.hostname).endswith(".ecmwf.int"))):
                    raise ValueError("ADS returned an unapproved download host.")
                with requests.get(location, stream=True, timeout=(30, 60), allow_redirects=False) as response:
                    response.raise_for_status()
                    if response.is_redirect:
                        raise ValueError("Unexpected ADS download redirect.")
                    with tmp.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            used += len(block)
                            atomic_json(budget_path, {"used_bytes": used, "submitted_requests": submitted})
                            if used > acquisition["max_download_bytes"]:
                                raise ValueError("Approved CAMS download byte limit exceeded.")
                            stream.write(block)
                os.replace(tmp, archive)
                files = safe_extract(archive, folder, int(limits.get("max_disk_bytes", acquisition["max_download_bytes"] * 4)))
                cached = {**item, "product": PRODUCT, "reference_kind": acquisition["reference_kind"],
                          "retrieved_at": datetime.now(timezone.utc).isoformat(),
                          "archive_sha256": sha256(archive), "root": str(folder),
                          "files": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in files]}
                atomic_json(receipt, cached)
                results.append(cached)
                break
            except (ValueError, zipfile.BadZipFile):
                tmp.unlink(missing_ok=True)
                raise
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                if attempt + 1 == attempts:
                    # Do not put request headers, PATs, or signed URLs from third-party exceptions in logs.
                    raise RuntimeError(f"ADS request failed ({type(exc).__name__}); check local credentials, terms, and availability.") from None
                time.sleep(min(2**attempt, 4))
    return {**acquisition, "chunks": results, "downloaded_bytes": used, "submitted_requests": submitted}


def canonical_units(value: str) -> str:
    value = str(value).strip().replace("**", "").replace("^", "").replace("{", "").replace("}", "")
    value = " ".join(value.split())
    return {"kg/kg": "kg kg-1", "kg/m2": "kg m-2", "kg/m3": "kg m-3", "m/s": "m s-1",
            "m2/s2": "m2 s-2", "kelvin": "K", "1": "dimensionless"}.get(value, value)


def canonicalize_cams(ds, *, require_backbone: bool = True):
    """Preserve initialization and lead dimensions while reusing legacy grid canonicalization."""
    import numpy as np
    from finetune.prepare_train_test_from_netcdf import _canonicalize_spatial_coordinates
    rename = {}
    for target, candidates in {"latitude": ("lat",), "longitude": ("lon",),
                               "level": ("pressure_level", "isobaricInhPa"),
                               "forecast_reference_time": ("time",),
                               "lead_time": ("forecast_period", "step", "leadtime")}.items():
        if target not in ds:
            for candidate in candidates:
                if candidate in ds:
                    rename[candidate] = target
                    break
    ds = ds.rename(rename)
    for name in ("forecast_reference_time", "lead_time"):
        if name not in ds:
            if name == "lead_time":
                raise ValueError("Missing lead provenance: explicitly add lead_time=0 only for verified lead0 inputs.")
            raise ValueError("Missing forecast_reference_time.")
        if name not in ds.dims:
            ds = ds.expand_dims(name)
    cycles = ds.forecast_reference_time.values.astype("datetime64[ns]")
    leads = ds.lead_time.values
    if np.issubdtype(leads.dtype, np.timedelta64):
        hours = leads / np.timedelta64(1, "h")
    else:
        if ds.lead_time.attrs.get("units") not in ("h", "hour", "hours"):
            raise ValueError("Numeric lead_time requires explicit hour units.")
        hours = leads.astype(float)
    if np.any(~np.isfinite(hours)) or np.any(hours < 0) or len(np.unique(hours)) != len(hours):
        raise ValueError("Invalid or duplicate forecast leads.")
    if np.isnat(cycles).any() or len(np.unique(cycles)) != len(cycles):
        raise ValueError("Missing or duplicate forecast cycles.")
    expected_valid = cycles[:, None] + np.rint(hours * 3600).astype("timedelta64[s]")[None, :]
    if "valid_time" in ds:
        actual = ds.valid_time
        if set(actual.dims) - {"forecast_reference_time", "lead_time"}:
            raise ValueError("Unexpected valid_time dimensions.")
        for name in ("forecast_reference_time", "lead_time"):
            if name not in actual.dims:
                actual = actual.expand_dims({name: ds[name]})
        actual = actual.transpose("forecast_reference_time", "lead_time")
        if actual.shape != expected_valid.shape or not np.array_equal(actual.values.astype("datetime64[ns]"), expected_valid):
            raise ValueError("valid_time does not equal forecast_reference_time + lead_time.")
        ds = ds.drop_vars("valid_time")
    ds = ds.assign_coords(lead_time=("lead_time", hours, {"units": "hours"}),
                          valid_time=(("forecast_reference_time", "lead_time"), expected_valid))
    # Explicit documented aliases only. Never infer chemistry units from magnitude.
    ds = ds.rename({old: new for old, new in {"tcno": "tc_no", "tco3": "gtco3", "o3": "go3"}.items()
                    if old in ds and new not in ds})
    if "level" in ds and ds.level.attrs.get("units", "hPa") not in ("hPa", "millibar", "mbar"):
        raise ValueError("Pressure coordinate must be hPa; implicit Pa conversion is not permitted.")
    if require_backbone:
        missing = [v[1] for v in VARIABLES if v[1] not in ds]
        if missing:
            raise ValueError(f"Missing backbone variables: {missing}")
        if "level" not in ds or set(ds.level.values.tolist()) != set(LEVELS):
            raise ValueError("Full thirteen-level backbone grid required.")
        ds = ds.sel(level=LEVELS)
    for _, name, _, units, kind in VARIABLES:
        if name in ds:
            if canonical_units(ds[name].attrs.get("units", "")) != units:
                raise ValueError(f"{name}: expected physical units {units!r}, found {ds[name].attrs.get('units')!r}.")
            expected_dims = {"forecast_reference_time", "lead_time", "latitude", "longitude"}
            if kind == "atmos":
                expected_dims.add("level")
            if set(ds[name].dims) != expected_dims:
                raise ValueError(f"{name}: unsupported dimensions {ds[name].dims}")
            ds[name].attrs["units"] = units
    wrapped = np.asarray(ds.longitude.values, float) % 360
    order = np.argsort(wrapped)
    for index in np.where(np.isclose(np.diff(wrapped[order]), 0, atol=1e-7, rtol=0))[0]:
        for name in ds.data_vars:
            if "longitude" in ds[name].dims:
                left = ds[name].isel(longitude=int(order[index])).values
                right = ds[name].isel(longitude=int(order[index + 1])).values
                if not np.array_equal(left, right, equal_nan=True):
                    raise ValueError("Duplicate cyclic endpoint fields disagree; refusing silent seam removal.")
    ds = _canonicalize_spatial_coordinates(ds, {"data": {}})
    lat = ds.latitude.values
    lon = ds.longitude.values
    if not np.all(np.isfinite(lat)) or np.any(np.abs(lat) > 90):
        raise ValueError("Invalid latitude coordinates.")
    if len(lon) < 2 or not np.allclose(np.diff(lon), np.diff(lon)[0], rtol=0, atol=1e-5):
        raise ValueError("Longitude grid must be regular and canonical.")
    return ds.sortby("forecast_reference_time").sortby("lead_time")


def split_cases(cycles: list[str], splits: dict, lead_hours: list[int] = LEADS) -> tuple[list[dict], list[dict]]:
    """Assign only cycles whose complete history and targets fit one split window."""
    windows = []
    for name, bounds in splits.items():
        if name not in ("train", "val", "test") or len(bounds) != 2:
            raise ValueError("Splits must map train/val/test to [inclusive_start,inclusive_end].")
        first, last = map(_datetime, bounds)
        if first > last:
            raise ValueError("Reversed split bounds.")
        windows.append((name, first, last))
    for i, (_, first, last) in enumerate(windows):
        if any(max(first, other_first) <= min(last, other_last) for _, other_first, other_last in windows[i + 1:]):
            raise ValueError("Overlapping split windows would leak forecast targets/history.")
    assigned, excluded = [], []
    for text in cycles:
        cycle = _datetime(text)
        history = cycle - timedelta(hours=12)
        end = cycle + timedelta(hours=max(lead_hours))
        matched = [name for name, first, last in windows if first <= history and end <= last]
        if len(matched) != 1:
            excluded.append({"cycle": text, "reason": "history/target window crosses split boundary or falls outside splits"})
        else:
            assigned.append({"cycle": text, "history_times": [history.isoformat(), cycle.isoformat()],
                             "lead_hours": lead_hours, "split": matched[0]})
    return assigned, excluded


def fit_training_statistics(values, split_labels):
    """Only training samples contribute; historical recipes keep learned normalization disabled."""
    import numpy as np
    values, split_labels = np.asarray(values), np.asarray(split_labels)
    if values.ndim < 2 or split_labels.ndim != 1 or values.shape[0] != len(split_labels):
        raise ValueError("Statistics require [sample,...,channel] data and one split label per sample.")
    selected = values[split_labels == "train"]
    if not len(selected) or not np.all(np.isfinite(selected)):
        raise ValueError("Finite training data required for normalization fitting.")
    axes = tuple(range(selected.ndim - 1))
    return {"mean": selected.mean(axis=axes).tolist(), "std": selected.std(axis=axes).tolist(),
            "fit_split": "train", "training_sample_count": len(selected)}


def audit_history(cycle: str, history_times: list[str], available_at: list[str] | None = None) -> None:
    when = _datetime(cycle)
    expected = [when - timedelta(hours=12), when]
    if list(map(_datetime, history_times)) != expected:
        raise ValueError("Autonomous rollout requires exactly [cycle-12h, cycle] history; future inputs prohibited.")
    if available_at and (len(available_at) != 2 or any(_datetime(v) > when for v in available_at)):
        raise ValueError("Input was not available at initialization.")


def prepare(spec: dict, run_dir: Path, acquisition: dict | None = None, limits: dict | None = None) -> dict:
    with ExitStack() as resources:
        return _prepare(spec, run_dir, acquisition, limits, resources)


def _prepare(spec: dict, run_dir: Path, acquisition: dict | None, limits: dict | None, resources: ExitStack) -> dict:
    import importlib.util
    import numpy as np
    import xarray as xr
    paths = [Path(v) for v in spec.get("raw_paths", [])]
    if acquisition:
        paths += [Path(c["root"]) / f["name"] for c in acquisition["chunks"] for f in c["files"]]
    if not paths:
        raise ValueError("Preparation requires acquired chunks or approved raw_paths.")
    limits = limits or {}
    lazy = importlib.util.find_spec("dask") is not None
    estimated_array_bytes = 0
    for path in paths:
        with xr.open_dataset(path) as opened:
            estimated_array_bytes += sum(v.size * v.dtype.itemsize for v in opened.data_vars.values())
    if not lazy and estimated_array_bytes * 3 > float(limits.get("memory_gb", 16)) * 1024**3:
        raise ValueError("Preparation exceeds approved memory estimate; install pinned dask for streaming or use smaller date chunks.")
    sources = []
    histories, forecasts = [], []
    for path in paths:
        if lazy:
            opened = resources.enter_context(xr.open_dataset(path, chunks={}))
            ds = opened
        else:
            with xr.open_dataset(path) as opened:
                ds = opened.load()
        # Surface and pressure-level files validate separately, then as a complete set below.
        ds = canonicalize_cams(ds, require_backbone=False)
        if lazy:
            ds = ds.chunk({"forecast_reference_time": 1, "lead_time": 1})
        if 0 in ds.lead_time:
            histories.append(ds.sel(lead_time=[0]))
        if np.any(ds.lead_time.values > 0):
            forecasts.append(ds.sel(lead_time=ds.lead_time.values[ds.lead_time.values > 0]))
        sources.append({"path": str(path), "sha256": sha256(path), "attrs":
                        {k: str(v) for k, v in ds.attrs.items() if k in ("GRIB_centre", "GRIB_subCentre", "model_cycle", "history", "source")}})
    if not histories:
        raise ValueError("Lead-zero initialization history is missing.")
    merged_history = xr.combine_by_coords(histories, combine_attrs="drop_conflicts", join="exact")
    merged_history = canonicalize_cams(merged_history)
    history_cycle_bytes = sum(v.size * v.dtype.itemsize for v in merged_history.data_vars.values()) / merged_history.sizes["forecast_reference_time"]
    if history_cycle_bytes * 3 > float(limits.get("memory_gb", 16)) * 1024**3:
        raise ValueError("Even one initialization cycle exceeds the approved preparation memory estimate.")
    merged_forecast = xr.combine_by_coords(forecasts, combine_attrs="drop_conflicts", join="exact") if forecasts else None
    cycles = spec.get("cycles")
    if cycles is None:
        planned = plan_cams(spec, limits)
        cycles = planned["cycles"]
    lead_hours = spec.get("lead_hours", LEADS)
    splits = spec.get("splits")
    if not splits:
        raise ValueError("Explicit split windows required; no historical validation interval is invented.")
    cases, excluded = split_cases(cycles, splits, lead_hours)
    if not cases:
        raise ValueError("No complete cases remain after split-boundary purging.")
    history = merged_history.sel(lead_time=0, drop=True).drop_vars("valid_time", errors="ignore")
    history = history.rename(forecast_reference_time="time")
    for case in cases:
        audit_history(case["cycle"], case["history_times"])
        for valid in case["history_times"]:
            if np.datetime64(valid, "ns") not in history.time.values:
                raise ValueError(f"Missing initialization history time {valid}.")
    kind = spec.get("reference_kind", "forecast")
    references = []
    for case in cases:
        cycle = np.datetime64(case["cycle"], "ns")
        if kind == "forecast":
            if merged_forecast is None:
                raise ValueError("Specified-cycle forecast reference fields are missing.")
            reference = merged_forecast.sel(forecast_reference_time=[cycle], lead_time=lead_hours)
        elif kind == "analysis":
            times = cycle + np.asarray(lead_hours, dtype="timedelta64[h]")
            reference = history.sel(time=times).rename(time="lead_time").assign_coords(lead_time=lead_hours)
            reference = reference.expand_dims(forecast_reference_time=[cycle])
            reference = reference.assign_coords(valid_time=(("forecast_reference_time", "lead_time"), times[None, :]))
        else:
            raise ValueError("Unsupported reference product.")
        references.append(reference)
    reference = xr.concat(references, dim="forecast_reference_time")
    # Backbone cannot ingest missing data. Reference missing cells stay masked for honest coverage.
    required_times = sorted({v for case in cases for v in case["history_times"]})
    for _, name, _, _, _ in VARIABLES:
        for valid in required_times:
            if not np.isfinite(history[name].sel(time=np.datetime64(valid, "ns")).values).all():
                raise ValueError(f"Nonfinite initialization field {name}; explicit imputation recipe required.")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for label, dataset in (("data", history), ("reference", reference)):
        path = run_dir / f"prepared-{label}.nc"
        tmp = path.with_suffix(".nc.partial")
        dataset.attrs.update(product=PRODUCT, reference_kind=kind, preparation_contract="cycle-preserving-v1")
        if lazy:
            import dask
            with dask.config.set(scheduler="synchronous"):
                dataset.to_netcdf(tmp, engine="netcdf4")
        else:
            dataset.to_netcdf(tmp, engine="netcdf4")
        os.replace(tmp, path)
        output_paths[label + "_path"] = str(path)
    result = {"schema_version": 1, **output_paths, "cases": cases, "excluded_cases": excluded,
              "source": {"product": PRODUCT, "reference_kind": kind, "inputs": sources,
                         "model_cycles": sorted({v["attrs"].get("model_cycle", "not supplied") for v in sources})},
              "sha256": {k: sha256(Path(v)) for k, v in output_paths.items()},
              "normalization": {"enabled": False, "mode": "none", "learned_statistics": None},
              "splits": splits, "history_audit": "all autonomous inputs at cycle-12h and cycle; no future targets"}
    atomic_json(run_dir / "prepared.json", result)
    return result


def execute_data(stage: str, plan: dict, run_dir: Path) -> dict:
    spec = resolve_data_spec(dict(plan.get("data", {})), plan.get("mode", "replay"))
    limits = plan.get("limits", {})
    if stage == "acquire":
        roots = plan.get("roots", {})
        result = acquire(spec, Path(roots.get("cache", run_dir / "cache")) / "cams", limits)
        atomic_json(run_dir / "acquisition.json", result)
        return {"state": "succeeded", "outputs": {"acquisition": str(run_dir / "acquisition.json")},
                "verification": {"requests": len(result["chunks"]), "downloaded_bytes": result["downloaded_bytes"]}}
    if stage == "prepare":
        inputs = plan.get("inputs", {})
        acquisition_path = inputs.get("acquisition", inputs.get("raw_manifest", run_dir / "acquisition.json"))
        acquisition = json.loads(Path(acquisition_path).read_text()) if Path(acquisition_path).exists() else None
        from .common import safe_path
        allowed_roots = [Path(v) for v in plan.get("roots", {}).values()]
        allowed_roots += [Path(v) for v in plan.get("allowed_read_roots", [])]
        allowed_roots += [run_dir]
        for raw_path in spec.get("raw_paths", []):
            safe_path(raw_path, allowed_roots, exists=True)
        if acquisition:
            if acquisition.get("product") != PRODUCT:
                raise ValueError("Raw manifest must identify the official CAMS operational product.")
            for chunk in acquisition["chunks"]:
                for record in chunk["files"]:
                    if Path(record["name"]).name != record["name"]:
                        raise ValueError("Raw manifest contains an unsafe file name.")
                    path = safe_path(Path(chunk["root"]) / record["name"], allowed_roots, exists=True)
                    if sha256(path) != record["sha256"]:
                        raise ValueError("Raw data changed after retrieval; checksum mismatch.")
        result = prepare(spec, run_dir, acquisition, limits)
        return {"state": "succeeded", "outputs": {"prepared": str(run_dir / "prepared.json"),
                "data": result["data_path"], "reference_data": result["reference_path"]},
                "verification": {"cases": len(result["cases"]), "excluded_cases": len(result["excluded_cases"]),
                                 "history_audit": result["history_audit"]}}
    raise ValueError(f"Unsupported data stage: {stage}")
