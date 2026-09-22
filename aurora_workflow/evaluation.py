"""Physical-unit, matched-case metrics for point forecasts and finite ensembles.

The original evaluation notebook remains available. This importable evaluator
adds explicitly weighted and cycle-preserving products, without relabeling CAMS
model agreement as observational forecast accuracy.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .data import atomic_json, canonical_units, sha256


def improvement(baseline: float, refined: float) -> float | None:
    """Positive means lower error; zero baseline has undefined percent change."""
    import math
    if not math.isfinite(baseline) or not math.isfinite(refined) or baseline <= 0:
        return None
    return 100.0 * (baseline - refined) / baseline


def area_weights(lat, lon):
    """Relative spherical cell area on regular longitude and monotone latitude."""
    import numpy as np
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 2:
        raise ValueError("One-dimensional latitude/longitude with at least two cells required.")
    if not np.isfinite(lat).all() or np.any(abs(lat) > 90) or not np.isfinite(lon).all():
        raise ValueError("Nonfinite/out-of-range spatial coordinates.")
    if not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)):
        raise ValueError("Latitude must be strictly monotone.")
    if not np.all(np.diff(lon) > 0) or not np.allclose(np.diff(lon), np.diff(lon)[0], atol=1e-5, rtol=0):
        raise ValueError("Longitude must be increasing and regular.")
    if len(np.unique(np.round(lon % 360, 7))) != len(lon):
        raise ValueError("Duplicate cyclic longitude endpoint.")
    edges = np.r_[lat[0] + (lat[0] - lat[1]) / 2,
                  (lat[:-1] + lat[1:]) / 2, lat[-1] + (lat[-1] - lat[-2]) / 2]
    edges = np.clip(edges, -90, 90)
    areas = np.abs(np.diff(np.sin(np.deg2rad(edges))))[:, None] * np.ones((1, len(lon)))
    return areas / areas.sum()


def _mean(values, weights, mask):
    import numpy as np
    weighted = np.broadcast_to(weights, np.shape(values))
    keep = np.asarray(mask) & np.isfinite(values) & (weighted > 0)
    denominator = weighted[keep].sum()
    return float(np.sum(np.asarray(values)[keep] * weighted[keep]) / denominator) if denominator > 0 else None


def point_metrics(prediction, reference, weights, mask=None) -> dict:
    """Pooled weighted errors; mean of per-case ordinary spatial correlations."""
    import numpy as np
    pred, ref = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    if pred.shape != ref.shape or pred.ndim < 2:
        raise ValueError("Point forecast/reference shapes differ.")
    common = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        common &= np.broadcast_to(mask, pred.shape)
    error = pred - ref
    mse = _mean(error**2, weights, common)
    correlations = []
    for p, r, valid in zip(pred.reshape(-1, *pred.shape[-2:]), ref.reshape(-1, *ref.shape[-2:]),
                           common.reshape(-1, *common.shape[-2:])):
        pm, rm = _mean(p, weights, valid), _mean(r, weights, valid)
        if pm is None or rm is None:
            continue
        pv, rv = _mean((p - pm)**2, weights, valid), _mean((r - rm)**2, weights, valid)
        if pv > 0 and rv > 0 and valid.sum() >= 2:
            correlations.append(_mean((p - pm) * (r - rm), weights, valid) / np.sqrt(pv * rv))
    return {"mae": _mean(abs(error), weights, common), "rmse": float(np.sqrt(mse)) if mse is not None else None,
            "bias": _mean(error, weights, common),
            "spatial_correlation": float(np.mean(correlations)) if correlations else None,
            "correlation_case_count": len(correlations), "valid_points": int(common.sum()),
            "excluded_points": int(common.size - common.sum())}


def ensemble_metrics(ensemble, reference, weights, mask=None) -> dict:
    """Empirical CRPS, population spread, equal-tail coverage and fractional ranks.

    CRPS = mean |X-y| - 0.5 mean |X-X'| for the finite empirical distribution.
    The sorted expression avoids allocating an M×M×grid array.
    """
    import numpy as np
    members, ref = np.asarray(ensemble, float), np.asarray(reference, float)
    if members.ndim != ref.ndim + 1 or members.shape[1:] != ref.shape or members.shape[0] < 2:
        raise ValueError("Probabilistic diagnostics require >=2 members with the reference shape.")
    valid = np.isfinite(ref) & np.all(np.isfinite(members), axis=0)
    if mask is not None:
        valid &= np.broadcast_to(mask, ref.shape)
    count = len(members)
    sorted_members = np.sort(members, axis=0)
    coefficients = (2 * np.arange(1, count + 1) - count - 1).reshape((count,) + (1,) * ref.ndim)
    crps = np.mean(abs(members - ref), axis=0) - np.sum(coefficients * sorted_members, axis=0) / count**2
    mean = np.mean(members, axis=0)
    variance = _mean(np.var(members, axis=0, ddof=0), weights, valid)
    mse = _mean((mean - ref)**2, weights, valid)
    spread = float(np.sqrt(variance)) if variance is not None else None
    skill = float(np.sqrt(mse)) if mse is not None else None
    intervals = []
    for nominal in (0.5, 0.8, 0.9):
        lower, upper = np.quantile(members, [(1 - nominal) / 2, (1 + nominal) / 2], axis=0)
        intervals.append({"nominal": nominal, "coverage": _mean((ref >= lower) & (ref <= upper), weights, valid),
                          "mean_width": _mean(upper - lower, weights, valid)})
    # Rank ties share their mass equally over admissible ranks. No artificial RNG jitter.
    lower_rank = np.sum(members < ref, axis=0)
    ties = np.sum(members == ref, axis=0)
    ranks = [_mean(((lower_rank <= rank) & (rank <= lower_rank + ties)) / (ties + 1), weights, valid)
             for rank in range(count + 1)]
    return {"member_count": count, "crps": _mean(crps, weights, valid), "spread": spread,
            "ensemble_mean_rmse": skill, "spread_skill_ratio": spread / skill if skill and spread is not None else None,
            "intervals": intervals, "rank_histogram_fraction": ranks,
            "valid_points": int(valid.sum()), "excluded_points": int(valid.size - valid.sum())}


def _aligned_npz(paths: dict) -> tuple[dict, dict]:
    """Exact grid/channel/units; intersection by unique cycle plus lead, with coverage."""
    import numpy as np
    arrays = {}
    required = ("fields", "case_cycle", "lead_hours", "lat", "lon", "channel", "units")
    for role, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            missing = [v for v in required if v not in data]
            if missing:
                raise ValueError(f"{role} missing evaluation metadata: {missing}")
            arr = {name: data[name] for name in data.files}
        for key in ("case_cycle", "lead_hours", "channel"):
            if len(np.unique(arr[key])) != len(arr[key]):
                raise ValueError(f"{role} has duplicate {key} identifiers.")
        expected = tuple(len(arr[v]) for v in ("case_cycle", "lead_hours", "channel", "lat", "lon"))
        if arr["fields"].shape != expected:
            raise ValueError(f"{role} field dimensions do not match coordinates.")
        if len(arr["units"]) != len(arr["channel"]) or any(not str(v).strip() for v in arr["units"]):
            raise ValueError("Explicit channel units required.")
        if (not np.isfinite(arr["lead_hours"]).all() or np.any(arr["lead_hours"] <= 0)
                or np.any(arr["lead_hours"] % 12) or np.any(arr["lead_hours"] > 72)):
            raise ValueError("Evaluation leads must be finite multiples of 12 through 72 hours.")
        cycles = arr["case_cycle"].astype("datetime64[ns]")
        if np.isnat(cycles).any():
            raise ValueError("Missing initialization timestamps.")
        expected_valid = cycles[:, None] + (arr["lead_hours"] * 3600).astype("timedelta64[s]")[None, :]
        if "valid_time" in arr and not np.array_equal(arr["valid_time"].astype("datetime64[ns]"), expected_valid):
            raise ValueError("Stored valid time differs from cycle plus lead.")
        arr["case_cycle"] = cycles.astype(str)
        arrays[role] = arr
    reference = arrays["reference"]
    for role, arr in arrays.items():
        for key in ("channel", "lat", "lon"):
            if not np.array_equal(arr[key], reference[key]):
                raise ValueError(f"Exact evaluation {key} mismatch in {role}; no implicit regridding.")
        if list(map(canonical_units, arr["units"])) != list(map(canonical_units, reference["units"])):
            raise ValueError(f"Physical units mismatch in {role}.")
    common_cycles = sorted(set.intersection(*(set(v["case_cycle"]) for v in arrays.values())))
    common_leads = sorted(set.intersection(*(set(v["lead_hours"]) for v in arrays.values())))
    if not common_cycles or not common_leads:
        raise ValueError("No identical cycle/lead cases across baseline, refined and reference.")
    coverage = {role: {"available_cycles": len(v["case_cycle"]), "available_leads": len(v["lead_hours"]),
                        "missing_cycles": sorted(set(reference["case_cycle"]) - set(v["case_cycle"])),
                        "excluded_cycles": sorted(set(v["case_cycle"]) - set(common_cycles)),
                        "excluded_leads": sorted(float(x) for x in set(v["lead_hours"]) - set(common_leads))}
                for role, v in arrays.items()}
    for arr in arrays.values():
        ci = [list(arr["case_cycle"]).index(v) for v in common_cycles]
        li = [list(arr["lead_hours"]).index(v) for v in common_leads]
        old_shape = arr["fields"].shape
        arr["fields"] = arr["fields"][ci][:, li]
        if "mask" in arr:
            arr["mask"] = np.broadcast_to(arr["mask"], old_shape)[ci][:, li]
        if "ensemble" in arr:
            if arr["ensemble"].shape[1:] != old_shape:
                raise ValueError("Ensemble coordinate shape mismatch.")
            arr["ensemble"] = arr["ensemble"][:, ci][:, :, li]
        if "split" in arr:
            arr["split"] = arr["split"][ci]
        arr["case_cycle"], arr["lead_hours"] = np.asarray(common_cycles), np.asarray(common_leads)
    for arr in arrays.values():
        if "split" in reference and ("split" not in arr or not np.array_equal(arr["split"], reference["split"])):
            raise ValueError("Forecast/reference split labels differ.")
    return arrays, coverage


def evaluate_arrays(baseline, refined, reference, metadata: dict, ensemble=None) -> dict:
    import numpy as np
    baseline, refined, reference = map(lambda v: np.asarray(v, dtype=float), (baseline, refined, reference))
    if baseline.shape != refined.shape or baseline.shape != reference.shape or baseline.ndim != 5:
        raise ValueError("Expected identically shaped [case,lead,channel,lat,lon] fields.")
    weights = area_weights(metadata["lat"], metadata["lon"])
    common = np.isfinite(baseline) & np.isfinite(refined) & np.isfinite(reference)
    if "mask" in metadata:
        common &= np.broadcast_to(metadata["mask"], common.shape)
    shape = reference.shape[-2:]
    boundary = np.zeros(shape, bool)
    width = min(2, max(1, min(shape) // 4))
    boundary[:width] = boundary[-width:] = True
    boundary[:, :width] = boundary[:, -width:] = True
    south = np.zeros(shape, bool)
    south[np.argsort(metadata["lat"])[:width], :] = True
    subsets = {"region": np.ones(shape, bool), "interior": ~boundary, "boundary": boundary, "southern_boundary": south}
    if "land_mask" in metadata:
        land = np.asarray(metadata["land_mask"]) >= 0.5
        if land.shape != shape:
            raise ValueError("Static land mask grid mismatch.")
        coast = np.zeros(shape, bool)
        coast[1:] |= land[1:] != land[:-1]
        coast[:-1] |= land[1:] != land[:-1]
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        coast[:, :-1] |= land[:, 1:] != land[:, :-1]
        subsets["coastal"] = coast
    cycles = np.asarray(metadata["case_cycle"], dtype="datetime64[h]")
    init_hours = cycles.astype(int) % 24
    rows = []
    for channel_index, channel in enumerate(metadata["channel"]):
        for lead_index, lead in enumerate(metadata["lead_hours"]):
            b, r, target = (v[:, lead_index, channel_index] for v in (baseline, refined, reference))
            valid = common[:, lead_index, channel_index]
            local_subsets = dict(subsets)
            threshold = metadata.get("hotspot_thresholds", {}).get(str(channel))
            if threshold is not None:
                local_subsets["hotspot"] = target >= float(threshold)
            if valid.any():
                # Descriptive reference-conditional upper tail, never used for training or tuning.
                local_subsets["reference_top_decile"] = target >= np.quantile(target[valid], 0.9)
            for label, subset in local_subsets.items():
                categories = [("all", np.ones(len(cycles), bool))]
                if label == "region":
                    categories += [(f"initialization_{hour:02d}UTC", init_hours == hour) for hour in sorted(set(init_hours))]
                    valid_hours = (init_hours + int(lead)) % 24
                    categories += [(f"valid_{hour:02d}UTC", valid_hours == hour) for hour in sorted(set(valid_hours))]
                for category, selection in categories:
                    mask = valid & subset & selection[:, None, None]
                    bm, rm = point_metrics(b, target, weights, mask), point_metrics(r, target, weights, mask)
                    row = {"channel": str(channel), "units": str(metadata["units"][channel_index]), "lead_hours": float(lead),
                           "subset": label, "utc_category": category, "selected_cases": int(selection.sum()),
                           "baseline": bm, "refined_point": rm,
                           "improvement_percent": {key: improvement(bm[key], rm[key]) if bm[key] is not None and rm[key] is not None else None
                                                   for key in ("mae", "rmse")}}
                    if ensemble is not None:
                        members = np.asarray(ensemble)[:, :, lead_index, channel_index]
                        emask = mask & np.all(np.isfinite(members), axis=0)
                        row["ensemble"] = ensemble_metrics(members, target, weights, emask)
                        row["baseline_on_ensemble_mask"] = point_metrics(b, target, weights, emask)
                        row["ensemble_mean"] = point_metrics(np.mean(members, axis=0), target, weights, emask)
                        if label == "region" and category == "all":
                            row["individual_members"] = [point_metrics(v, target, weights, emask) for v in members]
                    rows.append(row)
    diagnostics = []
    for row in rows:
        if row["subset"] == "southern_boundary" and row["channel"].lower() == "tcno2":
            b, r = row["baseline"], row["refined_point"]
            if b["rmse"] is not None and r["rmse"] is not None and r["rmse"] > b["rmse"]:
                diagnostics.append({"channel": row["channel"], "lead_hours": row["lead_hours"],
                                    "finding": "southern-boundary tcNO2 RMSE degradation", "baseline_bias": b["bias"], "refined_bias": r["bias"]})
        if row["subset"] != "region" or row["utc_category"] != "all":
            continue
        b, r = row["baseline"], row["refined_point"]
        if (b["spatial_correlation"] is not None and r["spatial_correlation"] is not None
                and r["spatial_correlation"] > b["spatial_correlation"]
                and any(r[k] > b[k] for k in ("mae", "rmse"))):
            diagnostics.append({"channel": row["channel"], "lead_hours": row["lead_hours"],
                                "finding": "correlation improved while absolute error degraded"})
    global_lon = abs((float(metadata["lon"][-1]) - float(metadata["lon"][0])) +
                     float(metadata["lon"][1]) - float(metadata["lon"][0]) - 360) < 1e-4
    for name, product_values in (("baseline", baseline), ("refined_point", refined), ("reference", reference)):
        for channel_index, channel in enumerate(metadata["channel"]):
            values = np.where(common[:, :, channel_index], product_values[:, :, channel_index], np.nan)
            finite_diffs = np.abs(np.diff(values, axis=-1))
            finite_diffs = finite_diffs[np.isfinite(finite_diffs)]
            diagnostic = {"product": name, "channel": str(channel), "units": str(metadata["units"][channel_index]),
                          "mean_adjacent_longitude_jump": float(finite_diffs.mean()) if finite_diffs.size else None}
            if global_lon:
                seam = abs(values[..., 0] - values[..., -1])
                diagnostic["mean_longitude_seam_jump"] = float(np.nanmean(seam)) if np.isfinite(seam).any() else None
            patch = int(metadata.get("patch_size", 3))
            if patch > 0:
                jumps = abs(np.diff(values, axis=-1))[..., patch - 1::patch]
                diagnostic["mean_patch_boundary_jump"] = float(np.nanmean(jumps)) if np.isfinite(jumps).any() else None
                latitude_jumps = abs(np.diff(values, axis=-2))[..., patch - 1::patch, :]
                diagnostic["mean_latitude_patch_boundary_jump"] = float(np.nanmean(latitude_jumps)) if np.isfinite(latitude_jumps).any() else None
            diagnostics.append(diagnostic)
    return {"schema_version": 1, "reference_interpretation": metadata.get("reference_interpretation", "CAMS operational model reference; not independent observations"),
            "aggregation": "pooled spherical-cell-area weighted MAE/RMSE/bias; arithmetic mean of per-case weighted spatial correlations",
            "correlation": "ordinary spatial pattern correlation; no climatology or anomaly correlation",
            "improvement_definition": "100 * (baseline_error - refined_error) / baseline_error; null for zero baseline; negative is degradation",
            "uncertainty_intervals": "not computed; forecast cycles overlap, so grid cells must not be treated as independent samples",
            "coverage": {"matched_cycles": len(cycles), "matched_leads": len(metadata["lead_hours"]),
                         "valid_points": int(common.sum()), "excluded_points": int(common.size - common.sum())},
            "subset_definitions": {"boundary": f"outer {width} grid cells", "southern_boundary": f"southernmost {width} latitude rows",
                                   "coastal": "cells adjacent to a supplied static land-mask transition; omitted without static data",
                                   "hotspot": "explicit per-channel physical-unit threshold; omitted when not supplied",
                                   "reference_top_decile": "per-channel/lead matched-reference 90th percentile, descriptive only"},
            "rows": rows, "diagnostics": diagnostics,
            "limitations": ["No independent observational accuracy claim.", "No temporal-adapter attribution without matched separate experiments.",
                            "Patch/seam jumps are descriptive diagnostics, not proof of a model artifact."]}


def write_report(metrics: dict, run_dir: Path) -> dict:
    rows = [v for v in metrics["rows"] if v["subset"] == "region" and v["utc_category"] == "all"]
    lines = ["# Aurora evaluation", "", metrics["reference_interpretation"], "",
             f"Matched {metrics['coverage']['matched_cycles']} cycles and {metrics['coverage']['matched_leads']} leads; "
             f"excluded {metrics['coverage']['excluded_points']} nonfinite/masked points.", "", metrics["aggregation"], "",
             "| Channel | Lead h | Baseline RMSE | Refined point RMSE | Improvement % |", "|---|---:|---:|---:|---:|"]
    fmt = lambda x: "undefined" if x is None else f"{x:.6g}"
    for row in rows:
        lines.append(f"| {row['channel']} ({row['units']}) | {row['lead_hours']:g} | {fmt(row['baseline']['rmse'])} | "
                     f"{fmt(row['refined_point']['rmse'])} | {fmt(row['improvement_percent']['rmse'])} |")
    lines += ["", metrics["improvement_definition"], "", metrics["uncertainty_intervals"], "",
              "Machine-readable subset, UTC, coverage, and ensemble diagnostics: [metrics.json](metrics.json).",
              "", "[RMSE improvement plot](metrics.svg)", "", *metrics["limitations"]]
    report = run_dir / "evaluation.md"
    report.write_text("\n".join(lines) + "\n")
    # Standalone SVG: no plotting dependency, executable scripts, or external resources.
    height = 70 + 28 * len(rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" role="img" aria-label="RMSE improvement by channel and lead">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="25" font-family="sans-serif" font-size="16">RMSE improvement (%) — negative values indicate degradation</text>',
           f'<line x1="560" x2="560" y1="45" y2="{height}" stroke="#555"/>']
    for i, row in enumerate(rows):
        value = row["improvement_percent"]["rmse"]
        y = 65 + 28 * i
        label = html.escape(f"{row['channel']} +{row['lead_hours']:g}h: {fmt(value)}")
        svg.append(f'<text x="20" y="{y}" font-family="sans-serif" font-size="13">{label}</text>')
        if value is not None:
            length = min(abs(value) * 2, 280)
            x = 560 if value >= 0 else 560 - length
            svg.append(f'<rect x="{x}" y="{y-12}" width="{length}" height="16" fill="{"#18765a" if value >= 0 else "#b43b39"}"/>')
    svg.append("</svg>")
    plot = run_dir / "metrics.svg"
    plot.write_text("\n".join(svg))
    return {"evaluation_summary": str(report), "evaluation_plot": str(plot)}


def execute_evaluation(plan: dict, run_dir: Path) -> dict:
    import numpy as np
    inputs = plan.get("inputs", {})
    paths = {name: Path(inputs.get(name, run_dir / f"{name}.npz")) for name in ("baseline", "refined", "reference")}
    if plan.get("head") == "none" and not paths["refined"].exists():
        paths["refined"] = paths["baseline"]
    arrays, coverage = _aligned_npz(paths)
    metadata = {k: v for k, v in arrays["reference"].items() if k != "fields"}
    metadata.update({k: v for k, v in plan.get("evaluation", {}).items()
                     if k in ("hotspot_thresholds", "patch_size", "reference_interpretation")})
    common = np.ones(arrays["reference"]["fields"].shape, bool)
    for arr in arrays.values():
        if "mask" in arr:
            common &= arr["mask"].astype(bool)
    metadata["mask"] = common
    if "split" in metadata:
        selected = plan.get("evaluation", {}).get("split", "test")
        split = np.asarray(metadata["split"])
        if selected not in ("train", "val", "test"):
            raise ValueError("Evaluation split must be train, val, or test.")
        keep = split == selected
        if not keep.any():
            raise ValueError(f"No matched {selected} evaluation cases.")
        for arr in arrays.values():
            arr["fields"] = arr["fields"][keep]
            if "ensemble" in arr:
                arr["ensemble"] = arr["ensemble"][:, keep]
        metadata["case_cycle"] = metadata["case_cycle"][keep]
        metadata["mask"] = common[keep]
    metrics = evaluate_arrays(*(arrays[v]["fields"] for v in ("baseline", "refined", "reference")),
                              metadata, arrays["refined"].get("ensemble"))
    metrics["matching_coverage"] = coverage
    metrics["evaluation_split"] = plan.get("evaluation", {}).get("split", "test") if "split" in metadata else "unspecified"
    recipe_id = plan.get("recipe_id", plan.get("recipe", ""))
    if isinstance(recipe_id, dict):
        recipe_id = recipe_id.get("id", "")
    metrics["evidence_kind"] = "synthetic_fixture" if recipe_id == "cpu-smoke-v1" or plan.get("fixture") else "executed_evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    atomic_json(metrics_path, metrics)
    artifacts = {"metrics": str(metrics_path), **write_report(metrics, run_dir)}
    receipt = {"schema_version": 1, "executor": "aurora_workflow.evaluation.execute_evaluation",
               "source_sha256": sha256(Path(__file__)), "configuration_hash": plan.get("configuration_hash", hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()),
               "dependency_sha256": {v: sha256(Path(__file__).with_name(v)) for v in ("data.py", "common.py")},
               "parameters": {"head": plan.get("head"), "recipe_id": recipe_id, "fixture": bool(plan.get("fixture")),
                              "evaluation": plan.get("evaluation", {})},
               "executed_at": datetime.now(timezone.utc).isoformat(), "execution_status": "executed",
               "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in paths.items()},
               "outputs": {k: {"path": str(v), "sha256": sha256(Path(v))} for k, v in artifacts.items()},
               "reference_interpretation": metrics["reference_interpretation"], "evidence_kind": metrics["evidence_kind"]}
    atomic_json(run_dir / "evaluation-receipt.json", receipt)
    artifacts["evaluation_receipt"] = str(run_dir / "evaluation-receipt.json")
    return {"state": "succeeded", "outputs": artifacts, "verification": metrics["coverage"],
            "warnings": metrics["limitations"]}
