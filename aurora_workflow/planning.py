"""Typed planning, immutable effective configurations, and input fingerprinting."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .common import REPO, artifact, digest, environment_fingerprint, identifier, read_json, safe_path, sha256, source_fingerprint
from .settings import Limits, Settings

Head = Literal["none", "flow_matching_unet", "flow_matching_transformer", "diffusion_unet",
               "diffusion_transformer", "flow_matching_conv_unet"]
Mode = Literal["replay", "resume", "warm-start", "reproduce", "evaluate"]
Stage = Literal["sources", "assets", "acquire", "prepare", "rollout", "train", "refine",
                "evaluate", "report", "smoke"]
HEAD_NAMES = {"none": "Unchanged Aurora baseline", "flow_matching_unet": "Flow matching with a UNet (legacy adapter)",
              "flow_matching_transformer": "Flow matching with a Transformer",
              "diffusion_unet": "Diffusion with a UNet", "diffusion_transformer": "Diffusion with a Transformer",
              "flow_matching_conv_unet": "Flow matching with a convolutional UNet (unified)"}


class EvaluationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    split: Literal["train", "val", "test"] = "test"
    hotspot_thresholds: dict[str, float] = Field(default_factory=dict)
    patch_size: int = Field(3, ge=1, le=128)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: str = "cpu-smoke-v1"
    head: Head | None = None
    mode: Mode | None = None
    product: Literal["point", "ensemble"] = "point"
    ensemble_members: int = Field(1, ge=1, le=1000)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    case_id: str = "reproduction"
    inputs: dict[str, str] = Field(default_factory=dict)
    limits: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)
    evaluation: EvaluationOptions = Field(default_factory=EvaluationOptions)
    stages: list[Stage] | None = None


def load_recipe(name: str) -> dict:
    return read_json(REPO / "recipes" / (identifier(name) + ".json"))


def validate_recipe(name: str, head: Head | None = None) -> dict:
    recipe = load_recipe(name)
    config = copy.deepcopy(recipe["config"])
    head = head or recipe["default_head"]
    if head not in HEAD_NAMES:
        raise ValueError("Unknown refinement selection")
    if head == "none":
        config["model"]["refinement"] = {"enabled": False, "type": "none", "feedback_to_rollout": False}
    else:
        if head not in recipe["head_configs"]:
            raise ValueError("Head does not have a versioned configuration in this recipe")
        config["model"]["refinement"] = copy.deepcopy(recipe["head_configs"][head])
    model = config["model"]
    if model["model_variant"] != "aurora_air_pollution":
        raise ValueError("This workflow only supports AuroraAirPollution")
    if model["refinement"].get("feedback_to_rollout", False):
        raise ValueError("Canonical recipe forbids refinement feedback")
    temporal = model.get("mamba_temporal", {})
    unified = head in {"flow_matching_conv_unet", "flow_matching_transformer", "diffusion_unet", "diffusion_transformer"}
    temporal_enabled = temporal.get("enabled", model.get("mamba_temporal_enabled", unified))
    if temporal_enabled or model.get("mamba_temporal_enabled", False):
        raise ValueError("Canonical recipe must be spatial-only; temporal experiments require a separate recipe")
    refinement = model["refinement"]
    if refinement.get("temporal", {}).get("backend", "none") != "none":
        raise ValueError("Portable recipes do not yet supply temporal context; use the documented experimental API")
    if any(refinement.get("conditioning", {}).get(key, False) for key in ("calendar", "solar_geometry")):
        raise ValueError("Portable recipes do not yet supply calendar/solar context; use the documented experimental API")
    if config["data"]["input_time_steps"] != 2 or config["rollout"]["rollout_step_hours"] != 12:
        raise ValueError("Air-pollution model requires two history times at 12 hour cadence")
    if set(config["data"]["atmos_levels"]) != {50,100,150,200,250,300,400,500,600,700,850,925,1000}:
        raise ValueError("Full 13-level backbone grid is required")
    # Recipes are immutable snapshots derived from a named historical commit.
    # Later notebook/YAML experiments must not silently change or invalidate them.
    source_commit, source_path = recipe["source_commit"], recipe["source_config"]
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Recipe source must be pinned to a full Git commit")
    if not isinstance(source_path, str) or not source_path.startswith("finetune/") or ".." in Path(source_path).parts:
        raise ValueError("Recipe source must be a repository-relative finetune configuration")
    try:
        source = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{source_commit}:{source_path}"],
            stderr=subprocess.DEVNULL, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Pinned recipe source is unavailable; fetch the source history") from exc
    if hashlib.sha256(source).hexdigest() != recipe["source_config_sha256"]:
        raise ValueError("Pinned source configuration checksum mismatch; version and review the recipe")
    return {"state": "validated", "recipe": recipe, "config": config, "head": head,
            "configuration_hash": digest(config), "warnings": recipe.get("changes_from_source", [])}


def fingerprint_inputs(inputs: dict[str, str], settings: Settings) -> tuple[dict, dict]:
    allowed = {"pretrained", "static", "refinement", "prepared", "baseline", "forecast", "reference",
               "refined", "raw_manifest", "normalization", "refinement_metadata", "packing", "acquisition"}
    if set(inputs) - allowed:
        raise ValueError("Unknown input roles: " + ", ".join(sorted(set(inputs) - allowed)))
    resolved, fingerprints = {}, {}
    for role, raw in inputs.items():
        path = safe_path(raw, settings.roots, exists=True)
        resolved[role] = str(path)
        fingerprints[role] = artifact(path, role)
        if path.suffix == ".json":
            # Account for companion files explicitly referenced by input manifests.
            def visit(obj, key="", depth=0):
                if depth > 15:
                    raise ValueError("Metadata nesting exceeds limit")
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        visit(v, k, depth + 1)
                elif isinstance(obj, list):
                    for v in obj:
                        visit(v, key, depth + 1)
                elif isinstance(obj, str) and (key in {"path", "weights_file", "data_path", "reference_path"}
                                              or key.endswith("_path")):
                    p = Path(obj)
                    if not p.is_absolute():
                        p = path.parent / p
                    p = safe_path(p, settings.roots, exists=True)
                    fingerprints[f"{role}:{len(fingerprints)}"] = artifact(p, key)
            visit(read_json(path))
    return resolved, fingerprints


def make_plan(request: PlanRequest, settings: Settings) -> dict:
    valid = validate_recipe(request.recipe, request.head)
    recipe, cfg, head = valid["recipe"], valid["config"], valid["head"]
    mode = request.mode or recipe["default_mode"]
    identifier(request.case_id)
    limits = Limits.model_validate({**recipe["limits"], **request.limits}).model_dump()
    if limits["device"].startswith("cuda") and limits["gpu_count"] != 1:
        if "gpu_count" in request.limits:
            raise ValueError("cuda execution requires gpu_count=1")
        limits["gpu_count"] = 1
    if limits["device"] == "cpu" and limits["gpu_count"]:
        raise ValueError("cpu recipe cannot allocate a GPU")
    if request.ensemble_members > limits["max_ensemble_members"]:
        raise ValueError("Ensemble exceeds resource bounds")
    if request.product == "point" and request.ensemble_members != 1:
        raise ValueError("Point product has one deterministic prediction")
    if request.product == "ensemble" and (request.ensemble_members < 2 or head == "none"):
        raise ValueError("Ensemble product requires a stochastic head and at least two members")
    configured_assets = {k: v["path"] for k,v in settings.assets.items()
                         if k in {"pretrained", "static", "refinement"} and v.get("path")}
    inputs, fingerprints = fingerprint_inputs({**configured_assets, **request.inputs}, settings)
    cfg["training"]["seed"] = request.seed
    cfg["case_name"] = request.case_id + "-" + head
    cfg["model"]["refinement"].update(seed=request.seed, ensemble_size=request.ensemble_members,
                                     deterministic_inference=request.product == "point")
    # Limits cap execution; retain the scientific scheduler horizon across interrupted runs.
    data = {**recipe["data"], **request.data}
    if not recipe.get("fixture"):
        from .data import resolve_data_spec
        data = resolve_data_spec(data, mode)
    if any(k in data for k in ("url", "api_key", "token", "executor", "command")):
        raise ValueError("Data requests cannot contain credentials, URLs or executors")
    for raw in data.get("raw_paths", []):
        p = safe_path(raw, settings.roots, exists=True)
        fingerprints[f"raw:{len(fingerprints)}"] = artifact(p, "raw_data")
    if recipe.get("fixture"):
        stages = ["smoke", "evaluate", "report"]
    elif mode == "evaluate":
        stages = ["evaluate", "report"]
    else:
        stages = ["sources"]
        if "baseline" not in inputs:
            stages.append("assets")
        if "prepared" not in inputs and "baseline" not in inputs:
            if not data.get("raw_paths") and "raw_manifest" not in inputs:
                stages.append("acquire")
            stages.append("prepare")
        if "baseline" not in inputs:
            stages.append("rollout")
        if head != "none" and mode in {"reproduce", "warm-start", "resume"}:
            stages.append("train")
        stages.extend(["refine", "evaluate", "report"])
    if request.stages is not None:
        if len(request.stages) != len(set(request.stages)) or not request.stages:
            raise ValueError("Stage list must be nonempty and unique")
        stages = list(request.stages)
    if mode in {"replay", "evaluate"} and any(s in stages for s in ("train", "smoke")):
        raise ValueError("Replay/evaluate mode cannot execute training; select a new experiment explicitly")
    if "smoke" in stages and not recipe.get("fixture"):
        raise ValueError("Smoke stage is restricted to the synthetic fixture recipe")
    blockers, warnings = [], list(valid["warnings"])
    if mode in {"replay", "resume", "warm-start"} and head != "none" and "refinement" not in inputs:
        blockers.append("requires_artifact: supplied trained refinement checkpoint; no fallback to training")
    if mode == "evaluate" and not any(k in inputs for k in ("forecast", "refined", "baseline")):
        blockers.append("requires_artifact: forecast archive")
    acquisition = None
    if "acquire" in stages:
        from .data import plan_cams
        acquisition = plan_cams(data, limits)
    if "assets" in stages:
        lock = read_json(REPO / "provenance/source-lock.json")["huggingface"]
        total = sum(v["bytes"] for k,v in lock["assets"].items()
                    if k not in inputs and not (settings.cache_root/"official-assets"/v["filename"]).is_file())
        if total > limits["max_download_bytes"]:
            blockers.append(f"Asset download needs {total} bytes, exceeding max_download_bytes")
    else:
        total = 0
    if acquisition and acquisition["estimated_uncompressed_bytes"] + total > limits["max_disk_bytes"]:
        warnings.append("Estimated acquisition plus assets exceeds disk bound; execution stops at the approved bound. Review capacity.")
    plan = {"schema_version": 1, "recipe": recipe["id"], "recipe_version": recipe["version"],
            "fixture": bool(recipe.get("fixture")), "config": cfg, "configuration_hash": digest(cfg),
            "head": head, "mode": mode, "product": request.product,
            "ensemble_members": request.ensemble_members, "seed": request.seed,
            "limits": limits, "roots": {"data": str(settings.data_root), "cache": str(settings.cache_root),
                                          "output": str(settings.output_root)},
            "inputs": inputs, "input_fingerprints": fingerprints,
            "asset_registry": settings.assets, "data": data, "stages": stages,
            "evaluation": request.evaluation.model_dump(),
            "allowed_read_roots": [str(p) for p in settings.read_roots],
            "reserved_asset_download_bytes": total,
            "source": source_fingerprint(), "settings_hash": digest(settings.model_dump(mode="json")),
            "environment": environment_fingerprint(),
            "warnings": warnings, "blockers": blockers, "acquisition": acquisition,
            "resources": {"wall_time_is_upper_bound": True, "limits": limits,
                          "currency_cost": None, "cost_note": "Local compute/storage and electricity cost unknown; operator review required.",
                          "gpu_peak_memory_gb": None, "dependencies": stages},
            "case_id": request.case_id}
    plan["plan_id"] = digest(plan)
    plan["run_id"] = request.case_id + "-" + head + "-" + plan["plan_id"][:12]
    plan["state"] = "blocked" if blockers else "planned"
    return plan


def verify_plan(plan: dict, settings: Settings) -> None:
    payload = {k:v for k,v in plan.items() if k not in {"plan_id", "run_id", "state"}}
    if digest(payload) != plan["plan_id"]:
        raise ValueError("Plan integrity failure")
    if digest(settings.model_dump(mode="json")) != plan["settings_hash"]:
        raise ValueError("Local configuration changed; replan and approve")
    current = source_fingerprint()
    if any(current[k] != plan["source"][k] for k in ("commit", "code_hash")):
        raise ValueError("Implementation changed; replan and approve")
    if environment_fingerprint() != plan["environment"]:
        raise ValueError("Execution environment changed; replan and approve")
    for record in plan["input_fingerprints"].values():
        p = safe_path(record["path"], settings.roots, exists=True)
        if sha256(p) != record["sha256"]:
            raise ValueError("Input artifact changed after planning")
