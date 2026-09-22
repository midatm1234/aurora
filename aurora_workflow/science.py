"""Portable adapters to the pinned fork's actual Aurora and refinement classes.

Heavy dependencies are imported only when a scientific stage runs. Numerical
arrays and training state use NPZ with an embedded JSON tree, never pickle.
The autonomous baseline is materialized before any refinement is constructed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "88f652f04aa75b65e410b8d00cf4ea07f2edd945"
HEADS = {
    "none": "Unchanged Aurora baseline",
    "flow_matching_unet": "Flow matching UNet (legacy numerical adapter)",
    "flow_matching_transformer": "Flow matching Transformer",
    "diffusion_unet": "Diffusion UNet",
    "diffusion_transformer": "Diffusion Transformer",
    "flow_matching_conv_unet": "Flow matching convolutional UNet (unified objective)",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _npz(path: Path, **arrays: Any) -> None:
    import numpy as np
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def _input(plan: dict, name: str, run_dir: Path, default: str | None = None) -> Path:
    value = plan.get("inputs", {}).get(name)
    if isinstance(value, dict):
        value = value.get("path")
    path = Path(value).expanduser().resolve() if value else run_dir / (default or name)
    # Manifest paths are checked too: passing an approved manifest cannot grant
    # a second, unrestricted filesystem capability.
    roots = [Path(p).expanduser().resolve() for p in plan.get("roots", {}).values()]
    roots.extend(Path(p).expanduser().resolve() for p in plan.get("allowed_read_roots", []))
    roots.append(run_dir.resolve())
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"{name} path is outside approved roots")
    if not path.is_file():
        raise FileNotFoundError(f"requires_artifact: {name}: {path}")
    return path


def effective_config(plan: dict) -> dict:
    """Resolve exact per-head settings before approval hashes are computed."""
    recipe = plan.get("recipe", {})
    if isinstance(recipe, str):
        if "/" in recipe or "\\" in recipe or recipe not in {
            "cpu-smoke-v1", "real-gpu-v1", "no2-us-west-v1", "o3-global-v1",
            "no2-historical-forecast-forced-v1", "o3-historical-forecast-forced-v1"
        }:
            raise ValueError("Unknown versioned recipe")
        recipe = json.loads((Path(__file__).parents[1] / "recipes" / f"{recipe}.json").read_text())
    cfg = copy.deepcopy(plan.get("config") or recipe.get("config", {}))
    head = plan.get("head", cfg.get("model", {}).get("refinement", {}).get("type", "none"))
    if head not in HEADS:
        raise ValueError(f"Unknown head {head!r}; use one of {list(HEADS)}")
    if head != cfg.get("model", {}).get("refinement", {}).get("type"):
        if head in recipe.get("head_configs", {}):
            cfg.setdefault("model", {})["refinement"] = copy.deepcopy(recipe["head_configs"][head])
        else:
            cfg.setdefault("model", {}).setdefault("refinement", {})["type"] = head
    cfg["model"]["refinement"]["enabled"] = head != "none"
    cfg["model"]["refinement"]["type"] = head
    return cfg


def validate_scientific_config(cfg: dict) -> dict:
    """Lightweight scientific checks; SDK discovery never imports torch."""
    model = cfg.get("model", {})
    ref = model.get("refinement", {})
    if model.get("model_variant") != "aurora_air_pollution":
        raise ValueError("Workflow requires AuroraAirPollution")
    if ref.get("type", "none") not in HEADS:
        raise ValueError("Unsupported refinement head")
    if ref.get("feedback_to_rollout", False):
        raise ValueError("Canonical workflow requires feedback_to_rollout=false")
    if not ref.get("freeze_aurora", True) or ref.get("joint_finetuning", False):
        raise ValueError("Phase 1 backbone must remain frozen")
    historical = cfg.get("workflow", {}).get("rollout_mode") == "historical_forecast_forced"
    if not historical and cfg.get("rollout", {}).get("keep_exogenous_predictors", "fixed") != "fixed":
        raise ValueError("Autonomous workflow rejects future CAMS exogenous refresh")
    if not cfg.get("rollout", {}).get("autoregressive_inputs", True):
        raise ValueError("Autonomous workflow rejects teacher forcing")
    if cfg.get("data", {}).get("input_time_steps") != 2:
        raise ValueError("Aurora air-pollution checkpoint requires two history times")
    if cfg.get("rollout", {}).get("rollout_step_hours") != 12:
        raise ValueError("Aurora air-pollution checkpoint cadence is 12 hours")
    if model.get("patch_size") != 3:
        raise ValueError("AuroraAirPollution patch size must be 3")
    unified_active = ref.get("enabled", True) and ref.get("type", "none") not in {"none", "flow_matching_unet"}
    temporal_enabled = model.get("mamba_temporal", {}).get(
        "enabled", model.get("mamba_temporal_enabled", unified_active)
    )
    if temporal_enabled or model.get("mamba_temporal_enabled", False):
        raise ValueError("Portable cached workflow currently supports spatial heads; use the preserved sequence-aware Mamba runner for a separately identified temporal experiment")
    if ref.get("temporal", {}).get("backend", "none") != "none":
        raise ValueError("Portable cached workflow does not supply temporal context; use the experimental sequence API")
    if any(ref.get("conditioning", {}).get(key, False) for key in ("calendar", "solar_geometry")):
        raise ValueError("Portable cached workflow does not supply calendar/solar context; use the experimental conditioning API")
    return {"valid": True, "model_family": "AuroraAirPollution", "head": ref["type"],
            "feedback_to_rollout": False, "input_provenance": "explicit initialization-issued forecast exogenous fields" if historical else "two past analysis times; autonomous thereafter"}


def _packing(data: dict):
    from finetune.refinement.packing import ChannelSpec, FieldPacking
    return FieldPacking(channels=tuple(ChannelSpec(**c) for c in data["channels"]),
                        lat=tuple(data["lat"]), lon=tuple(data["lon"]),
                        lead_times_hours=tuple(data["lead_times_hours"]),
                        lead_time_scale_hours=data["lead_time_scale_hours"],
                        lon_periodic=data["lon_periodic"])


def checkpoint_contract(cfg: dict, packing: Any) -> dict:
    from finetune.refinement.config import resolve_refinement_config
    from finetune.refinement.two_phase import resolve_temporal_config
    ref = resolve_refinement_config(cfg).to_dict()
    # Point and ensemble are separately requested products from identical
    # weights. Process, architecture and temporal contracts remain immutable.
    for key in ("ensemble_size", "deterministic_inference", "seed", "checkpoint"):
        ref.pop(key, None)
    return {"schema_version": 1, "model_family": "AuroraAirPollution",
            "source_commit": SOURCE_COMMIT, "refinement": ref,
            "packing": packing.to_dict(), "temporal": resolve_temporal_config(cfg),
            "backbone_levels": cfg["data"]["atmos_levels"],
            "predictors": cfg["data"]["predictor_variables"],
            "backbone_predictors": cfg.get("workflow", {}).get("backbone_predictor_variables"),
            "static_fields": cfg["data"]["static_variables"],
            "patch_size": cfg["model"]["patch_size"],
            "padding": cfg["data"].get("patch_alignment_strategy", "crop"),
            "nonnegative_variables": cfg["data"].get("nonnegative_target_variables", []),
            "rollout_mode": cfg.get("workflow", {}).get("rollout_mode", "autonomous_full_backbone"),
            "history_times": 2, "cadence_hours": 12}


def save_training_state(path: Path, payload: dict) -> str:
    """Lossless tensors + simple Python containers, including optimizer/RNG."""
    import numpy as np
    import torch
    arrays: dict[str, Any] = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            name = f"tensor_{len(arrays)}"
            tensor = value.detach().cpu()
            # BF16 has no NumPy dtype; retain bits and explicit tensor dtype.
            arrays[name] = tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
            return {"tensor": name, "dtype": str(tensor.dtype)}
        if isinstance(value, dict):
            return {"dict": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, tuple):
            return {"tuple": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise TypeError(f"Unsupported training-state value: {type(value).__name__}")

    metadata = encode(payload)
    arrays["__metadata__"] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    _npz(path, **arrays)
    return _hash(path)


def _comparable_checkpoint_contract(contract: dict) -> dict:
    """Hydrate only legacy-safe, inactive options added after schema v1.

    The reference branch predates optional calendar/vertical/context features.
    Their disabled defaults add no tensors or inputs. Active configurations,
    including feature order and every existing scientific field, remain strict.
    """
    result = copy.deepcopy(contract)
    refinement = result.get("refinement")
    if isinstance(refinement, dict):
        conditioning = refinement.get("conditioning")
        if isinstance(conditioning, dict):
            for key in ("calendar", "solar_geometry", "vertical_identity"):
                conditioning.setdefault(key, False)
        context = refinement.get("temporal", {"backend": "none"})
        if isinstance(context, dict) and context.get("backend", "none") == "none":
            refinement["temporal"] = {"backend": "none"}
    temporal = result.get("temporal")
    if isinstance(temporal, dict) and temporal.get("enabled") is False:
        result["temporal"] = {"enabled": False}
    return result


def load_training_state(path: Path, expected: dict, *, model=None, cfg=None) -> dict:
    import numpy as np
    import torch
    if path.suffix != ".npz":
        # Historical checkpoints often contain NumPy RNG pickle globals. Never
        # retry with weights_only=False. A trusted owner can explicitly export
        # those artifacts; incompatibility is not a request to retrain.
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(raw, dict) and isinstance(raw.get("contract"), dict) and _comparable_checkpoint_contract(raw["contract"]) == _comparable_checkpoint_contract(expected):
            payload = raw
        elif isinstance(raw, dict) and model is not None and cfg is not None:
            from finetune.model_factory import validate_unified_checkpoint_contract
            from finetune.model_factory import validate_loaded_residual_scaler_contract
            # Run the SAME strict architecture/normalization/temporal/lead
            # validator as notebook inference, before extracting any tensors.
            validate_unified_checkpoint_contract(model, raw, cfg)
            old_cfg = raw["config"]
            old_rollout = old_cfg.get("rollout", {})
            new_rollout = cfg.get("rollout", {})
            for key in ("keep_exogenous_predictors", "predicted_fields_get_fed_back", "autoregressive_inputs"):
                if old_rollout.get(key) != new_rollout.get(key):
                    raise ValueError(f"Historical checkpoint rollout contract mismatch: {key}")
            if cfg.get("workflow", {}).get("rollout_mode") != "historical_forecast_forced":
                raise ValueError("Historical combined checkpoint requires an explicitly identified historical rollout recipe")
            state = raw.get("model_state_dict", {})
            head_state = {k: v for k, v in state.items() if not k.startswith("aurora.")}
            model.load_state_dict(head_state, strict=True)
            validate_loaded_residual_scaler_contract(model, raw)
            payload = {"contract":expected,"state_dict":head_state,
                       "validated_for_inference":raw.get("validated_for_inference",False),
                       "validation":raw.get("validation"),"historical_import":True}
        else:
            raise ValueError("requires_artifact: checkpoint needs verified compatible architecture metadata")
    else:
        with np.load(path, allow_pickle=False) as archive:
            if archive["__metadata__"].nbytes > 16000000:
                raise ValueError("Checkpoint metadata exceeds bound")
            tree = json.loads(archive["__metadata__"].tobytes())

            def decode(value):
                if isinstance(value, list):
                    return [decode(v) for v in value]
                if not isinstance(value, dict):
                    return value
                if set(value) == {"tensor", "dtype"}:
                    a = archive[value["tensor"]]
                    if a.dtype.hasobject:
                        raise ValueError("Object arrays are forbidden")
                    t = torch.from_numpy(a.copy())
                    return t.view(torch.bfloat16) if value["dtype"] == "torch.bfloat16" else t
                if set(value) == {"dict"}:
                    return {decode(k): decode(v) for k, v in value["dict"]}
                if set(value) == {"tuple"}:
                    return tuple(decode(v) for v in value["tuple"])
                raise ValueError("Invalid checkpoint container")
            # Inspect plain JSON compatibility metadata before allocating the
            # first saved parameter or optimizer tensor.
            if not isinstance(tree,dict) or set(tree)!={"dict"}:
                raise ValueError("Checkpoint root must be a mapping")
            contract_tree=next((v for k,v in tree["dict"] if k=="contract"),None)
            actual_contract=decode(contract_tree)
            if not isinstance(actual_contract, dict) or _comparable_checkpoint_contract(actual_contract)!=_comparable_checkpoint_contract(expected):
                raise ValueError("Checkpoint incompatibility before tensor loading")
            payload = decode(tree)
    if not isinstance(payload.get("contract"), dict) or _comparable_checkpoint_contract(payload["contract"]) != _comparable_checkpoint_contract(expected):
        actual = payload.get("contract", {})
        differences = [k for k in expected if actual.get(k) != expected[k]]
        raise ValueError(f"Checkpoint incompatibility before state loading: {differences}")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint missing state_dict")
    return payload


def autonomous_rollout(model, batch, steps: int):
    """Extracted notebook/ft state-advance API; no dataset or targets accepted."""
    import torch
    from finetune.aurora_finetune_utils import _advance_batch_with_prediction
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    current = batch
    with torch.inference_mode():
        for _ in range(steps):
            prediction = model(current)
            yield prediction
            current = _advance_batch_with_prediction(current, prediction)


def _fixture(plan: dict, cfg: dict, run_dir: Path) -> dict:
    """Synthetic physical fields; actual head training happens in _train."""
    import numpy as np
    from types import SimpleNamespace
    from finetune.refinement.packing import FieldPacking
    from finetune.aurora_finetune_utils import compute_target_normalization_stats
    specs = SimpleNamespace(targets=[SimpleNamespace(**v) for v in cfg["data"]["target_variables"]])
    for spec in specs.targets:
        if not hasattr(spec, "loss_levels"):
            spec.loss_levels = None
    stats = compute_target_normalization_stats(None, specs, cfg)
    packing = FieldPacking.from_specs(specs.targets, norm_stats=stats,
        atmos_levels=cfg["data"]["atmos_levels"], lat=list(np.linspace(50, 32, 12)),
        lon=list(np.linspace(234, 258, 12)), lead_times_hours=[12, 24])
    rng = np.random.default_rng(plan.get("seed", 42))
    values = rng.uniform(.1, .9, (3, 2, packing.num_channels, 12, 12)).astype("float32")
    scale = np.asarray([c.std for c in packing.channels], dtype="float32")[None,None,:,None,None]
    baseline = values * scale
    truth = (values + .08 * np.sin(np.linspace(0, np.pi, 12))[None,None,None,:,None]).astype("float32") * scale
    coordinates = dict(case_cycle=np.array(["2024-01-01T00:00:00", "2024-01-06T00:00:00", "2024-01-11T00:00:00"]),
        lead_hours=np.array([12,24]), lat=np.array(packing.lat), lon=np.array(packing.lon),
        channel=np.array([c.aurora_name + (f"_{c.level:g}" if c.level else "") for c in packing.channels]),
        units=np.array([c.units for c in packing.channels]), split=np.array(["train","val","test"]))
    _npz(run_dir/"baseline.npz", fields=baseline, mask=np.ones_like(baseline,dtype=bool), **coordinates)
    _npz(run_dir/"reference.npz", fields=truth, mask=np.ones_like(truth,dtype=bool), **coordinates)
    _json(run_dir/"packing.json", packing.to_dict())
    _json(run_dir/"rollout-provenance.json", {"execution_status":"executed_synthetic_fixture", "actual_aurora_backbone_executed":False,
        "feedback_to_rollout":False,"source_commit":SOURCE_COMMIT,"normalization":"fixed Aurora constants"})
    return {"baseline":str(run_dir/"baseline.npz"),"reference":str(run_dir/"reference.npz"),"packing":str(run_dir/"packing.json")}


def _real_rollout(plan: dict, cfg: dict, run_dir: Path) -> dict:
    import numpy as np
    import torch
    import xarray as xr
    from aurora import AuroraAirPollution
    from finetune import aurora_finetune_utils as ft
    from finetune.refinement.integration import build_field_packing
    from finetune.longitude import longitude_is_periodic
    from inspect import signature
    device = plan.get("limits", {}).get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BLOCKED: CUDA-enabled PyTorch and local GPU required")
    prepared = _input(plan, "prepared", run_dir, "prepared.json")
    manifest = json.loads(prepared.read_text())
    nested = copy.deepcopy(plan)
    nested.setdefault("inputs", {}).update(data_path=manifest["data_path"], reference_path=manifest["reference_path"])
    data_path = _input(nested, "data_path", run_dir)
    reference_path = _input(nested, "reference_path", run_dir)
    checkpoint = _input(plan,"pretrained",run_dir)
    static_path = _input(plan,"static",run_dir)
    if static_path.suffix != ".npz":
        raise ValueError("Static data must be a verified numerical NPZ conversion; pickle loading is prohibited")
    nested["inputs"]["static_metadata"]=str(static_path.with_suffix(".json"))
    static_metadata=_input(nested,"static_metadata",run_dir)
    asset_identity=_verify_phase1_assets(checkpoint,static_path,static_metadata)
    fingerprints=manifest.get("sha256",{})
    if isinstance(fingerprints,str):fingerprints={"data_path":fingerprints}
    for role,path in (("data_path",data_path),("reference_path",reference_path)):
        if fingerprints.get(role) and _hash(path)!=fingerprints[role]:
            raise ValueError(f"Prepared {role} fingerprint mismatch")
    cases = manifest["cases"]
    if not cases or len(cases)>int(plan.get("limits",{}).get("max_cases",1000)):
        raise ValueError("Prepared cases exceed approved bound or are empty")
    # Keep model default complete variable set: the smaller NO2 predictor list
    # remains refinement metadata, not a reason to discard pretrained weights.
    defaults = signature(AuroraAirPollution.__init__).parameters
    full_cfg = copy.deepcopy(cfg)
    mapping = {v["aurora_name"]:v for v in cfg["workflow"]["backbone_predictor_variables"]}
    historical=cfg.get("workflow",{}).get("rollout_mode")=="historical_forecast_forced"
    if not historical:
        full_cfg["data"]["predictor_variables"] = [mapping[name] for name in defaults["surf_vars"].default+defaults["atmos_vars"].default]
        full_cfg["data"]["domain_type"]="global"
    model=None
    outputs=[]; references=[]; masks=[]; audit=[]; packing=None
    with xr.open_dataset(data_path) as data, xr.open_dataset(reference_path) as reference, np.load(static_path,allow_pickle=False) as static:
        data=ft._ensure_monotonic_lat_lon(data,"latitude","longitude")
        data=ft._subset_domain(data,full_cfg,"latitude","longitude")
        for item in cfg["data"]["static_variables"]:
            key=item["aurora_name"]
            if key not in static:
                raise ValueError(f"Static NPZ missing {key}")
            lat_key="lat" if "lat" in static else "latitude"
            lon_key="lon" if "lon" in static else "longitude"
            field=xr.DataArray(static[key],dims=("latitude","longitude"),coords={"latitude":static[lat_key],"longitude":static[lon_key]})
            field=field.assign_coords(longitude=field.longitude%360).sortby("longitude")
            # Named-coordinate intersection, no implicit resizing/interpolation.
            aligned=field.sel(latitude=data.latitude,longitude=data.longitude,method="nearest",tolerance=2e-5)
            data[item["dataset_name"]]=aligned.assign_coords(latitude=data.latitude,longitude=data.longitude)
        specs=ft.resolve_variable_specs(data,full_cfg)
        if historical:
            if manifest.get("source",{}).get("reference_kind")!="forecast":
                raise ValueError("Historical forecast-forced mode requires explicit forecast-cycle reference; future analyses are not operational inputs")
            base_cfg=copy.deepcopy(full_cfg)
            base_cfg["model"]["refinement"]={"type":"none","enabled":False}
            base_cfg["model"]["flow_refine_enabled"]=False
            base_cfg["model"]["conv_refine_enabled"]=False
            model=ft.build_finetune_model(base_cfg,specs,load_pretrained=False,autocast=str(device).startswith("cuda"))
            state=model._adapt_checkpoint(torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True))
            expected_keys=set(model.state_dict())
            if expected_keys-set(state):
                raise ValueError("Official checkpoint missing required historical backbone weights; random initialization refused")
            omitted_keys=sorted(set(state)-expected_keys)
            model.load_state_dict({key:state[key] for key in expected_keys},strict=True)
        else:
            model=AuroraAirPollution(autocast=str(device).startswith("cuda"))
            model.load_checkpoint_local(str(checkpoint),strict=True)
            omitted_keys=[]
        model.to(device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        stats=ft.compute_target_normalization_stats(data,specs,full_cfg)
        for case in cases:
            cycle=np.datetime64(case["cycle"].replace("Z",""),"ns")
            expected=[cycle-np.timedelta64(12,"h"),cycle]
            history=data.sel(time=expected)
            if list(history.time.values)!=expected:
                raise ValueError("History times are not exact t-12h,t")
            sample={"anchor_index":1,"history_indices":[0,1],"target_indices":{}}
            batch=ft.build_aurora_batch(history,sample,full_cfg,specs).to(device)
            if packing is None:
                grid=xr.Dataset(coords={"latitude":batch.metadata.lat.cpu().numpy(),"longitude":batch.metadata.lon.cpu().numpy()})
                grid=ft._subset_domain(grid,cfg,"latitude","longitude")
                packing=build_field_packing(cfg,specs,norm_stats=stats,lat=grid.latitude.values.tolist(),lon=grid.longitude.values.tolist(),lon_periodic=longitude_is_periodic(grid.longitude.values))
            packed_predictions=[]; packed_targets=[]
            if historical:
                future=reference.sel(forecast_reference_time=cycle).copy()
                lead_values=future.lead_time.values
                valid_times=cycle+(lead_values if np.issubdtype(lead_values.dtype,np.timedelta64) else lead_values.astype("timedelta64[h]"))
                future=future.assign_coords(time=("lead_time",valid_times)).swap_dims({"lead_time":"time"})
                future=ft._ensure_monotonic_lat_lon(future,"latitude","longitude")
                future=ft._subset_domain(future,full_cfg,"latitude","longitude")
                future=future.sel(time=[cycle+np.timedelta64(int(h),"h") for h in case["lead_hours"]])
                for item in cfg["data"]["static_variables"]:future[item["dataset_name"]]=history[item["dataset_name"]]
                forced=xr.concat([history,future],dim="time",data_vars="minimal",coords="minimal",compat="override")
                trajectory=ft.run_rollout(model,forced,sample,full_cfg,specs,device)
            else:
                trajectory=autonomous_rollout(model,batch,len(case["lead_hours"]))
            for lead_index,prediction in enumerate(trajectory):
                lead=case["lead_hours"][lead_index]
                if lead!=(lead_index+1)*12:
                    raise ValueError("Reference leads must be contiguous 12-hour steps")
                fields={}; target_fields={}
                reference_case=reference.sel(forecast_reference_time=cycle)
                lead_dtype=reference_case.lead_time.dtype
                selector=np.timedelta64(int(lead),"h") if np.issubdtype(lead_dtype,np.timedelta64) else lead
                reference_case=reference_case.sel(lead_time=selector)
                reference_case=reference_case.assign_coords(longitude=reference_case.longitude%360).sortby("longitude")
                # Autonomous backbone remains global; crop only the packed
                # refinement products, preserving the global baseline state.
                pred_lat=prediction.metadata.lat.cpu().numpy()
                pred_lon=prediction.metadata.lon.cpu().numpy()
                lat_indices=[int(np.argmin(abs(pred_lat-value))) for value in packing.lat]
                lon_indices=[int(np.argmin(abs(pred_lon-value))) for value in packing.lon]
                if any(abs(pred_lat[i]-v)>2e-5 for i,v in zip(lat_indices,packing.lat)) or any(abs(pred_lon[i]-v)>2e-5 for i,v in zip(lon_indices,packing.lon)):
                    raise ValueError("Prediction/refinement coordinate mismatch")
                for name in packing.variables:
                    channels=packing.channels_for(name); c=channels[0]
                    target=reference_case[c.dataset_name]
                    if c.kind=="surf":
                        fields[name]=prediction.surf_vars[name][:,0].cpu().float()
                    else:
                        levels=list(prediction.metadata.atmos_levels)
                        indices=[levels.index(float(s.level)) for s in channels]
                        fields[name]=prediction.atmos_vars[name][:,0,indices].cpu().float()
                        target=target.sel(level=[s.level for s in channels])
                    fields[name]=fields[name][...,lat_indices,:][...,lon_indices]
                    target=target.sel(latitude=list(packing.lat),longitude=list(packing.lon),method="nearest",tolerance=2e-5)
                    dims=("latitude","longitude") if c.kind=="surf" else ("level","latitude","longitude")
                    target_fields[name]=torch.from_numpy(np.asarray(target.transpose(*dims).values,dtype="float32").copy()).unsqueeze(0)
                packed_predictions.append(packing.pack(fields)[0].numpy())
                packed_targets.append(packing.pack(target_fields)[0].numpy())
            outputs.append(np.stack(packed_predictions)); references.append(np.stack(packed_targets))
            audit.append({"cycle":case["cycle"],"inputs":[str(t) for t in expected],"future_dataset_reads":len(case["lead_hours"]) if historical else 0,"advance":"target predictions plus forecast exogenous context" if historical else "all baseline output fields","exogenous_available_at":case["cycle"] if historical else None,"exogenous_variables":[v["aurora_name"] for v in cfg["data"]["predictor_variables"] if v["aurora_name"] not in {t["aurora_name"] for t in cfg["data"]["target_variables"]}] if historical else [],"feedback_to_rollout":False})
    assert packing is not None
    baseline=np.stack(outputs); truth=np.stack(references)
    coords=dict(case_cycle=np.array([c["cycle"] for c in cases]),lead_hours=np.array(cases[0]["lead_hours"]),lat=np.array(packing.lat),lon=np.array(packing.lon),channel=np.array([c.aurora_name+(f"_{c.level:g}" if c.level else "") for c in packing.channels]),units=np.array([c.units for c in packing.channels]),split=np.array([c["split"] for c in cases]))
    _npz(run_dir/"baseline.npz",fields=baseline,mask=np.isfinite(baseline),**coords)
    _npz(run_dir/"reference.npz",fields=truth,mask=np.isfinite(truth),**coords)
    _json(run_dir/"packing.json",packing.to_dict())
    _json(run_dir/"rollout-provenance.json",{"execution_status":"executed_real_aurora",**asset_identity,"prepared_sha256":_hash(prepared),"source_commit":SOURCE_COMMIT,"cases":audit,"frozen_backbone":True,"omitted_unconfigured_checkpoint_keys":omitted_keys,"rollout_mode":cfg["workflow"]["rollout_mode"]})
    return {"baseline":str(run_dir/"baseline.npz"),"reference":str(run_dir/"reference.npz"),"packing":str(run_dir/"packing.json")}


def _verify_phase1_assets(checkpoint: Path,static_path: Path,static_metadata: Path) -> dict:
    """Defense in depth for direct rollout-stage invocation after approval."""
    lock=json.loads((Path(__file__).parents[1]/"provenance/source-lock.json").read_text())["huggingface"]
    checkpoint_hash=_hash(checkpoint);static_hash=_hash(static_path)
    if checkpoint_hash!=lock["assets"]["pretrained"]["sha256"]:
        raise ValueError("Pretrained artifact is not the verified official air-pollution checkpoint")
    metadata=json.loads(static_metadata.read_text())
    if metadata.get("source_sha256")!=lock["assets"]["static"]["sha256"] or metadata.get("sha256")!=static_hash:
        raise ValueError("Static conversion metadata/hash does not match pinned official source")
    return {"checkpoint_sha256":checkpoint_hash,"static_sha256":static_hash,"static_source_sha256":metadata["source_sha256"],"huggingface_revision":lock["revision"]}


def _load_fields(plan: dict, run_dir: Path, *, need_reference: bool):
    import numpy as np
    import torch
    baseline_path=_input(plan,"baseline",run_dir,"baseline.npz")
    packing_path=_input(plan,"packing",run_dir,"packing.json")
    packing=_packing(json.loads(packing_path.read_text()))
    with np.load(baseline_path,allow_pickle=False) as archive:
        data={k:archive[k].copy() for k in archive.files}
    if data["fields"].ndim!=5 or data["fields"].shape[2]!=packing.num_channels:
        raise ValueError("Baseline must be [case,lead,channel,latitude,longitude]")
    if not np.array_equal(data["lat"],packing.lat) or not np.array_equal(data["lon"],packing.lon):
        raise ValueError("Baseline grid does not match checkpoint packing")
    if not np.array_equal(data["lead_hours"],packing.lead_times_hours):
        raise ValueError("Baseline leads exceed or differ from trained lead support")
    target=None
    if need_reference:
        with np.load(_input(plan,"reference",run_dir,"reference.npz"),allow_pickle=False) as archive:
            for key in ("case_cycle","lead_hours","lat","lon","channel","units","split"):
                if not np.array_equal(data[key],archive[key]):
                    raise ValueError(f"Baseline/reference {key} mismatch")
            target=torch.from_numpy(archive["fields"].copy())
    fields=torch.from_numpy(data["fields"])
    shape=fields.shape
    return data,packing,fields.flatten(0,1),None if target is None else target.flatten(0,1),shape


def _build_phase2(cfg: dict, packing, device: str):
    from finetune.refinement.two_phase import build_two_phase_refiner
    model=build_two_phase_refiner(None,packing,cfg,nonnegative_variables=cfg["data"].get("nonnegative_target_variables",()))
    if model.refinement_config.is_active:
        model.initialize_refiner(model.conditioning_channels())
    return model.to(device)


def _infer(model, baseline, leads, product: str, members: int, seed: int):
    import torch
    model.eval()
    with torch.inference_mode():
        normalized=model.target_space.encode(baseline)
        output=model.refine(normalized,forecast_lead_time=leads,ensemble_size=members,seed=seed)
    return output


def _selection(model, baseline, target, leads, cfg):
    """Use the pinned physical metrics and exact existing no-harm guard API."""
    from finetune.refinement.evaluation import evaluate_packed
    from finetune.aurora_finetune_distributed import _evaluate_checkpoint_guards
    output=_infer(model,baseline,leads,"point",1,0)
    refined=output.deterministic_refined_physical
    common=dict(packing=model.packing,lead_hours=leads,lead_index=(leads/12-1).long(),area_weighted=True)
    before=evaluate_packed(baseline,target,**common)
    after=evaluate_packed(refined,target,**common)
    comparisons={str(i):{"baseline":b,"refined":r} for i,(b,r) in enumerate(zip(before,after))}
    settings=cfg["training"]
    guards=_evaluate_checkpoint_guards(physical_channels=comparisons,expected_channels=list(comparisons),metrics=settings.get("checkpoint_guard_metrics",[]),relative_tolerance=settings.get("checkpoint_guard_relative_tolerance",0.),correlation_tolerance=settings.get("checkpoint_guard_correlation_tolerance",0.),bias_rmse_floor_fraction=settings.get("checkpoint_guard_bias_rmse_floor_fraction",.05))
    ratios=[r["rmse"]/b["rmse"] if b["rmse"]>0 else (1. if r["rmse"]==0 else float("inf")) for b,r in zip(before,after)]
    score=sum(ratios)/len(ratios)
    accepted=bool(guards["passed"] and math.isfinite(score))
    if settings.get("require_all_physical_channels_improve",False):
        accepted &= all(r<1 for r in ratios)
    if settings.get("require_refinement_improvement",False):
        accepted &= score<1
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        return value
    return finite({"score":score if math.isfinite(score) else None,"validated_for_inference":accepted,"guards":guards,"split":"val","channels":comparisons})


def _train(plan: dict, cfg: dict, run_dir: Path) -> dict:
    import numpy as np
    import torch
    from finetune import aurora_finetune_utils as ft
    from finetune.refinement.residual_scaling import ResidualScaler
    if cfg["model"]["refinement"]["type"]=="none":
        return {"skipped":"baseline selection has no trainable refinement"}
    mode=plan.get("mode","reproduce")
    if mode not in {"reproduce","resume","warm_start","warm-start"}:
        raise ValueError("Replay never falls back to training")
    data,packing,baseline,target,shape=_load_fields(plan,run_dir,need_reference=True)
    device=plan.get("limits",{}).get("device","cpu")
    seed=int(plan.get("seed",42)); torch.manual_seed(seed)
    model=_build_phase2(cfg,packing,device)
    baseline=baseline.to(device);target=target.to(device)
    lead=torch.tensor(data["lead_hours"],dtype=torch.float32,device=device).repeat(shape[0])
    split=np.repeat(data["split"],shape[1]); train=torch.tensor(np.flatnonzero(split=="train"),device=device); val=torch.tensor(np.flatnonzero(split=="val"),device=device)
    if not train.numel() or not val.numel():
        raise ValueError("Training requires explicitly separate train and val cases; test is never used for selection")
    contract=checkpoint_contract(cfg,packing)
    train_fingerprint=hashlib.sha256(baseline[train].cpu().numpy().tobytes()+target[train].cpu().numpy().tobytes()+json.dumps(data["case_cycle"][data["split"]=="train"].tolist()).encode()).hexdigest()
    optimizer=ft.create_optimizer(model,cfg)
    epochs=min(int(cfg["training"].get("num_epochs",1)),int(plan.get("limits",{}).get("max_epochs",1)))
    # Preserve one logical case containing all leads, as in the original trainer.
    case_batch=int(cfg["training"].get("batch_size",1));batch_size=case_batch*shape[1]
    accumulation=int(cfg["training"].get("accumulation_steps",1))
    updates_per_epoch=math.ceil(math.ceil(len(train)/batch_size)/accumulation)
    # Schedule is a scientific config, not a resource-limit side effect. An
    # approved one-epoch slice of a 100-epoch run keeps the 100-epoch schedule.
    scheduler=ft.create_scheduler(optimizer,cfg,int(cfg["training"].get("num_epochs",1))*updates_per_epoch)
    generator=torch.Generator(device=device).manual_seed(seed+17)
    start_epoch=0;start_chunk=0;step=0;best=None;history=[]
    last_path=run_dir/"last.npz";best_path=run_dir/"best.npz"
    recovery=last_path.exists() and mode=="reproduce"
    if mode in {"resume","warm_start","warm-start"} or recovery:
        saved=load_training_state(last_path if recovery else _input(plan,"refinement",run_dir),contract,model=model,cfg=cfg)
        if recovery and saved.get("plan_id")!=plan.get("plan_id"):
            raise ValueError("Checkpoint belongs to another plan; refusing overwrite")
        model.load_state_dict(saved["state_dict"],strict=True)
        if mode=="resume" or recovery:
            if saved.get("training_fingerprint")!=train_fingerprint or saved.get("training_config")!=cfg["training"]:
                raise ValueError("Resume requires unchanged training data/config; use separately identified warm-start")
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            generator.set_state(saved["generator_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            if str(device).startswith("cuda") and saved.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
            start_epoch=saved["epoch"];start_chunk=saved.get("next_chunk",0);step=saved["step"];best=saved.get("best_score");history=saved.get("history",[])
            if saved.get("validated_for_inference") and not best_path.exists():
                save_training_state(best_path,saved)
    else:
        # Fit only training corrections, using the original exact calibration.
        residual=model.target_space.encode(target[train])-model.target_space.encode(baseline[train])
        valid=torch.isfinite(residual)
        for module in model.modules():
            if isinstance(module,ResidualScaler) and module.is_active:
                module.begin_exact_training_split_calibration()
                module.observe(residual,valid)
                module.finalize_exact_training_split_calibration(logical_samples=int((data["split"]=="train").sum()),packed_examples=len(train),fingerprint=train_fingerprint)
    max_steps=int(plan.get("limits",{}).get("max_train_steps",100000))
    for epoch in range(start_epoch,epochs):
        if step>=max_steps:
            break
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        chunks=list(train.split(batch_size))
        for index,indices in enumerate(chunks):
            if epoch==start_epoch and index<start_chunk:continue
            result=model.training_step(model.target_space.encode(baseline[indices]),model.target_space.encode(target[indices]),forecast_lead_time=lead[indices],lead_index=(lead[indices]/12-1).long(),generator=generator)
            loss=result.losses["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite refinement training loss")
            actual_accumulation=min(accumulation,len(chunks)-(index//accumulation)*accumulation)
            (loss/actual_accumulation).backward();losses.append(float(loss.detach()))
            if (index+1)%accumulation==0 or index==len(chunks)-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"].get("gradient_clip_val",1.)))
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                if scheduler is not None:scheduler.step()
                if step>=max_steps:break
        selection=None
        if (epoch+1)%int(cfg["training"].get("validation_frequency",1))==0 or epoch+1==epochs:
            selection=_selection(model,baseline[val],target[val],lead[val],cfg)
        epoch_complete=index==len(chunks)-1
        history.append({"epoch":epoch+1,"epoch_complete":epoch_complete,"step":step,"training_loss":sum(losses)/len(losses),"validation":selection})
        promoted=bool(selection and selection["validated_for_inference"] and (best is None or selection["score"]<best))
        if promoted:best=selection["score"]
        payload={"contract":contract,"config":cfg,"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":None if scheduler is None else scheduler.state_dict(),"generator_state":generator.get_state(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],"epoch":epoch+1 if epoch_complete else epoch,"next_chunk":0 if epoch_complete else index+1,"step":step,"best_score":best,"training_config":cfg["training"],"training_fingerprint":train_fingerprint,"history":history,"validated_for_inference":bool(selection and selection["validated_for_inference"]),"validation":selection,"plan_id":plan.get("plan_id")}
        save_training_state(last_path,payload)
        if promoted:save_training_state(best_path,payload)
    if not last_path.exists():
        raise ValueError("Training has no remaining authorized epochs/steps")
    _json(run_dir/"training-history.json",history)
    return {"refinement":str(best_path if best_path.exists() else last_path),"last_checkpoint":str(last_path),"best_checkpoint":str(best_path) if best_path.exists() else None,"checkpoint_status":"validated" if best_path.exists() else "unvalidated_no_promotion","optimizer_steps":step,"history":str(run_dir/"training-history.json")}


def _refine(plan: dict, cfg: dict, run_dir: Path) -> dict:
    import numpy as np
    import torch
    data,packing,baseline,_,shape=_load_fields(plan,run_dir,need_reference=False)
    head=cfg["model"]["refinement"]["type"]
    product=plan.get("product","point")
    if product not in {"point","ensemble"}:
        raise ValueError("Product must be point or ensemble")
    members=int(plan.get("ensemble_members",plan.get("ensemble_size",2 if product=="ensemble" else 1)))
    if product=="ensemble" and (members<2 or members>int(plan.get("limits",{}).get("max_ensemble_members",20))):
        raise ValueError("Ensemble requires 2 or more members within approved bound")
    if head=="none":
        # Bitwise copy, including masks and physical units; no encode/decode.
        values=data["fields"].copy();ensemble=None;checkpoint_hash=None
    else:
        device=plan.get("limits",{}).get("device","cpu")
        inference_cfg=copy.deepcopy(cfg)
        inference_cfg["model"]["refinement"]["deterministic_inference"]=product=="point"
        inference_cfg["model"]["refinement"]["ensemble_size"]=members
        model=_build_phase2(inference_cfg,packing,device)
        checkpoint=_input(plan,"refinement",run_dir,"last.npz")
        saved=load_training_state(checkpoint,checkpoint_contract(cfg,packing),model=model,cfg=cfg)
        if cfg.get("inference",{}).get("require_validated_checkpoint",False) and not saved.get("validated_for_inference",False):
            raise ValueError("Checkpoint failed/has no independent validation gate; production inference not promoted")
        model.load_state_dict(saved["state_dict"],strict=True)
        baseline=baseline.to(device)
        leads=torch.tensor(data["lead_hours"],device=device,dtype=torch.float32).repeat(shape[0])
        output=_infer(model,baseline,leads,product,members,int(plan.get("seed",42)))
        values=output.deterministic_refined_physical.cpu().numpy().reshape(shape)
        ensemble=None
        if product=="ensemble":
            # Existing API [N,M,C,H,W] -> portable [M,case,lead,C,H,W].
            ensemble=output.members.cpu().numpy().reshape(shape[0],shape[1],members,*shape[2:]).transpose(2,0,1,3,4,5)
        checkpoint_hash=_hash(checkpoint)
    coords={k:v for k,v in data.items() if k not in {"fields","mask"}}
    _npz(run_dir/"refined.npz",fields=values,mask=np.isfinite(values),**coords,**({"ensemble":ensemble} if ensemble is not None else {}))
    _json(run_dir/"refinement-provenance.json",{"head":head,"product":product,"checkpoint_sha256":checkpoint_hash,"baseline_sha256":_hash(_input(plan,"baseline",run_dir,"baseline.npz")),"feedback_to_rollout":False,"future_targets_used":False,"point_product":"deterministic conditional mean","ensemble_size":members if ensemble is not None else 1})
    return {"refined":str(run_dir/"refined.npz"),"refinement_provenance":str(run_dir/"refinement-provenance.json")}


def execute_science(stage: str, plan: dict, run_dir: Path) -> dict:
    """Durable worker entry point shared by MCP and CLI via the workflow layer."""
    run_dir=Path(run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=True)
    cfg=effective_config(plan);validate_scientific_config(cfg)
    if stage not in {"rollout","train","refine","smoke"}:
        raise ValueError(f"Unknown science stage {stage}")
    import torch
    seed=int(plan.get("seed",42));torch.manual_seed(seed)
    if plan.get("limits",{}).get("device","cpu")=="cpu":
        torch.set_num_threads(min(int(plan.get("limits",{}).get("max_workers",1)),torch.get_num_threads()))
    recipe=plan.get("recipe",{})
    fixture=recipe=="cpu-smoke-v1" or (isinstance(recipe,dict) and recipe.get("fixture",False))
    if stage=="smoke" and not fixture:
        raise ValueError("Smoke stage requires explicitly synthetic recipe")
    local=copy.deepcopy(plan);local["config"]=cfg
    if stage=="rollout":
        artifacts=_fixture(local,cfg,run_dir) if fixture else _real_rollout(local,cfg,run_dir)
    elif stage=="train":artifacts=_train(local,cfg,run_dir)
    elif stage=="refine":artifacts=_refine(local,cfg,run_dir)
    else:
        artifacts=_fixture(local,cfg,run_dir);local.setdefault("inputs",{}).update(artifacts)
        trained=_train(local,cfg,run_dir);artifacts.update(trained);local["inputs"].update(trained)
        artifacts.update(_refine(local,cfg,run_dir))
    for name in ("rollout-provenance.json","packing.json","training-history.json","refinement-provenance.json"):
        if (run_dir/name).is_file():artifacts.setdefault(name.replace(".json", "").replace("-","_"),str(run_dir/name))
    files=[{"path":value,"sha256":_hash(Path(value)),"kind":key} for key,value in artifacts.items() if isinstance(value,str) and Path(value).is_file()]
    return {"stage":stage,"status":"succeeded","execution_status":"synthetic_actual_refinement" if fixture else "executed","artifacts":files,"outputs":{key:value for key,value in artifacts.items() if isinstance(value,str) and Path(value).is_file()},"details":{key:value for key,value in artifacts.items() if not isinstance(value,str) or not Path(value).is_file()},"warnings":["Synthetic fixture: Aurora backbone and real CAMS were not executed"] if fixture else [],"provenance":{"source_commit":SOURCE_COMMIT,"feedback_to_rollout":False,"model_family":"AuroraAirPollution"}}
