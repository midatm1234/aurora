"""CPU synthetic contract tests use real existing heads; no CAMS/model download."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from aurora_workflow.science import (
    HEADS, autonomous_rollout, checkpoint_contract, effective_config,
    execute_science, load_training_state, save_training_state,
    validate_scientific_config, _packing,
)


@pytest.fixture
def plan():
    recipe=json.loads((Path(__file__).parents[2]/"recipes/cpu-smoke-v1.json").read_text())
    return {"recipe":recipe,"head":"flow_matching_transformer","mode":"reproduce",
            "product":"point","limits":recipe["limits"],"seed":42}


@pytest.mark.parametrize("head",list(HEADS))
def test_actual_heads_train_save_reload_and_none_identity(tmp_path,plan,head):
    plan["head"]=head
    result=execute_science("smoke",plan,tmp_path)
    assert result["status"]=="succeeded"
    assert result["execution_status"]=="synthetic_actual_refinement"
    with np.load(result["outputs"]["baseline"]) as b,np.load(result["outputs"]["refined"]) as r:
        assert r["fields"].shape==b["fields"].shape
        assert np.isfinite(r["fields"]).all()
        if head=="none":np.testing.assert_array_equal(r["fields"],b["fields"])
    if head!="none":
        saved=load_training_state(Path(result["outputs"]["refinement"]),checkpoint_contract(effective_config(plan),_packing(json.loads(Path(result["outputs"]["packing"]).read_text()))))
        assert saved["step"]==1
        assert saved["optimizer"]["state"]
        assert all(not key.startswith("aurora.") for key in saved["state_dict"])


def test_checkpoint_incompatibility_rejected_before_tensor_load(tmp_path):
    import torch
    p=tmp_path/"state.npz"
    save_training_state(p,{"contract":{"head":"diffusion_unet","levels":[1000]},"state_dict":{"w":torch.ones(2)}})
    with pytest.raises(ValueError,match="Checkpoint incompatibility"):
        load_training_state(p,{"head":"flow_matching_transformer","levels":[850]})


def test_checkpoint_roundtrip_optimizer_integer_keys_and_bfloat(tmp_path):
    import torch
    p=tmp_path/"state.npz"
    original={"contract":{},"state_dict":{"w":torch.ones(2,dtype=torch.bfloat16)},"optimizer":{"state":{1:{"step":torch.tensor(2.)}}},"rng":torch.get_rng_state()}
    save_training_state(p,original)
    actual=load_training_state(p,{})
    assert torch.equal(actual["state_dict"]["w"],original["state_dict"]["w"])
    assert actual["optimizer"]["state"][1]["step"].item()==2
    with pytest.raises(TypeError):save_training_state(p,{"state_dict":object()})


def test_repeatability_and_inference_does_not_read_reference(tmp_path,plan):
    import aurora_workflow.science as science
    a=execute_science("smoke",plan,tmp_path/"a")
    b=execute_science("smoke",plan,tmp_path/"b")
    with np.load(a["outputs"]["refined"]) as aa,np.load(b["outputs"]["refined"]) as bb:
        np.testing.assert_array_equal(aa["fields"],bb["fields"])
    inputs=a["outputs"]
    Path(inputs["reference"]).unlink()
    plan["inputs"]=inputs;plan["roots"]={"output":str(tmp_path)};plan["mode"]="replay"
    replay=execute_science("refine",plan,tmp_path/"replay")
    assert Path(replay["outputs"]["refined"]).is_file()
    with pytest.raises(ValueError,match="Replay never"):
        science._train(plan,effective_config(plan),tmp_path/"replay")


def test_frozen_autonomous_backbone_matches_existing_state_advance():
    import dataclasses
    import torch
    from datetime import datetime,timedelta
    from aurora import Batch,Metadata
    from finetune.aurora_finetune_utils import _advance_batch_with_prediction

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
        def forward(self,batch):
            return dataclasses.replace(batch,surf_vars={"2t":batch.surf_vars["2t"][:,-1:]+self.weight},metadata=dataclasses.replace(batch.metadata,time=(batch.metadata.time[0]+timedelta(hours=12),)))
    model=TinyBackbone()
    batch=Batch(surf_vars={"2t":torch.ones(1,2,4,4)},atmos_vars={},static_vars={},metadata=Metadata(lat=torch.linspace(50,30,4),lon=torch.linspace(230,260,4),time=(datetime(2024,1,1),),atmos_levels=()))
    before=model.weight.detach().clone()
    predictions=list(autonomous_rollout(model,batch,3))
    current=batch
    for index,pred in enumerate(predictions):
        expected=model(current)
        torch.testing.assert_close(pred.surf_vars["2t"],expected.surf_vars["2t"])
        assert torch.all(pred.surf_vars["2t"]==1+2*(index+1))
        current=_advance_batch_with_prediction(current,expected)
    assert torch.equal(before,model.weight)
    assert not model.weight.requires_grad


def test_real_rollout_adapter_global_input_regional_output_with_synthetic_backbone(tmp_path,plan,monkeypatch):
    """Mocked backbone, real NetCDF/batch/packing path; not a pretrained test."""
    import dataclasses
    import inspect
    import torch
    import xarray as xr
    from datetime import timedelta
    import aurora
    monkeypatch.setattr("aurora_workflow.science._verify_phase1_assets",lambda *args:{"checkpoint_sha256":"explicit_synthetic_backbone"})
    original=aurora.AuroraAirPollution
    seen=[]
    class SyntheticBackbone(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        def load_checkpoint_local(self,path,strict):assert strict
        def forward(self,batch):
            seen.append((batch.spatial_shape,float(batch.surf_vars["2t"][0,-1,0,0])))
            return dataclasses.replace(batch,surf_vars={k:v[:,-1:]+1 for k,v in batch.surf_vars.items()},atmos_vars={k:v[:,-1:]+1 for k,v in batch.atmos_vars.items()},metadata=dataclasses.replace(batch.metadata,time=(batch.metadata.time[0]+timedelta(hours=12),)))
    SyntheticBackbone.__init__.__signature__=inspect.signature(original.__init__)
    monkeypatch.setattr(aurora,"AuroraAirPollution",SyntheticBackbone)
    cfg=effective_config(plan)
    cfg["data"].update(lat_min=31,lat_max=52,lon_min=180,lon_max=270)
    lat=np.array([90,60,52,45,38,31,0,-45,-90],dtype=float)
    lon=np.arange(12,dtype=float)*30
    times=np.array(["2024-01-01T00","2024-01-01T12"],dtype="datetime64[ns]")
    levels=cfg["data"]["atmos_levels"]
    fields={}
    for spec in cfg["workflow"]["backbone_predictor_variables"]:
        dims=("time","latitude","longitude") if spec["kind"]=="surf" else ("time","level","latitude","longitude")
        shape=(2,9,12) if spec["kind"]=="surf" else (2,13,9,12)
        fields[spec["dataset_name"]]=(dims,np.ones(shape,dtype="float32"))
    dataset=xr.Dataset(fields,coords={"time":times,"level":levels,"latitude":lat,"longitude":lon})
    dataset.to_netcdf(tmp_path/"history.nc")
    reference=dataset.isel(time=0,drop=True).expand_dims(forecast_reference_time=[times[1]],lead_time=[12,24])
    reference.to_netcdf(tmp_path/"target.nc")
    np.savez(tmp_path/"static.npz",lat=lat,lon=lon,**{s["aurora_name"]:np.ones((9,12),dtype="float32") for s in cfg["data"]["static_variables"]})
    (tmp_path/"static.json").write_text("{}")
    (tmp_path/"pretrained.ckpt").write_bytes(b"explicit synthetic model fixture")
    manifest={"data_path":str(tmp_path/"history.nc"),"reference_path":str(tmp_path/"target.nc"),"source":{"reference_kind":"forecast"},"cases":[{"cycle":"2024-01-01T12:00:00","lead_hours":[12,24],"split":"test"}]}
    (tmp_path/"prepared.json").write_text(json.dumps(manifest))
    plan.update(config=cfg,recipe={"id":"synthetic-backbone-integration","fixture":False},inputs={"prepared":str(tmp_path/"prepared.json"),"static":str(tmp_path/"static.npz"),"pretrained":str(tmp_path/"pretrained.ckpt")},roots={"output":str(tmp_path)})
    result=execute_science("rollout",plan,tmp_path)
    assert seen==[((9,12),1.),((9,12),2.)]
    with np.load(result["outputs"]["baseline"]) as result_data:
        assert result_data["fields"].shape[-2:]==(3,3)
        assert np.all(result_data["fields"][:,0]==2)
        assert np.all(result_data["fields"][:,1]==3)


@pytest.mark.parametrize("change",["feedback","family","future","teacher"])
def test_reject_invalid_scientific_contract(plan,change):
    cfg=effective_config(plan)
    if change=="feedback":cfg["model"]["refinement"]["feedback_to_rollout"]=True
    elif change=="family":cfg["model"]["model_variant"]="aurora_v1p5"
    elif change=="future":cfg["rollout"]["keep_exogenous_predictors"]="refresh_from_dataset"
    else:cfg["rollout"]["autoregressive_inputs"]=False
    with pytest.raises(ValueError):validate_scientific_config(cfg)


def test_resume_restores_optimizer_rng_without_warm_start(tmp_path,plan):
    plan["recipe"]["config"]["training"]["num_epochs"]=2
    plan["limits"]["max_train_steps"]=2
    first=execute_science("smoke",plan,tmp_path/"initial")
    plan["mode"]="resume";plan["inputs"]=first["outputs"];plan["roots"]={"output":str(tmp_path)}
    plan["limits"]["max_epochs"]=2
    continued=execute_science("train",plan,tmp_path/"continued")
    assert continued["details"]["optimizer_steps"]==2


def test_bounded_resume_matches_uninterrupted_scheduler_trajectory(tmp_path,plan):
    import torch
    plan["recipe"]["config"]["training"].update(num_epochs=2,scheduler="cosine")
    plan["limits"].update(max_epochs=2,max_train_steps=2)
    full=execute_science("smoke",plan,tmp_path/"full")
    plan["limits"]["max_epochs"]=1
    partial=execute_science("smoke",plan,tmp_path/"partial")
    plan["mode"]="resume";plan["inputs"]=partial["outputs"]
    plan["roots"]={"output":str(tmp_path)};plan["limits"]["max_epochs"]=2
    continued=execute_science("train",plan,tmp_path/"continued")
    packing=_packing(json.loads(Path(full["outputs"]["packing"]).read_text()))
    contract=checkpoint_contract(effective_config(plan),packing)
    expected=load_training_state(Path(full["outputs"]["last_checkpoint"]),contract)
    actual=load_training_state(Path(continued["outputs"]["last_checkpoint"]),contract)
    for name,value in expected["state_dict"].items():
        torch.testing.assert_close(value,actual["state_dict"][name],rtol=0,atol=0)
    assert expected["scheduler"]==actual["scheduler"]


@pytest.mark.parametrize("head",["flow_matching_transformer","diffusion_unet"])
def test_explicit_ensemble_keeps_separate_point_product(tmp_path,plan,head):
    plan["head"]=head
    point=execute_science("smoke",plan,tmp_path/"point")
    plan.update(product="ensemble",ensemble_members=3,mode="replay",inputs=point["outputs"],roots={"output":str(tmp_path)})
    ensemble=execute_science("refine",plan,tmp_path/"ensemble")
    with np.load(point["outputs"]["refined"]) as p,np.load(ensemble["outputs"]["refined"]) as e:
        np.testing.assert_array_equal(p["fields"],e["fields"])
        assert e["ensemble"].shape==(3,*p["fields"].shape)
        assert np.isfinite(e["ensemble"]).all()


@pytest.mark.skipif(os.environ.get("AURORA_RUN_REAL_GPU")!="1",reason="Opt-in real CAMS/GPU requires approved plan, verified assets, credentials and GPU")
def test_real_gpu_end_to_end():
    """Runs actual prepared plan stages, including ADS, through shared backend."""
    import time
    from aurora_workflow.jobs import JobStore
    config=os.environ.get("AURORA_WORKFLOW_CONFIG")
    plan_id=os.environ.get("AURORA_GPU_PLAN_ID")
    if not config or not plan_id:
        pytest.fail("AURORA_WORKFLOW_CONFIG and AURORA_GPU_PLAN_ID must select an already approved bounded plan")
    store=JobStore(config)
    approved_plan=store.plan(plan_id)
    assert approved_plan["recipe"]=="real-gpu-v1" and not approved_plan["fixture"]
    assert approved_plan["limits"]["device"].startswith("cuda")
    assert {"acquire","prepare","rollout","refine","evaluate"} <= set(approved_plan["stages"])
    assert "train" in approved_plan["stages"] or approved_plan["inputs"].get("refinement")
    assert approved_plan["limits"]["max_cases"] <= 3
    assert approved_plan["limits"]["max_train_steps"] <= 1
    job=store.submit(plan_id)
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        current=store.status(job["job_id"])
        if current["state"] in {"succeeded","failed","blocked","cancelled","interrupted"}:
            assert current["state"]=="succeeded",current
            manifest=json.loads(Path(current["manifest"]).read_text())
            assert manifest["execution_kind"]=="real"
            assert {"acquire","prepare","rollout","refine","evaluate"} <= {s["stage"] for s in manifest["steps"]}
            return
        time.sleep(1)
    pytest.fail("Bounded real integration timed out; query/cancel its durable job")
