"""CPU-only real orchestration tests; no network/model downloads or private credentials."""
import json
import os
from pathlib import Path
import time
import pytest
from aurora_workflow.common import atomic_json, digest, redact, safe_path, sha256
from aurora_workflow.jobs import JobStore, process_identity
from aurora_workflow.planning import PlanRequest, make_plan, validate_recipe, verify_plan
from aurora_workflow.settings import Settings


@pytest.fixture
def store(tmp_path):
    settings = Settings(data_root=tmp_path/"data",cache_root=tmp_path/"cache",
                        output_root=tmp_path/"runs",state_root=tmp_path/"state")
    path = tmp_path/"local.json"
    atomic_json(path,settings.model_dump(mode="json"))
    return JobStore(path)


def plan(store, **overrides):
    return store.save_plan(make_plan(PlanRequest(head="none",stages=["report"],**overrides),store.settings))


def wait_job(store, job, timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        row=store.status(job["id"])
        if row["state"] in {"succeeded","failed","blocked","cancelled","interrupted"}:
            return row
        time.sleep(.1)
    raise AssertionError(store.logs(job["id"]))


def test_planning_and_approval_enforced(store, monkeypatch):
    p=plan(store)
    assert p["state"]=="planned" and not store.settings.output_root.exists()
    with pytest.raises(PermissionError):store.submit(p["plan_id"])
    with pytest.raises(PermissionError):store.approve_in_terminal(p["plan_id"])
    with pytest.raises(ValueError):PlanRequest.model_validate({"approved":True})
    store._record_approval(p["plan_id"])
    changed=dict(p,seed=p["seed"]+1)
    with pytest.raises(ValueError,match="integrity"):store.require_approval(changed)
    with store.connect() as db:
        db.execute("UPDATE approvals SET signature='forged'")
    with pytest.raises(PermissionError):store.require_approval(p)


def test_real_detached_dispatch_persistence_idempotency(store):
    p=plan(store)
    store._record_approval(p["plan_id"])
    job=store.submit(p["plan_id"])
    result=wait_job(store,job)
    assert result["state"]=="succeeded",store.logs(job["id"])
    restarted=JobStore(store.config_path)
    assert restarted.submit(p["plan_id"])["id"]==job["id"]
    manifest=json.loads(Path(result["manifest"]).read_text())
    assert manifest["execution_kind"]=="metadata"
    assert manifest["steps"][0]["stage"]=="report"
    assert all(sha256(x["path"])==x["sha256"] for x in manifest["artifacts"])
    assert restarted.recover(job["id"])["state"]=="succeeded"


def test_cancel_recovery_and_bounds(store, monkeypatch):
    p=plan(store)
    store._record_approval(p["plan_id"])
    monkeypatch.setattr(store,"_launch",lambda _:None)
    job=store.submit(p["plan_id"])
    assert store.cancel(job["id"])["state"]=="cancelled"
    with pytest.raises(ValueError):store.recover(job["id"])
    with store.connect() as db:
        db.execute("UPDATE jobs SET state='running',pid=99999999,identity='1' WHERE id=?",(job["id"],))
    assert store.status(job["id"])["state"]=="interrupted"
    assert store.recover(job["id"])["state"]=="queued"
    with store.connect() as db:
        db.execute("UPDATE jobs SET state='failed',attempts=99 WHERE id=?",(job["id"],))
    with pytest.raises(ValueError,match="retry budget"):store.recover(job["id"])


def test_resource_input_and_path_restrictions(store,tmp_path,monkeypatch):
    with pytest.raises(ValueError):make_plan(PlanRequest(limits={"max_wall_seconds":0}),store.settings)
    with pytest.raises(ValueError):make_plan(PlanRequest(limits={"device":"cuda:5"}),store.settings)
    with pytest.raises(ValueError):safe_path(tmp_path/"state/approval.key",store.settings.roots)
    store.settings.data_root.mkdir()
    link=store.settings.data_root/"escape"
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError):safe_path(link/"state/approval.key",store.settings.roots)
    with pytest.raises(ValueError):safe_path(store.settings.data_root/"../state",store.settings.roots)
    raw=store.settings.data_root/"input.npz";raw.write_bytes(b"first")
    p=plan(store,inputs={"baseline":str(raw)})
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):verify_plan(p,store.settings)
    monkeypatch.setenv("CDSAPI_KEY","operator-secret-value")
    assert "operator-secret-value" not in redact("Failed operator-secret-value https://example.com/?key=operator-secret-value")
    assert redact({"api_key":"abc"})["api_key"]=="[REDACTED]"


@pytest.mark.parametrize("head",["none","flow_matching_conv_unet","flow_matching_transformer","diffusion_unet","diffusion_transformer","flow_matching_unet"])
def test_recipe_selections_and_replay_never_trains(store,head):
    result=validate_recipe("cpu-smoke-v1",head)
    assert result["config"]["model"]["refinement"]["type"]==head
    p=make_plan(PlanRequest(recipe="no2-us-west-v1",head=head,mode="replay",stages=["refine"]),store.settings)
    assert "train" not in p["stages"]
    if head!="none":assert p["blockers"]


def test_atomic_writes_process_identity_and_log_redaction(store,monkeypatch):
    assert process_identity(os.getpid())
    p=plan(store);store._record_approval(p["plan_id"])
    monkeypatch.setattr(store,"_launch",lambda _:None)
    job=store.submit(p["plan_id"])
    path=store.root/"logs"/(job["id"]+".log");path.parent.mkdir()
    monkeypatch.setenv("API_KEY","top-secret-test")
    path.write_text("x"*70000+" top-secret-test")
    result=store.logs(job["id"],999999)
    assert result["truncated"] and len(result["text"])<=65536 and "top-secret-test" not in result["text"]


def test_reject_write_root_symlink_escape(store,tmp_path):
    from aurora_workflow.worker import _run_directory
    p=plan(store)
    store.settings.cache_root.mkdir()
    (store.settings.cache_root/"official-assets").symlink_to(tmp_path)
    with pytest.raises(ValueError,match="symlink escapes"):
        _run_directory(store,p)
