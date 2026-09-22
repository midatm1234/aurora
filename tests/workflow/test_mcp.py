"""Actual official SDK client/server stdio integration, not mocked protocol traffic."""
import asyncio
import json
import os
from pathlib import Path
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from aurora_workflow.common import REPO, atomic_json
import pytest


def test_sdk_initialize_discovery_schemas_dispatch_and_clean_stdout(tmp_path):
    config=tmp_path/"local.json"
    atomic_json(config,{f"{key}_root":str(tmp_path/key) for key in ("data","cache","output","state")})
    async def exercise():
        params=StdioServerParameters(command=sys.executable,args=["-m","aurora_workflow.mcp_server","--config",str(config)],
                    cwd=str(REPO),env={**os.environ,"PYTHONPATH":str(REPO)})
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                initialized=await session.initialize()
                assert initialized.protocolVersion=="2025-11-25"
                listed=await session.list_tools()
                tools={t.name:t for t in listed.tools}
                assert len(tools)==24
                assert "approved" not in json.dumps(tools["execute_workflow"].inputSchema).lower()
                assert tools["plan_execution"].inputSchema["properties"]["request"]
                for tool in tools.values():assert tool.outputSchema and tool.inputSchema["type"]=="object"
                response=await session.call_tool("inspect_environment",{})
                assert response.structuredContent["fork_import_verified"]
                plan=await session.call_tool("plan_execution",{"request":{"head":"none","stages":["report"]}})
                assert plan.structuredContent["state"]=="planned"
                blocked=await session.call_tool("execute_workflow",{"plan_id":plan.structuredContent["plan_id"]})
                assert blocked.structuredContent["state"]=="approval_required"
                invalid=await session.call_tool("validate_recipe",{"recipe":"../../etc/passwd"})
                assert invalid.structuredContent["state"]=="failed"
                assert (await session.call_tool("list_runs",{})).structuredContent["jobs"]==[]
                return initialized.protocolVersion, sorted(tools)
    protocol,names=asyncio.run(exercise())
    assert protocol and "train_refinement" in names


@pytest.mark.parametrize("head", ["none", "flow_matching_conv_unet", "flow_matching_transformer",
                                  "diffusion_unet", "diffusion_transformer"])
def test_actual_fixture_mcp_execution_and_backend_parity(tmp_path, head):
    """Synthetic Aurora inputs, actual existing head training via MCP and direct backend."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from aurora_workflow.jobs import JobStore
    from aurora_workflow.science import execute_science
    config=tmp_path/"local.json"
    atomic_json(config,{f"{key}_root":str(tmp_path/key) for key in ("data","cache","output","state")})
    store=JobStore(config)
    async def exercise():
        params=StdioServerParameters(command=sys.executable,args=["-m","aurora_workflow.mcp_server","--config",str(config)],
                    cwd=str(REPO),env={**os.environ,"PYTHONPATH":str(REPO),"CUDA_VISIBLE_DEVICES":""})
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                planned=await session.call_tool("plan_execution",{"request":{"recipe":"cpu-smoke-v1","head":head}})
                p=planned.structuredContent
                assert p["state"]=="planned",p
                # Test fixture simulates operator approval internally; production exposes no such tool.
                store._record_approval(p["plan_id"])
                submitted=await session.call_tool("execute_workflow",{"plan_id":p["plan_id"]})
                job=submitted.structuredContent
                assert job.get("job_id"),job
                for _ in range(600):
                    status=(await session.call_tool("job_status",{"job_id":job["job_id"]})).structuredContent
                    if status["state"] in {"succeeded","failed","blocked","interrupted","cancelled"}:break
                    await asyncio.sleep(.1)
                assert status["state"]=="succeeded",store.logs(job["job_id"])
                registered=(await session.call_tool("knowledge_register",{"job_id":job["job_id"]})).structuredContent
                assert registered.get("ok"),registered
                record=(await session.call_tool("knowledge_read",{"concept":registered["concept_id"],"scope":"runs"})).structuredContent
                assert record.get("ok"),record
                return store.plan(p["plan_id"]),status
    plan,status=asyncio.run(exercise())
    run_dir=Path(status["manifest"]).parent
    manifest=json.loads(Path(status["manifest"]).read_text())
    assert manifest["execution_kind"]=="fixture"
    assert {s["stage"] for s in manifest["steps"]}=={"smoke","evaluate","report"}
    assert any(a["kind"]=="refined" for a in manifest["artifacts"])
    direct=tmp_path/"direct"
    execute_science("smoke",plan,direct)
    for name in ("baseline","refined"):
        with np.load(run_dir/(name+".npz"),allow_pickle=False) as wrapped, np.load(direct/(name+".npz"),allow_pickle=False) as plain:
            if name == "baseline" or head == "none":
                np.testing.assert_array_equal(wrapped["fields"],plain["fields"])
            else:
                # CPU reduction threading can change float32 rounding, never scientific units.
                np.testing.assert_allclose(wrapped["fields"],plain["fields"],rtol=1e-6,atol=0)
