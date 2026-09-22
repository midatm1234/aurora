"""Detached supervisor with a durable journal, bounded execution, and crash recovery."""
from __future__ import annotations
import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from .common import REPO, artifact, atomic_json, read_json, redact, safe_path, sha256, utcnow
from .jobs import JobStore, process_identity


def _log(stream, destination):
    with destination.open("a") as out:
        for line in iter(lambda: stream.readline(65536), ""):
            if out.tell() < 10_000_000:
                out.write(redact(line[:65536]))
                out.flush()


def _directory_bytes(root: Path):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def _run_directory(store, plan):
    run_dir = safe_path(store.settings.output_root / plan["run_id"], [store.settings.output_root])
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in run_dir.rglob("*")):
        raise ValueError("Run outputs must not contain symlinks")
    for root in (store.settings.cache_root, store.settings.data_root):
        # HF's internal cache links are legitimate only while remaining in the approved root.
        if root.exists() and any(p.is_symlink() and not p.resolve().is_relative_to(root.resolve())
                                 for p in root.rglob("*")):
            raise ValueError("Cache/data symlink escapes an approved writable root")
    return run_dir


def _valid_receipt(path: Path, plan: dict):
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt.get("plan_id") != plan["plan_id"] or receipt.get("state") != "succeeded":
        return None
    for record in receipt.get("result", {}).get("artifacts", []):
        p = Path(record["path"])
        if not p.is_file() or sha256(p) != record["sha256"]:
            raise ValueError("Previously completed stage artifact changed; refusing invalid reuse")
    return receipt


def execute_child(config: str, job_id: str, stage: str):
    # Linux parent-death signal prevents an orphan stage using resources after supervisor crash.
    parent = os.getppid()
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent:
            raise RuntimeError("Supervisor disappeared")
    from .backend import execute_stage
    store = JobStore(config)
    row = store.status(job_id)
    plan = store.plan(row["plan_id"])
    store.require_approval(plan)
    run_dir = _run_directory(store, plan)
    effective = copy.deepcopy(plan)
    effective["data"]["limits"] = {**plan["limits"], "max_download_bytes":
                                      plan["limits"]["max_download_bytes"] - plan.get("reserved_asset_download_bytes",0)}
    if stage == "acquire":
        effective["limits"]["max_download_bytes"] = effective["data"]["limits"]["max_download_bytes"]
    for s in plan["stages"]:
        if s == stage:
            break
        receipt = _valid_receipt(run_dir / "receipts" / (s + ".json"), plan)
        if receipt:
            effective["inputs"].update(receipt["result"].get("outputs", {}))
    if plan["limits"]["gpu_count"] and stage in {"rollout", "train", "refine"}:
        import torch
        if not torch.cuda.is_available():
            raise ModuleNotFoundError("Approved GPU stage needs CUDA-enabled PyTorch")
        fraction = min(1., plan["limits"]["gpu_memory_gb"] * 1e9 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    result = execute_stage(stage, effective, run_dir)
    # Normalize every output into a verified artifact. No unverified receipt-only reuse.
    paths = {}
    for record in result.get("artifacts", []):
        p = safe_path(record["path"], store.settings.roots, exists=True)
        paths[str(p)] = record.get("kind", "artifact")
    for role, value in result.get("outputs", {}).items():
        if isinstance(value, str):
            p = safe_path(value, store.settings.roots, exists=True)
            paths[str(p)] = role
    result["artifacts"] = [artifact(path,kind) for path,kind in paths.items()]
    _run_directory(store, plan)
    if result.get("state", "succeeded") not in {"succeeded", "completed", "passed"}:
        raise RuntimeError("Stage returned non-success: " + str(result.get("state")))
    atomic_json(run_dir / "receipts" / (stage + ".json"),
                {"schema_version":1,"stage":stage,"plan_id":plan["plan_id"],"job_id":job_id,
                 "state":"succeeded","completed_at":utcnow(),"result":redact(result)})


def supervise(config: str, job_id: str):
    from .backend import execution_environment
    store = JobStore(config)
    with store.connect() as db:
        changed = db.execute("UPDATE jobs SET state='running',pid=?,identity=?,started=COALESCE(started,?),updated=?,attempts=attempts+1 WHERE id=? AND state='queued'",
                 (os.getpid(), process_identity(os.getpid()), utcnow(), utcnow(), job_id)).rowcount
    if not changed:
        return
    row = store.status(job_id)
    plan = store.plan(row["plan_id"])
    run_dir = _run_directory(store, plan)
    logs = store.root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / (job_id + ".log")
    stage_list = plan["stages"] if row["stage"] == "workflow" else [row["stage"]]
    started = time.monotonic()
    old_elapsed = row["elapsed"]
    other_elapsed = store.plan_elapsed(plan["plan_id"], excluding=job_id)
    usage_file = run_dir / "initial-usage.json"
    monitored_roots = [store.settings.cache_root, store.settings.data_root]
    if usage_file.exists():
        initial_usage = read_json(usage_file)["bytes"]
    else:
        initial_usage = sum(_directory_bytes(p) for p in monitored_roots)
        atomic_json(usage_file, {"bytes":initial_usage})
    def disk_use():
        return _directory_bytes(run_dir) + max(0,sum(_directory_bytes(p) for p in monitored_roots)-initial_usage)
    state, error, child = "succeeded", None, None
    manifest = {"schema_version":1,"run_id":plan["run_id"],"job_id":job_id,"plan_id":plan["plan_id"],
                "state":"running","execution_status":"running","started_at":row["started"],
                "execution_kind":"fixture" if plan["fixture"] else "real",
                "provenance":{"source":plan["source"],"configuration_hash":plan["configuration_hash"],
                              "seed":plan["seed"],"environment":execution_environment(),
                              "execution_kind":"fixture" if plan["fixture"] else "real"},
                "artifacts":[],"steps":[],"warnings":plan["warnings"],"errors":[]}
    atomic_json(run_dir / "effective-plan.json", plan)
    try:
        store.require_approval(plan)
        for stage in stage_list:
            receipt_path = run_dir / "receipts" / (stage + ".json")
            receipt = _valid_receipt(receipt_path, plan)
            if not receipt:
                env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS=str(plan["limits"]["max_workers"]),
                           MKL_NUM_THREADS=str(plan["limits"]["max_workers"]),
                           CUDA_VISIBLE_DEVICES="0" if plan["limits"]["gpu_count"] else "")
                child = subprocess.Popen([sys.executable,"-m","aurora_workflow.worker","--config",config,
                            "--job",job_id,"--execute",stage], cwd=REPO, env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", start_new_session=True)
                reader = threading.Thread(target=_log,args=(child.stdout,log_path),daemon=True)
                reader.start()
                while child.poll() is None:
                    elapsed = old_elapsed + time.monotonic() - started
                    with store.connect() as db:
                        db.execute("UPDATE jobs SET elapsed=?,updated=? WHERE id=?", (elapsed,utcnow(),job_id))
                    status = store.status(job_id)["state"]
                    if status == "cancelled":
                        state = "cancelled"
                        raise RuntimeError("Cancelled by operator")
                    if elapsed + other_elapsed > plan["limits"]["max_wall_seconds"]:
                        raise RuntimeError("Approved wall-time bound exceeded")
                    if disk_use() > plan["limits"]["max_disk_bytes"]:
                        raise RuntimeError("Approved output byte bound exceeded")
                    try:
                        status_text = Path(f"/proc/{child.pid}/status").read_text()
                        rss = next(int(line.split()[1])*1024 for line in status_text.splitlines() if line.startswith("VmRSS:"))
                        if rss > plan["limits"]["memory_gb"] * 1e9:
                            raise RuntimeError("Approved host memory bound exceeded")
                    except (OSError, StopIteration):
                        pass
                    time.sleep(0.25)
                reader.join(timeout=2)
                if store.status(job_id)["state"] == "cancelled":
                    state = "cancelled"
                    raise RuntimeError("Cancelled by operator")
                if child.returncode:
                    state = "blocked" if child.returncode == 3 else "failed"
                    raise RuntimeError(f"Stage {stage} exited {child.returncode}; see bounded job logs")
                receipt = _valid_receipt(receipt_path, plan)
                if not receipt:
                    raise RuntimeError("Stage returned without verified execution receipt")
            if disk_use() > plan["limits"]["max_disk_bytes"]:
                raise RuntimeError("Completed stage exceeded approved disk bound")
            manifest["steps"].append({"stage":stage,"state":"succeeded","receipt":artifact(receipt_path,"receipt")})
            manifest["artifacts"].extend(receipt["result"].get("artifacts", []))
            atomic_json(run_dir / "manifest.json", manifest)
    except Exception as exc:
        error = redact(str(exc))
        if state == "succeeded":
            state = "failed"
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        with log_path.open("a") as f:
            f.write(error + "\n")
    elapsed = old_elapsed + time.monotonic() - started
    if store.status(job_id)["state"] == "cancelled":
        state = "cancelled"
    if not any(s["stage"] in {"smoke","rollout","train","refine","evaluate"} for s in manifest["steps"]):
        manifest["execution_kind"] = "metadata"
        manifest["provenance"]["execution_kind"] = "metadata"
    manifest.update(state=state,execution_status=state,completed_at=utcnow(),errors=[error] if error else [])
    atomic_json(run_dir / "manifest.json", manifest)
    with store.connect() as db:
        db.execute("UPDATE jobs SET state=?,error=?,elapsed=?,updated=? WHERE id=?",
                   (state,error,elapsed,utcnow(),job_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--execute")
    args = parser.parse_args()
    try:
        if args.execute:
            execute_child(args.config,args.job,args.execute)
        else:
            supervise(args.config,args.job)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print(redact(f"BLOCKED: {type(exc).__name__}: {exc}"),file=sys.stderr)
        sys.exit(3)
    except Exception as exc:
        if not args.execute:
            # Startup/preflight failures must not strand a running record indefinitely.
            try:
                store = JobStore(args.config)
                with store.connect() as db:
                    db.execute("UPDATE jobs SET state='failed',error=?,updated=? WHERE id=?",
                               (redact(str(exc)),utcnow(),args.job))
            except Exception:
                pass
        print(redact(f"FAILED: {type(exc).__name__}: {exc}"),file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
