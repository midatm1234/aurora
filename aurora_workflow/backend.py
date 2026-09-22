"""One scientific dispatch table shared by the CLI, workers, and MCP tools."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .common import REPO, artifact, atomic_json, read_json, redact, source_fingerprint, utcnow
from .settings import Settings


def doctor(settings: Settings, *, check_torch: bool = False) -> dict:
    versions = {}
    for package in ("mcp", "pydantic", "PyYAML", "numpy", "torch", "xarray", "netCDF4", "huggingface-hub", "cdsapi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    spec = importlib.util.find_spec("aurora")
    fork = bool(spec and Path(spec.origin).resolve().parent == REPO / "aurora")
    roots = {}
    for name in ("data_root", "cache_root", "output_root", "state_root"):
        p = getattr(settings, name)
        parent = p
        while not parent.exists():
            parent = parent.parent
        roots[name] = {"path": str(p), "free_bytes": shutil.disk_usage(parent).free,
                       "writable_parent": os.access(parent, os.W_OK)}
    gpu = {"cuda_available": None, "torch_checked": check_torch}
    if shutil.which("nvidia-smi"):
        try:
            gpu["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                "--format=csv,noheader"], text=True, timeout=10).strip()
        except (subprocess.SubprocessError, OSError):
            gpu["hardware"] = "Unavailable"
    if check_torch:
        try:
            import torch
            gpu["cuda_available"] = torch.cuda.is_available()
            gpu["cuda_version"] = torch.version.cuda
        except ImportError:
            gpu["cuda_available"] = False
    return {"state": "ready" if fork and versions["mcp"] else "attention_required",
            "python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
            "dependencies": versions, "aurora_import": spec.origin if spec else None,
            "fork_import_verified": fork, "roots": roots, "gpu": gpu,
            "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "credentials_present": {"ads_local_config": Path(os.environ.get("CDSAPI_RC", "~/.cdsapirc")).expanduser().is_file(),
                                    "ads_environment": bool(os.environ.get("CDSAPI_KEY"))},
            "warnings": ["Credential presence does not establish ADS access or acceptance of data terms.",
                         "GPU capability and checkpoint compatibility must also pass before scientific execution."]}


def execution_environment() -> dict:
    hardware = None
    if shutil.which("nvidia-smi"):
        try:
            hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                             "--format=csv,noheader"], text=True, timeout=10).strip()
        except (subprocess.SubprocessError, OSError):
            hardware = "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(), "gpu_hardware":hardware,
            "packages": {d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")},
            "execution_kind": "recorded_at_runtime"}


def execute_stage(stage: str, plan: dict, run_dir: Path) -> dict:
    if stage == "sources":
        from .assets import verify_source_references
        result = verify_source_references()
        p = run_dir / "sources.json"
        atomic_json(p, result)
        result["artifacts"] = [artifact(p, "sources")]
        return result
    if stage == "assets":
        from .assets import retrieve_assets
        return retrieve_assets(plan, run_dir)
    if stage in {"acquire", "prepare"}:
        from .data import execute_data
        return execute_data(stage, plan, run_dir)
    if stage in {"rollout", "train", "refine", "smoke"}:
        from .science import execute_science
        return execute_science(stage, plan, run_dir)
    if stage == "evaluate":
        from .evaluation import execute_evaluation
        return execute_evaluation(plan, run_dir)
    if stage == "report":
        path = run_dir / "report.md"
        lines = ["# Aurora air-pollution execution report", "", f"Run: `{plan['run_id']}`",
                 f"Plan: `{plan['plan_id']}`", f"Configuration: `{plan['configuration_hash']}`",
                 f"Head: `{plan['head']}`; mode: `{plan['mode']}`; product: `{plan['product']}`.", "",
                 "Synthetic fixture execution; no real Aurora skill claim." if plan["fixture"] else
                 "CAMS operational fields are a model reference, not independent observational truth.", "",
                 "See manifest.json for actual completed stages, output checksums, environment, and errors.",
                 "See evaluation.md (when present) and metrics.json for matched metrics and coverage.", "",
                 "Historical claims are not independently reproduced by writing this report."]
        path.write_text("\n\n".join(lines) + "\n")
        return {"state": "succeeded", "artifacts": [artifact(path, "report")], "outputs": {"report": str(path)}}
    raise ValueError("Executor is not allowlisted")
