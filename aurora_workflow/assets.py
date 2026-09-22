"""Only the pinned public air-pollution assets and operator-registered local artifacts."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
from .common import REPO, artifact, atomic_json, git, read_json, sha256


def verify_source_references() -> dict:
    lock = read_json(REPO / "provenance/source-lock.json")
    source = lock["fork"]["source_commit"]
    if git("merge-base", source, "HEAD") != source:
        raise ValueError("Implementation is not descended from the pinned fork source")
    import importlib.util
    spec = importlib.util.find_spec("aurora")
    if spec is None or Path(spec.origin).resolve().parent != REPO / "aurora":
        raise ValueError("Another aurora package shadows the intended fork")
    return {"state": "succeeded", "fork_source": source,
            "upstream_reference": lock["upstream"], "imported_aurora_path": spec.origin,
            "verification": "git ancestry and import resolution checked; upstream reference is not installed"}


def inspect_assets(registry: dict) -> dict:
    lock = read_json(REPO / "provenance/source-lock.json")
    results = {}
    for role in ("pretrained", "static", "refinement"):
        record = registry.get(role, {})
        p = Path(record["path"]) if record.get("path") else None
        results[role] = {"state": "available_unverified" if p and p.is_file() else "requires_artifact",
                         "path": str(p) if p else None,
                         "bytes": p.stat().st_size if p and p.is_file() else None,
                         "expected_sha256": record.get("sha256"),
                         "official": lock["huggingface"]["assets"].get(role)}
    return {"state": "inspected", "assets": results, "revision": lock["huggingface"]["revision"]}


def convert_official_static(source: Path, target: Path) -> dict:
    """Deserialize ONLY Microsoft's exact pinned, checksum-verified static pickle."""
    import pickle
    import numpy as np
    expected = read_json(REPO / "provenance/source-lock.json")["huggingface"]["assets"]["static"]
    if sha256(source) != expected["sha256"]:
        raise ValueError("Refusing untrusted static pickle: official checksum mismatch")
    # This is deliberately not a general-purpose pickle upload endpoint.
    with source.open("rb") as f:
        fields = pickle.load(f)
    if not isinstance(fields, dict):
        raise ValueError("Unexpected official static representation")
    arrays = {}
    for key, value in fields.items():
        if not isinstance(key, str):
            raise ValueError("Invalid static field name")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number):
            raise ValueError("Static asset contains nonnumeric fields")
        arrays[key] = value
    if any(v.shape != (451, 900) for v in arrays.values()):
        raise ValueError("Pinned 0.4-degree static grid must be 451 by 900")
    # Official CAMS example: north-to-south latitude, eastward [0,360) longitude.
    arrays["latitude"] = np.linspace(90.0, -90.0, 451)
    arrays["longitude"] = np.arange(900, dtype=np.float64) * 0.4
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, target)
    record = {**artifact(target, "static"), "source_sha256": expected["sha256"],
              "conversion": "numeric-only NPZ; unchanged fields plus official global 0.4-degree coordinates", "fields": sorted(arrays)}
    atomic_json(target.with_suffix(".json"), record)
    return record


def retrieve_assets(plan: dict, run_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download
    lock = read_json(REPO / "provenance/source-lock.json")["huggingface"]
    cache = Path(plan["roots"]["cache"]) / "official-assets"
    cache.mkdir(parents=True, exist_ok=True)
    outputs, records, downloaded = {}, [], 0
    for role, expected in lock["assets"].items():
        supplied = plan.get("inputs", {}).get(role)
        target = Path(supplied) if supplied else cache / expected["filename"]
        if role == "static" and supplied and target.suffix == ".npz":
            metadata = read_json(target.with_suffix(".json"))
            if metadata.get("source_sha256") != expected["sha256"] or metadata["sha256"] != sha256(target):
                raise ValueError("Static conversion provenance mismatch")
        else:
            if not target.exists():
                downloaded += expected["bytes"]
                if downloaded > plan["limits"]["max_download_bytes"]:
                    raise ValueError("Download exceeds approved byte bound")
                if shutil.disk_usage(cache).free < expected["bytes"] * 2:
                    raise ValueError("Insufficient storage for atomic asset retrieval")
                source = hf_hub_download(repo_id=lock["repository"], revision=lock["revision"],
                                         filename=expected["filename"], cache_dir=str(cache / "hf"), endpoint="https://huggingface.co")
                if sha256(source) != expected["sha256"]:
                    raise ValueError("Official download SHA256 mismatch")
                partial = target.with_suffix(target.suffix + ".partial")
                shutil.copyfile(source, partial)
                os.replace(partial, target)
            if sha256(target) != expected["sha256"]:
                raise ValueError(f"{role} checksum mismatch: weather or incompatible artifact refused")
            if role == "static":
                converted = cache / "aurora-0.4-air-pollution-static.npz"
                convert_official_static(target, converted)
                target = converted
        outputs[role] = str(target.resolve())
        records.append({**artifact(target, role), "repository": lock["repository"],
                        "revision": lock["revision"], "official_source_sha256": expected["sha256"]})
    if plan.get("inputs", {}).get("refinement"):
        p = Path(plan["inputs"]["refinement"])
        registration = plan["asset_registry"].get("refinement", {})
        if registration.get("sha256") and sha256(p) != registration["sha256"]:
            raise ValueError("Refinement artifact checksum mismatch")
        records.append(artifact(p, "refinement"))
        outputs["refinement"] = str(p)
    atomic_json(run_dir / "assets.json", {"artifacts": records, "outputs": outputs})
    return {"state": "succeeded", "artifacts": records, "outputs": outputs,
            "verification": "Exact public revision and SHA256, numeric static conversion"}
