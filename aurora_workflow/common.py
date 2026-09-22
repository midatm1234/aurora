"""Small shared filesystem, provenance, and redaction primitives."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path: str | Path, max_bytes: int = 8_000_000) -> Any:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("JSON exceeds metadata size limit")
    with path.open() as f:
        return json.load(f)


def safe_path(path: str | Path, roots: list[Path], *, exists: bool = False) -> Path:
    p = Path(path).expanduser()
    if ".." in p.parts or "\x00" in str(p):
        raise ValueError("Parent traversal is forbidden")
    p = p.resolve(strict=exists)
    if not any(p.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Path is outside configured roots")
    if exists and not p.is_file():
        raise ValueError("Expected a regular file")
    return p


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value) or ".." in value:
        raise ValueError("Invalid identifier")
    return value


@lru_cache(maxsize=1)
def _local_credential_values() -> tuple[str, ...]:
    try:
        import yaml
        p = Path(os.environ.get("CDSAPI_RC", "~/.cdsapirc")).expanduser()
        if p.is_file() and p.stat().st_size <= 16384:
            config = yaml.safe_load(p.read_text())
            if isinstance(config, dict) and isinstance(config.get("key"), str):
                return (config["key"],)
    except (OSError, ValueError, ImportError):
        pass
    return ()


def redact(value: Any) -> Any:
    """Never persist credential-bearing exception text, URLs, or configuration fields."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"(?i)(password|secret|token|api.?key|credential)$", k)
                else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if not isinstance(value, str):
        return value
    for key, secret in os.environ.items():
        if re.search(r"(?i)(TOKEN|PASSWORD|SECRET|API_KEY|CDSAPI_KEY)$", key) and len(secret) >= 6:
            value = value.replace(secret, "[REDACTED]")
    for secret in _local_credential_values():
        if len(secret) >= 6:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)((?:token|password|secret|api[_-]?key|authorization)\s*[=:]\s*)[^\s,;]+",
                   r"\1[REDACTED]", value)
    value = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", value)
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                   stderr=subprocess.DEVNULL, timeout=15).strip()


def source_fingerprint() -> dict:
    """Bind approvals to executable source as well as HEAD, including uncommitted edits."""
    # Private outputs and excluded local helpers are not executable dependencies
    # of the public workflow. Avoid traversing multi-terabyte production folders.
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
                "aurora_workflow", "aurora", "finetune", "recipes", "provenance").split("\0")
    paths = sorted({REPO / name for name in names if name
                    and (REPO / name).is_file() and Path(name).suffix in {".py", ".yaml", ".json"}})
    return {"commit": git("rev-parse", "HEAD"),
            "code_hash": digest({str(p.relative_to(REPO)): sha256(p) for p in paths}),
            "dirty": bool(git("status", "--porcelain")), "imported_workflow_path": str(REPO)}


def environment_fingerprint() -> dict:
    import platform
    packages = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
    return {"python":platform.python_version(), "platform":platform.platform(), "packages_hash":digest(packages)}


def artifact(path: str | Path, kind: str = "artifact") -> dict:
    p = Path(path)
    return {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size,
            "kind": kind}
