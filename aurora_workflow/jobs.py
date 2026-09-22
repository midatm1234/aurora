"""SQLite-backed jobs survive MCP disconnects; approvals are made only in a human terminal."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import sys
import time
from .common import REPO, atomic_json, identifier, read_json, redact, utcnow
from .planning import verify_plan
from .settings import Settings

TERMINAL = {"succeeded", "failed", "blocked", "cancelled", "interrupted"}


def process_identity(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # field 22 starttime, robust to spaces in process name
    except (OSError, IndexError):
        return None


class JobStore:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.settings = Settings.load(self.config_path)
        self.root = self.settings.state_root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS approvals(id TEXT PRIMARY KEY, payload TEXT NOT NULL, signature TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, stage TEXT NOT NULL,
              state TEXT NOT NULL, pid INTEGER, identity TEXT, started TEXT, updated TEXT,
              attempts INTEGER NOT NULL DEFAULT 0, elapsed REAL NOT NULL DEFAULT 0, error TEXT,
              UNIQUE(plan_id,stage));
            """)

    def connect(self):
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def save_plan(self, plan: dict):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO plans VALUES (?,?)", (plan["plan_id"], json.dumps(plan)))
        return plan

    def plan(self, plan_id: str):
        identifier(plan_id)
        with self.connect() as db:
            row = db.execute("SELECT payload FROM plans WHERE id=?", (plan_id,)).fetchone()
        if not row:
            raise ValueError("Unknown plan")
        return json.loads(row[0])

    def _key(self) -> bytes:
        keyfile = self.root / "approval.key"
        try:
            fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        return keyfile.read_bytes()

    def approve_in_terminal(self, plan_id: str) -> dict:
        plan = self.plan(plan_id)
        verify_plan(plan, self.settings)
        if plan["blockers"]:
            raise ValueError("Resolve plan blockers before approval: " + "; ".join(plan["blockers"]))
        # No approved=true option, environment switch, or MCP approval tool exists.
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise PermissionError("Human approval requires an interactive terminal")
        print(json.dumps({"plan_id":plan_id,"recipe":plan["recipe"],"head":plan["head"],
                          "mode":plan["mode"],"resources":plan["resources"],"stages":plan["stages"],
                          "inputs":plan["input_fingerprints"],"warnings":plan["warnings"]}, indent=2))
        answer = input(f"Approve these bounds by typing {plan_id[:12]}: ")
        if not hmac.compare_digest(answer.strip(), plan_id[:12]):
            raise PermissionError("Approval not granted")
        return self._record_approval(plan_id)

    def _record_approval(self, plan_id: str) -> dict:
        """Private method for terminal handler and isolated tests, never exposed over MCP."""
        payload = json.dumps({"plan_id": plan_id, "approved_at": utcnow(), "via": "local_operator_terminal"}, sort_keys=True)
        signature = hmac.new(self._key(), payload.encode(), hashlib.sha256).hexdigest()
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO approvals VALUES (?,?,?)", (plan_id, payload, signature))
        return {"state": "approved", "plan_id": plan_id}

    def require_approval(self, plan: dict):
        verify_plan(plan, self.settings)
        with self.connect() as db:
            row = db.execute("SELECT payload,signature FROM approvals WHERE id=?", (plan["plan_id"],)).fetchone()
        if not row or not hmac.compare_digest(hmac.new(self._key(), row[0].encode(), hashlib.sha256).hexdigest(), row[1]):
            raise PermissionError("Plan requires local human approval; run the approve CLI in your terminal")
        if plan["blockers"]:
            raise ValueError("Plan has unresolved blockers")

    def submit(self, plan_id: str, stage: str = "workflow") -> dict:
        plan = self.plan(plan_id)
        self.require_approval(plan)
        if stage != "workflow" and stage not in plan["stages"]:
            raise PermissionError("Stage is outside the approved plan")
        job_id = hashlib.sha256((plan_id + ":" + stage).encode()).hexdigest()[:32]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row:
                return {**dict(row), "job_id": row["id"], "run_id": plan["run_id"]}
            if db.execute("SELECT COUNT(*) FROM jobs WHERE plan_id=? AND state IN ('queued','running')", (plan_id,)).fetchone()[0]:
                raise ValueError("An execution for this plan is already active")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= self.settings.max_concurrent_jobs:
                raise ValueError("Concurrent job bound reached; inspect/recover existing jobs")
            db.execute("INSERT INTO jobs(id,plan_id,stage,state,updated) VALUES (?,?,?,?,?)",
                       (job_id, plan_id, stage, "queued", utcnow()))
        self._launch(job_id)
        return self.status(job_id)

    def _launch(self, job_id):
        env = dict(os.environ, PYTHONPATH=str(REPO))
        # Worker persists all output through its redacting logging wrapper.
        subprocess.Popen([sys.executable, "-m", "aurora_workflow.worker", "--config", str(self.config_path),
                          "--job", job_id], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def plan_elapsed(self, plan_id: str, excluding: str | None = None) -> float:
        with self.connect() as db:
            return db.execute("SELECT COALESCE(SUM(elapsed),0) FROM jobs WHERE plan_id=? AND id!=?",
                              (plan_id, excluding or "")).fetchone()[0]

    def status(self, job_id):
        identifier(job_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ValueError("Unknown job")
            row = dict(row)
            if row["state"] == "running" and process_identity(row["pid"]) != row["identity"]:
                db.execute("UPDATE jobs SET state='interrupted',error=?,updated=? WHERE id=?",
                           ("Worker disappeared; explicit recovery required", utcnow(), job_id))
                row.update(state="interrupted", error="Worker disappeared; explicit recovery required")
        plan = self.plan(row["plan_id"])
        manifest = self.settings.output_root / plan["run_id"] / "manifest.json"
        row["run_id"] = plan["run_id"]
        row["job_id"] = row["id"]
        row["manifest"] = str(manifest) if manifest.exists() else None
        return redact(row)

    def list_jobs(self, limit=50):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM jobs ORDER BY updated DESC LIMIT ?", (min(max(limit,1),100),)).fetchall()
        return {"state": "inspected", "jobs": [self.status(r[0]) for r in rows]}

    def cancel(self, job_id):
        row = self.status(job_id)
        if row["state"] in TERMINAL:
            return row
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='cancelled',updated=? WHERE id=?", (utcnow(), job_id))
        # Supervisor sees the cancellation, terminates its stage process group and writes a receipt.
        return self.status(job_id)

    def recover(self, job_id):
        row = self.status(job_id)
        if row["state"] == "succeeded":
            return row
        if row["state"] not in {"interrupted", "failed", "blocked", "queued"}:
            raise ValueError("Only interrupted, failed, blocked or orphaned queued jobs can recover")
        plan = self.plan(row["plan_id"])
        self.require_approval(plan)
        if row["attempts"] >= plan["limits"]["retries"] + 1:
            raise ValueError("Approved retry budget exhausted; make a new reviewed plan")
        if self.plan_elapsed(row["plan_id"]) >= plan["limits"]["max_wall_seconds"]:
            raise ValueError("Approved wall-time budget exhausted")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE plan_id=? AND id!=? AND state IN ('queued','running')",
                          (row["plan_id"], job_id)).fetchone()[0]:
                raise ValueError("An execution for this plan is already active")
            db.execute("UPDATE jobs SET state='queued',pid=NULL,identity=NULL,error=NULL,updated=? WHERE id=?",
                       (utcnow(), job_id))
        self._launch(job_id)
        return self.status(job_id)

    def logs(self, job_id: str, max_bytes=16000):
        self.status(job_id)
        p = self.root / "logs" / (identifier(job_id) + ".log")
        limit = min(max(max_bytes, 1), 65536)
        if not p.exists():
            return {"job_id": job_id, "text": "", "truncated": False}
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - limit))
            result = f.read(limit).decode(errors="replace")
        return {"job_id": job_id, "text": redact(result), "truncated": p.stat().st_size > limit}
