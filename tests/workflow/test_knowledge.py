"""Offline OKF/profile, path, receipt and canonical client-adapter checks."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import tomllib

import pytest
import yaml

from aurora_workflow.knowledge import (
    KnowledgeError, parse_document, read, refresh, register_run, search, validate_bundle,
)

REPO = Path(__file__).resolve().parents[2]


def concept(root, name="sample", **overrides):
    metadata = {"type": "Concept", "title": "NO2 sample", "description": "A pollutant contract.",
                "sources": [{"id": "source", "resource": "https://example.org/reference"}],
                "evidence_kind": "code_verified", "reporting_status": "background"}
    metadata.update(overrides)
    path = root / (name + ".md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + yaml.safe_dump(metadata) + "---\n\nNO2 details.[^source]\n\n[^source]: Reference.\n")
    return path


def test_repository_bundle_and_progressive_search():
    result = validate_bundle(REPO / "knowledge")
    assert result["ok"], result
    assert result["okf_conformant"]
    assert result["concepts"] >= 17
    results = search(REPO / "knowledge", "flow_matching_conv_unet")
    assert any(row["concept_id"] == "models/refinement" for row in results["matches"])
    record = read(REPO / "knowledge", "evidence/presentation")
    assert record["metadata"]["evidence_kind"] == "unavailable"
    assert record["metadata"]["slides_inspected"] is False
    assert record["trust"] == "unverified"


def test_minimal_okf_and_unknown_fields_are_conformant(tmp_path):
    (tmp_path / "minimal.md").write_text("---\ntype: Unknown vendor type\nunknown_extension: allowed\n---\n[Future](missing.md)\n")
    result = validate_bundle(tmp_path, project_profile=False)
    assert result["ok"] and result["okf_conformant"]
    assert result["warnings"]  # Broken links are tolerated by base OKF.
    strict = validate_bundle(tmp_path)
    assert strict["okf_conformant"] and not strict["ok"]


def test_reserved_filenames_and_missing_type(tmp_path):
    (tmp_path / "index.md").write_text("---\ntype: Concept\n---\n# Invalid\n")
    assert not validate_bundle(tmp_path)["okf_conformant"]
    (tmp_path / "index.md").unlink()
    (tmp_path / "x.md").write_text("---\ntitle: X\n---\nText")
    assert not validate_bundle(tmp_path)["okf_conformant"]


def test_duplicate_yaml_keys_and_unsafe_tags_rejected():
    for raw in ["---\ntype: Concept\ntype: Forged\n---\n", "---\ntype: !!python/object:os.system {}\n---\n"]:
        with pytest.raises(KnowledgeError):
            parse_document(raw)


def test_verified_mapping_and_staleness(tmp_path):
    concept(tmp_path, verified={"by": "process:unit-check", "at": "2020-01-01T00:00:00Z"}, stale_after="2020-01-02T00:00:00Z")
    record = read(tmp_path, "sample")
    assert record["trust"] == "machine-confirmed" and record["stale"]
    concept(tmp_path, verified={"by": "human:actual-reviewer", "at": "2020-01-01T00:00:00+00:00"})
    assert read(tmp_path, "sample")["trust"] == "human-reviewed"


@pytest.mark.parametrize("target", ["../secret", "%2e%2e/secret", "file:///etc/passwd", "foo\\bar", "x\x00y"])
def test_read_rejects_unsafe_paths(tmp_path, target):
    with pytest.raises(KnowledgeError):
        read(tmp_path, target)


def test_symlink_escape_and_size_limit(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("secret")
    (bundle / "escape.md").symlink_to(secret)
    with pytest.raises(KnowledgeError):
        read(bundle, "escape")
    (bundle / "large.md").write_text("x" * 524289)
    with pytest.raises(KnowledgeError):
        read(bundle, "large")


def test_index_refresh_is_deterministic_and_does_not_verify_claims(tmp_path):
    path = concept(tmp_path, "data/no2")
    raw = path.read_bytes()
    assert refresh(tmp_path)["changed"]
    assert not refresh(tmp_path)["changed"]
    assert path.read_bytes() == raw
    assert "verified" not in read(tmp_path, "data/no2")["metadata"]
    assert validate_bundle(tmp_path)["ok"]
    (tmp_path / "data" / "index.md").write_text("# Data\n")
    result = validate_bundle(tmp_path)
    assert result["okf_conformant"] and not result["ok"]
    assert any("missing index entry" in e for e in result["errors"])


@pytest.mark.parametrize("metadata", [
    {"evidence_kind": "invented"},
    {"reporting_status": "executed_real"},
    {"configuration_hash": "bad"},
    {"generated": {"by": "agent/1", "at": "2026-01-01"}},
    {"evidence_kind": "slide_reported"},
    {"model_track": "aurora_v1p5"},
    {"slide_references": [{"slide": 1, "file_sha256": "invented"}]},
])
def test_profile_rejects_false_or_malformed_evidence(tmp_path, metadata):
    concept(tmp_path, **metadata)
    refresh(tmp_path)
    result = validate_bundle(tmp_path)
    assert result["okf_conformant"] and not result["ok"]


def test_source_footnote_binding(tmp_path):
    path = concept(tmp_path)
    path.write_text(path.read_text() + "Claim.[^fabricated]\n")
    result = validate_bundle(tmp_path)
    assert any("Footnote lacks" in e for e in result["errors"])


def manifest_fixture(root):
    run = root / "run-1"
    run.mkdir()
    artifact = run / "metrics.json"
    artifact.write_text('{"fixture":true}\n')
    record = {"run_id": "run-1", "job_id": "job-1", "plan_id": "a" * 64,
              "state": "succeeded", "provenance": {"configuration_hash": "b" * 64, "execution_kind": "fixture"},
              "steps": [{"stage": "smoke", "state": "succeeded", "receipt": {"result": "fixture"}}],
              "artifacts": [{"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    receipt = run / "smoke-receipt.json"
    receipt.write_text(json.dumps({"stage": "smoke", "state": "succeeded", "plan_id": record["plan_id"]}))
    record["steps"][0]["receipt"] = {"path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
    manifest = run / "run_manifest.json"
    manifest.write_text(json.dumps(record))
    return manifest, record, artifact


def test_register_verified_runtime_evidence_is_idempotent(tmp_path):
    manifest, record, artifact = manifest_fixture(tmp_path)
    result = register_run(REPO / "knowledge", manifest, tmp_path)
    destination = Path(result["path"])
    assert destination.is_relative_to(tmp_path / "knowledge")
    assert not result["reused"]
    assert register_run(REPO / "knowledge", manifest, tmp_path)["reused"]
    readback = read(tmp_path / "knowledge", "runs/run-1")
    assert readback["metadata"]["reporting_status"] == "executed_fixture"
    assert readback["trust"] == "machine-confirmed"
    assert validate_bundle(tmp_path / "knowledge")["ok"]
    artifact.write_text("tampered")
    with pytest.raises(KnowledgeError, match="checksum"):
        register_run(REPO / "knowledge", manifest, tmp_path)


@pytest.mark.parametrize("change", ["state", "config", "execution_kind", "traversal", "steps", "job"])
def test_registration_rejects_unverified_or_unsafe_manifest(tmp_path, change):
    manifest, record, artifact = manifest_fixture(tmp_path)
    if change == "state": record["state"] = "failed"
    if change == "config": record["provenance"].pop("configuration_hash")
    if change == "execution_kind": record["provenance"].pop("execution_kind")
    if change == "traversal": record["artifacts"][0]["path"] = "/etc/passwd"
    if change == "steps": record["steps"] = []
    if change == "job": record.pop("job_id")
    manifest.write_text(json.dumps(record))
    with pytest.raises(KnowledgeError):
        register_run(REPO / "knowledge", manifest, tmp_path)


def test_registration_cannot_escape_output_overlay(tmp_path):
    allowed = tmp_path / "runs"
    allowed.mkdir()
    manifest, record, artifact = manifest_fixture(allowed)
    outside = tmp_path / "outside"
    outside.mkdir()
    (allowed / "knowledge").symlink_to(outside)
    with pytest.raises(KnowledgeError, match="escapes"):
        register_run(REPO / "knowledge", manifest, allowed)
    assert not list(outside.iterdir())


def test_canonical_skill_and_client_adapters():
    skill = REPO / ".agents/skills/msresearch-aurora-air-pollution/SKILL.md"
    metadata, body = parse_document(skill.read_text())
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata["name"])
    assert metadata["name"] == skill.parent.name and len(metadata["name"]) <= 64
    assert 1 <= len(metadata["description"]) <= 1024
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        assert (skill.parent / target).resolve().is_file(), target
    assert (REPO / ".claude/skills" / skill.parent.name).resolve() == skill.parent
    for adapter in ("AGENTS.md", "CLAUDE.md", ".github/copilot-instructions.md"):
        assert ".agents/skills/msresearch-aurora-air-pollution/SKILL.md" in (REPO / adapter).read_text()
    ui = yaml.safe_load((skill.parent / "agents/openai.yaml").read_text())
    assert "$" + metadata["name"] in ui["interface"]["default_prompt"]
    assert 25 <= len(ui["interface"]["short_description"]) <= 64
    assert ui["policy"]["allow_implicit_invocation"] is True


def test_client_registration_examples_share_server_and_use_distinct_schemas():
    clients = REPO / "docs/clients"
    codex = tomllib.loads((clients / "codex-config.example.toml").read_text())
    claude = json.loads((clients / "claude-mcp.example.json").read_text())
    vscode = json.loads((clients / "vscode-mcp.example.json").read_text())
    assert set(codex) == {"mcp_servers"}
    assert set(claude) == {"mcpServers"}
    assert set(vscode) == {"servers"}
    entries = [codex["mcp_servers"]["aurora_air_pollution"], claude["mcpServers"]["aurora_air_pollution"], vscode["servers"]["aurora_air_pollution"]]
    for entry in entries:
        assert Path(entry["command"]).is_absolute()
        assert entry["args"][:3] == ["-m", "aurora_workflow.mcp_server", "--config"]
        assert Path(entry["args"][3]).is_absolute()
        assert "env" not in entry  # No example credentials.
    assert entries[1]["type"] == entries[2]["type"] == "stdio"


def test_reference_lock_has_real_pins_and_no_slide_claims():
    lock = json.loads((REPO / "provenance/references.json").read_text())
    assert lock["presentation"]["status"] == "unavailable"
    assert lock["presentation"]["sha256"] is None
    for commit in lock["repositories"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert lock["documents"]["okf"]["revision"] == "22efaa5402775a7c4d4c37f89e41258daaf3cb65"
    for key in ("flow", "ddpm", "dit", "mamba", "corrdiff"):
        assert re.fullmatch(r"arXiv:[0-9.]+v[0-9]+", lock["documents"][key]["arxiv_version"])


def test_receipt_attestation_recomputes_and_rejects_forged_metrics(tmp_path):
    np = pytest.importorskip("numpy")
    from aurora_workflow.evaluation import execute_evaluation
    from aurora_workflow.knowledge import attest_evaluation_receipt
    meta = {"case_cycle": np.array(["2024-07-01T00"]), "lead_hours": np.array([12]),
            "channel": np.array(["tcno2"]), "units": np.array(["kg m-2"]),
            "lat": np.array([42., 41., 40., 39.]), "lon": np.array([240., 241., 242., 243.])}
    reference = np.arange(16, dtype=float).reshape(1, 1, 1, 4, 4)
    for name, fields in (("reference", reference), ("baseline", reference + 2), ("refined", reference + 1)):
        np.savez(tmp_path / f"{name}.npz", fields=fields, **meta)
    config_hash = "c" * 64
    execute_evaluation({"recipe_id": "cpu-smoke-v1", "configuration_hash": config_hash}, tmp_path)
    receipt_path = tmp_path / "evaluation-receipt.json"
    result = attest_evaluation_receipt(receipt_path, tmp_path, config_hash)
    assert result["provenance_valid"] and result["fidelity_valid"]
    with pytest.raises(KnowledgeError, match="configuration"):
        attest_evaluation_receipt(receipt_path, tmp_path, "d" * 64)
    with pytest.raises(KnowledgeError, match="budget"):
        attest_evaluation_receipt(receipt_path, tmp_path, config_hash, max_input_bytes=1)
    # Alter both metrics AND its receipt checksum: integrity alone could accept
    # that pair, while deterministic replay must reject the forged result.
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["rows"][0]["improvement_percent"]["rmse"] = 99
    metrics_path.write_text(json.dumps(metrics))
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"]["metrics"]["sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    assert attest_evaluation_receipt(receipt_path, tmp_path, config_hash, recompute=False)["fidelity_valid"] is False
    with pytest.raises(KnowledgeError, match="fidelity"):
        attest_evaluation_receipt(receipt_path, tmp_path, config_hash)


def test_registration_receipt_plan_binding(tmp_path):
    manifest, record, artifact = manifest_fixture(tmp_path)
    receipt = Path(record["steps"][0]["receipt"]["path"])
    changed = json.loads(receipt.read_text())
    changed["plan_id"] = "e" * 64
    receipt.write_text(json.dumps(changed))
    record["steps"][0]["receipt"]["sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(record))
    with pytest.raises(KnowledgeError, match="bound"):
        register_run(REPO / "knowledge", manifest, tmp_path)


def test_refresh_dry_run_never_writes(tmp_path):
    concept(tmp_path, "subject/detail")
    original = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    preview = refresh(tmp_path, dry_run=True)
    assert preview["dry_run"] and not preview["applied"] and preview["changed"]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == original
    assert not (tmp_path / "index.md").exists()


def test_metadata_jobs_do_not_become_scientific_knowledge(tmp_path):
    manifest, record, artifact = manifest_fixture(tmp_path)
    record["provenance"]["execution_kind"] = "metadata"
    manifest.write_text(json.dumps(record))
    with pytest.raises(KnowledgeError, match="Metadata-only"):
        register_run(REPO / "knowledge", manifest, tmp_path)
