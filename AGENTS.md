# Aurora air-pollution workflow

For CAMS acquisition, AuroraAirPollution rollout, refinement training/replay/resume, evaluation or workflow explanation, read the canonical [air-pollution skill](.agents/skills/msresearch-aurora-air-pollution/SKILL.md) and [knowledge index](knowledge/index.md). These are the authoritative workflow instructions; client adapters link them rather than maintain separate scientific procedures.

Start by inspecting environment, assets and existing runs. Use the local MCP server for agent scientific execution; review a bounded plan before human approval. Setup and registration are described in [docs/agent-workflow.md](docs/agent-workflow.md). Preserve historical notebooks, checkpoint contracts and unrelated working-tree changes. Do not run the Microsoft weather starter generator or substitute Aurora 1.5.

For implementation work, use the existing `finetune` models and the shared `aurora_workflow` backend. Default validation is CPU-only: `python -m pytest --confcutdir=tests/workflow tests/workflow`. Keep generated data, checkpoints, local configuration, credentials and run outputs out of commits. State what actually ran; source-reported studies and synthetic checks are not new scientific reproduction.
