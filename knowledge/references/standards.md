---
type: Format Reference
title: Pinned standards and organizational reference
description: OKF, Agent Skills, MCP and client documentation are versioned references.
model_track: aurora_air_pollution
evidence_kind: published_background
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: okf
  resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/22efaa5402775a7c4d4c37f89e41258daaf3cb65/okf/SPEC.md
  title: okf
- id: skills
  resource: https://agentskills.io/specification
  title: agentskills
- id: mcp
  resource: https://modelcontextprotocol.io/specification/2025-11-25
  title: mcp
- id: vibe
  resource: https://github.com/microsoft/vibe-kit/blob/6590f598c1a7294d7b3c8874721af4cca1908679/skills/msresearch-aurora/SKILL.md
  title: vibe-kit-skill
---

# Standards

OKF v0.2 is pinned to knowledge-catalog commit `22efaa5402775a7c4d4c37f89e41258daaf3cb65`. Concept IDs are bundle-relative file paths without `.md`; `index.md` and `log.md` are reserved. Only `type` is universally mandatory for concepts. Root `index.md` may declare `okf_version`; subordinate indexes have no frontmatter.[^okf]

Agent Skills is referenced at repository commit `69ef37e9424c0a7ea9dd2293b559e43ec8176379`; its public specification requires valid name/description frontmatter. MCP protocol documentation is pinned to revision `2025-11-25`; the implementation uses the supported Python SDK version recorded in the dependency lock and tests.[^skills][^mcp]

The Microsoft vibe-kit Aurora skill at commit `6590f598c1a7294d7b3c8874721af4cca1908679` is used only for progressive disclosure and task routing. Its weather starter generator was not run or copied. The canonical skill routes this fork's existing air-pollution workflow through MCP.[^vibe]

Client setup examples follow separately fetched official documentation. Configuration parse/schema checks are distinct from end-user application handshakes; uninstalled clients must be labeled untested. See [agent concepts](../operations/agents.md).

[^okf]: Pinned official SPEC.
[^skills]: Agent Skills specification.
[^mcp]: MCP protocol 2025-11-25.
[^vibe]: Pinned organizational reference skill.
