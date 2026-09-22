---
type: Extension Schema
title: Aurora extension profile for OKF v0.2
description: Project evidence checks supplement base OKF conformance without redefining
  it.
model_track: aurora_air_pollution
evidence_kind: code_verified
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
---

# Extension schema

The machine-readable `aurora-extension.schema.json` describes this project's extra fields. `aurora_workflow.knowledge.validate_bundle` reports base `okf_conformant` separately from the stricter `aurora-v1` profile. Missing optional OKF fields and broken links do not invalidate base conformance. Unknown types and keys remain consumable.[^okf]

Project concepts require title, description, sources, evidence_kind and reporting_status for useful attribution. The project profile additionally checks enums, timestamps, unique source IDs, footnote/source joins, SHA256 syntax, local links, progressive indexes and executed-evidence consistency. External links are checked structurally offline; their availability or scientific truth is not certified.

Extensions: `model_track`, `evidence_kind`, `implementation_status`, `execution_status`, `configuration_hash`, `reporting_status`, `slide_references`, `run_manifest`, and `artifact_hashes`. None is an OKF-wide mandatory field. `repository_reported` intentionally separates source-authored historical experiments from this session's `executed_run`. Absence of `verified` means unverified in OKF even when provenance identifies inspected source code. No human review is fabricated.

Runtime receipt registration writes to the configured run root's `knowledge/` overlay. It copies only allowlisted IDs, hashes, status and artifact references from a verified local manifest. Receipts/data remain outside the committed bundle. Index refresh never adds verification timestamps.

[^okf]: OKF v0.2 extension and conformance rules.
