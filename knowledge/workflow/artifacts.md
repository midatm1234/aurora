---
type: Artifact Contract
title: Artifact and run provenance
description: Successful claims require source, configuration, data, checkpoint and
  output linkage.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: checkpoint
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/refinement/checkpoint.py
  title: finetune/refinement/checkpoint.py
---

# Traceability

Record fork/upstream commits; pretrained/static revisions, names and SHA256; refinement architecture and checksum; effective configuration and hash; raw/prepared data fingerprints; forecast request identities; split/scaler provenance; seeds; software and hardware; and actual stage states. Missing dependencies remain blocked or `requires_artifact`, not success.

Runtime receipts live beside run artifacts. Knowledge registration re-reads a successful manifest, verifies artifact hashes under the configured run root and emits a controlled summary in the runtime knowledge overlay. It does not publish private data or rewrite historical prose. Hash checking establishes consistency with a receipt, not independent observational truth or proof against a malicious host.

Reuse only compatible verified artifacts. Resume continues supported persisted state; recover first inspects durable job status and partial artifacts. A cancelled or interrupted run is not a completed experiment. See [agent tools](../operations/agents.md).

The historical repository already records checkpoint-contract requirements; the workflow preserves that distinction rather than guessing an inaccessible checkpoint's architecture.[^checkpoint]

[^checkpoint]: Pinned checkpoint validation/serialization implementation.
