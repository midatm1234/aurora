---
title: Evaluation executor and receipt
description: The allowlisted Python evaluation function used by MCP and the human
  CLI.
type: Executor
evidence_kind: code_verified
reporting_status: background
implementation_status: implemented
execution_status: not_run
model_track: aurora_air_pollution
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:39:04.764227+00:00'
sources:
- id: okf
  resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/22efaa5402775a7c4d4c37f89e41258daaf3cb65/okf/SPEC.md
  title: OKF v0.2 attestation contract
---

# Executor

MCP `evaluate_forecasts` dispatches the shared backend to `aurora_workflow.evaluation.execute_evaluation(plan, run_dir)`. The function reads matched baseline/refined/reference NPZ arrays, computes the [metric contract](../evaluation/metrics.md), and writes `metrics.json`, `evaluation.md`, `metrics.svg` and `evaluation-receipt.json`.

Receipt inputs and outputs carry paths and content hashes, source and dependency hashes bind the sanctioned implementation, and parameters preserve evaluation scope. `aurora_workflow.knowledge.attest_evaluation_receipt` re-reads the receipt and repeats deterministic evaluation. Use the [computation concept](../evaluation/computation.md) for the typed contract. This is a project implementation of OKF's runtime-independent executor/attester interface.[^okf]

[^okf]: OKF v0.2.
