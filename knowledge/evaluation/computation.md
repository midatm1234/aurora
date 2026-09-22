---
title: Matched forecast evaluation
description: Sanctioned physical-unit evaluation with receipt and deterministic local
  attestation.
type: Attested Computation
runtime: python
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
parameters:
- name: plan
  type: approved PlanRequest/effective plan
  required: true
- name: run_dir
  type: approved output directory
  required: true
executor:
  resource: /references/evaluation-executor.md
  receipt:
  - schema_version
  - executor
  - source_sha256
  - dependency_sha256
  - configuration_hash
  - parameters
  - inputs
  - outputs
  - execution_status
attester:
  resource: /references/attesters/evaluation.py
---

# Computation

```python
from aurora_workflow.evaluation import execute_evaluation
result = execute_evaluation(plan, run_dir)
```

The installed fork fixes this function and its dependencies. Through MCP, call `evaluate_forecasts` with an approved plan; parameters do not select arbitrary code. The receipt binds effective configuration, evaluator/dependency source hashes, compared arrays and generated metrics/report/plot hashes.[^okf]

The deterministic attester checks those hashes under an explicitly allowed local root, replays only this evaluator with the recorded parameters in a temporary directory, and compares all output hashes. Default replay input budget is 256 MiB, configurable up to 1 GiB; larger runs need separately reviewed resource-aware attestation. Receipts remain in runtime run artifacts, not this bundle. Source changes fail attestation and require an explicit re-evaluation.

This local computation attestation checks reproducible parameter binding and output fidelity. It is not cryptographic proof against a host owner who controls both code and receipts, nor observational validation. Missing receipts or failed checks must be reported, never replaced by authored prose.

[^okf]: Official distinction between definition verification and per-run attestation.
