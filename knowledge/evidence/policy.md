---
type: Evidence Policy
title: Evidence and reporting policy
description: Authored knowledge, historical reports and executed artifacts have different
  evidential status.
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

# Evidence classes

`code_verified` means a claim was checked against pinned source/configuration; it is not a numerical rerun. `repository_reported` means a historical report says an experiment occurred. `published_background` is literature, `slide_reported` requires a file checksum and numbered slide, `executed_run` requires a successful receipt and verified artifacts, `hypothesis` is untested, and `unavailable` says evidence was not present.

`reporting_status` further distinguishes background, historical-unreproduced, executed-fixture, executed-real, unavailable and hypothesis. Only actual runtime receipts can justify executed status. Authorship (`generated`) never grants a verification tier. Human review is absent unless an actual human review event is recorded. Base OKF derives trust only from `verified`, and permits missing optional fields and unknown extensions.[^okf]

Keep scientific conclusions scoped to their data and product. Historical percentages/sample counts in repository prose are not imported as current results. Small synthetic tests establish software invariants, not air-quality skill. See [presentation availability](presentation.md), [historical experiment separation](historical.md), and [extension schema](../references/extension-schema.md).

[^okf]: OKF v0.2, pinned SPEC.
