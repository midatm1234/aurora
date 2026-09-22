---
type: Concept
title: Air-pollution reproduction scope
description: Frozen Aurora forecasts followed by residual correction, with explicit
  evidence limits.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: models
  resource: https://microsoft.github.io/aurora/models.html
  title: aurora-models
---

# Scope

This fork uses AuroraAirPollution: CAMS acquisition → preparation → frozen deterministic baseline → selected residual head → point or ensemble inference → matched evaluation → hashed reports. The weather Aurora 1.5 example is a separate model family; it is not a substitute checkpoint.[^models]

The reproducibility question is whether a named source/configuration/data/checkpoint contract can produce traceable forecasts on a recipient's resources. Better scores are an empirical question. Model agreement with CAMS operational forecasts is not observational validation.

Read [model contracts](../models/refinement.md), [recipes](../workflow/recipes.md), [evidence classes](../evidence/policy.md), or [agent routing](../operations/agents.md) as needed. A shared URL supplies neither compute nor credentials nor private trained artifacts.

[^models]: Official model-family description.
