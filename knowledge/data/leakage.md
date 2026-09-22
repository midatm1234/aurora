---
type: Scientific Constraint
title: Autonomous inference and leakage controls
description: Only information available at initialization may enter an autonomous
  forecast.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: review
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/NO2_REFINEMENT_REVIEW.md
  title: finetune/NO2_REFINEMENT_REVIEW.md
- id: utils
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/aurora_finetune_utils.py
  title: finetune/aurora_finetune_utils.py
---

# Availability boundary

The scientific baseline should consume exactly the two initialization-history states and frozen static inputs, then feed its own unrefined state into each next step. Refinement is post-processing (`feedback_to_rollout=false`). Future CAMS targets must never condition an autonomous rollout or refiner.[^review]

The historical utility `refresh_from_dataset` can replace future non-target predictors from a dataset. That is forecast-forced/exogenous inference. The canonical autonomous workflow is separately identified because removing this forcing changes the historical experiment. Record every supplied field, source cycle and earliest availability if replaying an explicitly supported forced experiment. The historical-named workflow recipes preserve forcing dynamics using a specified-cycle forecast rather than future analyses; that changes the baseline inputs and is a versioned approximation, not exact replay of the old analysis-refreshed baseline.[^utils]

Training/evaluation target masks also carry future information: condition the head on baseline-derived validity only. Use joint target/baseline validity to score predictions, not as a future target-conditioning feature. The repository review documents this distinction and prior mask leakage.[^review]

[^review]: Pinned NO2 refinement review.
[^utils]: Pinned rollout/dataset utility implementation.
