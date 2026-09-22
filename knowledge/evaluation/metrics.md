---
type: Metric
title: Matched physical-unit metrics
description: Score the same cases, coordinates, units and masks for baseline and correction.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: evaluation
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/refinement/evaluation.py
  title: finetune/refinement/evaluation.py
- id: loss
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/refinement/losses.py
  title: finetune/refinement/losses.py
---

# Definition

Match initialization cycle, forecast lead, valid time, variable, pressure level and named spatial coordinates. Report requested/matched/excluded cases and reason counts. Score common finite masks in physical units with consistent spherical latitude area weights. Report MAE, RMSE and signed `forecast - reference` bias. Spatial correlation is ordinary centered pattern correlation; it is not anomaly correlation unless an independently defined climatology was subtracted.[^evaluation]

Improvement percent is `100*(baseline_error-refined_error)/baseline_error`; negative means degradation. A zero baseline denominator produces undefined improvement with a recorded reason, never infinity or an invented win.

Report targets and leads (12,24,36,48,60,72 h where available), initialization/valid-time UTC groups, regional/boundary/interior and available coast/hotspot masks, and reference high-concentration subsets. Do not invent unavailable coastline labels. Diagnose tcNO2 southern-boundary overcorrection, global longitude seams, Transformer patch/high-frequency artifacts, profile versus column differences, temporal degradation, and correlation gains with worse bias/MAE/RMSE.

Keep deterministic forecasts, stochastic members, ensemble means and uncertainty products distinct. Empirical ensemble CRPS, spread/skill, interval coverage and rank diagnostics require multiple members. No confidence interval from treating spatial pixels as independent observations; use forecast-cycle/block resampling when supported.

CAMS operational forecasts are a model reference. Agreement is neither observational accuracy nor evidence of outperforming CAMS. Use validation for tuning/checkpoint selection and keep test results separate. Existing loss/CRPS options do not authorize silently changing historical objectives.[^loss]

[^evaluation]: Pinned scientific evaluation implementation.
[^loss]: Pinned loss implementation.
