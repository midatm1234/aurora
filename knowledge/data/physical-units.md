---
type: Data Contract
title: Fields, physical units and complete inputs
description: Backbone inputs exceed the small refinement target set.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: no2
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml
  title: finetune/aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml
- id: models
  resource: https://microsoft.github.io/aurora/models.html
  title: aurora-models
---

# Field and coordinate checks

NO2 and O3 atmospheric mixing ratios use `kg kg-1`; total columns use `kg m-2`. Geopotential uses `m2 s-2`, not geometric height. A 1000-hPa field is not universally ground-level concentration; three pressure levels cannot define an exact total column.[^no2]

Validate names/aliases, declared units, finite masks, coordinate order and pressure identity before comparisons. Retain missingness masks and align static fields by named coordinates. Longitude is periodic for the global grid; remove a duplicate cyclic endpoint deliberately and record the operation. Regional padding must not wrap the U.S.-West east and west boundaries.

The backbone retains 13 levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa. Its meteorological and chemical inputs and emission/static fields must satisfy AuroraAirPollution, even when the refinement predicts only NO2 at 1000/925/850 hPa and tcno2.[^models] Preserve the existing YAML dataset-to-Aurora aliases (`t2m→2t`, `u10→10u`, `v10→10v`, `z_static→z`) and static emission/log-emission names.[^no2]

[^no2]: Active NO2 Transformer YAML.
[^models]: Official Air Pollution input table; use that complete contract, not weather-only inputs.
