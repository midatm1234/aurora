---
type: Reproduction Recipe
title: Versioned reproduction recipes
description: Four bounded entry points retain source configuration lineage.
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
- id: o3
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/aurora_O3_global_finetune_3day_lead_config.yaml
  title: finetune/aurora_O3_global_finetune_3day_lead_config.yaml
---

# Recipe selection

| Recipe | Purpose | Scientific status |
| --- | --- | --- |
| `cpu-smoke-v1` | CPU fixture through workflow and real refinement classes | Synthetic integration evidence only |
| `real-gpu-v1` | Explicitly bounded real CAMS/GPU integration | Requires data access, GPU and official assets |
| `no2-us-west-v1` | U.S.-West NO2/tcNO2, two histories, 12–72-hour leads | Configuration-derived reproduction recipe |
| `o3-global-v1` | Global O3/column, six 12-hour leads | Configuration-derived reproduction recipe |

Select any of the [four unified heads](../models/refinement.md) or `none` through the same interface. Each experiment/head has a distinct run identifier. The active NO2 Transformer target levels are 1000, 925, 850 hPa, plus tcno2; predictor/backbone levels remain the full 13. Domain bounds are 31–52°N, 128–100°W.[^no2] Global O3 targets go3 at 1000, 500, 100 and 50 hPa plus gtco3.[^o3]

Historical combined checkpoints may instead require `no2-historical-forecast-forced-v1` or `o3-historical-forecast-forced-v1`, the separately identified exogenous-input recipes. Inspect actual checkpoint provenance before choosing. These historical-named variants use the source forcing dynamics with a specified-cycle CAMS forecast rather than the old future-analysis refresh. This is a versioned approximation: head checkpoint replay may be compatible while baseline inputs differ, so it is not exact historical numerical reproduction. Canonical autonomous recipes likewise change the historical forcing contract. “Transformer” alone does not distinguish flow from diffusion.

Replay requires an existing compatible trained checkpoint. Resume requires full resumable optimizer/scheduler/RNG state. Warm-start creates a named new experiment and does not imply resumed optimization. Reproduce-training starts from verified public Aurora assets and the explicit versioned recipe. None of these modes may silently become another.

The original O3 YAML points `val_data_path` and `test_data_path` at the same file. Record that historical limitation. A new purged split is a new documented experiment, not an invented historical held-out validation interval. Recipe dates and purging are machine-readable; never infer dates from slide captions.

[^no2]: Active NO2 Transformer source configuration.
[^o3]: Global O3 source configuration.
