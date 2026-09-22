---
type: Historical Findings
title: Historical experiments and discrepancies
description: Repository-reported outcomes are preserved without claiming independent
  reproduction.
model_track: aurora_air_pollution
evidence_kind: repository_reported
implementation_status: documented
execution_status: not_run
reporting_status: historical_unreproduced
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: no2
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/NO2_REFINEMENT_REVIEW.md
  title: finetune/NO2_REFINEMENT_REVIEW.md
- id: o3
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/REFINEMENT_REVIEW.md
  title: finetune/REFINEMENT_REVIEW.md
- id: mamba
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/MAMBA_ABLATION.md
  title: finetune/MAMBA_ABLATION.md
---

# Historical evidence

The NO2 review separates saved historical products from corrected controlled short training. It reports mixed profile/column behavior, tcno2 excess variance in Transformer products, weak mean-bias behavior at 850 hPa, and remaining checkpoint-promotion failures. Its controlled experiment is explicitly not a full production result.[^no2]

The O3 review reports invalid old flow endpoints, stochastic overrides, insufficient scaling, feedback mismatch and missing model-selection evidence. Correcting these changes a checkpoint contract; the old checkpoint cannot be relabeled as fixed merely by changing inference flags.[^o3]

The Mamba report concerns bounded, compact experiments, not identical native-resolution production YAMLs. Its baseline checkpoint linkage is asserted rather than proven. Its negative findings support keeping temporal modeling optional; they do not establish that temporal models can never help.[^mamba]

The available review reports do not supply the absent presentation's early-versus-one-year chronology. Do not merge those experimental phases or invent a historical validation split. Exact trained artifacts and data must be provided before numerical historical reproduction can be claimed. This implementation session has not independently rerun these historical studies.

[^no2]: NO2 review at the pinned fork commit.
[^o3]: Global O3 review at the pinned fork commit.
[^mamba]: Temporal ablation report at the pinned fork commit.
