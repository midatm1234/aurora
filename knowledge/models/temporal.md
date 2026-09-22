---
type: Experiment Context
title: Optional temporal Mamba
description: Spatial refinement remains canonical; temporal capabilities are separately
  evaluated.
model_track: aurora_air_pollution
evidence_kind: repository_reported
implementation_status: documented
execution_status: not_run
reporting_status: historical_unreproduced
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: ablation
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/MAMBA_ABLATION.md
  title: finetune/MAMBA_ABLATION.md
- id: mamba
  resource: https://arxiv.org/abs/2312.00752
  title: mamba
---

# Optional temporal adapter

Keep Mamba disabled in canonical recipes. Forecast lead embeddings describe lead identity; they do not constitute temporal sequence modeling. A temporal adapter needs ordered history and a checkpoint trained for that contract.

The repository's bounded ablation reports negative temporal results and explicitly distinguishes compact benchmark architectures/strided O3 grids from full production experiments. It also records asserted rather than proven baseline-checkpoint linkage and a corpus mtime/size fingerprint rather than content SHA256. These are historical report statements, not newly reproduced outcomes.[^ablation]

Mamba's published selective state-space architecture is background motivation, not evidence that adding it improves this pollutant task. Preserve optional capability and compare spatial-only, ordered temporal and shuffled-history controls using matched data/seeds.[^mamba]

[^ablation]: Pinned repository ablation report.
[^mamba]: Mamba arXiv v2 abstract/metadata.
