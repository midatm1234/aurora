---
type: Scientific Reference
title: Scientific foundations and limits
description: Literature motivates design choices without renaming this implementation.
model_track: aurora_air_pollution
evidence_kind: published_background
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: aurora
  resource: https://www.nature.com/articles/s41586-025-09005-y
  title: nature
- id: flow
  resource: https://arxiv.org/abs/2210.02747
  title: flow
- id: ddpm
  resource: https://arxiv.org/abs/2006.11239
  title: ddpm
- id: dit
  resource: https://arxiv.org/abs/2212.09748
  title: dit
- id: mamba
  resource: https://arxiv.org/abs/2312.00752
  title: mamba
- id: corrdiff
  resource: https://arxiv.org/abs/2309.15214
  title: corrdiff
---

# Distilled background

Aurora (Nature, published 21 May 2025) describes broad geophysical pretraining followed by task adaptation. Its atmospheric-chemistry study has its own CAMS periods and reference comparisons; its headline numbers do not measure this fork's residual refiners. CAMS model updates and changing emissions create distribution shift.[^aurora]

Flow Matching (arXiv:2210.02747v2) trains vector fields along specified conditional probability paths. That formalism is background; this fork's active clean-data residual parameterization and point query must be described from its actual code rather than renamed a velocity ODE model.[^flow]

DDPM (arXiv:2006.11239v2) motivates iterative denoising from a noising process. Training target parameterization and reverse schedule still have to match each checkpoint; diffusion terminology alone does not guarantee calibrated ensembles.[^ddpm]

Diffusion Transformers (arXiv:2212.09748v2) studies Transformer denoisers over image latent patches. A spatial Transformer refiner shares architectural ideas but is not automatically the paper's DiT model or an atmospheric temporal sequence model.[^dit]

Mamba (arXiv:2312.00752v2) describes selective state-space sequence modeling. Sequence-model results on other modalities do not predict pollutant refinement gains; use ordered/shuffled controls and matched spatial baselines.[^mamba]

Residual corrective diffusion (arXiv:2309.15214v4) separates a conditional mean and diffusion residual for atmospheric downscaling and notes calibration challenges. It motivates two-stage reasoning; this fork preserves a frozen Aurora baseline rather than reproducing that paper's U-Net mean model, target resolution or experiment.[^corrdiff]

The arXiv abstract/metadata pages were reviewed to bound these background claims; this is not a reimplementation or independent replication of those papers. Exact accessed versions and response hashes are recorded in the reference lock.

[^aurora]: Nature atmospheric-chemistry methods/results, DOI 10.1038/s41586-025-09005-y.
[^flow]: Flow Matching abstract and version metadata.
[^ddpm]: DDPM abstract and version metadata.
[^dit]: Diffusion Transformers abstract and version metadata.
[^mamba]: Mamba abstract and version metadata.
[^corrdiff]: Residual corrective diffusion abstract and version metadata.
