---
type: Model Contract
title: Unified refinement contracts
description: Four active residual heads, their parameterizations, and historical compatibility
  boundaries.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: config
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/refinement/config.py
  title: finetune/refinement/config.py
- id: flow
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/refinement/flow_matching.py
  title: finetune/refinement/flow_matching.py
- id: no2
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml
  title: finetune/aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml
- id: diffusion
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/refinement/diffusion.py
  title: finetune/refinement/diffusion.py
- id: review
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/NO2_REFINEMENT_REVIEW.md
  title: finetune/NO2_REFINEMENT_REVIEW.md
- id: readme
  resource: https://github.com/midatm1234/aurora/blob/88f652f04aa75b65e410b8d00cf4ea07f2edd945/finetune/README.md
  title: finetune/README.md
---

# Choices

| Workflow value | Human name | Current implementation |
| --- | --- | --- |
| `flow_matching_conv_unet` | Flow matching with UNet | Unified packed convolutional head |
| `flow_matching_transformer` | Flow matching with Transformer | Unified spatial-token head |
| `diffusion_unet` | Diffusion with UNet | Unified conditional denoiser |
| `diffusion_transformer` | Diffusion with Transformer | Unified spatial-token denoiser |
| `none` | Aurora baseline | Identity refinement |

These names are registered by the pinned configuration/implementation.[^config] Historical `flow_matching_unet` (alias `flow_matching`) is the legacy `AuroraFlowRefine` adapter, not a checkpoint alias for `flow_matching_conv_unet`.

The active NO2 flow Transformer selects x1/data clean-residual prediction. Training interpolates noise and clean correction; the shared-process deterministic forecast queries zero state at process time zero. The documented point product must not be replaced by an arbitrary multistep sample. Flow time and physical forecast lead are different clocks. Data-prediction straight-line updates are not a rectified-flow ODE solver; a separately configured velocity model has a different contract.[^flow][^no2]

Diffusion supports `epsilon`, `sample`, and `velocity`, with conversion in its noise schedule. Active corrected recipes select clean `sample`, centered residual scaling, and directly supervised shared-process deterministic prediction. Stochastic diffusion uses the configured reverse sampler/schedule; its endpoint is not obtained by blindly applying flow's `t=0` rule.[^diffusion]

Residual sign is `CAMS - Aurora`; reconstruction adds the decoded correction once. Centered scaled zero represents the residual mean, not necessarily physical zero. Preserve identity initialization, inverse transforms, physical clipping, masks and checkpointed per-channel scalers. Validate all active architecture/target/grid/temporal/lead contracts before loading. Omitted legacy keys preserve older behavior; broad `strict=False` must not hide incompatibility.[^review]

Discrepancy: the old `finetune/README.md` table calls the flow Transformer a velocity network. The active YAML and code select data prediction; configuration/checkpoint metadata decide behavior. Old legacy endpoints and feedback are versioned rather than silently migrated.[^readme]

[^config]: Enum and aliases.
[^flow]: Flow training and sampling implementation.
[^no2]: Active Transformer YAML.
[^diffusion]: Diffusion implementation.
[^review]: Normalization and compatibility findings.
[^readme]: Historical overview table.
