# AuroraAirPollution Fine-Tuning: NO2 / tcNO2 over North America

## Overview

This branch (`aurora_finetune`) adds a complete fine-tuning and inference pipeline for the **AuroraAirPollution** model, targeting **NO2** (atmospheric, 13 pressure levels) and **tcNO2** (total-column surface) prediction over a regional North America domain.

## Changes

### New files

| File | Description |
|------|-------------|
| `finetune/aurora_finetune_rollout_config.yaml` | YAML configuration for data paths, variable specs, model settings, training hyperparameters, and rollout options |
| `finetune/aurora_finetune_distributed.py` | Multi-GPU distributed training script (torchrun/gloo) with gradient accumulation and all-reduce |
| `finetune/aurora_finetune_utils.py` | Core utilities: batch construction, target building, supervised loss with per-variable normalization, autoregressive rollout, NetCDF output |
| `finetune/conv_refine.py` | Convolutional residual decoder (`AuroraConvRefine`) — deterministic 3-conv-block bias-correction head |
| `finetune/flow_refine.py` | Flow-matching residual decoder (`AuroraFlowRefine`) — see "Flow-Matching Residual Decoder" section below |
| `finetune/aurora_finetune_rollout.ipynb` | Training notebook — loads config, launches distributed training, plots loss curves, saves checkpoints |
| `finetune/aurora_inference_rollout.ipynb` | Inference notebook — loads fine-tuned checkpoint, runs multi-step rollout, saves NetCDF and plots |
| `finetune/prepare_train_test_from_netcdf.py` | Script to split raw CAMS NetCDF data into train/test sets |
| `finetune/__init__.py` | Package init |
| `finetune/README.md` | Quick-start guide for the fine-tuning pipeline |
| `scripts/foundry_test_request.py` | Foundry server test request script |

### Bug fixes in `aurora/model/aurora.py`

1. **Broadcasting bug in `AuroraAirPollution._post_decoder_hook`** — When batch size B > 1, the expression `prev[name][:, dim_lookup[name]]` reduced shape from `[B, T, H, W]` to `[B, H, W]` (3D). Multiplying this with a `[B, 1, H, W]` tensor triggered incorrect broadcasting: `[1, B, H, W] × [B, 1, H, W] → [B, B, H, W]`, doubling the history dimension. **Fixed** by using slice indexing `prev[name][:, idx : idx + 1]` to preserve the 4D shape `[B, 1, H, W]`.

2. **SO2 KeyError guard** — Added `and "so2" in pred.atmos_vars` check before the SO2 LoRA clipping block to prevent `KeyError` when SO2 is not in the variable set.

### Adapter in `finetune/aurora_finetune_utils.py`

**`_advance_batch_with_prediction`** — Rewritten to handle mismatched keys between model input (predictors) and model output (predictions). The AuroraAirPollution decoder only outputs variables it has heads for, but the encoder expects all predictor variables. The adapter now:
- Updates variables present in both input and output normally (shift history + append prediction)
- Carries forward exogenous predictor variables not in the model output by repeating the last history step

## Configuration

| Setting | Value |
|---------|-------|
| Model | `aurora_air_pollution` (1.27B params, LoRA) |
| Patch size | 3 |
| Timestep | 12 hours |
| Domain | North America, lat 25–50°N, lon 235–310°E (0.4° resolution, 63×188 grid) |
| Predictors (surface) | 2t, 10u, 10v, msl, tcno2 |
| Predictors (atmospheric) | z, u, v, t, q, no2 (13 levels: 50–1000 hPa) |
| Targets | no2 (atmos), tcno2 (surf) |
| Static variables | lsm, z, slt + 8 emission fields (ammonia, co, nox, so2 and their log transforms) |
| Freeze strategy | Backbone fully frozen (`pretrained_lr_scale: 0.0`); only newly-initialised refinement heads train. Default refinement head: `AuroraFlowRefine` (4.4 M params) |
| Training | AdamW, lr=3e-4, batch_size=1, accumulation_steps=8, ~30 epochs, MSE residual flow-matching loss in normalised space |
| Hardware | 4× A10G GPU (distributed via gloo backend, autocast bf16 backbone / fp32 master weights) |

## Flow-Matching Residual Decoder (default refinement head)

`finetune/flow_refine.py` implements `AuroraFlowRefine`, a small **conditional residual regressor** that wraps a frozen Aurora model and learns a bias correction `r = y_true − ŷ` in per-variable normalised space.

### Why a residual head at all

Pretrained Aurora is already near-optimal on the CAMS dates we fine-tune on. A direct AdamW step on Aurora's tiny per-level NO₂ weights (≈1e-9 RMS) shifts them by ~10³× their natural magnitude in a single update and destroys the pretrained mapping. The clean answer is: **freeze Aurora entirely** (`pretrained_lr_scale: 0.0`) and let a freshly-initialised residual head absorb the bias.

### Architecture

Per refined variable (`no2`, `tcno2`):

* A 3-level UNet (`ResidualFlowUNet`) with FiLM time conditioning, ~1.2 M params at `flow_refine_hidden=64`.
* **Output layer is zero-init** so `r̂ ≡ 0` at step 0 → wrapper output equals Aurora's prediction unchanged before any training happens. This is the identity-at-init guarantee that x₁-prediction (not v-prediction) gives you for free.
* For atmospheric variables, the level axis is collapsed into batch (per-level independent refinement) — keeps the head light while sharing the time/conditioning embedding across levels.

### Parameterisation: rectified flow with x₁ (data) prediction

Sampling a noise level `t ∈ [σ_min, 1 − σ_min]` and noise `x₀ ~ N(0, I)`, the head sees

```
x_t   = (1 − t) · x₀ + t · r          # interpolant
cond  = (ŷ − μ) / σ                   # Aurora prediction in normalised space
r̂    = head(x_t, t, cond)
loss = MSE(r̂, r)                      # x₁-prediction
```

Equivalent to velocity-prediction but with two practical wins:

1. **Identity-at-init** (above) — the model never produces noise as residuals before training.
2. **Single-step deterministic eval is principled.** At `sampling_steps = 1` the head reduces to a regression of `E[r | ŷ]`, exactly what a deterministic conv-refine learns, but trained over **all** noise levels — strong stochastic regularisation for free.

### Eval / inference

`AuroraFlowRefine.forward` in `eval()` mode iterates the FM ODE for `sampling_steps` steps and returns `aurora_pred + denorm(r̂)`. We use a **progressive sampling schedule**:

* Phase 1 (`epoch < num_epochs × phase_fraction`, default ⅓): `sampling_steps = 1` — deterministic regression, learn the residual mean fast.
* Phase 2 (rest): `sampling_steps = flow_refine_sampling_steps_late` (default 8) — multi-step stochastic refinement; lets the head model the conditional residual *distribution* (multi-modal corrections, calibrated spread).

### Loss-path integration

`compute_supervised_loss` auto-detects an `AuroraFlowRefine` wrapper and, in training mode, calls `flow_loss(pred_norm, target_norm, var, kind)` — replacing the standard MSE on Aurora's prediction with the FM regression on the *residual*. The displayed train-loss number is therefore residual-MSE at random `t`, NOT directly comparable to a baseline MSE on the prediction.

### Configuration knobs (`model:` block)

```yaml
model:
  flow_refine_enabled: true
  flow_refine_hidden: 64               # UNet base channels
  flow_refine_sampling_steps: 1        # phase-1 (eval) sampling steps
  flow_refine_sampling_steps_late: 8   # phase-2 (eval) sampling steps
  flow_refine_phase_fraction: 0.3333   # epoch fraction at which phase 2 begins
```

Mutually exclusive with `conv_refine_enabled: true` (the deterministic conv-refine head in `conv_refine.py`); the wrappers in `aurora_finetune_utils.maybe_wrap_*` self-gate on these flags.

### Checkpoints

Saved checkpoints carry keys prefixed `base.*` (frozen Aurora) plus `surf_flow.*` / `atmos_flow.*` (the trainable heads). Both `aurora_finetune_rollout.ipynb` and `aurora_inference_rollout.ipynb` rebuild the wrapped model with `ft.maybe_wrap_flow_refine(...)` before `load_state_dict`, so loading is symmetric with training. `norm_stats` are persisted inside the checkpoint payload and restored automatically (with a fallback that recomputes them from the train dataset for older checkpoints).

### Case folders

Set `case_name: my_experiment` at the top of the YAML and all artifacts (checkpoints, training history, rollout NetCDF, plots, manifest) land under `outputs/<case_name>/` and `outputs/checkpoints/<case_name>/`. `resume_from: "best"` resolves relative to the case folder so multi-stage runs chain naturally.

## Data

- **Training**: CAMS forecast, Feb 1 – Mar 31 2026, 112 timesteps (`data/train.nc`)
- **Test**: 6 timesteps (`data/test.nc`)
- **Static fields**: From HuggingFace pickle (`aurora-0.4-air-pollution-static.pickle`)

## Usage

1. Prepare data: `python finetune/prepare_train_test_from_netcdf.py`
2. Train: Run all cells in `finetune/aurora_finetune_rollout.ipynb`
3. Inference: Run all cells in `finetune/aurora_inference_rollout.ipynb`

See `finetune/README.md` for detailed instructions.
