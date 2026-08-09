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

* A 3-level UNet (`ResidualFlowUNet`) with separate FiLM embeddings for flow interpolation time and cumulative forecast lead, ~1.2 M params at `flow_refine_hidden=64`.
* **Output layer is zero-init** so `r̂ ≡ 0` at step 0 → wrapper output equals Aurora's prediction unchanged before any training happens. This is the identity-at-init guarantee that x₁-prediction (not v-prediction) gives you for free.
* For atmospheric variables, the level axis is collapsed into batch (per-level independent refinement) — keeps the head light while sharing the time/conditioning embedding across levels.

### Parameterisation: rectified flow with x₁ (data) prediction

Sampling a noise level `t ∈ [σ_min, 1 − σ_min]` and noise `x₀ ~ N(0, I)`, the head sees

```
x_t   = (1 − t) · x₀ + t · r          # flow interpolant
cond  = (ŷ − μ) / σ                   # Aurora prediction in normalised space
lead  = valid_time − initialization_time  # cumulative forecast hours
r̂    = head(x_t, t, cond, lead)
loss = MSE(r̂, r)                      # x₁-prediction
```

Equivalent to velocity-prediction but with two practical wins:

1. **Identity-at-init** (above) — the model never produces noise as residuals before training.
2. **Single-step deterministic eval is principled.** At `sampling_steps = 1`
   the head is queried at `t=0, x₀=0`. Source noise is independent of the
   target, so the squared-error optimum is `E[r | ŷ]`. The training loss
   explicitly supervises this exact query; querying `t=1, x_t=0` is invalid
   because the training interpolant equals `r` at `t=1`.

### Eval / inference

`AuroraFlowRefine.forward` in `eval()` mode returns
`aurora_pred + denorm(r̂)`. For deterministic bias correction, use:

* `sampling_steps = 1`: query the source mean once, with no artificial
  ensemble spread.
* `sampling_steps > 1`: opt-in stochastic sampling from a Gaussian source.
  Use this only when ensemble calibration is part of the objective and
  validation; it is not a drop-in replacement for deterministic correction.

### Loss-path integration

`compute_supervised_loss` auto-detects an `AuroraFlowRefine` wrapper and, in training mode, calls `flow_loss(pred_norm, target_norm, var, kind, lead_time_hours=...)` — replacing the standard MSE on Aurora's prediction with the FM regression on the *residual*. The displayed train-loss number is therefore residual-MSE at random `t`, NOT directly comparable to a baseline MSE on the prediction.

### Structural auxiliary losses (`training.flow_aux_loss` block)

The base flow loss is a **per-pixel / per-level residual MSE**. That objective is blind to spatial structure, extremes, distribution, vertical shape, and the column↔profile relationship — exactly the coherent biases left in the rollout-vs-test difference maps (e.g. `rollout_vs_test_gtco3.gif`). To close that gap, `flow_loss` layers weighted structural terms on top, all computed on the head's **clean residual estimate** `refined = ŷ + r̂` versus the target (so they penalise the *structured* error `r̂ − r`, not just its magnitude):

| Term | Weight key | What it penalises |
|------|------------|-------------------|
| Extreme-event | `extreme_weight`, `peak_weight` | Quantile-weighted squared error on tail cells (`extreme_quantile`, `extreme_intensity`) + spatial peak (max/min) magnitude mismatch — corrects under-predicted plumes. |
| Spatial-pattern | `spatial_grad_weight`, `spatial_acc_weight` | Finite-difference gradient-field MSE + `1 − anomaly-correlation (ACC)` — fixes displaced fronts/gradients. |
| Distributional | `dist_var_weight`, `dist_wasserstein_weight` | Spatial-std mismatch + sorted-value (1-D Wasserstein-2) distance — matches the value distribution. |
| Vertical-profile | `vertical_weight` | Level-to-level finite-difference MSE on the atmospheric profile — enforces a coherent vertical shape (the head is otherwise per-level independent). |
| First-moment / bias | `bias_weight` | Per-sample spatial-**mean** match `MSE(mean(refined), mean(target))`. The *only* term that pins the field's absolute level — ACC subtracts the mean and the variance/Wasserstein terms constrain only spread/shape, so without this nothing penalises a domain-wide offset (e.g. the systematic column-O₃ low bias). |
| Deterministic reconstruction | `deterministic_reconstruction_weight` | Direct MSE between the true residual and the exact inference correction at `t=0, x₀=0`. This closes the random-`t`/inference-point gap. |
| Degradation hinge | `degradation_weight` | `relu((r̂-r)²-r²)`, zero when correction is no worse than leaving the baseline unchanged and positive when it degrades a point. |
| Column/profile coherence | `coherence_weight` | Ties the column var (`gtco3`) to the pressure-weighted vertical integral of the profile var (`go3`): `MSE(D(refined), D(truth))` with `D = col − Σ_l w_l · prof_l`. Computed cross-variable in `compute_supervised_loss`. **Caveat:** total-column O₃ is stratosphere-dominated (~10–50 hPa), *above* the `loss_levels` used here, so the integral is a poor proxy and can drag `gtco3` toward a low bias — disable it (`coherence_weight: 0.0`) for the column-O₃ fine-tune. Note it is inert for the NO₂ case anyway (its hard-coded `gtco3`/`go3` pair never resolves against `no2`/`tcno2` targets). |

All weights default to `0.0` (pure residual MSE → backward compatible), and a single `enabled` flag toggles the whole feature on/off without re-zeroing weights. The block is exposed in every flow-refine config. Example block:

```yaml
training:
  flow_aux_loss:
    enabled: true                 # master on/off switch (single line)
    extreme_weight: 0.5
    extreme_quantile: 0.95
    extreme_intensity: 4.0
    peak_weight: 0.25
    spatial_grad_weight: 0.5
    spatial_acc_weight: 0.25
    dist_var_weight: 0.25
    dist_wasserstein_weight: 0.25
    bias_weight: 0.5              # spatial-mean match — counters systematic bias
    deterministic_reconstruction_weight: 1.0
    degradation_weight: 1.0
    aux_on_deterministic: true
    vertical_weight: 0.5
    coherence_weight: 0.25
    coherence_column_var: gtco3   # empty => auto-detect first surf target
    coherence_profile_var: go3    # empty => auto-detect first atmos target
```

When `enabled: false` (or the block is absent), `flow_loss` is exactly the original residual MSE regardless of the individual weights.

### Configuration knobs (`model:` block)

```yaml
model:
  flow_refine_enabled: true
  flow_refine_hidden: 64               # UNet base channels
  flow_refine_sampling_steps: 1        # deterministic source-mean correction
  flow_refine_sampling_steps_late: 1   # keep validation/inference deterministic
  flow_refine_lead_time_cond: true     # distinct 12/24/... h correction regimes
  flow_refine_lead_time_scale_hours: 72  # normalized scale and trained support
  flow_refine_doy_cond: false          # seasonal (day-of-year) conditioning
```

Mutually exclusive with `conv_refine_enabled: true` (the deterministic conv-refine head in `conv_refine.py`); the wrappers in `aurora_finetune_utils.maybe_wrap_*` self-gate on these flags.

### Forecast-lead conditioning (`flow_refine_lead_time_cond`)

Flow interpolation time `t` is not forecast lead: it says how far `x_t` lies
between source noise and the clean residual. Forecast lead says how old the
Aurora rollout is. Training computes exact cumulative hours from each sample
initialization and valid time, validates the 12-hour cadence, and repeats the
case lead across pressure levels after flattening the level axis. Inference
passes `step * rollout_step_hours` and refuses to exceed
`flow_refine_lead_time_scale_hours` for a lead-conditioned checkpoint.

The head encodes scaled linear, `log1p`, and square-root lead features with a
separate MLP, then adds that embedding to the FiLM context. The final MLP layer
is zero-initialized, preserving identity-at-initialization. This lets the head
learn larger or structurally different corrections at longer leads from the
true residual; it deliberately does not force magnitude to be monotonic.

### Seasonal (day-of-year) conditioning (`flow_refine_doy_cond`)

When `true`, each `ResidualFlowUNet` head gains a small `doy_mlp` that maps a
sin/cos encoding of the **fractional day-of-year** into the FM time-embedding
space and adds it to the flow-time embedding before FiLM modulation. The
day-of-year is read from `pred.metadata.time` (per batch element) at both train
and eval time, so one shared head can specialise its residual by season instead
of averaging, e.g., a summer-low ozone correction against a spring-high one. The
final `doy_mlp` layer is **zero-init**, so the feature preserves the
identity-at-init guarantee and is a no-op until trained. Default `false` keeps
the head's behaviour and checkpoint shape unchanged (backward compatible).

### Choosing the column-O₃ vs NO₂ recipe

The aux block and sampling schedule that work for sparse, multi-modal surface
NO₂ plumes are **not** optimal for the smooth, large-scale `gtco3` column field.
For a column-O₃ fine-tune, prefer:

* `flow_refine_sampling_steps_late: 1` — the x₁ parameterisation already returns
  the residual mean `E[r|ŷ]` in one step; extra steps only inject stochastic
  spread (observed as +18 % std inflation and a spurious low-ozone blob).
* `coherence_weight: 0.0` — the column↔profile integral is a poor proxy for
  stratosphere-dominated total-column O₃ (see the loss table caveat).
* `bias_weight > 0` — pin the absolute level to counter the systematic low bias.
* `flow_refine_doy_cond: true` — absorb the strong ozone seasonal cycle.

See `finetune/aurora_O3_finetune_US-WEST_3day_lead_config_v2.yaml` (subtraction:
drop coherence + multi-step) and `..._config_v3.yaml` (v2 + `bias_weight` +
seasonal conditioning).

### Checkpoints

Saved checkpoints carry keys prefixed `base.*` (frozen Aurora) plus `surf_flow.*` / `atmos_flow.*` (the trainable heads). Both `aurora_finetune_rollout.ipynb` and `aurora_inference_rollout.ipynb` rebuild the wrapped model with `ft.maybe_wrap_flow_refine(...)` before `load_state_dict`, so loading is symmetric with training. `norm_stats` are persisted inside the checkpoint payload and restored automatically (with a fallback that recomputes them from the train dataset for older checkpoints).

### Case folders

Set `case_name: my_experiment` at the top of the YAML and all artifacts (checkpoints, training history, rollout NetCDF, plots, manifest) land under `outputs/<case_name>/` and `outputs/checkpoints/<case_name>/`. `resume_from: "best"` resolves relative to the case folder so multi-stage runs chain naturally.

## Data

- **Training**: CAMS forecast, full year 2023-07-01 → 2024-06-30, 732 timesteps at 12 h spacing (per-case `data/<case>/train.nc`)
- **Test**: 2024-07-01 → 2024-09-30, ~184 timesteps (per-case `data/<case>/test.nc`); the July evaluation date is seasonally in-distribution
- **Static fields**: From HuggingFace pickle (`aurora-0.4-air-pollution-static.pickle`)

## Usage

1. Prepare data: `python finetune/prepare_train_test_from_netcdf.py`
2. Train: Run all cells in `finetune/aurora_finetune_rollout.ipynb`
3. Inference: Run all cells in `finetune/aurora_inference_rollout.ipynb`

See `finetune/README.md` for detailed instructions.
