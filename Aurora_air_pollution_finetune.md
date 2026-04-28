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
| Freeze strategy | Backbone + encoder + embeddings frozen; decoder unfrozen (17.7M / 1.27B trainable) |
| Training | AdamW, lr=3e-4, batch_size=2, accumulation_steps=8, 10 epochs, MSE loss with per-variable normalization |
| Hardware | 2× GPU (distributed via gloo backend, bf16 precision) |

## Data

- **Training**: CAMS forecast, Feb 1 – Mar 31 2026, 112 timesteps (`data/train.nc`)
- **Test**: 6 timesteps (`data/test.nc`)
- **Static fields**: From HuggingFace pickle (`aurora-0.4-air-pollution-static.pickle`)

## Usage

1. Prepare data: `python finetune/prepare_train_test_from_netcdf.py`
2. Train: Run all cells in `finetune/aurora_finetune_rollout.ipynb`
3. Inference: Run all cells in `finetune/aurora_inference_rollout.ipynb`

See `finetune/README.md` for detailed instructions.
