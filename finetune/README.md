# Aurora Notebook Fine-Tuning + Rollout

This folder provides a config-first, notebook-driven fine-tuning workflow:

- `aurora_finetune_rollout.ipynb`
- `aurora_finetune_rollout_config.yaml`
- `aurora_finetune_utils.py`

## Expected Input Format

Provide train/val/test datasets as NetCDF or Zarr with coordinates:

- `time` (or configured `data.time_dim`)
- `latitude` (descending)
- `longitude` (increasing; values can be `[-180, 180)` or `[0, 360)` in source)
- `level` for atmospheric variables (or configured `data.level_dim`)

Variable dims expected:

- Surface predictors/targets: `(time, latitude, longitude)`
- Atmospheric predictors/targets: `(time, level, latitude, longitude)`
- Static fields: `(latitude, longitude)` (or singleton extra dims indexed via `data.extra_dim_indexers`)

## Predictor/Target Mapping

Use:

- `data.predictor_variables`
- `data.target_variables`
- `data.static_variables`

Each entry supports:

```yaml
- dataset_name: t2m
  aurora_name: 2t
  kind: surf
```

You can also provide mapping dictionaries:

- `data.predictor_mapping_to_aurora_names`
- `data.target_mapping_from_aurora_outputs`

## Frozen Backbone vs Full Fine-Tuning

Switch in config:

- Frozen-backbone mode: `model.backbone_freeze: true`
- Full-model fine-tuning: `model.backbone_freeze: false`

Extra controls:

- `model.freeze_embeddings`
- `model.freeze_encoder`
- `model.freeze_decoder`
- `model.trainable_head_only`

The notebook prints total/trainable/frozen parameter counts before training.

## Refinement Heads (recommended for bias correction)

Pretrained Aurora is already near-optimal on dates seen during pretraining; nudging its tiny per-level weights with AdamW is a fast way to *destroy* the model. The recommended path is to **freeze the backbone fully** (`training.scale_aware_lr.pretrained_lr_scale: 0.0`) and attach a small refinement head whose only job is to learn the residual `r = y_true − ŷ`.

Two heads are available, mutually exclusive:

| Head | File | Behaviour | Best for |
|------|------|-----------|----------|
| `AuroraConvRefine` | `conv_refine.py` | Deterministic per-variable 3-conv stack. ~few-hundred-K params. | Cheapest baseline, single forward pass. |
| `AuroraFlowRefine` | `flow_refine.py` | Rectified-flow / x₁-prediction UNet (FiLM time-conditioned), zero-init output layer. ~4 M params. | Multi-modal residual structure; principled single-step regression *and* multi-step stochastic refinement from the same weights. |

Enable in YAML:

```yaml
model:
  conv_refine_enabled: false
  flow_refine_enabled: true
  flow_refine_hidden: 64
  flow_refine_sampling_steps: 1        # phase-1 eval (deterministic regression)
  flow_refine_sampling_steps_late: 8   # phase-2 eval (stochastic refinement)
  flow_refine_phase_fraction: 0.3333   # epoch fraction at which phase 2 begins

training:
  scale_aware_lr:
    enabled: true
    pretrained_lr_scale: 0.0           # fully freeze the Aurora backbone
```

The flow-matching head is described in detail in `Aurora_air_pollution_finetune.md`. Key properties:

- **Identity-at-init** (zero-init output): `eval()` output before any training equals Aurora's prediction exactly.
- **Train-loss is residual-MSE at random noise level** in normalised space — *not* directly comparable to a prediction MSE.
- **Eval uses an iterative sampler** controlled by `sampling_steps`. The trainer ramps it from 1 → `flow_refine_sampling_steps_late` at epoch ≥ `num_epochs × phase_fraction`.

## Mamba Temporal Module (optional, for long-lead rollouts)

The flow-matching head corrects each rollout step *independently* — it has no
memory of how the field (or Aurora's error in it) evolves through time, which is
exactly where 2–3 day rollouts drift relative to a 6 h forecast. The optional
**Mamba temporal module** (`mamba_temporal.py`) adds a selective state-space
(S6) corrector *on top of* the flow-matching output. For each refined variable
it runs a causal Mamba over the **time / rollout-step axis** at every pixel
(parameters shared across space and pressure levels) and predicts a temporal
correction added to the flow-corrected field.

```yaml
model:
  flow_refine_enabled: true            # temporal module rides on the flow head
  mamba_temporal_enabled: true         # default false → flow-matching-only
  mamba_temporal_channels: 16          # encoder / SSM feature width
  mamba_temporal_state: 8              # SSM hidden-state dim (N)
  mamba_temporal_layers: 2             # stacked Mamba blocks
  mamba_temporal_conv: 3               # causal temporal conv kernel
  mamba_temporal_expand: 2             # SSM inner expansion factor
training:
  mamba_temporal_weight: 1.0           # weight of the sequential temporal loss
rollout:
  verbose_provenance: true             # log CAMS-vs-model variables per step
```

Key properties:

- **Identity-at-init** (zero-init decoder): enabling the module changes nothing
  until trained, so existing flow-matching-only checkpoints are numerically
  unchanged and resume cleanly (checkpoint load is tolerant of the added/removed
  temporal keys).
- **Trained on sequential samples**: the temporal loss is computed over the
  *ordered* multi-lead rollout sequence (needs ≥ 2 `target_lead_times`), so the
  module learns temporal dependencies and variability rather than per-step
  behaviour. The flow-corrected frames are detached, so it trains only the Mamba
  parameters and leaves the flow head/backbone untouched.
- **Causal multi-step rollout**: applied step-by-step over the growing history;
  step *s* depends only on steps ≤ *s*. Works for any configured lead
  (6 h, 12 h, 1/2/3 days).
- **Portable**: a pure-PyTorch selective scan runs on CPU/GPU with no extra
  deps; the `mamba_ssm` CUDA kernel is used automatically when available.

### Rollout CAMS-vs-prediction safeguards

`run_rollout` enforces the correct data provenance at every step:

- bias-corrected **target** variables are advanced by the **model's**
  (flow + temporal) prediction and are **never** overwritten by future CAMS
  truth;
- all other context/exogenous predictors are refreshed from CAMS when
  `keep_exogenous_predictors: refresh_from_dataset`;
- assertions fail loudly if any target would be sourced from CAMS or is not
  model-advanced, and `verbose_provenance` logs the per-step split.

## Case Folders

Set `case_name: <experiment>` at the top of the YAML and all training artifacts land under `outputs/<experiment>/` and `outputs/checkpoints/<experiment>/`. The inference notebook auto-derives `CHECKPOINT_PATH` and `OUTPUT_DIR` from the same `case_name`, so a single config drives both ends. `resume_from: "best"` resolves relative to the case folder, so chained-stage finetunes don't trample each other.

## Regional Domains

Enable regional training with:

```yaml
data:
  domain_type: regional
  lat_min: 25
  lat_max: 50
  lon_min: -125
  lon_max: -50
```

The helper layer converts longitudes to `[0, 360)`, subsets the region, and preserves regional coordinates in rollout NetCDF outputs.

## Notebook Usage

From repo root:

```bash
jupyter lab finetune/aurora_finetune_rollout.ipynb
```

In the notebook:

1. Load YAML config.
2. Optionally patch values in the local override cell.
3. Run preprocess/sample build.
4. Train, validate, and save checkpoints (`best.ckpt`, `last.ckpt`).
5. Run rollout and write outputs (`rollout_predictions.nc`, figures, manifest, history).
