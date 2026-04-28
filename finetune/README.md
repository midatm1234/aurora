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
