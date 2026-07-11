# CAMS Prediction Without Finetuning

This note explains how to use `examples/cams_prediction_local.ipynb` to run **air-pollution forecasting with pretrained Aurora**, without doing any model finetuning.

## What this notebook does

`examples/cams_prediction_local.ipynb`:

- downloads CAMS atmospheric composition data for a chosen UTC initialization time,
- prepares an `aurora.Batch` using two history timesteps,
- loads the pretrained Aurora air-pollution model,
- runs rollout predictions for a configurable forecast window,
- optionally writes predictions to a NetCDF file.

The workflow uses pretrained checkpoints and static variables directly; it does not train or finetune weights.

## Prerequisites

- Python environment with Aurora installed.
- Additional packages used by this notebook:

```bash
pip install cdsapi matplotlib
```

- ADS account + API key in `$HOME/.cdsapirc` for CAMS downloads.
- CAMS dataset terms accepted in ADS.

## Main configuration (`parameters` cell)

Edit the first code cell in `examples/cams_prediction_local.ipynb`:

- `INIT_DATE`, `INIT_HOUR_UTC`: forecast start time (UTC)
- `PREDICTION_WINDOW_DAYS`: rollout length
- `AURORA_STEP_HOURS`: Aurora step size (12h in this example)
- `DO_DOWNLOAD_CAMS_INIT`, `DO_DOWNLOAD_CAMS_RANGE`, `DO_DOWNLOAD_CAMS_FORECAST`: data download switches
- `DOWNLOAD_DIR`: local storage directory
- `RUN_AURORA`: whether to run model inference
- `SAVE_PREDICTIONS_NETCDF`: whether to save outputs

Static variables configuration:

- `STATIC_VARS_PICKLE_PATH = ""`
- `DEFAULT_STATIC_FNAME = "aurora-0.4-air-pollution-static.pickle"` (can be a local file path or a Hugging Face filename)

Static pickle resolution order is:

1. `STATIC_VARS_PICKLE_PATH` (if non-empty)
2. Environment variable `AURORA_STATIC_PICKLE` (if set)
3. If `DEFAULT_STATIC_FNAME` exists locally, use it as a file path
4. Otherwise use `hf_hub_download(repo_id="microsoft/aurora", filename=DEFAULT_STATIC_FNAME)`

## Running locally

1. Open `examples/cams_prediction_local.ipynb`.
2. Update the parameters cell.
3. Run cells top to bottom.

Expected outputs include:

- downloaded CAMS `.zip` and extracted `.nc` files under `DOWNLOAD_DIR`,
- optional predictions file (for example `aurora-cams-predictions_*.nc`).

## Notes

- The notebook can compare Aurora outputs with CAMS forecast files when forecast download is enabled.
- If you already have static variables locally, set `STATIC_VARS_PICKLE_PATH` to avoid repeated hub fetch checks.
- This workflow is inference-only and intended for quick experimentation or baseline evaluation without finetuning.
