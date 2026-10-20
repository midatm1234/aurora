# Case Study: Fine-Tuning Aurora Air Pollution on Ozone

This page reports a small fine-tuning test of `AuroraAirPollution` (0.4°, 12-hour time step) on
one week of CAMS analyses, with a focus on ozone: `go3` (13 pressure levels) and `gtco3` (total
column).
It covers what worked, what did not, and suggestions for anyone doing something similar.
The goal was to check the fine-tuning pipeline end to end, not to train a better model.

## What Is Included

| File | Purpose |
|---|---|
| `finetuning/prepare_cams_ozone.py` | Download CAMS analyses (00/12 UTC, `leadtime_hour=0`) for a date range plus the static variables, using the same request as the [CAMS example](example_cams). |
| `finetuning/finetune_ozone.py` | Fine-tuning loop: data QC, baseline evaluation, training, periodic validation, saving the checkpoint and reloading it. `--synthetic N` runs on random data to test the pipeline. |
| `finetuning/submit_aml_ozone.py` | Submit the fine-tuning loop as a single-GPU Azure Machine Learning job. |

The loop follows [Fine-Tuning](finetuning): full fine-tuning with AdamW, `autocast=True` and
`configure_activation_checkpointing()`, run with
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`.
In addition, gradients are clipped to norm 1 and the gradient norm is logged at every step.
The loss is a latitude-weighted MAE in normalised space, averaged over all predicted variables.
Ozone metrics are logged separately.

## Setup

* **Data:** CAMS global composition analyses, 2022-06-01 to 2022-06-08 (16 time steps, about 3.5 GB).
  Every sample uses two history steps and a 12-hour target, which gives 14 samples.
  The last 2 samples are used for validation and the other 12 for training.
* **Compute:** one H100 (94 GB) on an Azure ML cluster (`Standard_NC40ads_H100_v5`).
  The environment is the curated `acpt-pytorch-2.8-cuda12.6` image with this repository
  installed via `pip install -e .`.
* **Steps:** 100 steps of batch size 1, which is about 8 passes over the training samples.

## Results

| Run | LR | Val. loss (step 0 → 100) | `gtco3` RMSE (kg m⁻²) | `go3` normalised MAE |
|---|---|---|---|---|
| Synthetic data (pipeline test) | 3e-4 | n/a | n/a | n/a |
| CAMS | 3e-4 | 0.0461 → **0.0671** (worse) | 9.27e-5 → 1.09e-4 (worse) | 0.0157 → 0.0239 (worse) |
| CAMS | 1e-5 | 0.0461 → **0.0443** (−4%) | 9.27e-5 → 8.75e-5 (−5.5%) | 0.0157 → 0.0158 (flat) |

Resources and stability:

* 1.27 B trainable parameters, 74 GB peak GPU memory and about 5.6 s per step.
* Gradient norms ranged from 0.06 to 0.47 at LR 1e-5 and from 0.11 to 1.9 at LR 3e-4. There were no NaNs.
* In both CAMS runs the saved checkpoint reloaded and gave identical validation metrics.

## What Worked

* The recipe in [Fine-Tuning](finetuning) fits on a single 94 GB GPU for the 0.4° air-pollution
  model with full fine-tuning and AdamW, with about 20 GB to spare.
* Running on synthetic data first (`--synthetic 16`) checked the environment, checkpoint
  download, memory, step time and checkpoint round-trip before any real data was needed.
* Downloading the data locally and mounting it into the AML job kept the ADS API key out of the job.
* At a low learning rate (1e-5), a short fine-tune slightly improved the validation loss and total-column ozone.

## What Did Not Work

* **The default learning rate (3e-4) from `finetuning/finetune.py` damaged the pretrained model.**
  The first steps increased the training loss eightfold (0.045 → 0.37).
  After 100 steps the validation loss was still 45% worse than the pretrained model.
  The pretrained model is already very good on CAMS analyses, so it needs a much gentler update.
* **Ozone on pressure levels (`go3`) did not improve** at 1e-5 within 100 steps.
  With only 2 validation samples from the same week, small differences are not meaningful.
* **ADS access:** a key in `.cdsapirc` authenticates against ADS, but requests fail with
  `403 ... user didn't accept all required site policies` until the ADS terms of use and privacy
  statement are accepted on the ADS website. The CAMS dataset licence must also be accepted.
  The CDS and ADS share the API key but not the URL, so `prepare_cams_ozone.py` always uses the ADS URL.
* **xarray versions:** older xarray versions, such as the one in the curated ACPT image, fail to
  decode the `forecast_period` timedelta in the CAMS NetCDF files.
  `finetune_ozone.py` opens the files with `decode_timedelta=False` to avoid this.

## Suggestions

1. **Start with a low learning rate (around 1e-5)** when fine-tuning on data close to the
   pretraining data. Consider warmup, or a higher learning rate only for new parameters, as
   suggested in [Fine-Tuning](finetuning#extending-aurora-with-new-variables).
2. **Always evaluate the pretrained model first** (step 0). Without a baseline, the falling loss
   at LR 3e-4 would have looked like learning, when it was only recovery from the first steps.
3. **Test the pipeline on synthetic data** with the real grid and variables before spending time
   on data downloads.
4. **Use more data and a separate validation period** for real experiments, for example months of
   training data and a validation period that does not overlap the training window.
   One week is only enough for a smoke test.
5. **Report ozone-specific metrics** per pressure level alongside the overall loss. Improvements
   in the overall loss can hide regressions for individual variables.
6. **Check the data** before training: shapes, NaNs and value ranges compared with the
   normalisation statistics. `finetune_ozone.py` logs this as a `data_qc` record.

## Reproducing

Download the data. This needs an ADS account with the terms and the CAMS licence accepted:

```bash
python finetuning/prepare_cams_ozone.py --start 2022-06-01 --end 2022-06-08 \
    --out-dir ~/cams_ozone --rc-file ~/.cdsapirc
```

Run locally on a GPU with at least 80 GB of memory:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    python finetuning/finetune_ozone.py --data-dir ~/cams_ozone --out-dir ~/ozone_ft \
    --max-steps 100 --lr 1e-5
```

Or submit to Azure ML, after making the data available on a datastore:

```bash
python finetuning/submit_aml_ozone.py --compute <gpu-cluster> \
    --data-uri azureml://datastores/<datastore>/paths/<path>/
```

Metrics are written to `metrics.jsonl` in the output directory, one JSON record per event
(`data_qc`, `eval`, `train` and `reload_eval`).
