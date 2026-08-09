# Flow-refinement reliability review

## Scope and conclusion

This review traced the global O₃ case through data selection, normalization,
training loss, autoregressive feedback, validation, checkpoint persistence,
inference, and evaluation.

The observed result was real: it was not caused by the evaluation notebook
silently resizing or interpolating fields. The principal failure was a
train–inference mismatch compounded by an inference override:

1. the old deterministic sampler queried an invalid endpoint
   (`t=1, x_t=0`);
2. the inference notebook then overrode the checkpoint's one-step setting with
   eight stochastic steps and averaged ten members;
3. residuals much smaller than unit Gaussian source noise were trained without
   residual z-scoring;
4. correlation-oriented auxiliary loss was active without direct supervision
   of the deterministic reconstruction, a degradation penalty, or sufficient
   mean-bias control;
5. later training leads consumed unrefined states, whereas inference consumed
   refined states;
6. the run had no validation/model-selection evidence;
7. forecast lead never reached the refinement head, even though residual
   magnitude and structure change materially from 12 to 72 hours;
8. batched autoregressive training loaded the first sample future exogenous
   frame and repeated it across every sample; and
9. the scheduler horizon counted batches instead of optimizer updates.

These mechanisms explain why spatial anomaly correlation could improve while
MAE and RMSE became worse: the head learned some useful correction pattern, but
applied it with the wrong mean, amplitude, endpoint, and stochastic spread.

The old checkpoint is **not compatible with the corrected deterministic
endpoint**. It must not be promoted as a fixed model. A validation-selected
gate applied to its old deterministic correction is included only as a
representative safeguard/evaluation and as evidence that useful signal exists
inside the learned correction.

## Mathematical contract

Let the normalized Aurora baseline be `y_hat` and the normalized true
correction be

```text
r = y_true - y_hat.
```

The refinement head uses the rectified-flow data/end-point parameterization:

```text
x0 ~ Normal(0, I)
x_t = (1 - t) * x0 + t * r_std
lead = valid_time - initialization_time
r_pred = head(x_t, t, y_hat, lead)
loss_random = MSE(r_pred, r_std)
```

where `r_std = r / sigma_r`. For atmospheric targets, `sigma_r` is now tracked
separately for every configured pressure level.

At `t=0`, source noise is independent of `r`. Under squared loss,

```text
head(x0, 0, y_hat) = E[r_std | x0, y_hat].
```

The deterministic source-mean query is therefore

```text
r_det = head(0, 0, y_hat, lead) ≈ E[r_std | y_hat, lead].
refined = y_hat + sigma_r * r_det.
```

Flow interpolation time `t` and forecast `lead` are separate clocks. The
former describes position on the noise-to-residual path; the latter is the
cumulative physical age of the Aurora rollout. The lead is encoded from scaled
linear, `log1p`, and square-root hour features and added through its own FiLM
embedding. No rule forces correction magnitude to grow monotonically: the
observed residual target and degradation loss determine whether a larger
correction is warranted at each lead.

The former query `head(0, 1, y_hat)` was inconsistent: the training
interpolant is `x_t=r_std` at `t=1`, so `x_t=0` describes a zero residual, not
an unknown residual whose conditional mean should be inferred.

The corrected objective additionally supervises the exact inference query:

```text
loss_det = MSE(r_det, r_std)
loss_degrade = mean(relu((r_det - r_std)^2 - r_std^2))
```

The degradation term is zero wherever applying the correction is no worse
than applying no correction and positive wherever it increases squared error.
It is an optimization constraint, not a guarantee on unseen data; validation
selection remains required.

## Issues and fixes

### 1. Invalid deterministic endpoint

**Issue.** One-step inference evaluated `x_t=0, t=1`, although training has
`x_t=r` at `t=1`.

**Change.** `AuroraFlowRefine._sample_residual` and
`refine_norm_deterministic` now query `x_t=0, t=0`.
`deterministic_reconstruction_weight` trains that exact point.

**Why correct.** At the source endpoint the source is independent of the
target residual; its conditional squared-loss solution is the mean correction.

**Affected file.** `finetune/flow_refine.py`.

### 2. Stochastic inference override

**Issue.** The committed inference notebook used `NUM_ENSEMBLE=10` and
`SAMPLING_STEPS_OVERRIDE=8`, overriding the checkpoint value of one. For a
deterministic bias-correction task this added unvalidated spread and changed
correction amplitude.

**Change.** Defaults are now one member, no sampler override, and one
deterministic step. The model and wrapper defaults are also one. Multi-step
sampling must be explicit.

**Affected files.** `finetune/aurora_inference_rollout.ipynb`,
`finetune/flow_refine.py`, `finetune/aurora_finetune_utils.py`,
`finetune/aurora_finetune_distributed.py`.

### 3. Residual/source scale mismatch

**Issue.** In the saved global checkpoint, normalized O₃ residual RMS was
roughly `0.003–0.04`, while `x0` had unit variance. Random source noise
therefore dominated much of the interpolation. One scalar atmospheric
residual scale would also allow large-residual levels to dominate smaller
levels.

**Change.** The global configuration enables residual z-scoring. Running
scales are computed from exact distributed sums/sums-of-squares over valid
points, independently by atmospheric loss level. Inference only reads the
saved buffers and cannot update them.

**Why correct.** Both flow endpoints have comparable numerical scale, each
level receives a balanced residual target, and de-normalization returns the
correction to the same normalized/physical units as the baseline.

**Affected files.** `finetune/flow_refine.py`,
`finetune/aurora_O3_global_finetune_3day_lead_config.yaml`.

### 4. Random-`t` loss did not constrain the evaluated prediction

**Issue.** Random-`t` endpoint regression and ACC/structure terms could improve
patterns without sufficiently constraining the deterministic output's
absolute error or mean.

**Change.** The O₃ recipe enables deterministic reconstruction, pointwise
degradation, and mean-bias losses. Structural losses operate on the
deterministic correction used by inference. Missing values use the same mask
as the primary loss; incomplete fields skip spatial structural terms instead
of treating filled zeros as observations.

**Affected files.** `finetune/flow_refine.py`,
`finetune/aurora_finetune_utils.py`,
`finetune/aurora_O3_global_finetune_3day_lead_config.yaml`.

### 5. Autoregressive train–inference mismatch

**Issue.** During training, later leads consumed the unrefined Aurora state
because `AuroraFlowRefine.forward()` deliberately returns the base prediction
in training mode. Inference fed the refined state back into the next step.

**Change.** The residual loss still references the unrefined base prediction,
but the next training step receives a detached deterministic refined state.

**Why correct.** The residual target remains `truth - baseline`, while the
autoregressive state distribution matches deterministic inference and avoids
back-propagating through every earlier rollout correction.

**Affected files.** `finetune/aurora_finetune_utils.py`,
`finetune/flow_refine.py`, global O₃ YAML.

### 6. No honest validation or safe model selection

**Issue.** The reviewed run had `skip_validation=true`; `last.ckpt` recorded
`best_val_loss=inf`. `best.ckpt` was a stale legacy epoch-6 artifact, while
inference loaded `last.ckpt`. The configured validation path also pointed at
the test set.

**Change.**

- validation defaults to a temporally separated training-tail split;
- a gap equal to the maximum forecast lead prevents history/target overlap;
- baseline and refined normalized validation MSE and percentage improvement
  are recorded together;
- a best checkpoint is eligible only if refinement does not degrade the
  baseline when `require_refinement_improvement=true`;
- inference can require a finite validated score;
- sidecar metadata carries a training-run identifier, and checkpoint selection
  refuses a stale `best.ckpt` from another run.

**Affected files.** `finetune/aurora_finetune_distributed.py`,
`finetune/aurora_finetune_utils.py`, both rollout notebooks, global O₃ YAML.

### 7. Scheduler horizon ignored gradient accumulation

**Issue.** With `accumulation_steps=32`, the scheduler treated every batch as
an optimizer update. The intended cosine schedule was approximately 32 times
too long; the saved history still showed learning rate near `3e-4` late in
training.

**Change.** Scheduler length is computed after world-size sharding, batch
grouping, and gradient accumulation. It advances only when an optimizer update
actually occurs.

**Affected file.** `finetune/aurora_finetune_distributed.py`.

### 8. Output-only smoothing changed the evaluated product

**Issue.** `smooth_sigma=1` modified only the saved refined output, not the
training target or baseline. This confounded model quality with an extra
post-processing filter.

**Change.** The global O₃ recipe sets smoothing to zero. Smoothing remains an
explicit optional experiment rather than an implicit part of refinement.

**Affected file.** Global O₃ YAML.

### 9. Checkpoint and preprocessing contract was incomplete

**Issue.** A compatible tensor shape did not prove compatible target order,
level order, longitude behavior, residual scaling, norm statistics, or
validation provenance. Inference could recompute missing normalization and
continue.

**Change.** Checkpoint loading validates target/level signatures,
architecture-affecting options, longitude behavior, normalization keys and
sizes, positive scales, and (when requested) validation provenance. Inference
raises instead of silently recomputing or swallowing restoration failures.
The corrected global recipe declares `flow_refine_contract_version: 3`, so
version-1 endpoint/scaling checkpoints and version-2 checkpoints without the
forecast-lead embedding are rejected before tensor loading.

**Affected files.** `finetune/aurora_finetune_utils.py`,
`finetune/aurora_inference_rollout.ipynb`,
`finetune/aurora_finetune_rollout.ipynb`.

### 10. Alignment and missing values

**Finding.** Ground truth and rollouts use different level ordering, and the
source test grid has one extra `-90°` latitude that is cropped for the model's
patch size. The evaluator and diagnostic tools match named coordinate values,
valid times, and level values. They explicitly intersect the common latitude
coordinates and perform no interpolation, resizing, or truncation by array
position. The reviewed test targets contained no NaNs.

**Change.** Flow loss now receives the same configured/finite mask as the
supervised loss. Diagnostics fail on ambiguous coordinate matches.

**Affected files.** `finetune/aurora_finetune_utils.py`,
`finetune/flow_refine.py`, `finetune/diagnose_refinement.py`.

### 11. Forecast lead was not a model input

**Issue.** Training knew the integer rollout step, and inference knew the
rollout loop index, but neither passed cumulative forecast age into the flow
head. Flow time `t` was incorrectly the only temporal input; it cannot identify
whether an otherwise similar field is a 12-hour or 72-hour Aurora forecast.
The existing full evaluation confirms the premise: mean baseline RMSE rises
from `6.33e-8` to `1.34e-7` for `go3` and from `1.23e-4` to `2.66e-4` for
`gtco3` between 12 and 72 hours.

**Change.** Each head now has a separate zero-initialized lead MLP. Training
computes exact hours from `valid_time - initialization_time` for every sample,
validates them against `rollout_step_hours`, and repeats the per-sample value
across pressure levels. Inference passes `step * rollout_step_hours`, validates
the dataset cadence, and refuses to exceed the configured trained support.
Lead conditioning and its hour scale are part of checkpoint contract version 3.

**Why correct.** The conditional target is now `E[r | y_hat, lead]`, allowing
correction mean, amplitude, and spatial structure to differ by forecast age.
Keeping this input separate from flow interpolation time preserves the
rectified-flow formulation.

**Affected files.** `finetune/flow_refine.py`,
`finetune/aurora_finetune_utils.py`,
`finetune/run_refinement_ablation.py`, global O3 YAML.

### 12. Batched future exogenous fields were misaligned

**Issue.** Autoregressive training used `sample_list[0].anchor_index` to load
one future CAMS context frame, then repeated it across the whole batch. With
different initialization times, samples after the first received the wrong
future predictors; the mismatch accumulates at later leads.

**Change.** Future frames are loaded from each sample’s own anchor-plus-lead time
and stacked in batch order. A regression test uses distinct time-coded values
to prove that the two batch items remain distinct.

**Why correct.** Every autoregressive trajectory now uses exogenous data at its
own valid time, while target variables still come exclusively from the model
and never leak future CAMS truth.

**Affected file.** `finetune/aurora_finetune_utils.py`.

## Why correlation improved while magnitude worsened

Correlation is unchanged by a constant offset and is relatively insensitive to
a positive amplitude rescaling. MAE/RMSE are not. The full old rollout
diagnostic found the following mean changes across requested leads:

| target | RMSE improvement | MAE improvement | correlation change | correction RMS / true residual RMS |
|---|---:|---:|---:|---:|
| go3 1000 hPa | -7.41% | -14.69% | +0.0108 | 0.84 |
| go3 500 hPa | -2.33% | -8.57% | +0.0279 | 0.80 |
| go3 100 hPa | -49.52% | -60.72% | -0.0035 | 1.51 |
| go3 50 hPa | -57.87% | -69.46% | +0.0017 | 1.59 |
| gtco3 surface | -13.28% | -18.04% | +0.0033 | 1.09 |

Negative improvement means the refined error is larger. The upper-level
correction has substantially more RMS energy than the true residual. For
`gtco3`, mean bias moved from about `+3.9e-5` to `-1.1e-4`, reversing sign and
increasing absolute bias. These are amplitude/offset failures even where the
correlation change is positive.

The full result contains 183 common initializations; 1,077
initialization/lead combinations had matching truth after end-of-period cases
were excluded.

## Representative safeguard evaluation

Because the old checkpoint was never trained at `t=0, x0=0`, merely changing
its query endpoint makes it worse (RMSE changes of roughly `-7.5%` to `-99%`
on the three representative test initializations). This is expected and is
why the corrected objective requires a fresh checkpoint.

To test a non-arbitrary degradation safeguard with the existing learned
correction, four training-tail initializations (2024-06-21, 23, 25, and 27)
were used to fit a separate scale for each variable, level, and lead:

```text
alpha = clip(sum(c * r) / sum(c^2), 0, 1)
c = refined - baseline
r = truth - baseline
```

Three test initializations (2024-07-01, 2024-08-08, and 2024-09-14) were then
evaluated without using their truth to choose `alpha`.

| target | old deterministic RMSE improvement | gated RMSE improvement | gated MAE improvement | gated correlation change |
|---|---:|---:|---:|---:|
| go3 1000 hPa | +0.13% | +3.66% | +3.20% | +0.0092 |
| go3 500 hPa | +7.58% | +11.10% | +7.90% | +0.0297 |
| go3 100 hPa | -2.68% | +2.46% | +1.78% | +0.0017 |
| go3 50 hPa | -20.69% | 0.00% | 0.00% | 0.0000 |
| gtco3 surface | -0.77% | +9.29% | +9.93% | +0.0042 |

At 50 hPa the calibration correction was anti-correlated, so `alpha=0`
selected the unchanged baseline. This is an honest abstention, not a claimed
improvement. At `gtco3`, the test mean bias magnitude fell from about
`3.9e-5` to `3.3e-5`.

This ablation demonstrates that the correction contains useful spatial signal
but needs amplitude control. It does **not** validate the newly trained
source-endpoint model; a full fresh training run is required for that claim.

## Diagnostics and outputs

The non-overwriting review outputs are under:

```text
finetune/outputs/O3_global_3day_lead/evaluation/refinement_review/
```

Important subdirectories:

- `legacy_forced8_full_diagnostics/`: full historical diagnosis, including
  per-case/per-lead CSV files, aggregate correction fields, eight-panel maps,
  correction scatterplots, and lead-time plots;
- `corrected_source_mean_diagnostics/`: proof that the old checkpoint cannot
  be retrofitted to the corrected endpoint;
- `legacy_target_zero_diagnostics/`: deterministic old-endpoint ablation;
- `calibration_train_tail_rollouts/`: held-out calibration forecasts;
- `validation_gated_legacy/`: fitted scales, calibrated test rollouts, and
  representative diagnostics.

`finetune/diagnose_refinement.py` reports:

- bias, MAE, RMSE, centered RMSE, spatial correlation;
- truth/baseline/refined spatial standard deviation and amplitude ratio;
- correction/residual correlation and RMS ratio;
- residual sign accuracy;
- fractions of grid points improved, worsened, or unchanged and fractions of
  forecast cases with lower/higher RMSE;
- least-squares optimal correction scale.

Its eight-panel maps show ground truth, baseline, refined prediction, both
errors, true residual, predicted correction, and remaining residual.

`finetune/calibrate_refinement_gate.py` fits scales only from explicitly
provided calibration truth and writes new rollouts without modifying inputs.

## Tests

Focused tests in `tests/test_flow_refine_contract.py` cover:

- deterministic sampling at `t=0, x0=0`;
- exact deterministic-reconstruction and degradation-hinge arithmetic;
- missing-value masks;
- per-pressure-level residual z-scoring;
- cumulative lead propagation through training, atmospheric levels, and rollout inference;
- exact initialization-to-valid-time hour conversion and cadence rejection;
- per-sample batched future-predictor alignment;
- optimizer-update scheduler horizon;
- rejection of unvalidated checkpoints;
- rejection of stale best checkpoints and acceptance of same-run validated
  checkpoints.

The focused suite plus existing data-path and longitude tests passes.

## Remaining limitations and required next run

1. The old checkpoint cannot be used to assess the corrected source-endpoint
   objective. Start a fresh run as configured; do not resume it.
2. A shared atmospheric head processes levels independently. It has no explicit
   pressure-level embedding, so it can only infer level identity from the
   conditioned field distribution. A future architecture should add pressure
   conditioning before expecting one head to transfer cleanly across levels.
3. The head now receives explicit cumulative forecast hours, but the
   architecture does not impose monotonic correction magnitude. This is
   intentional: baseline error often grows with lead, but the correct sign and
   magnitude remain variable-, level-, case-, and location-dependent. Validate
   every trained lead separately and retrain before extending beyond 72 hours.
4. The validation gate uses a scalar per variable/level/lead. It cannot fix a
   correction whose appropriate strength varies spatially. A learned bounded
   gate is a future option, but must be trained with the same degradation and
   validation constraints.
5. Four calibration and three test initializations are sufficient for the
   representative ablation, not for production scale selection. Production
   gating needs the complete training-tail validation period.
6. Promotion criteria should require positive aggregate RMSE/MAE improvement,
   non-increasing absolute bias, acceptable centered RMSE/variance, and no
   material variable/level/lead regressions. Correlation alone is insufficient.

After fresh training, run deterministic one-member inference, then
`diagnose_refinement.py` over the full test period. Only the same-run validated
best checkpoint should be evaluated or promoted.
