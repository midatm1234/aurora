# Aurora two-phase stochastic residual refinement

This document describes the unified stochastic residual-refinement framework
added on the `aurora_finetune_stochastic_refinement` branch, on top of the
existing `aurora_finetune_flow_matching` bias-correction workflow.

---

## 1. Scientific framing (read this first)

Aurora is a **forecasting and rollout** model. Every refined field belongs to a
specific forecast initialization time, forecast valid time and forecast lead
time, and those never change here.

Refinement is **postprocessing of each deterministic rollout step**:

```
Aurora state at step n
   -> deterministic Aurora prediction for step n+1     (Phase 1)
   -> optional stochastic residual correction for n+1  (Phase 2)
```

The refined field is returned *alongside* the deterministic prediction. The
deterministic state that produces later rollout steps is **not** replaced, so
the rollout trajectory of a refined run is identical to the unrefined one.

Three quantities are kept strictly separate and never share an embedding or a
configuration field:

| symbol | meaning | is **not** |
| --- | --- | --- |
| forecast lead time (hours) | physical time between initialization and valid time | a process variable |
| diffusion timestep `k` in `{0..K-1}` | index of a Gaussian noising process | a lead time or a timestamp |
| flow interpolation time `t` in `[0,1]` | integration coordinate of a probability path | a lead time or a timestamp |

`tests/test_refinement_transformer.py`, `tests/test_refinement_models.py` and
`tests/test_refinement_target_space.py` enforce these properties automatically.

---

## 2. Architecture

```
              +----------------- Phase 1 (deterministic) ------------------+
 batch  --->  | existing Aurora model + existing autoregressive rollout     | ---> rollout_phys
              +------------------------------------------------------------+
                                     |  encode()  (Aurora normalized space)
                                     v
              +----------------- Phase 2 (optional) -----------------------+
 conditioning | flow_matching_unet | flow_matching_conv_unet                | ---> r_hat
 (section 5)  | flow_matching_transformer | diffusion_unet / _transformer   |
              +------------------------------------------------------------+
                                     |
       refined_norm = rollout_norm + r_hat  ->  decode once  ->  constraints once
```

### Phase 1 — deterministic Aurora

Unchanged. Same architecture, checkpoints, input history, autoregressive
rollout, variable/level ordering, forecast timing, normalization and
postprocessing as `aurora_finetune_flow_matching`.

### Phase 2 — optional stochastic residual refinement

Selected with `model.refinement.type`:

| value | model | process variable | backend |
| --- | --- | --- | --- |
| `none` | *(disabled)* | – | – |
| `flow_matching_unet` (alias `flow_matching`) | the **existing** Aurora `ResidualFlowUNet` per-variable heads | flow time | legacy |
| `flow_matching_transformer` | spatial-token Transformer velocity network | flow time | unified |
| `flow_matching_conv_unet` (alias `flow_matching_conv`) | unified packed-field conditional UNet | flow time | unified |
| `diffusion_unet` | conditional convolutional UNet, DDPM training / DDIM sampling | diffusion timestep | unified |
| `diffusion_transformer` | spatial-token Transformer denoiser, DDPM/DDIM | diffusion timestep | unified |

The five options are **alternatives**. Diffusion and flow matching are never
stacked, and flow matching does not require diffusion to run first.
Construction goes through a registry
(`finetune.refinement.base.build_refiner`), so training, validation, inference,
evaluation, checkpoint loading and output writing contain no per-type
`if/elif` chains.

`flow_matching_unet` is the *legacy backend*: it keeps using the untouched
`finetune.aurora_finetune_utils.maybe_wrap_flow_refine` /
`finetune.flow_refine.AuroraFlowRefine` code path, so its numerics and its
checkpoints are preserved exactly. It is also exposed through the common
interface by `finetune.refinement.legacy_flow.LegacyFlowMatchingUNetRefiner`,
which *delegates* to the same `AuroraFlowRefine` module rather than
reimplementing it.

`flow_matching_conv_unet` is deliberately a different type. It uses the same
packed target space, conditioning, residual scaler, unified losses and
checkpoint contract as the other unified refiners, with a convolutional UNet
backbone. It is not an alias for, nor checkpoint-compatible with, the legacy
per-variable `AuroraFlowRefine` wrapper.

---

## 3. Reference implementation and attribution

The unified two-phase design is adapted from the Prithvi stochastic
residual-refinement reference implementation:

> <https://github.com/midatm1234/Prithvi-UNet-stocahstic>, branch
> `Prithvi-UNet-stochastic_refinement`, package `granitewxc.refinement`
> (Apache-2.0). See `docs/STOCHASTIC_REFINEMENT.md` in that repository.

Ported or closely adapted, with attribution in each module docstring:

| Aurora module | Prithvi source |
| --- | --- |
| `finetune/refinement/base.py` | `granitewxc/refinement/base.py` (registry, `ResidualRefiner`, `ChunkNoiseSource`, `masked_loss`) |
| `finetune/refinement/config.py` | `granitewxc/refinement/config.py` (schema, aliases, validation) |
| `finetune/refinement/target_space.py` | `granitewxc/refinement/target_space.py` (`NormalizedTargetSpace`) |
| `finetune/refinement/schedules.py` | `granitewxc/refinement/schedules.py` (`DiffusionSchedule`, `ProcessTimeEmbedding`) |
| `finetune/refinement/backbones.py` | `granitewxc/refinement/backbones.py` (`ConditionalResidualUNet`, `SpatialResidualTransformer`, patchify/unpatchify, 2-D positional encoding) |
| `finetune/refinement/diffusion.py` | `granitewxc/refinement/diffusion.py` |
| `finetune/refinement/flow_matching.py` | `granitewxc/refinement/flow_matching.py` |
| `finetune/refinement/two_phase.py` | `granitewxc/refinement/two_phase.py` |
| `finetune/refinement/checkpoint.py` | `granitewxc/refinement/checkpoint.py` |
| `finetune/refinement/io.py` | `granitewxc/refinement/io.py` |
| `finetune/refinement/cache.py` | `granitewxc/refinement/cache.py` |

**Aurora-specific adaptations** (not present in the reference, which is a
spatial-downscaling problem with no forecast lead time):

* forecast lead time as an explicit conditioning input with its own
  `LeadTimeEmbedding`, separate from the process-time embedding;
* rollout-step bookkeeping (`LeadStepBuffer`) that folds lead times into the
  effective batch dimension in ascending order while preserving the
  initialization/valid/lead-time triple;
* a variable **and pressure-level** packing abstraction (`FieldPacking`);
* a target space derived from Aurora's own per-variable / per-level location and
  scale constants instead of a downscaling head's output scalers;
* the `existing_aurora` flow-matching interpolation path (`x1`/data
  parameterisation), preserved as the default;
* `windowed_2d` spatial attention with longitude periodicity;
* bias-aware auxiliary losses grouped by variable, pressure level and forecast
  lead time;
* the rollout cache key (initialization time, valid time, lead time, rollout
  interval, input history);
* NetCDF output with `rollout_step`, `init_time`, `time` and `lead_time_hours`
  coordinates and a pressure-level dimension.

Prithvi-specific data loaders, model wrappers, target variables, same-time
predictor assumptions and checkpoint names were **not** copied.

---

## 4. Residual target and reconstruction

The repository uses one correction sign and one output meaning everywhere:

```python
correction_target_physical = cams_target - aurora_forecast
predicted_correction_physical = refinement.predict_correction(...)
refined_forecast = aurora_forecast + predicted_correction_physical
```

A refiner never returns a complete atmospheric field, diffusion epsilon, flow
velocity, normalized tensor, or latent state through the forecast interface. It
returns the explicitly denormalized physical correction shown above. Internally,
training and sampling happen in **Aurora's normalized target space**
(`finetune.refinement.target_space.NormalizedTargetSpace`), which is derived
from the same statistics the supervised loss already uses
(`compute_target_normalization_stats`, i.e. `aurora.normalisation` locations and
scales, per variable and per pressure level).

```
rollout_norm    = encode(aurora_rollout)        # physical -> normalized
target_norm     = encode(ground_truth)          # physical -> normalized
residual_target = target_norm - rollout_norm    # ONE space

predicted_residual = refiner(...)               # same space
refined_norm       = rollout_norm + predicted_residual
refined_physical   = decode(refined_norm)       # inverse-normalize ONCE
refined_physical   = apply_constraints(...)     # constraints ONCE
refined_physical   = apply_mask(...)            # mask ONCE
```

Guarantees, all covered by `tests/test_refinement_target_space.py`:

* `decode(encode(x)) ≈ x` and `encode(decode(n)) ≈ n` within `rtol = 1e-5`;
* a normalized residual is never added to a physical field;
* physical and normalized fields are never subtracted from each other;
* inverse normalization, the physical constraints and the mask are each applied
  exactly once;
* invalid target cells get exactly zero residual, are excluded from the masked
  loss denominator, and stay `NaN` in the output.

Because the Aurora supervised loss already normalizes the prediction and the
target with the same statistics, the training path uses
`residual_target_from_normalized`, which avoids a redundant decode/encode round
trip and therefore any risk of double normalization.

### Optional residual scaling

Unified refiners can map that normalized residual into a better-conditioned
generative space with `target_space.residual_scaling`: `none` is an exact
identity, `global` uses one scale, and `per_channel` uses one scale for each
packed variable/pressure-level channel. `auto` currently resolves to
`per_channel`. The transform is inverted before reconstruction, so the refiner
still returns an Aurora-normalized residual.

The legacy-safe default is `none`; existing YAMLs and checkpoints therefore do
not silently gain scaler state. New experiments must opt in explicitly. Active
scalers estimate masked, finite training-only statistics, preserve channels
with no observations, and store their calibration state in checkpoints.
Validation and inference never update it.

The production distributed trainer calibrates before optimizer step 1 with a
deterministic pass over the complete training split. Ranks consume disjoint,
unpadded sample shards, accumulate finite/masked per-channel count, sum and
sum-of-squares locally, and perform one final sufficient-statistic all-reduce.
The resulting transform is frozen and checkpointed together with the logical
sample count and a SHA-256 split/coordinate/channel fingerprint. Resume and
inference reject missing, partial or mismatched calibration. The
`residual_scaling_warmup_batches` and momentum keys remain only for standalone
legacy compatibility; production calibration does not select the first N
shuffled optimization batches and never trains under a moving transform.

### NO2 Diffusion Transformer failure: root cause and migration

The pre-audit US-WEST NO2 Diffusion Transformer run is not a scientifically
valid checkpoint. Its repeated speckle and horizontal bands were systematic,
not plausible stochastic uncertainty. The same configured seed replayed the
same initial diffusion latent for every forecast initialization, so fixed-grid
noise survived the mean over 1,083 cases. Four additional configuration and
selection problems amplified that failure:

* it combined an epsilon-prediction objective with a zero-initialized output
  projection. A zero epsilon estimate is not a zero correction: converting it
  to x0 divides the retained Gaussian latent by `sqrt(alpha_bar)`, creating a
  large random correction before the head has learned anything;
* it mixed a small CAMS-minus-Aurora correction with unit-variance diffusion
  noise without centered per-channel correction scaling, strongly imbalancing
  `tcno2` and the pressure-level `no2` channels;
* all direct deterministic, bias, structure, tail, and degradation loss weights
  were zero, while temporal Mamba was enabled at full weight, so the deployed
  forecast field was not the quantity principally supervised;
* checkpoint acceptance used an aggregate validation loss that could hide
  physical-space degradation in individual variable/level/lead channels.

That diagnosis is consistent with the saved pre-audit held-out artifacts: the
reported RMSE improvement relative to Aurora was -42.451% for `tcno2`, -83.769%
for `no2` at 1000 hPa, -52.376% for `no2` at 925 hPa, and -61.229%
for `no2` at 850 hPa (negative means worse), despite checkpoint metadata reporting a
positive aggregate validation improvement. Those numbers are a failure
baseline, not post-fix results.

For a new unified refinement run, the shipped YAMLs now require the following
migration profile:

1. keep `train_on_residual: true` under the explicit
   `correction_target = CAMS - Aurora` contract and form the refined forecast
   by addition;
2. fit centered `per_channel` correction scaling on masked finite training data
   and restore its checkpointed mean and scale exactly once;
3. use clean-sample/x0 prediction for diffusion with zero-initialized heads;
4. deploy the deterministic conditional mean by default (`ensemble_size: 1`),
   and use initialization- and member-specific seeds with at least two members
   only for an explicitly stochastic uncertainty product;
5. keep temporal Mamba disabled unless its separate causal objective is
   explicitly trained and validated;
6. select checkpoints by `mean_physical_rmse_ratio`, require improvement over
   matched Aurora, and require every physical variable/level/lead channel to
   improve on the purged training-tail validation split.

Do not resume the former NO2 epsilon checkpoint: prediction type, residual
scaler state, deployed estimator, temporal architecture, and checkpoint metric
are part of the scientific contract. Start a clean run. The known-good legacy
flow-matching YAMLs and checkpoints retain their old defaults and are not
silently migrated.

For US-WEST NO2, use one consolidated unified file for new runs. Keep the
known-good old flow file only when reproducing its legacy checkpoint contract:

| requested head | safe NO2 recipe |
| --- | --- |
| flow-matching convolutional U-Net (new unified default) | consolidated file below with `model.refinement.type: flow_matching_conv_unet` |
| flow-matching Transformer | consolidated file below with `model.refinement.type: flow_matching_transformer` |
| diffusion U-Net | consolidated file below with `model.refinement.type: diffusion_unet` |
| Diffusion Transformer | `finetune/aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml` with its fresh `_corrected` case name |
| legacy flow-matching U-Net (compatibility only) | `finetune/aurora_NO2_finetune_US-WEST_3day_lead_config.yaml`; do not treat this as the unified safe default |

The consolidated file contains explicit `flow_matching`, `diffusion`, `unet`,
and `transformer` blocks; its correction convention, scaler, deterministic
mean, losses, conditioning, physical constraints, seeding, and checkpoint gate
are shared by the four unified heads. Each head is a distinct checkpoint
contract: give switched runs a fresh `case_name`, keep `resume_training: false`,
and never load weights produced by another type.

---

## 5. Conditioning

`model.refinement.conditioning` selects which fields are concatenated along the
channel axis and aligned to the target grid:

| key | default | content |
| --- | --- | --- |
| `aurora_rollout` | `true` | deterministic Aurora rollout in normalized target space |
| `aurora_input_state` | `false` | the Aurora *input* state of the rollout step |
| `aurora_features` | `false` | not implemented (raises); Aurora exposes no stable frozen feature map |
| `static_fields` | `true` | static/orography-style fields |
| `masks` | `true` | validity of the inputs (**never** the target mask) |
| `forecast_lead_time` | `true` | scalar hours, injected through a separate embedding |

Never used as conditioning: future target values, future Aurora states
unavailable at inference, and any statistic derived from the inference-period
target.

---

## 6. Flow matching

### Existing Aurora formulation (default, `interpolation_path: existing_aurora`)

* source `x0 ~ N(0, I)`;
* path `x_t = (1 - t) * x0 + t * r`;
* flow time `t = sigmoid(N(-0.5, 1.2))`, clamped to `[sigma_min, 1 - sigma_min]`;
* the network predicts the **clean residual** `r` (the `x1`/data
  parameterisation), trained with a masked MSE against `r`;
* `integration_steps == 1` is a single deterministic query at the source mean
  (`x_t = 0`, `t = 0`), which is the squared-error optimum `E[r | rollout]`;
* more steps run the DDIM-style straight-line update.

### Prithvi/rectified formulation (`interpolation_path: rectified_flow`)

* the network regresses the velocity `u_t = r - x0`;
* inference integrates `dx/dt = v_theta` from `t = 0` to `t = 1` with `euler`,
  `midpoint` or `heun`.

**The two formulations differ**, and the Aurora one is the default. The legacy
`flow_matching_unet` type accepts *only* `existing_aurora` (any other value is a
configuration error), so a legacy configuration or checkpoint can never be
silently switched to the Prithvi formulation. `rectified_flow` is available on
the unified flow refiners as an explicit, documented choice.

---

## 7. Diffusion

Trained on the same normalized residual:

```
noisy_residual = sqrt(alpha_bar_k) * residual_target
               + sqrt(1 - alpha_bar_k) * noise
```

* prediction types: `epsilon`, `velocity`, `sample`, with a single set of
  conversion helpers (`to_clean` / `to_epsilon`) used by the loss, the clean
  residual reconstruction, the auxiliary bias losses and the sampler;
* schedules: `cosine`, `linear`, `scaled_linear`;
* sampling: DDIM with configurable `inference_steps`, `eta`, generator and
  ensemble size; `sampler: ddpm` requires `eta: 1.0` and gives the ancestral
  update;
* schedule coefficients are non-persistent buffers built once per configuration
  and kept on the target device, so nothing is rebuilt inside a sampling loop
  and nothing stale can be restored from a checkpoint;
* `snr_weighting` supports `none`, `min_snr`, `snr`, `truncated_snr` and
  `auto`; `timestep_distribution` supports `uniform`, `low_noise`,
  `high_noise` and `auto`, with `timestep_bias` controlling the bias strength;
* `deterministic_estimator` selects `ode`, `posterior_mean` or `auto`, and
  `deterministic_training_steps` bounds the differentiable trajectory used by
  `loss.deterministic_weight`; epsilon prediction with `posterior_mean` is
  rejected because the clean-sample conversion is ill-conditioned there;
* backward-compatible omitted-key defaults are `prediction_type: epsilon`,
  `snr_weighting: none`, and `timestep_distribution: uniform`. The example
  configurations opt in explicitly to improved settings such as sample
  prediction and automatic weighting/sampling; these are scientific choices,
  not migrations applied to old checkpoints.

Nothing is silently reduced: the configured step count, schedule, prediction
type, `eta` and stochastic initialization are used verbatim
(`tests/test_refinement_performance.py::test_step_counts_are_never_silently_reduced`).

---

## 8. Spatial-only Transformers

`SpatialResidualTransformer` pipeline, per sample **and per forecast lead time**:

1. pack the residual state and the configured conditioning;
2. align everything to the target lat/lon grid;
3. concatenate along the channel dimension;
4. pad the trailing edges to a multiple of the patch size (circular in
   longitude when the domain is periodic, replicate otherwise);
5. patchify into `grid_h x grid_w` 2-D tokens in row-major `(lat, lon)` order;
6. project tokens to the embedding dimension;
7. add the 2-D positional encoding (`latlon_2d` separable learned tables, or
   fixed `sincos_2d`);
8. apply spatial self-attention blocks;
9. inject the process time (diffusion timestep or flow time) through adaptive
   layer norm;
10. inject the forecast lead time **separately**, through its own embedding
    added to the process-time embedding;
11. project tokens back to the residual/velocity field;
12. unpatchify;
13. crop exactly to the original `(H, W)`.

`transformer.local_refinement: true` adds an overlapping 3-by-3 convolutional
stem before patchification and a zero-initialized local residual head after
unpatchification. It defaults to `false` so older Transformer checkpoints keep
their original architecture.

### Attention modes

| mode | behaviour |
| --- | --- |
| `global_2d` | exact full bidirectional attention over all spatial tokens |
| `windowed_2d` | exact attention restricted to (optionally shifted) 2-D token windows |

`windowed_2d` handles partial edge windows by padding the token grid and
excluding padded keys from every softmax; shifted windows use a circular roll
with a region mask so wrapped and non-wrapped content never attend to each
other across a seam. When the longitude axis is periodic, the longitude padding
is circular (real tokens acting as keys only, cropped from the output), so
wrap-around adjacency is genuine and no dateline seam appears. Selecting a
window that divides the token grid avoids the partial-window case entirely; a
`RuntimeWarning` is emitted once when it does not.

Explicitly absent, by design and enforced by tests: temporal self-attention,
cross-time attention, cross-lead-time attention, causal temporal masks,
autoregressive Transformer decoding, recurrent temporal state and temporal token
sequences. When a tensor carries several rollout lead times, the caller folds the
lead-time axis into the effective batch dimension, so lead-time items are
processed completely independently.

`optimized_attention` chooses between PyTorch's fused
`scaled_dot_product_attention` (`sdpa`) and an explicit reference implementation
(`math`). Both compute the *same* operation; no sparse, local or linearised
approximation is ever substituted for a requested exact mode.

### Optional temporal Mamba wrapper

The spatial Transformer itself remains per-lead. Setting the existing top-level
`model.mamba_temporal_enabled: true` adds `PackedMambaTemporalAdapter` around an
active refiner, including `diffusion_transformer`. The same wrapper also works
with the unified diffusion/flow UNets, `flow_matching_transformer`, and the
legacy `flow_matching_unet` path. It does not turn the Transformer tokens into a
temporal attention sequence: it applies a separate causal selective state-space
model over the ordered rollout-lead axis at every pixel after spatial
refinement.

```yaml
model:
  refinement:
    enabled: true
    type: diffusion_transformer
  mamba_temporal_enabled: true         # optional; default false
  mamba_temporal_channels: 16
  mamba_temporal_state: 8
  mamba_temporal_layers: 2
  mamba_temporal_conv: 3
  mamba_temporal_expand: 2

training:
  mamba_temporal_weight: 1.0
```

Temporal training consumes target-independent spatial-refiner outputs in
lead-major order. Those frames are evaluated with the same deterministic or
stochastic inference mode configured for rollout and detached, so the temporal
loss updates Mamba without back-propagating into the spatial refiner. Inference
maintains a growing causal history separately for every ensemble member before
forming ensemble statistics.

Enabling Mamba requires at least two consecutive `data.target_lead_times`
starting at 1, a positive `rollout.rollout_step_hours`, a rollout horizon no
longer than the trained temporal horizon, and
`training.mamba_temporal_weight > 0`. Its decoder is zero-initialized, so a new
module starts as an identity correction. The enabled flag and all architecture
values are part of the checkpoint contract; inference must reconstruct the
same temporal architecture. With `mamba_temporal_enabled: false` (the default),
the spatial head retains its existing per-lead behavior and checkpoint state.

Forecast-lead-time conditioning and temporal Mamba solve different problems:
the former tells one spatial pass which physical lead it represents, whereas
the latter explicitly models dependencies among the sequence of predicted
leads.

---

## 9. Variable and pressure-level packing

`finetune.refinement.packing.FieldPacking` is the single channel mapping used by
every refiner, by training, validation, inference, evaluation, NetCDF output and
checkpoint metadata. Channel order is deterministic:

1. surface targets, in `resolved_specs.targets` order;
2. atmospheric targets, in the same order, expanded over their configured
   (loss-)levels.

Each `ChannelSpec` records the variable name, surface/atmospheric type, pressure
level, tensor channel, units, normalization method, normalization statistics and
level index; the packing additionally records latitude, longitude, the rollout
lead times and the lead-time conditioning scale. Round-trip tests assert that
values, variable names, pressure levels, channel order, spatial dimensions,
coordinates and units all survive `pack` → `unpack`.

---

## 10. Bias-aware auxiliary losses

```
total_loss = generative_loss
           + reconstruction_weight        * reconstruction_loss
           + bias_weight                  * bias_loss
           + gradient_weight              * gradient_loss
           + pattern_correlation_weight   * pattern_correlation_loss
           + deterministic_weight     * deterministic_endpoint_loss
           + mae/extreme/peak/quantile/variance/spectral terms
           + degradation_weight       * over-correction_hinge
           + magnitude_weight         * residual_magnitude
```

* all weights default to `0.0`, so an existing configuration keeps its exact
  objective; the legacy `flow_matching_unet` type *rejects* non-zero weights and
  keeps its `training.flow_aux_loss` objective;
* diffusion reconstructs the clean residual from the configured prediction
  parameterisation before computing auxiliary terms;
* flow matching derives the endpoint estimate consistently with the selected
  path (`r_hat` directly for `existing_aurora`, `x_t + (1 - t) * v` for
  `rectified_flow`);
* the mean bias is accumulated **separately** per configured group (variable,
  pressure level, forecast lead time) and only then aggregated, so unrelated
  errors cannot cancel;
* cosine-latitude area weighting is applied when a latitude vector is available;
* every component is returned and logged separately;
* `mae_weight` targets absolute error; `extreme_weight`/`extreme_quantile`/
  `extreme_intensity`, `peak_weight`, `quantile_weight`, `variance_weight` and
  `spectral_weight` expose complementary tail, distribution and structure
  objectives;
* `degradation_weight` penalizes corrections that make a valid cell worse than
  the unchanged rollout, while `magnitude_weight` regularizes correction size;
  `aux_on_deterministic` applies structural terms to the deterministic endpoint
  that inference emits.

---

## 11. Checkpoint compatibility

* Existing Aurora pretrained, fine-tuned and flow-matching checkpoints are
  read-only inputs and are never converted in place.
* Aurora keys are loaded **strictly** after an explicit prefix migration
  (`aurora.`); the only tolerated missing keys are those of the newly introduced
  `refiner.*` module. Missing Aurora keys, unexpected keys and shape mismatches
  all raise.
* Legacy `AuroraFlowRefine` checkpoints map explicitly: `base.* -> aurora.*` and
  `surf_flow.* / atmos_flow.* / _res_std__* -> refiner.legacy.*`.
* Refinement-only checkpoints record the Aurora checkpoint path and SHA-256
  fingerprint, the refinement type, the variable/level packing, the target-space
  normalization, the forecast lead-time configuration, the resolved
  configuration, the precision settings, optimizer/scheduler/scaler state,
  epoch, global step and RNG states.
* `save_checkpoint` additionally records `resolved_refinement_config`,
  `resolved_performance_config`, `refinement_backend`, `refinement_type`,
  `rng_state` and (for the unified backend) `field_packing` and
  `aurora_fingerprint`.
* Checkpoints are written atomically through a same-directory temporary file.

### Legacy numerical parity

`finetune/refinement_parity_check.py` is a fixed regression case: a real
flow-matching checkpoint, its recorded configuration, a fixed synthetic input
and target batch, fixed device, fixed float32 precision, a fixed seed, a fixed
initial stochastic sample and a fixed number of flow-integration steps. It
compares the untouched `AuroraFlowRefine` against the same weights loaded
through the checkpoint migration into the unified adapter.

```
python -m finetune.refinement_parity_check \
    --checkpoint finetune/outputs/checkpoints/O3_global_3day_lead/best.ckpt \
    --report /tmp/flow_parity.json
```

Documented tolerance: tensors bitwise identical (`atol = 0`); the scalar loss
within `1e-6` absolute/relative, because the two call sites aggregate the
per-variable losses slightly differently.

---

## 12. Configuration schema

The block below is the recommended safe profile for a **new unified run**, not
a statement of omitted-key defaults. Backward-compatible defaults are listed
after the block and remain unchanged for legacy configurations/checkpoints.

```yaml
model:
  refinement:
    enabled: true
    type: diffusion_transformer       # none | flow_matching_unet | flow_matching_conv_unet |
                                      # flow_matching_transformer | diffusion_unet |
                                      # diffusion_transformer (aliases also accepted)
    correction_convention: cams_minus_aurora_add
    checkpoint: null
    freeze_aurora: true
    joint_finetuning: false
    # correction_target = CAMS - Aurora
    # refined_forecast = Aurora + predicted_correction
    train_on_residual: true
    feedback_to_rollout: false        # EXPERIMENTAL when true
    seed: 1234
    ensemble_size: 1                  # deterministic default product
    deterministic_inference: true

    target_space:
      use_existing_normalization: true
      residual_space: normalized
      residual_scaling: per_channel   # none | global | per_channel | auto
      residual_scaling_center: true
      residual_scaling_momentum: 0.05
      residual_scaling_warmup_batches: 32
      residual_scaling_target_std: 1.0

    conditioning:
      aurora_rollout: true
      aurora_input_state: false
      aurora_features: false
      static_fields: true
      masks: true
      forecast_lead_time: true

    loss:
      generative: mse                 # mse | l1 | huber
      reconstruction_weight: 0.0
      bias_weight: 1.0
      gradient_weight: 0.25
      pattern_correlation_weight: 0.5
      area_weighted: true
      deterministic_weight: 1.0      # directly supervise deployed mean correction
      mae_weight: 0.0
      extreme_weight: 1.0
      extreme_quantile: 0.95
      extreme_intensity: 4.0
      peak_weight: 0.0
      quantile_weight: 1.0
      variance_weight: 2.0
      spectral_weight: 0.0
      degradation_weight: 1.0
      magnitude_weight: 0.0
      aux_on_deterministic: true
      separate_by_variable: true
      separate_by_level: true
      separate_by_lead_time: true

    flow_matching:
      source_distribution: gaussian
      interpolation_path: existing_aurora   # or rectified_flow
      integration_steps: 50
      solver: euler                   # euler | midpoint | heun
      stochastic_initialization: true
      time_sampling: logit_normal     # uniform | logit_normal
      sigma_min: 1.0e-3
      logit_normal_mean: -0.5
      logit_normal_std: 1.2
      residual_zscore: false
      res_std_momentum: 0.99

    diffusion:
      training_timesteps: 1000
      inference_steps: 50
      prediction_type: sample         # epsilon | velocity | sample; sample = x0
      schedule: cosine                # cosine | linear | scaled_linear
      sampler: ddim                   # ddim | ddpm (ddpm requires eta: 1.0)
      eta: 0.0
      clip_sample: false
      clip_sample_range: 10.0
      snr_weighting: auto             # none | min_snr | snr | truncated_snr | auto
      snr_gamma: 5.0
      timestep_distribution: auto     # uniform | low_noise | high_noise | auto
      timestep_bias: 3.0
      deterministic_training_steps: 4
      deterministic_estimator: auto   # auto | ode | posterior_mean

    unet:
      hidden_channels: 64
      num_levels: 3
      num_residual_blocks: 2
      time_embedding_dim: 128
      dropout: 0.0
      bottleneck_attention: true
      attention_heads: 4
      zero_init_output: true

    transformer:
      patch_size: [8, 8]
      embedding_dim: 256
      num_heads: 8
      num_blocks: 6
      mlp_ratio: 4.0
      dropout: 0.0
      positional_encoding: sincos_2d  # latlon_2d | sincos_2d
      max_tokens_lat: 256
      max_tokens_lon: 512
      attention_mode: windowed_2d     # global_2d | windowed_2d
      window_size: [8, 8]
      shifted_windows: true
      gradient_checkpointing: false
      optimized_attention: auto       # auto | sdpa | math
      zero_init_output: true
      local_refinement: true

  mamba_temporal_enabled: false
  mamba_temporal_channels: 16
  mamba_temporal_state: 8
  mamba_temporal_layers: 2
  mamba_temporal_conv: 3
  mamba_temporal_expand: 2

training:
  mamba_temporal_weight: 0.0          # set > 0 only with explicit Mamba opt-in
  validation_refinement_ensemble_size: 1
  validation_source: train_tail       # grouped and purged; never held-out test.nc
  checkpoint_metric: mean_physical_rmse_ratio
  require_refinement_improvement: true
  require_all_physical_channels_improve: true

performance:
  profile: false
  dataloader: {num_workers: auto, pin_memory: true, persistent_workers: true,
               prefetch_factor: 2, non_blocking_transfer: true}
  precision: {mode: fp32, allow_tf32: false}
  compile: {enabled: false, aurora: false, refinement: false, mode: default}
  rollout_cache: {enabled: false, path: null, validate_cache: true}
  conditioning_cache: {enabled: false, path: null, validate_cache: true}
  ensemble: {batch_members: true, chunk_size: auto}
  io: {atomic_checkpoints: true, netcdf_compression: true, netcdf_compression_level: 4}
```

### Backward compatibility

* A configuration **without** a `refinement` section keeps the existing
  behaviour. When the historical `model.flow_refine_*` keys are present they are
  translated into the new schema (`type: flow_matching_unet`, legacy backend,
  `unet.hidden_channels = flow_refine_hidden`,
  `flow_matching.integration_steps = flow_refine_sampling_steps`, ...), so every
  existing YAML file and checkpoint keeps working unchanged.
* `flow_matching` is a permanent alias for `flow_matching_unet`.
* `flow_matching_conv` is an alias for the unified
  `flow_matching_conv_unet`, not for the legacy wrapper.
* Omitted new options keep the legacy-safe values: no residual scaling,
  epsilon prediction, no SNR reweighting, uniform diffusion-timestep sampling,
  no Transformer local stem/head, and temporal Mamba disabled. The improved
  example YAMLs opt in explicitly where intended. Because scaling, local
  refinement and Mamba add checkpoint state or parameters, training and
  inference must agree on those resolved architecture settings.
* `refinement.enabled: false` and `refinement.type: none` both disable Phase 2.
* Invalid refiner names, Transformer geometry, patch/window settings, diffusion
  schedules, prediction parameterizations, flow solvers and incompatible option
  pairs raise `ConfigValidationError` (surfaced as a `ValueError` by
  `validate_config`) rather than being silently coerced.
* Scientific settings (`model.refinement`) and workflow settings
  (`performance`) are separate sections. Nothing in `performance` may change
  architecture, loss, sampler, solver, ensemble size, diffusion/flow step counts
  or evaluation data.
* The fully resolved configuration is saved in checkpoints and in the output
  metadata.

---

## 13. Rollout caching

When Aurora is frozen, deterministic rollouts can be cached
(`finetune.refinement.cache`). The key includes the Aurora checkpoint
fingerprint, dataset and split, initialization time, valid time, lead time,
rollout interval, input history length, variables, pressure levels, spatial
domain, normalization and the cache-schema version. Stale or incompatible
entries are rejected, cached rollouts can be validated against online Aurora
output before use, and caching is refused entirely when Aurora is trainable
(joint fine-tuning) — enforced at configuration-resolution time.

---

## 14. Output

`finetune.refinement.io.build_refined_dataset` writes, per variable:

| variable | content |
| --- | --- |
| `<var>` | deterministic Aurora rollout |
| `<var>_residual` | predicted normalized correction (legacy diagnostic name) |
| `<var>_predicted_correction_physical` | physical correction actually added to Aurora |
| `<var>_refined` | refined physical prediction (ensemble mean when `M > 1`) |
| `<var>_members` | every ensemble member, explicit `member` dimension, draw order |
| `<var>_ensemble_mean` | ensemble mean |
| `<var>_ensemble_spread` | unbiased ensemble standard deviation |
| `<var>_truth` | ground truth, when provided |

The dataset attribute `aurora_default_forecast_variable_map` points evaluators to
`<var>_refined` (or the compatible physical fallback), never to `_residual`, raw
epsilon/velocity, or a latent.

Coordinates: `rollout_step` (1-based), `init_time`, `time` (valid time),
`lead_time_hours`, `latitude`, `longitude`, `level`, `member`. Ensemble
statistics are accumulated in float32 regardless of the compute dtype, and
masked cells are excluded from the mean and spread instead of poisoning them.

---

## 15. Commands

Training (all use the existing distributed driver):

```bash
# raw Aurora rollout, no refinement
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_rollout_no_refinement.yaml

# compatibility filename using the unified flow-matching convolutional UNet
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_flow_matching_unet.yaml

# unified packed-field flow-matching convolutional UNet
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_flow_matching_conv_unet.yaml

# flow-matching Transformer
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_flow_matching_transformer.yaml

# diffusion UNet
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_diffusion_unet.yaml

# diffusion Transformer
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_diffusion_transformer.yaml
```

Resume works for every refinement type through the existing
`--resume` / checkpoint-directory mechanism of the driver: the refinement
checkpoint restores the model, optimizer, scheduler, gradient scaler, epoch,
global step and RNG states, and the recorded Aurora fingerprint is verified.

The shipped unified examples emit the deterministic conditional mean with
`deterministic_inference: true` and `ensemble_size: 1`. For uncertainty
quantification, set `deterministic_inference: false` and `ensemble_size >= 2`;
individual members, their mean, and their spread remain distinct products. A
configured base seed is deterministically mixed with forecast initialization
time and member index, so reruns reproduce exactly without replaying one spatial
latent across every case.

Diagnostics and validation:

```bash
# legacy flow-matching numerical parity (read-only on the checkpoint)
python -m finetune.refinement_parity_check \
  --checkpoint finetune/outputs/checkpoints/O3_global_3day_lead/best.ckpt \
  --report /tmp/flow_parity.json

# minimal non-destructive synthetic smoke tests
python -m finetune.refinement_smoke_test --report /tmp/refinement_smoke.json

# trace one real failed NO2 batch through exact time/coordinate/level matching;
# this writes strict JSON, NetCDF tensors, and unsmoothed per-field PNGs
python -m finetune.trace_refinement_batch \
  --config finetune/aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml \
  --initialization 20240701T000000 --lead-hours 12 \
  --finetuned-dir finetune/outputs/NO2_US-WEST_3day_lead_diffusion_transformer \
  --legacy-refined-dir finetune/outputs/NO2_US-WEST_3day_lead \
  --output-dir /tmp/no2_batch_trace --overwrite

# controlled smooth synthetic reconstruction plus 1-8-sample real overfit;
# runs the Diffusion Transformer and the same backbone as direct regression
python -m finetune.run_refinement_diagnostics --mode both \
  --heads diffusion_transformer direct_regression_transformer \
  --synthetic-samples 6 --real-samples 4 --train-steps 400 \
  --batch-size 4 --device cuda \
  --output-dir /tmp/refinement_controlled

# short real-data execution check (not a scientific-duration training claim)
python -m finetune.refinement.benchmark --case no2_uswest --compare \
  --epochs 8 --max-initializations 24 \
  --output /tmp/no2_refinement_benchmark.json

# automated tests
python -m pytest tests/test_refinement_config.py tests/test_refinement_target_space.py \
                 tests/test_refinement_models.py tests/test_refinement_transformer.py \
                 tests/test_refinement_checkpoint.py tests/test_refinement_performance.py \
                 tests/test_refinement_io.py tests/test_refinement_oracle_contract.py \
                 tests/test_refinement_controlled_diagnostics.py -q
```

Rollout-cache generation is enabled per run with
`performance.rollout_cache.{enabled,path}` (requires
`refinement.freeze_aurora: true`); profiling with `performance.profile: true`.

---

## 16. Evaluation

`finetune.refinement.evaluation` compares raw Aurora against each refiner
**separately by** variable, pressure level, forecast lead time and (when
configured) spatial region:

```python
from finetune.refinement.evaluation import compare_raw_and_refined, summarize

rows = compare_raw_and_refined(
    truth,                       # [N, C, H, W] packed physical ground truth
    packing=packing,
    candidates={
        "raw": deterministic,
        "flow_matching_unet": refined_fm_unet,
        "flow_matching_transformer": refined_fm_transformer,
        "flow_matching_conv_unet": refined_fm_conv_unet,
        "diffusion_unet": refined_diff_unet,
        "diffusion_transformer": refined_diff_transformer,
    },
    ensembles={"diffusion_unet": members},   # [N, M, C, H, W], draw order
    lead_index=lead_index,
    lead_hours=lead_hours,
    regions={"all": None, "tropics": tropics_mask},
)
summary = summarize(rows, baseline="raw")
```

Reported per group: `bias`, `mae`, `rmse`, `pattern_correlation`,
`ensemble_mean_bias`, `ensemble_mean_rmse`, `ensemble_spread`,
`spread_skill_ratio` and the fair (unbiased) ensemble `crps`, each optionally
cosine-latitude area weighted, with masked and non-finite cells excluded.

`summarize` only reports `improvement: true` when the aggregate absolute bias
decreases **and** no group degrades by more than the tolerance on any other
metric; every degraded group is listed in `degraded_groups`. A bias reduction
therefore can never hide an RMSE, MAE or correlation regression, a degraded
variable/level, or a degraded long-lead forecast. Report the tradeoffs
explicitly.

Smoke tests and short benchmark runs establish **software correctness only**.
The built-in `o3_global` and `no2_uswest` benchmark cases currently read
provenance-validated raw Aurora rollouts from
`examples/outputs/cams_rollouts`; their baseline stage is recorded as
`pretrained`. That is not a Stage-1 fine-tuned, no-refinement baseline. If the
scientific comparison requires that baseline, first produce or supply matching
unrefined rollout files and retain that limitation in any report until they are
available.

The benchmark groups by forecast initialization and purges the train/test
boundary by forecast horizon, records source/config/environment provenance, and
reports every variable, pressure level and lead separately. Those safeguards do
not make an eight-epoch execution check a performance result. Claims that a
refiner reduces CAMS bias require a full-duration, case-tuned training run and
an independent held-out period covering the intended seasons and forecast
horizons; report MAE, RMSE, bias, correlation, extremes and ensemble calibration,
not training loss alone.

---

## 17. Known limitations

* `refinement.conditioning.aurora_features` is rejected: Aurora does not expose
  a stable frozen spatial feature map through the rollout API.
* `refinement.feedback_to_rollout: true` is experimental, disabled by default,
  routed through a separate code path in `run_rollout`, and must be evaluated
  independently for stability and error accumulation. The legacy backend's
  resolved configuration reports the historical inline-refinement behaviour of
  `training.flow_refine_autoregressive_feedback` faithfully rather than changing
  it.
* `windowed_2d` attention with a periodic longitude and a token grid that is not
  a multiple of the window size handles the final partial window with circular
  padding and warns once; choose a dividing window size to avoid it.
* Temporal Mamba's authoritative pure-PyTorch scan vectorizes the spatial
  pixels and selected pressure levels. The regional NO2 contract is covered by
  executable tests; large global grids need case-specific memory profiling and
  may require a future chunked/checkpointed scan before full-resolution training.
* Joint Aurora/refiner fine-tuning is supported only when explicitly configured
  and is incompatible with rollout/conditioning caching.
* `performance.compile` and reduced-precision modes are exposed but not enabled
  by default; validate numerical parity and stability on the exact target
  hardware before production use.
