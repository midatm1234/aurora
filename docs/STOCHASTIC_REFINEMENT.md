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
 conditioning | flow_matching_unet | flow_matching_transformer              | ---> r_hat
 (section 5)  | diffusion_unet     | diffusion_transformer                  |
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
| `diffusion_unet` | conditional convolutional UNet, DDPM training / DDIM sampling | diffusion timestep | unified |
| `diffusion_transformer` | spatial-token Transformer denoiser, DDPM/DDIM | diffusion timestep | unified |

The four options are **alternatives**. Diffusion and flow matching are never
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

Refinement happens entirely in **Aurora's normalized target space**
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
`flow_matching_transformer` as an explicit, documented choice.

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
  and nothing stale can be restored from a checkpoint.

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
* every component is returned and logged separately.

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

```yaml
model:
  refinement:
    enabled: true
    type: flow_matching_transformer   # none | flow_matching_unet | flow_matching |
                                      # flow_matching_transformer | diffusion_unet |
                                      # diffusion_transformer
    checkpoint: null
    freeze_aurora: true
    joint_finetuning: false
    train_on_residual: true
    feedback_to_rollout: false        # EXPERIMENTAL when true
    seed: 1234
    ensemble_size: 10

    target_space:
      use_existing_normalization: true
      residual_space: normalized

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
      bias_weight: 0.0
      gradient_weight: 0.0
      pattern_correlation_weight: 0.0
      area_weighted: true
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
      prediction_type: epsilon        # epsilon | velocity | sample
      schedule: cosine                # cosine | linear | scaled_linear
      sampler: ddim                   # ddim | ddpm (ddpm requires eta: 1.0)
      eta: 0.0
      clip_sample: false
      clip_sample_range: 10.0

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
      positional_encoding: latlon_2d  # latlon_2d | sincos_2d
      max_tokens_lat: 256
      max_tokens_lon: 512
      attention_mode: global_2d       # global_2d | windowed_2d
      window_size: [8, 8]
      shifted_windows: false
      gradient_checkpointing: false
      optimized_attention: auto       # auto | sdpa | math
      zero_init_output: true

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
| `<var>_residual` | predicted residual (normalized target space) |
| `<var>_refined` | refined prediction (ensemble mean when `M > 1`) |
| `<var>_members` | every ensemble member, explicit `member` dimension, draw order |
| `<var>_ensemble_mean` | ensemble mean |
| `<var>_ensemble_spread` | unbiased ensemble standard deviation |
| `<var>_truth` | ground truth, when provided |

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

# existing flow-matching UNet (unchanged behaviour)
python finetune/aurora_finetune_distributed.py \
  --config finetune/examples/stochastic_refinement/aurora_O3_global_flow_matching_unet.yaml

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

Single-member versus ensemble inference is selected with
`model.refinement.ensemble_size` (1 versus N); `model.refinement.seed` makes the
draws reproducible.

Diagnostics and validation:

```bash
# legacy flow-matching numerical parity (read-only on the checkpoint)
python -m finetune.refinement_parity_check \
  --checkpoint finetune/outputs/checkpoints/O3_global_3day_lead/best.ckpt \
  --report /tmp/flow_parity.json

# minimal non-destructive smoke tests for all four refiners
python -m finetune.refinement_smoke_test --report /tmp/refinement_smoke.json

# profiling + numerical-parity benchmark
python -m finetune.refinement_benchmark --try-bf16 --report /tmp/refinement_bench.json

# automated tests
python -m pytest tests/test_refinement_config.py tests/test_refinement_target_space.py \
                 tests/test_refinement_models.py tests/test_refinement_transformer.py \
                 tests/test_refinement_checkpoint.py tests/test_refinement_performance.py \
                 tests/test_refinement_io.py -q
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

Smoke tests establish **software correctness only**. No claim of scientific bias
reduction should be made without an appropriately trained model evaluated over a
validation period.

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
* Joint Aurora/refiner fine-tuning is supported only when explicitly configured
  and is incompatible with rollout/conditioning caching.
* `performance.compile` and reduced-precision modes are exposed but not enabled
  by default; validate them with `finetune/refinement_benchmark.py` before use.
* The bf16 parity column of the benchmark is only meaningful on hardware where
  autocast actually engages for these operators.
