# Joint spatiotemporal O3 refinement — implementation review

> Historical worktree review retained during branch synchronization. The GPU memory measurements and pilot percentages below were already present in the original local notes; their execution receipts were not supplied with this cleanup and they were not independently rerun here. They are reported historical evidence, not new validation of this published branch. Current executed checks and known limits are recorded in [the synchronization report](../docs/refinement-branch-synchronization.md). Local one-off memory and full-backbone training-step helpers were preserved outside Git tracking; maintained tests, the configuration generator, and the synthetic pilot remain public.

**Repository** `midatm1234/aurora`
**Branch** `aurora_finetune_stochastic_refinement`
**Base revision** `88f652f04aa75b65e410b8d00cf4ea07f2edd945` ("Expand temporal Mamba refinement workflows")
**Scope** the four unified refiners: `flow_matching_conv_unet`,
`flow_matching_transformer`, `diffusion_unet`, `diffusion_transformer`.

The legacy `flow_matching_unet` / `AuroraFlowRefine` wrapper is **not** in scope.
Its formulation and checkpoint format are preserved unchanged, and
`refinement.temporal` is rejected for it by the configuration validator.

---

## 1. Repository state actually observed

No `AGENTS.md` exists at any level of the tree. The working tree carried
unrelated modifications under `examples/` (CAMS download logs and notebook
outputs); these were left untouched.

`tests/test_finetune_config_compatibility.py::test_finetune_notebook_default_config_loads_as_unified_refinement`
**already failed on the unmodified tree** (the notebook default is
`flow_matching_transformer`, the test asserts `flow_matching_conv_unet`). It is
reported here as a pre-existing defect and was deliberately not "fixed" as part
of this work, because changing either side alters a workflow that is out of
scope. Verified by stashing all changes and re-running.

The O3 recipe resolves to `model_variant: aurora_air_pollution`, i.e. the
`AuroraAirPollution` subclass.

---

## 2. Demonstrated defects

### 2.1 The pretrained backbone's day-of-year channels are day-of-**month**

`aurora/model/encoder.py` computes, inside the `self.dynamic_vars` branch:

```python
ones * np.cos(2 * np.pi * time[b].day / 365.25),
ones * np.sin(2 * np.pi * time[b].day / 365.25),
...
dynamic_vars = ("tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin")
```

`datetime.day` is the day of the **month** (1–31), not the day of the year
(1–366). Divided by 365.25 the two channels therefore span only about 8.5° of a
circle and repeat every calendar month, so they encode no seasonal information
at all.

**This path is active for the O3 recipe.** `Aurora.__init__` defaults
`dynamic_vars=False` and the standard released checkpoints keep that default,
but `AuroraAirPollution` sets `dynamic_vars: bool = True`. The O3 configuration
selects exactly that variant.

**Deliberately not changed.** The pretrained air-pollution checkpoint was
trained *with* these channels. Correcting them in place would silently alter the
frozen backbone's input convention and invalidate the checkpoint. Section 4
below describes the correct refinement-side replacement. A backbone correction
would need an explicit versioned option, a matched raw-Aurora baseline and a
retraining analysis, and is out of scope here.

Two further properties of the same code path, recorded for completeness:

* the features are broadcast with `ones * ...` over the history axis `T`, so
  every history frame receives the **initialization** timestamp;
* they are constant in space, so no UTC × longitude interaction is represented.

### 2.2 Temporal conditioning never reached the generative process

Before this change the only temporal component was
`PackedMambaTemporalAdapter`, applied in `two_phase.py` **after** sampling:

```python
member_normalized = rollout.unsqueeze(1) + member_corrections
if self.temporal is not None:
    ...
    temporal_correction = self.temporal.causal_residual(flat_sequence, ...)
    member_normalized = member_normalized.float() + temporal_correction
```

Three consequences, all of which the historical `MAMBA_ABLATION.md` study
inherited and none of which it could separate from "temporal modelling does not
help":

1. The denoiser/velocity network never saw temporal context. The generative
   distribution was conditioned on a single frame; the temporal module could
   only apply a deterministic post-hoc shift to already-drawn samples.
2. The history frames were `detach()`ed (`current_raw = member_normalized.detach()`),
   so no gradient flowed from a later lead back through an earlier one.
3. Mamba ran per grid cell after a single 3×3 spatial encoder, which cannot
   represent transport between locations.

The legacy adapter is retained unchanged. The new path is additive and
separately configured.

### 2.3 No calendar, solar or vertical identity in the refiner

`ConditioningConfig` offered only `aurora_rollout`, `aurora_input_state`,
`aurora_features` (rejected as unimplemented), `static_fields`, `masks`,
`forecast_lead_time`, `latitude` and `longitude`.

* Nothing derived from the **valid** timestamp.
* Latitude entered as `lat / 90`, longitude as `sin/cos` — separately, never as
  a seam-safe spherical position and never interacting with UTC.
* The packed channel axis carried pressure only as an **index**. `gtco3`
  (units `kg m-2`, a total column) was indistinguishable from a surface
  concentration.
* Lead time was embedded and **added** to the process-time embedding
  (`LeadTimeEmbedding` → `emb + ...`), which can only express additive
  corrections — a set of disconnected lookups, not interactions.

### 2.4 Defects introduced and fixed during this work

Recorded because they are the failure modes this task warns about, and they were
caught by the tests rather than by inspection:

* **Normalization broke causality.** The first draft of the temporal mixers used
  `nn.GroupNorm` on `[N, D, S]` tensors. `GroupNorm` reduces over the trailing
  axis, which here is the **lead axis**, so each frame's statistics depended on
  future frames. Prefix invariance failed numerically for `causal_conv` and
  `conv_gru` (max deviation ≈ 4e-2 and 5e-2). Replaced by `_StepNorm`, which
  normalizes each frame independently. Now pinned by
  `test_temporal_context_is_prefix_invariant`.
* **Temporal weights were silently dropped from checkpoints.**
  `build_refinement_checkpoint` filtered state-dict keys by the prefixes
  `refiner.` and `temporal.`. The new encoder is registered as
  `temporal_context.`, which does **not** start with `temporal.`. Every temporal
  parameter would have been discarded on save and silently re-initialized on
  load. Fixed by adding an explicit `temporal_context_prefix`.
* **Duplicate parameter registration.** Exposing the refiner-owned
  `VerticalChannelEncoder` as an attribute on the wrapper registered a second
  copy under a top-level prefix that refinement checkpoints do not save, so
  `load_state_dict` reported missing keys. Converted to a read-only property.
* **Ensemble conditioning misalignment.** Batched ensemble generation expands
  the conditioning with `repeat_interleave`, but the bound per-frame calendar
  and temporal context were not expanded, so ensemble inference raised a shape
  error. Fixed with `_expand_to_members`, which reproduces the
  `repeat_interleave` layout exactly. A plain `repeat` would have produced the
  right shape while pairing frame 0's calendar with sample 1's field; the
  alignment is therefore pinned numerically by
  `test_ensemble_sampling_aligns_conditioning_with_members`.
* **Temporal weights were dropped on *load* as well as on save.** Fixing the
  save side was not enough: `load_refinement_state_dict` applied the same
  `temporal.`-only prefix filter to incoming keys *and* to the `missing`
  report. A `temporal_context.*` tensor was therefore silently discarded and
  not even reported, so a resumed run restored a freshly initialized temporal
  encoder. Because the context projection is zero-initialized, the run would
  degrade to the **spatial-only control** with no error, no NaN and no
  warning — making a trained temporal model numerically indistinguishable from
  its own ablation control. Found by an independent review of this diff, not by
  the original tests, which only exercised `load_state_dict(..., strict=False)`
  directly. Now covered by
  `test_refinement_checkpoint_round_trips_through_the_public_loader`.
* **Two incompatible flattening conventions.**
  `LeadStepBuffer.pack` (the production trainer path) emits rows **lead
  blocked**, `position * batch + sample`, whereas the temporal encoder works on
  `[B, S, ...]` and flattens **lead major**, `sample * steps + position`.
  Pairing them naively yields correct shapes while attaching every frame's
  calendar and temporal context to the wrong sample. The two orders are now
  separate, named, tested converters
  (`expand_initializations` / `expand_initializations_lead_blocked`,
  `sequence_to_lead_blocked` / `lead_blocked_to_sequence`) rather than a flag,
  and §9 records that the production trainer is not yet wired.
* **`peak_timing_error` inferred an axis from a size coincidence.** It decided
  whether a mask was member-shaped by testing `mask.shape[1] != steps`, so an
  ensemble whose member count happened to equal the rollout length took the
  peak over the **member** axis instead of the lead axis (or raised, depending
  on the batch size). Replaced by an explicit separate `truth_mask` argument.

---

### 2.5 Bottleneck attention OOMs at global resolution (fixed)

Measured on the idle H100 NVL (93.1 GiB) at the real O3 geometry — 451×900,
5 packed channels, 6 rollout leads folded into the batch — the **shipped
spatial-only recipe ran out of memory**, before any of this work's additions
were enabled.

`_SpatialSelfAttention2d.forward` built its q/k/v as
`t.reshape(...).transpose(-2, -1)`, a **non-contiguous** view. PyTorch's
`scaled_dot_product_attention` cannot dispatch a non-contiguous layout to its
flash / memory-efficient kernels, so it silently fell back to the *math*
backend and materialized the complete score matrix. With `num_levels: 3` the
bottleneck holds `ceil(451/4) × ceil(900/4) = 113 × 225 = 25,425` tokens, so

```
6 leads × 4 heads × 25,425² × 4 B = 57.8 GiB
```

in a single allocation — exactly the figure in the OOM. The same call with a
contiguous layout peaks below 1 GiB. `_SpatialAttention._attend` (the
Transformer heads) had the identical pattern.

Fixed by adding `.contiguous()` at all three attention sites. Verified
numerically identical in float64 (max abs difference **4.4e-16**), so it is a
kernel-selection fix and not a change of mathematics; no checkpoint is
invalidated.

**Historically reported peak memory, refinement components only** (the frozen
1.27 B Aurora backbone is excluded). The original measurement used a local
diagnostic helper excluded from the public branch. These numbers do not verify
the full model's memory requirements on another environment:

| case | peak |
|---|---|
| spatial-only, `bottleneck_attention: true` (shipped) — **before** the fix | **OOM** (57.8 GiB attempted) |
| spatial-only, `bottleneck_attention: true` — after the fix | 22.63 GiB |
| spatial-only, `bottleneck_attention: false` | 22.21 GiB |
| spatial-only, attention on, `num_levels: 5` | 26.25 GiB |
| + calendar/solar/vertical, `temporal.backend: none` | 23.01 GiB |
| + full spatiotemporal, `causal_conv`, stride 4 | 25.92 GiB |
| + full spatiotemporal, `causal_conv`, stride 8 | 25.72 GiB |
| + full spatiotemporal, `conv_gru`, stride 4 | 26.16 GiB |
| + full spatiotemporal, `causal_conv`, `num_levels: 5`, attention on | 29.97 GiB |

The joint spatiotemporal conditioning costs **~3.7 GiB** on top of the
spatial-only baseline. stride 4 and stride 8 differ by only 0.2 GiB, which
locates the cost in the full-resolution convolutional stem rather than in
temporal mixing — so `spatial_stride` is not a useful memory lever, but
`temporal.hidden_channels` would be.

Caveats: measured in float32 without autocast, so the trainer's bf16 path
should use less; and the frozen backbone's activations add on top. See §2.7 for
the end-to-end number.

### 2.7 End-to-end single-step verification

The original local training-step diagnostic used the **real** pipeline — the real
config through `load_config`, the real `train.nc` and static pickle, the real
model from `build_finetune_model` including the frozen 1.27 B `AuroraAirPollution`
backbone, and the real `compute_supervised_loss` over all six rollout leads plus
one backward. The notes report no optimizer update or checkpoint write. That
one-off helper is retained locally and is not a public entry point; the results
below were not repeated during branch cleanup.

```
device        : NVIDIA H100 NVL  total 93.08 GiB
model         : 1,273,091,691 total, 397,493 trainable
after model   : 4.75 GiB allocated
```

| effective batch (samples × 6 leads) | peak allocated | headroom | result |
|---|---|---|---|
| 1 — **without** the §2.5 contiguity fix | 72.57 GiB at failure | — | **OOM** |
| 1 — with the fix | 32.12 GiB | 60.49 GiB | OK |
| 2 | 59.15 GiB | 33.21 GiB | OK |
| 3 | 86.22 GiB | 5.67 GiB | OK, no margin |

The end-to-end run confirms §2.5 on the production path, not just in isolation:
**the shipped recipe OOMs on its first step without the fix and uses 32 GiB with
it.** Memory scales close to linearly in the sample count (≈27 GiB per sample),
so `batch_size: 1` with `accumulation_steps: 32` — the shipped setting — is the
right choice; `batch_size: 2` fits, and `3` leaves no margin for fragmentation.

### 2.6 Calendar features crashed on GPU (fixed)

`CalendarFeatureBuilder.scalar_features` built its phase tensors on the CPU
(they derive from Python `datetime` objects) but kept `lead_hours` on whatever
device the caller supplied, so `torch.stack` raised
`Expected all tensors to be on the same device`. Every test in the suite runs on
CPU, so this only appeared when the memory measurement above first exercised the
path on CUDA. All calendar arithmetic is now pinned to the CPU in float64 and
the finished matrix is moved to the requested device once.



Recorded because they are the failure modes this task warns about, and they were
caught by the tests rather than by inspection:

## 3. Four time coordinates

`finetune/refinement/calendar_features.py` is the single feature builder used by
training, inference and evaluation. It keeps four coordinates strictly separate:

| coordinate | meaning | representation |
|---|---|---|
| `t0` | initialization time | `init_cycle_{sin,cos}`, `init_season_{sin,cos}` |
| `ell` | cumulative forecast lead (hours) | `lead_hours_{scaled,log1p,sqrt}` + `LeadTimeEmbedding` |
| `t_valid = t0 + ell` | valid time of **this** frame | `valid_season_{sin,cos}`, `valid_utc_{sin,cos}` |
| `tau` / `k` | flow coordinate / diffusion index | `ProcessTimeEmbedding` (separate module, separate parameters) |

`tau`/`k` are deliberately absent from the calendar module: nothing there can be
indexed by them.

Seasonal encoding, as specified:

```
year_phase = (day_of_year - 1 + fractional_UTC_hour / 24) / days_in_year
season     = [sin(2*pi*year_phase), cos(2*pi*year_phase)]
utc        = [sin(2*pi*fractional_UTC_hour/24), cos(2*pi*fractional_UTC_hour/24)]
```

* `days_in_year` uses the Gregorian rule: 366 in 2024 and 2000, 365 in 1900 and
  2100.
* `day_of_year` comes from `timetuple().tm_yday`, so 1 March is day 60 in a leap
  year and 59 otherwise — the exact quantity the backbone gets wrong.
* Fractional UTC hour carries minutes, seconds and microseconds.
* Aware timestamps are converted with an explicit `timezone.utc`; naive ones are
  documented as UTC. `datetime.now`, `utcnow` and bare `astimezone()` are never
  called, so the machine's local zone cannot leak in.
* `cftime` calendars `360_day`, `noleap`, `all_leap`, `julian` are rejected with
  a named error instead of being mis-encoded.

`valid_times(init_time, lead_hours)` requires one lead per frame, which is what
structurally prevents reusing the initialization hour across a rollout.
`elapsed_hours_from_leads` reports the **actual** gap between adjacent frames, so
a missing step appears as a larger elapsed time rather than being compressed;
`has_predecessor` marks trajectory starts and post-gap frames.

---

## 4. Geography, vertical structure and fusion

**Spatial channels** (`SPATIAL_FEATURE_NAMES`, 7 channels):
seam-safe spherical position `[cos φ cos λ, cos φ sin λ, sin φ]`; explicit
non-periodic `latitude_normalized`; cyclic **local mean solar hour**

```
local_mean_solar_hour = (UTC_hour + longitude_east_degrees / 15) mod 24
```

labelled as mean solar time — not civil local time (no zones, no DST) and not
apparent solar time (no equation of time); and `cos_solar_zenith`.

`cos_solar_zenith_angle` reproduces the orbital approximation already used by
`aurora/insolation.py` (1995 elements, first-order equation of centre) in torch,
rather than duplicating a new one. It is validated against
`aurora.insolation.insolation` per date: correlation > 0.99999 with a uniform
ratio, the ratio being the Earth–Sun distance factor `rho**-2` that the Aurora
helper additionally applies and that is a function of day-of-year alone.
Longitude 0° and 360° give identical values, so the dateline seam is exact.
Night-time values are **not** clipped: the sign carries the day/night
transition.

**Vertical channels** (`finetune/refinement/vertical.py`): each packed channel
receives a continuous `log(p / 1000 hPa)` coordinate, an explicit
`pressure_applicable` mask, `is_atmos_level` / `is_surface_field` /
`is_column_field` type flags, and a learned variable-identity embedding.

* `gtco3` is classified as a **column** quantity from its `kg m-2` units and is
  given `pressure_applicable = 0` and `log_pressure = 0`. It is never assigned a
  fabricated surface or zero-pressure level.
* No column/profile integral relationship is imposed anywhere. Four supervised
  levels do not integrate to a total column; the two are represented jointly and
  left statistically coupled.
* Pressure is validated to a plausible hPa range, so a Pa-valued packing fails
  loudly instead of producing a wrong `log(p)`.
* Because the coordinate is continuous, the representation does not rely on the
  packed channel *index* for pressure generalization.

**Fusion.** `_ConditioningEmbedding` no longer merely sums embeddings. Process
time, lead and the metadata vector are concatenated and passed through a joint
nonlinear MLP whose output is added through a **single** zero-initialized
projection:

```python
parts = [process_time_emb, metadata_emb] (+ lead_emb)
return emb + self.fusion(torch.cat(parts, dim=-1))
```

This allows season × hour × lead × level interactions rather than three
independent lookup corrections. There is no year-ID table anywhere: multi-year
behaviour is carried by continuous seasonal phase, not by memorizing a year.

---

## 5. Temporal sequence learning

`finetune/refinement/temporal.py` provides one interface shared by all four
heads. It consumes
`[batch, lead_time, packed_variable_level, latitude, longitude]` and emits
per-frame context `[batch, lead_time, context_channels, latitude, longitude]`,
which is concatenated onto the spatial conditioning — so temporal information
enters the denoiser/velocity network at **every** process evaluation.

Backends, selected by `refinement.temporal.backend`:

| backend | kind | attention? |
|---|---|---|
| `none` | spatial-only control, builds nothing | no |
| `causal_conv` | dilated causal temporal convolution | **no** |
| `conv_gru` | forward-only gated recurrence | no |
| `attention` | causal attention over lead tokens | yes (temporal only) |
| `mamba` | reference selective scan, reused unchanged | no |

The temporal backend is independent of `refinement.type` (spatial backbone) and
of `transformer.attention_mode` (spatial attention), so `causal_conv` really is
an attention-free temporal mode that adds no attention elsewhere. **Mamba is
optional and nothing depends on it.**

**Causality.** Every backend is prefix invariant, asserted numerically for all
four (`test_temporal_context_is_prefix_invariant`) after one optimizer step so
the property is not trivially satisfied by an all-zero output. Left-padding
only; upper-triangular attention mask; forward-only recurrence; per-frame
`_StepNorm`. `assert_increasing_leads` refuses a trajectory whose leads are not
strictly increasing, which is what stops two initializations from being merged
because their valid times overlap.

**Spatial coupling.** A convolutional stem runs *before* temporal mixing and a
second spatial stage runs *after* it, so information moves between locations as
well as along time. `test_temporal_context_is_spatially_contextual` asserts that
perturbing one grid cell changes its neighbour's context — a property an
isolated per-pixel recurrence does not have.

**Memory.** Temporal mixing runs on a strided grid (`spatial_stride`, default 4)
and the context is interpolated back with longitude wrapping. Cost is
`O(B·S·D·H·W / stride²)`, not attention over every global space–time–level
token. `max_sequence_length` fails loudly rather than exhausting memory.

**Process/physical axis separation.** The context is a function of the causal
trajectory only. `TemporalContextEncoder.forward` has **no argument** through
which a noised residual could be passed, so it cannot drift across solver
evaluations of one frame. That makes caching exact rather than approximate;
`TemporalContextCache` keys on `(initialization, member, lead_index)` so state
can never leak between members or initializations.

**Trainable identity initialization.** The context output projection is
zero-initialized, so enabling temporal conditioning does not perturb a freshly
built refiner. It is the **only** zero in that path and its input activation is
non-zero, so the first backward pass produces a non-zero gradient there and the
branch trains. The end-to-end chain (`out_proj` → trunk → temporal/calendar)
resolves within two optimizer steps, verified for all four heads by
`test_temporal_and_calendar_parameters_receive_gradient`.

---

## 6. Causality and generative-state handling

Default is **causal** streaming refinement: the correction at lead `j` may use
initialization-time information, raw Aurora forecasts through `j`, and preceding
refined states. It may not use future verifying CAMS values.

`refinement.temporal.mode: full_trajectory` is the separately labelled optional
postprocessor that may also use **later raw Aurora forecasts** of the same
initialization, which are available when the whole rollout is produced before
refinement. It never substitutes later CAMS truth. The mode and the flag
`uses_future_raw_aurora` are persisted in the checkpoint contract so a
deployment cannot confuse the two.

Conditioning is bound through `use_frame_conditioning`, and a mismatch between
configuration and supplied tensors is a hard error in both directions: refining
a temporal checkpoint without context raises, and supplying context to a
spatial-only checkpoint raises. A run therefore cannot train and deploy
different conditioning contracts.

Refinement remains postprocessing with no feedback into Aurora.

---

## 7. Trajectory objectives and scores

`finetune/refinement/trajectory.py` supplements — never replaces — the marginal
CRPS already in `evaluation.py`.

`tendency_loss` implements

```
L = mean rho( ((yhat_j - yhat_{j-1}) - (y_j - y_{j-1})) / delta_hours )
```

with documented robust/squared penalties, a training-derived per-channel scale
for mixed units, validity requiring **both** frames of a pair, and real elapsed
hours. It is exactly zero for a perfect trajectory and positive for a smoothed
one, which is the property that makes it a penalty on incorrect evolution rather
than on change itself (`test_tendency_loss_penalizes_smoothing_not_change`). It
applies to the declared point product only.

Event functionals are computed **member-wise before** ensemble scoring:
`member_window_maximum` and `member_time_weighted_mean` reduce each member's
trajectory, and `ensemble_crps` then scores the resulting distribution. The test
suite asserts that the mean of per-member maxima differs from the maximum of the
ensemble mean.

`ensemble_crps` implements the empirical and fair estimators explicitly and
**rejects** the fair estimator for `M = 1` rather than silently degenerating;
the empirical `M = 1` value is verified to equal the MAE, and is documented as
saying nothing about calibration.

`threshold_weighted_crps` applies `v(x) = max(x, u)` (or `min` for the lower
tail relevant to total-column ozone) to members **and** truth alike and scores
**all** cases, not only observed exceedances. Thresholds are caller-supplied in
absolute ozone units, from training data only.

`exceedance_brier` returns reliability bins; `residual_autocorrelation` and
`peak_timing_error` diagnose persistent drift and realization timing, with an
explicit note that a 12-hour verification cadence quantizes peak timing to 12
hours and must not be presented as hourly skill.

---

## 8. Bounded pilot evidence

`finetune/pilot_spatiotemporal.py` is a **capability test on a synthetic problem
with a known answer**, not a CAMS skill measurement. The synthetic residual is
built from three additive terms chosen so that each conditioning stage is
provably necessary:

* a fixed **spatial** dipole, recoverable from the Aurora field alone;
* a **calendar** term `sin(2π·year_phase) · cos(solar_zenith)` that depends on
  the frame's own valid timestamp and on longitude through solar position;
* a **temporal** first-order autoregressive drift whose amplitude depends on the
  *realized history*, so two trajectories with identical leads but different
  histories require different corrections. This is the discriminator between
  genuine temporal modelling and a lead-indexed lookup — precisely the
  distinction the historical Mamba ablation could not draw.

All variants share data, seeds, optimizer settings and backbone budget. Training
uses 48 independent synthetic initializations with a fresh minibatch each step;
evaluation uses 8 held-out trajectories generated from a different generator
state. A stage that fails here is broken; a stage that succeeds is *capable*,
which is necessary but **not** sufficient for CAMS skill. No claim about ozone
accuracy follows from it.

### Observed result

`python -m finetune.pilot_spatiotemporal --head diffusion_unet --steps 1500`
(CPU, single seed 1234, ~2–4 min per variant):

| stage | backend | MAE skill vs raw | RMSE skill vs raw | tendency (raw 0.0007) | params |
|---|---|---|---|---|---|
| `spatial_only` | `none` | 12.3 % | 9.2 % | 0.0008 | 10,795 |
| `calendar_only` | `none` | 25.5 % | −117.7 % | 0.0001 | 13,253 |
| `temporal_causal_conv` | `causal_conv` | 24.8 % | 11.8 % | 0.0006 | 28,085 |
| `temporal_conv_gru` | `conv_gru` | 60.2 % | 49.7 % | 0.0001 | 33,461 |
| `temporal_attention` | `attention` | 41.3 % | −37.2 % | 0.0003 | 31,205 |
| `temporal_mamba` | `mamba` | 51.6 % | 25.7 % | 0.0001 | 36,389 |

**What this does and does not show.**

* The ordering is as constructed: `spatial_only` recovers roughly the spatial
  dipole and little else; adding calendar conditioning roughly doubles MAE
  skill; the temporal backends, which are the only variants with access to the
  realized history, reach substantially higher MAE skill. That is direct
  evidence that each conditioning stage actually delivers its information to the
  generative network — the property this pilot exists to test.
* The **negative RMSE skill** for `calendar_only` and `temporal_attention` is a
  real result and is not smoothed over: at this budget the deterministic query
  of an under-trained diffusion network produces occasional large outliers, and
  RMSE is far more sensitive to them than MAE. It means these variants are *not*
  converged at 1500 steps, not that the conditioning is harmful.
* This is a **single seed on one head**. The relative ranking of the four
  temporal backends is **not** established by it and must not be used to select
  a backend. Backend selection requires the matched CAMS ablation of §11.
* `tendency` falls from 0.0007 (raw) to 0.0001 for the converged variants,
  i.e. the refined trajectories evolve closer to the reference — but on a
  synthetic target, so this measures capability, not forecast skill.

Run it with:

```bash
python -m finetune.pilot_spatiotemporal --head diffusion_unet --steps 1500 --json pilot.json
python -m finetune.pilot_spatiotemporal --all-heads --steps 1500 --json pilot_all.json
```

---

## 9. Status of each recipe

Classification is deliberately conservative.

| recipe | status |
|---|---|
| the four unified heads + calendar/solar/vertical + any temporal backend | **implementation-ready, pending one trainer wiring** (see below) |
| spatial-only control (`temporal.backend: none`, new keys off) | **unchanged and runnable today** — this is what the shipped O3 recipe selects |
| `temporal.mode: full_trajectory` | **implementation-ready**, horizon-dependent; must be reported separately from streaming |
| frozen Aurora latent-feature conditioning (`conditioning.aurora_features`) | **blocked** — still rejected by the validator; Aurora exposes no stable frozen spatial feature map through the rollout API, and inventing one without a tested extraction point would mislabel latent vertical tokens as pressure levels |
| any claim of O3 skill improvement | **not demonstrated.** No CAMS training was run. |

### The one remaining wiring step

`finetune/aurora_O3_global_finetune_3day_lead_config.yaml` ships with
`calendar`, `solar_geometry`, `vertical_identity` set to `false` and
`temporal.backend: none`. That is deliberate: the production training call site
(`unified_refiner.training_step(...)` in `finetune/aurora_finetune_utils.py`)
does not yet pass `calendar=` or `temporal_context=`, and the new guards raise
immediately if a checkpoint's declared conditioning is not supplied. Shipping
the keys enabled would ship a recipe that crashes on its first refinement step.

To enable them, that call site must additionally build:

```python
init_times = CalendarFeatureBuilder.expand_initializations_lead_blocked(t0_per_sample, steps)
calendar   = unified_refiner.calendar_spec(init_times, lead_hours_n)
sequence   = CalendarFeatureBuilder.lead_blocked_to_sequence(rollout_n, batch, steps)
context    = unified_refiner.build_temporal_context(sequence, lead_hours=..., init_time=t0_per_sample)
context    = CalendarFeatureBuilder.sequence_to_lead_blocked(context)
```

**Use the named converters, never a bare `reshape`.** `LeadStepBuffer.pack`
emits rows lead-**blocked** (`position * batch + sample`); the temporal encoder
is lead-**major**. A bare reshape between them has the correct shape and
attaches every frame's calendar and context to the wrong sample, with no error.
`test_lead_blocked_and_lead_major_orders_are_distinct_and_invertible` pins this.

The generated ablation configurations
(`python -m finetune.generate_spatiotemporal_configs`) enable the keys and are
therefore also blocked on the same wiring; the bounded pilot
(`finetune/pilot_spatiotemporal.py`) drives the full path directly and is
runnable today.

Nothing here is "scientifically validated": the pilot establishes capability on
a synthetic problem, and no multi-year job was started.

---

## 10. Compatibility and retraining

Every new key defaults to off:

```yaml
refinement:
  conditioning:
    calendar: false
    solar_geometry: false
    vertical_identity: false
  temporal:
    backend: none
```

With those defaults the conditioning width, the network, the state-dict keys and
the numerical path are unchanged. The full existing suite passes (823 tests
before the new file, 892 after; the single failure is the pre-existing one in
§1).

Enabling any of them **changes the first conditioning layer's input width** and
therefore requires retraining. This is intentional and is not silently
absorbed: `_metadata_for` and `augment_conditioning` raise if a checkpoint's
declared widths do not match what the caller supplies.

The checkpoint now persists a `spatiotemporal_contract` block recording the
calendar convention (`gregorian_utc_v1`), the exact scalar and spatial feature
**order**, the pressure reference and vertical feature order, the column channel
indices, the conditioning widths, the temporal backend and causal mode, and
`uses_future_raw_aurora`.

---

## 11. Verified commands

The maintained commands below are public entry points. Historical local-only
diagnostic commands have been removed from this list. See the synchronization
report for the exact checks executed during publication.

Preflight — resolve the configuration and print the contracts:
```bash
python -c "
import yaml
from finetune.refinement.config import resolve_refinement_config
c = yaml.safe_load(open('finetune/aurora_O3_global_finetune_3day_lead_config.yaml'))
r = resolve_refinement_config(c)
print('type      ', r.type)
print('cond      ', r.conditioning.to_dict())
print('temporal  ', r.temporal.to_dict())
print('meta/solar', r.conditioning.metadata_feature_count, r.conditioning.solar_channel_count)
"
```

Memory / OOM validation requires a separately approved bounded run on the
recipient's resources. Start with the portable `real-gpu-v1` plan through MCP;
its spatial-only scope does not validate these experimental temporal settings.

Targeted tests for this work:
```bash
python -m pytest tests/test_refinement_spatiotemporal.py -q
python -m pytest tests/test_refinement_models.py tests/test_refinement_config.py \
                 tests/test_refinement_conditioning.py tests/test_refinement_checkpoint.py -q
```

Generate the matched ablation grid (7 variants × 4 heads = 28 configurations):

```bash
python -m finetune.generate_spatiotemporal_configs --list
python -m finetune.generate_spatiotemporal_configs --out finetune/ablations
```

Bounded matched pilot:

```bash
python -m finetune.pilot_spatiotemporal --head diffusion_unet --steps 1500 --json pilot_du.json
python -m finetune.pilot_spatiotemporal --all-heads --steps 1500 --json pilot_all.json
```

Full multi-year training — **prepared, intentionally not started.** Check the
runner's own options before launching:

```bash
python finetune/aurora_finetune_distributed.py --help
python finetune/aurora_finetune_distributed.py \
    --config finetune/aurora_O3_global_finetune_3day_lead_config.yaml
```

---

## 12. Work not done

Stated explicitly so it is not mistaken for completed work.

* **No CAMS training or evaluation was run.** Every number in §8 comes from the
  synthetic capability pilot.
* **The production trainer is not wired for the new conditioning.** See §9. The
  shipped O3 recipe therefore keeps the new keys off and is unchanged.
* **Frozen Aurora latent-feature conditioning is still not implemented.**
  `conditioning.aurora_features` remains rejected.
* **The backbone `doy` defect is documented, not fixed.** Fixing it requires a
  versioned option plus a matched baseline and retraining analysis.
* **Data-coverage auditing, purged chronological splits and training-only
  threshold persistence** are unchanged; the new scores accept
  caller-supplied thresholds but nothing here estimates or persists them.
* **The CRPS/twCRPS terms are scoring functions, not yet training objectives.**
  They are not wired into `LossConfig`, the trainer or the logger.
* **`tendency_loss` is implemented and tested but not yet wired into
  `LossConfig`.**
* The historical `MAMBA_ABLATION.md` conclusions were **not** re-run; §2.2
  explains why that study could not have separated "temporal modelling does not
  help" from "temporal context never reached the generative process".
