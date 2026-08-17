# Controlled temporal-Mamba ablation

This study tests whether temporal Mamba adds skill to a spatial refinement head. It is not a comparison of unrelated historical runs.

## Completed bounded study (2026-08-14)

### Outcome and scope

The controlled result is negative: temporal Mamba did not improve any of the
eight dataset/head combinations consistently enough to promote. All 24
case/head/seed candidates failed the validation no-harm guard. Relative to the
exactly matched Mamba-off spatial refiner, ordered Mamba worsened the
equal-target-by-lead held-out RMSE by 2.25% on average for NO₂ and 1.17% for O₃.
Keep Mamba disabled by default for all four heads on both datasets.

Every seed/head pair trained one spatial refiner and reused its exact checkpoint
for Mamba-off, ordered Mamba-on, and shuffled-history evaluation. Spatial
parameters were frozen before temporal training. All arms used the same cached
raw Aurora fields, chronological split, normalization, seed, batch construction,
deterministic sampler, and held-out initializations. The invariant was:

~~~text
target residual = CAMS truth - original Aurora rollout
refined forecast = original Aurora rollout + spatial residual + temporal residual
~~~

The study used seeds 11, 29, and 47, six leads (12 through 72 hours), 12 spatial
epochs, 24 temporal-control epochs, deterministic ensemble-size-one inference,
and a 72-hour purge around split boundaries.

| Case | Train initializations | Validation initializations | Test initializations | Spatial grid |
| --- | ---: | ---: | ---: | --- |
| NO₂ US-WEST | 26, 2024-06-30 through 2024-08-16 | 10, 2024-08-20 through 2024-09-06 | 10, 2024-09-10 through 2024-09-27 | native 53 x 70 |
| Global O₃ | 19, 2024-06-30 through 2024-08-15 | 7, 2024-08-20 through 2024-09-04 | 8, 2024-09-09 through 2024-09-27 | latitude/longitude stride 8 |

The asserted phase-1 checkpoint SHA256 is
**e0186f46fff92df0eb5975471e6117b0812a2ed01558544930b7ec38666c85f5**.
The legacy raw-rollout NetCDFs do not record the checkpoint that generated
them, so every controlled manifest correctly records
**phase1_rollout_linkage.status=asserted_not_proven**. The raw-corpus digest is
a path/size/mtime fingerprint, not a content SHA256.

The runner calls the repository's production diffusion/flow and U-Net/
Transformer classes, but this bounded experiment uses compact benchmark
architectures: U-Net hidden width 32 and Transformer width 192 with four
blocks, versus the larger and longer-running example YAMLs (for example,
Transformer width 256 with six blocks and NO₂ U-Net width 64). Flow integration
uses one step in the bounded runner, and global O₃ is strided. These results are
controlled regression evidence about the temporal adapter, not exact
production-workflow or native-resolution evidence.

### Exact completed commands

~~~bash
/home/azureuser/miniforge3/envs/aurora/bin/python finetune/run_mamba_ablation_study.py \
  --case no2_uswest \
  --heads diffusion_unet diffusion_transformer flow_matching_conv_unet flow_matching_transformer \
  --seeds 11 29 47 \
  --output-dir finetune/outputs/mamba_ablation_controlled_20260814 \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --spatial-epochs 12 --temporal-epochs 24 --max-initializations 48 \
  --lat-stride 1 --lon-stride 1 --batch-size 12 \
  --spatial-learning-rate 3e-4 --temporal-learning-rate 5e-4 \
  --validation-fraction .2 --test-fraction .2 --purge-hours 72 --device cuda

/home/azureuser/miniforge3/envs/aurora/bin/python finetune/run_mamba_ablation_study.py \
  --case o3_global \
  --heads diffusion_unet diffusion_transformer flow_matching_conv_unet flow_matching_transformer \
  --seeds 11 29 47 \
  --output-dir finetune/outputs/mamba_ablation_controlled_20260814 \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --spatial-epochs 12 --temporal-epochs 24 --max-initializations 36 \
  --lat-stride 8 --lon-stride 8 --batch-size 12 \
  --spatial-learning-rate 3e-4 --temporal-learning-rate 5e-4 \
  --validation-fraction .2 --test-fraction .2 --purge-hours 72 --device cuda
~~~

The strict evaluator was run once per head. This exact NO₂ Diffusion U-Net
invocation shows the three-seed pairing; change only case/config/head/output to
run the other completed heads, as in the general evaluator examples below.

~~~bash
.venv/bin/python finetune/evaluate_mamba_ablation.py \
  --config finetune/aurora_NO2_finetune_US-WEST_3day_lead_config.yaml \
  --aurora-dir examples/outputs/cams_rollouts \
  --off-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_11/off \
  --on-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_11/on \
  --shuffled-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_11/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --off-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_29/off \
  --on-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_29/on \
  --shuffled-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_29/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --off-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_47/off \
  --on-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_47/on \
  --shuffled-dir finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest/diffusion_unet/seed_47/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --output-dir finetune/outputs/mamba_ablation_controlled_20260814/evaluation/no2_uswest/diffusion_unet \
  --bootstrap-resamples 2000 --bootstrap-block-length 6
~~~

The NO₂ regional evaluations additionally use
**--region california,32,42,235,246**,
**--region pacific_northwest,42,50,235,246**, and
**--region interior_west,32,50,246,259.6**, and write below
**evaluation_regions/no2_uswest**.

### Validation selection and held-out RMSE

Ratios below are equal-weight means over target x forecast lead; lower is
better. Ordered/off isolates the temporal adapter. Off/Aurora and
ordered/Aurora verify skill relative to the raw deterministic rollout.
Candidate names are in seed order 11/29/47.

| Case | Head | Selected candidate(s) | Validation ordered/off | Test ordered/off (seed range) | Shuffled/off | Off/Aurora | Ordered/Aurora | Promoted |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NO₂ | Diffusion U-Net | wider/wider/wider | 1.0082 | 1.0100 (1.0053-1.0132) | 1.0113 | 0.8015 | 0.8094 | 0/3 |
| NO₂ | Diffusion Transformer | wider/wider/wider | 1.0196 | 1.0265 (1.0122-1.0343) | 1.0337 | 0.8190 | 0.8394 | 0/3 |
| NO₂ | Flow U-Net | wider/compact/wider | 1.0128 | 1.0254 (1.0232-1.0280) | 1.0291 | 0.7799 | 0.7939 | 0/3 |
| NO₂ | Flow Transformer | compact/wider/reference | 1.0168 | 1.0279 (1.0164-1.0408) | 1.0695 | 0.8026 | 0.8186 | 0/3 |
| O₃ | Diffusion U-Net | wider/wider/wider | 1.0104 | 1.0090 (1.0064-1.0113) | 1.0090 | 0.9932 | 1.0020 | 0/3 |
| O₃ | Diffusion Transformer | wider/wider/wider | 1.0134 | 1.0138 (1.0073-1.0242) | 1.0138 | 0.9926 | 1.0063 | 0/3 |
| O₃ | Flow U-Net | wider/wider/wider | 1.0109 | 1.0102 (1.0066-1.0144) | 1.0102 | 0.9904 | 1.0004 | 0/3 |
| O₃ | Flow Transformer | wider/wider/wider | 1.0133 | 1.0138 (1.0078-1.0255) | 1.0139 | 0.9866 | 1.0002 | 0/3 |

Across all 24 seed/head pairs, ordered Mamba-on improved none. The composite
temporal objective nevertheless decreased by 7.20% for NO₂ and 1.00% for O₃,
while validation physical RMSE worsened by 1.43% and 1.20%, respectively.
Optimizing the composite objective did not optimize physical forecast quality.

### Strict paired-bootstrap diagnostics

Positive percentages below mean Mamba-on error increased relative to matched
Mamba-off; negative percentages mean it decreased. Correlations are absolute
deltas. The physical per-target estimates and paired 95% confidence intervals
are in **paired_summary_overall.csv** and **paired_summary_by_lead.csv**.

The evaluator's anomaly-correlation field is the spatial-mean anomaly/pattern
correlation for each individual field. It is numerically the same as spatial
Pearson correlation and is not independent climatological ACC evidence.

| NO₂ head | MAE | RMSE | Pattern RMSE | Absolute bias | Spatial corr. | Gradient RMSE | W1 distance | P99 error | P99.9 error | Tendency RMSE | Temporal corr. | Lag-1 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Diffusion U-Net | +9.87% | +0.91% | +0.05% | +31.57% | -0.00654 | -0.11% | +28.12% | -3.13% | -1.76% | +1.35% | -0.00875 | +72.36% |
| Diffusion Transformer | +15.90% | +2.63% | +1.30% | +58.75% | -0.01268 | -0.02% | +45.56% | -0.52% | -2.07% | +3.16% | -0.00549 | +64.93% |
| Flow U-Net | +15.54% | +2.33% | +0.20% | +35.99% | -0.01410 | -0.33% | +36.39% | -5.91% | -1.92% | +0.99% | -0.00370 | +65.96% |
| Flow Transformer | +22.75% | +2.62% | +0.68% | +55.54% | -0.02138 | -1.19% | +54.39% | -10.93% | -4.91% | -0.65% | -0.00925 | +43.98% |

NO₂ cell-better fractions were only 0.387, 0.349, 0.347, and 0.340 in the
same head order; RMSE case-worse fractions were 0.667, 0.917, 0.842, and
0.758. Upper-tail and occasional gradient/tendency gains came with materially
worse MAE, bias, distribution distance, spatial correlation, and temporal
autocorrelation.

| O₃ head | MAE | RMSE | Pattern RMSE | Absolute bias | Spatial corr. | Gradient RMSE | W1 distance | P99 error | P99.9 error | Tendency RMSE | Temporal corr. | Lag-1 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Diffusion U-Net | +1.71% | +0.75% | +0.32% | +46.59% | -0.00049 | +0.01% | +6.33% | -18.93% | -9.48% | -0.02% | +0.00312 | -0.47% |
| Diffusion Transformer | +2.26% | +1.25% | +0.83% | +56.43% | -0.00067 | +0.04% | +9.14% | -12.94% | -6.49% | -0.04% | +0.00162 | +0.37% |
| Flow U-Net | +1.84% | +0.89% | +0.35% | +47.86% | -0.00048 | +0.02% | +7.61% | -18.56% | -9.58% | +0.00% | +0.00224 | -0.24% |
| Flow Transformer | +2.28% | +1.27% | +0.70% | +54.68% | -0.00056 | +0.03% | +9.75% | -16.73% | -8.50% | -0.03% | +0.00165 | +0.04% |

O₃ cell-better fractions were 0.469, 0.458, 0.473, and 0.460; RMSE
case-worse fractions were 0.767, 0.875, 0.783, and 0.883. Ordered and shuffled
histories were virtually indistinguishable in O₃: ordered/shuffled RMSE changed
by only +0.01% to +0.05%. This is not useful lead-order learning. NO₂ was
order-sensitive, especially for Flow Transformer, but ordered Mamba was still
worse than Mamba-off.

### Per-target, lead, and region RMSE

An asterisk marks a paired 95% interval excluding zero in the degradation
direction; no target had a significant improvement.

| NO₂ target | Diffusion U-Net | Diffusion Transformer | Flow U-Net | Flow Transformer |
| --- | ---: | ---: | ---: | ---: |
| no2 1000 hPa | +2.119% | +4.046%* | +0.553% | -2.296% |
| no2 925 hPa | -0.230% | +1.253% | +0.175% | +3.103% |
| no2 850 hPa | +0.800%* | +1.357%* | +1.024%* | +1.499%* |
| tcno2 | +0.961%* | +3.854%* | +7.561%* | +8.170%* |

| O₃ target | Diffusion U-Net | Diffusion Transformer | Flow U-Net | Flow Transformer |
| --- | ---: | ---: | ---: | ---: |
| go3 1000 hPa | +0.164% | +0.180% | +0.187% | +0.205% |
| go3 500 hPa | +1.161%* | +1.512%* | +1.125%* | +1.167%* |
| go3 100 hPa | +0.471% | +0.969%* | +0.560% | +1.262%* |
| go3 50 hPa | +0.531% | +1.429%* | +1.027% | +1.520%* |
| gtco3 | +1.428% | +2.173%* | +1.534% | +2.184% |

Every head's mean RMSE worsened at every lead. At 12/24/36/48/60/72 hours:

- NO₂ Diffusion U-Net: +1.01%, +1.47%, +0.90%, +1.04%, +0.68%, +0.78%;
- NO₂ Diffusion Transformer: +2.56%, +3.02%, +2.86%, +2.62%, +2.53%, +2.39%;
- NO₂ Flow U-Net: +1.63%, +4.48%, +1.67%, +3.19%, +1.69%, +2.40%;
- NO₂ Flow Transformer: +1.96%, +5.90%, +1.24%, +2.95%, +2.02%, +3.22%;
- O₃ Diffusion U-Net: +1.37%, +1.51%, +1.01%, +0.69%, +0.46%, +0.36%;
- O₃ Diffusion Transformer: +1.95%, +1.79%, +1.38%, +1.30%, +1.02%, +0.93%;
- O₃ Flow U-Net: +1.50%, +1.57%, +1.09%, +0.79%, +0.61%, +0.56%;
- O₃ Flow Transformer: +1.86%, +1.81%, +1.44%, +1.28%, +1.01%, +0.97%.

Region-stratified mean RMSE changes were also uniformly harmful:

| NO₂ region | Diffusion U-Net | Diffusion Transformer | Flow U-Net | Flow Transformer |
| --- | ---: | ---: | ---: | ---: |
| California | +0.739% | +1.769% | +1.241% | +0.766% |
| Pacific Northwest | +0.768% | +1.110% | +1.339% | +1.522% |
| Interior West | +0.313% | +3.472% | +2.095% | +4.428% |

| O₃ region | Diffusion U-Net | Diffusion Transformer | Flow U-Net | Flow Transformer |
| --- | ---: | ---: | ---: | ---: |
| Northern extratropics | +4.277% | +4.303% | +4.240% | +5.205% |
| Tropics | +1.611% | +2.334% | +1.659% | +1.918% |
| Southern extratropics | +0.115% | +0.108% | +0.202% | +0.183% |

Raw-baseline checks remain essential. NO₂ Mamba-off improved raw Aurora by
18.71% to 22.71% RMSE, and Mamba-on still improved it by 16.70% to 21.44%,
but every on arm was worse than its matched off arm. For O₃, Mamba-off improved
Aurora by 0.86% to 1.59% RMSE; Mamba-on retained only a 0.19% to 0.36% pooled
gain for three heads and made Diffusion Transformer 0.37% worse than Aurora.

### Root causes and safeguards

Historical results cannot support a Mamba-skill claim:

- commit 9acff6b trained temporal correction on deterministic spatial outputs
  with raw-Aurora feedback, while inference used stochastic corrections and
  recursively fed refined plus Mamba output;
- validation never applied legacy Mamba, so checkpoint selection ignored
  deployed temporal forecast quality;
- exposure-bias fix e966ce3 is on a side branch, not an ancestor of this branch;
- the former optional CUDA module was instantiated lazily after optimizer
  construction and did not share registered reference-S6 weights; available
  checkpoints contain no lazy-module weights;
- historical notebooks loaded last.ckpt, changed sampler steps, and omitted
  checkpoint/sampler/temporal/trajectory/seed provenance.

The repaired packed adapter has explicit [batch, lead, channel, latitude,
longitude] semantics. It spatially encodes each lead, scans only the lead axis
for corresponding latent grid locations, enforces chronological physical leads
and prefix masks, keeps initializations separate, and conditions on lead,
spacing, masks, fixed packed channel/level identity, and optional coordinates.
Its causal per-channel ReZero fusion gate starts at zero. Parameters are eagerly
registered, optimizer-visible, checkpointed, and restored. The interface is
shared by all four refinement heads.

Fail-closed behavior is deliberate:

- stateless AuroraTwoPhaseRefiner.forward rejects an enabled temporal module
  and directs callers to the chronological sequence-aware rollout path;
- a legacy temporal checkpoint missing top-level flow_sampling_steps provenance
  is rejected even if an unsafe override is supplied; a proven checkpoint
  sampler is restored, and changing it requires explicit unsafe diagnostic
  opt-in;
- noncausal mode, fewer than two leads, nonconsecutive rollout indices,
  out-of-order leads, invalid masks, and incompatible checkpoints are rejected;
- Mamba-disabled configurations instantiate no temporal module and preserve the
  spatial output exactly.

The unified NO₂ and global-O₃ example YAMLs are intentionally a one-key toggle:
model.mamba_temporal.enabled remains false, while
training.mamba_temporal_weight is positive but ignored while disabled. Changing
only enabled instantiates and trains the temporal branch; the paired runner,
not these default YAMLs (whose mamba_temporal_only remains false), handles
temporal-only freezing. The loader requires the positive temporal weight once
enabled. This is configuration convenience, not a recommendation to enable it.

The corrected implementation establishes real temporal computation, but not
utility. Fusion gates stayed small, limiting damage, while the tail/
structure-heavy objective rewarded changes that improved some extremes but
worsened bulk error, bias, distribution distance, and spatial agreement. O₃
shuffled parity shows mostly local/lead-conditioned calibration rather than
order skill. NO₂ has order sensitivity, but its learned tendency is not
consistently aligned with the true residual tendency. The post-head adapter
also does not yet consume internal U-Net multi-scale or Transformer patch
latents.

### Recommendation, limitations, and artifacts

Set model.mamba_temporal.enabled to false for all four heads on both datasets.
Every retained temporal checkpoint records deployment_recommendation=disable.
The best bounded non-Mamba spatial head was Flow U-Net for NO₂ (off/Aurora RMSE
-22.71%, MAE -23.41%, spatial-correlation delta +0.00737) and Flow Transformer
for O₃ (RMSE -1.59%, MAE -2.24%, spatial-correlation delta +0.00026).

Reconsider Mamba only after a validation-only redesign that selects checkpoints
on physical forecast metrics, integrates temporal blocks at native head latents,
and passes the same no-harm guard at native O₃ resolution over more seasons and
seeds. Ordered history must beat both off and shuffled without harming later
leads, extremes, distribution, or spatial structure.

Remaining evidence limitations are compact benchmark architectures, short
training, stride-8 O₃, three seeds in one late-summer period, deterministic
one-member scientific evaluation, post-head rather than internal-latent
integration, and asserted rather than proven phase-1/raw-rollout lineage. The
optimized mamba_ssm backend is not installed in the available environment; the
self-contained reference S6 implementation is the backend tested here.

Run records and strict outputs are under:

~~~text
finetune/outputs/mamba_ablation_controlled_20260814/no2_uswest_paired_study.json
finetune/outputs/mamba_ablation_controlled_20260814/o3_global_paired_study.json
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/no2_uswest/<head>/
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/o3_global/<head>/
finetune/outputs/mamba_ablation_controlled_20260814/evaluation_regions/no2_uswest/<head>/
~~~

Each head directory contains **evaluation_summary.md**,
**paired_summary_overall.csv**, **paired_summary_by_lead.csv**, per-case and
temporal CSVs, an **evaluation_manifest.json**, and a **figures/** directory.
Representative exact plot paths are:

~~~text
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/no2_uswest/diffusion_unet/figures/representative_map_tcno2_surface.png
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/no2_uswest/diffusion_unet/figures/temporal_trajectory_tcno2_surface.png
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/no2_uswest/diffusion_unet/figures/hotspot_trajectory_tcno2_surface.png
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/o3_global/diffusion_unet/figures/representative_map_gtco3_surface.png
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/o3_global/diffusion_unet/figures/temporal_trajectory_gtco3_surface.png
finetune/outputs/mamba_ablation_controlled_20260814/evaluation/o3_global/diffusion_unet/figures/hotspot_trajectory_gtco3_surface.png
~~~

Implementation paths are grouped as follows:

- temporal model and workflow:
  **finetune/mamba_temporal.py**, **finetune/model_factory.py**,
  **finetune/aurora_finetune_utils.py**,
  **finetune/aurora_finetune_distributed.py**, and
  **finetune/refinement/{integration,two_phase,benchmark}.py**;
- controlled study and evaluation:
  **finetune/run_mamba_ablation_study.py** and
  **finetune/evaluate_mamba_ablation.py**;
- workflow compatibility: the four primary NO₂ YAMLs, the global-O₃ YAML and
  four unified stochastic O₃ examples, **finetune/aurora_inference_rollout.ipynb**,
  and the O₃ launch-script overwrite guards;
- verification: **tests/test_mamba_temporal.py**,
  **tests/test_mamba_ablation_study.py**,
  **tests/test_mamba_ablation_evaluation.py**, and the refinement/config/
  checkpoint compatibility suites.

The full local suite passed: 793 passed, 4 skipped, with 257 warnings in 177.36 seconds. The skips were CUDA
mixed-precision cases in .venv; the same four all-head CUDA/bfloat16 cases
passed in the available Aurora environment on an H100 NVL. No environment
literally named Prithvi is installed; the working environment is
/home/azureuser/miniforge3/envs/aurora.

~~~bash
.venv/bin/pytest -q
/home/azureuser/miniforge3/envs/aurora/bin/python -m pytest -q \
  tests/test_mamba_temporal.py::test_packed_temporal_all_heads_cuda_mixed_precision_forward_backward
~~~

## Fair comparison contract

For every case, spatial head, and random seed:

1. Select initialization dates uniformly over the eligible raw-rollout period.
2. Split those initialization groups chronologically into train, validation, and held-out test sets, with a purge gap.
3. Train the spatial refinement head once on train and save its exact checkpoint.
4. Evaluate Mamba-off from that checkpoint.
5. Freeze every spatial parameter, train only temporal parameters on train, and select the temporal checkpoint on validation.
6. Evaluate ordered Mamba-on and shuffled-history control on the identical held-out test initializations.
7. Compare raw Aurora, Mamba-off, Mamba-on, and shuffled history only after exact target, lead, initialization, valid-time, and coordinate matching.

The cached-rollout phase has no Aurora autoregressive feedback. Its manifest must record:

```json
{
  "trajectory_policy": {
    "source": "cached_raw_rollout",
    "aurora_autoregressive_feedback": false
  }
}
```

Never tune loss weights, temporal architecture, promotion thresholds, or checkpoint selection on the held-out test metrics.

## Phase-1 artifact and provenance limitation

The available asserted phase-1 artifact is:

```text
/home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt
SHA256 e0186f46fff92df0eb5975471e6117b0812a2ed01558544930b7ec38666c85f5
```

The evaluator hashes the supplied file and verifies every paired manifest against its bytes. The legacy raw-rollout NetCDFs do not carry a source manifest proving they were generated by this checkpoint. Manifests therefore correctly record `phase1_rollout_linkage.status=asserted_not_proven`; do not claim stronger lineage. The raw corpus digest is a reproducible path/size/mtime fingerprint, not a content SHA256.

## Generate controlled paired runs

The runner refuses to overwrite a non-empty case/head/seed directory unless `--allow-overwrite` is explicitly supplied.

Confirmatory NO₂ US-WEST run:

```bash
.venv/bin/python finetune/run_mamba_ablation_study.py \
  --case no2_uswest \
  --heads diffusion_unet diffusion_transformer flow_matching_conv_unet flow_matching_transformer \
  --seeds 11 29 47 \
  --output-dir finetune/outputs/mamba_ablation_confirmatory \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --max-initializations 120 \
  --spatial-epochs 12 \
  --temporal-epochs 12 \
  --batch-size 12 \
  --validation-fraction 0.2 \
  --test-fraction 0.2 \
  --purge-hours 72 \
  --device cuda
```

Confirmatory global O₃ run with a reproducible 4×4 spatial stride:

```bash
.venv/bin/python finetune/run_mamba_ablation_study.py \
  --case o3_global \
  --heads diffusion_unet diffusion_transformer flow_matching_conv_unet flow_matching_transformer \
  --seeds 11 29 47 \
  --output-dir finetune/outputs/mamba_ablation_confirmatory \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --max-initializations 120 \
  --lat-stride 4 \
  --lon-stride 4 \
  --spatial-epochs 12 \
  --temporal-epochs 12 \
  --batch-size 12 \
  --validation-fraction 0.2 \
  --test-fraction 0.2 \
  --purge-hours 72 \
  --device cuda
```

A short plumbing pilot may use `--max-initializations 36 --spatial-epochs 2 --temporal-epochs 2`, but its small held-out sample is not confirmatory evidence. Global O₃ at native 451×900 resolution is substantially more expensive; the spatial stride is part of the study definition and must be identical for all arms.

For each head and seed, artifacts are written under:

```text
<output>/<case>/<head>/seed_<seed>/
  spatial_checkpoint.pt
  temporal_checkpoint.pt
  off/
  on/
  shuffled/
  study_result.json
```

Each arm directory contains timestamped NetCDFs and `mamba_ablation_manifest.json`. Mamba-off/on/shuffled share the exact spatial checkpoint hash. Ordered and shuffled share the exact phase-2 checkpoint hash.

The shuffled arm deliberately destroys full-sequence lead order, permutes fields, masks, and lead values together, then inverse-maps the correction before reconstruction. It can expose a later physical lead after permutation, so it is a test-only order-sensitivity control, not deployable causal inference.

## Evaluate one head strictly

The raw Aurora input is the existing corpus, not a fourth runner arm. `--aurora-dir` defaults to the YAML's `evaluation.baseline_rollout_path`; specifying it makes the evidence path explicit. The directory may cover a larger grid or contain additional dates/leads, but every selected test initialization and lead must match exactly within configured coordinate tolerance. The evaluator never interpolates.

Example for NO₂ diffusion U-Net:

```bash
.venv/bin/python finetune/evaluate_mamba_ablation.py \
  --config finetune/aurora_NO2_finetune_US-WEST_3day_lead_config.yaml \
  --aurora-dir examples/outputs/cams_rollouts \
  --off-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_11/off \
  --on-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_11/on \
  --shuffled-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_11/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --off-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_29/off \
  --on-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_29/on \
  --shuffled-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_29/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --off-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_47/off \
  --on-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_47/on \
  --shuffled-dir finetune/outputs/mamba_ablation_confirmatory/no2_uswest/diffusion_unet/seed_47/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --output-dir finetune/outputs/mamba_ablation_evaluation/no2_diffusion_unet \
  --bootstrap-resamples 2000 \
  --bootstrap-block-length 6
```

Example for the corresponding strided global-O₃ head:

```bash
.venv/bin/python finetune/evaluate_mamba_ablation.py \
  --config finetune/aurora_O3_global_finetune_3day_lead_config.yaml \
  --aurora-dir examples/outputs/cams_rollouts \
  --off-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_11/off \
  --on-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_11/on \
  --shuffled-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_11/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --off-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_29/off \
  --on-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_29/on \
  --shuffled-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_29/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --off-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_47/off \
  --on-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_47/on \
  --shuffled-dir finetune/outputs/mamba_ablation_confirmatory/o3_global/diffusion_unet/seed_47/shuffled \
  --phase1-checkpoint /home/azureuser/.cache/huggingface/hub/models--microsoft--aurora/snapshots/1764d5630a53d3d7a7d169ca335236fc343e4bfc/aurora-0.4-air-pollution.ckpt \
  --output-dir finetune/outputs/mamba_ablation_evaluation/o3_diffusion_unet_stride4 \
  --bootstrap-resamples 2000 \
  --bootstrap-block-length 6
```

Run the evaluator separately for each of the four head names. Do not combine different spatial heads in one paired invocation.

When evaluating an older full held-out corpus rather than controlled runner output, `--initialization-stride 6 --bootstrap-block-length 1` provides a season-spanning lower-cost O₃ pass. Strict runner manifests remain mandatory for controlled attribution.

## Statistical and diagnostic outputs

The evaluator writes:

- per-case field, tendency, and temporal-sequence CSVs;
- paired summaries by lead and overall;
- cell-improvement and predicted-correction-tendency summaries;
- a concise `evaluation_summary.md`;
- an `evaluation_manifest.json` with checkpoint, input, split, selected-date, output, and figure provenance;
- bounded lead-time plots for domain-level key comparisons;
- one longest-lead CAMS/Aurora/off/on map plus difference panel per configured target;
- one representative domain-mean/RMSE temporal trajectory per configured target.

Metrics cover every configured variable/level/lead and requested region: MAE, pooled RMSE, bias, centered/pattern RMSE, spatial/anomaly correlation, standard-deviation ratio, gradient structure error, Wasserstein-1 and CDF distances, P50/P75/P90/P95/P99/P99.9, tail MAE, exceedance bias, maxima, cell-better/worse fractions, tendency RMSE/correlation/variance, correction-vs-true-residual tendency amplitude/correlation, temporal correlation, and lag-one autocorrelation.

Confidence intervals use a paired hierarchical bootstrap: resample seeds, then circular chronological initialization blocks within each selected seed. Pixels are never bootstrap units. Summary rows report seed count, unique initialization count, and seed×initialization pair count. RMSE-family metrics pool squared errors before taking the square root.

A temporal-skill claim requires ordered Mamba-on to improve over raw Aurora and Mamba-off, and to beat shuffled history, without materially worsening tail, distribution, spatial-structure, or case-worse diagnostics. A temporal candidate retained for analysis is not automatically recommended for deployment; use the recorded validation promotion decision.

## Focused verification

```bash
.venv/bin/ruff check finetune/evaluate_mamba_ablation.py tests/test_mamba_ablation_evaluation.py
.venv/bin/pytest -q tests/test_mamba_ablation_evaluation.py
```
