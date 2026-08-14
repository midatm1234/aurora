# NO2 US-WEST three-day refinement review

## Executive conclusion

This review traced the NO2_US-WEST_3day_lead workflow from Aurora rollout generation through target alignment, residual construction, normalization, refinement training, deterministic and stochastic inference, autoregressive feedback, checkpoint selection, and evaluation.

The saved historical products do contain useful correction signal, but their headline improvements conceal distinct physical failures:

- The legacy Flow Matching result corrects the tcno2 mean but heavily smooths every field, harms vertical-level MAE and spatial correlation, worsens most vertical distributions and tails, and changes the column field too abruptly between leads.
- Historical Diffusion is the strongest existing saved product. It improves MAE and RMSE for all four targets, improves vertical-level correlation, and gives the best column variance and distribution agreement. It still degrades tcno2 spatial and temporal correlation and overcorrects the nearly unbiased 850 hPa mean.
- Both historical Transformer products improve vertical-level MAE, RMSE, correlation, tails, and Wasserstein distance. They inject excess tcno2 variance and high-frequency structure, worsen column pattern and temporal agreement, and overcorrect the 850 hPa mean.
- The corrected controlled experiment selects unified Flow Matching with a convolutional U-Net as the best-performing architecture. By 24 short epochs it lowers aggregate vertical MAE by 9.4–14.4% and RMSE by 15.6–17.3%, reduces aggregate bias and tails, and improves MAE and spatial correlation at every target and lead. The exact guard still withholds promotion because two 12-hour vertical biases exceed its conservative floor.
- The Transformer heads also lower every MAE and RMSE after 12 short epochs, but spatial correlation remains 0.0026–0.0072 below Aurora. Strict checkpoint guards therefore refuse to promote them until pattern agreement also stops degrading.

All pre-existing saved rollouts and checkpoints predate at least part of the corrected contract. They must be retrained. The controlled run demonstrates that the corrected implementation can learn useful residuals; it is not a substitute for a full 100-epoch, full-training-period production result.

## Evidence and scope

The full historical audit used all 184 common rollout initializations. There were 1,083 valid initialization/lead combinations after 21 requested steps beyond the CAMS truth time range were excluded by exact-time matching. Four targets were evaluated at each combination, producing 4,332 variable/level/case/lead rows:

- total-column NO2, tcno2;
- NO2 at 1000 hPa;
- NO2 at 925 hPa;
- NO2 at 850 hPa.

No nearest-time matching, interpolation, spatial resizing, or array-position level matching was used. The regenerated summaries are under:

~~~text
finetune/outputs/refinement_review/full_diagnostics/
~~~

The controlled comparison uses 24 raw Aurora initializations, a grouped and temporally purged split with 78 training and 36 held-out lead samples, the same seed 42, batch size 8, learning rate 3e-4, and CUDA for every head. The ignored machine-readable results are:

~~~text
finetune/outputs/refinement_review/before_four_head.json
finetune/outputs/refinement_review/controlled/after_four_head.json
finetune/outputs/refinement_review/controlled/after_unets_12epoch.json
finetune/outputs/refinement_review/controlled/after_unets_24epoch.json
finetune/outputs/refinement_review/controlled/after_transformers_12epoch.json
~~~

## Historical saved-product results

Positive MAE/RMSE percentages mean improvement relative to the exact same Aurora field. Correlation is refined minus Aurora.

| saved method | tcno2 MAE / RMSE / correlation | vertical NO2 MAE range | vertical NO2 RMSE range | vertical correlation range |
|---|---:|---:|---:|---:|
| Legacy Flow Matching | +62.98% / +38.37% / -0.0931 | -16.87% to -2.91% | +6.12% to +7.41% | -0.0635 to -0.0517 |
| Diffusion | +65.54% / +41.67% / -0.0533 | +11.43% to +18.42% | +15.30% to +19.28% | +0.0063 to +0.0133 |
| Diffusion Transformer | +53.66% / +30.92% / -0.0885 | +5.65% to +11.16% | +15.28% to +19.25% | +0.0111 to +0.0187 |
| Flow Matching Transformer | +52.34% / +29.49% / -0.0847 | +8.50% to +12.90% | +16.39% to +20.59% | +0.0170 to +0.0251 |

Bias behavior is not uniform:

- Legacy Flow reduces absolute bias for tcno2, 1000 hPa, and 925 hPa, but changes the nearly unbiased 850 hPa Aurora field into a 5.66e-11 biased field.
- Diffusion reduces absolute bias for tcno2, 1000 hPa, and 925 hPa, but worsens 850 hPa from 1.70e-12 to 4.99e-11.
- Diffusion Transformer reduces tcno2 and 1000 hPa bias but worsens 925 and 850 hPa.
- Flow Matching Transformer reduces tcno2 and 1000 hPa bias but worsens 925 and 850 hPa.

### Distribution, extremes, and artifacts

Legacy Flow is strongly underdispersed. Its refined standard-deviation ratios are 0.666 for tcno2 and 0.635–0.675 for the vertical levels. Its vertical Wasserstein distances and P99 tail MAE are worse than Aurora, only 34–42% of vertical grid points improve, and only 31–36% of vertical forecast cases improve. It produced 8–12% negative concentration values in the audited output. The field maps and gradient/Laplacian diagnostics show plume loss and broad smoothing rather than a physically coherent local correction.

Historical Diffusion is the most balanced saved result. Its tcno2 standard-deviation ratio is 0.922 and its vertical ratios are 0.704–0.793. It improves most Wasserstein and P99 diagnostics, although no2 at 850 hPa remains slightly worse in Wasserstein distance and the vertical fields remain underdispersed.

The saved Diffusion Transformer has a tcno2 standard-deviation ratio of 1.165; Flow Matching Transformer reaches 1.228. Their vertical ratios, 0.794–0.897, are closer to CAMS than the legacy flow result, but the column field has excess variance and high-frequency energy. On the 51 by 69 regional grid, the old patch size of 8 reduced the field to only 7 by 9 tokens and compressed 576 patch scalars into a 256-dimensional token, explaining plume smoothing, patch texture, and weak local reconstruction.

All saved methods underrepresent the highly skewed vertical NO2 distribution. CAMS vertical skewness is about 7.7–9.8 with excess kurtosis about 173–405. Legacy Flow collapses this to skewness 2.3–3.1 and excess kurtosis 7.7–17.1. Diffusion retains more tail structure, and the Transformers retain still more, but none reproduces the full CAMS tail.

### Temporal behavior

Tendencies are per-hour changes between consecutive evaluated leads.

| saved method | tcno2 tendency-error change | vertical tendency-error improvement | main temporal finding |
|---|---:|---:|---|
| Legacy Flow | -9.82% | +10.5% to +12.3% | column transitions worsen; vertical fields become too smooth |
| Diffusion | -9.37% | +20.8% to +24.3% | best vertical temporal behavior; column still worsens |
| Diffusion Transformer | -29.4% | +18.9% to +23.2% | excessive column correction-tendency amplitude |
| Flow Matching Transformer | -29.2% | +20.0% to +24.6% | excessive column correction-tendency amplitude |

The Transformer tcno2 correction-tendency amplitude ratios are 1.26 and 1.29, while their correlation with the true residual tendency is only 0.226 and 0.249. This is temporally structured noise, not merely a mean-bias issue.

## Root causes

### Residual and normalization contract

The current target path is correctly defined as:

~~~text
true_correction = CAMS_truth - Aurora_rollout
refined = Aurora_rollout + predicted_correction
~~~

The same valid lead, packed variable order, pressure-level order, mask, and normalization metadata are used on both sides, and the correction is decoded and added once.

The important bug was subtler. Every unified recipe centers the residual scaler. In that coordinate system, scaled value zero decodes to the training residual mean, not to a literal zero correction. The magnitude and degradation losses nevertheless treated scaled zero as the Aurora identity. With the saved production calibration, the error was material: tcno2 scaled identity is about +0.480, while scaled zero applies a physical correction of -1.56e-6 kg m-2.

This is fixed. No-harm losses construct the scaled value that decodes to physical zero, and an untouched zero-initialized shared-process projection returns a literal zero correction before centered decoding. After any projection parameter is trained, normal centered decoding resumes. Regression tests cover the nonzero-shift round trip and prove Aurora plus the decoded true correction equals CAMS to floating-point tolerance.

### Deployed estimator did not train the advertised process

The unified implementation formerly built an independent mean network. All shipped NO2 recipes used deterministic inference, so the evaluated point forecast bypassed diffusion reverse sampling and flow integration entirely. A comparison labeled diffusion versus flow was therefore mainly a comparison of separately trained mean backbones.

A versioned deterministic_head option now supports shared_process. All four NO2 recipes use it so the same conditioned process backbone is trained and deployed for the deterministic correction. The legacy separate_mean behavior remains the default for old configurations and is part of the checkpoint architecture contract.

### Diffusion regression

Commit a0bd0cf combined epsilon prediction, a zero-initialized output, and no effective residual scaling. A zero epsilon prediction is not an identity prediction: converting a unit-variance noisy latent to x0 divides by the diffusion alpha factor and yields a large random correction. The controlled legacy arm consequently degrades diffusion errors by tens to hundreds of millions of percent.

The corrected recipes predict the clean residual sample, use per-channel centered scaling, Min-SNR-compatible weighting, deterministic supervision, DDIM with eta zero, and a statistically bounded correction. Training and sampling equations now share the same parameterization and reverse-time grid. Checkpoints trained under the old epsilon contract must be retrained.

### Legacy Flow Matching regressions

Recent history exposed several compatibility and train/inference mismatches:

- a0bd0cf silently changed the old single-step query from x=0, t=1 to x=0, t=0 for every legacy checkpoint without gating behavior by contract version;
- the historical NO2 recipe switched from one deterministic step to an eight-step single-member stochastic sample late in training and used a global random generator;
- training fed unrefined Aurora states to later leads, while rollout inference recursively fed stochastic refined states;
- optional Mamba was trained on deterministic frames but inferred on stochastic recursively refined frames;
- a side-branch exposure-bias fix, e966ce3, was never inherited by the current NO2 lineage;
- a physically invalid tcno2/NO2 coherence term was accidentally activated by a0bd0cf. It combined independently normalized total column mass with only three mixing-ratio levels, so the claimed cancellation of normalization constants was false.

Legacy runtime behavior is now contract-versioned. Version 1 retains its historical endpoint and feedback semantics; version 2 and newer use the source endpoint, deterministic point default, explicit seeded stochastic opt-in, and raw-Aurora feedback unless explicitly trained otherwise. Generator state is forwarded through the legacy adapter. The production NO2 Flow recipe is migrated to unified flow_matching_conv_unet, so it no longer uses the invalid coherence or per-level legacy architecture.

### Configuration and checkpoint regressions

Commit 2dc57bc left the shipped Diffusion Transformer YAML with Mamba enabled but temporal weight zero, while its new validator rejects exactly that combination. The evaluated manifest had Mamba disabled, so the committed recipe could not reproduce its own run. Mamba is now explicitly disabled and remains optional.

The same lineage correctly stopped using test.nc for checkpoint selection, but the main NO2 Flow YAML was not migrated and pointed at a nonexistent val.nc. It now uses a temporally purged training-tail split. Earlier saved evaluations must be interpreted with the knowledge that test.nc had also served as validation.

In a0bd0cf, unified Flow examples advertised residual z-scoring through a legacy-only key that the unified refiner never consumed. That no-op left small residual endpoints competing with unit Gaussian source noise. The generic target-space scaler introduced later is now the sole authoritative scaler, is fitted exactly on training data, and is checkpointed.

Checkpoint validation also omitted deterministic_head even though switching between a separate mean network and shared process changes the deployed estimator. The active head and only its active process/backbone sections are now part of the versioned scientific contract; ignored configuration sections do not create false incompatibilities.

### Weak spatial and geophysical conditioning

The old legacy flow model handled every target and pressure level independently. It had no cross-level packed context, no latitude, no mask, and no shared tcno2/vertical representation.

All unified heads now condition on the packed Aurora tcno2 and three NO2 levels, the validity mask, forecast lead, latitude divided by 90, and periodic sine/cosine longitude. The Transformer already uses adaptive LayerNorm conditioning; explicit coordinates and lead embedding now make that conditioning geophysically identifiable. The regional Transformer patch size is reduced from 8 to 4, yielding about 13 by 18 tokens rather than 7 by 9, with local convolutional refinement retained.

Longitude remains periodic for global cases through periodic convolution/padding and sine/cosine position. The US-WEST case uses non-periodic regional padding and explicit coordinates, avoiding a false wrap across its boundaries.

### Controlled benchmark target-mask leakage

The controlled benchmark originally sanitized rollout NaNs and then used the joint Aurora/CAMS validity mask as a conditioning channel. That exposed CAMS-only missingness to the model and made an intermediate U-Net result look materially better than the production path. This was evaluation leakage, not model skill.

The dataset now preserves a separate rollout_valid tensor before sanitization. Training and inference use only that rollout-derived mask, and an exact-parity regression compares every benchmark conditioning channel with AuroraTwoPhaseRefiner.build_conditioning. All controlled results below were regenerated after this fix.

### Training distribution shift

The full production calibration used 646 logical training samples and 3,876 packed lead examples. Physical training correction mean and standard deviation were:

| target | training mean | training standard deviation |
|---|---:|---:|
| tcno2 | -1.560e-6 kg m-2 | 3.252e-6 |
| no2 1000 hPa | +8.85e-11 kg kg-1 | 2.154e-9 |
| no2 925 hPa | +1.012e-10 kg kg-1 | 1.619e-9 |
| no2 850 hPa | +1.414e-10 kg kg-1 | 1.102e-9 |

The historical test-period residual means have the opposite sign for tcno2, 1000 hPa, and 925 hPa. This seasonal drift explains why a global mean correction, especially one accidentally applied as identity, overcorrects a reasonable Aurora state.

The fixes reduce this risk through strong Aurora-state conditioning, exact lead conditioning, literal-zero initialization, a no-harm loss, a training-residual envelope, temporally purged train-tail validation, and strict per-channel/per-lead promotion guards. A full future-season retrain is still required; the current dataset cannot prove extrapolation beyond its seasonal support.

## Architecture, loss, inference, and training changes

Common production defaults now include:

- a packed four-channel residual and conditioning representation;
- shared-process deterministic heads;
- per-channel centered residual scaling fitted only on the training split;
- a correction envelope equal to training mean plus or minus four training residual standard deviations;
- direct deterministic MSE and MAE supervision;
- configurable bias, spatial-gradient, pattern-correlation, variance, quantile, upper-tail, and pointwise degradation losses;
- upper-tail-only extreme weighting at the CAMS P95 threshold;
- nonnegative physical constraints for no2 and tcno2;
- deterministic one-member validation and inference for the point product;
- stochastic generation only as an explicit, seeded ensemble experiment;
- no autoregressive feedback of an unvalidated correction;
- optional Mamba support preserved but disabled by default;
- AdamW at 3e-4, cosine warmup, gradient clipping at 1.0, bf16, batch size 1 with accumulation 8, and 100 epochs;
- a purged 10% training-tail validation split, not test.nc;
- checkpoint ranking by physical RMSE ratio, with zero tolerated degradation in MAE, absolute bias, or spatial correlation for any target/level/lead.

Checkpoint contracts now include deterministic-head architecture, correction parameterization, scaler state, target and level signature, geophysical conditioning, and validation provenance. Resume restores serialized CPU and CUDA random state after calibration so stochastic training is reproducible across interruption.

## Evaluation changes

Every diagnostic run now reports CAMS, Aurora, and refined values on the same finite cells, by variable, level, lead, and overall period. Added metrics and artifacts include:

- MAE, RMSE, centered RMSE, mean bias, pattern and anomaly correlation;
- truth/prediction standard deviation and standard-deviation ratio;
- P90, P95, P99, tail MAE, exceedance frequency, and exceedance bias;
- empirical Wasserstein distance, skewness, and excess kurtosis;
- correction amplitude/correlation/sign accuracy and improved/worsened point fractions;
- forecast-case improvement fractions and optimal correction strength;
- exact per-lead curves and target summaries;
- per-hour lead-to-lead tendency RMS, error RMSE, correlation, amplitude ratio, and correction/residual tendency correlation;
- CAMS/Aurora/refined maps, both error maps, true and predicted residual maps, scatterplots, spatial bias/RMSE maps, histograms, PDFs, CDFs, QQ plots, and upper-tail plots.

Trace-gas correlations no longer use an absolute 1e-12 variance cutoff, which incorrectly returned NaN for valid NO2 patterns. Aggregated RMSE is computed by pooling MSE and taking one square root, rather than averaging percentages or RMSE values.

## Controlled post-fix results

### Equal six-epoch budget

The table reports MAE improvement, RMSE improvement, and correlation change relative to raw Aurora. Vertical ranges cover 1000, 925, and 850 hPa.

| corrected head | tcno2 MAE / RMSE / correlation | vertical MAE range | vertical RMSE range | vertical correlation range |
|---|---:|---:|---:|---:|
| Diffusion U-Net | +67.85% / +53.21% / -0.0005 | -0.15% to +0.62% | +0.72% to +1.07% | -0.0034 to -0.0001 |
| Diffusion Transformer | +68.26% / +53.66% / -0.0028 | -0.66% to -0.41% | +0.85% to +0.99% | -0.0043 to -0.0022 |
| Flow Matching U-Net | +67.84% / +53.20% / -0.0009 | -0.29% to +0.52% | +0.19% to +0.54% | -0.0014 to +0.0001 |
| Flow Matching Transformer | +68.48% / +53.86% / -0.0031 | +0.17% to +0.20% | +1.07% to +1.18% | -0.0037 to -0.0022 |

These production-parity results supersede the more favorable intermediate U-Net proxy, which had received a target-derived validity mask. At six epochs, only Flow Matching Transformer lowers every MAE and RMSE, and no head preserves every spatial correlation.

The zero-tolerance promotion guard therefore rejects all four six-epoch candidates. That abstention is intentional: an undertrained correction cannot replace Aurora merely because its aggregate training loss fell.

### Equal twelve-epoch check

With no data, seed, batch, or learning-rate change, all four heads improve every target in aggregate MAE and RMSE after 12 epochs:

| corrected head | tcno2 MAE / RMSE / correlation | vertical MAE range | vertical RMSE range | vertical correlation range |
|---|---:|---:|---:|---:|
| Diffusion U-Net | +71.11% / +55.32% / +0.0049 | +5.16% to +6.74% | +7.06% to +10.75% | +0.0038 to +0.0056 |
| Diffusion Transformer | +74.00% / +57.57% / -0.0032 | +0.80% to +0.86% | +2.70% to +3.47% | -0.0072 to -0.0040 |
| Flow Matching U-Net | +72.93% / +56.49% / +0.0039 | +5.19% to +7.27% | +8.02% to +12.85% | +0.0036 to +0.0055 |
| Flow Matching Transformer | +74.09% / +57.57% / -0.0026 | +1.63% to +2.01% | +3.65% to +4.94% | -0.0066 to -0.0041 |

The U-Nets reduce aggregate absolute bias, P99 absolute error, and upper-tail MAE for every target while increasing every aggregate spatial correlation. At exact target/lead granularity, however, both still have sub-1.1% MAE regressions at 12 hours and are rejected by the configured guard.

Both Transformers improve P99 and tail errors at every level, but worsen 850 hPa absolute bias and every spatial correlation. The zero-tolerance checkpoint guard correctly abstains even though their tcno2 and training-loss improvements are larger.

### U-Net convergence at 24 epochs

A longer run was used only to test whether the U-Net pattern was normal convergence rather than a formulation failure:

| corrected head | tcno2 MAE / RMSE / correlation | vertical MAE range | vertical RMSE range | vertical correlation range |
|---|---:|---:|---:|---:|
| Diffusion U-Net | +78.48% / +59.80% / +0.0116 | +9.20% to +13.41% | +14.32% to +17.16% | +0.0113 to +0.0172 |
| Flow Matching U-Net | +78.71% / +59.93% / +0.0115 | +9.37% to +14.38% | +15.60% to +17.31% | +0.0124 to +0.0164 |

Both U-Nets now improve MAE and correlation for every target and every lead, and improve aggregate MAE, RMSE, absolute bias, P99 error, and tail MAE for all four targets. Flow Matching U-Net remains the strongest error/tail model. The tcno2 standard-deviation ratio also moves toward one, but vertical ratios move farther below one, so complete distribution agreement is still mixed rather than solved.

The exact guard remains deliberately stricter than these aggregates. Flow Matching U-Net exceeds the 5%-of-baseline-RMSE absolute-bias floor at 12 hours for 1000 and 925 hPa; Diffusion U-Net does so for all three 12-hour vertical levels. The Transformer correlation failures remain. Therefore no controlled checkpoint is represented as production-approved: if validation repeats these results, checkpoint promotion is vetoed and Aurora remains unchanged.

## Current method status

This table separates evidence from the historical full rollout and the post-fix controlled proxy.

| method | MAE | RMSE | bias | spatial correlation | extremes | full distribution |
|---|---|---|---|---|---|---|
| Diffusion U-Net | improves every target/lead at 24 epochs | improves every aggregate target | aggregate improves all; 12-hour guard fails three levels | improves every target/lead at 24 epochs | P99 and tail improve for all | mixed: tcno2 variance improves, vertical underdispersion increases |
| Diffusion Transformer | improves all aggregates at 12 epochs; rejected | improves all aggregates | improves three; 850 hPa degrades | degrades every target and is rejected | P99 and tail improve for all | mixed; tcno2 is overvariable historically |
| Flow Matching U-Net | improves every target/lead at 24 epochs and is best vertically | improves every aggregate target and is best vertically | aggregate improves all; 12-hour guard fails two levels | improves every target/lead at 24 epochs | P99 and tail improve for all | mixed: tcno2 variance improves, vertical underdispersion increases |
| Flow Matching Transformer | improves all aggregates at 12 epochs; rejected | improves all aggregates | improves three; 850 hPa degrades | degrades every target and is rejected | P99 and tail improve for all | mixed; tcno2 is overvariable historically |

No post-fix full three-day rollout exists yet, so temporal improvement and complete post-fix distribution agreement remain unproven for every method.

## Best method and recommended defaults

Unified Flow Matching with the convolutional U-Net is the best current architecture and the recommended production training default for NO2_US-WEST_3day_lead. It provides the strongest controlled vertical MAE, RMSE, and upper-tail improvement, reduces aggregate absolute bias, and raises spatial correlation at every target and lead while retaining the conservative shared process, scaling, clipping, validation, and checkpoint safeguards. No post-fix controlled checkpoint is yet authorized for deployment because the lead-local bias guard still vetoes it; the safe runtime outcome remains the unmodified Aurora forecast. Diffusion U-Net is the close secondary, while the Transformer heads remain experimental.

| head | recommended status | head-specific default |
|---|---|---|
| Diffusion U-Net | supported secondary candidate | clean-sample prediction, cosine schedule, deterministic DDIM-compatible point estimator, 50 stochastic steps only for explicit ensembles; promote only if every guard passes |
| Diffusion Transformer | experimental until guard passes | patch 4, embedding 256, six blocks, adaptive conditioning, local refinement; train longer but never promote on loss alone |
| Flow Matching U-Net | primary training candidate | existing-Aurora interpolation, unified packed U-Net, deterministic shared-process point output; seeded stochastic integration only for uncertainty; promote only if every guard passes |
| Flow Matching Transformer | experimental secondary | same flow contract plus patch 4 Transformer; requires non-degrading validation correlation before use |

For every head, use the committed YAML rather than copying individual numbers. Old checkpoints lack the current deterministic-head, scaling, conditioning, and promotion contract and must not be resumed.

## Files changed

Production and configuration changes:

- finetune/aurora_finetune_utils.py
- finetune/aurora_finetune_distributed.py
- finetune/flow_refine.py
- finetune/model_factory.py
- finetune/refinement/checkpoint.py
- finetune/refinement/config.py
- finetune/refinement/legacy_flow.py
- finetune/refinement/losses.py
- finetune/refinement/packed.py
- finetune/refinement/two_phase.py
- the four finetune/aurora_NO2_finetune_US-WEST_3day_lead configuration YAMLs

Evaluation and controlled diagnostics:

- finetune/diagnose_refinement.py
- finetune/run_refinement_diagnostics.py
- finetune/refinement/benchmark.py
- finetune/refinement/evaluation.py

Regression coverage:

- tests/test_diagnose_refinement.py
- tests/test_distributed_refinement_contract.py
- tests/test_finetune_config_compatibility.py
- tests/test_flow_refine_contract.py
- tests/test_no2_diffusion_transformer_workflow.py
- tests/test_refinement_benchmark.py
- tests/test_refinement_checkpoint.py
- tests/test_refinement_conditioning.py
- tests/test_refinement_config.py
- tests/test_refinement_controlled_diagnostics.py
- tests/test_refinement_correction_process_regression.py
- tests/test_refinement_evaluation.py
- tests/test_refinement_models.py
- tests/test_refinement_performance.py
- tests/test_refinement_residual_scaling.py
- tests/test_refinement_spatial_math.py

Small header/shebang maintenance changes keep executable scripts and the repository header test compatible. Backup .orig artifacts were deliberately left untracked and are not part of the change.

## Automated safeguards and verification

Regression coverage includes:

- residual sign and exact Aurora plus true residual reconstruction;
- normalization and centered residual-scaler round trips;
- lead, target, channel, and pressure-level alignment;
- Transformer patch/token reshape and reconstruction;
- deterministic identity and seeded stochastic reconstruction;
- diffusion and flow formulation parity;
- legacy endpoint and autoregressive-feedback compatibility;
- checkpoint architecture, scaler, validation, and deterministic-head loading;
- CPU/CUDA random-state resume and chunk-independent ensemble ordering;
- YAML compatibility, including loading every shipped NO2 recipe;
- tiny known-correction learning and four-head tiny-dataset overfit;
- trace-gas spatial metrics, masks, tails, distribution metrics, and temporal aggregation.

The complete repository suite passed 743 tests. The 243 emitted warnings are existing dependency deprecations, optional-section configuration notices, and documented numerical/runtime warnings; there were no test failures.

## Remaining limitations

1. Full production retraining was not computationally feasible in this review. All historical 184-initialization results describe old checkpoints; all post-fix results are controlled proxies.
2. The test-period residual mean changes sign relative to the full training calibration. Aurora-state and lead conditioning plus train-tail selection reduce the risk, but future-season generalization needs a fresh full rollout and preferably rolling seasonal validation.
3. The Transformer heads still trade small RMSE gains for small pattern-correlation losses in the controlled run. They are intentionally ineligible for promotion until the strict guard passes.
4. Fixed four-standard-deviation clipping is statistically justified by the training correction distribution but cannot repair a spatially wrong correction. It is a last boundary, not a calibration method.
5. The standalone validation-fit correction-strength utility remains useful for analysis, but learned or fitted spatial gating is not integrated into production inference. The committed default instead abstains at checkpoint level.
6. Optional Mamba remains disabled. Before enabling it, train it on the exact deployed upstream correction distribution and feedback policy, then validate the same temporal metrics.
7. The corrected deterministic point product is the conditional correction estimate. Stochastic reverse diffusion or flow integration should be evaluated as a seeded ensemble with probabilistic scores, not substituted as one noisy member.
8. Automated kurtosis and gradient/Laplacian spatial diagnostics are present; a full isotropic power-spectrum diagnostic remains a useful follow-up for global cases.
