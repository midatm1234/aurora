# Inventory of the pinned scientific workflow

Source: fork branch `aurora_finetune_stochastic_refinement`, commit
`88f652f04aa75b65e410b8d00cf4ea07f2edd945`. This inventory describes code, not
independently reproduced historical scientific results. Existing notebooks and
scientific modules are retained. Implementation commits are recorded by the
runtime source manifest rather than a self-referential commit hash in this file.

## Actual stages and interfaces

| Stage | Existing implementation and configuration | Inputs → outputs | New shared executor and checks |
| --- | --- | --- | --- |
| CAMS acquisition | `examples/cams_download_2025_range.py`, `examples/cams_download_between_dates.ipynb`; historical downloader uses half-month chunks and `/data/cams` | ADS operational meteorology/chemistry, 13 pressure levels → raw NetCDF | `aurora_workflow.data.execute_data('acquire', ...)`: approved bounded requests, cycle provenance, retries/cache/integrity; old hard-coded dates/paths are not defaults |
| Preparation | `finetune/prepare_train_test_from_netcdf.py`: `_canonicalize_spatial_coordinates`, `_split_train_val_test`; `ft.resolve_variable_specs`, `build_aurora_batch` | Operational cycles, two history times, static alignment → `prepared.json`, separate history/reference NetCDF | `execute_data('prepare', ...)`: explicit split windows and purging, named coordinate/variable/unit validation |
| Backbone assets | `aurora/model/aurora.py`: `AuroraAirPollution.load_checkpoint_local` | Official `aurora-0.4-air-pollution.ckpt`, static pickle converted to verified NPZ → frozen model | `aurora_workflow.assets`: pinned model revision/file hashes; source package-path verification; weights-only loading and strict required-state loading |
| Baseline rollout | `examples/cams_prediction_local.ipynb`, `finetune/aurora_inference_rollout.ipynb`, `ft.run_rollout`, `ft._advance_batch_with_prediction` | Past CAMS history → deterministic physical forecasts | `execute_science('rollout', ...)`: baseline NPZ and packing/provenance; autonomous mode accepts no target input inside its state-advance loop; historical forecast-forced mode calls original `ft.run_rollout` explicitly |
| Phase 2 training | `finetune/aurora_finetune_distributed.py`, `ft.compute_supervised_loss`, `ft.create_optimizer/create_scheduler`, `refinement.two_phase.AuroraTwoPhaseRefiner.training_step` | Immutable baseline + CAMS training reference, exact train-only residual calibration → head parameters and optimizer/scheduler/RNG state | `execute_science('train', ...)`: local cached head training uses the existing objective and per-case six-lead packing, configured accumulation, original physical selection guards; checkpoints are NPZ+JSON; no backbone gradient graph |
| Resume/warm-start | `ft.restore_checkpoint_rng_state`, original distributed resume handling | Exact previous run state or identified warm-start checkpoint → explicit new continuation | Resume requires identical training config and data fingerprint; warm-start restores weights/scalers and starts a new optimizer experiment. No replay-to-training fallback |
| Refined inference | `refinement.two_phase.refine`, `flow_matching`, `diffusion`, `legacy_flow`, `target_space` | Baseline + compatible head → physical point/ensemble products | `execute_science('refine', ...)`: strict compatibility checked before loading head parameters; no reference target file is opened; baseline survives unchanged |
| Evaluation | `evaluate_finetuned_cams_rollouts.ipynb`, `refinement/evaluation.py`, `diagnose_refinement.py` | Matched baseline/refinement/reference cases → metrics and diagnostics | `aurora_workflow.evaluation.execute_evaluation`: physical scores, masks, area weighting, separate point/ensemble interpretation, reference qualification, reports/plots |
| Tracing and diagnostics | `trace_refinement_batch.py`, `refinement_parity_check.py`, `run_refinement_diagnostics.py`, `run_mamba_ablation_study.py` | Saved fields/checkpoint/config → residual/normalization/temporal evidence | Existing human entry points remain available. Agent execution uses shared MCP workflow stages; authored review prose is not an execution receipt |

Here `ft` means `finetune.aurora_finetune_utils`. The new executor signature is
`execute_science(stage: str, plan: dict, run_dir: Path) -> dict`; the CLI and MCP
server dispatch through the same durable worker. Scientific functions remain
inside the fork. No shell arguments are invented for notebook cells.

## Unified heads and legacy names

| Human choice | Actual configuration enum | Numerical implementation and inference contract |
| --- | --- | --- |
| Flow matching with convolutional UNet | `flow_matching_conv_unet` | Unified `FlowMatchingResidualRefiner` with convolutional backbone; source YAML uses clean residual/data parameterization; exact trained point product is distinct from stochastic sampling |
| Flow matching with Transformer | `flow_matching_transformer` | Unified flow implementation + spatial Transformer. Active NO2 YAML: `interpolation_path: existing_aurora`, `deterministic_head: shared_process`, clean residual x1/data prediction; deployed source-mean query is **x=0, t=0**, regardless of the stored 50-step stochastic setting |
| Diffusion with UNet | `diffusion_unet` | `DiffusionResidualRefiner` + convolutional backbone. Active YAML's sample/clean-correction parameterization and schedule are preserved; dedicated deterministic product follows saved `deterministic_head` settings; DDIM/DDPM members are explicit products |
| Diffusion with Transformer | `diffusion_transformer` | Same diffusion contract with Transformer geometry; architecture, diffusion schedule, prediction parameterization, normalization and inference estimator are checkpoint metadata |
| No refinement | `none` | Physical baseline copied bit-for-bit, including original mask; no normalize/decode/clipping round trip |
| Legacy flow compatibility | `flow_matching_unet` (permanent alias `flow_matching`) | `LegacyFlowMatchingUNetRefiner` delegates to the original `AuroraFlowRefine`/`ResidualFlowUNet`, per variable and atmospheric level. It does **not** mean the unified convolutional objective above. Legacy auxiliary weights live under `training.flow_aux_loss`, and incompatible unified auxiliary keys are rejected |

`AuroraConvRefine` is the older deterministic convenience head, not one of the
four stochastic alternatives. The legacy flow adapter is an additional
compatibility selection. Current NO2 and global O3 configs named “flow matching”
often actually select `flow_matching_conv_unet`; filenames alone are not an
architecture contract. In particular,
`examples/stochastic_refinement/aurora_O3_global_flow_matching_unet.yaml` is a
compatibility filename whose content selects **conv_unet**.

All corrections use `cams_minus_aurora_add`: target minus raw Aurora in the
same normalized space, added once to Aurora, inverse-transformed once, with
saved constraints/masks. Flow interpolation time and physical forecast lead
are separate quantities. Lead embeddings are not temporal sequence modeling.
The `existing_aurora` straight-line x1/data sampling updates are not described
as a rectified-flow ODE solver. Explicit `rectified_flow` velocity paths remain
an alternative configuration and cannot load a data-prediction checkpoint.

## Recipes, inputs, and dates

`recipes/*.json` contains complete machine-readable, YAML-derived configuration,
per-head overrides, source file SHA256, source commit, input contract, limits,
historical dates, and explicit changes from source. The four required entries
are `cpu-smoke-v1`, `real-gpu-v1`, `no2-us-west-v1`, and `o3-global-v1`.
Separate `no2-historical-forecast-forced-v1` and
`o3-historical-forecast-forced-v1` preserve the old exogenous-refresh code path.
Plans supply recipient-controlled input/cache/output roots and distinct run IDs.

The current NO2 Transformer configuration is
`finetune/aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml`:
31–52°N, 128–100°W; two history times; 12-hour cadence; six leads through 72
hours; NO2 targets at **1000, 925, 850 hPa** plus **tcNO2**. The backbone levels
remain `[50,100,150,200,250,300,400,500,600,700,850,925,1000]` hPa. The historical
regional grid is approximately 53×70 before crop, 51×69 after Aurora patch-3
alignment. Exact coordinates are recorded, not inferred from grid size.

The NO2 refinement predictor mapping contains 5 surface fields and 6 atmospheric
fields. This is distinct from complete pretrained backbone inputs: 12 surface
fields (`2t,10u,10v,msl,pm1,pm2p5,pm10,tcco,tc_no,tcno2,gtco3,tcso2`), 10 atmospheric
fields (`z,u,v,t,q,co,no,no2,go3,so2`), and all 13 levels. Both tracks use static
`lsm,z,slt` plus ammonia/CO/NOx/SO2 emissions and their four log fields.
The new autonomous variant runs the full global backbone and crops only its cached
outputs to the refinement domain; the historical mode
preserves the original configured predictor subset, explicitly filters only
unconfigured pretrained keys, and strictly requires every remaining weight.

Global O3 uses the full predictor set, O3 `go3` at **1000,500,100,50 hPa** and
total-column `gtco3`, six 12-hour leads, and periodic longitude without duplicate
cyclic endpoints. Its current source config defaults to unified convolutional
flow matching and 50 epochs; current NO2 source configs use 100 epochs.

The preparation script's recorded defaults are training **2023-07-01 00 UTC
through 2024-06-30 12 UTC**, test **2024-07-01 00 UTC through 2024-09-30 12 UTC**.
Current YAMLs use `validation_source: train_tail`, `validation_fraction: 0.1`.
There is no explicit historical validation date list to reproduce blindly.
New preparation derives and records disjoint cycle/valid-time windows with
72-hour purging. The old global YAML also contains a `val_data_path: test.nc`
convenience path; it is superseded by train-tail selection and is not evidence
that a reused training interval was independently held out.

## Autonomous versus historical exogenous rollout

The pinned active YAMLs have `keep_exogenous_predictors: refresh_from_dataset`
and an empty target-feedback list interpreted as **targets only**. Existing
code protects NO2/tcNO2 (or O3/column) targets against overwrite, but refreshes
meteorology and other non-target predictors from later CAMS data. Calling that
historical trajectory autonomous would be inaccurate.

The canonical new autonomous recipes explicitly version a scientific change:
only the two past history frames enter the backbone; every available raw Aurora
output advances the state; no future dataset enters `autonomous_rollout`.
Refined fields are never fed back. Training from this trajectory is comparable
retraining, not an exact replay of the historical forecast-forced trajectory.

The two named historical recipes call the original `ft.run_rollout` with target
feedback and `refresh_from_dataset`. Their future exogenous fields come from
the **same initialization-issued CAMS forecast**; provenance records their
variable names, availability time and valid leads. The tool rejects analysis
or reanalysis substitution in this mode. Historical inputs that actually used
later analyses would define a retrospective hindcast, and are not silently
relabelled as operationally available forecast context. Exact historical
numerical reproduction requires verifying which product generated the old
data and preserving it; this linkage is not proven by old filenames.

## Checkpoints and scientific evidence available during implementation

The official model class pins Hugging Face revision
`1764d5630a53d3d7a7d169ca335236fc343e4bfc`; acquisition records whichever explicitly
verified revision it retrieves. Both filename and model family are checked.
Static conversion accepts only the official verified artifact or an explicitly
verified equivalent and writes numerical arrays with coordinates. The agent
workflow never calls the old unguarded static `pickle.load` paths.

New training checkpoints contain JSON plus numerical NPZ arrays for all head
parameters, normalization buffers, optimizer integer-key states, scheduler,
RNG, train data fingerprint, config, lead/grid/static/temporal contract and
validation decision. Replay checks metadata first and loads head state with
`strict=True`. A checkpoint is never silently reinitialized or replaced by
fresh training. Warm-start and resume are different modes; weights with active
residual scaling retain their trained transform.

Existing combined `.ckpt` files can be consumed using `torch.load` with
`weights_only=True, mmap=True`, followed by the same
`validate_unified_checkpoint_contract` and
`validate_loaded_residual_scaler_contract` used by notebook inference, and
strict extraction of head/temporal keys. Unsupported pickle globals or missing
metadata require owner conversion; there is no unsafe fallback. Historic
rollout context must match the explicitly selected historical recipe.

Read-only inspection found local NO2 Transformer `best.ckpt` and `last.ckpt`
artifacts outside Git. The best checkpoint is from epoch 29 of a different
training run than the unvalidated epoch-99 last checkpoint. A CPU-only,
weights-only **strict load of all 92 head tensors and exact residual-calibration
validation passed** for the best checkpoint against the historical recipe.
This verifies compatibility, not historical numerical reproduction or public
availability. No private path or public download URL is supplied as a portable
default. Recipients need their own authorized artifact and recorded SHA256.

`REFINEMENT_REVIEW.md` documents an old invalid endpoint, inference sampler
override, missing validation, residual scaling, train/inference feedback, and
scheduler issues. `NO2_REFINEMENT_REVIEW.md` reports mixed historical results:
column improvements coexist with profile bias, texture and spatial correlation
degradation. Those are historical repository claims. `MAMBA_ABLATION.md`
reports negative controlled temporal results and `phase1_rollout_linkage:
asserted_not_proven`; its path/size/mtime corpus digest is not a content hash.
Mamba stays disabled in canonical recipes. Existing temporal classes remain
available, but the portable cached training adapter currently supports the
spatial-only reproduction track; temporal experiments require the existing
sequence-aware runner and a separately approved integration.

The original `AuroraAirPollution(simulate_indexing_bug=True)` default is part
of the published checkpoint contract. This adapter does not silently “fix” it.
Older README text describing 25–50°N, 235–310°E or all 13 NO2 refined levels is
superseded by the active source YAML; archived slide target lists do not replace
it. Aurora 1.5 is a separate weather track, not this model.

## Validation and practical bounds

`tests/workflow/test_science.py` exercises real CPU refinement classes with
explicitly synthetic fields, strict NPZ save/reload, resume, deterministic
repeatability, full frozen toy-backbone state-advance parity, `none` identity,
rejected feedback/teacher forcing/model-family mismatches, and inference that
works after the reference file is removed. Existing tests inspected include
`test_flow_refine_contract.py`, `test_refinement_checkpoint.py`,
`test_distributed_refinement_contract.py`, `test_refinement_target_space.py`,
`test_refinement_models.py`, and `test_prepare_train_test_from_netcdf.py`.
These fixtures are not pretrained Aurora or CAMS scientific validation.

The opt-in GPU test uses an **already approved** durable plan selected by
`AURORA_WORKFLOW_CONFIG` and `AURORA_GPU_PLAN_ID`, with `AURORA_RUN_REAL_GPU=1`.
It covers acquisition, preparation, actual backbone rollout, bounded head
training or compatible checkpoint loading, refinement and evaluation through
the shared workflow. It is skipped without those prerequisites. Download
counts, storage, cases, elapsed time and optimizer updates remain bounded by
the approved plan. Historical ADS dates may require an accessible archive;
current operational retention does not imply old files are still downloadable.

The local adapter caches packed arrays in memory; large global/year-long work
must be divided into approved case windows that fit the recipient's RAM and
storage. It is not a new distributed trainer. Cached autonomous retraining is
a versioned variant preserving head objectives; it is not claimed bitwise equal
to distributed historical training. Production inference retains validation
guards: a completed but unpromoted checkpoint can remain unavailable for
production inference, which is an honest negative outcome rather than a failed
promise to improve. CPU synthetic recipes explicitly permit diagnostic
inference for wiring validation. Hardware/software changes may alter floating
point reductions and stochastic trajectories even with identical seeds.

Recent source history inspected: `88f652f` (temporal Mamba workflows), `df61d4a`
(NO2 correction contracts), `2dc57bc` (stochastic workflow expansion), `691d083`
(archived O3 configs), `a0bd0cf` (stochastic tooling), `878b23c` (config-driven
CAMS evaluation), and `20841e9` (scheduler restore). Presentation evidence and
primary scientific sources are separately attributed in the knowledge bundle.
