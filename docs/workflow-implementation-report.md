# Implementation and validation handoff

The branch `agent_skills_to_finetune_aurora` adds a working local MCP/CLI backend around the existing fork. Source: `88f652f04aa75b65e410b8d00cf4ea07f2edd945` on `aurora_finetune_stochastic_refinement`. [Source and asset pins](../provenance/source-lock.json), [reference access records](../provenance/references.json), and the [post-commit implementation receipt](../provenance/implementation.json) identify the exact lineage. The feature implementation commit is `032a6326a2d05a200ca1963b9fc00a35892d5719`. The original working tree, uncommitted experiments, notebooks and trained artifacts were preserved in place; development used a separate worktree.

Publication succeeded using the authenticated `midatm1234` account. Share [the published branch](https://github.com/midatm1234/aurora/tree/agent_skills_to_finetune_aurora). Published implementation and packaging commit `68c32207a964decc739c44c533ccfc17f42b942d` passed both [GitHub CI jobs](https://github.com/midatm1234/aurora/actions/runs/35778286127) and the complete workflow suite from a fresh HTTPS clone. The [publication receipt](../provenance/publication-validation.json) records that exact tested commit; subsequent documentation commits preserve the result as historical evidence. Resolve the current documentation/branch head with `git rev-parse HEAD`. Only the requested new branch was pushed, without force.

## Implementation

`aurora_workflow/` separates planning/settings, filesystem/provenance safeguards, SQLite jobs and detached workers, assets, CAMS preparation, scientific adapters, physical evaluation, knowledge, and the MCP/CLI entry points. It reuses the fork's Aurora model, state advance, packing, residual heads, losses, optimizer/scheduler functions and compatibility validators. There is no weather starter project or hosted-service dependency.

The 24 local stdio tools below dispatch to actual code. Inspection stays lightweight. Execution tools return durable IDs and run only after a local operator approves the hashed effective plan. Workers verify code, environment and input hashes; retain stage receipts; enforce time, process, host/GPU-memory, request, download, output/storage and retry bounds; and support cancellation and interrupted-job recovery. Successful stage outputs are hashed before reuse. Approval is an orchestration boundary, not a sandbox against arbitrary programs with the same Unix account or unrestricted agent shell privileges. Tested worker support is Linux/local CPU or CUDA; cluster scheduling and remote execution are not implemented.

| MCP tool | Backend mapping |
| --- | --- |
| `inspect_environment` | `backend.doctor` |
| `inspect_assets` | `assets.inspect_assets` |
| `list_runs` | `jobs.JobStore.list_jobs` |
| `validate_recipe` | `planning.validate_recipe` |
| `plan_execution` | `planning.make_plan` → persisted immutable plan |
| `execute_workflow` | `JobStore.submit` → `worker.supervise` → approved stages |
| `prepare_sources` | job → `assets.verify_source_references` |
| `retrieve_assets` | job → `assets.retrieve_assets`, verified numeric static conversion |
| `acquire_cams` | job → `data.execute_data('acquire')` → current ADS client |
| `prepare_dataset` | job → `data.execute_data('prepare')` → cycle-preserving NetCDF |
| `run_rollout` | job → `science.execute_science('rollout')` → actual AuroraAirPollution |
| `train_refinement` | job → `science.execute_science('train')` → existing head/objective |
| `run_refinement` | job → `science.execute_science('refine')` → checkpoint point/ensemble product |
| `evaluate_forecasts` | job → `evaluation.execute_evaluation` |
| `generate_report` | job → `backend.execute_stage('report')` |
| `job_status`, `job_logs`, `cancel_job`, `recover_job` | corresponding `JobStore` persistence/process methods |
| `knowledge_search`, `knowledge_read` | `knowledge.search/read`, bundle or runtime scope |
| `knowledge_validate`, `knowledge_refresh` | `knowledge.validate_bundle/refresh`; refresh previews changes |
| `knowledge_register` | `knowledge.register_run`, verifies successful scientific receipts before controlled runtime writes |

The workflow inventory maps existing notebooks/scripts, actual APIs, checkpoint requirements and known differences in greater detail. The [data contract](data-contract.md) and [evaluation contract](evaluation.md) define cycle/lead/valid-time handling, purged splits, matching/masks/units, weighting and probabilistic diagnostics. Evaluation emits `metrics.json`, `metrics.svg`, `evaluation.md`, an attested computation receipt, and a run manifest. Runtime knowledge records link receipts; prose is not an execution receipt.

## Setup and exact selections

Follow [the onboarding guide](agent-workflow.md) for the clone/setup/credentials/registration/approval sequence, including separate examples for **Codex local CLI/IDE**, **Claude Code local CLI**, and **GitHub Copilot in VS Code**. The canonical skill is `.agents/skills/msresearch-aurora-air-pollution/SKILL.md`; AGENTS.md, CLAUDE.md and Copilot instructions route to it. Python 3.12–3.14 is supported by the bootstrap; lightweight inspection does not require model dependencies.

```bash
python3.12 scripts/bootstrap_workflow.py --venv .venv-science --science
.venv-science/bin/python -m aurora_workflow --config /absolute/aurora-local.json configure --workspace /absolute/aurora-workspace
.venv-science/bin/python -m aurora_workflow --config /absolute/aurora-local.json doctor --check-torch
```

The CPU bootstrap uses pinned PyTorch CPU wheels. Choose and verify a compatible CUDA environment separately for real GPU work; the source checkout must be installed editable with `--no-deps` so upstream/PyPI code does not replace this fork. Use its absolute Python in the client's registration example. Credentials stay in approved local ADS configuration/environment.

Exact MCP planning requests for the four required choices (each result has its own run identifier):

```json
{"request":{"recipe":"no2-us-west-v1","head":"flow_matching_conv_unet","mode":"reproduce","case_id":"no2-flow-unet"}}
{"request":{"recipe":"no2-us-west-v1","head":"flow_matching_transformer","mode":"reproduce","case_id":"no2-flow-transformer"}}
{"request":{"recipe":"no2-us-west-v1","head":"diffusion_unet","mode":"reproduce","case_id":"no2-diffusion-unet"}}
{"request":{"recipe":"no2-us-west-v1","head":"diffusion_transformer","mode":"reproduce","case_id":"no2-diffusion-transformer"}}
```

These are arguments to `plan_execution`, not four automatic training submissions. Select `cpu-smoke-v1` for the small synthetic route, `real-gpu-v1` for the bounded sparse real-data route, or `o3-global-v1` for global O3. Set `head: "none"` for unchanged baseline output. The additional `flow_matching_unet` enum preserves the historical convenience adapter; it is not the unified convolutional UNet.

Review the returned plan and effective configuration, then the human approves once:

```bash
.venv-science/bin/python -m aurora_workflow --config /absolute/aurora-local.json approve PLAN_ID
```

The agent calls `execute_workflow({"plan_id":"PLAN_ID"})`, then `job_status`, `job_logs`, and when justified `recover_job` with its returned job ID. Changes to code, environment, assets, configuration or bounds require a new reviewed plan. For replay supply a compatible `inputs.refinement`, `inputs.baseline`, `inputs.packing`, and reference archive (or the upstream data/assets to generate them); select `mode:"replay"`. `resume` requires optimizer/scheduler/RNG-compatible state; `warm-start` identifies a new experiment. Replay never triggers fresh training.

## Scientific contracts and available assets

Canonical NO2 keeps U.S.-West, two history times, 12-hour cadence and six leads through 72 hours. The full backbone keeps all 13 pressure levels and all pretrained input/static fields. NO2 refinement targets 1000/925/850 hPa plus tcNO2. Mamba is off. The frozen raw baseline is immutable; refined fields are never fed back. Flow-matching Transformer point inference preserves the clean-residual/data parameterization and its checkpoint-specific zero-state/zero-time query. Stochastic sampling is explicit and separately evaluated; no rectified-flow ODE label is applied to the x1 updates.

Historical YAMLs refreshed some future predictors from the dataset. The autonomous recipes are explicitly versioned changes. The separately named `no2-historical-forecast-forced-v1` and `o3-historical-forecast-forced-v1` reuse that forcing structure with specified-cycle operational forecasts, excluding unavailable future analyses. They approximate the old forcing contract; compatible head replay can still have a changed baseline. Neither route establishes exact historical numerical reproduction. Existing temporal/Mamba code and notebook entry points remain intact; portable temporal execution is explicitly unsupported rather than silently reduced to spatial training.

Official public checkpoint and static files were pinned to Hugging Face revision `a96afd7ee6d65e3bd2d476f3be798a25a56f2296`. An existing cached 5,096,147,215-byte air-pollution checkpoint was hashed and matched the pinned official checksum. The exact 17,860,299-byte public static file was retrieved, hash-verified, and converted to numeric NPZ with the documented global coordinates. The entire model repository was not downloaded.

A locally available historical NO2 best checkpoint passed an actual safe `weights_only`/strict CPU head load, including 92 head tensors and calibrated residual metadata. Its private bytes and production datasets were not copied into Git; no public trained-refinement URL was established. Recipients must supply authorized compatible artifacts. An absent checkpoint is `requires_artifact`; metadata/checksum verification is not evidence of forecast skill.

No requested presentation was present in the attachment directory or repository. No figures, slide settings or slide sample counts were invented. The attributed 19-concept OKF 0.2 bundle separates pinned-code facts, published background, unavailable evidence, hypotheses and executed fixtures. It has offline structural/profile validation, not blanket verification of external-source scientific truth.

## Executed validation

The clean-machine CPU setup was actually installed from `scripts/bootstrap_workflow.py`, using Python 3.12.3, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu and the pinned scientific requirements, including Dask. Direct dependency pins and bootstrap live in `requirements/` and `scripts/`; every scientific run records the resolved environment and hardware. Exact floating-point equality across different hardware, thread counts, library builds or kernels is not promised. Wrapped/direct fixture comparisons use relative tolerance 1e-6 for refined floating-point fields; baseline/none identity is exact.

| Validation | Actual result |
| --- | --- |
| New complete workflow suite | **PASSED 103**, **SKIPPED 1** (104 collected) |
| Selected existing refinement/checkpoint/normalization/model/flow/distributed/config/IO regression tests | **PASSED 250** |
| Separate clean lightweight environment (no NumPy/PyTorch) | **PASSED 47**, **SKIPPED 6** optional numerical checks; real MCP and knowledge available |
| Published-branch fresh clone, complete workflow suite | **PASSED 103**, **SKIPPED 1**; pinned source ancestry and all knowledge files present |
| GitHub hosted CI at `68c3220` | **PASSED** both jobs: metadata **47 passed / 6 skipped**; CPU science **103 passed / 1 skipped** |
| Actual SDK stdio initialize/discovery/schemas/structured results/dispatch | **PASSED**, SDK 1.30.0, negotiated protocol 2025-11-25, 24 tools |
| Four unified heads plus none through MCP → durable worker → actual fixture train/refine/evaluate/report → registered run knowledge | **PASSED**, synthetic backbone/CAMS fixtures; actual refinement implementations |
| Direct science versus MCP numerical parity, strict checkpoint reload, exact interrupted-resume comparison, no feedback/future-target input | **PASSED** within documented scope/tolerance |
| ADS acquisition retry tests | **PASSED**, explicitly mocked network failure; real API not exercised |
| Cycle-preserving NetCDF preparation, Dask path, known-value metrics and ensemble diagnostics | **PASSED** with synthetic fixtures |
| OKF bundle/profile, skills/adapters, controlled attestation/registration | **PASSED**, 19 concepts; no human review fabricated |
| Public asset checksums/static conversion | **PASSED**, actual files |
| Existing historical best checkpoint head compatibility | **PASSED**, actual local artifact on CPU; no new forecast |
| Codex registration arguments | **PASSED**, read-only CLI 0.154.0-alpha.6.2 registration parsing; no paid agent session launched |
| Claude Code / Copilot app sessions | **BLOCKED / NOT EXECUTED**, applications/client sessions unavailable; templates and source documentation checked |
| Real CAMS → actual GPU Aurora → trained refinement → evaluation | **SKIPPED**, opt-in bounded execution not approved; no real forecast generated |
| Presentation extraction | **BLOCKED**, requested presentation unavailable |
| Final failing tests in the executed suites | **0** |

The broadened regression run initially found five existing config-only tests requiring private train/test paths. All five were reproduced on a clean detached checkout of the source commit. The test now validates the identical YAML through `validate_config` without calling the convenience loader that checks data files and creates output directories. Scientific code and the historical YAMLs were unchanged. The resulting 250-test regression run passes. A nonfatal installed NumPy/NetCDF extension compatibility warning was observed; fixture values and round trips passed. Real forecast validation remains necessary.

The first hosted runs exposed two packaging failures: the inherited `data/` ignore rule omitted four knowledge documents, and shallow CI checkout prevented pinned-source ancestry verification. Both were corrected; the published clean clone and hosted rerun passed. The suite also skips clearly when its optional dependencies are absent in the inherited legacy Python test matrix. A separate clean-archive audit passed all 35 knowledge tests and verified client/skill links. These checks do not require private datasets.

Commands used:

```bash
.venv-science/bin/python -m pytest --confcutdir=tests/workflow tests/workflow -q
.venv-science/bin/python -m pytest tests/test_refinement_checkpoint.py tests/test_refinement_target_space.py tests/test_refinement_models.py tests/test_flow_refine_contract.py tests/test_distributed_refinement_contract.py tests/test_refinement_config.py tests/test_refinement_io.py -q
```

No real scientific accuracy result is claimed. Generated fixture metrics/plots/receipts establish executable plumbing and numerical checks only. CAMS is a model reference, not independent observational truth; agreement must not be described as outperforming CAMS. Real-data quality, runtime/memory needs, training convergence, and statistical skill remain unvalidated in this handoff. Large historical global preparation requires substantial disk and runtime, which appear in the plan and require operator approval.

## Remaining operator actions

Review the published implementation commit, install the chosen environment, configure dedicated roots/ADS permissions and trustworthy trained artifacts, and register the local MCP server. Start with the CPU smoke recipe. For real science, inspect the sparse integration plan and approve its actual resource limits, then execute it and retain metrics, coverage, receipts and manifests. Use a compatible historical checkpoint or explicitly choose retraining; keep validation selection separate from final test evaluation. Sharing a URL supplies neither private artifacts nor resource authorization.
