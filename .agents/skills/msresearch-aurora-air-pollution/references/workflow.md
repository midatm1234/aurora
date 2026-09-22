# MCP execution routes

Use discovered typed schemas for arguments; do not invent CLI flags or tool names. `plan_execution` returns the immutable plan/run identifiers and resource bounds. Mutating stage tools require that approved plan; inspect returned blockers before execution. Any missing scientific executor is a blocker, not permission to write an alternate model.

| Phase | MCP tool | Required evidence / output |
| --- | --- | --- |
| Inspect | `inspect_environment`, `inspect_assets`, `list_runs` | Imported fork path, capabilities, roots, manifest/checkpoint availability |
| Validate/plan | `validate_recipe`, `plan_execution` | Effective head/configuration hash, dependencies, bounded resources, unknown costs |
| Complete chain | `execute_workflow` | Durable job for all approved stages with verified reuse |
| Source | `prepare_sources` | Pinned fork/upstream identity; fork remains executable |
| Assets | `retrieve_assets` | Trusted air-pollution model/static and supplied refinement checksums |
| CAMS | `acquire_cams` | Bounded request plan, two-time initialization history, cycle/lead provenance |
| Prepare | `prepare_dataset` | Valid coordinates/units/masks, complete backbone inputs, purged split manifest |
| Baseline | `run_rollout` | Frozen autonomous deterministic baseline, audited inputs, no refined feedback |
| Fit/resume | `train_refinement` | Mode, full checkpoint contract, train-only statistics, actual optimizer progress |
| Inference | `run_refinement` | Compatible head, point/ensemble separation, baseline preserved |
| Evaluate | `evaluate_forecasts` | Matched cases, physical metrics, coverage, excluded cases and diagnostics |
| Report | `generate_report` | Metrics JSON, plot, Markdown summary and hashes linked to receipts |
| Manage | `job_status`, `job_logs`, `cancel_job`, `recover_job` | Durable job ID, bounded logs, explicit state and recovery evidence |
| Knowledge | `knowledge_search`, `knowledge_read`, `knowledge_validate`, `knowledge_refresh`, `knowledge_register` | Attributed concepts and controlled verified runtime summaries |

Request a recipe using its exact identifier: `cpu-smoke-v1`, `real-gpu-v1`, `no2-us-west-v1`, or `o3-global-v1`. A synthetic smoke run validates interfaces, not real Aurora/CAMS science. The bounded real test still needs authorized ADS access, GPU and official assets. Choose a unique case ID for each head/product/mode.

For a historical combined checkpoint, inspect the recorded forcing contract first. `no2-historical-forecast-forced-v1` and `o3-historical-forecast-forced-v1` preserve the separately named historical exogenous-input route. These historical-named recipes reuse the source forcing dynamics with a specified-cycle CAMS forecast instead of future analysis refresh. This is a versioned approximation: checkpoint-head replay can be compatible while baseline inputs and results differ from the old analysis-forced experiment. Do not claim exact historical numerical reproduction. The canonical NO2/O3 recipes are autonomous and also cannot be substituted without identifying that changed contract. A historical route still needs forecast-cycle reference inputs available according to its declared availability contract; future analyses are not silently authorized as operational inputs. “Transformer” alone is ambiguous between flow and diffusion: checkpoint/configuration metadata identify the head.

Replay never trains implicitly. Resume requires optimizer/scheduler/RNG state when supported; warm-start is a named new experiment. Evaluate-only reuses outputs. The human CLI and MCP share the backend, but agent scientific execution uses MCP; local setup and human plan approval are separate.

Inspect knowledge concepts `data/cams`, `data/leakage`, `data/physical-units`, `models/refinement`, `workflow/recipes`, `workflow/artifacts`, and `evaluation/metrics` only as needed. The historical notebook's forecast-forced/exogenous path is distinct from canonical autonomous inference and must be labeled as such.
