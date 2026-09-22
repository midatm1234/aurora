# Refinement branch synchronization

`aurora_finetune_stochastic_refinement` includes the public agent workflow from
`agent_skills_to_finetune_aurora` at
`7e534b08216d62c1ffbc77e9423b1cdef6fccf05`, together with the reviewed research
changes already present in the original worktree. Both branches started from
`88f652f04aa75b65e410b8d00cf4ea07f2edd945`. The target was fast-forwarded through
the six reference commits before committing its own changes. The reference
branch was not modified. Resolve this checkout with `git rev-parse HEAD`.

## Public workflow and branch differences

The shared CLI, 24-tool local stdio MCP server, bounded approval/jobs, pinned
assets, data/evaluation interfaces, skills, knowledge bundle and dependencies
are inherited from the reference. Follow [onboarding](agent-workflow.md) on
this branch. Agent scientific execution uses MCP and an approved effective
plan; notebook entry points remain available to human users.

All four unified heads remain selectable through that interface:
`flow_matching_conv_unet`, `flow_matching_transformer`, `diffusion_unet`, and
`diffusion_transformer`, plus `none`. The separately named
`flow_matching_unet` convenience head retains its legacy meaning.

| Area | Deliberate target-branch behavior |
| --- | --- |
| Portable reproduction recipes | The same six JSON recipes as the reference, derived from pinned source YAMLs at `88f652f`. Validation reads those Git blobs and checks their hashes; current research YAML edits cannot silently redefine a recipe. Both temporal routes remain off. |
| Research YAMLs and notebooks | Preserve the existing experiment names, train/test dates, explicit resume choices, and enabled legacy `model.mamba_temporal` settings. Several NO2 case names carry `_mamba`; the diffusion YAML resumes `last`. These are distinct experiments requiring their own data/checkpoints, not drop-in replacements for portable recipes. |
| New temporal APIs | Preserve calendar/solar/vertical conditioning, causal convolution, GRU, attention and optional Mamba context, and trajectory metrics. Production trainer wiring for calendar/solar/new temporal context is incomplete. MCP rejects unsupported context requests before execution; the lower-level API and synthetic pilot exercise them explicitly. |
| Optional defaults/checkpoints | New unified research configurations default legacy Mamba on when undeclared; portable recipes explicitly opt out and reject missing opt-outs. Historical checkpoints without temporal declarations retain legacy-safe interpretation. New disabled fields can be hydrated for old spatial checkpoints; active architecture, packing, time/context, calibration and tensor incompatibilities still fail. New temporal contract metadata is saved and checked. |
| Evaluation/inference | Preserve configuration-defined evaluation windows, explicit latest-checkpoint diagnostics, provenance-aware run selection, and corrected overall-map indexing. Notebook use of an unvalidated `last` checkpoint is labelled diagnostic; portable promotion checks remain strict. |
| Data preparation | Preserve split definitions from YAML, cycle/history handling, overwrite protection and tests. No private prepared data are required to inspect or validate configurations. |
| Local paths | The downloader/preparer accept `CAMS_DATA_DIR` and default to repository `data/cams`. The 11 reviewed research YAMLs use a relative static-asset path; operators can retain their existing files and supply a local `paths.static_data_path` override. Portable MCP roots/assets remain independently configurable. |
| Experiment tools | Retain the documented synthetic pilot and ablation generator. Generated variants have distinct output/checkpoint names and reuse the declared data case; legacy Mamba is explicitly off when comparing the new temporal backends. |
| Scientific environment/CI | CPU requirements also pin Matplotlib 3.10.9 for maintained diagnostic figures. CI runs both the portable workflow and the broader refinement/notebook contract regressions, with model downloads disabled. Lightweight metadata/MCP checks still require neither Matplotlib nor PyTorch. |

Attention layout changes preserve the mathematical operation while permitting
efficient kernels. Additional tests cover masked tendency-loss gradients,
checkpoint metadata, and solar feature device handling. Cross-hardware bitwise
identity and scientific improvements are not inferred from those unit tests.

The historical [spatiotemporal review](../finetune/O3_SPATIOTEMPORAL_REVIEW.md)
retains the original notes and reported measurements with an explicit evidence
label. Its GPU/pilot tables were not independently repeated in this cleanup.
The [original workflow implementation report](workflow-implementation-report.md)
continues to identify the reference branch's historical validation precisely.

## Preservation and exclusions

Before changing any original file, a restricted local backup outside the
repository captured all 62 modified/untracked files, the index, binary patches,
and all original Git refs in a bundle. The index was initially empty. Ignored
multi-terabyte datasets, checkpoints, environments and other private artifacts
were inventoried and preserved in place; no reset, clean, or destructive data
operation was used.

Scoped repository ignore rules cover downloader logs, generated ablation YAMLs,
and root-level synthetic pilot JSON results. Two untracked exploratory GPU
diagnostics remain locally excluded through `.git/info/exclude`; they are not
public entry points. Existing maintained smoke tests and test infrastructure
remain tracked. No tracked production artifact needed removal. Workflow source
fingerprints include tracked and eligible new source files while excluding
ignored outputs and local helpers.

Seven notebooks retain all 85 scientific output objects and the original plot
image. Cleanup redacts machine prefixes in 35 outputs, removes 70 timing
metadata entries, and compacts one redundant filename inventory. Saved-output
notes distinguish historical flow/Mamba results from the notebook's current
diffusion selection. Numerical values and deliberate experiment dates remain.
No notebook training/inference was rerun to refresh those saved results.

Outgoing reference commits and the final staged changes are reviewed for
credentials, private artifacts, unexpected binary/large files and excluded
helper references. Local backup paths and credentials are not public recipe
dependencies. GitHub SSH and the existing author/committer identity were both
verified as `midatm1234`; no identity or co-author trailer was invented.

## Validation

The reviewed staged files were applied to a fresh detached checkout with no
private data or excluded helpers. On 2026-09-22, that public snapshot passed:

| Check | Result |
| --- | --- |
| Complete workflow suite, including real stdio MCP dispatch and all four fixture heads | **117 passed, 1 skipped**; real CAMS/GPU opt-in not executed |
| Broader refinement, checkpoint, data preparation, evaluation, notebook contracts and maintained smoke tests | **787 passed, 4 skipped**; CUDA/development-header cases unavailable in the CPU environment |
| Separate lightweight environment, without NumPy/PyTorch | **54 passed, 6 skipped**; optional numerical checks skipped |
| Research configuration validation | **11 YAMLs passed**, without opening private train/test files |
| Pinned portable recipes | **30 selections passed**: six recipes × four unified heads plus baseline |
| Knowledge/entry points | **19 OKF concepts passed**; handoff and client/skill links resolve in the public snapshot |
| Notebook structure | **7 notebooks validated**, all **91 code cells parsed**; saved scientific outputs retained |
| Maintained synthetic pilot | Two CPU diffusion-UNet variants completed one update each; both had negative MAE/RMSE skill at this tiny budget, with no CAMS accuracy claim |
| Final failing tests in these suites | **0** |

The expanded first regression run exposed two missing-plotting-dependency
failures and three stale notebook/case-name assertions. Matplotlib is now
pinned; tests check the deliberate diffusion/temporal settings and use AST
contracts rather than fragile notebook line layout. Configuration-only tests
validate the YAML and synthetic datasets without a private-file precondition.
All scientific correction and compatibility assertions remain enforced.

Validation used Python 3.12.3, PyTorch 2.10.0+cpu and the pinned CPU requirements.
Tests use temporary output directories outside the checkout; model downloads
were disabled for the public-snapshot suites. A nonfatal NumPy/NetCDF extension
warning and existing configuration advisory warnings were observed. Data round
trips and numerical checks passed; this is not a cross-platform guarantee.

The exact maintained test selections are in
[the CI workflow](../.github/workflows/agent-workflow.yml). To run the core
workflow manually after bootstrap:

```bash
.venv-science/bin/python -m pytest --confcutdir=tests/workflow tests/workflow -q
```

No full training, large data/model downloads, or real CAMS/GPU execution was
launched for this cleanup. A pre-existing training process and its artifacts
were left alone. Historical notebook/review results are retained evidence from
the original worktree, not new scientific results from these tests.

The operator still supplies compute, ADS permissions/credentials, datasets and
any unpublished trained checkpoints. Use the portable CPU recipe first, then
review and approve a bounded real-data plan. The new experimental temporal
features require additional trainer integration and independent validation
before they can be treated as an operational reproduction recipe.
