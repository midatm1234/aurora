# Reproduce Aurora air-pollution workflows on your resources

This branch wraps the existing fork implementation. Read [AGENTS.md](../AGENTS.md), the [canonical skill](../.agents/skills/msresearch-aurora-air-pollution/SKILL.md), and [knowledge/index.md](../knowledge/index.md). A branch URL supplies code and recipes; it does not supply a GPU, ADS permission/credentials, operational CAMS data or unpublished refinement checkpoints.

## Clean-machine setup

After reviewing and approving local setup, clone the intended branch and create the lightweight environment. Use Python 3.12–3.14 compatible with the pinned requirements; the tested Python/package versions are recorded by environment preflight and the implementation validation report.

```bash
git clone --branch agent_skills_to_finetune_aurora --single-branch https://github.com/midatm1234/aurora.git
cd aurora
python scripts/bootstrap_workflow.py --venv .venv-workflow
```

The clone command applies after the branch has actually been published. For a local handoff, use the existing checked-out branch or a Git bundle. Check its resolved commit before running setup; [provenance/source-lock.json](../provenance/source-lock.json) identifies source lineage. Keep the fork as the imported package. The weather starter project and PyPI-only replacement are not this workflow.

The lightweight environment supports metadata/knowledge/MCP inspection. For CPU synthetic science checks, approve the larger dependency installation and add `--science` to the initial bootstrap command. Bootstrap preserves existing environments: if the lightweight environment already exists, create a distinct environment such as `.venv-science` with `--science` and register that Python for scientific execution. Real GPU work additionally needs a compatible CUDA-enabled PyTorch environment and sufficient local device memory; doctor reports actual availability. The documented worker runs on Linux with local CPU/CUDA resources. Native Windows/macOS worker management, hosted services and cluster schedulers are not validated.

Configure a dedicated workspace and local configuration, using your own absolute paths:

```bash
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json configure --workspace /absolute/path/aurora-workspace
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json doctor
```

Configuration defines independent `data_root`, `cache_root`, `output_root`, `state_root`, optional `read_roots` and an asset registry. Keep local configuration outside version control. Credentials belong in the locally configured ADS client environment/configuration, never in chat, recipes or reports. Read the current [ADS setup instructions](https://ads.atmosphere.copernicus.eu/how-to-api) and accept the selected product's access conditions in your own account. The tool checks credential presence without revealing values. Supply trained checkpoints as trusted local artifacts with checksums/architecture metadata; inaccessible historical checkpoints remain `requires_artifact`.

Doctor checks the Python/dependency environment, imported fork path, CUDA/GPU, memory/storage, credential presence and available asset metadata. Compatibility is also checked against the effective recipe before checkpoint loading. Use `doctor --check-torch` in the scientific environment to query actual PyTorch CUDA availability; the lightweight doctor avoids importing PyTorch by default. No expensive download or training is authorized by passing preflight.

## Register one supported local client

The same server runs on the recipient's machine:

```text
/absolute/path/aurora/.venv-workflow/bin/python -m aurora_workflow.mcp_server --config /absolute/path/aurora-local.json
```

Use the absolute environment Python, an editable fork installation, and an absolute local config path. Stdout is MCP protocol traffic; diagnostic logs go to stderr. The examples below were checked against separately fetched official documentation and are parsed/validated by repository tests. MCP SDK initialization/dispatch is tested separately. The local Codex CLI `0.154.0-alpha.6.2` also passed a read-only, per-invocation registration check using `mcp list --json` with explicit command/args; it recognized the enabled stdio entry without changing user configuration. The official MCP Python SDK `1.30.0` separately initialized this server with protocol `2025-11-25` and discovered 24 typed tools. This does not claim a paid/client-agent session or a Claude Code/VS Code application handshake.

| Client surface | Skill discovery/entry | MCP registration |
| --- | --- | --- |
| Codex local CLI / IDE extension | Canonical `.agents/skills`; root AGENTS.md | `~/.codex/config.toml`, table `mcp_servers` |
| Claude Code local CLI | CLAUDE.md imports AGENTS.md; `.claude/skills` symlink points to canonical skill | Local `claude mcp add`, or reviewed project `.mcp.json` with `mcpServers` |
| GitHub Copilot in VS Code | `.github/copilot-instructions.md` and canonical `.agents/skills` | `.vscode/mcp.json` with `servers` |

These are different configuration formats. GitHub cloud coding agents, browser-only clients and hosted GPU/server deployment are not covered by these local registration examples.

For Codex, merge the table in [codex-config.example.toml](clients/codex-config.example.toml), replacing paths. Or register locally:

```bash
codex mcp add aurora_air_pollution -- /absolute/path/aurora/.venv-workflow/bin/python -m aurora_workflow.mcp_server --config /absolute/path/aurora-local.json
codex mcp list
```

Restart the session when required by the client and inspect the server/tool list. The current official [Codex MCP guide](https://developers.openai.com/codex/mcp/) documents local stdio registration; the [skill guide](https://developers.openai.com/codex/skills/) documents `.agents/skills` discovery.

For Claude Code, use local scope to keep machine-specific paths out of tracked project files:

```bash
claude mcp add --transport stdio --scope local aurora_air_pollution -- /absolute/path/aurora/.venv-workflow/bin/python -m aurora_workflow.mcp_server --config /absolute/path/aurora-local.json
claude mcp get aurora_air_pollution
```

The [Claude JSON example](clients/claude-mcp.example.json) shows the project `.mcp.json` shape if desired; review trust prompts rather than committing an auto-approval setting. The `.claude/skills` adapter is a repository symlink, whose target is tested. If a checkout does not preserve symlinks, read the canonical skill through CLAUDE.md; restore the link for slash-command discovery. See official [MCP](https://code.claude.com/docs/en/mcp) and [skills](https://code.claude.com/docs/en/skills) documentation.

For GitHub Copilot in VS Code, merge [vscode-mcp.example.json](clients/vscode-mcp.example.json) into `.vscode/mcp.json` locally, replace paths, then use **MCP: List Servers** to start/review the server and inspect tools. Follow official [VS Code MCP](https://code.visualstudio.com/docs/agent-customization/mcp-servers) and [GitHub Copilot skills](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills) documentation. Avoid committing workstation-specific `.vscode/mcp.json` paths or secrets.

## Inspect, plan, authorize, execute

The agent starts with MCP `inspect_environment`, `inspect_assets`, `list_runs`; validates the selected recipe/head; then calls `plan_execution`. Reuse existing compatible artifacts before downloads or training. Four recipe IDs are `cpu-smoke-v1`, `real-gpu-v1`, `no2-us-west-v1`, `o3-global-v1`.

Use one of these four canonical refinement selections, or `none`:

| Human choice | `head` value |
| --- | --- |
| Flow matching with UNet | `flow_matching_conv_unet` |
| Flow matching with Transformer | `flow_matching_transformer` |
| Diffusion with UNet | `diffusion_unet` |
| Diffusion with Transformer | `diffusion_transformer` |

`flow_matching_unet` is the historical adapter, not an alias for the unified convolutional UNet. The common modes are `replay`, `resume`, `warm-start`, `reproduce`, `evaluate`; exact checkpoint requirements differ. Replay never silently creates a newly trained model. A requested `product: ensemble` needs at least two members and a compatible stochastic configuration; canonical point inference remains deterministic and spatial-only. For historical combined checkpoints, choose `no2-historical-forecast-forced-v1` or `o3-historical-forecast-forced-v1` only when their recorded exogenous-input contract matches. The historical-named variants reuse source forcing dynamics with specified-cycle CAMS forecasts, excluding unavailable future analyses. They are a versioned approximation to the old analysis-refresh behavior: checkpoint-head replay may be compatible while the baseline is shifted. Neither these variants nor the canonical autonomous recipes establish exact historical numerical reproduction. Identify flow versus diffusion from checkpoint/configuration metadata, not the word “Transformer”.

For an agent, use MCP. The following human CLI examples show the shared backend and allow a local operator to inspect a plan without executing it:

```bash
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json plan --recipe cpu-smoke-v1 --head flow_matching_conv_unet --mode reproduce --case-id smoke-flow-unet
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json plan --recipe cpu-smoke-v1 --head flow_matching_transformer --mode reproduce --case-id smoke-flow-transformer
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json plan --recipe cpu-smoke-v1 --head diffusion_unet --mode reproduce --case-id smoke-diffusion-unet
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json plan --recipe cpu-smoke-v1 --head diffusion_transformer --mode reproduce --case-id smoke-diffusion-transformer
```

For a real NO2 plan, replace the recipe with `no2-us-west-v1`; for O3 use `o3-global-v1`. Bound time, disk, download bytes, requests, steps, epochs, cases, memory, device and ensemble members in a local JSON request passed with `plan --request /absolute/path/request.json`. This is a typed `PlanRequest`, for example:

```json
{
  "recipe": "no2-us-west-v1",
  "head": "flow_matching_transformer",
  "mode": "replay",
  "case_id": "no2-replay-flow-transformer",
  "product": "point",
  "seed": 42,
  "inputs": {
    "pretrained": "/absolute/path/assets/aurora-0.4-air-pollution.ckpt",
    "static": "/absolute/path/assets/aurora-0.4-air-pollution-static.npz",
    "refinement": "/absolute/path/assets/trained-refinement.ckpt"
  },
  "limits": {
    "max_wall_seconds": 7200,
    "max_download_bytes": 0,
    "max_requests": 0,
    "gpu_count": 1,
    "device": "cuda:0"
  }
}
```

Paths must be under configured permitted roots; the trained checkpoint must carry the expected compatibility metadata. The static NPZ is the numerical representation produced by the approved asset tool from the exact official pickle, with its checksum/provenance sidecar. Do not rename or deserialize an arbitrary pickle. This request deliberately does not authorize downloads; missing prepared data/baseline/reference will appear as plan prerequisites. Use `real-gpu-v1` for the bounded acquisition-through-evaluation integration recipe and inspect its actual storage/request bounds before approval. Historical production dates, target levels, complete backbone fields and changes from old exogenous-input behavior are in recipes and the workflow inventory; do not overwrite them with slide assumptions.

The operator reviews the returned plan hash, resource limits, costs/unknowns, input fingerprints and stage list, then approves that plan in their own interactive terminal:

```bash
.venv-workflow/bin/python -m aurora_workflow --config /absolute/path/aurora-local.json approve PLAN_ID
```

No MCP tool approves a plan, and a model-authored `approved=true` is not authorization. Do not have the agent automate the terminal confirmation or call internal approval methods. Approval uses local HMAC state and is tied to the effective plan and inputs; this protects the MCP capability boundary, not against the machine owner or an agent separately granted unrestricted local shell access. Configure client permissions accordingly.

After approval, the agent can invoke MCP `execute_workflow` for the full approved chain or the explicitly planned individual stage tools. Long operations return a durable job ID. A human can use `execute PLAN_ID`, `status JOB_ID`, `logs JOB_ID`, `cancel JOB_ID`, or `recover JOB_ID` through the same CLI. Do not repeat individual approval requests for unchanged stages. Changed code/configuration/input hashes or bounds require a new plan. Recovery reuses supported persisted artifacts/state within the original budget; incompatible or missing evidence remains blocked.

## Read results and knowledge

Inspect `metrics.json`, the SVG plot, Markdown evaluation summary, manifests and evaluation receipt under the configured run output. Baseline fields remain separately available. Evaluate identical cases/coordinates/units/masks, report exclusions, and distinguish deterministic points from stochastic members and ensemble statistics. Positive error improvement is `100*(baseline-refined)/baseline`; negative means worse. CAMS operational reference agreement is not independent observational validation.

Run `knowledge_validate` through MCP (or human CLI `knowledge-validate`). Search/read accept `scope: "bundle"` for committed concepts or `scope: "runs"` for the runtime overlay. They use offline Markdown; no database, embedding service or paid API is needed. `knowledge_refresh` previews index changes without mutating the bundled documents; the controlled local helper can apply them. `knowledge_register` verifies a successful scientific run's artifact hashes (metadata/report-only jobs do not count) and writes a compact concept to the runtime knowledge overlay, retaining receipts outside committed knowledge. Never label synthetic smoke metrics, historical prose or an unavailable presentation as new real-data evidence.

Default validation is `python -m pytest --confcutdir=tests/workflow tests/workflow`, without large downloads/private credentials. Opt-in real GPU testing is explicitly gated by local resources and the documented test configuration; skips/blockers must retain their reason. Check the implementation report for exactly which tests and client sessions actually ran.

Reference revisions and accessed-page hashes are in [provenance/references.json](../provenance/references.json). They record documentation research, not an assertion that every upstream page is immutable.
