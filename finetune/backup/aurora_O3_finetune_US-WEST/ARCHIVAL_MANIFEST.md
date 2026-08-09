# Aurora O3 Finetune US-WEST Configuration Archive

**Archival Date:** 2026-08-09  
**Archived By:** Copilot  
**Reason for Archival:** Legacy O3 regional US-WEST fine-tuning case, superseded by:
- `aurora_O3_global_finetune_3day_lead_config.yaml` (active global O3 configuration)
- New stochastic-refinement example configurations in `finetune/examples/stochastic_refinement/`

---

## Archived Files

| Filename | Original Path | New Backup Path | Version | Status |
|----------|---------------|-----------------|---------|--------|
| `aurora_O3_finetune_US-WEST_3day_lead_config.yaml` | `finetune/` | `finetune/backup/aurora_O3_finetune_US-WEST/` | v1 (base) | Archived |
| `aurora_O3_finetune_US-WEST_3day_lead_config_v2.yaml` | `finetune/` | `finetune/backup/aurora_O3_finetune_US-WEST/` | v2 | Archived |
| `aurora_O3_finetune_US-WEST_3day_lead_config_v3.yaml` | `finetune/` | `finetune/backup/aurora_O3_finetune_US-WEST/` | v3 | Archived |
| `aurora_O3_finetune_US-WEST_3day_lead_config_v4.yaml` | `finetune/` | `finetune/backup/aurora_O3_finetune_US-WEST/` | v4 (final) | Archived |

---

## Retention & Reproducibility

These files are **retained indefinitely** for reproducibility and historical reference. They should **not be deleted**. However:

- ✓ They are **no longer** used as active training examples
- ✓ They are **not** included in automatic configuration discovery
- ✓ They will **not** be run by CI smoke tests or batch launchers
- ✓ They **can** still be explicitly loaded by their backup path if needed for historical reproduction

### Loading a Archived Configuration

To reproduce a legacy O3 US-WEST fine-tuning run:

```bash
python /data/aurora/finetune/aurora_finetune_distributed.py \
  --config /data/aurora/finetune/backup/aurora_O3_finetune_US-WEST/aurora_O3_finetune_US-WEST_3day_lead_config_v4.yaml
```

Or in a notebook:

```python
import yaml
config_path = "finetune/backup/aurora_O3_finetune_US-WEST/aurora_O3_finetune_US-WEST_3day_lead_config_v4.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
```

---

## Active Replacements

**For O3 Fine-Tuning:**
- Use `aurora_O3_global_finetune_3day_lead_config.yaml` (global, recommended)
- See `finetune/examples/stochastic_refinement/` for stochastic-refinement variations

**For NO₂ Fine-Tuning:**
- Use `aurora_NO2_finetune_US-WEST_3day_lead_config.yaml` (regional US-WEST)

**For General Refinement Examples:**
- See `finetune/examples/stochastic_refinement/` and `finetune/REFINEMENT_REVIEW.md`

---

## Update History

### Moves & References Updated (2026-08-09)

The following files and references were updated to point to the backup location:

1. **Shell Scripts:**
   - `finetune/run_O3_v2_papermill.sh` → Updated to `finetune/backup/aurora_O3_finetune_US-WEST/aurora_O3_finetune_US-WEST_3day_lead_config_v2.yaml`
   - `finetune/run_O3_v3_papermill.sh` → Updated to `finetune/backup/aurora_O3_finetune_US-WEST/aurora_O3_finetune_US-WEST_3day_lead_config_v3.yaml`
   - `finetune/run_O3_v4_papermill.sh` → Updated to `finetune/backup/aurora_O3_finetune_US-WEST/aurora_O3_finetune_US-WEST_3day_lead_config_v4.yaml`

2. **Documentation:**
   - `Aurora_air_pollution_finetune.md` → References updated from main directory to backup paths where appropriate

3. **Notebooks in Output Artifacts:**
   - Existing execution notebooks in `finetune/outputs/O3_US-WEST_3day_lead_v*/` retain original paths for historical accuracy but are not re-run

---

## No Automatic Discovery

Code that automatically discovers YAML files for training excludes `backup/` directories:

- ✓ `finetune/refinement_smoke_test.py` does not enumerate archived YAMLs
- ✓ Configuration-validation matrices skip `backup/` subdirectories
- ✓ Default example listings in documentation use active configs only

---

## Files NOT Archived

Intentionally left in the active configuration directory:

- ✓ `aurora_O3_global_finetune_3day_lead_config.yaml` (active global O3)
- ✓ `aurora_NO2_finetune_US-WEST_3day_lead_config.yaml` (active NO₂ US-WEST)
- ✓ `aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml` (active NO₂ diffusion variant)
- ✓ `aurora_finetune_US-WEST_1.5day_lead_config.yaml` (generic regional config, not O3-specific)
- ✓ All configs in `finetune/examples/stochastic_refinement/`

---

## Validation Checklist

- ✓ All four O3 US-WEST configs moved to backup
- ✓ No configs with different case names were moved
- ✓ Global O3 and NO₂ configs remain active
- ✓ Shell scripts updated to reference backup paths
- ✓ Documentation references updated
- ✓ Automatic discovery excludes backup directory
- ✓ Archived YAMLs remain parseable and loadable
- ✓ Git history preserved via `git mv`

---

## Questions or Issues?

If you need to restore an archived configuration to active status or report an issue with the archival, contact the repository maintainers.
