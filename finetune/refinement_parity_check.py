"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Fixed numerical regression case for the existing Aurora flow-matching refiner.

Purpose
-------
Prove that routing the **existing** ``flow_matching_unet`` head through the
unified Phase-2 interface (:class:`finetune.refinement.legacy_flow.\
LegacyFlowMatchingUNetRefiner`) and through the explicit checkpoint migration in
:mod:`finetune.refinement.checkpoint` does not change its numerical behaviour.

Everything that could perturb the result is pinned:

* a real flow-matching checkpoint (read-only; never modified),
* the resolved YAML configuration recorded inside that checkpoint,
* a fixed synthetic input batch and target batch,
* a fixed device, a fixed precision (float32),
* a fixed random seed and a fixed initial stochastic sample,
* a fixed number of flow-integration steps.

Recorded quantities
-------------------
* the flow-matching training loss,
* the predicted (deterministic, source-mean) residual,
* the sampled residual for a fixed initial state,
* the final refined field in normalized target space,
* summary evaluation metrics (bias / MAE / RMSE against the fixed target).

Usage::

    python -m finetune.refinement_parity_check \\
        --checkpoint finetune/outputs/checkpoints/O3_global_3day_lead/best.ckpt \\
        --report /tmp/flow_parity.json

``--checkpoint`` may also point at a small head-only extract produced by
``--extract-heads``, which avoids re-reading a multi-gigabyte payload.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Mapping

import torch
from torch import nn

from finetune.refinement.checkpoint import migrate_legacy_flow_state_dict
from finetune.refinement.config import resolve_refinement_config
from finetune.refinement.packing import ChannelSpec, FieldPacking

_FLOW_MODULE_PREFIXES = ("surf_flow.", "atmos_flow.")
_FLOW_HEAD_PREFIXES = (*_FLOW_MODULE_PREFIXES, "_res_std__")
_HEAD_PREFIXES = (*_FLOW_HEAD_PREFIXES, "temporal.")

#: Tolerances used to declare parity. Both paths execute the *same* modules on
#: the same inputs, so tensor-valued results are expected to match bitwise
#: (``atol = 0``). The scalar loss is aggregated slightly differently by the two
#: call sites (a Python mean over per-variable floats versus a tensor mean), so
#: it is compared with a float32 round-off tolerance.
ATOL = 0.0
LOSS_ATOL = 1e-6
LOSS_RTOL = 1e-6


def _load_payload(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "heads" in payload:
        heads = dict(payload["heads"])
        _require_flow_heads(path, heads)
        return {
            "heads": heads,
            "config": payload.get("config"),
            "norm_stats": payload.get("norm_stats"),
            "flow_sampling_steps": payload.get("flow_sampling_steps"),
        }
    state = payload.get("model_state_dict", payload)
    heads = {k: v for k, v in state.items() if k.startswith(_HEAD_PREFIXES)}
    _require_flow_heads(path, heads)
    return {
        "heads": {k: v.clone() for k, v in heads.items()},
        "config": payload.get("config"),
        "norm_stats": payload.get("norm_stats"),
        "flow_sampling_steps": payload.get("flow_sampling_steps"),
    }


def _require_flow_heads(path: str, heads: Mapping[str, torch.Tensor]) -> None:
    """Reject temporal-only/statistics-only payloads before building a packing."""
    if not any(key.startswith(_FLOW_MODULE_PREFIXES) for key in heads):
        raise SystemExit(
            f"{path!r} does not contain flow-matching refinement heads "
            "(surf_flow.* / atmos_flow.*)."
        )


def _head_layout(heads: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Recover the surface/atmospheric variables and per-variable level counts."""
    surf = sorted({k.split(".")[1] for k in heads if k.startswith("surf_flow.")})
    atmos = sorted({k.split(".")[1] for k in heads if k.startswith("atmos_flow.")})
    levels: dict[str, int] = {}
    for name in atmos:
        buffer = heads.get(f"_res_std__atmos__{name}")
        levels[name] = int(buffer.numel()) if buffer is not None and buffer.ndim else 1
    return {"surf": surf, "atmos": atmos, "levels": levels}


def build_packing(
    heads: Mapping[str, torch.Tensor],
    *,
    height: int,
    width: int,
    lead_scale_hours: float,
    lon_periodic: bool,
) -> FieldPacking:
    """Build the packing implied by a legacy checkpoint's head layout."""
    layout = _head_layout(heads)
    channels: list[ChannelSpec] = []
    levels_by_var: dict[str, tuple[float, ...]] = {}
    index = 0
    for name in layout["surf"]:
        channels.append(
            ChannelSpec(
                index=index,
                aurora_name=name,
                dataset_name=name,
                kind="surf",
                level=None,
                level_index=None,
                mean=0.0,
                std=1.0,
            )
        )
        index += 1
    for name in layout["atmos"]:
        count = int(layout["levels"].get(name, 1))
        levels = tuple(float(1000 - 100 * i) for i in range(count))
        levels_by_var[name] = levels
        for level_index, level in enumerate(levels):
            channels.append(
                ChannelSpec(
                    index=index,
                    aurora_name=name,
                    dataset_name=name,
                    kind="atmos",
                    level=level,
                    level_index=level_index,
                    mean=0.0,
                    std=1.0,
                )
            )
            index += 1
    return FieldPacking(
        channels=tuple(channels),
        lat=tuple(float(90 - i * (180 / max(height - 1, 1))) for i in range(height)),
        lon=tuple(float(i * (360 / width)) for i in range(width)),
        lead_times_hours=(lead_scale_hours,),
        lead_time_scale_hours=lead_scale_hours,
        lon_periodic=lon_periodic,
        _levels_by_var=levels_by_var,
    )


def _build_reference(config: Mapping[str, Any], packing: FieldPacking, heads):
    """Construct the untouched :class:`AuroraFlowRefine` heads."""
    from finetune.flow_refine import AuroraFlowRefine

    model_cfg = dict((config or {}).get("model", {}))
    surf = tuple(spec.aurora_name for spec in packing.channels if spec.kind == "surf")
    atmos = tuple(
        dict.fromkeys(spec.aurora_name for spec in packing.channels if spec.kind == "atmos")
    )
    reference = AuroraFlowRefine(
        base=nn.Identity(),
        target_surf_vars=surf,
        target_atmos_vars=atmos,
        hidden=int(model_cfg.get("flow_refine_hidden", 64)),
        time_dim=int(model_cfg.get("flow_refine_time_dim", 128)),
        sampling_steps=int(model_cfg.get("flow_refine_sampling_steps", 1)),
        atmos_loss_levels={n: list(range(len(packing.levels_for(n)))) for n in atmos} or None,
        lead_time_cond=bool(model_cfg.get("flow_refine_lead_time_cond", False)),
        lead_time_scale_hours=float(model_cfg.get("flow_refine_lead_time_scale_hours", 72.0)),
        residual_zscore=bool(model_cfg.get("flow_refine_residual_zscore", False)),
        res_std_momentum=float(model_cfg.get("flow_refine_res_std_momentum", 0.99)),
        lon_periodic=packing.lon_periodic,
    )
    missing, unexpected = reference.load_state_dict(dict(heads), strict=False)
    unexpected = [k for k in unexpected if k.startswith(_FLOW_HEAD_PREFIXES)]
    missing = [k for k in missing if k.startswith(_FLOW_HEAD_PREFIXES)]
    if missing or unexpected:
        raise SystemExit(
            f"Legacy head weights did not load cleanly: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    reference.eval()
    return reference


def _build_adapter(config: Mapping[str, Any], packing: FieldPacking, heads):
    """Construct the unified adapter and load the same weights through migration."""
    from finetune.refinement.base import build_refiner

    refinement = resolve_refinement_config(config)
    refiner = build_refiner(
        refinement,
        residual_channels=packing.num_channels,
        cond_channels=packing.num_channels,
        metadata=packing,
    )
    assert refiner is not None
    # This utility compares only the spatial flow head. Optional temporal Mamba
    # weights are loaded by the production wrapper, not by the bare refiner used
    # for this isolated migration/parity comparison.
    migrated, renames = migrate_legacy_flow_state_dict(
        {key: value for key, value in heads.items() if key.startswith(_FLOW_HEAD_PREFIXES)}
    )
    stripped = {
        key[len("refiner.") :]: value
        for key, value in migrated.items()
        if key.startswith("refiner.")
    }
    missing, unexpected = refiner.load_state_dict(stripped, strict=False)
    unexpected = list(unexpected)
    missing = [k for k in missing if "legacy." in k]
    if missing or unexpected:
        raise SystemExit(
            f"Migrated head weights did not load cleanly: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    refiner.eval()
    return refiner, renames


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.double() - b.double()).abs()
    finite = torch.isfinite(diff)
    return float(diff[finite].max()) if bool(finite.any()) else 0.0


def run_parity(
    checkpoint: str,
    *,
    height: int = 32,
    width: int = 48,
    batch: int = 2,
    seed: int = 1234,
    integration_steps: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the fixed regression case and return a JSON-serialisable report."""
    torch.use_deterministic_algorithms(False)
    payload = _load_payload(checkpoint)
    heads = payload["heads"]
    config = payload.get("config") or {}
    model_cfg = dict(config.get("model", {}))
    if integration_steps is not None:
        model_cfg["flow_refine_sampling_steps"] = int(integration_steps)
        config = {**config, "model": model_cfg}

    packing = build_packing(
        heads,
        height=height,
        width=width,
        lead_scale_hours=float(model_cfg.get("flow_refine_lead_time_scale_hours", 72.0)),
        lon_periodic=bool(model_cfg.get("lon_periodic_resolved", True)),
    )

    reference = _build_reference(config, packing, heads).to(device)
    adapter, renames = _build_adapter(config, packing, heads)
    adapter = adapter.to(device)

    # Fixed input / target batch, fixed precision.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rollout = torch.randn(
        batch, packing.num_channels, height, width, generator=generator, dtype=torch.float32
    ).to(device)
    target = rollout + 0.05 * torch.randn(
        rollout.shape, generator=generator, dtype=torch.float32
    ).to(device)
    lead = torch.full((batch,), 72.0, dtype=torch.float32, device=device)

    rollout_fields = packing.unpack(rollout)
    target_fields = packing.unpack(target)

    # 1) Flow-matching training loss, identical RNG stream on both paths.
    torch.manual_seed(seed)
    reference_losses = {
        name: float(
            reference.flow_loss(
                pred_norm=pred,
                target_norm=target_fields[name],
                var_name=name,
                kind="surf" if pred.dim() == 3 else "atmos",
                lead_time_hours=lead,
            ).detach()
        )
        for name, pred in rollout_fields.items()
    }
    reference_loss = sum(reference_losses.values()) / max(len(reference_losses), 1)

    torch.manual_seed(seed)
    adapter_out = adapter.compute_training_loss(
        residual_target=target - rollout,
        conditioning=rollout,
        forecast_lead_time=lead,
        rollout_normalized=rollout,
    )
    adapter_loss = float(adapter_out.total_loss.detach())

    # 2) Deterministic (source-mean) residual and refined field.
    with torch.no_grad():
        reference_refined = packing.pack(
            {
                name: reference.refine_norm_deterministic(
                    pred, name, "surf" if pred.dim() == 3 else "atmos", lead_time_hours=lead
                )
                for name, pred in rollout_fields.items()
            }
        )
        reference_residual = reference_refined - rollout
        adapter_residual = adapter.deterministic_residual(
            rollout, forecast_lead_time=lead, rollout_normalized=rollout
        )
        adapter_refined = rollout + adapter_residual

    # 3) Sampled residual from a fixed initial stochastic sample. Mirrors the
    # normalized-space arithmetic of ``AuroraFlowRefine.refine_prediction``:
    # sample the standardised residual, then un-standardise with sigma_r.
    torch.manual_seed(seed)
    with torch.no_grad():
        sampled_fields = {}
        for name, pred in rollout_fields.items():
            kind = "surf" if pred.dim() == 3 else "atmos"
            heads_module = reference.surf_flow if kind == "surf" else reference.atmos_flow
            flat = pred.reshape(-1, 1, height, width)
            lead_n = lead.repeat_interleave(pred.shape[1]) if kind == "atmos" else lead
            raw = reference._sample_residual(flat, heads_module[name], lead_hours=lead_n)
            sigma = reference._residual_std(kind, name, pred, update=False)
            raw = raw.reshape(pred.shape)
            if kind == "atmos":
                sigma = sigma.reshape(1, -1, 1, 1) if sigma.ndim else sigma
            else:
                sigma = sigma.reshape(-1)[0] if sigma.ndim else sigma
            sampled_fields[name] = raw * sigma
        sampled_reference = packing.pack(sampled_fields)
    torch.manual_seed(seed)
    with torch.no_grad():
        sampled_adapter = adapter.sample_residual(
            rollout, forecast_lead_time=lead, rollout_normalized=rollout
        )

    def _metrics(refined: torch.Tensor) -> dict[str, float]:
        error = (refined - target).double()
        return {
            "bias": float(error.mean()),
            "mae": float(error.abs().mean()),
            "rmse": float(error.pow(2).mean().sqrt()),
        }

    report = {
        "checkpoint": os.path.abspath(checkpoint),
        "device": device,
        "precision": "float32",
        "seed": seed,
        "batch": batch,
        "grid": [height, width],
        "integration_steps": int(model_cfg.get("flow_refine_sampling_steps", 1)),
        "channels": packing.num_channels,
        "variables": list(packing.variables),
        "renamed_keys": len(renames),
        "flow_loss": {
            "reference": reference_loss,
            "unified_adapter": adapter_loss,
            "abs_diff": abs(reference_loss - adapter_loss),
            "per_variable_reference": reference_losses,
        },
        "predicted_residual_max_abs_diff": _max_abs_diff(reference_residual, adapter_residual),
        "sampled_residual_max_abs_diff": _max_abs_diff(sampled_reference, sampled_adapter),
        "refined_output_max_abs_diff": _max_abs_diff(reference_refined, adapter_refined),
        "metrics_reference": _metrics(reference_refined),
        "metrics_unified_adapter": _metrics(adapter_refined),
        "tolerance": {"tensor_atol": ATOL, "loss_atol": LOSS_ATOL, "loss_rtol": LOSS_RTOL},
    }
    report["parity"] = (
        report["flow_loss"]["abs_diff"] <= max(LOSS_ATOL, LOSS_RTOL * abs(reference_loss))
        and report["predicted_residual_max_abs_diff"] <= ATOL
        and report["sampled_residual_max_abs_diff"] <= ATOL
        and report["refined_output_max_abs_diff"] <= ATOL
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="flow-matching checkpoint (read-only)")
    parser.add_argument("--report", default="", help="write the JSON report here")
    parser.add_argument("--extract-heads", default="", help="write a head-only extract here")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--integration-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.extract_heads:
        payload = _load_payload(args.checkpoint)
        torch.save(payload, args.extract_heads)
        print(f"Wrote head-only extract to {args.extract_heads}")

    report = run_parity(
        args.checkpoint,
        height=args.height,
        width=args.width,
        batch=args.batch,
        seed=args.seed,
        integration_steps=args.integration_steps,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0 if report["parity"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
