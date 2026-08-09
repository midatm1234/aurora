"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Minimal, non-destructive smoke tests for Aurora stochastic residual refinement.

Runs one tiny forward/backward pass per refiner plus one ensemble inference and
one NetCDF write, on a very small synthetic subset, into a temporary output
directory. It never launches a full training run, never reads or writes an
existing checkpoint, dataset, rollout output or YAML file, and never overwrites
anything.

Optionally, ``--flow-checkpoint`` points at an existing flow-matching checkpoint
(read-only) to additionally run the legacy numerical-parity case from
:mod:`finetune.refinement_parity_check`.

Usage::

    python -m finetune.refinement_smoke_test
    python -m finetune.refinement_smoke_test \\
        --flow-checkpoint finetune/outputs/checkpoints/O3_global_3day_lead/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from finetune.refinement.integration import LeadStepBuffer
from finetune.refinement.io import build_refined_dataset, write_refined_netcdf
from finetune.refinement.packing import FieldPacking
from finetune.refinement.two_phase import build_two_phase_refiner

REFINERS = (
    "flow_matching_unet",
    "flow_matching_transformer",
    "diffusion_unet",
    "diffusion_transformer",
)


@dataclass
class _Spec:
    aurora_name: str
    dataset_name: str
    kind: str
    loss_levels: list[float] | None = None
    units: str = ""


def build_packing(height: int, width: int) -> FieldPacking:
    specs = [
        _Spec("gtco3", "gtco3", "surf", units="kg m-2"),
        _Spec("go3", "go3", "atmos", [500.0, 850.0], units="kg kg-1"),
    ]
    norm_stats = {
        "gtco3": {"mean": torch.tensor([1.0e-3]), "std": torch.tensor([2.0e-4])},
        "go3": {"mean": torch.tensor([0.0, 0.0]), "std": torch.tensor([1.0e-7, 2.0e-7])},
    }
    return FieldPacking.from_specs(
        specs,
        norm_stats=norm_stats,
        atmos_levels=[500.0, 850.0],
        lat=[90.0 - i * (180.0 / max(height - 1, 1)) for i in range(height)],
        lon=[i * (360.0 / width) for i in range(width)],
        lead_times_hours=[24.0, 48.0, 72.0],
        lon_periodic=True,
    )


def smoke_config(refinement_type: str, ensemble: int) -> dict[str, Any]:
    """Smoke-test-only overrides, deliberately kept out of the example YAMLs."""
    refinement: dict[str, Any] = {
        "type": refinement_type,
        "enabled": True,
        "ensemble_size": ensemble,
        "seed": 1234,
        "conditioning": {
            "aurora_rollout": True,
            "static_fields": False,
            "masks": True,
            "forecast_lead_time": True,
        },
        "unet": {
            "hidden_channels": 8,
            "num_levels": 2,
            "num_residual_blocks": 1,
            "time_embedding_dim": 16,
            "attention_heads": 4,
        },
        "transformer": {
            "patch_size": [4, 4],
            "embedding_dim": 32,
            "num_heads": 4,
            "num_blocks": 2,
            "max_tokens_lat": 32,
            "max_tokens_lon": 32,
            "attention_mode": "windowed_2d",
            "window_size": [2, 2],
            "shifted_windows": True,
        },
    }
    if refinement_type.startswith("diffusion"):
        # Minimal smoke-test-specific process step counts.
        refinement["diffusion"] = {"training_timesteps": 20, "inference_steps": 3}
    else:
        refinement["flow_matching"] = {"integration_steps": 2}
    return {"model": {"refinement": refinement}}


def run_smoke(
    *,
    height: int = 16,
    width: int = 24,
    batch: int = 1,
    leads: tuple[int, ...] = (1, 2),
    ensemble: int = 3,
    output_dir: str | None = None,
    flow_checkpoint: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(0)
    packing = build_packing(height, width)
    results: dict[str, Any] = {"refiners": {}}

    for refinement_type in REFINERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build_two_phase_refiner(None, packing, smoke_config(refinement_type, ensemble))
            model.initialize_refiner(model.conditioning_channels())

        # --- one deterministic "rollout" (synthetic) ---------------------
        buffer = LeadStepBuffer(packing)
        for position, lead in enumerate(leads):
            surf = torch.randn(batch, height, width)
            atmos = torch.randn(batch, 2, height, width)
            hours = 24.0 * (position + 1)
            buffer.add(
                lead,
                "gtco3",
                rollout_normalized=surf,
                target_normalized=surf + 0.05,
                lead_hours=hours,
            )
            buffer.add(
                lead,
                "go3",
                rollout_normalized=atmos,
                target_normalized=atmos + 0.05,
                lead_hours=hours,
            )
        rollout, target, mask, lead_hours, lead_index = buffer.pack()

        # --- one training step -------------------------------------------
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(7)
        step = model.training_step(
            rollout,
            target,
            valid_mask=mask,
            forecast_lead_time=lead_hours,
            lead_index=lead_index,
            generator=generator,
        )
        loss = step.losses["total_loss"]
        loss.backward()
        grad_norm = float(
            torch.sqrt(
                sum(p.grad.pow(2).sum() for p in model.refiner.parameters() if p.grad is not None)
            )
        )
        train_seconds = time.perf_counter() - started

        # --- one inference pass with an ensemble --------------------------
        started = time.perf_counter()
        inference = model.refine(
            rollout,
            forecast_lead_time=lead_hours,
            ensemble_size=ensemble,
            seed=99,
        )
        infer_seconds = time.perf_counter() - started

        assert inference.members.shape == (
            rollout.shape[0],
            ensemble,
            packing.num_channels,
            height,
            width,
        )
        # Deterministic rollout untouched by refinement (postprocessing).
        assert torch.equal(inference.deterministic_normalized, rollout)

        results["refiners"][refinement_type] = {
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "refiner_parameters": model.refine_parameter_count(),
            "members_shape": list(inference.members.shape),
            "ensemble_spread_mean": float(inference.ensemble_spread.mean()),
            "residual_finite": bool(torch.isfinite(inference.residual).all()),
            "refined_finite": bool(torch.isfinite(inference.refined_physical).all()),
            "train_step_seconds": round(train_seconds, 4),
            "ensemble_inference_seconds": round(infer_seconds, 4),
            "feedback_to_rollout": model.refinement_config.feedback_to_rollout,
        }

        # --- one NetCDF write (temporary directory only) ------------------
        if refinement_type == REFINERS[-1]:
            steps = rollout.shape[0]
            import numpy as np

            with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
                dataset = build_refined_dataset(
                    packing=packing,
                    deterministic=model.target_space.decode(rollout),
                    init_time=np.array(["2024-01-01T00"] * steps, dtype="datetime64[h]"),
                    valid_time=np.array(
                        [f"2024-01-0{1 + i // 2}T{(i % 2) * 12:02d}" for i in range(steps)],
                        dtype="datetime64[h]",
                    ),
                    lead_time_hours=lead_hours.tolist(),
                    refined=inference.refined_physical,
                    residual=inference.residual,
                    members=inference.members,
                    ensemble_mean=inference.ensemble_mean,
                    ensemble_spread=inference.ensemble_spread,
                )
                path = write_refined_netcdf(
                    dataset, Path(tmp) / "smoke_refined.nc", compression_level=1
                )
                results["netcdf"] = {
                    "path": path,
                    "variables": sorted(dataset.data_vars),
                    "sizes": {k: int(v) for k, v in dataset.sizes.items()},
                }

    # --- optional: legacy checkpoint parity -------------------------------
    if flow_checkpoint:
        from finetune.refinement_parity_check import run_parity

        results["legacy_parity"] = run_parity(flow_checkpoint)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="parent directory for the temporary NetCDF output (never overwritten)",
    )
    parser.add_argument(
        "--flow-checkpoint",
        default=None,
        help="existing flow-matching checkpoint, opened read-only for the parity case",
    )
    parser.add_argument("--report", default="", help="write the JSON report here")
    args = parser.parse_args()

    results = run_smoke(
        height=args.height,
        width=args.width,
        ensemble=args.ensemble,
        output_dir=args.output_dir,
        flow_checkpoint=args.flow_checkpoint,
    )
    print(json.dumps(results, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2) + "\n")
    parity = results.get("legacy_parity")
    return 0 if (parity is None or parity["parity"]) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
