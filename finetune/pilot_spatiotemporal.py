"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Bounded capability pilot for joint spatiotemporal refinement.

Purpose
-------
This is a **controlled capability test on a synthetic problem with a known
answer**, not a CAMS skill measurement. It answers one question that a CAMS run
cannot answer cheaply or unambiguously:

    Does each conditioning stage actually give the refiner access to the
    information it claims to add?

The synthetic residual is constructed so that each stage is *provably*
necessary:

``spatial``
    a fixed spatial dipole. Recoverable from the Aurora field alone.
``calendar``
    a term proportional to ``sin(2*pi*year_phase) * cos(solar_zenith)``. It
    depends on the frame's own **valid** timestamp and on longitude through
    local solar position. A spatial-only refiner sees the same Aurora input at
    every valid time and therefore *cannot* separate it; a calendar-conditioned
    refiner can.
``temporal``
    a first-order autoregressive drift whose amplitude at lead ``j`` depends on
    the *realized history* of the trajectory, not on the lead label. A refiner
    conditioned only on lead ``j`` cannot recover it, because two trajectories
    with identical leads but different histories require different corrections.
    This is the discriminator that separates genuine temporal modelling from
    lead-indexed lookup, which is exactly the distinction the historical Mamba
    ablation could not make.

Because the ground truth is known, a stage that fails here is broken; a stage
that succeeds here is *capable*, which is a necessary -- not sufficient --
condition for CAMS skill. No claim about ozone accuracy is made or implied.

Every variant sees identical data, identical seeds, identical optimizer
settings and an identical parameter budget for the shared backbone, so a
difference is attributable to the conditioning factor alone.

Usage::

    python -m finetune.pilot_spatiotemporal --steps 150 --head diffusion_unet
    python -m finetune.pilot_spatiotemporal --steps 150 --all-heads --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from tests.refinement_fixtures import build_packing, refinement_config

from finetune.refinement.calendar_features import (
    CalendarFeatureBuilder,
    cos_solar_zenith_angle,
    valid_times,
    year_phase,
)
from finetune.refinement.trajectory import (
    ensemble_crps,
    member_window_maximum,
    residual_autocorrelation,
    tendency_loss,
)
from finetune.refinement.two_phase import build_two_phase_refiner

__all__ = ["PilotResult", "run_pilot", "main"]

HEADS = (
    "flow_matching_conv_unet",
    "flow_matching_transformer",
    "diffusion_unet",
    "diffusion_transformer",
)

STAGES: dict[str, dict[str, Any]] = {
    "spatial_only": {"calendar": False, "backend": "none"},
    "calendar_only": {"calendar": True, "backend": "none"},
    "temporal_causal_conv": {"calendar": True, "backend": "causal_conv"},
    "temporal_conv_gru": {"calendar": True, "backend": "conv_gru"},
    "temporal_attention": {"calendar": True, "backend": "attention"},
    "temporal_mamba": {"calendar": True, "backend": "mamba"},
}

HEIGHT, WIDTH = 16, 24
STEPS = 4
LEAD_STEP_HOURS = 12.0


class PilotResult(dict):
    """A single variant's metrics."""


def _make_dataset(
    packing,
    *,
    batch: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Synthesize trajectories whose residual needs each conditioning stage."""
    channels = packing.num_channels
    latitude = torch.tensor(packing.lat, dtype=torch.float32)
    longitude = torch.tensor(packing.lon, dtype=torch.float32)

    # Initializations spread across the year and across UTC cycles, so calendar
    # features are not confounded with the trajectory index.
    init_time = [
        datetime(2024, 1, 1) + timedelta(days=37 * i, hours=6 * (i % 4))
        for i in range(batch)
    ]
    leads = torch.arange(1, STEPS + 1, dtype=torch.float32) * LEAD_STEP_HOURS
    lead_hours = leads[None, :].expand(batch, STEPS).contiguous()

    # Phase 1 "Aurora" forecast: smooth random fields.
    aurora = torch.randn(
        batch, STEPS, channels, HEIGHT, WIDTH, generator=generator
    ) * 0.5

    # --- spatial component: a fixed dipole -----------------------------
    dipole = torch.sin(torch.deg2rad(latitude))[:, None] * torch.cos(
        torch.deg2rad(longitude)
    )[None, :]
    spatial = 0.30 * dipole[None, None, None].expand(batch, STEPS, channels, HEIGHT, WIDTH)

    # --- calendar component: season x solar geometry -------------------
    calendar_term = torch.zeros_like(spatial)
    for b in range(batch):
        stamps = valid_times([init_time[b]] * STEPS, lead_hours[b])
        phase = torch.tensor(
            [math.sin(2.0 * math.pi * year_phase(s)) for s in stamps], dtype=torch.float32
        )
        zenith = cos_solar_zenith_angle(stamps, latitude, longitude)  # [S, H, W]
        calendar_term[b] = (0.45 * phase[:, None, None] * zenith)[:, None].expand(
            STEPS, channels, HEIGHT, WIDTH
        )

    # --- temporal component: history-dependent AR drift ----------------
    # The drift at lead j is a decaying accumulation of the *realized* Aurora
    # field, so it cannot be predicted from the lead label alone.
    temporal = torch.zeros_like(spatial)
    state = torch.zeros(batch, channels, HEIGHT, WIDTH)
    for step in range(STEPS):
        state = 0.65 * state + 0.55 * aurora[:, step]
        temporal[:, step] = 0.40 * torch.tanh(state)

    residual = spatial + calendar_term + temporal
    target = aurora + residual
    return {
        "aurora": aurora,
        "target": target,
        "residual": residual,
        "lead_hours": lead_hours,
        "init_time": init_time,
        "components": {
            "spatial": spatial,
            "calendar": calendar_term,
            "temporal": temporal,
        },
    }


def _build_model(head: str, stage: str, packing):
    options = STAGES[stage]
    config = refinement_config(
        head,
        ensemble_size=4,
        deterministic_head="shared_process",
        loss={"deterministic_weight": 1.0, "generative": "mse"},
        conditioning={
            "calendar": options["calendar"],
            "solar_geometry": options["calendar"],
            "vertical_identity": options["calendar"],
            "latitude": True,
            "longitude": True,
        },
        temporal={
            "backend": options["backend"],
            "context_channels": 8,
            "hidden_channels": 24,
            "layers": 2,
            "spatial_stride": 4,
        },
    )
    # These ablations isolate the new conditioning path. Keep the separate
    # legacy post-sampling adapter off even when its global defaults change.
    config["model"]["mamba_temporal"] = {"enabled": False}
    config["model"]["mamba_temporal_enabled"] = False
    model = build_two_phase_refiner(None, packing, config)
    model.initialize_refiner(model.conditioning_channels())
    return model


def _frame_inputs(model, data, *, channels: int):
    """Build the lead-major tensors and the causal conditioning for one batch."""
    batch = data["aurora"].shape[0]
    flat = batch * STEPS
    aurora_flat = data["aurora"].reshape(flat, channels, HEIGHT, WIDTH)
    target_flat = data["target"].reshape(flat, channels, HEIGHT, WIDTH)
    calendar = None
    if model.refinement_config.conditioning.calendar or (
        model.refinement_config.conditioning.solar_geometry
    ):
        calendar = model.calendar_spec(
            CalendarFeatureBuilder.expand_initializations(data["init_time"], STEPS),
            data["lead_hours"].reshape(-1),
        )
    context = None
    if model.has_temporal_context:
        context = model.build_temporal_context(
            data["aurora"],
            lead_hours=data["lead_hours"],
            init_time=data["init_time"],
        ).reshape(flat, model.refiner.temporal_context_channels, HEIGHT, WIDTH)
    return aurora_flat, target_flat, calendar, context


@torch.no_grad()
def _evaluate(model, data, *, channels: int) -> dict[str, float]:
    batch = data["aurora"].shape[0]
    aurora_flat, target_flat, calendar, context = _frame_inputs(
        model, data, channels=channels
    )
    output = model.refine(
        aurora_flat,
        forecast_lead_time=data["lead_hours"].reshape(-1),
        calendar=calendar,
        temporal_context=context,
        ensemble_size=4,
        return_members=True,
        seed=7,
    )
    refined = output.deterministic_refined_normalized
    raw_error = (aurora_flat - target_flat).abs().mean()
    refined_error = (refined - target_flat).abs().mean()
    raw_rmse = (aurora_flat - target_flat).pow(2).mean().sqrt()
    refined_rmse = (refined - target_flat).pow(2).mean().sqrt()

    shape = (batch, STEPS, channels, HEIGHT, WIDTH)
    refined_trajectory = refined.reshape(shape)
    target_trajectory = target_flat.reshape(shape)
    delta = torch.full((batch, STEPS), LEAD_STEP_HOURS)

    metrics = {
        "raw_mae": float(raw_error),
        "refined_mae": float(refined_error),
        "mae_skill_percent": float(100.0 * (1.0 - refined_error / raw_error)),
        "raw_rmse": float(raw_rmse),
        "refined_rmse": float(refined_rmse),
        "rmse_skill_percent": float(100.0 * (1.0 - refined_rmse / raw_rmse)),
        "refined_bias": float((refined - target_flat).mean()),
        "tendency_loss": float(
            tendency_loss(refined_trajectory, target_trajectory, delta, penalty="l2")
        ),
        "raw_tendency_loss": float(
            tendency_loss(
                aurora_flat.reshape(shape), target_trajectory, delta, penalty="l2"
            )
        ),
        "residual_autocorrelation": float(
            residual_autocorrelation(refined_trajectory, target_trajectory)
        ),
    }
    members = output.members
    if members is not None and members.shape[1] >= 2:
        member_trajectory = members.reshape(
            batch, STEPS, members.shape[1], channels, HEIGHT, WIDTH
        ).permute(0, 2, 1, 3, 4, 5)
        metrics["crps_horizon_maximum"] = float(
            ensemble_crps(
                member_window_maximum(member_trajectory),
                target_trajectory.amax(dim=1),
            )
        )
    return metrics


def run_pilot(
    head: str,
    stage: str,
    *,
    steps: int,
    batch: int,
    seed: int,
    learning_rate: float,
    train_trajectories: int = 48,
) -> PilotResult:
    """Train one variant on the synthetic problem and score it.

    The training set contains ``train_trajectories`` independent
    initializations and each step draws a fresh minibatch from it, so the score
    measures generalization rather than memorization of one fixed batch. The
    evaluation set is generated from a different generator state and is never
    trained on.
    """
    if head not in HEADS or stage not in STAGES:
        raise ValueError("Unknown pilot head or conditioning stage.")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if type(batch) is not int or type(train_trajectories) is not int or not 1 <= batch <= train_trajectories:
        raise ValueError("Require 1 <= batch <= train_trajectories.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if type(seed) is not int or not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1).")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    packing = build_packing(HEIGHT, WIDTH, lon_periodic=True)
    channels = packing.num_channels

    train = _make_dataset(packing, batch=train_trajectories, generator=generator)
    evaluation = _make_dataset(packing, batch=max(batch, 8), generator=generator)

    model = _build_model(head, stage, packing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    sampler = torch.Generator().manual_seed(seed + 1)

    started = time.time()
    losses: list[float] = []
    for _ in range(steps):
        index = torch.randperm(train_trajectories, generator=sampler)[:batch]
        minibatch = {
            "aurora": train["aurora"][index],
            "target": train["target"][index],
            "lead_hours": train["lead_hours"][index],
            "init_time": [train["init_time"][int(i)] for i in index],
        }
        optimizer.zero_grad()
        aurora_flat, target_flat, calendar, context = _frame_inputs(
            model, minibatch, channels=channels
        )
        output = model.training_step(
            aurora_flat,
            target_flat,
            forecast_lead_time=minibatch["lead_hours"].reshape(-1),
            lead_index=torch.arange(aurora_flat.shape[0]) % STEPS,
            calendar=calendar,
            temporal_context=context,
        )
        loss = output.losses["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    metrics = _evaluate(model, evaluation, channels=channels)
    metrics.update(
        {
            "head": head,
            "stage": stage,
            "temporal_backend": model.refinement_config.temporal.backend,
            "calendar_conditioning": model.refinement_config.conditioning.calendar,
            "trainable_parameters": int(sum(p.numel() for p in trainable)),
            "train_steps": steps,
            "train_trajectories": train_trajectories,
            "final_train_loss": (
                float(sum(losses[-20:]) / len(losses[-20:])) if losses else float("nan")
            ),
            "seconds": round(time.time() - started, 2),
        }
    )
    return PilotResult(metrics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=HEADS, default="diffusion_unet")
    parser.add_argument("--all-heads", action="store_true")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-trajectories", type=int, default=48)
    parser.add_argument("--json", type=Path, help="write machine-readable metrics here")
    args = parser.parse_args(argv)

    heads = list(HEADS) if args.all_heads else [args.head]
    rows: list[PilotResult] = []
    for head in heads:
        for stage in args.stages:
            row = run_pilot(
                head,
                stage,
                steps=args.steps,
                batch=args.batch,
                seed=args.seed,
                learning_rate=args.learning_rate,
                train_trajectories=args.train_trajectories,
            )
            rows.append(row)
            print(
                f"{head:28s} {stage:22s} "
                f"MAE skill {row['mae_skill_percent']:7.2f}%  "
                f"RMSE skill {row['rmse_skill_percent']:7.2f}%  "
                f"tendency {row['tendency_loss']:.4f} "
                f"(raw {row['raw_tendency_loss']:.4f})  "
                f"params {row['trainable_parameters']:,}  "
                f"{row['seconds']:.1f}s"
            )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
