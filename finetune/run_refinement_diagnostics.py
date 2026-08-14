"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Controlled stochastic-refinement experiments with auditable artifacts.

This runner deliberately uses the same packed target space, residual scaler,
conditioning stack, Transformer backbone, diffusion objective, and deterministic
inference path as Phase-2 training.  It provides two bounded experiments:

``synthetic``
    Fits a known, smooth, four-channel ``CAMS - Aurora`` correction containing
    a broad bias, latitude gradient, and local positive/negative features.

``real``
    Overfits one to eight exactly matched NO2 US-WEST Aurora/CAMS samples.  A
    failure here is a data/layout/conditioning/objective bug, not a data-volume
    problem.

Both ``diffusion_transformer`` and a direct-regression ablation of the identical
Transformer backbone are supported.  Every run writes JSON statistics and one
six-panel map per channel: Aurora, CAMS, true correction, predicted correction,
refined forecast, and remaining error.

Examples::

    python -m finetune.run_refinement_diagnostics --mode synthetic --train-steps 400
    python -m finetune.run_refinement_diagnostics --mode real --real-samples 4 \
        --train-steps 600 --output-dir finetune/outputs/refinement_diagnostics/tiny
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from finetune.refinement.base import build_refiner, masked_loss
from finetune.refinement.benchmark import (
    CASES,
    BenchmarkDataset,
    _conditioning,
    _deep_update,
    build_config,
    correction_statistics,
    evaluate_channels,
    load_case,
)
from finetune.refinement.losses import area_weights_from_latitudes
from finetune.refinement.packing import ChannelSpec, FieldPacking
from finetune.refinement.target_space import NormalizedTargetSpace

SUPPORTED_HEADS = (
    "diffusion_unet",
    "diffusion_transformer",
    "flow_matching_conv_unet",
    "flow_matching_transformer",
    "direct_regression_transformer",
)


def _channel_label(spec: ChannelSpec) -> str:
    return (
        spec.dataset_name if spec.level is None else f"{spec.dataset_name}@{float(spec.level):g}hPa"
    )


def _safe_label(value: str) -> str:
    return value.replace("@", "_").replace(".", "p").replace("/", "_")


def tensor_statistics(value: torch.Tensor) -> dict[str, Any]:
    """JSON-safe tensor diagnostics without hiding non-finite values."""
    detached = value.detach().float()
    finite = torch.isfinite(detached)
    selected = detached[finite]
    stats: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "nan_count": int(torch.isnan(detached).sum()),
        "positive_inf_count": int(torch.isposinf(detached).sum()),
        "negative_inf_count": int(torch.isneginf(detached).sum()),
        "finite_count": int(finite.sum()),
    }
    if selected.numel():
        stats.update(
            min=float(selected.min()),
            max=float(selected.max()),
            mean=float(selected.mean()),
            std=float(selected.std(unbiased=False)),
            rms=float(selected.square().mean().sqrt()),
        )
    else:
        stats.update(min=None, max=None, mean=None, std=None, rms=None)
    return stats


def spatial_texture_statistics(value: torch.Tensor) -> dict[str, float | None]:
    """Report pixel-scale energy and row/column signatures for ``[N,C,H,W]``."""
    x = value.detach().float()
    if x.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(x.shape)}.")
    x = torch.nan_to_num(x)
    centered = x - x.mean(dim=(-2, -1), keepdim=True)
    total = centered.square().mean().clamp(min=1e-20)
    dy = (centered[..., 1:, :] - centered[..., :-1, :]).square().mean()
    dx = (centered[..., :, 1:] - centered[..., :, :-1]).square().mean()
    row = centered.mean(dim=-1).square().mean()
    column = centered.mean(dim=-2).square().mean()

    spectrum = torch.fft.rfft2(centered, norm="ortho").abs().square()
    height, width = centered.shape[-2:]
    fy = torch.fft.fftfreq(height, device=x.device).abs()[:, None]
    fx = torch.fft.rfftfreq(width, device=x.device).abs()[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    high = radius >= 0.35
    spectral_total = spectrum.sum().clamp(min=1e-20)
    return {
        "horizontal_neighbor_mse_over_variance": float(dx / total),
        "vertical_neighbor_mse_over_variance": float(dy / total),
        "row_mean_energy_fraction": float(row / total),
        "column_mean_energy_fraction": float(column / total),
        "high_frequency_power_fraction": float(spectrum[..., high].sum() / spectral_total),
    }


def _synthetic_packing(height: int, width: int) -> FieldPacking:
    lat = np.linspace(52.0, 31.2, height, dtype=np.float64)
    lon = np.linspace(232.0, 259.6, width, dtype=np.float64)
    channels = (
        ChannelSpec(
            index=0,
            aurora_name="tcno2",
            dataset_name="tcno2",
            kind="surf",
            level=None,
            level_index=None,
            units="kg m**-2",
            mean=0.0,
            std=6.141e-5,
        ),
        ChannelSpec(
            index=1,
            aurora_name="no2",
            dataset_name="no2",
            kind="atmos",
            level=1000.0,
            level_index=0,
            units="kg kg**-1",
            mean=0.0,
            std=1.931e-7,
        ),
        ChannelSpec(
            index=2,
            aurora_name="no2",
            dataset_name="no2",
            kind="atmos",
            level=925.0,
            level_index=1,
            units="kg kg**-1",
            mean=0.0,
            std=1.379e-7,
        ),
        ChannelSpec(
            index=3,
            aurora_name="no2",
            dataset_name="no2",
            kind="atmos",
            level=850.0,
            level_index=2,
            units="kg kg**-1",
            mean=0.0,
            std=9.422e-8,
        ),
    )
    return FieldPacking(
        channels=channels,
        lat=tuple(float(v) for v in lat),
        lon=tuple(float(v) for v in lon),
        lon_periodic=False,
    )


def make_smooth_synthetic_dataset(
    *, samples: int = 6, height: int = 32, width: int = 40
) -> BenchmarkDataset:
    """Known smooth, distinct, conditioning-dependent four-channel correction."""
    if not 1 <= samples <= 8:
        raise ValueError(f"samples must be in [1, 8], got {samples}.")
    if height < 12 or width < 12:
        raise ValueError("Synthetic spatial dimensions must both be >= 12.")
    packing = _synthetic_packing(height, width)
    yy = torch.linspace(-1.0, 1.0, height).view(1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width).view(1, 1, width)
    rollouts: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    leads: list[float] = []
    init_ids: list[str] = []
    init_times: list[np.datetime64] = []
    valid_times: list[np.datetime64] = []
    base_time = np.datetime64("2024-07-01T00:00:00", "ns")

    for sample in range(samples):
        phase = 2.0 * math.pi * sample / max(samples, 2)
        cx = 0.42 * math.sin(phase)
        cy = 0.30 * math.cos(phase)
        plume = torch.exp(-((xx - cx) ** 2 / 0.12 + (yy - cy) ** 2 / 0.20))
        secondary = torch.exp(
            -((xx + 0.45 - 0.10 * math.cos(phase)) ** 2 / 0.045 + (yy - 0.36) ** 2 / 0.075)
        )
        negative = torch.exp(
            -((xx - 0.45) ** 2 / 0.060 + (yy + 0.42 - 0.08 * math.sin(phase)) ** 2 / 0.055)
        )
        wave = torch.sin(math.pi * (xx + 0.12 * sample)) * torch.cos(math.pi * yy)
        broad = torch.exp(-((xx + 0.05) ** 2 + (yy - 0.05) ** 2) / 0.85)

        # Positive, structured Aurora fields.  The moving plume makes the true
        # correction depend on the conditioning rather than being climatology.
        rollout = torch.cat(
            [
                0.070 + 0.016 * plume + 0.006 * broad + 0.002 * wave,
                0.024 + 0.010 * plume + 0.004 * secondary + 0.0015 * wave,
                0.026 + 0.008 * plume + 0.006 * secondary - 0.0010 * wave,
                0.028 + 0.006 * plume + 0.003 * negative + 0.0010 * wave,
            ],
            dim=0,
        )
        lat_gradient = yy.expand(1, height, width)
        correction = torch.cat(
            [
                0.0070
                + 0.0045 * broad
                + 0.0040 * plume
                + 0.0020 * secondary
                - 0.0030 * negative
                + 0.0020 * lat_gradient,
                -0.0015
                + 0.0040 * plume
                + 0.0030 * secondary
                - 0.0020 * negative
                - 0.0015 * lat_gradient,
                0.0005
                + 0.0025 * broad
                - 0.0020 * plume
                + 0.0040 * secondary
                + 0.0010 * lat_gradient,
                0.0010
                + 0.0015 * broad
                + 0.0015 * plume
                - 0.0040 * negative
                - 0.0020 * lat_gradient,
            ],
            dim=0,
        )
        rollouts.append(rollout)
        corrections.append(correction)
        # Hold lead fixed so conditioning shuffle cannot exploit lead as sample ID.
        lead = 24.0
        initialization = base_time + np.timedelta64(12 * sample, "h")
        valid_time = initialization + np.timedelta64(int(lead), "h")
        leads.append(lead)
        init_ids.append(np.datetime_as_string(initialization, unit="s"))
        init_times.append(initialization)
        valid_times.append(valid_time)

    rollout_tensor = torch.stack(rollouts)
    correction_tensor = torch.stack(corrections)
    case = replace(
        CASES["no2_uswest"],
        name="synthetic_no2_uswest",
        rollout_dir="synthetic://aurora",
        truth_path="synthetic://cams",
    )
    return BenchmarkDataset(
        packing=packing,
        rollout=rollout_tensor,
        target=rollout_tensor + correction_tensor,
        valid=torch.ones_like(rollout_tensor, dtype=torch.bool),
        lead_hours=torch.tensor(leads, dtype=torch.float32),
        lead_index=torch.zeros(samples, dtype=torch.long),
        lat=np.asarray(packing.lat),
        lon=np.asarray(packing.lon),
        case=case,
        initialization_ids=tuple(init_ids),
        initialization_times=np.asarray(init_times, dtype="datetime64[ns]"),
        valid_times=np.asarray(valid_times, dtype="datetime64[ns]"),
        source_validation={
            "declared_source_kind": "synthetic",
            "correction_convention": "CAMS - Aurora",
            "features": [
                "broad_regional_bias",
                "localized_positive_feature",
                "localized_negative_feature",
                "latitudinal_gradient",
                "distinct_pressure_level_patterns",
                "conditioning_dependent_moving_plume",
                "forecast_lead_held_constant_for_condition_test",
            ],
        },
    )


def _experiment_overrides() -> dict[str, Any]:
    """Small but scientifically identical Transformer for bounded diagnostics."""
    return {
        "deterministic_inference": True,
        "ensemble_size": 1,
        "target_space": {
            "residual_scaling": "per_channel",
            "residual_scaling_center": True,
            "residual_scaling_warmup_batches": 0,
        },
        "transformer": {
            "patch_size": [4, 5],
            "embedding_dim": 96,
            "num_heads": 4,
            "num_blocks": 3,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "positional_encoding": "sincos_2d",
            "attention_mode": "global_2d",
            "zero_init_output": True,
            "local_refinement": True,
        },
        "diffusion": {
            "training_timesteps": 64,
            "inference_steps": 8,
            "prediction_type": "sample",
            "schedule": "cosine",
            "sampler": "ddim",
            "eta": 0.0,
            "snr_weighting": "none",
            "timestep_distribution": "high_noise",
            "timestep_bias": 5.0,
            "deterministic_estimator": "posterior_mean",
            "deterministic_training_steps": 1,
        },
        "loss": {
            "generative": "huber",
            "deterministic_weight": 2.0,
            "bias_weight": 0.25,
            "mae_weight": 0.25,
            "gradient_weight": 0.05,
            "pattern_correlation_weight": 0.0,
            "variance_weight": 0.0,
            "extreme_weight": 0.0,
            "quantile_weight": 0.0,
            "aux_on_deterministic": True,
        },
    }


@dataclass
class TinyFitResult:
    head: str
    refiner: torch.nn.Module
    config: Any
    predicted_correction: torch.Tensor
    refined: torch.Tensor
    history: list[dict[str, float | None]] = field(default_factory=list)
    train_seconds: float = 0.0
    parameters: int = 0
    train_steps: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0


def _predict_correction(
    refiner: torch.nn.Module,
    config: Any,
    data: BenchmarkDataset,
    *,
    head: str,
    device: torch.device,
    conditioning_override: torch.Tensor | None = None,
) -> torch.Tensor:
    rollout = data.rollout.to(device)
    input_valid_mask = None if data.rollout_valid is None else data.rollout_valid.to(device)
    static_fields = None if data.static is None else data.static.to(device)
    lead = data.lead_hours.to(device)
    conditioning = (
        _conditioning(
            rollout,
            packing=data.packing,
            config=config,
            static_fields=static_fields,
            input_valid_mask=input_valid_mask,
        )
        if conditioning_override is None
        else conditioning_override.to(device)
    )
    refiner.eval()
    with torch.no_grad():
        predicted = refiner.deterministic_mean_correction_normalized(
            conditioning, forecast_lead_time=lead
        )
    return predicted.float().cpu()


def fit_tiny_dataset(
    head: str,
    data: BenchmarkDataset,
    *,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device | str,
    seed: int,
    overrides: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> TinyFitResult:
    """Overfit an exact tiny dataset and return the deployed correction field."""
    if head not in SUPPORTED_HEADS:
        raise ValueError(f"Unsupported head {head!r}; expected {SUPPORTED_HEADS}.")
    if not 1 <= len(data) <= 8:
        raise ValueError(f"Tiny overfit requires 1-8 samples; actual {len(data)}.")
    if train_steps < 1 or batch_size < 1:
        raise ValueError("train_steps and batch_size must be positive.")
    device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    merged = _experiment_overrides()
    if overrides:
        _deep_update(merged, overrides)
    configured_head = "diffusion_transformer" if head == "direct_regression_transformer" else head
    config = build_config(configured_head, lon_periodic=data.case.lon_periodic, overrides=merged)
    conditioning_example = _conditioning(
        data.rollout[:1],
        packing=data.packing,
        config=config,
        static_fields=None if data.static is None else data.static[:1],
        input_valid_mask=(None if data.rollout_valid is None else data.rollout_valid[:1]),
    )
    cond_channels = int(conditioning_example.shape[1])
    refiner = build_refiner(
        config,
        residual_channels=data.packing.num_channels,
        cond_channels=cond_channels,
        metadata=data.packing,
    )
    assert refiner is not None
    refiner = refiner.to(device)
    correction = torch.where(data.valid, data.target - data.rollout, torch.zeros_like(data.target))
    with torch.no_grad():
        refiner.fit_residual_scale(correction.to(device), data.valid.to(device))
        refiner.freeze_residual_scale()

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, train_steps), eta_min=learning_rate * 0.05
    )
    process_generator = torch.Generator(device=device)
    process_generator.manual_seed(seed + 17)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(seed + 29)
    area = area_weights_from_latitudes(
        data.packing.lat,
        len(data.packing.lat),
        device=device,
        dtype=torch.float32,
    )
    history: list[dict[str, float | None]] = []

    def record(step: int, loss: float | None) -> None:
        predicted = _predict_correction(refiner, config, data, head=head, device=device)
        truth = correction.float()
        mask = data.valid
        error = (predicted - truth)[mask]
        target_values = truth[mask]
        rmse = float(error.square().mean().sqrt())
        target_rms = float(target_values.square().mean().sqrt())
        row = {
            "step": float(step),
            "training_loss": None if loss is None else float(loss),
            "normalized_correction_rmse": rmse,
            "relative_correction_rmse": rmse / max(target_rms, 1e-12),
        }
        history.append(row)
        if verbose:
            print(
                f"    [{head}] step={step:4d}/{train_steps} "
                f"loss={'n/a' if loss is None else f'{loss:.6g}'} "
                f"correction_rel_rmse={row['relative_correction_rmse']:.4f}",
                flush=True,
            )

    record(0, None)
    started = time.time()
    refiner.train()
    last_loss = float("nan")
    log_every = max(1, train_steps // 10)
    n = len(data)
    for step in range(1, train_steps + 1):
        # Cover distinct tiny cases instead of drawing duplicates with replacement.
        indices = torch.randperm(n, generator=batch_generator)[: min(batch_size, n)]
        rollout = data.rollout[indices].to(device)
        target = data.target[indices].to(device)
        valid = data.valid[indices].to(device)
        lead = data.lead_hours[indices].to(device)
        lead_index = data.lead_index[indices].to(device)
        conditioning = _conditioning(
            rollout,
            packing=data.packing,
            config=config,
            static_fields=None if data.static is None else data.static[indices].to(device),
            input_valid_mask=(
                None if data.rollout_valid is None else data.rollout_valid[indices].to(device)
            ),
        )
        target_correction = torch.where(valid, target - rollout, torch.zeros_like(target))

        optimizer.zero_grad(set_to_none=True)
        if head == "direct_regression_transformer":
            scaled_target = refiner.residual_scaler.encode(target_correction)
            process_time = torch.zeros(rollout.shape[0], device=device, dtype=torch.float32)
            predicted_scaled = refiner.net(
                torch.zeros_like(scaled_target), conditioning, process_time, lead
            )
            loss = masked_loss(predicted_scaled, scaled_target, valid, "huber")
        else:
            output = refiner.compute_training_loss(
                target_correction,
                conditioning,
                forecast_lead_time=lead,
                mask=valid,
                generator=process_generator,
                lead_index=lead_index,
                area_weight=area,
                rollout_normalized=rollout,
            )
            assert output.total_loss is not None
            loss = output.total_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"{head} produced a non-finite loss at step {step}: {float(loss)}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % log_every == 0 or step == train_steps:
            record(step, last_loss)
        refiner.train()

    train_seconds = time.time() - started
    predicted_correction = _predict_correction(refiner, config, data, head=head, device=device)
    refined = NormalizedTargetSpace(data.packing).decode(data.rollout + predicted_correction)
    return TinyFitResult(
        head=head,
        refiner=refiner,
        config=config,
        predicted_correction=predicted_correction,
        refined=refined,
        history=history,
        train_seconds=train_seconds,
        parameters=sum(p.numel() for p in refiner.parameters()),
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def _masked_correlation(
    left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor
) -> float | None:
    a = left[valid].double()
    b = right[valid].double()
    if a.numel() < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt()
    return float((a * b).sum() / denom) if float(denom) > 0.0 else None


def conditioning_diagnostics(
    result: TinyFitResult,
    data: BenchmarkDataset,
    *,
    device: torch.device | str,
    seed: int,
) -> dict[str, Any]:
    """Condition shuffle and fixed/different-seed tests on the fitted head."""
    device = torch.device(device)
    original = result.predicted_correction
    rollout = data.rollout.to(device)
    input_valid_mask = None if data.rollout_valid is None else data.rollout_valid.to(device)
    static_fields = None if data.static is None else data.static.to(device)
    condition = _conditioning(
        rollout,
        packing=data.packing,
        config=result.config,
        static_fields=static_fields,
        input_valid_mask=input_valid_mask,
    )
    permutation = torch.arange(len(data) - 1, -1, -1, device=device)
    shuffled_condition = condition[permutation]
    shuffled = _predict_correction(
        result.refiner,
        result.config,
        data,
        head=result.head,
        device=device,
        conditioning_override=shuffled_condition,
    )
    truth = data.target - data.rollout
    original_error = (original - truth)[data.valid].square().mean().sqrt()
    shuffled_error = (shuffled - truth)[data.valid].square().mean().sqrt()
    sensitivity = (original - shuffled)[data.valid].square().mean().sqrt()
    out: dict[str, Any] = {
        "shuffle_permutation": permutation.cpu().tolist(),
        "prediction_change_rms_normalized": float(sensitivity),
        "prediction_change_over_true_correction_rms": float(
            sensitivity / truth[data.valid].square().mean().sqrt().clamp(min=1e-12)
        ),
        "original_correction_rmse_normalized": float(original_error),
        "shuffled_correction_rmse_normalized": float(shuffled_error),
        "shuffle_rmse_degradation_percent": float(
            100.0 * (shuffled_error - original_error) / original_error.clamp(min=1e-12)
        ),
    }

    if result.head == "diffusion_transformer":
        result.refiner.eval()
        lead = data.lead_hours.to(device)

        def sample(sample_seed: int) -> torch.Tensor:
            generator = torch.Generator(device=device)
            generator.manual_seed(sample_seed)
            with torch.no_grad():
                return result.refiner.sample_correction_normalized(
                    condition,
                    forecast_lead_time=lead,
                    generator=generator,
                    num_steps=result.config.diffusion.inference_steps,
                ).cpu()

        same_a = sample(seed + 101)
        same_b = sample(seed + 101)
        different = sample(seed + 102)
        deterministic = original
        out["fixed_seed"] = {
            "repeat_max_abs_difference": float((same_a - same_b).abs().max()),
            "repeat_bitwise_equal": bool(torch.equal(same_a, same_b)),
            "different_seed_correction_rms_difference": float(
                (same_a - different)[data.valid].square().mean().sqrt()
            ),
            "member_innovation_rms_seed_a": float(
                (same_a - deterministic)[data.valid].square().mean().sqrt()
            ),
            "member_innovation_rms_seed_b": float(
                (different - deterministic)[data.valid].square().mean().sqrt()
            ),
        }
    return out


def _metric_dict(row: Any) -> dict[str, Any]:
    return row.as_dict()


def _plot_six_panel(
    output: Path,
    *,
    data: BenchmarkDataset,
    result: TinyFitResult,
    sample_index: int,
    channel_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    space = NormalizedTargetSpace(data.packing)
    rollout_physical = space.decode(data.rollout)[sample_index, channel_index].numpy()
    target_physical = space.decode(data.target)[sample_index, channel_index].numpy()
    true_correction = target_physical - rollout_physical
    predicted_correction = result.refined[sample_index, channel_index].numpy() - rollout_physical
    refined = result.refined[sample_index, channel_index].numpy()
    error = refined - target_physical
    fields = (
        ("Aurora", rollout_physical, "viridis", None),
        ("CAMS", target_physical, "viridis", None),
        ("true correction (CAMS - Aurora)", true_correction, "RdBu_r", "symmetric"),
        ("predicted correction", predicted_correction, "RdBu_r", "symmetric"),
        ("refined = Aurora + correction", refined, "viridis", None),
        ("remaining error (refined - CAMS)", error, "RdBu_r", "symmetric"),
    )
    field_min = min(
        float(np.nanmin(rollout_physical)),
        float(np.nanmin(target_physical)),
        float(np.nanmin(refined)),
    )
    field_max = max(
        float(np.nanmax(rollout_physical)),
        float(np.nanmax(target_physical)),
        float(np.nanmax(refined)),
    )
    correction_max = max(
        float(np.nanmax(np.abs(true_correction))),
        float(np.nanmax(np.abs(predicted_correction))),
        float(np.nanmax(np.abs(error))),
        1e-30,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (title, values, cmap, scale) in zip(axes.flat, fields):
        kwargs = (
            {"vmin": -correction_max, "vmax": correction_max}
            if scale == "symmetric"
            else {"vmin": field_min, "vmax": field_max}
        )
        mesh = axis.pcolormesh(data.lon, data.lat, values, shading="auto", cmap=cmap, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        fig.colorbar(mesh, ax=axis, shrink=0.82)
    spec = data.packing.channels[channel_index]
    fig.suptitle(
        f"{result.head}: {_channel_label(spec)}; sample {sample_index}; "
        "correction convention = CAMS - Aurora"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _radial_power_spectrum(
    field: np.ndarray, valid: np.ndarray, *, bins: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Mean radial 2-D spectrum for sample-height-width maps."""
    values = np.asarray(field, dtype=np.float64).copy()
    mask = np.asarray(valid, dtype=bool)
    for sample in range(values.shape[0]):
        selected = values[sample][mask[sample]]
        fill = float(selected.mean()) if selected.size else 0.0
        values[sample][~mask[sample]] = fill
        values[sample] -= values[sample].mean()
    power = np.abs(np.fft.rfft2(values, norm="ortho")) ** 2
    mean_power = power.mean(axis=0)
    height, width = values.shape[-2:]
    fy = np.abs(np.fft.fftfreq(height))[:, None]
    fx = np.abs(np.fft.rfftfreq(width))[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    edges = np.linspace(0.0, float(radius.max()) + 1e-12, bins + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    numerator = np.bincount(index, weights=mean_power.ravel(), minlength=bins)
    denominator = np.bincount(index, minlength=bins)
    radial = numerator / np.maximum(denominator, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = (centers > 0.0) & (radial > 0.0)
    return centers[keep], radial[keep]


def _plot_distribution_and_spectrum(
    output: Path, *, data: BenchmarkDataset, result: TinyFitResult
) -> None:
    """Plot correction histograms, scatter, and radial spectra for every channel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    space = NormalizedTargetSpace(data.packing)
    rollout = space.decode(data.rollout)
    target = space.decode(data.target)
    truth = (target - rollout).numpy()
    predicted = (result.refined - rollout).numpy()
    valid = data.valid.numpy()
    rows = data.packing.num_channels
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(14, max(3.0 * rows, 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for spec in data.packing.channels:
        row = spec.index
        mask = valid[:, row]
        true_flat = truth[:, row][mask]
        predicted_flat = predicted[:, row][mask]
        error_flat = predicted_flat - true_flat
        axes[row, 0].hist(
            true_flat,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.5,
            label="true",
        )
        axes[row, 0].hist(
            predicted_flat,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.5,
            label="predicted",
        )
        axes[row, 0].hist(
            error_flat,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.0,
            label="error",
        )
        axes[row, 0].set_title(f"{_channel_label(spec)} correction PDF")
        axes[row, 0].legend(fontsize=8)

        count = min(5000, true_flat.size)
        select = np.linspace(0, true_flat.size - 1, count, dtype=int)
        axes[row, 1].scatter(
            true_flat[select],
            predicted_flat[select],
            s=3,
            alpha=0.25,
            rasterized=True,
        )
        low = float(min(true_flat.min(), predicted_flat.min()))
        high = float(max(true_flat.max(), predicted_flat.max()))
        axes[row, 1].plot([low, high], [low, high], color="black", linewidth=1)
        axes[row, 1].set_xlabel("true correction")
        axes[row, 1].set_ylabel("predicted correction")
        axes[row, 1].set_title("predicted vs true")

        true_k, true_power = _radial_power_spectrum(truth[:, row], mask)
        pred_k, pred_power = _radial_power_spectrum(predicted[:, row], mask)
        axes[row, 2].loglog(true_k, true_power, label="true", linewidth=1.5)
        axes[row, 2].loglog(pred_k, pred_power, label="predicted", linewidth=1.5)
        axes[row, 2].set_xlabel("normalized spatial wavenumber")
        axes[row, 2].set_ylabel("power")
        axes[row, 2].set_title("radial correction spectrum")
        axes[row, 2].legend(fontsize=8)
        axes[row, 0].ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))
        axes[row, 1].ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))
    fig.suptitle(f"{result.head}: correction distributions and spatial power")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_heldout_eight_panel(
    output: Path,
    *,
    data: BenchmarkDataset,
    head: str,
    refined_physical: torch.Tensor,
    sample_index: int,
    channel_index: int,
) -> None:
    """Save the complete physical-field/correction contract for one held-out case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    space = NormalizedTargetSpace(data.packing)
    aurora = space.decode(data.rollout)[sample_index, channel_index].numpy()
    cams = space.decode(data.target)[sample_index, channel_index].numpy()
    refined = refined_physical[sample_index, channel_index].numpy()
    true_correction = cams - aurora
    predicted_correction = refined - aurora
    baseline_error = aurora - cams
    refined_error = refined - cams
    correction_error = predicted_correction - true_correction
    panels = (
        ("CAMS", cams, "viridis", False),
        ("Aurora", aurora, "viridis", False),
        ("refined", refined, "viridis", False),
        ("Aurora - CAMS", baseline_error, "RdBu_r", True),
        ("refined - CAMS", refined_error, "RdBu_r", True),
        ("true correction", true_correction, "RdBu_r", True),
        ("predicted correction", predicted_correction, "RdBu_r", True),
        ("correction error", correction_error, "RdBu_r", True),
    )
    field_min = min(float(np.nanmin(aurora)), float(np.nanmin(cams)), float(np.nanmin(refined)))
    field_max = max(float(np.nanmax(aurora)), float(np.nanmax(cams)), float(np.nanmax(refined)))
    difference_max = max(
        *(float(np.nanmax(np.abs(values))) for _, values, _, diverging in panels if diverging),
        1e-30,
    )
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for axis, (title, values, cmap, diverging) in zip(axes.flat, panels):
        limits = (
            {"vmin": -difference_max, "vmax": difference_max}
            if diverging
            else {"vmin": field_min, "vmax": field_max}
        )
        mesh = axis.pcolormesh(data.lon, data.lat, values, shading="auto", cmap=cmap, **limits)
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        fig.colorbar(mesh, ax=axis, shrink=0.78)
    spec = data.packing.channels[channel_index]
    fig.suptitle(
        f"{head}: held-out {_channel_label(spec)}; "
        "correction = CAMS - Aurora; refined = Aurora + correction"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_heldout_diagnostic_artifacts(
    output_dir: str | Path,
    *,
    data: BenchmarkDataset,
    head: str,
    refined_physical: torch.Tensor,
) -> dict[str, Any]:
    """Save optional held-out maps plus aggregate distribution/scale diagnostics."""
    refined = refined_physical.detach().float().cpu()
    expected = tuple(data.target.shape)
    if tuple(refined.shape) != expected:
        raise ValueError(
            f"Held-out refined field expected shape {expected}, got {tuple(refined.shape)}."
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    maps: list[str] = []
    for spec in data.packing.channels:
        path = root / f"{_safe_label(_channel_label(spec))}_heldout_eight_panel.png"
        _plot_heldout_eight_panel(
            path,
            data=data,
            head=head,
            refined_physical=refined,
            sample_index=0,
            channel_index=spec.index,
        )
        maps.append(str(path.resolve()))
    distribution = root / "heldout_distribution_scatter_spectrum.png"
    view = SimpleNamespace(head=head, refined=refined)
    _plot_distribution_and_spectrum(distribution, data=data, result=view)
    return {
        "representative_sample_index": 0,
        "eight_panel_maps": maps,
        "distribution_scatter_spectrum": str(distribution.resolve()),
    }


def summarize_fit(
    result: TinyFitResult,
    data: BenchmarkDataset,
    *,
    output_dir: Path,
    experiment: str,
    device: torch.device | str,
    seed: int,
) -> dict[str, Any]:
    space = NormalizedTargetSpace(data.packing)
    rollout_physical = space.decode(data.rollout)
    target_physical = space.decode(data.target)
    predicted_physical = result.refined - rollout_physical
    true_physical = target_physical - rollout_physical
    area = area_weights_from_latitudes(data.packing.lat, len(data.packing.lat), dtype=torch.float64)
    baseline = evaluate_channels(
        rollout_physical,
        target_physical,
        data.valid,
        data.packing,
        area_weight=area,
        label="Aurora",
    )
    refined = evaluate_channels(
        result.refined,
        target_physical,
        data.valid,
        data.packing,
        area_weight=area,
        label=result.head,
    )
    per_channel: dict[str, Any] = {}
    map_paths: list[str] = []
    for spec in data.packing.channels:
        label = _channel_label(spec)
        mask = data.valid[:, spec.index : spec.index + 1]
        pred = predicted_physical[:, spec.index : spec.index + 1]
        truth = true_physical[:, spec.index : spec.index + 1]
        error = (pred - truth)[mask]
        target_values = truth[mask]
        map_path = output_dir / f"{experiment}_{result.head}_{_safe_label(label)}_six_panel.png"
        _plot_six_panel(
            map_path,
            data=data,
            result=result,
            sample_index=0,
            channel_index=spec.index,
        )
        map_paths.append(str(map_path.resolve()))
        base_row = baseline[label]
        refined_row = refined[label]
        per_channel[label] = {
            "baseline": _metric_dict(base_row),
            "refined": _metric_dict(refined_row),
            "mae_improvement_percent": float(
                100.0 * (base_row.mae - refined_row.mae) / max(base_row.mae, 1e-30)
            ),
            "rmse_improvement_percent": float(
                100.0 * (base_row.rmse - refined_row.rmse) / max(base_row.rmse, 1e-30)
            ),
            "correction_rmse_physical": float(error.square().mean().sqrt()),
            "relative_correction_rmse": float(
                error.square().mean().sqrt() / target_values.square().mean().sqrt().clamp(min=1e-30)
            ),
            "correction_correlation": _masked_correlation(pred, truth, mask),
            "true_correction_texture": spatial_texture_statistics(truth),
            "predicted_correction_texture": spatial_texture_statistics(pred),
        }

    distribution_path = output_dir / f"{experiment}_{result.head}_distribution_scatter_spectrum.png"
    _plot_distribution_and_spectrum(distribution_path, data=data, result=result)

    report = {
        "experiment": experiment,
        "head": result.head,
        "training_objective": (
            "direct_supervised_scaled_correction_huber"
            if result.head == "direct_regression_transformer"
            else "diffusion_innovation_plus_direct_supervised_conditional_mean"
        ),
        "evaluation_product": "Aurora plus deterministic conditional-mean correction",
        "seed": int(seed),
        "case": {
            "name": data.case.name,
            "rollout_dir": data.case.rollout_dir,
            "rollout_glob": data.case.rollout_glob,
            "truth_path": data.case.truth_path,
            "source_kind": data.case.source_kind,
            "source_files": list(data.source_files),
            "source_validation": dict(data.source_validation),
        },
        "split": {
            "method": "intentional_tiny_overfit_same_cases_for_fit_and_reconstruction",
            "training_sample_indices": list(range(len(data))),
            "evaluation_sample_indices": list(range(len(data))),
            "held_out": False,
        },
        "correction_convention": (
            "correction_target = CAMS - Aurora; " "refined = Aurora + predicted_correction"
        ),
        "samples": len(data),
        "channel_order": [_channel_label(spec) for spec in data.packing.channels],
        "forecast_lead_hours": [float(value) for value in data.lead_hours],
        "initialization_times": [str(value) for value in data.initialization_times],
        "valid_times": [str(value) for value in data.valid_times],
        "latitude_order": "descending" if data.lat[0] > data.lat[-1] else "ascending",
        "longitude_order": "ascending" if data.lon[0] < data.lon[-1] else "descending",
        "pressure_level_order": [
            None if spec.level is None else float(spec.level) for spec in data.packing.channels
        ],
        "parameters": result.parameters,
        "train_seconds": result.train_seconds,
        "optimization": {
            "train_steps": result.train_steps,
            "batch_size": result.batch_size,
            "learning_rate": result.learning_rate,
        },
        "resolved_config": result.config.to_dict(),
        "training_history": result.history,
        "training_correction_statistics": correction_statistics(data),
        "tensor_statistics": {
            "aurora_normalized": tensor_statistics(data.rollout),
            "cams_normalized": tensor_statistics(data.target),
            "true_correction_normalized": tensor_statistics(data.target - data.rollout),
            "predicted_correction_normalized": tensor_statistics(result.predicted_correction),
            "aurora_physical": tensor_statistics(rollout_physical),
            "cams_physical": tensor_statistics(target_physical),
            "true_correction_physical": tensor_statistics(true_physical),
            "predicted_correction_physical": tensor_statistics(predicted_physical),
            "refined_physical": tensor_statistics(result.refined),
            "remaining_error_physical": tensor_statistics(result.refined - target_physical),
        },
        "conditioning": conditioning_diagnostics(result, data, device=device, seed=seed),
        "channels": per_channel,
        "maps": map_paths,
        "distribution_scatter_spectrum": str(distribution_path.resolve()),
    }
    checkpoint_path = output_dir / f"{experiment}_{result.head}_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": experiment,
            "head": result.head,
            "seed": int(seed),
            "correction_convention": "CAMS - Aurora",
            "resolved_config": result.config.to_dict(),
            "channel_order": report["channel_order"],
            "state_dict": {
                name: value.detach().cpu() for name, value in result.refiner.state_dict().items()
            },
        },
        checkpoint_path,
    )
    report["checkpoint_path"] = str(checkpoint_path.resolve())

    report_path = output_dir / f"{experiment}_{result.head}_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    report["report_path"] = str(report_path.resolve())
    return report


def _real_tiny_dataset(samples: int, *, verbose: bool) -> BenchmarkDataset:
    if not 1 <= samples <= 8:
        raise ValueError(f"real_samples must be in [1, 8], got {samples}.")
    # A raw rollout contains six configured leads; two initializations cover the
    # maximum requested sample count while retaining exact load_case assertions.
    full = load_case(
        CASES["no2_uswest"],
        max_initializations=max(1, math.ceil(samples / 6)),
        verbose=verbose,
    )
    return full.subset(range(samples))


def run_controlled_experiments(
    *,
    mode: str,
    heads: Sequence[str],
    train_steps: int,
    synthetic_samples: int,
    real_samples: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    output_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    if mode not in {"synthetic", "real", "both"}:
        raise ValueError(f"mode must be synthetic, real, or both; actual {mode!r}.")
    unknown = sorted(set(heads) - set(SUPPORTED_HEADS))
    if unknown:
        raise ValueError(f"Unsupported heads {unknown}; expected {SUPPORTED_HEADS}.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiments: dict[str, BenchmarkDataset] = {}
    if mode in {"synthetic", "both"}:
        experiments["synthetic_smooth"] = make_smooth_synthetic_dataset(samples=synthetic_samples)
    if mode in {"real", "both"}:
        experiments["real_no2_tiny_overfit"] = _real_tiny_dataset(real_samples, verbose=verbose)

    report: dict[str, Any] = {
        "purpose": "controlled residual-refinement reconstruction diagnostics",
        "correction_convention": "CAMS - Aurora, added exactly once",
        "device": device,
        "seed": seed,
        "train_steps": train_steps,
        "experiments": {},
    }
    for experiment, data in experiments.items():
        if verbose:
            print(
                f"\n=== {experiment}: samples={len(data)} shape={tuple(data.rollout.shape)} ===",
                flush=True,
            )
        report["experiments"][experiment] = {}
        for head_index, head in enumerate(heads):
            if verbose:
                print(f"  fitting {head} ...", flush=True)
            result = fit_tiny_dataset(
                head,
                data,
                train_steps=train_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                seed=seed + 1000 * head_index,
                verbose=verbose,
            )
            report["experiments"][experiment][head] = summarize_fit(
                result,
                data,
                output_dir=root,
                experiment=experiment,
                device=device,
                seed=seed + 1000 * head_index,
            )
    summary_path = root / "controlled_experiments_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    report["summary_path"] = str(summary_path.resolve())
    if verbose:
        print(f"\nWrote {summary_path.resolve()}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real", "both"), default="both")
    parser.add_argument(
        "--heads", nargs="+", choices=SUPPORTED_HEADS, default=list(SUPPORTED_HEADS)
    )
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--synthetic-samples", type=int, default=6)
    parser.add_argument("--real-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        default="finetune/outputs/refinement_diagnostics/controlled",
    )
    args = parser.parse_args(argv)
    run_controlled_experiments(
        mode=args.mode,
        heads=args.heads,
        train_steps=args.train_steps,
        synthetic_samples=args.synthetic_samples,
        real_samples=args.real_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
