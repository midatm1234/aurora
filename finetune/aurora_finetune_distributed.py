"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Distributed Aurora fine-tuning across multiple GPUs.

Automatically detects idle GPUs (≥90 % free) and distributes training across
them.  Gradients are all-reduced manually after each backward pass — no DDP
wrapper needed, which avoids all compatibility issues with Aurora's custom
``Batch`` dataclass.

The key OOM fix: model parameters are stored in **bfloat16** instead of float32.
This halves param, gradient *and* AdamW-state memory (fp32 ~20 GB → bf16 ~7.5 GB
per GPU), well within the 22 GB A10G budget.

Usage
-----
    python finetune/aurora_finetune_distributed.py               # auto-detect GPUs
    python finetune/aurora_finetune_distributed.py --gpus 5,6,7  # explicit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# This workload is launch-bound on a small spatial grid: thousands of tiny CUDA
# kernels per step mean the bottleneck is CPU-side kernel dispatch, not GPU math.
# Letting OpenMP/torch spawn one thread per core (48 on this box) oversubscribes
# the CPU and slows dispatch. Cap intra-op threads to keep dispatch fast.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
try:
    torch.set_num_threads(8)
except Exception:
    pass


def _progress_bars_disabled() -> bool:
    value = (
        os.environ.get("AURORA_DISABLE_PROGRESS_BARS")
        or os.environ.get("TQDM_DISABLE")
        or ""
    )
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import finetune.aurora_finetune_utils as ft  # noqa: E402

FREE_MEMORY_FRACTION = 0.90


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------

def _detect_idle_gpus(min_free_fraction: float = FREE_MEMORY_FRACTION) -> list[int]:
    """Return indices of GPUs whose free memory exceeds *min_free_fraction*."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    idle: list[int] = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 3:
            continue
        idx, free, total = int(parts[0]), float(parts[1]), float(parts[2])
        if total > 0 and free / total >= min_free_fraction:
            idle.append(idx)
    return idle


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def _print0(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


def _allreduce_grads(model: torch.nn.Module, world_size: int):
    """Average gradients across all ranks.

    Uses CPU float32 tensors for gloo backend compatibility, then copies back.
    """
    for param in model.parameters():
        if param.grad is not None:
            grad_cpu = param.grad.data.float().cpu()
            dist.all_reduce(grad_cpu, op=dist.ReduceOp.SUM)
            param.grad.data.copy_((grad_cpu / world_size).to(param.grad.dtype))


def _active_residual_scalers(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Return every active unified-refinement scaler in stable module order."""
    from finetune.refinement.residual_scaling import ResidualScaler

    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ResidualScaler) and module.is_active
    ]


def _disjoint_rank_samples(samples: list[dict], rank: int, world_size: int) -> list[dict]:
    """Shard without padding, duplication, or overlap for calibration."""
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid calibration rank/world_size: rank={rank}, world_size={world_size}."
        )
    return samples[rank::world_size]


def _prepare_exact_calibration_forward(
    model: torch.nn.Module,
    scalers: list[tuple[str, torch.nn.Module]],
) -> None:
    """Collect training targets with deterministic Aurora/network behavior."""
    # `compute_supervised_loss` keys target collection off the outer unified
    # wrapper's training flag. Set flags directly (rather than recursively
    # calling eval()) so every stochastic/dropout/BatchNorm submodule is in eval
    # while the wrapper and scaler still execute their training-only target path.
    model.train()
    for module in model.modules():
        module.training = False
    model.training = True
    inner = model.module if hasattr(model, "module") else model
    inner.training = True
    for _, scaler in scalers:
        scaler.training = True


def _training_split_calibration_fingerprint(
    ds,
    train_samples: list[dict],
    cfg: dict,
    packing,
) -> bytes:
    """Hash the exact logical split, coordinates, and packed channel contract."""
    data_cfg = cfg.get("data", {})
    coordinate_names = (
        str(data_cfg.get("time_dim", "time")),
        str(data_cfg.get("lat_dim", "latitude")),
        str(data_cfg.get("lon_dim", "longitude")),
        str(data_cfg.get("level_dim", "level")),
    )
    coordinates = {}
    for name in coordinate_names:
        if name not in ds.coords:
            continue
        values = np.asarray(ds.coords[name].values)
        coordinates[name] = {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "values": values.tolist(),
        }
    payload = {
        "version": 1,
        "correction_convention": "CAMS_minus_Aurora",
        "train_data_path": str(
            Path(cfg.get("paths", {}).get("train_data_path", "")).resolve()
        ),
        "dataset_sizes": {str(k): int(v) for k, v in ds.sizes.items()},
        "coordinates": coordinates,
        "samples": train_samples,
        "target_lead_times": [
            int(value) for value in data_cfg.get("target_lead_times", ())
        ],
        "rollout_step_hours": float(
            cfg.get("rollout", {}).get("rollout_step_hours", 0.0)
        ),
        "packing": packing.to_dict(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _residual_calibration_report(
    *,
    scalers: list[tuple[str, torch.nn.Module]],
    packing,
    fingerprint: bytes,
    logical_samples: int,
    packed_examples: int,
) -> dict:
    """Build checkpoint/JSON provenance plus physical per-channel statistics."""
    scaler_reports = []
    for module_name, scaler in scalers:
        count, normalized_mean, normalized_std = scaler.raw_statistics()
        count_values = count.reshape(-1).cpu().tolist()
        mean_values = normalized_mean.reshape(-1).cpu().tolist()
        std_values = normalized_std.reshape(-1).cpu().tolist()
        scale_values = scaler.scale.detach().double().reshape(-1).cpu().tolist()
        shift_values = scaler.shift.detach().double().reshape(-1).cpu().tolist()
        channels = []
        for channel, n, mean, std, scale, shift in zip(
            packing.channels,
            count_values,
            mean_values,
            std_values,
            scale_values,
            shift_values,
        ):
            physical_scale = float(channel.std)
            channels.append(
                {
                    "index": int(channel.index),
                    "variable": str(channel.dataset_name),
                    "aurora_variable": str(channel.aurora_name),
                    "level_hpa": (
                        None if channel.level is None else float(channel.level)
                    ),
                    "units": str(channel.units),
                    "valid_cell_count": int(n),
                    "normalized_correction_mean": float(mean),
                    "normalized_correction_std": float(std),
                    "physical_correction_mean": float(mean) * physical_scale,
                    "physical_correction_std": float(std) * physical_scale,
                    "normalizer_scale": float(scale),
                    "normalizer_shift": float(shift),
                    "physical_normalizer_scale": float(scale) * physical_scale,
                    "physical_normalizer_shift": float(shift) * physical_scale,
                }
            )
        scaler_reports.append(
            {
                "module": module_name,
                "mode": str(scaler.mode),
                "center": bool(scaler.center),
                "complete": bool(scaler.calibration_complete.item()),
                "frozen": bool(scaler.frozen.item()),
                "method": "exact_training_split",
                "logical_samples": int(scaler.calibration_logical_samples.item()),
                "packed_examples": int(scaler.calibration_examples.item()),
                "channels": channels,
            }
        )
    return {
        "version": 1,
        "correction_convention": "CAMS_minus_Aurora",
        "method": "exact_training_split",
        "complete": True,
        "training_split_fingerprint_sha256": fingerprint.hex(),
        "logical_samples": int(logical_samples),
        "packed_examples": int(packed_examples),
        "lead_times_hours": [float(value) for value in packing.lead_times_hours],
        "scalers": scaler_reports,
    }


def _calibrate_residual_scalers_from_training_split(
    *,
    model: torch.nn.Module,
    train_ds,
    train_samples: list[dict],
    cfg: dict,
    resolved_specs,
    device: torch.device,
    rank: int,
    world_size: int,
    norm_stats,
    global_step: int,
) -> dict | None:
    """Fit exact train-only correction statistics before any optimiser step.

    Ranks consume disjoint, unpadded sample shards. No collective occurs in the
    forward loop; each scaler performs exactly one final sufficient-statistic
    all-reduce after all local examples have been observed.
    """
    scalers = _active_residual_scalers(model)
    if not scalers:
        return None
    inner = model.module if hasattr(model, "module") else model
    packing = getattr(inner, "packing", None)
    if packing is None:
        raise RuntimeError(
            "An active ResidualScaler requires canonical field-packing metadata."
        )
    logical_samples = len(train_samples)
    if logical_samples <= 0:
        raise RuntimeError("Residual calibration requires a non-empty training split.")
    configured_leads = [
        int(value) for value in cfg.get("data", {}).get("target_lead_times", ())
    ]
    if not configured_leads:
        raise RuntimeError(
            "Residual calibration requires configured data.target_lead_times."
        )
    packed_examples = logical_samples * len(configured_leads)
    fingerprint = _training_split_calibration_fingerprint(
        train_ds, train_samples, cfg, packing
    )

    if int(global_step) > 0:
        for module_name, scaler in scalers:
            try:
                scaler.validate_exact_training_split_calibration(
                    logical_samples=logical_samples,
                    packed_examples=packed_examples,
                    fingerprint=fingerprint,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Cannot resume trained stochastic-refinement weights with an "
                    "unverified or incompatible residual normalizer at module "
                    f"{module_name!r}. Recalibrating it would change the learned "
                    "coordinate system. Restart Phase-2 training from step zero "
                    "with the current training split."
                ) from exc
        return _residual_calibration_report(
            scalers=scalers,
            packing=packing,
            fingerprint=fingerprint,
            logical_samples=logical_samples,
            packed_examples=packed_examples,
        )

    _prepare_exact_calibration_forward(model, scalers)
    for _, scaler in scalers:
        scaler.begin_exact_training_split_calibration()
    local_samples = _disjoint_rank_samples(train_samples, rank, world_size)
    batch_size = max(1, int(cfg.get("training", {}).get("batch_size", 1)))
    try:
        with torch.inference_mode():
            for offset in range(0, len(local_samples), batch_size):
                sample_batch = local_samples[offset : offset + batch_size]
                loss, _ = ft.compute_supervised_loss(
                    model=model,
                    ds=train_ds,
                    samples=sample_batch,
                    config=cfg,
                    resolved_specs=resolved_specs,
                    device=device,
                    norm_stats=norm_stats,
                )
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise RuntimeError(
                        "Residual calibration forward produced a non-finite loss."
                    )
        for _, scaler in scalers:
            scaler.finalize_exact_training_split_calibration(
                logical_samples=logical_samples,
                packed_examples=packed_examples,
                fingerprint=fingerprint,
                distributed=world_size > 1,
            )
    except Exception:
        for _, scaler in scalers:
            scaler.abort_exact_training_split_calibration()
        raise

    return _residual_calibration_report(
        scalers=scalers,
        packing=packing,
        fingerprint=fingerprint,
        logical_samples=logical_samples,
        packed_examples=packed_examples,
    )



def _optimizer_updates_per_epoch(
    num_samples: int,
    *,
    world_size: int,
    batch_size: int,
    accumulation_steps: int,
) -> int:
    """Return scheduler steps per epoch after sharding and accumulation."""
    samples_per_rank = max(1, int(math.ceil(num_samples / max(1, world_size))))
    batches_per_rank = max(
        1, int(math.ceil(samples_per_rank / max(1, batch_size))),
    )
    return max(
        1, int(math.ceil(batches_per_rank / max(1, accumulation_steps))),
    )


_PHYSICAL_VALIDATION_STAT_NAMES = (
    "count",
    "error_sum",
    "abs_error_sum",
    "sq_error_sum",
    "spatial_correlation_sum",
    "spatial_correlation_count",
)


def _checkpoint_candidate_improves(
    *,
    should_validate: bool,
    checkpoint_metric_value: float,
    non_degrading: bool,
    physical_channel_coverage_complete: bool,
    best_value: float,
    min_delta: float,
    checkpoint_guards_passed: bool = True,
) -> bool:
    """Reject checkpoint candidates unless coverage and all safety gates pass."""
    return bool(
        should_validate
        and physical_channel_coverage_complete
        and np.isfinite(checkpoint_metric_value)
        and non_degrading
        and checkpoint_guards_passed
        and checkpoint_metric_value < (best_value - min_delta)
    )


def _merge_physical_validation_sums(target, batch_metrics) -> None:
    """Merge one validation batch, accepting legacy error-only accumulators."""
    for channel, values in batch_metrics.get("physical_error_sums", {}).items():
        slot = target.setdefault(
            channel,
            {name: 0.0 for name in _PHYSICAL_VALIDATION_STAT_NAMES},
        )
        for name in _PHYSICAL_VALIDATION_STAT_NAMES:
            slot[name] = float(slot.get(name, 0.0)) + float(
                values.get(name, 0.0)
            )


def _finalize_physical_validation_channel(sums) -> dict[str, float]:
    """Convert additive cell errors and sample correlations to channel metrics."""
    count = float(sums.get("count", 0.0))
    correlation_count = float(sums.get("spatial_correlation_count", 0.0))
    if count <= 0:
        return {
            "count": 0.0,
            "bias": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "pattern_correlation_count": correlation_count,
            "pattern_correlation": math.nan,
        }
    correlation = (
        float(sums.get("spatial_correlation_sum", 0.0)) / correlation_count
        if correlation_count > 0
        else math.nan
    )
    return {
        "count": count,
        "bias": float(sums.get("error_sum", 0.0)) / count,
        "mae": float(sums.get("abs_error_sum", 0.0)) / count,
        "rmse": math.sqrt(max(0.0, float(sums.get("sq_error_sum", 0.0)) / count)),
        "pattern_correlation_count": correlation_count,
        "pattern_correlation": correlation,
    }


def _evaluate_checkpoint_guards(
    *,
    physical_channels,
    expected_channels,
    metrics,
    relative_tolerance: float,
    correlation_tolerance: float,
    bias_rmse_floor_fraction: float,
) -> dict[str, object]:
    """Evaluate additive scientific guards for every expected physical channel."""
    configured_metrics = [str(metric).strip().lower() for metric in metrics]
    allowed_metrics = {"mae", "absolute_bias", "pattern_correlation"}
    unknown = sorted(set(configured_metrics) - allowed_metrics)
    if unknown:
        raise ValueError(f"Unknown checkpoint guard metrics: {unknown}.")
    report: dict[str, object] = {
        "status": "disabled" if not configured_metrics else "passed",
        "passed": True,
        "configured_metrics": configured_metrics,
        "relative_tolerance": float(relative_tolerance),
        "correlation_tolerance": float(correlation_tolerance),
        "bias_rmse_floor_fraction": float(bias_rmse_floor_fraction),
        "channels": {},
        "failures": [],
        "failed_channels": [],
    }
    if not configured_metrics:
        return report

    channel_details: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    for channel in sorted(set(expected_channels)):
        comparison = physical_channels.get(channel, {})
        baseline = comparison.get("baseline", {})
        candidate = comparison.get("refined", {})
        metric_details: dict[str, object] = {}
        for metric in configured_metrics:
            detail: dict[str, object]
            if metric == "mae":
                baseline_value = float(baseline.get("mae", math.nan))
                candidate_value = float(candidate.get("mae", math.nan))
                threshold = baseline_value * (1.0 + relative_tolerance)
                comparison_name = "candidate <= baseline * (1 + tolerance)"
                finite = all(
                    np.isfinite(value)
                    for value in (baseline_value, candidate_value, threshold)
                )
                passed = bool(finite and candidate_value <= threshold)
                detail = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "threshold": threshold,
                    "comparison": comparison_name,
                    "passed": passed,
                }
            elif metric == "absolute_bias":
                baseline_value = abs(float(baseline.get("bias", math.nan)))
                candidate_value = abs(float(candidate.get("bias", math.nan)))
                baseline_rmse = float(baseline.get("rmse", math.nan))
                relative_threshold = baseline_value * (1.0 + relative_tolerance)
                rmse_floor_threshold = (
                    bias_rmse_floor_fraction * baseline_rmse
                )
                threshold = max(relative_threshold, rmse_floor_threshold)
                comparison_name = (
                    "abs(candidate bias) <= max(abs(baseline bias) * "
                    "(1 + tolerance), floor * baseline RMSE)"
                )
                finite = all(
                    np.isfinite(value)
                    for value in (
                        baseline_value,
                        candidate_value,
                        baseline_rmse,
                        threshold,
                    )
                )
                passed = bool(finite and candidate_value <= threshold)
                detail = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "baseline_rmse": baseline_rmse,
                    "relative_threshold": relative_threshold,
                    "rmse_floor_threshold": rmse_floor_threshold,
                    "threshold": threshold,
                    "comparison": comparison_name,
                    "passed": passed,
                }
            else:
                baseline_value = float(
                    baseline.get("pattern_correlation", math.nan)
                )
                candidate_value = float(
                    candidate.get("pattern_correlation", math.nan)
                )
                threshold = baseline_value - correlation_tolerance
                comparison_name = (
                    "candidate >= baseline - correlation tolerance"
                )
                finite = all(
                    np.isfinite(value)
                    for value in (baseline_value, candidate_value, threshold)
                )
                passed = bool(finite and candidate_value >= threshold)
                detail = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "threshold": threshold,
                    "comparison": comparison_name,
                    "passed": passed,
                }
            metric_details[metric] = detail
            if not passed:
                failures.append(
                    {
                        "channel": channel,
                        "metric": metric,
                        **detail,
                        "reason": (
                            "threshold_failed" if finite else "non_finite_metric"
                        ),
                    }
                )
        channel_details[channel] = metric_details

    failed_channels = sorted({str(item["channel"]) for item in failures})
    report.update(
        {
            "status": "failed" if failures else "passed",
            "passed": not failures,
            "channels": channel_details,
            "failures": failures,
            "failed_channels": failed_channels,
        }
    )
    return report


# ---------------------------------------------------------------------------
# Distributed validation
# ---------------------------------------------------------------------------

def _validation_baseline_model(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the deterministic Aurora model used for baseline validation."""
    inner = model.module if hasattr(model, "module") else model
    from finetune.refinement.two_phase import AuroraTwoPhaseRefiner

    if isinstance(inner, AuroraTwoPhaseRefiner):
        return inner.aurora
    return getattr(inner, "base", None)

def _expected_physical_validation_channels(
    model: torch.nn.Module,
    cfg: dict,
    resolved_specs,
) -> tuple[str, ...]:
    """Derive the exact variable×level×configured-lead metric key contract."""
    inner = model.module if hasattr(model, "module") else model
    packing = getattr(inner, "packing", None)
    if packing is not None:
        channels = list(packing.channels)
        lead_hours = [float(value) for value in packing.lead_times_hours]
    else:
        channels = []
        full_levels = [
            float(value)
            for value in cfg.get("data", {}).get(
                "atmos_levels",
                cfg.get("data", {}).get("pressure_levels", ()),
            )
        ]
        for spec in resolved_specs.targets:
            if spec.kind == "surf":
                channels.append(spec)
            else:
                levels = list(spec.loss_levels or full_levels)
                for level in levels:
                    channels.append((spec, float(level)))
        step_hours = float(cfg.get("rollout", {}).get("rollout_step_hours", 0.0))
        lead_hours = [
            int(value) * step_hours
            for value in cfg.get("data", {}).get("target_lead_times", ())
        ]
    if not lead_hours or any(not np.isfinite(value) or value <= 0 for value in lead_hours):
        raise ValueError(
            "Physical validation coverage requires positive configured lead hours."
        )

    names = []
    for channel in channels:
        if isinstance(channel, tuple):
            spec, level = channel
            prefix = f"{spec.dataset_name}@{float(level):g}hPa"
        elif getattr(channel, "kind", None) == "surf":
            prefix = f"{channel.dataset_name}@surface"
        else:
            prefix = f"{channel.dataset_name}@{float(channel.level):g}hPa"
        names.extend(f"{prefix}@lead{float(hours):g}h" for hours in lead_hours)
    if not names or len(set(names)) != len(names):
        raise RuntimeError(
            "Physical validation channel contract is empty or contains duplicates."
        )
    return tuple(sorted(names))


def _summarize_physical_channel_coverage(
    *,
    expected_channels,
    observed_refined,
    observed_baseline,
    physical_channels,
) -> dict[str, object]:
    """Report missing, extra, and zero-count validation groups exactly."""
    expected = set(expected_channels)
    refined = set(observed_refined)
    baseline = set(observed_baseline)
    missing_refined = sorted(expected - refined)
    missing_baseline = sorted(expected - baseline)
    extra_refined = sorted(refined - expected)
    extra_baseline = sorted(baseline - expected)
    zero_count_refined = sorted(
        name
        for name in expected
        if name not in physical_channels
        or float(physical_channels[name]["refined"]["count"]) <= 0
    )
    zero_count_baseline = sorted(
        name
        for name in expected
        if name not in physical_channels
        or float(physical_channels[name]["baseline"]["count"]) <= 0
    )
    complete = not any(
        (
            missing_refined,
            missing_baseline,
            extra_refined,
            extra_baseline,
            zero_count_refined,
            zero_count_baseline,
        )
    )
    return {
        "complete": complete,
        "expected": sorted(expected),
        "observed_refined": sorted(refined),
        "observed_baseline": sorted(baseline),
        "missing_refined": missing_refined,
        "missing_baseline": missing_baseline,
        "extra_refined": extra_refined,
        "extra_baseline": extra_baseline,
        "zero_count_refined": zero_count_refined,
        "zero_count_baseline": zero_count_baseline,
    }


def _distributed_validation(

    model, ds_val, val_samples, cfg, resolved_specs, device, rank, world_size,
    norm_stats=None, pbar=None,
):
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    model.eval()

    per_rank = math.ceil(len(val_samples) / world_size)
    my_samples = val_samples[rank * per_rank : min((rank + 1) * per_rank, len(val_samples))]

    loss_sum = torch.zeros(1)
    baseline_loss_sum = torch.zeros(1)
    count = torch.zeros(1)
    refined_physical_sums: dict[str, dict[str, float]] = {}
    baseline_physical_sums: dict[str, dict[str, float]] = {}

    baseline_model = _validation_baseline_model(model)
    refinement_generator = ft._validation_refinement_generator(
        model, device, stream=rank,
    )

    with torch.inference_mode():
        for i in range(0, len(my_samples), batch_size):
            sample_batch = my_samples[i : i + batch_size]
            loss, refined_batch_metrics = ft.compute_supervised_loss(
                model=model, ds=ds_val, samples=sample_batch,
                config=cfg, resolved_specs=resolved_specs, device=device,
                norm_stats=norm_stats,
                refinement_generator=refinement_generator,
            )
            _merge_physical_validation_sums(
                refined_physical_sums, refined_batch_metrics
            )
            loss_sum += loss.detach().cpu()
            if baseline_model is not None:
                baseline_loss, baseline_batch_metrics = ft.compute_supervised_loss(
                    model=baseline_model, ds=ds_val, samples=sample_batch,
                    config=cfg, resolved_specs=resolved_specs, device=device,
                    norm_stats=norm_stats,
                )
            else:
                baseline_loss = loss
                baseline_batch_metrics = refined_batch_metrics
            _merge_physical_validation_sums(
                baseline_physical_sums, baseline_batch_metrics
            )
            baseline_loss_sum += baseline_loss.detach().cpu()
            count += 1
            if pbar is not None and rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    val_loss=loss_sum.item() / max(count.item(), 1.0),
                    base_loss=baseline_loss_sum.item() / max(count.item(), 1.0),
                    refresh=True,
                )

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(baseline_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    expected_channels = set(
        _expected_physical_validation_channels(model, cfg, resolved_specs)
    )
    local_channel_sets = {
        "refined": sorted(refined_physical_sums),
        "baseline": sorted(baseline_physical_sums),
    }
    gathered_channel_sets: list[dict[str, list[str]] | None] = [None] * world_size
    dist.all_gather_object(gathered_channel_sets, local_channel_sets)
    observed_refined = {
        name
        for rank_sets in gathered_channel_sets
        for name in ((rank_sets or {}).get("refined", ()))
    }
    observed_baseline = {
        name
        for rank_sets in gathered_channel_sets
        for name in ((rank_sets or {}).get("baseline", ()))
    }
    channel_names = sorted(
        expected_channels | observed_refined | observed_baseline
    )
    stat_names = _PHYSICAL_VALIDATION_STAT_NAMES

    def reduce_channel(source, channel):
        local = source.get(channel, {})
        packed = torch.tensor(
            [float(local.get(name, 0.0)) for name in stat_names],
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return {name: float(value) for name, value in zip(stat_names, packed.tolist())}

    def metric_ratio(refined, baseline, metric, *, absolute=False):
        numerator = float(refined[metric])
        denominator = float(baseline[metric])
        if absolute:
            numerator = abs(numerator)
            denominator = abs(denominator)
        return (
            numerator / denominator
            if np.isfinite(numerator)
            and np.isfinite(denominator)
            and denominator > 0.0
            else math.nan
        )

    physical_channels = {}
    rmse_ratios = []
    for channel in channel_names:
        refined_channel = _finalize_physical_validation_channel(
            reduce_channel(refined_physical_sums, channel)
        )
        baseline_channel = _finalize_physical_validation_channel(
            reduce_channel(baseline_physical_sums, channel)
        )
        bias_ratio = metric_ratio(
            refined_channel, baseline_channel, "bias", absolute=True
        )
        mae_ratio = metric_ratio(refined_channel, baseline_channel, "mae")
        rmse_ratio = metric_ratio(refined_channel, baseline_channel, "rmse")
        if channel in expected_channels and np.isfinite(rmse_ratio):
            rmse_ratios.append(rmse_ratio)
        physical_channels[channel] = {
            "baseline": baseline_channel,
            "refined": refined_channel,
            "absolute_bias_ratio": bias_ratio,
            "mae_ratio": mae_ratio,
            "rmse_ratio": rmse_ratio,
            "pattern_correlation_improvement": (
                refined_channel["pattern_correlation"]
                - baseline_channel["pattern_correlation"]
                if np.isfinite(refined_channel["pattern_correlation"])
                and np.isfinite(baseline_channel["pattern_correlation"])
                else math.nan
            ),
            "absolute_bias_improvement_percent": (
                100.0 * (1.0 - bias_ratio)
                if np.isfinite(bias_ratio)
                else math.nan
            ),
            "mae_improvement_percent": (
                100.0 * (1.0 - mae_ratio)
                if np.isfinite(mae_ratio)
                else math.nan
            ),
            "rmse_improvement_percent": (
                100.0 * (1.0 - rmse_ratio)
                if np.isfinite(rmse_ratio)
                else math.nan
            ),
        }

    physical_channel_coverage = _summarize_physical_channel_coverage(
        expected_channels=expected_channels,
        observed_refined=observed_refined,
        observed_baseline=observed_baseline,
        physical_channels=physical_channels,
    )
    coverage_complete = bool(physical_channel_coverage["complete"])

    mean_physical_rmse_ratio = (
        float(np.mean(rmse_ratios))
        if coverage_complete and len(rmse_ratios) == len(expected_channels)
        else math.nan
    )

    val_loss = (loss_sum / count).item() if count.item() > 0 else math.nan
    baseline_val_loss = (
        (baseline_loss_sum / count).item() if count.item() > 0 else math.nan
    )
    improvement_percent = (
        100.0 * (baseline_val_loss - val_loss) / abs(baseline_val_loss)
        if np.isfinite(baseline_val_loss) and abs(baseline_val_loss) > 1.0e-12
        else math.nan
    )
    return {
        "val_loss": val_loss,
        "baseline_val_loss": baseline_val_loss,
        "val_improvement_percent": improvement_percent,
        "physical_channels": physical_channels,
        "physical_channel_coverage": physical_channel_coverage,
        "mean_physical_rmse_ratio": mean_physical_rmse_ratio,
        "all_physical_channels_improved": (
            coverage_complete
            and len(rmse_ratios) == len(expected_channels)
            and all(ratio < 1.0 for ratio in rmse_ratios)
        ),
        "num_val_batches": int(count.item()),
    }


# ---------------------------------------------------------------------------
# Worker (one per GPU)
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, local_gpu: int, cfg: dict):
    # Set device early but defer dist.init_process_group until after model
    # construction — NCCL init can trigger SIGSEGV in erfinv_().
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local_gpu)
    device = torch.device(f"cuda:{local_gpu}")
    physical_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    train_cfg = cfg["training"]
    cfg.setdefault("runtime", {})["training_run_id"] = os.environ.get(
        "_AURORA_TRAINING_RUN_ID", ""
    )
    skip_validation = bool(train_cfg.get("skip_validation", False))
    validation_source = str(
        train_cfg.get("validation_source", "configured")
    ).lower()
    if validation_source not in {"configured", "train_tail"}:
        raise ValueError(
            "training.validation_source must be `configured` or `train_tail`."
        )

    # ---- data (every rank, read-only) ----
    train_ds = ft.open_dataset(cfg["paths"]["train_data_path"], cfg)
    val_ds = (
        None
        if skip_validation or validation_source == "train_tail"
        else ft.open_dataset(cfg["paths"]["val_data_path"], cfg)
    )
    test_ds = ft.open_dataset(cfg["paths"]["test_data_path"], cfg)

    longitude_datasets = [train_ds, test_ds]
    if val_ds is not None:
        longitude_datasets.insert(1, val_ds)
    lon_periodic = ft.validate_longitude_consistency(longitude_datasets, cfg)
    lon_dim = str(cfg.get("data", {}).get("lon_dim", "longitude"))
    train_lon = train_ds[lon_dim].values
    lat_dim = str(cfg.get("data", {}).get("lat_dim", "latitude"))
    train_lat = train_ds[lat_dim].values if lat_dim in train_ds.coords else None
    _print0(
        rank,
        f"Longitude grid: {len(train_lon)} columns | periodic={lon_periodic} | "
        f"range=[{float(train_lon[0]):.6g}, {float(train_lon[-1]):.6g}]",
    )

    static_path = cfg["paths"].get("static_data_path", "")
    if static_path:
        train_ds = ft.merge_external_static_vars(train_ds, static_path, cfg)
        if val_ds is not None:
            val_ds = ft.merge_external_static_vars(val_ds, static_path, cfg)
        test_ds = ft.merge_external_static_vars(test_ds, static_path, cfg)
    if not skip_validation and validation_source == "train_tail":
        val_ds = train_ds

    resolved_specs = ft.resolve_variable_specs(train_ds, cfg)
    norm_stats = ft.compute_target_normalization_stats(train_ds, resolved_specs, cfg)
    _print0(rank, f"Norm stats: { {k: {sk: sv.tolist() for sk, sv in v.items()} for k, v in norm_stats.items()} }")
    train_samples = ft.build_training_samples(train_ds, cfg, split_name="train")
    if skip_validation:
        val_samples = []
    elif validation_source == "train_tail":
        validation_fraction = float(
            train_cfg.get("validation_fraction", 0.1)
        )
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError(
                "training.validation_fraction must be between 0 and 0.5 "
                "for validation_source=train_tail."
            )
        all_samples = train_samples
        val_count = max(1, int(math.ceil(len(all_samples) * validation_fraction)))
        split_index = len(all_samples) - val_count
        gap = max(int(x) for x in cfg["data"].get("target_lead_times", [1]))
        train_end = split_index - gap
        if train_end <= 0:
            raise ValueError(
                "Not enough training samples for a disjoint train-tail "
                f"validation split (samples={len(all_samples)}, "
                f"validation={val_count}, temporal_gap={gap})."
            )
        train_samples = all_samples[:train_end]
        val_samples = all_samples[split_index:]
        _print0(
            rank,
            f"Validation source: train tail | train={len(train_samples)} | "
            f"gap={gap} sample(s) | val={len(val_samples)}",
        )
    else:
        val_samples = ft.build_training_samples(val_ds, cfg, split_name="val")
    val_label = "skipped" if skip_validation else str(len(val_samples))
    _print0(rank, f"Train samples: {len(train_samples)} | Val samples: {val_label}")

    # ---- model ----
    model_cfg = cfg["model"]
    variant = str(model_cfg.get("model_variant", "aurora_pretrained")).lower()
    mixed_precision_mode = str(model_cfg.get("mixed_precision", "none")).lower()
    use_bf16 = mixed_precision_mode in {"bf16", "bfloat16"}
    _print0(rank, f"Building configured model {variant} through shared factory")
    model = ft.build_finetune_model(
        cfg,
        resolved_specs,
        lon=train_lon,
        lat=train_lat,
        norm_stats=norm_stats,
        load_pretrained=True,
        autocast=False,
    )

    # If the flow wrapper is in use, hand it the per-variable normalisation
    # stats so its de-normalisation step at inference time matches the
    # space the FM head was trained in.
    from finetune.flow_refine import AuroraFlowRefine as _AFR
    if isinstance(model, _AFR):
        model.set_norm_stats(norm_stats)
        _print0(
            rank,
            f"Flow-refine head active: {model.refine_parameter_count():,} "
            f"trainable refine params, sampling_steps={model.sampling_steps}",
        )
        if getattr(model, "has_temporal", False):
            _print0(
                rank,
                f"Mamba temporal module active: "
                f"{model.temporal_parameter_count():,} params "
                f"(surf={list(model.target_surf_vars)}, "
                f"atmos={list(model.target_atmos_vars)})",
            )

    from finetune.refinement.two_phase import AuroraTwoPhaseRefiner as _ATP

    if isinstance(model, _ATP):
        _print0(
            rank,
            f"Stochastic refinement active: type="
            f"{model.refinement_config.type}, "
            f"{model.refine_parameter_count():,} refiner params, "
            f"aurora_frozen={model.aurora_frozen}, "
            f"channels={model.packing.num_channels}, "
            f"feedback_to_rollout={model.refinement_config.feedback_to_rollout}",
        )
        if model.has_temporal:
            _print0(
                rank,
                "Mamba temporal module active above unified refiner: "
                f"{model.temporal_parameter_count():,} params, "
                f"sequence_channels={model.packing.num_channels}",
            )

    param_summary = ft.configure_trainable_parameters(model, cfg)
    _print0(rank, f"Parameters: {json.dumps(param_summary)}")

    if use_bf16:
        # Per Aurora paper §B.7 ("32-bit floating-point computation"):
        # encoder, decoder, heads, and normalise/unnormalise are run in fp32
        # — only the backbone uses bf16.  We achieve this by keeping all
        # *parameters* in fp32 (so master weights and AdamW moments are
        # precise) and relying on torch.autocast(bf16) inside the loss
        # function to cast backbone activations to bf16 transparently.
        # This avoids the catastrophic precision loss observed when the
        # whole model is cast to bf16: with output scales ~1.9e-7, a 7-bit
        # bf16 mantissa introduces ~1.5e-9 quantisation noise per element —
        # the same magnitude as the upper-level NO2 signal itself.
        _print0(
            rank,
            "Params kept in fp32; using autocast(bf16) for backbone compute.",
        )

    model = model.to(device)
    _print0(rank, f"GPU memory after model load: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    # Initialize distributed (deferred until after model construction to avoid
    # SIGSEGV in NCCL/erfinv_ interaction on PyTorch 2.6 + Python 3.13).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    dist.barrier()
    _print0(rank, f"World size: {world_size} | Physical GPUs: {physical_gpus}")

    # ---- optimizer / scheduler ----
    ft.set_seed(int(train_cfg.get("seed", 42)))

    lr = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()

    # ---- per-parameter (trust-ratio-style) LR scaling --------------------
    # Aurora's pretrained per-level heads/embeds for top-of-atmosphere NO2
    # have weight magnitudes ~1e-9. AdamW's step is independent of the
    # parameter's own scale (≈ ±lr · sign(g) / sqrt(v)), so an unscaled step
    # at lr=2e-6 perturbs a 1e-9 weight by ~10^3× its natural magnitude and
    # destroys the pretrained mapping (this is exactly what produced
    # |pred|max=1283 at step 8). The fix is to scale each parameter's LR by
    # its own RMS, so tiny channels learn slowly and large channels learn at
    # the configured base rate.
    sa_cfg = train_cfg.get("scale_aware_lr", {}) or {}
    scale_aware_enabled = bool(sa_cfg.get("enabled", False))
    sa_min_scale = float(sa_cfg.get("min_scale", 1.0e-3))
    sa_rms_floor = float(sa_cfg.get("rms_floor", 1.0e-4))
    sa_ref_rms_cfg = sa_cfg.get("reference_rms", None)
    # Multiplicative LR scale applied to Aurora-pretrained params on top of
    # the RMS-based bucket scale. Default 1.0 = no extra damping. Set this
    # to a small value (e.g. 1e-2) to "anchor" the pretrained model at its
    # near-optimum and let the newly-initialised refinement heads carry
    # most of the learning signal.
    pretrained_lr_scale = float(sa_cfg.get("pretrained_lr_scale", 1.0))

    # Identify Aurora-pretrained vs newly-initialised params. The model is
    # AuroraConvRefine(base=AuroraAirPollution, surf_refine=..., atmos_refine=...)
    # under DDP; everything reachable via `.base` is pretrained, everything
    # else (surf_refine / atmos_refine) is newly initialised in this run.
    pretrained_param_ids: set[int] = set()
    inner = model.module if hasattr(model, "module") else model
    base_attr = getattr(inner, "base", None)
    if base_attr is not None:
        pretrained_param_ids = {id(p) for p in base_attr.parameters()}

    # If the user wants pretrained params frozen, set requires_grad=False on
    # them BEFORE collecting the trainable list, so they are excluded from
    # both the optimizer and the manual all-reduce in the training loop.
    if scale_aware_enabled and pretrained_lr_scale == 0.0 and pretrained_param_ids:
        n_frozen = 0
        for _, p in model.named_parameters():
            if id(p) in pretrained_param_ids and p.requires_grad:
                p.requires_grad = False
                n_frozen += 1
        if rank == 0:
            print(
                f"[opt] pretrained_lr_scale=0 -> froze {n_frozen} Aurora-pretrained "
                "tensors (only the conv-refine heads will train)"
            )

    named_trainable = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    from finetune.refinement.two_phase import resolve_temporal_config

    temporal_lr_multiplier = resolve_temporal_config(cfg)[
        "learning_rate_multiplier"
    ]
    temporal_module = getattr(inner, "temporal", None)
    temporal_param_ids = (
        {id(param) for param in temporal_module.parameters()}
        if temporal_module is not None
        else set()
    )

    if scale_aware_enabled:
        with torch.no_grad():
            rms_values = [
                float(p.detach().float().pow(2).mean().sqrt().item())
                for _, p in named_trainable
            ]
        if sa_ref_rms_cfg is not None:
            ref_rms = float(sa_ref_rms_cfg)
        else:
            # Compute reference RMS only over pretrained params so the
            # newly-initialised refine heads don't pull the median up/down.
            well_cond = [
                r for r, (_, p) in zip(rms_values, named_trainable)
                if r > sa_rms_floor and id(p) in pretrained_param_ids
            ]
            if not well_cond:
                well_cond = [r for r in rms_values if r > sa_rms_floor]
            ref_rms = float(np.median(well_cond)) if well_cond else 1.0e-2

        is_pretrained_flags = [
            (id(p) in pretrained_param_ids) for _, p in named_trainable
        ]

        # Per-parameter raw lr-scale.
        # - Pretrained params: clamp(rms/ref, [min_scale, 1.0]) * pretrained_lr_scale.
        # - New (refine) params: full LR (1.0); these were init'd from scratch
        #   and need the configured base LR to actually learn.
        raw_scales = [
            (
                max(min(rms / ref_rms, 1.0), sa_min_scale) * pretrained_lr_scale
                if is_pre
                else 1.0
            )
            for rms, is_pre in zip(rms_values, is_pretrained_flags)
        ]
        if temporal_lr_multiplier != 1.0:
            raw_scales = [
                scale
                * (
                    temporal_lr_multiplier
                    if id(param) in temporal_param_ids
                    else 1.0
                )
                for scale, (_, param) in zip(raw_scales, named_trainable)
            ]
        # Bucket by (log10(scale) at 0.5-dex, is_pretrained) so we can label
        # each group clearly in the printout.
        bucket_keys = [
            (int(np.floor(np.log10(max(s, 1e-12)) * 2.0)), is_pre)
            for s, is_pre in zip(raw_scales, is_pretrained_flags)
        ]
        unique_buckets = sorted(set(bucket_keys), key=lambda k: (-k[0], k[1]))

        if rank == 0:
            n_pre = sum(is_pretrained_flags)
            n_new = len(is_pretrained_flags) - n_pre
            print(
                f"[opt] scale-aware LR: ref_rms={ref_rms:.3e}, "
                f"pretrained_lr_scale={pretrained_lr_scale:.2e}, "
                f"{len(named_trainable)} params "
                f"({n_pre} pretrained / {n_new} new) -> {len(unique_buckets)} groups"
            )
        param_groups = []
        for bkt_key in unique_buckets:
            member_idxs = [i for i, k in enumerate(bucket_keys) if k == bkt_key]
            scales_in_bucket = [raw_scales[i] for i in member_idxs]
            lr_scale = float(np.exp(np.mean(np.log(scales_in_bucket))))
            group_lr = lr * lr_scale
            tag = "pre" if bkt_key[1] else "NEW"
            param_groups.append({
                "params": [named_trainable[i][1] for i in member_idxs],
                "lr": group_lr,
                "weight_decay": weight_decay,
                "name": f"{tag}_bkt[{bkt_key[0]}]",
            })
            if rank == 0:
                bk_rms = [rms_values[i] for i in member_idxs]
                example = named_trainable[member_idxs[0]][0]
                print(f"  [{tag}] bucket={bkt_key[0]:>+3d} n={len(member_idxs):>5d} "
                      f"lr_scale={lr_scale:.3e} lr={group_lr:.2e} "
                      f"rms[min={min(bk_rms):.2e},max={max(bk_rms):.2e}] "
                      f"e.g. {example}")
        opt_target = param_groups
    else:
        if temporal_lr_multiplier == 1.0 or not temporal_param_ids:
            opt_target = [param for _, param in named_trainable]
        else:
            spatial_params = [
                param
                for _, param in named_trainable
                if id(param) not in temporal_param_ids
            ]
            temporal_params = [
                param
                for _, param in named_trainable
                if id(param) in temporal_param_ids
            ]
            opt_target = []
            if spatial_params:
                opt_target.append({"params": spatial_params, "lr": lr})
            if temporal_params:
                opt_target.append(
                    {
                        "params": temporal_params,
                        "lr": lr * temporal_lr_multiplier,
                    }
                )

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(opt_target, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(opt_target, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            opt_target, lr=lr,
            momentum=float(train_cfg.get("sgd_momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Keep independent snapshots: scheduler construction mutates each optimizer
    # group's live LR (for example, LambdaLR applies its first warmup factor).
    # Referring back to ``param_groups`` during resume is therefore unsafe.
    configured_group_lrs = [float(g["lr"]) for g in optimizer.param_groups]

    batch_size = int(train_cfg.get("batch_size", 1))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    grad_clip = float(train_cfg.get("gradient_clip_val", 0.0))
    validation_frequency = max(1, int(train_cfg.get("validation_frequency", 1)))
    save_best_only = bool(train_cfg.get("save_best_only", True))
    save_last = bool(train_cfg.get("save_last", True))
    diagnostics_enabled = bool(train_cfg.get("diagnostics_enabled", False))
    checkpoint_guard_metrics = tuple(
        str(metric).strip().lower()
        for metric in train_cfg.get("checkpoint_guard_metrics", ())
    )
    checkpoint_guard_relative_tolerance = float(
        train_cfg.get("checkpoint_guard_relative_tolerance", 0.0)
    )
    checkpoint_guard_correlation_tolerance = float(
        train_cfg.get("checkpoint_guard_correlation_tolerance", 0.0)
    )
    checkpoint_guard_bias_rmse_floor_fraction = float(
        train_cfg.get("checkpoint_guard_bias_rmse_floor_fraction", 0.0)
    )

    # The scheduler advances once per *optimizer update*, not once per sample
    # batch. Account for both data-parallel sharding and gradient accumulation;
    # omitting accumulation made a 32-step accumulated run decay its LR about
    # 32x too slowly.
    updates_per_epoch = _optimizer_updates_per_epoch(
        len(train_samples),
        world_size=world_size,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
    )
    scheduler = ft.create_scheduler(
        optimizer, cfg, num_training_steps=updates_per_epoch * max(1, num_epochs),
    )
    fresh_scheduler_lrs = [float(g["lr"]) for g in optimizer.param_groups]

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    best_ckpt_path = checkpoint_dir / "best.ckpt"
    last_ckpt_path = checkpoint_dir / "last.ckpt"

    history: list[dict] = []
    best_val_loss = float("inf")
    latest_validation: dict[str, object] | None = None
    global_step = 0
    start_epoch = 0
    resume_checkpoint: dict | None = None

    # ---- resume from checkpoint ----
    resume_training = bool(train_cfg.get("resume_training", False))
    resume_from = str(train_cfg.get("resume_from", "")).strip()
    resume_path = None
    if resume_training and resume_from:
        if resume_from.lower() == "best":
            resume_path = best_ckpt_path
        elif resume_from.lower() == "last":
            resume_path = last_ckpt_path
        else:
            resume_path = Path(resume_from)

    if resume_path and resume_path.exists():
        _print0(rank, f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        resume_checkpoint = ckpt
        saved_run_id = (
            ckpt.get("config", {}).get("runtime", {}).get("training_run_id")
        )
        if saved_run_id:
            cfg["runtime"]["training_run_id"] = str(saved_run_id)
        ft.validate_checkpoint_longitude(model, ckpt)
        saved_config = ckpt.get("config", {})
        saved_temporal = resolve_temporal_config(saved_config)
        serialized_temporal = ckpt.get("resolved_temporal_config")
        if isinstance(serialized_temporal, dict):
            saved_temporal.update(serialized_temporal)
        current_temporal = resolve_temporal_config(cfg)
        saved_temporal_enabled = bool(saved_temporal["enabled"])
        current_temporal_enabled = bool(current_temporal["enabled"])
        temporal_migration = (
            saved_temporal_enabled != current_temporal_enabled
            or (
                saved_temporal_enabled
                and current_temporal_enabled
                and saved_temporal != current_temporal
            )
        )
        ft.validate_checkpoint_refinement_contract(
            model,
            ckpt,
            cfg,
            resolved_specs,
            allow_temporal_migration=True,
            require_validated=False,
        )
        # Only an explicitly detected temporal-only migration may be tolerant,
        # for either the legacy or unified wrapper. An unchanged architecture
        # (including disabled temporal options that merely differ in YAML) loads
        # strictly so truncated or incompatible restarts fail early.
        load_result = model.load_state_dict(
            ckpt["model_state_dict"], strict=not temporal_migration,
        )
        missing = [k for k in load_result.missing_keys]
        unexpected = [k for k in load_result.unexpected_keys]
        if temporal_migration:
            def _is_temporal_key(key: str) -> bool:
                while key.startswith("module."):
                    key = key[len("module.") :]
                return key.startswith("temporal.")

            invalid_migration_keys = [
                key
                for key in (*missing, *unexpected)
                if not _is_temporal_key(key)
            ]
            if invalid_migration_keys:
                raise RuntimeError(
                    "Temporal warm-start migration may only add or remove "
                    "temporal.* state. Non-temporal missing/unexpected keys were "
                    f"found: {invalid_migration_keys[:8]!r}."
                )
        if missing:
            _print0(rank, f"  checkpoint missing {len(missing)} key(s) "
                          f"(kept fresh init), e.g. {missing[:3]}")
        if unexpected:
            _print0(rank, f"  checkpoint has {len(unexpected)} unexpected key(s) "
                          f"(ignored), e.g. {unexpected[:3]}")
        if temporal_migration:
            _print0(
                rank,
                "  Temporal architecture changed; keeping the current optimizer "
                "and scheduler state so new/removed Mamba parameters are handled safely.",
            )
            checkpoint_group_lrs = configured_group_lrs
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            checkpoint_group_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        skip_sched = bool(train_cfg.get("resume_skip_scheduler", False))
        scheduler_state = ckpt.get("scheduler_state_dict")
        if (
            scheduler is not None
            and scheduler_state is not None
            and not skip_sched
            and not temporal_migration
        ):
            scheduler.load_state_dict(scheduler_state)
            restored_lrs = scheduler.get_last_lr()
            if not isinstance(restored_lrs, (list, tuple)) or len(restored_lrs) != len(
                optimizer.param_groups
            ):
                restored_lrs = checkpoint_group_lrs
            for group, restored_lr in zip(optimizer.param_groups, restored_lrs):
                group["lr"] = float(restored_lr)
            _print0(
                rank,
                f"  Restored scheduler at step {scheduler.last_epoch}, "
                f"lr={[float(g['lr']) for g in optimizer.param_groups]}",
            )
        else:
            # A skipped/missing scheduler state is a deliberate fresh stage:
            # use the current YAML's group scales and fresh warmup position.
            for group, group_lr, initial_lr in zip(
                optimizer.param_groups, fresh_scheduler_lrs, configured_group_lrs
            ):
                group["lr"] = group_lr
                group["initial_lr"] = initial_lr
        if skip_sched:
            _print0(rank, "  resume_skip_scheduler=true -> using fresh scheduler "
                          f"(t_max={train_cfg.get('scheduler_t_max')})")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        saved_validation = ckpt.get("validation")
        if isinstance(saved_validation, dict):
            latest_validation = dict(saved_validation)
        # Load previous history if available
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            import json as _json
            history = _json.loads(history_path.read_text())
        _print0(rank, f"  Resuming at epoch {start_epoch}, step {global_step}, "
                       f"best_val_loss={best_val_loss:.4e}")
    elif resume_training and resume_from:
        _print0(rank, f"WARNING: resume_from='{resume_from}' but checkpoint not found — training from scratch")

    # Calibrate CAMS-minus-Aurora corrections over every logical training sample
    # before the first optimiser step. Ranks use disjoint, unpadded shards and
    # synchronize sufficient statistics only once at the end of the pass.
    residual_calibration = _calibrate_residual_scalers_from_training_split(
        model=model,
        train_ds=train_ds,
        train_samples=train_samples,
        cfg=cfg,
        resolved_specs=resolved_specs,
        device=device,
        rank=rank,
        world_size=world_size,
        norm_stats=norm_stats,
        global_step=global_step,
    )
    if residual_calibration is not None:
        cfg.setdefault("runtime", {})["residual_calibration"] = residual_calibration
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "residual_calibration.json").write_text(
                json.dumps(residual_calibration, indent=2, allow_nan=False) + "\n"
            )
            print(
                "Residual correction statistics "
                f"(CAMS - Aurora; {residual_calibration['method']}; "
                f"samples={residual_calibration['logical_samples']}; "
                f"fingerprint={residual_calibration['training_split_fingerprint_sha256']}):",
                flush=True,
            )
            for scaler_report in residual_calibration["scalers"]:
                for channel in scaler_report["channels"]:
                    level = (
                        "surface"
                        if channel["level_hpa"] is None
                        else f"{channel['level_hpa']:g} hPa"
                    )
                    print(
                        f"  {channel['variable']} @ {level}: n={channel['valid_cell_count']} "
                        f"mean={channel['physical_correction_mean']:.6e} "
                        f"std={channel['physical_correction_std']:.6e} "
                        f"{channel['units']}",
                        flush=True,
                    )
    # The pre-pass exercises stochastic training forwards. A resumed job must
    # continue the exact serialized stream; a fresh job starts from its
    # configured seed after the pre-pass.
    if resume_checkpoint is None or not ft.restore_checkpoint_rng_state(
        resume_checkpoint,
        device=device,
    ):
        ft.set_seed(int(train_cfg.get("seed", 42)))

    # Mark the current logical run independently of best/last. Inference uses
    # this marker to reject a stale best checkpoint when a fresh run produces
    # no acceptable validation improvement.
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "training_run.metadata.json").write_text(
            json.dumps(
                {
                    "training_run_id": cfg.get("runtime", {}).get(
                        "training_run_id", ""
                    ),
                    "resume_path": str(resume_path) if resume_path else None,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()

    patience_cfg = train_cfg.get("early_stopping", {})
    early_stop_enabled = bool(patience_cfg.get("enabled", False))
    patience = int(patience_cfg.get("patience", 5))
    min_delta = float(patience_cfg.get("min_delta", 0.0))
    no_improve_epochs = 0

    # ---- training loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()

        # Optional sampling-step schedule. The safe default keeps deterministic
        # one-step validation throughout. Multi-step stochastic validation must
        # be requested explicitly and is inappropriate for deterministic
        # bias-correction model selection unless ensemble metrics are used.
        # Training loss itself is independent of `sampling_steps`.
        inner = model.module if hasattr(model, "module") else model
        if isinstance(inner, _AFR):
            init_steps = int(model_cfg.get("flow_refine_sampling_steps", 1))
            late_steps = int(
                model_cfg.get("flow_refine_sampling_steps_late", init_steps)
            )
            phase_frac = float(
                model_cfg.get("flow_refine_phase_fraction", 1.0 / 3.0)
            )
            switch_epoch = int(round(num_epochs * phase_frac))
            desired = late_steps if epoch >= switch_epoch else init_steps
            if desired != inner.sampling_steps:
                if rank == 0:
                    print(
                        f"  [flow] epoch {epoch}: sampling_steps "
                        f"{inner.sampling_steps} -> {desired} "
                        f"(switch at epoch {switch_epoch})"
                    )
                inner.sampling_steps = desired

        epoch_samples = list(train_samples)
        rng = np.random.default_rng(int(train_cfg.get("seed", 42)) + epoch)
        rng.shuffle(epoch_samples)

        # Shard samples across ranks for data parallelism.
        # Pad so every rank processes the same number of batches, preventing
        # collective-operation mismatches in _allreduce_grads (gloo SIGABRT).
        per_rank = math.ceil(len(epoch_samples) / world_size)
        rank_samples = epoch_samples[rank * per_rank : min((rank + 1) * per_rank, len(epoch_samples))]
        # Duplicate the last sample to pad short ranks to per_rank length.
        while len(rank_samples) < per_rank:
            rank_samples.append(rank_samples[-1])

        epoch_loss_sum = 0.0
        epoch_batches = 0
        optimizer.zero_grad(set_to_none=True)

        n_train_batches = math.ceil(len(rank_samples) / batch_size)
        should_validate = (
            not skip_validation
            and (((epoch + 1) % validation_frequency == 0) or (epoch == num_epochs - 1))
        )
        if should_validate:
            val_per_rank = math.ceil(len(val_samples) / world_size)
            my_val_count = len(
                val_samples[rank * val_per_rank : min((rank + 1) * val_per_rank, len(val_samples))]
            )
            n_val_batches = math.ceil(my_val_count / batch_size)
        else:
            n_val_batches = 0

        pbar = tqdm(
            range(0, len(rank_samples), batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
            disable=(rank != 0 or _progress_bars_disabled()),
            unit="batch",
            total=n_train_batches,
            colour="blue",
            dynamic_ncols=True,
            leave=False,
            position=0,
        )
        for step_i in pbar:
            sample_batch = rank_samples[step_i : step_i + batch_size]

            # Activate one-shot per-variable loss breakdown on rank 0
            # for the first batch and the batch right after the first update.
            diag_on = (
                diagnostics_enabled
                and rank == 0
                and global_step in (0, accumulation_steps, accumulation_steps + 1)
            )
            if diag_on:
                ft._DIAG_LOSS_BREAKDOWN["active"] = True
                ft._DIAG_LOSS_BREAKDOWN["rows"] = []

            loss, _ = ft.compute_supervised_loss(
                model=model, ds=train_ds, samples=sample_batch,
                config=cfg, resolved_specs=resolved_specs, device=device,
                norm_stats=norm_stats,
            )

            if diag_on:
                ft._DIAG_LOSS_BREAKDOWN["active"] = False
                rows = ft._DIAG_LOSS_BREAKDOWN.get("rows", [])
                print(f"  [diag] loss breakdown @ global_step={global_step}:")
                for r in rows:
                    print(
                        f"    {r['var']:>8s}@{str(r['lead']):>5s} kind={r['kind']:>5s} "
                        f"|pred|max={r['pred_abs_max']:.3e} |tgt|max={r['tgt_abs_max']:.3e} "
                        f"|diff|max={r['diff_abs_max']:.3e} loss={r['loss_var']:.4e} "
                        f"w={r['weight']:.2g}"
                    )

            (loss / accumulation_steps).backward()

            update_now = ((epoch_batches + 1) % accumulation_steps == 0) or (
                (step_i + batch_size) >= len(rank_samples)
            )
            if update_now:
                _allreduce_grads(model, world_size)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer_updated = False
                # ---- DIAGNOSTIC: locate top-gradient params on first update ----
                if diagnostics_enabled and global_step <= 1 and rank == 0:
                    grad_info = []
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach()
                            grad_info.append((n, float(g.norm().item()), tuple(p.shape)))
                    grad_info.sort(key=lambda x: -x[1])
                    print(f"  [diag] top-10 grad-norm params at step {global_step}:")
                    for n, gn, sh in grad_info[:10]:
                        print(f"    {gn:.3e}  {n}  {sh}")
                    total_sq = sum(gn * gn for _, gn, _ in grad_info)
                    print(f"  [diag] total trainable grad L2 = {total_sq ** 0.5:.4e} "
                          f"({len(grad_info)} tensors)")
                # ---- end diagnostic ----
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=grad_clip if grad_clip > 0 else float("inf"),
                )
                # Skip optimizer step when gradients contain NaN/Inf (bf16 overflow).
                if not torch.isfinite(pre_clip_norm):
                    if rank == 0:
                        pbar.set_postfix(
                            phase="train",
                            loss=epoch_loss_sum / max(epoch_batches, 1),
                            lr=optimizer.param_groups[0]["lr"],
                            grad="nan/inf skip",
                            refresh=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                else:
                    post_clip = (
                        min(float(pre_clip_norm), grad_clip) if grad_clip > 0 else float(pre_clip_norm)
                    )
                    optimizer.step()
                    optimizer_updated = True
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        pbar.set_postfix(
                            phase="train",
                            loss=epoch_loss_sum / max(epoch_batches, 1),
                            lr=optimizer.param_groups[0]["lr"],
                            grad=f"{float(pre_clip_norm):.2e}",
                            clip=f"{post_clip:.2e}",
                            refresh=True,
                        )
                if scheduler is not None and optimizer_updated:
                    scheduler.step()

            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_batches += 1
            global_step += 1
            pbar.set_postfix(
                phase="train",
                loss=epoch_loss_sum / epoch_batches,
                lr=optimizer.param_groups[0]["lr"],
            )

        # All-reduce train loss for logging (CPU tensors for gloo)
        train_loss_t = torch.tensor([epoch_loss_sum, float(epoch_batches)])
        dist.all_reduce(train_loss_t, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_t[0] / train_loss_t[1]).item() if train_loss_t[1] > 0 else math.nan

        # Training bar is complete; close it before validation gets its own bar.
        pbar.close()

        # ---- validation ----
        val_loss = float("nan")
        baseline_val_loss = float("nan")
        val_improvement_percent = float("nan")
        mean_physical_rmse_ratio = float("nan")
        all_physical_channels_improved = False
        physical_channels = {}
        physical_channel_coverage = {
            "complete": False,
            "expected": [],
        }
        if should_validate:
            val_pbar = tqdm(
                total=n_val_batches,
                desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
                disable=(rank != 0 or _progress_bars_disabled()),
                unit="batch",
                colour="green",
                dynamic_ncols=True,
                leave=False,
                position=0,
            )
            val_metrics = _distributed_validation(
                model, val_ds, val_samples, cfg, resolved_specs, device, rank, world_size,
                norm_stats=norm_stats, pbar=val_pbar,
            )
            val_loss = float(val_metrics["val_loss"])
            baseline_val_loss = float(val_metrics["baseline_val_loss"])
            val_improvement_percent = float(
                val_metrics["val_improvement_percent"]
            )
            mean_physical_rmse_ratio = float(val_metrics["mean_physical_rmse_ratio"])
            all_physical_channels_improved = bool(
                val_metrics["all_physical_channels_improved"]
            )
            physical_channels = val_metrics["physical_channels"]
            physical_channel_coverage = val_metrics[
                "physical_channel_coverage"
            ]
            val_pbar.close()
            model.train()

        if should_validate:
            checkpoint_guards = _evaluate_checkpoint_guards(
                physical_channels=physical_channels,
                expected_channels=physical_channel_coverage["expected"],
                metrics=checkpoint_guard_metrics,
                relative_tolerance=checkpoint_guard_relative_tolerance,
                correlation_tolerance=checkpoint_guard_correlation_tolerance,
                bias_rmse_floor_fraction=(
                    checkpoint_guard_bias_rmse_floor_fraction
                ),
            )
        else:
            checkpoint_guards = {
                "status": (
                    "not_evaluated" if checkpoint_guard_metrics else "disabled"
                ),
                "passed": not checkpoint_guard_metrics,
                "configured_metrics": list(checkpoint_guard_metrics),
                "relative_tolerance": checkpoint_guard_relative_tolerance,
                "correlation_tolerance": (
                    checkpoint_guard_correlation_tolerance
                ),
                "bias_rmse_floor_fraction": (
                    checkpoint_guard_bias_rmse_floor_fraction
                ),
                "channels": {},
                "failures": [],
                "failed_channels": [],
            }

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_val_loss": baseline_val_loss,
            "val_improvement_percent": val_improvement_percent,
            "mean_physical_rmse_ratio": mean_physical_rmse_ratio,
            "physical_channel_coverage_complete": bool(physical_channel_coverage["complete"]),
            "physical_rmse_improvement_percent": 100.0 * (1.0 - mean_physical_rmse_ratio),
            "checkpoint_guard_status": checkpoint_guards["status"],
            "checkpoint_guard_passed": bool(checkpoint_guards["passed"]),
            "checkpoint_guard_failed_channels": checkpoint_guards[
                "failed_channels"
            ],
            "checkpoint_guards": checkpoint_guards,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        # Persistent one-line epoch summary (progress bars use leave=False and
        # vanish on completion, so log the epoch result explicitly).
        _print0(
            rank,
            f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | baseline_val_loss={baseline_val_loss:.4f} | "
            f"improvement={val_improvement_percent:+.2f}% | "
            f"physical_RMSE_improvement={100.0 * (1.0 - mean_physical_rmse_ratio):+.2f}% | "
            f"checkpoint_guard={checkpoint_guards['status']} | "
            f"mem={torch.cuda.memory_allocated(device) / 1e9:.1f}GB",
        )

        checkpoint_metric_name = str(
            train_cfg.get("checkpoint_metric", "validation_loss")
        ).strip().lower()
        if checkpoint_metric_name == "mean_physical_rmse_ratio":
            checkpoint_metric_value = mean_physical_rmse_ratio
            checkpoint_baseline_value = 1.0
        elif checkpoint_metric_name in {"validation_loss", "val_loss"}:
            checkpoint_metric_name = "validation_loss"
            checkpoint_metric_value = val_loss
            checkpoint_baseline_value = baseline_val_loss
        else:
            raise ValueError(
                "training.checkpoint_metric must be validation_loss or "
                f"mean_physical_rmse_ratio; got {checkpoint_metric_name!r}."
            )
        require_improvement = bool(
            train_cfg.get("require_refinement_improvement", False)
        )
        require_all_channels = bool(
            train_cfg.get("require_all_physical_channels_improve", False)
        )
        coverage_complete = bool(physical_channel_coverage["complete"])
        non_degrading = coverage_complete and (
            not require_improvement
            or (
                np.isfinite(checkpoint_baseline_value)
                and checkpoint_metric_value <= checkpoint_baseline_value
            )
        )
        if require_all_channels and not all_physical_channels_improved:
            non_degrading = False
        improved = _checkpoint_candidate_improves(
            should_validate=should_validate,
            checkpoint_metric_value=checkpoint_metric_value,
            non_degrading=non_degrading,
            physical_channel_coverage_complete=coverage_complete,
            checkpoint_guards_passed=bool(checkpoint_guards["passed"]),
            best_value=best_val_loss,
            min_delta=min_delta,
        )
        if should_validate:
            if not coverage_complete:
                validation_status = "incomplete_physical_channel_coverage"
            elif not np.isfinite(checkpoint_metric_value):
                validation_status = "non_finite_checkpoint_metric"
            elif require_all_channels and not all_physical_channels_improved:
                validation_status = "physical_channel_degradation"
            elif not bool(checkpoint_guards["passed"]):
                validation_status = "checkpoint_guard_failed"
            elif require_improvement and checkpoint_metric_value > checkpoint_baseline_value:
                validation_status = "degrades_deterministic_baseline"
            elif not improved:
                validation_status = "not_better_than_current_best"
            else:
                validation_status = "accepted_best"
            latest_validation = {
                "status": validation_status,
                "accepted": bool(improved),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "val_loss": float(val_loss),
                "baseline_val_loss": float(baseline_val_loss),
                "val_improvement_percent": float(val_improvement_percent),
                "physical_channels": physical_channels,
                "physical_channel_coverage": physical_channel_coverage,
                "mean_physical_rmse_ratio": float(mean_physical_rmse_ratio),
                "physical_rmse_improvement_percent": float(
                    100.0 * (1.0 - mean_physical_rmse_ratio)
                ),
                "checkpoint_metric": checkpoint_metric_name,
                "checkpoint_metric_value": float(checkpoint_metric_value),
                "validation_refinement_ensemble_size": int(
                    train_cfg.get("validation_refinement_ensemble_size", 1)
                ),
                "min_delta": float(min_delta),
                "require_refinement_improvement": bool(require_improvement),
                "require_all_physical_channels_improve": bool(require_all_channels),
                "checkpoint_guards": checkpoint_guards,
            }
            row["checkpoint_promotion_status"] = validation_status
            row["checkpoint_promoted"] = bool(improved)
        else:
            row["checkpoint_promotion_status"] = "not_validated"
            row["checkpoint_promoted"] = False
        if improved:
            best_val_loss = checkpoint_metric_value
            no_improve_epochs = 0

        if should_validate and improved:
            if rank == 0:
                ft.save_checkpoint(
                    best_ckpt_path, model, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, config=cfg,
                    norm_stats=norm_stats,
                    validation=latest_validation,
                    validated_for_inference=True,
                )
            dist.barrier()
        elif should_validate:
            no_improve_epochs += 1

        if should_validate and not save_best_only:
            if rank == 0:
                ft.save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.ckpt",
                    model, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, config=cfg,
                    norm_stats=norm_stats,
                    validation=latest_validation,
                    validated_for_inference=bool(improved),
                )
            dist.barrier()

        if save_last:
            if rank == 0:
                ft.save_checkpoint(
                    last_ckpt_path, model, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, config=cfg,
                    norm_stats=norm_stats,
                    validation=latest_validation,
                    validated_for_inference=False,
                )
            dist.barrier()

        if rank == 0 and should_validate:
            (checkpoint_dir / "training_run.metadata.json").write_text(
                json.dumps(
                    {
                        "training_run_id": cfg.get("runtime", {}).get(
                            "training_run_id", ""
                        ),
                        "resume_path": str(resume_path) if resume_path else None,
                        "validation": latest_validation,
                    },
                    indent=2,
                    allow_nan=True,
                )
                + "\n"
            )
        if should_validate:
            dist.barrier()

        if early_stop_enabled and no_improve_epochs >= patience:
            _print0(rank, f"Early stopping triggered at epoch {epoch}.")
            break

    # ---- post-training (rank 0 only) ----
    if rank == 0:
        ft.write_training_history(history, output_dir)

        rollout_cfg = cfg.get("rollout", {})
        post_training_rollout_status = "disabled"
        post_training_rollout_checkpoint: str | None = None
        if bool(rollout_cfg.get("run_rollout_after_training", True)):
            require_validated = bool(
                cfg.get("inference", {}).get(
                    "require_validated_checkpoint", False
                )
            )
            try:
                rollout_ckpt_path = ft.select_refinement_checkpoint(
                    checkpoint_dir,
                    require_validated=require_validated,
                )
            except (ValueError, FileNotFoundError) as exc:
                post_training_rollout_status = "skipped_checkpoint_gate"
                print(
                    "Skipping post-training rollout because no eligible "
                    f"checkpoint was selected: {exc}"
                )
            else:
                post_training_rollout_status = "completed"
                post_training_rollout_checkpoint = str(rollout_ckpt_path)
                state = torch.load(
                    str(rollout_ckpt_path), map_location=device, weights_only=False,
                )
                ft.validate_checkpoint_longitude(model, state)
                ft.validate_checkpoint_refinement_contract(
                    model, state, cfg, resolved_specs,
                )
                model.load_state_dict(state["model_state_dict"])
                _print0(rank, f"Post-training rollout checkpoint: {rollout_ckpt_path}")

                input_steps = int(cfg["data"].get("input_time_steps", 2))
                test_samples = [{
                    "anchor_index": input_steps - 1,
                    "history_indices": list(range(input_steps)),
                    "target_indices": {},
                }]
                model.eval()
                predictions = ft.run_rollout(
                    model=model, ds=test_ds, start_sample=test_samples[0],
                    config=cfg, resolved_specs=resolved_specs, device=device,
                )
                print(f"Rollout prediction steps: {len(predictions)}")

                rollout_path = output_dir / "rollout_predictions.nc"
                if predictions and bool(
                    rollout_cfg.get("save_predictions_to_netcdf", True)
                ):
                    # Convert bf16 predictions to float32 for numpy/NetCDF.
                    predictions = [p.type(torch.float32) for p in predictions]
                    ft.save_predictions(
                        predictions, rollout_path, save_netcdf=True,
                        resolved_specs=resolved_specs,
                        smooth_sigma=float(rollout_cfg.get("smooth_sigma", 0.0)),
                        patch_size=int(model_cfg.get("patch_size", 3)),
                        lon_periodic=lon_periodic,
                    )
                    print(f"Saved rollout NetCDF: {rollout_path}")

        ft.write_run_manifest(
            cfg, output_dir,
            extras={
                "distributed": True, "world_size": world_size,
                "gpu_ids": physical_gpus,
                "history_rows": len(history),
                "parameter_summary": param_summary,
                "num_train_samples": len(train_samples),
                "num_val_samples": len(val_samples),
                "post_training_rollout_status": post_training_rollout_status,
                "post_training_rollout_checkpoint": post_training_rollout_checkpoint,
            },
        )
        print("Training complete.")

    _cleanup_dist()


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _torchrun_worker():
    """Entry-point when launched via ``torchrun`` (one process per GPU)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    config_path = Path(os.environ["_AURORA_CONFIG"])
    cfg = ft.load_config(config_path)

    # With torchrun + CUDA_VISIBLE_DEVICES, local_rank maps to logical GPU
    _worker(rank, world_size, local_rank, cfg)


def main():
    parser = argparse.ArgumentParser(description="Distributed Aurora fine-tuning")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (e.g. '5,6,7')")
    parser.add_argument("--min-free-fraction", type=float, default=FREE_MEMORY_FRACTION)
    parser.add_argument("--_torchrun_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._torchrun_worker:
        _torchrun_worker()
        return

    if args.config:
        config_path = Path(args.config)
    else:
        for candidate in [
            _SCRIPT_DIR / "aurora_finetune_rollout_config.yaml",
            _PROJECT_ROOT / "finetune" / "aurora_finetune_rollout_config.yaml",
        ]:
            if candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError("Could not find aurora_finetune_rollout_config.yaml")

    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = _detect_idle_gpus(min_free_fraction=args.min_free_fraction)
        if not gpu_ids:
            print("No idle GPUs detected — falling back to GPU 0", flush=True)
            gpu_ids = [0]

    world_size = len(gpu_ids)
    print(f"Launching on {world_size} GPU(s): {gpu_ids}", flush=True)

    master_port = str(_find_free_port())
    visible_devices = ",".join(str(g) for g in gpu_ids)

    # Build torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        f"--master_port={master_port}",
        "--master_addr=127.0.0.1",
        # The script itself, in worker mode
        str(Path(__file__).resolve()),
        "--_torchrun_worker",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["_AURORA_GPU_IDS"] = visible_devices
    env["_AURORA_CONFIG"] = str(config_path.resolve())
    env["_AURORA_TRAINING_RUN_ID"] = uuid.uuid4().hex
    # Mitigate fragmentation in the long, multi-stage rollout fine-tune. The
    # FM head allocates many small tensors per (lead × level × variable);
    # the default best-fit allocator fragments enough to OOM on a 22 GB A10G
    # by epoch ~6. expandable_segments lets the allocator grow contiguously.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import subprocess as _sp
    proc = _sp.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
