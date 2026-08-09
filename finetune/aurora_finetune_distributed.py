#!/usr/bin/env python
"""Distributed Aurora fine-tuning across multiple GPUs.

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
from aurora import (  # noqa: E402
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraPretrained,
    AuroraSmallPretrained,
    AuroraWave,
)

MODEL_REGISTRY = {
    "aurora": Aurora,
    "aurora_pretrained": AuroraPretrained,
    "aurora_small_pretrained": AuroraSmallPretrained,
    "aurora_12h_pretrained": Aurora12hPretrained,
    "aurora_highres": AuroraHighRes,
    "aurora_air_pollution": AuroraAirPollution,
    "aurora_wave": AuroraWave,
}

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


# ---------------------------------------------------------------------------
# Distributed validation
# ---------------------------------------------------------------------------

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
    inner = model.module if hasattr(model, "module") else model
    baseline_model = getattr(inner, "base", None)

    with torch.inference_mode():
        for i in range(0, len(my_samples), batch_size):
            sample_batch = my_samples[i : i + batch_size]
            loss, _ = ft.compute_supervised_loss(
                model=model, ds=ds_val, samples=sample_batch,
                config=cfg, resolved_specs=resolved_specs, device=device,
                norm_stats=norm_stats,
            )
            loss_sum += loss.detach().cpu()
            if baseline_model is not None:
                baseline_loss, _ = ft.compute_supervised_loss(
                    model=baseline_model, ds=ds_val, samples=sample_batch,
                    config=cfg, resolved_specs=resolved_specs, device=device,
                    norm_stats=norm_stats,
                )
            else:
                baseline_loss = loss
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
    model_var_cfg = ft.derive_model_variable_config(resolved_specs, cfg)
    model_kwargs = dict(model_cfg.get("model_kwargs", {}))

    if "patch_size" in model_cfg and "patch_size" not in model_kwargs:
        model_kwargs["patch_size"] = int(model_cfg["patch_size"])

    mixed_precision_mode = str(model_cfg.get("mixed_precision", "none")).lower()
    use_bf16 = mixed_precision_mode in {"bf16", "bfloat16"}
    # Disable Aurora's internal autocast — we store params directly in bf16.
    model_kwargs["autocast"] = False

    model = MODEL_REGISTRY[variant](
        surf_vars=model_var_cfg["surf_vars"],
        static_vars=model_var_cfg["static_vars"],
        atmos_vars=model_var_cfg["atmos_vars"],
        **model_kwargs,
    )

    if bool(model_cfg.get("use_pretrained_weights", True)):
        ckpt_path = cfg["paths"].get("pretrained_checkpoint")
        if ckpt_path:
            _print0(rank, f"Loading local checkpoint: {ckpt_path}")
            model.load_checkpoint_local(ckpt_path, strict=False)
        else:
            _print0(rank, f"Loading default HF checkpoint for {variant}")
            model.load_checkpoint(strict=False)

    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.configure_activation_checkpointing()

    # Optionally wrap with convolutional refinement heads.
    model = ft.maybe_wrap_conv_refine(model, cfg, resolved_specs, lon=train_lon)

    # Optionally wrap with rectified-flow residual refine heads. Mutually
    # exclusive with conv-refine (the helper checks the flag itself).
    model = ft.maybe_wrap_flow_refine(model, cfg, resolved_specs, lon=train_lon)
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

    # Optionally wrap with the unified stochastic residual refiner
    # (flow_matching_transformer / diffusion_unet / diffusion_transformer).
    # Returns the model unchanged for `none` and for the legacy
    # flow_matching_unet path handled above.
    model = ft.maybe_wrap_stochastic_refine(
        model,
        cfg,
        resolved_specs,
        lon=train_lon,
        lat=train_lat,
        norm_stats=norm_stats,
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

    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

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
        opt_target = [p for _, p in named_trainable]

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
    global_step = 0
    start_epoch = 0

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
        saved_run_id = (
            ckpt.get("config", {}).get("runtime", {}).get("training_run_id")
        )
        if saved_run_id:
            cfg["runtime"]["training_run_id"] = str(saved_run_id)
        ft.validate_checkpoint_longitude(model, ckpt)
        saved_temporal_enabled = bool(
            ckpt.get("config", {}).get("model", {}).get(
                "mamba_temporal_enabled", False,
            )
        )
        current_temporal_enabled = bool(
            cfg.get("model", {}).get("mamba_temporal_enabled", False)
        )
        temporal_migration = saved_temporal_enabled != current_temporal_enabled
        ft.validate_checkpoint_refinement_contract(
            model,
            ckpt,
            cfg,
            resolved_specs,
            allow_temporal_migration=True,
        )
        # Tolerant load so checkpoints from flow-matching-only runs (which lack
        # the Mamba temporal parameters) resume cleanly into a temporal-enabled
        # model, and vice-versa. Missing keys keep their fresh init; unexpected
        # keys (e.g. a disabled temporal module) are ignored. Any *other*
        # mismatch would still surface here for inspection.
        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        missing = [k for k in load_result.missing_keys]
        unexpected = [k for k in load_result.unexpected_keys]
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
        # Load previous history if available
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            import json as _json
            history = _json.loads(history_path.read_text())
        _print0(rank, f"  Resuming at epoch {start_epoch}, step {global_step}, "
                       f"best_val_loss={best_val_loss:.4e}")
    elif resume_training and resume_from:
        _print0(rank, f"WARNING: resume_from='{resume_from}' but checkpoint not found — training from scratch")

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
            val_pbar.close()
            model.train()

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_val_loss": baseline_val_loss,
            "val_improvement_percent": val_improvement_percent,
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
            f"mem={torch.cuda.memory_allocated(device) / 1e9:.1f}GB",
        )

        require_improvement = bool(
            train_cfg.get("require_refinement_improvement", False)
        )
        non_degrading = (
            not require_improvement
            or (
                np.isfinite(baseline_val_loss)
                and val_loss <= baseline_val_loss
            )
        )
        improved = (
            should_validate
            and np.isfinite(val_loss)
            and non_degrading
            and (val_loss < (best_val_loss - min_delta))
        )
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0

        if should_validate and (improved or not save_best_only):
            if rank == 0:
                ft.save_checkpoint(
                    best_ckpt_path, model, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, config=cfg,
                    norm_stats=norm_stats,
                )
            dist.barrier()
        elif should_validate:
            no_improve_epochs += 1

        if save_last:
            if rank == 0:
                ft.save_checkpoint(
                    last_ckpt_path, model, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, config=cfg,
                    norm_stats=norm_stats,
                )
            dist.barrier()

        if early_stop_enabled and no_improve_epochs >= patience:
            _print0(rank, f"Early stopping triggered at epoch {epoch}.")
            break

    # ---- post-training (rank 0 only) ----
    if rank == 0:
        ft.write_training_history(history, output_dir)

        rollout_cfg = cfg.get("rollout", {})
        if bool(rollout_cfg.get("run_rollout_after_training", True)):
            rollout_ckpt_path = (
                best_ckpt_path
                if np.isfinite(best_val_loss) and best_ckpt_path.exists()
                else last_ckpt_path
            )
            if rollout_ckpt_path.exists():
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
            if predictions and bool(rollout_cfg.get("save_predictions_to_netcdf", True)):
                # Convert bf16 predictions to float32 for numpy/netCDF compatibility.
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
