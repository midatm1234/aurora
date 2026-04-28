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
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# Distributed validation
# ---------------------------------------------------------------------------

def _distributed_validation(
    model, ds_val, val_samples, cfg, resolved_specs, device, rank, world_size,
    norm_stats=None,
):
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    model.eval()

    per_rank = math.ceil(len(val_samples) / world_size)
    my_samples = val_samples[rank * per_rank : min((rank + 1) * per_rank, len(val_samples))]

    loss_sum = torch.zeros(1)
    count = torch.zeros(1)

    with torch.inference_mode():
        for i in range(0, len(my_samples), batch_size):
            sample_batch = my_samples[i : i + batch_size]
            loss, _ = ft.compute_supervised_loss(
                model=model, ds=ds_val, samples=sample_batch,
                config=cfg, resolved_specs=resolved_specs, device=device,
                norm_stats=norm_stats,
            )
            loss_sum += loss.detach().cpu()
            count += 1

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / count).item() if count.item() > 0 else math.nan
    return {"val_loss": val_loss, "num_val_batches": int(count.item())}


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

    # ---- data (every rank, read-only) ----
    train_ds = ft.open_dataset(cfg["paths"]["train_data_path"], cfg)
    val_ds = ft.open_dataset(cfg["paths"]["val_data_path"], cfg)
    test_ds = ft.open_dataset(cfg["paths"]["test_data_path"], cfg)

    static_path = cfg["paths"].get("static_data_path", "")
    if static_path:
        train_ds = ft.merge_external_static_vars(train_ds, static_path, cfg)
        val_ds = ft.merge_external_static_vars(val_ds, static_path, cfg)
        test_ds = ft.merge_external_static_vars(test_ds, static_path, cfg)

    resolved_specs = ft.resolve_variable_specs(train_ds, cfg)
    norm_stats = ft.compute_target_normalization_stats(train_ds, resolved_specs, cfg)
    _print0(rank, f"Norm stats: { {k: {sk: sv.tolist() for sk, sv in v.items()} for k, v in norm_stats.items()} }")
    train_samples = ft.build_training_samples(train_ds, cfg, split_name="train")
    val_samples = ft.build_training_samples(val_ds, cfg, split_name="val")
    _print0(rank, f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

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

    param_summary = ft.configure_trainable_parameters(model, cfg)
    _print0(rank, f"Parameters: {json.dumps(param_summary)}")

    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
        _print0(rank, "Model converted to bfloat16")

    model = model.to(device)
    _print0(rank, f"GPU memory after model load: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    # Initialize distributed (deferred until after model construction to avoid
    # SIGSEGV in NCCL/erfinv_ interaction on PyTorch 2.6 + Python 3.13).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    dist.barrier()
    _print0(rank, f"World size: {world_size} | Physical GPUs: {physical_gpus}")

    # ---- optimizer / scheduler ----
    train_cfg = cfg["training"]
    ft.set_seed(int(train_cfg.get("seed", 42)))

    lr = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=lr,
            momentum=float(train_cfg.get("sgd_momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    batch_size = int(train_cfg.get("batch_size", 1))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    grad_clip = float(train_cfg.get("gradient_clip_val", 0.0))
    validation_frequency = max(1, int(train_cfg.get("validation_frequency", 1)))
    save_best_only = bool(train_cfg.get("save_best_only", True))
    save_last = bool(train_cfg.get("save_last", True))

    updates_per_epoch = max(1, int(np.ceil(len(train_samples) / max(1, batch_size))))
    scheduler = ft.create_scheduler(
        optimizer, cfg, num_training_steps=updates_per_epoch * max(1, num_epochs),
    )

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
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
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

    patience_cfg = train_cfg.get("early_stopping", {})
    early_stop_enabled = bool(patience_cfg.get("enabled", False))
    patience = int(patience_cfg.get("patience", 5))
    min_delta = float(patience_cfg.get("min_delta", 0.0))
    no_improve_epochs = 0

    # ---- training loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()
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

        n_batches = math.ceil(len(rank_samples) / batch_size)
        pbar = tqdm(
            range(0, len(rank_samples), batch_size),
            desc=f"Epoch {epoch}/{num_epochs}",
            disable=(rank != 0),
            unit="batch",
            total=n_batches,
        )
        for step_i in pbar:
            sample_batch = rank_samples[step_i : step_i + batch_size]

            loss, _ = ft.compute_supervised_loss(
                model=model, ds=train_ds, samples=sample_batch,
                config=cfg, resolved_specs=resolved_specs, device=device,
                norm_stats=norm_stats,
            )

            (loss / accumulation_steps).backward()

            update_now = ((epoch_batches + 1) % accumulation_steps == 0) or (
                (step_i + batch_size) >= len(rank_samples)
            )
            if update_now:
                _allreduce_grads(model, world_size)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=grad_clip,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{epoch_loss_sum / epoch_batches:.4e}")

        # All-reduce train loss for logging (CPU tensors for gloo)
        train_loss_t = torch.tensor([epoch_loss_sum, float(epoch_batches)])
        dist.all_reduce(train_loss_t, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_t[0] / train_loss_t[1]).item() if train_loss_t[1] > 0 else math.nan

        # ---- validation ----
        should_validate = ((epoch + 1) % validation_frequency == 0) or (epoch == num_epochs - 1)
        val_loss = float("nan")
        if should_validate:
            val_metrics = _distributed_validation(
                model, val_ds, val_samples, cfg, resolved_specs, device, rank, world_size,
                norm_stats=norm_stats,
            )
            val_loss = float(val_metrics["val_loss"])
            model.train()

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        pbar.set_postfix(
            train=f"{train_loss:.4e}", val=f"{val_loss:.4e}",
            mem=f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB",
            refresh=True,
        )
        _print0(rank, f"  Epoch {epoch}: train_loss={train_loss:.4e} val_loss={val_loss:.4e} "
                       f"lr={float(optimizer.param_groups[0]['lr']):.2e} "
                       f"mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB")

        improved = should_validate and np.isfinite(val_loss) and (val_loss < (best_val_loss - min_delta))
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0

        if should_validate and (improved or not save_best_only):
            if rank == 0:
                ft.save_checkpoint(
                    best_ckpt_path, model, optimizer, scheduler,
                    epoch=epoch, global_step=global_step,
                    best_val_loss=best_val_loss, config=cfg,
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
            if best_ckpt_path.exists():
                state = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
                model.load_state_dict(state["model_state_dict"])

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
                ft.save_predictions(predictions, rollout_path, save_netcdf=True, resolved_specs=resolved_specs)
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

    import subprocess as _sp
    proc = _sp.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
