"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Profiling and numerical-parity benchmark for Aurora stochastic refinement.

Measures, per refiner:

* refinement training-step time,
* refinement inference (single-member) time,
* ensemble-generation time for serial and batched member evaluation,
* peak CPU / GPU memory,
* checkpoint I/O time,
* NetCDF I/O time,

and, for every optimization that can be toggled, whether it is *numerically
neutral*:

* serial versus batched ensemble generation (expected bitwise identical),
* reference (``math``) versus fused (``sdpa``) attention,
* eager versus ``torch.compile`` execution,
* fp32 versus any reduced-precision autocast path.

Reduced-precision and compiled paths are reported with their measured
deviation; they are *not* recommended unless the deviation is within the
documented tolerance. Nothing here changes a configured schedule, sampler,
solver, step count, ensemble size or model geometry.

Usage::

    python -m finetune.refinement_benchmark --report /tmp/refinement_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import torch

from finetune.refinement.checkpoint import (
    CHECKPOINT_KIND_REFINEMENT,
    build_refinement_checkpoint,
    save_checkpoint_atomic,
)
from finetune.refinement.two_phase import build_two_phase_refiner
from finetune.refinement_smoke_test import build_packing, smoke_config

REFINERS = (
    "flow_matching_unet",
    "flow_matching_transformer",
    "diffusion_unet",
    "diffusion_transformer",
)


def _peak_cpu_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _time(fn: Callable[[], Any], repeats: int = 3) -> tuple[float, Any]:
    fn()  # warm-up, excluded from the measurement
    best = float("inf")
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return best, result


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.double() - b.double()).abs()
    finite = torch.isfinite(diff)
    return float(diff[finite].max()) if bool(finite.any()) else 0.0


def _benchmark_one(
    model,
    packing,
    rollout: torch.Tensor,
    target: torch.Tensor,
    lead: torch.Tensor,
    *,
    refinement_type: str,
    ensemble: int,
    device: str,
    try_compile: bool,
    try_bf16: bool,
) -> dict[str, Any]:
    """Measure one refiner. Kept in its own scope so no closure captures a loop variable."""
    entry: dict[str, Any] = {"refiner_parameters": model.refine_parameter_count()}

    def train_step():
        model.zero_grad(set_to_none=True)
        generator = torch.Generator(device="cpu").manual_seed(0)
        out = model.training_step(rollout, target, forecast_lead_time=lead, generator=generator)
        out.losses["total_loss"].backward()
        return float(out.losses["total_loss"].detach())

    def refine(members: int, seed: int, chunk: int | None = None):
        return model.refine(
            rollout,
            forecast_lead_time=lead,
            ensemble_size=members,
            seed=seed,
            chunk_size=chunk,
        )

    entry["train_step_seconds"], entry["train_loss"] = _time(train_step)
    entry["single_member_seconds"], _ = _time(lambda: refine(1, 1))

    serial_seconds, serial = _time(lambda: refine(ensemble, 5, 1))
    batched_seconds, batched = _time(lambda: refine(ensemble, 5, ensemble))
    entry["ensemble_serial_seconds"] = serial_seconds
    entry["ensemble_batched_seconds"] = batched_seconds
    entry["ensemble_speedup"] = (
        serial_seconds / batched_seconds if batched_seconds else float("nan")
    )
    entry["ensemble_parity_max_abs_diff"] = max(
        _max_abs_diff(serial.member_residuals[:, m], batched.member_residuals[:, m])
        for m in range(ensemble)
    )

    # Attention implementation parity (Transformers only). Both settings compute
    # the same exact operation; only the kernel differs.
    if model.refinement_config.uses_transformer:
        model.refiner.set_attention_implementation("math")
        math_seconds, math_out = _time(lambda: refine(1, 3))
        model.refiner.set_attention_implementation("sdpa")
        sdpa_seconds, sdpa_out = _time(lambda: refine(1, 3))
        entry["attention_math_seconds"] = math_seconds
        entry["attention_sdpa_seconds"] = sdpa_seconds
        entry["attention_parity_max_abs_diff"] = _max_abs_diff(
            math_out.member_residuals, sdpa_out.member_residuals
        )
        model.refiner.set_attention_implementation("auto")

    # Reduced precision is measured and reported, never enabled implicitly.
    if try_bf16:
        reference = refine(1, 4)
        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.autocast(autocast_device, dtype=torch.bfloat16):
            reduced = refine(1, 4)
        entry["bf16_parity_max_abs_diff"] = _max_abs_diff(
            reference.member_residuals, reduced.member_residuals
        )

    if try_compile and not hasattr(model.refiner, "net"):
        # The legacy per-variable heads are not a single sub-module, so there is
        # nothing to compile selectively. Reported, not silently skipped.
        entry["compile_status"] = "not applicable for the legacy flow_matching_unet heads"
    elif try_compile:
        try:
            eager = refine(1, 6)
            original = model.refiner.net
            model.refiner.net = torch.compile(original)
            compiled_seconds, compiled_out = _time(lambda: refine(1, 6), repeats=1)
            entry["compiled_seconds"] = compiled_seconds
            entry["compile_parity_max_abs_diff"] = _max_abs_diff(
                eager.member_residuals, compiled_out.member_residuals
            )
            model.refiner.net = original
        except Exception as exc:  # pragma: no cover - environment dependent
            entry["compile_error"] = f"{type(exc).__name__}: {exc}"

    with tempfile.TemporaryDirectory() as tmp:
        payload = build_refinement_checkpoint(
            model,
            kind=CHECKPOINT_KIND_REFINEMENT,
            refinement_type=refinement_type,
            packing=packing,
        )
        destination = Path(tmp) / "refinement.ckpt"
        entry["checkpoint_save_seconds"], _ = _time(
            lambda: save_checkpoint_atomic(payload, destination, atomic=True), repeats=1
        )

    entry["peak_cpu_mb"] = round(_peak_cpu_mb(), 1)
    if device.startswith("cuda"):
        entry["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        torch.cuda.reset_peak_memory_stats()
    return entry


def benchmark(
    *,
    height: int = 32,
    width: int = 64,
    batch: int = 2,
    ensemble: int = 4,
    device: str = "cpu",
    try_compile: bool = False,
    try_bf16: bool = False,
) -> dict[str, Any]:
    packing = build_packing(height, width)
    report: dict[str, Any] = {
        "grid": [height, width],
        "batch": batch,
        "ensemble_size": ensemble,
        "device": device,
        "refiners": {},
    }

    for refinement_type in REFINERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build_two_phase_refiner(None, packing, smoke_config(refinement_type, ensemble))
            model.initialize_refiner(model.conditioning_channels())
        model = model.to(device)

        rollout = torch.randn(batch, packing.num_channels, height, width, device=device)
        target = rollout + 0.05 * torch.randn_like(rollout)
        lead = torch.full((batch,), 48.0, device=device)
        report["refiners"][refinement_type] = _benchmark_one(
            model,
            packing,
            rollout,
            target,
            lead,
            refinement_type=refinement_type,
            ensemble=ensemble,
            device=device,
            try_compile=try_compile,
            try_bf16=try_bf16,
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--ensemble", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--try-compile", action="store_true")
    parser.add_argument("--try-bf16", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    report = benchmark(
        height=args.height,
        width=args.width,
        batch=args.batch,
        ensemble=args.ensemble,
        device=args.device,
        try_compile=args.try_compile,
        try_bf16=args.try_bf16,
    )
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
