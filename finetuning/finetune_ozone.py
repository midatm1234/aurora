"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Small fine-tuning test of `AuroraAirPollution` on CAMS analyses, with a focus on ozone
(`go3` on pressure levels and `gtco3` total column).

Follows the recipe in `docs/finetuning.md`: full fine-tuning with AdamW, `autocast=True` and
activation checkpointing, but with a lower default learning rate (see
`docs/finetuning_ozone.md`). Gradients are clipped and their norm is logged to detect exploding
gradients. The loss is a latitude-weighted MAE in normalised space averaged over all predicted
variables; ozone-specific metrics are logged separately.

Expects `--data-dir` to contain the output of `finetuning/prepare_cams_ozone.py`, or use
`--synthetic N` to test the pipeline on random data. Run with
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`.
"""

import argparse
import json
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora import AuroraAirPollution, Batch, Metadata
from aurora.normalisation import locations, normalise_atmos_var, normalise_surf_var, scales

SURF_MAP = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
    "pm1": "pm1",
    "pm2p5": "pm2p5",
    "pm10": "pm10",
    "tcco": "tcco",
    "tc_no": "tc_no",
    "tcno2": "tcno2",
    "gtco3": "gtco3",
    "tcso2": "tcso2",
}
ATMOS_VARS = ("t", "u", "v", "q", "z", "co", "no", "no2", "go3", "so2")
POSITIVE_VARS = {
    *("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2"),
    *("co", "no", "no2", "go3", "so2"),
}
STATIC_FILE = "aurora-0.4-air-pollution-static.pickle"


class CamsData:
    """All time steps of the downloaded days, held in memory as float32 arrays."""

    def __init__(
        self,
        times: list[datetime],
        surf: dict[str, np.ndarray],
        atmos: dict[str, np.ndarray],
        lat: torch.Tensor,
        lon: torch.Tensor,
        levels: tuple[int, ...],
        static: dict[str, torch.Tensor],
    ) -> None:
        self.times, self.surf, self.atmos = times, surf, atmos
        self.lat, self.lon, self.levels, self.static = lat, lon, levels, static

    @classmethod
    def synthetic(cls, n_times: int, seed: int = 0) -> "CamsData":
        """Random data on the CAMS 0.4° grid, scaled with the normalisation statistics.

        Only meant to test the pipeline (memory, speed, checkpointing), not to learn anything.
        """
        from huggingface_hub import hf_hub_download

        rng = np.random.default_rng(seed)
        lat = np.linspace(90, -90, 451, dtype=np.float32)
        lon = np.arange(900, dtype=np.float32) * 0.4
        levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

        def _field(keys: list[str], shape: tuple[int, ...], positive: bool) -> np.ndarray:
            loc = np.array([locations[k] for k in keys], np.float32)[:, None, None]
            scale = np.array([scales[k] for k in keys], np.float32)[:, None, None]
            z = rng.standard_normal(shape, dtype=np.float32)
            return loc + scale * (np.abs(z) if positive else z)

        positive = POSITIVE_VARS
        surf = {
            k: np.stack([_field([k], (1, 451, 900), k in positive)[0] for _ in range(n_times)])
            for k in SURF_MAP
        }
        atmos = {
            k: np.stack(
                [
                    _field([f"{k}_{lv}" for lv in levels], (13, 451, 900), k in positive)
                    for _ in range(n_times)
                ]
            )
            for k in ATMOS_VARS
        }
        static_path = hf_hub_download(repo_id="microsoft/aurora", filename=STATIC_FILE)
        with open(static_path, "rb") as f:
            static = {k: torch.from_numpy(v) for k, v in pickle.load(f).items()}
        start = datetime(2022, 6, 1)
        times = [start + i * timedelta(hours=12) for i in range(n_times)]
        return cls(times, surf, atmos, torch.from_numpy(lat), torch.from_numpy(lon), levels, static)

    @classmethod
    def from_dir(cls, data_dir: Path) -> "CamsData":
        sfc_files = sorted(data_dir.glob("*-cams-surface-level.nc"))
        plev_files = sorted(data_dir.glob("*-cams-atmospheric.nc"))
        if not sfc_files or len(sfc_files) != len(plev_files):
            raise FileNotFoundError(f"Missing or unmatched CAMS files in {data_dir}.")

        def _open(files: list[Path]) -> xr.Dataset:
            ds = [
                xr.open_dataset(f, engine="netcdf4", decode_timedelta=False).isel(forecast_period=0)
                for f in files
            ]
            return xr.concat(ds, dim="forecast_reference_time").sortby("forecast_reference_time")

        sfc, plev = _open(sfc_files), _open(plev_files)
        times = sfc.valid_time.values.astype("datetime64[s]").tolist()
        steps = {b - a for a, b in zip(times[:-1], times[1:])}
        if steps != {timedelta(hours=12)}:
            raise ValueError(f"Expected consecutive 12-hourly times, got steps {steps}.")

        with open(data_dir / STATIC_FILE, "rb") as f:
            static = {k: torch.from_numpy(v) for k, v in pickle.load(f).items()}
        return cls(
            times,
            {k: sfc[v].values.astype(np.float32) for k, v in SURF_MAP.items()},
            {k: plev[k].values.astype(np.float32) for k in ATMOS_VARS},
            torch.from_numpy(plev.latitude.values.astype(np.float32)),
            torch.from_numpy(plev.longitude.values.astype(np.float32)),
            tuple(int(level) for level in plev.pressure_level.values),
            static,
        )

    def __len__(self) -> int:
        """Number of samples: two history steps plus one 12 h target."""
        return len(self.times) - 2

    def sample(self, i: int) -> tuple[Batch, Batch]:
        hist, tgt = slice(i, i + 2), slice(i + 2, i + 3)

        def _batch(s: slice) -> Batch:
            return Batch(
                surf_vars={k: torch.from_numpy(v[s])[None] for k, v in self.surf.items()},
                static_vars=self.static,
                atmos_vars={k: torch.from_numpy(v[s])[None] for k, v in self.atmos.items()},
                metadata=Metadata(
                    lat=self.lat,
                    lon=self.lon,
                    time=(self.times[s][-1],),
                    atmos_levels=self.levels,
                ),
            )

        return _batch(hist), _batch(tgt)

    def check(self) -> dict:
        """Basic data QC: shapes, NaNs and ozone ranges."""
        report: dict = {"times": [str(self.times[0]), str(self.times[-1]), len(self.times)]}
        for name, arrays in (("surf", self.surf), ("atmos", self.atmos)):
            for k, v in arrays.items():
                n_nan = int(np.isnan(v).sum())
                if n_nan:
                    raise ValueError(f"{name} variable {k} has {n_nan} NaNs.")
                report[k] = {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max())}
        for k, v in self.static.items():
            if torch.isnan(v).any():
                raise ValueError(f"Static variable {k} has NaNs.")
        return report


def lat_weights(lat: torch.Tensor) -> torch.Tensor:
    w = torch.cos(torch.deg2rad(lat)).clamp(min=0)
    return (w / w.mean())[:, None]


def compare(model: AuroraAirPollution, pred: Batch, target: Batch) -> tuple[torch.Tensor, dict]:
    """Latitude-weighted normalised MAE averaged over all variables, plus ozone metrics."""
    h = pred.spatial_shape[0]
    w = lat_weights(pred.metadata.lat).to(pred.metadata.lat.device)
    levels = pred.metadata.atmos_levels
    losses, metrics = [], {}

    for k, p in pred.surf_vars.items():
        t = target.surf_vars[k][..., :h, :].to(p.device)
        err = normalise_surf_var(p, k, stats=model.surf_stats) - normalise_surf_var(
            t, k, stats=model.surf_stats
        )
        losses.append((w * err.abs()).mean())
        if k == "gtco3":
            metrics["gtco3_nmae"] = losses[-1].item()
            metrics["gtco3_rmse"] = (w * (p - t) ** 2).mean().sqrt().item()

    for k, p in pred.atmos_vars.items():
        t = target.atmos_vars[k][..., :h, :].to(p.device)
        err = normalise_atmos_var(p, k, levels) - normalise_atmos_var(t, k, levels)
        losses.append((w * err.abs()).mean())
        if k == "go3":
            metrics["go3_nmae"] = losses[-1].item()
            rmse = (w * (p - t) ** 2).mean(dim=(0, 1, 3, 4)).sqrt()
            metrics["go3_rmse_by_level"] = {lv: r.item() for lv, r in zip(levels, rmse)}

    loss = torch.stack(losses).mean()
    metrics["loss"] = loss.item()
    return loss, metrics


@torch.no_grad()
def evaluate(model: AuroraAirPollution, data: CamsData, indices: list[int]) -> dict:
    model.eval()
    results = []
    for i in indices:
        batch, target = data.sample(i)
        pred = model(batch.to("cuda"))
        results.append(compare(model, pred, target)[1])
    model.train()
    out = {
        k: float(np.mean([r[k] for r in results]))
        for k in ("loss", "go3_nmae", "gtco3_nmae", "gtco3_rmse")
    }
    out["go3_rmse_by_level"] = {
        lv: float(np.mean([r["go3_rmse_by_level"][lv] for r in results]))
        for lv in results[0]["go3_rmse_by_level"]
    }
    return out


def log(record: dict, log_file: Path) -> None:
    line = json.dumps(record)
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AuroraAirPollution on CAMS ozone.")
    parser.add_argument("--data-dir", type=Path, help="Output of `prepare_cams_ozone.py`.")
    parser.add_argument(
        "--synthetic",
        type=int,
        metavar="N_TIMES",
        help="Use N_TIMES random 12-hourly time steps instead of `--data-dir` (pipeline test).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=2, help="Last N samples for val.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="3e-4 (as in `finetune.py`) degraded the pretrained model in our tests.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="One train step, then stop.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (args.data_dir is None) == (args.synthetic is None):
        parser.error("Provide exactly one of --data-dir and --synthetic.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.out_dir / "metrics.jsonl"

    if args.synthetic:
        data = CamsData.synthetic(args.synthetic, seed=args.seed)
    else:
        data = CamsData.from_dir(args.data_dir)
    log({"event": "data_qc", **data.check()}, log_file)
    n = len(data)
    if n <= args.val_samples:
        raise ValueError(f"Only {n} samples; need more than --val-samples={args.val_samples}.")
    train_idx = list(range(n - args.val_samples))
    val_idx = list(range(n - args.val_samples, n))
    log({"event": "split", "train": len(train_idx), "val": len(val_idx)}, log_file)

    model = AuroraAirPollution(autocast=True)
    model.load_checkpoint()
    model.configure_activation_checkpointing()
    model = model.to("cuda")
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log({"event": "model", "trainable_params": n_params}, log_file)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if not args.dry_run:
        log({"event": "eval", "step": 0, **evaluate(model, data, val_idx)}, log_file)

    max_steps = 1 if args.dry_run else args.max_steps
    order: list[int] = []
    for step in range(1, max_steps + 1):
        if not order:
            order = random.sample(train_idx, len(train_idx))
        batch, target = data.sample(order.pop())

        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        pred = model(batch.to("cuda"))
        loss, metrics = compare(model, pred, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        opt.step()
        torch.cuda.synchronize()

        metrics.pop("go3_rmse_by_level")
        log(
            {
                "event": "train",
                "step": step,
                **metrics,
                "grad_norm": grad_norm.item(),
                "step_time_s": round(time.time() - t0, 2),
                "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            },
            log_file,
        )
        if not args.dry_run and (step % args.eval_every == 0 or step == max_steps):
            log({"event": "eval", "step": step, **evaluate(model, data, val_idx)}, log_file)

    if args.dry_run:
        return

    ckpt = args.out_dir / "aurora-0.4-air-pollution-ozone-ft.pt"
    torch.save(model.state_dict(), ckpt)
    del model, opt
    torch.cuda.empty_cache()
    reloaded = AuroraAirPollution(autocast=True)
    reloaded.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    reloaded = reloaded.to("cuda")
    log({"event": "reload_eval", **evaluate(reloaded, data, val_idx)}, log_file)


if __name__ == "__main__":
    main()
