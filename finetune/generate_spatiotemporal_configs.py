"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Generate the matched joint-spatiotemporal O3 ablation configurations.

Every variant is derived from one base recipe by changing *only* the factor
under test, so a difference in skill cannot be attributed to an incidental
change in capacity, data, seed or schedule. The generated grid is deliberately
small: this is a controlled comparison, not a factorial sweep.

Factors
-------
``head``
    which of the four unified refiners provides the spatial backbone.
``conditioning``
    ``spatial_only``  previous behaviour: no calendar, no solar, no vertical
                      identity, no temporal context, bottleneck attention OFF.
    ``calendar``      + calendar / solar / vertical identity, still no temporal
                      context. Isolates "metadata channels" from real temporal
                      modelling.
    ``temporal``      + a causal temporal backend feeding the denoiser.
``temporal_backend``
    ``causal_conv`` (attention-free), ``conv_gru``, ``attention``, ``mamba``.

Controls that must be run before any claim of temporal skill:
``no_history``   temporal backend built but fed a single frame.
``shuffled``     past frames permuted; the current lead stays last.
These require additional inference-time ablations; neither control is exposed
as a switch by this generator or the current synthetic pilot.

Usage::

    python -m finetune.generate_spatiotemporal_configs --list
    python -m finetune.generate_spatiotemporal_configs --out finetune/ablations
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["VARIANTS", "build_variant", "main"]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPOSITORY_ROOT / "finetune" / "aurora_O3_global_finetune_3day_lead_config.yaml"
)

HEADS: tuple[str, ...] = (
    "flow_matching_conv_unet",
    "flow_matching_transformer",
    "diffusion_unet",
    "diffusion_transformer",
)

#: (name, conditioning stage, temporal backend, bottleneck attention)
VARIANTS: tuple[tuple[str, str, str, bool], ...] = (
    # Stage 1: the spatial-only control. Attention is explicitly OFF so the
    # U-Net control really is attention-free.
    ("spatial_only_attn_off", "spatial_only", "none", False),
    # Stage 1b: the same control with bottleneck attention ON, which is what
    # the historical O3 recipe actually used. Keeping both stops "U-Net" from
    # being described as attention-free by default.
    ("spatial_only_attn_on", "spatial_only", "none", True),
    # Stage 2: calendar / solar / vertical identity only. Adding metadata is
    # not temporal modelling, and this variant exists to measure exactly that.
    ("calendar_only", "calendar", "none", True),
    # Stage 3: genuine causal temporal refinement, one variant per backend.
    ("temporal_causal_conv", "temporal", "causal_conv", True),
    ("temporal_conv_gru", "temporal", "conv_gru", True),
    ("temporal_attention", "temporal", "attention", True),
    ("temporal_mamba", "temporal", "mamba", True),
)


def _set_conditioning(refinement: dict[str, Any], stage: str) -> None:
    conditioning = refinement.setdefault("conditioning", {})
    enabled = stage in {"calendar", "temporal"}
    conditioning["calendar"] = enabled
    conditioning["solar_geometry"] = enabled
    conditioning["vertical_identity"] = enabled


def build_variant(
    base: dict[str, Any],
    *,
    head: str,
    name: str,
    stage: str,
    backend: str,
    bottleneck_attention: bool,
    source_config: str | None = None,
) -> dict[str, Any]:
    """Return one ablation configuration derived from ``base``."""
    if head not in HEADS or stage not in {"spatial_only", "calendar", "temporal"}:
        raise ValueError("Unknown ablation head or conditioning stage.")
    if backend not in {"none", "causal_conv", "conv_gru", "attention", "mamba"}:
        raise ValueError("Unknown temporal backend.")
    if (stage == "temporal") != (backend != "none"):
        raise ValueError("Only temporal-stage variants may enable a temporal backend.")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Ablation name must be a single nonempty path component.")
    config = copy.deepcopy(base)
    base_case = str(base.get("case_name", "")).strip()
    if not base_case:
        raise ValueError("The base recipe must define case_name.")
    # Reuse the original prepared data, while isolating every experiment's
    # checkpoints, rollouts and reports in its own case directory.
    paths = config.setdefault("paths", {})
    paths["data_case_name"] = paths.get("data_case_name") or base_case
    config["case_name"] = f"{base_case}__{head}__{name}"
    model = config.setdefault("model", {})
    refinement = model.setdefault("refinement", {})
    refinement["enabled"] = True
    refinement["type"] = head

    _set_conditioning(refinement, stage)
    refinement.setdefault("temporal", {})["backend"] = backend
    refinement["temporal"]["mode"] = "causal"

    # Spatial attention is a *separate* factor from the temporal backend.
    refinement.setdefault("unet", {})["bottleneck_attention"] = bool(bottleneck_attention)

    # The legacy post-sampling Mamba adapter stays off in every variant: the
    # temporal comparison here is about conditioning the generative process,
    # not about filtering already-sampled frames.
    model.setdefault("mamba_temporal", {})["enabled"] = False
    model["mamba_temporal_enabled"] = False

    config["_ablation"] = {
        "name": f"{head}__{name}",
        "head": head,
        "conditioning_stage": stage,
        "temporal_backend": backend,
        "bottleneck_attention": bool(bottleneck_attention),
        "temporal_mode": "causal",
        "derived_from": source_config or BASE_CONFIG.name,
    }
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=BASE_CONFIG, help="base O3 configuration"
    )
    parser.add_argument("--out", type=Path, help="directory to write configurations to")
    parser.add_argument(
        "--heads", nargs="+", default=list(HEADS), choices=list(HEADS)
    )
    parser.add_argument(
        "--list", action="store_true", help="print the variant grid and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        for head in args.heads:
            for name, stage, backend, attention in VARIANTS:
                print(
                    f"{head}__{name}\tstage={stage}\tbackend={backend}\t"
                    f"bottleneck_attention={attention}"
                )
        return 0

    if args.out is None:
        parser.error("--out is required unless --list is given")

    base = yaml.safe_load(args.base.read_text())
    if not isinstance(base, dict):
        parser.error("--base must contain a YAML mapping")
    base_root = Path(base.get("paths", {}).get("project_root", ".")).expanduser()
    if not base_root.is_absolute():
        base_root = args.base.resolve().parent / base_root
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for head in args.heads:
        for name, stage, backend, attention in VARIANTS:
            config = build_variant(
                base,
                head=head,
                name=name,
                stage=stage,
                backend=backend,
                bottleneck_attention=attention,
                source_config=args.base.name,
            )
            config["paths"]["project_root"] = os.path.relpath(base_root.resolve(), args.out.resolve())
            path = args.out / f"O3_{head}__{name}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            written += 1
            print(f"wrote {path}")
    print(f"{written} configuration(s) written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
