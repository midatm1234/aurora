"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Run non-overwriting representative refinement inference from a checkpoint.

This utility is intended for sampler/formulation ablations, not training. It
reconstructs the architecture from the checkpoint contract, runs selected
initializations, disables output-only smoothing, and writes one deterministic
rollout per initialization to a new directory.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aurora import (  # noqa: E402
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraPretrained,
    AuroraSmallPretrained,
    AuroraWave,
)
from finetune import aurora_finetune_utils as ft  # noqa: E402
from finetune.flow_refine import AuroraFlowRefine  # noqa: E402


MODEL_REGISTRY = {
    "aurora": Aurora,
    "aurora_pretrained": AuroraPretrained,
    "aurora_small_pretrained": AuroraSmallPretrained,
    "aurora_12h_pretrained": Aurora12hPretrained,
    "aurora_highres": AuroraHighRes,
    "aurora_air_pollution": AuroraAirPollution,
    "aurora_wave": AuroraWave,
}


def _timestamp_tag(value: Any) -> str:
    return str(np.datetime64(value, "s")).replace("-", "").replace(":", "")


def _checkpoint_architecture_config(
    current: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Use current paths/data but checkpoint-saved model architecture."""
    result = copy.deepcopy(current)
    saved = checkpoint.get("config")
    if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
        raise ValueError("Checkpoint has no saved model configuration.")
    architecture_keys = (
        "model_variant",
        "patch_size",
        "model_kwargs",
        "conv_refine_enabled",
        "conv_refine_hidden",
        "flow_refine_enabled",
        "flow_refine_contract_version",
        "flow_refine_hidden",
        "flow_refine_time_dim",
        "flow_refine_doy_cond",
        "flow_refine_lead_time_cond",
        "flow_refine_lead_time_scale_hours",
        "flow_refine_lon_encoding",
        "flow_refine_residual_zscore",
        "flow_refine_res_std_momentum",
        "mamba_temporal_enabled",
        "mamba_temporal_channels",
        "mamba_temporal_state",
        "mamba_temporal_layers",
        "mamba_temporal_conv",
        "mamba_temporal_expand",
        "lon_periodic",
        "lon_periodic_resolved",
        "longitude_grid_signature",
        "mixed_precision",
    )
    saved_model = saved["model"]
    for key in architecture_keys:
        if key in saved_model:
            result["model"][key] = copy.deepcopy(saved_model[key])
        elif key in {
            "flow_refine_contract_version",
            "flow_refine_doy_cond",
            "flow_refine_lead_time_cond",
            "flow_refine_lead_time_scale_hours",
            "flow_refine_lon_encoding",
            "flow_refine_residual_zscore",
            "mamba_temporal_enabled",
        }:
            result["model"][key] = (
                1
                if key == "flow_refine_contract_version"
                else 72.0
                if key == "flow_refine_lead_time_scale_hours"
                else False
            )
    result.setdefault("inference", {})["require_validated_checkpoint"] = False
    result.setdefault("rollout", {})["smooth_sigma"] = 0.0
    return result


def _install_legacy_target_zero_endpoint(model: AuroraFlowRefine) -> None:
    """Reproduce the former incorrect t=1, x_t=0 deterministic query."""

    def legacy(
        self: AuroraFlowRefine,
        cond_norm: torch.Tensor,
        head: torch.nn.Module,
        doy: torch.Tensor | None = None,
        lead_hours: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, _, h, w = cond_norm.shape
        x_t = torch.zeros(
            n, 1, h, w, device=cond_norm.device, dtype=cond_norm.dtype,
        )
        t = torch.ones(n, device=cond_norm.device, dtype=torch.float32)
        return head(
            x_t,
            t,
            cond_norm,
            doy=doy,
            lead_hours=lead_hours,
            coords=coords,
        )

    model._deterministic_residual = MethodType(legacy, model)


def run_ablation(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    endpoint: str,
    dataset_split: str,
    initialization_indices: list[int],
    device: torch.device,
) -> list[Path]:
    """Run selected deterministic rollouts and return their paths."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory; ablation outputs never overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    cfg = _checkpoint_architecture_config(ft.load_config(config_path), checkpoint)
    if dataset_split not in {"train", "test"}:
        raise ValueError("dataset_split must be `train` or `test`.")
    dataset_path = cfg["paths"][f"{dataset_split}_data_path"]
    source_ds = ft.open_dataset(dataset_path, cfg)
    static_path = cfg["paths"].get("static_data_path")
    if static_path:
        source_ds = ft.merge_external_static_vars(source_ds, static_path, cfg)
    resolved_specs = ft.resolve_variable_specs(source_ds, cfg)
    lon_name = str(cfg.get("data", {}).get("lon_dim", "longitude"))
    model_longitude = source_ds[lon_name].values
    model_cfg = cfg["model"]
    variant = str(model_cfg.get("model_variant", "aurora_pretrained")).lower()
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model variant: {variant}")
    variable_config = ft.derive_model_variable_config(resolved_specs, cfg)
    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    if "patch_size" in model_cfg and "patch_size" not in model_kwargs:
        model_kwargs["patch_size"] = int(model_cfg["patch_size"])
    model_kwargs["autocast"] = str(
        model_cfg.get("mixed_precision", "none")
    ).lower() in {"bf16", "bfloat16", "fp16"}
    model = MODEL_REGISTRY[variant](
        surf_vars=variable_config["surf_vars"],
        static_vars=variable_config["static_vars"],
        atmos_vars=variable_config["atmos_vars"],
        **model_kwargs,
    )
    model = ft.maybe_wrap_conv_refine(
        model, cfg, resolved_specs, lon=model_longitude,
    )
    model = ft.maybe_wrap_flow_refine(
        model, cfg, resolved_specs, lon=model_longitude,
    )
    if not isinstance(model, AuroraFlowRefine):
        raise TypeError("Ablation requires an AuroraFlowRefine checkpoint.")
    ft.validate_checkpoint_longitude(model, checkpoint)
    ft.validate_checkpoint_refinement_contract(
        model, checkpoint, cfg, resolved_specs,
    )
    model.set_norm_stats(checkpoint["norm_stats"])
    model.sampling_steps = 1
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if endpoint == "legacy-target-zero":
        _install_legacy_target_zero_endpoint(model)
    elif endpoint != "source-mean":
        raise ValueError(
            "endpoint must be `source-mean` or `legacy-target-zero`."
        )
    model = model.to(device)
    model.eval()

    time_name = str(cfg.get("data", {}).get("time_dim", "time"))
    input_steps = int(cfg.get("data", {}).get("input_time_steps", 2))
    n_time = int(source_ds.sizes[time_name])
    valid_anchor_indices = list(range(input_steps - 1, n_time))
    selected_anchors: list[int] = []
    for index in initialization_indices:
        resolved = index if index >= 0 else len(valid_anchor_indices) + index
        if not 0 <= resolved < len(valid_anchor_indices):
            raise IndexError(
                f"Initialization index {index} is outside "
                f"[{-len(valid_anchor_indices)}, {len(valid_anchor_indices)-1}]."
            )
        selected_anchors.append(valid_anchor_indices[resolved])

    paths: list[Path] = []
    try:
        for anchor_index in selected_anchors:
            sample = {
                "anchor_index": anchor_index,
                "history_indices": list(
                    range(anchor_index - input_steps + 1, anchor_index + 1)
                ),
                "target_indices": {},
            }
            initialization = np.datetime64(
                source_ds[time_name].values[anchor_index], "s",
            )
            path = output_dir / (
                f"rollout_predictions_init_{_timestamp_tag(initialization)}.nc"
            )
            predictions = ft.run_rollout(
                model,
                source_ds,
                sample,
                cfg,
                resolved_specs,
                device,
            )
            predictions = [prediction.type(torch.float32) for prediction in predictions]
            dataset = ft.save_predictions(
                predictions,
                path,
                save_netcdf=True,
                resolved_specs=resolved_specs,
                smooth_sigma=0.0,
                patch_size=int(model_cfg.get("patch_size", 3)),
                lon_periodic=bool(getattr(model, "lon_periodic", False)),
            )
            dataset.close()
            paths.append(path)
            print(
                f"Saved {endpoint} rollout for initialization "
                f"{initialization}: {path}"
            )
    finally:
        source_ds.close()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--endpoint",
        choices=("source-mean", "legacy-target-zero"),
        default="source-mean",
    )
    parser.add_argument(
        "--dataset-split",
        choices=("train", "test"),
        default="test",
        help="Dataset whose initialization indices are selected (default: test).",
    )
    parser.add_argument(
        "--initialization-index",
        action="append",
        type=int,
        dest="indices",
        help="Zero-based initialization index; repeatable (default: 0).",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_ablation(
        args.config.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        endpoint=args.endpoint,
        dataset_split=args.dataset_split,
        initialization_indices=args.indices or [0],
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
