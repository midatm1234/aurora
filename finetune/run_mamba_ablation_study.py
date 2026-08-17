"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Controlled paired Mamba-on/off training and rollout generation.

The spatial refiner is trained once for each case, head, and seed. Its exact
checkpoint is then used by both arms. The off arm is evaluated unchanged while
the on arm trains only an optional packed temporal model on chronological
multi-lead sequences. Checkpoint selection uses validation initializations;
held-out test initializations are written once for strict paired evaluation.

The correction contract is invariant throughout:

    correction_target = CAMS_truth - original_Aurora_rollout
    refined = original_Aurora_rollout + spatial_correction + temporal_correction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xarray as xr

from finetune.mamba_temporal import PackedMambaTemporalAdapter
from finetune.refinement.benchmark import (
    CASES,
    DEFAULT_HEADS,
    BenchmarkDataset,
    _predict_refined,
    evaluate_channels,
    evaluate_channels_by_lead,
    grouped_purged_split,
    load_case,
    train_and_evaluate,
)
from finetune.refinement.losses import area_weights_from_latitudes
from finetune.refinement.target_space import NormalizedTargetSpace

SUPPORTED_HEADS = tuple(name for name in DEFAULT_HEADS if name != "direct_regression_transformer")
MANIFEST_CASE_NAMES = {
    "no2_uswest": "NO2_US-WEST_3day_lead",
    "o3_global": "O3_global_3day_lead",
}


@dataclass
class SequenceData:
    """Initialization-major chronological tensors."""

    base: torch.Tensor
    rollout: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    input_valid: torch.Tensor
    lead_hours: torch.Tensor
    initialization_ids: tuple[str, ...]
    initialization_times: np.ndarray
    valid_times: np.ndarray
    flat_indices: tuple[int, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_corpus_hash(data: BenchmarkDataset) -> str:
    """Hash selected source identities without rereading multi-terabyte payloads."""
    identities = []
    for name in data.source_files:
        path = Path(name)
        stat = path.stat()
        identities.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return _json_hash(identities)


def _three_way_split(
    data: BenchmarkDataset,
    *,
    validation_fraction: float,
    test_fraction: float,
    purge_hours: float,
) -> tuple[BenchmarkDataset, BenchmarkDataset, BenchmarkDataset, dict[str, Any]]:
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction and test_fraction must be positive.")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("At least 20% of initialization groups must remain for training.")
    development, test, outer = grouped_purged_split(
        data,
        test_fraction=test_fraction,
        purge_hours=purge_hours,
    )
    inner_fraction = validation_fraction / (1.0 - test_fraction)
    train, validation, inner = grouped_purged_split(
        development,
        test_fraction=inner_fraction,
        purge_hours=purge_hours,
    )
    split = {
        "method": "nested_chronological_initialization_split_with_purge",
        "train_initialization_ids": inner["train_initialization_ids"],
        "validation_initialization_ids": inner["test_initialization_ids"],
        "test_initialization_ids": outer["test_initialization_ids"],
        "inner": inner,
        "outer": outer,
    }
    memberships = [
        set(split["train_initialization_ids"]),
        set(split["validation_initialization_ids"]),
        set(split["test_initialization_ids"]),
    ]
    if (
        memberships[0] & memberships[1]
        or memberships[0] & memberships[2]
        or memberships[1] & memberships[2]
    ):
        raise RuntimeError("Train, validation, and test initialization groups overlap.")
    return train, validation, test, split


def _as_sequences(
    data: BenchmarkDataset,
    base_normalized: torch.Tensor,
) -> SequenceData:
    if base_normalized.shape != data.rollout.shape:
        raise ValueError(
            f"base_normalized expected {tuple(data.rollout.shape)}, got "
            f"{tuple(base_normalized.shape)}."
        )
    groups: dict[str, list[int]] = {}
    group_time: dict[str, np.datetime64] = {}
    for index, initialization_id in enumerate(data.initialization_ids):
        groups.setdefault(str(initialization_id), []).append(index)
        group_time[str(initialization_id)] = np.datetime64(data.initialization_times[index], "ns")
    ordered_ids = sorted(groups, key=lambda key: (group_time[key], key))
    expected_leads = tuple(float(value) for value in data.case.expected_lead_hours)
    ordered_indices: list[list[int]] = []
    ordered_valid_times: list[np.ndarray] = []
    for initialization_id in ordered_ids:
        indices = sorted(groups[initialization_id], key=lambda i: float(data.lead_hours[i]))
        actual = tuple(float(data.lead_hours[index]) for index in indices)
        if actual != expected_leads:
            raise ValueError(
                f"Initialization {initialization_id} expected chronological leads "
                f"{expected_leads}, got {actual}."
            )
        init_values = {np.datetime64(data.initialization_times[index], "ns") for index in indices}
        if len(init_values) != 1:
            raise ValueError(f"Initialization {initialization_id} mixes samples.")
        ordered_indices.append(indices)
        ordered_valid_times.append(data.valid_times[indices].astype("datetime64[ns]"))
    index_tensor = torch.as_tensor(ordered_indices, dtype=torch.long)
    input_valid_flat = (
        torch.isfinite(data.rollout)
        if data.rollout_valid is None
        else data.rollout_valid.to(dtype=torch.bool)
    )
    return SequenceData(
        base=base_normalized[index_tensor],
        rollout=data.rollout[index_tensor],
        target=data.target[index_tensor],
        valid=data.valid[index_tensor].to(dtype=torch.bool),
        input_valid=input_valid_flat[index_tensor] & torch.isfinite(base_normalized[index_tensor]),
        lead_hours=data.lead_hours[index_tensor],
        initialization_ids=tuple(ordered_ids),
        initialization_times=np.asarray(
            [group_time[key] for key in ordered_ids], dtype="datetime64[ns]"
        ),
        valid_times=np.stack(ordered_valid_times),
        flat_indices=tuple(int(value) for value in index_tensor.reshape(-1).tolist()),
    )


def _spatial_prediction(
    result,
    data: BenchmarkDataset,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if result.trained_refiner is None or result.refinement_config is None:
        raise RuntimeError("Spatial benchmark did not retain its exact trained refiner.")
    result.trained_refiner.eval()
    physical, _ = _predict_refined(
        result.trained_refiner,
        result.refinement_config,
        data,
        NormalizedTargetSpace(data.packing),
        batch_size=batch_size,
        device=device,
        seed=seed,
        ensemble_size=1,
        direct_regression=False,
    )
    return NormalizedTargetSpace(data.packing).encode(physical).float()


def _masked_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    area: torch.Tensor,
    extra_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = valid.to(values.dtype) * area
    if extra_weight is not None:
        weights = weights * extra_weight
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


def _temporal_objective(
    corrected: torch.Tensor,
    batch: SequenceData,
    indices: torch.Tensor,
    *,
    area: torch.Tensor,
    tail_threshold: torch.Tensor,
    tail_weight: float,
    tendency_weight: float,
    structure_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch.target[indices].to(corrected.device)
    valid = batch.valid[indices].to(corrected.device)
    target = torch.where(valid, target, torch.zeros_like(target))
    extreme = target >= tail_threshold
    weights = 1.0 + float(tail_weight) * extreme.to(corrected.dtype)
    base_loss = _masked_mean(
        F.smooth_l1_loss(corrected, target, reduction="none", beta=0.5),
        valid,
        area,
        weights,
    )

    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    predicted_tendency = corrected[:, 1:] - corrected[:, :-1]
    target_tendency = target[:, 1:] - target[:, :-1]
    tendency_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_tendency,
            target_tendency,
            reduction="none",
            beta=0.5,
        ),
        adjacent_valid,
        area,
    )

    prediction_anomaly = corrected - corrected.mean(dim=1, keepdim=True)
    target_anomaly = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction_anomaly * target_anomaly).sum(dim=1)
    denominator = (
        (prediction_anomaly.square().sum(dim=1) * target_anomaly.square().sum(dim=1))
        .clamp(min=1.0e-12)
        .sqrt()
    )
    correlation = numerator / denominator
    structure_valid = valid.all(dim=1) & torch.isfinite(correlation)
    structure_area = area[:, 0].expand_as(correlation)
    structure_loss = _masked_mean(
        1.0 - correlation,
        structure_valid,
        structure_area,
    )
    total = (
        base_loss
        + float(tendency_weight) * tendency_loss
        + float(structure_weight) * structure_loss
    )
    diagnostics = {
        "base": float(base_loss.detach()),
        "tendency": float(tendency_loss.detach()),
        "structure": float(structure_loss.detach()),
        "total": float(total.detach()),
    }
    return total, diagnostics


def _validation_score(
    model: PackedMambaTemporalAdapter,
    data: SequenceData,
    *,
    device: torch.device,
    area: torch.Tensor,
    tail_threshold: torch.Tensor,
    tail_weight: float,
    tendency_weight: float,
    structure_weight: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        base = data.base.to(device)
        corrected = model.corrected_sequence(
            base,
            lead_hours=data.lead_hours.to(device),
            valid_cell_mask=data.input_valid.to(device),
        )
        indices = torch.arange(base.shape[0], device="cpu")
        loss, diagnostics = _temporal_objective(
            corrected,
            data,
            indices,
            area=area,
            tail_threshold=tail_threshold,
            tail_weight=tail_weight,
            tendency_weight=tendency_weight,
            structure_weight=structure_weight,
        )
    return float(loss), diagnostics


def _validation_skill_guard(
    model: PackedMambaTemporalAdapter,
    data: SequenceData,
    *,
    device: torch.device,
    relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Require broad validation improvement before deploying Mamba."""
    model.eval()
    with torch.no_grad():
        base = data.base.to(device)
        target = data.target.to(device)
        valid = data.valid.to(device)
        corrected = model.corrected_sequence(
            base,
            lead_hours=data.lead_hours.to(device),
            valid_cell_mask=data.input_valid.to(device),
        )
        area = area_weights_from_latitudes(
            model.packing.lat,
            len(model.packing.lat),
            device=device,
            dtype=torch.float64,
        ).view(1, 1, 1, len(model.packing.lat), 1)
        weights = valid.to(torch.float64) * area
        off_squared = (base.double() - target.double()).square()
        on_squared = (corrected.double() - target.double()).square()
        dimensions = (0, 1, 3, 4)
        denominator = weights.sum(dim=dimensions).clamp(min=1.0)
        off_channel = ((off_squared * weights).sum(dim=dimensions) / denominator).sqrt()
        on_channel = ((on_squared * weights).sum(dim=dimensions) / denominator).sqrt()
        lead_dimensions = (0, 3, 4)
        lead_denominator = weights.sum(dim=lead_dimensions).clamp(min=1.0)
        off_by_lead = ((off_squared * weights).sum(dim=lead_dimensions) / lead_denominator).sqrt()
        on_by_lead = ((on_squared * weights).sum(dim=lead_dimensions) / lead_denominator).sqrt()

    channel_ratio = on_channel / off_channel.clamp(min=1.0e-12)
    channel_lead_ratio = on_by_lead / off_by_lead.clamp(min=1.0e-12)
    finite_channel = torch.isfinite(channel_ratio)
    finite_pairs = torch.isfinite(channel_lead_ratio)
    fraction_channels_improved = float((channel_ratio[finite_channel] < 1.0).float().mean())
    fraction_pairs_improved = float((channel_lead_ratio[finite_pairs] < 1.0).float().mean())
    maximum_degradation_ratio = float(channel_lead_ratio[finite_pairs].max())
    passed = (
        fraction_channels_improved >= 0.5
        and fraction_pairs_improved >= 0.6
        and maximum_degradation_ratio <= 1.0 + float(relative_tolerance)
    )
    return {
        "passed": bool(passed),
        "relative_tolerance": float(relative_tolerance),
        "fraction_channels_improved": fraction_channels_improved,
        "fraction_channel_lead_pairs_improved": fraction_pairs_improved,
        "maximum_channel_lead_rmse_ratio": maximum_degradation_ratio,
        "channel_rmse_off": [float(value) for value in off_channel.cpu()],
        "channel_rmse_on": [float(value) for value in on_channel.cpu()],
        "channel_rmse_ratio": [float(value) for value in channel_ratio.cpu()],
        "channel_lead_rmse_ratio": channel_lead_ratio.cpu().tolist(),
    }


def _fit_temporal(
    train: SequenceData,
    validation: SequenceData,
    *,
    packing,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    channels: int,
    state_dim: int,
    layers: int,
    expansion_factor: int,
    conv_kernel: int,
    dropout: float,
    gate_init: float,
    tail_weight: float,
    tendency_weight: float,
    structure_weight: float,
) -> tuple[PackedMambaTemporalAdapter, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PackedMambaTemporalAdapter(
        packing,
        channels=channels,
        d_state=state_dim,
        n_layers=layers,
        d_conv=conv_kernel,
        expand=expansion_factor,
        dropout=dropout,
        mode="packed_joint",
        gated_fusion=True,
        gate_init=gate_init,
        lead_time_conditioning=True,
        mask_conditioning=True,
        coordinate_conditioning=True,
        causal=True,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1.0e-4)
    steps_per_epoch = max(1, math.ceil(train.base.shape[0] / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * steps_per_epoch),
        eta_min=learning_rate * 0.05,
    )
    area = area_weights_from_latitudes(
        packing.lat,
        len(packing.lat),
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 1, len(packing.lat), 1)
    thresholds = []
    for channel in range(train.target.shape[2]):
        values = train.target[:, :, channel][train.valid[:, :, channel]].float()
        thresholds.append(torch.quantile(values, 0.95))
    tail_threshold = torch.stack(thresholds).to(device).view(1, 1, -1, 1, 1)

    off_model = PackedMambaTemporalAdapter(
        packing,
        channels=channels,
        d_state=state_dim,
        n_layers=layers,
        d_conv=conv_kernel,
        expand=expansion_factor,
        dropout=dropout,
        mode="packed_joint",
        gated_fusion=True,
        gate_init=0.0,
        lead_time_conditioning=True,
        mask_conditioning=True,
        coordinate_conditioning=True,
        causal=True,
    ).to(device)
    off_score, off_diagnostics = _validation_score(
        off_model,
        validation,
        device=device,
        area=area,
        tail_threshold=tail_threshold,
        tail_weight=tail_weight,
        tendency_weight=tendency_weight,
        structure_weight=structure_weight,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 31)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train.base.shape[0], generator=generator)
        epoch_losses = []
        for begin in range(0, order.numel(), batch_size):
            index = order[begin : begin + batch_size]
            base = train.base[index].to(device)
            corrected = model.corrected_sequence(
                base,
                lead_hours=train.lead_hours[index].to(device),
                valid_cell_mask=train.input_valid[index].to(device),
            )
            loss, _ = _temporal_objective(
                corrected,
                train,
                index,
                area=area,
                tail_threshold=tail_threshold,
                tail_weight=tail_weight,
                tendency_weight=tendency_weight,
                structure_weight=structure_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach()))
        score, diagnostics = _validation_score(
            model,
            validation,
            device=device,
            area=area,
            tail_threshold=tail_threshold,
            tail_weight=tail_weight,
            tendency_weight=tendency_weight,
            structure_weight=structure_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_loss": score,
                **{f"validation_{key}": value for key, value in diagnostics.items()},
            }
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Temporal training produced no finite validation checkpoint.")
    model.load_state_dict(best_state, strict=True)
    skill_guard = _validation_skill_guard(
        model,
        validation,
        device=device,
    )
    promoted = best_score < off_score and bool(skill_guard["passed"])
    report = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "off_validation_loss": off_score,
        "off_validation_diagnostics": off_diagnostics,
        "candidate_best_validation_loss": min(row["validation_loss"] for row in history),
        "mamba_on_validation_loss": best_score,
        "promoted": promoted,
        "deployment_recommendation": "enable" if promoted else "disable",
        "validation_skill_guard": skill_guard,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": time.time() - started,
        "upper_concentration_threshold_normalized": [
            float(value) for value in tail_threshold.reshape(-1)
        ],
        "history": history,
        "fusion_strength": [
            float(value) for value in model.core.fusion_strength.detach().cpu().reshape(-1)
        ],
    }

    return model, report


def _fit_temporal_search(
    train: SequenceData,
    validation: SequenceData,
    *,
    packing,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[PackedMambaTemporalAdapter, dict[str, Any]]:
    """Run a bounded validation-only search over three diagnostic candidates."""
    candidates = (
        {
            "name": "compact",
            "channels": 12,
            "state_dim": 4,
            "layers": 1,
            "expansion_factor": 2,
            "conv_kernel": 3,
            "dropout": 0.0,
            "gate_init": 0.0,
            "learning_rate_multiplier": 1.0,
            "tail_weight": 2.0,
            "tendency_weight": 0.10,
            "structure_weight": 0.02,
        },
        {
            "name": "reference",
            "channels": 16,
            "state_dim": 8,
            "layers": 2,
            "expansion_factor": 2,
            "conv_kernel": 3,
            "dropout": 0.0,
            "gate_init": 0.0,
            "learning_rate_multiplier": 1.0,
            "tail_weight": 2.0,
            "tendency_weight": 0.25,
            "structure_weight": 0.05,
        },
        {
            "name": "wider_regularized",
            "channels": 24,
            "state_dim": 16,
            "layers": 2,
            "expansion_factor": 2,
            "conv_kernel": 4,
            "dropout": 0.05,
            "gate_init": 0.0,
            "learning_rate_multiplier": 0.5,
            "tail_weight": 2.0,
            "tendency_weight": 0.50,
            "structure_weight": 0.05,
        },
    )
    trained: list[PackedMambaTemporalAdapter] = []
    reports: list[dict[str, Any]] = []
    for candidate in candidates:
        model, report = _fit_temporal(
            train,
            validation,
            packing=packing,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=(learning_rate * float(candidate["learning_rate_multiplier"])),
            seed=seed,
            device=device,
            channels=int(candidate["channels"]),
            state_dim=int(candidate["state_dim"]),
            layers=int(candidate["layers"]),
            expansion_factor=int(candidate["expansion_factor"]),
            conv_kernel=int(candidate["conv_kernel"]),
            dropout=float(candidate["dropout"]),
            gate_init=float(candidate["gate_init"]),
            tail_weight=float(candidate["tail_weight"]),
            tendency_weight=float(candidate["tendency_weight"]),
            structure_weight=float(candidate["structure_weight"]),
        )
        report = {
            **report,
            "candidate_name": str(candidate["name"]),
            "candidate_config": {
                **candidate,
                "learning_rate": (learning_rate * float(candidate["learning_rate_multiplier"])),
            },
        }
        ratios = np.asarray(
            report["validation_skill_guard"]["channel_lead_rmse_ratio"],
            dtype=np.float64,
        )
        finite_ratios = ratios[np.isfinite(ratios)]
        report["validation_selection_score"] = float(finite_ratios.mean())
        trained.append(model)
        reports.append(report)

    promoted = [index for index, report in enumerate(reports) if report["promoted"]]
    selection_pool = promoted if promoted else list(range(len(reports)))
    selected_index = min(
        selection_pool,
        key=lambda index: float(reports[index]["validation_selection_score"]),
    )
    selected_model = trained[selected_index]
    selected_report = dict(reports[selected_index])
    selected_report["hyperparameter_search"] = {
        "selection_split": "validation",
        "selection_rule": (
            "lowest mean channel-by-lead RMSE ratio among no-harm candidates; "
            "lowest-ratio diagnostic candidate if none pass"
        ),
        "any_candidate_promoted": bool(promoted),
        "selected_candidate": selected_report["candidate_name"],
        "candidates": reports,
    }
    for index, model in enumerate(trained):
        if index != selected_index:
            model.cpu()
    return selected_model, selected_report


def _temporal_prediction(
    model: PackedMambaTemporalAdapter,
    data: SequenceData,
    *,
    device: torch.device,
    lead_permutation: Sequence[int] | None = None,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        base = data.base.to(device)
        leads = data.lead_hours.to(device)
        valid = data.input_valid.to(device)
        if lead_permutation is None:
            return model.corrected_sequence(
                base,
                lead_hours=leads,
                valid_cell_mask=valid,
            ).cpu()
        permutation = torch.as_tensor(lead_permutation, device=device, dtype=torch.long)
        shuffled_correction = model.temporal_residual(
            base[:, permutation],
            lead_hours=leads[:, permutation],
            valid_cell_mask=valid[:, permutation],
            validate_lead_order=False,
        )
        inverse = torch.argsort(permutation)
        # Restore correction fields to chronological positions before scoring.
        return (base + shuffled_correction[:, inverse]).cpu()


def _metric_payload(
    prediction: torch.Tensor,
    sequence: SequenceData,
    *,
    packing,
    label: str,
) -> dict[str, Any]:
    flat_prediction = prediction.reshape(-1, *prediction.shape[2:])
    flat_truth = NormalizedTargetSpace(packing).decode(
        sequence.target.reshape(-1, *sequence.target.shape[2:])
    )
    physical = NormalizedTargetSpace(packing).decode(flat_prediction)
    flat_valid = sequence.valid.reshape(-1, *sequence.valid.shape[2:])
    leads = sequence.lead_hours.reshape(-1)
    area = area_weights_from_latitudes(
        packing.lat,
        len(packing.lat),
        dtype=torch.float64,
    )
    channels = evaluate_channels(
        physical,
        flat_truth,
        flat_valid,
        packing,
        area_weight=area,
        label=label,
    )
    by_lead = evaluate_channels_by_lead(
        physical,
        flat_truth,
        flat_valid,
        leads,
        packing,
        area_weight=area,
        label=label,
    )
    return {
        "channels": {key: value.as_dict() for key, value in channels.items()},
        "per_lead": {
            key: {str(lead): row.as_dict() for lead, row in rows.items()}
            for key, rows in by_lead.items()
        },
    }


def _write_rollout_directory(
    output: Path,
    prediction: torch.Tensor,
    sequence: SequenceData,
    *,
    packing,
    manifest: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    physical = (
        NormalizedTargetSpace(packing)
        .decode(prediction.reshape(-1, *prediction.shape[2:]))
        .reshape_as(prediction)
    )
    for batch_index, initialization_id in enumerate(sequence.initialization_ids):
        fields = packing.unpack(physical[batch_index])
        coords: dict[str, Any] = {
            "time": sequence.valid_times[batch_index],
            "lead_time": ("time", sequence.lead_hours[batch_index].numpy()),
            "latitude": np.asarray(packing.lat),
            "longitude": np.asarray(packing.lon),
        }
        variables: dict[str, Any] = {}
        for name in packing.variables:
            specs = packing.channels_for(name)
            values = fields[name].detach().cpu().numpy()
            if specs[0].kind == "surf":
                variables[name] = (("time", "latitude", "longitude"), values)
            else:
                coords["level"] = np.asarray([spec.level for spec in specs])
                variables[name] = (
                    ("time", "level", "latitude", "longitude"),
                    values,
                )
        dataset = xr.Dataset(
            variables,
            coords=coords,
            attrs={
                "initialization_time": initialization_id,
                "step_hours": float(sequence.lead_hours[batch_index, 0]),
                "mamba_ablation_mode": str(manifest["mode"]),
                "spatial_head": str(manifest["spatial_head"]),
                "seed": int(manifest["seed"]),
                "phase1_checkpoint_sha256": str(manifest["phase1_checkpoint_sha256"]),
                "spatial_checkpoint_sha256": str(manifest["spatial_checkpoint_sha256"]),
                "phase2_checkpoint_sha256": str(manifest.get("phase2_checkpoint_sha256", "")),
                "temporal_config": json.dumps(manifest["temporal_config"], sort_keys=True),
            },
        )
        safe_initialization = initialization_id.replace("-", "").replace(":", "")
        safe_initialization = safe_initialization.replace("T", "T")
        dataset.to_netcdf(output / f"rollout_predictions_init_{safe_initialization}.nc")
        dataset.close()
    (output / "mamba_ablation_manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, allow_nan=False)
    )


def _save_spatial_checkpoint(path: Path, result, packing, seed: int) -> str:
    if result.trained_refiner is None:
        raise RuntimeError("No trained spatial refiner is available.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in result.trained_refiner.state_dict().items()
            },
            "resolved_config": result.resolved_config,
            "packing": packing.to_dict(),
            "seed": seed,
        },
        path,
    )
    return _sha256_file(path)


def run_study(
    *,
    case_name: str,
    heads: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    phase1_checkpoint: Path,
    spatial_epochs: int,
    temporal_epochs: int,
    max_initializations: int,
    lat_stride: int,
    lon_stride: int,
    batch_size: int,
    spatial_learning_rate: float,
    temporal_learning_rate: float,
    validation_fraction: float,
    test_fraction: float,
    purge_hours: float,
    device: str,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    invalid_heads = sorted(set(heads) - set(SUPPORTED_HEADS))
    if invalid_heads:
        raise ValueError(f"Unsupported heads {invalid_heads}; expected {SUPPORTED_HEADS}.")
    torch_device = torch.device(device)
    data = load_case(
        CASES[case_name],
        max_initializations=max_initializations,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        initialization_selection="uniform",
        verbose=True,
    )
    train, validation, test, split = _three_way_split(
        data,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        purge_hours=purge_hours,
    )
    phase1_hash = _sha256_file(phase1_checkpoint)
    raw_hash = _source_corpus_hash(data)
    split_hash = _json_hash(split)
    results: dict[str, Any] = {
        "schema_version": 1,
        "case_name": case_name,
        "phase1_checkpoint": str(phase1_checkpoint.resolve()),
        "phase1_checkpoint_sha256": phase1_hash,
        "raw_rollout_corpus_sha256": raw_hash,
        "raw_rollout_directory": str(Path(data.case.rollout_dir).resolve()),
        "raw_rollout_corpus_hash_method": "selected_absolute_path+size+mtime",
        "phase1_rollout_linkage": {
            "status": "asserted_not_proven",
            "reason": ("cached rollout NetCDF metadata omits its generating checkpoint hash"),
        },
        "data_split_sha256": split_hash,
        "split": split,
        "heads": {},
    }
    for head in heads:
        results["heads"][head] = {}
        for seed in seeds:
            run_root = output_dir / case_name / head / f"seed_{seed}"
            if run_root.exists() and any(run_root.iterdir()) and not allow_overwrite:
                raise FileExistsError(
                    "Refusing to overwrite existing Mamba study artifacts under "
                    f"{run_root}. Choose a new --output-dir or pass "
                    "--allow-overwrite explicitly."
                )
            print(f"[{case_name}] head={head} seed={seed}: train spatial", flush=True)
            spatial_result = train_and_evaluate(
                head,
                train,
                validation,
                epochs=spatial_epochs,
                batch_size=batch_size,
                learning_rate=spatial_learning_rate,
                device=torch_device,
                seed=seed,
                ensemble_size=1,
                verbose=True,
            )
            spatial_path = run_root / "spatial_checkpoint.pt"
            spatial_hash = _save_spatial_checkpoint(
                spatial_path,
                spatial_result,
                data.packing,
                seed,
            )
            base_train = _spatial_prediction(
                spatial_result,
                train,
                batch_size=batch_size,
                device=torch_device,
                seed=seed,
            )
            base_validation = _spatial_prediction(
                spatial_result,
                validation,
                batch_size=batch_size,
                device=torch_device,
                seed=seed,
            )
            base_test = _spatial_prediction(
                spatial_result,
                test,
                batch_size=batch_size,
                device=torch_device,
                seed=seed,
            )
            train_sequence = _as_sequences(train, base_train)
            validation_sequence = _as_sequences(validation, base_validation)
            test_sequence = _as_sequences(test, base_test)
            print(f"[{case_name}] head={head} seed={seed}: train temporal", flush=True)
            temporal, temporal_report = _fit_temporal_search(
                train_sequence,
                validation_sequence,
                packing=data.packing,
                epochs=temporal_epochs,
                batch_size=max(1, batch_size // len(data.case.expected_lead_hours)),
                learning_rate=temporal_learning_rate,
                seed=seed + 10000,
                device=torch_device,
            )
            temporal_path = run_root / "temporal_checkpoint.pt"
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu() for key, value in temporal.state_dict().items()
                    },
                    "spatial_checkpoint_sha256": spatial_hash,
                    "report": temporal_report,
                    "seed": seed,
                },
                temporal_path,
            )
            temporal_hash = _sha256_file(temporal_path)
            on_prediction = _temporal_prediction(
                temporal,
                test_sequence,
                device=torch_device,
            )
            permutation = tuple(
                [0, 2, 1, 4, 3, 5]
                if len(data.case.expected_lead_hours) == 6
                else reversed(range(len(data.case.expected_lead_hours)))
            )
            shuffled_prediction = _temporal_prediction(
                temporal,
                test_sequence,
                device=torch_device,
                lead_permutation=permutation,
            )
            off_prediction = test_sequence.base
            test_initializations = list(test_sequence.initialization_ids)
            selected_temporal = temporal_report["candidate_config"]
            temporal_config = {
                "enabled": True,
                "mode": "packed_joint",
                "causal": True,
                "selected_candidate": temporal_report["candidate_name"],
                "channels": int(selected_temporal["channels"]),
                "state_dim": int(selected_temporal["state_dim"]),
                "num_layers": int(selected_temporal["layers"]),
                "conv_kernel": int(selected_temporal["conv_kernel"]),
                "expansion_factor": int(selected_temporal["expansion_factor"]),
                "dropout": float(selected_temporal["dropout"]),
                "gated_fusion": True,
                "gate_init": float(selected_temporal["gate_init"]),
                "lead_time_conditioning": True,
                "mask_conditioning": True,
                "coordinate_conditioning": True,
                "learning_rate": float(selected_temporal["learning_rate"]),
                "objective": {
                    "base": "huber",
                    "huber_delta": 0.5,
                    "extreme_quantile": 0.95,
                    "tail_weight": float(selected_temporal["tail_weight"]),
                    "tendency_weight": float(selected_temporal["tendency_weight"]),
                    "structure_weight": float(selected_temporal["structure_weight"]),
                },
                "semantic_version": 2,
                "scan_backend": "selective_scan_ref_v1",
            }

            common = {
                "raw_rollout_corpus_hash_method": ("selected_absolute_path+size+mtime"),
                "phase1_rollout_linkage": {
                    "status": "asserted_not_proven",
                    "reason": "cached rollout metadata omits checkpoint hash",
                },
                "schema_version": 1,
                "case_name": MANIFEST_CASE_NAMES[case_name],
                "spatial_head": head,
                "seed": int(seed),
                "phase1_checkpoint_sha256": phase1_hash,
                "raw_rollout_corpus_sha256": raw_hash,
                "raw_rollout_directory": str(Path(data.case.rollout_dir).resolve()),
                "data_split_sha256": split_hash,
                "evaluation_split": "test",
                "spatial_refinement_config_sha256": _json_hash(spatial_result.resolved_config),
                "sampling_configuration": {
                    "deterministic_inference": True,
                    "ensemble_size": 1,
                    "seed": int(seed),
                },
                "normalization_contract": {
                    "space": "aurora_location_scale",
                    "residual_target": "CAMS_truth - original_Aurora_rollout",
                    "reconstruction": (
                        "original_Aurora_rollout + spatial_residual + " "temporal_residual"
                    ),
                },
                "training_schedule": {
                    "spatial_epochs": int(spatial_epochs),
                    "temporal_control_epochs": int(temporal_epochs),
                },
                "selected_test_initializations": test_initializations,
                "spatial_checkpoint_sha256": spatial_hash,
                "trajectory_policy": {
                    "source": "cached_raw_rollout",
                    "aurora_autoregressive_feedback": False,
                },
                "spatial_parameters_frozen": True,
            }
            manifests = {
                "off": {
                    **common,
                    "mode": "off",
                    "temporal_config": {
                        "enabled": False,
                        "semantic_version": 2,
                    },
                },
                "on": {
                    **common,
                    "mode": "on",
                    "temporal_training_split": "train",
                    "checkpoint_selection_split": "validation",
                    "phase2_checkpoint_sha256": temporal_hash,
                    "temporal_promoted": bool(temporal_report["promoted"]),
                    "temporal_config": temporal_config,
                },
                "shuffled": {
                    **common,
                    "mode": "shuffled",
                    "temporal_training_split": "train",
                    "checkpoint_selection_split": "validation",
                    "phase2_checkpoint_sha256": temporal_hash,
                    "lead_permutation": list(permutation),
                    "temporal_promoted": bool(temporal_report["promoted"]),
                    "temporal_config": temporal_config,
                },
            }
            for mode, prediction in (
                ("off", off_prediction),
                ("on", on_prediction),
                ("shuffled", shuffled_prediction),
            ):
                _write_rollout_directory(
                    run_root / mode,
                    prediction,
                    test_sequence,
                    packing=data.packing,
                    manifest=manifests[mode],
                )
            baseline_sequence = SequenceData(
                base=test_sequence.rollout,
                rollout=test_sequence.rollout,
                target=test_sequence.target,
                valid=test_sequence.valid,
                input_valid=test_sequence.input_valid,
                lead_hours=test_sequence.lead_hours,
                initialization_ids=test_sequence.initialization_ids,
                initialization_times=test_sequence.initialization_times,
                valid_times=test_sequence.valid_times,
                flat_indices=test_sequence.flat_indices,
            )
            seed_payload = {
                "spatial_checkpoint": str(spatial_path.resolve()),
                "spatial_checkpoint_sha256": spatial_hash,
                "temporal_checkpoint": str(temporal_path.resolve()),
                "temporal_checkpoint_sha256": temporal_hash,
                "temporal_training": temporal_report,
                "metrics": {
                    "aurora": _metric_payload(
                        test_sequence.rollout,
                        baseline_sequence,
                        packing=data.packing,
                        label="aurora",
                    ),
                    "off": _metric_payload(
                        off_prediction,
                        test_sequence,
                        packing=data.packing,
                        label="mamba_off",
                    ),
                    "on": _metric_payload(
                        on_prediction,
                        test_sequence,
                        packing=data.packing,
                        label="mamba_on",
                    ),
                    "shuffled": _metric_payload(
                        shuffled_prediction,
                        test_sequence,
                        packing=data.packing,
                        label="mamba_shuffled",
                    ),
                },
                "rollout_directories": {
                    mode: str((run_root / mode).resolve()) for mode in ("off", "on", "shuffled")
                },
            }
            results["heads"][head][str(seed)] = seed_payload
            (run_root / "study_result.json").write_text(
                json.dumps(seed_payload, indent=2, allow_nan=False)
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{case_name}_paired_study.json"
    summary_path.write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--heads", nargs="+", default=list(SUPPORTED_HEADS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--output-dir", type=Path, default=Path("finetune/outputs/mamba_ablation"))
    parser.add_argument("--phase1-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-epochs", type=int, default=12)
    parser.add_argument("--temporal-epochs", type=int, default=12)
    parser.add_argument("--max-initializations", type=int, default=36)
    parser.add_argument("--lat-stride", type=int, default=1)
    parser.add_argument("--lon-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--spatial-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--temporal-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--purge-hours", type=float, default=72.0)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Explicitly allow replacing files in an existing case/head/seed directory.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    run_study(
        case_name=args.case,
        heads=args.heads,
        seeds=args.seeds,
        output_dir=args.output_dir,
        phase1_checkpoint=args.phase1_checkpoint,
        spatial_epochs=args.spatial_epochs,
        temporal_epochs=args.temporal_epochs,
        max_initializations=args.max_initializations,
        lat_stride=args.lat_stride,
        lon_stride=args.lon_stride,
        batch_size=args.batch_size,
        spatial_learning_rate=args.spatial_learning_rate,
        temporal_learning_rate=args.temporal_learning_rate,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        purge_hours=args.purge_hours,
        allow_overwrite=args.allow_overwrite,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
