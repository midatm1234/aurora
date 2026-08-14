"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Raw-versus-refined evaluation contract.
"""

from __future__ import annotations

import math

import pytest
import torch
from finetune.refinement.evaluation import compare_raw_and_refined, evaluate_packed, summarize

from tests.refinement_fixtures import build_packing


def make_case(steps: int = 2, batch: int = 1, height: int = 6, width: int = 8):
    packing = build_packing(height, width)
    torch.manual_seed(0)
    total = steps * batch
    truth = torch.randn(total, packing.num_channels, height, width)
    lead_index = torch.arange(steps).repeat_interleave(batch)
    lead_hours = 24.0 * (lead_index + 1).float()
    return packing, truth, lead_index, lead_hours


def test_metrics_are_reported_per_variable_level_and_lead() -> None:
    packing, truth, lead_index, lead_hours = make_case(steps=3)
    prediction = truth + 0.1
    rows = evaluate_packed(
        prediction,
        truth,
        packing=packing,
        lead_index=lead_index,
        lead_hours=lead_hours,
    )
    # 3 rollout steps x (1 surface + 2 pressure levels)
    assert len(rows) == 3 * packing.num_channels
    keys = {(r["variable"], r["level"], r["rollout_step"]) for r in rows}
    assert ("gtco3", None, 1) in keys
    assert ("go3", 500.0, 2) in keys
    assert ("go3", 850.0, 3) in keys
    for row in rows:
        assert row["lead_time_hours"] == 24.0 * row["rollout_step"]
        assert row["units"] in {"kg m-2", "kg kg-1"}
        assert row["bias"] == pytest.approx(0.1, abs=1e-5)
        assert row["mae"] == pytest.approx(0.1, abs=1e-5)
        assert row["rmse"] == pytest.approx(0.1, abs=1e-5)


def test_perfect_prediction_scores_are_ideal() -> None:
    packing, truth, lead_index, lead_hours = make_case()
    rows = evaluate_packed(
        truth.clone(), truth, packing=packing, lead_index=lead_index, lead_hours=lead_hours
    )
    for row in rows:
        assert row["bias"] == pytest.approx(0.0, abs=1e-6)
        assert row["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert row["pattern_correlation"] == pytest.approx(1.0, abs=1e-5)


def test_masked_cells_are_excluded() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    prediction = truth.clone()
    prediction[:, :, 0, 0] = 1000.0
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, :, 0, 0] = False
    rows = evaluate_packed(prediction, truth, packing=packing, lead_index=lead_index, mask=mask)
    for row in rows:
        assert row["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_nonfinite_cells_do_not_contaminate_weighted_means_or_correlation() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    prediction = truth + 0.1
    truth[:, :, 0, 0] = float("nan")
    prediction[:, :, 0, 0] = float("nan")

    rows = evaluate_packed(prediction, truth, packing=packing, lead_index=lead_index)

    for row in rows:
        assert row["bias"] == pytest.approx(0.1, abs=1.0e-5)
        assert row["rmse"] == pytest.approx(0.1, abs=1.0e-5)
        assert math.isfinite(row["pattern_correlation"])


def test_no_valid_cells_are_reported_as_nan_not_zero() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    mask = torch.zeros_like(truth, dtype=torch.bool)

    rows = evaluate_packed(
        truth,
        truth,
        packing=packing,
        lead_index=lead_index,
        mask=mask,
    )

    for row in rows:
        assert math.isnan(row["bias"])
        assert math.isnan(row["mae"])
        assert math.isnan(row["rmse"])


def test_area_weighting_changes_the_score() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    prediction = truth.clone()
    prediction[:, :, 0] += 1.0  # perturb the northernmost row only
    weighted = evaluate_packed(
        prediction, truth, packing=packing, lead_index=lead_index, area_weighted=True
    )
    plain = evaluate_packed(
        prediction, truth, packing=packing, lead_index=lead_index, area_weighted=False
    )
    # cos(90 deg) ~ 0, so the polar row contributes almost nothing when weighted.
    assert weighted[0]["rmse"] < plain[0]["rmse"]


def test_ensemble_scores() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    members = truth.unsqueeze(1) + 0.05 * torch.randn(truth.shape[0], 4, *truth.shape[1:])
    rows = evaluate_packed(
        members.mean(dim=1),
        truth,
        packing=packing,
        lead_index=lead_index,
        members=members,
    )
    for row in rows:
        assert row["ensemble_spread"] > 0
        assert row["crps"] >= 0
        assert math.isfinite(row["spread_skill_ratio"])
        assert "ensemble_mean_bias" in row


def test_nonfinite_ensemble_cell_is_excluded_from_all_ensemble_scores() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    members = truth.unsqueeze(1).repeat(1, 3, 1, 1, 1)
    members[:, 0, :, 0, 0] = float("nan")

    rows = evaluate_packed(
        truth,
        truth,
        packing=packing,
        lead_index=lead_index,
        members=members,
    )

    for row in rows:
        assert row["ensemble_mean_bias"] == pytest.approx(0.0, abs=1.0e-6)
        assert row["ensemble_mean_rmse"] == pytest.approx(0.0, abs=1.0e-6)
        assert row["ensemble_spread"] == pytest.approx(0.0, abs=1.0e-6)
        assert row["crps"] == pytest.approx(0.0, abs=1.0e-6)


def test_crps_of_a_perfect_deterministic_ensemble_is_zero() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    members = truth.unsqueeze(1).repeat(1, 3, 1, 1, 1)
    rows = evaluate_packed(
        truth.clone(), truth, packing=packing, lead_index=lead_index, members=members
    )
    for row in rows:
        assert row["crps"] == pytest.approx(0.0, abs=1e-6)
        assert row["ensemble_spread"] == pytest.approx(0.0, abs=1e-6)


def test_regions_are_reported_separately() -> None:
    packing, truth, lead_index, _ = make_case(steps=1, height=6, width=8)
    north = torch.zeros(1, 1, 6, 8, dtype=torch.bool)
    north[..., :3, :] = True
    rows = compare_raw_and_refined(
        truth,
        packing=packing,
        candidates={"raw": truth + 0.2},
        lead_index=lead_index,
        regions={"north": north, "all": torch.ones(1, 1, 6, 8, dtype=torch.bool)},
    )
    regions = {row["region"] for row in rows}
    assert regions == {"north", "all"}


def test_summary_flags_a_bias_only_improvement_as_a_tradeoff() -> None:
    packing, truth, lead_index, lead_hours = make_case(steps=1)
    raw = truth + 0.1  # small constant bias, zero-variance error
    # Lower mean bias but much larger random error: a classic false improvement.
    torch.manual_seed(1)
    refined = truth + 0.01 + 0.5 * torch.randn_like(truth)
    rows = compare_raw_and_refined(
        truth,
        packing=packing,
        candidates={"raw": raw, "diffusion_unet": refined},
        lead_index=lead_index,
        lead_hours=lead_hours,
    )
    summary = summarize(rows, baseline="raw")
    entry = summary["models"]["diffusion_unet"]
    assert entry["bias_improved"] is True
    assert entry["degraded_groups"], "RMSE degradation must be reported"
    assert entry["improvement"] is False


def test_summary_accepts_a_genuine_improvement() -> None:
    packing, truth, lead_index, lead_hours = make_case(steps=1)
    raw = truth + 0.2
    refined = truth + 0.02
    rows = compare_raw_and_refined(
        truth,
        packing=packing,
        candidates={"raw": raw, "flow_matching_unet": refined},
        lead_index=lead_index,
        lead_hours=lead_hours,
    )
    summary = summarize(rows, baseline="raw")
    entry = summary["models"]["flow_matching_unet"]
    assert entry["degraded_groups"] == []
    assert entry["improvement"] is True


def test_summary_requires_the_baseline() -> None:
    packing, truth, lead_index, _ = make_case(steps=1)
    rows = compare_raw_and_refined(
        truth, packing=packing, candidates={"refined": truth}, lead_index=lead_index
    )
    with pytest.raises(KeyError, match="Baseline"):
        summarize(rows, baseline="raw")


def test_shape_mismatch_is_rejected() -> None:
    packing, truth, _, _ = make_case(steps=1)
    with pytest.raises(ValueError, match="same shape"):
        evaluate_packed(truth[:, :1], truth, packing=packing)


def test_member_shape_mismatch_is_rejected() -> None:
    packing, truth, _, _ = make_case(steps=1)
    members = torch.zeros(1, 2, packing.num_channels + 1, *truth.shape[-2:])
    with pytest.raises(ValueError, match="members must have non-member shape"):
        evaluate_packed(truth, truth, packing=packing, members=members)


def test_one_lead_group_cannot_mix_physical_lead_times() -> None:
    packing, truth, _, _ = make_case(steps=1, batch=2)
    with pytest.raises(ValueError, match="mixes physical lead times"):
        evaluate_packed(
            truth,
            truth,
            packing=packing,
            lead_index=torch.tensor([0, 0]),
            lead_hours=torch.tensor([12.0, 24.0]),
        )
