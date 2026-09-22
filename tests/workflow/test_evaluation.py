"""Known-value metric and artifact tests; these fixtures are not Aurora forecasts."""
import json

import numpy as np
import pytest

from aurora_workflow.evaluation import (
    _aligned_npz, area_weights, ensemble_metrics, evaluate_arrays,
    execute_evaluation, improvement, point_metrics,
)


def test_known_value_errors_and_degenerate_correlation():
    reference = np.array([[1., 2.], [3., 4.]])
    result = point_metrics(reference + 2, reference, np.ones((2, 2)))
    assert result["mae"] == result["rmse"] == result["bias"] == 2
    assert result["spatial_correlation"] == pytest.approx(1)
    assert point_metrics(np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2)))["spatial_correlation"] is None
    assert point_metrics(np.full((2, 2), np.nan), reference, np.ones((2, 2)))["rmse"] is None


def test_improvement_sign_and_zero_denominator():
    assert improvement(2, 1) == 50
    assert improvement(1, 2) == -100
    assert improvement(0, 1) is None
    assert improvement(float("nan"), 1) is None


def test_cell_area_weighting_not_unweighted_latitude():
    weights = area_weights([0., 60.], [0., 1.])
    assert weights[0].sum() > weights[1].sum()
    assert weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="cyclic"):
        area_weights([20, 10], [0, 180, 360])


def test_empirical_crps_spread_coverage_and_tie_ranks():
    members = np.array([np.zeros((2, 2)), np.full((2, 2), 2.)])
    reference = np.ones((2, 2))
    result = ensemble_metrics(members, reference, np.ones((2, 2)))
    assert result["crps"] == pytest.approx(0.5)
    assert result["spread"] == 1
    assert result["ensemble_mean_rmse"] == 0
    assert result["spread_skill_ratio"] is None
    assert result["intervals"][-1]["coverage"] == 1
    assert result["rank_histogram_fraction"] == [0, 1, 0]
    ties = ensemble_metrics(np.ones((2, 2, 2)), reference, np.ones((2, 2)))
    assert ties["rank_histogram_fraction"] == pytest.approx([1 / 3] * 3)
    with pytest.raises(ValueError, match=">=2"):
        ensemble_metrics(members[:1], reference, np.ones((2, 2)))


def metadata():
    return {"case_cycle": np.array(["2024-07-01T00", "2024-07-01T12"]), "lead_hours": np.array([12, 24]),
            "channel": np.array(["no2_1000", "tcno2"]), "units": np.array(["kg kg-1", "kg m-2"]),
            "lat": np.array([42., 41., 40., 39.]), "lon": np.array([240., 241., 242., 243.]),
            "split": np.array(["test", "test"])}


def test_identical_masks_and_diagnostics():
    rng = np.random.default_rng(23)
    ref = rng.random((2, 2, 2, 4, 4))
    baseline, refined = ref + 1, ref + 2
    refined[0, 0, 0, 0, 0] = np.nan
    result = evaluate_arrays(baseline, refined, ref, metadata())
    assert result["coverage"]["excluded_points"] == 1
    row = result["rows"][0]
    assert row["baseline"]["valid_points"] == row["refined_point"]["valid_points"]
    assert row["improvement_percent"]["rmse"] == pytest.approx(-100)
    assert {v["subset"] for v in result["rows"]} >= {"southern_boundary", "reference_top_decile", "interior"}
    assert "coastal" not in {v["subset"] for v in result["rows"]}
    assert any(v["utc_category"] == "valid_00UTC" for v in result["rows"])


def test_exact_matching_unit_and_cycle_provenance(tmp_path):
    meta = metadata()
    fields = np.ones((2, 2, 2, 4, 4))
    paths = {v: tmp_path / f"{v}.npz" for v in ("reference", "baseline", "refined")}
    for p in paths.values():
        np.savez(p, fields=fields, **meta)
    changed = dict(meta, units=np.array(["ppb", "kg m-2"]))
    np.savez(paths["refined"], fields=fields, **changed)
    with pytest.raises(ValueError, match="units"):
        _aligned_npz(paths)
    np.savez(paths["refined"], fields=fields[:1], **dict(meta, case_cycle=meta["case_cycle"][:1], split=meta["split"][:1]))
    arrays, coverage = _aligned_npz(paths)
    assert arrays["baseline"]["fields"].shape[0] == 1
    assert len(coverage["refined"]["missing_cycles"]) == 1


def test_execution_writes_real_metrics_plot_report_receipt(tmp_path):
    meta = metadata()
    ref = np.random.default_rng(7).random((2, 2, 2, 4, 4))
    for name, fields in (("reference", ref), ("baseline", ref + 2), ("refined", ref + 1)):
        np.savez(tmp_path / f"{name}.npz", fields=fields, **meta)
    result = execute_evaluation({"recipe_id": "cpu-smoke-v1"}, tmp_path)
    assert result["state"] == "succeeded"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["evidence_kind"] == "synthetic_fixture"
    assert metrics["rows"][0]["improvement_percent"]["rmse"] == pytest.approx(50)
    assert (tmp_path / "metrics.svg").read_text().startswith("<svg")
    receipt = json.loads((tmp_path / "evaluation-receipt.json").read_text())
    assert receipt["execution_status"] == "executed"
    assert len(receipt["outputs"]["metrics"]["sha256"]) == 64
