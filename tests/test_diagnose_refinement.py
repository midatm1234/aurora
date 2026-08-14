"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Regression tests for baseline-relative refinement diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from finetune.diagnose_refinement import (
    _aggregate_target_metrics,
    _excess_kurtosis,
    aggregate_case_metrics,
    case_metrics,
    plot_temporal_metrics_by_lead,
    temporal_case_metrics,
    write_summary,
)


def _row(lead_time_hours: float) -> dict[str, object]:
    truth = np.linspace(0.0, 9.0, 100, dtype=np.float64)
    baseline = 1.4 * truth + 2.0
    refined = truth + 0.1
    return {
        **case_metrics(truth, baseline, refined),
        "initialization_time": "2025-01-01T00:00:00",
        "valid_time": "2025-01-01T12:00:00",
        "variable": "no2",
        "level": "1000",
        "lead_time_hours": lead_time_hours,
        "units": "kg kg-1",
    }


def test_case_metrics_cover_distribution_and_upper_tail() -> None:
    row = _row(12.0)

    assert row["truth_p99"] == pytest.approx(8.91)
    assert row["refined_p99_error"] == pytest.approx(0.1)
    assert row["refined_p99_tail_mae"] == pytest.approx(0.1)
    assert row["refined_wasserstein_distance"] == pytest.approx(0.1)
    assert row["refined_p99_exceedance_frequency"] == pytest.approx(0.03)
    assert row["baseline_p99_error"] > row["refined_p99_error"]
    assert row["baseline_wasserstein_distance"] > row["refined_wasserstein_distance"]


def test_excess_kurtosis_is_scale_independent_and_aggregated() -> None:
    truth_shape = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    baseline_shape = np.asarray([-5.0, -1.0, 0.0, 1.0, 5.0])
    for physical_scale in (1.0, 1.0e-200):
        assert _excess_kurtosis(physical_scale * truth_shape) == pytest.approx(-1.3)

    case = case_metrics(
        truth_shape,
        baseline_shape,
        truth_shape,
    )
    assert case["truth_excess_kurtosis"] == pytest.approx(-1.3)
    assert case["refined_excess_kurtosis"] == pytest.approx(-1.3)
    assert case["baseline_excess_kurtosis"] != pytest.approx(-1.3)

    rows = []
    for lead in (12.0, 24.0):
        rows.append(
            {
                **case,
                "initialization_time": "2025-01-01T00:00:00",
                "valid_time": f"2025-01-01T{int(lead):02d}:00:00",
                "variable": "no2",
                "level": "1000",
                "lead_time_hours": lead,
                "units": "kg kg-1",
            }
        )
    lead_frame = aggregate_case_metrics(pd.DataFrame(rows))
    target = _aggregate_target_metrics(lead_frame).iloc[0]

    assert (lead_frame["baseline_excess_kurtosis_error"] > 0.0).all()
    assert (lead_frame["refined_excess_kurtosis_error"] == 0.0).all()
    assert (lead_frame["excess_kurtosis_error_improvement"] > 0.0).all()
    assert target.baseline_excess_kurtosis_error > 0.0
    assert target.refined_excess_kurtosis_error == pytest.approx(0.0)
    assert target.excess_kurtosis_error_improvement == pytest.approx(
        target.baseline_excess_kurtosis_error
    )


def test_aggregation_and_markdown_report_absolute_and_relative_skill(
    tmp_path,
) -> None:
    lead_frame = aggregate_case_metrics(pd.DataFrame([_row(12.0), _row(24.0)]))

    assert len(lead_frame) == 2
    assert (lead_frame["mae_improvement_percent"] > 0.0).all()
    assert (lead_frame["rmse_improvement_percent"] > 0.0).all()
    assert (lead_frame["wasserstein_distance_improvement"] > 0.0).all()
    assert (lead_frame["p99_tail_mae_improvement"] > 0.0).all()

    output = tmp_path / "evaluation_summary.md"
    write_summary(lead_frame, output)
    report = output.read_text()
    assert "## Absolute forecast skill and change from Aurora" in report
    assert "## Distribution and upper-tail agreement" in report
    assert "### Distribution shape" in report
    assert "## Correction behavior" in report
    assert "## Lead-time metrics" in report
    assert "Aurora abs bias" in report


def test_target_summary_recomputes_pooled_improvement() -> None:
    truth = np.linspace(0.0, 9.0, 100, dtype=np.float64)
    second = _row(24.0)
    second.update(case_metrics(truth, truth + 0.2, truth + 0.15))
    lead_frame = aggregate_case_metrics(pd.DataFrame([_row(12.0), second]))

    target = _aggregate_target_metrics(lead_frame).iloc[0]
    expected_baseline_mae = (
        lead_frame.loc[0, "baseline_mae"] + lead_frame.loc[1, "baseline_mae"]
    ) / 2.0
    expected_refined_mae = (
        lead_frame.loc[0, "refined_mae"] + lead_frame.loc[1, "refined_mae"]
    ) / 2.0
    expected_improvement = (
        100.0 * (expected_baseline_mae - expected_refined_mae) / expected_baseline_mae
    )

    assert target.baseline_mae == pytest.approx(expected_baseline_mae)
    assert target.refined_mae == pytest.approx(expected_refined_mae)
    assert target.mae_improvement_percent == pytest.approx(expected_improvement)
    assert target.mae_improvement_percent != pytest.approx(
        lead_frame.mae_improvement_percent.mean()
    )


def test_temporal_case_metrics_match_consecutive_per_hour_changes() -> None:
    for physical_scale in (1.0, 1.0e-15):
        previous = np.zeros(4, dtype=np.float64)
        truth = physical_scale * np.asarray([0.0, 1.0, 2.0, 3.0])
        baseline = 2.0 * truth
        refined = truth.copy()

        metrics = temporal_case_metrics(
            previous,
            previous,
            previous,
            truth,
            baseline,
            refined,
            lead_interval_hours=2.0,
        )

        truth_tendency_rms = physical_scale * np.sqrt(0.875)
        assert metrics["number_of_temporal_valid_points"] == 4
        assert metrics["truth_tendency_rms"] == pytest.approx(truth_tendency_rms)
        assert metrics["baseline_tendency_rms"] == pytest.approx(2.0 * truth_tendency_rms)
        assert metrics["refined_tendency_rms"] == pytest.approx(truth_tendency_rms)
        assert metrics["baseline_tendency_rmse"] == pytest.approx(truth_tendency_rms)
        assert metrics["refined_tendency_rmse"] == pytest.approx(0.0)
        assert metrics["baseline_tendency_spatial_correlation"] == pytest.approx(1.0)
        assert metrics["refined_tendency_spatial_correlation"] == pytest.approx(1.0)
        assert metrics["baseline_tendency_std_ratio"] == pytest.approx(2.0)
        assert metrics["refined_tendency_std_ratio"] == pytest.approx(1.0)
        assert metrics["correction_tendency_amplitude_ratio"] == pytest.approx(1.0)
        assert metrics["correction_tendency_to_residual_tendency_correlation"] == pytest.approx(1.0)


def test_temporal_aggregation_pools_mse_and_ignores_first_leads(
    tmp_path,
) -> None:
    missing = np.asarray([np.nan])
    empty_temporal = temporal_case_metrics(
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        lead_interval_hours=12.0,
    )
    first_a = {**_row(12.0), **empty_temporal}
    first_b = {
        **_row(12.0),
        **empty_temporal,
        "initialization_time": "2025-01-02T00:00:00",
    }

    previous_a = np.zeros(1)
    truth_a = np.zeros(1)
    baseline_a = np.ones(1)
    second_a = {
        **_row(24.0),
        **temporal_case_metrics(
            previous_a,
            previous_a,
            previous_a,
            truth_a,
            baseline_a,
            truth_a,
            lead_interval_hours=1.0,
        ),
    }

    previous_b = np.zeros(3)
    truth_b = np.arange(3, dtype=np.float64)
    baseline_b = truth_b + 3.0
    second_b = {
        **_row(24.0),
        **temporal_case_metrics(
            previous_b,
            previous_b,
            previous_b,
            truth_b,
            baseline_b,
            truth_b,
            lead_interval_hours=1.0,
        ),
        "initialization_time": "2025-01-02T00:00:00",
    }

    lead_frame = aggregate_case_metrics(pd.DataFrame([first_a, first_b, second_a, second_b]))
    first_lead = lead_frame.loc[lead_frame.lead_time_hours == 12.0].iloc[0]
    second_lead = lead_frame.loc[lead_frame.lead_time_hours == 24.0].iloc[0]
    target = _aggregate_target_metrics(lead_frame).iloc[0]

    assert first_lead.number_of_temporal_valid_points == 0
    assert np.isnan(first_lead.baseline_tendency_rmse)
    assert second_lead.number_of_temporal_valid_points == 4
    assert second_lead.number_of_temporal_forecasts == 2
    assert second_lead.baseline_tendency_rmse == pytest.approx(np.sqrt(7.0))
    assert second_lead.refined_tendency_rmse == pytest.approx(0.0)
    assert target.number_of_temporal_valid_points == 4
    assert target.baseline_tendency_rmse == pytest.approx(np.sqrt(7.0))
    assert target.refined_tendency_rmse == pytest.approx(0.0)
    assert target.temporal_rmse_improvement_percent == pytest.approx(100.0)

    output = tmp_path / "evaluation_summary.md"
    write_summary(lead_frame, output)
    assert "## Temporal consistency" in output.read_text()

    plot_temporal_metrics_by_lead(lead_frame, tmp_path)
    assert (tmp_path / "no2_1000_temporal_metrics_by_lead.png").is_file()


@pytest.mark.parametrize("lead_interval_hours", [0.0, -1.0, np.nan])
def test_temporal_case_metrics_reject_non_positive_or_non_finite_intervals(
    lead_interval_hours: float,
) -> None:
    field = np.zeros(1)
    with pytest.raises(ValueError, match="strictly positive"):
        temporal_case_metrics(
            field,
            field,
            field,
            field,
            field,
            field,
            lead_interval_hours=lead_interval_hours,
        )
