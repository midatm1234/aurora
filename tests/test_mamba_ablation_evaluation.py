"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from finetune.evaluate_mamba_ablation import (
    MetricSpec,
    _hotspot_trajectory,
    _weighted_distribution_distances,
    build_study_pairs,
    chronological_purged_split,
    evaluate_mamba_ablation,
    paired_bootstrap_summary,
    paired_field_metrics,
    paired_tendency_metrics,
    select_initializations,
    sha256_file,
    temporal_sequence_metrics,
    validate_study_manifests,
)


def test_field_metrics_cover_structure_distribution_tails_and_cell_pairing() -> None:
    truth = np.asarray([[0.0, 1.0, 2.0], [3.0, 5.0, 9.0]])
    off = np.asarray([[0.0, 0.0, 4.0], [1.0, 8.0, 5.0]])
    on = truth.copy()
    shuffled = truth + np.asarray([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    metrics = paired_field_metrics(
        truth,
        {"aurora": truth + 2.0, "off": off, "on": on, "shuffled": shuffled},
        np.ones_like(truth),
        longitude_periodic=False,
    )

    for percentile in ("p50", "p75", "p90", "p95", "p99", "p99_9"):
        assert f"truth_{percentile}" in metrics
        assert f"on_{percentile}_abs_error" in metrics
    assert metrics["on_rmse"] < metrics["off_rmse"]
    assert metrics["on_centered_rmse"] == pytest.approx(metrics["on_pattern_rmse"])
    assert metrics["on_anomaly_correlation"] == pytest.approx(metrics["on_spatial_correlation"])
    assert metrics["on_gradient_rmse"] < metrics["off_gradient_rmse"]
    assert metrics["on_wasserstein_distance"] < metrics["off_wasserstein_distance"]
    assert metrics["on_cdf_distance"] < metrics["off_cdf_distance"]
    assert metrics["fraction_cells_on_better_than_off"] > 0.5
    assert metrics["fraction_cells_on_better_than_aurora"] == pytest.approx(1.0)
    assert 0.0 <= metrics["fraction_cells_on_better_than_shuffled"] <= 1.0


def test_weighted_empirical_distribution_distances_are_exact() -> None:
    wasserstein, cdf = _weighted_distribution_distances(
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 2.0]),
        np.ones(2),
    )
    assert wasserstein == pytest.approx(1.0)
    assert cdf == pytest.approx(0.5)


def test_hotspot_trajectory_uses_longest_lead_cams_maximum() -> None:
    truth_by_lead = [
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.asarray([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]),
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 99.0]]),
    ]
    lead_fields = []
    for lead, truth in zip((12.0, 24.0, 36.0), truth_by_lead, strict=True):
        lead_fields.append(
            {
                "lead_time_hours": lead,
                "fields": {
                    "truth": truth,
                    "aurora": truth + 2.0,
                    "off": truth + 1.0,
                    "on": truth.copy(),
                    "shuffled": truth - 1.0,
                },
            }
        )
    trajectory = _hotspot_trajectory(
        {
            "latitude": np.asarray([35.0, 36.0]),
            "longitude": np.asarray([-120.0, -119.0, -118.0]),
            "lead_fields": lead_fields,
        }
    )

    assert trajectory is not None
    assert trajectory["latitude_index"] == 1
    assert trajectory["longitude_index"] == 2
    assert trajectory["latitude"] == pytest.approx(36.0)
    assert trajectory["longitude"] == pytest.approx(-118.0)
    np.testing.assert_array_equal(
        trajectory["lead_time_hours"],
        np.asarray([12.0, 24.0, 36.0]),
    )
    np.testing.assert_array_equal(
        trajectory["values"]["truth"],
        np.asarray([6.0, 16.0, 99.0]),
    )
    np.testing.assert_array_equal(
        trajectory["values"]["aurora"],
        np.asarray([8.0, 18.0, 101.0]),
    )


def test_tendency_and_temporal_sequence_diagnostics_detect_ordered_skill() -> None:
    previous_truth = np.zeros((2, 3))
    truth = np.asarray([[0.0, 1.0, 3.0], [1.0, 2.0, 5.0]])
    previous = {
        "off": np.zeros((2, 3)),
        "on": np.zeros((2, 3)),
        "shuffled": np.zeros((2, 3)),
    }
    candidates = {
        "off": np.zeros((2, 3)),
        "on": truth.copy(),
        "shuffled": -truth,
    }
    metrics = paired_tendency_metrics(
        previous_truth,
        previous,
        truth,
        candidates,
        np.ones_like(truth),
        12.0,
    )
    assert metrics["on_tendency_mse"] == pytest.approx(0.0)
    assert metrics["on_tendency_rmse"] == pytest.approx(0.0)
    assert metrics["on_tendency_spatial_correlation"] == pytest.approx(1.0)
    assert metrics["on_correction_tendency_amplitude_ratio"] == pytest.approx(1.0)
    assert metrics["on_correction_tendency_amplitude_ratio_error"] == pytest.approx(0.0)
    assert metrics["on_correction_tendency_residual_correlation"] == pytest.approx(1.0)

    sequence = np.stack(
        [
            truth * 0.25,
            truth * 0.75 + 0.1,
            truth * 1.5 + 0.4,
            truth * 2.5 + 0.9,
        ]
    )
    temporal = temporal_sequence_metrics(
        sequence,
        {
            "off": sequence[::-1],
            "on": sequence.copy(),
            "shuffled": np.roll(sequence, 1, axis=0),
        },
        np.ones_like(truth),
    )
    assert temporal["on_temporal_correlation"] == pytest.approx(1.0)
    assert temporal["on_lag1_autocorrelation_abs_error"] == pytest.approx(0.0)
    assert temporal["off_temporal_correlation"] < 0.0


def test_uniform_initialization_selection_spans_full_period() -> None:
    values = list(range(10))
    assert select_initializations(values, 3, mode="uniform") == [0, 4, 9]
    assert select_initializations(values, 3, mode="first") == [0, 1, 2]
    with pytest.raises(ValueError, match="at least one"):
        select_initializations(values, 0, mode="uniform")


def test_chronological_split_groups_dates_and_applies_purge() -> None:
    times = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-02-10"),
        np.timedelta64(1, "D"),
    ).astype("datetime64[ns]")
    split = chronological_purged_split(
        [int(value.astype(np.int64)) for value in times],
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_hours=72,
    )
    train = np.asarray(split["train"], dtype="datetime64[ns]")
    validation = np.asarray(split["validation"], dtype="datetime64[ns]")
    test = np.asarray(split["test"], dtype="datetime64[ns]")
    assert train.max() < validation.min() - np.timedelta64(72, "h")
    assert validation.max() < test.min() - np.timedelta64(72, "h")
    assert set(split) == {"train", "validation", "test"}


def _manifest(
    mode: str,
    *,
    phase1: str = "a" * 64,
    spatial: str = "e" * 64,
    phase2: str = "d" * 64,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "case_name": "NO2_US-WEST_3day_lead",
        "spatial_head": "flow_matching",
        "seed": 3,
        "phase1_checkpoint_sha256": phase1,
        "spatial_checkpoint_sha256": spatial,
        "raw_rollout_corpus_sha256": "b" * 64,
        "raw_rollout_corpus_hash_method": "selected_absolute_path+size+mtime",
        "phase1_rollout_linkage": {
            "status": "asserted_not_proven",
        },
        "data_split_sha256": "c" * 64,
        "evaluation_split": "test",
        "selected_test_initializations": [
            "2024-08-01T00:00:00",
            "2024-08-02T00:00:00",
        ],
        "trajectory_policy": {
            "source": "cached_raw_rollout",
            "aurora_autoregressive_feedback": False,
        },
    }
    if mode in {"on", "shuffled"}:
        manifest.update(
            {
                "spatial_parameters_frozen": True,
                "temporal_training_split": "train",
                "checkpoint_selection_split": "validation",
                "phase2_checkpoint_sha256": phase2,
            }
        )
    if mode == "shuffled":
        manifest["lead_permutation"] = [1, 0, 2]
    return manifest


def test_strict_manifests_enforce_two_stage_and_trajectory_fairness() -> None:
    manifests = [_manifest("off"), _manifest("on"), _manifest("shuffled")]
    result = validate_study_manifests(manifests, require_shuffled=True)
    assert set(result[3]) == {"off", "on", "shuffled"}

    bad = [_manifest("off"), _manifest("on")]
    bad[1]["trajectory_policy"] = {
        "source": "cached_raw_rollout",
        "aurora_autoregressive_feedback": True,
    }
    with pytest.raises(ValueError, match="trajectory_policy"):
        validate_study_manifests(bad, require_shuffled=False)


def test_build_study_pairs_hashes_supplied_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "phase1.ckpt"
    checkpoint.write_bytes(b"same phase one bytes")
    digest = sha256_file(checkpoint)
    spatial_checkpoint = tmp_path / "spatial_checkpoint.pt"
    spatial_checkpoint.write_bytes(b"same spatial bytes")
    spatial_digest = sha256_file(spatial_checkpoint)
    phase2_checkpoint = tmp_path / "temporal_checkpoint.pt"
    phase2_checkpoint.write_bytes(b"same temporal bytes")
    phase2_digest = sha256_file(phase2_checkpoint)
    directories = {mode: tmp_path / mode for mode in ("off", "on", "shuffled")}
    for mode, directory in directories.items():
        directory.mkdir()
        (directory / "mamba_ablation_manifest.json").write_text(
            json.dumps(
                _manifest(
                    mode,
                    phase1=digest,
                    spatial=spatial_digest,
                    phase2=phase2_digest,
                )
            )
        )

    pairs = build_study_pairs(
        [directories["off"]],
        [directories["on"]],
        [directories["shuffled"]],
        [checkpoint],
        unsafe_skip_provenance_checks=False,
    )
    assert len(pairs) == 1
    assert pairs[0].phase1_checkpoint_sha256 == digest
    assert pairs[0].spatial_checkpoint_sha256 == spatial_digest
    assert pairs[0].phase2_checkpoint_sha256 == phase2_digest

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        build_study_pairs(
            [directories["off"]],
            [directories["on"]],
            [directories["shuffled"]],
            [checkpoint],
            unsafe_skip_provenance_checks=False,
        )


def test_paired_bootstrap_clusters_initializations_and_pools_rmse_squared() -> None:
    frame = pd.DataFrame(
        {
            "group": ["x"] * 4,
            "initialization_time": ["a", "a", "b", "b"],
            "off_rmse": [1.0, 1.0, 3.0, 3.0],
            "on_rmse": [0.0, 0.0, 0.0, 0.0],
        }
    )
    summary = paired_bootstrap_summary(
        frame,
        [MetricSpec("rmse", "RMSE", False, True)],
        group_keys=["group"],
        reference="off",
        candidate="on",
        resamples=500,
        block_length=1,
        seed=5,
    ).iloc[0]
    assert summary["number_of_seeds"] == 1
    assert summary["number_of_initializations"] == 2
    assert summary["number_of_seed_initialization_pairs"] == 2
    assert summary["number_of_rows"] == 4
    assert summary["reference_mean"] == pytest.approx(math.sqrt(5.0))
    assert summary["candidate_mean"] == pytest.approx(0.0)
    assert summary["mean_improvement"] == pytest.approx(math.sqrt(5.0))
    assert summary["fraction_cases_candidate_better"] == pytest.approx(1.0)


def test_bootstrap_resamples_seed_before_time_blocks() -> None:
    frame = pd.DataFrame(
        {
            "group": ["x"] * 4,
            "seed": [0, 0, 1, 1],
            "initialization_time": ["a", "b", "a", "b"],
            "off_rmse": [1.0, 1.0, 9.0, 9.0],
            "on_rmse": [0.0, 0.0, 0.0, 0.0],
        }
    )
    summary = paired_bootstrap_summary(
        frame,
        [MetricSpec("rmse", "RMSE", False, True)],
        group_keys=["group"],
        reference="off",
        candidate="on",
        resamples=1000,
        block_length=1,
        seed=11,
    ).iloc[0]

    assert summary["number_of_seeds"] == 2
    assert summary["number_of_initializations"] == 2
    assert summary["number_of_seed_initialization_pairs"] == 4
    assert summary["reference_mean"] == pytest.approx(math.sqrt(41.0))
    assert summary["ci_lower"] == pytest.approx(1.0)
    assert summary["ci_upper"] == pytest.approx(9.0)


def test_scale_independent_guards_work_at_tiny_magnitudes() -> None:
    scale = 1.0e-30
    truth = scale * np.asarray([[0.0, 1.0], [2.0, 4.0]])
    metrics = paired_field_metrics(
        truth,
        {
            "off": truth[::-1],
            "on": truth.copy(),
        },
        np.ones_like(truth),
    )
    assert metrics["on_spatial_correlation"] == pytest.approx(1.0)
    assert metrics["on_rmse"] == pytest.approx(0.0, abs=0.0)
    assert metrics["fraction_cells_on_better_than_off"] > 0.0


def test_end_to_end_paired_netcdf_evaluation_writes_artifacts(
    tmp_path: Path,
) -> None:
    latitude = np.asarray([35.0, 36.0])
    longitude = np.asarray([-120.0, -119.0, -118.0])
    initializations = [
        np.datetime64("2024-08-01T00:00:00", "ns"),
        np.datetime64("2024-08-03T00:00:00", "ns"),
    ]
    lead_hours = np.asarray([12, 24, 36])
    valid_times: list[np.datetime64] = []
    truth_fields: list[np.ndarray] = []
    base_field = np.asarray([[0.0, 1.0, 2.0], [1.0, 3.0, 6.0]])
    for initialization_index, initialization in enumerate(initializations):
        for lead_index, lead in enumerate(lead_hours):
            valid_times.append(initialization + np.timedelta64(int(lead), "h"))
            truth_fields.append(base_field + 0.5 * lead_index + 0.25 * initialization_index)
    truth_path = tmp_path / "truth.nc"
    xr.Dataset(
        {
            "no2": (
                ("time", "latitude", "longitude"),
                np.stack(truth_fields),
            )
        },
        coords={
            "time": np.asarray(valid_times, dtype="datetime64[ns]"),
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_netcdf(truth_path)

    checkpoint = tmp_path / "phase1.ckpt"
    checkpoint.write_bytes(b"controlled phase one")
    checkpoint_digest = sha256_file(checkpoint)
    spatial_checkpoint = tmp_path / "spatial_checkpoint.pt"
    spatial_checkpoint.write_bytes(b"controlled spatial checkpoint")
    spatial_digest = sha256_file(spatial_checkpoint)
    phase2_checkpoint = tmp_path / "temporal_checkpoint.pt"
    phase2_checkpoint.write_bytes(b"controlled temporal checkpoint")
    phase2_digest = sha256_file(phase2_checkpoint)
    directories = {mode: tmp_path / mode for mode in ("aurora", "off", "on", "shuffled")}
    for directory in directories.values():
        directory.mkdir()

    for initialization in initializations:
        indices = [
            index
            for index, valid_time in enumerate(valid_times)
            if initialization < valid_time <= initialization + np.timedelta64(36, "h")
        ]
        truth_values = np.stack([truth_fields[index] for index in indices])
        values_by_mode = {
            "aurora": truth_values + 2.0,
            "off": truth_values + 1.0,
            "on": truth_values.copy(),
            "shuffled": truth_values + 0.75,
        }
        tag = np.datetime_as_string(initialization, unit="s").replace("-", "").replace(":", "")
        for mode, values in values_by_mode.items():
            rollout = xr.Dataset(
                {
                    "no2": (
                        ("time", "latitude", "longitude"),
                        values,
                    )
                },
                coords={
                    "time": np.asarray(
                        [valid_times[index] for index in indices],
                        dtype="datetime64[ns]",
                    ),
                    "latitude": latitude,
                    "longitude": longitude,
                },
                attrs={
                    "initialization_time": np.datetime_as_string(
                        initialization,
                        unit="s",
                    )
                },
            )
            rollout.to_netcdf(directories[mode] / f"rollout_predictions_init_{tag}.nc")

    selected_dates = [np.datetime_as_string(value, unit="s") for value in initializations]
    for mode, directory in ((name, directories[name]) for name in ("off", "on", "shuffled")):
        manifest = _manifest(
            mode,
            phase1=checkpoint_digest,
            spatial=spatial_digest,
            phase2=phase2_digest,
        )
        manifest["selected_test_initializations"] = selected_dates
        (directory / "mamba_ablation_manifest.json").write_text(json.dumps(manifest))

    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case_name": "NO2_US-WEST_3day_lead",
                "paths": {
                    "project_root": str(tmp_path),
                    "data_dir": str(tmp_path / "data"),
                    "output_dir": str(tmp_path / "unused"),
                },
                "evaluation": {
                    "ground_truth_path": str(truth_path),
                    "baseline_rollout_path": str(directories["aurora"]),
                    "lead_times_hours": lead_hours.tolist(),
                    "target_variables": [{"name": "no2", "kind": "surface"}],
                },
            }
        )
    )
    output_directory = tmp_path / "evaluation"
    paths = evaluate_mamba_ablation(
        config_path,
        [directories["off"]],
        [directories["on"]],
        [directories["shuffled"]],
        [checkpoint],
        output_directory,
        bootstrap_resamples=40,
        bootstrap_block_length=1,
        make_plots=True,
    )

    assert paths["evaluation_manifest"].is_file()
    assert paths["evaluation_summary"].is_file()
    summary = pd.read_csv(paths["paired_summary_overall"])
    for comparison in ("off_vs_aurora", "on_vs_aurora", "on_vs_off"):
        rmse = summary.loc[(summary["comparison"] == comparison) & (summary["metric"] == "rmse")]
        assert not rmse.empty
        assert (rmse["mean_improvement"] > 0.0).all()
    report = paths["evaluation_summary"].read_text()
    assert "Mamba-off versus raw Aurora" in report
    assert "Ordered Mamba-on versus raw Aurora" in report
    manifest = json.loads(paths["evaluation_manifest"].read_text())
    assert manifest["bootstrap"]["unit"] == "seed_and_forecast_initialization_id"
    assert manifest["bootstrap"]["method"].startswith("paired_hierarchical")
    assert manifest["aurora_baseline"]["directory"] == str(directories["aurora"].resolve())
    assert len(manifest["aurora_baseline"]["selected_files"]) == 2
    assert len(manifest["selection"]["selected_initializations_by_seed"]["3"]) == 2
    figure_names = {Path(item["path"]).name for item in manifest["figures"]}
    assert "representative_map_no2_surface.png" in figure_names
    assert "hotspot_trajectory_no2_surface.png" in figure_names
    assert "temporal_trajectory_no2_surface.png" in figure_names
