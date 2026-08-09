"""Tests for the standalone overall evaluation-map helper."""

from pathlib import Path

from finetune.generate_overall_evaluation_maps import (
    configured_map_extent,
    load_settings,
)


def test_load_settings_matches_notebook_path_defaults(tmp_path: Path) -> None:
    """Training paths and the example-baseline fallback match the notebook."""
    root = tmp_path / "repository"
    config_dir = root / "finetune"
    (root / ".git").mkdir(parents=True)
    (root / "examples" / "outputs" / "cams_rollouts").mkdir(parents=True)
    config_dir.mkdir()
    config_path = config_dir / "case.yaml"
    config_path.write_text(
        """\
case_name: example_case
paths:
  project_root: .
  data_dir: ../data
  output_dir: ./outputs
data:
  target_lead_times: [1, 2]
  lat_min: 31
  lat_max: 52
  lon_min: -128
  lon_max: -100
rollout:
  rollout_step_hours: 12
"""
    )

    settings = load_settings(config_path)

    assert settings["truth_path"] == root / "data" / "example_case" / "test.nc"
    assert settings["baseline_dir"] == root / "examples" / "outputs" / "cams_rollouts"
    assert settings["finetuned_dir"] == config_dir / "outputs" / "example_case"
    assert (
        settings["output_dir"]
        == config_dir / "outputs" / "example_case" / "evaluation"
    )
    assert settings["lead_hours"] == [12.0, 24.0]
    assert settings["map_extent"] == (-128.0, -100.0, 31.0, 52.0)
    assert settings["coordinate_tolerance"] == 2.0e-5


def test_none_boundaries_select_global_map() -> None:
    """O3-style None boundaries select Cartopy's true global extent."""
    raw = {
        "data": {
            "domain_type": "global",
            "lat_min": "None",
            "lat_max": "None",
            "lon_min": "None",
            "lon_max": "None",
        }
    }

    assert configured_map_extent(raw) is None
