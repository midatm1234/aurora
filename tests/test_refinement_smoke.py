"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Minimal end-to-end smoke coverage for the stochastic refinement workflow.

Runs the non-destructive smoke script at a tiny size in a temporary directory
so CI exercises the same path an operator would run manually.
"""

from __future__ import annotations

import pytest

pytest.importorskip("xarray")

from finetune.refinement_smoke_test import REFINERS, run_smoke  # noqa: E402


def test_smoke_covers_every_refiner(tmp_path) -> None:
    results = run_smoke(
        height=8,
        width=12,
        batch=1,
        leads=(1, 2),
        ensemble=2,
        output_dir=str(tmp_path),
    )
    assert set(results["refiners"]) == set(REFINERS)
    for name, entry in results["refiners"].items():
        assert entry["residual_finite"], name
        assert entry["refined_finite"], name
        assert entry["members_shape"][1] == 2, name
        assert entry["refiner_parameters"] > 0, name
        # Refinement is postprocessing by default in every refiner.
        assert entry["feedback_to_rollout"] is False, name


def test_smoke_writes_a_complete_netcdf(tmp_path) -> None:
    results = run_smoke(
        height=8, width=12, batch=1, leads=(1,), ensemble=2, output_dir=str(tmp_path)
    )
    netcdf = results["netcdf"]
    for expected in (
        "gtco3",
        "gtco3_refined",
        "gtco3_residual",
        "gtco3_members",
        "gtco3_ensemble_mean",
        "gtco3_ensemble_spread",
        "go3",
        "go3_members",
    ):
        assert expected in netcdf["variables"], expected
    assert netcdf["sizes"]["member"] == 2
    assert netcdf["sizes"]["level"] == 2
    # The temporary directory is cleaned up: nothing is left behind.
    assert list(tmp_path.iterdir()) == []
