"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the bounded single-batch refinement trace."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from finetune.generate_overall_evaluation_maps import Selection
from finetune.trace_refinement_batch import (
    _json_safe,
    _load_aurora_scales,
    _normalization_scale,
    _render_map,
)


def test_trace_uses_exact_aurora_no2_channel_scales() -> None:
    repository = Path(__file__).resolve().parents[1]
    scales = _load_aurora_scales(repository)

    key, scale = _normalization_scale(
        Selection("no2", 925.0, "atmos"), scales=scales
    )

    assert key == "no2_925"
    assert scale == 1.378600e-07
    assert scales["tcno2"] == 6.141358e-05


def test_trace_png_writer_handles_nan_without_smoothing(tmp_path: Path) -> None:
    path = tmp_path / "trace.png"
    values = np.array([[0.0, 1.0, np.nan], [-1.0, 0.5, 2.0]])

    _render_map(
        values,
        latitude=np.array([52.0, 51.6]),
        longitude=np.array([232.0, 232.4, 232.8]),
        title="unit trace",
        units="kg kg-1",
        path=path,
        limits=(-2.0, 2.0),
        diverging=True,
    )

    with Image.open(path) as image:
        assert image.size == (960, 680)
        assert image.format == "PNG"


def test_trace_manifest_is_strict_json() -> None:
    value = _json_safe(
        {"finite": np.float32(1.25), "nan": float("nan"), "inf": np.inf}
    )

    encoded = json.dumps(value, allow_nan=False)

    assert json.loads(encoded) == {"finite": 1.25, "nan": None, "inf": None}
