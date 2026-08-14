"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Compatibility contracts for regional NO2 and global O3 workflows."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from aurora import Batch, Metadata
from finetune import aurora_finetune_utils as ft


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "finetune"
NO2_CONFIG = CONFIG_DIR / "aurora_NO2_finetune_US-WEST_3day_lead_config.yaml"
NO2_DT_CONFIG = (
    CONFIG_DIR
    / "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml"
)
O3_CONFIG = CONFIG_DIR / "aurora_O3_global_finetune_3day_lead_config.yaml"


@pytest.mark.parametrize(
    "notebook_name",
    ["aurora_finetune_rollout.ipynb", "aurora_inference_rollout.ipynb"],
)
def test_notebook_config_is_resolved_after_papermill_parameters(notebook_name: str) -> None:
    notebook = json.loads((CONFIG_DIR / notebook_name).read_text())
    parameter_index, parameter_cell = next(
        (index, cell)
        for index, cell in enumerate(notebook["cells"])
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    )
    parameter_source = "".join(parameter_cell["source"])
    assert "CONFIG_PATH_NAME" in parameter_source
    assert "ft.load_config" not in parameter_source
    if notebook_name == "aurora_inference_rollout.ipynb":
        for name in ("CHECKPOINT_PATH", "ROLLOUT_NUM_STEPS", "OUTPUT_DIR"):
            assignment = next(
                line for line in parameter_cell["source"] if line.startswith(f"{name} =")
            )
            assert "#" not in assignment

    later_source = "".join(
        line
        for cell in notebook["cells"][parameter_index + 1 :]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )
    assert "CONFIG_PATH =" in later_source
    assert "ft.load_config(CONFIG_PATH" in later_source


def _raw_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    return value


def _synthetic_dataset(config: dict) -> xr.Dataset:
    data_cfg = config["data"]
    levels = np.asarray(data_cfg["atmos_levels"], dtype=np.float64)
    times = np.arange(
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-05T00:00:00"),
        np.timedelta64(12, "h"),
    )
    if data_cfg["domain_type"] == "global":
        latitude = np.asarray([90, 54, 18, -18, -54, -90], dtype=np.float64)
        longitude = np.arange(0.0, 360.0, 60.0, dtype=np.float64)
    else:
        latitude = np.asarray([52.0, 51.6, 51.2, 50.8, 50.4, 50.0], dtype=np.float64)
        longitude = np.asarray([232.0, 232.4, 232.8, 233.2, 233.6, 234.0], dtype=np.float64)

    coords = {
        "time": times,
        "latitude": latitude,
        "longitude": longitude,
        "level": levels,
    }
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for item in (*data_cfg["predictor_variables"], *data_cfg["target_variables"]):
        name = item["dataset_name"]
        if name in variables:
            continue
        if item["kind"] == "atmos":
            shape = (len(times), len(levels), len(latitude), len(longitude))
            dims = ("time", "level", "latitude", "longitude")
        else:
            shape = (len(times), len(latitude), len(longitude))
            dims = ("time", "latitude", "longitude")
        variables[name] = (dims, np.zeros(shape, dtype=np.float32))
    for item in data_cfg.get("static_variables", []):
        name = item["dataset_name"]
        variables[name] = (
            ("latitude", "longitude"),
            np.zeros((len(latitude), len(longitude)), dtype=np.float32),
        )
    return xr.Dataset(variables, coords=coords)


@pytest.mark.parametrize("path", [NO2_CONFIG, NO2_DT_CONFIG, O3_CONFIG])
def test_repository_configs_pass_shared_schema_and_dataset_contract(path: Path) -> None:
    config = _raw_config(path)
    ft.validate_config(config, path)
    dataset = _synthetic_dataset(config)
    specs = ft.resolve_variable_specs(dataset, config)
    assert specs.predictors
    assert specs.targets
    assert tuple(dataset.level.values) == tuple(config["data"]["atmos_levels"])


def test_no2_config_uses_no2_auxiliary_and_plot_targets() -> None:
    config = _raw_config(NO2_CONFIG)
    aux = config["training"]["flow_aux_loss"]
    assert aux["coherence_column_var"] == "tcno2"
    assert aux["coherence_profile_var"] == "no2"
    assert config["notebook"]["plot_variables"] == ["no2", "tcno2"]
    ft.validate_config(config, NO2_CONFIG)


def test_config_validation_rejects_o3_names_in_no2_recipe() -> None:
    config = _raw_config(NO2_CONFIG)
    config["training"]["flow_aux_loss"]["coherence_column_var"] = "gtco3"
    with pytest.raises(ValueError, match="coherence_column_var.*not a configured target"):
        ft.validate_config(config)

    config = _raw_config(NO2_CONFIG)
    config["notebook"]["plot_variables"] = ["go3", "gtco3"]
    with pytest.raises(ValueError, match="notebook.plot_variables"):
        ft.validate_config(config)


def test_config_validation_rejects_missing_levels_and_unused_normalization() -> None:
    config = _raw_config(NO2_CONFIG)
    config["data"]["target_variables"][0]["loss_levels"].append(775)
    with pytest.raises(ValueError, match="absent from data.atmos_levels"):
        ft.validate_config(config)

    config = _raw_config(NO2_CONFIG)
    config["data"]["normalization"]["enabled"] = True
    config["data"]["normalization"]["mode"] = "zscore"
    with pytest.raises(ValueError, match="Custom data.normalization is not implemented"):
        ft.validate_config(config)


def test_legacy_no2_checkpoint_reports_temporal_contract_mismatch() -> None:
    config = _raw_config(NO2_CONFIG)
    ft.validate_config(config)
    dataset = _synthetic_dataset(config)
    specs = ft.resolve_variable_specs(dataset, config)
    model = ft.maybe_wrap_flow_refine(
        torch.nn.Identity(), config, specs, lon=dataset.longitude.values,
    )

    saved_config = copy.deepcopy(config)
    for key in tuple(saved_config["model"]):
        if key.startswith("mamba_temporal_"):
            saved_config["model"].pop(key)
    saved_config["model"].pop("lon_periodic_resolved", None)
    saved_config["model"].pop("longitude_grid_signature", None)
    checkpoint = {
        "config": saved_config,
        "best_val_loss": 1.0,
        "norm_stats": {
            "no2": {"mean": torch.zeros(3), "std": torch.ones(3)},
            "tcno2": {"mean": torch.zeros(1), "std": torch.ones(1)},
        },
    }

    ft.validate_checkpoint_longitude(model, checkpoint)
    with pytest.raises(ValueError, match="Mamba temporal configuration mismatch"):
        ft.validate_checkpoint_refinement_contract(model, checkpoint, config, specs)

    ft.validate_checkpoint_refinement_contract(
        model,
        checkpoint,
        config,
        specs,
        allow_temporal_migration=True,
    )


def test_no2_prediction_output_is_target_only_and_regionally_shaped() -> None:
    config = _raw_config(NO2_CONFIG)
    ft.validate_config(config)
    specs = ft.resolve_variable_specs(_synthetic_dataset(config), config)
    latitude = torch.tensor([52.0, 51.6, 51.2], dtype=torch.float64)
    longitude = torch.tensor([232.0, 232.4, 232.8], dtype=torch.float64)
    levels = tuple(float(value) for value in config["data"]["atmos_levels"])
    prediction = Batch(
        surf_vars={
            "tcno2": torch.ones(1, 1, 3, 3),
            "2t": torch.zeros(1, 1, 3, 3),
        },
        static_vars={},
        atmos_vars={
            "no2": torch.ones(1, 1, len(levels), 3, 3),
            "q": torch.zeros(1, 1, len(levels), 3, 3),
        },
        metadata=Metadata(
            lat=latitude,
            lon=longitude,
            time=(datetime(2024, 1, 1, 12),),
            atmos_levels=levels,
        ),
    )

    output = ft.save_predictions(
        [prediction],
        "unused.nc",
        save_netcdf=False,
        resolved_specs=specs,
        lon_periodic=False,
    )
    assert set(output.data_vars) == {"no2", "tcno2"}
    assert output.tcno2.dims == ("time", "latitude", "longitude")
    assert output.no2.dims == ("time", "level", "latitude", "longitude")
    assert output.tcno2.shape == (1, 3, 3)
    assert output.no2.shape == (1, len(levels), 3, 3)
    assert "modulo" not in output.longitude.attrs
