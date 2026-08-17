"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Compatibility contracts for regional NO2 and global O3 workflows."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml
from finetune import aurora_finetune_utils as ft
from finetune.refinement.config import resolve_refinement_config

from aurora import Batch, Metadata

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "finetune"
NO2_CONFIG = CONFIG_DIR / "aurora_NO2_finetune_US-WEST_3day_lead_config.yaml"
NO2_DT_CONFIG = (
    CONFIG_DIR / "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml"
)
NO2_DIFFUSION_CONFIG = CONFIG_DIR / "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_config.yaml"
NO2_FLOW_CONFIG = CONFIG_DIR / "aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_config.yaml"
NO2_FLOW_TRANSFORMER_CONFIG = (
    CONFIG_DIR / "aurora_NO2_finetune_US-WEST_3day_lead_flow_matching_transformer_config.yaml"
)
O3_CONFIG = CONFIG_DIR / "aurora_O3_global_finetune_3day_lead_config.yaml"
CANONICAL_UNIFIED_REFINEMENT_TYPES = frozenset(
    {
        "diffusion_unet",
        "diffusion_transformer",
        "flow_matching_conv_unet",
        "flow_matching_transformer",
    }
)
NO_REFINEMENT_EXAMPLE = (
    CONFIG_DIR
    / "examples"
    / "stochastic_refinement"
    / "aurora_O3_global_rollout_no_refinement.yaml"
)


def _active_pollutant_configs() -> tuple[Path, ...]:
    """Return active NO2/O3 recipes, excluding preserved backup trees."""
    configs = {
        path
        for pattern in (
            "aurora_NO2*.yaml",
            "aurora_NO2*.yml",
            "aurora_O3*.yaml",
            "aurora_O3*.yml",
        )
        for path in CONFIG_DIR.rglob(pattern)
        if "backup" not in path.parts
        and not any(part.startswith("yaml_backups") for part in path.parts)
    }
    return tuple(sorted(configs))


ACTIVE_POLLUTANT_CONFIGS = _active_pollutant_configs()
PRIMARY_FINETUNE_CONFIGS = (
    NO2_CONFIG,
    NO2_DIFFUSION_CONFIG,
    NO2_DT_CONFIG,
    NO2_FLOW_CONFIG,
    NO2_FLOW_TRANSFORMER_CONFIG,
    O3_CONFIG,
)
UNIFIED_NO2_CONFIGS = (
    NO2_CONFIG,
    NO2_DIFFUSION_CONFIG,
    NO2_DT_CONFIG,
    NO2_FLOW_CONFIG,
    NO2_FLOW_TRANSFORMER_CONFIG,
)


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
    if notebook_name == "aurora_inference_rollout.ipynb":
        assert "flow_sampling_steps_override=SAMPLING_STEPS_OVERRIDE" in later_source
        assert (
            "allow_unsafe_temporal_sampling_override=" "ALLOW_UNSAFE_MAMBA_SAMPLING_OVERRIDE"
        ) in later_source
        assert "model.sampling_steps = int(SAMPLING_STEPS_OVERRIDE)" not in later_source
        assert "temporal_control=TEMPORAL_CONTROL" in later_source
        for provenance_key in (
            "checkpoint_sha256",
            "resolved_flow_sampling_steps",
            "resolved_temporal_config",
            "trajectory_policy",
            "member_seeds",
        ):
            assert f"rollout_ds.attrs['{provenance_key}']" in later_source


def test_finetune_notebook_default_config_loads_as_unified_refinement() -> None:
    notebook = json.loads((CONFIG_DIR / "aurora_finetune_rollout.ipynb").read_text())
    parameter_cell = next(
        cell
        for cell in notebook["cells"]
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    )
    parameter_tree = ast.parse("".join(parameter_cell["source"]))
    assignment = next(
        node
        for node in parameter_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CONFIG_PATH_NAME"
            for target in node.targets
        )
    )
    config_name = ast.literal_eval(assignment.value)

    config = ft.load_config(CONFIG_DIR / config_name)
    refinement = resolve_refinement_config(config)
    assert refinement.type == "flow_matching_conv_unet"
    assert refinement.backend == "unified"
    assert refinement.target_space.residual_clip_standard_deviations == 4.0


def _raw_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    "path",
    ACTIVE_POLLUTANT_CONFIGS,
    ids=lambda path: str(path.relative_to(CONFIG_DIR)),
)
def test_active_pollutant_configs_select_canonical_unified_heads(path: Path) -> None:
    """Keep runnable NO2/O3 recipes off legacy refinement aliases/backends."""
    raw = _raw_config(path)
    refinement_section = raw.get("model", {}).get("refinement")
    assert isinstance(
        refinement_section, dict
    ), f"{path.relative_to(ROOT)} must declare model.refinement explicitly"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = resolve_refinement_config(raw)
    ignored_family_warnings = [
        str(item.message)
        for item in caught
        if "settings are ignored for type=" in str(item.message)
    ]
    assert not ignored_family_warnings, (
        f"{path.relative_to(ROOT)} contains inactive refinement settings: "
        f"{ignored_family_warnings}"
    )
    if path == NO_REFINEMENT_EXAMPLE:
        assert refinement_section.get("enabled") is False
        assert refinement_section.get("type") == "none"
        assert config.is_active is False
        assert config.backend == "none"
        return

    model_section = raw.get("model", {})
    training_section = raw.get("training", {})
    assert (
        model_section.get("flow_refine_enabled") is not True
    ), f"{path.relative_to(ROOT)} must not enable the legacy flow wrapper"
    legacy_aux = training_section.get("flow_aux_loss") or {}
    assert (
        legacy_aux.get("enabled") is not True
    ), f"{path.relative_to(ROOT)} must not enable ignored legacy flow_aux_loss"

    declared_type = refinement_section.get("type")
    assert declared_type in CANONICAL_UNIFIED_REFINEMENT_TYPES, (
        f"{path.relative_to(ROOT)} declares non-canonical refinement type " f"{declared_type!r}"
    )
    opposite_family = "flow_matching" if declared_type.startswith("diffusion") else "diffusion"
    assert opposite_family not in refinement_section, (
        f"{path.relative_to(ROOT)} must not configure inactive "
        f"model.refinement.{opposite_family} settings"
    )
    assert config.type == declared_type
    assert config.is_active is True
    assert config.backend == "unified"


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


@pytest.mark.parametrize("path", PRIMARY_FINETUNE_CONFIGS)
def test_repository_configs_pass_shared_schema_and_dataset_contract(path: Path) -> None:
    # load_config is the production entrypoint and includes schema validation;
    # call validate_config explicitly as a regression guard for callers that
    # already hold a parsed mapping.
    config = ft.load_config(path)
    ft.validate_config(config, path)
    dataset = _synthetic_dataset(config)
    specs = ft.resolve_variable_specs(dataset, config)
    assert specs.predictors
    assert specs.targets
    assert tuple(dataset.level.values) == tuple(config["data"]["atmos_levels"])


@pytest.mark.parametrize(
    "path",
    UNIFIED_NO2_CONFIGS,
)
def test_unified_no2_configs_share_packed_geophysical_temporal_preset(
    path: Path,
) -> None:
    temporal = _raw_config(path)["model"]["mamba_temporal"]
    assert temporal["enabled"] is False
    assert temporal["mode"] == "packed_joint"
    assert temporal["lead_time_conditioning"] is True
    assert temporal["mask_conditioning"] is True
    assert temporal["coordinate_conditioning"] is True
    assert temporal["causal"] is True


def test_shared_validation_rejects_string_legacy_feedback_boolean() -> None:
    config = _raw_config(O3_CONFIG)
    config["training"]["flow_refine_autoregressive_feedback"] = "false"

    with pytest.raises(
        ValueError,
        match="training.flow_refine_autoregressive_feedback must be a boolean",
    ):
        ft.validate_config(config, O3_CONFIG)


@pytest.mark.parametrize("path", UNIFIED_NO2_CONFIGS)
def test_unified_no2_mamba_is_a_single_enable_toggle(path: Path) -> None:
    config = _raw_config(path)
    assert config["training"]["mamba_temporal_weight"] > 0.0
    config["model"]["mamba_temporal"]["enabled"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ft.validate_config(config, path)


def test_no2_config_uses_unified_packed_flow_and_no2_targets() -> None:
    config = _raw_config(NO2_CONFIG)
    refinement = config["model"]["refinement"]
    assert refinement["type"] == "flow_matching_conv_unet"
    assert refinement["deterministic_head"] == "shared_process"
    assert refinement["conditioning"]["latitude"] is True
    assert refinement["conditioning"]["longitude"] is True
    assert refinement["loss"]["extreme_tail"] == "upper"
    assert config["data"]["nonnegative_target_variables"] == ["no2", "tcno2"]
    assert "flow_refine_enabled" not in config["model"]
    assert "flow_aux_loss" not in config["training"]
    assert config["notebook"]["plot_variables"] == ["no2", "tcno2"]
    ft.validate_config(config, NO2_CONFIG)


def test_config_validation_rejects_o3_names_in_no2_recipe() -> None:
    config = _raw_config(NO2_CONFIG)
    config["notebook"]["plot_variables"] = ["go3", "gtco3"]
    with pytest.raises(ValueError, match="notebook.plot_variables"):
        ft.validate_config(config)


def test_unified_extreme_tail_rejects_unknown_mode() -> None:
    config = _raw_config(NO2_CONFIG)
    config["model"]["refinement"]["loss"]["extreme_tail"] = "lower"

    with pytest.raises(ValueError, match="refinement.loss.extreme_tail"):
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
    config["model"].pop("refinement")
    config["model"].update(
        {
            "flow_refine_enabled": True,
            "flow_refine_contract_version": 3,
            "flow_refine_hidden": 64,
            "flow_refine_time_dim": 128,
            "flow_refine_sampling_steps": 1,
            "flow_refine_residual_zscore": True,
            "flow_refine_lead_time_cond": True,
            "flow_refine_lead_time_scale_hours": 72,
            "flow_refine_lon_encoding": True,
        }
    )
    # Exercise the migration contract independently of the repository recipe's
    # current temporal default. A legacy checkpoint omits these keys, whereas
    # the model being constructed here explicitly enables temporal refinement.
    config["model"].pop("mamba_temporal", None)
    config["model"]["mamba_temporal_enabled"] = True
    config["training"]["mamba_temporal_weight"] = 1.0
    ft.validate_config(config)
    dataset = _synthetic_dataset(config)
    specs = ft.resolve_variable_specs(dataset, config)
    model = ft.maybe_wrap_flow_refine(
        torch.nn.Identity(),
        config,
        specs,
        lon=dataset.longitude.values,
    )

    saved_config = copy.deepcopy(config)
    for key in tuple(saved_config["model"]):
        if key.startswith("mamba_temporal_"):
            saved_config["model"].pop(key)
    # This fixture isolates a checkpoint that predates the temporal module,
    # not one that predates longitude-aware training. Retain the resolved
    # longitude contract so strict longitude validation passes before the
    # intended temporal-contract mismatch is checked.
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


@pytest.mark.parametrize("version", ["v2", "v3", "v4"])
def test_o3_papermill_launchers_guard_existing_case_artifacts(version: str) -> None:
    script = CONFIG_DIR / f"run_O3_{version}_papermill.sh"
    source = script.read_text()
    guard_position = source.index("AURORA_ALLOW_CASE_OVERWRITE")
    mkdir_position = source.index("mkdir -p")
    assert guard_position < mkdir_position
    assert "Refusing to reuse existing" in source
    assert "CASE_CHECKPOINT_DIR" in source
    subprocess.run(["bash", "-n", str(script)], check=True)
