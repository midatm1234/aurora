"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Executable train-to-inference smoke coverage for the NO2 diffusion Transformer."""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml
from finetune import aurora_finetune_utils as ft
from finetune import model_factory
from finetune.mamba_temporal import MambaTemporalModule
from finetune.refinement import integration as refinement_integration
from finetune.refinement.two_phase import AuroraTwoPhaseRefiner

from aurora import Batch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "finetune"
NO2_DT_CONFIG = (
    CONFIG_DIR
    / "aurora_NO2_finetune_US-WEST_3day_lead_diffusion_transformer_config.yaml"
)


class TinyAurora(torch.nn.Module):
    """Shape-faithful deterministic stand-in for an unavailable 1.3B checkpoint."""

    def __init__(
        self,
        *,
        surf_vars,
        static_vars,
        atmos_vars,
        patch_size=3,
        autocast=False,
        **_,
    ):
        super().__init__()
        self.surf_vars = tuple(surf_vars)
        self.static_vars = tuple(static_vars)
        self.atmos_vars = tuple(atmos_vars)
        self.patch_size = int(patch_size)
        self.autocast = bool(autocast)
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.surf_stats = {}

    def forward(self, batch: Batch) -> Batch:
        metadata = dataclasses.replace(
            batch.metadata,
            time=tuple(value + timedelta(hours=12) for value in batch.metadata.time),
        )
        return Batch(
            surf_vars={
                name: value[:, -1:] + self.bias.to(value.dtype)
                for name, value in batch.surf_vars.items()
            },
            static_vars=batch.static_vars,
            atmos_vars={
                name: value[:, -1:] + self.bias.to(value.dtype)
                for name, value in batch.atmos_vars.items()
            },
            metadata=metadata,
        )


def _raw_config() -> dict:
    config = yaml.safe_load(NO2_DT_CONFIG.read_text())
    assert isinstance(config, dict)
    return config


def _smoke_config() -> dict:
    config = copy.deepcopy(_raw_config())
    config["model"]["model_variant"] = "tiny_aurora"
    config["model"]["mixed_precision"] = "none"
    config["model"].update(
        {
            "mamba_temporal_enabled": True,
            "mamba_temporal_channels": 4,
            "mamba_temporal_state": 2,
            "mamba_temporal_layers": 1,
            "mamba_temporal_conv": 2,
            "mamba_temporal_expand": 1,
        }
    )
    transformer = config["model"]["refinement"]["transformer"]
    transformer.update(
        {
            "patch_size": [2, 2],
            "embedding_dim": 16,
            "num_heads": 4,
            "num_blocks": 1,
            "mlp_ratio": 2.0,
            "max_tokens_lat": 16,
            "max_tokens_lon": 16,
            "window_size": [2, 2],
            "gradient_checkpointing": False,
            "optimized_attention": "math",
        }
    )
    diffusion = config["model"]["refinement"]["diffusion"]
    diffusion.update({"training_timesteps": 8, "inference_steps": 2})
    # This smoke split has exactly one training batch; calibrate and freeze the
    # per-channel correction statistics before validation/rollout enters eval.
    config["model"]["refinement"]["target_space"][
        "residual_scaling_warmup_batches"
    ] = 1
    # The shipped product is deterministic; the smoke test explicitly opts in
    # to stochastic ensemble behavior so both contracts stay covered.
    config["model"]["refinement"].update(
        {"ensemble_size": 2, "deterministic_inference": False}
    )
    config["training"].update(
        {
            "batch_size": 1,
            "accumulation_steps": 1,
            "scheduler": "none",
            "learning_rate": 1.0e-3,
            "mamba_temporal_weight": 1.0,
            "validation_refinement_ensemble_size": 2,
        }
    )
    config["rollout"].update(
        {
            "verbose_provenance": False,
            "save_plots": False,
            "smooth_sigma": 0.0,
        }
    )
    return config


def _synthetic_dataset(config: dict) -> xr.Dataset:
    levels = np.asarray(config["data"]["atmos_levels"], dtype=np.float64)
    times = np.arange(
        np.datetime64("2024-01-01T00"),
        np.datetime64("2024-01-05T00"),
        np.timedelta64(12, "h"),
    )
    latitude = np.linspace(52.0, 50.0, 6, dtype=np.float64)
    longitude = np.linspace(232.0, 234.0, 6, dtype=np.float64)
    coords = {
        "time": times,
        "level": levels,
        "latitude": latitude,
        "longitude": longitude,
    }
    variables = {}
    seen = set()
    time_ramp = np.arange(times.size, dtype=np.float32)[:, None, None]
    spatial = (
        np.arange(latitude.size, dtype=np.float32)[None, :, None]
        + np.arange(longitude.size, dtype=np.float32)[None, None, :]
    ) / 100.0
    for item in (
        *config["data"]["predictor_variables"],
        *config["data"]["target_variables"],
    ):
        name = item["dataset_name"]
        if name in seen:
            continue
        seen.add(name)
        if item["kind"] == "atmos":
            base = 1.0e-9 if name == "no2" else 1.0
            values = base * (
                1.0
                + 0.02 * time_ramp[:, None]
                + 0.001 * np.arange(levels.size, dtype=np.float32)[None, :, None, None]
                + spatial[:, None]
            )
            variables[name] = (
                ("time", "level", "latitude", "longitude"),
                values.astype(np.float32),
            )
        else:
            base = 1.0e-5 if name == "tcno2" else 1.0
            values = base * (1.0 + 0.02 * time_ramp + spatial)
            variables[name] = (
                ("time", "latitude", "longitude"),
                values.astype(np.float32),
            )
    for item in config["data"]["static_variables"]:
        name = item["dataset_name"]
        variables[name] = (
            ("latitude", "longitude"),
            np.broadcast_to(spatial[0], (latitude.size, longitude.size)).astype(np.float32),
        )
    dataset = xr.Dataset(variables, coords=coords)
    dataset["no2"].attrs["units"] = "kg kg-1"
    dataset["tcno2"].attrs["units"] = "kg m-2"
    return dataset


def test_target_config_and_notebooks_share_the_factory() -> None:
    config = _raw_config()
    ft.validate_config(config, NO2_DT_CONFIG)
    assert config["paths"]["data_case_name"] == "NO2_US-WEST_3day_lead"
    assert config["data"]["target_lead_times"] == [1, 2, 3, 4, 5, 6]
    assert config["rollout"]["rollout_step_hours"] == 12

    text = NO2_DT_CONFIG.read_text()
    refinement = config["model"]["refinement"]
    target_space = refinement["target_space"]
    loss = refinement["loss"]
    diffusion = refinement["diffusion"]
    transformer = refinement["transformer"]
    training = config["training"]
    assert "correction_target = CAMS - Aurora" in text
    assert "refined_forecast = Aurora + predicted_correction" in text
    assert refinement["train_on_residual"] is True
    assert refinement["feedback_to_rollout"] is False
    assert refinement["deterministic_inference"] is True
    assert refinement["ensemble_size"] == 1
    assert target_space["residual_scaling"] == "per_channel"
    assert target_space["residual_scaling_center"] is True
    assert loss["deterministic_weight"] > 0.0
    assert loss["aux_on_deterministic"] is True
    assert loss["bias_weight"] > 0.0
    assert loss["degradation_weight"] > 0.0
    assert transformer["zero_init_output"] is True
    assert diffusion["prediction_type"] == "sample"
    assert not (
        transformer["zero_init_output"]
        and diffusion["prediction_type"] == "epsilon"
    )
    assert config["model"]["mamba_temporal_enabled"] is False
    assert training["mamba_temporal_weight"] == 0.0
    assert training["validation_refinement_ensemble_size"] == 1
    assert training["validation_source"] == "train_tail"
    assert training["checkpoint_metric"] == "mean_physical_rmse_ratio"
    assert training["require_all_physical_channels_improve"] is True
    assert training["require_refinement_improvement"] is True
    assert training["resume_training"] is False
    assert training["resume_from"] == ""

    for notebook_name in (
        "aurora_finetune_rollout.ipynb",
        "aurora_inference_rollout.ipynb",
    ):
        notebook = json.loads((CONFIG_DIR / notebook_name).read_text())
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        for index, cell_source in enumerate(code_cells):
            compile(cell_source, f"{notebook_name}:cell-{index}", "exec")
        source = "\n".join(code_cells)
        assert "ft.load_model_from_checkpoint(" in source
        assert "MODEL_REGISTRY" not in source
        assert "maybe_wrap_flow_refine(" not in source
        if notebook_name == "aurora_finetune_rollout.ipynb":
            assert "require_validated=False" in source
            assert "state.get('validated_for_inference')" in source
            assert "validated_for_inference is None" in source
            assert "UNVALIDATED last checkpoint" in source


def test_refinement_seed_is_stable_and_mixed_by_initialization_and_member() -> None:
    initialization_a = np.datetime64("2024-01-01T00:00:00")
    initialization_b = np.datetime64("2024-01-01T12:00:00")
    seed = ft.derive_refinement_seed(1234, initialization_a, member_index=0)

    assert seed == ft.derive_refinement_seed(1234, initialization_a, member_index=0)
    assert len(
        {
            seed,
            ft.derive_refinement_seed(1234, initialization_b, member_index=0),
            ft.derive_refinement_seed(1234, initialization_a, member_index=1),
            ft.derive_refinement_seed(4321, initialization_a, member_index=0),
        }
    ) == 4
    assert 0 <= seed <= 0x7FFF_FFFF_FFFF_FFFF
    with pytest.raises(ValueError, match="member_index must be non-negative"):
        ft.derive_refinement_seed(1234, initialization_a, member_index=-1)


def test_no2_diffusion_transformer_train_checkpoint_inference_netcdf(
    tmp_path, monkeypatch
) -> None:
    config = _smoke_config()
    dataset = _synthetic_dataset(config)
    ft.validate_config(config, NO2_DT_CONFIG)
    specs = ft.resolve_variable_specs(dataset, config)
    norm_stats = ft.compute_target_normalization_stats(dataset, specs, config)
    samples = ft.build_training_samples(dataset, config)
    assert len(samples) == 1

    monkeypatch.setattr(
        model_factory, "_model_registry", lambda: {"tiny_aurora": TinyAurora}
    )
    model = ft.build_finetune_model(
        config,
        specs,
        lon=dataset.longitude.values,
        lat=dataset.latitude.values,
        norm_stats=norm_stats,
        load_pretrained=False,
        autocast=False,
    )
    from finetune.aurora_finetune_distributed import (
        _calibrate_residual_scalers_from_training_split,
        _validation_baseline_model,
    )

    assert _validation_baseline_model(model) is model.aurora
    calibration = _calibrate_residual_scalers_from_training_split(
        model=model,
        train_ds=dataset,
        train_samples=samples,
        cfg=config,
        resolved_specs=specs,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        norm_stats=norm_stats,
        global_step=0,
    )
    assert calibration is not None
    assert calibration["method"] == "exact_training_split"
    assert model.refiner.residual_scaler.has_exact_training_split_calibration
    config.setdefault("runtime", {})["residual_calibration"] = calibration
    summary = ft.configure_trainable_parameters(model, config)
    assert summary["trainable_parameters"] > 0
    assert all(not parameter.requires_grad for parameter in model.aurora.parameters())
    assert all(parameter.requires_grad for parameter in model.refiner.parameters())
    assert model.has_temporal
    assert model.temporal is not None
    assert all(parameter.requires_grad for parameter in model.temporal.parameters())

    optimizer = ft.create_optimizer(model, config)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimizer_ids for parameter in model.temporal.parameters())
    before = {
        name: value.detach().clone()
        for name, value in model.refiner.named_parameters()
    }
    temporal_before = {
        name: value.detach().clone()
        for name, value in model.temporal.named_parameters()
    }
    training_sequence_shapes = []
    hook = model.temporal.core.surf_heads["tcno2"].register_forward_hook(
        lambda _module, args, _output: training_sequence_shapes.append(
            tuple(args[0].shape)
        )
    )
    model.train()
    loss, metrics = ft.compute_supervised_loss(
        model,
        dataset,
        samples[0],
        config,
        specs,
        "cpu",
        norm_stats=norm_stats,
    )
    assert torch.isfinite(loss)
    assert "refinement" in metrics
    assert set(metrics["temporal"]) == {
        "temporal_loss/tcno2",
        "temporal_loss/no2",
    }
    assert training_sequence_shapes == [(1, 6, 6, 6)]
    hook.remove()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.refiner.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in model.temporal.named_parameters()
        if ".decoder." in name
    )
    optimizer.step()
    assert any(
        not torch.equal(before[name], value.detach())
        for name, value in model.refiner.named_parameters()
    )
    assert any(
        not torch.equal(temporal_before[name], value.detach())
        for name, value in model.temporal.named_parameters()
    )

    validation_sequence_lengths = []
    validation_lead_hours = []
    validation_generators = []
    validation_ensemble_sizes = []
    validation_hook = model.temporal.core.surf_heads["tcno2"].register_forward_hook(
        lambda _module, args, _output: validation_sequence_lengths.append(
            int(args[0].shape[1])
        )
    )
    original_validation_refine = refinement_integration.refine_batch_prediction

    def _record_validation_refine(*args, **kwargs):
        lead = kwargs["forecast_lead_time_hours"]
        validation_lead_hours.append(float(lead.reshape(-1)[0].item()))
        validation_generators.append(kwargs.get("generator"))
        validation_ensemble_sizes.append(kwargs.get("ensemble_size"))
        return original_validation_refine(*args, **kwargs)

    monkeypatch.setattr(
        refinement_integration,
        "refine_batch_prediction",
        _record_validation_refine,
    )
    validation_metrics = ft.run_validation(
        model,
        dataset,
        samples,
        config,
        specs,
        "cpu",
        max_batches=1,
        norm_stats=norm_stats,
    )
    monkeypatch.setattr(
        refinement_integration,
        "refine_batch_prediction",
        original_validation_refine,
    )
    validation_hook.remove()
    assert np.isfinite(validation_metrics["val_loss"])
    assert validation_sequence_lengths == [1, 2, 3, 4, 5, 6]
    assert validation_lead_hours == [12.0, 24.0, 36.0, 48.0, 60.0, 72.0]
    assert validation_ensemble_sizes == [2] * 6
    assert validation_generators[0] is not None
    assert all(
        generator is validation_generators[0]
        for generator in validation_generators
    )

    checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
    config.setdefault("runtime", {})["training_run_id"] = "synthetic-no2-dt"
    ft.save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        None,
        epoch=0,
        global_step=1,
        best_val_loss=float(loss.detach()),
        config=config,
        norm_stats=norm_stats,
    )
    assert checkpoint_path.is_file()
    saved_temporal = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith("temporal.")
    }
    assert saved_temporal
    ft.write_run_manifest(config, tmp_path / "outputs")
    assert (tmp_path / "outputs" / "resolved_config.yaml").is_file()

    loaded, checkpoint = ft.load_model_from_checkpoint(
        config,
        specs,
        checkpoint_path,
        lon=dataset.longitude.values,
        lat=dataset.latitude.values,
        map_location="cpu",
        autocast=False,
    )
    assert checkpoint["global_step"] == 1
    assert checkpoint["resolved_temporal_config"]["enabled"] is True
    assert loaded.has_temporal
    # The validator runs only after state restoration. A frozen legacy-online
    # scaler has compatible shapes and passes the ordinary eval readiness check,
    # but it must never enter validated production inference.
    legacy_online = copy.deepcopy(checkpoint)
    method_keys = [
        key
        for key in legacy_online["model_state_dict"]
        if key.endswith("residual_scaler.calibration_method")
    ]
    assert len(method_keys) == 1
    method_key = method_keys[0]
    legacy_online["model_state_dict"][method_key] = torch.ones_like(
        legacy_online["model_state_dict"][method_key]
    )
    legacy_online_path = tmp_path / "checkpoints" / "legacy_online.ckpt"
    torch.save(legacy_online, legacy_online_path)
    with pytest.raises(ValueError, match="exact training-split provenance"):
        ft.load_model_from_checkpoint(
            config,
            specs,
            legacy_online_path,
            lon=dataset.longitude.values,
            lat=dataset.latitude.values,
            map_location="cpu",
            autocast=False,
        )
    diagnostic_model, _ = ft.load_model_from_checkpoint(
        config,
        specs,
        legacy_online_path,
        lon=dataset.longitude.values,
        lat=dataset.latitude.values,
        map_location="cpu",
        autocast=False,
        require_validated=False,
    )
    assert diagnostic_model.refiner.residual_scaler.is_ready
    assert not (
        diagnostic_model.refiner.residual_scaler
        .has_exact_training_split_calibration
    )

    bad_fingerprint = copy.deepcopy(checkpoint)
    bad_fingerprint["config"]["runtime"]["residual_calibration"][
        "training_split_fingerprint_sha256"
    ] = "00" * 32
    with pytest.raises(ValueError, match="provenance"):
        model_factory.validate_loaded_residual_scaler_contract(
            loaded, bad_fingerprint
        )

    bad_channel = copy.deepcopy(checkpoint)
    bad_channel["config"]["runtime"]["residual_calibration"]["scalers"][0][
        "channels"
    ][0]["valid_cell_count"] += 1
    with pytest.raises(ValueError, match="metadata/count"):
        model_factory.validate_loaded_residual_scaler_contract(loaded, bad_channel)

    for key, value in saved_temporal.items():
        assert torch.equal(loaded.state_dict()[key], value)
    assert loaded.packing.num_channels == 4
    assert [channel.aurora_name for channel in loaded.packing.channels] == [
        "tcno2",
        "no2",
        "no2",
        "no2",
    ]
    assert [channel.level for channel in loaded.packing.channels] == [
        None,
        1000.0,
        925.0,
        850.0,
    ]

    # The lower-level inference API keeps a stable, separate history for every
    # stochastic member and temporal-corrects members before aggregation.
    packed_rollout = torch.zeros(1, 4, 6, 6)
    direct_history = []
    loaded.eval()
    direct_first = loaded.refine(
        packed_rollout,
        forecast_lead_time=torch.tensor([12.0]),
        ensemble_size=2,
        seed=77,
        num_steps=2,
        temporal_history=direct_history,
    )
    assert direct_first.members.shape == (1, 2, 4, 6, 6)
    assert direct_history[0].shape == (1, 2, 4, 6, 6)
    assert not torch.equal(direct_history[0][:, 0], direct_history[0][:, 1])
    corrected_members = (
        packed_rollout.unsqueeze(1) + direct_first.member_residuals
    )
    assert not torch.equal(corrected_members, direct_history[0])
    loaded.refine(
        packed_rollout,
        forecast_lead_time=torch.tensor([24.0]),
        ensemble_size=2,
        seed=78,
        num_steps=2,
        temporal_history=direct_history,
    )
    assert len(direct_history) == 2

    member_datasets = []
    member_predictions = []
    anchor_time = dataset.time.values[samples[0]["anchor_index"]]
    loaded.eval()
    inference_sequence_lengths = []
    aurora_rollout_inputs = []
    inference_hook = loaded.temporal.core.surf_heads["tcno2"].register_forward_hook(
        lambda _module, args, _output: inference_sequence_lengths.append(
            int(args[0].shape[1])
        )
    )
    aurora_hook = loaded.aurora.register_forward_pre_hook(
        lambda _module, args: aurora_rollout_inputs.append(
            args[0].surf_vars["tcno2"].detach().clone()
        )
    )
    original_refine_batch_prediction = refinement_integration.refine_batch_prediction
    generator_calls = []
    rollout_ensemble_sizes = []

    def _record_generator(*args, **kwargs):
        generator = kwargs.get("generator")
        rollout_ensemble_sizes.append(kwargs.get("ensemble_size"))
        generator_calls.append(
            (
                generator,
                None if generator is None else generator.get_state().clone(),
            )
        )
        return original_refine_batch_prediction(*args, **kwargs)

    monkeypatch.setattr(
        refinement_integration,
        "refine_batch_prediction",
        _record_generator,
    )
    for member in range(2):
        predictions = ft.run_rollout(
            loaded,
            dataset,
            samples[0],
            config,
            specs,
            "cpu",
            refinement_seed=100 + member,
        )
        assert len(predictions) == 6
        member_predictions.append(predictions)
        member_datasets.append(
            ft.save_predictions(
                predictions,
                tmp_path / "unused.nc",
                save_netcdf=False,
                resolved_specs=specs,
                lon_periodic=False,
                initialization_time=anchor_time,
            )
        )
    monkeypatch.setattr(
        refinement_integration,
        "refine_batch_prediction",
        original_refine_batch_prediction,
    )
    inference_hook.remove()
    aurora_hook.remove()
    assert len(generator_calls) == 12
    assert rollout_ensemble_sizes == [2] * 12
    first_generator = generator_calls[0][0]
    second_generator = generator_calls[6][0]
    assert first_generator is not None
    assert second_generator is not None
    assert first_generator is not second_generator
    assert all(call[0] is first_generator for call in generator_calls[:6])
    assert all(call[0] is second_generator for call in generator_calls[6:])
    generator_states = [call[1] for call in generator_calls]
    assert all(state is not None for state in generator_states)
    assert all(
        not torch.equal(generator_states[left], generator_states[right])
        for left in range(len(generator_states))
        for right in range(left + 1, len(generator_states))
    )
    assert inference_sequence_lengths == [1, 2, 3, 4, 5, 6] * 2
    assert len(aurora_rollout_inputs) == 12
    for first_member_input, second_member_input in zip(
        aurora_rollout_inputs[:6], aurora_rollout_inputs[6:], strict=True
    ):
        # feedback_to_rollout=false: different stochastic/Mamba members are
        # recorded, but the deterministic Aurora trajectory remains identical.
        assert torch.equal(first_member_input, second_member_input)

    feedback_config = copy.deepcopy(config)
    feedback_config["rollout"]["rollout_num_steps"] = 2
    feedback_config["model"]["refinement"]["feedback_to_rollout"] = True
    original_refinement_config = loaded.refinement_config
    loaded.refinement_config = dataclasses.replace(
        original_refinement_config,
        feedback_to_rollout=True,
    )
    feedback_inputs = []
    deterministic_outputs = []
    feedback_temporal_lengths = []
    feedback_input_hook = loaded.aurora.register_forward_pre_hook(
        lambda _module, args: feedback_inputs.append(
            {
                "tcno2": args[0].surf_vars["tcno2"].detach().cpu().clone(),
                "no2": args[0].atmos_vars["no2"].detach().cpu().clone(),
            }
        )
    )
    deterministic_output_hook = loaded.aurora.register_forward_hook(
        lambda _module, _args, output: deterministic_outputs.append(
            {
                "tcno2": output.surf_vars["tcno2"].detach().cpu().clone(),
                "no2": output.atmos_vars["no2"].detach().cpu().clone(),
            }
        )
    )
    feedback_temporal_hook = loaded.temporal.core.surf_heads[
        "tcno2"
    ].register_forward_hook(
        lambda _module, args, _output: feedback_temporal_lengths.append(
            int(args[0].shape[1])
        )
    )
    try:
        feedback_predictions = ft.run_rollout(
            loaded,
            dataset,
            samples[0],
            feedback_config,
            specs,
            "cpu",
            refinement_seed=100,
        )
    finally:
        feedback_input_hook.remove()
        deterministic_output_hook.remove()
        feedback_temporal_hook.remove()
        loaded.refinement_config = original_refinement_config
    assert feedback_temporal_lengths == [1, 2]
    assert len(feedback_inputs) == len(deterministic_outputs) == 2
    torch.testing.assert_close(
        feedback_inputs[1]["tcno2"][:, -1:],
        feedback_predictions[0].surf_vars["tcno2"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        feedback_inputs[1]["no2"][:, -1:],
        feedback_predictions[0].atmos_vars["no2"],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        feedback_predictions[0].surf_vars["tcno2"],
        deterministic_outputs[0]["tcno2"],
    )
    refined_levels = set(loaded.packing.levels_for("no2"))
    unselected_level_indices = torch.tensor(
        [
            index
            for index, level in enumerate(config["data"]["atmos_levels"])
            if float(level) not in refined_levels
        ],
        dtype=torch.long,
    )
    assert unselected_level_indices.numel() > 0
    for prediction, deterministic in zip(
        feedback_predictions, deterministic_outputs, strict=True
    ):
        torch.testing.assert_close(
            prediction.atmos_vars["no2"].index_select(
                2, unselected_level_indices
            ),
            deterministic["no2"].index_select(2, unselected_level_indices),
            rtol=0.0,
            atol=0.0,
        )

    repeat_config = copy.deepcopy(config)
    repeat_config["rollout"]["rollout_num_steps"] = 2
    repeated_predictions = ft.run_rollout(
        loaded,
        dataset,
        samples[0],
        repeat_config,
        specs,
        "cpu",
        refinement_seed=100,
    )
    for expected, actual in zip(
        member_predictions[0][:2], repeated_predictions, strict=True
    ):
        torch.testing.assert_close(
            expected.surf_vars["tcno2"],
            actual.surf_vars["tcno2"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            expected.atmos_vars["no2"],
            actual.atmos_vars["no2"],
            rtol=0.0,
            atol=0.0,
        )

    derived_seed_calls = []

    def _record_derived_seed(base_seed, initialization_time, member_index=0):
        derived_seed_calls.append(
            (int(base_seed), np.datetime64(initialization_time, "ns"), member_index)
        )
        return 909

    monkeypatch.setattr(ft, "derive_refinement_seed", _record_derived_seed)
    automatic_seed_config = copy.deepcopy(config)
    automatic_seed_config["rollout"]["rollout_num_steps"] = 1
    automatic_seed_predictions = ft.run_rollout(
        loaded,
        dataset,
        samples[0],
        automatic_seed_config,
        specs,
        "cpu",
    )
    expected_initialization = np.datetime64(
        dataset.time.values[samples[0]["anchor_index"]], "ns"
    )
    assert len(automatic_seed_predictions) == 1
    assert derived_seed_calls == [(1234, expected_initialization, 0)]

    original_refinement_config = loaded.refinement_config
    loaded.refinement_config = dataclasses.replace(
        original_refinement_config, seed=None
    )
    null_seed_config = copy.deepcopy(config)
    null_seed_config["rollout"]["rollout_num_steps"] = 1
    try:
        null_seed_predictions = ft.run_rollout(
            loaded,
            dataset,
            samples[0],
            null_seed_config,
            specs,
            "cpu",
        )
    finally:
        loaded.refinement_config = original_refinement_config
    assert len(null_seed_predictions) == 1

    too_long_rollout = copy.deepcopy(config)
    too_long_rollout["rollout"]["rollout_num_steps"] = 7
    with pytest.raises(
        ValueError,
        match="exceeds the trained temporal-Mamba horizon",
    ):
        ft.run_rollout(
            loaded,
            dataset,
            samples[0],
            too_long_rollout,
            specs,
            "cpu",
        )

    combined = xr.concat(member_datasets, dim="member", join="exact")
    combined = combined.assign_coords(member=np.arange(2))
    output_path = tmp_path / "rollout_predictions.nc"
    combined.to_netcdf(output_path)
    combined.close()
    for member_dataset in member_datasets:
        member_dataset.close()

    with xr.open_dataset(output_path) as output:
        assert set(output.data_vars) == {"no2", "tcno2"}
        assert output.sizes == {
            "member": 2,
            "time": 6,
            "level": 13,
            "latitude": 6,
            "longitude": 6,
        }
        np.testing.assert_allclose(output.lead_time.values, [12, 24, 36, 48, 60, 72])
        assert output.time.values[0] == anchor_time + np.timedelta64(12, "h")
        assert output.time.values[-1] == anchor_time + np.timedelta64(72, "h")
        assert output.no2.attrs["units"] == "kg kg-1"
        assert output.tcno2.attrs["units"] == "kg m-2"
        assert output.level.attrs["units"] == "hPa"
        np.testing.assert_allclose(
            output.level.values,
            np.asarray(config["data"]["atmos_levels"], dtype=np.float64),
        )
        np.testing.assert_allclose(output.latitude.values, dataset.latitude.values)
        np.testing.assert_allclose(output.longitude.values, dataset.longitude.values)
        assert np.isfinite(output.no2.values).all()
        assert np.isfinite(output.tcno2.values).all()
        assert float(output.no2.min()) >= 0.0
        assert float(output.tcno2.min()) >= 0.0
        assert float(output.no2.max()) < 1.0e-3
        assert float(output.tcno2.max()) < 1.0
        assert not np.array_equal(
            output.tcno2.isel(member=0).values,
            output.tcno2.isel(member=1).values,
        )

    incompatible = copy.deepcopy(config)
    incompatible["model"]["refinement"]["transformer"]["embedding_dim"] = 32
    with pytest.raises(ValueError, match="Checkpoint unified refinement mismatch"):
        ft.load_model_from_checkpoint(
            incompatible,
            specs,
            checkpoint_path,
            lon=dataset.longitude.values,
            lat=dataset.latitude.values,
            map_location="cpu",
            autocast=False,
        )

    incompatible_temporal = copy.deepcopy(config)
    incompatible_temporal["model"]["mamba_temporal_channels"] = 8
    with pytest.raises(
        ValueError,
        match="model.mamba_temporal_channels: saved=4, current=8",
    ):
        ft.load_model_from_checkpoint(
            incompatible_temporal,
            specs,
            checkpoint_path,
            lon=dataset.longitude.values,
            lat=dataset.latitude.values,
            map_location="cpu",
            autocast=False,
        )

    incompatible_temporal_enabled = copy.deepcopy(config)
    incompatible_temporal_enabled["model"]["mamba_temporal_enabled"] = False
    with pytest.raises(
        ValueError,
        match="model.mamba_temporal_enabled: saved=True, current=False",
    ):
        ft.load_model_from_checkpoint(
            incompatible_temporal_enabled,
            specs,
            checkpoint_path,
            lon=dataset.longitude.values,
            lat=dataset.latitude.values,
            map_location="cpu",
            autocast=False,
        )
    # Training resume may explicitly add/remove only the temporal module; the
    # inference loader above remains strict.
    model_factory.validate_unified_checkpoint_contract(
        loaded,
        checkpoint,
        incompatible_temporal_enabled,
        allow_temporal_migration=True,
    )


def test_lead_major_unfolding_preserves_batch_trajectories() -> None:
    folded = torch.tensor(
        [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
        dtype=torch.float32,
    ).reshape(12, 1, 1, 1)
    lead_index = torch.arange(6).repeat_interleave(2)
    sequence = AuroraTwoPhaseRefiner.lead_major_to_sequence(
        folded, lead_index, field_name="sentinel"
    )
    assert sequence.shape == (2, 6, 1, 1, 1)
    assert sequence[0, :, 0, 0, 0].tolist() == [10, 20, 30, 40, 50, 60]
    assert sequence[1, :, 0, 0, 0].tolist() == [11, 21, 31, 41, 51, 61]


def test_temporal_mamba_is_causal_and_accepts_arbitrary_positive_width() -> None:
    torch.manual_seed(7)
    temporal = MambaTemporalModule(
        surf_vars=["tcno2"],
        channels=10,
        d_state=2,
        n_layers=1,
        d_conv=2,
        expand=1,
        lon_periodic=False,
    ).eval()
    with torch.no_grad():
        temporal.surf_heads["tcno2"].decoder.weight.fill_(0.1)
    prefix = torch.randn(2, 3, 4, 5)
    first = torch.cat([prefix, torch.randn(2, 3, 4, 5)], dim=1)
    second = torch.cat([prefix, torch.randn(2, 3, 4, 5) + 5.0], dim=1)
    first_correction = temporal.temporal_residual(first, "tcno2", "surf")
    second_correction = temporal.temporal_residual(second, "tcno2", "surf")
    torch.testing.assert_close(
        first_correction[:, :3], second_correction[:, :3], rtol=0.0, atol=0.0
    )
    assert not torch.equal(first_correction[:, 3:], second_correction[:, 3:])


def _enable_temporal_for_validation(config: dict) -> dict:
    config["model"]["mamba_temporal_enabled"] = True
    config["training"]["mamba_temporal_weight"] = 1.0
    return config


def test_temporal_config_defaults_off_and_rejects_invalid_sequence_contract() -> None:
    disabled = _raw_config()
    for key in (
        "mamba_temporal_enabled",
        "mamba_temporal_channels",
        "mamba_temporal_state",
        "mamba_temporal_layers",
        "mamba_temporal_conv",
        "mamba_temporal_expand",
    ):
        disabled["model"].pop(key, None)
    disabled["training"].pop("mamba_temporal_weight", None)
    ft.validate_config(disabled, NO2_DT_CONFIG)

    sparse = _enable_temporal_for_validation(_raw_config())
    sparse["data"]["target_lead_times"] = [1, 3, 6]
    with pytest.raises(ValueError, match=r"expected \[1, 2, 3, 4, 5, 6\]"):
        ft.validate_config(sparse, NO2_DT_CONFIG)

    too_long = _enable_temporal_for_validation(_raw_config())
    too_long["rollout"]["rollout_num_steps"] = 7
    with pytest.raises(
        ValueError,
        match=r"expected 0 or a value <= 6, got 7",
    ):
        ft.validate_config(too_long, NO2_DT_CONFIG)

    no_refinement = _enable_temporal_for_validation(_raw_config())
    no_refinement["model"]["refinement"].update(
        {"enabled": False, "type": "none"}
    )
    no_refinement["model"]["refinement"]["target_space"][
        "residual_clip_standard_deviations"
    ] = 0.0
    with pytest.raises(
        ValueError,
        match="mamba_temporal_enabled requires an active refinement backend",
    ):
        ft.validate_config(no_refinement, NO2_DT_CONFIG)


@pytest.mark.parametrize(
    "field",
    [
        "mamba_temporal_channels",
        "mamba_temporal_state",
        "mamba_temporal_layers",
        "mamba_temporal_conv",
        "mamba_temporal_expand",
    ],
)
def test_temporal_config_rejects_nonpositive_architecture_fields(field) -> None:
    config = _enable_temporal_for_validation(_raw_config())
    config["model"][field] = 0
    with pytest.raises(ValueError, match=rf"model\.{field}"):
        ft.validate_config(config, NO2_DT_CONFIG)


def test_temporal_config_rejects_nonboolean_enable_and_zero_weight() -> None:
    invalid_boolean = _raw_config()
    invalid_boolean["model"]["mamba_temporal_enabled"] = "true"
    with pytest.raises(ValueError, match="must be true or false"):
        ft.validate_config(invalid_boolean, NO2_DT_CONFIG)

    zero_weight = _enable_temporal_for_validation(_raw_config())
    zero_weight["training"]["mamba_temporal_weight"] = 0.0
    with pytest.raises(ValueError, match="mamba_temporal_weight > 0"):
        ft.validate_config(zero_weight, NO2_DT_CONFIG)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_validation_refinement_ensemble_size_must_be_positive_integer(value) -> None:
    config = _raw_config()
    config["training"]["validation_refinement_ensemble_size"] = value
    with pytest.raises(
        ValueError,
        match="training.validation_refinement_ensemble_size",
    ):
        ft.validate_config(config, NO2_DT_CONFIG)


def test_validation_rng_advances_across_batches_and_repeats_across_runs(
    monkeypatch,
) -> None:
    draws_by_run: list[list[float]] = []

    def _run_once() -> None:
        draws: list[float] = []
        generator_ids: list[int] = []

        def _generator(_model, _device):
            return torch.Generator().manual_seed(17)

        def _loss(*_args, refinement_generator=None, **_kwargs):
            assert refinement_generator is not None
            generator_ids.append(id(refinement_generator))
            draws.append(float(torch.rand((), generator=refinement_generator)))
            return torch.tensor(0.0), {}

        monkeypatch.setattr(ft, "_validation_refinement_generator", _generator)
        monkeypatch.setattr(ft, "compute_supervised_loss", _loss)
        metrics = ft.run_validation(
            torch.nn.Identity(),
            xr.Dataset(),
            [{"anchor_index": 0}, {"anchor_index": 1}],
            {"training": {"batch_size": 1}},
            None,
            "cpu",
        )
        assert metrics["num_val_batches"] == 2.0
        assert len(set(generator_ids)) == 1
        assert draws[0] != draws[1]
        draws_by_run.append(draws)

    _run_once()
    _run_once()
    assert draws_by_run[0] == draws_by_run[1]


def test_optional_conditioning_flows_through_shared_training_and_rollout(
    monkeypatch,
) -> None:
    config = _smoke_config()
    config["data"]["target_lead_times"] = [1, 2]
    config["rollout"]["rollout_num_steps"] = 2
    config["model"]["refinement"]["conditioning"].update(
        {"aurora_input_state": True, "static_fields": True}
    )
    dataset = _synthetic_dataset(config)
    ft.validate_config(config, NO2_DT_CONFIG)
    specs = ft.resolve_variable_specs(dataset, config)
    norm_stats = ft.compute_target_normalization_stats(dataset, specs, config)
    samples = ft.build_training_samples(dataset, config)

    monkeypatch.setattr(
        model_factory, "_model_registry", lambda: {"tiny_aurora": TinyAurora}
    )
    model = ft.build_finetune_model(
        config,
        specs,
        lon=dataset.longitude.values,
        lat=dataset.latitude.values,
        norm_stats=norm_stats,
        load_pretrained=False,
        autocast=False,
    )
    assert isinstance(model, AuroraTwoPhaseRefiner)
    assert model.refiner is not None
    conditioning_cfg = model.refinement_config.conditioning
    expected_width = (
        2 * model.packing.num_channels
        + len(specs.static)
        + 1  # configured finite-input mask
        + int(conditioning_cfg.latitude)
        + 2 * int(conditioning_cfg.longitude)
    )
    assert model.refiner.cond_channels == expected_width

    conditioning_widths: list[int] = []
    hook = model.refiner.net.register_forward_pre_hook(
        lambda _module, args: conditioning_widths.append(int(args[1].shape[1]))
    )
    model.train()
    loss, metrics = ft.compute_supervised_loss(
        model,
        dataset,
        samples[0],
        config,
        specs,
        "cpu",
        norm_stats=norm_stats,
    )
    hook.remove()
    assert torch.isfinite(loss)
    assert "refinement" in metrics and "temporal" in metrics
    assert conditioning_widths and set(conditioning_widths) == {expected_width}
    loss.backward()

    original = refinement_integration.refine_batch_prediction
    input_batches = []

    def _record_input_batch(*args, **kwargs):
        input_batches.append(kwargs.get("aurora_input_batch"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        refinement_integration,
        "refine_batch_prediction",
        _record_input_batch,
    )
    predictions = ft.run_rollout(
        model,
        dataset,
        samples[0],
        config,
        specs,
        "cpu",
        refinement_seed=17,
    )
    assert len(predictions) == 2
    assert len(input_batches) == 2 and all(batch is not None for batch in input_batches)
