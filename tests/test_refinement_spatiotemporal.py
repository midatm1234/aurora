"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Contracts for joint spatiotemporal O3 refinement.

These tests assert the properties that cannot be established by inspection:
calendar correctness across leap days and year rollover, longitude-seam safety,
strict temporal causality (prefix invariance) for every backend, trainable
identity initialization, and exact preservation of the previous behaviour when
the new conditioning is disabled.
"""

from __future__ import annotations

import math
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from finetune.refinement.calendar_features import (
    CALENDAR_CONVENTION,
    SCALAR_FEATURE_NAMES,
    SPATIAL_FEATURE_NAMES,
    CalendarFeatureBuilder,
    CalendarSpec,
    cos_solar_zenith_angle,
    days_in_year,
    fractional_day_of_year,
    fractional_utc_hour,
    local_mean_solar_hour,
    require_gregorian,
    valid_times,
    year_phase,
)
from finetune.refinement.config import resolve_refinement_config
from finetune.refinement.temporal import (
    TEMPORAL_BACKENDS,
    TemporalContextCache,
    TemporalContextEncoder,
    assert_increasing_leads,
    build_temporal_encoder,
    elapsed_hours_from_leads,
)
from finetune.refinement.trajectory import (
    ensemble_crps,
    exceedance_brier,
    member_time_weighted_mean,
    member_window_maximum,
    residual_autocorrelation,
    tendency_loss,
    threshold_weighted_crps,
)
from finetune.refinement.two_phase import build_two_phase_refiner
from finetune.refinement.vertical import VerticalChannelEncoder, is_column_channel

from tests.refinement_fixtures import DummySpec, build_packing, refinement_config

REAL_BACKENDS = tuple(b for b in TEMPORAL_BACKENDS if b != "none")


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_solar_geometry_follows_coordinate_device(device) -> None:
    # Meta exercises accelerator-style placement without allocating a GPU.
    result = cos_solar_zenith_angle(
        [datetime(2024, 6, 21, 12)],
        torch.tensor([30.0, 60.0], device=device),
        torch.tensor([0.0, 90.0], device=device),
    )
    assert result.device.type == device
    assert result.shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# A. Calendar correctness
# ---------------------------------------------------------------------------


def test_days_in_year_uses_the_gregorian_leap_rule() -> None:
    assert days_in_year(2024) == 366
    assert days_in_year(2023) == 365
    assert days_in_year(2000) == 366
    assert days_in_year(1900) == 365
    assert days_in_year(2100) == 365


def test_day_of_year_is_leap_aware_not_day_of_month() -> None:
    # The historical Aurora encoder defect used ``.day`` (day of month). On
    # 1 March the two differ by 59/60, which is what this pins down.
    assert fractional_day_of_year(datetime(2024, 3, 1)) == pytest.approx(60.0)
    assert fractional_day_of_year(datetime(2023, 3, 1)) == pytest.approx(59.0)
    assert fractional_day_of_year(datetime(2024, 2, 29)) == pytest.approx(59.0)


def test_year_phase_is_continuous_across_the_year_boundary() -> None:
    before = year_phase(datetime(2023, 12, 31, 23, 59))
    after = year_phase(datetime(2024, 1, 1, 0, 1))
    assert before > 0.999
    assert after < 0.001
    # The encoded angle must be continuous even though the phase wraps.
    two_pi = 2.0 * math.pi
    gap = math.hypot(
        math.sin(two_pi * before) - math.sin(two_pi * after),
        math.cos(two_pi * before) - math.cos(two_pi * after),
    )
    assert gap < 1e-3


def test_leap_day_has_a_distinct_finite_phase() -> None:
    phase = year_phase(datetime(2024, 2, 29, 12))
    assert 0.0 < phase < 1.0
    assert phase != year_phase(datetime(2024, 2, 28, 12))


def test_fractional_utc_hour_includes_sub_hour_precision() -> None:
    assert fractional_utc_hour(datetime(2024, 1, 1, 3, 30, 30)) == pytest.approx(
        3 + 30 / 60 + 30 / 3600
    )
    assert fractional_utc_hour(datetime(2024, 1, 1, 0, 0, 0)) == 0.0


def test_aware_timestamps_are_converted_to_utc_not_local_time() -> None:
    aware = datetime(2024, 6, 1, 12, tzinfo=timezone(timedelta(hours=5)))
    assert fractional_utc_hour(aware) == pytest.approx(7.0)


def test_non_gregorian_calendars_are_rejected_explicitly() -> None:
    class FakeCftime:
        calendar = "360_day"
        year, month, day, hour = 2024, 1, 1, 0

    with pytest.raises(ValueError, match="360_day"):
        require_gregorian(FakeCftime())
    require_gregorian(datetime(2024, 1, 1))


def test_local_mean_solar_hour_wraps_and_tracks_longitude() -> None:
    utc = torch.tensor([12.0])
    lon = torch.tensor([0.0, 180.0, 359.0])
    solar = local_mean_solar_hour(utc[:, None], lon[None, :])
    assert solar[0, 0] == pytest.approx(12.0)
    assert solar[0, 1] == pytest.approx(0.0)
    assert 0.0 <= float(solar[0, 2]) < 24.0


def test_valid_times_uses_each_frames_own_lead() -> None:
    init = (datetime(2024, 12, 31, 18),) * 3
    stamps = valid_times(init, torch.tensor([6.0, 12.0, 30.0]))
    assert stamps[0] == datetime(2025, 1, 1, 0)
    assert stamps[2] == datetime(2025, 1, 2, 0)
    # Different leads must give different calendar features.
    assert len({year_phase(s) for s in stamps}) == 3


def test_cos_solar_zenith_matches_the_aurora_insolation_geometry() -> None:
    from aurora.insolation import insolation

    lat = np.array([60.0, 0.0, -60.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    # Compare one date at a time: ``insolation`` additionally multiplies by the
    # Earth-Sun distance factor rho**-2, which is constant for a given day but
    # differs between solstices. Within a date the two must agree exactly up to
    # that constant, so the correlation is 1 and the ratio is uniform.
    for stamp in (datetime(2024, 6, 21, 12), datetime(2024, 12, 21, 0)):
        reference = insolation([stamp], lat, lon, enforce_2d=True).ravel()
        ours = (
            cos_solar_zenith_angle(
                [stamp], torch.from_numpy(lat).float(), torch.from_numpy(lon).float()
            )
            .numpy()
            .ravel()
        )
        assert np.corrcoef(reference, ours)[0, 1] > 0.99999
        ratio = reference / np.where(np.abs(ours) < 1e-6, np.nan, ours)
        assert np.nanstd(ratio) < 1e-3


def test_cos_solar_zenith_is_seam_safe_at_the_dateline() -> None:
    stamps = [datetime(2024, 3, 21, 6)]
    lat = torch.tensor([45.0])
    wrapped = cos_solar_zenith_angle(stamps, lat, torch.tensor([0.0]))
    same = cos_solar_zenith_angle(stamps, lat, torch.tensor([360.0]))
    assert torch.allclose(wrapped, same, atol=1e-5)


def test_scalar_features_keep_the_four_time_coordinates_separate() -> None:
    builder = CalendarFeatureBuilder([10.0, 0.0], [0.0, 90.0])
    init = (datetime(2024, 5, 1, 0), datetime(2024, 5, 1, 0))
    spec = CalendarSpec(init_time=init, lead_hours=torch.tensor([12.0, 18.0]))
    features = builder.scalar_features(spec)
    assert features.shape == (2, len(SCALAR_FEATURE_NAMES))
    names = list(SCALAR_FEATURE_NAMES)
    # Same initialization: the init-cycle features agree ...
    for key in ("init_cycle_sin", "init_cycle_cos", "init_season_sin"):
        index = names.index(key)
        assert features[0, index] == pytest.approx(float(features[1, index]))
    # ... while the valid-time UTC features differ, because the leads differ.
    utc = names.index("valid_utc_sin")
    assert abs(float(features[0, utc]) - float(features[1, utc])) > 1e-3
    # A lead of exactly one day returns to the same UTC hour but a different
    # seasonal phase, which is why the two are separate features.
    day_apart = builder.scalar_features(
        CalendarSpec(init_time=init, lead_hours=torch.tensor([12.0, 36.0]))
    )
    assert day_apart[0, utc] == pytest.approx(float(day_apart[1, utc]), abs=1e-6)
    season = names.index("valid_season_sin")
    assert abs(float(day_apart[0, season]) - float(day_apart[1, season])) > 1e-4


def test_spatial_features_have_the_documented_order_and_extent() -> None:
    builder = CalendarFeatureBuilder([90.0, 0.0, -90.0], [0.0, 120.0, 240.0])
    spec = CalendarSpec(
        init_time=(datetime(2024, 1, 15, 0),), lead_hours=torch.tensor([24.0])
    )
    spatial = builder.spatial_features(spec)
    assert spatial.shape == (1, len(SPATIAL_FEATURE_NAMES), 3, 3)
    names = list(SPATIAL_FEATURE_NAMES)
    unit = (
        spatial[0, names.index("geo_x")] ** 2
        + spatial[0, names.index("geo_y")] ** 2
        + spatial[0, names.index("geo_z")] ** 2
    )
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5)
    latitude = spatial[0, names.index("latitude_normalized")]
    assert latitude[0, 0] == pytest.approx(1.0)
    assert latitude[2, 0] == pytest.approx(-1.0)


def test_calendar_convention_is_recorded_for_the_checkpoint() -> None:
    builder = CalendarFeatureBuilder([0.0], [0.0])
    convention = builder.convention()
    assert convention["calendar_convention"] == CALENDAR_CONVENTION
    assert convention["scalar_feature_names"] == list(SCALAR_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# B. Vertical / column identity
# ---------------------------------------------------------------------------


def test_column_quantities_are_detected_by_units_not_invented_levels() -> None:
    assert is_column_channel("gtco3", "surf", None, "kg m-2")
    assert is_column_channel("tco3", "surf", None, "DU")
    assert not is_column_channel("go3", "atmos", 500.0, "kg kg-1")
    assert not is_column_channel("t2m", "surf", None, "K")
    assert is_column_channel("weird", "surf", None, "unknown", column_variables=["weird"])


def test_vertical_encoder_marks_the_column_as_pressure_not_applicable() -> None:
    packing = build_packing(8, 12)
    encoder = VerticalChannelEncoder(packing, 8)
    assert encoder.column_channels == (0,)  # gtco3
    assert encoder.pressure_channels == (1, 2)  # go3 at 500 / 850 hPa
    names = list(encoder.describe()["vertical_feature_names"])
    applicable = encoder.static_features[:, names.index("pressure_applicable")]
    assert float(applicable[0]) == 0.0
    # The column must not be given a fabricated pressure of any kind.
    log_pressure = encoder.static_features[:, names.index("log_pressure_normalized")]
    assert float(log_pressure[0]) == 0.0
    assert float(log_pressure[1]) < 0.0  # 500 hPa is below the 1000 hPa reference


def test_vertical_encoder_orders_pressure_continuously() -> None:
    packing = build_packing(8, 12)
    encoder = VerticalChannelEncoder(packing, 8)
    names = list(encoder.describe()["vertical_feature_names"])
    log_pressure = encoder.static_features[:, names.index("log_pressure_normalized")]
    # 500 hPa is higher in the atmosphere than 850 hPa, so its coordinate is smaller.
    assert float(log_pressure[1]) < float(log_pressure[2])


def test_vertical_encoder_rejects_pascal_valued_levels() -> None:
    from finetune.refinement.packing import ChannelSpec, FieldPacking

    packing = FieldPacking(
        channels=(
            ChannelSpec(0, "go3", "go3", "atmos", 50_000.0, 0, units="kg kg-1"),
        ),
        lat=(0.0,),
        lon=(0.0,),
    )
    with pytest.raises(ValueError, match="hPa"):
        VerticalChannelEncoder(packing, 8)


def test_vertical_modulation_is_identity_at_construction() -> None:
    packing = build_packing(8, 12)
    encoder = VerticalChannelEncoder(packing, 8)
    state = torch.randn(2, packing.num_channels, 8, 12)
    assert torch.equal(encoder.apply_channel_modulation(state), state)


# ---------------------------------------------------------------------------
# C. Temporal causality
# ---------------------------------------------------------------------------


def _encoder(backend: str, **kwargs) -> TemporalContextEncoder:
    options = {
        "input_channels": 3,
        "context_channels": 4,
        "backend": backend,
        "hidden_channels": 16,
        "layers": 2,
        "spatial_stride": 4,
        "metadata_features": 5,
        "lon_periodic": True,
    }
    options.update(kwargs)
    return TemporalContextEncoder(**options)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_temporal_context_is_prefix_invariant(backend: str) -> None:
    """The context at lead j must not change when later leads change."""
    torch.manual_seed(0)
    encoder = _encoder(backend)
    # Train one step so the zero-initialized output projection is non-zero and
    # the invariance test is not trivially satisfied by an all-zero output.
    sequence = torch.randn(2, 5, 3, 16, 24)
    metadata = torch.randn(2, 5, 5)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5)
    (encoder(sequence, metadata=metadata) - 1.0).pow(2).mean().backward()
    optimizer.step()
    encoder.zero_grad()

    with torch.no_grad():
        full = encoder(sequence, metadata=metadata)
        assert float(full.abs().max()) > 0.0, "context is still identically zero"
        altered = sequence.clone()
        altered[:, 3:] = torch.randn_like(altered[:, 3:])
        altered_metadata = metadata.clone()
        altered_metadata[:, 3:] = torch.randn_like(altered_metadata[:, 3:])
        changed = encoder(altered, metadata=altered_metadata)
    assert torch.allclose(full[:, :3], changed[:, :3], atol=1e-6)
    assert not torch.allclose(full[:, 3:], changed[:, 3:], atol=1e-6)


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_temporal_context_is_identity_at_init_and_then_trainable(backend: str) -> None:
    torch.manual_seed(0)
    encoder = _encoder(backend)
    sequence = torch.randn(2, 4, 3, 16, 24)
    metadata = torch.randn(2, 4, 5)
    with torch.no_grad():
        assert float(encoder(sequence, metadata=metadata).abs().max()) == 0.0

    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5)
    for _ in range(3):
        optimizer.zero_grad()
        (encoder(sequence, metadata=metadata) - 1.0).pow(2).mean().backward()
        optimizer.step()
    optimizer.zero_grad()
    (encoder(sequence, metadata=metadata) - 1.0).pow(2).mean().backward()

    # No double-zero deadlock: gradient must reach the temporal mixer itself.
    mixer_gradient = sum(
        float(p.grad.abs().sum()) for p in encoder.mixer.parameters() if p.grad is not None
    )
    stem_gradient = sum(
        float(p.grad.abs().sum()) for p in encoder.stem.parameters() if p.grad is not None
    )
    assert mixer_gradient > 0.0
    assert stem_gradient > 0.0


@pytest.mark.parametrize("backend", REAL_BACKENDS)
def test_temporal_context_is_spatially_contextual(backend: str) -> None:
    """A change at one grid cell must be able to affect its neighbours."""
    torch.manual_seed(0)
    encoder = _encoder(backend, spatial_stride=1)
    torch.nn.init.normal_(encoder.out_proj.weight, std=0.1)
    sequence = torch.zeros(1, 3, 3, 8, 8)
    metadata = torch.zeros(1, 3, 5)
    with torch.no_grad():
        base = encoder(sequence, metadata=metadata)
        poked = sequence.clone()
        poked[0, :, :, 4, 4] = 5.0
        changed = encoder(poked, metadata=metadata)
    difference = (base - changed).abs().sum(dim=(0, 1, 2))
    assert float(difference[4, 4]) > 0.0
    assert float(difference[3, 4]) > 0.0, "per-pixel recurrence cannot move information"


def test_temporal_encoder_rejects_over_long_trajectories() -> None:
    encoder = _encoder("causal_conv", max_sequence_length=3)
    with pytest.raises(ValueError, match="max_sequence_length"):
        encoder(torch.randn(1, 4, 3, 8, 8), metadata=torch.randn(1, 4, 5))


def test_temporal_encoder_requires_the_metadata_it_was_built_with() -> None:
    encoder = _encoder("causal_conv")
    with pytest.raises(ValueError, match="metadata"):
        encoder(torch.randn(1, 3, 3, 8, 8))


def test_none_backend_builds_the_spatial_only_control() -> None:
    assert (
        build_temporal_encoder(
            "none", input_channels=3, lon_periodic=False, metadata_features=0
        )
        is None
    )


def test_elapsed_hours_expose_gaps_instead_of_compressing_them() -> None:
    leads = torch.tensor([[12.0, 24.0, 60.0]])
    elapsed = elapsed_hours_from_leads(leads)
    assert torch.allclose(elapsed, torch.tensor([[12.0, 12.0, 36.0]]))


def test_separate_initializations_cannot_be_merged_into_one_trajectory() -> None:
    assert_increasing_leads(torch.tensor([[12.0, 24.0, 36.0]]))
    with pytest.raises(ValueError, match="strictly increasing"):
        assert_increasing_leads(torch.tensor([[12.0, 24.0, 12.0]]))


def test_temporal_context_cache_keys_isolate_members_and_initializations() -> None:
    cache = TemporalContextCache()
    cache.put(("init-a", 0, 1), torch.ones(2))
    assert cache.get(("init-a", 1, 1)) is None
    assert cache.get(("init-b", 0, 1)) is None
    assert torch.equal(cache.get(("init-a", 0, 1)), torch.ones(2))
    with pytest.raises(ValueError, match="triples"):
        cache.get(("init-a", 0))


# ---------------------------------------------------------------------------
# D. Integration with the four unified refiners
# ---------------------------------------------------------------------------

ALL_HEADS = (
    "flow_matching_conv_unet",
    "flow_matching_transformer",
    "diffusion_unet",
    "diffusion_transformer",
)


def _spatiotemporal_model(head: str, backend: str = "causal_conv"):
    packing = build_packing(16, 24, lon_periodic=True)
    config = refinement_config(
        head,
        conditioning={
            "calendar": True,
            "solar_geometry": True,
            "vertical_identity": True,
        },
        temporal={
            "backend": backend,
            "context_channels": 4,
            "hidden_channels": 16,
            "spatial_stride": 4,
        },
    )
    model = build_two_phase_refiner(None, packing, config)
    model.initialize_refiner(model.conditioning_channels())
    return model, packing


def _trajectory_batch(model, packing, batch: int = 2, steps: int = 3):
    from finetune.refinement.calendar_features import CalendarFeatureBuilder as Builder

    torch.manual_seed(0)
    height, width = 16, 24
    leads = torch.tensor([[24.0, 48.0, 72.0]][0][:steps]).expand(batch, steps).contiguous()
    init = [datetime(2024, 2, 28, 12), datetime(2024, 7, 4, 0)][:batch]
    sequence = torch.randn(batch, steps, packing.num_channels, height, width)
    context = model.build_temporal_context(sequence, lead_hours=leads, init_time=init)
    flat = batch * steps
    rollout = sequence.reshape(flat, packing.num_channels, height, width)
    calendar = model.calendar_spec(
        Builder.expand_initializations(init, steps), leads.reshape(-1)
    )
    return {
        "rollout": rollout,
        "target": rollout + 0.3 * torch.randn_like(rollout),
        "leads": leads,
        "calendar": calendar,
        "context": context.reshape(flat, 4, height, width),
        "lead_index": torch.arange(flat) % steps,
    }


@pytest.mark.parametrize("head", ALL_HEADS)
def test_every_head_accepts_joint_spatiotemporal_conditioning(head: str) -> None:
    model, packing = _spatiotemporal_model(head)
    batch = _trajectory_batch(model, packing)
    output = model.training_step(
        batch["rollout"],
        batch["target"],
        forecast_lead_time=batch["leads"].reshape(-1),
        lead_index=batch["lead_index"],
        calendar=batch["calendar"],
        temporal_context=batch["context"],
    )
    assert torch.isfinite(output.losses["total_loss"])


@pytest.mark.parametrize("head", ALL_HEADS)
def test_temporal_and_calendar_parameters_receive_gradient(head: str) -> None:
    """Guards against a zero-gate / zero-projection deadlock end to end."""
    model, packing = _spatiotemporal_model(head)
    data = _trajectory_batch(model, packing)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    height, width = 16, 24
    steps = data["leads"].shape[1]
    sequence = data["rollout"].reshape(2, steps, packing.num_channels, height, width)
    init = [datetime(2024, 2, 28, 12), datetime(2024, 7, 4, 0)]

    for _ in range(4):
        optimizer.zero_grad()
        context = model.build_temporal_context(
            sequence, lead_hours=data["leads"], init_time=init
        )
        output = model.training_step(
            data["rollout"],
            data["target"],
            forecast_lead_time=data["leads"].reshape(-1),
            lead_index=data["lead_index"],
            calendar=data["calendar"],
            temporal_context=context.reshape(2 * steps, 4, height, width),
        )
        output.losses["total_loss"].backward()
        optimizer.step()

    def _gradient(module) -> float:
        return sum(
            float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None
        )

    assert _gradient(model.temporal_context) > 0.0
    assert _gradient(model.temporal_context.mixer) > 0.0
    assert _gradient(model.refiner.net.cond_embed) > 0.0
    assert _gradient(model.vertical_encoder) > 0.0


@pytest.mark.parametrize("head", ALL_HEADS)
def test_disabled_conditioning_preserves_the_previous_contract(head: str) -> None:
    """A recipe that does not opt in keeps its exact widths and code path."""
    packing = build_packing(16, 24)
    model = build_two_phase_refiner(None, packing, refinement_config(head))
    model.initialize_refiner(model.conditioning_channels())
    assert model.temporal_context is None
    assert model.vertical_encoder is None
    assert model.refiner.metadata_features == 0
    assert model.refiner.temporal_context_channels == 0
    assert model.refiner.net.cond_channels == model.conditioning_channels()


def test_supplying_context_without_a_temporal_backend_is_an_error() -> None:
    packing = build_packing(16, 24)
    model = build_two_phase_refiner(None, packing, refinement_config("diffusion_unet"))
    model.initialize_refiner(model.conditioning_channels())
    rollout = torch.randn(2, packing.num_channels, 16, 24)
    with pytest.raises(RuntimeError, match="temporal.backend is"):
        model.training_step(
            rollout,
            rollout.clone(),
            forecast_lead_time=torch.tensor([24.0, 48.0]),
            temporal_context=torch.zeros(2, 4, 16, 24),
        )


def test_enabled_temporal_backend_requires_context() -> None:
    model, packing = _spatiotemporal_model("diffusion_unet")
    data = _trajectory_batch(model, packing)
    with pytest.raises(RuntimeError, match="no temporal context"):
        model.training_step(
            data["rollout"],
            data["target"],
            forecast_lead_time=data["leads"].reshape(-1),
            calendar=data["calendar"],
        )


def test_solar_conditioning_requires_a_calendar_spec() -> None:
    model, packing = _spatiotemporal_model("diffusion_unet")
    with pytest.raises(RuntimeError, match="CalendarSpec"):
        model.build_conditioning(torch.randn(2, packing.num_channels, 16, 24))


def test_temporal_config_is_rejected_for_the_legacy_wrapper() -> None:
    with pytest.raises(Exception, match="legacy"):
        resolve_refinement_config(
            {
                "model": {
                    "refinement": {
                        "enabled": True,
                        "type": "flow_matching_unet",
                        "temporal": {"backend": "causal_conv"},
                    }
                }
            }
        )


def test_ensemble_sampling_aligns_conditioning_with_members() -> None:
    """Member replication uses repeat_interleave; per-frame conditioning must follow.

    A plain ``repeat`` would pair frame 0's calendar with sample 1's field. The
    misalignment is invisible in the output shape, so it is pinned numerically:
    every member of one frame must receive that frame's own conditioning.
    """
    model, packing = _spatiotemporal_model("diffusion_unet")
    data = _trajectory_batch(model, packing)
    refiner = model.refiner
    frames = data["rollout"].shape[0]
    members = 4

    calendar = model.calendar_metadata(data["calendar"], device=data["rollout"].device)
    with refiner.use_frame_conditioning(
        metadata=calendar, temporal_context=data["context"]
    ):
        expanded = refiner._metadata_for(frames * members)
        context = refiner.augment_conditioning(
            torch.zeros(
                frames * members, refiner.cond_channels, 16, 24
            )
        )[:, refiner.cond_channels :]

    for frame in range(frames):
        for member in range(members):
            row = frame * members + member
            assert torch.equal(
                expanded[row, : refiner.calendar_features], calendar[frame]
            )
            assert torch.equal(context[row], data["context"][frame])


def test_ensemble_refinement_runs_end_to_end_with_members() -> None:
    model, packing = _spatiotemporal_model("diffusion_unet")
    data = _trajectory_batch(model, packing)
    output = model.refine(
        data["rollout"],
        forecast_lead_time=data["leads"].reshape(-1),
        calendar=data["calendar"],
        temporal_context=data["context"],
        ensemble_size=3,
        return_members=True,
        seed=11,
    )
    assert output.members is not None
    assert output.members.shape[1] == 3
    assert torch.isfinite(output.members).all()


def test_mismatched_frame_count_is_rejected_rather_than_broadcast() -> None:
    model, packing = _spatiotemporal_model("diffusion_unet")
    data = _trajectory_batch(model, packing)
    refiner = model.refiner
    calendar = model.calendar_metadata(data["calendar"], device=data["rollout"].device)
    with refiner.use_frame_conditioning(
        metadata=calendar, temporal_context=data["context"]
    ), pytest.raises(ValueError, match="divisor"):
        refiner._metadata_for(calendar.shape[0] + 1)


def test_refinement_checkpoint_round_trips_through_the_public_loader() -> None:
    """`load_refinement_state_dict` must restore the temporal encoder.

    ``temporal_context.`` does not start with ``temporal.`` (the ninth
    character is ``_``), so a prefix filter that only knows the legacy Mamba
    prefix silently discards every temporal tensor. Because the context
    projection is zero-initialized, the result is an exact fallback to the
    spatial-only control with no error, no NaN and no warning -- which would
    make a trained temporal model indistinguishable from its ablation control.
    """
    from finetune.refinement.checkpoint import (
        build_refinement_checkpoint,
        load_refinement_state_dict,
    )

    trained, packing = _spatiotemporal_model("diffusion_unet", backend="conv_gru")
    with torch.no_grad():
        for parameter in trained.temporal_context.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
        for parameter in trained.vertical_encoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    payload = build_refinement_checkpoint(
        trained,
        refinement_type="diffusion_unet",
        resolved_config=trained.refinement_config.to_dict(),
        packing=packing,
        aurora_fingerprint="test",
    )

    restored, _ = _spatiotemporal_model("diffusion_unet", backend="conv_gru")
    report = load_refinement_state_dict(restored, payload)
    assert report.missing == []

    for (name, expected), (_, actual) in zip(
        trained.temporal_context.state_dict().items(),
        restored.temporal_context.state_dict().items(),
    ):
        assert torch.equal(expected, actual), f"temporal_context.{name} not restored"
    for name, expected in trained.vertical_encoder.state_dict().items():
        assert torch.equal(expected, restored.vertical_encoder.state_dict()[name])


def test_lead_blocked_and_lead_major_orders_are_distinct_and_invertible() -> None:
    """The trainer packs lead-blocked; the temporal encoder works lead-major.

    Confusing the two produces correct shapes while attaching each frame's
    calendar and temporal context to the wrong sample, so the converters are
    explicit and pinned here.
    """
    batch, steps = 3, 4
    init = [datetime(2024, 1, 1 + i) for i in range(batch)]

    lead_major = CalendarFeatureBuilder.expand_initializations(init, steps)
    lead_blocked = CalendarFeatureBuilder.expand_initializations_lead_blocked(
        init, steps
    )
    assert lead_major[:steps] == (init[0],) * steps
    assert lead_blocked[:batch] == tuple(init)
    assert lead_major != lead_blocked

    sequence = torch.arange(batch * steps, dtype=torch.float32).reshape(batch, steps)
    blocked = CalendarFeatureBuilder.sequence_to_lead_blocked(sequence)
    # Lead-blocked row (position, sample) lives at position * batch + sample.
    for position in range(steps):
        for sample in range(batch):
            assert float(blocked[position * batch + sample]) == float(
                sequence[sample, position]
            )
    restored = CalendarFeatureBuilder.lead_blocked_to_sequence(blocked, batch, steps)
    assert torch.equal(restored, sequence)
    # A bare reshape transposes the axes and would fabricate trajectories.
    assert not torch.equal(blocked.reshape(batch, steps), sequence)


def test_peak_timing_error_uses_the_lead_axis_when_members_equal_steps() -> None:
    """Guards against inferring the truth-mask axis from ``M != S``."""
    from finetune.refinement.trajectory import peak_timing_error

    batch = members = steps = 3
    torch.manual_seed(0)
    ensemble = torch.randn(batch, members, steps, 1, 2, 2)
    truth = torch.randn(batch, steps, 1, 2, 2)
    leads = torch.tensor([[12.0, 24.0, 36.0]]).expand(batch, steps).contiguous()

    unmasked = peak_timing_error(ensemble, truth, leads)
    all_valid = peak_timing_error(
        ensemble,
        truth,
        leads,
        mask=torch.ones_like(ensemble, dtype=torch.bool),
        truth_mask=torch.ones_like(truth, dtype=torch.bool),
    )
    assert float(unmasked) == pytest.approx(float(all_valid))

    with pytest.raises(ValueError, match="truth_mask must match"):
        peak_timing_error(
            ensemble, truth, leads, truth_mask=torch.ones_like(ensemble, dtype=torch.bool)
        )


def test_production_o3_recipe_resolves_and_matches_the_trainer_contract() -> None:
    """The shipped recipe must not enable conditioning the trainer cannot supply."""
    import yaml

    path = Path(__file__).parents[1] / "finetune"
    config = yaml.safe_load(
        (path / "aurora_O3_global_finetune_3day_lead_config.yaml").read_text()
    )
    resolved = resolve_refinement_config(config)
    assert resolved.conditioning.metadata_feature_count == 0
    assert resolved.conditioning.solar_channel_count == 0
    assert resolved.temporal.context_channels_if_active == 0


def test_regional_non_periodic_domain_supports_the_full_stack() -> None:
    """The NO2 US-WEST case is regional: longitude does not wrap.

    The periodic path is covered by the other tests in this file. This one pins
    the replicate-padding path used by every regional recipe, and checks that a
    column quantity is still detected on a non-``gtco3`` variable name.
    """
    from finetune.refinement.packing import FieldPacking

    height, width = 12, 16
    packing = FieldPacking.from_specs(
        [
            DummySpec("no2", "no2", "atmos", [1000.0, 500.0], units="kg kg-1"),
            DummySpec("tcno2", "tcno2", "surf", units="kg m-2"),
        ],
        norm_stats={
            "no2": {"mean": torch.zeros(2), "std": torch.full((2,), 1e-9)},
            "tcno2": {"mean": torch.tensor([1e-5]), "std": torch.tensor([2e-6])},
        },
        atmos_levels=[1000.0, 500.0],
        # US-WEST style window: longitudes in [232, 259.6], no wrap.
        lat=[49.0 - i * (28.0 / (height - 1)) for i in range(height)],
        lon=[232.0 + i * (27.6 / (width - 1)) for i in range(width)],
        lead_times_hours=(12.0, 24.0, 36.0),
        lon_periodic=False,
    )
    assert packing.lon_periodic is False

    config = refinement_config(
        "diffusion_transformer",
        conditioning={
            "calendar": True,
            "solar_geometry": True,
            "vertical_identity": True,
        },
        temporal={
            "backend": "causal_conv",
            "context_channels": 4,
            "hidden_channels": 16,
            "spatial_stride": 4,
        },
    )
    model = build_two_phase_refiner(None, packing, config)
    model.initialize_refiner(model.conditioning_channels())
    assert model.temporal_context.lon_periodic is False
    # A per-area column variable is detected by units, not by the name gtco3.
    # FieldPacking orders surface fields before profile levels, so it is
    # channel 0 here.
    assert model.vertical_encoder.column_channels == (0,)
    assert model.vertical_encoder.pressure_channels == (1, 2)

    batch, steps = 2, 3
    channels = packing.num_channels
    leads = torch.tensor([[12.0, 24.0, 36.0]]).expand(batch, steps).contiguous()
    init = [datetime(2024, 7, 4, 0), datetime(2024, 1, 15, 12)]
    sequence = torch.randn(batch, steps, channels, height, width)
    context = model.build_temporal_context(
        sequence, lead_hours=leads, init_time=init
    )

    flat = batch * steps
    rollout = sequence.reshape(flat, channels, height, width)
    calendar = model.calendar_spec(
        CalendarFeatureBuilder.expand_initializations(init, steps), leads.reshape(-1)
    )
    output = model.training_step(
        rollout,
        rollout + 0.2 * torch.randn_like(rollout),
        forecast_lead_time=leads.reshape(-1),
        lead_index=torch.arange(flat) % steps,
        calendar=calendar,
        temporal_context=context.reshape(flat, 4, height, width),
    )
    assert torch.isfinite(output.losses["total_loss"])

    # Causality must hold on a non-periodic grid too.
    with torch.no_grad():
        altered = sequence.clone()
        altered[:, 2:] = torch.randn_like(altered[:, 2:])
        changed = model.build_temporal_context(
            altered, lead_hours=leads, init_time=init
        )
    assert torch.allclose(context[:, :2], changed[:, :2], atol=1e-6)


def test_spatiotemporal_contract_round_trips_through_a_checkpoint() -> None:
    from finetune.refinement.checkpoint import build_refinement_checkpoint

    model, packing = _spatiotemporal_model("flow_matching_conv_unet", backend="conv_gru")
    payload = build_refinement_checkpoint(
        model,
        refinement_type="flow_matching_conv_unet",
        resolved_config=model.refinement_config.to_dict(),
        packing=packing,
        aurora_fingerprint="test",
    )
    contract = payload["spatiotemporal_contract"]
    assert contract["calendar"]["calendar_convention"] == CALENDAR_CONVENTION
    assert contract["temporal_context"]["temporal_backend"] == "conv_gru"
    assert contract["temporal_context"]["causal"] is True
    assert contract["temporal_mode"] == "causal"
    assert contract["uses_future_raw_aurora"] is False
    assert contract["vertical"]["column_channels"] == [0]

    state = payload["model_state_dict"]
    assert any(key.startswith("temporal_context.") for key in state)
    assert any("vertical_encoder" in key for key in state)

    reloaded, _ = _spatiotemporal_model("flow_matching_conv_unet", backend="conv_gru")
    missing, unexpected = reloaded.load_state_dict(state, strict=False)
    assert [k for k in missing if not k.startswith("aurora")] == []
    assert list(unexpected) == []


@pytest.mark.parametrize("change", ["calendar", "temporal_mode", "missing"])
def test_checkpoint_rejects_changed_spatiotemporal_semantics_before_loading(change) -> None:
    from finetune.refinement.checkpoint import (
        build_refinement_checkpoint,
        load_refinement_state_dict,
    )

    model, _ = _spatiotemporal_model("flow_matching_conv_unet")
    payload = build_refinement_checkpoint(model)
    if change == "missing":
        payload.pop("spatiotemporal_contract")
    elif change == "calendar":
        payload["spatiotemporal_contract"]["calendar"]["calendar_convention"] = "360_day"
    else:
        payload["spatiotemporal_contract"]["temporal_mode"] = "full_trajectory"
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="spatiotemporal_contract"):
        load_refinement_state_dict(model, payload)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_original_training_checkpoint_persists_spatiotemporal_contract(tmp_path) -> None:
    from finetune import aurora_finetune_utils as ft
    from finetune.refinement.checkpoint import load_refinement_state_dict

    model, _ = _spatiotemporal_model("flow_matching_conv_unet")
    path = tmp_path / "best.pt"
    cfg = {"model": {"refinement": model.refinement_config.to_dict()}}
    cfg["model"]["refinement"].pop("from_legacy_keys")
    ft.save_checkpoint(
        path, model, torch.optim.Adam(model.parameters()), None,
        epoch=0, global_step=0, best_val_loss=1.0, config=cfg,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["spatiotemporal_contract"]["temporal_context"]["temporal_backend"] == "causal_conv"
    load_refinement_state_dict(model, payload)


@pytest.mark.parametrize("change,value", [("mode", "full_trajectory"), ("spatial_stride", 2), ("dropout", 0.1)])
def test_unified_contract_checks_context_options_without_tensor_shape_changes(change, value) -> None:
    from finetune.model_factory import validate_unified_checkpoint_contract

    model, packing = _spatiotemporal_model("flow_matching_conv_unet")
    saved_config = {"model": {"refinement": model.refinement_config.to_dict()}}
    saved_config["model"]["refinement"].pop("from_legacy_keys")
    current = copy.deepcopy(saved_config)
    current["model"]["refinement"]["temporal"][change] = value
    checkpoint = {
        "config": saved_config,
        "resolved_refinement_config": model.refinement_config.to_dict(),
        "field_packing": packing.to_dict(),
    }
    with pytest.raises(ValueError, match="unified refinement mismatch: temporal"):
        validate_unified_checkpoint_contract(model, checkpoint, current)


# ---------------------------------------------------------------------------
# E. Trajectory objectives and scores
# ---------------------------------------------------------------------------


def test_tendency_loss_is_zero_for_a_perfect_trajectory() -> None:
    torch.manual_seed(0)
    reference = torch.randn(2, 4, 3, 8, 10)
    delta = torch.full((2, 4), 12.0)
    assert float(tendency_loss(reference.clone(), reference, delta)) == 0.0


def test_tendency_loss_penalizes_smoothing_not_change() -> None:
    torch.manual_seed(0)
    reference = torch.randn(2, 4, 3, 8, 10) * 3.0
    delta = torch.full((2, 4), 12.0)
    smoothed = reference.clone()
    smoothed[:, 1:] = 0.5 * (reference[:, 1:] + reference[:, :-1])
    assert float(tendency_loss(smoothed, reference, delta)) > 0.0
    # A sharp but correct trajectory is not penalized for being sharp.
    sharp = reference * 1.0
    assert float(tendency_loss(sharp, reference, delta)) == 0.0


def test_tendency_loss_uses_real_elapsed_hours() -> None:
    reference = torch.zeros(1, 2, 1, 2, 2)
    prediction = torch.zeros(1, 2, 1, 2, 2)
    prediction[:, 1] = 1.0
    close = tendency_loss(prediction, reference, torch.tensor([[12.0, 6.0]]), penalty="l2")
    far = tendency_loss(prediction, reference, torch.tensor([[12.0, 24.0]]), penalty="l2")
    assert float(close) > float(far)


def test_tendency_loss_rejects_non_positive_gaps() -> None:
    reference = torch.zeros(1, 2, 1, 2, 2)
    with pytest.raises(ValueError, match="strictly positive"):
        tendency_loss(reference, reference, torch.tensor([[12.0, 0.0]]))


@pytest.mark.parametrize("penalty", ["huber", "l2", "l1"])
def test_tendency_loss_masks_missing_observations_before_differentiating(penalty) -> None:
    prediction = torch.arange(4.0).reshape(1, 4, 1, 1, 1).requires_grad_()
    reference = torch.zeros_like(prediction)
    reference[:, 1] = float("nan")
    mask = torch.isfinite(reference)
    weight = torch.ones_like(reference)
    weight[:, 1] = float("nan")
    loss = tendency_loss(
        prediction, reference, torch.full((1, 4), 12.0),
        mask=mask, weight=weight, penalty=penalty,
    )
    expected = tendency_loss(
        prediction[:, 2:], reference[:, 2:], torch.full((1, 2), 12.0),
        penalty=penalty,
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[:, :2], torch.zeros_like(prediction.grad[:, :2]))


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -1.0])
def test_tendency_loss_rejects_invalid_normalization(scale) -> None:
    value = torch.zeros(1, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        tendency_loss(value, value, torch.full((1, 2), 12.0), scale=scale)


def test_tendency_loss_without_any_pair_is_finite_zero() -> None:
    value = torch.full((1, 1, 1, 1, 1), float("nan"), requires_grad=True)
    loss = tendency_loss(value, value, torch.ones(1, 1))
    assert loss.item() == 0.0
    loss.backward()
    assert value.grad.item() == 0.0


def test_event_functionals_are_computed_member_wise_first() -> None:
    torch.manual_seed(0)
    members = torch.randn(2, 5, 4, 1, 3, 3)
    per_member_max = member_window_maximum(members)
    assert per_member_max.shape == (2, 5, 1, 3, 3)
    # The mean of per-member maxima is not the maximum of the ensemble mean.
    max_of_mean = members.mean(dim=1).amax(dim=1)
    assert not torch.allclose(per_member_max.mean(dim=1), max_of_mean)


def test_window_maximum_respects_the_declared_window() -> None:
    members = torch.zeros(1, 1, 4, 1, 2, 2)
    members[0, 0, 3] = 9.0
    assert float(member_window_maximum(members, window=(0, 3)).max()) == 0.0
    assert float(member_window_maximum(members, window=(0, 4)).max()) == 9.0


def test_time_weighted_mean_weights_by_actual_hours() -> None:
    members = torch.zeros(1, 1, 2, 1, 1, 1)
    members[0, 0, 0] = 0.0
    members[0, 0, 1] = 10.0
    even = member_time_weighted_mean(members, torch.tensor([[12.0, 12.0]]))
    skewed = member_time_weighted_mean(members, torch.tensor([[1.0, 23.0]]))
    assert float(even) == pytest.approx(5.0)
    assert float(skewed) > float(even)


def test_fair_crps_requires_at_least_two_members() -> None:
    members = torch.randn(2, 1, 3, 3)
    truth = torch.randn(2, 3, 3)
    with pytest.raises(ValueError, match="at least two"):
        ensemble_crps(members, truth, estimator="fair")
    assert torch.isfinite(ensemble_crps(members, truth, estimator="empirical"))


def test_empirical_crps_with_one_member_equals_mae() -> None:
    members = torch.randn(2, 1, 3, 3)
    truth = torch.randn(2, 3, 3)
    score = ensemble_crps(members, truth, estimator="empirical")
    assert float(score) == pytest.approx(float((members[:, 0] - truth).abs().mean()), abs=1e-6)


def test_threshold_weighted_crps_scores_every_case_not_only_exceedances() -> None:
    # All truth below the threshold: a perfect forecast still scores 0, and a
    # forecast that wrongly predicts exceedance is penalized.
    truth = torch.zeros(1, 2, 2)
    good = torch.zeros(1, 2, 2, 2)
    bad = torch.full((1, 2, 2, 2), 5.0)
    threshold = 1.0
    assert float(threshold_weighted_crps(good, truth, threshold)) == pytest.approx(0.0)
    assert float(threshold_weighted_crps(bad, truth, threshold)) > 0.0


def test_threshold_weighted_crps_supports_a_lower_tail() -> None:
    truth = torch.full((1, 2, 2), -5.0)
    members = torch.zeros(1, 2, 2, 2)
    assert float(threshold_weighted_crps(members, truth, -1.0, tail="lower")) > 0.0


def test_exceedance_brier_reports_reliability_bins() -> None:
    torch.manual_seed(0)
    members = torch.randn(2, 4, 3, 3)
    truth = torch.randn(2, 3, 3)
    score, reliability = exceedance_brier(members, truth, 0.0, return_reliability=True)
    assert 0.0 <= float(score) <= 1.0
    assert int(reliability["count"].sum()) == truth.numel()


def test_residual_autocorrelation_detects_persistent_drift() -> None:
    torch.manual_seed(0)
    reference = torch.randn(4, 5, 2, 4, 4)
    drifting = reference + 3.0  # constant bias: perfectly correlated error
    noisy = reference + torch.randn_like(reference)
    assert float(residual_autocorrelation(noisy, reference)) < 0.3
    # A constant offset has zero variance, so use a slowly varying drift.
    drift = reference + torch.linspace(0.0, 2.0, 5)[None, :, None, None, None]
    assert float(residual_autocorrelation(drift, reference)) > float(
        residual_autocorrelation(noisy, reference)
    )
    assert torch.isfinite(residual_autocorrelation(drifting, reference))
