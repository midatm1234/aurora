"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Regression tests for periodic longitude in the Flow Matching pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune import prepare_train_test_from_netcdf as prep
from finetune.longitude import (
    PeriodicConv2d,
    add_cyclic_column,
    canonical_longitudes,
    dateline_discontinuity,
    lon_cyclic_features,
    longitude_grid_signature,
    longitude_is_periodic,
    periodic_avg_pool2d,
    periodic_bilinear_interpolate,
    resolve_lon_periodic,
)


class LongitudeCoordinateTest(unittest.TestCase):
    def test_conventions_and_duplicate_endpoint_canonicalize_identically(self) -> None:
        lon_360 = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        lon_180 = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])

        canonical_360, index_360 = canonical_longitudes(lon_360)
        canonical_180, index_180 = canonical_longitudes(lon_180)

        np.testing.assert_array_equal(canonical_360, [0.0, 90.0, 180.0, 270.0])
        np.testing.assert_array_equal(canonical_180, canonical_360)
        self.assertEqual(len(index_360), 4)
        self.assertEqual(len(index_180), 4)
        self.assertTrue(longitude_is_periodic(lon_360))
        self.assertTrue(longitude_is_periodic(lon_180))
        self.assertEqual(
            longitude_grid_signature(lon_360), longitude_grid_signature(lon_180),
        )
        self.assertFalse(longitude_is_periodic([250.0, 251.0, 252.0, 253.0]))

    def test_preprocessing_reorders_data_with_longitude(self) -> None:
        ds = xr.Dataset(
            {"v": (("latitude", "longitude"), [[-180.0, -90.0, 0.0, 90.0, 180.0]])},
            coords={"latitude": [0.0], "longitude": [-180.0, -90.0, 0.0, 90.0, 180.0]},
        )
        # Spatial canonicalisation requires at least two latitudes, as real
        # gridded model inputs do.
        ds = xr.concat([ds.assign_coords(latitude=[1.0]), ds], dim="latitude")
        canonical = prep._canonicalize_spatial_coordinates(ds, {"data": {}})

        np.testing.assert_array_equal(canonical.longitude, [0.0, 90.0, 180.0, 270.0])
        np.testing.assert_array_equal(canonical.v.isel(latitude=0), [0.0, 90.0, -180.0, -90.0])
        self.assertEqual(canonical.longitude.attrs["units"], "degrees_east")

    def test_mixed_source_conventions_align_without_dropping_longitudes(self) -> None:
        time = np.array(["2024-01-01"], dtype="datetime64[ns]")
        lat = [1.0, 0.0]
        surface_lon = np.array([-180.0, -90.0, 0.0, 90.0])
        atmos_lon = np.array([0.0, 90.0, 180.0, 270.0])
        surface = xr.Dataset(
            {
                "s": (
                    ("time", "latitude", "longitude"),
                    np.broadcast_to(surface_lon % 360.0, (1, 2, 4)).copy(),
                )
            },
            coords={"time": time, "latitude": lat, "longitude": surface_lon},
        )
        atmos = xr.Dataset(
            {
                "a": (
                    ("time", "level", "latitude", "longitude"),
                    np.broadcast_to(atmos_lon, (1, 1, 2, 4)).copy(),
                )
            },
            coords={
                "time": time,
                "level": [1000.0],
                "latitude": lat,
                "longitude": atmos_lon,
            },
        )
        config = {
            "data": {
                "predictor_variables": [
                    {"dataset_name": "s", "kind": "surf"},
                    {"dataset_name": "a", "kind": "atmos"},
                ],
                "target_variables": [],
                "atmos_levels": [1000.0],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            surface_path = Path(directory) / "surface.nc"
            atmos_path = Path(directory) / "atmos.nc"
            surface.to_netcdf(surface_path)
            atmos.to_netcdf(atmos_path)
            merged = prep._build_merged_dataset(
                config,
                [surface_path],
                [atmos_path],
                prep._extract_requested_variables(config),
            ).load()
        np.testing.assert_array_equal(merged.longitude, atmos_lon)
        np.testing.assert_array_equal(merged.s.isel(time=0, latitude=0), atmos_lon)
        np.testing.assert_array_equal(
            merged.a.isel(time=0, level=0, latitude=0), atmos_lon,
        )

    def test_multifile_concat_snaps_float_precision_without_nan_columns(self) -> None:
        lat = np.array([1.0, 0.0])
        lon64 = np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64)
        lon32 = lon64.astype(np.float32)

        def dataset(time: str, lon: np.ndarray) -> xr.Dataset:
            return xr.Dataset(
                {"s": (("time", "latitude", "longitude"), np.ones((1, 2, 4)))},
                coords={
                    "time": np.array([time], dtype="datetime64[ns]"),
                    "latitude": lat,
                    "longitude": lon,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.nc"
            second = Path(directory) / "second.nc"
            dataset("2024-01-01", lon64).to_netcdf(first)
            dataset("2024-01-02", lon32).to_netcdf(second)
            merged = prep._load_and_merge_files(
                [first, second], {"data": {}}, label="surface", var_names=["s"],
            ).load()
        self.assertEqual(merged.sizes["longitude"], 4)
        self.assertFalse(bool(merged.s.isnull().any()))

    def test_cyclic_features_are_convention_invariant(self) -> None:
        a = lon_cyclic_features(torch.tensor([0.0, 90.0, 180.0]), height=2)
        b = lon_cyclic_features(torch.tensor([360.0, 450.0, -180.0]), height=2)
        torch.testing.assert_close(a, b)

    def test_invalid_periodicity_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lon_periodic"):
            resolve_lon_periodic({"model": {"lon_periodic": "globla"}}, [0, 90, 180, 270])


class PeriodicOperatorTest(unittest.TestCase):
    def test_periodic_convolution_reads_across_dateline(self) -> None:
        x = torch.tensor([[[[1.0, 2.0, 3.0, 7.0], [1.0, 2.0, 3.0, 7.0]]]])
        periodic = PeriodicConv2d(1, 1, 3, bias=False, lon_periodic=True)
        regional = PeriodicConv2d(1, 1, 3, bias=False, lon_periodic=False)
        with torch.no_grad():
            periodic.weight.zero_()
            periodic.weight[0, 0, 1, 0] = 1.0  # left longitude neighbour
            regional.weight.copy_(periodic.weight)

        self.assertEqual(float(periodic(x)[0, 0, 0, 0].detach()), 7.0)
        self.assertEqual(float(regional(x)[0, 0, 0, 0].detach()), 1.0)

    def test_nonperiodic_convolution_matches_legacy_replicate_padding(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(2, 3, 5, 8)
        legacy = nn.Conv2d(3, 4, 3, padding=1, padding_mode="replicate")
        replacement = PeriodicConv2d(3, 4, 3, lon_periodic=False)
        replacement.load_state_dict(legacy.state_dict())
        torch.testing.assert_close(replacement(x), legacy(x), rtol=0.0, atol=0.0)

    def test_periodic_resampling_is_roll_equivariant(self) -> None:
        torch.manual_seed(2)
        x = torch.randn(1, 2, 4, 8)
        up = periodic_bilinear_interpolate(x, (8, 16), lon_periodic=True)
        up_rolled = periodic_bilinear_interpolate(
            torch.roll(x, 1, -1), (8, 16), lon_periodic=True,
        )
        torch.testing.assert_close(up_rolled, torch.roll(up, 2, -1))

        down = periodic_avg_pool2d(x, lon_periodic=True)
        down_rolled = periodic_avg_pool2d(torch.roll(x, 2, -1), lon_periodic=True)
        torch.testing.assert_close(down_rolled, torch.roll(down, 1, -1))

    def test_spatial_gradient_loss_includes_wrap_edge(self) -> None:
        try:
            from finetune.flow_refine import _spatial_gradient_loss
        except RuntimeError as exc:  # pragma: no cover - broken optional torchvision builds
            self.skipTest(str(exc))

        refined = torch.tensor([[[[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]]])
        target = torch.zeros_like(refined)
        regional = _spatial_gradient_loss(refined, target, lon_periodic=False)
        global_loss = _spatial_gradient_loss(refined, target, lon_periodic=True)
        self.assertGreater(float(global_loss), float(regional))

    def test_flow_unet_is_equivariant_to_patch_aligned_longitude_roll(self) -> None:
        try:
            from finetune.flow_refine import ResidualFlowUNet
        except RuntimeError as exc:  # pragma: no cover - broken optional torchvision builds
            self.skipTest(str(exc))

        torch.manual_seed(3)
        model = ResidualFlowUNet(
            hidden=8, time_dim=16, lon_periodic=True, lon_encoding=False,
        ).eval()
        # The output is zero-initialised; randomise only this projection so the
        # test exercises all periodic encoder/decoder paths non-trivially.
        with torch.no_grad():
            model.out.weight.normal_()
            model.out.bias.normal_()
        x = torch.randn(1, 1, 8, 16)
        cond = torch.randn_like(x)
        t = torch.tensor([0.4])
        expected = torch.roll(model(x, t, cond), shifts=4, dims=-1)
        actual = model(
            torch.roll(x, shifts=4, dims=-1),
            t,
            torch.roll(cond, shifts=4, dims=-1),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class PipelineConsistencyTest(unittest.TestCase):
    def _imports(self):
        try:
            from aurora.batch import Batch, Metadata
            from finetune import aurora_finetune_utils as ft
        except RuntimeError as exc:  # pragma: no cover - broken optional torchvision builds
            self.skipTest(str(exc))
        return Batch, Metadata, ft

    def test_global_patch_alignment_never_crops_longitude(self) -> None:
        _, _, ft = self._imports()
        surf = {"x": torch.zeros(1, 1, 4, 10)}
        static = {"x": torch.zeros(4, 10)}
        atmos = {"x": torch.zeros(1, 1, 1, 4, 10)}
        with self.assertRaisesRegex(ValueError, "Cropping longitude"):
            ft._align_spatial_dims_for_patch(
                surf,
                static,
                atmos,
                lat=torch.arange(4.0),
                lon=torch.arange(10.0),
                patch_size=4,
                strategy="crop",
                lon_periodic=True,
            )

    def test_checkpoint_padding_mismatch_is_rejected(self) -> None:
        _, _, ft = self._imports()

        class Model(nn.Module):
            lon_periodic = True
            lon_encoding = False

        matching = {
            "config": {
                "data": {"domain_type": "global"},
                "model": {"lon_periodic_resolved": True},
            }
        }
        ft.validate_checkpoint_longitude(Model(), matching)
        mismatched = {
            "config": {
                "data": {"domain_type": "regional"},
                "model": {"lon_periodic_resolved": False},
            }
        }
        with self.assertRaisesRegex(ValueError, "Checkpoint longitude mismatch"):
            ft.validate_checkpoint_longitude(Model(), mismatched)

    def test_legacy_checkpoint_is_only_allowed_for_unchanged_regional_path(self) -> None:
        _, _, ft = self._imports()

        class GlobalModel(nn.Module):
            lon_periodic = True
            lon_encoding = False

        class RegionalModel(nn.Module):
            lon_periodic = False
            lon_encoding = False

        legacy = {"config": {"data": {"domain_type": "regional"}, "model": {}}}
        ft.validate_checkpoint_longitude(RegionalModel(), legacy)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            ft.validate_checkpoint_longitude(GlobalModel(), legacy)

    def test_prediction_writer_reorders_and_deduplicates_longitude(self) -> None:
        Batch, Metadata, ft = self._imports()
        metadata = Metadata(
            lat=torch.tensor([1.0, 0.0], dtype=torch.float64),
            lon=torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float64),
            time=(datetime(2024, 1, 1),),
            atmos_levels=(),
        )
        # Emulate a third-party prediction carrying a redundant endpoint. The
        # public Metadata constructor correctly disallows this; the writer is
        # still defensive at its external file boundary.
        metadata.lon = torch.tensor(
            [-180.0, -90.0, 0.0, 90.0, 180.0], dtype=torch.float64,
        )
        values = torch.tensor(
            [[[[180.0, 270.0, 0.0, 90.0, 180.0], [180.0, 270.0, 0.0, 90.0, 180.0]]]]
        )
        prediction = Batch(
            surf_vars={"x": values},
            static_vars={},
            atmos_vars={},
            metadata=metadata,
        )
        output = ft.save_predictions(
            [prediction], "unused.nc", save_netcdf=False, lon_periodic=True,
        )
        np.testing.assert_array_equal(output.longitude, [0.0, 90.0, 180.0, 270.0])
        np.testing.assert_array_equal(output.x.isel(time=0, latitude=0), [0.0, 90.0, 180.0, 270.0])
        self.assertEqual(output.longitude.attrs["standard_name"], "longitude")
        self.assertEqual(output.attrs["Conventions"], "CF-1.10")
        self.assertIsNone(output.longitude.encoding["_FillValue"])

    def test_output_smoothing_wraps_longitude_only_for_global_grid(self) -> None:
        _, _, ft = self._imports()
        field = np.zeros((5, 16), dtype=np.float32)
        field[2, 0] = 1.0
        global_smoothed = ft._smooth_patch_artifacts(
            field, sigma=1.0, patch_size=3, lon_periodic=True,
        )
        regional_smoothed = ft._smooth_patch_artifacts(
            field, sigma=1.0, patch_size=3, lon_periodic=False,
        )
        self.assertGreater(float(global_smoothed[2, -1]), 0.0)
        self.assertEqual(float(regional_smoothed[2, -1]), 0.0)


class LongitudeDiagnosticTest(unittest.TestCase):
    def test_diagnostic_distinguishes_smooth_wrap_from_injected_seam(self) -> None:
        lon = np.arange(360.0)
        smooth = np.sin(np.deg2rad(lon))[None, :]
        smooth_metric = dateline_discontinuity(smooth)
        self.assertAlmostEqual(smooth_metric["local_ratio"], 1.0, places=2)

        seam = smooth.copy()
        seam[:, 0] += 2.0
        seam_metric = dateline_discontinuity(seam)
        self.assertGreater(seam_metric["local_ratio"], 1.8)

    def test_plotting_column_is_only_added_for_global_grid(self) -> None:
        global_lon = np.arange(0.0, 360.0, 90.0)
        field = np.arange(4.0)[None, :]
        lon_plot, field_plot = add_cyclic_column(global_lon, field)
        np.testing.assert_array_equal(lon_plot, [0.0, 90.0, 180.0, 270.0, 360.0])
        np.testing.assert_array_equal(field_plot[..., -1], field[..., 0])

        regional_lon = np.array([250.0, 251.0, 252.0, 253.0])
        lon_same, field_same = add_cyclic_column(regional_lon, field)
        np.testing.assert_array_equal(lon_same, regional_lon)
        np.testing.assert_array_equal(field_same, field)

    def test_prepared_netcdf_has_unique_cf_longitude(self) -> None:
        ds = xr.Dataset(
            {"v": (("time", "latitude", "longitude"), np.zeros((1, 2, 5)))},
            coords={
                "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                "latitude": [1.0, 0.0],
                "longitude": [0.0, 90.0, 180.0, 270.0, 360.0],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.nc"
            prep._write_netcdf(ds, path, compression_level=0)
            with xr.open_dataset(path) as written:
                np.testing.assert_array_equal(
                    written.longitude, [0.0, 90.0, 180.0, 270.0],
                )
                self.assertEqual(written.longitude.attrs["standard_name"], "longitude")
                self.assertEqual(written.longitude.attrs["units"], "degrees_east")
                self.assertEqual(written.attrs["Conventions"], "CF-1.10")
            import netCDF4

            with netCDF4.Dataset(path) as raw:
                for coord_name in ("time", "latitude", "longitude"):
                    self.assertNotIn("_FillValue", raw.variables[coord_name].ncattrs())

    def test_streaming_append_rejects_changed_longitude_grid(self) -> None:
        def dataset(lon: list[float]) -> xr.Dataset:
            return xr.Dataset(
                {"v": (("time", "latitude", "longitude"), np.zeros((1, 2, 2)))},
                coords={
                    "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                    "latitude": [1.0, 0.0],
                    "longitude": lon,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.nc"
            prep._append_netcdf(
                dataset([0.0, 1.0]), path, time_dim="time", compression_level=0,
            )
            with self.assertRaisesRegex(ValueError, "coordinate `longitude` changed"):
                prep._append_netcdf(
                    dataset([0.0, 2.0]), path, time_dim="time", compression_level=0,
                )

    def test_streaming_append_rejects_overlapping_time(self) -> None:
        def dataset(time: str) -> xr.Dataset:
            return xr.Dataset(
                {"v": (("time", "latitude", "longitude"), np.zeros((1, 2, 2)))},
                coords={
                    "time": np.array([time], dtype="datetime64[ns]"),
                    "latitude": [1.0, 0.0],
                    "longitude": [0.0, 1.0],
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.nc"
            prep._append_netcdf(
                dataset("2024-01-02"), path, time_dim="time", compression_level=0,
            )
            with self.assertRaisesRegex(ValueError, "not later than"):
                prep._append_netcdf(
                    dataset("2024-01-02"), path,
                    time_dim="time", compression_level=0,
                )


if __name__ == "__main__":
    unittest.main()
