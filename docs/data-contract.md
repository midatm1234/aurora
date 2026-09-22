# CAMS acquisition and preparation contract

The execution backend is `aurora_workflow.data`. `plan_cams(spec, limits)` is
read-only; `execute_data("acquire" | "prepare", plan, run_dir)` is called by the
approved durable worker. Set local roots in the workflow configuration. No
credentials, data paths from the original author's machine, or downloaded data
belong in Git.

## Source and interface verification

Inspected fork commit: `88f652f04aa75b65e410b8d00cf4ea07f2edd945`.
The existing downloader is `examples/cams_download_2025_range.py`; its actual
CLI has `--start-date`, `--end-date`, `--output-dir`, and `--overwrite`. It
requests lead-zero operational forecasts, extracts surface and atmospheric
files, and checks time bounds. Its default start in 2014 precedes the current
operational product archive. Its `format` field is historical: this backend
uses the currently advertised `data_format` field.

Official references accessed 2026-09-22:

- [ADS product](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts).
- [ADS API setup](https://ads.atmosphere.copernicus.eu/how-to-api).
- [Machine-readable product catalogue](https://ads.atmosphere.copernicus.eu/api/catalogue/v1/collections/cams-global-atmospheric-composition-forecasts).
- [Exact request form snapshot](https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/cams-global-atmospheric-composition-forecasts/form_ff578604ee4b3615add68329e0630fe5c463621ce4904fef6ba5fe0d533694e5.json),
  SHA256 `ff578604ee4b3615add68329e0630fe5c463621ce4904fef6ba5fe0d533694e5`.
- [Exact constraint snapshot](https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/cams-global-atmospheric-composition-forecasts/constraints_aa53d12f3e084eed42044864f2e047cbd76e0ce8f21932b217cb00ca458ea267.json).

This is a live operational archive with changing model cycles, not an immutable
reanalysis. The form lists dates from 2015-01-01, 00/12 UTC initialization,
`type: forecast`, `leadtime_hour`, `variable`, `pressure_level`, `date`, `time`,
and `data_format: netcdf_zip` (experimental conversion) or `grib`. The wrapper
uses NetCDF ZIP only. Service constraints and availability can change after the
recorded snapshot; rejection is an execution failure, never permission to use
another product. Very recent meteorological variables may be unavailable and
older requests may be tape-backed. Real acquisition was not tested without
local authorization, credentials and accepted terms.

The official setup specifies `cdsapi>=0.7.7`, ADS URL
`https://ads.atmosphere.copernicus.eu/api`, a personal access token in local
`.cdsapirc` or approved environment, and manually accepting dataset terms.
Do that outside chat. The backend fixes the service URL, allows only ADS/ECMWF
result hosts, suppresses third-party exception text that could expose signed
URLs, and never serializes credentials. It uses the client's supported result
location and size interfaces, and streams result bytes with its own cap.

## Request planning and bounds

For every selected forecast cycle, history is exactly `[cycle-12h, cycle]`.
The planner expands the first requested date backwards to include the earlier
history time. Date-only end bounds include both 00 and 12 UTC cycles.
Initialization lead zero is requested separately from reference leads
12/24/36/48/60/72. Explicit sparse case lists request just their required history
snapshots. References can use selected target variables/levels and the refinement
domain; initialization always retains all global backbone fields. Each request
uses chunks of 1–15 days, a content-derived cache key and an explicit request
count. Storage estimates count full global 451×900 float32 fields, 12 surface
fields and 10×13 pressure-level fields for history (or the selected targets), plus 25%
overhead. ZIP compression and actual service costs remain unknown. The planner
reports estimates; the approved byte, disk, wall-clock and request bounds are
enforced by backend/worker checks.

Verified ZIP chunks are reused by checksum. Interrupted jobs reuse completed
chunks; a partial current download restarts within the persistent download
budget and bounded retry count. Each extracted file has a hash. Extraction
rejects nested/absolute/traversal paths, symlinks, non-NetCDF files and archives
over the extraction cap. Failed or incomplete requests are not reported as
successful data preparation. Byte resumption within an incomplete HTTP response
is not supported; resumption is at verified chunk boundaries.

## Full backbone versus refinement fields

The complete `AuroraAirPollution` input contract at the pinned source is:

| Kind | CAMS/NetCDF names | Aurora names |
|---|---|---|
| Meteorological surface | t2m, u10, v10, msl | 2t, 10u, 10v, msl |
| Pollutant surface/column | pm1, pm2p5, pm10, tcco, tc_no, tcno2, gtco3, tcso2 | same |
| Atmospheric | z, u, v, t, q, co, no, no2, go3, so2 | same |
| Static | lsm, z_static, slt, static_ammonia, static_ammonia_log, static_co, static_co_log, static_nox, static_nox_log, static_so2, static_so2_log | z_static maps to z; others unchanged |

Pressure grid in hPa: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
1000. Static data come from the verified Microsoft air-pollution asset, not from
an inferred pressure-level geopotential slice. Acquisition is global; regional
refinement is applied to the deterministic backbone output. The NO2 YAML's
smaller predictor mapping and refined target NO2 at 1000/925/850 hPa plus tcNO2
do not reduce these backbone requirements. The adapter checks full input fields
before rollout and aligns static coordinates in the science stage.

Accepted explicit aliases include `tcno→tc_no`, `tco3→gtco3`, `o3→go3`.
Surface wind is m s-1, temperature K, pressure Pa; pollutant mixing ratios are
kg kg-1, particulate matter kg m-3 and total columns kg m-2. Geopotential is
m2 s-2. No implicit conversion to ppb, geometric height or ground-level
concentration is performed. A 1000-hPa field is not a universal ground-level
measurement, and a partial pressure integral is not an exact total column.

## Preparation, time provenance and leakage

The original `finetune/prepare_train_test_from_netcdf.py` has reusable spatial
canonicalization and merging helpers. Its `_collapse_forecast_dims` stacks
cycle and lead onto valid time and drops explicit valid-time metadata. That
historical timeseries contract is retained in the existing script, but cannot
represent distinct forecasts that share valid time. The wrapper imports the
original spatial canonicalizer and preserves `forecast_reference_time`,
`lead_time` in hours, and `valid_time = cycle + lead` explicitly for references.
It does not call the lossy forecast-collapse helper on reference forecasts.

Preparation rejects missing cycle/lead provenance, inconsistent valid time,
duplicate cycle/lead keys, unsupported dimensions, incomplete backbone fields,
wrong pressure levels/units and irregular coordinates. The original spatial
canonicalizer reorders complete fields with coordinates, stores descending
latitude and increasing `[0,360)` longitude, and handles duplicate cyclic
endpoints through the pinned longitude helper, after rejecting unequal duplicate
endpoint field values. Initialization missing values
are rejected, requiring a separately declared imputation experiment. Reference
missing values remain available for exclusion and coverage reporting.

`reference_kind=forecast` selects each reference from the specified initialization
and requested forecast lead. `reference_kind=analysis` explicitly selects
lead-zero initial-state fields at each future valid time; it is operational
analysis/initial-state verification, not an alternative API `type` value.
Reanalysis is rejected. Input provenance records raw-file hashes, selected
source metadata and model-cycle labels when supplied; missing labels are
recorded as unavailable. A historical NetCDF without cycle/lead provenance must
not be silently relabeled as an operational forecast reference.

`prepared.json` schema version 1 contains:

```json
{
  "schema_version": 1,
  "data_path": "/recipient/run/prepared-data.nc",
  "reference_path": "/recipient/run/prepared-reference.nc",
  "cases": [{"cycle": "2024-07-01T12:00:00",
             "history_times": ["2024-07-01T00:00:00", "2024-07-01T12:00:00"],
             "lead_hours": [12,24,36,48,60,72], "split": "test"}],
  "excluded_cases": [],
  "source": {"product": "cams-global-atmospheric-composition-forecasts",
             "reference_kind": "forecast", "inputs": [], "model_cycles": []},
  "sha256": {}, "normalization": {"enabled": false, "mode": "none"}
}
```

`data_path` contains only lead-zero initialization fields on a `time` dimension;
the science stage reads exactly each case's two history times. Targets are read
separately from `reference_path`, whose dimensions are
`forecast_reference_time,lead_time,[level],latitude,longitude`. Preparation never
supplies future targets to autonomous inference. Forecast-forced historical
behavior is a different scientific contract and cannot be inferred from the
presence of future fields in a file.

Splits are explicit nonoverlapping inclusive valid-time windows. A case is
kept only if its entire `[cycle-12h, cycle+max_lead]` interval fits one window;
boundary cases are recorded as purged. This prevents shared forecast support
between train/validation/test. No historical validation interval is invented. Historical
normalization is disabled; the optional statistics helper fits only entries
marked train and records that split. `resolve_data_spec` makes dates and splits
explicit in the approved plan. Reproduce mode uses the recorded train/test
periods and derives the configured chronological training-tail validation
window; this is labeled a new deterministic derivation, since historical exact
validation dates are unavailable. Replay defaults to the historical test period.
The real integration recipe instead supplies exactly three disjoint cases.

With Dask from the scientific environment, preparation uses lazy arrays,
one-cycle/lead chunks and synchronous writes; history finite checks read one
variable/time at a time. Without Dask, small fixtures use an eager fallback that
fails above the approved memory estimate. Multi-year streaming is implemented
but was not validated with production CAMS data here; wall-clock, disk and
source-data limits still apply.

## Available validation

`tests/workflow/test_data.py` runs synthetic NetCDF fixtures covering request
history, wrong products/levels/units, same-valid-time separate forecasts,
split purging, future-input rejection, training-only statistics, ZIP traversal
and size limits, preparation output hashes and missing initialization fields.
These tests do not download CAMS or prove numerical reproduction of Aurora.
