---
type: Dataset Contract
title: CAMS acquisition and time identity
description: Operational analysis, forecast-cycle reference, and reanalysis are distinct
  data products.
model_track: aurora_air_pollution
evidence_kind: code_verified
implementation_status: documented
execution_status: not_run
reporting_status: background
generated:
  by: codex/gpt-6
  at: '2026-09-22T19:30:40.538537+00:00'
sources:
- id: ads
  resource: https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview
  title: cams
---

# CAMS contract

The selected ADS collection is `cams-global-atmospheric-composition-forecasts`. The catalogue describes forecasts and analyses, archive access conditions, temporal coverage and model updates. Consult the live form/API request schema for availability; pin and save each effective request and response provenance.[^ads]

Analysis at a valid time may initialize or verify forecasts; an operational forecast is also identified by its initialization (`forecast_reference_time`) and lead. Preserve these plus `valid_time`, product identity, retrieval date and model cycle when supplied. Distinct cycles with the same valid time are distinct forecast records. Do not substitute EAC4 reanalysis silently.

Plan history before acquisition: two inputs require initialization and initialization minus 12 hours. For six 12-hour leads, verification reaches initialization plus 72 hours. Keep analysis and forecast requests explicit. Chunk requests, bound storage/retries, reuse checksum-verified cache objects, and retain unfinished-download state.

Fit learned scalers on training rows only. Split by complete forecast cycles and valid-time windows; purge overlapping lead windows at boundaries. A test interval previously used for validation remains contaminated for historical selection claims. See [historical experiments](../evidence/historical.md).

[^ads]: ADS operational product catalogue, response hash in provenance/references.json.
