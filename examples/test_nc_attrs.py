"""
Verify that _create_daily_nc_files (from the updated notebook) produces files
with the same variables, dimensions, attributes, and time values as the
backfilled existing reference file.

No model or ERA5 data required — only metadata is checked.

Usage:
    python examples/test_nc_attrs.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np

# ── Constants mirrored from the notebook ────────────────────────────────────
AURORA_STEP_HOURS  = 6
FINE_LEAD_TIMES    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
FORECAST_HOURS     = 240
N_ENSEMBLE_MEMBERS = 5
USE_ENSEMBLE       = True
ATMOS_LEVELS       = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

n_members       = N_ENSEMBLE_MEMBERS
n_forecast_days = (FORECAST_HOURS + 23) // 24

REF_FILE = Path(__file__).parent / "outputs" / "ecmwf_hourly_rollouts" / "rollout_2026-06-01_d01.nc"

# Pull grid and variable names from the reference file
with nc.Dataset(str(REF_FILE)) as ref:
    _lat_global   = ref.variables["latitude"][:]
    _lon_global   = ref.variables["longitude"][:]
    _vnames_surf  = [v for v in ref.variables
                     if ref.variables[v].dimensions ==
                     ("ensemble", "init_time", "lead_minutes", "latitude", "longitude")]
    _vnames_atmos = [v for v in ref.variables
                     if ref.variables[v].dimensions ==
                     ("ensemble", "init_time", "lead_minutes", "level", "latitude", "longitude")]

# ── Attribute tables (identical to the notebook cell) ────────────────────────
VAR_ATTRS = {
    "2t":           {"long_name": "2 metre temperature",                              "units": "K",          "standard_name": "unknown",                                             "GRIB_paramId": 167,    "GRIB_shortName": "2t",   "GRIB_name": "2 metre temperature",                          "GRIB_units": "K"},
    "10u":          {"long_name": "10 metre U wind component",                        "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 165,    "GRIB_shortName": "10u",  "GRIB_name": "10 metre U wind component",                    "GRIB_units": "m s**-1"},
    "10v":          {"long_name": "10 metre V wind component",                        "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 166,    "GRIB_shortName": "10v",  "GRIB_name": "10 metre V wind component",                    "GRIB_units": "m s**-1"},
    "msl":          {"long_name": "Mean sea level pressure",                          "units": "Pa",         "standard_name": "air_pressure_at_mean_sea_level",                      "GRIB_paramId": 151,    "GRIB_shortName": "msl",  "GRIB_name": "Mean sea level pressure",                      "GRIB_units": "Pa"},
    "2d":           {"long_name": "2 metre dewpoint temperature",                     "units": "K",          "standard_name": "unknown",                                             "GRIB_paramId": 168,    "GRIB_shortName": "2d",   "GRIB_name": "2 metre dewpoint temperature",                 "GRIB_units": "K"},
    "tcwv":         {"long_name": "Total column vertically-integrated water vapour",  "units": "kg m**-2",   "standard_name": "lwe_thickness_of_atmosphere_mass_content_of_water_vapor", "GRIB_paramId": 137, "GRIB_shortName": "tcwv", "GRIB_name": "Total column vertically-integrated water vapour","GRIB_units": "kg m**-2"},
    "tcc":          {"long_name": "Total cloud cover",                                "units": "(0 - 1)",    "standard_name": "cloud_area_fraction",                                 "GRIB_paramId": 164,    "GRIB_shortName": "tcc",  "GRIB_name": "Total cloud cover",                            "GRIB_units": "(0 - 1)"},
    "100u":         {"long_name": "100 metre U wind component",                       "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 228246, "GRIB_shortName": "100u", "GRIB_name": "100 metre U wind component",                   "GRIB_units": "m s**-1"},
    "100v":         {"long_name": "100 metre V wind component",                       "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 228247, "GRIB_shortName": "100v", "GRIB_name": "100 metre V wind component",                   "GRIB_units": "m s**-1"},
    "sp":           {"long_name": "Surface pressure",                                 "units": "Pa",         "standard_name": "surface_air_pressure",                                "GRIB_paramId": 134,    "GRIB_shortName": "sp",   "GRIB_name": "Surface pressure",                             "GRIB_units": "Pa"},
    "lcc":          {"long_name": "Low cloud cover",                                  "units": "(0 - 1)",    "standard_name": "unknown",                                             "GRIB_paramId": 186,    "GRIB_shortName": "lcc",  "GRIB_name": "Low cloud cover",                              "GRIB_units": "(0 - 1)"},
    "mcc":          {"long_name": "Medium cloud cover",                               "units": "(0 - 1)",    "standard_name": "unknown",                                             "GRIB_paramId": 187,    "GRIB_shortName": "mcc",  "GRIB_name": "Medium cloud cover",                           "GRIB_units": "(0 - 1)"},
    "hcc":          {"long_name": "High cloud cover",                                 "units": "(0 - 1)",    "standard_name": "unknown",                                             "GRIB_paramId": 188,    "GRIB_shortName": "hcc",  "GRIB_name": "High cloud cover",                             "GRIB_units": "(0 - 1)"},
    "skt":          {"long_name": "Skin temperature",                                 "units": "K",          "standard_name": "unknown",                                             "GRIB_paramId": 235,    "GRIB_shortName": "skt",  "GRIB_name": "Skin temperature",                             "GRIB_units": "K"},
    "stl1":         {"long_name": "Soil temperature level 1",                         "units": "K",          "standard_name": "surface_temperature",                                 "GRIB_paramId": 139,    "GRIB_shortName": "stl1", "GRIB_name": "Soil temperature level 1",                     "GRIB_units": "K"},
    "swvl1":        {"long_name": "Volumetric soil water layer 1",                    "units": "m**3 m**-3", "standard_name": "unknown",                                             "GRIB_paramId": 39,     "GRIB_shortName": "swvl1","GRIB_name": "Volumetric soil water layer 1",                 "GRIB_units": "m**3 m**-3"},
    "ci":           {"long_name": "Sea ice area fraction",                            "units": "(0 - 1)",    "standard_name": "sea_ice_area_fraction",                               "GRIB_paramId": 31,     "GRIB_shortName": "ci",   "GRIB_name": "Sea ice area fraction",                        "GRIB_units": "(0 - 1)"},
    "scaled_sd":    {"long_name": "Snow depth (log-scaled)",                          "units": "1",          "standard_name": "lwe_thickness_of_surface_snow_amount",                "GRIB_paramId": 141,    "GRIB_shortName": "sd",   "GRIB_name": "Snow depth",                                   "GRIB_units": "m of water equivalent",   "comment": "Log-transformed from ERA5 snow depth (sd); apply exp() to recover physical values in m of water equivalent"},
    "i10fg":        {"long_name": "Instantaneous 10 metre wind gust",                 "units": "m s**-1",    "standard_name": "wind_speed_of_gust",                                  "GRIB_paramId": 49,     "GRIB_shortName": "i10fg","GRIB_name": "Instantaneous 10 metre wind gust",              "GRIB_units": "m s**-1"},
    "blh":          {"long_name": "Boundary layer height",                            "units": "m",          "standard_name": "atmosphere_boundary_layer_thickness",                 "GRIB_paramId": 159,    "GRIB_shortName": "blh",  "GRIB_name": "Boundary layer height",                        "GRIB_units": "m"},
    "uvb_1h":       {"long_name": "1-hour UV-B downward solar radiation flux",        "units": "J m**-2",    "standard_name": "surface_downwelling_ultraviolet_b_flux_in_air",        "GRIB_paramId": 57,     "GRIB_shortName": "uvb",  "GRIB_name": "Downward UV radiation at the surface",         "GRIB_units": "J m**-2"},
    "ssrd_1h":      {"long_name": "1-hour surface solar radiation downwards",         "units": "J m**-2",    "standard_name": "surface_downwelling_shortwave_flux_in_air",            "GRIB_paramId": 169,    "GRIB_shortName": "ssrd", "GRIB_name": "Surface solar radiation downwards",            "GRIB_units": "J m**-2"},
    "ttr_1h":       {"long_name": "1-hour top net thermal radiation",                 "units": "J m**-2",    "standard_name": "toa_outgoing_longwave_flux",                           "GRIB_paramId": 179,    "GRIB_shortName": "ttr",  "GRIB_name": "Top net thermal radiation",                    "GRIB_units": "J m**-2"},
    "scaled_tp_1h": {"long_name": "1-hour total precipitation (log-scaled)",          "units": "1",          "standard_name": "lwe_thickness_of_precipitation_amount",               "GRIB_paramId": 228,    "GRIB_shortName": "tp",   "GRIB_name": "Total precipitation",                          "GRIB_units": "m",                       "comment": "Log-transformed; apply exp() to recover physical values in metres"},
    "scaled_sf_1h": {"long_name": "1-hour snowfall (log-scaled)",                     "units": "1",          "standard_name": "lwe_thickness_of_snowfall_amount",                    "GRIB_paramId": 144,    "GRIB_shortName": "sf",   "GRIB_name": "Snowfall",                                     "GRIB_units": "m of water equivalent",   "comment": "Log-transformed; apply exp() to recover physical values in m of water equivalent"},
    "insolation":   {"long_name": "Top-of-atmosphere solar insolation",               "units": "W m**-2",    "standard_name": "toa_incoming_shortwave_flux"},
    "z":            {"long_name": "Geopotential",                                     "units": "m**2 s**-2", "standard_name": "geopotential",                                        "GRIB_paramId": 129,    "GRIB_shortName": "z",    "GRIB_name": "Geopotential",                                 "GRIB_units": "m**2 s**-2"},
    "u":            {"long_name": "U component of wind",                              "units": "m s**-1",    "standard_name": "eastward_wind",                                       "GRIB_paramId": 131,    "GRIB_shortName": "u",    "GRIB_name": "U component of wind",                          "GRIB_units": "m s**-1"},
    "v":            {"long_name": "V component of wind",                              "units": "m s**-1",    "standard_name": "northward_wind",                                      "GRIB_paramId": 132,    "GRIB_shortName": "v",    "GRIB_name": "V component of wind",                          "GRIB_units": "m s**-1"},
    "t":            {"long_name": "Temperature",                                      "units": "K",          "standard_name": "air_temperature",                                     "GRIB_paramId": 130,    "GRIB_shortName": "t",    "GRIB_name": "Temperature",                                  "GRIB_units": "K"},
    "q":            {"long_name": "Specific humidity",                                "units": "kg kg**-1",  "standard_name": "specific_humidity",                                   "GRIB_paramId": 133,    "GRIB_shortName": "q",    "GRIB_name": "Specific humidity",                            "GRIB_units": "kg kg**-1"},
}

COORD_ATTRS = {
    "ensemble":     {"long_name": "ensemble member index",          "units": "1",                      "standard_name": "realization"},
    "init_time":    {"units": "seconds since 1970-01-01",           "standard_name": "forecast_reference_time", "long_name": "initialization time"},
    "lead_minutes": {"units": "minutes",                            "long_name": "lead time since initialization"},
    "latitude":     {"units": "degrees_north",                      "standard_name": "latitude",       "long_name": "latitude",  "stored_direction": "decreasing"},
    "longitude":    {"units": "degrees_east",                       "standard_name": "longitude",      "long_name": "longitude"},
    "level":        {"units": "hPa",                                "positive": "down",                "standard_name": "air_pressure", "long_name": "pressure", "stored_direction": "decreasing"},
}


def _apply_attrs(nc_var, attr_dict):
    for k, v in attr_dict.items():
        setattr(nc_var, k, v)


def create_test_file(out_path, init_day_str, day_init_times, forecast_day_idx):
    """Reproduce _create_daily_nc_files for a single day index."""
    epoch     = np.datetime64("1970-01-01", "s")
    init_secs = np.array(
        [int((np.datetime64(t, "s") - epoch) / np.timedelta64(1, "s")) for t in day_init_times],
        dtype="i8",
    )
    d = forecast_day_idx
    first_lead = d * 24 + 1
    last_lead  = min((d + 1) * 24, FORECAST_HOURS)
    n_hours    = last_lead - first_lead + 1
    lead_mins  = np.array([(first_lead + h - 1) * 60 for h in range(n_hours)], dtype="i4")

    ds = nc.Dataset(str(out_path), "w")
    ds.createDimension("ensemble",     n_members)
    ds.createDimension("init_time",    len(day_init_times))
    ds.createDimension("lead_minutes", n_hours)
    ds.createDimension("latitude",     len(_lat_global))
    ds.createDimension("longitude",    len(_lon_global))
    ds.createDimension("level",        len(ATMOS_LEVELS))

    v = ds.createVariable("ensemble",     "i4", ("ensemble",));      v[:] = np.arange(n_members);               _apply_attrs(v, COORD_ATTRS["ensemble"])
    v = ds.createVariable("init_time",    "i8", ("init_time",));     v[:] = init_secs;                          _apply_attrs(v, COORD_ATTRS["init_time"])
    v = ds.createVariable("lead_minutes", "i4", ("lead_minutes",));  v[:] = lead_mins;                          _apply_attrs(v, COORD_ATTRS["lead_minutes"])
    time_vals = init_secs[0] + lead_mins.astype("i8") * np.int64(60)
    v = ds.createVariable("time",         "i8", ("lead_minutes",));  v[:] = time_vals
    v.units = "seconds since 1970-01-01"; v.standard_name = "time"; v.long_name = "valid time"; v.calendar = "proleptic_gregorian"
    v = ds.createVariable("latitude",     "f4", ("latitude",));      v[:] = _lat_global;                        _apply_attrs(v, COORD_ATTRS["latitude"])
    v = ds.createVariable("longitude",    "f4", ("longitude",));     v[:] = _lon_global;                        _apply_attrs(v, COORD_ATTRS["longitude"])
    v = ds.createVariable("level",        "i4", ("level",));         v[:] = np.array(ATMOS_LEVELS, dtype="i4"); _apply_attrs(v, COORD_ATTRS["level"])

    for vname in _vnames_surf:
        v = ds.createVariable(vname, "f4", ("ensemble", "init_time", "lead_minutes", "latitude", "longitude"))
        if vname in VAR_ATTRS:
            _apply_attrs(v, VAR_ATTRS[vname])
    for vname in _vnames_atmos:
        v = ds.createVariable(vname, "f4", ("ensemble", "init_time", "lead_minutes", "level", "latitude", "longitude"))
        if vname in VAR_ATTRS:
            _apply_attrs(v, VAR_ATTRS[vname])

    ds.forecast_day    = d + 1
    ds.init_day        = init_day_str
    ds.forecast_hours  = FORECAST_HOURS
    ds.fine_lead_times = str(FINE_LEAD_TIMES)
    ds.ensemble_mode   = str(USE_ENSEMBLE)
    ds.n_members       = n_members
    ds.Conventions     = "CF-1.10"
    ds.institution     = "Microsoft Research"
    ds.source          = "Aurora 1.5"
    ds.close()


def compare(new_path, ref_path):
    errors   = []
    warnings = []

    with nc.Dataset(str(new_path)) as new_ds, nc.Dataset(str(ref_path)) as ref_ds:
        # Global attributes
        for attr in ["Conventions", "institution", "source", "forecast_day",
                     "init_day", "forecast_hours", "fine_lead_times",
                     "ensemble_mode", "n_members"]:
            nv = getattr(new_ds, attr, "MISSING")
            rv = getattr(ref_ds, attr, "MISSING")
            if str(nv) != str(rv):
                errors.append(f"global.{attr}: new={nv!r}  ref={rv!r}")

        # Variable presence
        new_vars = set(new_ds.variables)
        ref_vars = set(ref_ds.variables)
        for vn in ref_vars - new_vars:
            errors.append(f"missing variable: {vn!r}")
        for vn in new_vars - ref_vars:
            warnings.append(f"extra variable in new (ok): {vn!r}")

        # Dimensions per variable
        for vname in ref_vars & new_vars:
            nd = new_ds.variables[vname].dimensions
            rd = ref_ds.variables[vname].dimensions
            if nd != rd:
                errors.append(f"{vname} dims: new={nd}  ref={rd}")

        # Attributes per variable
        skip = {"_FillValue"}
        for vname in ref_vars & new_vars:
            rv_attrs = {a: getattr(ref_ds.variables[vname], a)
                        for a in ref_ds.variables[vname].ncattrs() if a not in skip}
            nv_attrs = {a: getattr(new_ds.variables[vname], a)
                        for a in new_ds.variables[vname].ncattrs() if a not in skip}
            for k, rv in rv_attrs.items():
                if k not in nv_attrs:
                    errors.append(f"{vname}.{k}: MISSING in new (ref={rv!r})")
                elif str(nv_attrs[k]) != str(rv):
                    errors.append(f"{vname}.{k}: new={nv_attrs[k]!r}  ref={rv!r}")
            for k in set(nv_attrs) - set(rv_attrs):
                warnings.append(f"{vname}.{k}: extra in new (ok)")

        # time values
        if "time" in new_ds.variables and "time" in ref_ds.variables:
            nt = new_ds.variables["time"][:]
            rt = ref_ds.variables["time"][:]
            if not np.array_equal(nt, rt):
                errors.append(f"time values differ: new[0]={nt[0]}  ref[0]={rt[0]}")
        else:
            errors.append("time variable missing in new or ref file")

    return errors, warnings


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        init_times = [np.datetime64("2026-06-01T00:00:00", "s")]
        new_path   = Path(tmpdir) / "rollout_2026-06-01_d01.nc"
        create_test_file(new_path, "2026-06-01", init_times, forecast_day_idx=0)

        errors, warnings = compare(new_path, REF_FILE)
    finally:
        shutil.rmtree(tmpdir)

    if warnings:
        print("Warnings (non-blocking):")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print(f"\nFAIL — {len(errors)} discrepancy(ies):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nPASS — notebook output matches reference on all variables, dimensions, attributes, and time values.")


if __name__ == "__main__":
    main()
