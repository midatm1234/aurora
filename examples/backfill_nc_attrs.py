"""
Backfill variable/coordinate attributes and add a 1-D `time` coordinate
to existing rollout netCDF files in outputs/ecmwf_hourly_rollouts/.

Opens each file in r+ (append) mode — NO data copying, only attribute writes
and a single new variable, so it completes in seconds per file.

Usage:
    python examples/backfill_nc_attrs.py
"""

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs" / "ecmwf_hourly_rollouts"

VAR_ATTRS = {
    "2t":   {"long_name": "2 metre temperature",                   "units": "K",          "standard_name": "unknown",                                             "GRIB_paramId": 167, "GRIB_shortName": "2t",   "GRIB_name": "2 metre temperature",                         "GRIB_units": "K"},
    "10u":  {"long_name": "10 metre U wind component",             "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 165, "GRIB_shortName": "10u",  "GRIB_name": "10 metre U wind component",                   "GRIB_units": "m s**-1"},
    "10v":  {"long_name": "10 metre V wind component",             "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 166, "GRIB_shortName": "10v",  "GRIB_name": "10 metre V wind component",                   "GRIB_units": "m s**-1"},
    "msl":  {"long_name": "Mean sea level pressure",               "units": "Pa",         "standard_name": "air_pressure_at_mean_sea_level",                      "GRIB_paramId": 151, "GRIB_shortName": "msl",  "GRIB_name": "Mean sea level pressure",                     "GRIB_units": "Pa"},
    "2d":   {"long_name": "2 metre dewpoint temperature",          "units": "K",          "standard_name": "unknown",                                             "GRIB_paramId": 168, "GRIB_shortName": "2d",   "GRIB_name": "2 metre dewpoint temperature",                "GRIB_units": "K"},
    "tcwv": {"long_name": "Total column vertically-integrated water vapour", "units": "kg m**-2", "standard_name": "lwe_thickness_of_atmosphere_mass_content_of_water_vapor", "GRIB_paramId": 137, "GRIB_shortName": "tcwv", "GRIB_name": "Total column vertically-integrated water vapour", "GRIB_units": "kg m**-2"},
    "tcc":  {"long_name": "Total cloud cover",                     "units": "(0 - 1)",    "standard_name": "cloud_area_fraction",                                 "GRIB_paramId": 164, "GRIB_shortName": "tcc",  "GRIB_name": "Total cloud cover",                           "GRIB_units": "(0 - 1)"},
    "100u": {"long_name": "100 metre U wind component",            "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 228246, "GRIB_shortName": "100u", "GRIB_name": "100 metre U wind component",               "GRIB_units": "m s**-1"},
    "100v": {"long_name": "100 metre V wind component",            "units": "m s**-1",    "standard_name": "unknown",                                             "GRIB_paramId": 228247, "GRIB_shortName": "100v", "GRIB_name": "100 metre V wind component",               "GRIB_units": "m s**-1"},
    "sp":   {"long_name": "Surface pressure",                      "units": "Pa",         "standard_name": "surface_air_pressure",                                "GRIB_paramId": 134, "GRIB_shortName": "sp",   "GRIB_name": "Surface pressure",                            "GRIB_units": "Pa"},
    "lcc":  {"long_name": "Low cloud cover",                       "units": "(0 - 1)",    "standard_name": "unknown",                                             "GRIB_paramId": 186, "GRIB_shortName": "lcc",  "GRIB_name": "Low cloud cover",                             "GRIB_units": "(0 - 1)"},
    "mcc":  {"long_name": "Medium cloud cover",                    "units": "(0 - 1)",    "standard_name": "unknown",                                             "GRIB_paramId": 187, "GRIB_shortName": "mcc",  "GRIB_name": "Medium cloud cover",                          "GRIB_units": "(0 - 1)"},
    "hcc":  {"long_name": "High cloud cover",                      "units": "(0 - 1)",    "standard_name": "unknown",                                             "GRIB_paramId": 188, "GRIB_shortName": "hcc",  "GRIB_name": "High cloud cover",                            "GRIB_units": "(0 - 1)"},
    "skt":  {"long_name": "Skin temperature",                      "units": "K",          "standard_name": "unknown",                                             "GRIB_paramId": 235, "GRIB_shortName": "skt",  "GRIB_name": "Skin temperature",                            "GRIB_units": "K"},
    "stl1": {"long_name": "Soil temperature level 1",              "units": "K",          "standard_name": "surface_temperature",                                 "GRIB_paramId": 139, "GRIB_shortName": "stl1", "GRIB_name": "Soil temperature level 1",                    "GRIB_units": "K"},
    "swvl1":{"long_name": "Volumetric soil water layer 1",         "units": "m**3 m**-3", "standard_name": "unknown",                                             "GRIB_paramId": 39,  "GRIB_shortName": "swvl1","GRIB_name": "Volumetric soil water layer 1",                "GRIB_units": "m**3 m**-3"},
    "ci":   {"long_name": "Sea ice area fraction",                 "units": "(0 - 1)",    "standard_name": "sea_ice_area_fraction",                               "GRIB_paramId": 31,  "GRIB_shortName": "ci",   "GRIB_name": "Sea ice area fraction",                       "GRIB_units": "(0 - 1)"},
    "scaled_sd":    {"long_name": "Snow depth (log-scaled)",               "units": "1",       "standard_name": "lwe_thickness_of_surface_snow_amount",    "GRIB_paramId": 141, "GRIB_shortName": "sd",   "GRIB_name": "Snow depth",                   "GRIB_units": "m of water equivalent",    "comment": "Log-transformed from ERA5 snow depth (sd); apply exp() to recover physical values in m of water equivalent"},
    "i10fg":        {"long_name": "Instantaneous 10 metre wind gust",      "units": "m s**-1", "standard_name": "wind_speed_of_gust",                      "GRIB_paramId": 49,  "GRIB_shortName": "i10fg","GRIB_name": "Instantaneous 10 metre wind gust", "GRIB_units": "m s**-1"},
    "blh":          {"long_name": "Boundary layer height",                 "units": "m",       "standard_name": "atmosphere_boundary_layer_thickness",     "GRIB_paramId": 159, "GRIB_shortName": "blh",  "GRIB_name": "Boundary layer height",         "GRIB_units": "m"},
    "uvb_1h":       {"long_name": "1-hour UV-B downward solar radiation flux","units": "J m**-2","standard_name": "surface_downwelling_ultraviolet_b_flux_in_air","GRIB_paramId": 57, "GRIB_shortName": "uvb",  "GRIB_name": "Downward UV radiation at the surface","GRIB_units": "J m**-2"},
    "ssrd_1h":      {"long_name": "1-hour surface solar radiation downwards","units": "J m**-2","standard_name": "surface_downwelling_shortwave_flux_in_air",  "GRIB_paramId": 169,"GRIB_shortName": "ssrd", "GRIB_name": "Surface solar radiation downwards",    "GRIB_units": "J m**-2"},
    "ttr_1h":       {"long_name": "1-hour top net thermal radiation",      "units": "J m**-2", "standard_name": "toa_outgoing_longwave_flux",               "GRIB_paramId": 179, "GRIB_shortName": "ttr",  "GRIB_name": "Top net thermal radiation",     "GRIB_units": "J m**-2"},
    "scaled_tp_1h": {"long_name": "1-hour total precipitation (log-scaled)","units": "1",      "standard_name": "lwe_thickness_of_precipitation_amount",   "GRIB_paramId": 228, "GRIB_shortName": "tp",   "GRIB_name": "Total precipitation",           "GRIB_units": "m",                        "comment": "Log-transformed; apply exp() to recover physical values in metres"},
    "scaled_sf_1h": {"long_name": "1-hour snowfall (log-scaled)",          "units": "1",       "standard_name": "lwe_thickness_of_snowfall_amount",        "GRIB_paramId": 144, "GRIB_shortName": "sf",   "GRIB_name": "Snowfall",                      "GRIB_units": "m of water equivalent",    "comment": "Log-transformed; apply exp() to recover physical values in m of water equivalent"},
    "insolation":   {"long_name": "Top-of-atmosphere solar insolation",    "units": "W m**-2", "standard_name": "toa_incoming_shortwave_flux"},
    "z": {"long_name": "Geopotential",       "units": "m**2 s**-2", "standard_name": "geopotential",     "GRIB_paramId": 129, "GRIB_shortName": "z", "GRIB_name": "Geopotential",       "GRIB_units": "m**2 s**-2"},
    "u": {"long_name": "U component of wind","units": "m s**-1",    "standard_name": "eastward_wind",    "GRIB_paramId": 131, "GRIB_shortName": "u", "GRIB_name": "U component of wind", "GRIB_units": "m s**-1"},
    "v": {"long_name": "V component of wind","units": "m s**-1",    "standard_name": "northward_wind",   "GRIB_paramId": 132, "GRIB_shortName": "v", "GRIB_name": "V component of wind", "GRIB_units": "m s**-1"},
    "t": {"long_name": "Temperature",        "units": "K",          "standard_name": "air_temperature",  "GRIB_paramId": 130, "GRIB_shortName": "t", "GRIB_name": "Temperature",         "GRIB_units": "K"},
    "q": {"long_name": "Specific humidity",  "units": "kg kg**-1",  "standard_name": "specific_humidity","GRIB_paramId": 133, "GRIB_shortName": "q", "GRIB_name": "Specific humidity",   "GRIB_units": "kg kg**-1"},
}

COORD_ATTRS = {
    "ensemble":     {"long_name": "ensemble member index",   "units": "1",                     "standard_name": "realization"},
    "init_time":    {"units": "seconds since 1970-01-01",    "standard_name": "forecast_reference_time", "long_name": "initialization time"},
    "lead_minutes": {"units": "minutes",                     "long_name": "lead time since initialization"},
    "latitude":     {"units": "degrees_north",               "standard_name": "latitude",      "long_name": "latitude",  "stored_direction": "decreasing"},
    "longitude":    {"units": "degrees_east",                "standard_name": "longitude",     "long_name": "longitude"},
    "level":        {"units": "hPa",                         "positive": "down",               "standard_name": "air_pressure", "long_name": "pressure", "stored_direction": "decreasing"},
}


def backfill_file(path: Path):
    with nc.Dataset(str(path), "r+") as ds:
        # Global attributes
        ds.Conventions = "CF-1.10"
        ds.institution = "Microsoft Research"
        ds.source      = "Aurora 1.5"

        # Coordinate attributes
        for cname, attrs in COORD_ATTRS.items():
            if cname in ds.variables:
                for k, v in attrs.items():
                    setattr(ds.variables[cname], k, v)

        # Add 1-D time variable if absent
        if "time" not in ds.variables:
            init_secs = int(ds.variables["init_time"][0])
            lead_mins = ds.variables["lead_minutes"][:]
            time_vals = init_secs + lead_mins.astype("i8") * np.int64(60)
            tv = ds.createVariable("time", "i8", ("lead_minutes",))
            tv[:] = time_vals
            tv.units         = "seconds since 1970-01-01"
            tv.standard_name = "time"
            tv.long_name     = "valid time"
            tv.calendar      = "proleptic_gregorian"

        # Data variable attributes
        all_coords = set(COORD_ATTRS) | {"time"}
        for vname, var in ds.variables.items():
            if vname in all_coords:
                continue
            if vname in VAR_ATTRS:
                for k, v in VAR_ATTRS[vname].items():
                    setattr(var, k, v)

    print(f"  updated: {path.name}")


def main():
    nc_files = sorted(OUTPUT_DIR.glob("rollout_*.nc"))
    if not nc_files:
        print(f"No rollout_*.nc files found in {OUTPUT_DIR}")
        return
    print(f"Backfilling {len(nc_files)} files in {OUTPUT_DIR} ...")
    for p in nc_files:
        backfill_file(p)
    print("Done.")


if __name__ == "__main__":
    main()
