"""Adapters from FastF1 laps to the frame the simulator fits on.

The simulator itself never imports FastF1. Only `load_session` does, lazily.
Frame columns: driver, lap, stint, compound, tyre_age, lap_time_s, track_status,
pit_in, pit_out, start_s, end_s, team, field_ref_s, pace_resid_s, clean.
"""
import os

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
K_SLOW_S = 1.0         # a lap is an outlier if it is this much slower than the driver's usual deficit
MIN_FIELD_LAPS = 5     # laps needed on a lap number before the field reference is trusted


WEATHER_COLUMNS = ["time_s", "air_temp_c", "track_temp_c", "humidity_pct", "pressure_hpa",
                   "rainfall", "wind_speed_ms", "wind_dir_deg"]


def load_session(season, gp, kind="R", cache_dir=CACHE_DIR, weather=False, messages=False):
    import fastf1
    fastf1.set_log_level("ERROR")
    fastf1.Cache.configure(cache_dir=cache_dir)
    fastf1.Cache.offline_mode(True)
    session = fastf1.get_session(season, gp, kind)
    session.load(telemetry=False, weather=weather, messages=messages)
    return session


def session_weather(session):
    """Weather samples (about one per minute) in simulator units, empty if not loaded."""
    from fastf1.exceptions import DataNotLoadedError
    try:
        w = session.weather_data
    except DataNotLoadedError:  # weather=False, or absent from the cache
        w = None
    if w is None or len(w) == 0:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    return pd.DataFrame({
        "time_s": w["Time"].dt.total_seconds().to_numpy(),
        "air_temp_c": w["AirTemp"].to_numpy(),
        "track_temp_c": w["TrackTemp"].to_numpy(),
        "humidity_pct": w["Humidity"].to_numpy(),
        "pressure_hpa": w["Pressure"].to_numpy(),  # FastF1 reports mbar, same as hPa
        "rainfall": w["Rainfall"].fillna(False).astype(bool).to_numpy(),
        "wind_speed_ms": w["WindSpeed"].to_numpy(),
        "wind_dir_deg": w["WindDirection"].to_numpy(),
    })


def lap_frame(laps, outlier_factor=None, k_slow_s=K_SLOW_S):
    """Convert a FastF1 Laps table into the simulator frame.

    A lap is clean if it is a green flag, non pit lap after lap 1 and is not unusually slow
    for its driver. Slowness is measured against the median lap time of the field on that
    lap number, less the driver's median deficit to it, so slow cars and heavy fuel laps
    stay clean. Only slow laps are outliers. `outlier_factor` is deprecated and ignored.
    """
    df = pd.DataFrame({
        "driver": laps["Driver"].to_numpy(),
        "lap": laps["LapNumber"].astype(int).to_numpy(),
        "stint": laps["Stint"].to_numpy(),
        "compound": laps["Compound"].to_numpy(),
        "tyre_age": laps["TyreLife"].to_numpy(),
        "lap_time_s": laps["LapTime"].dt.total_seconds().to_numpy(),
        "track_status": laps["TrackStatus"].astype(str).to_numpy(),
        "pit_in": laps["PitInTime"].notna().to_numpy(),
        "pit_out": laps["PitOutTime"].notna().to_numpy(),
        "start_s": laps["LapStartTime"].dt.total_seconds().to_numpy(),
        "end_s": laps["Time"].dt.total_seconds().to_numpy(),
        "team": laps["Team"].to_numpy(),
    })
    base = (
        (df["track_status"] == "1") & ~df["pit_in"] & ~df["pit_out"]
        & df["lap_time_s"].notna() & df["tyre_age"].notna() & df["compound"].notna()
    )
    base = base & (df["lap"] > 1)
    base_time = df["lap_time_s"].where(base)
    by_lap = base_time.groupby(df["lap"])
    df["field_ref_s"] = by_lap.transform("median").where(by_lap.transform("count") >= MIN_FIELD_LAPS)
    deficit = base_time - df["field_ref_s"]
    driver_offset = deficit.groupby(df["driver"]).transform("median")
    df["pace_resid_s"] = df["lap_time_s"] - df["field_ref_s"] - driver_offset
    df["clean"] = base & (df["pace_resid_s"] < k_slow_s)
    return df


def race_stints(frame, driver):
    """Actual stints for a driver as (compound, laps, start_age, first_lap)."""
    g = frame[frame["driver"] == driver].sort_values("lap")
    out = []
    for _, s in g.groupby("stint", sort=True):
        out.append((s["compound"].iloc[0], len(s), int(s["tyre_age"].iloc[0]) - 1, int(s["lap"].iloc[0])))
    return out
