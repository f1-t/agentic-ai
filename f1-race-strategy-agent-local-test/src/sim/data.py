"""Adapters from FastF1 laps to the frame the simulator fits on.

The simulator itself never imports FastF1. Only `load_session` does, lazily.
Frame columns: driver, lap, stint, compound, tyre_age, lap_time_s, track_status,
pit_in, pit_out, start_s, end_s, team, clean.
"""
import os

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
OUTLIER_FACTOR = 1.07  # drop laps slower than 107% of the compound's fastest clean lap


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


def lap_frame(laps, outlier_factor=OUTLIER_FACTOR):
    """Convert a FastF1 Laps table into the simulator frame."""
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
    fastest = df["lap_time_s"].where(base).groupby(df["compound"]).transform("min")
    df["clean"] = base & (df["lap_time_s"] < fastest * outlier_factor)
    return df


def race_stints(frame, driver):
    """Actual stints for a driver as (compound, laps, start_age, first_lap)."""
    g = frame[frame["driver"] == driver].sort_values("lap")
    out = []
    for _, s in g.groupby("stint", sort=True):
        out.append((s["compound"].iloc[0], len(s), int(s["tyre_age"].iloc[0]) - 1, int(s["lap"].iloc[0])))
    return out
