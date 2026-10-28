"""Per-lap weather from the session weather feed and the lap frame.

`lap_weather` bins the feed (FastF1 samples roughly once a minute) onto lap windows.
`snapshot_weather` turns the row at a decision lap into the kwargs for `Snapshot.weather`.
Feed columns: time_s, air_temp_c, track_temp_c, humidity_pct, pressure_hpa, rainfall,
wind_speed_ms, wind_dir_deg. Wind is m/s in the feed and km/h here, as in the contract.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RAIN_THRESHOLD = 0.5   # a lap is wet when more than this share of its samples show rain
MS_TO_KPH = 3.6
COLUMNS = ["rain_frac", "rain", "track_temp_c", "air_temp_c", "wind_kph", "n_samples"]


def _mean(x):
    return np.nan if np.isnan(x).all() else float(np.nanmean(x))


def lap_weather(weather, frame, rain_threshold=RAIN_THRESHOLD):
    """Weather over each lap, indexed by lap number.

    A lap's window runs from the earliest start to the latest finish of any driver on it.
    A lap with no sample inside carries the last earlier sample forward (n_samples = 0).
    """
    win = frame.groupby("lap").agg(start=("start_s", "min"), end=("end_s", "max"))
    out = pd.DataFrame(
        {"rain_frac": 0.0, "rain": False, "track_temp_c": np.nan, "air_temp_c": np.nan,
         "wind_kph": np.nan, "n_samples": 0},
        index=win.index.rename("lap"),
    )
    if weather is None or len(weather) == 0:
        log.warning("empty weather frame: lap weather is unknown and treated as dry")
        return out[COLUMNS]

    w = weather.sort_values("time_s")
    t = w["time_s"].to_numpy(dtype=float)
    rain = w["rainfall"].to_numpy(dtype=float)
    track = w["track_temp_c"].to_numpy(dtype=float)
    air = w["air_temp_c"].to_numpy(dtype=float)
    wind = w["wind_speed_ms"].to_numpy(dtype=float) * MS_TO_KPH

    for lap, start, end in zip(win.index, win["start"], win["end"]):
        if np.isnan(start) or np.isnan(end):
            continue
        lo = np.searchsorted(t, start, "left")
        hi = np.searchsorted(t, end, "right")
        n = hi - lo
        if n == 0:
            lo, hi = lo - 1, lo   # the last sample before the window
            if lo < 0:
                continue
        out.loc[lap, "n_samples"] = n
        out.loc[lap, "rain_frac"] = rain[lo:hi].mean()
        out.loc[lap, "track_temp_c"] = _mean(track[lo:hi])
        out.loc[lap, "air_temp_c"] = _mean(air[lo:hi])
        out.loc[lap, "wind_kph"] = _mean(wind[lo:hi])
    out["rain"] = out["rain_frac"] > rain_threshold
    return out[COLUMNS]


def rain_laps(lw):
    """Lap numbers flagged wet, for `Snapshot.weather.rain_laps`."""
    return [int(lap) for lap in lw.index[lw["rain"]]]


def snapshot_weather(lw, lap):
    """Kwargs for the `Weather` contract at `lap`, using only laps up to it."""
    seen = lw[lw.index <= lap]
    if seen.empty:
        return {"rainfall_now": False, "track_temp_c": None, "air_temp_c": None,
                "wind_kph": None, "rain_laps": []}
    now = seen.iloc[-1]

    def val(col):
        return None if pd.isna(now[col]) else float(now[col])

    return {
        "rainfall_now": bool(now["rain"]),
        "track_temp_c": val("track_temp_c"),
        "air_temp_c": val("air_temp_c"),
        "wind_kph": val("wind_kph"),
        "rain_laps": rain_laps(seen),
    }
