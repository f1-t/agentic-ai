"""Lap weather binning on synthetic feeds. Offline, no FastF1."""
import logging

import numpy as np
import pandas as pd
import pytest

from src.contracts import Weather
from src.sim.weather import lap_weather, rain_laps, snapshot_weather

LAP_S = 90.0


def make_frame(n_laps=20, drivers=("AAA", "BBB")):
    """90 s laps; each later driver runs 4 s behind the one before."""
    rows = []
    for i, d in enumerate(drivers):
        for lap in range(1, n_laps + 1):
            rows.append({"driver": d, "lap": lap,
                         "start_s": (lap - 1) * LAP_S + 4 * i, "end_s": lap * LAP_S + 4 * i})
    return pd.DataFrame(rows)


def make_weather(times, rain_from=None, rain_to=None):
    t = np.asarray(times, dtype=float)
    wet = np.zeros(len(t), dtype=bool) if rain_from is None else (t >= rain_from) & (t < rain_to)
    return pd.DataFrame({
        "time_s": t, "air_temp_c": 20.0 + t / 1000, "track_temp_c": 30.0 + t / 100,
        "humidity_pct": 50.0, "pressure_hpa": 1000.0, "rainfall": wet,
        "wind_speed_ms": 2.0, "wind_dir_deg": 90.0,
    })


def test_rain_block_gives_full_and_partial_laps():
    # 60 s samples; wet from t=810 (start of lap 10) to t=1275 (inside lap 15's window)
    lw = lap_weather(make_weather(np.arange(0, 1800, 60), 810, 1275), make_frame())
    assert lw.index.name == "lap"
    assert (lw.loc[:9, "rain_frac"] == 0).all()
    assert (lw.loc[10:14, "rain_frac"] == 1.0).all()
    assert (lw.loc[16:, "rain_frac"] == 0).all()
    # lap 15 window is [1260, 1354]: samples 1260 (wet), 1320 (dry)
    assert lw.loc[15, "rain_frac"] == pytest.approx(0.5)
    assert not lw.loc[15, "rain"]
    assert rain_laps(lw) == [10, 11, 12, 13, 14]


def test_boundary_lap_is_partial():
    # lap 10 spans [810, 904] and holds samples 840 and 900; only 900 onwards is wet
    lw = lap_weather(make_weather(np.arange(0, 1800, 60), 870, 1300), make_frame())
    assert lw.loc[10, "rain_frac"] == pytest.approx(0.5)
    assert lw.loc[11, "rain_frac"] == 1.0


def test_threshold_is_exclusive_and_adjustable():
    weather = make_weather(np.arange(0, 1800, 60), 870, 1300)
    frame = make_frame()
    assert not lap_weather(weather, frame).loc[10, "rain"]
    assert lap_weather(weather, frame, rain_threshold=0.4).loc[10, "rain"]


def test_window_spans_all_drivers():
    # a sample at 92 s is inside lap 1 only because the slower driver finishes it at 94 s
    lw = lap_weather(make_weather([92.0]), make_frame(n_laps=2))
    assert lw.loc[1, "n_samples"] == 1


def test_means_and_wind_conversion():
    lw = lap_weather(make_weather([0.0, 30.0, 60.0]), make_frame(n_laps=1))
    assert lw.loc[1, "n_samples"] == 3
    assert lw.loc[1, "track_temp_c"] == pytest.approx(30.3)
    assert lw.loc[1, "air_temp_c"] == pytest.approx(20.03)
    assert lw.loc[1, "wind_kph"] == pytest.approx(7.2)


def test_sparse_feed_carries_last_sample_forward():
    # samples every 300 s: laps 2, 3 and 5 hold none of their own
    lw = lap_weather(make_weather([0.0, 300.0], 300, 600), make_frame(n_laps=5))
    assert lw.loc[1, "n_samples"] == 1
    assert lw.loc[2, "n_samples"] == 0
    assert lw.loc[2, "track_temp_c"] == pytest.approx(30.0)
    assert lw.loc[2, "rain_frac"] == 0.0
    assert lw.loc[4, "n_samples"] == 1     # window [270, 364] holds the 300 s sample
    assert lw.loc[5, "n_samples"] == 0
    assert lw.loc[5, "rain_frac"] == 1.0   # carried from the wet 300 s sample
    assert lw.loc[5, "rain"]


def test_laps_before_the_first_sample_are_unknown():
    lw = lap_weather(make_weather([500.0]), make_frame(n_laps=3))
    assert (lw["n_samples"] == 0).all()
    assert lw["track_temp_c"].isna().all()
    assert (lw["rain_frac"] == 0).all()


def test_empty_weather_warns_and_returns_nan(caplog):
    with caplog.at_level(logging.WARNING):
        lw = lap_weather(make_weather([]), make_frame(n_laps=3))
    assert "empty weather" in caplog.text
    assert list(lw.index) == [1, 2, 3]
    assert (lw["rain_frac"] == 0).all() and not lw["rain"].any()
    assert lw[["track_temp_c", "air_temp_c", "wind_kph"]].isna().all().all()
    assert (lw["n_samples"] == 0).all()
    assert rain_laps(lw) == []


def test_snapshot_weather_matches_contract():
    lw = lap_weather(make_weather(np.arange(0, 1800, 60), 810, 1260), make_frame())
    w = snapshot_weather(lw, 12)
    assert w["rainfall_now"] is True
    assert w["rain_laps"] == [10, 11, 12]          # nothing past the decision lap
    assert w["wind_kph"] == pytest.approx(7.2)
    assert Weather(**w).track_temp_c == pytest.approx(lw.loc[12, "track_temp_c"])
    assert snapshot_weather(lw, 5)["rainfall_now"] is False


def test_snapshot_weather_before_first_lap_and_unknown_values():
    lw = lap_weather(make_weather([]), make_frame(n_laps=3))
    assert Weather(**snapshot_weather(lw, 0)) == Weather()
    w = snapshot_weather(lw, 2)
    assert w["track_temp_c"] is None and w["wind_kph"] is None
    Weather(**w)
