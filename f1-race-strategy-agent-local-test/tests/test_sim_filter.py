"""Outlier filter checks for lap_frame. Synthetic laps, no cache, no network."""
import numpy as np
import pandas as pd
import pytest

from src.sim.data import lap_frame

DRIVERS = ["A", "B", "C", "D", "E", "F"]
SLOW_CAR = "F"
OFFSETS = {"A": 0.0, "B": 0.2, "C": -0.3, "D": 0.1, "E": 0.4, SLOW_CAR: 4.0}


def field_time(lap):
    return 100.0 - 0.5 * lap  # fuel burn: early laps are slow for everybody


def synthetic_laps(n_laps=12, bumps=None, drop=()):
    """FastF1-shaped Laps table. `bumps` maps (driver, lap) to seconds added to that lap,
    `drop` lists (driver, lap) rows to leave out."""
    bumps = bumps or {}
    rows = []
    for d in DRIVERS:
        for lap in range(1, n_laps + 1):
            if (d, lap) in drop:
                continue
            t = field_time(lap) + OFFSETS[d] + bumps.get((d, lap), 0.0)
            rows.append({
                "Driver": d, "LapNumber": float(lap), "Stint": 1.0, "Compound": "SOFT",
                "TyreLife": float(lap), "LapTime": pd.Timedelta(seconds=t), "TrackStatus": "1",
                "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
                "LapStartTime": pd.Timedelta(seconds=0), "Time": pd.Timedelta(seconds=t),
                "Team": "T",
            })
    return pd.DataFrame(rows)


def clean_at(frame, driver, lap):
    return bool(frame.loc[(frame["driver"] == driver) & (frame["lap"] == lap), "clean"].iloc[0])


def test_early_race_slow_laps_stay_clean():
    f = lap_frame(synthetic_laps())
    early = f[(f["lap"] >= 2) & (f["lap"] <= 4)]
    assert len(early) == 3 * len(DRIVERS)
    assert early["clean"].all()


def test_slow_car_laps_stay_clean():
    f = lap_frame(synthetic_laps())
    slow = f[(f["driver"] == SLOW_CAR) & (f["lap"] > 1)]
    assert slow["clean"].all()
    assert slow["pace_resid_s"].abs().max() < 1e-9


def test_traffic_lap_is_dropped():
    f = lap_frame(synthetic_laps(bumps={("B", 6): 3.0}))
    assert not clean_at(f, "B", 6)
    assert clean_at(f, "B", 5) and clean_at(f, "B", 7)
    assert f.loc[(f["driver"] == "B") & (f["lap"] == 6), "pace_resid_s"].iloc[0] == pytest.approx(3.0, abs=0.2)


def test_fast_lap_is_never_dropped():
    f = lap_frame(synthetic_laps(bumps={("C", 6): -3.0}))
    assert clean_at(f, "C", 6)
    assert f.loc[(f["driver"] == "C") & (f["lap"] == 6), "pace_resid_s"].iloc[0] < -2.0


def test_laps_with_fewer_than_five_field_laps_are_not_clean():
    # lap 12 is run by 4 drivers only
    f = lap_frame(synthetic_laps(drop=[("E", 12), (SLOW_CAR, 12)]))
    last = f[f["lap"] == 12]
    assert len(last) == 4
    assert not last["clean"].any()
    assert last["field_ref_s"].isna().all()
    assert f.loc[f["lap"] == 11, "clean"].all()  # five or more are fine: all six present


def test_exactly_five_field_laps_is_enough():
    f = lap_frame(synthetic_laps(drop=[(SLOW_CAR, 12)]))
    assert f.loc[f["lap"] == 12, "clean"].all()


def test_lap_one_and_pit_laps_are_never_clean():
    laps = synthetic_laps()
    laps.loc[(laps["Driver"] == "A") & (laps["LapNumber"] == 5), "PitInTime"] = pd.Timedelta(seconds=1)
    laps.loc[(laps["Driver"] == "A") & (laps["LapNumber"] == 6), "PitOutTime"] = pd.Timedelta(seconds=1)
    f = lap_frame(laps)
    assert not f.loc[f["lap"] == 1, "clean"].any()
    assert not clean_at(f, "A", 5) and not clean_at(f, "A", 6)
    assert np.isnan(f.loc[f["lap"] == 1, "field_ref_s"]).all()


def test_outlier_factor_is_accepted_and_ignored():
    laps = synthetic_laps(bumps={("B", 6): 3.0})
    assert lap_frame(laps, outlier_factor=1.5).equals(lap_frame(laps))
