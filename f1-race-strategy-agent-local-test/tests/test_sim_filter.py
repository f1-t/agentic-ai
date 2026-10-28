"""Outlier filter checks for lap_frame. Synthetic laps unless marked, no network."""
import os

import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, Simulator
from src.sim.data import CACHE_DIR, LEGACY_OUTLIER_FACTOR, lap_frame, load_session

N_LAPS = 66
OFFSETS = {"A": 0.0, "B": 0.2, "C": -0.3, "D": 0.1, "E": 0.4, "F": 0.8}  # F is the slow car
# one stoppers run old tyres late in the race while the two stoppers are on fresh ones
PIT_LAPS = {"A": [12], "B": [20, 45], "C": [18], "D": [30, 50], "E": [24], "F": [22, 44]}
STINTS = ["SOFT", "MEDIUM", "SOFT"]
BASE = {"SOFT": 80.0, "MEDIUM": 80.6}
FUEL_S = 0.06    # per lap of fuel still on board: lap 2 is about 4 s slower than the last lap
NOISE_S = 0.2    # clipped at two sigma, so with k_resid = 2.5 noise alone never cuts a lap


def synthetic_laps(wear=0.1, bumps=None, seed=0):
    """FastF1-shaped Laps table, stints in STINTS order split at PIT_LAPS.
    `bumps` maps (driver, lap) to seconds added to that lap."""
    bumps = bumps or {}
    rng = np.random.default_rng(seed)
    rows = []
    for d, off in OFFSETS.items():
        pits = PIT_LAPS[d]
        for lap in range(1, N_LAPS + 1):
            stint = sum(lap > p for p in pits)
            comp, age = STINTS[stint], lap - ([0] + pits)[stint]
            noise = float(np.clip(rng.normal(0, NOISE_S), -2 * NOISE_S, 2 * NOISE_S))
            t = BASE[comp] + wear * age + FUEL_S * (N_LAPS - lap) + off + noise + bumps.get((d, lap), 0.0)
            rows.append({
                "Driver": d, "LapNumber": float(lap), "Stint": float(stint + 1), "Compound": comp,
                "TyreLife": float(age), "LapTime": pd.Timedelta(seconds=t), "TrackStatus": "1",
                "PitInTime": pd.Timedelta(seconds=1) if lap in pits else pd.NaT,
                "PitOutTime": pd.Timedelta(seconds=1) if lap - 1 in pits else pd.NaT,
                "LapStartTime": pd.Timedelta(seconds=0), "Time": pd.Timedelta(seconds=t),
                "Team": "T", "Position": float(list(OFFSETS).index(d) + 1),
            })
    return pd.DataFrame(rows)


def row(frame, driver, lap):
    return frame.loc[(frame["driver"] == driver) & (frame["lap"] == lap)].iloc[0]


def base_laps(frame):
    return frame[frame["pace_resid_s"].notna()]


def test_heavy_fuel_early_laps_stay_clean():
    f = lap_frame(synthetic_laps())
    early = f[(f["lap"] >= 2) & (f["lap"] <= 4)]
    assert len(early) == 3 * len(OFFSETS)
    assert early["clean"].all()
    assert base_laps(f)["clean"].all()


def test_slow_car_laps_stay_clean():
    f = lap_frame(synthetic_laps())
    slow = base_laps(f)[base_laps(f)["driver"] == "F"]
    assert slow["clean"].all()
    expected = OFFSETS["F"] - np.mean(list(OFFSETS.values()))
    assert row(f, "F", 10)["driver_offset_s"] == pytest.approx(expected, abs=0.05)


def test_traffic_lap_is_dropped():
    f = lap_frame(synthetic_laps(bumps={("B", 10): 3.0}))
    assert not row(f, "B", 10)["clean"]
    assert row(f, "B", 9)["clean"] and row(f, "B", 11)["clean"]
    assert row(f, "B", 10)["pace_resid_s"] == pytest.approx(3.0, abs=0.5)


def test_fast_lap_is_never_dropped():
    f = lap_frame(synthetic_laps(bumps={("C", 10): -3.0}))
    assert row(f, "C", 10)["clean"]
    assert row(f, "C", 10)["pace_resid_s"] < -2.0


def test_cut_share_does_not_depend_on_tyre_age():
    # traffic on every sixth lap of every driver, so the true cut share is the same at
    # every tyre age; a field relative reference cuts old tyre laps far more often
    bumps = {(d, lap): 3.0 for i, d in enumerate(OFFSETS) for lap in range(2 + i, N_LAPS + 1, 6)}
    f = base_laps(lap_frame(synthetic_laps(bumps=bumps)))
    cut = ~f["clean"]
    old, fresh = cut[f["tyre_age"] > 30], cut[f["tyre_age"] < 10]
    assert len(old) >= 50 and len(fresh) >= 50
    assert 0 < fresh.mean() and abs(old.mean() - fresh.mean()) <= 0.05
    bumped = [(d, lap) in bumps for d, lap in zip(f["driver"], f["lap"])]
    assert (cut.to_numpy() == np.array(bumped)).all()


def test_rare_compound_laps_stay_clean():
    laps = synthetic_laps()
    rare = (laps["Driver"] == "A") & laps["LapNumber"].between(2, 5)
    laps.loc[rare, "Compound"] = "HARD"
    laps.loc[rare, "LapTime"] = laps.loc[rare, "LapTime"] + pd.Timedelta(seconds=5)
    f = lap_frame(laps, min_samples=5)
    hard = f[f["compound"] == "HARD"]
    assert len(hard) == 4
    assert hard["clean"].all() and hard["pace_resid_s"].isna().all()


def test_lap_one_and_pit_laps_are_never_clean():
    f = lap_frame(synthetic_laps())
    assert not f.loc[f["lap"] == 1, "clean"].any()
    pit = PIT_LAPS["A"][0]
    assert not row(f, "A", pit)["clean"] and not row(f, "A", pit + 1)["clean"]
    assert f.loc[f["lap"] == 1, "pace_resid_s"].isna().all()


@pytest.mark.skipif(not os.path.isdir(os.path.join(CACHE_DIR, "2021")), reason="no cache")
def test_legacy_path_reproduces_baseline_fit():
    frame = lap_frame(load_session(2021, "Spain").laps, LEGACY_OUTLIER_FACTOR)
    sim = Simulator(frame, SimConfig(degradation={"name": "linear"}))
    assert sim.degradation.coefs["SOFT"] == pytest.approx((81.93, 0.0512), abs=1e-3)
