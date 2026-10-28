"""Outlier filter checks for lap_frame and practice_frame. Synthetic laps unless marked, no network."""
import os

import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, Simulator
from src.sim.data import CACHE_DIR, HINGE_KNOT, LEGACY_OUTLIER_FACTOR, lap_frame, load_session, practice_frame

N_LAPS = 66
OFFSETS = {"A": 0.0, "B": 0.2, "C": -0.3, "D": 0.1, "E": 0.4, "F": 0.8}  # F is the slow car
# one stoppers run old tyres late in the race while the two stoppers are on fresh ones
PIT_LAPS = {"A": [12], "B": [20, 45], "C": [18], "D": [30, 50], "E": [24], "F": [22, 44]}
STINTS = ["SOFT", "MEDIUM", "SOFT"]
BASE = {"SOFT": 80.0, "MEDIUM": 80.6}
FUEL_S = 0.06    # per lap of fuel still on board: lap 2 is about 4 s slower than the last lap
NOISE_S = 0.2    # clipped at two sigma, so with k_resid = 2.5 noise alone never cuts a lap


def synthetic_laps(wear=0.1, bumps=None, seed=0, noise_s=NOISE_S, cliff=None):
    """FastF1-shaped Laps table, stints in STINTS order split at PIT_LAPS.
    `bumps` maps (driver, lap) to seconds added to that lap, `cliff` maps compound to
    extra wear per lap of tyre age past HINGE_KNOT."""
    bumps, cliff = bumps or {}, cliff or {}
    rng = np.random.default_rng(seed)
    rows = []
    for d, off in OFFSETS.items():
        pits = PIT_LAPS[d]
        for lap in range(1, N_LAPS + 1):
            stint = sum(lap > p for p in pits)
            comp, age = STINTS[stint], lap - ([0] + pits)[stint]
            noise = float(np.clip(rng.normal(0, noise_s), -2 * noise_s, 2 * noise_s))
            t = (BASE[comp] + wear * age + cliff.get(comp, 0.0) * max(0, age - HINGE_KNOT) + FUEL_S * (N_LAPS - lap)
                 + off + noise + bumps.get((d, lap), 0.0))
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


def test_wear_cliff_past_the_knot_is_modelled_not_cut():
    # MEDIUM runs well past the knot, so it gets a hinge; without one the old laps read as slow
    bumps = {(d, lap): 3.0 for i, d in enumerate(OFFSETS) for lap in range(2 + i, N_LAPS + 1, 6)}
    laps = synthetic_laps(bumps=bumps, cliff={"MEDIUM": 0.15})
    f = base_laps(lap_frame(laps))
    bumped = np.array([(d, lap) in bumps for d, lap in zip(f["driver"], f["lap"])])
    assert ((~f["clean"]).to_numpy() == bumped).all()
    straight = base_laps(lap_frame(laps, knot=1000))
    old = (straight["tyre_age"] > 30).to_numpy()
    assert (~straight["clean"].to_numpy() & ~bumped & old).sum() > 0


def test_no_hinge_for_a_compound_with_few_laps_past_the_knot():
    # SOFT has four base laps past the knot (D, ages 26 to 29): too few for a hinge, so a
    # SOFT cliff there stays in the residual instead of being fitted away
    f = lap_frame(synthetic_laps(cliff={"SOFT": 0.3}), min_samples=5)
    past = f[(f["compound"] == "SOFT") & (f["tyre_age"] > HINGE_KNOT) & f["pace_resid_s"].notna()]
    assert len(past) == 4
    assert row(f, "D", 29)["pace_resid_s"] > 1.0


def test_noise_free_laps_are_all_clean():
    f = base_laps(lap_frame(synthetic_laps(noise_s=0.0)))
    assert f["clean"].all()


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


# --- practice_frame ----------------------------------------------------------
def practice_laps(stints, bumps=None, wear=0.1, seed=0):
    """FastF1-shaped practice Laps. `stints` is [(compound, timed laps)] per driver, each
    run framed by an out lap and an in lap; `bumps` maps (driver, LapNumber) to seconds."""
    bumps = bumps or {}
    rng = np.random.default_rng(seed)
    rows = []
    for d, off in OFFSETS.items():
        lap = 0
        for k, (comp, n) in enumerate(stints):
            for age in range(n + 2):  # out lap, n timed laps, in lap
                lap += 1
                noise = float(np.clip(rng.normal(0, NOISE_S), -2 * NOISE_S, 2 * NOISE_S))
                t = BASE[comp] + wear * age + off + noise + bumps.get((d, lap), 0.0) + (8.0 if age in (0, n + 1) else 0.0)
                rows.append({
                    "Driver": d, "LapNumber": float(lap), "Stint": float(k + 1), "Compound": comp,
                    "TyreLife": float(age + 1), "LapTime": pd.Timedelta(seconds=t), "TrackStatus": "1",
                    "PitInTime": pd.Timedelta(seconds=1) if age == n + 1 else pd.NaT,
                    "PitOutTime": pd.Timedelta(seconds=1) if age == 0 else pd.NaT,
                    "LapStartTime": pd.Timedelta(seconds=0), "Time": pd.Timedelta(seconds=t),
                    "Team": "T", "Position": np.nan,
                })
    return pd.DataFrame(rows)


def test_practice_frame_columns_and_long_runs():
    sessions = {"FP1": practice_laps([("SOFT", 3), ("MEDIUM", 8)]), "FP2": practice_laps([("SOFT", 10)], seed=1)}
    f = practice_frame(sessions)
    assert list(f.columns) == list(lap_frame(sessions["FP1"]).columns) + ["session", "stint_len", "lap_in_stint"]
    assert f["lap"].isna().all()
    assert len(f) == len(OFFSETS) * (5 + 10 + 12)
    run = f[(f["session"] == "FP1") & (f["driver"] == "A") & (f["stint"] == 2)]
    assert run["stint_len"].eq(8).all() and list(run["lap_in_stint"]) == list(range(1, 11))
    short = f[(f["session"] == "FP1") & (f["stint"] == 1)]
    assert short["stint_len"].eq(3).all() and not short["clean"].any() and short["pace_resid_s"].isna().all()
    timed = f[~f["pit_in"] & ~f["pit_out"] & (f["stint_len"] >= 5)]
    assert timed["clean"].all() and timed["pace_resid_s"].notna().all()
    assert not f.loc[f["pit_in"] | f["pit_out"], "clean"].any()


def test_practice_frame_cuts_slow_laps_with_the_race_filter():
    laps = practice_laps([("SOFT", 10), ("MEDIUM", 10)], bumps={("B", 5): 3.0, ("C", 6): -3.0})
    f = practice_frame({"FP2": laps})
    lap5 = f[(f["driver"] == "B") & (f["lap_in_stint"] == 5) & (f["stint"] == 1)].iloc[0]
    assert not lap5["clean"] and lap5["pace_resid_s"] == pytest.approx(3.0, abs=0.5)
    assert f[(f["driver"] == "C") & (f["lap_in_stint"] == 6) & (f["stint"] == 1)]["clean"].all()  # fast stays
    assert int((~f["clean"] & f["pace_resid_s"].notna()).sum()) == 1
    slow = f.loc[f["driver"] == "F", "driver_offset_s"].iloc[0]
    assert slow == pytest.approx(OFFSETS["F"] - np.mean(list(OFFSETS.values())), abs=0.05)
    fit = Simulator(lap_frame(synthetic_laps()), SimConfig(fit_window="pre_race"), f).degradation
    assert fit.coefs["SOFT"] == pytest.approx((BASE["SOFT"] - 0.1 + np.mean(list(OFFSETS.values())), 0.1), abs=0.1)
