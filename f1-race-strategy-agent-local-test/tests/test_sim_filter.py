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
def practice_laps(stints, bumps=None, wear=0.1, seed=0, evo=0.0, noise_s=NOISE_S, fuel=0.0):
    """FastF1-shaped practice Laps. `stints` is [(compound, timed laps)] per driver, each
    run framed by an out lap and an in lap; `bumps` maps (driver, LapNumber) to seconds;
    `evo` is track evolution in seconds per minute of the run's start time; `fuel` is the
    gain per lap of the run. Every driver starts at session time 0 and runs without breaks."""
    bumps = bumps or {}
    rng = np.random.default_rng(seed)
    rows = []
    for d, off in OFFSETS.items():
        lap, clock = 0, 0.0
        for k, (comp, n) in enumerate(stints):
            run_start = clock
            for age in range(n + 2):  # out lap, n timed laps, in lap
                lap += 1
                noise = float(np.clip(rng.normal(0, noise_s), -2 * noise_s, 2 * noise_s))
                t = (BASE[comp] + wear * age + fuel * (age + 1) + off + noise + evo * run_start / 60.0
                     + bumps.get((d, lap), 0.0) + (8.0 if age in (0, n + 1) else 0.0))
                rows.append({
                    "Driver": d, "LapNumber": float(lap), "Stint": float(k + 1), "Compound": comp,
                    "TyreLife": float(age + 1), "LapTime": pd.Timedelta(seconds=t), "TrackStatus": "1",
                    "PitInTime": pd.Timedelta(seconds=1) if age == n + 1 else pd.NaT,
                    "PitOutTime": pd.Timedelta(seconds=1) if age == 0 else pd.NaT,
                    "LapStartTime": pd.Timedelta(seconds=clock), "Time": pd.Timedelta(seconds=clock + t),
                    "Team": "T", "Position": np.nan,
                })
                clock += t
    return pd.DataFrame(rows)


def run_class(f, session, driver, stint):
    return f.loc[(f["session"] == session) & (f["driver"] == driver) & (f["stint"] == stint), "run_class"].iloc[0]


def test_practice_frame_columns_and_long_runs():
    sessions = {"FP1": practice_laps([("SOFT", 3), ("MEDIUM", 8)]), "FP2": practice_laps([("SOFT", 10)], seed=1)}
    f = practice_frame(sessions)
    assert list(f.columns) == list(lap_frame(sessions["FP1"]).columns) + [
        "session", "stint_len", "lap_in_stint", "session_time_min", "stint_start_min", "run_class"]
    assert f["lap"].isna().all()
    assert len(f) == len(OFFSETS) * (5 + 10 + 12)
    run = f[(f["session"] == "FP1") & (f["driver"] == "A") & (f["stint"] == 2)]
    assert run["stint_len"].eq(8).all() and list(run["lap_in_stint"]) == list(range(1, 11))
    assert run["run_class"].eq("long_run").all()
    short = f[(f["session"] == "FP1") & (f["stint"] == 1)]
    assert short["stint_len"].eq(3).all() and short["run_class"].eq("short").all()
    assert not short["clean"].any() and short["pace_resid_s"].isna().all()
    timed = f[~f["pit_in"] & ~f["pit_out"] & (f["run_class"] == "long_run")]
    assert timed["clean"].all() and timed["pace_resid_s"].notna().all()
    assert not f.loc[f["pit_in"] | f["pit_out"], "clean"].any()
    for _, g in f.groupby("session"):
        assert g["session_time_min"].min() == 0.0
    a = f[(f["session"] == "FP2") & (f["driver"] == "A")]
    assert a["session_time_min"].is_monotonic_increasing
    assert a["session_time_min"].iloc[1] == pytest.approx(a["lap_time_s"].iloc[0] / 60.0)
    assert (a["stint_start_min"] == a.groupby("stint")["session_time_min"].transform("min")).all()


def test_long_run_test_drops_short_alternating_and_off_pace_stints():
    alternating = {("A", lap): 6.0 for lap in range(14, 24, 2)}   # A, stint 2: push / cool
    low_fuel = {("B", lap): -4.0 for lap in range(26, 36)}        # B, stint 3: consistent, 4 s quick
    traffic = {("C", 16): 4.0, ("C", 19): 5.0}                    # C, stint 2: two traffic laps
    laps = practice_laps([("SOFT", 10), ("SOFT", 10), ("MEDIUM", 10), ("MEDIUM", 5)],
                         bumps={**alternating, **low_fuel, **traffic})
    f = practice_frame({"FP2": laps})
    assert run_class(f, "FP2", "A", 2) == "inconsistent"
    assert run_class(f, "FP2", "B", 3) == "off_pace"
    assert run_class(f, "FP2", "C", 4) == "short"
    assert SimConfig().long_run.consistent_share == 0.8  # frozen: at most one lap in five off-pace
    assert run_class(f, "FP2", "C", 2) == "long_run"  # 8 of 10 laps within 2 s of the median
    assert run_class(f, "FP2", "A", 1) == run_class(f, "FP2", "B", 2) == "long_run"
    dropped = f[f["run_class"] != "long_run"]
    assert dropped["pace_resid_s"].isna().all() and not dropped["clean"].any()
    stints = f.groupby(["driver", "stint"])["run_class"].first().value_counts().to_dict()
    assert stints == {"long_run": 3 * len(OFFSETS) - 2, "short": len(OFFSETS), "inconsistent": 1, "off_pace": 1}
    strict = practice_frame({"FP2": laps}, SimConfig(long_run={"min_timed_laps": 11}).long_run)
    assert set(strict["run_class"]) == {"short"}
    tight = practice_frame({"FP2": laps}, SimConfig(long_run={"consistent_share": 0.9}).long_run)
    assert run_class(tight, "FP2", "C", 2) == "inconsistent"


def test_session_evolution_between_runs_is_not_read_as_base_or_wear():
    # two SOFT runs back to back: the tyre age resets, the run start time does not
    laps = practice_laps([("SOFT", 10), ("SOFT", 10), ("MEDIUM", 10)], evo=-0.02, noise_s=0.05)
    f = practice_frame({"FP2": laps})
    cfg = SimConfig(fit_window="pre_race", degradation={"name": "linear"}, pre_race_base_delta_s=0.0)
    fit = Simulator(lap_frame(synthetic_laps()), cfg, f).degradation
    level = np.mean(list(OFFSETS.values())) - 0.1  # TyreLife is the age index plus one
    assert fit.coefs["SOFT"] == pytest.approx((BASE["SOFT"] + level, 0.1), abs=0.05)
    # the slope is identified across run start times only; the 0.02 s/min ridge pulls it part way to 0
    assert -0.02 < fit.fit_report["evolution_s_per_min"]["FP2"] < -0.01
    blind = Simulator(lap_frame(synthetic_laps()), cfg, f.drop(columns=["stint_start_min"])).degradation
    err, blind_err = (abs(m.coefs["MEDIUM"][0] - BASE["MEDIUM"] - level) for m in (fit, blind))
    assert blind_err > 2 * err  # without the term the later run reads as a faster tyre
    assert blind.fit_report["evolution_s_per_min"] == {}


def test_practice_in_stint_fuel_is_taken_at_the_prior():
    prior = SimConfig().pre_race_fuel_slope_s_per_lap
    laps = practice_laps([("SOFT", 10), ("SOFT", 10), ("MEDIUM", 10)], fuel=prior, noise_s=0.05)
    f = practice_frame({"FP2": laps})
    cfg = dict(fit_window="pre_race", pre_race_base_delta_s=0.0)
    fit = Simulator(lap_frame(synthetic_laps()), SimConfig(**cfg), f).degradation
    assert fit.coefs["SOFT"][1] == pytest.approx(0.1, abs=0.01)
    assert fit.fit_report["fuel_slope_source"] == "prior"
    net = Simulator(lap_frame(synthetic_laps()), SimConfig(**cfg, degradation={"name": "linear"}), f).degradation
    assert net.coefs["SOFT"][1] == pytest.approx(0.1 + prior, abs=0.01)  # no fuel term: wear net of fuel


def test_practice_compound_without_a_long_run_is_unmodelled():
    laps = practice_laps([("SOFT", 10), ("MEDIUM", 10)])
    laps.loc[(laps["Compound"] == "MEDIUM") & (laps["Driver"] == "A"), "Compound"] = "HARD"
    laps.loc[laps["Compound"] == "HARD", "LapTime"] += pd.to_timedelta(np.tile([0.0, 6.0], 6)[:12], unit="s")
    f = practice_frame({"FP1": laps})
    assert set(f.loc[f["compound"] == "HARD", "run_class"]) == {"inconsistent"}
    fit = Simulator(lap_frame(synthetic_laps()), SimConfig(fit_window="pre_race"), f).degradation
    assert "HARD" not in fit.coefs and fit.fit_report["unmodelled"] == {"HARD": "no long run"}


@pytest.mark.parametrize("practice", [False, True])
def test_a_lap_faster_than_the_model_is_never_cut(practice):
    if practice:
        bumps = {("C", 6): -3.0, ("E", 30): -2.0, ("B", 5): 3.0}
        f = practice_frame({"FP2": practice_laps([("SOFT", 10), ("SOFT", 10), ("MEDIUM", 10)], bumps=bumps)})
    else:
        bumps = {("C", 10): -3.0, ("E", 40): -2.0, ("B", 5): 3.0}
        f = lap_frame(synthetic_laps(bumps=bumps))
    fitted = f[f["pace_resid_s"].notna()]
    assert (fitted["pace_resid_s"] < -1.5).sum() >= 2
    assert fitted.loc[fitted["pace_resid_s"] <= 0, "clean"].all()
    assert not fitted.loc[fitted["pace_resid_s"] > 2.5, "clean"].any()


def test_practice_frame_cuts_slow_laps_with_the_race_filter():
    laps = practice_laps([("SOFT", 10), ("SOFT", 10), ("MEDIUM", 10)], bumps={("B", 5): 3.0})
    f = practice_frame({"FP2": laps})
    lap5 = f[(f["driver"] == "B") & (f["lap_in_stint"] == 5) & (f["stint"] == 1)].iloc[0]
    assert not lap5["clean"] and lap5["pace_resid_s"] == pytest.approx(3.0, abs=0.5)
    assert int((~f["clean"] & f["pace_resid_s"].notna()).sum()) == 1
    slow = f.loc[f["driver"] == "F", "driver_offset_s"].iloc[0]
    assert slow == pytest.approx(OFFSETS["F"] - np.mean(list(OFFSETS.values())), abs=0.05)
    cfg = SimConfig(fit_window="pre_race", degradation={"name": "linear"}, pre_race_base_delta_s=0.0)
    fit = Simulator(lap_frame(synthetic_laps()), cfg, f).degradation
    assert fit.coefs["SOFT"] == pytest.approx((BASE["SOFT"] - 0.1 + np.mean(list(OFFSETS.values())), 0.1), abs=0.1)
