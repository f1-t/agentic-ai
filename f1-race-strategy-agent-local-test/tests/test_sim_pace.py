"""Pre-race pace term (pace.py) and the pace argument of the position check. Synthetic data, offline."""
import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, Simulator, calibration, pace
from src.sim.data import practice_frame
from src.sim.degradation import LinearDegradation
from src.sim.tyre_sets import stand_in
from test_sim import RACE, synthetic_frame
from test_sim_position import three_driver_frame

BASE = {"SOFT": 80.0, "MEDIUM": 80.6}
TEAMS = {"A": "T1", "B": "T1", "C": "T2", "D": "T2"}
SLOW = {"B": 0.5}   # B is half a second a lap slower than the field


def practice_laps(stints=(("SOFT", 10), ("MEDIUM", 10)), drivers=TEAMS, offsets=SLOW, seed=0, noise_s=0.05):
    """FastF1-shaped practice Laps: per driver, each run is an out lap, the timed laps and an in lap."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, team in drivers.items():
        lap, clock = 0, 0.0
        for k, (comp, n) in enumerate(stints):
            for age in range(n + 2):
                lap += 1
                noise = float(np.clip(rng.normal(0, noise_s), -2 * noise_s, 2 * noise_s))
                t = BASE[comp] + 0.1 * age + offsets.get(d, 0.0) + noise + (8.0 if age in (0, n + 1) else 0.0)
                rows.append({
                    "Driver": d, "LapNumber": float(lap), "Stint": float(k + 1), "Compound": comp,
                    "TyreLife": float(age + 1), "LapTime": pd.Timedelta(t, unit="s"), "TrackStatus": "1",
                    "PitInTime": pd.Timedelta(1, unit="s") if age == n + 1 else pd.NaT,
                    "PitOutTime": pd.Timedelta(1, unit="s") if age == 0 else pd.NaT,
                    "LapStartTime": pd.Timedelta(clock, unit="s"), "Time": pd.Timedelta(clock + t, unit="s"),
                    "Team": team, "Position": np.nan,
                })
                clock += t
    return pd.DataFrame(rows)


def practice_fit(practice):
    sim = Simulator(synthetic_frame(), SimConfig(fit_window="pre_race"), practice)
    return pace.practice_model(sim, practice)


@pytest.fixture(scope="module")
def practice():
    return practice_frame({"FP2": practice_laps(), "FP3": practice_laps(seed=1)})


def test_practice_pace_recovers_a_slow_driver_relative_to_the_field(practice):
    table = pace.practice_pace(practice, practice_fit(practice))
    assert list(table.columns) == ["offset_s", "n_laps", "team", "source"]
    assert table.loc["B", "offset_s"] == pytest.approx(0.5, abs=0.05)
    for d in "ACD":
        assert table.loc[d, "offset_s"] == pytest.approx(0.0, abs=0.05)
    assert (table["source"] == "practice").all() and (table["n_laps"] == 40).all()
    assert table.loc["A", "team"] == "T1"


def test_team_offset_is_the_constructor_mean(practice):
    table = pace.practice_pace(practice, practice_fit(practice))
    team = pace.team_offset(table)
    assert team["A"] == team["B"] == pytest.approx((table.loc["A", "offset_s"] + table.loc["B", "offset_s"]) / 2)
    assert team["C"] == team["D"] == pytest.approx(0.0, abs=0.05)


def test_drivers_without_laps_take_the_team_mate_or_zero(practice):
    roster = pd.Series({**TEAMS, "E": "T3", "F": "T3"})
    no_a = practice[practice["driver"] != "A"]
    table = pace.practice_pace(no_a, practice_fit(no_a), roster=roster)
    assert list(table.index) == list(roster.index)
    assert table.loc["A", "source"] == "team" and table.loc["A", "n_laps"] == 0
    assert table.loc["A", "offset_s"] == table.loc["B", "offset_s"]
    assert table.loc["E", "source"] == table.loc["F", "source"] == "none"
    assert table.loc["E", "offset_s"] == table.loc["F", "offset_s"] == 0.0
    few = pace.practice_pace(practice, practice_fit(practice), min_laps=41)
    assert (few["source"] == "none").all() and (few["offset_s"] == 0.0).all()


def test_a_reserve_who_ran_the_car_counts_for_the_race_driver():
    laps = practice_laps()
    laps.loc[laps["Driver"] == "B", "Driver"] = "R"   # a reserve ran B's car under his own code
    f = practice_frame({"FP1": laps})
    table = pace.practice_pace(f, practice_fit(f), roster=pd.Series(TEAMS))
    assert "R" not in table.index
    assert table.loc["B", "source"] == "practice" and table.loc["B", "offset_s"] == pytest.approx(0.5, abs=0.05)
    assert table.attrs["notes"] == ["FP1: no B laps, took R laps (T1)"]
    assert stand_in({"A", "R"}, {"A", "B"}, "B") == "R"
    assert stand_in({"A", "R", "S"}, {"A", "B"}, "B") is None   # two candidates: nobody


def test_one_compound_session_and_an_unknown_compound():
    f = practice_frame({"FP2": practice_laps(stints=(("SOFT", 10),))})
    table = pace.practice_pace(f, practice_fit(f))
    assert table.loc["B", "offset_s"] == pytest.approx(0.5, abs=0.05)
    two = practice_frame({"FP2": practice_laps()})
    soft_only = LinearDegradation({"SOFT": (BASE["SOFT"] - 0.1, 0.1)})
    table = pace.practice_pace(two, soft_only)
    assert (table["n_laps"] == 10).all()  # the MEDIUM laps are left out
    assert table.loc["B", "offset_s"] == pytest.approx(0.5, abs=0.05)


def test_wet_session_needs_more_than_half_its_laps_wet():
    laps = practice_laps()
    end = laps["Time"].dt.total_seconds().max()
    t = np.arange(0.0, end + 60.0, 60.0)

    def feed(rain):
        return pd.DataFrame({"time_s": t, "rainfall": rain, "track_temp_c": 30.0, "air_temp_c": 20.0,
                             "wind_speed_ms": 1.0})

    assert pace.wet_session(laps, feed(np.ones(len(t), dtype=bool)))
    assert not pace.wet_session(laps, feed(np.zeros(len(t), dtype=bool)))
    assert not pace.wet_session(laps, feed(t < end * 0.3))   # rain early on, most laps dry
    assert not pace.wet_session(laps, pd.DataFrame(columns=["time_s", "rainfall"]))


def test_positions_with_a_pace_series_shifts_the_predicted_order():
    sim = Simulator(synthetic_frame())
    frame = three_driver_frame()
    base, _ = calibration.positions(sim, frame, RACE)
    assert base["pred_pos"].tolist() == [1, 2, 3]
    # AAA a second a lap slower: 50 s over the race, so BBB is ahead; CCC is not listed and gets 0
    shifted, _ = calibration.positions(sim, frame, RACE, pace=pd.Series({"AAA": 1.0, "BBB": 0.0}))
    assert shifted["pred_s"]["AAA"] == pytest.approx(base["pred_s"]["AAA"] + 50.0)
    assert shifted["pred_s"]["CCC"] == pytest.approx(base["pred_s"]["CCC"])
    assert shifted["pred_pos"].tolist() == [2, 1, 3]
    assert shifted["abs_err"].tolist() == [0, 0, 0]


def test_gate_table_rows():
    sim = Simulator(synthetic_frame())
    paces = {"none": None, "insample": pd.Series({"AAA": 1.0})}
    gate = pace.gate_table(sim, three_driver_frame(), RACE, None, paces)
    assert gate[["mode", "pace"]].values.tolist() == [
        ["simulated_order", "none"], ["simulated_order", "insample"], ["simulated_order", "grid"],
        ["vs_real", "none"], ["vs_real", "insample"], ["vs_real", "grid"]]
    assert gate.loc[1, "exact_rate"] == 1.0
