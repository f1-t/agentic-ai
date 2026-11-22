"""Pit-stop durations (SimConfig.pit_stops, pit_loss.lane_deltas / team_delta)."""
import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, Simulator, calibration, pit_loss
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_practice
from test_traffic_future_blindness import DRIVER, K, alter_others_after, cut, field


def _stops(lanes, teams, kinds=None):
    n = len(lanes)
    return pd.DataFrame({"driver": [f"D{i}" for i in range(n)], "team": teams, "lane_s": lanes,
                         "kind": kinds or ["green"] * n})


def test_lane_deltas_against_the_green_median():
    s = _stops([22.0, 23.0, 25.0, 30.0], ["A", "A", "B", "B"], ["green", "green", "green", "sc"])
    assert list(pit_loss.lane_deltas(s)) == [-1.0, 0.0, 2.0, 7.0]


def test_team_delta_is_shrunk_toward_zero():
    s = _stops([22.0, 22.0, 24.0, 24.0, 23.0], ["A", "A", "B", "B", "C"])
    d, n = pit_loss.team_delta(s, "B", prior_stops=3.0)
    assert n == 2 and d == pytest.approx(2 / 5 * 1.0)
    assert pit_loss.team_delta(s, "Z", 3.0) == (0.0, 0)


def _with_lanes(df):
    df = df.copy()
    df["pit_in_time_s"] = np.where(df["pit_in"].astype(bool), df["end_s"] - 4.0, np.nan)
    rng = np.random.default_rng(2)
    df["pit_out_time_s"] = np.where(df["pit_out"].astype(bool), df["start_s"] + 18.0 + rng.uniform(0, 3, len(df)), np.nan)
    return df


@pytest.fixture(scope="module")
def pair():
    df = _with_lanes(field())
    alt = alter_others_after(df)
    late = (alt["end_s"] > cut(df)) & (alt["driver"] != DRIVER)
    alt.loc[late, "pit_out_time_s"] += 3.0
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, K)
    return df, alt, snap


def _times(cfg, frame, snap):
    rest = snap.remaining_laps
    plans = [[["SOFT", 3], ["MEDIUM", rest - 3]], [["SOFT", 3], ["MEDIUM", 10], ["SOFT", rest - 13]]]
    return Simulator(frame, cfg, synthetic_practice()).simulate_batch(snap, plans, n_samples=2, seed=1).times


def test_belief_team_stop_ignores_stops_after_the_cut(pair):
    df, alt, snap = pair
    cfg = SimConfig.from_file("configs/sim/belief.yaml")
    cfg = cfg.model_copy(update={"pit_stops": cfg.pit_stops.model_copy(update={"team": True})})
    assert np.array_equal(_times(cfg, df, snap), _times(cfg, alt, snap), equal_nan=True)


def test_judge_team_stop_reads_the_whole_race_and_off_is_unchanged(pair):
    df, alt, snap = pair
    on = SimConfig(pit_stops={"team": True})
    assert not np.allclose(_times(on, df, snap), _times(on, alt, snap))   # the canary: the judge reads them
    assert np.array_equal(_times(SimConfig(), df, snap), _times(SimConfig(pit_stops={"team": False}), df, snap))
