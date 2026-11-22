"""No future information in anything a policy touches, with belief traffic on.

At snapshot lap K a policy may know positions, gaps, pace and tyres up to K, but not what
the other cars do after K. Every other car's laps after K are altered here (lap times, the
session clock, compounds, tyre ages, pit stops, track status, positions and gaps, and one
car retires), and the belief's traffic state, the tools' outputs and every policy's plan must
come out identical. A synthetic race, no cache, no model call.

R8 (the stint-limit rule) is off in this test: its limits are read off the whole race, a known
leak fixed on its own branch (fix/tyre-life-no-hindsight), not a traffic one.
"""
import json

import numpy as np
import pytest

from src.eval import decisions
from src.eval.policies import BASELINE_POLICIES, RULE_POLICIES
from src.eval.scenarios import Scenario
from src.sim import Simulator, agent_tools, calibration
from src.sim.data import gap_columns
from src.sim.traffic import GhostTraffic
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_practice
from test_sim_agent_tools import PROFILE, outputs
from test_sim_position import race_frame

K = 20
DRIVER = "CCC"
BELIEF = agent_tools.belief_config().model_copy(update={"stint_limit_margin_laps": None})
N = 8


def field():
    df = race_frame(("AAA", "T1", [("SOFT", 15), ("MEDIUM", 35)], -0.3),
                    ("BBB", "T2", [("SOFT", 10), ("MEDIUM", 40)], 0.2, (2, 2)),
                    (DRIVER, "T3", [("SOFT", 25), ("MEDIUM", 25)], 0.4, (3, 3)),
                    ("DDD", "T3", [("SOFT", 18), ("MEDIUM", 32)], 0.5, (4, 4)),
                    ("EEE", "T4", [("SOFT", 12), ("MEDIUM", 20), ("SOFT", 18)], 0.1, (5, 5)),
                    ("FFF", "T4", [("SOFT", 22), ("MEDIUM", 28)], 0.9, (6, 6)))
    df["lap_time_s"] += np.random.default_rng(0).normal(0.0, 0.3, len(df))
    # the session clock follows the noisy lap times
    for d, g in df.groupby("driver"):
        end = g["lap_time_s"].cumsum().to_numpy()
        df.loc[g.index, "end_s"], df.loc[g.index, "start_s"] = end, end - g["lap_time_s"].to_numpy()
    return gap_columns(df)


def alter_others_after(df, k, driver=DRIVER):
    """Every other car's laps after k altered in every column a race record has, and FFF retired."""
    df = df.copy()
    late = (df["lap"] > k) & (df["driver"] != driver)
    rng = np.random.default_rng(1)
    df.loc[late, "lap_time_s"] += rng.normal(2.0, 1.5, late.sum())
    df.loc[late, ["start_s", "end_s"]] += 41.0 + rng.uniform(0, 5, late.sum())[:, None]
    df.loc[late, "compound"] = np.where(df.loc[late, "compound"] == "SOFT", "MEDIUM", "SOFT")
    df.loc[late, "tyre_age"] = df.loc[late, "tyre_age"] + 7
    df.loc[late, "pit_in"] = ~df.loc[late, "pit_in"].astype(bool) & (df.loc[late, "lap"] % 9 == 0)
    df.loc[late, "pit_out"] = df.loc[late, "lap"] % 9 == 1
    df.loc[late, "track_status"] = np.where(df.loc[late, "lap"] % 7 == 0, "4", "1")
    df.loc[late, "position"] = 99.0
    df.loc[late, ["gap_ahead_s", "gap_behind_s"]] += 0.7
    df.loc[late, "free_air"] = ~df.loc[late, "free_air"].astype(bool)
    return df[~((df["driver"] == "FFF") & (df["lap"] > k + 5))]


@pytest.fixture(scope="module")
def pair():
    df = field()
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, K)
    return df, alter_others_after(df, K), snap


def belief(df):
    return Simulator(df, BELIEF, synthetic_practice())


def test_the_belief_runs_projected_traffic(pair):
    df, _, snap = pair
    assert BELIEF.traffic.name == "ghost" and BELIEF.traffic.params["source"] == "projected"
    b = belief(df).simulate_batch(snap, [calibration.remaining_plan(df, DRIVER, K)], n_samples=N)
    assert b.traffic_on and np.isfinite(b.final_pos).all()


def test_traffic_state_ignores_the_other_cars_after_k(pair):
    df, alt, snap = pair
    laps = np.arange(K + 1, SYN_RACE.total_laps + 1)
    model = belief(df).fitted_for(snap).degradation
    tr = GhostTraffic(**BELIEF.traffic.params)
    a, b = (tr.prepare(snap, laps, f, DRIVER, ref_lap_s=90.0, model=model) for f in (df, alt))
    assert a.table.ghosts == b.table.ghosts == ["AAA", "BBB", "DDD", "EEE", "FFF"]   # FFF has not retired yet
    for va, vb in zip(a.variants, b.variants):
        for name in ("G", "G_prev", "g", "in_lap", "out_lap"):
            assert np.array_equal(getattr(va.table, name), getattr(vb.table, name), equal_nan=True), name
    assert (a.own_offset_s, a.dirty_air_s, a.anchor) == (b.own_offset_s, b.dirty_air_s, b.anchor)


def test_the_recorded_source_would_read_them(pair):
    # the canary bites: the world's recorded ghosts do read the laps after k
    df, alt, snap = pair
    laps = np.arange(K + 1, SYN_RACE.total_laps + 1)
    a, b = (GhostTraffic().prepare(snap, laps, f, DRIVER) for f in (df, alt))
    assert not np.array_equal(a.table.G, b.table.G, equal_nan=True)


def test_batches_and_positions_ignore_the_other_cars_after_k(pair):
    df, alt, snap = pair
    rest = snap.remaining_laps
    plans = [[["SOFT", 5], ["MEDIUM", rest - 5]], [["SOFT", 1], ["MEDIUM", rest - 1]], [["SOFT", rest]]]
    a, b = (belief(f).simulate_batch(snap, plans, seed=3, n_samples=N, rules=["lap_sum", "known_compounds"])
            for f in (df, alt))
    for name in ("times", "traffic_s", "passes", "final_pos", "rejoin_pos", "pit_window_cars"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    assert a.support_warnings == b.support_warnings


def test_tool_outputs_ignore_the_other_cars_after_k(pair):
    df, alt, snap = pair
    a, b = outputs(belief(df), snap), outputs(belief(alt), snap)
    assert a == b
    got = json.loads(a["compare_plans"])
    assert got["traffic_basis"] and got["traffic"][0]["final_position"] is not None
    assert json.loads(a["model_report"])["traffic"]["modelled"] is True
    assert json.loads(a["evaluate_pit"])["traffic"]["action"]["expected_rejoin_position"] is not None


def test_policy_plans_ignore_the_other_cars_after_k(pair):
    df, alt, snap = pair
    sc = Scenario(scenario_id=f"syn:{DRIVER}:{K}", race_id="syn", driver=DRIVER, lap=K, snapshot=snap,
                  actual_stints=[("SOFT", 25), ("MEDIUM", 25)])
    sims = belief(df), belief(alt)
    opt = [decisions.belief_optimum(snap, s, 2, n_samples=N).as_pairs() for s in sims]
    assert opt[0] == opt[1]
    for name, pol in {**BASELINE_POLICIES, **RULE_POLICIES}.items():
        if name == "actual_strategy":   # replays the real strategy: the future by definition
            continue
        got = []
        for s in sims:
            try:
                d = pol(sc, s, 2)
                got.append(d.plan.as_pairs() if hasattr(d.plan, "as_pairs") else d.plan)
            except Exception as exc:   # PolicyUnavailable and the like: the same either way
                got.append(type(exc).__name__)
        assert got[0] == got[1], name
