"""Pit loss without hindsight (src/sim/pit_loss.py; docs/eval.md "Pit loss and track status")."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.contracts import Snapshot
from src.sim import agent_tools, calibration, pit_loss
from src.sim.compute import load_profile
from src.sim.config import PitLossModel, SimConfig
from src.sim.data import CACHE_DIR
from src.sim.engine import Simulator

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))
LOCAL = load_profile("local")
PROFILE = LOCAL.model_copy(update={"sim": LOCAL.sim.model_copy(update={"n_samples": 20})})
BELIEF = agent_tools.belief_config()
L = 30


# --- the prior ---------------------------------------------------------------------------------
def table(**races):
    # the table only: no run-time cache discovery, so nothing is loaded
    return PitLossModel(prior={"races": races}, cache_lookup=False)


TABLE = {"2021_04_spain": {"pit_loss_s": 23.0}, "2019_10_united_kingdom": {"pit_loss_s": 25.0},
         "2022_02_saudi_arabia": {"pit_loss_s": 21.0}, "2023_07_spain": {"pit_loss_s": 30.0}}


def test_prior_fallback_chain_tiers_and_widths():
    cfg = table(**TABLE)
    same = pit_loss.pit_loss_prior("2023_07_spain", cfg, held=())
    assert (same["mean_s"], same["sd_s"], same["source"]) == (23.0, cfg.same_circuit_sd_s, "same_circuit_asof")
    # a circuit with no history (a layout used once): earlier races elsewhere, a wider interval
    cross = pit_loss.pit_loss_prior("2023_01_nowhere", cfg, held=())
    assert cross["source"] == "cross_circuit_asof"
    assert cross["races"] == ["2019_10_united_kingdom", "2021_04_spain", "2022_02_saudi_arabia"]
    assert cross["mean_s"] == pytest.approx(23.0) and cross["sd_s"] == pytest.approx(2.0)
    # nothing earlier: the constant, the widest; never a later race, never the race itself
    const = pit_loss.pit_loss_prior("2019_10_united_kingdom", cfg, held=())
    assert (const["mean_s"], const["sd_s"], const["source"]) == (cfg.default_s, cfg.default_sd_s, "constant")
    assert same["sd_s"] < cross["sd_s"] < const["sd_s"]
    assert pit_loss.pit_loss_prior("syn", cfg, held=())["source"] == "constant"   # no time order


def test_as_of_estimate_exists_on_a_circuit_with_no_history_and_learns_from_its_stops():
    """A synthetic race at an unseen circuit: the cross-circuit prior, then this race's stops."""
    from test_sim_agent_tools import syn_frame
    cfg = table(**TABLE)
    df = syn_frame()   # AAA stops after lap 15, BBB after 10, CCC after 25
    early = pit_loss.estimate(df, "2023_01_nowhere", "CCC", 5, None, cfg)
    assert early["prior_source"] == "cross_circuit_asof" and early["n_green_stops"] == 0
    assert early["pit_loss_s"] == pytest.approx(23.0) and early["sd_s"] > cfg.min_cross_sd_s
    none = pit_loss.estimate(df, "2018_01_nowhere", "CCC", 5, None, cfg)
    assert none["prior_source"] == "constant" and none["sd_s"] > early["sd_s"]
    for k in ("sc", "vsc"):   # the SC / VSC factors exist too, from their constant prior
        assert none[f"{k}_factor"] == getattr(cfg, f"default_{k}_factor")
    model = Simulator(df, SimConfig()).degradation
    late = pit_loss.estimate(df, "2018_01_nowhere", "CCC", 30, model, cfg)
    assert late["n_green_stops"] == 3 and late["sd_s"] < none["sd_s"]
    assert late["pit_loss_s"] != none["pit_loss_s"]


def test_world_value_is_the_median_green_stop_else_the_prior():
    from test_sim_agent_tools import syn_frame
    cfg = table(**TABLE)
    df = syn_frame()
    model = Simulator(df, SimConfig()).degradation
    out = pit_loss.whole_race(df, model, "2023_01_nowhere", cfg)
    assert out["source"] == "green_median" and out["n_green_stops"] == 3
    thin = pit_loss.whole_race(df[df["driver"] != "AAA"], model, "2023_01_nowhere", cfg)   # 2 stops
    assert thin["source"] == "prior" and thin["pit_loss_s"] == pytest.approx(23.0)


def test_prior_pool_reads_the_cache_and_keeps_held_out_races_out_of_development_runs():
    cfg = PitLossModel(prior={"races": TABLE})
    cached = ["2019_06_monaco", "2024_11_austria", "2025_11_austria", "2023_01_bahrain"]
    held = ["2019_06_monaco", "2024_11_austria", "2025_11_austria"]
    pool = pit_loss.prior_pool("2023_07_spain", cfg, cached=cached, held=held)
    assert set(pool) == {*TABLE, "2023_01_bahrain"} - {"2023_07_spain"}
    # the held-out run: an earlier held-out race at the circuit is in the pool, the race is not
    pool = pit_loss.prior_pool("2025_11_austria", cfg, cached=cached, held=held)
    assert "2024_11_austria" in pool and "2025_11_austria" not in pool
    assert pit_loss.prior_pool("2023_07_spain", cfg.model_copy(update={"cache_lookup": False}),
                               cached=cached, held=held) == sorted(set(TABLE) - {"2023_07_spain"})


def test_committed_prior_holds_calibration_races_only():
    # the nineteen calibration races sampled into dev20 (configs/eval/rerun_sets.yaml), 2022 Austria
    # included since it loads; never a held-out race
    from src.eval.rerun import set_spec
    from src.eval.scenarios import load_eval_set
    races = set(PitLossModel().prior["races"])
    assert races == set(set_spec("dev20")["races"]) and not races & set(load_eval_set()[0])


# --- the as-of estimate on a real race -----------------------------------------------------
@pytest.fixture(scope="module")
def spain():
    return agent_tools._race_data("2021_04_spain", LOCAL)


def alter_after(frame, driver, lap):
    """Every stop and lap that ends after `driver` ended lap `lap`, in a lap past it, changed:
    stops moved, added and slowed, in- and out-lap times changed, an SC added."""
    t = pit_loss.snapshot_time(frame, driver, lap)
    df = frame.copy()
    late = (df["lap"] > lap) & (df["end_s"] > t)
    df.loc[late, "pit_in"] = ~df.loc[late, "pit_in"].astype(bool) & (df.loc[late, "lap"] % 7 == 0)
    df.loc[late, "pit_out"] = df.loc[late, "lap"] % 7 == 1
    df.loc[late, "lap_time_s"] = df.loc[late, "lap_time_s"] + np.where(df.loc[late, "pit_in"], 40.0, 3.0)
    df.loc[late & (df["lap"] % 11 == 0), "track_status"] = "4"
    return df


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_altering_every_stop_after_the_snapshot_changes_nothing_a_policy_sees(spain):
    snap = calibration.snapshot_at(spain.frame, spain.race, "HAM", L)
    a = Simulator(spain.frame, BELIEF, spain.practice, qualifying=spain.qualifying)
    b = Simulator(alter_after(spain.frame, "HAM", L), BELIEF, spain.practice, qualifying=spain.qualifying)
    pa, pb = (s.fitted_for(snap).pit_loss_inputs(snap) for s in (a, b))
    assert pa == pb and pa["source"] == "asof" and pa["n_green_stops"] > 0
    assert pa["pit_loss_s"] != spain.race.pit_loss_s
    kw = {"profile": PROFILE}
    for fn, args in ((agent_tools.pit_loss, ()), (agent_tools.simulate_plan, ([["SOFT", 10], ["MEDIUM", 26]],)),
                     (agent_tools.evaluate_action, ({"action": "pit", "compound": "MEDIUM"},))):
        assert json.dumps(fn(snap, *args, sim=a, **kw), sort_keys=True) == \
            json.dumps(fn(snap, *args, sim=b, **kw), sort_keys=True), fn.__name__
    from src.eval import decisions
    plans = [decisions.belief_optimum(snap, s, 2, seed=0, n_samples=20).as_pairs() for s in (a, b)]
    assert plans[0] == plans[1]


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_the_estimate_moves_with_stops_already_made(spain):
    """The canary bites: slowing the stops made before the snapshot moves the estimate."""
    snap = calibration.snapshot_at(spain.frame, spain.race, "HAM", L)
    df = spain.frame.copy()
    t = pit_loss.snapshot_time(df, "HAM", L)
    early = df["pit_out"].astype(bool) & (df["end_s"] <= t)
    df.loc[early, "lap_time_s"] += 4.0
    a = Simulator(spain.frame, BELIEF, spain.practice, qualifying=spain.qualifying)
    b = Simulator(df, BELIEF, spain.practice, qualifying=spain.qualifying)
    assert b.fitted_for(snap).pit_loss_inputs(snap)["pit_loss_s"] > a.fitted_for(snap).pit_loss_inputs(snap)["pit_loss_s"] + 1


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_world_takes_the_whole_race_value_and_policies_the_belief_one(spain):
    from src.eval.scenarios import Scenario as EvalScenario
    from src.eval.score import policy_scenario
    snap = calibration.snapshot_at(spain.frame, spain.race, "HAM", L)
    world = Simulator(spain.frame, SimConfig(), spain.practice, pit_loss_s=spain.race.pit_loss_s)
    odd = snap.model_copy(update={"race": snap.race.model_copy(update={"pit_loss_s": 99.0})})
    assert world.pit_loss_inputs(odd)["pit_loss_s"] == spain.race.pit_loss_s
    belief = Simulator(spain.frame, BELIEF, spain.practice, qualifying=spain.qualifying)
    sc = EvalScenario(scenario_id="x", race_id="2021_04_spain", driver="HAM", lap=L, snapshot=snap,
                      actual_stints=[("SOFT", 28)])
    seen = policy_scenario(sc, belief)
    assert seen.snapshot.race.pit_loss_s == pytest.approx(belief.fitted_for(snap).pit_loss_inputs(snap)["pit_loss_s"],
                                                          abs=1e-3)
    assert seen.snapshot.race.pit_loss_s != spain.race.pit_loss_s
    assert sc.snapshot.race.pit_loss_s == calibration.pre_race_pit_loss("2021_04_spain")   # the snapshot: the prior
    # a continuation's advanced snapshot reads no stop beyond the fit's lap
    fit = belief.fitted_for(snap)
    later = snap.model_copy(update={"laps_completed": L + 5, "tyre_age": snap.tyre_age + 5})
    assert fit.pit_loss_inputs(later)["pit_loss_s"] == fit.pit_loss_inputs(snap)["pit_loss_s"]


def test_pit_loss_tool_under_a_safety_car_is_compact_and_as_of():
    from test_sim_agent_tools import K, syn_frame, syn_sim, syn_snap
    df = syn_frame()
    snap = syn_snap(df).model_copy(update={"track_status": "sc", "sc_laps": [K - 1, K]})
    out = agent_tools.pit_loss(snap, sim=syn_sim(df), profile=PROFILE)
    assert out["source"] == "asof" and out["factor_source"] == "asof" and out["track_status"] == "sc"
    assert out["neutralised_for_laps"] == 2 and out["pit_loss_now_factor"] == out["sc_pit_loss_factor"]
    assert out["basis"]["prior_source"] == "constant"     # a synthetic race: no time order
    o = out["outlook"]["next_5_laps"]["sc"]
    assert 0 < o["low"] < o["median"] < o["high"] < 1
    json.dumps(out, allow_nan=False)
