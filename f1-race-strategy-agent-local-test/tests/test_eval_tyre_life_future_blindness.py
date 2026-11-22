"""Tyre life without hindsight (docs/eval.md "Information contract", docs/rules.md R8 retired).

A policy at (race, lap L) may use only the weekend's practice, qualifying and sprint, race laps
up to L, the published rules and priors fixed in config. Here every lap after L is truncated
away or scrambled, and nothing a policy sees may change: the belief's cliff ages, every agent
tool output and error string, the belief optimizer's plan, the continuation of a PIT and a
STAY_OUT decision, and every baseline plan (actual_strategy excepted: it replays the team's real
strategy by definition). The same check with the legacy R8 setting bites, so the test can see
the leak it guards against. The world (the scorer) is allowed the whole race, and its cliff
moves when the race changes.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.baselines.domain import PolicyUnavailable
from src.eval import decisions
from src.eval.policies import BASELINE_POLICIES, RULE_POLICIES
from src.eval.scenarios import Scenario
from src.sim import SimConfig, Simulator, agent_tools, calibration, cliff
from src.sim.data import CACHE_DIR
from src.sim.optimizer_bridge import optimize_batch
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_practice
from test_sim_agent_tools import PROFILE, syn_frame


def outputs(sim, snap):
    """Every agent tool's output for one snapshot, as JSON text."""
    kw = {"sim": sim, "profile": PROFILE}
    rest = snap.remaining_laps
    out = {
        "race_state": agent_tools.race_state(snap, config=sim.config),
        "model_report": agent_tools.model_report(snap, ages={"MEDIUM": 30}, **kw),
        "pit_loss": agent_tools.pit_loss(snap, **kw),
        "simulate_plan": agent_tools.simulate_plan(snap, [["SOFT", 5], ["MEDIUM", rest - 5]], **kw),
        "compare_plans": agent_tools.compare_plans(
            snap, [[["SOFT", 5], ["MEDIUM", rest - 5]], {"action": "stay_out", "laps": 2},
                   {"action": "pit", "compound": "MEDIUM"}], **kw),
        "evaluate_pit": agent_tools.evaluate_action(snap, {"action": "pit", "compound": "MEDIUM"}, **kw),
        "evaluate_stay": agent_tools.evaluate_action(snap, {"action": "stay", "laps": 3}, **kw),
        "evaluate_long_stay": agent_tools.evaluate_action(snap, {"action": "stay", "laps": 25}, **kw),
    }
    return {name: json.dumps(v, sort_keys=True, allow_nan=False, default=str) for name, v in out.items()}

BELIEF = agent_tools.belief_config()
LEGACY = {"stint_limit_margin_laps": 3, "wear_cliff": {"enabled": False}}
DRIVER = "CCC"
SNAPS = (8, 20, 30)
HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))


def _late(df, k, driver=None):
    """The rows not yet visible when `driver` completes lap k: by session time (end_s after that
    moment) when the frame has it, else by lap number."""
    if driver is not None and "end_s" in df and "driver" in df:
        t = df.loc[(df["driver"] == driver) & (df["lap"] == k), "end_s"]
        if len(t) and pd.notna(t.iloc[0]):
            return (pd.to_numeric(df["end_s"], errors="coerce") > float(t.iloc[0])).to_numpy()
    return (df["lap"] > k).to_numpy()


def truncate_after(df, k, driver=None):
    return df[~_late(df, k, driver)].copy()


def scramble_after(df, k, seed=1, driver=None):
    """Every lap not yet visible at the snapshot rewritten: tyre ages stretched far past any real
    stint, compounds swapped, lap times shuffled and shifted."""
    df = df.copy()
    late = _late(df, k, driver)
    rng = np.random.default_rng(seed)
    df.loc[late, "tyre_age"] = df.loc[late, "tyre_age"].to_numpy() * 3 + 40
    swap = {"SOFT": "MEDIUM", "MEDIUM": "SOFT"}
    df.loc[late, "compound"] = df.loc[late, "compound"].map(lambda c: swap.get(c, c))
    df.loc[late, "lap_time_s"] = rng.permutation(df.loc[late, "lap_time_s"].to_numpy()) + 2.5
    return df


def _scenario(snap, k):
    return Scenario(scenario_id=f"syn:{DRIVER}:{k:02d}", race_id="syn", driver=DRIVER, lap=k, snapshot=snap,
                    actual_stints=[("SOFT", 25), ("MEDIUM", 25)])


def policy_view(sim, snap, k, n=8):
    """Everything a policy can get at the snapshot, as JSON text."""
    out = {"cliffs": {c: vars(cl) for c, cl in sorted(sim.cliffs_for(snap).items())}}
    out["tools"] = outputs(sim, snap)
    out["belief_optimum"] = decisions.belief_optimum(snap, sim, 2, n_samples=n).as_pairs()
    for name, sub in (("pit", {"decision": "PIT", "compound": "MEDIUM", "confidence": 0.6}),
                      ("stay", {"decision": "STAY_OUT", "reassess_after_laps": 2, "confidence": 0.6})):
        out[f"continuation_{name}"] = decisions.to_plan(snap, sub, sim, 2, n_samples=n).as_pairs()
    sc = _scenario(snap, k)
    for name, pol in {**BASELINE_POLICIES, **RULE_POLICIES}.items():
        if name == "actual_strategy":
            continue
        try:
            d = pol(sc, sim, 2)
        except PolicyUnavailable as exc:
            out[name] = ["unavailable", str(exc)]
            continue
        out[name] = [d.plan.as_pairs() if hasattr(d.plan, "as_pairs") else d.plan, d.meta]
    # a plan past every tyre life: the belief prices it, it is never an error
    rest = snap.remaining_laps
    res = sim.simulate(snap, [["SOFT", rest]], n_samples=n, seed=agent_tools.SEED)
    out["overlong"] = [res.valid, res.errors]
    return json.dumps(out, sort_keys=True, default=str)


def _sim(df, config):
    return Simulator(df, config, synthetic_practice())


@pytest.mark.parametrize("k", SNAPS)
@pytest.mark.parametrize("perturb", [truncate_after, scramble_after])
def test_nothing_a_policy_sees_reads_a_lap_after_the_snapshot(k, perturb):
    df = syn_frame()
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, k)
    assert policy_view(_sim(df, BELIEF), snap, k) == policy_view(_sim(perturb(df, k), BELIEF), snap, k)


def test_the_legacy_stint_limit_is_the_leak_the_test_catches():
    """R8 restored (legacy setting): the policy view changes when the laps after the snapshot do."""
    df, k = syn_frame(), 20
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, k)
    legacy = BELIEF.model_copy(update={"stint_limit_margin_laps": 3,
                                       "wear_cliff": BELIEF.wear_cliff.model_copy(update={"enabled": False})})
    a, b = _sim(df, legacy), _sim(scramble_after(df, k), legacy)
    assert a.stint_limits != b.stint_limits
    assert policy_view(a, snap, k) != policy_view(b, snap, k)


def test_the_default_belief_carries_no_whole_race_limit():
    sim = _sim(syn_frame(), BELIEF)
    assert sim.config.stint_limit_margin_laps is None and sim.stint_limits == {}


def test_the_world_cliff_reads_the_whole_race():
    """The scorer may know the whole race: its cliff ages move with the laps after the snapshot."""
    df, k = syn_frame(), 20
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, k)
    world = SimConfig()
    assert _sim(df, world).cliffs_for(snap) != _sim(truncate_after(df, k), world).cliffs_for(snap)


def test_belief_cliff_never_below_what_was_seen_and_grows_with_the_race():
    df = syn_frame()
    sim = _sim(df, BELIEF)
    ages = []
    for k in (5, 15, 30, 45):
        snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, k)
        cl = sim.cliffs_for(snap)
        # the rows visible when the driver completes lap k (the fit window's time cut)
        seen = sim.fitted_for(snap).window_rows(snap, k).groupby("compound")["tyre_age"].max()
        for c, a in seen.items():
            assert cl[c].lower >= a and cl[c].age >= a
        ages.append(cl["MEDIUM"].lower)
    assert ages == sorted(ages)


def test_tools_report_tyre_life_as_an_estimate():
    df, k = syn_frame(), 20
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, k)
    rep = agent_tools.model_report(snap, sim=_sim(df, BELIEF), profile=PROFILE)
    life = rep["compounds"]["SOFT"]["tyre_life"]
    assert "longest run seen so far on this compound" in life["note"] and "uncertain" in life["note"]
    sim = _sim(df, BELIEF)
    rows = sim.fitted_for(snap).window_rows(snap, k)
    assert life["longest_run_seen_laps"] == rows.loc[rows["compound"] == "SOFT", "tyre_age"].max()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CACHE, reason="needs the committed FastF1 cache")
@pytest.mark.parametrize("race_id", ["2021_04_spain", "2023_07_spain"])
def test_development_races_belief_is_blind_to_the_future(race_id):
    from src.sim import cross_race
    from src.sim.compute import load_profile
    data = cross_race.load_race(race_id, load_profile())
    driver = cross_race.field_drivers(data.frame, data.race.total_laps)[0]
    for k in (10, 25, 40):
        snap = calibration.snapshot_at(data.frame, data.race, driver, k)
        views = []
        for frame in (data.frame, truncate_after(data.frame, k, driver=driver),
                      scramble_after(data.frame, k, driver=driver)):
            sim = Simulator(frame, BELIEF, data.practice, qualifying=data.qualifying)
            cl = {c: vars(x) for c, x in sorted(sim.cliffs_for(snap).items())}
            plan = optimize_batch(snap, sim.fitted_for(snap), seed=agent_tools.SEED, n_samples=4, max_stops=2).plan
            views.append(json.dumps([cl, plan.as_pairs()], sort_keys=True))
        assert views[0] == views[1] == views[2], (race_id, k)


# --- the belief's tyre-life prior: an explicit fallback chain that never fails ------------------
HISTORY = {"races": {
    "2020_01_alpha": {"circuit": "alpha", "total_laps": 50, "oldest_age": {"SOFT": 20.0, "MEDIUM": 30.0, "HARD": 40.0}},
    "2021_01_beta": {"circuit": "beta", "total_laps": 60, "oldest_age": {"SOFT": 24.0, "MEDIUM": 30.0}},
    "2023_01_alpha": {"circuit": "alpha", "total_laps": 50, "oldest_age": {"SOFT": 99.0}},   # later: never read for 2022
}}


def _wc(**kw):   # the stub prior on a synthetic history table
    return SimConfig().wear_cliff.model_copy(update={"prior_source": "stub", "scan_cache": False,
                                                     "history": HISTORY, **kw})


def test_the_default_prior_is_the_shared_history_model_and_never_fails():
    """prior_source tyre_history: every dry compound gets an edge at any race id, a synthetic one
    (no date) through the stub's constant tier."""
    wc = SimConfig().wear_cliff
    assert wc.prior_source == "tyre_history"
    for race_id in ("2023_07_spain", "2018_03_nowhere", "syn"):
        prior = cliff.history_prior(race_id, 60, wc)
        assert set(prior) >= {"SOFT", "MEDIUM", "HARD"}
        assert all(e > 0 and sd > 0 for e, sd, _ in prior.values())
    assert cliff.history_prior("syn", 60, wc)["SOFT"][2] == "config_fraction"
    assert cliff.history_prior("2023_07_spain", 60, wc)["SOFT"][2].startswith("tyre_history_tier")


def test_prior_tiers_fall_back_in_order_with_widening_sd():
    sd = SimConfig().wear_cliff.prior_sd_fraction
    assert [sd[t] for t in cliff.PRIOR_TIERS] == sorted(sd[t] for t in cliff.PRIOR_TIERS)   # wider down the chain
    same = cliff.life_prior("2022_05_alpha", 60, _wc())
    assert {c: s for c, (_, _, s) in same.items()} == dict.fromkeys(["SOFT", "MEDIUM", "HARD"],
                                                                    "circuit_history:2020_01_alpha")
    assert same["SOFT"][0] == pytest.approx(20.0 * 60 / 50) and same["SOFT"][1] == pytest.approx(sd["circuit_history"] * 60)
    other = cliff.life_prior("2022_05_gamma", 60, _wc())   # a circuit with no history: earlier races anywhere
    assert {s for _, _, s in other.values()} == {"history_fraction"}
    assert other["SOFT"][0] == pytest.approx(60 * (20 / 50 + 24 / 60) / 2) and other["SOFT"][1] > same["SOFT"][1]
    # a compound no earlier race ran: a neighbour's prior times the as-of life ratio
    rows = [{"race_id": "2021_01_beta", "pair": "MEDIUM-HARD", "thin": False, "life_ratio": 0.75}]
    hist = {"races": {"2021_01_beta": HISTORY["races"]["2021_01_beta"]}}
    ratio = cliff.life_prior("2022_05_gamma", 60, _wc(history=hist, life_ratio_rows=rows))
    assert ratio["HARD"][2] == "life_ratio_as_of:MEDIUM" and ratio["HARD"][0] == pytest.approx(30.0 / 0.75)
    assert ratio["HARD"][1] > other["SOFT"][1]


@pytest.mark.parametrize("race_id", ["2018_03_nowhere", "syn"])
def test_a_circuit_with_no_history_gets_the_constant_prior_with_the_widest_sd(race_id):
    """No earlier race at all (or an id with no date): the documented constant, never a failure
    and never a whole-race quantity."""
    wc = _wc()
    prior = cliff.life_prior(race_id, 60, wc)
    widest = max(wc.prior_sd_fraction.values()) * 60
    for c, f in wc.prior_life_fraction.items():
        assert prior[c] == (pytest.approx(f * 60), pytest.approx(widest), "config_fraction")


def test_the_belief_builds_at_a_circuit_with_no_history():
    """A synthetic race at a circuit nobody raced before: the belief builds, plans, and reports the
    tier and its widest uncertainty to the tools; the prior reads nothing of this race."""
    df, k = syn_frame(), 5
    race = SYN_RACE.model_copy(update={"race_id": "2018_03_nowhere"})
    snap = calibration.snapshot_at(df, race, DRIVER, k)
    cfg = BELIEF.model_copy(update={"wear_cliff": _wc()})
    sim = _sim(df, cfg)
    cl = sim.cliffs_for(snap)
    assert cl["HARD"].source == "config_fraction" and cl["HARD"].lower == 0.0
    assert cl["HARD"].prior_sd == pytest.approx(max(cfg.wear_cliff.prior_sd_fraction.values()) * race.total_laps)
    assert decisions.belief_optimum(snap, sim, 2, n_samples=4) is not None
    rep = agent_tools.model_report(snap, sim=sim, profile=PROFILE)
    assert all("tyre_life" in e for e in rep["compounds"].values() if e.get("modelled"))


@pytest.mark.skipif(not HAS_CACHE, reason="needs the committed FastF1 cache")
def test_the_cache_scan_lists_earlier_races_and_never_a_held_out_one():
    """The run-time history reads whatever earlier seasons are cached, by date; held-out races
    (data/eval_set.json) are left out before anything is loaded."""
    from src.sim.cross_race import held_out_races
    got = cliff._scan_candidates("2025_12_x")
    assert got and not set(got) & set(held_out_races())
    assert "2024_07_emilia_romagna" in got and "2025_23_qatar" not in got   # strictly earlier only
