"""SPEC 0015: src/sim/agent_tools.py, the simulator behind the agent's tools.

The information-boundary canary and the unit tests run on a synthetic race; the Spain
tests (HAM lap 20, the end-to-end agent run) on the committed cache.
"""
import json
import os

import numpy as np
import pytest

from src import tools
from src.agent import run_agent
from src.contracts import StintRecord, TyreSet
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.sim import SimConfig, Simulator, agent_tools, calibration
from src.sim.compute import load_profile
from src.sim.config import ComponentSpec
from src.sim.data import CACHE_DIR, gap_columns
from src.snapshot import RaceSnapshot
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_practice
from test_sim_position import race_frame

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))
K = 20
DRIVER = "CCC"   # on SOFT, 20 laps old at K, stops at 25
LOCAL = load_profile("local")
PROFILE = LOCAL.model_copy(update={"sim": LOCAL.sim.model_copy(update={"n_samples": 20})})
BELIEF = agent_tools.belief_config()


def syn_frame():
    df = race_frame(("AAA", "T1", [("SOFT", 15), ("MEDIUM", 35)]),
                    ("BBB", "T2", [("SOFT", 10), ("MEDIUM", 40)], 0.2, (2, 2)),
                    (DRIVER, "T3", [("SOFT", 25), ("MEDIUM", 25)], 0.4, (3, 3)))
    df["lap_time_s"] += np.random.default_rng(0).normal(0.0, 0.3, len(df))
    return gap_columns(df)


def perturb_after(df, k):
    """Every car's laps after k changed: lap times, the session clock, positions and gaps."""
    df = df.copy()
    late = df["lap"] > k
    df.loc[late, "lap_time_s"] += 1.3
    df.loc[late, ["start_s", "end_s"]] += 37.0
    df.loc[late, "position"] = 99.0
    df.loc[late, ["gap_ahead_s", "gap_behind_s"]] += 0.7
    return df


def syn_sim(df, config=BELIEF):
    return Simulator(df, config, synthetic_practice())


def syn_snap(df=None, k=K):
    return calibration.snapshot_at(syn_frame() if df is None else df, SYN_RACE, DRIVER, k)


def outputs(sim, snap):
    """Every function's output for one snapshot, as JSON text."""
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
        "recommendation_fields": agent_tools.recommendation_fields(snap, {"decision": "PIT", "confidence": 0.7}),
    }
    return {name: json.dumps(v, sort_keys=True, allow_nan=False) for name, v in out.items()}


# --- the information-boundary canary -----------------------------------------------------
def test_canary_outputs_ignore_every_lap_after_the_snapshot():
    df = syn_frame()
    snap = syn_snap(df)
    a = outputs(syn_sim(df), snap)
    b = outputs(syn_sim(perturb_after(df, K)), snap)
    assert a == b
    got = {name: json.loads(v) for name, v in a.items()}
    assert got["simulate_plan"]["valid"] and got["compare_plans"]["valid"] == [True, True, True]
    assert all(got[e]["gap_mean_s"] is not None and got[e]["alternative"]["valid"]
               for e in ("evaluate_pit", "evaluate_stay"))


def test_canary_bites_with_the_full_window():
    df = syn_frame()
    snap = syn_snap(df)
    full = BELIEF.model_copy(update={"fit_window": "full"})
    a = outputs(syn_sim(df, full), snap)
    b = outputs(syn_sim(perturb_after(df, K), full), snap)
    for name in ("model_report", "simulate_plan", "compare_plans", "evaluate_pit", "evaluate_stay"):
        assert a[name] != b[name], name


# --- per function ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def syn():
    df = syn_frame()
    return syn_sim(df), syn_snap(df)


def test_race_state_dumps_the_snapshot_with_legal_compounds(syn):
    sim, snap = syn
    out = agent_tools.race_state(snap, config=sim.config)
    assert out["remaining_laps"] == SYN_RACE.total_laps - K and out["laps_completed"] == K
    assert out["legal_compounds"] == ["SOFT", "MEDIUM"]   # race.compounds when unconstrained
    sets = [TyreSet(compound="HARD", source="assumed"), TyreSet(compound="MEDIUM", laps_run=2)]
    with_sets = snap.model_copy(update={"sets_available": sets})
    assert agent_tools.race_state(with_sets, config=sim.config)["legal_compounds"] == ["HARD", "MEDIUM"]
    observed = sim.config.model_copy(update={"inventory_sources": ["observed"]})
    assert agent_tools.race_state(with_sets, config=observed)["legal_compounds"] == ["MEDIUM"]


def test_model_report_adds_intervals_bound_and_correlation(syn):
    sim, snap = syn
    out = agent_tools.model_report(snap, sim=sim, profile=PROFILE)
    json.dumps(out, allow_nan=False)
    report = sim.model_report(snap)
    assert out["flags"] == report.flags and out["fit_window"] == "to_snapshot"
    for c in report.modelled:
        e = out["compounds"][c]
        wear = report.compounds[c].wear_s_per_lap
        assert e["point_estimate"]["deg_per_lap_s"] == wear.value
        d = e["deg_per_lap_s"]
        assert d["low"] <= d["median"] <= d["high"]                                   # always ordered
        assert e["interval_basis"]["deg_per_lap_s"] == agent_tools.BASIS_DRAWS     # belief draws on
        assert e["deg_per_lap_s"]["low"] >= 0.0                                     # the draws keep the bound
        base = report.compounds[c].base_s
        assert e["base_s_interval"] == pytest.approx(
            {"low": base.value - 1.64 * base.sd, "median": base.value, "high": base.value + 1.64 * base.sd})
        assert e["wear_at_bound"] == (f"wear_at_bound:{c}" in report.flags)
        assert e["fuel_wear_corr"] == pytest.approx(report.fuel_wear_corr[c])
        assert e["flags"] == [f for f in report.flags if f.endswith(f":{c}")]
    fuel = report.fuel_s_per_lap
    assert out["fuel_s_per_lap_interval"]["median"] == fuel.value
    # the fit's draw_stats are the simulator's own: drawing the report's percentiles leaves them alone
    stats = sim.fitted_for(snap).degradation.fit_report["draw_stats"]
    before = dict(stats)
    agent_tools.model_report(snap, sim=sim, profile=PROFILE)
    assert stats == before or (np.isnan(stats["acceptance_rate"]) and np.isnan(before["acceptance_rate"]))


def test_model_report_without_draws_uses_the_sd(syn):
    _, snap = syn
    params = {k: v for k, v in BELIEF.degradation.params.items() if k != "draw"}
    cfg = BELIEF.model_copy(update={"degradation": BELIEF.degradation.model_copy(update={"params": params})})
    out = agent_tools.model_report(snap, sim=syn_sim(syn_frame(), cfg), profile=PROFILE)
    for c, e in out["compounds"].items():
        if e["modelled"]:
            wear = e["wear_s_per_lap"]
            assert e["interval_basis"]["deg_per_lap_s"] == agent_tools.BASIS_SD
            assert e["deg_per_lap_s"]["median"] == e["point_estimate"]["deg_per_lap_s"] == wear["value"]
            assert e["deg_per_lap_s"]["low"] == pytest.approx(wear["value"] - 1.64 * wear["sd"])


def test_interval_without_sd_is_flagged():
    assert agent_tools._interval(0.05, float("nan")) == ({"low": 0.05, "median": 0.05, "high": 0.05}, "no_sd")
    iv, basis = agent_tools._interval(0.0, 0.01, draws=np.linspace(0.0, 0.1, 101))
    # the median is the draws': a point estimate on the bound (0) never sits below low
    assert basis == agent_tools.BASIS_DRAWS and iv == pytest.approx({"low": 0.005, "median": 0.05, "high": 0.095})


def test_pit_loss_config_band_and_calibrator_sd(syn, monkeypatch):
    sim, snap = syn
    out = agent_tools.pit_loss(snap, sim=sim, profile=PROFILE)
    assert out["source"] == "config" and out["sd_s"] is None
    assert out["pit_loss_s"] == pytest.approx({"low": 18.0, "median": 20.0, "high": 22.0})
    assert (out["sc_pit_loss_factor"], out["vsc_pit_loss_factor"]) == (0.5, 0.6)
    assert out["track_status"] == "green"
    assert out["pit_loss_now_factor"] == 1.0 and out["pit_loss_now_s"] == out["pit_loss_s"]
    for status, factor in (("sc", 0.5), ("vsc", 0.6)):
        now = agent_tools.pit_loss(snap.model_copy(update={"track_status": status}), sim=sim, profile=PROFILE)
        assert now["pit_loss_s"] == out["pit_loss_s"] and now["pit_loss_now_factor"] == factor
        assert now["pit_loss_now_s"] == pytest.approx({k: v * factor for k, v in out["pit_loss_s"].items()})
    monkeypatch.setattr(agent_tools.calibrators, "effective_factors", lambda s, race: {"pit_loss_s": (21.0, 1.0)})
    cal = agent_tools.pit_loss(snap, sim=sim, profile=PROFILE)
    assert cal["source"] == "calibrator" and cal["sd_s"] == 1.0
    assert cal["pit_loss_s"] == pytest.approx({"low": 19.36, "median": 21.0, "high": 22.64})


def test_simulate_plan_is_the_simresult(syn):
    sim, snap = syn
    plan = [["SOFT", 5], ["MEDIUM", snap.remaining_laps - 5]]
    out = agent_tools.simulate_plan(snap, plan, sim=sim, profile=PROFILE)
    ref = sim.simulate(snap, plan, seed=agent_tools.SEED, n_samples=20).model_dump(mode="json")
    assert out == json.loads(json.dumps(ref).replace("NaN", "null"))
    assert out["remaining"]["n_samples"] == 20
    bad = agent_tools.simulate_plan(snap, [["SOFT", 5]], sim=sim, profile=PROFILE)
    assert bad["valid"] is False and bad["errors"]


def test_compare_plans_expands_actions_with_the_optimizer(syn):
    sim, snap = syn
    out = agent_tools.compare_plans(snap, [{"action": "stay_out", "laps": 2}, {"action": "pit", "compound": "MEDIUM"},
                                           [["SOFT", 1]]], sim=sim, profile=PROFILE)
    json.dumps(out, allow_nan=False)
    assert out["valid"] == [True, True, False] and out["n_samples"] == 20 and out["seed"] == agent_tools.SEED
    stay, pit = out["candidates"][0]["stints"], out["candidates"][1]["stints"]
    assert stay[0]["compound"] == "SOFT" and stay[0]["laps"] >= 2
    assert (pit[0]["laps"], pit[1]["compound"]) == (1, "MEDIUM")
    assert len(pit) - 1 <= PROFILE.sim.max_stops   # a pit spends one of the stops
    to_flag = agent_tools.compare_plans(snap, [{"action": "stay", "laps": 99}], sim=sim, profile=PROFILE)
    assert to_flag["candidates"][0]["stints"][0]["laps"] == snap.remaining_laps


def test_evaluate_action_pit_against_staying_out(syn):
    sim, snap = syn
    out = agent_tools.evaluate_action(snap, {"action": "pit", "compound": "MEDIUM"}, sim=sim, profile=PROFILE)
    json.dumps(out, allow_nan=False)
    assert out["alternative"]["action"] == {"action": "stay", "laps": 1} and out["alternative"]["valid"]
    d = out["relative_race_time_delta_s"]
    assert set(d) == {"optimistic", "median", "pessimistic"}
    assert d["optimistic"] <= d["median"] <= d["pessimistic"]
    c = agent_tools.compare_plans(snap, [{"action": "pit", "compound": "MEDIUM"}, {"action": "stay", "laps": 1}],
                                  sim=sim, profile=PROFILE)
    assert out["gap_mean_s"] == pytest.approx(c["gap_mean_s"][0][1])
    assert out["gap_std_s"] == pytest.approx(c["gap_std_s"][0][1])
    assert out["probability_improves_outcome"] == c["p_beats"][0][1]
    assert out["probability_covers"] == agent_tools.PROBABILITY_COVERS
    assert "traffic_probability" not in out and "not modelled" in out["traffic"]
    assert out["dominated"] == c["dominated"][0] and out["support_warnings"] == c["support_warnings"][0]
    cal = agent_tools.calibration_error_s(sim, snap)   # the midrace summary's to_snapshot row
    assert out["calibration_error_s"] == cal
    assert out["within_sim_error"] == (abs(out["gap_mean_s"]) < max(2 * out["gap_std_s"], cal))


def test_evaluate_action_stay_takes_the_pit_compare_prefers(syn):
    sim, snap = syn
    out = agent_tools.evaluate_action(snap, {"action": "stay", "laps": 3}, sim=sim, profile=PROFILE)
    alts = [{"action": "pit", "compound": c} for c in ("SOFT", "MEDIUM")]
    c = agent_tools.compare_plans(snap, [{"action": "stay", "laps": 3}, *alts], sim=sim, profile=PROFILE)
    best = max((1, 2), key=lambda i: (c["p_best"][i], -c["summaries"][i]["mean_s"], -i))
    assert out["alternative"]["action"] == alts[best - 1]
    assert out["gap_mean_s"] == pytest.approx(c["gap_mean_s"][0][best])


def test_within_sim_error_and_its_basis(syn, monkeypatch):
    sim, snap = syn
    act = {"action": "pit", "compound": "MEDIUM"}
    base = agent_tools.evaluate_action(snap, act, sim=sim, profile=PROFILE)
    gap, std = abs(base["gap_mean_s"]), base["gap_std_s"]
    monkeypatch.setattr(agent_tools, "calibration_error_s", lambda s, snapshot: gap + 2 * std + 1.0)
    big = agent_tools.evaluate_action(snap, act, sim=sim, profile=PROFILE)
    assert big["within_sim_error"] is True and big["confidence_basis"] == "model error"
    monkeypatch.setattr(agent_tools, "calibration_error_s", lambda s, snapshot: std)   # below 2 sd
    small = agent_tools.evaluate_action(snap, act, sim=sim, profile=PROFILE)
    assert small["confidence_basis"] == "sampling spread" and small["within_sim_error"] == (gap < 2 * std)
    assert small["calibration_error_s"] == std


def test_evaluate_action_errors_and_edges(syn):
    sim, snap = syn
    bad = agent_tools.evaluate_action(snap, {"action": "pit", "compound": "HARD"}, sim=sim, profile=PROFILE)
    assert bad["valid"] is False and bad["error"]
    early = syn_snap(k=2)   # 2 laps on the set: a pit now is a 3-lap stint, below the 8-lap minimum
    out = agent_tools.evaluate_action(early, {"action": "stay", "laps": 1}, sim=sim, profile=PROFILE)
    assert out["alternative"]["valid"] is False and out["alternative"]["errors"]
    assert out["relative_race_time_delta_s"] == {"optimistic": None, "median": None, "pessimistic": None}
    assert out["within_sim_error"] is None and out["probability_improves_outcome"] is None
    # a stay past the flag runs the current set to the end, which breaks R3 on one compound
    flag = agent_tools.evaluate_action(snap, {"action": "stay_out", "laps": 99}, sim=sim, profile=PROFILE)
    assert "two different compounds" in flag["error"]
    later = snap.model_copy(update={"compound": "MEDIUM", "tyre_age": 0, "stints": [
        StintRecord(compound="SOFT", laps=K, start_lap=1)]})
    flag = agent_tools.evaluate_action(later, {"action": "stay_out", "laps": 99}, sim=sim, profile=PROFILE)
    assert flag["plan"] == [["MEDIUM", snap.remaining_laps]] and "error" not in flag


def test_recommendation_fields_give_p_pit(syn):
    _, snap = syn
    assert agent_tools.recommendation_fields(snap, {"decision": "PIT", "confidence": 0.7})["p_pit"] == 0.7
    stay = agent_tools.recommendation_fields(snap, {"decision": "stay_out", "confidence": 0.75})
    assert stay["decision"] == "STAY_OUT" and stay["p_pit"] == pytest.approx(0.25)
    assert agent_tools.recommendation_fields(snap, {"decision": "maybe"}) == {
        "decision": None, "p_pit": None, "laps_completed": K}


# --- the wiring in tools.py --------------------------------------------------------------
def test_tools_keep_the_contract_and_read_min_stint_from_it():
    assert not hasattr(tools, "MIN_STINT_LAPS")
    names = [t["name"] for t in tools.TOOL_SPECS]
    assert {"get_model_report", "compare_plans"} <= set(names)
    snap = syn_snap()
    view = tools.set_snapshot(snap)
    assert tools._contract() is snap and view == RaceSnapshot.from_contract(snap)
    assert view.min_stint_laps == SYN_RACE.min_stint_laps
    # the synthetic race is not in the cache: the tools say so instead of failing
    out = tools.estimate_pit_loss()
    assert out["recoverable"] is False and "syn" in out["error"]
    desc = next(t for t in tools.TOOL_SPECS if t["name"] == "evaluate_strategy")["description"]
    assert "not model error" in desc and "confidence_basis" in desc


def test_view_of_an_unknown_race_rebuilds_its_contract():
    view = RaceSnapshot(season=2099, event="nowhere", lap=3, total_laps=20, driver="X", position=1,
                        compound="SOFT", tyre_age_laps=3)
    assert agent_tools.race_id_for_view(view) is None
    assert agent_tools.contract_of(view).race.race_id == "2099_nowhere"


# --- Spain ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spain():
    return agent_tools._race_data("2021_04_spain", load_profile())


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_view_of_spain_takes_the_race_constants(spain):
    view = RaceSnapshot(season=2021, event="spain", lap=30, total_laps=60, driver="TST", position=4,
                        compound="SOFT", tyre_age_laps=22)
    assert agent_tools.race_id_for_view(view) == "2021_04_spain"
    race = agent_tools.contract_of(view).race
    assert (race.race_id, race.total_laps, race.pit_loss_s) == ("2021_04_spain", 60, spain.race.pit_loss_s)


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_ham_lap_20_is_within_sim_error_on_model_error(spain):
    """At HAM lap 20 the sampling spread alone would claim certainty (|gap| > 2 sd); the
    calibration error says the gap is inside the model's error."""
    snap = calibration.snapshot_at(spain.frame, spain.race, "HAM", 20)
    # the continuations without the stint-length support filter (SPEC 0023 B): at lap 20 the
    # belief has seen SOFT to age 24 and MEDIUM to 22, and the filtered continuations spread
    # the gap past the claim this test is about
    sim = agent_tools.belief_simulator(
        snap, config=agent_tools.belief_config().model_copy(update={"support_margin_laps": None}))
    out = agent_tools.evaluate_action(snap, {"action": "stay", "laps": 1}, sim=sim)
    assert abs(out["gap_mean_s"]) > 2 * out["gap_std_s"]
    assert out["within_sim_error"] is True and out["confidence_basis"] == "model error"
    assert out["calibration_error_s"] > 2 * out["gap_std_s"]


class Scripted(ModelBackend):
    name = "test"
    model_id = "scripted"

    def __init__(self, calls):
        self.turns = iter([ModelTurn(tool_calls=[ToolCall(str(i), n, a)]) for i, (n, a) in enumerate(calls)])

    def run_turn(self, **kwargs):
        return next(self.turns)


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_run_agent_end_to_end_through_the_belief(spain):
    snap = calibration.snapshot_at(spain.frame, spain.race, "HAM", 20)
    rest = snap.remaining_laps
    calls = [("get_race_snapshot", {}), ("estimate_tyre_degradation", {"compound": "SOFT"}),
             ("estimate_pit_loss", {}), ("get_model_report", {"ages": {"MEDIUM": 30}}),
             ("compare_plans", {"candidates": [[["SOFT", 8], ["MEDIUM", rest - 8]],
                                               {"action": "stay_out", "laps": 3}]}),
             ("evaluate_strategy", {"action": {"action": "pit", "compound": "MEDIUM"}}),
             ("evaluate_strategy", {"action": {"action": "stay_out", "laps": 3}}),
             ("submit_recommendation", {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6,
                                        "rationale": "Staying out is faster inside the model error.",
                                        "key_risks": ["model error"]})]
    rec, traj = run_agent(snap, backend=Scripted(calls), verbose=False)
    assert rec["decision"] == "STAY_OUT"
    assert [t["tool"] for t in traj] == [n for n, _ in calls]
    assert all(t["error"] is None for t in traj), [t["output"] for t in traj if t["error"]]
    assert tools._contract() is snap
    out = {t["tool"]: t["output"] for t in traj}
    assert out["get_race_snapshot"]["min_stint_laps"] == snap.race.min_stint_laps
    assert out["estimate_tyre_degradation"]["wear_at_bound"] is True
    assert out["estimate_pit_loss"]["pit_loss_s"]["median"] == snap.race.pit_loss_s
    assert out["compare_plans"]["valid"] == [True, True]
    ev = traj[-2]["output"]
    assert ev["action"] == {"action": "stay_out", "laps": 3} and ev["within_sim_error"] is True
    for t in traj:
        json.dumps(t["output"], allow_nan=False)


# --- the belief's seed offset ---------------------------------------------------------------
def _recording(sim):
    """`sim` with its events model's draws recorded, in sample order."""
    seen = []
    draw = sim.events.draw

    def record(snapshot, rng):
        out = draw(snapshot, rng)
        seen.append((tuple(out.sc_laps), tuple(out.vsc_laps)))
        return out
    sim.events.draw = record
    return seen


def test_belief_never_draws_the_worlds_event_samples():
    """With events on in both, the belief's scenario for (seed 0, sample k) is never the
    world's: the tools draw from seed 0 + BELIEF_SEED_OFFSET. High rates, so every sample has
    a period and an equal pair would mean a shared stream."""
    assert agent_tools.SEED == agent_tools.BELIEF_SEED_OFFSET == 1_000_003
    events = ComponentSpec(name="rate", params={"p_sc": 0.2, "p_vsc": 0.2})
    df = syn_frame()
    belief = syn_sim(df, BELIEF.model_copy(update={"events": events}))
    world = Simulator(df, SimConfig(events=events))
    snap = syn_snap(df, k=0)
    plan = [["SOFT", 20], ["MEDIUM", 30]]
    b_seen, w_seen = _recording(belief.fitted_for(snap)), _recording(world)
    agent_tools.simulate_plan(snap, plan, n_samples=200, sim=belief, profile=PROFILE)
    world.simulate_batch(snap, [plan], seed=0, n_samples=200)
    assert len(b_seen) == len(w_seen) == 200
    assert all(b != w for b, w in zip(b_seen, w_seen))
    assert all(sc or vsc for sc, vsc in b_seen + w_seen)


def test_decision_continuations_draw_with_the_offset(monkeypatch):
    from src.eval import decisions

    seeds = []
    real = decisions.as_continuation

    def spy(fit, **kw):
        seeds.append(kw["seed"])
        return real(fit, **kw)
    monkeypatch.setattr(decisions, "as_continuation", spy)
    df = syn_frame()
    decisions.to_plan(syn_snap(df), {"decision": "STAY_OUT", "reassess_after_laps": 2, "confidence": 0.5},
                      syn_sim(df), 2, seed=7, n_samples=5)
    assert seeds == [7 + agent_tools.BELIEF_SEED_OFFSET]


def test_eval_rows_unchanged_with_world_events_off(monkeypatch):
    """The baselines plan on the belief's fit, never on its samples, so with the world's
    events off their rows are the rows of a belief without events and without the offset."""
    from src.eval.policies import BASELINE_POLICIES
    from src.eval.scenarios import Scenario
    from src.eval.score import score_scenario

    df = syn_frame()
    world = Simulator(df, SimConfig(), synthetic_practice())
    snaps = [syn_snap(df, k) for k in (0, 10, 20)]
    items = [Scenario(scenario_id=f"syn:{DRIVER}:{s.laps_completed}", race_id="syn", driver=DRIVER,
                      lap=s.laps_completed, snapshot=s, actual_stints=[["SOFT", 25], ["MEDIUM", 25]])
             for s in snaps]
    no_events = BELIEF.model_copy(update={"events": SimConfig().events})

    def rows(belief_config):
        belief = syn_sim(df, belief_config)
        return [r for sc in items for r in score_scenario(sc, world, BASELINE_POLICIES, belief=belief,
                                                           n_samples=10)]
    new = rows(BELIEF)
    monkeypatch.setattr(agent_tools, "BELIEF_SEED_OFFSET", 0)
    old = rows(no_events)
    assert BELIEF.events.name == "rate" and len(new) == 12
    assert any(r["status"] == "ok" for r in new)
    def strip(rs):   # latency is wall time
        return json.dumps([{k: v for k, v in r.items() if k != "latency_s"} for r in rs], sort_keys=True, default=str)
    assert strip(new) == strip(old)


def test_event_rates_block_and_its_source(syn):
    sim, snap = syn
    kw = {"sim": sim, "profile": PROFILE}
    want = {"p_sc": 0.015, "d_sc": 3.0, "p_vsc": 0.02, "d_vsc": 2.0,
            "source": "provisional: SC from 2021 Spain, VSC placeholder"}
    assert agent_tools.race_state(snap, config=sim.config)["event_rates"] == want
    assert agent_tools.model_report(snap, **kw)["event_rates"] == want
    act = {"action": "pit", "compound": "MEDIUM"}
    assert f"SC and VSC rates: {want['source']}" in agent_tools.evaluate_action(snap, act, **kw)["assumptions"]
    circuit = snap.model_copy(update={"race": snap.race.model_copy(update={"sc_rate_per_lap": 0.03,
                                                                           "sc_mean_laps": 4.0})})
    got = agent_tools.race_state(circuit, config=sim.config)["event_rates"]
    assert got == {**want, "p_sc": 0.03, "d_sc": 4.0, "source": "circuit"}
    assert agent_tools.model_report(circuit, **kw)["event_rates"]["source"] == "circuit"
    assert "SC and VSC rates: circuit" in agent_tools.evaluate_action(circuit, act, **kw)["assumptions"]
