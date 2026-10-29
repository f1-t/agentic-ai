"""SPEC 0014 B and C: belief and world simulators in the eval, and agent decisions scored as
an action plus a continuation. Synthetic races, plus the committed Spain cache where marked."""
import os

import numpy as np
import pandas as pd
import pytest
from test_sim import RACE, TRUE, mid_race, start, stub, synthetic_frame

from src.contracts import Plan, Snapshot
from src.eval import Decision, ScenarioSet, resolve, summarize
from src.eval import decisions
from src.eval.scenarios import Scenario
from src.eval.score import score_scenario, score_set
from src.sim import SimConfig, Simulator, calibration
from src.sim.data import CACHE_DIR
from src.sim.optimizer_bridge import optimize_batch

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))
# the belief's tyres: the same SOFT, a MEDIUM that wears four times as fast
BELIEF_TRUE = {"SOFT": TRUE["SOFT"], "MEDIUM": (TRUE["MEDIUM"][0], 4 * TRUE["MEDIUM"][1])}


def frame_for(true):
    rows = []
    for comp, (base, slope) in true.items():
        for first in range(0, 30, 3):
            for age in range(1, 21):
                rows.append({"compound": comp, "tyre_age": age, "lap": first + age, "clean": True,
                             "lap_time_s": base + slope * age})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def world():
    return Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))


@pytest.fixture(scope="module")
def belief():
    return Simulator(frame_for(BELIEF_TRUE), SimConfig(degradation={"name": "linear"}, min_samples=6))


def scenario(snap: Snapshot, sid="syn:X:00"):
    return Scenario(scenario_id=sid, race_id=snap.race.race_id, driver=snap.driver, lap=snap.laps_completed,
                    snapshot=snap, actual_stints=[("SOFT", 20), ("MEDIUM", 30)])


# --- B: who sees what ------------------------------------------------------------
def test_policies_plan_on_the_belief_and_the_world_scores(world, belief):
    seen = []

    def probe(sc, sim, max_stops):
        seen.append(sim)
        return Decision([["SOFT", 20], ["MEDIUM", 30]])

    snap = start()
    (row,) = score_scenario(scenario(snap), world, {"probe": probe}, belief=belief)
    assert seen == [belief]
    assert row["opt_time_s"] == pytest.approx(optimize_batch(snap, world).evaluation.mean_time_s)
    assert row["time_s"] == pytest.approx(world.simulate(snap, [["SOFT", 20], ["MEDIUM", 30]]).remaining.mean_s)
    assert row["time_s"] != pytest.approx(belief.simulate(snap, [["SOFT", 20], ["MEDIUM", 30]]).remaining.mean_s)


def test_without_a_belief_one_simulator_plays_both_roles(world):
    seen = []

    def probe(sc, sim, max_stops):
        seen.append(sim)
        return Decision(None)

    score_scenario(scenario(start()), world, {"probe": probe})
    assert seen == [world]


def test_greedy_crossover_follows_the_belief_pace(world, belief):
    snap = start()
    pols = resolve(["greedy_crossover"])
    (alone,) = score_scenario(scenario(snap), world, pols)
    (split,) = score_scenario(scenario(snap), world, pols, belief=belief)
    assert alone["status"] == split["status"] == "ok"
    assert alone["plan"] != split["plan"]                       # the belief's MEDIUM wears faster
    assert alone["opt_time_s"] == split["opt_time_s"]           # the reference is the world's either way


def test_score_set_stamps_both_hashes_and_the_belief_window(world, belief):
    snaps = [start(), mid_race()]
    s = ScenarioSet(version="t", items=[scenario(x, f"syn:X:{x.laps_completed:02d}") for x in snaps])
    s.digest = s.compute_digest()
    roles = []

    def sim_for(race_id, role):
        roles.append(role)
        return {"belief": belief, "world": world}[role]

    rows = score_set(s, sim_for, resolve(["fixed_one_stop"]))
    assert set(roles) == {"belief", "world"}
    assert set(rows["belief_config_hash"]) == {calibration.config_hash(belief.config)}
    assert set(rows["world_config_hash"]) == {calibration.config_hash(world.config)}
    assert rows["belief_config_hash"].iloc[0] != rows["world_config_hash"].iloc[0]
    assert set(rows["belief_window"]) == {"full"}
    assert "config_hash" not in rows


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_race_simulators_default_roles():
    from src.eval.score import race_simulators
    from src.sim.compute import load_profile

    sim_for = race_simulators(load_profile("local"))
    b, w = sim_for("2021_04_spain", "belief"), sim_for("2021_04_spain", "world")
    assert (b.config.fit_window, b.config.traffic.name) == ("to_snapshot", "none")
    assert (w.config.fit_window, w.config.traffic.name) == ("full", "none")
    assert sim_for("2021_04_spain", "belief") is b and b.laps is w.laps
    with pytest.raises(ValueError, match="role"):
        sim_for("2021_04_spain", "oracle")
    world_yaml = SimConfig.from_file(os.path.join(os.path.dirname(__file__), "..", "configs", "sim", "world.yaml"))
    assert (world_yaml.fit_window, world_yaml.traffic.name) == ("full", "ghost")


# --- C: the action plus continuation unit ------------------------------------------
def test_expand_action_is_public_and_stay_takes_laps(world):
    assert Simulator._expand is Simulator.expand_action
    seen = []

    def cont(adv):
        seen.append(adv)
        return stub(7)(adv)

    plan, errors = world.expand_action(mid_race(), {"action": "stay", "laps": 4}, cont)
    assert errors == []
    adv = seen[0]
    assert (adv.laps_completed, adv.tyre_age, adv.compound, adv.stints) == (14, 14, "SOFT", [])
    assert plan.as_pairs() == [["SOFT", 4 + 7], ["MEDIUM", 50 - 10 - 11]]
    one, _ = world.expand_action(mid_race(), {"action": "stay"}, stub(7))
    assert one.stints[0].laps == 1 + 7


def test_a_pit_action_takes_one_lap(world):
    plan, errors = world.expand_action(mid_race(), {"action": "pit", "compound": "MEDIUM", "laps": 3}, stub())
    assert plan is None and any("end of this lap" in e for e in errors)


def test_to_plan_pit_and_stay_with_a_stub_continuation(world, monkeypatch):
    calls = []

    def fake(sim, **kw):
        calls.append(kw)
        return stub(12)

    monkeypatch.setattr(decisions, "as_continuation", fake)
    snap = mid_race()
    pit = decisions.to_plan(snap, {"decision": "PIT", "compound": "MEDIUM", "confidence": 0.8}, world, 2)
    assert pit.as_pairs() == [["SOFT", 1], ["MEDIUM", 12], ["SOFT", 27]]
    stay = decisions.to_plan(snap, {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6},
                             world, 1, seed=5, n_samples=4)
    assert stay.as_pairs() == [["SOFT", 3 + 12], ["MEDIUM", 50 - 10 - 15]]
    assert calls[-1] == {"seed": 5, "n_samples": 4, "max_stops": 1}


def test_to_plan_continuation_is_the_belief_optimizer(world):
    snap = mid_race()
    plan = decisions.to_plan(snap, {"decision": "STAY_OUT", "reassess_after_laps": 2}, world, 2)
    after = snap.model_copy(update={"laps_completed": 12, "tyre_age": 12})
    best = optimize_batch(after, world, max_stops=2).plan
    assert plan.stints[0].laps == 2 + best.stints[0].laps
    assert [s.compound for s in plan.stints] == [s.compound for s in best.stints]


def test_to_plan_stay_to_the_flag_and_invalid_submissions(world):
    snap = Snapshot(race=RACE, driver="X", laps_completed=44, compound="MEDIUM", tyre_age=10,
                    stints=[{"compound": "SOFT", "laps": 34, "start_lap": 1}])
    assert decisions.to_plan(snap, {"decision": "STAY_OUT", "reassess_after_laps": 9}, world, 2).as_pairs() == \
        [["MEDIUM", 6]]
    for bad, match in [({"decision": "BOX"}, "PIT or STAY_OUT"), ({"decision": "PIT"}, "compound"),
                       ({"decision": "STAY_OUT", "reassess_after_laps": 0}, "at least 1"),
                       ({"decision": "STAY_OUT"}, "at least 1"),
                       ({"decision": "PIT", "compound": "HARD"}, "does not model")]:
        with pytest.raises(decisions.InvalidDecision, match=match):
            decisions.to_plan(mid_race(), bad, world, 2)


def test_p_pit():
    assert decisions.p_pit({"decision": "PIT", "confidence": 0.8}) == 0.8
    assert decisions.p_pit({"decision": "stay_out", "confidence": 0.8}) == pytest.approx(0.2)
    assert decisions.p_pit({"decision": "PIT", "confidence": 1.5}) is None
    assert decisions.p_pit({"decision": "PIT"}) is None


def test_submissions_score_as_expanded_plans_with_brier_columns(world, belief):
    def agent(sub):
        return lambda sc, sim, m: Decision(None, submission=sub, tool_calls=3)

    snap = mid_race()
    pols = {"pit": agent({"decision": "PIT", "compound": "MEDIUM", "confidence": 0.9}),
            "stay": agent({"decision": "STAY_OUT", "reassess_after_laps": 5, "confidence": 0.7}),
            "bad": agent({"decision": "PIT", "confidence": 0.4}),
            "plan": lambda sc, sim, m: Decision([["SOFT", 20], ["MEDIUM", 20]])}
    rows = {r["policy"]: r for r in score_scenario(scenario(snap), world, pols, belief=belief)}
    expected = decisions.to_plan(snap, {"decision": "STAY_OUT", "reassess_after_laps": 5}, belief, 2)
    assert rows["stay"]["status"] == "ok"
    assert rows["stay"]["plan"] == str(expected.as_pairs()).replace("'", '"')
    assert rows["stay"]["time_s"] == pytest.approx(world.simulate(snap, expected).remaining.mean_s)
    assert rows["pit"]["status"] == "ok" and rows["pit"]["first_pit_lap"] == 11
    assert rows["bad"]["status"] == "invalid" and "compound" in rows["bad"]["reason"]
    opt_now = rows["plan"]["opt_first_pit_lap"] == 11
    assert all(r["pit_now_opt"] == opt_now for r in rows.values())
    assert (rows["pit"]["p_pit"], rows["stay"]["p_pit"]) == (0.9, pytest.approx(0.3))
    assert rows["pit"]["decision"] == "PIT" and "p_pit" not in rows["plan"]

    table = summarize(pd.DataFrame(rows.values())).query("race_id == 'ALL'").set_index("policy")
    truth = float(opt_now)
    assert table.loc["pit", "brier"] == pytest.approx((0.9 - truth) ** 2)
    assert table.loc["stay", "brier"] == pytest.approx((0.3 - truth) ** 2)
    assert table.loc["pit", "pit_now_agreement"] == float(opt_now)
    assert table.loc["stay", "pit_now_agreement"] == float(not opt_now)
    assert np.isnan(table.loc["plan", "brier"]) and np.isnan(table.loc["plan", "pit_now_agreement"])


def test_summary_has_no_brier_columns_without_agent_rows(world):
    rows = pd.DataFrame(score_scenario(scenario(start()), world, resolve(["fixed_one_stop"])))
    assert "brier" not in summarize(rows) and "pit_now_agreement" not in summarize(rows)
    assert rows["pit_now_opt"].notna().all()
