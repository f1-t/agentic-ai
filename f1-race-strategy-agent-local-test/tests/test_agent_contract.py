"""The agent's tools agree with the scorer (2021 Spain, the committed cache, the leak-free builder):

- anything submit_recommendation accepts is a decision the scorer turns into a valid, scored plan
  (a property over submissions at lap 0, mid-race and with three laps left), and what it refuses is
  refused with the cause and the legal alternative;
- where pitting now is illegal, a stay-out evaluation is compared with the earliest legal stop, and
  with no legal stop it says STAY_OUT to the flag is the only legal action;
- a stay of k laps names its continuation, a second request resolving to the same plan says so, and
  a stay carries a compact stop-lap scan;
- the snapshot view and the model report state the pit window.
"""
import json
import os

import pytest

from src import tools
from src.agent import run_agent
from src.eval.decisions import to_plan
from src.eval.policies import Decision
from src.eval.scenarios import Scenario
from src.eval.score import policy_scenario, race_simulators, score_scenario
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.sim import agent_tools, jobs
from src.sim.calibration import snapshot_as_of
from src.sim.compute import load_profile

RACE, DRIVER = "2021_04_spain", "HAM"
HAS_CACHE = os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "cache"))
pytestmark = pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")


@pytest.fixture(scope="module")
def setup():
    prof = load_profile("local")
    data = jobs.load_race(RACE, prof)
    total = data.race.total_laps
    laps = {"lap0": 0, "early": 1, "mid": total // 2, "late3": total - 3}
    scenarios = {name: Scenario(scenario_id=f"{RACE}:{DRIVER}:{k:02d}", race_id=RACE, driver=DRIVER, lap=k,
                                snapshot=snapshot_as_of(data.frame, data.race, DRIVER, k), actual_stints=[])
                 for name, k in laps.items()}
    return prof, race_simulators(prof), scenarios


def submissions(snapshot):
    rem = snapshot.remaining_laps
    out = [{"decision": "PIT", "compound": c, "confidence": 0.6, "reassess_after_laps": 3}
           for c in ("SOFT", "MEDIUM", "HARD", "WET", None)]
    out += [{"decision": "STAY_OUT", "reassess_after_laps": k, "confidence": 0.6}
            for k in (0, 1, 2, 3, 5, rem - 1, rem, rem + 4, 2.0) if not isinstance(k, int) or k >= 0]
    out += [{"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": c} for c in (1.2, -0.1, 1.0)]
    return out


@pytest.mark.parametrize("name", ["lap0", "early", "mid", "late3"])
def test_what_the_tool_accepts_the_scorer_scores(setup, name):
    """The property: a submission check_submission accepts is an "ok" row; a refused one names the
    legal alternative."""
    prof, sim_for, scenarios = setup
    sc = scenarios[name]
    belief, world = sim_for(RACE, "belief"), sim_for(RACE, "world")
    seen = policy_scenario(sc, belief)
    tools.set_snapshot(seen.snapshot)
    accepted, refused = [], []
    for i, rec in enumerate(submissions(seen.snapshot)):
        error = tools.check_submission(rec)
        (refused if error else accepted).append((i, rec, error))
    assert accepted and refused
    for _, rec, error in refused:
        assert "legal" in error or "0 to 1" in error or "one of" in error or "1 to" in error, error
    policies = {f"s{i}": (lambda r: (lambda scenario, b, m: Decision(None, submission=r)))(rec)
                for i, rec, _ in accepted}
    rows = score_scenario(sc, world, policies, belief=belief, n_samples=prof.sim.n_samples,
                          max_stops=prof.sim.max_stops)
    bad = [(r["policy"], r["status"], r.get("reason")) for r in rows if r["status"] != "ok"]
    assert not bad, bad


def test_reassess_zero_is_refused_with_the_fix(setup):
    _, sim_for, scenarios = setup
    tools.set_snapshot(scenarios["late3"].snapshot)
    msg = tools.check_submission({"decision": "STAY_OUT", "reassess_after_laps": 0, "confidence": 0.9})
    assert "only STAY_OUT to the flag is legal: use reassess_after_laps = 3" in msg
    tools.set_snapshot(scenarios["lap0"].snapshot)
    msg = tools.check_submission({"decision": "PIT", "compound": "MEDIUM", "confidence": 0.6, "reassess_after_laps": 3})
    assert msg.startswith("Pitting now is not legal") and "reassess_after_laps =" in msg


def test_a_stay_where_pitting_now_is_illegal_is_compared_with_the_earliest_legal_stop(setup):
    _, _, scenarios = setup
    tools.set_snapshot(scenarios["lap0"].snapshot)
    win = tools.pit_window()
    assert win["pit_now_legal"] is False and win["earliest_stop_lap"] > 1
    out = tools.evaluate_strategy({"action": "stay_out", "laps": 3})
    assert out["gap_reading"] and f"at the end of lap {win['earliest_stop_lap']}" in out["gap_reading"]
    assert out["probability_improves_outcome"] is not None and "earliest legal stop" in out["note"]
    assert out["alternative"]["action"]["at_end_of_lap"] == win["earliest_stop_lap"]
    tools.set_snapshot(scenarios["late3"].snapshot)
    out = tools.evaluate_strategy({"action": "stay_out", "laps": 3})
    assert out["alternative"] is None and "only STAY_OUT to the flag is legal" in out["note"]


def test_a_stay_names_its_continuation_and_a_repeat_plan_says_so(setup):
    _, _, scenarios = setup
    snap = scenarios["mid"].snapshot
    tools.set_snapshot(snap)
    outs = {k: tools.evaluate_strategy({"action": "stay_out", "laps": k}) for k in (2, 3, 4)}
    for k, out in outs.items():
        assert out["continuation"].startswith(f"staying out at least {k} lap")
        assert "first stop is at the end of lap" in out["continuation"] or "to the flag" in out["continuation"]
    same = [o for o in outs.values() if o.get("same_plan_as")]
    plans = {json.dumps(o["plan"]) for o in outs.values()}
    assert len(same) == len(outs) - len(plans)   # every repeat of a plan says so, a new plan does not
    scan = outs[2].get("stop_lap_scan")
    if scan is not None:
        assert 1 <= len(scan["rows"]) <= agent_tools.SCAN_POINTS
        assert all(r["stop_lap"] >= tools.pit_window()["earliest_stop_lap"] for r in scan["rows"])


def test_the_view_and_the_report_state_the_pit_window(setup):
    _, _, scenarios = setup
    tools.set_snapshot(scenarios["lap0"].snapshot)
    view = tools.get_race_snapshot()
    assert set(view["pit_window"]) >= {"pit_now_legal", "earliest_stop_lap", "latest_stop_lap", "summary"}
    assert tools.get_model_report()["pit_window"] == view["pit_window"]


class Scripted(ModelBackend):
    name, model_id = "test", "scripted"

    def __init__(self, turns):
        self.turns, self.seen = iter(turns), []

    def run_turn(self, **kw):
        self.seen.append(kw["messages"])
        return next(self.turns)


def test_the_loop_refuses_an_unscoreable_submission_and_takes_the_correction(setup):
    _, _, scenarios = setup
    sub = {"decision": "STAY_OUT", "confidence": 0.9, "rationale": "r", "key_risks": []}
    backend = Scripted([ModelTurn(tool_calls=[ToolCall("1", "submit_recommendation", {**sub, "reassess_after_laps": 0})]),
                        ModelTurn(tool_calls=[ToolCall("2", "submit_recommendation", {**sub, "reassess_after_laps": 3})])])
    rec, traj = run_agent(scenarios["late3"].snapshot, verbose=False, backend=backend, max_steps=3)
    assert rec["reassess_after_laps"] == 3 and len(traj) == 2
    assert "use reassess_after_laps = 3" in traj[0]["output"]["error"]
    assert '"pit_window"' in backend.seen[0][0]["content"]   # the first message states the window
    snap = scenarios["late3"].snapshot
    assert to_plan(snap, rec, race_simulators(load_profile("local"))(RACE, "belief"), 2) is not None
