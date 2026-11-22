"""Agent protocol v2 (src/agent.py AGENT_PROTOCOL, docs/eval.md "Agent protocol v2"), on 2021 Spain
from the committed cache:

- v1 is the prompt and tool surface as they were; v2 offers rank_plans, evaluate_plan and the two
  per-estimate tools, and submit_recommendation takes the same arguments under both;
- rank_plans' plan 1 is the belief_optimizer policy's plan, both immediate actions are on show when
  both are legal, and each plan's submission is the one that keeps it;
- evaluate_plan evaluates a stop at a lap onto a compound or a full plan against plan 1, and says
  what is legal when the lap is not;
- under v2 an unknown argument is refused with the valid ones named and nothing runs;
- the consistency check holds a submission that contradicts the latest rank_plans evidence once,
  and the next submission is final, changed or not.
"""
import json
import os

import pytest

from src import agent, tools
from src.agent import run_agent
from src.eval.decisions import belief_optimum
from src.eval.score import race_simulators
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.sim import agent_tools, jobs
from src.sim.calibration import snapshot_as_of
from src.sim.compute import load_profile

RACE, DRIVER = "2021_04_spain", "HAM"
HAS_CACHE = os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "cache"))
pytestmark = [pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present"),
              pytest.mark.agent_protocol_v2]

V1_TOOLS = ["get_race_snapshot", "estimate_tyre_degradation", "estimate_pit_loss", "get_model_report",
            "compare_plans", "evaluate_strategy", "submit_recommendation"]


@pytest.fixture(scope="module")
def snaps():
    prof = load_profile("local")
    data = jobs.load_race(RACE, prof)
    total = data.race.total_laps
    return {name: snapshot_as_of(data.frame, data.race, DRIVER, k)
            for name, k in {"lap0": 0, "mid": total // 2, "late3": total - 3}.items()}


class Recording(ModelBackend):
    name, model_id = "test", "scripted"

    def __init__(self, turns):
        self.turns, self.prompts, self.offered, self.seen = iter(turns), [], [], []

    def run_turn(self, **kw):
        self.prompts.append(kw["system_prompt"])
        self.offered.append([t["name"] for t in kw["tools"]])
        self.seen.append(list(kw["messages"]))
        return next(self.turns)


def submit(**kw):
    return ToolCall("s", "submit_recommendation", {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6,
                                                   "rationale": "r", "key_risks": [], **kw})


def test_v1_is_the_surface_it_was_and_v2_the_new_one(snaps, monkeypatch):
    assert [t["name"] for t in agent._api_tools("v1")] == V1_TOOLS
    assert [t["name"] for t in agent._api_tools("v2")] == ["rank_plans", "evaluate_plan", "estimate_tyre_degradation",
                                                           "estimate_pit_loss", "submit_recommendation"]
    v1, v2 = agent._api_tools("v1")[-1], agent._api_tools("v2")[-1]
    assert v1 is agent.SUBMIT_TOOL and set(v1["input_schema"]["properties"]) == set(v2["input_schema"]["properties"])
    assert v1["input_schema"]["required"] == v2["input_schema"]["required"]
    for proto, prompt in (("v1", agent.SYSTEM_PROMPT), ("v2", agent.SYSTEM_PROMPT_V2)):
        monkeypatch.setenv("AGENT_PROTOCOL", proto)
        backend = Recording([ModelTurn(tool_calls=[submit(reassess_after_laps=3)])])
        run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=2)
        assert backend.prompts[0] == prompt and backend.offered[0] == [t["name"] for t in agent._api_tools(proto)]
    monkeypatch.setenv("AGENT_PROTOCOL", "v3")
    with pytest.raises(ValueError):
        agent.protocol()


def test_rank_plans_plan_one_is_the_optimizers_and_both_actions_show(snaps):
    prof = load_profile("local")
    belief = race_simulators(prof)(RACE, "belief")
    snap = snaps["mid"]
    with agent_tools.belief_config_as(belief.config):
        tools.set_snapshot(snap)
        out = tools.rank_plans()
    opt = belief_optimum(snap, belief, prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples).as_pairs()
    plans = out["plans"]
    assert plans[0]["plan"] == agent_tools.plan_text(opt) and plans[0]["gap_s"] == 0.0
    assert 1 <= len(plans) <= agent_tools.RANK_TOP
    kinds = {p["immediate"].split()[0] for p in plans}
    assert kinds == {"PIT", "STAY_OUT"} and out["decision"]["favoured"]
    assert 0.5 <= out["decision"]["share"] <= 1.0
    for p in plans:
        first = p["plan"].split(" > ")[0]
        laps = int(first.split()[1])
        if p["stops"] and laps == 1:
            assert p["submit"] == {"decision": "PIT", "compound": p["plan"].split(" > ")[1].split()[0]}
        elif p["stops"]:
            assert p["submit"] == {"decision": "STAY_OUT", "reassess_after_laps": laps - 1}
        assert abs(sum(int(s.split()[1]) for s in p["plan"].split(" > ")) - snap.remaining_laps) == 0
    assert len(json.dumps(out)) < 4000   # compact: the reply cap and the context are real limits


def test_rank_plans_at_lap_zero_shows_only_legal_actions(snaps):
    tools.set_snapshot(snaps["lap0"])
    out = tools.rank_plans()
    assert all(p["immediate"] == "STAY_OUT" for p in out["plans"])
    assert out["decision"]["favoured"] == "STAY_OUT" and "no PIT now is legal" in out["decision"]["reading"]


def test_evaluate_plan_takes_a_stop_lap_or_a_plan_and_says_what_is_legal(snaps):
    snap = snaps["mid"]
    tools.set_snapshot(snap)
    lap = snap.laps_completed + 4
    out = tools.evaluate_plan(first_stop_lap=lap, compound="HARD")
    assert out["first_stop"].startswith(f"the first stop is at the end of lap {lap} onto HARD")
    assert set(out["vs_best"]) >= {"best_plan", "gap_s", "gap_90", "faster_share"}
    assert out["submit"] == {"decision": "STAY_OUT", "reassess_after_laps": 3}
    every = tools.evaluate_plan(first_stop_lap=lap)
    assert every["other_compounds"] and every["vs_best"]["gap_s"] <= out["vs_best"]["gap_s"] + 1e-9
    rest = snap.remaining_laps
    full = tools.evaluate_plan(plan=[[snap.compound, 1], ["MEDIUM", rest - 1]])
    assert full["immediate"] == "PIT MEDIUM"
    late = tools.evaluate_plan(first_stop_lap=snap.race.total_laps)
    assert "error" in late and "latest" in late["error"]
    assert "error" in tools.evaluate_plan()


def test_an_unknown_argument_is_refused_with_the_valid_ones(snaps):
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "evaluate_plan", {"stop_lap": 30, "compound": "HARD"})]),
                         ModelTurn(tool_calls=[submit()])])
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3)
    err = traj[0]["output"]["error"]
    assert "Unknown argument(s) for evaluate_plan: stop_lap" in err and "first_stop_lap" in err
    assert rec is not None


def test_v1_still_drops_an_unknown_argument(snaps, monkeypatch):
    monkeypatch.setenv("AGENT_PROTOCOL", "v1")
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "estimate_pit_loss", {"probe": 1})]),
                         ModelTurn(tool_calls=[submit()])])
    _, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3)
    assert "error" not in traj[0]["output"]


EVIDENCE = {"plans": [{"rank": 1, "plan": "X 6 > Y 20", "stops": 1, "immediate": "STAY_OUT",
                       "submit": {"decision": "STAY_OUT", "reassess_after_laps": 5}}],
            "decision": {"favoured": "STAY_OUT", "share": 0.93, "reading": "STAY_OUT (best plan) is faster in 93%"}}


def fake_rank(monkeypatch):
    real = agent._dispatch
    monkeypatch.setattr(agent, "_dispatch", lambda name, args, proto="v1": EVIDENCE if name == "rank_plans"
                        else real(name, args, proto))


@pytest.mark.parametrize("final,changed", [({"decision": "PIT", "compound": "MEDIUM"}, False),
                                           ({"decision": "STAY_OUT", "reassess_after_laps": 4}, True)])
def test_a_contradicting_submission_is_held_once_and_the_next_is_final(snaps, monkeypatch, tmp_path, final, changed):
    fake_rank(monkeypatch)
    pit = {"decision": "PIT", "compound": "MEDIUM"}
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {})]),
                         ModelTurn(tool_calls=[submit(**pit)]),
                         ModelTurn(tool_calls=[submit(**final)])])
    stats, log = {}, tmp_path / "t.jsonl"
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=4, stats=stats, log_path=log)
    assert traj[1]["output"]["accepted"] is False and "favours STAY_OUT" in traj[1]["output"]["held"]
    assert "favours STAY_OUT" in backend.seen[2][-1]["content"]   # the model is shown the evidence
    assert rec["decision"] == final["decision"]
    assert stats["check_fired"] is True and stats["check_changed"] is changed
    events = [json.loads(x) for x in log.read_text().splitlines()]
    checks = [e for e in events if e.get("event") == "consistency_check"]
    assert len(checks) == 2 and checks[-1]["changed"] is changed


def test_a_stay_that_forbids_plan_ones_stop_is_held_and_a_consistent_one_is_not(snaps, monkeypatch):
    fake_rank(monkeypatch)
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}), submit(reassess_after_laps=9)]),
                         ModelTurn(tool_calls=[submit(reassess_after_laps=5)])])
    stats = {}
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3, stats=stats)
    assert "reassess_after_laps 5 keeps that stop" in traj[1]["output"]["held"]
    assert rec["reassess_after_laps"] == 5 and stats["check_changed"] is True
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}), submit(reassess_after_laps=5)])])
    stats = {}
    rec, _ = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3, stats=stats)
    assert rec["reassess_after_laps"] == 5 and stats["check_fired"] is False


def test_with_no_turn_left_the_submission_is_not_held(snaps, monkeypatch):
    fake_rank(monkeypatch)
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}),
                                               submit(decision="PIT", compound="MEDIUM")])])
    stats = {}
    rec, _ = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=1, submit_grace_steps=0, stats=stats)
    assert rec["decision"] == "PIT" and stats["check_fired"] is False
