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
from src.eval.decisions import belief_optimum, to_plan
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
    assert v1 is agent.SUBMIT_TOOL
    assert set(v1["input_schema"]["properties"]) | {"first_stop_lap"} == set(v2["input_schema"]["properties"])
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
        nxt = p["plan"].split(" > ")[1].split()[0] if p["stops"] else None
        stop = snap.laps_completed + laps
        if p["stops"] and laps == 1:
            assert p["submit"] == {"decision": "PIT", "compound": nxt, "first_stop_lap": stop}
        elif p["stops"]:
            assert p["submit"] == {"decision": "STAY_OUT", "reassess_after_laps": laps - 1, "compound": nxt,
                                   "first_stop_lap": stop}
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
    assert out["submit"] == {"decision": "STAY_OUT", "reassess_after_laps": 3, "compound": "HARD", "first_stop_lap": lap}
    assert "scored_plan" not in out   # naming the stop keeps the evaluated plan
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


PLAN1 = {"decision": "STAY_OUT", "reassess_after_laps": 5, "compound": "HARD", "first_stop_lap": None}
EVIDENCE = {"plans": [{"rank": 1, "plan": "X 6 > HARD 20", "stops": 1, "immediate": "STAY_OUT", "submit": PLAN1}],
            "decision": {"favoured": "STAY_OUT", "share": 0.93, "reading": "STAY_OUT (best plan) is faster in 93%"}}


@pytest.fixture
def plan1(snaps):
    PLAN1["first_stop_lap"] = snaps["mid"].laps_completed + 6
    return dict(PLAN1)


def fake_rank(monkeypatch):
    real = agent._dispatch
    monkeypatch.setattr(agent, "_dispatch", lambda name, args, proto="v1": EVIDENCE if name == "rank_plans"
                        else real(name, args, proto))


@pytest.mark.parametrize("final_is_plan1", [False, True])
def test_a_submission_that_leaves_plan_one_is_held_once_and_the_next_is_final(snaps, plan1, monkeypatch, tmp_path,
                                                                             final_is_plan1):
    fake_rank(monkeypatch)
    pit = {"decision": "PIT", "compound": "MEDIUM"}
    final = plan1 if final_is_plan1 else pit
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {})]),
                         ModelTurn(tool_calls=[submit(**pit)]),
                         ModelTurn(tool_calls=[submit(**final)])])
    stats, log = {}, tmp_path / "t.jsonl"
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=4, stats=stats, log_path=log)
    held = traj[1]["output"]["held"]
    assert traj[1]["output"]["accepted"] is False and "does not keep plan 1" in held and '"first_stop_lap"' in held
    assert "does not keep plan 1" in backend.seen[2][-1]["content"]   # the model is shown its evidence
    assert rec["decision"] == final["decision"] and rec.get("first_stop_lap") == final.get("first_stop_lap")
    assert stats["check_fired"] is True and stats["check_changed"] is final_is_plan1
    events = [json.loads(x) for x in log.read_text().splitlines()]
    checks = [e for e in events if e.get("event") == "consistency_check"]
    assert len(checks) == 2 and checks[-1]["changed"] is final_is_plan1


def test_plan_ones_own_submission_is_not_held(snaps, plan1, monkeypatch):
    fake_rank(monkeypatch)
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}), submit(reassess_after_laps=9)]),
                         ModelTurn(tool_calls=[submit(**plan1)])])
    stats = {}
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3, stats=stats)
    assert "Without first_stop_lap the rest of the race is planned again" in traj[1]["output"]["held"]
    assert rec["first_stop_lap"] == plan1["first_stop_lap"] and stats["check_changed"] is True
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}), submit(**plan1)])])
    stats = {}
    rec, _ = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3, stats=stats)
    assert rec["reassess_after_laps"] == 5 and stats["check_fired"] is False


def test_v1_drops_a_first_stop_lap_from_the_submission(snaps, plan1, monkeypatch):
    monkeypatch.setenv("AGENT_PROTOCOL", "v1")
    rec, _ = run_agent(snaps["mid"], verbose=False, backend=Recording([ModelTurn(tool_calls=[submit(**plan1)])]),
                       max_steps=2)
    assert "first_stop_lap" not in rec and rec["reassess_after_laps"] == 5


@pytest.mark.parametrize("name", ["mid", "late3", "lap0"])
def test_naming_the_optimizers_first_stop_scores_its_plan(snaps, name):
    """The agreement ceiling: plan 1's submit fields expand (decisions.to_plan, as the scorer does) to
    belief_optimum's plan exactly, which a bare STAY_OUT k does not guarantee."""
    prof = load_profile("local")
    belief = race_simulators(prof)(RACE, "belief")
    snap = snaps[name]
    with agent_tools.belief_config_as(belief.config):
        tools.set_snapshot(snap)
        sub = tools.rank_plans()["plans"][0]["submit"]
    rec = {**sub, "confidence": 0.6}
    assert tools.check_submission(rec) is None
    opt = belief_optimum(snap, belief, prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples)
    assert to_plan(snap, rec, belief, prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples).as_pairs() == opt.as_pairs()


def test_a_named_first_stop_that_breaks_the_action_is_refused_with_the_fix(snaps):
    snap = snaps["mid"]
    tools.set_snapshot(snap)
    done = snap.laps_completed
    msg = tools.check_submission({"decision": "STAY_OUT", "reassess_after_laps": 5, "compound": "HARD",
                                  "first_stop_lap": done + 3, "confidence": 0.6})
    assert "needs reassess_after_laps of at most 2" in msg
    msg = tools.check_submission({"decision": "PIT", "compound": "HARD", "first_stop_lap": done + 4, "confidence": 0.6})
    assert f"first_stop_lap must be {done + 1}" in msg
    msg = tools.check_submission({"decision": "STAY_OUT", "reassess_after_laps": 2, "first_stop_lap": done + 5,
                                  "confidence": 0.6})
    assert "needs the compound" in msg


def test_a_broken_named_stop_says_how_to_fix_it_and_on_the_last_turn_is_dropped(snaps, tmp_path):
    """A live check-sample episode repeated a first_stop_lap without its compound until the turns ran
    out (no submission): the reply now says what to copy, and on the very last turn the submission is
    scored without the named stop rather than not at all."""
    snap = snaps["mid"]
    bad = {"decision": "STAY_OUT", "reassess_after_laps": 3, "compound": None,
           "first_stop_lap": snap.laps_completed + 4}
    backend = Recording([ModelTurn(tool_calls=[submit(**bad)]) for _ in range(3)])
    stats = {}
    rec, traj = run_agent(snap, verbose=False, backend=backend, max_steps=1, submit_grace_steps=2, stats=stats)
    assert "Copy first_stop_lap and compound together" in traj[0]["output"]["error"]
    assert "first_stop_lap" not in rec and rec["reassess_after_laps"] == 3
    assert "kept_stop_dropped" in traj[-1]["output"] and traj[-1]["output"]["accepted"] is True


def test_what_v2_accepts_the_scorer_scores(snaps):
    """test_agent_contract's property for the kept-stop submissions: accepted means an ok row."""
    from src.eval.policies import Decision
    from src.eval.scenarios import Scenario
    from src.eval.score import policy_scenario, score_scenario
    prof = load_profile("local")
    sim_for = race_simulators(prof)
    belief, world = sim_for(RACE, "belief"), sim_for(RACE, "world")
    snap = snaps["mid"]
    sc = Scenario(scenario_id=f"{RACE}:{DRIVER}:{snap.laps_completed:02d}", race_id=RACE, driver=DRIVER,
                  lap=snap.laps_completed, snapshot=snap, actual_stints=[])
    seen = policy_scenario(sc, belief)
    tools.set_snapshot(seen.snapshot)
    done = seen.snapshot.laps_completed
    subs = [{"decision": "PIT", "compound": c, "first_stop_lap": done + 1, "confidence": 0.6} for c in ("SOFT", "MEDIUM", "HARD")]
    subs += [{"decision": "STAY_OUT", "reassess_after_laps": k, "compound": c, "first_stop_lap": done + w, "confidence": 0.6}
             for k, w in ((1, 2), (3, 5), (4, 4), (5, 12)) for c in ("MEDIUM", "HARD", "WET")]
    accepted = [s for s in subs if tools.check_submission(s) is None]
    assert accepted and len(accepted) < len(subs)
    policies = {f"s{i}": (lambda r: (lambda scenario, b, m: Decision(None, submission=r)))(s) for i, s in enumerate(accepted)}
    rows = score_scenario(sc, world, policies, belief=belief, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops)
    assert not [(r["policy"], r["status"], r.get("reason")) for r in rows if r["status"] != "ok"]


def test_with_no_turn_left_the_submission_is_not_held(snaps, monkeypatch):
    fake_rank(monkeypatch)
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}),
                                               submit(decision="PIT", compound="MEDIUM")])])
    stats = {}
    rec, _ = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=1, submit_grace_steps=0, stats=stats)
    assert rec["decision"] == "PIT" and stats["check_fired"] is False


# --- v2free (the default): v2's tools plus the full picture, the choice left to the model ----------
def test_v2free_is_the_default_and_leaves_the_choice_to_the_model(snaps, monkeypatch):
    monkeypatch.delenv("AGENT_PROTOCOL", raising=False)
    assert agent.protocol() == "v2free"
    assert "copy its submit" not in agent.SYSTEM_PROMPT_V2FREE and "plan 1 is the default" not in agent.SYSTEM_PROMPT_V2FREE
    backend = Recording([ModelTurn(tool_calls=[submit(reassess_after_laps=3)])])
    run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=2)
    assert backend.prompts[0] == agent.SYSTEM_PROMPT_V2FREE
    assert {"rank_plans", "evaluate_plan", "race_picture"} <= set(backend.offered[0])


@pytest.mark.agent_protocol_v2free
def test_v2free_does_not_hold_a_departure_from_plan_one(snaps, plan1, monkeypatch):
    fake_rank(monkeypatch)
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}), submit(reassess_after_laps=9)])])
    stats = {}
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3, stats=stats)
    assert rec["reassess_after_laps"] == 9 and stats["check_fired"] is False and traj[-1]["output"]["accepted"]


@pytest.mark.agent_protocol_v2free
@pytest.mark.parametrize("confirm", [True, False])
def test_v2free_holds_once_a_submission_its_reading_contradicts(snaps, plan1, monkeypatch, confirm):
    fake_rank(monkeypatch)   # the reading: STAY_OUT's best plan faster in 93% of draws
    pit = {"decision": "PIT", "compound": "MEDIUM"}
    final = pit if confirm else plan1
    backend = Recording([ModelTurn(tool_calls=[ToolCall("1", "rank_plans", {}), submit(**pit)]),
                         ModelTurn(tool_calls=[submit(**final)])])
    stats = {}
    rec, traj = run_agent(snaps["mid"], verbose=False, backend=backend, max_steps=3, stats=stats)
    held = traj[1]["output"]["held"]
    assert "the reading you received says otherwise" in held and "93%" in held
    assert rec["decision"] == final["decision"]
    assert stats["check_fired"] is True and stats["check_changed"] is (not confirm)


@pytest.mark.parametrize("name", ["mid", "late3", "lap0"])
def test_a_named_plan_is_scored_exactly_and_the_optimizers_is_its_own(snaps, name):
    prof = load_profile("local")
    belief = race_simulators(prof)(RACE, "belief")
    snap = snaps[name]
    opt = belief_optimum(snap, belief, prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples)
    pairs = opt.as_pairs()
    w = pairs[0][1]
    rec = {"decision": "PIT" if len(pairs) > 1 and w == 1 else "STAY_OUT", "reassess_after_laps": max(1, w - 1),
           "compound": pairs[1][0] if len(pairs) > 1 else None, "plan": pairs, "confidence": 0.6}
    with agent_tools.belief_config_as(belief.config):
        tools.set_snapshot(snap)
        assert tools.check_submission(rec) is None
    assert to_plan(snap, rec, belief, prof.sim.max_stops, seed=0, n_samples=prof.sim.n_samples) == opt


def test_a_named_plan_that_breaks_the_decision_is_refused_with_the_fix(snaps):
    snap = snaps["mid"]
    tools.set_snapshot(snap)
    rest, cur, done = snap.remaining_laps, snap.compound, snap.laps_completed
    other = "HARD" if cur != "HARD" else "MEDIUM"
    cases = {
        "sum to": {"decision": "STAY_OUT", "reassess_after_laps": 3, "plan": [[cur, 5], [other, rest - 4]]},
        "first stint continues": {"decision": "STAY_OUT", "reassess_after_laps": 3, "plan": [[other, 5], [cur, rest - 5]]},
        "1 lap": {"decision": "PIT", "compound": other, "plan": [[cur, 3], [other, rest - 3]]},
        "allows no stop before": {"decision": "STAY_OUT", "reassess_after_laps": 6, "plan": [[cur, 5], [other, rest - 5]]},
        "not the plan's first stop": {"decision": "STAY_OUT", "reassess_after_laps": 3, "first_stop_lap": done + 6,
                                      "compound": other, "plan": [[cur, 5], [other, rest - 5]]},
    }
    for needle, rec in cases.items():
        msg = tools.check_submission({**rec, "confidence": 0.6})
        assert msg and msg.startswith("The named plan cannot be scored") and needle in msg, (needle, msg)
    start = snaps["lap0"]   # no stop yet: a one-stint plan uses a single compound
    tools.set_snapshot(start)
    single = tools.check_submission({"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6,
                                     "plan": [[start.compound, start.remaining_laps]]})
    assert single and "two different compounds" in single   # the rules still apply to a named plan


def test_v2_and_v1_drop_a_named_plan(snaps, monkeypatch):
    snap = snaps["mid"]
    bad = {"plan": [["WET", 1]]}   # scored, it would be refused
    for proto in ("v1", "v2"):
        monkeypatch.setenv("AGENT_PROTOCOL", proto)
        rec, _ = run_agent(snap, verbose=False, backend=Recording([ModelTurn(tool_calls=[submit(**bad)])]), max_steps=2)
        assert rec is not None and "plan" not in rec


@pytest.mark.agent_protocol_v2free
def test_race_picture_is_compact_anonymous_and_says_the_sets_wear(snaps):
    tools.set_snapshot(snaps["mid"])
    out = tools.present(tools.race_picture())
    text = json.dumps(out)
    assert len(text) < 3000
    assert set(out) >= {"car", "nearby", "rejoin", "compounds"}
    assert set(out["car"]["set_wear"]) == {"observed_s_per_lap", "clean_laps", "model"}
    assert all(c["car"].startswith("Car ") for c in out["nearby"])
    assert DRIVER not in text and "spain" not in text.lower() and "2021" not in text


@pytest.mark.agent_protocol_v2free
def test_what_v2free_accepts_the_scorer_scores(snaps):
    """test_agent_contract's property for named plans: accepted means an ok row, scored as named."""
    from src.eval.policies import Decision
    from src.eval.scenarios import Scenario
    from src.eval.score import policy_scenario, score_scenario
    prof = load_profile("local")
    sim_for = race_simulators(prof)
    belief, world = sim_for(RACE, "belief"), sim_for(RACE, "world")
    snap = snaps["mid"]
    sc = Scenario(scenario_id=f"{RACE}:{DRIVER}:{snap.laps_completed:02d}", race_id=RACE, driver=DRIVER,
                  lap=snap.laps_completed, snapshot=snap, actual_stints=[])
    seen = policy_scenario(sc, belief)
    tools.set_snapshot(seen.snapshot)
    cur, rest = seen.snapshot.compound, seen.snapshot.remaining_laps
    subs = []
    for c in ("SOFT", "MEDIUM", "HARD", "WET"):
        subs.append({"decision": "PIT", "compound": c, "plan": [[cur, 1], [c, rest - 1]]})
        for k, w in ((2, 3), (5, 12), (5, 6)):
            subs.append({"decision": "STAY_OUT", "reassess_after_laps": k, "plan": [[cur, w], [c, rest - w]]})
        subs.append({"decision": "STAY_OUT", "reassess_after_laps": 1, "plan": [[cur, 4], [c, 10], [cur, rest - 14]]})
    subs = [{**s, "confidence": 0.6} for s in subs]
    accepted = [s for s in subs if tools.check_submission(s) is None]
    assert accepted and len(accepted) < len(subs)
    policies = {f"s{i}": (lambda r: (lambda scenario, b, m: Decision(None, submission=r)))(s) for i, s in enumerate(accepted)}
    rows = score_scenario(sc, world, policies, belief=belief, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops)
    assert not [(r["policy"], r["status"], r.get("reason")) for r in rows if r["status"] != "ok"]
    for r in rows:
        named = accepted[int(r["policy"][1:])]["plan"]
        assert json.loads(r["plan"]) == named if isinstance(r["plan"], str) else r["plan"] == named


@pytest.mark.agent_protocol_v2free
def test_v2free_keeps_the_named_stop_refusal_and_the_last_turn_fallback(snaps):
    test_a_broken_named_stop_says_how_to_fix_it_and_on_the_last_turn_is_dropped(snaps, None)
