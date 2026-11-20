import itertools
import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from src import agent
from src.agent import run_agent
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.snapshot import RaceSnapshot
from src import tools


@pytest.fixture(autouse=True)
def default_loop_options(monkeypatch):
    # the loop options take their defaults unless a test sets them, whatever a local .env says
    monkeypatch.delenv("AGENT_TURN_STATUS", raising=False)
    monkeypatch.delenv("AGENT_REPEAT_MODE", raising=False)


@pytest.fixture
def cache_mode(monkeypatch):
    monkeypatch.setenv("AGENT_REPEAT_MODE", "cache")


def snap():
    # a race in the committed cache: the tools run on its belief simulator (SPEC 0015)
    return RaceSnapshot(season=2021, event="spain", lap=30, total_laps=60, driver="TST",
                        position=4, compound="SOFT", tyre_age_laps=22, gap_ahead_s=1.2,
                        gap_behind_s=2.5, rain_probability=0.2)


class ScriptedBackend(ModelBackend):
    name = "test"
    model_id = "scripted"
    def __init__(self, turns): self.turns = iter(turns)
    def run_turn(self, **kwargs): return next(self.turns)


def tc(name, args=None):
    return ToolCall("id", name, args or {})


def test_tools_return_uncertainty():
    tools.set_snapshot(snap())
    d = tools.estimate_tyre_degradation("SOFT")
    assert set(d["deg_per_lap_s"]) == {"low", "median", "high"}
    r = tools.evaluate_strategy({"action": "pit", "compound": "MEDIUM"})
    assert "probability_improves_outcome" in r
    assert set(r["relative_race_time_delta_s"]) == {"optimistic", "median", "pessimistic"}


def test_invalid_plan_recovery_and_logging():
    backend = ScriptedBackend([
        ModelTurn(tool_calls=[tc("evaluate_strategy", {"action": {"action": "pit", "compound": "WET"}})]),
        ModelTurn(tool_calls=[tc("evaluate_strategy", {"action": {"action": "pit", "compound": "MEDIUM"}})]),
        ModelTurn(tool_calls=[tc("submit_recommendation", {"decision": "PIT", "compound": "MEDIUM", "reassess_after_laps": 3, "confidence": .7, "rationale": "Range favors pitting.", "key_risks": ["traffic"]})]),
    ])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trajectory.jsonl")
        rec, traj = run_agent(snap(), backend=backend, verbose=False, log_path=path)
        assert rec["decision"] == "PIT"
        assert traj[0]["error"]
        rows = [json.loads(x) for x in open(path)]
        assert len(rows) == 3
        assert rows[0]["tool"] == "evaluate_strategy"
        assert rows[-1]["tool"] == "submit_recommendation"


def test_step_budget_stops_run():
    backend = ScriptedBackend([ModelTurn(tool_calls=[tc("get_race_snapshot")])])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=1, submit_grace_steps=0)
    assert rec is None
    assert len(traj) == 1


# --- the evidence budget: a submission is always processed, a turn's results are kept ---------
SUBMIT = {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": .6,
          "rationale": "Enough evidence.", "key_risks": ["traffic"]}


class RecordingBackend(ScriptedBackend):
    """ScriptedBackend that keeps a copy of the messages and the tools it was shown on each turn."""
    def __init__(self, turns, name="test"):
        super().__init__(turns)
        self.name = name
        self.seen = []
        self.offered = []   # the tool names offered on each turn

    def run_turn(self, **kwargs):
        self.seen.append(list(kwargs["messages"]))
        self.offered.append([t["name"] for t in kwargs["tools"]])
        return super().run_turn(**kwargs)


_PROBE = itertools.count()


def evidence_turn(n):
    # distinct arguments (dropped by _dispatch), so the repeat guard leaves these calls alone
    return ModelTurn(tool_calls=[ToolCall(f"e{i}", "get_race_snapshot", {"probe": next(_PROBE)}) for i in range(n)])


def counting_dispatch(monkeypatch):
    calls = []
    real = agent._dispatch

    def dispatch(name, args):
        calls.append(name)
        return real(name, args)
    monkeypatch.setattr(agent, "_dispatch", dispatch)
    return calls


def test_submission_after_a_spent_budget_is_accepted(monkeypatch):
    dispatched = counting_dispatch(monkeypatch)
    backend = ScriptedBackend([evidence_turn(12), evidence_turn(12),
                               ModelTurn(tool_calls=[tc("submit_recommendation", SUBMIT)])])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=24)
    assert rec is not None and rec["decision"] == "STAY_OUT"
    assert len(dispatched) == 24
    assert traj[-1]["tool"] == "submit_recommendation" and traj[-1]["output"] == {"accepted": True}
    assert traj[-1]["call_index"] == 25


def test_submission_in_the_turn_that_spends_the_budget_is_accepted(monkeypatch):
    dispatched = counting_dispatch(monkeypatch)
    turn = evidence_turn(24)
    turn.tool_calls.append(tc("submit_recommendation", SUBMIT))
    rec, traj = run_agent(snap(), backend=ScriptedBackend([turn]), verbose=False, max_tool_calls=24)
    assert rec is not None and len(dispatched) == 24 and len(traj) == 25


def test_evidence_after_the_budget_is_refused_and_logged(monkeypatch, tmp_path):
    dispatched = counting_dispatch(monkeypatch)
    path = tmp_path / "trajectory.jsonl"
    backend = ScriptedBackend([evidence_turn(24), evidence_turn(1),
                               ModelTurn(tool_calls=[tc("submit_recommendation", SUBMIT)])])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=24, log_path=path)
    assert rec is not None
    assert len(dispatched) == 24                       # the 25th evidence call never ran
    refused = traj[24]
    assert refused["tool"] == "get_race_snapshot" and "budget exhausted" in refused["error"]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["call_index"] for r in rows] == list(range(1, 27))
    assert "budget exhausted" in rows[24]["error"] and rows[25]["tool"] == "submit_recommendation"


def test_invalid_submission_after_the_budget_gets_its_error_back(monkeypatch):
    counting_dispatch(monkeypatch)
    backend = RecordingBackend([evidence_turn(2),
                                ModelTurn(tool_calls=[tc("submit_recommendation", {**SUBMIT, "decision": "MAYBE"})]),
                                ModelTurn(tool_calls=[tc("submit_recommendation", SUBMIT)])])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=2)
    assert rec is not None and len(traj) == 4
    last = backend.seen[-1][-1]["content"]
    assert "decision must be PIT or STAY_OUT" in last and "MUST call submit_recommendation" in last


def test_a_turn_that_spends_the_budget_keeps_its_results():
    # text transcript: 2 calls, then a turn of 2 with 1 left -> one runs, one is refused
    backend = RecordingBackend([evidence_turn(2), evidence_turn(2),
                                ModelTurn(tool_calls=[tc("submit_recommendation", SUBMIT)])])
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=3)
    assert rec is not None
    msg = backend.seen[-1][-1]
    assert msg["role"] == "user" and msg["content"].startswith("Tool results:\n")
    body, instruction = msg["content"][len("Tool results:\n"):].split("\n\n")
    results = json.loads(body)
    assert len(results) == 2
    assert results[0]["result"]["driver"] == "TST"
    assert "budget exhausted" in results[1]["result"]["error"]
    assert "MUST call submit_recommendation" in instruction


def test_a_turn_that_spends_the_budget_keeps_its_results_anthropic():
    def turn(calls):
        return ModelTurn(tool_calls=calls, raw=SimpleNamespace(content=[{"type": "tool_use", "id": c.id} for c in calls]))
    # distinct arguments: an identical second call would be a repeat, served from the cache
    first = [ToolCall("a1", "get_race_snapshot", {}), ToolCall("a2", "get_race_snapshot", {"probe": "a2"})]
    backend = RecordingBackend([turn(first), turn([ToolCall("s1", "submit_recommendation", SUBMIT)])],
                               name="anthropic")
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=1)
    assert rec is not None
    content = backend.seen[-1][-1]["content"]
    assert [b["type"] for b in content] == ["tool_result", "tool_result", "text"]
    assert [b["tool_use_id"] for b in content[:2]] == ["a1", "a2"]   # every tool_use is answered
    assert json.loads(content[0]["content"])["driver"] == "TST"
    assert "budget exhausted" in json.loads(content[1]["content"])["error"]
    assert "MUST call submit_recommendation" in content[2]["text"]


# --- the turn budget: a submission grace, an optional status line, repeat handling --------------
ALL_TOOLS = [t["name"] for t in tools.TOOL_SPECS] + ["submit_recommendation"]
SPENT = "evidence turns are spent"


def submit_turn(args=SUBMIT):
    return ModelTurn(tool_calls=[tc("submit_recommendation", args)])


def stub_dispatch(monkeypatch):
    """_dispatch replaced by a stub that records the calls it runs and returns their count."""
    calls = []

    def dispatch(name, args):
        calls.append((name, args))
        return {"ran": len(calls)}
    monkeypatch.setattr(agent, "_dispatch", dispatch)
    return calls


def test_a_submission_in_the_grace_is_accepted(monkeypatch, tmp_path):
    stub_dispatch(monkeypatch)
    path = tmp_path / "trajectory.jsonl"
    backend = RecordingBackend([evidence_turn(1) for _ in range(10)] + [submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=10, log_path=path)
    assert rec is not None and rec["decision"] == "STAY_OUT"
    assert [t["phase"] for t in traj] == ["evidence"] * 10 + ["grace"]
    assert traj[-1]["step"] == 11 and traj[-1]["output"] == {"accepted": True}
    assert backend.offered[:10] == [ALL_TOOLS] * 10 and backend.offered[10] == ["submit_recommendation"]
    assert SPENT in backend.seen[10][-1]["content"]          # what the model read before the grace turn
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["phase"] for r in rows] == [t["phase"] for t in traj]


def test_the_grace_offers_only_the_submission_and_refuses_evidence(monkeypatch):
    ran = stub_dispatch(monkeypatch)
    backend = RecordingBackend([evidence_turn(1) for _ in range(3)] + [evidence_turn(1), submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=3)
    assert rec is not None
    assert len(ran) == 3                                     # the grace turn's evidence call never ran
    refused = traj[3]
    assert refused["phase"] == "grace" and refused["tool"] == "get_race_snapshot"
    assert "Evidence turns are spent" in refused["error"] and refused["output"]["recoverable"] is True
    assert backend.offered[3:] == [["submit_recommendation"]] * 2
    assert traj[-1]["phase"] == "grace" and traj[-1]["step"] == 5


def test_an_invalid_submission_in_the_grace_gets_its_error_and_the_next_turn(monkeypatch):
    stub_dispatch(monkeypatch)
    backend = RecordingBackend([evidence_turn(1), evidence_turn(1),
                                submit_turn({**SUBMIT, "decision": "MAYBE"}), submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=2)
    assert rec is not None and rec["decision"] == "STAY_OUT"
    assert [t["phase"] for t in traj] == ["evidence", "evidence", "grace", "grace"]
    assert traj[2]["error"] == "decision must be PIT or STAY_OUT"
    last = backend.seen[-1][-1]["content"]
    assert "decision must be PIT or STAY_OUT" in last and SPENT in last and "1 submission turn left" in last


def test_no_submission_at_all_returns_none_after_the_grace(monkeypatch):
    ran = stub_dispatch(monkeypatch)
    stats = {}
    # exactly max_steps + grace turns: a 13th run_turn would raise StopIteration
    backend = RecordingBackend([evidence_turn(1) for _ in range(12)])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=10, stats=stats)
    assert rec is None
    assert len(traj) == 12 and len(ran) == 10 and stats["steps"] == 12
    assert [t["phase"] for t in traj[10:]] == ["grace", "grace"]
    assert all("Evidence turns are spent" in t["error"] for t in traj[10:])


def test_the_grace_length_comes_from_the_environment(monkeypatch):
    stub_dispatch(monkeypatch)
    monkeypatch.setenv("AGENT_SUBMIT_GRACE_STEPS", "1")
    backend = RecordingBackend([evidence_turn(1), evidence_turn(1)])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=1)
    assert rec is None and len(backend.seen) == 2


def test_an_identical_evidence_call_gets_the_cached_result_without_using_budget(monkeypatch, cache_mode):
    ran = stub_dispatch(monkeypatch)
    stats = {}
    first = {"ages": {"MEDIUM": 30, "SOFT": 10}}
    same = {"ages": {"SOFT": 10, "MEDIUM": 30}}            # the same call once the keys are sorted
    backend = RecordingBackend([
        ModelTurn(tool_calls=[tc("get_model_report", first)]),
        ModelTurn(tool_calls=[tc("get_model_report", same), tc("get_model_report", {"ages": {"MEDIUM": 31}})]),
        submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=2, stats=stats, turn_status=True)
    assert rec is not None
    assert ran == [("get_model_report", first), ("get_model_report", {"ages": {"MEDIUM": 31}})]
    assert traj[1]["output"] == {"ran": 1, "repeat_of": 1} and traj[1]["error"] is None
    assert traj[1]["repeat_of"] == 1 and "repeat_of" not in traj[0]
    assert traj[0]["output"] == {"ran": 1}                  # the cached output is not mutated
    assert traj[2]["output"] == {"ran": 2}                  # the repeat did not spend the last call
    assert (stats["evidence_calls"], stats["repeat_calls"]) == (2, 1)
    assert "Evidence calls used 2 of 2." in backend.seen[-1][-1]["content"]


def test_a_repeat_runs_the_tool_once_and_is_served_after_the_budget(monkeypatch, tmp_path, cache_mode):
    runs = []
    spec = next(s for s in tools.TOOL_SPECS if s["name"] == "estimate_pit_loss")

    def counted(**kwargs):
        runs.append(kwargs)
        return {"pit_loss_s": 21.5}
    monkeypatch.setitem(spec, "fn", counted)
    path = tmp_path / "trajectory.jsonl"
    stats = {}
    backend = RecordingBackend([ModelTurn(tool_calls=[tc("estimate_pit_loss")]),
                                ModelTurn(tool_calls=[tc("estimate_pit_loss"), tc("get_race_snapshot")]),
                                submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=1, log_path=path, stats=stats)
    assert rec is not None and len(runs) == 1             # the tool function ran once
    assert traj[1]["output"] == {"pit_loss_s": 21.5, "repeat_of": 1}   # served with the budget spent
    assert "budget exhausted" in traj[2]["error"]            # a new call is still refused
    assert (stats["evidence_calls"], stats["repeat_calls"]) == (1, 1)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r.get("repeat_of") for r in rows] == [None, 1, None, None]
    results = json.loads(backend.seen[-1][-1]["content"][len("Tool results:\n"):].split("\n\n")[0])
    assert results[0] == {"tool": "estimate_pit_loss", "result": {"pit_loss_s": 21.5, "repeat_of": 1}}


def test_a_repeat_of_a_call_that_errored_gets_the_same_error(monkeypatch, cache_mode):
    dispatched = counting_dispatch(monkeypatch)
    stats = {}
    bad = {"action": {"action": "pit", "compound": "WET"}}
    backend = RecordingBackend([ModelTurn(tool_calls=[tc("evaluate_strategy", bad)]),
                                ModelTurn(tool_calls=[tc("evaluate_strategy", bad)]), submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, stats=stats)
    assert rec is not None and dispatched == ["evaluate_strategy"]
    assert traj[0]["error"] and traj[1]["error"] == traj[0]["error"]
    assert traj[1]["output"] == {**traj[0]["output"], "repeat_of": 1} and traj[1]["repeat_of"] == 1
    assert (stats["evidence_calls"], stats["repeat_calls"]) == (1, 1)


def test_a_repeat_of_a_non_dict_result_is_wrapped(monkeypatch, cache_mode):
    monkeypatch.setattr(agent, "_dispatch", lambda name, args: ["SOFT", "MEDIUM"])
    backend = RecordingBackend([ModelTurn(tool_calls=[tc("get_race_snapshot")] * 2), submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False)
    assert rec is not None
    assert traj[0]["output"] == ["SOFT", "MEDIUM"]
    assert traj[1]["output"] == {"result": ["SOFT", "MEDIUM"], "repeat_of": 1} and traj[1]["error"] is None


@pytest.mark.parametrize("mode", ["run", "cache"])
def test_a_repeat_in_the_grace_is_still_refused(monkeypatch, mode):
    monkeypatch.setenv("AGENT_REPEAT_MODE", mode)
    ran = stub_dispatch(monkeypatch)
    stats = {}
    backend = RecordingBackend([ModelTurn(tool_calls=[tc("estimate_pit_loss")]),
                                ModelTurn(tool_calls=[tc("estimate_pit_loss")]), submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=1, stats=stats)
    assert rec is not None and len(ran) == 1
    assert traj[1]["phase"] == "grace" and "Evidence turns are spent" in traj[1]["error"]
    assert "repeat_of" not in traj[1] and stats["repeat_calls"] == 0


def test_a_repeat_on_the_anthropic_path(monkeypatch, cache_mode):
    stub_dispatch(monkeypatch)

    def turn(calls):
        return ModelTurn(tool_calls=calls, raw=SimpleNamespace(content=[{"type": "tool_use", "id": c.id} for c in calls]))
    backend = RecordingBackend([turn([ToolCall("a1", "estimate_pit_loss", {})]),
                                turn([ToolCall("a2", "estimate_pit_loss", {})]),
                                turn([ToolCall("s1", "submit_recommendation", SUBMIT)])], name="anthropic")
    rec, traj = run_agent(snap(), backend=backend, verbose=False)
    assert rec is not None
    content = backend.seen[-1][-1]["content"]
    assert content[0]["type"] == "tool_result" and content[0]["tool_use_id"] == "a2"
    assert json.loads(content[0]["content"]) == {"ran": 1, "repeat_of": 1}


def test_evidence_calls_count_only_the_calls_that_ran(monkeypatch, cache_mode):
    ran = stub_dispatch(monkeypatch)
    stats = {}
    backend = RecordingBackend([
        ModelTurn(tool_calls=[tc("estimate_pit_loss"), tc("estimate_pit_loss")]),            # ran, repeat
        ModelTurn(tool_calls=[tc("get_race_snapshot"), tc("get_model_report"),                # ran, budget
                              tc("submit_recommendation", {**SUBMIT, "decision": "MAYBE"})]),  # rejected
        ModelTurn(tool_calls=[tc("get_model_report"), tc("submit_recommendation", SUBMIT)])])  # grace
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=2, max_tool_calls=2, stats=stats)
    assert rec is not None and len(traj) == 7
    assert stats["evidence_calls"] == len(ran) == 2
    assert stats["repeat_calls"] == 1


def test_the_status_line_counts_turns_and_calls(monkeypatch):
    stub_dispatch(monkeypatch)
    backend = RecordingBackend([evidence_turn(2), evidence_turn(1), evidence_turn(1), submit_turn()])
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_steps=3, max_tool_calls=24, turn_status=True)
    assert rec is not None
    after = [seen[-1]["content"] for seen in backend.seen[1:]]   # the message closing turns 1, 2, 3
    assert after[0].endswith("\n\nTurn 1 of 3. Evidence calls used 2 of 24.")
    assert after[1].endswith("\n\nTurn 2 of 3. Evidence calls used 3 of 24. One evidence turn is left after this one.")
    assert "Turn 3 of 3. Evidence calls used 4 of 24." in after[2] and SPENT in after[2]
    assert "2 submission turns left" in after[2]


def test_the_status_line_and_the_grace_on_the_anthropic_path(monkeypatch):
    stub_dispatch(monkeypatch)

    def turn(calls):
        return ModelTurn(tool_calls=calls, raw=SimpleNamespace(content=[{"type": "tool_use", "id": c.id} for c in calls]))
    backend = RecordingBackend([turn([ToolCall("a1", "get_race_snapshot", {})]),
                                turn([ToolCall("g1", "estimate_pit_loss", {}),
                                      ToolCall("g2", "submit_recommendation", {**SUBMIT, "decision": "MAYBE"})]),
                                turn([ToolCall("g3", "submit_recommendation", SUBMIT)])], name="anthropic")
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=1, turn_status=True)
    assert rec is not None and [t["phase"] for t in traj] == ["evidence", "grace", "grace", "grace"]
    assert backend.offered == [ALL_TOOLS, ["submit_recommendation"], ["submit_recommendation"]]
    first = backend.seen[1][-1]["content"]
    assert [b["type"] for b in first] == ["tool_result", "text"] and first[0]["tool_use_id"] == "a1"
    assert first[1]["text"].startswith("Turn 1 of 1. Evidence calls used 1 of 24.") and SPENT in first[1]["text"]
    grace = backend.seen[2][-1]["content"]
    assert [b.get("tool_use_id") for b in grace] == ["g1", "g2", None]   # every tool_use is answered
    assert "Evidence turns are spent" in json.loads(grace[0]["content"])["error"]
    assert "1 submission turn left" in grace[2]["text"]


# --- the defaults: no status line, a repeat runs again (AGENT_TURN_STATUS, AGENT_REPEAT_MODE) -----
def test_by_default_the_tool_results_carry_no_status_line(monkeypatch):
    stub_dispatch(monkeypatch)
    backend = RecordingBackend([evidence_turn(2), evidence_turn(1), evidence_turn(1), submit_turn()])
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_steps=3, max_tool_calls=24)
    assert rec is not None
    after = [seen[-1]["content"] for seen in backend.seen[1:]]   # the message closing turns 1, 2, 3
    for content, n in zip(after[:2], (2, 1)):                     # only the results, as before #72
        assert content.startswith("Tool results:\n") and "\n\n" not in content
        assert len(json.loads(content[len("Tool results:\n"):])) == n
    body, note = after[2].split("\n\n")                          # the grace message stays
    assert note.startswith("The evidence turns are spent") and "Turn " not in note


def test_by_default_the_anthropic_tool_results_carry_no_status_line(monkeypatch):
    stub_dispatch(monkeypatch)

    def turn(calls):
        return ModelTurn(tool_calls=calls, raw=SimpleNamespace(content=[{"type": "tool_use", "id": c.id} for c in calls]))
    backend = RecordingBackend([turn([ToolCall("a1", "get_race_snapshot", {}), ToolCall("a2", "estimate_pit_loss", {})]),
                                turn([ToolCall("a3", "get_model_report", {})]),
                                turn([ToolCall("s1", "submit_recommendation", SUBMIT)])], name="anthropic")
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_steps=3)
    assert rec is not None
    first, second = backend.seen[1][-1]["content"], backend.seen[2][-1]["content"]
    assert [(b["type"], b["tool_use_id"]) for b in first] == [("tool_result", "a1"), ("tool_result", "a2")]
    assert [(b["type"], b["tool_use_id"]) for b in second] == [("tool_result", "a3")]


@pytest.mark.parametrize("value", ["1", "true", "on", "ON"])
def test_the_status_line_comes_from_the_environment(monkeypatch, value):
    stub_dispatch(monkeypatch)
    monkeypatch.setenv("AGENT_TURN_STATUS", value)
    backend = RecordingBackend([evidence_turn(1), evidence_turn(1), submit_turn()])
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_steps=3, max_tool_calls=24)
    assert rec is not None
    assert backend.seen[1][-1]["content"].endswith("\n\nTurn 1 of 3. Evidence calls used 1 of 24.")
    assert backend.seen[2][-1]["content"].endswith(
        "\n\nTurn 2 of 3. Evidence calls used 2 of 24. One evidence turn is left after this one.")


def test_the_status_line_argument_overrides_the_environment(monkeypatch):
    stub_dispatch(monkeypatch)
    monkeypatch.setenv("AGENT_TURN_STATUS", "1")
    backend = RecordingBackend([evidence_turn(1), submit_turn()])
    rec, _ = run_agent(snap(), backend=backend, verbose=False, max_steps=3, turn_status=False)
    assert rec is not None and "Turn " not in backend.seen[1][-1]["content"]


def test_by_default_a_repeat_runs_again_and_uses_budget(monkeypatch, tmp_path):
    runs = []
    spec = next(s for s in tools.TOOL_SPECS if s["name"] == "estimate_pit_loss")

    def counted(**kwargs):
        runs.append(kwargs)
        return {"pit_loss_s": 21.5}
    monkeypatch.setitem(spec, "fn", counted)
    path = tmp_path / "trajectory.jsonl"
    stats = {}
    backend = RecordingBackend([ModelTurn(tool_calls=[tc("estimate_pit_loss")]),
                                ModelTurn(tool_calls=[tc("estimate_pit_loss"), tc("estimate_pit_loss")]),
                                submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_tool_calls=2, log_path=path, stats=stats)
    assert rec is not None and len(runs) == 2              # the tool function ran twice
    assert traj[1]["output"] == {"pit_loss_s": 21.5}       # the plain result, no repeat_of in it
    assert traj[1]["repeat_of"] == 1 and "repeat_of" not in traj[0]
    assert "budget exhausted" in traj[2]["error"]          # the next repeat found the budget spent
    assert "repeat_of" not in traj[2]
    assert (stats["evidence_calls"], stats["repeat_calls"]) == (2, 1)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r.get("repeat_of") for r in rows] == [None, 1, None, None]
    results = json.loads(backend.seen[-1][-1]["content"][len("Tool results:\n"):].split("\n\n")[0])
    assert results[0] == {"tool": "estimate_pit_loss", "result": {"pit_loss_s": 21.5}}


def test_by_default_a_repeat_on_the_anthropic_path_runs_again(monkeypatch):
    ran = stub_dispatch(monkeypatch)

    def turn(calls):
        return ModelTurn(tool_calls=calls, raw=SimpleNamespace(content=[{"type": "tool_use", "id": c.id} for c in calls]))
    backend = RecordingBackend([turn([ToolCall("a1", "estimate_pit_loss", {})]),
                                turn([ToolCall("a2", "estimate_pit_loss", {})]),
                                turn([ToolCall("s1", "submit_recommendation", SUBMIT)])], name="anthropic")
    rec, _ = run_agent(snap(), backend=backend, verbose=False)
    assert rec is not None and len(ran) == 2
    content = backend.seen[-1][-1]["content"]
    assert [(b["type"], b["tool_use_id"]) for b in content] == [("tool_result", "a2")]
    assert json.loads(content[0]["content"]) == {"ran": 2}


@pytest.mark.parametrize("env, arg, runs", [("cache", "run", 2), ("run", "cache", 1), (None, "cache", 1)])
def test_the_repeat_mode_argument_overrides_the_environment(monkeypatch, env, arg, runs):
    ran = stub_dispatch(monkeypatch)
    if env:
        monkeypatch.setenv("AGENT_REPEAT_MODE", env)
    backend = RecordingBackend([ModelTurn(tool_calls=[tc("estimate_pit_loss")] * 2), submit_turn()])
    stats = {}
    rec, _ = run_agent(snap(), backend=backend, verbose=False, stats=stats, repeat_mode=arg)
    assert rec is not None and len(ran) == runs
    assert (stats["evidence_calls"], stats["repeat_calls"]) == (runs, 1)


class NoTurns(ModelBackend):
    name = "test"
    def run_turn(self, **kwargs): raise AssertionError("an invalid repeat mode must fail before any turn")


@pytest.mark.parametrize("env, arg", [(None, "skip"), ("refuse", None), ("CACHE", None)])
def test_an_invalid_repeat_mode_raises(monkeypatch, env, arg):
    if env:
        monkeypatch.setenv("AGENT_REPEAT_MODE", env)
    with pytest.raises(ValueError, match="run, cache"):
        run_agent(snap(), backend=NoTurns(), verbose=False, repeat_mode=arg)
