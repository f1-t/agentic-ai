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


# --- turns with no usable tool call: logged, and the reminder says what was wrong ---------------
NO_CALL = "No tool call was detected"
# dev6 run 3, 2021_04_spain:OCO:00, turns 2 to 10: one "}" too many after compare_plans' candidates
GARBLED = ('{"tool_calls": [{"name": "evaluate_strategy", "arguments": {"action": {"action": "pit", '
           '"compound": "MEDIUM"}}}, {"name": "compare_plans", "arguments": {"candidates": [{"action": '
           '"pit", "compound": "MEDIUM"}, {"action": "stay_out", "laps": 3}]}}}]}')
FIXED = GARBLED[:-6] + "]}}]}"


def openai_reply(content, finish_reason="stop", completion_tokens=101):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                    finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=4000, completion_tokens=completion_tokens))


def voyager(replies):
    """The real VoyagerBackend (prompt, parser), its client replaced by scripted chat completions."""
    from src.llm.voyager_backend import VoyagerBackend
    backend = VoyagerBackend()
    replies, backend.seen = iter(replies), []

    def create(**kwargs):
        backend.seen.append(kwargs["messages"])
        return next(replies)
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return backend


def test_the_garbled_reply_is_what_run_3_saw():
    with pytest.raises(json.JSONDecodeError):
        json.loads(GARBLED)
    assert len(json.loads(FIXED)["tool_calls"]) == 2


@pytest.mark.parametrize("tail, fixed", [
    ("}]}}}]}", "}]}}]}"),        # one brace too many (run 3's stall)
    ("}]}]}", "}]}}]}"),          # one too few in the run
    ("}]}}]", "}]}}]}"),          # the last one missing
    ("}]}}]}}", "}]}}]}"),        # a stray one after the end
    ("}]}}]}\n}", "}]}}]}"),      # whitespace in the run
])
def test_only_the_final_run_of_closers_is_rewritten(tail, fixed):
    from src.llm.voyager_backend import _reclose_tail
    head = FIXED[:-6]
    assert head.endswith('"laps": 3')
    assert _reclose_tail(head + tail) == (head + fixed, "".join(tail.split()), fixed)
    assert _reclose_tail(FIXED) is None                                   # nothing to change


@pytest.mark.parametrize("text", [
    '{"a": [{"x": 1}}, {"y": 2}]}',                # a wrong closer before the end: two readings
    '{"a": {"b": [1, 2}, "c": 3}}',
    '{"tool_calls": [{"name": "x", "arguments": {"note": "unclosed}]}',   # ends inside a string
])
def test_a_closer_out_of_place_before_the_end_is_not_guessed(text):
    from src.llm.voyager_backend import _reclose_tail
    assert _reclose_tail(text) is None


def test_brackets_inside_strings_are_not_counted():
    from src.llm.voyager_backend import _reclose_tail
    text = '{"tool_calls": [{"name": "x", "arguments": {"note": "a } and \\" ] here"}}]}}'
    fixed, old, new = _reclose_tail(text)
    assert json.loads(fixed)["tool_calls"][0]["arguments"]["note"] == 'a } and " ] here'
    assert (old, new) == ("}}]}}", "}}]}")


def test_a_turn_with_no_tool_call_is_logged_but_is_not_a_trajectory_event(monkeypatch, tmp_path):
    stub_dispatch(monkeypatch)
    path = tmp_path / "t.jsonl"
    backend = RecordingBackend([ModelTurn(text="I think staying out is best.",
                                          raw={"usage": {"prompt_tokens": 9, "completion_tokens": 7},
                                               "choices": [{"finish_reason": "stop"}]}),
                                evidence_turn(1), submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, log_path=path)
    assert rec is not None and [t["step"] for t in traj] == [2, 3]   # the trajectory is the calls only
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    empty = rows[0]
    assert {k: empty[k] for k in ("step", "phase", "event", "text", "text_chars", "parse_error",
                                  "finish_reason", "output_tokens")} == {
        "step": 1, "phase": "evidence", "event": "no_tool_call", "text": "I think staying out is best.",
        "text_chars": 28, "parse_error": None, "finish_reason": "stop", "output_tokens": 7}
    assert "tool" not in empty and "call_index" not in empty
    assert [r.get("tool") for r in rows[1:]] == ["get_race_snapshot", "submit_recommendation"]
    # a deliberate non-answer gets the reminder it always got
    assert backend.seen[1][-1] == {"role": "user", "content": NO_CALL + ". Respond with a valid tool call; "
                                                                        "submit if you have enough evidence."}


def test_a_long_reply_is_clipped_in_the_log_head_and_tail(monkeypatch, tmp_path):
    stub_dispatch(monkeypatch)
    path = tmp_path / "t.jsonl"
    text = "a" * 3000 + "THE_END"
    run_agent(snap(), backend=RecordingBackend([ModelTurn(text=text), submit_turn()]), verbose=False,
              log_path=path)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["text_chars"] == 3007 and len(row["text"]) < 1600
    assert row["text"].startswith("aaa") and row["text"].endswith("THE_END") and "chars ...]" in row["text"]
    assert row["finish_reason"] is None and row["output_tokens"] is None   # nothing reported


EVIDENCE = ('{"tool_calls": [{"name": "get_model_report", "arguments": {}}, {"name": '
            '"estimate_tyre_degradation", "arguments": {"compound": "SOFT"}}, '
            '{"name": "estimate_pit_loss", "arguments": {}}]}')
SUBMIT_JSON = json.dumps({"tool_calls": [{"name": "submit_recommendation", "arguments": SUBMIT}]})


def test_run_3s_stalled_episode_now_runs_its_plan_evaluation(tmp_path):
    # run 3, 2021_04_spain:OCO:00: evidence in turn 1, then GARBLED in each of turns 2 to 10 (no
    # call ran) and a grace-turn PIT with no plan evaluated. The final run of closers is now read
    # as the one that closes the JSON, so turn 2's calls run and it submits in turn 3.
    backend = voyager([openai_reply(EVIDENCE), openai_reply(GARBLED), openai_reply(SUBMIT_JSON)])
    path = tmp_path / "t.jsonl"
    rec, traj = run_agent(snap(), backend=backend, verbose=False, log_path=path)
    assert rec is not None and rec["decision"] == "STAY_OUT"
    assert [(t["step"], t["phase"], t["tool"]) for t in traj] == [
        (1, "evidence", "get_model_report"), (1, "evidence", "estimate_tyre_degradation"),
        (1, "evidence", "estimate_pit_loss"), (2, "evidence", "evaluate_strategy"),
        (2, "evidence", "compare_plans"), (3, "evidence", "submit_recommendation")]
    assert all(t["error"] is None for t in traj)
    assert traj[4]["input"]["candidates"][-1] == {"action": "stay_out", "laps": 3}
    assert [t.get("parse_repair") for t in traj[3:5]] == ["final closing brackets '}]}}}]}' read as '}]}}]}'"] * 2
    assert "parse_repair" not in traj[0] and "parse_repair" not in traj[5]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert not any("event" in r for r in rows) and rows[3]["parse_repair"]
    # the model's history holds the calls as they ran, re-serialised
    assert json.loads(backend.seen[2][-2]["content"]) == json.loads(FIXED)


# run 3-like, 2022_02_saudi_arabia:OCO:05 before the fix: two whole envelopes in one reply
TWO_ENVELOPES = ('{"tool_calls": [{"name": "evaluate_strategy", "arguments": {"action": {"action": "pit", '
                 '"compound": "MEDIUM"}}}]}\n\n{"tool_calls": [{"name": "compare_plans", "arguments": '
                 '{"candidates": [{"action": "pit", "compound": "MEDIUM"}, {"action": "stay_out", "laps": 3}]}}]}')


def test_an_unparsed_reply_is_logged_and_the_model_is_told_what_was_wrong(tmp_path):
    backend = voyager([openai_reply(EVIDENCE), openai_reply(TWO_ENVELOPES), openai_reply(FIXED),
                       openai_reply(SUBMIT_JSON)])
    path = tmp_path / "t.jsonl"
    rec, traj = run_agent(snap(), backend=backend, verbose=False, log_path=path)
    assert rec is not None and not any(t.get("parse_repair") for t in traj)
    assert [(t["step"], t["tool"]) for t in traj][3:] == [(3, "evaluate_strategy"), (3, "compare_plans"),
                                                          (4, "submit_recommendation")]
    reminder = backend.seen[2][-1]
    assert reminder["role"] == "user" and NO_CALL not in reminder["content"]
    assert "not valid JSON" in reminder["content"] and "Extra data" in reminder["content"]
    assert "one corrected JSON object" in reminder["content"]
    assert "followed by '{\"tool_calls\": [{\"na'" in reminder["content"]   # where it went wrong
    empty = [json.loads(x) for x in path.read_text().splitlines()][3]
    assert empty["event"] == "no_tool_call" and empty["step"] == 2 and empty["finish_reason"] == "stop"
    assert empty["output_tokens"] == 101 and empty["parse_error"].startswith("JSONDecodeError: Extra data")
    assert empty["text"].startswith(TWO_ENVELOPES)   # the backend's text, its parse-error note after it


def test_a_reply_cut_off_at_the_output_limit_is_told_so_and_not_completed(tmp_path):
    # well nested so far: closing it would parse, but a cut-off reply is unfinished, not mis-closed
    cut = '{"tool_calls": [{"name": "compare_plans", "arguments": {"candidates": [{"action": "pit"}'
    backend = voyager([openai_reply(cut, "length", 1024), openai_reply(SUBMIT_JSON)])
    path = tmp_path / "t.jsonl"
    rec, traj = run_agent(snap(), backend=backend, verbose=False, log_path=path)
    assert rec is not None and [t["tool"] for t in traj] == ["submit_recommendation"]
    reminder = backend.seen[1][-1]["content"]
    assert "cut off at the output limit" in reminder and "fewer tool calls" in reminder
    empty = json.loads(path.read_text().splitlines()[0])
    assert (empty["finish_reason"], empty["output_tokens"]) == ("length", 1024) and empty["parse_error"]


def test_the_parse_error_reminder_keeps_the_grace_note(monkeypatch):
    stub_dispatch(monkeypatch)
    backend = RecordingBackend([evidence_turn(1), ModelTurn(text="{", parse_error="JSONDecodeError: x"),
                                submit_turn()])
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=2)
    assert rec is not None and traj[-1]["phase"] == "grace"
    last = backend.seen[-1][-1]["content"]
    assert "not valid JSON" in last and "JSONDecodeError: x" in last and SPENT in last


def test_a_cut_off_anthropic_turn_is_told_so(monkeypatch):
    stub_dispatch(monkeypatch)
    raw = SimpleNamespace(content=[], stop_reason="max_tokens", usage=SimpleNamespace(input_tokens=5, output_tokens=4000))
    submit = ModelTurn(tool_calls=[tc("submit_recommendation", SUBMIT)],
                       raw=SimpleNamespace(content=[{"type": "tool_use", "id": "id"}]))
    backend = RecordingBackend([ModelTurn(raw=raw), submit], name="anthropic")
    rec, _ = run_agent(snap(), backend=backend, verbose=False)
    assert rec is not None
    assert "cut off at the output limit" in backend.seen[1][-1]["content"]
