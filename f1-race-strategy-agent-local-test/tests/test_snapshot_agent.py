import json
import os
import tempfile
from types import SimpleNamespace

from src import agent
from src.agent import run_agent
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.snapshot import RaceSnapshot
from src import tools


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
    rec, traj = run_agent(snap(), backend=backend, verbose=False, max_steps=1)
    assert rec is None
    assert len(traj) == 1


# --- the evidence budget: a submission is always processed, a turn's results are kept ---------
SUBMIT = {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": .6,
          "rationale": "Enough evidence.", "key_risks": ["traffic"]}


class RecordingBackend(ScriptedBackend):
    """ScriptedBackend that keeps a copy of the messages it was shown on each turn."""
    def __init__(self, turns, name="test"):
        super().__init__(turns)
        self.name = name
        self.seen = []

    def run_turn(self, **kwargs):
        self.seen.append(list(kwargs["messages"]))
        return super().run_turn(**kwargs)


def evidence_turn(n):
    return ModelTurn(tool_calls=[ToolCall(f"e{i}", "get_race_snapshot", {}) for i in range(n)])


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
    first = [ToolCall("a1", "get_race_snapshot", {}), ToolCall("a2", "get_race_snapshot", {})]
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
