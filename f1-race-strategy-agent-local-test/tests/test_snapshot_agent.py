import json
import os
import tempfile

from src.agent import run_agent
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.snapshot import RaceSnapshot
from src import tools


def snap():
    return RaceSnapshot(season=2026, event="Test GP", lap=30, total_laps=60, driver="TST",
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
