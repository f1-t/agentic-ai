"""A scripted model backend for tests/test_eval_rerun_agent.py, importable by name in spawned
workers ("agent_pass_scripted:make"). Per scenario (Runner calls on_scenario): one evidence call,
then a submission that depends on the scenario only (PIT on MEDIUM at an odd lap, else STAY_OUT).
Failures are injected through the environment, which spawned workers inherit:

    AGENT_PASS_TEST_FAIL=<id>    every turn of that scenario raises RuntimeError (an API error)
    AGENT_PASS_TEST_FLAKY=<id>   its first turn raises a 503 once, then the script runs
    AGENT_PASS_TEST_AUTH=<id>    every turn raises a 401
    AGENT_PASS_TEST_CRASH=<id>   the worker process exits hard (os._exit)
"""
import os

from src.llm.base import ModelBackend, ModelTurn, ToolCall


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class Scripted(ModelBackend):
    name = "test"
    model_id = "scripted"

    def __init__(self):
        self.sid, self.turn, self.flaked = None, 0, set()

    def on_scenario(self, sid):
        self.sid, self.turn = sid, 0

    def run_turn(self, **kw):
        env = os.environ.get
        if self.sid == env("AGENT_PASS_TEST_CRASH"):
            os._exit(3)
        if self.sid == env("AGENT_PASS_TEST_FAIL"):
            raise RuntimeError("injected API failure")
        if self.sid == env("AGENT_PASS_TEST_AUTH"):
            raise StatusError(401)
        if self.sid == env("AGENT_PASS_TEST_FLAKY") and self.sid not in self.flaked:
            self.flaked.add(self.sid)
            raise StatusError(503)
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(tool_calls=[ToolCall("1", "get_race_snapshot", {})])
        lap = int(self.sid.rsplit(":", 1)[1])
        sub = ({"decision": "PIT", "compound": "MEDIUM"} if lap % 2 else {"decision": "STAY_OUT"})
        return ModelTurn(tool_calls=[ToolCall("2", "submit_recommendation", {
            **sub, "reassess_after_laps": 3, "confidence": 0.6, "rationale": "scripted", "key_risks": []})])


def make():
    return Scripted()
