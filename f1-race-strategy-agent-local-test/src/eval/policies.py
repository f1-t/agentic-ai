"""Policies the eval scores. A policy maps a scenario to a plan for the remaining laps.

    policy(scenario, belief, max_stops) -> Decision      raises PolicyUnavailable to skip

`belief` is the belief simulator (docs/eval.md "Who sees what"): fitted only on what is
known at the snapshot. Policies plan on it; the world simulator that scores them and plans
the reference is not passed in. The four baselines are Quan's reference policies
(src/baselines) run through the optimizer bridge on the belief's fit. `recorded_policy`
replays plans saved from earlier agent runs, and `agent_policy` (an AgentPolicy) asks the
live agent loop (src/agent.py) through whatever LLM_BACKEND is configured, at any lap of any
race: it sees the scenario's snapshot through RaceSnapshot.from_contract, and its
submission (PIT or STAY_OUT) is scored as an action plus a continuation (eval.decisions).

Policies whose rows are decisions rather than baseline plans (the agent, recorded runs)
carry `agent_decisions = True`; score_set warns when they meet a no-stop reference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..baselines import domain, policies
from ..baselines.domain import PolicyUnavailable
from ..sim.optimizer_bridge import from_contract, pace_for_crossover, to_contract

BASELINES = ("actual_strategy", "fixed_one_stop", "fixed_two_stop", "greedy_crossover")
TRAJECTORY_DIR = Path(__file__).resolve().parents[2] / "reports" / "eval" / "trajectories"


@dataclass
class Decision:
    """What a policy decided. `plan` is anything contracts.Plan.of accepts, or None when the
    policy produced no plan (an invalid decision). `submission` is an agent decision
    (PIT or STAY_OUT, eval.decisions) given in place of a plan: the scorer expands it on
    the belief simulator into an action plus a continuation. `error` marks a failure of the
    policy itself (an API error, say), which is reported apart from invalid plans."""

    plan: object
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)
    submission: dict | None = None


def _bridged(choose):
    """A baseline that plans on the bridge's small snapshot of the belief's fit and returns a
    contracts.Plan."""
    def policy(scenario, belief, max_stops):
        state = from_contract(scenario.snapshot, belief)
        plan = choose(scenario, state, belief, max_stops)
        if plan.stints[0].laps == 0:
            raise PolicyUnavailable("The plan stops immediately; a first stint needs at least one lap "
                                    "(docs/rules.md R1).")
        return Decision(to_contract(plan, scenario.snapshot, sources=belief.config.inventory_sources))
    return policy


BASELINE_POLICIES = {
    "actual_strategy": _bridged(lambda sc, st, sim, m: policies.actual_strategy(
        st, [domain.Stint(c, n) for c, n in sc.actual_stints])),
    "fixed_one_stop": _bridged(lambda sc, st, sim, m: policies.fixed_one_stop(st)),
    "fixed_two_stop": _bridged(lambda sc, st, sim, m: policies.fixed_two_stop(st)),
    "greedy_crossover": _bridged(lambda sc, st, belief, m: policies.greedy_crossover(
        st, pace_for_crossover(belief, sc.snapshot), m)),
}


def recorded_policy(path):
    """(name, policy) replaying a json file of saved decisions:

        {"name": "...", "source": "...", "decisions": {"<scenario_id>": {"stint_plan": [...],
         "tool_calls": 12, "input_tokens": null, "output_tokens": null}}}

    Scenarios the file has no decision for are unavailable."""
    data = json.loads(Path(path).read_text())
    decisions = data["decisions"]

    def policy(scenario, sim, max_stops):
        rec = decisions.get(scenario.scenario_id)
        if rec is None:
            raise PolicyUnavailable(f"no recorded decision for {scenario.scenario_id}")
        return Decision(rec.get("stint_plan"), tool_calls=rec.get("tool_calls"),
                        input_tokens=rec.get("input_tokens"), output_tokens=rec.get("output_tokens"))
    policy.agent_decisions = True
    return data["name"], policy


class AgentPolicy:
    """The live agent on the snapshot loop, as an eval policy.

    The agent's view is RaceSnapshot.from_contract(scenario.snapshot); run_agent gets
    max_steps (profile.agent.max_steps unless given) and logs its trajectory to
    <log_dir>/<set_name>/<scenario_id>.jsonl, ":" in the id written "_" (a rerun replaces
    the file). The submission goes back as Decision.submission, which the scorer expands on
    the belief (eval.decisions.to_plan). tool_calls is the trajectory's length, tokens are
    the backend's reported usage (None when it reports none), and a backend exception is a
    status "error" row, not a crash. `backend` is for tests; None takes LLM_BACKEND."""

    agent_decisions = True

    def __init__(self, *, set_name: str = "adhoc", max_steps: int | None = None, profile=None,
                 log_dir=TRAJECTORY_DIR, backend=None):
        self.set_name, self.max_steps, self.profile = set_name, max_steps, profile
        self.log_dir, self.backend = Path(log_dir), backend

    def log_path(self, scenario) -> Path:
        return self.log_dir / self.set_name / f"{scenario.scenario_id.replace(':', '_')}.jsonl"

    def __call__(self, scenario, belief, max_stops):
        from ..agent import run_agent
        from ..sim.compute import load_profile
        from ..snapshot import RaceSnapshot

        max_steps = self.max_steps or (self.profile or load_profile()).agent.max_steps
        path = self.log_path(scenario)
        path.unlink(missing_ok=True)
        stats = {}
        try:
            rec, trajectory = run_agent(RaceSnapshot.from_contract(scenario.snapshot), verbose=False,
                                        backend=self.backend, max_steps=max_steps, log_path=path, stats=stats)
        except Exception as exc:  # noqa: BLE001  a backend failure must not end an overnight run
            return Decision(None, error=f"{type(exc).__name__}: {exc}", input_tokens=stats.get("input_tokens"),
                            output_tokens=stats.get("output_tokens"))
        return Decision(None, submission=rec, tool_calls=len(trajectory), input_tokens=stats.get("input_tokens"),
                        output_tokens=stats.get("output_tokens"),
                        meta={"submitted": rec is not None, "steps": stats.get("steps"), "trajectory": str(path)})


agent_policy = AgentPolicy()


def resolve(names, recorded=(), *, set_name: str | None = None, profile=None) -> dict:
    """Policies by name. "baselines" expands to the four reference policies, "agent" is the
    live agent (an AgentPolicy logging under `set_name`, with `profile`'s agent budget),
    and each recorded file adds the policy it names."""
    out = {}
    for name in names:
        if name == "baselines":
            out.update(BASELINE_POLICIES)
        elif name == "agent":
            out["agent"] = AgentPolicy(set_name=set_name or "adhoc", profile=profile)
        elif name in BASELINE_POLICIES:
            out[name] = BASELINE_POLICIES[name]
        else:
            raise KeyError(f"unknown policy {name!r}; choose from baselines, agent, {', '.join(BASELINES)}")
    for path in recorded:
        name, policy = recorded_policy(path)
        out[name] = policy
    return out
