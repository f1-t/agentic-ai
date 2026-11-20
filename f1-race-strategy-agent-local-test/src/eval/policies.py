"""Policies the eval scores. A policy maps a scenario to a plan for the remaining laps.

    policy(scenario, belief, max_stops) -> Decision      raises PolicyUnavailable to skip

`belief` is the belief simulator (docs/eval.md "Who sees what"): fitted only on what is
known at the snapshot. Policies plan on it; the world simulator that scores them and plans
the reference is not passed in. The four baselines are Quan's reference policies
(src/baselines) run through the optimizer bridge on the belief's fit; under the R8 stint
limit this wrapper makes fixed_one_stop and greedy_crossover limit-aware (_bridged), with
src/baselines unchanged. `recorded_policy`
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
from ..sim.rules import stint_limit_violations

BASELINES = ("actual_strategy", "fixed_one_stop", "fixed_two_stop", "greedy_crossover")
TRAJECTORY_DIR = Path(__file__).resolve().parents[2] / "reports" / "eval" / "trajectories"


@dataclass
class Decision:
    """What a policy decided. `plan` is anything contracts.Plan.of accepts, or None when the
    policy produced no plan (an invalid decision). `submission` is an agent decision
    (PIT or STAY_OUT, eval.decisions) given in place of a plan: the scorer expands it on
    the belief simulator into an action plus a continuation. `error` marks a failure of the
    policy itself (an API error, say), which is reported apart from invalid plans.
    `submitted_in_grace` is True when an agent's submission came in a grace turn after its
    evidence turns ran out (src/agent.py), False for any other agent run, None otherwise.
    `tool_calls` counts every attempted call; `evidence_calls` only the evidence calls that
    ran (no submissions, no refused calls, no repeats); `repeat_calls` the repeated evidence
    calls served from the cache."""

    plan: object
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)
    submission: dict | None = None
    submitted_in_grace: bool | None = None
    evidence_calls: int | None = None
    repeat_calls: int | None = None
    belief_optimum: bool = False   # no plan given: the scorer plans the belief optimum (eval.decisions.belief_optimum)


def submitted_in_grace(rec, trajectory) -> bool:
    """True when run_agent's accepted submission (the trajectory's last event) came in a
    grace turn."""
    return rec is not None and bool(trajectory) and trajectory[-1].get("phase") == "grace"


PAST_LIMIT_S = 1e6   # the lap time greedy_crossover sees on a set past its R8 limit


def _stint_limit(belief):
    """(limits, margin) of R8 when the belief applies it, else None (docs/rules.md R8)."""
    margin, limits = belief.config.stint_limit_margin_laps, getattr(belief, "stint_limits", None)
    if margin is None or not limits or "stint_limit" not in belief.config.rules:
        return None
    return limits, margin


def _bridged(choose, within_limit=None):
    """A baseline that plans on the bridge's small snapshot of the belief's fit and returns a
    contracts.Plan.

    R8 (docs/rules.md) is a simulator rule the baselines in src/baselines do not plan around,
    and they stay unchanged: the adjustment lives here, in the eval wrapper. When the
    baseline's plan breaks R8, `within_limit(scenario, state, belief, max_stops, plan, limit)`
    gives the policy's limit-aware plan (a baselines Plan, or None when there is none), which
    replaces it. Whatever plan comes out is scored by the world, which checks every rule, so a
    baseline plan that still breaks R8 is an invalid row, as an agent's is. With R8 off
    (stint_limit_margin_laps None) the plan is the baseline's own."""
    def policy(scenario, belief, max_stops):
        snap = scenario.snapshot
        state = from_contract(snap, belief)
        plan = choose(scenario, state, belief, max_stops)
        if plan.stints[0].laps == 0:
            raise PolicyUnavailable("The plan stops immediately; a first stint needs at least one lap "
                                    "(docs/rules.md R1).")
        sources = belief.config.inventory_sources
        contract = to_contract(plan, snap, sources=sources)
        limit = _stint_limit(belief)
        if limit is None or not stint_limit_violations(snap, contract, *limit):
            return Decision(contract)
        adjusted = None
        if within_limit is not None:
            try:
                adjusted = within_limit(scenario, state, belief, max_stops, plan, limit)
            except PolicyUnavailable:
                adjusted = None
        if adjusted is None or adjusted.stints[0].laps == 0 or adjusted == plan:
            return Decision(contract, meta={"stint_limit_adjusted": False})
        return Decision(to_contract(adjusted, snap, sources=sources), meta={"stint_limit_adjusted": True})
    return policy


def _one_stop_within_limit(scenario, state, belief, max_stops, plan, limit):
    """fixed_one_stop under R8: the same compounds, the first stop moved earlier, to the last
    lap at which the whole plan passes every rule on the belief; None when no lap does."""
    snap, sources = scenario.snapshot, belief.config.inventory_sources
    (first, second), remaining = plan.stints, state.remaining_laps
    for laps in range(first.laps - 1, 0, -1):
        moved = domain.Plan((domain.Stint(first.compound, laps), domain.Stint(second.compound, remaining - laps)))
        if not belief.validate(snap, to_contract(moved, snap, sources=sources)):
            return moved
    return None


def _crossover_within_limit(scenario, state, belief, max_stops, plan, limit):
    """greedy_crossover under R8: the same policy with a stop forced on the lap its set would
    pass the limit. It reads the belief's pace, except that a lap on a set older than its
    limit costs PAST_LIMIT_S, so the crossover test fires there whenever a stop is allowed
    (a stop left, R2). A set already past its limit pits at its first legal stop."""
    limits, margin = limit
    pace = pace_for_crossover(belief, scenario.snapshot)

    def bounded(compound, age):
        seen = limits.get(compound)
        return PAST_LIMIT_S if seen is not None and age > seen + margin else pace(compound, age)
    return policies.greedy_crossover(state, bounded, max_stops)


BASELINE_POLICIES = {
    "actual_strategy": _bridged(lambda sc, st, sim, m: policies.actual_strategy(
        st, [domain.Stint(c, n) for c, n in sc.actual_stints])),
    "fixed_one_stop": _bridged(lambda sc, st, sim, m: policies.fixed_one_stop(st), _one_stop_within_limit),
    "fixed_two_stop": _bridged(lambda sc, st, sim, m: policies.fixed_two_stop(st)),
    "greedy_crossover": _bridged(lambda sc, st, belief, m: policies.greedy_crossover(
        st, pace_for_crossover(belief, sc.snapshot), m), _crossover_within_limit),
}


def belief_optimizer(scenario, belief, max_stops):
    """The optimizer on what the agent can see, with no model call: the scorer plans the belief's
    optimal full plan from the snapshot (eval.decisions.belief_optimum, the agent
    continuation's belief, seed and rules, nothing imposed) and the world scores it. Not in
    "baselines"; select it by name."""
    return Decision(None, belief_optimum=True)


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

    run_agent gets the scenario's contracts.Snapshot, which the tools keep (their belief
    simulator reads the stint history and the sets); the model sees its
    RaceSnapshot.from_contract view. run_agent gets max_steps (profile.agent.max_steps
    unless given) and logs its trajectory to <log_dir>/<set_name>/<scenario_id>.jsonl, ":"
    in the id written "_" (a rerun replaces the file). The submission goes back as Decision.submission, which the scorer expands on
    the belief (eval.decisions.to_plan). tool_calls is the trajectory's length: every
    attempted call, refused ones (budget, grace), repeats and submission attempts included;
    evidence_calls counts only the evidence calls that ran, repeat_calls the repeats served
    from the cache. submitted_in_grace says the submission came in a grace turn. Tokens are the backend's
    reported usage (None when it reports none), and a backend exception is a status "error"
    row, not a crash. `backend` is for tests; None takes LLM_BACKEND."""

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

        max_steps = self.max_steps or (self.profile or load_profile()).agent.max_steps
        path = self.log_path(scenario)
        path.unlink(missing_ok=True)
        stats = {}
        try:
            rec, trajectory = run_agent(scenario.snapshot, verbose=False, backend=self.backend,
                                        max_steps=max_steps, log_path=path, stats=stats)
        except Exception as exc:  # noqa: BLE001  a backend failure must not end an overnight run
            return Decision(None, error=f"{type(exc).__name__}: {exc}", input_tokens=stats.get("input_tokens"),
                            output_tokens=stats.get("output_tokens"))
        return Decision(None, submission=rec, tool_calls=len(trajectory), input_tokens=stats.get("input_tokens"),
                        output_tokens=stats.get("output_tokens"),
                        submitted_in_grace=submitted_in_grace(rec, trajectory),
                        evidence_calls=stats.get("evidence_calls"), repeat_calls=stats.get("repeat_calls"),
                        meta={"submitted": rec is not None, "steps": stats.get("steps"), "trajectory": str(path)})


agent_policy = AgentPolicy()


def resolve(names, recorded=(), *, set_name: str | None = None, profile=None) -> dict:
    """Policies by name. "baselines" expands to the four reference policies, "belief_optimizer" is the optimizer on
    the belief (apart from the group), "agent" is the
    live agent (an AgentPolicy logging under `set_name`, with `profile`'s agent budget),
    and each recorded file adds the policy it names."""
    out = {}
    for name in names:
        if name == "baselines":
            out.update(BASELINE_POLICIES)
        elif name == "agent":
            out["agent"] = AgentPolicy(set_name=set_name or "adhoc", profile=profile)
        elif name == "belief_optimizer":
            out[name] = belief_optimizer
        elif name in BASELINE_POLICIES:
            out[name] = BASELINE_POLICIES[name]
        else:
            raise KeyError(f"unknown policy {name!r}; choose from baselines, agent, belief_optimizer, "
                           f"{', '.join(BASELINES)}")
    for path in recorded:
        name, policy = recorded_policy(path)
        out[name] = policy
    return out
