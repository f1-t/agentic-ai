"""Policies the eval scores. A policy maps a scenario to a plan for the remaining laps.

    policy(scenario, belief, max_stops) -> Decision      raises PolicyUnavailable to skip

`belief` is the belief simulator (docs/eval.md "Who sees what"): fitted only on what is
known at the snapshot. Policies plan on it; the world simulator that scores them and plans
the reference is not passed in. The four baselines are Quan's reference policies
(src/baselines) run through the optimizer bridge on the belief's fit, src/baselines unchanged.
The wrapper makes fixed_one_stop and greedy_crossover tyre-life-aware (_bridged): by default
with the belief's as-of tyre-life estimate at the snapshot (the support edge of its wear cliff,
sim/cliff.py, the information every policy has), under the retired R8 setting with R8's limit.
A baseline plan is still invalid only for a published-regulation violation: the world prices
tyre life, it never rejects a plan for it. `belief_optimizer` and the rule baselines `run_to_flag` and
`one_stop_total` are named policies outside the baselines group. `recorded_policy`
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
from itertools import product
from pathlib import Path

from ..baselines import domain, policies
from ..baselines.domain import PolicyUnavailable
from ..sim.optimizer_bridge import from_contract, pace_for_crossover, to_contract
from ..sim.rules import assign_sets, stint_limit_violations

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
    ran (no submissions, no refused calls, no cached repeats; a re-run repeat counts);
    `repeat_calls` the evidence calls identical to an earlier one, re-run or served from the
    cache (AGENT_REPEAT_MODE)."""

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


def _stint_limit(belief, snapshot=None):
    """(limits, margin) a rule baseline plans its tyre life with, else None: R8's (limits, margin)
    under the legacy setting (docs/rules.md R8); otherwise, with the wear cliff on, the belief's
    as-of support edge per compound at the snapshot (Simulator.cliffs_for: the point estimate,
    max(ages seen so far, the pre-race prior), nothing after the snapshot), margin 0."""
    margin, limits = belief.config.stint_limit_margin_laps, getattr(belief, "stint_limits", None)
    if margin is not None and limits and "stint_limit" in belief.config.rules:
        return limits, margin
    if snapshot is None or not belief.config.wear_cliff.enabled:
        return None
    edges = {c: float(cl.age) for c, cl in belief.cliffs_for(snapshot).items()}
    return (edges, 0) if edges else None


def _legal(belief, snapshot, contract, limit) -> bool:
    """Every rule passes on the belief, and the plan keeps inside `limit` when there is one."""
    return not belief.validate(snapshot, contract) and (
        limit is None or not stint_limit_violations(snapshot, contract, *limit))


def _no_set(belief, snapshot, contract):
    """SimConfig.tyre_inventory: the first inventory error of `contract` (a stint on a compound
    with no set left), else None."""
    if not getattr(belief.config, "tyre_inventory", False) or snapshot.sets_available is None:
        return None
    errors = assign_sets(snapshot, contract, belief.config.inventory_sources, best=True)[1]
    return errors[0] if errors else None


def _bridged(choose, within_limit=None, inventory_check=True):
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
        # SimConfig.tyre_inventory: a rule that names a compound with no set left has no plan here
        missing = _no_set(belief, snap, contract) if inventory_check else None
        if missing is not None:
            raise PolicyUnavailable(f"No tyre set for the rule's plan: {missing}")
        limit = _stint_limit(belief, snap)
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
        moved = to_contract(adjusted, snap, sources=sources)
        if inventory_check and _no_set(belief, snap, moved) is not None:
            return Decision(contract, meta={"stint_limit_adjusted": False})
        return Decision(moved, meta={"stint_limit_adjusted": True})
    return policy


def _real_ages(policy):
    """actual_strategy under SimConfig.tyre_inventory: each later stint on its recorded set's age
    (data.race_stints: the set's laps run when fitted), so the scorer's inventory matches the real
    set (exactly, else the nearest held one; a set the plan took off earlier may be refitted). A
    recorded strategy the inventory still cannot fit is unavailable, with the cause."""
    def wrapped(scenario, belief, max_stops):
        d = policy(scenario, belief, max_stops)
        snap = scenario.snapshot
        if not getattr(belief.config, "tyre_inventory", False) or d.plan is None:
            return d
        from ..contracts import Plan, PlanStint
        from ..sim.data import race_stints
        later = [r for r in race_stints(belief.laps, scenario.driver) if r[3] > snap.laps_completed]
        plan = Plan.of(d.plan)
        stints = [PlanStint(compound=st.compound, laps=st.laps,
                            start_age=(later[j - 1][2] if j > 0 and j - 1 < len(later)
                                       and later[j - 1][0] == st.compound else 0))
                  for j, st in enumerate(plan.stints)]
        aged = Plan(stints=stints)
        missing = _no_set(belief, snap, aged)
        if missing is not None:
            raise PolicyUnavailable(f"The recorded strategy needs a set the inventory does not hold: {missing}")
        return Decision(aged, meta=d.meta)
    return wrapped


def _one_stop_within_limit(scenario, state, belief, max_stops, plan, limit):
    """fixed_one_stop under R8: the same compounds, the first stop moved earlier, to the last
    lap at which the whole plan passes every rule on the belief; None when no lap does."""
    snap, sources = scenario.snapshot, belief.config.inventory_sources
    (first, second), remaining = plan.stints, state.remaining_laps
    for laps in range(first.laps - 1, 0, -1):
        moved = domain.Plan((domain.Stint(first.compound, laps), domain.Stint(second.compound, remaining - laps)))
        if _legal(belief, snap, to_contract(moved, snap, sources=sources), limit):
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


def _actual_to_distance(actual_stints, total_laps):
    """The driver's recorded stints as domain.Stints, the last one run on to `total_laps` when
    the car covered fewer laps (lapped, or retired): a snapshot carries the scheduled distance
    (calibration.snapshot_as_of), and the comparator keeps the last set to the flag."""
    stints = [domain.Stint(c, n) for c, n in actual_stints]
    short = total_laps - sum(s.laps for s in stints)
    if stints and short > 0:
        stints[-1] = domain.Stint(stints[-1].compound, stints[-1].laps + short)
    return stints


BASELINE_POLICIES = {
    "actual_strategy": _real_ages(_bridged(lambda sc, st, sim, m: policies.actual_strategy(
        st, _actual_to_distance(sc.actual_stints, st.total_laps)), inventory_check=False)),
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


COMPOUND_PREFERENCE = ("SOFT", "MEDIUM", "HARD")   # run_to_flag's forced stops, softest first


def _preferred(compounds):
    """`compounds` in run_to_flag's order: SOFT, MEDIUM, HARD, then any other alphabetically."""
    rank = {c: i for i, c in enumerate(COMPOUND_PREFERENCE)}
    return sorted(compounds, key=lambda c: (rank.get(c, len(rank)), c))


def _latest_stops(remaining: int, stops: int):
    """Stint lengths of every `stops`-stop split of `remaining` laps, each stint at least one
    lap, latest pit laps first (the first stop compared first, then the second, ...)."""
    if stops == 0:
        yield (remaining,)
        return
    for first in range(remaining - stops, 0, -1):
        for rest in _latest_stops(remaining - first, stops - 1):
            yield (first, *rest)


def run_to_flag(scenario, belief, max_stops):
    """No further stop when the rules allow it; otherwise the fewest stops (at most
    `max_stops`) that make the plan legal, each as late as the rules allow, the new stints on
    the softest compound that keeps it legal (docs/eval.md "Rule baselines"). Legal means
    every rule passes on the belief (belief.validate, R1-R4, known compounds) and no stint passes
    the belief's as-of tyre-life estimate (_stint_limit; R8's limit under the legacy setting). With no
    legal plan the current set runs to the flag and the world rejects it (an invalid row).
    Reads the snapshot, the belief's modelled compounds and the rules, no pace. Not in
    "baselines"; select it by name."""
    snap = scenario.snapshot
    sources = belief.config.inventory_sources
    state = from_contract(snap, belief)
    limit = _stint_limit(belief, snap)
    flag = to_contract(domain.Plan((domain.Stint(state.current_compound, state.remaining_laps),)), snap,
                       sources=sources)
    if _legal(belief, snap, flag, limit):
        return Decision(flag, meta={"forced_stops": 0})
    compounds = _preferred(state.available_compounds)
    for stops in range(1, max_stops + 1):
        for lengths in _latest_stops(state.remaining_laps, stops):
            for later in product(compounds, repeat=stops):
                plan = domain.Plan(tuple(domain.Stint(c, n) for c, n in
                                         zip((state.current_compound, *later), lengths)))
                contract = to_contract(plan, snap, sources=sources)
                if _legal(belief, snap, contract, limit):
                    return Decision(contract, meta={"forced_stops": stops})
    return Decision(flag, meta={"forced_stops": None})


def one_stop_total(scenario, belief, max_stops):
    """One stop in the whole race. A driver who has not stopped yet (no completed stint in
    the snapshot) gets fixed_one_stop's decision through the same wrapper, its limit-aware
    adjustment included; a driver who has stopped gets run_to_flag. Not in "baselines";
    select it by name."""
    if not scenario.snapshot.stints:
        return BASELINE_POLICIES["fixed_one_stop"](scenario, belief, max_stops)
    return run_to_flag(scenario, belief, max_stops)


RULE_POLICIES = {"run_to_flag": run_to_flag, "one_stop_total": one_stop_total}


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
    evidence_calls counts only the evidence calls that ran, repeat_calls the repeats re-run
    or served from the cache. submitted_in_grace says the submission came in a grace turn. Tokens are the backend's
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
        import contextlib

        from ..agent import run_agent
        from ..sim import agent_tools
        from ..sim.compute import load_profile

        max_steps = self.max_steps or (self.profile or load_profile()).agent.max_steps
        path = self.log_path(scenario)
        path.unlink(missing_ok=True)
        stats = {}
        # one belief: the tools plan on the scorer's belief config (the one to_plan expands the
        # submission on and belief_optimizer plans with); a full-window simulator passed as the
        # belief (score_scenario with no belief: the world in both roles) is never handed to them
        cfg = getattr(belief, "config", None)
        same = (agent_tools.belief_config_as(cfg) if cfg is not None and cfg.fit_window != "full"
                else contextlib.nullcontext())
        try:
            with same:
                rec, trajectory = run_agent(scenario.snapshot, verbose=False, backend=self.backend,
                                            max_steps=max_steps, log_path=path, stats=stats)
        except Exception as exc:  # noqa: BLE001  a backend failure must not end an overnight run
            return Decision(None, error=f"{type(exc).__name__}: {exc}", input_tokens=stats.get("input_tokens"),
                            output_tokens=stats.get("output_tokens"))
        return Decision(None, submission=rec, tool_calls=len(trajectory), input_tokens=stats.get("input_tokens"),
                        output_tokens=stats.get("output_tokens"),
                        submitted_in_grace=submitted_in_grace(rec, trajectory),
                        evidence_calls=stats.get("evidence_calls"), repeat_calls=stats.get("repeat_calls"),
                        meta={"submitted": rec is not None, "steps": stats.get("steps"), "trajectory": str(path),
                              "identity": stats.get("identity")})


agent_policy = AgentPolicy()


def resolve(names, recorded=(), *, set_name: str | None = None, profile=None) -> dict:
    """Policies by name. "baselines" expands to the four reference policies, "belief_optimizer" is the optimizer on
    the belief and "run_to_flag" and "one_stop_total" are the rule baselines (all three apart
    from the group), "agent" is the
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
        elif name in RULE_POLICIES:
            out[name] = RULE_POLICIES[name]
        elif name in BASELINE_POLICIES:
            out[name] = BASELINE_POLICIES[name]
        else:
            raise KeyError(f"unknown policy {name!r}; choose from baselines, agent, belief_optimizer, "
                           f"{', '.join(RULE_POLICIES)}, {', '.join(BASELINES)}")
    for path in recorded:
        name, policy = recorded_policy(path)
        out[name] = policy
    return out
