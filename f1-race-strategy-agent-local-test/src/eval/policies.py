"""Policies the eval scores. A policy maps a scenario to a plan for the remaining laps.

    policy(scenario, sim, max_stops) -> Decision      raises PolicyUnavailable to skip

The four baselines are Quan's reference policies (src/baselines) run through the optimizer
bridge, so they see the same snapshot and the same fit as the optimizer they are measured
against. `recorded_policy` replays plans saved from earlier agent runs, and `agent_policy`
asks the live agent loop (src/agent.py) through whatever LLM_BACKEND is configured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..baselines import domain, policies
from ..baselines.domain import PolicyUnavailable
from ..sim.optimizer_bridge import from_contract, pace_for_crossover, to_contract
from ..sim.race_ids import canonical_race_id

BASELINES = ("actual_strategy", "fixed_one_stop", "fixed_two_stop", "greedy_crossover")


@dataclass
class Decision:
    """What a policy decided. `plan` is anything contracts.Plan.of accepts, or None when the
    policy produced no plan (an invalid decision). `error` marks a failure of the policy
    itself (an API error, say), which is reported apart from invalid plans."""

    plan: object
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


def _bridged(choose):
    """A baseline that plans on the bridge's small snapshot and returns a contracts.Plan."""
    def policy(scenario, sim, max_stops):
        state = from_contract(scenario.snapshot, sim)
        plan = choose(scenario, state, sim, max_stops)
        if plan.stints[0].laps == 0:
            raise PolicyUnavailable("The plan stops immediately; a first stint needs at least one lap "
                                    "(docs/rules.md R1).")
        return Decision(to_contract(plan, scenario.snapshot, sources=sim.config.inventory_sources))
    return policy


BASELINE_POLICIES = {
    "actual_strategy": _bridged(lambda sc, st, sim, m: policies.actual_strategy(
        st, [domain.Stint(c, n) for c, n in sc.actual_stints])),
    "fixed_one_stop": _bridged(lambda sc, st, sim, m: policies.fixed_one_stop(st)),
    "fixed_two_stop": _bridged(lambda sc, st, sim, m: policies.fixed_two_stop(st)),
    "greedy_crossover": _bridged(lambda sc, st, sim, m: policies.greedy_crossover(
        st, pace_for_crossover(sim, sc.snapshot), m)),
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
    return data["name"], policy


AGENT_QUESTION = ("{driver} starts the {season} {race} on the {compound} tyre. Should they run a one-stop "
                  "or a two-stop strategy, and how should the stints be split, to minimise total race "
                  "time? Justify it with simulated times.")


def agent_policy(scenario, sim, max_stops):
    """The live agent. Its tools are fixed to one race and plan whole races from the start,
    so only lap-0 scenarios of that race with the full race length are available until the
    loop takes a snapshot."""
    from .. import tools

    if (scenario.lap != 0 or scenario.race_id != canonical_race_id(tools._race_params().race_id)
            or scenario.snapshot.race.total_laps != tools.race_laps()):
        raise PolicyUnavailable("the agent loop plans full races of the tools' race only; mid-race "
                                "scenarios wait for the snapshot-driven loop")
    from ..agent import run_agent

    info = tools.get_race_info()
    question = AGENT_QUESTION.format(driver=scenario.driver, season=info["season"], race=info["race"],
                                     compound=scenario.snapshot.compound)
    try:
        rec, trajectory = run_agent(question, verbose=False)
    except Exception as exc:  # noqa: BLE001  a backend failure must not end an overnight run
        return Decision(None, error=f"{type(exc).__name__}: {exc}")
    plan = (rec or {}).get("stint_plan")
    return Decision(plan, tool_calls=len(trajectory), meta={"submitted": rec is not None})


def resolve(names, recorded=()) -> dict:
    """Policies by name. "baselines" expands to the four reference policies, "agent" is the
    live agent, and each recorded file adds the policy it names."""
    out = {}
    for name in names:
        if name == "baselines":
            out.update(BASELINE_POLICIES)
        elif name == "agent":
            out["agent"] = agent_policy
        elif name in BASELINE_POLICIES:
            out[name] = BASELINE_POLICIES[name]
        else:
            raise KeyError(f"unknown policy {name!r}; choose from baselines, agent, {', '.join(BASELINES)}")
    for path in recorded:
        name, policy = recorded_policy(path)
        out[name] = policy
    return out
