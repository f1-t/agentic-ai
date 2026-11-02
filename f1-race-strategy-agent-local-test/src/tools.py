"""Uncertainty-aware F1 strategy tools.

Tools evaluate a *current race snapshot*. They deliberately return ranges and
probabilities: the simulator is a decision aid, not an oracle.

The bodies are src/sim/agent_tools.py, the belief simulator at the snapshot (SPEC 0015):
fitted only on the laps up to the snapshot, with posterior parameter draws and lap noise.
The minimum stint (R2, docs/rules.md) is the snapshot's race.min_stint_laps.
"""
from __future__ import annotations

from typing import Any

from .contracts import Snapshot
from .sim import agent_tools
from .snapshot import RaceSnapshot

_ACTIVE_SNAPSHOT: RaceSnapshot | None = None
_ACTIVE_CONTRACT: Snapshot | None = None


def set_snapshot(snapshot: Snapshot | RaceSnapshot | dict[str, Any]) -> RaceSnapshot:
    """Make `snapshot` the state every tool reads. A contracts.Snapshot is kept as it is and
    the agent's view built from it; a RaceSnapshot or dict is the view, and its contract is
    rebuilt (agent_tools.contract_of). Returns the view."""
    global _ACTIVE_SNAPSHOT, _ACTIVE_CONTRACT
    if isinstance(snapshot, Snapshot):
        contract, view = snapshot, RaceSnapshot.from_contract(snapshot)
    else:
        view = snapshot if isinstance(snapshot, RaceSnapshot) else RaceSnapshot.from_dict(snapshot)
        contract = agent_tools.contract_of(view)
    _ACTIVE_SNAPSHOT, _ACTIVE_CONTRACT = view, contract
    return _ACTIVE_SNAPSHOT


def _snapshot() -> RaceSnapshot:
    if _ACTIVE_SNAPSHOT is None:
        raise ValueError("No race snapshot is active for this agent run.")
    return _ACTIVE_SNAPSHOT


def _contract() -> Snapshot:
    _snapshot()
    return _ACTIVE_CONTRACT


def _belief(fn, *args, **kwargs):
    """fn(contract, ...), or a non-recoverable error when the belief simulator cannot be
    built for this snapshot's race (no race data in the cache)."""
    contract = _contract()
    try:
        agent_tools.belief_simulator(contract)
    except (ValueError, KeyError, OSError) as exc:
        return {"error": f"No simulator for race {contract.race.race_id!r}: {exc}", "recoverable": False}
    return fn(contract, *args, **kwargs)


def get_race_snapshot():
    """Return the immutable decision-time state supplied to this run."""
    return _snapshot().to_dict()


def estimate_tyre_degradation(compound):
    """One compound's entry of the belief's model report (agent_tools.model_report)."""
    comp = str(compound).upper()
    rep = _belief(agent_tools.model_report)
    if "error" in rep:
        return rep
    entry = rep["compounds"].get(comp)
    if entry is None:
        return {"error": f"Unknown compound '{comp}'.", "recoverable": True,
                "modelled": sorted(c for c, e in rep["compounds"].items() if e["modelled"])}
    return {"compound": comp, **entry, "fuel_s_per_lap_interval": rep.get("fuel_s_per_lap_interval"),
            "fit_window": rep["fit_window"], "calibration_error_s": rep["calibration_error_s"],
            "flags": entry.get("flags", []), "report_flags": rep["flags"],
            "warning": "This is an estimate, not a guaranteed degradation rate."}


def estimate_pit_loss():
    return _belief(agent_tools.pit_loss)


def get_model_report(ages=None):
    return _belief(agent_tools.model_report, ages)


def compare_plans(candidates):
    return _belief(agent_tools.compare_plans, list(candidates))


def _validate_action(action: dict[str, Any]):
    s = _snapshot()
    kind = str(action.get("action", "")).lower()
    if kind == "stay_out":
        horizon = int(action.get("laps", 1))
        if horizon < 1 or horizon > max(1, s.laps_remaining):
            return None, f"stay_out laps must be between 1 and {max(1, s.laps_remaining)}."
        return {"action": kind, "laps": horizon}, None
    if kind == "pit":
        comp = str(action.get("compound", "")).upper()
        if comp not in s.available_compounds:
            return None, f"Compound '{comp}' is not available; choose from {s.available_compounds}."
        return {"action": kind, "compound": comp}, None
    return None, "action must be 'pit' or 'stay_out'."


def evaluate_strategy(action):
    """Evaluate one immediate decision using a distribution, never a single truth value
    (agent_tools.evaluate_action: the action against its alternative on the belief)."""
    clean, error = _validate_action(action)
    if error:
        return {"error": error, "recoverable": True, "hint": "Correct the action and call evaluate_strategy again."}
    sim_action = ({"action": "stay", "laps": clean["laps"]} if clean["action"] == "stay_out"
                  else {"action": "pit", "compound": clean["compound"]})
    out = _belief(agent_tools.evaluate_action, sim_action)
    if "error" in out:
        return {**out, "recoverable": out.get("recoverable", True),
                "hint": "Correct the action and call evaluate_strategy again."}
    return {
        **out,
        "action": clean,
        "assumptions": ["no safety car or other incident is drawn after the snapshot",
                        "the plan after the action is the optimizer's best continuation on the same model",
                        "traffic is not modelled"],
        "warning": "Compare this distribution with alternatives; do not treat the median as ground truth.",
    }


TOOL_SPECS = [
    {
        "name": "get_race_snapshot",
        "description": "Read the current decision-time race snapshot: lap, position, tyre state, gaps, weather, nearby cars and recent context.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": get_race_snapshot,
    },
    {
        "name": "estimate_tyre_degradation",
        "description": "Return the simulator's low/median/high degradation per lap for a compound, fitted only on laps up to now, with its flags. It is evidence, not truth.",
        "input_schema": {"type": "object", "properties": {"compound": {"type": "string"}}, "required": ["compound"], "additionalProperties": False},
        "fn": estimate_tyre_degradation,
    },
    {
        "name": "estimate_pit_loss",
        "description": "Estimate the pit-lane loss as a low/median/high range with its source, plus the safety-car and VSC pit-loss factors.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": estimate_pit_loss,
    },
    {
        "name": "get_model_report",
        "description": "Report what the simulator's tyre model knows: per compound base pace and degradation ranges, data support, flags such as extrapolation or wear_at_bound, and the model's calibration error.",
        "input_schema": {
            "type": "object",
            "properties": {"ages": {"type": "object", "description": "Optional {compound: tyre age} to check for extrapolation.",
                                    "additionalProperties": {"type": "integer"}}},
            "additionalProperties": False,
        },
        "fn": get_model_report,
    },
    {
        "name": "compare_plans",
        "description": "Score several candidates against each other under shared random draws. Each candidate is a remaining-race plan [[compound, laps], ...] whose first stint continues the current set, or an action {action:'pit', compound:'MEDIUM'} / {action:'stay_out', laps:3} completed by the optimizer. Returns per candidate summaries, p_best, pairwise p_beats and gaps.",
        "input_schema": {
            "type": "object",
            "properties": {"candidates": {"type": "array", "items": {"type": ["array", "object"]}}},
            "required": ["candidates"],
            "additionalProperties": False,
        },
        "fn": compare_plans,
    },
    {
        "name": "evaluate_strategy",
        "description": "Evaluate an immediate PIT or STAY_OUT decision probabilistically against its alternative. Returns ranges, probabilities, assumptions and uncertainty; never an oracle score. probability_improves_outcome covers parameter and lap-noise uncertainty under this model, not model error; within_sim_error is true when the gap is smaller than max(2 x its sd, the model's calibration error), and confidence_basis says which of the two decided.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "object",
                    "description": "Either {action:'pit', compound:'MEDIUM'} or {action:'stay_out', laps:3}.",
                    "properties": {"action": {"type": "string"}, "compound": {"type": "string"}, "laps": {"type": "integer"}},
                    "required": ["action"],
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "fn": evaluate_strategy,
    },
]
