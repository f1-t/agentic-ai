"""Uncertainty-aware F1 strategy tools.

Tools evaluate a *current race snapshot*. They deliberately return ranges and
probabilities: the simulator is a decision aid, not an oracle.

The bodies are src/sim/agent_tools.py, the belief simulator at the snapshot (SPEC 0015):
fitted only on the laps up to the snapshot, with posterior parameter draws and lap noise.
The minimum stint (R2, docs/rules.md) is the snapshot's race.min_stint_laps.
"""
from __future__ import annotations

from typing import Any

import re

from .contracts import Snapshot
from .sim import agent_tools
from .snapshot import OTHER_LABEL, RACE_LABEL, RaceSnapshot, pseudonyms, show_identity

_ACTIVE_SNAPSHOT: RaceSnapshot | None = None
_ACTIVE_CONTRACT: Snapshot | None = None
_SHOWN: RaceSnapshot | None = None   # the view the model sees (anonymised unless show_identity)
_IDENTITY: dict = {}                 # the harness's mapping: {"race_id", "labels": {code: label}}
_SCRUB: list = []                    # (compiled pattern, replacement), applied by present()
YEAR_LABEL = "<year>"


def set_snapshot(snapshot: Snapshot | RaceSnapshot | dict[str, Any], *,
                 identity: bool | None = None) -> RaceSnapshot:
    """Make `snapshot` the state every tool reads. A contracts.Snapshot is kept as it is and
    the agent's view built from it; a RaceSnapshot or dict is the view, and its contract is
    rebuilt (agent_tools.contract_of). Returns the view the model is shown: anonymised
    (RaceSnapshot.anonymised) unless show_identity(identity), the AGENT_SHOW_IDENTITY ablation.
    The real names stay here (identity()); present() applies the same mapping to every tool
    output and error string."""
    global _ACTIVE_SNAPSHOT, _ACTIVE_CONTRACT, _SHOWN, _IDENTITY
    if isinstance(snapshot, Snapshot):
        contract, view = snapshot, RaceSnapshot.from_contract(snapshot)
    else:
        view = snapshot if isinstance(snapshot, RaceSnapshot) else RaceSnapshot.from_dict(snapshot)
        contract = agent_tools.contract_of(view)
    _ACTIVE_SNAPSHOT, _ACTIVE_CONTRACT = view, contract
    if show_identity(identity):
        _SHOWN, _IDENTITY = view, {"shown": True, "race_id": contract.race.race_id, "labels": {}}
        _SCRUB.clear()
    else:
        labels = pseudonyms(view.driver, [c.get("driver") for c in view.nearby_cars])
        _SHOWN = view.anonymised(labels)
        _IDENTITY = {"shown": False, "race_id": contract.race.race_id, "labels": labels}
        _SCRUB[:] = _scrub_rules(contract, view, labels)
    return _SHOWN


def identity() -> dict:
    """The active snapshot's real identity, kept in the harness and never shown to the model:
    {"shown": AGENT_SHOW_IDENTITY was on, "race_id", "labels": {real code: label}}."""
    return dict(_IDENTITY)


def _field_codes(contract) -> set[str]:
    """Every driver code of the race (the belief's frame and qualifying), empty when the race is
    not in the cache."""
    try:
        sim = agent_tools.belief_simulator(contract)
    except (ValueError, KeyError, OSError):   # as _belief: no race data in the cache
        return set()
    codes = set(sim.laps["driver"].dropna().astype(str)) if "driver" in sim.laps else set()
    if getattr(sim, "qualifying", None) is not None and "driver" in sim.qualifying:
        codes |= set(sim.qualifying["driver"].dropna().astype(str))
    return codes


def _schedule_names(season, rnd, slug) -> set[str]:
    """The cached schedule's event name, country and location for (season, round); empty when
    the schedule is not cached. A synthetic race (cross_race.is_synthetic) is never looked up:
    FastF1 would reach for a schedule that does not exist."""
    from .sim.cross_race import is_synthetic
    from .sim.data import season_schedule
    from .sim.race_ids import make_race_id
    if is_synthetic(make_race_id(season, rnd, slug)):
        return set()
    sched = season_schedule(season)
    if sched is None:
        return set()
    row = sched[sched["RoundNumber"] == int(rnd)]
    return {str(row.iloc[0][c]) for c in ("EventName", "Country", "Location")} if len(row) else set()


def _scrub_rules(contract, view, labels) -> list:
    """present()'s replacements: the race id, its season and event slug (and any year), each
    labelled driver code by its label, every other code of the field by OTHER_LABEL."""
    from .sim.race_ids import canonical_race_id, parse_race_id

    rid = str(contract.race.race_id)
    names = {rid, canonical_race_id(rid)}
    words = set()
    try:
        season, rnd, slug = parse_race_id(canonical_race_id(rid))
        words |= {slug, slug.replace("_", " ")}
        words |= _schedule_names(season, rnd, slug)
    except ValueError:
        pass
    if view.event:
        words |= {str(view.event), str(view.event).replace("_", " ")}
    rules = [(re.compile(re.escape(n)), RACE_LABEL) for n in sorted(names, key=len, reverse=True) if n]
    rules += [(re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE), RACE_LABEL)
              for w in sorted(words, key=len, reverse=True) if len(w) >= 3]
    rules.append((re.compile(r"(?<![\w.])(?:19|20)\d{2}(?![\w.]?\d)"), YEAR_LABEL))   # a year, not a digit run
    others = (_field_codes(contract) | set(labels)) - set(labels.values())
    for code in sorted(others, key=len, reverse=True):
        if code and code.isalnum():
            rules.append((re.compile(r"\b" + re.escape(code) + r"\b"), labels.get(code, OTHER_LABEL)))
    return rules


def present(value):
    """`value` (a tool output or error) as the model is shown it: every string, keys included,
    with the active snapshot's real race and driver names replaced (set_snapshot); unchanged
    under AGENT_SHOW_IDENTITY."""
    if not _SCRUB:
        return value
    if isinstance(value, str):
        for pattern, label in _SCRUB:
            value = pattern.sub(label, value)
        return value
    if isinstance(value, dict):
        return {present(k): present(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [present(v) for v in value]
    return value


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
    """Return the immutable decision-time state supplied to this run, as the model is shown it
    (anonymised unless AGENT_SHOW_IDENTITY)."""
    _snapshot()
    return _SHOWN.to_dict()


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
        "assumptions": [*out.get("assumptions", []),
                        "no incident other than the drawn SC and VSC periods changes the race",
                        "the plan after the action is the optimizer's best continuation on the same model",
                        ("other cars are projected from what is known now (positions, gaps, pace, tyres); "
                         "they do not react to us" if agent_tools.traffic_on(agent_tools.belief_config())
                         else "traffic is not modelled")],
        "warning": "Compare this distribution with alternatives; do not treat the median as ground truth.",
    }


TOOL_SPECS = [
    {
        "name": "get_race_snapshot",
        "description": "Read the current decision-time race snapshot: lap, laps remaining, position, tyre state, track status (GREEN, SC or VSC), gaps, weather, nearby cars and your recent lap times. The race is not named and cars are labelled: yours is Car A, the cars around you Car B, Car C, ...",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": get_race_snapshot,
    },
    {
        "name": "estimate_tyre_degradation",
        "description": "Return the simulator's low/median/high degradation per lap for a compound, fitted only on what is known up to now, with its flags. A compound nobody has raced is priced by proxy: pace_source and wear_source say whether a number comes from race laps, practice long runs, qualifying or the typical step from a neighbouring compound in earlier races, and a proxied number has no race laps behind it. It is evidence, not truth.",
        "input_schema": {"type": "object", "properties": {"compound": {"type": "string"}}, "required": ["compound"], "additionalProperties": False},
        "fn": estimate_tyre_degradation,
    },
    {
        "name": "estimate_pit_loss",
        "description": "Estimate the pit-lane loss from what is known now: a low/median/high range built from earlier races (at this circuit when there are any) and this race's green stops so far, with its basis. Also the share of it paid under a safety car or VSC, the cost of a stop right now (pit_loss_now_s) given the current track status and how many laps it has lasted, and the chance of a new safety car or VSC in the next 5 and 10 laps with a range. Later neutralisations are not known.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": estimate_pit_loss,
    },
    {
        "name": "get_model_report",
        "description": "Report what the simulator's tyre model knows: per compound base pace and degradation ranges, data support, where each compound's pace and wear come from (pace_source, wear_source: race laps, or a proxy from practice, qualifying or the step seen in earlier races for a compound nobody raced), flags such as extrapolation or wear_at_bound, per compound tyre_life (the longest run seen so far and a pre-race tyre-life estimate with its sd from earlier races; a set run past its life loses time quickly, there is no hard stint limit), and the model's calibration error (None when no earlier race measured it). traffic: whether other cars are modelled; when they are (projected from what is known now), cars_within_pit_loss_behind_now (a stop now drops behind them), dirty_air_s and own_pace_s.",
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
        "description": "Score several candidates against each other under shared random draws. Each candidate is a remaining-race plan [[compound, laps], ...] whose first stint continues the current set, or an action {action:'pit', compound:'MEDIUM'} / {action:'stay_out', laps:3} completed by the optimizer. Returns per candidate summaries, p_best, pairwise p_beats and gaps. When traffic is modelled the times include it and traffic gives per candidate expected_rejoin_position and its band, cars_in_pit_window (cars that come by during the first stop), time_lost_s in traffic and final_position, from other cars projected from what is known now; otherwise traffic says it is not modelled.",
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
        "description": "Evaluate an immediate PIT or STAY_OUT decision probabilistically against its alternative. Returns ranges, probabilities, assumptions and uncertainty; never an oracle score. probability_improves_outcome covers parameter and lap-noise uncertainty under this model, not model error; within_sim_error is true when the gap is smaller than max(2 x its sd, the model's calibration error; its sd alone when that error is None), and confidence_basis says which decided. When traffic is modelled, traffic gives for the action and its alternative the expected rejoin position (with a band), cars in the pit-loss window behind, time lost in traffic (included in the times) and final position, from other cars projected from what is known now (they do not react to us); otherwise it says traffic is not modelled.",
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
