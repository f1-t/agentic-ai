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
_PLANS: dict = {}                    # this snapshot's evaluated plans: plan -> the first action label
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
    _PLANS.clear()
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


def pit_window():
    """agent_tools.pit_window for the active snapshot: whether pitting now is legal and the earliest
    and latest legal stop laps (the regulations and the laps so far only)."""
    return agent_tools.pit_window(_contract())


def shown_snapshot() -> dict:
    """The snapshot as the model is first shown it: the view plus the pit window."""
    _snapshot()
    return {**_SHOWN.to_dict(), "pit_window": pit_window()}


def get_race_snapshot():
    """Return the immutable decision-time state supplied to this run, as the model is shown it
    (anonymised unless AGENT_SHOW_IDENTITY), with the pit window."""
    return shown_snapshot()


def check_submission(rec) -> str | None:
    """agent_tools.submission_error for the active snapshot: why the scorer would reject `rec`, said
    with the legal alternative, or None when it would accept it (None too when no belief can be built
    for the race: the scorer decides then)."""
    contract = _contract()
    try:
        agent_tools.belief_simulator(contract)
    except (ValueError, KeyError, OSError):
        return None
    return present(agent_tools.submission_error(contract, rec))


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
    # the remaining-time calibration error only under the legacy gap yardstick: with the paired one it
    # is in get_model_report, with the note that it is not a yardstick for a gap between plans
    legacy = {"calibration_error_s": rep["calibration_error_s"]} if agent_tools.gap_yardstick() == "calibration" else {}
    return {"compound": comp, **entry, "fuel_s_per_lap_interval": rep.get("fuel_s_per_lap_interval"),
            "fit_window": rep["fit_window"], **legacy,
            "flags": entry.get("flags", []), "report_flags": rep["flags"],
            "warning": "This is an estimate, not a guaranteed degradation rate."}


def estimate_pit_loss():
    return _belief(agent_tools.pit_loss)


def get_model_report(ages=None):
    out = _belief(agent_tools.model_report, ages)
    return out if "error" in out else {**out, "pit_window": pit_window()}


def compare_plans(candidates):
    return _belief(agent_tools.compare_plans, list(candidates))


def rank_plans():
    """agent_tools.rank_plans for the active snapshot (protocol v2)."""
    return _belief(agent_tools.rank_plans)


def evaluate_plan(plan=None, first_stop_lap=None, compound=None):
    """agent_tools.evaluate_plan for the active snapshot (protocol v2): a full plan, or a first stop
    at the end of a lap onto a compound, against the optimizer's best plan."""
    if plan is None and first_stop_lap is None:
        return {"error": "Give plan ([[compound, laps], ...] for the remaining laps) or first_stop_lap "
                         "(with compound, optional).", "recoverable": True}
    if plan is not None and first_stop_lap is not None:
        return {"error": "Give either plan or first_stop_lap (with compound), not both.", "recoverable": True}
    return _belief(agent_tools.evaluate_plan, plan, first_stop_lap, compound)


def rank_plans_families():
    """agent_tools.rank_plans(families=True) for the active snapshot (protocol v2free's rank_plans)."""
    return _belief(agent_tools.rank_plans, families=True)


def race_picture():
    """agent_tools.race_picture for the active snapshot (protocol v2free)."""
    return _belief(agent_tools.race_picture)


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
    # two requests that resolve to the same plan (a stay of k is a minimum): said once, plainly
    label = agent_tools.action_label(sim_action)
    if out.get("plan") is not None:
        first = _PLANS.setdefault(str(out["plan"]), label)
        if first != label:
            out = {**out, "same_plan_as": f"same plan as {first}: this request resolves to the plan evaluated there"}
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


# How the descriptions say to read a gap between plans: AGENT_GAP_YARDSTICK as the tools are imported
# (agent_tools.gap_yardstick; "calibration" restores the legacy text and within_sim_error)
_LEGACY_GAPS = agent_tools.gap_yardstick() == "calibration"
_REPORT_CALIBRATION = (
    "the model's calibration error (None when no earlier race measured it)" if _LEGACY_GAPS else
    "the model's calibration error of the absolute remaining race time (None when no earlier race measured it; "
    "not a yardstick for the gap between two plans, whose common errors cancel)")
_COMPARE_GAPS = (
    "" if _LEGACY_GAPS else
    " gap_readings: the fastest candidate against each other valid one, the share of draws it is faster in, "
    "the gap and its 90% interval (from the paired draws; model error not included).")
_EVALUATE_GAPS = (
    "within_sim_error is true when the gap is smaller than max(2 x its sd, the model's calibration error; its sd "
    "alone when that error is None), and confidence_basis says which decided." if _LEGACY_GAPS else
    "gap_reading says which side is faster in what share of the draws, and the gap (action minus alternative, "
    "negative: the action is faster) with its 90% interval; the interval is the model's own spread of the "
    "difference over shared draws, so errors common to both plans cancel in it.")
_EVALUATE_STAY = (
    " A STAY_OUT of k laps is a minimum: continuation says the plan it resolves to (where its first stop is), and "
    "same_plan_as says when two requests are the same plan. When pitting now is not legal the alternative is the "
    "earliest legal stop (note says so); with no legal stop left, note says STAY_OUT to the flag is the only legal "
    "action. stop_lap_scan gives, in one call, a stop at each of several laps against this plan, so there is no "
    "need to ask k = 3, 5, 7, 10 one at a time.")

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
        "description": "Report what the simulator's tyre model knows: per compound base pace and degradation ranges, data support, where each compound's pace and wear come from (pace_source, wear_source: race laps, or a proxy from practice, qualifying or the step seen in earlier races for a compound nobody raced), flags such as extrapolation or wear_at_bound, per compound tyre_life (the longest run seen so far and a pre-race tyre-life estimate with its sd from earlier races; a set run past its life loses time quickly, there is no hard stint limit), and " + _REPORT_CALIBRATION + ". traffic: whether other cars are modelled; when they are (projected from what is known now), cars_within_pit_loss_behind_now (a stop now drops behind them), dirty_air_s and own_pace_s.",
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
        "description": "Score several candidates against each other under shared random draws. Each candidate is a remaining-race plan [[compound, laps], ...] whose first stint continues the current set, or an action {action:'pit', compound:'MEDIUM'} / {action:'stay_out', laps:3} completed by the optimizer. Returns per candidate summaries, p_best, pairwise p_beats and gaps; errors say which stint breaks a rule and what is legal instead." + _COMPARE_GAPS + " When traffic is modelled the times include it and traffic gives per candidate expected_rejoin_position and its band, cars_in_pit_window (cars that come by during the first stop), time_lost_s in traffic and final_position, from other cars projected from what is known now; otherwise traffic says it is not modelled.",
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
        "description": "Evaluate an immediate PIT or STAY_OUT decision probabilistically against its alternative. Returns ranges, probabilities, assumptions and uncertainty; never an oracle score. probability_improves_outcome covers parameter and lap-noise uncertainty under this model, not model error; " + _EVALUATE_GAPS + _EVALUATE_STAY + " When traffic is modelled, traffic gives for the action and its alternative the expected rejoin position (with a band), cars in the pit-loss window behind, time lost in traffic (included in the times) and final position, from other cars projected from what is known now (they do not react to us); otherwise it says traffic is not modelled.",
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


# Protocol v2 (src/agent.py AGENT_PROTOCOL): the ranked plans and one plan evaluated in place of
# probing one action at a time; the per-estimate tools stay. TOOL_SPECS above is protocol v1, unchanged.
_V2_KEPT = ("estimate_tyre_degradation", "estimate_pit_loss")
TOOL_SPECS_V2 = [
    {
        "name": "rank_plans",
        "description": "Call first. The optimizer's best distinct plans for the rest of the race from this snapshot, in one call (it searches every stop lap and compound). Per plan: plan (stints, the first continuing the current set), first_stop, immediate (the action it implies now), submit (the submission that keeps it), gap_s to plan 1 with gap_90 (its 90% interval over shared draws) and faster_share (share of draws faster than plan 1), best_share (share of draws in which it is the fastest plan shown), first_stop_range (the stop laps over which a plan of that shape stays within 1 s), flags (a stint past the estimated tyre life, or past the longest run seen when the life beyond it is a generic prior) and extrapolates (a stint older than any seen so far: a note, not a flag). decision: the best PIT-now plan against the best STAY_OUT plan, the favoured action and its share of draws. flags: weak estimates (a compound with no race laps, a prior-dominated compound, a safety car or VSC in progress). Plan 1 is what the optimizer alone would play.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": rank_plans,
    },
    {
        "name": "evaluate_plan",
        "description": "One plan against plan 1 of rank_plans under the same draws. Give plan: [[compound, laps], ...] for the remaining laps, the first stint continuing the current set; or first_stop_lap (a stop at the end of that lap) with compound (omit it to try every legal compound), the rest planned by the optimizer. Returns the plan, vs_best (gap_s, gap_90, faster_share), immediate, submit, scored_plan when the scorer would play another plan for that submission, flags and extrapolates. Rule errors say what is legal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {"type": "array", "items": {"type": "array"},
                         "description": "[[compound, laps], ...] summing to the laps remaining."},
                "first_stop_lap": {"type": "integer", "description": "The lap at whose end the first stop is made."},
                "compound": {"type": "string", "description": "The compound fitted at that stop."},
            },
            "additionalProperties": False,
        },
        "fn": evaluate_plan,
    },
    *[spec for spec in TOOL_SPECS if spec["name"] in _V2_KEPT],
]

# Protocol v2free (the default): v2's tools plus the race as a strategist sees it.
TOOL_SPECS_V2FREE = [
    {
        "name": "rank_plans",
        "description": "Call first. The simulator's look-ahead across the strategy space, in one call: the best plan of each strategy family (stops left and the compound sequence after the current set; it searches every stop lap), plan 1 the lowest mean time, at most 8 families and none more than 25 s behind (families_dropped counts the rest), plus the best PIT-now plan. Per plan: family, plan (stints, the first continuing the current set), immediate (the action it implies now), submit (the submission fields that keep it), gap_s to plan 1 with gap_90 (its 90% interval over shared draws) and faster_share (share of draws faster than plan 1), best_share (share of draws in which it is the fastest plan shown), first_stop_range (the stop laps over which a plan of that shape stays within 1 s), flags (a stint past the estimated tyre life, or past the longest run seen when the life beyond it is a generic prior) and extrapolates (a stint older than any seen so far: a note, not a flag). decision: the best PIT-now plan against the best STAY_OUT plan. flags: weak estimates (a compound with no race laps, a prior-dominated compound, a safety car or VSC in progress).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": rank_plans_families,
    },
    TOOL_SPECS_V2[1],
    {
        "name": "race_picture",
        "description": "Everything visible now that the look-ahead does not fully weigh, in one call. car: position, gaps, laps left, track status and how long a safety car or VSC has lasted, the pit window, the stints so far, the current set and its last lap times, and set_wear (the set's observed wear in s/lap of tyre age from its clean laps, fuel-corrected, against the model's value and sd: is this set wearing faster or slower than the simulator thinks?). nearby: the cars around (position, gap, compound, tyre age, stops so far). rejoin: cars within the pit loss behind now and the position a stop now would rejoin at, from the gaps. compounds: per compound the race laps behind the fit, the longest run seen in the race, the field's observed wear against the model's, the base pace sd, how a compound with no race laps is priced, and the tyre-life estimate with its sd and source.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": race_picture,
    },
    *TOOL_SPECS_V2[2:],
]
