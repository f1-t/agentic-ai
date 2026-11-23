"""The simulator behind the agent's tools (SPEC 0015).

Every function takes a contracts.Snapshot and returns a JSON-ready dict: NaN and infinite
values come back as None, and every number carries an interval, a source or a flag. They
run on the belief simulator, configs/sim/belief.yaml as it stands (to_snapshot window,
traffic off, SC and VSC periods from the rate model, the config's calibrator, posterior
parameter draws, residual lap noise, the wear lower bound), or the yaml AGENT_BELIEF_CONFIG
names. The tools never build their own config. Traffic is a switch (docs/eval.md
"Traffic"): with configs/sim/belief_traffic.yaml the other cars are projected from what was
visible at the snapshot (traffic.py, source "projected"; belief_simulator refuses the
recorded source), and compare_plans, evaluate_action and model_report carry it compactly
(traffic_view, traffic_report); with it off they say traffic is not modelled.

    race_state(snapshot)                       the snapshot, legal_compounds, remaining_laps, event_rates
    model_report(snapshot, ages)               the fit's coefficients with intervals and flags
    pit_loss(snapshot)                         pit loss low / median / high, SC and VSC factors
    simulate_plan(snapshot, stints)            SimResult of one plan
    compare_plans(snapshot, candidates)        CompareResult of plans and actions
    evaluate_action(snapshot, action)          one action against its alternative
    recommendation_fields(snapshot, decision)  p_pit for the Brier column

belief_simulator(snapshot) is the Simulator for snapshot.race.race_id, built once per race
(and profile and config) from sim.jobs.load_race. Its frame is the whole race; fitted_for
cuts the fit window at the snapshot, so no lap after the snapshot reaches any output (the
information-boundary canary in tests/test_sim_agent_tools.py checks it).

Every call uses seed SEED = 0 + BELIEF_SEED_OFFSET (docs/rules.md R6): every belief-side
simulation draws from seed + BELIEF_SEED_OFFSET (eval.decisions.to_plan's continuation
too), so with events on in both simulators the belief never plans on the event, parameter
or lap-noise samples the world scores on (seed s, samples 0..n-1). n_samples None takes the compute profile's
sim.n_samples (compare_n_samples for compare_plans and evaluate_action), and the
continuations are optimizer_bridge.optimize_batch on the belief fit at the snapshot, with
the profile's sim.max_stops counted from the snapshot as eval.decisions counts them: a pit
spends one stop.

Actions are {"action": "pit", "compound": c} or {"action": "stay", "laps": n}; the agent's
"stay_out" is read as "stay". A stay of at least the remaining laps runs the current set
to the flag.
"""
from __future__ import annotations

import contextlib
import copy
import math
import os
import re
from pathlib import Path

import numpy as np

from ..contracts import Plan, Snapshot
from . import calibrators, jobs
from . import cliff as _cliff
from . import model_report as _report
from . import proxy as _proxy
from . import rules as _rules
from .compute import load_profile
from .config import SimConfig
from .engine import DEGRADATION_RETRY_STREAM, DEGRADATION_STREAM, Simulator, batch_normals
from .optimizer_bridge import as_continuation
from .race_ids import RACE_ID_ALIASES, canonical_race_id, parse_race_id, slug

BELIEF_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sim" / "belief.yaml"
BELIEF_CONFIG_ENV = "AGENT_BELIEF_CONFIG"   # a belief yaml for the tools in place of BELIEF_CONFIG
# added to the seed of every belief-side simulation, so the belief's samples are never the
# world's (a prime, far from any seed a run uses)
BELIEF_SEED_OFFSET = 1_000_003
SEED = 0 + BELIEF_SEED_OFFSET
Z90 = 1.64              # a 90 percent normal band, mean -+ Z90 sd
PIT_LOSS_BAND = 0.10    # the band around a pit loss with no calibrated sd, source "config"
PROBABILITY_COVERS = "parameter and lap-noise uncertainty under this model; does not include model error"
# interval_meaning (docs/eval.md "Honest uncertainty"): what a 90 percent interval of the fit means,
# widened (the fit's sd_scale, chosen by coverage on past races) or not
INTERVAL_MEANING_WIDENED = "90% range, widened so about 9 in 10 such ranges held the full-race value on past races"
INTERVAL_MEANING_RAW = "90% range from this race's laps alone; on past races such ranges held the full-race value far less often"
# event_rates source: the EventModel defaults, or the race's own (circuit) rates
EVENT_RATES_PROVISIONAL = "provisional: fixed config rates, not fitted to earlier races"   # events as_of off (legacy)
EVENT_RATES_CIRCUIT = "circuit"
TRAFFIC_NOTE = "traffic is not modelled belief-side: the gaps and probabilities ignore cars on track"
# belief traffic (traffic.py, source "projected"): how the numbers were made, in words
TRAFFIC_BASIS = ("other cars projected from what is known at this lap only (their positions, gaps, pace and "
                 "tyres so far); each is assumed to stop once more when its tyres reach the field's typical "
                 "life so far; the bands cover that projection's spread, not cars reacting to us")
CALIBRATION_NOTE = ("mean absolute error of the remaining race time for this fit window and horizon; "
                    "a conservative tolerance for a gap between plans, since the fuel error cancels between them")
# The yardstick for a gap between two plans (AGENT_GAP_YARDSTICK). "paired" (the default): the
# belief's own spread of the difference between the two plans over shared draws (common random
# numbers), reported as the gap, its 90% interval and the share of draws each side wins
# (gap_reading). "calibration": the legacy within_sim_error, the gap against max(2 sd, the
# remaining-time calibration error), kept to reproduce runs made before the switch.
GAP_YARDSTICK_ENV = "AGENT_GAP_YARDSTICK"
GAP_YARDSTICKS = ("paired", "calibration")
CALIBRATION_NOTE_PAIRED = ("mean absolute error of the absolute remaining race time for this fit window and "
                           "horizon; not a yardstick for the gap between two plans, whose common errors cancel "
                           "(read gaps from compare_plans or evaluate_strategy)")


def gap_yardstick() -> str:
    """AGENT_GAP_YARDSTICK, "paired" when unset; any other value than GAP_YARDSTICKS is an error."""
    mode = os.environ.get(GAP_YARDSTICK_ENV, "").strip().lower() or GAP_YARDSTICKS[0]
    if mode not in GAP_YARDSTICKS:
        raise ValueError(f"{GAP_YARDSTICK_ENV} must be one of {', '.join(GAP_YARDSTICKS)}; got {mode!r}")
    return mode


def calibration_note() -> str:
    return CALIBRATION_NOTE if gap_yardstick() == "calibration" else CALIBRATION_NOTE_PAIRED

_SIMS: dict = {}
_RACES: dict = {}
_BELIEF: dict = {}


# --- the belief simulator ---------------------------------------------------------------
def belief_config() -> SimConfig:
    """configs/sim/belief.yaml, read once; the yaml AGENT_BELIEF_CONFIG names instead when set
    (the agent's traffic switch: configs/sim/belief_traffic.yaml, docs/eval.md "Traffic")."""
    if "config" not in _BELIEF:
        _BELIEF["config"] = SimConfig.from_file(os.environ.get(BELIEF_CONFIG_ENV) or BELIEF_CONFIG)
    return _BELIEF["config"]


@contextlib.contextmanager
def belief_config_as(config: SimConfig):
    """The tools plan on `config` inside the block (eval.policies.AgentPolicy passes the scorer's
    belief), so the agent's tools, the continuation the scorer expands its submission with and the
    belief_optimizer policy run one belief config whatever --belief-config or AGENT_BELIEF_CONFIG
    say. The previous config comes back after the block."""
    had, prev = "config" in _BELIEF, _BELIEF.get("config")
    _BELIEF["config"] = config
    try:
        yield config
    finally:
        if had:
            _BELIEF["config"] = prev
        else:
            _BELIEF.pop("config", None)


def _race_data(race_id: str, profile):
    key = (race_id, profile.hash)
    if key not in _RACES:
        _RACES[key] = jobs.load_race(race_id, profile)
    return _RACES[key]


def belief_simulator(snapshot: Snapshot, profile=None, config: SimConfig | None = None) -> Simulator:
    """The belief Simulator for snapshot.race.race_id: the race's frame and practice from
    jobs.load_race, `config` (default configs/sim/belief.yaml). Cached per race, profile and
    config. fitted_for(snapshot) is the window. Raises ValueError for a race id that is not
    a race in the cache."""
    profile = profile or load_profile()
    config = config or belief_config()
    race_id = canonical_race_id(snapshot.race.race_id)
    if config.traffic.name != "none" and config.traffic.params.get("source", "recorded") != "projected":
        raise ValueError("belief traffic must be the projected source: recorded ghosts read the race's future")
    key = (race_id, profile.hash, config.model_dump_json())
    if key not in _SIMS:
        data = _race_data(race_id, profile)
        _SIMS[key] = Simulator(data.frame, config, data.practice, qualifying=getattr(data, "qualifying", None))
    return _SIMS[key]


def _setup(snapshot, sim, profile):
    profile = profile or load_profile()
    return (sim or belief_simulator(snapshot, profile)), profile


def _compare_n(profile, n_samples):
    return n_samples or profile.sim.compare_n_samples or profile.sim.n_samples


def race_id_for_view(view) -> str | None:
    """The canonical race id of an agent view (snapshot.RaceSnapshot), which keeps only the
    season and the event slug: "<season>_<event>" when that is a race id, else the known
    race (an alias target or jobs.DEFAULT_RACES) with that season and slug, else None."""
    raw = f"{view.season}_{view.event}"
    try:
        parse_race_id(raw)
        return canonical_race_id(raw)
    except ValueError:
        pass
    for rid in sorted({*RACE_ID_ALIASES.values(), *jobs.DEFAULT_RACES}):
        season, _, gp = parse_race_id(rid)
        if season == view.season and gp == slug(view.event):
            return rid
    return None


def contract_of(view, profile=None) -> Snapshot:
    """A contracts.Snapshot for an agent view. A view carries no pit loss (R5) and no
    canonical race id, so when its race is known (race_id_for_view) the race constants
    come from jobs.load_race, with the view's total_laps and min_stint_laps; otherwise
    RaceSnapshot.to_contract's rebuilt race (pit loss 0), which the belief cannot load."""
    race_id = race_id_for_view(view)
    if race_id is None:
        return view.to_contract()
    profile = profile or load_profile()
    race = _race_data(race_id, profile).race
    update = {"total_laps": view.total_laps}
    if view.min_stint_laps is not None:
        update["min_stint_laps"] = view.min_stint_laps
    snap = view.to_contract(race=race.model_copy(update=update))
    # the race's whole-race pit loss is the world's; the contract carries the belief's as-of one
    pl = belief_simulator(snap, profile).fitted_for(snap).pit_loss_inputs(snap)
    return snap.model_copy(update={"race": snap.race.model_copy(update={"pit_loss_s": round(pl["pit_loss_s"], 3)})})


# --- helpers ------------------------------------------------------------------------------
def _jsonable(x):
    """Plain JSON values: NaN and infinities to None, numpy scalars to Python ones."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def _nan(x) -> bool:
    return x is None or not math.isfinite(float(x))


BASIS_DRAWS = "posterior_draws_p05_p50_p95"   # low, median, high: the draws' percentiles
BASIS_SD = "point_estimate_-+1.64_sd"          # median: the point estimate
BASIS_NO_SD = "no_sd"                          # low = median = high = the point estimate


def _interval(value, sd, draws=None):
    """({low, median, high}, basis), always ordered low <= median <= high. With `draws` the
    5th, 50th and 95th percentiles of the draws (BASIS_DRAWS; under a wear bound the point
    estimate, the constrained mode, can sit below the draws' median); else value -+ Z90 sd
    around the point estimate (BASIS_SD); low = median = high = value when sd is NaN
    (BASIS_NO_SD)."""
    value = float(value)
    if _nan(sd):
        return {"low": value, "median": value, "high": value}, BASIS_NO_SD
    if draws is not None:
        lo, med, hi = np.percentile(draws, [5, 50, 95])
        return {"low": float(lo), "median": float(med), "high": float(hi)}, BASIS_DRAWS
    return {"low": value - Z90 * float(sd), "median": value, "high": value + Z90 * float(sd)}, BASIS_SD


def _wear_draws(model, n, seed=SEED):
    """{compound: (n,) drawn wear slopes}: the posterior draws the engine takes for samples
    0..n-1 of `seed` (the same normals, so the same draws, the wear bound respected), None
    when the model draws nothing. Drawn on a copy, so the fit's draw_stats are untouched."""
    if getattr(model, "param_draw", "none") != "posterior" or not hasattr(model, "draw_z"):
        return None
    m = copy.copy(model)
    if "draw_stats" in (model.fit_report or {}):
        from .degradation import new_draw_stats
        m.fit_report = {**model.fit_report, "draw_stats": new_draw_stats()}
    z = batch_normals(seed, DEGRADATION_STREAM, n, m.n_params)
    tries = m.draw_tries
    retry = batch_normals(seed, DEGRADATION_RETRY_STREAM, n, (tries - 1) * m.n_params) if tries > 1 else None
    drawn = [m.draw_z(z[k] if retry is None else np.concatenate([z[k], retry[k]])) for k in range(n)]
    return {c: np.array([d.coefs[c][1] for d in drawn]) for c in model.coefs}


def _action(raw) -> dict:
    """An action for Simulator.expand_action, "stay_out" read as "stay"."""
    act = dict(raw)
    if str(act.get("action", "")).lower() in ("stay_out", "stay"):
        act["action"] = "stay"
    elif str(act.get("action", "")).lower() == "pit":
        act["action"] = "pit"
    return act


def _candidate(snapshot: Snapshot, raw):
    """A compare candidate: actions normalised, a stay of at least the remaining laps as the
    plan that runs the current set to the flag (eval.decisions.to_plan does the same)."""
    if not (isinstance(raw, dict) and "action" in raw):
        return raw
    act = _action(raw)
    laps = act.get("laps", 1)
    if act["action"] == "stay" and isinstance(laps, int) and laps >= snapshot.remaining_laps:
        return Plan.of([[snapshot.compound, snapshot.remaining_laps]])
    return act


def _continuation(fit, n_samples, max_stops):
    """optimize_batch on `fit` after an action: max_stops future stops after a stay, one
    fewer after a pit (eval.decisions). When no plan is feasible inside the limit, or the
    pit fits a compound the fit does not model, the current set runs to the flag and the
    simulator's rules say why that is illegal."""
    from ..baselines.domain import NoFeasiblePlan

    stay = as_continuation(fit, seed=SEED, n_samples=n_samples, max_stops=max_stops)
    pit = as_continuation(fit, seed=SEED, n_samples=n_samples, max_stops=max(0, max_stops - 1))

    def continuation(after: Snapshot):
        just_pitted = after.laps_completed - sum(s.laps for s in after.stints) == 0
        to_flag = Plan.of([[after.compound, after.remaining_laps, after.tyre_age if just_pitted else 0]])
        if after.compound not in fit.modelled:
            return to_flag
        try:
            return (pit if just_pitted else stay)(after)
        except NoFeasiblePlan:
            return to_flag

    def stay_plan(snapshot: Snapshot, laps: int):
        """A stay under SC or VSC (Simulator.expand_action): the whole plan from the decision
        snapshot, the current set to the flag on the same terms as above."""
        to_flag = Plan.of([[snapshot.compound, snapshot.remaining_laps]])
        if snapshot.compound not in fit.modelled:
            return to_flag
        try:
            return stay.stay_plan(snapshot, laps)
        except NoFeasiblePlan:
            return to_flag

    def stop_plan(snapshot: Snapshot, lap: int, compound: str):
        """A stop at the end of a later lap under SC or VSC (_stop_at): the whole plan from the
        decision snapshot; with no feasible plan, the one stop run to the flag, whose legality
        the simulator's rules then decide."""
        w = lap - snapshot.laps_completed
        one_stop = [[snapshot.compound, w], [compound, snapshot.remaining_laps - w]]
        if snapshot.compound not in fit.modelled or compound not in fit.modelled:
            return one_stop
        try:
            return stay.stop_plan(snapshot, lap, compound)
        except NoFeasiblePlan:
            return one_stop
    continuation.stay_plan = stay_plan
    continuation.stop_plan = stop_plan
    return continuation


def legal_compounds(snapshot: Snapshot, config: SimConfig | None = None) -> list[str]:
    """Compounds a pit may fit: those with a set in sets_available whose source the config's
    inventory rule counts (rules.inventory), else race.compounds (the three dry compounds
    when the race names none, as the agent's view does)."""
    if snapshot.sets_available is not None:
        sources = (config or belief_config()).inventory_sources
        return sorted({t.compound for t in snapshot.sets_available if t.source in sources})
    from ..snapshot import DEFAULT_COMPOUNDS
    return list(snapshot.race.compounds) or list(DEFAULT_COMPOUNDS)


def event_rates(snapshot: Snapshot, config: SimConfig | None = None) -> dict:
    """{p_sc, d_sc, p_vsc, d_vsc, source}: the SC and VSC rates the belief draws periods
    with for this race (events.RateEvents.rates). source is "circuit" when RaceParams carries
    circuit rates (sc_rate_per_lap and the others), else EVENT_RATES_PROVISIONAL, the
    EventModel defaults. p_sc_lap1 / p_vsc_lap1 (the chance a period starts on lap 1 of a
    race walked from lap 0) appear only when the events params set them. An events model other
    than "rate" draws no periods: the rates are None and source names the model."""
    from .events import RateEvents

    spec = (config or belief_config()).events
    if spec.name != "rate":
        return {"p_sc": None, "d_sc": None, "p_vsc": None, "d_vsc": None, "source": f"events model {spec.name!r}"}
    race = snapshot.race
    model = RateEvents(**spec.params)
    p_sc, d_sc, p_vsc, d_vsc = model.rates(race)
    circuit = any(v is not None for v in (race.sc_rate_per_lap, race.sc_mean_laps, race.vsc_rate_per_lap,
                                          race.vsc_mean_laps))
    out = {"p_sc": p_sc, "d_sc": d_sc, "p_vsc": p_vsc, "d_vsc": d_vsc}
    # the lap-1 rates (SPEC 0011b), only when set: lap 1 from a standing start carries its own risk
    for key in ("p_sc_lap1", "p_vsc_lap1"):
        if getattr(model.params, key) is not None:
            out[key] = getattr(model.params, key)
    if circuit:
        source = EVENT_RATES_CIRCUIT
    elif model.as_of:   # race_history.as_of_event_rates: the races before this one, else a priori
        from .race_history import as_of_event_rates
        source = as_of_event_rates(race.race_id)["source"]
    else:
        source = EVENT_RATES_PROVISIONAL
    return {**out, "source": source}


def calibration_error_s(sim: Simulator, snapshot: Snapshot) -> float:
    """The tools' tolerance for a gap between plans: the offset-adjusted mean absolute error
    of the remaining race time for the belief's fit window at the snapshot's horizon bucket,
    from reports/calibration/midrace_summary.csv (ModelReport.calibration_error_s), as of the
    race (model_report.calibration_error: NaN when the summary's races are not all before it,
    unless SimConfig.cross_race_as_of is off). It is a conservative tolerance for a gap, since
    the fuel error cancels between plans; SPEC 0016 adds a gap-specific one. NaN without a row."""
    race_id = snapshot.race.race_id if sim.config.cross_race_as_of and sim.config.fit_window != "full" else None
    return _report.calibration_error(sim.config.fit_window, snapshot.remaining_laps, race_id=race_id)


# --- rule replies -----------------------------------------------------------------------------
# rules.py's messages (a physics module, unchanged): matched here and restated for the agent
_MIN_STINT = re.compile(r"^Stint of \d+ laps(?: \(\d+ already run on the set\))? is below the \d+-lap minimum\.$")
_TWO_COMPOUNDS = "A valid dry strategy must use at least two different compounds."
_LAP_SUM = re.compile(r"^Stint laps sum to (\d+), must equal remaining laps (\d+)\.$")


def stop_window(snapshot: Snapshot) -> tuple[int, int]:
    """(first, last): the laps at whose end a stop may be made under the minimum stint (R2):
    the current set must reach race.min_stint_laps (laps already run on it counted) and the
    stint after the stop must too. first > last when no stop fits in the laps left. Only the
    snapshot and the regulations enter (laps so far, the scheduled distance, the minimum)."""
    floor, done = snapshot.race.min_stint_laps, snapshot.laps_completed
    first = done + max(1, floor - _rules.laps_on_set(snapshot))
    return first, snapshot.race.total_laps - floor


def _laps(n: int) -> str:
    return f"{n} lap{'s' if n != 1 else ''}"


def _window_text(snapshot: Snapshot, config=None) -> str:
    """The legal alternative in words: the earliest and latest stop, or that only staying out
    to the flag is left (or that nothing is, when that leaves one dry compound)."""
    first, last = stop_window(snapshot)
    done = snapshot.laps_completed
    if first > last:
        to_flag = Plan.of([[snapshot.compound, snapshot.remaining_laps]])
        if _rules.two_compounds(_rules.RuleContext(snapshot, to_flag, set(), config)):
            return (f"no stop fits in the {_laps(snapshot.remaining_laps)} left, and running to the flag leaves "
                    f"a single dry compound: no legal plan remains")
        return f"no stop fits in the {_laps(snapshot.remaining_laps)} left: only STAY_OUT to the flag is legal"
    wait = first - done - 1
    now = "pitting now is legal" if wait == 0 else f"stay out {_laps(wait)} first, then pit"
    return f"the earliest legal stop is at the end of lap {first} ({now}), the latest at the end of lap {last}"


def pit_window(snapshot: Snapshot, config=None) -> dict:
    """Whether pitting now (a stop at the end of the next lap) is legal, and the earliest and
    latest laps at whose end a stop is legal (None when no stop fits), from the regulations and
    the laps so far only (stop_window), with the same in words (_window_text)."""
    first, last = stop_window(snapshot)
    done = snapshot.laps_completed
    fits = first <= last
    return {"pit_now_legal": bool(fits and first == done + 1), "earliest_stop_lap": first if fits else None,
            "latest_stop_lap": last if fits else None, "laps_remaining": snapshot.remaining_laps,
            "summary": _window_text(snapshot, config)}


def _stop_at(fit, snapshot: Snapshot, lap: int, compound: str, continuation):
    """(plan, errors) of a stop at the end of lap `lap` onto `compound`, the current set run until
    then and the continuation after the stop: expand_action's pit, preceded by a stay when the
    stop is later than the end of the next lap. Under SC or VSC a later stop is instead the best
    full plan from this snapshot whose first stop is at the end of `lap` onto `compound`
    (continuation.stop_plan), so the period in progress runs on as the belief draws it per sample
    rather than being carried to the stop by an advanced snapshot; a stop now is unchanged."""
    pit = {"action": "pit", "compound": compound}
    wait = lap - snapshot.laps_completed - 1
    if wait <= 0:
        return fit.expand_action(snapshot, pit, continuation)
    stop_plan = getattr(continuation, "stop_plan", None)
    if snapshot.track_status != "green" and stop_plan is not None:
        from pydantic import ValidationError
        try:
            return Plan.of(stop_plan(snapshot, lap, compound)), []
        except ValidationError as exc:
            return None, [e["msg"] for e in exc.errors()]
    inner = {}

    def after_stay(after):
        plan, errors = fit.expand_action(after, pit, continuation)
        if errors:
            inner["errors"] = errors
            return [[after.compound, after.remaining_laps]]   # a placeholder; the errors are returned
        return plan
    plan, errors = fit.expand_action(snapshot, {"action": "stay", "laps": wait}, after_stay)
    return (None, inner["errors"]) if inner.get("errors") else (plan, errors)


def describe_plan(snapshot: Snapshot, pairs) -> str:
    """A plan in words from the snapshot: its first stop (the lap at whose end and the compound),
    or that it runs the current set to the flag."""
    pairs = [list(p) for p in pairs]
    if len(pairs) < 2:
        return "the current set runs to the flag, no further stop"
    stop = snapshot.laps_completed + int(pairs[0][1])
    more = len(pairs) - 2
    return (f"the first stop is at the end of lap {stop} onto {pairs[1][0]}"
            + (f", then {more} more stop{'s' if more != 1 else ''}" if more else ", no further stop"))


def explain_rule_errors(snapshot: Snapshot, plan, errors, config=None) -> list[str]:
    """`errors` (a plan's rule replies) as the agent is shown them: the minimum-stint, two-compound
    and lap-count messages of rules.py restated with which stint and why, and the nearest legal
    alternative (_window_text); any other message unchanged. `plan` is the plan the errors are
    about (a contracts.Plan; None when it did not parse, and then nothing changes). Only the
    snapshot, the plan and the regulations enter, never a lap after the snapshot."""
    errors = list(errors or [])
    if not errors or plan is None or not hasattr(plan, "stints"):
        return errors
    floor, run, done = snapshot.race.min_stint_laps, _rules.laps_on_set(snapshot), snapshot.laps_completed
    stints = plan.stints
    ctx = _rules.RuleContext(snapshot, plan, set(), config)
    window = _window_text(snapshot, config)
    window = window[0].upper() + window[1:]
    stated = []
    for i, s in enumerate(stints):   # rules.min_stint's test, stint by stint
        laps = run + s.laps if i == 0 else s.laps
        if (i == 0 and len(stints) == 1) or laps >= floor or _rules._class_change(ctx, i):
            continue
        if i == 0:
            cause = (f"stopping at the end of lap {done + s.laps} makes the current {s.compound} stint "
                     f"{_laps(laps)} ({run} run so far + {s.laps})")
        elif i == len(stints) - 1:
            cause = f"the final stint ({s.compound}) would be {_laps(laps)}"
        else:   # a stint between two stops: the window is not the fix
            stated.append(f"Below the {floor}-lap minimum stint: stint {i + 1} ({s.compound}) would be "
                          f"{_laps(laps)}. Make it at least {floor} laps or drop one of the stops around it.")
            continue
        stated.append(f"Below the {floor}-lap minimum stint: {cause}. {window}.")
    # restated in order when the count agrees with rules.py's; otherwise as written plus the window
    restate = sum(bool(_MIN_STINT.match(e)) for e in errors) == len(stated)
    out = []
    for e in errors:
        if _MIN_STINT.match(e):
            out.append(stated.pop(0) if restate else f"{e} {window}.")
        elif e == _TWO_COMPOUNDS:
            used = sorted(snapshot.compounds_used | {s.compound for s in stints})
            first, last = stop_window(snapshot)
            if first <= last:
                fix = f"so it needs a stop onto another dry compound. {window}"
                if (stay := last - done - 1) >= 1:
                    fix += f"; a stay_out of at most {_laps(stay)} keeps that stop possible"
            else:
                fix = f"and no stop fits in the {_laps(snapshot.remaining_laps)} left: no legal dry plan remains"
            out.append(f"Single compound: this plan runs only {', '.join(used)} over the race; a dry race must use "
                       f"at least two different compounds, {fix}.")
        elif (m := _LAP_SUM.match(e)):
            have, need = int(m.group(1)), int(m.group(2))
            last_laps = stints[-1].laps + need - have
            fix = f"e.g. make the last stint {_laps(last_laps)}" if last_laps >= 1 else "shorten the stints"
            out.append(f"Lap count: the stints sum to {_laps(have)} but {need} remain ({done} of "
                       f"{snapshot.race.total_laps} done); they must sum to exactly {need}, {fix}.")
        else:
            out.append(e)
    return out


# --- the functions --------------------------------------------------------------------------
def race_state(snapshot: Snapshot, *, config: SimConfig | None = None) -> dict:
    """The snapshot dump plus legal_compounds, remaining_laps and event_rates."""
    return _jsonable({**snapshot.model_dump(mode="json"), "legal_compounds": legal_compounds(snapshot, config),
                      "remaining_laps": snapshot.remaining_laps, "event_rates": event_rates(snapshot, config)})


def interval_meaning(degradation) -> str:
    """One short phrase for what the tools' intervals mean: INTERVAL_MEANING_WIDENED when the fit
    carries sd_scale (degradation._scale_sd), else INTERVAL_MEANING_RAW."""
    rep = getattr(degradation, "fit_report", None) or {}
    return INTERVAL_MEANING_WIDENED if rep.get("sd_scale") else INTERVAL_MEANING_RAW


def model_report(snapshot: Snapshot, ages=None, *, n_samples=None, sim=None, profile=None) -> dict:
    """Simulator.model_report's dump plus, per modelled compound:

      deg_per_lap_s             {low, median, high}, ordered: the 5th, 50th and 95th
                                percentiles of the posterior draws the simulator takes (SEED,
                                n_samples, the wear bound respected) when draws are on, else
                                the fitted wear -+ 1.64 sd; sd NaN gives low = median = high =
                                the fitted wear and a no_sd flag
      base_s_interval           the base lap time, always -+ 1.64 sd
      hinge_s_per_lap_interval  the hinge slope -+ 1.64 sd, where the model has one
      point_estimate            {interval name: the fitted value}, the constrained mode for a
                                bounded wear (the fit the simulator runs without draws)
      interval_basis            {interval name: BASIS_DRAWS, BASIS_SD or BASIS_NO_SD}
      wear_at_bound             the fit put the wear on its lower bound
      fuel_wear_corr            corr(fuel slope, wear slope), None without a fuel term
      flags                     the report's flags for this compound, plus no_sd:<name>
      pace_source, wear_source  with the compound proxy on (proxy.py): where the base and the
                                wear come from: race (laps), or for a compound without race
                                laps a blend of practice (long runs this weekend), qualifying
                                (pace only) and cross_race_step (the typical step from a
                                neighbouring compound), joined by "+"
      pace_sources, wear_sources  the blend's parts: [{source, weight, anchor, value, step}]
      qualifying_gap_s          the measured qualifying gap to the nearest raced compound on
                                fresh tyres, {anchor, gap_s, sd_s, drivers}, None without one
      price_basis               all of it in words, e.g. "SOFT: pace from practice long runs
                                40%, qualifying 35%, the cross-race step 25% (anchor MEDIUM);
                                ...". A proxied compound's price has no race laps behind it.
      tyre_life                 with the wear cliff on (sim/cliff.py): the belief's estimate of
                                the compound's tyre life at the snapshot, {longest_run_seen_laps
                                (practice and this race so far, None when nothing was seen),
                                seen_source, prior_life_laps and prior_sd_laps (the pre-race
                                estimate from earlier races, None when the runs seen already
                                exceed it), prior_source, cost_past_life_s_per_lap (the extra lap time
                                past it, in words), note}. An estimate with uncertainty, never a
                                limit: it reads nothing after the snapshot.

    and fuel_s_per_lap_interval (-+ 1.64 sd, its fuel_interval_basis and
    fuel_point_estimate) for models with a
    fuel term, a no_sd:fuel_s_per_lap_interval flag when its sd is NaN. flags and
    calibration_error_s are the report's; calibration_error_note (calibration_note()) says how
    to read it: under the paired gap yardstick (the default) an error of the absolute remaining
    time, not a tolerance for gaps between plans; under the legacy one a conservative tolerance
    for them (calibration_error_s()). This is estimate_tyre_degradation's body."""
    sim, profile = _setup(snapshot, sim, profile)
    rep = sim.model_report(snapshot, ages)
    deg = sim.fitted_for(snapshot).degradation
    fit_report = getattr(deg, "fit_report", None) or {}
    draws = _wear_draws(deg, n_samples or profile.sim.n_samples)
    bounded = fit_report.get("bounded") or {}
    cliffs = sim.cliffs_for(snapshot)
    out = rep.model_dump()
    for c, r in rep.compounds.items():
        if not r.modelled:
            continue
        entry = out["compounds"][c]
        flags = [f for f in rep.flags if f.endswith(f":{c}")]
        wear = r.wear_s_per_lap
        pairs = {"deg_per_lap_s": _interval(wear.value, wear.sd, None if draws is None else draws.get(c)),
                 "base_s_interval": _interval(r.base_s.value, r.base_s.sd)}
        hinge = (rep.hinge or {}).get(c)
        if hinge is not None:
            pairs["hinge_s_per_lap_interval"] = _interval(hinge["value"], hinge["sd"])
        flags += [f"no_sd:{name}" for name, (_, basis) in pairs.items() if basis == BASIS_NO_SD]
        point = {"deg_per_lap_s": wear.value, "base_s_interval": r.base_s.value}
        if hinge is not None:
            point["hinge_s_per_lap_interval"] = hinge["value"]
        entry.update({name: iv for name, (iv, _) in pairs.items()}, point_estimate=point,
                     interval_basis={name: basis for name, (_, basis) in pairs.items()},
                     wear_at_bound=bool(bounded.get(c, False)),
                     fuel_wear_corr=(rep.fuel_wear_corr or {}).get(c), flags=flags)
        life = cliffs.get(c)
        if life is not None:   # SimConfig.wear_cliff: the as-of tyre-life estimate, never a limit
            entry["tyre_life"] = _tyre_life(life)
        tiers = (fit_report.get("proxy") or {}).get(c)
        if tiers is not None:   # SimConfig.compound_proxy: where this compound's pace and wear come from
            entry.update(pace_source=tiers["pace_source"], wear_source=tiers["wear_source"],
                         pace_sources=tiers.get("pace_sources", []), wear_sources=tiers.get("wear_sources", []),
                         qualifying_gap_s=tiers.get("qualifying_gap_s"), price_basis=_proxy.describe(c, tiers))
    out["interval_meaning"] = interval_meaning(deg)
    if rep.fuel_s_per_lap is not None:
        iv, basis = _interval(rep.fuel_s_per_lap.value, rep.fuel_s_per_lap.sd)
        out["fuel_s_per_lap_interval"], out["fuel_interval_basis"] = iv, basis
        out["fuel_point_estimate"] = rep.fuel_s_per_lap.value
        if basis == BASIS_NO_SD:
            out["flags"] = [*out["flags"], "no_sd:fuel_s_per_lap_interval"]
    out["calibration_error_note"] = calibration_note()
    out["event_rates"] = event_rates(snapshot, sim.config)
    out["traffic"] = traffic_report(snapshot, sim)
    return _jsonable(out)


def traffic_on(config: SimConfig) -> bool:
    return config.traffic.name != "none"


def _band(b, digits=1):
    return None if b is None else {k: round(float(v), digits) for k, v in b.items()}


def traffic_view(est) -> dict | None:
    """A plan's TrafficEstimate, compact: expected_rejoin_position and its 5-95 band,
    cars_in_pit_window (cars within the first stop's pit loss behind, which come by while we
    are in the pit lane), time_lost_s (included in the remaining time; negative: passes that
    net gain) and final_position. None for an invalid plan."""
    if est is None:
        return None
    est = est if isinstance(est, dict) else est.model_dump()
    rejoin = est.get("rejoin_position")
    return {"expected_rejoin_position": None if rejoin is None else round(float(rejoin["median"]), 1),
            "rejoin_position_band": None if rejoin is None else [round(float(rejoin["low"]), 1),
                                                                 round(float(rejoin["high"]), 1)],
            "cars_in_pit_window": (None if est.get("cars_in_pit_window") is None
                                   else round(float(est["cars_in_pit_window"]), 1)),
            "time_lost_s": _band(est["time_lost_s"]), "final_position": _band(est["final_position"])}


def traffic_report(snapshot: Snapshot, sim) -> dict:
    """The belief's traffic at the snapshot: {modelled: False, note} with traffic off; else
    modelled, the basis in words, cars_within_pit_loss_behind_now (other cars whose line
    crossing at the end of this lap is within the pit loss behind ours: a stop now drops
    behind them), dirty_air_s (the time a lap in dirty air costs, estimated from the laps so
    far) and own_pace_s (our pace over the model on the track). Everything is from the laps up
    to the snapshot (traffic.py, source "projected")."""
    if not traffic_on(sim.config):
        return {"modelled": False, "note": TRAFFIC_NOTE}
    fit = sim.fitted_for(snapshot)
    laps = np.arange(snapshot.laps_completed + 1, snapshot.race.total_laps + 1)
    state = fit.traffic.prepare(snapshot, laps, fit.laps, snapshot.driver, ref_lap_s=fit.ref_lap_s,
                                model=fit.degradation)
    if state is None:
        return {"modelled": True, "available": False, "basis": TRAFFIC_BASIS,
                "note": "no lap completed yet: the field cannot be placed"}
    pit = float(snapshot.race.pit_loss_s or 0.0)
    start = state.table.G_prev[:, 0] if len(state.table.laps) else np.empty(0)
    behind = int(((start > state.anchor) & (start <= state.anchor + pit)).sum())
    out = {"modelled": True, "available": True, "basis": TRAFFIC_BASIS,
           "cars_within_pit_loss_behind_now": behind, "dirty_air_s": round(float(state.dirty_air_s), 2),
           "own_pace_s": round(float(state.own_offset_s), 2)}
    if getattr(state, "difficulty", None) is not None:   # traffic difficulty "asof" (overtaking.py)
        out["passing"] = passing_words(state.difficulty)
    return out


# the passing difficulty in generic words (no circuit, season or event): the per-lap chance that a
# car within a second of the one ahead passes it, from overtaking.difficulty_prior
PASSING_BANDS = ((0.05, "very hard: a faster car usually stays stuck behind for many laps"),
                 (0.12, "hard: a car needs a large pace advantage to pass"),
                 (0.20, "average"),
                 (1.01, "easy: a faster car usually gets by within a lap or two"))
PASSING_SOURCES = {1: "earlier races at this track", 2: "earlier races at other tracks only",
                   3: "a general prior (no earlier race)"}


def passing_words(d) -> dict:
    """{difficulty, pass_chance_per_lap_close, pace_advantage_to_pass_s, from}: overtaking.Difficulty
    in words a tool may show."""
    words = next(w for cut, w in PASSING_BANDS if d.rate < cut)
    return {"difficulty": words,
            "pass_chance_per_lap_close": [round(float(d.rate_lo), 3), round(float(d.rate), 3), round(float(d.rate_hi), 3)],
            "pace_advantage_to_pass_s": round(float(d.theta_s), 2),
            "from": PASSING_SOURCES.get(int(d.tier), PASSING_SOURCES[3])}


NEUTRAL_HORIZONS = (5, 10)   # laps ahead for the chance of a new SC / VSC (pit_loss)
RATE_BAND = 2.0              # the rate's uncertainty: a factor of 2 either way (provisional rates)


def _neutralised_for(snapshot: Snapshot) -> int:
    """Consecutive laps up to the snapshot under its current status (0 when green)."""
    laps = set(snapshot.sc_laps if snapshot.track_status == "sc" else
               snapshot.vsc_laps if snapshot.track_status == "vsc" else [])
    n, lap = 0, snapshot.laps_completed
    while lap in laps:
        n, lap = n + 1, lap - 1
    return n


def neutralisation_outlook(snapshot: Snapshot, config: SimConfig | None = None) -> dict:
    """The chance a new SC or VSC period starts within each of NEUTRAL_HORIZONS laps (capped at
    the laps left), from the rates the belief draws with (event_rates): 1 - (1 - p)^n, with a
    low / high band from the rate divided and multiplied by RATE_BAND. Only the rates as of the
    race start and the status so far enter; nothing about later neutralisations."""
    r = event_rates(snapshot, config)
    if r["p_sc"] is None:
        return {"source": r["source"]}
    left = snapshot.remaining_laps

    def chance(p, n):
        return {"low": round(1 - (1 - p / RATE_BAND) ** n, 3), "median": round(1 - (1 - p) ** n, 3),
                "high": round(1 - (1 - min(1.0, p * RATE_BAND)) ** n, 3)}
    out = {f"next_{h}_laps": {"sc": chance(r["p_sc"], min(h, left)), "vsc": chance(r["p_vsc"], min(h, left))}
           for h in NEUTRAL_HORIZONS}
    return {**out, "mean_period_laps": {"sc": r["d_sc"], "vsc": r["d_vsc"]}, "rates_source": r["source"],
            "band": f"rate / {RATE_BAND:g} to rate x {RATE_BAND:g}"}


def _tyre_life(life) -> dict:
    """model_report's tyre_life entry of one compound's belief cliff (cliff.Cliff)."""
    seen = life.lower if life.prior is not None else life.age
    seen_src = life.seen_source if life.prior is not None else life.source
    return {"longest_run_seen_laps": seen if seen_src is not None else None, "seen_source": seen_src,
            "prior_life_laps": life.prior, "prior_sd_laps": life.prior_sd if life.prior is not None else None,
            # the tier only: a source never names an earlier race (the agent's view is anonymous)
            "prior_source": str(life.source).split(":")[0] if life.prior is not None else None,
            "cost_past_life_s_per_lap": f"min({life.kappa:g} x laps_past^2, {life.cap:g})",
            "note": _cliff.describe(life)}


def pit_loss(snapshot: Snapshot, *, sim=None, profile=None) -> dict:
    """pit_loss_s {low, median, high}: the pit loss the belief prices a green stop with at this
    snapshot (Simulator.pit_loss_inputs). Under the as-of belief (SimConfig.pit_loss.source
    "asof", configs/sim/belief.yaml) it is pit_loss.estimate: a prior from earlier races
    updated with this race's green stops seen so far, -+ 1.64 sd, source "asof", with `basis`
    (prior, its source, stops seen). A calibrator's sd (calibrators.effective_factors), when it
    gives one, takes over (source "calibrator"); with no sd at all a fixed 10 percent band,
    source "config". pit_loss_now_s is that interval times pit_loss_now_factor, the factor for
    the current track status (the as-of SC / VSC factor under SC / VSC, 1 when green), and
    neutralised_for_laps how long the current SC / VSC has lasted. `outlook` is the chance of a
    new SC or VSC in the coming laps (neutralisation_outlook). Never the whole-race value the
    world scores with. estimate_pit_loss's body."""
    sim, _ = _setup(snapshot, sim, profile)
    fit = sim.fitted_for(snapshot)
    pl = fit.pit_loss_inputs(snapshot)
    mean, sd = float(pl["pit_loss_s"]), float(pl["sd_s"])
    source = "asof" if pl["source"] == "asof" else None
    if fit.calibration is not None and pl["source"] != "asof":
        mean, sd = calibrators.effective_factors(fit, snapshot.race)["pit_loss_s"]
    if not _nan(sd) and sd > 0:
        band, source = Z90 * sd, source or "calibrator"
    else:
        band, source, sd = PIT_LOSS_BAND * mean, "config", None
    green = {"low": mean - band, "median": mean, "high": mean + band}
    sc, vsc = pl["sc_factor"], pl["vsc_factor"]
    factor = {"sc": sc, "vsc": vsc}.get(snapshot.track_status, 1.0)
    out = {"pit_loss_s": green, "source": source, "sd_s": sd,
           "pit_loss_now_s": {k: v * factor for k, v in green.items()}, "pit_loss_now_factor": factor,
           "sc_pit_loss_factor": sc, "vsc_pit_loss_factor": vsc,
           "factor_source": "asof" if pl["source"] == "asof" else "config",
           "track_status": snapshot.track_status, "neutralised_for_laps": _neutralised_for(snapshot),
           "outlook": neutralisation_outlook(snapshot, sim.config)}
    if pl["source"] == "asof":
        out["basis"] = {"prior_s": pl["prior_s"], "prior_source": pl["prior_source"],
                        "green_stops_seen": pl["n_green_stops"], "sc_stops_seen": pl["n_sc_stops"],
                        "vsc_stops_seen": pl["n_vsc_stops"],
                        "sc_factor_sd": round(pl["sc_factor_sd"], 3), "vsc_factor_sd": round(pl["vsc_factor_sd"], 3)}
    return _jsonable(out)


def simulate_plan(snapshot: Snapshot, stints, n_samples=None, *, sim=None, profile=None) -> dict:
    """SimResult of one plan for the remaining laps ([[compound, laps], ...] or a Plan)."""
    sim, profile = _setup(snapshot, sim, profile)
    res = sim.simulate(snapshot, stints, seed=SEED, n_samples=n_samples or profile.sim.n_samples)
    return _jsonable(res.model_dump())


def compare_plans(snapshot: Snapshot, candidates, n_samples=None, *, sim=None, profile=None) -> dict:
    """CompareResult of plans and actions under shared draws; actions are expanded with the
    optimizer's continuation on the belief fit at the snapshot."""
    sim, profile = _setup(snapshot, sim, profile)
    n = _compare_n(profile, n_samples)
    fit = sim.fitted_for(snapshot)
    cands = [_candidate(snapshot, c) for c in candidates]
    res = fit.compare(snapshot, cands, seed=SEED, n_samples=n,
                      continuation=_continuation(fit, n, profile.sim.max_stops))
    out = res.model_dump()
    out["errors"] = [explain_rule_errors(snapshot, p, e, sim.config) for p, e in zip(res.candidates, res.errors)]
    if gap_yardstick() == "paired" and res.best is not None:
        # the fastest candidate (lowest mean) against each other valid one, over the shared draws
        i = res.best
        out["gap_readings"] = [
            gap_reading(f"candidates[{i}]", f"candidates[{j}]", res.p_beats[i][j], res.gap_mean_s[i][j],
                        res.gap_mean_s[i][j] - Z90 * res.gap_std_s[i][j], res.gap_mean_s[i][j] + Z90 * res.gap_std_s[i][j])
            for j in range(len(res.candidates)) if j != i and res.valid[j]]
    if res.traffic is not None:   # traffic on: compact per-candidate views and the basis
        out["traffic"] = [traffic_view(t) for t in res.traffic]
        out["traffic_basis"] = TRAFFIC_BASIS
    else:
        out["traffic"] = TRAFFIC_NOTE
    return _jsonable(out)


def action_label(raw) -> str:
    """An action in the agent's words: "PIT MEDIUM", "PIT MEDIUM at the end of lap 3",
    "STAY_OUT 3 laps"."""
    act = _action(raw)
    if act.get("action") == "pit":
        if act.get("at_end_of_lap") is not None:
            return f"PIT {act.get('compound')} at the end of lap {act['at_end_of_lap']}"
        return f"PIT {act.get('compound')}"
    return f"STAY_OUT {_laps(int(act.get('laps', 1)))}"


def gap_reading(a: str, b: str, p_a, gap, lo, hi) -> str | None:
    """The gap between a and b in words: which is faster in what share of the shared draws, then
    a minus b and its 90% interval (lo, hi), e.g. "STAY_OUT 1 lap is faster than PIT MEDIUM in 82%
    of draws; STAY_OUT 1 lap minus PIT MEDIUM: -3.1 s (90%: -6.0 to -0.2 s)". None when a number
    is missing."""
    if any(v is None or _nan(v) for v in (p_a, gap, lo, hi)):
        return None
    fav, other, p = (a, b, p_a) if p_a >= 0.5 else (b, a, 1 - p_a)
    return (f"{fav} is faster than {other} in {p:.0%} of draws; {a} minus {b}: {gap:+.1f} s "
            f"(90%: {lo:+.1f} to {hi:+.1f} s)")


def evaluate_action(snapshot: Snapshot, action, n_samples=None, *, sim=None, profile=None) -> dict:
    """One action against its alternative, from one compare: a pit against staying out one
    lap; a stay against pitting now on the legal, modelled compound whose pit compare's
    p_best prefers (ties: the lower mean, then the order of legal_compounds).

      relative_race_time_delta_s   {optimistic, median, pessimistic}: the 5th, 50th and 95th
                                   percentiles of action time - alternative time over the
                                   samples (negative: the action is faster)
      gap_mean_s, gap_std_s        the same gap's mean and sd
      probability_improves_outcome p_beats, the share of samples where the action is faster;
                                   it covers parameter and lap-noise uncertainty under this
                                   model and does not include model error (probability_covers)
      gap_reading                  (AGENT_GAP_YARDSTICK "paired", the default) the gap in words
                                   (gap_reading): which side is faster in what share of the
                                   shared draws, the median gap and its 5-95 percentile
                                   interval, the model's own paired spread
      within_sim_error             (AGENT_GAP_YARDSTICK "calibration", legacy) |gap_mean| <
                                   max(2 gap_std, calibration_error_s), with
      confidence_basis             "model error" when calibration_error_s exceeds 2 gap_std,
                                   else "sampling spread", and
      calibration_error_s          a conservative tolerance for gaps (calibration_error_s())
      dominated, support_warnings  the action's, from the compare
      alternative                  its action, plan and remaining-time summary

      traffic                      with belief traffic on: {action, alternative} as
                                   traffic_view (expected rejoin position and band, cars in
                                   the pit window, time lost in traffic, final position) and
                                   basis (TRAFFIC_BASIS); TRAFFIC_NOTE with it off

    The numbers are None when no alternative is legal; an illegal action returns {"error": ...}.
    Rule replies are explain_rule_errors'."""
    sim, profile = _setup(snapshot, sim, profile)
    n = _compare_n(profile, n_samples)
    fit = sim.fitted_for(snapshot)
    act = _action(action)
    cont = _continuation(fit, n, profile.sim.max_stops)
    window = pit_window(snapshot, sim.config)
    note = None
    pits = [c for c in legal_compounds(snapshot, sim.config) if c in fit.modelled]
    if act.get("action") == "pit":
        alts = [{"action": "stay", "laps": 1}]
    elif window["earliest_stop_lap"] is None:
        # no stop is legal: staying out to the flag is the only action, there is nothing to compare
        alts = []
        note = (f"{window['summary'][0].upper()}{window['summary'][1:]}; there is no other legal action to "
                f"compare this with.")
    elif window["pit_now_legal"]:
        alts = [{"action": "pit", "compound": c} for c in pits]
    else:
        # pitting now is illegal: the nearest legal alternative is a stop at the earliest legal lap
        first = window["earliest_stop_lap"]
        alts = [{"action": "pit", "compound": c, "at_end_of_lap": first} for c in pits]
        note = (f"Pitting now is not legal; the alternative compared is a stop at the end of lap {first}, the "
                f"earliest legal stop.")

    def candidate(a):
        if a.get("at_end_of_lap") is None:
            return _candidate(snapshot, a)
        plan, errors = _stop_at(fit, snapshot, a["at_end_of_lap"], a["compound"], cont)
        return plan if plan is not None else {"action": "pit", "compound": a["compound"]}   # invalid either way
    res = fit.compare(snapshot, [candidate(a) for a in [act, *alts]], seed=SEED, n_samples=n, continuation=cont)
    if not res.valid[0]:
        errors = explain_rule_errors(snapshot, res.candidates[0], res.errors[0], sim.config)
        return _jsonable({"action": act, "valid": False, "errors": errors, "error": "; ".join(errors)})
    ok = [i for i in range(1, len(res.candidates)) if res.valid[i]]
    j = (max(ok, key=lambda i: (res.p_best[i], -res.summaries[i].mean_s, -i)) if ok
         else 1 if len(res.candidates) > 1 else None)
    legacy = gap_yardstick() == "calibration"
    gap_fields = ({"within_sim_error": None, "confidence_basis": None,
                   "calibration_error_s": calibration_error_s(sim, snapshot), "calibration_error_note": CALIBRATION_NOTE}
                  if legacy else {"gap_reading": None})
    out = {"action": act, "plan": res.candidates[0].as_pairs(), "remaining": res.summaries[0].model_dump(),
           "relative_race_time_delta_s": {"optimistic": None, "median": None, "pessimistic": None},
           "gap_mean_s": None, "gap_std_s": None, "probability_improves_outcome": None,
           "probability_covers": PROBABILITY_COVERS, "interval_meaning": interval_meaning(fit.degradation),
           **gap_fields,
           "dominated": res.dominated[0], "support_warnings": res.support_warnings[0],
           "traffic": TRAFFIC_NOTE, "alternative": None, "n_samples": res.n_samples, "seed": res.seed,
           "assumptions": [f"SC and VSC rates: {event_rates(snapshot, sim.config)['source']}"],
           "continuation": continuation_text(snapshot, act, res.candidates[0].as_pairs()),
           "pit_window": window["summary"]}
    if note:
        out["note"] = note
    if res.traffic is not None:
        out["traffic"] = {"action": traffic_view(res.traffic[0]),
                          "alternative": traffic_view(res.traffic[j]) if j is not None else None,
                          "basis": TRAFFIC_BASIS}
    if j is None:
        return _jsonable(out)
    alt = alts[j - 1]
    if not res.valid[j]:
        out["alternative"] = {"action": alt, "valid": False,
                              "errors": explain_rule_errors(snapshot, res.candidates[j], res.errors[j], sim.config)}
        return _jsonable(out)
    out["alternative"] = {"action": alt, "valid": True, "plan": res.candidates[j].as_pairs(),
                          "remaining": res.summaries[j].model_dump()}
    # the gap's percentiles: the two expanded plans again under the same draws (R6)
    times = fit.simulate_batch(snapshot, [res.candidates[0], res.candidates[j]], seed=SEED, n_samples=n).times
    lo, med, hi = np.percentile(times[0] - times[1], [5, 50, 95])
    gap, std, p = res.gap_mean_s[0][j], res.gap_std_s[0][j], res.p_beats[0][j]
    out.update(relative_race_time_delta_s={"optimistic": float(lo), "median": float(med), "pessimistic": float(hi)},
               gap_mean_s=gap, gap_std_s=std, probability_improves_outcome=p)
    if legacy:
        cal = out["calibration_error_s"]
        model_error = not _nan(cal) and cal > 2 * std
        out.update(within_sim_error=bool(abs(gap) < (cal if model_error else 2 * std)),
                   confidence_basis="model error" if model_error else "sampling spread")
    else:
        out["gap_reading"] = gap_reading(action_label(act), action_label(alt), p, float(med), float(lo), float(hi))
    if act.get("action") == "stay" and window["earliest_stop_lap"] is not None:
        out["stop_lap_scan"] = stop_lap_scan(fit, snapshot, res.candidates[0], alt["compound"], window, cont, n)
    return _jsonable(out)


SCAN_POINTS = 5     # stop laps in evaluate_action's stop_lap_scan
SCAN_SPAN = 20      # laps after the earliest legal stop the scan reaches, at most


def stop_lap_scan(fit, snapshot: Snapshot, plan, compound: str, window: dict, cont, n: int) -> dict:
    """The evaluated plan against a stop at the end of each of up to SCAN_POINTS laps, evenly
    spread from the earliest legal stop over at most SCAN_SPAN laps (and no later than the latest),
    onto `compound` with the best continuation after it, under the same draws: per stop lap the
    mean of that plan's time minus this plan's and the share of draws it is faster. One compare."""
    first, last = window["earliest_stop_lap"], min(window["latest_stop_lap"], window["earliest_stop_lap"] + SCAN_SPAN)
    laps = sorted({int(round(x)) for x in np.linspace(first, last, SCAN_POINTS)})
    plans, kept = [], []
    for lap in laps:
        p, errors = _stop_at(fit, snapshot, lap, compound, cont)
        if p is not None and not errors:
            plans.append(p)
            kept.append(lap)
    if not plans:
        return {"compound": compound, "rows": []}
    res = fit.compare(snapshot, [plan, *plans], seed=SEED, n_samples=n)
    rows = []
    for i, lap in enumerate(kept, start=1):
        if res.valid[i] and res.gap_mean_s[i][0] is not None:
            rows.append({"stop_lap": lap, "minus_this_plan_s": round(float(res.gap_mean_s[i][0]), 1),
                         "faster_share": round(float(res.p_beats[i][0]), 2)})
    return {"compound": compound,
            "meaning": f"a stop at the end of each lap onto {compound} with the best continuation after it, against "
                       "this plan: its mean time minus this plan's (negative: that stop lap is faster) and the share "
                       "of draws it is faster",
            "rows": rows}


def continuation_text(snapshot: Snapshot, act: dict, pairs) -> str:
    """What an evaluated action's plan is, in words: a stay of k laps is a minimum (the best
    continuation then picks its own stop), so the plan's first stop is named."""
    if act.get("action") == "stay":
        k = int(act.get("laps", 1))
        return f"staying out at least {_laps(k)}; with the best continuation after that, {describe_plan(snapshot, pairs)}"
    return f"pitting now: {describe_plan(snapshot, pairs)}"


def submission_error(snapshot: Snapshot, decision: dict, *, sim=None, profile=None) -> str | None:
    """Why the scorer would reject this submission, said with the legal alternative; None when it
    accepts it. The checks are the scorer's own (eval.decisions: the decision, a compound for a PIT,
    a whole reassess_after_laps of at least 1 for a STAY_OUT, the confidence in [0, 1] for p_pit),
    then the decision is expanded exactly as the scorer expands it (decisions.to_plan on this belief,
    seed 0, the profile's samples and stops) and the plan is checked against the rules."""
    from ..eval import decisions
    sim, profile = _setup(snapshot, sim, profile)
    window = pit_window(snapshot, sim.config)
    legal = window["summary"][0].upper() + window["summary"][1:]
    remaining, done = snapshot.remaining_laps, snapshot.laps_completed
    try:
        kind = decisions.kind(decision)
    except decisions.InvalidDecision:
        return "decision must be PIT or STAY_OUT."
    conf = decision.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        return f"confidence must be a number from 0 to 1, got {conf!r}."
    fit = sim.fitted_for(snapshot)
    pits = [c for c in legal_compounds(snapshot, sim.config) if c in fit.modelled]
    if kind == "PIT":
        compound = decision.get("compound")
        if not compound or str(compound).upper() not in pits:
            return f"PIT needs a compound that can be fitted now: one of {pits}, got {compound!r}."
        if not window["pit_now_legal"]:
            first = window["earliest_stop_lap"]
            fix = (f"submit STAY_OUT with reassess_after_laps = {first - done - 1} to reassess when the stop becomes "
                   f"legal" if first is not None else f"submit STAY_OUT with reassess_after_laps = {remaining}")
            return f"Pitting now is not legal: {window['summary']}. {fix[0].upper()}{fix[1:]}."
    else:
        laps = decision.get("reassess_after_laps")
        if isinstance(laps, bool) or not isinstance(laps, int) or laps < 1:
            fix = (f"only STAY_OUT to the flag is legal: use reassess_after_laps = {remaining}"
                   if window["earliest_stop_lap"] is None else
                   f"use a whole number from 1 to {remaining} ({remaining} stays out to the flag)")
            return f"STAY_OUT needs reassess_after_laps of at least 1, got {laps!r}; {remaining} laps remain: {fix}."
    try:
        plan = decisions.to_plan(snapshot, decision, sim, profile.sim.max_stops, seed=0,
                                 n_samples=profile.sim.n_samples)
    except decisions.InvalidDecision as exc:
        if decision.get("plan") is not None:   # protocol v2free's named plan (decisions.named_plan)
            return (f"The named plan cannot be scored: {exc}. Fix the plan ([[compound, laps], ...] for the "
                    f"{remaining} laps left, the first stint on the current set) or leave plan out.")
        if decision.get("first_stop_lap") is not None:   # protocol v2's kept stop (decisions.kept_plan)
            return (f"The named first stop cannot be kept: {exc}. Copy first_stop_lap and compound together from "
                    f"rank_plans' submit field, or leave first_stop_lap out (the rest is then planned again).")
        return f"This decision has no legal plan ({exc}). {legal}."
    errors = sim.validate(snapshot, plan)
    if errors:
        return f"This decision has no legal plan: {'; '.join(explain_rule_errors(snapshot, plan, errors, sim.config))}"
    return None


# --- protocol v2 (src/agent.py AGENT_PROTOCOL): ranked plans, one plan evaluated -----------------------
RANK_TOP = 4        # distinct plans rank_plans shows
FLEX_S = 1.0        # a shown plan's first stop may move over the laps whose plan of the same shape is within this
GENERIC_LIFE_TIER = 3   # tyre_history tiers from here on are generic priors (cliff.life_prior: no circuit history)
_RANKED: dict = {}  # (snapshot, n, stops, config) -> optimize_batch result with times, per snapshot


def _ranked(snapshot: Snapshot, fit, n: int, max_stops: int):
    """optimize_batch from the snapshot exactly as the belief_optimizer policy calls it
    (eval.decisions.belief_optimum: the belief fit, SEED, n samples, max_stops), with every legal plan
    in tie-break order (return_times). The first plan is that policy's plan. Cached per snapshot."""
    from .optimizer_bridge import optimize_batch
    key = (snapshot.model_dump_json(), n, max_stops)
    hit = _RANKED.get(key)
    if hit is None or hit[0] is not fit:   # the fit itself is held, so a cached result is never another fit's
        if len(_RANKED) > 32:
            _RANKED.clear()
        hit = _RANKED[key] = (fit, optimize_batch(snapshot, fit, seed=SEED, n_samples=n, max_stops=max_stops,
                                                  return_times=True))
    return hit[1]


def plan_text(pairs) -> str:
    """A plan compactly: "MEDIUM 11 > SOFT 26"."""
    return " > ".join(f"{c} {int(n)}" for c, n in [list(p)[:2] for p in pairs])


def immediate_action(snapshot: Snapshot, pairs) -> dict:
    """What a plan from the snapshot does now and the submission that keeps it: a first stint of one
    lap is PIT onto the next compound now; a longer one is STAY_OUT with reassess_after_laps one
    less than the laps to its first stop (a STAY_OUT of k laps allows no stop before the end of lap
    laps_completed + k + 1, eval.decisions); no stop is STAY_OUT to the flag. A submission with a
    stop names it (first_stop_lap and compound), so the scorer keeps this plan (decisions.kept_plan)
    rather than planning again after the stay."""
    pairs = [list(p) for p in pairs]
    done = snapshot.laps_completed
    if len(pairs) < 2:
        k = snapshot.remaining_laps
        return {"action": "STAY_OUT", "first_stop_lap": None,
                "submit": {"decision": "STAY_OUT", "reassess_after_laps": k}}
    w, nxt = int(pairs[0][1]), pairs[1][0]
    # first_stop_lap and compound name the stop the submission keeps (eval.decisions.kept_plan)
    if w == 1:
        return {"action": f"PIT {nxt}", "first_stop_lap": done + 1,
                "submit": {"decision": "PIT", "compound": nxt, "first_stop_lap": done + 1}}
    return {"action": "STAY_OUT", "first_stop_lap": done + w,
            "submit": {"decision": "STAY_OUT", "reassess_after_laps": w - 1, "compound": nxt, "first_stop_lap": done + w}}


def _shape(snapshot, pairs) -> tuple:
    """Plans of one shape differ only in their stop laps: the immediate action kind and the compounds."""
    return ("PIT" if len(pairs) > 1 and int(pairs[0][1]) == 1 else "STAY", tuple(p[0] for p in pairs[1:]))


def _gap(d) -> dict:
    """A paired difference over the draws (plan minus reference): mean, 90% interval, share faster."""
    d = np.asarray(d, dtype=float)
    lo, hi = np.percentile(d, [5, 95])
    return {"gap_s": round(float(d.mean()), 1), "gap_90": [round(float(lo), 1), round(float(hi), 1)],
            "faster_share": round(float((d < 0).mean() + 0.5 * (d == 0).mean()), 2)}


def _stint_ages(snapshot: Snapshot, plan) -> list[tuple[str, float]]:
    """(compound, age at the end of the stint) per stint of a contract plan from the snapshot."""
    out = []
    for i, s in enumerate(plan.stints):
        start = snapshot.tyre_age if i == 0 else (s.start_age or 0)
        out.append((s.compound, float(start + s.laps)))
    return out


def uncertainty_flags(snapshot: Snapshot, sim, plans, *, profile=None) -> tuple[list[str], list[list[str]]]:
    """(general flags, per plan flags) in words, from what the belief knows at the snapshot: a compound
    of the plans priced without race laps (the compound proxy), a compound whose estimate leans on the
    prior (prior_dominated), a stint run past the compound's longest run seen so far when the tyre
    life beyond it is a generic prior (cliff tiers GENERIC_LIFE_TIER on) or past the life estimate
    itself, and a safety car or VSC in progress. Nothing after the snapshot enters."""
    fit = sim.fitted_for(snapshot)
    rep = sim.model_report(snapshot)
    fit_report = getattr(fit.degradation, "fit_report", None) or {}
    proxy = fit_report.get("proxy") or {}
    cliffs = sim.cliffs_for(snapshot)
    used = sorted({s.compound for p in plans for s in p.stints})
    general = []
    for c in used:
        tiers = proxy.get(c)
        if tiers is not None and (tiers.get("pace_source") != "race" or tiers.get("wear_source") != "race"):
            general.append(f"{c} has no race laps so far: pace from {tiers.get('pace_source')}, wear from "
                           f"{tiers.get('wear_source')} (a proxy)")
        elif f"prior_dominated:{c}" in rep.flags:
            general.append(f"{c}: few race laps so far, its estimate leans on the prior")
    if snapshot.track_status != "green":
        n = _neutralised_for(snapshot)
        general.append(f"{snapshot.track_status.upper()} in progress ({_laps(n)} so far): a stop now is cheaper; "
                       f"how long it lasts is drawn by the model")
    per_plan = []
    for p in plans:
        flags = []
        for c, age in _stint_ages(snapshot, p):
            life = cliffs.get(c)
            if life is None:
                continue
            seen = life.lower if life.prior is not None else life.age
            est = life.prior if life.prior is not None else life.age
            tier = str(life.source or "")
            generic = life.prior is not None and any(f"tier{t}" in tier for t in range(GENERIC_LIFE_TIER, 10))
            if est is not None and age > est:
                flags.append(f"runs {c} to age {age:.0f}, past its estimated life of {est:.0f} laps")
            elif seen and age > seen and generic:
                flags.append(f"runs {c} to age {age:.0f}, past the longest run seen ({seen:.0f}); the life "
                             f"beyond that is a generic prior")
        per_plan.append(flags)
    return general, per_plan


FAMILY_TOP = 8      # rank_plans(families=True): at most this many strategy families shown
FAMILY_GAP_S = 25.0  # ... and none more than this far behind plan 1 (the rest are counted as dropped)


def family(pairs) -> str:
    """A plan's strategy family: its stops left and the compounds after the current set, in words."""
    later = [p[0] for p in pairs[1:]]
    return "no more stops" if not later else f"{len(later)} stop{'s' if len(later) > 1 else ''}: {' > '.join(later)}"


def rank_plans(snapshot: Snapshot, *, families=False, n_samples=None, sim=None, profile=None) -> dict:
    """The optimizer's best distinct plans from the snapshot (protocol v2's first call), compact.

    The search is the belief_optimizer policy's (_ranked), so plans[0] is that policy's plan. At most
    RANK_TOP plans, the best of each shape (_shape: the immediate action kind and the compounds),
    with the best PIT-now and the best STAY_OUT plan always among them when both are legal. Per plan:
    its stints, first stop, the immediate action and the submission that keeps it (immediate_action),
    gap_to_best (mean, 90% interval over the paired draws, share of draws faster than the best),
    best_share (the share of draws in which it is the fastest of those shown), first_stop_range
    (the first-stop laps over which a plan of the same shape stays within FLEX_S of it), flags (the
    plan's uncertainty_flags) and extrapolates (the fit's support warnings: a note, not a flag).
    `decision` is plan 1's immediate action with the best PIT-now plan read against the best STAY_OUT
    plan (gap_reading, the choice the submission makes); `flags` are uncertainty_flags. Nothing after the snapshot enters.

    With `families` (protocol v2free) the plans shown span the strategy space instead: the best plan of
    each family (family(): stops left and compound sequence), in the order of their best plans, at most
    FAMILY_TOP and none more than FAMILY_GAP_S behind plan 1, plus the best PIT-now plan when it is in
    none of them; each entry names its family, and families_dropped counts the families left out."""
    from ..baselines.domain import NoFeasiblePlan
    sim, profile = _setup(snapshot, sim, profile)
    n = n_samples or profile.sim.n_samples
    fit = sim.fitted_for(snapshot)
    window = pit_window(snapshot, sim.config)
    try:
        res = _ranked(snapshot, fit, n, profile.sim.max_stops)
    except NoFeasiblePlan as exc:
        return {"error": f"No legal plan within the stop limit: {exc}", "recoverable": False,
                "pit_window": window["summary"]}
    s = res.diagnostics["searched"]
    plans, means, times = s["plans"], np.asarray(s["mean_time_s"]), np.asarray(s["times"])
    pairs = [p.as_pairs() for p in plans]
    shapes = [_shape(snapshot, p) for p in pairs]
    shown, seen = [], set()
    for i, sh in enumerate(shapes):
        if sh not in seen:
            seen.add(sh)
            shown.append(i)
    best_of = {k: next((i for i, sh in enumerate(shapes) if sh[0] == k), None) for k in ("PIT", "STAY")}
    dropped = None
    if families:
        fams, seen = [], set()
        for i, pp in enumerate(pairs):
            if (f := family(pp)) not in seen:
                seen.add(f)
                fams.append(i)
        top = [i for i in fams if means[i] - means[0] <= FAMILY_GAP_S][:FAMILY_TOP]
        dropped = len(fams) - len(top)
        if best_of["PIT"] is not None and best_of["PIT"] not in top:   # pitting now is always on show
            top = sorted({*top, best_of["PIT"]})
    else:
        top = shown[:RANK_TOP]
        for k in ("PIT", "STAY"):   # both immediate actions on show when both are legal
            if best_of[k] is not None and best_of[k] not in top:
                top[-1] = best_of[k]
                top = sorted(set(top))
    sub = times[top]
    win = sub == sub.min(axis=0)
    share = (win / win.sum(axis=0)).mean(axis=1)
    general, per_plan = uncertainty_flags(snapshot, sim, [plans[i] for i in top], profile=profile)
    out = []
    for r, (i, sh_share, flags) in enumerate(zip(top, share, per_plan), start=1):
        same = [j for j, sh in enumerate(shapes) if sh == shapes[i] and means[j] <= means[i] + FLEX_S]
        stops = sorted({snapshot.laps_completed + int(pairs[j][0][1]) for j in same if len(pairs[j]) > 1})
        act = immediate_action(snapshot, pairs[i])
        entry = {"rank": r, **({"family": family(pairs[i])} if families else {}),
                 "plan": plan_text(pairs[i]), "stops": len(pairs[i]) - 1,
                 "first_stop": describe_plan(snapshot, pairs[i]), "immediate": act["action"],
                 "submit": act["submit"]}
        if families:   # the family and the submit fields say it; kept compact
            del entry["stops"], entry["first_stop"]
        entry.update(_gap(times[i] - times[0]) if i else {"gap_s": 0.0})
        entry["best_share"] = round(float(sh_share), 2)
        if len(stops) > 1:
            entry["first_stop_range"] = [stops[0], stops[-1]]
        if flags:
            entry["flags"] = flags
        warn = _report.support_warnings(fit.degradation, snapshot, plans[i])
        if warn:   # a stint older than any seen: a note on the plan, not an uncertainty flag
            entry["extrapolates"] = warn
        out.append(entry)
    # the immediate action is plan 1's (the lowest mean time, as the optimizer ranks); share is the
    # share of draws in which the best plan of that action beats the best plan of the other one, which
    # can be under one half when a few slow draws decide the mean
    a, b = best_of["PIT"], best_of["STAY"]
    favoured = immediate_action(snapshot, pairs[0])["action"]
    if a is not None and b is not None:
        d = times[a] - times[b]
        lo, hi = np.percentile(d, [5, 95])
        p = float((d < 0).mean() + 0.5 * (d == 0).mean())
        pit = immediate_action(snapshot, pairs[a])["action"]
        decision = {"favoured": favoured, "share": round(p if favoured.startswith("PIT") else 1 - p, 2),
                    "reading": gap_reading(f"{pit} (best plan)", "STAY_OUT (best plan)", p, float(d.mean()),
                                           float(lo), float(hi))}
    elif b is not None:
        decision = {"favoured": favoured, "share": 1.0, "reading": f"no PIT now is legal or modelled: {window['summary']}"}
    else:
        decision = {"favoured": favoured, "share": 1.0, "reading": "every legal plan pits now"}
    more = {"families_dropped": dropped} if families else {}
    from .optimizer_bridge import _stops_prior
    if _stops_prior(snapshot, fit) is not None:   # SimConfig.prerace.stops_prior_s: the order is not time alone
        more["ranking"] = ("plan 1 ranks on mean time plus a pre-race prior on the number of stops; gap_s is "
                           "time alone, so a plan below plan 1 can show a negative gap")
    return _jsonable({"plans": out, "decision": decision, "flags": general, "pit_window": window["summary"],
                      **more, "n_draws": int(times.shape[1]),
                      "note": "gaps: plan minus the best plan over shared draws (negative: faster); shares cover "
                              "parameter and lap noise under this model, not model error"})


def evaluate_plan(snapshot: Snapshot, plan=None, first_stop_lap=None, compound=None, *, n_samples=None,
                  sim=None, profile=None) -> dict:
    """One plan against the optimizer's best (rank_plans' plan 1) under the same draws: a full plan
    [[compound, laps], ...] whose first stint continues the current set, or a first stop at the end of
    lap first_stop_lap onto compound (every legal modelled compound when None, the fastest shown): the
    best full plan with that first stop, the plan a submission naming it is scored on
    (eval.decisions.kept_plan). Returns the plan, its gap to the best (_gap), the immediate action and
    submission that keep it (immediate_action), scored_plan when the scorer would play another plan for
    that submission (eval.decisions.to_plan on this belief: a full plan's later stints are the
    optimizer's again) and flags; rule replies say what is legal instead."""
    from ..baselines.domain import NoFeasiblePlan
    from ..eval import decisions
    sim, profile = _setup(snapshot, sim, profile)
    n = n_samples or profile.sim.n_samples
    fit = sim.fitted_for(snapshot)
    window = pit_window(snapshot, sim.config)
    if (plan is None) == (first_stop_lap is None):
        raise ValueError("give either plan or first_stop_lap (with compound), not both")
    try:
        best = _ranked(snapshot, fit, n, profile.sim.max_stops)
    except NoFeasiblePlan as exc:
        return {"error": f"No legal plan within the stop limit: {exc}", "recoverable": False}
    best_plan = best.plan
    if plan is not None:
        cands = [plan]
    else:
        lap = int(first_stop_lap)
        if not (window["earliest_stop_lap"] or 0) <= lap <= (window["latest_stop_lap"] or -1):
            return {"error": f"No legal stop at the end of lap {lap}: {window['summary']}.", "recoverable": True}
        pits = [c for c in legal_compounds(snapshot, sim.config) if c in fit.modelled]
        comps = [str(compound).upper()] if compound else pits
        bad = [c for c in comps if c not in pits]
        if bad:
            return {"error": f"Compound {bad[0]} cannot be fitted; choose from {pits}.", "recoverable": True}
        # the plan the scorer plays for a submission naming this stop (decisions.kept_plan)
        done = snapshot.laps_completed
        act = {"action": "pit"} if lap == done + 1 else {"action": "stay", "laps": lap - done - 1}
        cands = []
        for c in comps:
            try:
                cands.append(decisions.kept_plan(snapshot, {"first_stop_lap": lap, "compound": c}, act, sim,
                                                 profile.sim.max_stops, seed=0, n_samples=n))
            except decisions.InvalidDecision as exc:
                if len(comps) == 1:
                    return {"error": f"A stop at the end of lap {lap} onto {c}: {exc}", "recoverable": True}
        if not cands:
            return {"error": f"No legal plan stops at the end of lap {lap}: {window['summary']}.", "recoverable": True}
    res = fit.compare(snapshot, [*cands, best_plan], seed=SEED, n_samples=n)
    k = len(cands)
    ok = [i for i in range(k) if res.valid[i]]
    if not ok:
        errors = explain_rule_errors(snapshot, res.candidates[0], res.errors[0], sim.config)
        return _jsonable({"valid": False, "error": "; ".join(errors) or "not a legal plan", "recoverable": True,
                          "pit_window": window["summary"]})
    i = min(ok, key=lambda j: res.summaries[j].mean_s)
    pairs = res.candidates[i].as_pairs()
    times = fit.simulate_batch(snapshot, [res.candidates[i], best_plan], seed=SEED, n_samples=n).times
    act = immediate_action(snapshot, pairs)
    out = {"plan": plan_text(pairs), "stops": len(pairs) - 1, "first_stop": describe_plan(snapshot, pairs),
           "vs_best": {"best_plan": plan_text(best_plan.as_pairs()), **_gap(times[0] - times[1])},
           "immediate": act["action"], "submit": act["submit"]}
    if k > 1:
        out["other_compounds"] = {res.candidates[j].stints[1].compound: round(float(res.gap_mean_s[j][i]), 1)
                                  for j in ok if j != i}
    try:
        scored = decisions.to_plan(snapshot, {**act["submit"], "confidence": 0.5}, sim, profile.sim.max_stops,
                                   seed=0, n_samples=profile.sim.n_samples)
        if scored.as_pairs() != pairs:
            out["scored_plan"] = (f"the scorer plays {plan_text(scored.as_pairs())} for this submission (after a "
                                  f"STAY_OUT the rest is planned again)")
    except decisions.InvalidDecision as exc:
        out["scored_plan"] = f"this submission has no legal plan: {exc}"
    general, per_plan = uncertainty_flags(snapshot, sim, [res.candidates[i]], profile=profile)
    if (flags := [*general, *per_plan[0]]):
        out["flags"] = flags
    if res.support_warnings[i]:
        out["extrapolates"] = res.support_warnings[i]
    return _jsonable(out)


# --- protocol v2free: the race as a strategist sees it at the snapshot -----------------------------------
def _wear_slope(rows, fuel: float):
    """(observed wear s/lap, clean laps): the slope of fuel-corrected lap time on tyre age within each
    (driver, stint) of `rows`' clean laps, pooled; None with fewer than 4 laps or no age spread."""
    if rows is None or not len(rows) or "clean" not in rows:
        return None, 0
    r = rows[rows["clean"].fillna(False).astype(bool) & rows["tyre_age"].notna() & rows["lap_time_s"].notna()]
    if len(r) < 4:
        return None, int(len(r))
    keys = [r["driver"], r["stint"].fillna(-1)]
    y = r["lap_time_s"].astype(float) - fuel * r["lap"].astype(float)
    a = r["tyre_age"].astype(float)
    yd, ad = y - y.groupby(keys).transform("mean"), a - a.groupby(keys).transform("mean")
    den = float((ad * ad).sum())
    return (None if den <= 0 else round(float((yd * ad).sum()) / den, 3)), int(len(r))


def race_picture(snapshot: Snapshot, *, sim=None, profile=None) -> dict:
    """What a strategist weighs at the snapshot beyond the look-ahead's ranking, compactly, from the
    snapshot and the race laps visible at it (Simulator.window_rows: cut at the snapshot's moment,
    data.visible_rows) and the belief fit at it; nothing after the snapshot enters. Driver codes are
    the snapshot's (tools.present labels them):

      car         position, gaps, laps left, track status and how long a neutralisation has lasted,
                  the pit window, the stint history, the current set and its last lap times, and the
                  set's observed wear (fuel-corrected lap time on tyre age) against the model's
      nearby      the cars around: position, gap, compound, tyre age and stops so far
      rejoin      cars within the pit loss behind at the end of the last lap (a stop now drops behind
                  them) and the position that implies, from the gaps alone
      compounds   per compound: race laps behind the fit, the longest run seen in the race and the
                  field's observed wear against the model's (value and sd), the base pace sd, where pace
                  and wear come from when proxied, and the tyre-life estimate with its sd and source"""
    sim, profile = _setup(snapshot, sim, profile)
    fit = sim.fitted_for(snapshot)
    done = snapshot.laps_completed
    rows = sim.window_rows(snapshot, done) if done > 0 else None
    if rows is not None and len(rows):
        rows = rows[rows["lap"] <= done]
    deg = fit.degradation
    fuel = float(getattr(deg, "fuel_slope", 0.0) or 0.0)
    rep = sim.model_report(snapshot)
    fit_report = getattr(deg, "fit_report", None) or {}
    proxy = fit_report.get("proxy") or {}
    cliffs = sim.cliffs_for(snapshot)
    loss = pit_loss(snapshot, sim=sim, profile=profile)
    window = pit_window(snapshot, sim.config)

    def model_wear(c):
        r = rep.compounds.get(c)
        if r is None or not r.modelled:
            return None
        sd = r.wear_s_per_lap.sd
        return {"value": round(float(r.wear_s_per_lap.value), 3), "sd": None if _nan(sd) else round(float(sd), 3)}

    def r1(x):
        return None if x is None or _nan(x) else round(float(x), 1)
    car = {"position": snapshot.position, "gap_ahead_s": r1(snapshot.gap_ahead_s), "gap_behind_s": r1(snapshot.gap_behind_s),
           "laps_left": snapshot.remaining_laps, "track_status": snapshot.track_status.upper(),
           "neutralised_for_laps": _neutralised_for(snapshot), "pit_window": window["summary"],
           "stints_so_far": [f"{s.compound} {s.laps}" for s in snapshot.stints],
           "current_set": f"{snapshot.compound}, age {snapshot.tyre_age}",
           "last_laps_s": [round(float(t), 2) for t in snapshot.recent_lap_times_s[-5:]]}
    own = None
    if rows is not None and len(rows):
        mine = rows[rows["driver"] == snapshot.driver]
        if len(mine):
            own = mine[mine["stint"] == mine["stint"].max()]
    slope, n = _wear_slope(own, fuel)
    car["set_wear"] = {"observed_s_per_lap": slope, "clean_laps": n, "model": model_wear(snapshot.compound)}

    nearby = []
    for c in snapshot.field.nearby:
        entry = {"car": c.driver, "position": c.position, "gap_s": None if c.gap_s is None else round(float(c.gap_s), 1)}
        if rows is not None and len(rows):
            theirs = rows[rows["driver"] == c.driver]
            if len(theirs):
                last = theirs.iloc[-1]
                entry.update(compound=last["compound"], tyre_age=None if _nan(last["tyre_age"]) else int(last["tyre_age"]),
                             stops=int(theirs["stint"].nunique()) - 1)
        nearby.append(entry)

    rejoin = None
    now = float(loss["pit_loss_now_s"]["median"]) if loss.get("pit_loss_now_s") else None
    if rows is not None and len(rows) and now is not None and "end_s" in rows:
        at = rows[rows["lap"] == done]
        ours = at.loc[at["driver"] == snapshot.driver, "end_s"]
        if len(ours) and snapshot.position is not None:
            behind = (pd_numeric(at["end_s"]) - float(ours.iloc[0]))
            passing = int(((behind > 0) & (behind < now)).sum())
            rejoin = {"pit_loss_now_s": round(now, 1), "cars_within_it_behind": passing,
                      "rejoin_position_about": int(snapshot.position) + passing}

    compounds = {}
    for c, r in rep.compounds.items():
        if not r.modelled:
            continue
        on = rows[rows["compound"] == c] if rows is not None and len(rows) else None
        slope, n = _wear_slope(on, fuel)
        entry = {"race_laps": int(r.n_race_rows or 0),
                 "longest_run_seen": None if on is None or not len(on) else int(on["tyre_age"].max()),
                 "field_wear_observed": slope, "field_clean_laps": n, "model_wear": model_wear(c),
                 "base_pace_sd_s": None if _nan(r.base_s.sd) else round(float(r.base_s.sd), 3)}
        tiers = proxy.get(c)
        if tiers is not None and (tiers.get("pace_source") != "race" or tiers.get("wear_source") != "race"):
            entry["priced_by"] = f"pace {tiers.get('pace_source')}, wear {tiers.get('wear_source')}"
        life = cliffs.get(c)
        if life is not None:
            t = _tyre_life(life)
            entry["tyre_life"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in
                                  {"seen": t["longest_run_seen_laps"], "prior": t["prior_life_laps"],
                                   "prior_sd": t["prior_sd_laps"], "source": t["prior_source"]}.items()
                                  if v is not None}
        compounds[c] = entry
    out = {"car": car, "nearby": nearby, "rejoin": rejoin, "compounds": compounds,
           "note": "observed wear: s/lap of tyre age from clean laps so far, fuel-corrected; compare with "
                   "model_wear (the simulator's value and sd); race_laps are the laps behind the fit"}
    if prerace_shown():
        block = prerace_block(snapshot.race.race_id)
        if block is not None:
            out["pre_race"] = block
    return _jsonable(out)


def prerace_shown() -> bool:
    """AGENT_PRERACE "on": race_picture carries the supplier's pre-race block (prerace.prerace_block:
    the circuit's tyre ratings in words and the expected strategies, published before the race,
    nothing naming the race). Off (the default): race_picture as before."""
    import os
    mode = os.environ.get("AGENT_PRERACE", "").strip().lower() or "off"
    if mode not in ("on", "off"):
        raise ValueError(f"AGENT_PRERACE must be on or off; got {mode!r}")
    return mode == "on"


def pd_numeric(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def recommendation_fields(snapshot: Snapshot, decision: dict) -> dict:
    """The decision (PIT or STAY_OUT, None when malformed) and p_pit, the stated probability
    that pitting now is right, for the Brier column (eval.decisions.p_pit)."""
    from ..eval import decisions
    try:
        kind = decisions.kind(decision)
    except decisions.InvalidDecision:
        kind = None
    return _jsonable({"decision": kind, "p_pit": decisions.p_pit(decision),
                      "laps_completed": snapshot.laps_completed})


# --- the supplier's pre-race block (src/sim/prerace.py tables; AGENT_PRERACE) ----------------------------------------------------------------------------
_LEVEL = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}
_WORDS = {"asphalt_abrasion": "abrasion", "tyre_stress": "tyre stress", "lateral": "lateral load",
          "traction": "traction demand", "braking": "braking", "asphalt_grip": "asphalt grip",
          "track_evolution": "track evolution", "downforce": "downforce"}


def prerace_ratings_words(race_id) -> str | None:
    """'abrasion low, tyre stress high, ...' from the ratings, None without them."""
    from .prerace import load_ratings
    t = load_ratings()
    from .prerace import _rid
    rid = _rid(race_id)
    if rid is None or rid not in t.index:
        return None
    r = t.loc[rid]
    parts = [f"{_WORDS[c]} {_LEVEL[int(r[c])]}" for c in
             ("asphalt_abrasion", "tyre_stress", "lateral", "traction", "braking", "asphalt_grip",
              "track_evolution", "downforce")
             if c in r and r[c] == r[c] and int(r[c]) in _LEVEL]
    return ", ".join(parts) if parts else None


def prerace_guidance_words(race_id) -> list[str]:
    """One plain line per expected strategy, e.g. 'expected fastest: one stop, MEDIUM about laps
    1-24 then HARD'; [] without guidance."""
    out = []
    names = {0: "no stop", 1: "one stop", 2: "two stops", 3: "three stops"}
    from .prerace import strategies_of
    for s in strategies_of(race_id):
        label = "expected fastest" if s["rank"] == 1 else f"expected option {s['rank']}"
        seq = s["sequence"]
        stops = names.get(s["stops"], f"{s['stops']} stops")
        if not seq:
            out.append(f"{label}: {stops} (compounds not stated)")
            continue
        if s["windows"]:
            stints = []
            for j, role in enumerate(seq):
                if j < len(s["windows"]):
                    lo, hi = s["windows"][j]
                    stints.append(f"{role} to a stop around lap {lo:g}" + (f"-{hi:g}" if hi != lo else ""))
                else:
                    stints.append(f"{role} to the flag")
            body = ", then ".join(stints)
        else:
            body = " then ".join(seq)
        out.append(f"{label}: {stops}, {body}")
    return out


def prerace_block(race_id) -> dict | None:
    """The agent's compact pre-race block: the circuit's tyre ratings in words and the supplier's
    expected strategies, with its stated pace gaps and degradation note. Nothing that names the
    race, circuit, season or a driver. None when neither table has the race."""
    from .prerace import _rid
    rid = _rid(race_id)
    words = prerace_ratings_words(rid)
    lines = prerace_guidance_words(rid)
    from .prerace import load_guidance
    t = load_guidance()
    extra = {}
    if rid is not None and rid in t.index:
        r = t.loc[rid]
        for col, name in (("pace_gap_soft_medium_s", "soft_vs_medium_s_per_lap"),
                          ("pace_gap_medium_hard_s", "medium_vs_hard_s_per_lap")):
            try:
                v = float(r.get(col, ""))
                if v == v:
                    extra[name] = v
            except (TypeError, ValueError):
                pass
    if not words and not lines and not extra:
        return None
    out = {"source": "the tyre supplier's pre-race publications (before the start; may be wrong)"}
    if words:
        out["circuit_tyre_ratings"] = words
    if lines:
        out["expected_strategies"] = lines
    if extra:
        out["stated_pace_gaps"] = extra
    return out
