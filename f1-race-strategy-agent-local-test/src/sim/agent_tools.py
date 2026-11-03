"""The simulator behind the agent's tools (SPEC 0015).

Every function takes a contracts.Snapshot and returns a JSON-ready dict: NaN and infinite
values come back as None, and every number carries an interval, a source or a flag. They
run on the belief simulator, configs/sim/belief.yaml as it stands (to_snapshot window,
traffic off, SC and VSC periods from the rate model, the config's calibrator, posterior
parameter draws, residual lap noise, the wear lower bound). The tools never build their
own config.

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

import copy
import math
from pathlib import Path

import numpy as np

from ..contracts import Plan, Snapshot
from . import calibrators, jobs
from . import model_report as _report
from .compute import load_profile
from .config import SimConfig
from .engine import DEGRADATION_RETRY_STREAM, DEGRADATION_STREAM, Simulator, batch_normals
from .optimizer_bridge import as_continuation
from .race_ids import RACE_ID_ALIASES, canonical_race_id, parse_race_id, slug

BELIEF_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sim" / "belief.yaml"
# added to the seed of every belief-side simulation, so the belief's samples are never the
# world's (a prime, far from any seed a run uses)
BELIEF_SEED_OFFSET = 1_000_003
SEED = 0 + BELIEF_SEED_OFFSET
Z90 = 1.64              # a 90 percent normal band, mean -+ Z90 sd
PIT_LOSS_BAND = 0.10    # the band around a pit loss with no calibrated sd, source "config"
PROBABILITY_COVERS = "parameter and lap-noise uncertainty under this model; does not include model error"
# event_rates source: the EventModel defaults, or the race's own (circuit) rates
EVENT_RATES_PROVISIONAL = "provisional: SC from 2021 Spain, VSC placeholder"
EVENT_RATES_CIRCUIT = "circuit"
TRAFFIC_NOTE = "traffic is not modelled belief-side: the gaps and probabilities ignore cars on track"
CALIBRATION_NOTE = ("mean absolute error of the remaining race time for this fit window and horizon; "
                    "a conservative tolerance for a gap between plans, since the fuel error cancels between them")

_SIMS: dict = {}
_RACES: dict = {}
_BELIEF: dict = {}


# --- the belief simulator ---------------------------------------------------------------
def belief_config() -> SimConfig:
    """configs/sim/belief.yaml, read once."""
    if "config" not in _BELIEF:
        _BELIEF["config"] = SimConfig.from_file(BELIEF_CONFIG)
    return _BELIEF["config"]


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
    key = (race_id, profile.hash, config.model_dump_json())
    if key not in _SIMS:
        data = _race_data(race_id, profile)
        _SIMS[key] = Simulator(data.frame, config, data.practice)
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
    race = _race_data(race_id, profile or load_profile()).race
    update = {"total_laps": view.total_laps}
    if view.min_stint_laps is not None:
        update["min_stint_laps"] = view.min_stint_laps
    return view.to_contract(race=race.model_copy(update=update))


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
    EventModel defaults. An events model other than "rate" draws no periods: the rates are
    None and source names the model."""
    from .events import RateEvents

    spec = (config or belief_config()).events
    if spec.name != "rate":
        return {"p_sc": None, "d_sc": None, "p_vsc": None, "d_vsc": None, "source": f"events model {spec.name!r}"}
    race = snapshot.race
    p_sc, d_sc, p_vsc, d_vsc = RateEvents(**spec.params).rates(race)
    circuit = any(v is not None for v in (race.sc_rate_per_lap, race.sc_mean_laps, race.vsc_rate_per_lap,
                                          race.vsc_mean_laps))
    return {"p_sc": p_sc, "d_sc": d_sc, "p_vsc": p_vsc, "d_vsc": d_vsc,
            "source": EVENT_RATES_CIRCUIT if circuit else EVENT_RATES_PROVISIONAL}


def calibration_error_s(sim: Simulator, snapshot: Snapshot) -> float:
    """The tools' tolerance for a gap between plans: the offset-adjusted mean absolute error
    of the remaining race time for the belief's fit window at the snapshot's horizon bucket,
    from reports/calibration/midrace_summary.csv (ModelReport.calibration_error_s). It is a
    conservative tolerance for a gap, since the fuel error cancels between plans; SPEC 0016
    adds a gap-specific one. NaN without a row."""
    return _report.calibration_error(sim.config.fit_window, snapshot.remaining_laps)


# --- the functions --------------------------------------------------------------------------
def race_state(snapshot: Snapshot, *, config: SimConfig | None = None) -> dict:
    """The snapshot dump plus legal_compounds, remaining_laps and event_rates."""
    return _jsonable({**snapshot.model_dump(mode="json"), "legal_compounds": legal_compounds(snapshot, config),
                      "remaining_laps": snapshot.remaining_laps, "event_rates": event_rates(snapshot, config)})


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

    and fuel_s_per_lap_interval (-+ 1.64 sd, its fuel_interval_basis and
    fuel_point_estimate) for models with a
    fuel term, a no_sd:fuel_s_per_lap_interval flag when its sd is NaN. flags and
    calibration_error_s are the report's; calibration_error_s is a conservative tolerance
    for gaps between plans (calibration_error_s()). This is estimate_tyre_degradation's body."""
    sim, profile = _setup(snapshot, sim, profile)
    rep = sim.model_report(snapshot, ages)
    deg = sim.fitted_for(snapshot).degradation
    fit_report = getattr(deg, "fit_report", None) or {}
    draws = _wear_draws(deg, n_samples or profile.sim.n_samples)
    bounded = fit_report.get("bounded") or {}
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
    if rep.fuel_s_per_lap is not None:
        iv, basis = _interval(rep.fuel_s_per_lap.value, rep.fuel_s_per_lap.sd)
        out["fuel_s_per_lap_interval"], out["fuel_interval_basis"] = iv, basis
        out["fuel_point_estimate"] = rep.fuel_s_per_lap.value
        if basis == BASIS_NO_SD:
            out["flags"] = [*out["flags"], "no_sd:fuel_s_per_lap_interval"]
    out["calibration_error_note"] = CALIBRATION_NOTE
    out["event_rates"] = event_rates(snapshot, sim.config)
    return _jsonable(out)


def pit_loss(snapshot: Snapshot, *, sim=None, profile=None) -> dict:
    """pit_loss_s {low, median, high}: race.pit_loss_s, or the calibrator's value, with the
    calibrator's sd (calibrators.effective_factors) as -+ 1.64 sd, source "calibrator";
    when the calibrator gives no sd, the value with a fixed 10 percent band, source
    "config". pit_loss_now_s is that interval times pit_loss_now_factor, the factor for the
    current track status: sc_pit_loss_factor under SC, vsc_pit_loss_factor under VSC, 1 when
    green. The SC and VSC factors are config.effects'. estimate_pit_loss's body."""
    sim, _ = _setup(snapshot, sim, profile)
    fit = sim.fitted_for(snapshot)
    mean, sd = float(snapshot.race.pit_loss_s), 0.0
    if fit.calibration is not None:
        mean, sd = calibrators.effective_factors(fit, snapshot.race)["pit_loss_s"]
    if not _nan(sd) and sd > 0:
        band, source = Z90 * sd, "calibrator"
    else:
        band, source, sd = PIT_LOSS_BAND * mean, "config", None
    fx = sim.config.effects
    green = {"low": mean - band, "median": mean, "high": mean + band}
    factor = {"sc": fx.sc_pit_loss_factor, "vsc": fx.vsc_pit_loss_factor}.get(snapshot.track_status, 1.0)
    return _jsonable({
        "pit_loss_s": green, "source": source, "sd_s": sd,
        "pit_loss_now_s": {k: v * factor for k, v in green.items()}, "pit_loss_now_factor": factor,
        "sc_pit_loss_factor": fx.sc_pit_loss_factor, "vsc_pit_loss_factor": fx.vsc_pit_loss_factor,
        "factor_source": "config", "track_status": snapshot.track_status})


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
    return _jsonable(res.model_dump())


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
      within_sim_error             |gap_mean| < max(2 gap_std, calibration_error_s)
      confidence_basis             "model error" when calibration_error_s exceeds 2 gap_std,
                                   else "sampling spread"
      calibration_error_s          a conservative tolerance for gaps (calibration_error_s())
      dominated, support_warnings  the action's, from the compare
      alternative                  its action, plan and remaining-time summary

    No traffic_probability: traffic is not modelled belief-side (`traffic`). The numbers
    are None when no alternative is legal; an illegal action returns {"error": ...}."""
    sim, profile = _setup(snapshot, sim, profile)
    n = _compare_n(profile, n_samples)
    fit = sim.fitted_for(snapshot)
    act = _action(action)
    if act.get("action") == "pit":
        alts = [{"action": "stay", "laps": 1}]
    else:
        alts = [{"action": "pit", "compound": c} for c in legal_compounds(snapshot, sim.config) if c in fit.modelled]
    res = fit.compare(snapshot, [_candidate(snapshot, a) for a in [act, *alts]], seed=SEED, n_samples=n,
                      continuation=_continuation(fit, n, profile.sim.max_stops))
    if not res.valid[0]:
        return _jsonable({"action": act, "valid": False, "errors": res.errors[0], "error": "; ".join(res.errors[0])})
    ok = [i for i in range(1, len(res.candidates)) if res.valid[i]]
    j = (max(ok, key=lambda i: (res.p_best[i], -res.summaries[i].mean_s, -i)) if ok
         else 1 if len(res.candidates) > 1 else None)
    out = {"action": act, "plan": res.candidates[0].as_pairs(), "remaining": res.summaries[0].model_dump(),
           "relative_race_time_delta_s": {"optimistic": None, "median": None, "pessimistic": None},
           "gap_mean_s": None, "gap_std_s": None, "probability_improves_outcome": None,
           "probability_covers": PROBABILITY_COVERS, "within_sim_error": None, "confidence_basis": None,
           "calibration_error_s": calibration_error_s(sim, snapshot), "calibration_error_note": CALIBRATION_NOTE,
           "dominated": res.dominated[0], "support_warnings": res.support_warnings[0],
           "traffic": TRAFFIC_NOTE, "alternative": None, "n_samples": res.n_samples, "seed": res.seed,
           "assumptions": [f"SC and VSC rates: {event_rates(snapshot, sim.config)['source']}"]}
    if j is None:
        return _jsonable(out)
    alt = alts[j - 1]
    if not res.valid[j]:
        out["alternative"] = {"action": alt, "valid": False, "errors": res.errors[j]}
        return _jsonable(out)
    out["alternative"] = {"action": alt, "valid": True, "plan": res.candidates[j].as_pairs(),
                          "remaining": res.summaries[j].model_dump()}
    # the gap's percentiles: the two expanded plans again under the same draws (R6)
    times = fit.simulate_batch(snapshot, [res.candidates[0], res.candidates[j]], seed=SEED, n_samples=n).times
    lo, med, hi = np.percentile(times[0] - times[1], [5, 50, 95])
    gap, std, cal = res.gap_mean_s[0][j], res.gap_std_s[0][j], out["calibration_error_s"]
    model_error = not _nan(cal) and cal > 2 * std
    out.update(relative_race_time_delta_s={"optimistic": float(lo), "median": float(med), "pessimistic": float(hi)},
               gap_mean_s=gap, gap_std_s=std, probability_improves_outcome=res.p_beats[0][j],
               within_sim_error=bool(abs(gap) < (cal if model_error else 2 * std)),
               confidence_basis="model error" if model_error else "sampling spread")
    return _jsonable(out)


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
