"""Regret scoring: each policy's plan against the optimizer's, from the same snapshot.

Two simulators per race (docs/eval.md "Who sees what"): policies plan on the belief
simulator, fitted only on what is known at the snapshot; the reference optimizer and every
scoring call use the world simulator. For every scenario the optimizer
(sim.optimizer_bridge.optimize_batch) finds the world's best plan with at most `max_stops`
future stops, and every policy's plan is scored by the world under the same draws
(docs/rules.md R6: seed s, samples 0..n-1). Then

    regret_s = mean remaining time of the policy's plan - mean remaining time of the optimizer's

A plan with more stops than the optimizer may search (an actual three-stop, say) can come in
below the reference, so regret can be negative; it is reported as it is. One row per
(scenario, policy); `summarize` turns rows into the per-race and overall tables.

The reference comes from eval.reference (cached, or computed the same way). Its per-draw
times give the Monte Carlo precision of each regret under the shared draws:

    regret_se_s = sd(policy_times - reference_times) / sqrt(N)
    within_mc   = |regret_s| <= 2 * regret_se_s

At N = 1 regret_se_s is 0, so within_mc is the tie rule (regret_s 0). summarize reports
share_within_mc, the share of scored rows the draws cannot tell from the reference.

Near ties (information only, eval.reference.near_ties): every row carries the size of the
optimal set (the searched plans within 2 SE of the reference), the best pit-now and
stay-out means, their gap, its paired SE and whether the pit-now call is decisive; ok rows
add stop_agree_set and first_pit_error_set against the closest member of the optimal set.
summarize does not read them; summarize_near_ties is the separate view `report` prints.

A policy may return an agent decision (Decision.submission, PIT or STAY_OUT) instead of a
plan. eval.decisions.to_plan expands it on the belief into the action plus an optimizer
continuation, and the world scores that plan. Such rows also carry `decision` and `p_pit`
(the stated probability that pitting now is right) against `pit_now_opt` (the world
optimizer's first stop is at the end of this lap), for the Brier score. A decision's plan
stays inside the reference's search space, so an agent row with negative regret is a bug
and `summarize` warns about it (AgentRegretWarning).

Position regret (world traffic on, configs/sim/world.yaml): with ghost traffic the world
places us among the recorded field, so each sample has a finishing position,
1 + the ghosts that finished the last lap before the plan's world time (traffic.place); a
DNF sample is behind every finisher, ordered by dnf_lap against the ghosts that retired.
final_position_mean is the plan's mean over the samples, opt_final_position_mean the
optimizer's, and

    position_regret = final_position_mean - opt_final_position_mean

(positive: the policy finishes behind the reference on average). All three are NaN when the
world runs without traffic (the default world config) or traffic cannot be placed (no
elapsed_s), since without ghosts there is no field to finish among.

With the world's compound proxy on (SimConfig.compound_proxy, sim/proxy.py, docs/eval.md
"Proxied compounds") every row carries opt_proxied_compounds and every ok row
proxied_compounds: the reference's and the plan's compounds that the world prices by proxy
(no race laps behind them), sorted and joined by ";", "" for none. With it off neither
column exists.

Row status: ok, invalid (the plan broke a rule or no plan was submitted), unavailable (the
policy does not apply to this scenario), error (the policy itself failed, an API error say),
no_reference (no legal plan exists for the optimizer).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..baselines.domain import NoFeasiblePlan, PolicyUnavailable
from ..sim import calibration, proxy, traffic
from ..sim.optimizer_bridge import optimize_batch
from . import decisions
from .policies import Decision
from .reference import REFERENCE_DIR, Reference, ReferenceStore, compute_reference, near_ties, paired_se, \
    set_agreement

STATUSES = ("ok", "invalid", "unavailable", "error", "no_reference")
REGRET_TOL_S = 1e-6
REF_AGREE_LAPS = 2   # E: first pit laps this close count as the same reference call
ROLES = ("belief", "world")
BELIEF_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sim" / "belief.yaml"
# the traffic switch (docs/eval.md "Traffic"): opt-in world and belief configs with traffic on
WORLD_TRAFFIC_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sim" / "world.yaml"
BELIEF_TRAFFIC_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sim" / "belief_traffic.yaml"


def _first_pit_error(pits, ref_pits):
    """|first pit lap - optimizer's first pit lap|; 0 when neither stops, None when only one does."""
    if not pits and not ref_pits:
        return 0
    if not pits or not ref_pits:
        return None
    return abs(pits[0] - ref_pits[0])


def finishing_positions(frame: pd.DataFrame, snapshot, times, dnf_lap, at_flag=None) -> np.ndarray:
    """(n_samples,) finishing position per sample of one plan. The other drivers in `frame` are
    classified at the flag (traffic.flag_classification): a car whose last crossing is on the
    leader's last lap or later took the flag, a lapped one a lap or more short; one that stopped
    crossing earlier retired. A finisher is ahead when it crossed its own final lap's line
    before we did: our crossing of that lap is our session time (anchor + remaining time) less
    the laps we still had to run, at our mean remaining lap. `at_flag` (the engine's
    BatchResult.final_pos row) gives the same count on the traffic model's own lap-by-lap clock
    (the driver's own pace and the neutral-lap alignment, traffic.py) and replaces it. The last
    lap is snapshot.race.total_laps (the scheduled distance under snapshot_as_of), and a lapped
    finisher is no longer taken for a retirement. A DNF sample (finite dnf_lap) is behind every
    finisher and behind the cars that retired no earlier than its dnf_lap; its penalised time is
    not used."""
    k, last = snapshot.laps_completed, snapshot.race.total_laps
    laps = np.arange(k + 1, last + 1)
    table = traffic.ghost_table(frame, snapshot.driver, laps)
    G = table.G
    fin = traffic.flag_classification(G, None)
    classified = fin >= 0
    if at_flag is not None:
        pos = np.asarray(at_flag, dtype=float).astype(int)
    else:
        rem = np.asarray(times, dtype=float)
        t = traffic.anchor(frame, snapshot) + rem
        per_lap = rem / max(len(laps), 1)
        idx = np.flatnonzero(classified)
        g_end = G[idx, fin[idx]]                            # each finisher's final crossing
        short = (len(laps) - 1 - fin[idx]).astype(float)    # laps it finished short of our last
        ours = t[:, None] - short[None, :] * per_lap[:, None]
        pos = 1 + (g_end[None, :] < ours).sum(axis=1)
    dnf_lap = np.asarray(dnf_lap, dtype=float)
    dnf = np.isfinite(dnf_lap)
    if dnf.any():
        ghosts = frame[(frame["driver"] != snapshot.driver) & np.isfinite(frame["end_s"].astype(float))]
        reached = ghosts.groupby("driver")["lap"].max()
        out = [d for d, c in zip(table.ghosts, classified) if not c]
        retired = reached.reindex(out).dropna().to_numpy(dtype=float)   # lap count of each non-finisher
        pos[dnf] = 1 + int(classified.sum()) + (retired[None, :] >= dnf_lap[dnf][:, None]).sum(axis=1)
    return pos


def _positions(world, snap, batch, i):
    """finishing_positions of plan i of a world batch, None when traffic was not placed."""
    if not batch.traffic_on:
        return None
    at_flag = batch.final_pos[i] if batch.final_pos is not None else None
    return finishing_positions(world.laps, snap, batch.times[i], batch.dnf_lap[i], at_flag=at_flag)


def references_agree(plan, other, snapshot, within: int = REF_AGREE_LAPS) -> bool:
    """E: two reference plans agree when they make the same number of stops and their first
    pit laps are at most `within` laps apart (two no-stop plans agree)."""
    a, b = plan.pit_laps(snapshot), other.pit_laps(snapshot)
    return len(a) == len(b) and (not a or abs(a[0] - b[0]) <= within)


def proxied_compounds(world, snapshot) -> dict | None:
    """{compound: {"pace", "wear"} sources} the world prices by proxy at this snapshot
    (sim.proxy.proxied), None when the world's compound_proxy is off (the rows then carry no
    proxy columns, as before proxying)."""
    px = getattr(getattr(world, "config", None), "compound_proxy", None)
    if px is None or not px.enabled:
        return None
    fit = world.fitted_for(snapshot) if hasattr(world, "fitted_for") else world
    return proxy.proxied(getattr(getattr(fit, "degradation", None), "fit_report", None))


def _on_proxy(plan, borrowed) -> str:
    """The plan's compounds that the world prices by proxy, sorted and joined by ";" ("" for none)."""
    return ";".join(sorted({s.compound for s in plan.stints} & set(borrowed)))


class HindsightPitLossWarning(UserWarning):
    """Policies are handed a frozen scenario's whole-race pit loss (a belief without the as-of one)."""


def policy_scenario(scenario, belief):
    """The scenario a policy is handed: its snapshot's race.pit_loss_s replaced by the pit loss
    the belief prices stops with at that snapshot (Simulator.pit_loss_inputs: the as-of
    estimate under SimConfig.pit_loss.source "asof"). A frozen scenario set carries the
    whole-race value, which no policy may read (docs/eval.md "Pit loss and track status").
    Unchanged when the belief reads the snapshot's own value."""
    if getattr(scenario, "stratum", None) is not None or getattr(scenario, "weight", None) is not None:
        # a stratified set's stratum and weight are scorer-side (scenarios.sample_stratified)
        scenario = scenario.model_copy(update={"stratum": None, "weight": None})
    if not hasattr(belief, "pit_loss_inputs"):
        return scenario
    snap = scenario.snapshot
    pl = belief.fitted_for(snap).pit_loss_inputs(snap) if hasattr(belief, "fitted_for") else belief.pit_loss_inputs(snap)
    if pl.get("source") == "snapshot":
        warnings.warn("the belief reads the snapshot's pit loss (SimConfig.pit_loss.source 'snapshot'): a frozen "
                      "scenario carries the whole-race value, so policies see hindsight", HindsightPitLossWarning,
                      stacklevel=2)
        return scenario
    race = snap.race.model_copy(update={"pit_loss_s": round(float(pl["pit_loss_s"]), 3)})
    return scenario.model_copy(update={"snapshot": snap.model_copy(update={"race": race})})


def score_scenario(scenario, world, policies: dict, *, belief=None, seed: int = 0,
                   n_samples: int | None = None, max_stops: int = 2, ref_worlds=(),
                   reference: Reference | None = None) -> list[dict]:
    """One row per policy for one scenario. Policies plan on `belief` (`world` when None, one
    simulator in both roles); the reference and the scoring use `world`. `reference` is a
    cached eval.reference.Reference for these inputs; None computes it (compute_reference),
    the same record either way. With `ref_worlds` (E, optional) each of those world models
    plans its own reference too, and the rows gain ref_models_agree: every one of them agrees
    with the world's reference (references_agree). They score nothing."""
    belief = world if belief is None else belief
    snap = scenario.snapshot
    base = {"scenario_id": scenario.scenario_id, "race_id": scenario.race_id, "driver": scenario.driver,
            "lap": scenario.lap, "compound": snap.compound, "tyre_age": snap.tyre_age,
            "remaining_laps": snap.remaining_laps, "seed": seed, "max_stops": max_stops}
    if getattr(scenario, "stratum", None) is not None:   # a stratified set: for weighted summaries
        base.update(stratum=scenario.stratum, weight=scenario.weight)
    opt = reference if reference is not None else compute_reference(scenario, world, seed=seed,
                                                                    n_samples=n_samples, max_stops=max_stops)
    if opt.status == "no_reference":
        return [{**base, "policy": name, "status": "no_reference", "reason": opt.reason} for name in policies]
    ref = opt.mean_s
    ref_pits = opt.plan.pit_laps(snap)
    n = len(opt.times)
    if ref_worlds:
        agree = True
        for other in ref_worlds:
            try:
                alt = optimize_batch(snap, other, seed=seed, n_samples=n, max_stops=max_stops)
            except NoFeasiblePlan:
                agree = False
                continue
            agree = agree and references_agree(opt.plan, alt.plan, snap)
        base["ref_models_agree"] = agree
    ref_pos = None
    if world.config.traffic.name != "none":   # D: the reference's positions, same draws
        ref_pos = _positions(world, snap, world.simulate_batch(snap, [opt.plan], seed=seed, n_samples=n), 0)
    # SimConfig.compound_proxy: the compounds the world prices by proxy (no race laps behind them)
    borrowed = proxied_compounds(world, snap)
    if borrowed is not None:
        base["opt_proxied_compounds"] = _on_proxy(opt.plan, borrowed)
    base.update(n_samples=n, opt_plan=json.dumps(opt.plan.as_pairs()), opt_num_stops=opt.plan.num_stops,
                opt_first_pit_lap=ref_pits[0] if ref_pits else None, opt_time_s=ref,
                pit_now_opt=opt.pit_now_opt,
                opt_final_position_mean=float(ref_pos.mean()) if ref_pos is not None else np.nan,
                **near_ties(opt, snap))
    rows = []
    # what the policies see: the scenario with the belief's as-of pit loss in its snapshot
    # (policy_scenario); the world and the reference keep the frozen scenario
    seen = policy_scenario(scenario, belief)
    psnap = seen.snapshot
    for name, policy in policies.items():
        row = {**base, "policy": name}
        started = perf_counter()
        try:
            decision = policy(seen, belief, max_stops)
        except PolicyUnavailable as exc:
            rows.append({**row, "status": "unavailable", "reason": str(exc)})
            continue
        if not isinstance(decision, Decision):
            decision = Decision(decision)
        row.update(latency_s=perf_counter() - started, tool_calls=decision.tool_calls,
                   evidence_calls=decision.evidence_calls, input_tokens=decision.input_tokens, output_tokens=decision.output_tokens,
                   submitted_in_grace=decision.submitted_in_grace, repeat_calls=decision.repeat_calls)
        if decision.error is not None:
            rows.append({**row, "status": "error", "reason": decision.error})
            continue
        plan = decision.plan
        if plan is None and decision.submission is not None:
            sub = decision.submission
            row.update(decision=str(sub.get("decision", "")).strip().upper() or None,
                       p_pit=decisions.p_pit(sub))
            try:
                plan = decisions.to_plan(psnap, sub, belief, max_stops, seed=seed, n_samples=n)
            except decisions.InvalidDecision as exc:
                rows.append({**row, "status": "invalid", "reason": str(exc)})
                continue
        if plan is None and decision.belief_optimum:
            try:
                plan = decisions.belief_optimum(psnap, belief, max_stops, seed=seed, n_samples=n)
            except decisions.InvalidDecision as exc:
                rows.append({**row, "status": "invalid", "reason": str(exc)})
                continue
        if plan is None:
            rows.append({**row, "status": "invalid", "reason": "no plan submitted"})
            continue
        batch = world.simulate_batch(snap, [plan], seed=seed, n_samples=n)
        if not batch.valid[0]:
            shown = plan.as_pairs() if hasattr(plan, "as_pairs") else plan
            rows.append({**row, "status": "invalid", "reason": "; ".join(batch.errors[0]),
                         "plan": json.dumps(shown, default=str)})
            continue
        res = batch.result(0, snap)
        mean = res.remaining.mean_s
        pos = _positions(world, snap, batch, 0) if ref_pos is not None else None
        final_pos = float(pos.mean()) if pos is not None else np.nan
        # the same plan summed in another order differs by ~1e-12 s; call that no regret
        regret = 0.0 if abs(mean - ref) < REGRET_TOL_S else mean - ref
        se = paired_se(batch.times[0], opt.times)
        stop_agree_set, first_pit_error_set = set_agreement(opt, res.num_stops, res.pit_laps, _first_pit_error)
        if borrowed is not None:
            row["proxied_compounds"] = _on_proxy(batch.plans[0], borrowed)
        rows.append({**row, "status": "ok", "plan": json.dumps(batch.plans[0].as_pairs()),
                     "num_stops": res.num_stops, "first_pit_lap": res.pit_laps[0] if res.pit_laps else None,
                     "time_s": mean, "std_s": res.remaining.std_s, "regret_s": regret,
                     "regret_se_s": se, "within_mc": bool(abs(regret) <= 2 * se),
                     "regret_pct": 100.0 * regret / ref, "stop_agree": res.num_stops == opt.plan.num_stops,
                     "first_pit_error": _first_pit_error(res.pit_laps, ref_pits),
                     "stop_agree_set": stop_agree_set, "first_pit_error_set": first_pit_error_set,
                     "support_warnings": len(res.support_warnings),
                     "final_position_mean": final_pos, "position_regret": final_pos - base["opt_final_position_mean"]})
    return rows


class NoStopReferenceWarning(UserWarning):
    """max_stops 0 with agent or recorded decisions in the run: the reference never stops."""


def warn_no_stop_reference(policies: dict, max_stops: int) -> bool:
    """Warn when max_stops is 0 and an agent or recorded policy is scored, so that comparing
    decisions against a reference that never stops is a deliberate choice. True if it warned."""
    if max_stops == 0 and any(getattr(p, "agent_decisions", False) for p in policies.values()):
        warnings.warn("agent decisions are compared against a no-stop reference", NoStopReferenceWarning,
                      stacklevel=3)
        return True
    return False


def score_set(scenario_set, sim_for, policies: dict, *, seed: int = 0, n_samples: int | None = None,
              max_stops: int = 2, items=None, ref_sims=(), warn: bool = True,
              references: ReferenceStore | None = None) -> pd.DataFrame:
    """Rows for every scenario (or `items`), stamped with the set's version and digest, both
    simulators' config hashes, the belief's fit window, the world's R8 stint_limit_rank and
    stint_limit_margin_laps (None: R8 off) and the code version.
    `sim_for(race_id, role)` returns the race's Simulator for role "belief" or "world"
    (race_simulators). config_hash leaves fit_window out, so belief_window says which
    window the policies planned on. `ref_sims` are more sim_for (E, optional): their world
    simulators plan the alternative references behind ref_models_agree, and their hashes
    are stamped as ref_world_config_hashes. `warn` runs warn_no_stop_reference first; the
    jobs runner turns it off, as a profile's max_stops is at least 1. `references` (an
    eval.reference.ReferenceStore, as `score` passes) supplies cached references; None
    computes every one, which gives the same rows."""
    if warn:
        warn_no_stop_reference(policies, max_stops)
    rows = []
    for scenario in scenario_set.items if items is None else items:
        belief, world = sim_for(scenario.race_id, "belief"), sim_for(scenario.race_id, "world")
        alts = [f(scenario.race_id, "world") for f in ref_sims]
        stamp = {"scenario_set": scenario_set.version, "scenario_digest": scenario_set.digest[:12],
                 "belief_config_hash": calibration.config_hash(belief.config),
                 "world_config_hash": calibration.config_hash(world.config),
                 "belief_window": belief.config.fit_window,
                 "stint_limit_rank": world.config.stint_limit_rank,
                 "stint_limit_margin_laps": world.config.stint_limit_margin_laps,
                 "sim_code_version": calibration.SIM_CODE_VERSION}
        if alts:
            stamp["ref_world_config_hashes"] = ";".join(calibration.config_hash(a.config) for a in alts)
        ref = None if references is None else references.get(scenario_set, scenario, world, seed=seed,
                                                              n_samples=n_samples, max_stops=max_stops)
        rows += [{**r, **stamp} for r in score_scenario(scenario, world, policies, belief=belief, seed=seed,
                                                         n_samples=n_samples, max_stops=max_stops,
                                                         ref_worlds=alts, reference=ref)]
    return pd.DataFrame(rows)


# The scoring world's SC and VSC (docs/eval.md "Pit loss and track status"): the period in
# progress at the snapshot ends where it really ended, with the field's recorded lap times on
# it; later periods are drawn at the rate model's rates, the same draws for every plan and the
# reference (common random numbers). The replay accuracy that chose the recorded lap times over
# the lap factors is in reports/calibration/neutralised (src/sim/neutral_report.py).
# Later periods (after the one in progress) are drawn at the rates and with the period lengths
# pooled over the dry races of configs/sim/neutralised_history.csv (events.RateEvents duration
# "history", pooled; docs/eval.md "Safety-car duration and rates: the belief against the judge",
# adopted 2026-11-22). The world before: configs/sim/legacy/world_pre_sc_alignment.yaml.
WORLD_EVENTS = {"name": "hybrid", "params": {"lap_times": "recorded", "duration": "history", "pooled": True}}
# ON by default since the fidelity scorecard chose it (docs/eval.md "What the ablation says": close
# pairs +0.016 [+0.000, +0.038], Spearman +0.011, relative gap -2.4 s, belief-favourite regret
# -0.36 s on 18 races). The world without neutralisations: `default_world_config(c, on=False)`, or
# `--world-config configs/sim/legacy/world_pre_scorecard.yaml` (the whole world before the change).
WORLD_EVENTS_DEFAULT = True


def default_world_config(config, on: bool | None = None):
    """The world when no world config is given: `config` (the profile's SimConfig), with
    WORLD_EVENTS when `on` (default WORLD_EVENTS_DEFAULT, on). A config whose events are not
    "none" is returned as it is."""
    on = WORLD_EVENTS_DEFAULT if on is None else on
    if not on or config.events.name != "none":
        return config
    return config.model_copy(update={"events": config.events.model_copy(
        update={"name": WORLD_EVENTS["name"], "params": dict(WORLD_EVENTS["params"])})})


def world_pit_loss(data, world) -> float:
    """The world's whole-race pit loss, read from the loaded race, never from a snapshot:
    under world.config.pit_loss.world "median" (the default) sim.pit_loss.whole_race on the
    world's full-window fit, else the race's replay mean (data.race.pit_loss_s)."""
    from ..sim import pit_loss
    cfg = world.config.pit_loss
    if cfg.world == "replay_mean":
        return float(data.race.pit_loss_s)
    # the world's root fit: every race lap under its (full) window
    return round(pit_loss.whole_race(data.frame, world.degradation, data.race_id, cfg)["pit_loss_s"], 3)


def race_simulators(profile=None, belief_config=None, world_config=None):
    """sim_for(race_id, role) over the committed cache, role "belief" or "world". Both share
    the race's frame and practice from sim.jobs.load_race. The belief runs `belief_config`,
    by default agent_tools.belief_config(), the agent tools' belief: configs/sim/belief.yaml unless
    AGENT_BELIEF_CONFIG names another (to_snapshot window, traffic off, posterior draws and
    persistent lap noise on, driver effects in the fits); the world runs
    `world_config`, by default the profile's SimConfig (full window, traffic off; the wear fit
    config.WORLD_DEGRADATION, the tyre-life edge config.WORLD_EDGE and the thin-fit wear guard
    config.WORLD_WEAR_GUARD, config.world_default;
    default_world_config, WORLD_EVENTS on by default). The world alone
    gets the race's whole-race pit loss (world_pit_loss); the belief prices stops with the as-of
    estimate its config asks for (docs/eval.md "Pit loss and track status"). Cached per race
    and role."""
    from ..sim import agent_tools, jobs
    from ..sim.config import world_default
    from ..sim.compute import load_profile
    from ..sim.engine import Simulator

    profile = profile or load_profile()
    # one source of truth: the agent tools' belief (configs/sim/belief.yaml, or AGENT_BELIEF_CONFIG);
    # AgentPolicy runs the tools on whatever belief this returns
    belief_config = belief_config or agent_tools.belief_config()
    races, cache = {}, {}

    def sim_for(race_id, role):
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if (race_id, role) not in cache:
            if race_id not in races:
                races[race_id] = jobs.load_race(race_id, profile)
            data = races[race_id]
            config = (belief_config if role == "belief"
                      else world_config or default_world_config(world_default(data.config)))
            sim = Simulator(data.frame, config, data.practice, qualifying=getattr(data, "qualifying", None))
            if role == "world":
                sim.race_pit_loss_s = world_pit_loss(data, sim)
            cache[race_id, role] = sim
        return cache[race_id, role]
    return sim_for


# --- result tables ------------------------------------------------------------------
def _summary(g: pd.DataFrame) -> dict:
    count = g["status"].value_counts()
    ok = g[g["status"] == "ok"]
    attempted = int(count.get("ok", 0) + count.get("invalid", 0))

    def mean(col):
        return float(ok[col].dropna().mean()) if col in ok and ok[col].notna().any() else np.nan

    def g_mean(col):
        return float(g[col].dropna().mean()) if col in g and g[col].notna().any() else np.nan

    regret = ok["regret_s"] if len(ok) else pd.Series(dtype=float)
    out = {
        "scenarios": len(g), "scored": len(ok), "unavailable": int(count.get("unavailable", 0)),
        "errors": int(count.get("error", 0)),
        "invalid_rate": count.get("invalid", 0) / attempted if attempted else np.nan,
        "mean_regret_s": float(regret.mean()) if len(ok) else np.nan,
        "median_regret_s": float(regret.median()) if len(ok) else np.nan,
        "p90_regret_s": float(regret.quantile(0.9)) if len(ok) else np.nan,
        "max_regret_s": float(regret.max()) if len(ok) else np.nan,
        "mean_regret_pct": mean("regret_pct"),
        "stop_agreement": float(ok["stop_agree"].astype(float).mean()) if len(ok) else np.nan,
        "share_within_mc": (float(ok["within_mc"].astype(float).mean())
                            if "within_mc" in ok and ok["within_mc"].notna().any() else np.nan),
        "mean_first_pit_error": mean("first_pit_error"),
        "mean_latency_s": g_mean("latency_s"),
        "mean_tool_calls": g_mean("tool_calls"),
        "mean_evidence_calls": g_mean("evidence_calls"),
        "mean_repeat_calls": g_mean("repeat_calls"),
        "mean_input_tokens": g_mean("input_tokens"),
        "mean_output_tokens": g_mean("output_tokens"),
    }
    if "position_regret" in g and g["position_regret"].notna().any():   # world traffic on (D)
        out["mean_position_regret"] = mean("position_regret")
        out["mean_final_position"] = mean("final_position_mean")
    if "p_pit" in g:   # rows from agent decisions: every row with a usable p_pit
        agent = g[g["p_pit"].notna() & g["pit_now_opt"].notna()]
        truth = agent["pit_now_opt"].astype(float)
        out["brier"] = float(((agent["p_pit"].astype(float) - truth) ** 2).mean()) if len(agent) else np.nan
        out["pit_now_agreement"] = (float(((agent["decision"] == "PIT").astype(float) == truth).mean())
                                    if len(agent) else np.nan)
    return out


class AgentRegretWarning(UserWarning):
    """An agent row came in below the reference by more than REGRET_TOL_S."""


def check_agent_regret(rows: pd.DataFrame) -> pd.DataFrame:
    """The scored agent rows (those with p_pit) whose regret is below -REGRET_TOL_S, with a
    warning when there are any. A decision's plan lies inside the reference's search space
    (eval.decisions counts max_stops the same way), so such a row means a bug. Baselines may
    go below the reference (an actual three-stop against a two-stop search) and are not checked."""
    if "p_pit" not in rows or "regret_s" not in rows:
        return rows.iloc[0:0]
    bad = rows[rows["p_pit"].notna() & (rows["status"] == "ok") & (rows["regret_s"] < -REGRET_TOL_S)]
    if len(bad):
        ids = ", ".join(f"{s} ({p}, {r:.3f} s)" for s, p, r in
                        bad[["scenario_id", "policy", "regret_s"]].head(5).itertuples(index=False))
        warnings.warn(f"{len(bad)} agent rows beat the reference optimizer, which their plans cannot "
                      f"do unless the scoring is wrong: {ids}", AgentRegretWarning, stacklevel=2)
    return bad


SUMMARY_STAMPS = ("stint_limit_rank", "stint_limit_margin_laps")


def _stamps(g: pd.DataFrame) -> dict:
    """The SUMMARY_STAMPS the rows carry, each as its one value (NaN when R8 is off, the
    distinct values joined by ";" if rows of several settings are summarized together)."""
    out = {}
    for col in SUMMARY_STAMPS:
        if col in g:
            vals = sorted(g[col].dropna().unique().tolist())
            out[col] = np.nan if not vals else vals[0] if len(vals) == 1 else ";".join(map(str, vals))
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, policy), then one per policy over every race with race_id "ALL",
    with the R8 stamps (SUMMARY_STAMPS) when the rows carry them, so tables scored at
    different stint_limit_rank or margin can be told apart side by side.
    Agent rows below the reference raise an AgentRegretWarning (check_agent_regret)."""
    check_agent_regret(rows)
    out = [{"race_id": race, "policy": pol, **_summary(g), **_stamps(g)}
           for (race, pol), g in rows.groupby(["race_id", "policy"], sort=True)]
    out += [{"race_id": "ALL", "policy": pol, **_summary(g), **_stamps(g)}
            for pol, g in rows.groupby("policy", sort=True)]
    return pd.DataFrame(out)


def summarize_agreeing(rows: pd.DataFrame) -> pd.DataFrame | None:
    """E: summarize over the rows whose reference models agree (ref_models_agree), None when
    the rows carry no flag. Beside summarize(rows) it shows how much of a policy's regret sits
    where the world models disagree about the right call."""
    if "ref_models_agree" not in rows:
        return None
    return summarize(rows[rows["ref_models_agree"].astype(bool)])


def summarize_near_ties(rows: pd.DataFrame) -> pd.DataFrame | None:
    """The near-tie view (information only), one row per policy over every race: of the
    scenarios with near-tie columns, the share whose optimal set holds more than one plan
    (opt_set_size > 1) and the share where pitting now is decisive (of those with both a
    pit-now and a stay-out plan); then over the scored rows the stop agreement and mean first
    pit error against the reference and against the closest member of the optimal set. None
    when the rows carry no near-tie columns (rows scored before them, or a format-1 cache)."""
    if "opt_set_size" not in rows or rows["opt_set_size"].isna().all():
        return None
    out = []
    for pol, g in rows[rows["opt_set_size"].notna()].groupby("policy", sort=True):
        ok = g[g["status"] == "ok"]
        decisive = g["pit_now_decisive"].dropna().astype(float)

        def mean(col):
            return float(ok[col].dropna().astype(float).mean()) if ok[col].notna().any() else np.nan
        out.append({"policy": pol, "scenarios": len(g),
                    "share_opt_set_gt1": float((g["opt_set_size"] > 1).mean()),
                    "share_pit_now_decisive": float(decisive.mean()) if len(decisive) else np.nan,
                    "scored": len(ok), "stop_agreement": mean("stop_agree"),
                    "stop_agreement_set": mean("stop_agree_set"),
                    "mean_first_pit_error": mean("first_pit_error"),
                    "mean_first_pit_error_set": mean("first_pit_error_set")})
    return pd.DataFrame(out)


def to_markdown(table: pd.DataFrame, digits: int = 2) -> str:
    """A github table, without the optional tabulate dependency."""
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


# --- sim.jobs eval_chunk -------------------------------------------------------------
def _sim_config(value):
    """A SimConfig from None, a SimConfig, a dict or a yaml / json path."""
    from ..sim import SimConfig
    if value is None or isinstance(value, SimConfig):
        return value
    if isinstance(value, dict):
        return SimConfig.model_validate(value)
    return SimConfig.from_file(value)


def _spec_value(value):
    """A config as a plain picklable value for the jobs payload: a path string or a dict."""
    if value is None or isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(Path(value).resolve())


def install_job_runner(scenarios_path, policy_names=("baselines",), recorded=(), seed=0,
                       belief_config=None, world_config=None, ref_world_configs=(), reference_dir=None,
                       require_reference=False):
    """Register the eval with sim.jobs so `eval_chunk` tasks score slices of a scenario set.
    Chunk i is items[i * size:(i + 1) * size], size = profile.parallel.eval_chunk_size.
    The configs (SimConfig, dict or file path) go to race_simulators. The runner reads the
    reference cache under `reference_dir` (default data/references) whenever a reference
    matches its inputs and computes it otherwise; `require_reference` makes a miss an error
    (eval.reference.ReferenceMissing), so a report run fails on its first chunk. The
    runner's spec (scenarios path, policy names, recorded files, configs, reference_dir,
    require_reference) is registered with it, so run-all carries it in each task's payload
    and every worker installs the same runner. Returns the number of scenario items."""
    from ..sim import jobs
    from .policies import resolve
    from .scenarios import ScenarioSet

    scenario_set = ScenarioSet.load(scenarios_path)
    pols = resolve(policy_names, recorded, set_name=scenario_set.version)
    belief, world = _sim_config(belief_config), _sim_config(world_config)
    alts = [_sim_config(c) for c in ref_world_configs]
    reference_dir = Path(reference_dir or REFERENCE_DIR).resolve()
    store = ReferenceStore(reference_dir, require=require_reference)

    def runner(chunk_id, profile):
        index = int(chunk_id.rsplit("_", 1)[1])
        items = scenario_set.chunk(index, profile.parallel.eval_chunk_size)
        refs = [race_simulators(profile, belief, a) for a in alts]
        return score_set(scenario_set, race_simulators(profile, belief, world), pols, seed=seed,
                         n_samples=profile.sim.n_samples, max_stops=profile.sim.max_stops, items=items,
                         ref_sims=refs, warn=False, references=store)
    spec = {"installer": f"{__name__}:install_job_runner", "scenarios_path": str(Path(scenarios_path).resolve()),
            "policy_names": list(policy_names), "recorded": [str(Path(r).resolve()) for r in recorded],
            "seed": seed, "belief_config": _spec_value(belief_config), "world_config": _spec_value(world_config),
            "ref_world_configs": [_spec_value(c) for c in ref_world_configs],
            "reference_dir": str(reference_dir), "require_reference": bool(require_reference)}
    jobs.set_eval_runner(runner, spec=spec)
    return len(scenario_set.items)
