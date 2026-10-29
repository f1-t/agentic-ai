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
from ..sim import calibration, traffic
from ..sim.optimizer_bridge import optimize_batch
from . import decisions
from .policies import Decision

STATUSES = ("ok", "invalid", "unavailable", "error", "no_reference")
REGRET_TOL_S = 1e-6
REF_AGREE_LAPS = 2   # E: first pit laps this close count as the same reference call
ROLES = ("belief", "world")
BELIEF_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sim" / "belief.yaml"


def _first_pit_error(pits, ref_pits):
    """|first pit lap - optimizer's first pit lap|; 0 when neither stops, None when only one does."""
    if not pits and not ref_pits:
        return 0
    if not pits or not ref_pits:
        return None
    return abs(pits[0] - ref_pits[0])


def finishing_positions(frame: pd.DataFrame, snapshot, times, dnf_lap) -> np.ndarray:
    """(n_samples,) finishing position per sample of one plan: 1 + the ghosts (every other
    driver in `frame`) that finished lap race.total_laps before our session time, anchor +
    remaining time (traffic.place, vectorised over the samples). A DNF sample (finite
    dnf_lap) is behind every finisher and behind the ghosts that retired no earlier than its
    dnf_lap; its penalised time is not used."""
    last = snapshot.race.total_laps
    t = traffic.anchor(frame, snapshot) + np.asarray(times, dtype=float)
    G = traffic.ghost_table(frame, snapshot.driver, [last]).G[:, 0]
    finished = G[np.isfinite(G)]
    pos = 1 + (finished[None, :] < t[:, None]).sum(axis=1)
    dnf_lap = np.asarray(dnf_lap, dtype=float)
    dnf = np.isfinite(dnf_lap)
    if dnf.any():
        ghosts = frame[(frame["driver"] != snapshot.driver) & np.isfinite(frame["end_s"].astype(float))]
        reached = ghosts.groupby("driver")["lap"].max()
        retired = reached[reached < last].to_numpy(dtype=float)   # lap count of each non-finisher
        pos[dnf] = 1 + len(finished) + (retired[None, :] >= dnf_lap[dnf][:, None]).sum(axis=1)
    return pos


def _positions(world, snap, batch, i):
    """finishing_positions of plan i of a world batch, None when traffic was not placed."""
    if not batch.traffic_on:
        return None
    return finishing_positions(world.laps, snap, batch.times[i], batch.dnf_lap[i])


def references_agree(plan, other, snapshot, within: int = REF_AGREE_LAPS) -> bool:
    """E: two reference plans agree when they make the same number of stops and their first
    pit laps are at most `within` laps apart (two no-stop plans agree)."""
    a, b = plan.pit_laps(snapshot), other.pit_laps(snapshot)
    return len(a) == len(b) and (not a or abs(a[0] - b[0]) <= within)


def score_scenario(scenario, world, policies: dict, *, belief=None, seed: int = 0,
                   n_samples: int | None = None, max_stops: int = 2, ref_worlds=()) -> list[dict]:
    """One row per policy for one scenario. Policies plan on `belief` (`world` when None, one
    simulator in both roles); the reference and the scoring use `world`. With `ref_worlds`
    (E, optional) each of those world models plans its own reference too, and the rows gain
    ref_models_agree: every one of them agrees with the world's reference (references_agree).
    They score nothing."""
    belief = world if belief is None else belief
    snap = scenario.snapshot
    base = {"scenario_id": scenario.scenario_id, "race_id": scenario.race_id, "driver": scenario.driver,
            "lap": scenario.lap, "compound": snap.compound, "tyre_age": snap.tyre_age,
            "remaining_laps": snap.remaining_laps, "seed": seed, "max_stops": max_stops}
    try:
        opt = optimize_batch(snap, world, seed=seed, n_samples=n_samples, max_stops=max_stops)
    except NoFeasiblePlan as exc:
        return [{**base, "policy": name, "status": "no_reference", "reason": str(exc)} for name in policies]
    ref = opt.evaluation.mean_time_s
    ref_pits = opt.plan.pit_laps(snap)
    n = opt.diagnostics["n_samples"]
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
    base.update(n_samples=n, opt_plan=json.dumps(opt.plan.as_pairs()), opt_num_stops=opt.plan.num_stops,
                opt_first_pit_lap=ref_pits[0] if ref_pits else None, opt_time_s=ref,
                pit_now_opt=bool(ref_pits) and ref_pits[0] == snap.laps_completed + 1,
                opt_final_position_mean=float(ref_pos.mean()) if ref_pos is not None else np.nan)
    rows = []
    for name, policy in policies.items():
        row = {**base, "policy": name}
        started = perf_counter()
        try:
            decision = policy(scenario, belief, max_stops)
        except PolicyUnavailable as exc:
            rows.append({**row, "status": "unavailable", "reason": str(exc)})
            continue
        if not isinstance(decision, Decision):
            decision = Decision(decision)
        row.update(latency_s=perf_counter() - started, tool_calls=decision.tool_calls,
                   input_tokens=decision.input_tokens, output_tokens=decision.output_tokens)
        if decision.error is not None:
            rows.append({**row, "status": "error", "reason": decision.error})
            continue
        plan = decision.plan
        if plan is None and decision.submission is not None:
            sub = decision.submission
            row.update(decision=str(sub.get("decision", "")).strip().upper() or None,
                       p_pit=decisions.p_pit(sub))
            try:
                plan = decisions.to_plan(snap, sub, belief, max_stops, seed=seed, n_samples=n)
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
        rows.append({**row, "status": "ok", "plan": json.dumps(batch.plans[0].as_pairs()),
                     "num_stops": res.num_stops, "first_pit_lap": res.pit_laps[0] if res.pit_laps else None,
                     "time_s": mean, "std_s": res.remaining.std_s, "regret_s": regret,
                     "regret_pct": 100.0 * regret / ref, "stop_agree": res.num_stops == opt.plan.num_stops,
                     "first_pit_error": _first_pit_error(res.pit_laps, ref_pits),
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
              max_stops: int = 2, items=None, ref_sims=(), warn: bool = True) -> pd.DataFrame:
    """Rows for every scenario (or `items`), stamped with the set's version and digest, both
    simulators' config hashes, the belief's fit window and the code version.
    `sim_for(race_id, role)` returns the race's Simulator for role "belief" or "world"
    (race_simulators). config_hash leaves fit_window out, so belief_window says which
    window the policies planned on. `ref_sims` are more sim_for (E, optional): their world
    simulators plan the alternative references behind ref_models_agree, and their hashes
    are stamped as ref_world_config_hashes. `warn` runs warn_no_stop_reference first; the
    jobs runner turns it off, as a profile's max_stops is at least 1."""
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
                 "sim_code_version": calibration.SIM_CODE_VERSION}
        if alts:
            stamp["ref_world_config_hashes"] = ";".join(calibration.config_hash(a.config) for a in alts)
        rows += [{**r, **stamp} for r in score_scenario(scenario, world, policies, belief=belief, seed=seed,
                                                         n_samples=n_samples, max_stops=max_stops,
                                                         ref_worlds=alts)]
    return pd.DataFrame(rows)


def race_simulators(profile=None, belief_config=None, world_config=None):
    """sim_for(race_id, role) over the committed cache, role "belief" or "world". Both share
    the race's frame and practice from sim.jobs.load_race. The belief runs `belief_config`,
    by default configs/sim/belief.yaml (to_snapshot window, traffic off); the world runs
    `world_config`, by default the profile's SimConfig (full window, traffic off, as before
    the split). Cached per race and role."""
    from ..sim import SimConfig, jobs
    from ..sim.compute import load_profile
    from ..sim.engine import Simulator

    profile = profile or load_profile()
    belief_config = belief_config or SimConfig.from_file(BELIEF_CONFIG)
    races, cache = {}, {}

    def sim_for(race_id, role):
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if (race_id, role) not in cache:
            if race_id not in races:
                races[race_id] = jobs.load_race(race_id, profile)
            data = races[race_id]
            config = belief_config if role == "belief" else world_config or data.config
            cache[race_id, role] = Simulator(data.frame, config, data.practice)
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
        "mean_first_pit_error": mean("first_pit_error"),
        "mean_latency_s": g_mean("latency_s"),
        "mean_tool_calls": g_mean("tool_calls"),
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


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, policy), then one per policy over every race with race_id "ALL".
    Agent rows below the reference raise an AgentRegretWarning (check_agent_regret)."""
    check_agent_regret(rows)
    out = [{"race_id": race, "policy": pol, **_summary(g)}
           for (race, pol), g in rows.groupby(["race_id", "policy"], sort=True)]
    out += [{"race_id": "ALL", "policy": pol, **_summary(g)} for pol, g in rows.groupby("policy", sort=True)]
    return pd.DataFrame(out)


def summarize_agreeing(rows: pd.DataFrame) -> pd.DataFrame | None:
    """E: summarize over the rows whose reference models agree (ref_models_agree), None when
    the rows carry no flag. Beside summarize(rows) it shows how much of a policy's regret sits
    where the world models disagree about the right call."""
    if "ref_models_agree" not in rows:
        return None
    return summarize(rows[rows["ref_models_agree"].astype(bool)])


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
                       belief_config=None, world_config=None, ref_world_configs=()):
    """Register the eval with sim.jobs so `eval_chunk` tasks score slices of a scenario set.
    Chunk i is items[i * size:(i + 1) * size], size = profile.parallel.eval_chunk_size.
    The configs (SimConfig, dict or file path) go to race_simulators. The runner's spec
    (scenarios path, policy names, recorded files, configs) is registered with it, so
    run-all carries it in each task's payload and every worker installs the same runner.
    Returns the number of scenario items."""
    from ..sim import jobs
    from .policies import resolve
    from .scenarios import ScenarioSet

    scenario_set = ScenarioSet.load(scenarios_path)
    pols = resolve(policy_names, recorded, set_name=scenario_set.version)
    belief, world = _sim_config(belief_config), _sim_config(world_config)
    alts = [_sim_config(c) for c in ref_world_configs]

    def runner(chunk_id, profile):
        index = int(chunk_id.rsplit("_", 1)[1])
        items = scenario_set.chunk(index, profile.parallel.eval_chunk_size)
        refs = [race_simulators(profile, belief, a) for a in alts]
        return score_set(scenario_set, race_simulators(profile, belief, world), pols, seed=seed,
                         n_samples=profile.sim.n_samples, max_stops=profile.sim.max_stops, items=items,
                         ref_sims=refs, warn=False)
    spec = {"installer": f"{__name__}:install_job_runner", "scenarios_path": str(Path(scenarios_path).resolve()),
            "policy_names": list(policy_names), "recorded": [str(Path(r).resolve()) for r in recorded],
            "seed": seed, "belief_config": _spec_value(belief_config), "world_config": _spec_value(world_config),
            "ref_world_configs": [_spec_value(c) for c in ref_world_configs]}
    jobs.set_eval_runner(runner, spec=spec)
    return len(scenario_set.items)
