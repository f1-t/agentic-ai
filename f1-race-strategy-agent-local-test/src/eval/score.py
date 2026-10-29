"""Regret scoring: each policy's plan against the optimizer's, from the same snapshot.

For every scenario the optimizer (sim.optimizer_bridge.optimize_batch) finds the best plan
with at most `max_stops` future stops, and every policy's plan is scored by the same
simulator under the same draws (docs/rules.md R6: seed s, samples 0..n-1). Then

    regret_s = mean remaining time of the policy's plan - mean remaining time of the optimizer's

A plan with more stops than the optimizer may search (an actual three-stop, say) can come in
below the reference, so regret can be negative; it is reported as it is. One row per
(scenario, policy); `summarize` turns rows into the per-race and overall tables.

Row status: ok, invalid (the plan broke a rule or no plan was submitted), unavailable (the
policy does not apply to this scenario), error (the policy itself failed, an API error say),
no_reference (no legal plan exists for the optimizer).
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd

from ..baselines.domain import NoFeasiblePlan, PolicyUnavailable
from ..sim import calibration
from ..sim.optimizer_bridge import optimize_batch
from .policies import Decision

STATUSES = ("ok", "invalid", "unavailable", "error", "no_reference")
REGRET_TOL_S = 1e-6


def _first_pit_error(pits, ref_pits):
    """|first pit lap - optimizer's first pit lap|; 0 when neither stops, None when only one does."""
    if not pits and not ref_pits:
        return 0
    if not pits or not ref_pits:
        return None
    return abs(pits[0] - ref_pits[0])


def score_scenario(scenario, sim, policies: dict, *, seed: int = 0, n_samples: int | None = None,
                   max_stops: int = 2) -> list[dict]:
    """One row per policy for one scenario."""
    snap = scenario.snapshot
    base = {"scenario_id": scenario.scenario_id, "race_id": scenario.race_id, "driver": scenario.driver,
            "lap": scenario.lap, "compound": snap.compound, "tyre_age": snap.tyre_age,
            "remaining_laps": snap.remaining_laps, "seed": seed, "max_stops": max_stops}
    try:
        opt = optimize_batch(snap, sim, seed=seed, n_samples=n_samples, max_stops=max_stops)
    except NoFeasiblePlan as exc:
        return [{**base, "policy": name, "status": "no_reference", "reason": str(exc)} for name in policies]
    ref = opt.evaluation.mean_time_s
    ref_pits = opt.plan.pit_laps(snap)
    n = opt.diagnostics["n_samples"]
    base.update(n_samples=n, opt_plan=json.dumps(opt.plan.as_pairs()), opt_num_stops=opt.plan.num_stops,
                opt_first_pit_lap=ref_pits[0] if ref_pits else None, opt_time_s=ref)
    rows = []
    for name, policy in policies.items():
        row = {**base, "policy": name}
        started = perf_counter()
        try:
            decision = policy(scenario, sim, max_stops)
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
        if decision.plan is None:
            rows.append({**row, "status": "invalid", "reason": "no plan submitted"})
            continue
        batch = sim.simulate_batch(snap, [decision.plan], seed=seed, n_samples=n)
        if not batch.valid[0]:
            rows.append({**row, "status": "invalid", "reason": "; ".join(batch.errors[0]),
                         "plan": json.dumps(decision.plan, default=str)})
            continue
        res = batch.result(0, snap)
        mean = res.remaining.mean_s
        # the same plan summed in another order differs by ~1e-12 s; call that no regret
        regret = 0.0 if abs(mean - ref) < REGRET_TOL_S else mean - ref
        rows.append({**row, "status": "ok", "plan": json.dumps(batch.plans[0].as_pairs()),
                     "num_stops": res.num_stops, "first_pit_lap": res.pit_laps[0] if res.pit_laps else None,
                     "time_s": mean, "std_s": res.remaining.std_s, "regret_s": regret,
                     "regret_pct": 100.0 * regret / ref, "stop_agree": res.num_stops == opt.plan.num_stops,
                     "first_pit_error": _first_pit_error(res.pit_laps, ref_pits),
                     "support_warnings": len(res.support_warnings)})
    return rows


def score_set(scenario_set, sim_for, policies: dict, *, seed: int = 0, n_samples: int | None = None,
              max_stops: int = 2, items=None) -> pd.DataFrame:
    """Rows for every scenario (or `items`), stamped with the set's version and digest and the
    simulator's config hash and code version. `sim_for(race_id)` returns the race's Simulator."""
    rows = []
    for scenario in scenario_set.items if items is None else items:
        sim = sim_for(scenario.race_id)
        stamp = {"scenario_set": scenario_set.version, "scenario_digest": scenario_set.digest[:12],
                 "config_hash": calibration.config_hash(sim.config),
                 "sim_code_version": calibration.SIM_CODE_VERSION}
        rows += [{**r, **stamp} for r in score_scenario(scenario, sim, policies, seed=seed,
                                                         n_samples=n_samples, max_stops=max_stops)]
    return pd.DataFrame(rows)


def race_simulators(profile=None, config=None):
    """sim_for(race_id) over the committed cache: the race's frame and practice from
    sim.jobs.load_race, the profile's SimConfig unless `config` is given. Cached per race."""
    from ..sim import jobs
    from ..sim.compute import load_profile
    from ..sim.engine import Simulator

    profile = profile or load_profile()
    cache = {}

    def sim_for(race_id):
        if race_id not in cache:
            data = jobs.load_race(race_id, profile)
            cache[race_id] = Simulator(data.frame, config or data.config, data.practice)
        return cache[race_id]
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
    return {
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


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, policy), then one per policy over every race with race_id "ALL"."""
    out = [{"race_id": race, "policy": pol, **_summary(g)}
           for (race, pol), g in rows.groupby(["race_id", "policy"], sort=True)]
    out += [{"race_id": "ALL", "policy": pol, **_summary(g)} for pol, g in rows.groupby("policy", sort=True)]
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
def install_job_runner(scenarios_path, policy_names=("baselines",), recorded=(), seed=0, config=None):
    """Register the eval with sim.jobs so `eval_chunk` tasks score slices of a scenario set.
    Chunk i is items[i * size:(i + 1) * size], size = profile.parallel.eval_chunk_size."""
    from ..sim import jobs
    from .policies import resolve
    from .scenarios import ScenarioSet

    scenario_set = ScenarioSet.load(scenarios_path)
    pols = resolve(policy_names, recorded)

    def runner(chunk_id, profile):
        index = int(chunk_id.rsplit("_", 1)[1])
        items = scenario_set.chunk(index, profile.parallel.eval_chunk_size)
        return score_set(scenario_set, race_simulators(profile, config), pols, seed=seed,
                         n_samples=profile.sim.n_samples, max_stops=profile.sim.max_stops, items=items)
    jobs.set_eval_runner(runner)
    return len(scenario_set.items)
