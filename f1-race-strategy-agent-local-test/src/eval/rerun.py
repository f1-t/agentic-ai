"""The evaluation rerun under the corrected protocol, one command per stage (docs/eval.md
"Rerunning the evaluation").

    python -m src.eval.rerun scenarios  --set dev20 [--rule as_of|stratified] [--final]
    python -m src.eval.rerun references --set dev20 --world-config default [--jobs 4]
    python -m src.eval.rerun baselines  --set dev20 --world-config default --belief-config default [--jobs 4]
    python -m src.eval.rerun agent      --set dev20 --world-config default --belief-config default [--jobs 6]
                                        [--timing-sample 24] [--resume] [--force] [--only ids.txt]
    python -m src.eval.rerun tables     --set dev20
    python -m src.eval.rerun smoke      [--n 8] [--jobs 4]      # a live check after a merge (src/eval/smoke.py)

Every stage reads and writes under one directory, --out-dir (default reports/eval/rerun/<set>):

    scenarios.json            the set (stage scenarios; never rebuilt over an existing file)
    references/               the world's references, the cache score reads (stage references)
    parts/references/<race>.pkl, parts/baselines/<race>.csv   per-race results: a rerun skips a
                              race whose part exists, so an interrupted stage resumes
    baselines.csv             every non-LLM policy's rows (stage baselines)
    agent.csv, trajectories/  the live agent (stage agent, src/eval/agent_pass.py: --jobs workers, resumable,
                              never overwritten without --force); parts/agent/ its per-worker rows;
                              timing_sample.csv, timing_sample/  the sequential timing sample; agent_run.json
    tables/                   combined rows, summaries, paired tables, chart and tail tables (stage tables)
    provenance.json           commit, tag, configs and hashes, SIM_CODE_VERSION, set digest, model
                              and loop settings, and every command run here (commands.log)

The sets are defined in configs/eval/rerun_sets.yaml. A report set (development false: test9,
the held-out races) is built only with --final, on the owner's word: building it loads the
held-out races. --world-config / --belief-config take a SimConfig file or "default" (the world:
score.default_world_config of config.world_default, the scorecard-chosen judge with driver effects,
the history-only tyre-life edge and hybrid SC/VSC; the belief: configs/sim/belief.yaml, which the
agent's tools read too). provenance.json names each of those choices. The parallel stages run --jobs
worker processes, one race at a time each, with BLAS capped at one thread per worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SETS_FILE = ROOT / "configs" / "eval" / "rerun_sets.yaml"
OUT_ROOT = ROOT / "reports" / "eval" / "rerun"
BASELINE_POLICIES = ("fixed_one_stop", "fixed_two_stop", "greedy_crossover", "actual_strategy", "run_to_flag",
                     "one_stop_total", "belief_optimizer")
RULES = ("as_of", "stratified")
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
TAIL_S = (10.0, 20.0, 30.0)
# the environment provenance.json records as the model: the backend and the model id of each backend
# (and the voyager reply cap), never a key, token or endpoint
MODEL_ENV = ("LLM_BACKEND", "ANTHROPIC_MODEL", "HF_MODEL", "VOYAGER_MODEL", "VOYAGER_MAX_TOKENS", "VOYAGER_TEMPERATURE")
LOOP_ENV = ("AGENT_MAX_TOOL_CALLS", "AGENT_SUBMIT_GRACE_STEPS", "AGENT_REPEAT_MODE", "AGENT_TURN_STATUS",
            "AGENT_SHOW_IDENTITY", "AGENT_GAP_YARDSTICK", "AGENT_TURN_REPEAT")


def _loop_env() -> dict:
    """LOOP_ENV as set, except AGENT_TURN_REPEAT, recorded as the mode in effect (the agent's
    default when unset): the default moved from "end" to "run" on 2026-11-22, so an unset value
    would not say which mode a pass ran with."""
    from ..agent import DEFAULT_TURN_REPEAT
    env = {k: os.environ.get(k) for k in LOOP_ENV}
    env["AGENT_TURN_REPEAT"] = env["AGENT_TURN_REPEAT"] or DEFAULT_TURN_REPEAT
    return env


ERAS = (("2019-20", 2019, 2020), ("2021", 2021, 2021), ("2022", 2022, 2022), ("2023", 2023, 2023),
        ("2024-25", 2024, 2025))


# --- the set definitions ------------------------------------------------------------------------
def load_sets(path=SETS_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_spec(name: str, path=SETS_FILE) -> dict:
    """{races, per_race, seed, development, eras} of set `name`; races_from (a json with a "races"
    list, or a bare list) is read into races."""
    sets = load_sets(path)
    if name not in sets:
        raise KeyError(f"unknown set {name!r}; defined in {path}: {sorted(sets)}")
    spec = dict(sets[name])
    if "races_from" in spec:
        data = json.loads((ROOT / spec["races_from"]).read_text())
        spec["races"] = list(data if isinstance(data, list) else data["races"])
    spec.setdefault("eras", False)
    return spec


def out_dir(args) -> Path:
    return Path(args.out_dir) if args.out_dir else OUT_ROOT / args.set


def _config_path(value, default):
    """A config file path, or `default` (None for the race's own) for "default"/None."""
    return default if value in (None, "default") else str(Path(value).resolve())


def belief_path(value):
    from .score import BELIEF_CONFIG
    return _config_path(value, str(BELIEF_CONFIG))


def world_path(value):
    return _config_path(value, None)


def _sim_config(path):
    if path is None:
        return None
    from ..sim import SimConfig
    return SimConfig.from_file(path)


def _log_command(directory: Path, argv):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "commands.log", "a") as f:
        f.write(" ".join(["python -m src.eval.rerun", *map(str, argv)]) + "\n")


# --- stage 1: scenarios ----------------------------------------------------------------------------
def stage_scenarios(args, build=None) -> Path:
    """Build the set with the leak-free sampler (as_of, or stratified) and write scenarios.json.
    A report set (development false) needs --final; an existing file is never overwritten."""
    spec = set_spec(args.set, args.sets)
    if not spec["development"] and not args.final:
        raise SystemExit(f"set {args.set!r} is a report set (development false): building it loads held-out races. "
                         "Pass --final, once, on the owner's word.")
    path = out_dir(args) / "scenarios.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a scenario set is built once (remove it to rebuild a development set)")
    from ..sim.compute import load_profile
    from .scenarios import build_set
    build = build or build_set
    kw = {"max_per_race": spec["per_race"]} if args.rule == "as_of" else {}
    out = build(spec["races"], args.set, profile=load_profile(args.profile), rule=args.rule, seed=spec["seed"],
                development=bool(spec["development"]), **kw)
    out.save(path)
    print(f"{args.set} ({out.kind}, rule {args.rule}): {len(out.items)} scenarios, {len(out.skipped)} skipped, "
          f"digest {out.digest[:12]} -> {path}")
    return path


# --- the race-parallel runner ------------------------------------------------------------------------
def _races(scenario_set) -> list[str]:
    return sorted({s.race_id for s in scenario_set.items})


def run_parallel(fn, races, jobs: int, **kw):
    """fn(race_id, **kw) for every race, `jobs` spawned worker processes with BLAS at one thread
    each (the env is set before the workers import numpy); in this process when jobs is 1."""
    if jobs <= 1 or len(races) <= 1:
        return [fn(r, **kw) for r in races]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    saved = {k: os.environ.get(k) for k in BLAS_ENV}
    os.environ.update({k: "1" for k in BLAS_ENV})
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(fn, r, **kw) for r in races]
            return [f.result() for f in futures]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _profile_for(name):
    from ..sim.compute import load_profile
    return load_profile(name)


# --- stage 2: references -----------------------------------------------------------------------------
def _reference_part(race_id, *, scenarios, world, profile, part_dir):
    """One race's references, pickled to part_dir/<race>.pkl (skipped when it exists)."""
    from .reference import compute_reference
    from .scenarios import ScenarioSet
    from .score import race_simulators
    part = Path(part_dir) / f"{race_id}.pkl"
    if part.exists():
        return race_id, "exists", 0.0
    started = time.perf_counter()
    prof = _profile_for(profile)
    s = ScenarioSet.load(scenarios)
    sim_for = race_simulators(prof, None, _sim_config(world))
    w = sim_for(race_id, "world")
    refs = [compute_reference(sc, w, seed=0, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops)
            for sc in s.items if sc.race_id == race_id]
    tmp = part.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(refs))
    tmp.replace(part)
    return race_id, "built", time.perf_counter() - started


def stage_references(args) -> Path:
    """Every scenario's reference on the world config, race-parallel and resumable; then the
    references cache (references/) score reads."""
    from .reference import reference_inputs, write_references
    from .scenarios import ScenarioSet
    from .score import race_simulators
    d = out_dir(args)
    s = ScenarioSet.load(d / "scenarios.json")
    world = world_path(args.world_config)
    parts = d / "parts" / "references"
    parts.mkdir(parents=True, exist_ok=True)
    for race_id, status, secs in run_parallel(_reference_part, _races(s), args.jobs, scenarios=str(d / "scenarios.json"),
                                              world=world, profile=args.profile, part_dir=str(parts)):
        print(f"references {race_id}: {status} {secs:.0f} s", flush=True)
    refs = [r for race in _races(s) for r in pickle.loads((parts / f"{race}.pkl").read_bytes())]
    prof = _profile_for(args.profile)
    w = race_simulators(prof, None, _sim_config(world))(_races(s)[0], "world")
    inputs = reference_inputs(s, w, seed=0, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops)
    path = write_references(refs, {sc.scenario_id: sc.snapshot for sc in s.items}, inputs, s.version,
                            root=d / "references")
    print(f"{len(refs)} references -> {path}")
    return path


# --- stage 3: baselines --------------------------------------------------------------------------------
def _score_part(race_id, *, scenarios, world, belief, profile, policies, part_dir, references):
    """One race's rows for `policies`, to part_dir/<race>.csv (skipped when it exists)."""
    from .policies import resolve
    from .reference import ReferenceStore
    from .scenarios import ScenarioSet
    from .score import race_simulators, score_set
    part = Path(part_dir) / f"{race_id}.csv"
    if part.exists():
        return race_id, "exists", 0.0
    started = time.perf_counter()
    prof = _profile_for(profile)
    s = ScenarioSet.load(scenarios)
    items = [sc for sc in s.items if sc.race_id == race_id]
    rows = score_set(s, race_simulators(prof, _sim_config(belief), _sim_config(world)), resolve(policies),
                     n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops, items=items, warn=False,
                     references=ReferenceStore(references, require=True))
    tmp = part.with_suffix(".tmp")
    rows.to_csv(tmp, index=False)
    tmp.replace(part)
    return race_id, "scored", time.perf_counter() - started


def stage_baselines(args) -> Path:
    """Every non-LLM policy (BASELINE_POLICIES) on the set, race-parallel and resumable, the
    references required; writes baselines.csv."""
    from .scenarios import ScenarioSet
    d = out_dir(args)
    s = ScenarioSet.load(d / "scenarios.json")
    parts = d / "parts" / "baselines"
    parts.mkdir(parents=True, exist_ok=True)
    for race_id, status, secs in run_parallel(
            _score_part, _races(s), args.jobs, scenarios=str(d / "scenarios.json"), world=world_path(args.world_config),
            belief=belief_path(args.belief_config), profile=args.profile, policies=list(BASELINE_POLICIES),
            part_dir=str(parts), references=str(d / "references")):
        print(f"baselines {race_id}: {status} {secs:.0f} s", flush=True)
    rows = pd.concat([pd.read_csv(parts / f"{r}.csv") for r in _races(s)], ignore_index=True)
    path = d / "baselines.csv"
    rows.to_csv(path, index=False)
    print(f"{len(rows)} rows -> {path}")
    return path


# --- stage 4: agent ---------------------------------------------------------------------------------------
def _only_ids(value) -> list[str] | None:
    """--only: a file of scenario ids (one per line, # comments) or a comma-separated list."""
    if not value:
        return None
    p = Path(value)
    text = p.read_text() if p.is_file() else value.replace(",", "\n")
    return [x.strip() for x in text.splitlines() if x.strip() and not x.strip().startswith("#")]


def _agent_outputs(d: Path) -> dict:
    return {"agent.csv": d / "agent.csv", "trajectories": d / "trajectories", "parts": d / "parts" / "agent",
            "timing_sample.csv": d / "timing_sample.csv", "timing_sample": d / "timing_sample",
            "agent_run.json": d / "agent_run.json"}


def _nonempty(p: Path) -> bool:
    return p.is_file() or (p.is_dir() and any(p.iterdir()))


def stage_agent(args) -> Path:
    """The live agent (LLM_BACKEND) on every scenario of the set (or --only's), its tools on the belief
    config (AGENT_BELIEF_CONFIG), with --jobs worker processes (src/eval/agent_pass.py): the rows merged
    in the set's order into agent.csv, one trajectory per scenario in trajectories/. With
    --timing-sample K, K scenarios first run sequentially into timing_sample.csv (timing only; the pass
    runs them again). --resume skips finished scenarios; a finished or a started pass is never
    overwritten without --force. agent_run.json records the passes."""
    from . import agent_pass as ap
    from .policies import AgentPolicy
    from .scenarios import ScenarioSet
    from .score import warn_no_stop_reference
    d = out_dir(args)
    out = _agent_outputs(d)
    resume, force = getattr(args, "resume", False), getattr(args, "force", False)
    rerun_errors, every = getattr(args, "rerun_errors", False), int(getattr(args, "progress_every", 10) or 10)
    jobs, k = max(int(getattr(args, "jobs", 1) or 1), 1), int(getattr(args, "timing_sample", 0) or 0)
    seed = int(getattr(args, "timing_seed", 0) or 0)
    if resume:
        if out["agent.csv"].exists():
            if not force:
                raise SystemExit(f"{out['agent.csv']} exists: the pass is finished; --force to merge it again")
            ap.clear([out["agent.csv"]])
    elif any(_nonempty(p) for p in out.values()):
        if not force:
            raise SystemExit(f"an agent pass exists under {d} (agent.csv, trajectories/, parts/agent/ or the timing "
                             "sample): --resume to finish it, --force to start over, or use a new --out-dir")
        ap.clear(out.values())
    belief = belief_path(args.belief_config)
    os.environ["AGENT_BELIEF_CONFIG"] = belief
    items = ScenarioSet.load(d / "scenarios.json").items
    only = _only_ids(getattr(args, "only", None))
    if only is not None:
        missing = sorted(set(only) - {sc.scenario_id for sc in items})
        if missing:
            raise SystemExit(f"--only names scenarios not in {d / 'scenarios.json'}: {missing[:5]}")
        items = [sc for sc in items if sc.scenario_id in set(only)]
    ap.check_names(items)
    prof = _profile_for(args.profile)
    warn_no_stop_reference({"agent": AgentPolicy}, prof.sim.max_stops)
    cfg = {"scenarios": str(d / "scenarios.json"), "profile": args.profile, "belief": belief,
           "world": world_path(args.world_config), "references": str(d / "references"),
           "backend": getattr(args, "backend", None) or ap.DEFAULT_BACKEND}
    if getattr(args, "backoff", None) is not None:   # tests
        cfg["backoff"] = tuple(args.backoff)
    run = json.loads(out["agent_run.json"].read_text()) if out["agent_run.json"].exists() else {"passes": []}

    def save_run():
        out["agent_run.json"].write_text(json.dumps(run, indent=1, default=str) + "\n")

    def record(p):
        # the model id and loop switches each pass ran with (provenance.json reads them later, at tables)
        p["env"] = {**{k: os.environ.get(k) for k in MODEL_ENV}, **_loop_env()}
        run["passes"].append(p)
        save_run()

    try:
        if k:
            sample = ap.timing_sample(items, k, seed=seed)
            run["timing_sample"] = {"k": k, "seed": seed, "jobs": 1,
                                    "scenarios": {sc.scenario_id: ap.phase_of(sc) for sc in sample}}
            ts = out["timing_sample"]
            record(ap.run_pass(sample, cfg=cfg, jobs=1, traj_dir=ts / "trajectories", parts=ts / "parts",
                               label="timing sample", rerun_errors=rerun_errors, progress_every=every))
            ap.merge_rows(sample, ap.read_records(ts / "parts")).to_csv(out["timing_sample.csv"], index=False)
            print(f"{len(sample)} timing-sample rows -> {out['timing_sample.csv']} (timing only, not results)")
        record(ap.run_pass(items, cfg=cfg, jobs=jobs, traj_dir=out["trajectories"], parts=out["parts"],
                           label="agent", rerun_errors=rerun_errors, progress_every=every))
    except (ap.PassStopped, KeyboardInterrupt) as exc:
        if getattr(exc, "info", None):
            run["passes"].append(exc.info)
        run["stopped"] = str(exc) or type(exc).__name__
        save_run()
        raise
    recs = ap.read_records(out["parts"])
    missing = [sc.scenario_id for sc in items if sc.scenario_id not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} scenarios have no row (first {missing[:3]}); rerun with --resume")
    rows = ap.merge_rows(items, recs)
    rows.to_csv(out["agent.csv"], index=False)
    passes = run["passes"]
    run.update(agent_jobs=jobs, scenarios=len(items), only=only is not None, headline=ap.HEADLINE,
               latency_measured_under=f"{jobs} concurrent episode{'s' if jobs != 1 else ''} (agent_jobs) in the "
                                      "full pass; other users of the endpoint are not controlled",
               agent_wall_s=round(sum(p["wall_s"] for p in passes if p["label"] == "agent"), 1),
               timing_sample_wall_s=round(sum(p["wall_s"] for p in passes if p["label"] == "timing sample"), 1),
               retry_policy={"tries": ap.TRIES, "backoff_s": list(cfg.get("backoff", ap.BACKOFF_S)),
                             "request_timeout_s": ap.REQUEST_TIMEOUT_S,
                             "retried": "429, 5xx, timeouts, dropped connections; 401/403 stop the pass"},
               rows_status={str(st): int(n) for st, n in rows["status"].value_counts().items()})
    run.pop("stopped", None)
    save_run()
    print(f"{len(rows)} agent rows -> {out['agent.csv']}")
    return out["agent.csv"]


def agent_timing(d: Path):
    """(per-run latency table, paired table, per-scenario pairs) from timing_sample.csv (isolated, one
    job) and agent.csv (the pass, --jobs): latency mean, median and p90 with tokens and tool calls per
    run; on the sampled scenarios the concurrent minus isolated latency (mean, 95% t interval, ratio
    of means) and whether the decision was the same (two runs of one scenario: also a small
    run-to-run check). None when either file is missing."""
    ts, ag = d / "timing_sample.csv", d / "agent.csv"
    if not (ts.exists() and ag.exists()):
        return None
    iso, con = pd.read_csv(ts), pd.read_csv(ag)
    iso, con = iso[iso["policy"] == "agent"], con[con["policy"] == "agent"]
    for g in (iso, con):
        if "decision" not in g:
            g["decision"] = None

    def describe(g, name):
        lat = pd.to_numeric(g["latency_s"], errors="coerce").dropna()

        def mean(col):
            return float(pd.to_numeric(g[col], errors="coerce").mean()) if col in g else np.nan
        return {"run": name, "episodes": len(g), "latency_mean_s": float(lat.mean()) if len(lat) else np.nan,
                "latency_median_s": float(lat.median()) if len(lat) else np.nan,
                "latency_p90_s": float(lat.quantile(0.9)) if len(lat) else np.nan,
                "input_tokens_mean": mean("input_tokens"), "output_tokens_mean": mean("output_tokens"),
                "tool_calls_mean": mean("tool_calls"), "errors": int((g["status"] == "error").sum()),
                "submitted": int(g["decision"].notna().sum()),
                "api_retries": int(pd.to_numeric(g["api_retries"], errors="coerce").fillna(0).sum())
                if "api_retries" in g else 0}
    pairs = iso[["scenario_id", "latency_s", "decision"]].merge(
        con[["scenario_id", "latency_s", "decision"]], on="scenario_id", suffixes=("_isolated", "_concurrent"))
    pairs["latency_diff_s"] = pairs["latency_s_concurrent"] - pairs["latency_s_isolated"]
    pairs["same_decision"] = (pairs["decision_isolated"].fillna("none").astype(str)
                              == pairs["decision_concurrent"].fillna("none").astype(str))
    runs = pd.DataFrame([describe(iso, "isolated (timing sample, 1 job)"),
                         describe(con[con["scenario_id"].isin(pairs["scenario_id"])], "concurrent, same scenarios"),
                         describe(con, "concurrent, all scenarios")])
    diff = pairs["latency_diff_s"].dropna()
    n = len(diff)
    half = np.nan
    if n > 1:
        from scipy import stats
        half = float(stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n))
    mean_iso, mean_con = pairs["latency_s_isolated"].mean(), pairs["latency_s_concurrent"].mean()
    paired = pd.DataFrame([{
        "pairs": n, "mean_diff_s": float(diff.mean()) if n else np.nan,
        "ci95_low_s": float(diff.mean() - half) if n > 1 else np.nan,
        "ci95_high_s": float(diff.mean() + half) if n > 1 else np.nan,
        "ratio_of_means": float(mean_con / mean_iso) if n and mean_iso else np.nan,
        "median_ratio": float((pairs["latency_s_concurrent"] / pairs["latency_s_isolated"]).median()) if n else np.nan,
        "same_decision": int(pairs["same_decision"].sum()), "of": len(pairs)}])
    return runs, paired, pairs


# --- stage 5: tables --------------------------------------------------------------------------------------
def era_of(race_id: str) -> str:
    season = int(str(race_id)[:4])
    return next((name for name, lo, hi in ERAS if lo <= season <= hi), str(season))


def tail_table(rows: pd.DataFrame, by=("policy",)) -> pd.DataFrame:
    """Per group: scored rows, mean, median, p90 and worst regret, and the share over each TAIL_S."""
    ok = rows[rows["status"] == "ok"].copy()
    ok["regret_s"] = pd.to_numeric(ok["regret_s"], errors="coerce")
    out = []
    for key, g in ok.groupby(list(by), sort=True):
        r = g["regret_s"].dropna()
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(scored=len(r), mean_s=float(r.mean()), median_s=float(r.median()),
                   p90_s=float(r.quantile(0.9)), worst_s=float(r.max()))
        row.update({f"share_over_{int(t)}s": float((r > t).mean()) for t in TAIL_S})
        out.append(row)
    return pd.DataFrame(out)


def status_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Counts of non-ok rows per policy, status and reason."""
    bad = rows[rows["status"] != "ok"]
    if not len(bad):
        return pd.DataFrame(columns=["policy", "status", "reason", "rows"])
    reason = bad["reason"].fillna("").astype(str).str.slice(0, 160) if "reason" in bad else ""
    return (bad.assign(reason=reason).groupby(["policy", "status", "reason"]).size()
            .rename("rows").reset_index().sort_values(["policy", "rows"], ascending=[True, False]))


def stage_tables(args) -> Path:
    """The combined rows, per-race and pooled summaries, agent-minus-X paired tables with
    (race, driver) cluster intervals for every comparator (belief_optimizer stands in for the agent
    when there are no agent rows, said so in the file name), the chart CSV, the tail table (by era
    too for a set with eras), status counts, position metrics when traffic is on; provenance.json."""
    from .compare import metric_table, paired_table
    from .score import summarize
    d = out_dir(args)
    spec = set_spec(args.set, args.sets)
    parts = [pd.read_csv(p) for p in (d / "baselines.csv", d / "agent.csv") if p.exists()]
    if not parts:
        raise SystemExit(f"no rows under {d}: run the baselines (and agent) stages first")
    rows = pd.concat(parts, ignore_index=True)
    t = d / "tables"
    t.mkdir(parents=True, exist_ok=True)
    rows["era"] = rows["race_id"].map(era_of)
    rows.to_csv(t / "combined.csv", index=False)
    summarize(rows).to_csv(t / "summary.csv", index=False)
    reference = "agent" if "agent" in set(rows["policy"]) else "belief_optimizer"
    if reference in set(rows["policy"]):
        paired_table(rows, reference, cluster="driver").to_csv(t / f"paired_{reference}_minus_x.csv", index=False)
    metric_table(rows, cluster="driver").to_csv(t / "chart_regret_s.csv", index=False)
    tail_table(rows).to_csv(t / "tail.csv", index=False)
    if spec.get("eras"):
        tail_table(rows, by=("era", "policy")).to_csv(t / "tail_by_era.csv", index=False)
        pd.concat([metric_table(g, cluster="driver").assign(era=era) for era, g in rows.groupby("era")],
                  ignore_index=True).to_csv(t / "chart_regret_s_by_era.csv", index=False)
        for era, g in rows.groupby("era"):
            summarize(g).to_csv(t / f"summary_era_{era}.csv", index=False)
            if reference in set(g["policy"]):
                paired_table(g, reference, cluster="driver").to_csv(t / f"paired_{reference}_minus_x_era_{era}.csv",
                                                                     index=False)
    status_table(rows).to_csv(t / "status.csv", index=False)
    if "position_regret" in rows and pd.to_numeric(rows["position_regret"], errors="coerce").notna().any():
        metric_table(rows, metric="position_regret", cluster="driver").to_csv(t / "chart_position_regret.csv",
                                                                               index=False)
        if reference in set(rows["policy"]):
            paired_table(rows, reference, metric="position_regret", cluster="driver") \
                .to_csv(t / f"paired_{reference}_minus_x_position.csv", index=False)
    timing = agent_timing(d)
    if timing is not None:
        for name, table in zip(("agent_latency.csv", "agent_latency_paired.csv", "agent_latency_pairs.csv"), timing):
            table.to_csv(t / name, index=False)
    write_provenance(args, rows)
    print(f"tables -> {t}")
    return t


# --- provenance ----------------------------------------------------------------------------------------------
def _git(*cmd):
    try:
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12] if path and Path(path).is_file() else None


HARNESS = ROOT / "tests" / "test_future_blindness.py"


def future_blindness(run: bool = False) -> dict:
    """The future-blindness harness on this commit: the leaks tests/future_blindness.py lists as
    fixed (FIXED_LEAKS, regression checks) and as still known (KNOWN_LEAKS, pinned xfails), and with
    `run` the default part's outcome counts (python -m pytest tests/test_future_blindness.py, about
    16 s)."""
    import importlib
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        fb = importlib.import_module("future_blindness")
        out = {"fixed_leaks": sorted(fb.FIXED_LEAKS), "known_leaks": sorted(fb.KNOWN_LEAKS)}
    except Exception as exc:   # noqa: BLE001  recorded, never fatal
        out = {"error": f"{type(exc).__name__}: {exc}"}
    if run:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HARNESS)],
                           cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        counts = {k: int(n) for n, k in __import__("re").findall(r"(\d+) (passed|failed|xfailed|xpassed|error|"
                                                                  r"errors|skipped|deselected)", last)}
        out.update(default_part=counts, default_part_summary=last, returncode=r.returncode)
    return out


def as_of_tables(races) -> dict:
    """Per policy-side artifact (src/sim/provenance.py): its kind, file hashes, every race it
    records, and per race of the set the races pricing that race reads."""
    from ..sim import provenance
    out = {}
    for name, art in provenance.POLICY_SIDE.items():
        out[name] = {"kind": art.kind, "files": provenance.file_hashes(name), "recorded": provenance.recorded(name),
                     "reads": {r: provenance.races_for(name, r) for r in races}}
    return out


def _timing_doc(d: Path):
    """agent_timing's tables as records, and which latency is the headline (the isolated sample's)."""
    timing = agent_timing(d)
    if timing is None:
        return None
    runs, paired, _ = timing
    return {"headline": "isolated (timing sample, 1 job)", "runs": runs.to_dict("records"),
            "paired": paired.to_dict("records")[0]}


def default_world():
    """The world `--world-config default` scores with (score.race_simulators on the default profile)."""
    from ..sim import SimConfig
    from ..sim.config import world_default
    from .score import default_world_config
    return default_world_config(world_default(SimConfig()))


def config_choices(c, role: str) -> dict:
    """Each simulator choice the fidelity scorecard decides (docs/eval.md "Simulator fidelity"), named
    as it stands in config `c`, so provenance.json says which world and belief a run used without
    reading the hash. `role` "world" adds the judge-only choices (the tyre-life edge)."""
    deg = c.degradation.params or {}
    out = {"wear_fit": c.degradation.name,
           "driver_effects": deg.get("driver_effects", "off"),
           "hinge": c.degradation.name == "linear_fuel_hinge",
           "wear_lower_bound": deg.get("wear_lower_bound"),
           "sd_scale": deg.get("sd_scale"),
           "wear_cliff": c.wear_cliff.enabled,
           "stint_limit_margin_laps (R8)": c.stint_limit_margin_laps,
           "events": c.events.name, **({"events_params": dict(c.events.params)} if c.events.params else {}),
           "traffic": c.traffic.name if c.traffic.name == "none" else
           f"{c.traffic.name} ({c.traffic.params.get('source', 'recorded')})",
           "lap_noise": c.lap_noise, "plan_on": c.plan_on, "fit_window": c.fit_window}
    if c.fit_window == "full":
        out["tyre_life_edge"] = c.wear_cliff.world_edge
    else:   # a windowed fit places the edge as of the snapshot (sim/cliff.py), whatever world_edge says
        out["tyre_life_edge"] = "as_of_snapshot (history extended by the tyre ages seen so far)"
    if role != "world":
        out.update(calibrator=c.calibrator.name, fuel_anchor=c.fuel_anchor.model_dump(mode="json"),
                   pit_loss=c.pit_loss.source)
    return out


def write_provenance(args, rows: pd.DataFrame | None = None) -> Path:
    """provenance.json beside the outputs: commit, tag, the world and belief configs with their
    file sha256 and config_hash, SIM_CODE_VERSION, the set digest, the model id and loop settings,
    the stamps the rows carry, and every command run here."""
    from ..sim import SimConfig, calibration
    from .scenarios import ScenarioSet
    d = out_dir(args)
    s = ScenarioSet.load(d / "scenarios.json") if (d / "scenarios.json").exists() else None
    belief, world = belief_path(args.belief_config), world_path(args.world_config)

    def cfg(path, role):
        c = SimConfig.from_file(path) if path else default_world()
        return {"path": path or "default (score.default_world_config(config.world_default(SimConfig())))",
                "file_sha256": _file_sha(path), "config_hash": calibration.config_hash(c),
                "choices": config_choices(c, role)}
    stamps = {}
    if rows is not None:
        for col in ("world_config_hash", "belief_config_hash", "sim_code_version", "scenario_digest", "belief_window"):
            if col in rows:
                stamps[col] = sorted(map(str, rows[col].dropna().unique()))
    prof = _profile_for(args.profile)
    doc = {"commit": _git("rev-parse", "HEAD"), "tag": _git("describe", "--tags", "--exact-match"),
           "dirty": bool(_git("status", "--porcelain")), "set": args.set,
           "set_digest": s.digest if s else None, "set_kind": s.kind if s else None,
           "sampling": s.sampling if s else None, "world": cfg(world, "world"), "belief": cfg(belief, "belief"),
           "sim_code_version": calibration.SIM_CODE_VERSION, "profile": prof.model_dump(mode="json"),
           # the backend and model id only: never a key, a token or an endpoint
           "model": {k: os.environ.get(k) for k in MODEL_ENV},
           "loop": {"max_steps": prof.agent.max_steps,
                    **_loop_env()},
           "row_stamps": stamps,
           "builder": s.builder if s else None,
           "future_blindness": future_blindness(run=getattr(args, "harness", False)),
           "as_of_tables": as_of_tables(_races(s) if s else []),
           "agent_run": json.loads((d / "agent_run.json").read_text()) if (d / "agent_run.json").exists() else None,
           "agent_timing": _timing_doc(d),
           "commands": (d / "commands.log").read_text().splitlines() if (d / "commands.log").exists() else []}
    path = d / "provenance.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return path


# --- smoke: a handful of scenarios through every stage, then the lint ------------------------------------
SMOKE_SET = "smoke"


def stage_smoke(args) -> Path:
    """The smoke run (src/eval/smoke.py): the `smoke` set's races (development only, checked against
    data/eval_set.json; --final is refused), sampled by the leak-free builder, --n scenarios picked
    (lap 0, under SC, late, mid-race, the rest spread over races), then references, baselines, the
    agent with --jobs workers and the tables for just those; then the lint. Prints the summary and one
    line per finding (also smoke.txt) and exits 1 on any finding."""
    from types import SimpleNamespace
    from . import smoke
    from .scenarios import build_set
    if args.final:
        raise SystemExit("smoke never builds a report set: --final is refused")
    spec = set_spec(SMOKE_SET, args.sets)
    smoke.check_development(spec, spec["races"])
    d = out_dir(args)
    if (d / "scenarios.json").exists():
        raise SystemExit(f"{d} has a smoke run already; each run takes a new --out-dir (the default is new each time)")
    sub = SimpleNamespace(**{**vars(args), "set": SMOKE_SET, "out_dir": str(d), "final": False, "resume": False,
                             "force": False, "only": None, "timing_sample": 0, "harness": False,
                             "progress_every": max(args.n // 2, 1)})
    timings = {}

    def timed(name, fn, *a):
        t = time.perf_counter()
        out = fn(*a)
        timings[name] = time.perf_counter() - t
        return out

    def scenarios():
        prof = _profile_for(args.profile)
        pool = build_set(spec["races"], "smoke_pool", profile=prof, rule="as_of", seed=spec["seed"],
                         development=True, max_per_race=spec["per_race"])
        chosen = smoke.pick(pool.items, args.n, seed=spec["seed"])
        s = pool.model_copy(update={"version": SMOKE_SET, "items": chosen, "skipped": []})
        s.save(d / "scenarios.json")
        print(f"smoke: {len(chosen)} of {len(pool.items)} sampled scenarios: "
              + ", ".join(f"{x.scenario_id} ({x.snapshot.track_status}, {x.snapshot.race.total_laps - x.lap} left)"
                          for x in chosen), flush=True)
    timed("scenarios", scenarios)
    timed("references", stage_references, sub)
    timed("baselines", stage_baselines, sub)
    timed("agent", stage_agent, sub)
    timed("tables", stage_tables, sub)
    findings = timed("lint", smoke.lint, d, _profile_for(args.profile))
    lines = smoke.summary(d, findings, timings, sub.jobs) + [f"FINDING {f}" for f in findings]
    (d / "smoke.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if findings:
        raise SystemExit(1)
    return d


# --- command line ------------------------------------------------------------------------------------------
STAGES = {"scenarios": stage_scenarios, "references": stage_references, "baselines": stage_baselines,
          "agent": stage_agent, "tables": stage_tables, "smoke": stage_smoke}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.eval.rerun", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--set", help=f"a set of {SETS_FILE.relative_to(ROOT)}: dev20, test9 (every stage but smoke, "
                                  "which runs set smoke)")
    ap.add_argument("--sets", default=str(SETS_FILE), help="the set definitions file")
    ap.add_argument("--out-dir", help="default reports/eval/rerun/<set>")
    ap.add_argument("--world-config", default="default", help="SimConfig file for the reference and the scoring, or "
                                                               "'default' (the race's own SimConfig)")
    ap.add_argument("--belief-config", default="default", help="SimConfig file the policies plan on, or 'default' "
                                                                "(configs/sim/belief.yaml)")
    ap.add_argument("--rule", choices=RULES, default="as_of", help="scenarios: the leak-free sampler's rule")
    ap.add_argument("--final", action="store_true",
                    help="scenarios: required to build a report set (development false, e.g. test9), which loads the "
                         "held-out races; pass it once, on the owner's word")
    ap.add_argument("--jobs", type=int, default=None, help="default 1 (smoke: 4). references, baselines: worker processes (at most 4 on a "
                                                         "shared machine), one race each, BLAS at one thread; agent: "
                                                         "worker processes running episodes concurrently, one "
                                                         "scenario at a time each")
    ap.add_argument("--resume", action="store_true", help="agent: finish a started pass, skipping every scenario "
                                                           "with a row and a trajectory")
    ap.add_argument("--rerun-errors", action="store_true", help="agent, with --resume: also rerun error rows")
    ap.add_argument("--force", action="store_true", help="agent: start over, deleting the earlier pass's outputs "
                                                          "(with --resume: merge a finished pass again)")
    ap.add_argument("--timing-sample", type=int, default=0, metavar="K",
                    help="agent: first run K scenarios (seeded, stratified by race and early/mid/late laps) "
                         "sequentially into timing_sample.csv, the decision-time figure; timing only, the pass "
                         "runs them again")
    ap.add_argument("--timing-seed", type=int, default=0, help="agent: the timing sample's seed")
    ap.add_argument("--only", help="agent: only these scenario ids (a file, one per line, or comma-separated)")
    ap.add_argument("--progress-every", type=int, default=10, help="agent: a progress line every N scenarios")
    ap.add_argument("--n", type=int, default=8, help="smoke: the number of scenarios")
    ap.add_argument("--backend", help=argparse.SUPPRESS)   # agent: a "module:function" backend factory (tests)
    ap.add_argument("--profile", help="compute profile (default $SIM_PROFILE or local)")
    ap.add_argument("--harness", action="store_true",
                    help="tables: also run the future-blindness harness's default part (about 16 s) and record its "
                         "counts in provenance.json")
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = parser()
    args = ap.parse_args(argv)
    if args.stage == "smoke":
        args.set = SMOKE_SET
        args.out_dir = args.out_dir or str(OUT_ROOT / SMOKE_SET / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    elif not args.set:
        ap.error("--set is required")
    if args.jobs is None:
        args.jobs = 4 if args.stage == "smoke" else 1
    _log_command(out_dir(args), argv)
    started = time.perf_counter()
    STAGES[args.stage](args)
    print(f"stage {args.stage}: {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
