"""The evaluation rerun under the corrected protocol, one command per stage (docs/eval.md
"Rerunning the evaluation").

    python -m src.eval.rerun scenarios  --set dev20 [--rule as_of|stratified] [--final]
    python -m src.eval.rerun references --set dev20 --world-config default [--jobs 4]
    python -m src.eval.rerun baselines  --set dev20 --world-config default --belief-config default [--jobs 4]
    python -m src.eval.rerun agent      --set dev20 --world-config default --belief-config default
    python -m src.eval.rerun tables     --set dev20

Every stage reads and writes under one directory, --out-dir (default reports/eval/rerun/<set>):

    scenarios.json            the set (stage scenarios; never rebuilt over an existing file)
    references/               the world's references, the cache score reads (stage references)
    parts/references/<race>.pkl, parts/baselines/<race>.csv   per-race results: a rerun skips a
                              race whose part exists, so an interrupted stage resumes
    baselines.csv             every non-LLM policy's rows (stage baselines)
    agent.csv, trajectories/  the live agent (stage agent; refuses to overwrite either)
    tables/                   combined rows, summaries, paired tables, chart and tail tables (stage tables)
    provenance.json           commit, tag, configs and hashes, SIM_CODE_VERSION, set digest, model
                              and loop settings, and every command run here (commands.log)

The sets are defined in configs/eval/rerun_sets.yaml. A report set (development false: test9,
the held-out races) is built only with --final, on the owner's word: building it loads the
held-out races. --world-config / --belief-config take a SimConfig file or "default" (the world:
the race's own SimConfig; the belief: configs/sim/belief.yaml). The parallel stages run --jobs
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
def stage_agent(args) -> Path:
    """One sequential pass of the live agent (LLM_BACKEND), its tools on the belief config
    (AGENT_BELIEF_CONFIG), trajectories to trajectories/; refuses to overwrite agent.csv or a
    non-empty trajectories/."""
    from .policies import AgentPolicy
    from .reference import ReferenceStore
    from .scenarios import ScenarioSet
    from .score import race_simulators, score_set
    d = out_dir(args)
    traj, path = d / "trajectories", d / "agent.csv"
    if path.exists() or (traj.exists() and any(traj.iterdir())):
        raise SystemExit(f"{path} or {traj} exists: an agent pass is never overwritten; use a new --out-dir")
    belief = belief_path(args.belief_config)
    os.environ["AGENT_BELIEF_CONFIG"] = belief
    prof = _profile_for(args.profile)
    s = ScenarioSet.load(d / "scenarios.json")
    policy = AgentPolicy(set_name=".", profile=prof, log_dir=traj)
    rows = score_set(s, race_simulators(prof, _sim_config(belief), _sim_config(world_path(args.world_config))),
                     {"agent": policy}, n_samples=prof.sim.n_samples, max_stops=prof.sim.max_stops,
                     references=ReferenceStore(d / "references", require=True))
    rows.to_csv(path, index=False)
    print(f"{len(rows)} agent rows -> {path}")
    return path


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


def write_provenance(args, rows: pd.DataFrame | None = None) -> Path:
    """provenance.json beside the outputs: commit, tag, the world and belief configs with their
    file sha256 and config_hash, SIM_CODE_VERSION, the set digest, the model id and loop settings,
    the stamps the rows carry, and every command run here."""
    from ..sim import SimConfig, calibration
    from .scenarios import ScenarioSet
    d = out_dir(args)
    s = ScenarioSet.load(d / "scenarios.json") if (d / "scenarios.json").exists() else None
    belief, world = belief_path(args.belief_config), world_path(args.world_config)

    def cfg(path):
        c = SimConfig.from_file(path) if path else SimConfig()
        return {"path": path or "default (the race's own SimConfig)", "file_sha256": _file_sha(path),
                "config_hash": calibration.config_hash(c)}
    stamps = {}
    if rows is not None:
        for col in ("world_config_hash", "belief_config_hash", "sim_code_version", "scenario_digest", "belief_window"):
            if col in rows:
                stamps[col] = sorted(map(str, rows[col].dropna().unique()))
    prof = _profile_for(args.profile)
    doc = {"commit": _git("rev-parse", "HEAD"), "tag": _git("describe", "--tags", "--exact-match"),
           "dirty": bool(_git("status", "--porcelain")), "set": args.set,
           "set_digest": s.digest if s else None, "set_kind": s.kind if s else None,
           "sampling": s.sampling if s else None, "world": cfg(world), "belief": cfg(belief),
           "sim_code_version": calibration.SIM_CODE_VERSION, "profile": prof.model_dump(mode="json"),
           # the backend and model id only: never a key, a token or an endpoint
           "model": {k: os.environ.get(k) for k in MODEL_ENV},
           "loop": {"max_steps": prof.agent.max_steps,
                    **{k: os.environ.get(k) for k in ("AGENT_MAX_TOOL_CALLS", "AGENT_SUBMIT_GRACE_STEPS",
                                                       "AGENT_REPEAT_MODE", "AGENT_TURN_STATUS",
                                                       "AGENT_SHOW_IDENTITY", "AGENT_GAP_YARDSTICK",
                                                       "AGENT_TURN_REPEAT")}},
           "row_stamps": stamps,
           "builder": s.builder if s else None,
           "future_blindness": future_blindness(run=getattr(args, "harness", False)),
           "as_of_tables": as_of_tables(_races(s) if s else []),
           "commands": (d / "commands.log").read_text().splitlines() if (d / "commands.log").exists() else []}
    path = d / "provenance.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return path


# --- command line ------------------------------------------------------------------------------------------
STAGES = {"scenarios": stage_scenarios, "references": stage_references, "baselines": stage_baselines,
          "agent": stage_agent, "tables": stage_tables}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.eval.rerun", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--set", required=True, help=f"a set of {SETS_FILE.relative_to(ROOT)}: dev20, test9")
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
    ap.add_argument("--jobs", type=int, default=1, help="references, baselines: worker processes (at most 4 on a "
                                                         "shared machine), one race each, BLAS at one thread")
    ap.add_argument("--profile", help="compute profile (default $SIM_PROFILE or local)")
    ap.add_argument("--harness", action="store_true",
                    help="tables: also run the future-blindness harness's default part (about 16 s) and record its "
                         "counts in provenance.json")
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    _log_command(out_dir(args), argv)
    started = time.perf_counter()
    STAGES[args.stage](args)
    print(f"stage {args.stage}: {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
