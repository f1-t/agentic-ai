"""One command for the real calibration runs (SPEC 0021). See docs/runbook_real_runs.md.

    python -m src.sim.calibrate real-run --profile {local,sol} [--races auto|<ids>] [--out DIR]

Stages, in order, over the races (auto, the default: every race in the FastF1 cache,
cached_races, less the races data/eval_set.json holds out, each left out with a note; a
held-out race given by id is refused unless --allow-held-out, cross_race.guard_held_out):
  eligibility  calibration.eligibility per race (cross_race.eligibility_records), printed
               as a table and written to real_run/eligibility.json.
  cross_race   the cross-race pass and leave-one-race-out per arm (SPEC 0016, cross_race.run)
               on the eligible races, under cross_race/.
  event_rates  SC and VSC rates by circuit and era (SPEC 0011b, event_rates.report) from
               data/derived/neutralised_periods.csv, under event_rates/; without that file,
               from the cross-race factors.csv when it holds MIN_HISTORY_RACES eligible races
               (source "cached races only (n)").
  fuel_prior   the race-pace prior for the fuel term (SPEC 0020, fuel_prior.run) on the
               eligible races, under fuel_prior/, when MIN_HISTORY_RACES are eligible.
  references   `python -m src.eval reference --check`, then every stale cached reference
               rebuilt at the current code with its own inputs (rebuild_reference).
  report       the cross-race report (cross_race.report): factors.csv, summary.md and the
               tables beside them, once the cross-race pass has finished.

OUT defaults to reports/calibration, so the stages write where their own commands do; the
manifest, the eligibility table and any Slurm scripts go in OUT/real_run/.

Every stage runs, or skips with a printed reason when its input is missing. Resumable: a
stage whose inputs, code version (sim_code_version and git commit) and profile hash are
those of its last completed entry in the manifest, with its outputs present, is not rerun
(status "current"); the cross-race pass also resumes per unit (jobs.run_task). The manifest
(OUT/real_run/manifest.json, rewritten after every stage) has one entry per stage: stage,
status (ran, current, skipped, pending or failed), reason, inputs (each with `present`),
outputs, code version and wall time.

With a profile whose scheduler is local the cross-race units run in-process on its
parallel.n_jobs workers. With scheduler slurm (the sol profile) they are not run here: SlurmEmitter writes one array
script per phase from the committed templates (slurm/g_race.sbatch for the per-race units,
slurm/h_fold.sbatch for the per-fold units), prints the sbatch commands and leaves the pass
pending. Nothing here submits anything. Rerun the same command once the arrays finish: the
finished units are skipped and the next phase (the prior-sd fix, then the tau grid) is
emitted the same way, until the pass completes and the report runs.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import calibration, cross_race, event_rates, fuel_prior, jobs
from .compute import load_profile
from .race_ids import canonical_race_id, make_race_id_from_schedule, parse_race_id

ROOT = cross_race.ROOT
REPORTS_DIR = ROOT / "reports" / "calibration"
RUN_SUBDIR = "real_run"
SLURM_DIR = ROOT / "slurm"
SCENARIO_DIR = ROOT / "data" / "scenarios"
SIM_CONFIG_DIR = ROOT / "configs" / "sim"
STAGES = ("eligibility", "cross_race", "event_rates", "fuel_prior", "references", "report")
RAN, CURRENT, SKIPPED, PENDING, FAILED = "ran", "current", "skipped", cross_race.PENDING, "failed"
MIN_HISTORY_RACES = 8       # SPEC 0021: the fuel prior, and the event rates from factors.csv, need this many
ARRAY_MAX = 1000            # tasks per sbatch line; a longer phase gets several, with an offset
ARRAY_CONCURRENT = 50       # %K on the array: tasks running at once, as slurm/README.md
TEMPLATES = {"cross_race": "g_race.sbatch", "h_fold": "h_fold.sbatch"}


# --- races ----------------------------------------------------------------------------
def cached_races(cache_dir=None) -> tuple[list[str], list[str]]:
    """(canonical ids, notes) of every race in the FastF1 cache: an event directory
    <cache>/<season>/<date>_<Event_Name>/ holding a *_Race session directory, named by
    race_ids.make_race_id_from_schedule from the cached season schedule (data.season_schedule),
    offline. Sorted by season and round; an event the schedule does not name, a season
    without a cached schedule, or a sprint weekend (no FP2 or FP3 for the practice prior,
    calibrate.PRACTICE_SESSIONS; sprint support is its own change) is left out with a note."""
    from .data import CACHE_DIR, season_schedule
    cache = Path(cache_dir or CACHE_DIR)
    ids, notes = [], []
    if not cache.is_dir():
        return [], [f"no cache directory {cache}"]
    for season_dir in sorted(p for p in cache.iterdir() if p.is_dir() and p.name.isdigit()):
        events = [d for d in sorted(season_dir.iterdir())
                  if d.is_dir() and any(s.is_dir() and s.name.endswith("_Race") for s in d.iterdir())]
        if not events:
            continue
        season = int(season_dir.name)
        schedule = season_schedule(season, str(cache))
        if schedule is None:
            notes.append(f"{season_dir.name}: no cached event schedule, its {len(events)} race(s) left out")
            continue
        by_name = {str(r["EventName"]).replace(" ", "_"): r for _, r in schedule.iterrows()}
        for d in events:
            name = d.name.split("_", 1)[1] if "_" in d.name else d.name
            if name in by_name and any(s.is_dir() and s.name.endswith("_Sprint") for s in d.iterdir()):
                notes.append(f"{season_dir.name}/{d.name}: sprint weekend (no FP2/FP3 practice prior), left out")
            elif name in by_name:
                ids.append(make_race_id_from_schedule(by_name[name], schedule, season))
            else:
                notes.append(f"{season_dir.name}/{d.name}: not in the {season} event schedule, left out")
    return sorted(set(ids), key=lambda r: parse_race_id(r)[:2]), notes


# --- manifest -----------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if ROOT in p.parents else str(p)


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def file_input(path) -> dict:
    """An input file: its path, whether it is there, and its sha256 (12 hex) when it is."""
    path = Path(path) if path else None
    present = bool(path) and path.is_file()
    return {"path": _rel(path) if path else None, "present": present, "sha256": _sha12(path) if present else None}


def value_input(value) -> dict:
    """An input value; present when it is not empty."""
    return {"value": value, "present": bool(value)}


def code_version() -> dict:
    """sim_code_version (the physics) and the git commit (everything else), None outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"sim_code_version": calibration.SIM_CODE_VERSION, "git_commit": commit}


class RealRun:
    """One real-run: the stages' shared state and the manifest they write to."""

    def __init__(self, races, profile, out=REPORTS_DIR, *, profile_ref=None, race_source="given", race_notes=(),
                 periods=event_rates.NEUTRALISED_PERIODS, race_features=cross_race.RACE_FEATURES,
                 reference_dir=None, tau_multipliers=cross_race.TAU_MULTIPLIERS, tau_grid=True, n_boot=None,
                 slurm_dir=SLURM_DIR, log=print, eval_set=cross_race.EVAL_SET, allow_held_out=False,
                 held_out_skipped=()):
        from ..eval.reference import REFERENCE_DIR
        self.races = [canonical_race_id(r) for r in races]
        # a held-out race (data/eval_set.json) is refused here, before its eligibility loads it
        cross_race.guard_held_out(self.races, eval_set, allow_held_out)
        self.eval_set, self.allow_held_out = eval_set, allow_held_out
        self.held_out_skipped = [canonical_race_id(r) for r in held_out_skipped]
        self.profile, self.out, self.log = profile, Path(out), log
        self.profile_ref = profile_ref or profile.name
        self.race_source, self.race_notes = race_source, list(race_notes)
        self.periods, self.race_features = Path(periods), Path(race_features)
        self.reference_dir = Path(reference_dir or REFERENCE_DIR)
        self.tau_multipliers, self.tau_grid, self.n_boot = tuple(tau_multipliers), tau_grid, n_boot
        self.slurm = profile.scheduler == "slurm"
        self.slurm_dir = Path(slurm_dir)
        self.run_dir = self.out / RUN_SUBDIR
        self.manifest_path = self.run_dir / "manifest.json"
        self.code = code_version()
        self.elig: list[dict] | None = None
        self.entries: dict[str, dict] = {}
        self.previous = self._previous()
        self.started = _now()

    @property
    def eligible(self) -> list[str]:
        return [e["race_id"] for e in self.elig or [] if e["eligible"]]

    def _previous(self) -> dict:
        if not self.manifest_path.is_file():
            return {}
        try:
            return {e["stage"]: e for e in json.loads(self.manifest_path.read_text()).get("stages", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}

    def fingerprint(self, inputs: dict) -> str:
        payload = json.dumps([inputs, self.code, self.profile.name, self.profile.hash], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def resume(self, stage: str, inputs: dict) -> bool:
        """Record `stage` as current and return True when its last completed entry has these
        inputs, this code and profile, and every output still exists."""
        prev = self.previous.get(stage)
        if not (prev and prev.get("status") in (RAN, CURRENT) and prev.get("fingerprint") == self.fingerprint(inputs)
                and all((ROOT / p if not Path(p).is_absolute() else Path(p)).exists() for p in prev.get("outputs", []))):
            return False
        ran_at = prev.get("ran_at") or prev.get("finished")
        self._record(stage, inputs, CURRENT, f"inputs, code and profile unchanged since it ran at {ran_at}; "
                     "outputs present", prev.get("outputs", []), time.perf_counter(), prev.get("detail"),
                     ran_at=ran_at, ran_wall_time_s=prev.get("ran_wall_time_s"))
        return True

    def _record(self, stage, inputs, status, reason=None, outputs=(), t0=None, detail=None, ran_at=None,
                ran_wall_time_s=None):
        """The stage's manifest entry. wall_time_s is this invocation's; ran_at and
        ran_wall_time_s are those of the run that made the outputs (kept by a current stage)."""
        finished = _now()
        wall = round(time.perf_counter() - t0, 2) if t0 is not None else 0.0
        entry = {"stage": stage, "status": status, "reason": reason, "inputs": inputs,
                 "outputs": [_rel(p) for p in outputs], "code_version": self.code,
                 "profile": self.profile.name, "profile_hash": self.profile.hash,
                 "fingerprint": self.fingerprint(inputs), "finished": finished,
                 "ran_at": ran_at or (finished if status == RAN else None),
                 "wall_time_s": wall, "ran_wall_time_s": wall if status == RAN else ran_wall_time_s,
                 "detail": detail or {}}
        self.entries[stage] = entry
        word = {RAN: "ran", CURRENT: "current, not rerun", SKIPPED: "skipped", PENDING: "pending",
                FAILED: "failed"}[status]
        self.log(f"[{stage}] {word}" + (f": {reason}" if reason else "") + f" ({entry['wall_time_s']} s)")
        self.write_manifest()
        return entry

    def done(self, stage, inputs, outputs, t0, detail=None, reason=None):
        return self._record(stage, inputs, RAN, reason, outputs, t0, detail)

    def skip(self, stage, inputs, reason):
        return self._record(stage, inputs, SKIPPED, reason)

    def write_manifest(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "command": "python -m src.sim.calibrate real-run", "started": self.started, "updated": _now(),
            "profile": self.profile.name, "profile_ref": self.profile_ref, "profile_hash": self.profile.hash,
            "mode": "slurm (scripts emitted, nothing submitted)" if self.slurm
            else f"in-process, n_jobs {self.profile.parallel.n_jobs}",
            "out": _rel(self.out), "races": self.races, "race_source": self.race_source,
            "race_notes": self.race_notes, "code_version": self.code,
            "complete": all(s in self.entries for s in STAGES),
            "stages": [self.entries[s] for s in STAGES if s in self.entries]}
        tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=1, default=str))
        os.replace(tmp, self.manifest_path)


# --- stages -------------------------------------------------------------------------------
def _eligibility_table(elig) -> str:
    return cross_race.md_table(pd.DataFrame([
        {"race": e["race_id"], "eligible": e["eligible"], "drivers": e["n_finishers"],
         "reasons": "; ".join(e["reasons"]) or "-", "notes": "; ".join(e.get("notes", [])) or "-"} for e in elig]))


def stage_eligibility(run: RealRun):
    stage, path = "eligibility", run.run_dir / "eligibility.json"
    inputs = {"races": value_input(run.races)}
    if not run.races:
        return run.skip(stage, inputs, "no race: the FastF1 cache (cache/) holds no race session"
                        + (f" ({'; '.join(run.race_notes)})" if run.race_notes else "") + " and --races gave none")
    if run.resume(stage, inputs):
        run.elig = json.loads(path.read_text())
    else:
        t0 = time.perf_counter()
        run.elig = cross_race.eligibility_records(run.races, run.profile, run.log)
        run.run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.elig, indent=1))
        run.done(stage, inputs, [path], t0, {"races": len(run.races), "eligible": len(run.eligible)})
    run.log(f"\neligibility: {len(run.eligible)} of {len(run.elig)} race(s) eligible\n\n"
            + _eligibility_table(run.elig))


def stage_cross_race(run: RealRun):
    stage, out = "cross_race", run.out / "cross_race"
    inputs = {"eligible": value_input(run.eligible), "races": value_input(run.races),
              "tau_multipliers": value_input(list(run.tau_multipliers)), "tau_grid": value_input(run.tau_grid)}
    if run.elig is None:
        return run.skip(stage, inputs, "no eligibility table (the eligibility stage was skipped)")
    if not run.eligible:
        return run.skip(stage, inputs, "no eligible race (real_run/eligibility.json gives each race's reasons)")
    if run.resume(stage, inputs):
        return
    t0 = time.perf_counter()
    emitter = SlurmEmitter(run.run_dir / "slurm", run.profile_ref, run.slurm_dir) if run.slurm else None
    record = cross_race.run(run.races, run.profile, out, run.tau_multipliers, run.tau_grid, log=run.log,
                            elig=run.elig, runner=emitter, eval_set=run.eval_set,
                            allow_held_out=run.allow_held_out, held_out_skipped=run.held_out_skipped)
    if record.get("pending"):
        run.log("\nSubmit these from f1-race-strategy-agent-local-test/ (nothing has been submitted), then rerun "
                "this command once they finish:\n\n" + "\n".join(f"  {c}" for c in emitter.commands) + "\n")
        return run._record(stage, inputs, PENDING,
                           f"{len(record['pending'])} unit(s) of phase {record['phase']} left to Slurm: submit the "
                           "printed sbatch commands, then rerun real-run", emitter.scripts, t0,
                           {"phase": record["phase"], "pending": len(record["pending"]), "sbatch": emitter.commands})
    factors = cross_race.write_factors(cross_race.race_outputs(out, record["eligible"], run.profile), out,
                                       run.race_features)
    return run.done(stage, inputs, [out / "run.json", out / "eligibility.json", factors], t0,
                    {k: record[k] for k in ("eligible", "belief_bound", "belief_choice", "s2_arm", "grid_arm")}
                    | {"grid": len(record["grid"]), "notes": record["notes"]})


EVENT_RATES_OUTPUTS = ("rates.csv", "forward_nll.csv", "loro_nll.csv", "selection.csv", "strengths.csv", "summary.md")


def stage_event_rates(run: RealRun):
    """From the neutralised periods table; without it, from the cross-race factors.csv of this
    run when that holds MIN_HISTORY_RACES eligible races; else skipped."""
    stage, out = "event_rates", run.out / "event_rates"
    factors_path = run.out / "cross_race" / "factors.csv"
    cross_done = run.entries.get("cross_race", {}).get("status") in (RAN, CURRENT)
    inputs = {"periods": file_input(run.periods), "race_features": file_input(run.race_features),
              "factors": file_input(factors_path) if cross_done else {"path": _rel(factors_path), "present": False}}
    if inputs["periods"]["present"]:
        source = None
    elif inputs["factors"]["present"]:
        factors = pd.read_csv(factors_path)
        if len(factors) < MIN_HISTORY_RACES:
            return run.skip(stage, inputs, f"no {inputs['periods']['path']} (the data team's neutralised periods "
                            f"table), and the cross-race factors.csv holds {len(factors)} eligible race(s), fewer "
                            f"than {MIN_HISTORY_RACES}")
        source = f"cached races only ({len(factors)})"
    else:
        return run.skip(stage, inputs, f"no {inputs['periods']['path']} (the data team's neutralised periods "
                        "table), and no cross-race factors.csv from this run")
    if run.resume(stage, inputs):
        return
    t0 = time.perf_counter()
    if source is None:
        stats, source = event_rates.load_stats(run.periods, None, run.race_features)
    else:
        features = pd.read_csv(run.race_features) if run.race_features.is_file() else None
        stats = event_rates.stats_from_factors(factors, features)
    kw = {} if run.n_boot is None else {"n_boot": run.n_boot}
    event_rates.report(stats, out, source, **kw)
    return run.done(stage, inputs, [out / f for f in EVENT_RATES_OUTPUTS], t0,
                    {"races": int(len(stats)), "source": source})


FUEL_OUTPUTS = ("history.csv", "predictions.csv", "f_table.csv", "selection.csv", "summary.md")


def stage_fuel_prior(run: RealRun):
    stage, out = "fuel_prior", run.out / "fuel_prior"
    inputs = {"eligible": value_input(run.eligible), "race_features": file_input(run.race_features)}
    if len(run.eligible) < MIN_HISTORY_RACES:
        return run.skip(stage, inputs, f"{len(run.eligible)} eligible race(s), fewer than {MIN_HISTORY_RACES}: the "
                        "race-pace prior (SPEC 0020) needs a history of races")
    if run.resume(stage, inputs):
        return
    t0 = time.perf_counter()
    kw = {} if run.n_boot is None else {"n_boot": run.n_boot}
    fuel_prior.run(run.eligible, out, run.profile, features_path=run.race_features, log=run.log, **kw)
    return run.done(stage, inputs, [out / f for f in FUEL_OUTPUTS], t0, {"races": len(run.eligible)})


def world_config_for(config_hash: str):
    """(label, SimConfig or None) of the world config with this config_hash: None for the
    races' own default config (score.race_simulators' world default), else a
    configs/sim/*.yaml file. None when no candidate has that hash."""
    from .config import SimConfig
    if calibration.config_hash(SimConfig()) == config_hash:
        return "default (SimConfig())", None
    for path in sorted(SIM_CONFIG_DIR.glob("*.yaml")):
        cfg = SimConfig.from_file(path)
        if calibration.config_hash(cfg) == config_hash:
            return _rel(path), cfg
    return None


def rebuild_reference(meta: dict, profile, reference_dir, scenario_dir=SCENARIO_DIR, build=None) -> dict:
    """Rebuild one stale reference (its sidecar `meta`) at the current code: the same scenario
    set (by digest), world config (by hash), seed, n_samples, max_stops and hindsight. Returns
    {"status": rebuilt, already rebuilt or not rebuilt, "path" or "reason"}. `build` replaces
    eval.reference.build_references (tests)."""
    from ..eval.reference import build_references, reference_key, reference_path
    from ..eval.scenarios import ScenarioSet
    from ..eval.score import race_simulators
    version = meta.get("scenario_set")
    spath = Path(scenario_dir) / f"{version}.json"
    if not spath.is_file():
        return {"status": "not rebuilt", "reason": f"no scenario set {_rel(spath)}"}
    scenario_set = ScenarioSet.load(spath)
    if (scenario_set.digest or scenario_set.compute_digest()) != meta.get("set_digest"):
        return {"status": "not rebuilt", "reason": f"{_rel(spath)} has another digest than the reference's set"}
    world = world_config_for(meta.get("world_config_hash"))
    if world is None:
        return {"status": "not rebuilt", "reason": f"no world config with hash {meta.get('world_config_hash')} "
                                                   "(the default or configs/sim/*.yaml)"}
    label, cfg = world
    kw = {k: meta[k] for k in ("seed", "n_samples", "max_stops", "hindsight")}
    key = reference_key(meta["set_digest"], meta["world_config_hash"], calibration.SIM_CODE_VERSION, **kw)
    target = reference_path(reference_dir, version, key)
    if target.exists():
        return {"status": "already rebuilt", "path": _rel(target), "world_config": label}
    written = (build or build_references)(scenario_set, race_simulators(profile, None, cfg), root=reference_dir, **kw)
    return {"status": "rebuilt", "path": " ".join(_rel(p) for p, _ in written), "world_config": label}


def stage_references(run: RealRun, build=None):
    from ..eval.reference import stale_references
    stage = "references"
    sidecars = sorted(run.reference_dir.glob("*/*.json")) if run.reference_dir.is_dir() else []
    inputs = {"reference_dir": {"path": _rel(run.reference_dir), "present": run.reference_dir.is_dir()},
              "references": value_input({_rel(p): json.loads(p.read_text()).get("sim_code_version")
                                         for p in sidecars})}
    if not run.reference_dir.is_dir():
        return run.skip(stage, inputs, f"no reference directory {_rel(run.reference_dir)}")
    if run.resume(stage, inputs):
        return
    t0 = time.perf_counter()
    stale = stale_references(run.reference_dir)
    run.log(f"reference --check: {len(stale)} cached reference(s) with a sim_code_version other than "
            f"{calibration.SIM_CODE_VERSION}" + (":" if stale else ""))
    results, outputs = [], []
    for path, meta in stale:
        res = {"stale": _rel(path), "stale_sim_code_version": meta.get("sim_code_version"),
               **rebuild_reference(meta, run.profile, run.reference_dir, build=build)}
        results.append(res)
        outputs += res.get("path", "").split()
        run.log(f"  {res['stale']} ({res['stale_sim_code_version']}): {res['status']}"
                + (f", {res['path']}" if res.get("path") else f": {res.get('reason')}"))
    not_rebuilt = sum(r["status"] == "not rebuilt" for r in results)
    return run.done(stage, inputs, outputs, t0, {"stale": results},
                    f"{len(stale)} stale, {not_rebuilt} not rebuilt" if stale else "no stale reference")


CROSS_REPORT_OUTPUTS = ("summary.md", "factors.csv", "arms.csv", "selection.csv", "tau_grid.csv", "g_by_arm.csv",
                        "stop_gain.csv")


def stage_report(run: RealRun):
    stage, out = "report", run.out / "cross_race"
    cross = run.entries.get("cross_race", {})
    inputs = {"run_json": file_input(out / "run.json"), "race_features": file_input(run.race_features),
              "cross_race_ran_at": value_input(cross.get("ran_at"))}
    if cross.get("status") not in (RAN, CURRENT):
        why = cross.get("reason") or "it did not run"
        return run.skip(stage, inputs, f"the cross-race pass is {cross.get('status', 'missing')} ({why})")
    if not inputs["run_json"]["present"]:
        return run.skip(stage, inputs, f"no {inputs['run_json']['path']}")
    if run.resume(stage, inputs):
        return
    t0 = time.perf_counter()
    kw = {} if run.n_boot is None else {"n_boot": run.n_boot}
    path = cross_race.report(out, run.profile, run.race_features, **kw)
    lines = [ln for ln in path.read_text().splitlines() if ln.startswith(("**PROVISIONAL", "Decision for"))]
    run.log("\n".join(["", f"{_rel(path)}, first look:", *lines, ""]))
    return run.done(stage, inputs, [out / f for f in CROSS_REPORT_OUTPUTS], t0, {"decisions": lines})


STAGE_FNS = {"eligibility": stage_eligibility, "cross_race": stage_cross_race, "event_rates": stage_event_rates,
             "fuel_prior": stage_fuel_prior, "references": stage_references, "report": stage_report}


def execute(run: RealRun) -> dict:
    """Every stage in STAGES order; a stage that raises is recorded as failed and re-raised."""
    run.log(f"real-run: profile {run.profile.name} ({run.profile.hash}), "
            + ("Slurm: array scripts are emitted, nothing is submitted" if run.slurm
               else f"in-process with n_jobs {run.profile.parallel.n_jobs}")
            + f"; {len(run.races)} race(s) ({run.race_source}); out {_rel(run.out)}")
    for note in run.race_notes:
        run.log(f"  {note}")
    for stage in STAGES:
        try:
            STAGE_FNS[stage](run)
        except Exception as exc:
            run._record(stage, {}, FAILED, f"{type(exc).__name__}: {exc}")
            raise
    run.log("\n" + summary_table(run) + f"\nmanifest: {_rel(run.manifest_path)}")
    return json.loads(run.manifest_path.read_text())


def summary_table(run: RealRun) -> str:
    return cross_race.md_table(pd.DataFrame([
        {"stage": e["stage"], "status": e["status"], "reason": e["reason"] or "-", "outputs": len(e["outputs"]),
         "wall time (s)": e["wall_time_s"]} for e in (run.entries[s] for s in STAGES if s in run.entries)]))


# --- Slurm ----------------------------------------------------------------------------------
class SlurmEmitter:
    """cross_race.run's runner on Sol. The units of a phase that are not done (jobs.is_done)
    go to an array script written from the job's committed template, and come back PENDING;
    the done ones come back skipped. Writes, per phase, <name>.sbatch, the race list the
    task's --manifest indexes (<name>_races.json) and the tasks (<name>_tasks.txt, one
    "ARM INDEX" line each, "-" for no arm); <name> is cross_race, or h_fold_<phase> (arms, s2,
    tau_grid). Collects the scripts and the sbatch commands; it never runs sbatch."""

    def __init__(self, directory, profile_ref: str, templates_dir=SLURM_DIR):
        self.dir, self.profile_ref, self.templates_dir = Path(directory), profile_ref, Path(templates_dir)
        self.scripts: list[Path] = []
        self.commands: list[str] = []

    def __call__(self, items, phase):
        todo = [i for i in items if not jobs.is_done(jobs.JOBS[i[0]].output(i[3], i[1], i[4]), i[2])]
        if todo:
            self.emit(todo, phase)
        waiting = {id(i) for i in todo}
        return [(cross_race.item_label(i), PENDING if id(i) in waiting else jobs.SKIPPED) for i in items]

    def emit(self, items, phase):
        job = items[0][0]
        out = Path(items[0][3]).resolve()
        pool = list(items[0][4].pool)
        name = job if job == "cross_race" else f"{job}_{phase}"
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest, tasks, script = (self.dir / f"{name}_races.json", self.dir / f"{name}_tasks.txt",
                                   self.dir / f"{name}.sbatch")
        manifest.write_text(json.dumps(pool, indent=1))
        tasks.write_text("".join(f"{i[4].arm or '-'} {pool.index(i[1])}\n" for i in items))
        profile = shlex.quote(self.profile_ref)
        loader = f"--race-loader {cross_race.LOADER}"
        if job == "cross_race":
            command = [f'python -m src.sim.jobs cross_race --profile {profile} --manifest "$MANIFEST" '
                       f'--task-id "$I" --out "$OUT" {loader}']
        else:
            command = [f"POOL=({' '.join(shlex.quote(r) for r in pool)})",
                       f'python -m src.sim.jobs h_fold --profile {profile} --manifest "$MANIFEST" --task-id "$I" '
                       f'--out "$OUT" --arm "$ARM" {loader} --pool "${{POOL[@]}}"']
        body = [f"TASKS={shlex.quote(str(tasks))}",
                'LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + ${SIM_TASK_OFFSET:-0} + 1))p" "$TASKS")',
                'read -r ARM I <<< "$LINE"', *command]
        template = self.templates_dir / TEMPLATES[job]
        note = [f"# real-run, cross-race phase {phase}: {len(items)} {job} task(s), one per line of $TASKS.",
                "# Written by `python -m src.sim.calibrate real-run`; rerun it rather than editing this file.",
                f"# The #SBATCH header and the environment are slurm/{template.name}'s: its partition, QOS, time",
                "# limit and module lines are placeholders to verify against Sol's documentation (slurm/README.md)."]
        script.write_text(from_template(template, f"sim-real-{phase.replace('_', '-')}", out, manifest, note, body))
        self.scripts += [script, manifest, tasks]
        for off in range(0, len(items), ARRAY_MAX):
            n = min(ARRAY_MAX, len(items) - off)
            export = f" --export=ALL,SIM_TASK_OFFSET={off}" if off else ""
            self.commands.append(f"sbatch --array=0-{n - 1}%{ARRAY_CONCURRENT}{export} {_rel(script)}")


def from_template(template: Path, job_name: str, out: Path, manifest: Path, note: list[str], body: list[str]) -> str:
    """An array script from a committed template: its #SBATCH lines and environment set-up (up
    to its python command), with the job name, OUT and MANIFEST replaced, then `body`."""
    lines = template.read_text().splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("#SBATCH"))
    cut = next(i for i, ln in enumerate(lines) if ln.startswith("python -m "))
    kept = lines[first:cut]
    while kept and (kept[-1].startswith("#") or not kept[-1].strip()):
        kept.pop()   # the template's comment on its own command
    replaced, lines_out = set(), []
    for ln in kept:
        for prefix, value in (("#SBATCH --job-name=", job_name), ("OUT=", shlex.quote(str(out))),
                              ("MANIFEST=", shlex.quote(str(manifest)))):
            if ln.startswith(prefix):
                ln = prefix + value
                replaced.add(prefix)
        lines_out.append(ln)
    if len(replaced) != 3:
        raise ValueError(f"{template}: expected #SBATCH --job-name=, OUT= and MANIFEST= lines to replace")
    return "\n".join(["#!/bin/bash", *note, *lines_out, "", *body]) + "\n"


# --- cli ------------------------------------------------------------------------------------
def main(argv=None):
    from ..eval.reference import REFERENCE_DIR
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m src.sim.calibrate real-run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", help="compute profile: local (scheduler local, in-process) or sol (scheduler "
                                      "slurm, emits Slurm arrays); default $SIM_PROFILE, else local")
    ap.add_argument("--races", nargs="+", default=["auto"], help="auto (every cached race, the default) or race ids")
    ap.add_argument("--out", default=str(REPORTS_DIR), help="root of the stage outputs (default reports/calibration)")
    ap.add_argument("--periods", default=str(event_rates.NEUTRALISED_PERIODS), help="the neutralised periods table")
    ap.add_argument("--race-features", default=str(cross_race.RACE_FEATURES))
    ap.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    ap.add_argument("--tau-multipliers", nargs="+", type=float, default=list(cross_race.TAU_MULTIPLIERS))
    ap.add_argument("--no-tau-grid", action="store_true")
    ap.add_argument("--allow-held-out", action="store_true",
                    help="run races data/eval_set.json holds out (refused without it; auto skips them)")
    args = ap.parse_args(argv)
    if "auto" in args.races and len(args.races) > 1:
        ap.error("--races takes auto or race ids, not both")
    profile = load_profile(args.profile)
    ref = str(Path(args.profile).resolve()) if args.profile and args.profile.endswith((".yaml", ".yml")) else None
    races, notes, source = (*cached_races(), "auto: every cached race") if args.races == ["auto"] else \
        (args.races, [], "given")
    skipped = []
    if args.races == ["auto"] and not args.allow_held_out:
        races, skipped = cross_race.split_held_out(races)
        notes = notes + [f"{r}: held out in data/eval_set.json, left out" for r in skipped]
    run = RealRun(races, profile, args.out, profile_ref=ref, race_source=source, race_notes=notes,
                  periods=args.periods, race_features=args.race_features, reference_dir=args.reference_dir,
                  tau_multipliers=args.tau_multipliers, tau_grid=not args.no_tau_grid,
                  allow_held_out=args.allow_held_out, held_out_skipped=skipped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        execute(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
