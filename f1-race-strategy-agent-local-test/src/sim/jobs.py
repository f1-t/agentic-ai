"""Resumable batch jobs. One task is one unit of work, written to files, skipped when already done.

    python -m src.sim.jobs manifest g_race --races 2021_Spain 2022_Bahrain --out manifest.json
    python -m src.sim.jobs g_race --profile sol --manifest manifest.json --task-id 0 --out $SCRATCH/sim
    python -m src.sim.jobs run-all g_race --manifest manifest.json --out out   # in-process, n_jobs workers
    python -m src.sim.jobs aggregate g_race --manifest manifest.json --out out

A task's output is <out>/<job dir>/<unit>.parquet with a <unit>.parquet.meta.json beside it
holding the sim code version and the profile hash. The meta is written last; a task is
skipped only when the output and its meta exist and both values match, so a code or profile
change reruns it and an interrupted run leaves nothing that looks finished.

Jobs: g_race (SPEC G rows for one race), h_fold (one held-out race or season for a
calibrator), eval_chunk (one chunk of scenario items through a pluggable runner),
build_weekend (not here: owned by the data pipeline).
"""
import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Protocol

import pandas as pd

from ..contracts import RaceParams
from . import calibrate, calibration
from .compute import ComputeProfile, load_profile, wandb_run
from .config import SimConfig
from .data import session_results
from .engine import Simulator

log = logging.getLogger(__name__)

DEFAULT_RACES = ["2021_Spain"]
SKIPPED, DONE = "skipped", "done"


# --- race data -------------------------------------------------------------------
@dataclass
class RaceData:
    """Everything a job needs about one race, loaded from the committed cache."""
    race_id: str
    frame: pd.DataFrame
    race: RaceParams
    practice: pd.DataFrame | None
    results: pd.DataFrame
    config: SimConfig = field(default_factory=SimConfig)


def split_race_id(race_id: str) -> tuple[int, str]:
    season, _, gp = race_id.partition("_")
    return int(season), gp


def load_race(race_id: str, profile: ComputeProfile) -> RaceData:
    """Frame, race constants (observed pit loss, as `calibrate midrace` does), practice and results."""
    season, gp = split_race_id(race_id)
    args = SimpleNamespace(season=season, gp=gp, pit_loss="observed", degradation=SimConfig().degradation.name,
                           config=None, min_stint=profile.sim.min_stint or RaceParams.model_fields["min_stint_laps"].default)
    session, frame, cfg, race = calibrate._race_setup(args)
    pit_loss, _ = calibrate._pit_loss(args, frame, cfg, race)
    race = race.model_copy(update={"pit_loss_s": pit_loss})
    return RaceData(race_id, frame, race, calibrate._practice(season, gp, cfg.long_run),
                    session_results(session), cfg)


# --- calibrators (SPEC H plugs in here) ---------------------------------------------
@dataclass
class Prediction:
    """What a calibrator says about one race: SimConfig field updates, and per target the
    predicted and the actual value (for the per-target rmse)."""
    config_update: dict = field(default_factory=dict)
    predicted: dict[str, float] = field(default_factory=dict)
    actual: dict[str, float] = field(default_factory=dict)


class Calibrator(Protocol):
    def fit(self, train: dict[str, RaceData], profile: ComputeProfile) -> None: ...
    def predict(self, race: RaceData) -> Prediction: ...


CALIBRATORS: dict[str, Callable[[ComputeProfile], Calibrator]] = {}


def register_calibrator(name: str):
    """Decorator: register a factory(profile) -> Calibrator under `name`."""
    def deco(factory):
        CALIBRATORS[name] = factory
        return factory
    return deco


def get_calibrator(name: str, profile: ComputeProfile) -> Calibrator:
    if name not in CALIBRATORS:
        try:   # SPEC H's module registers its calibrators on import, once it exists
            import importlib
            importlib.import_module("src.sim.calibrators")
        except ModuleNotFoundError:
            pass
    if name not in CALIBRATORS:
        raise KeyError(f"unknown calibrator {name!r}, registered: {sorted(CALIBRATORS)}")
    return CALIBRATORS[name](profile)


# --- eval runner (the eval harness plugs in here) --------------------------------------
EvalRunner = Callable[[str, ComputeProfile], pd.DataFrame]
_EVAL_RUNNER: list[EvalRunner | None] = [None]


def set_eval_runner(fn: EvalRunner | None):
    """Install fn(chunk_id, profile) -> DataFrame of that chunk's results."""
    _EVAL_RUNNER[0] = fn


# --- job definitions ---------------------------------------------------------------
def laps_k_for(frame: pd.DataFrame, total_laps: int, profile: ComputeProfile) -> list[int]:
    """Snapshot laps: every `laps_every` from 0 to total_laps - 10, plus the lap before and after each SC / VSC period."""
    g = profile.g
    laps = set(range(0, total_laps - 10 + 1, g.laps_every))
    if g.sc_adjacent:
        status = frame["track_status"].astype(str)
        for pattern in ("4", "[67]"):
            hit = sorted(int(x) for x in frame.loc[status.str.contains(pattern), "lap"].unique())
            for i, lap in enumerate(hit):
                if i == 0 or hit[i - 1] != lap - 1:
                    laps.add(lap - 1)
                if i == len(hit) - 1 or hit[i + 1] != lap + 1:
                    laps.add(lap + 1)
    return sorted(x for x in laps if 0 <= x < total_laps)


def cadence_label(n) -> str:
    return "none" if n is None else str(n)


def _stamp(df: pd.DataFrame, data: RaceData, cfg: SimConfig, profile: ComputeProfile) -> pd.DataFrame:
    return df.assign(pit_loss_s=data.race.pit_loss_s, config_hash=calibration.config_hash(cfg),
                     sim_code_version=calibration.SIM_CODE_VERSION, profile=profile.name)


def g_rows(data: RaceData, cfg: SimConfig, profile: ComputeProfile, windows=None, laps_k=None) -> pd.DataFrame:
    """SPEC G rows for one race, one copy per cadence N: the snapshots at multiples of N
    (None: lap 0 only) out of the profile's lap set. Rows do not depend on N, so the
    midrace replay runs once and the copies are tagged."""
    laps_k = laps_k_for(data.frame, data.race.total_laps, profile) if laps_k is None else laps_k

    def factory(window):
        return Simulator(data.frame, cfg.model_copy(update={"fit_window": window}), data.practice)

    rows = calibration.midrace(factory, data.frame, data.race, laps_k=laps_k,
                               windows=windows or profile.g.windows, results=data.results)
    parts = []
    for n in profile.g.cadences:
        keep = rows["k"] == 0 if n is None else rows["k"] % n == 0
        parts.append(rows[keep].assign(cadence=cadence_label(n)))
    return _stamp(pd.concat(parts, ignore_index=True), data, cfg, profile)


class Job:
    """A unit-of-work type: how to list units, where a unit's output goes, how to run it."""
    name: str
    subdir: str
    suffix = ".parquet"

    def units(self, args) -> list[str]:
        raise NotImplementedError

    def stem(self, unit: str, args=None) -> str:
        return unit.replace(":", "_").replace("/", "_")

    def output(self, out, unit: str, args=None) -> Path:
        return Path(out) / self.subdir / f"{self.stem(unit, args)}{self.suffix}"

    def run(self, unit: str, profile: ComputeProfile, out, args=None) -> pd.DataFrame:
        """Do the unit and return its table (written by the runner); extra files go under out."""
        raise NotImplementedError

    def summarize(self, tables: dict[str, pd.DataFrame], profile: ComputeProfile, out, args=None) -> list[Path]:
        """Merged summary files from the tables that exist, by unit. Returns the paths written."""
        raise NotImplementedError


class GRace(Job):
    name, subdir = "g_race", "g"

    def units(self, args):
        return list(args.races or DEFAULT_RACES)

    def run(self, unit, profile, out, args=None):
        data = load_race(unit, profile)
        return g_rows(data, data.config.model_copy(update={"n_samples": profile.sim.n_samples}), profile)

    def summarize(self, tables, profile, out, args=None):
        rows = pd.concat(tables.values(), ignore_index=True)
        summaries = []
        for cadence, part in rows.groupby("cadence", sort=False):
            summaries.append(calibration.aggregate(part, n_boot=profile.g.n_boot).assign(cadence=cadence))
        summary = pd.concat(summaries)
        path = Path(out) / "g" / "summary.csv"
        summary.to_csv(path)
        paths = [path]
        fig = _g_figure(summary, Path(out) / "g" / "mae_by_horizon.png")
        return paths + ([fig] if fig else [])


def _g_figure(summary: pd.DataFrame, path: Path):
    """mae_s by horizon bucket and window for the densest cadence, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    s = summary.reset_index()
    s = s[s["cadence"] == s["cadence"].iloc[0]]
    bucket = next((c for c in s.columns if "bucket" in c or c == "horizon"), None)
    if bucket is None or "window" not in s:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for window, part in s.groupby("window"):
        ax.plot(part[bucket].astype(str), part["mae_s"], marker="o", label=window)
    ax.set_xlabel("horizon bucket (laps)")
    ax.set_ylabel("mean absolute error (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


class HFold(Job):
    name, subdir = "h_fold", "h"

    def output(self, out, unit, args=None):
        return Path(out) / self.subdir / args.calibrator / f"{self.stem(unit)}{self.suffix}"

    def units(self, args):
        """Held-out races (a seeded subsample of loro_folds when set) and, when loso is on, season:YYYY folds."""
        profile = load_profile(args.profile)
        pool = list(args.races or DEFAULT_RACES)
        folds = pool
        if profile.h.loro_folds is not None and profile.h.loro_folds < len(pool):
            folds = sorted(random.Random(profile.h.seed).sample(pool, profile.h.loro_folds))
        if profile.h.loso:
            folds = folds + [f"season:{s}" for s in sorted({split_race_id(r)[0] for r in pool})]
        return folds

    def run(self, unit, profile, out, args=None):
        pool = list(args.pool or args.races or DEFAULT_RACES)
        held = [r for r in pool if r.startswith(unit.split(":", 1)[1] + "_")] if unit.startswith("season:") else [unit]
        data = {r: load_race(r, profile) for r in pool}
        train = {r: d for r, d in data.items() if r not in held}
        cal = get_calibrator(args.calibrator, profile)
        cal.fit(train, profile)
        tables, sse, counts = [], {}, {}
        for r in held:
            pred = cal.predict(data[r])
            cfg = data[r].config.model_copy(update={"n_samples": profile.sim.n_samples, **pred.config_update})
            rows = g_rows(data[r], cfg, profile, windows=[w for w in ("pre_race", "to_snapshot") if w in profile.g.windows])
            tables.append(rows.assign(calibrator=args.calibrator, fold=unit))
            for t, p in pred.predicted.items():
                if t in pred.actual:
                    sse[t] = sse.get(t, 0.0) + (p - pred.actual[t]) ** 2
                    counts[t] = counts.get(t, 0) + 1
        rmse = {t: {"rmse": math.sqrt(sse[t] / counts[t]), "sse": sse[t], "n": counts[t]} for t in sse}
        path = self.output(out, unit, args).with_suffix(".rmse.json")
        path.write_text(json.dumps({"fold": unit, "held_out": held, "targets": rmse}, indent=1))
        return pd.concat(tables, ignore_index=True)

    def summarize(self, tables, profile, out, args=None):
        rows = pd.concat(tables.values(), ignore_index=True)
        paths = []
        base = Path(out) / "h" / (args.calibrator if args else "all")
        summary = calibration.aggregate(rows[rows["cadence"] == rows["cadence"].iloc[0]], n_boot=profile.g.n_boot)
        paths.append(base / "summary.csv")
        summary.to_csv(paths[-1])
        sse, counts = {}, {}
        for f in sorted(base.glob("*.rmse.json")):
            for t, v in json.loads(f.read_text())["targets"].items():
                sse[t], counts[t] = sse.get(t, 0.0) + v["sse"], counts.get(t, 0) + v["n"]
        paths.append(base / "rmse.json")
        paths[-1].write_text(json.dumps({t: {"rmse": math.sqrt(sse[t] / counts[t]), "n": counts[t]} for t in sse}, indent=1))
        return paths


class EvalChunk(Job):
    name, subdir = "eval_chunk", "eval"

    def units(self, args):
        size = load_profile(args.profile).parallel.eval_chunk_size
        return [f"chunk_{i:04d}" for i in range(math.ceil(args.n_items / size))]

    def run(self, unit, profile, out, args=None):
        runner = _EVAL_RUNNER[0]
        if runner is None:
            raise NotImplementedError("no eval runner installed; call jobs.set_eval_runner(fn)")
        return runner(unit, profile)

    def summarize(self, tables, profile, out, args=None):
        path = Path(out) / "eval" / "all.parquet"
        pd.concat(tables.values(), ignore_index=True).to_parquet(path)
        return [path]


class BuildWeekend(Job):
    """Stub. The bulk data build belongs to the data pipeline (SPEC I), not to this module."""
    name, subdir = "build_weekend", "build"

    def units(self, args):
        raise NotImplementedError("owned by the data pipeline")

    def run(self, unit, profile, out, args=None):
        raise NotImplementedError("owned by the data pipeline")

    def summarize(self, tables, profile, out, args=None):
        raise NotImplementedError("owned by the data pipeline")


JOBS: dict[str, Job] = {j.name: j for j in (GRace(), HFold(), EvalChunk(), BuildWeekend())}


# --- runner ---------------------------------------------------------------------------
def read_manifest(path) -> list[str]:
    """Units from a json list (or {"units": [...]}) or a text file with one per line."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [str(u) for u in (data["units"] if isinstance(data, dict) else data)]


def _meta_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def is_done(output: Path, profile: ComputeProfile) -> bool:
    """Output and meta exist, and the meta matches this sim code version and profile hash."""
    meta = _meta_path(output)
    if not (output.is_file() and meta.is_file()):
        return False
    try:
        m = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return False
    return m.get("sim_code_version") == calibration.SIM_CODE_VERSION and m.get("profile_hash") == profile.hash


def run_task(job: Job, unit: str, profile: ComputeProfile, out, args=None) -> str:
    """Run one unit unless its output is current. Returns "done" or "skipped"."""
    output = job.output(out, unit, args)
    if is_done(output, profile):
        return SKIPPED
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    table = job.run(unit, profile, out, args)
    tmp = output.with_name(output.name + ".tmp")
    table.to_parquet(tmp)
    os.replace(tmp, output)
    _meta_path(output).write_text(json.dumps({
        "job": job.name, "unit": unit, "profile": profile.name, "profile_hash": profile.hash,
        "sim_code_version": calibration.SIM_CODE_VERSION, "rows": len(table),
        "runtime_s": round(time.perf_counter() - started, 2)}, indent=1))
    return DONE


def aggregate(job: Job, units: list[str], profile: ComputeProfile, out, args=None) -> dict:
    """Merge the outputs that exist into the job's summary tables. Missing or stale tasks are listed, not fatal."""
    present, missing = {}, []
    for u in units:
        output = job.output(out, u, args)
        if is_done(output, profile):
            present[u] = pd.read_parquet(output)
        else:
            missing.append(u)
    status = {"job": job.name, "profile": profile.name, "expected": len(units), "done": len(present),
              "missing": missing, "files": []}
    if present:
        status["files"] = [str(p) for p in job.summarize(present, profile, out, args)]
    path = Path(out) / job.subdir / ("status.json" if not isinstance(job, HFold) else f"{args.calibrator}/status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=1))
    return status


def _run_one(item):
    name, unit, profile, out, args = item
    return unit, run_task(JOBS[name], unit, profile, out, args)


# --- cli --------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m src.sim.jobs", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, task=False):
        p.add_argument("--profile", help="compute profile name, else $SIM_PROFILE, else local")
        p.add_argument("--manifest", help="text or json list of units")
        p.add_argument("--out", required=True, help="output directory (scratch on Sol)")
        p.add_argument("--races", nargs="+", help="race ids such as 2021_Spain (manifest, h_fold pool, aggregate)")
        p.add_argument("--pool", nargs="+", help="h_fold: training pool, default --races")
        p.add_argument("--calibrator", default="default", help="h_fold: registered calibrator name")
        p.add_argument("--n-items", type=int, default=0, help="eval_chunk manifest: number of scenario items")
        p.add_argument("--wandb", action="store_true", help="log to wandb as the profile's wandb section says")
        if task:
            p.add_argument("--task-id", type=int, required=True, help="index into the manifest")

    for name in JOBS:
        common(sub.add_parser(name, help=f"run one {name} task"), task=True)
    for cmd in ("manifest", "run-all", "aggregate"):
        p = sub.add_parser(cmd)
        p.add_argument("job", choices=sorted(JOBS))
        common(p)
    return ap


def _units(args, job: Job, profile) -> list[str]:
    if args.manifest:
        return read_manifest(args.manifest)
    return job.units(args)


def main(argv=None):
    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    args.profile = profile.name   # units() reloads by name
    cmd = args.command
    job = JOBS[args.job if cmd in ("manifest", "run-all", "aggregate") else cmd]
    if cmd == "manifest":
        units = job.units(args)
        text = json.dumps(units, indent=1)
        if args.manifest:
            Path(args.manifest).write_text(text)
        print(text)
        return 0
    if args.pool is None and isinstance(job, HFold):
        args.pool = args.races
    if cmd == "aggregate":
        status = aggregate(job, _units(args, job, profile), profile, args.out, args)
        print(f"{job.name} [{profile.name}]: {status['done']}/{status['expected']} tasks, missing: {status['missing']}")
        for f in status["files"]:
            print("  wrote", f)
        return 0
    with wandb_run(profile, job.name, {"out": str(args.out)}) if args.wandb else _null():
        if cmd == "run-all":
            items = [(job.name, u, profile, args.out, args) for u in _units(args, job, profile)]
            if profile.parallel.n_jobs > 1:
                with ProcessPoolExecutor(profile.parallel.n_jobs) as pool:
                    results = list(pool.map(_run_one, items))
            else:
                results = [_run_one(i) for i in items]
            for unit, status in results:
                print(f"{job.name} {unit}: {status}")
            return 0
        units = _units(args, job, profile)
        unit = units[args.task_id]
        status = run_task(job, unit, profile, args.out, args)
        print(f"{job.name} {unit} [{profile.name}]: {status}")
    return 0


def _null():
    return contextlib.nullcontext()


if __name__ == "__main__":
    sys.exit(main())
