"""Reference cache: the world optimizer's plan per scenario, computed once and reused.

The reference of a scenario is optimizer_bridge.optimize_batch on the world simulator
(docs/eval.md "Who sees what"). It depends only on the scenario set, the world config, the
simulator code, the seed, n_samples, max_stops and whether the events are fixed to the
record, so

    python -m src.eval reference --scenarios S [--world-config W] [--profile P] [--hindsight]

computes it once per scenario and writes data/references/<set version>/<key>.parquet, with
the inputs in <key>.json beside it, where

    key = sha256(set digest, world config hash, sim_code_version, seed, n_samples,
                 max_stops, hindsight)[:12]

Per scenario the table holds the reference plan, its per-draw remaining times (float32),
their mean, sd and cvar_90, pit_now_opt, the runner-up's mean (the next plan in the
optimizer's tie-break order) and the paired standard error between the two.

`score` loads a matching reference and otherwise computes it, printing "reference computed,
not cached" once; with --require-reference a miss is an error (report runs). `reference
--check` lists cached references whose sim_code_version differs from the current one.

`--hindsight` fixes the SC and VSC laps of every draw to the ones the snapshot driver
recorded over the remaining laps. score never reads a hindsight reference: policies are
scored under the world's own events, and a regret needs both plans under the same draws.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np

from ..baselines.domain import NoFeasiblePlan
from ..contracts import Plan
from ..sim import calibration
from ..sim.engine import summarise
from ..sim.events import Scenario as EventScenario
from ..sim.optimizer_bridge import optimize_batch

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "references"
MISS_NOTICE = "reference computed, not cached"


class ReferenceMissing(LookupError):
    """--require-reference: a scenario has no cached reference for this run's inputs."""


def paired_se(a, b) -> float:
    """Standard error of mean(a - b) over paired draws: sd(a - b) / sqrt(N), sd with N - 1.
    0 at N = 1, where there is no spread to measure."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.std(ddof=1) / sqrt(len(d))) if len(d) > 1 else 0.0


def reference_key(set_digest: str, world_config_hash: str, sim_code_version: str, seed: int,
                  n_samples: int, max_stops: int, hindsight: bool) -> str:
    """The first 12 hex digits of sha256 over the inputs as a json list, in this order."""
    payload = json.dumps([set_digest, world_config_hash, sim_code_version, int(seed), int(n_samples),
                          int(max_stops), bool(hindsight)])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def reference_inputs(scenario_set, world, *, seed: int = 0, n_samples: int | None = None,
                     max_stops: int = 2, hindsight: bool = False) -> dict:
    """The key's inputs for one world simulator; n_samples None is the world's own."""
    return {"set_digest": scenario_set.digest or scenario_set.compute_digest(),
            "world_config_hash": calibration.config_hash(world.config),
            "sim_code_version": calibration.SIM_CODE_VERSION, "seed": int(seed),
            "n_samples": int(n_samples or world.config.n_samples), "max_stops": int(max_stops),
            "hindsight": bool(hindsight)}


@dataclass(frozen=True)
class Reference:
    """One scenario's reference. status "ok" or "no_reference" (no legal plan, `reason`)."""

    scenario_id: str
    status: str = "ok"
    reason: str | None = None
    plan: Plan | None = None
    times: np.ndarray | None = None        # (n_samples,) float32 remaining times
    mean_s: float = np.nan                  # from the float64 times, as optimize_batch has it
    sd_s: float = np.nan
    cvar_90_s: float = np.nan
    pit_now_opt: bool | None = None
    runner_up_plan: str | None = None       # json pairs
    runner_up_mean_s: float = np.nan
    runner_up_se_s: float = np.nan          # paired_se(runner-up times, reference times)


def recorded_events(world, snapshot) -> EventScenario:
    """The SC and VSC laps the snapshot driver recorded (track status 4, and 6 or 7)."""
    laps = world.laps
    if "driver" not in laps or "track_status" not in laps:
        return EventScenario()
    g = laps[laps["driver"] == snapshot.driver]
    status = g["track_status"].astype(str)
    return EventScenario(sc_laps=[int(x) for x in g.loc[status.str.contains("4"), "lap"]],
                         vsc_laps=[int(x) for x in g.loc[status.str.contains("[67]"), "lap"]])


def compute_reference(scenario, world, *, seed: int = 0, n_samples: int | None = None,
                      max_stops: int = 2, hindsight: bool = False) -> Reference:
    """The reference of one scenario on `world`, as the cache stores it (times in float32)."""
    snap = scenario.snapshot
    events = recorded_events(world, snap) if hindsight else None
    try:
        opt = optimize_batch(snap, world, seed=seed, n_samples=n_samples, max_stops=max_stops,
                             scenario=events, return_times=True)
    except NoFeasiblePlan as exc:
        return Reference(scenario.scenario_id, status="no_reference", reason=str(exc))
    times = opt.diagnostics["times"]
    summary = summarise(times)
    pits = opt.plan.pit_laps(snap)
    second = opt.diagnostics["runner_up"]
    return Reference(
        scenario.scenario_id, plan=opt.plan, times=times.astype(np.float32),
        mean_s=opt.evaluation.mean_time_s, sd_s=summary.std_s, cvar_90_s=summary.cvar_90_s,
        pit_now_opt=bool(pits) and pits[0] == snap.laps_completed + 1,
        runner_up_plan=None if second is None else json.dumps(second["plan"].as_pairs()),
        runner_up_mean_s=np.nan if second is None else float(second["mean_time_s"]),
        runner_up_se_s=np.nan if second is None else paired_se(second["times"], times))


# --- the parquet table ----------------------------------------------------------------
def _schema():
    import pyarrow as pa
    return pa.schema([
        ("scenario_id", pa.string()), ("status", pa.string()), ("reason", pa.string()),
        ("plan", pa.string()), ("opt_plan", pa.string()), ("opt_num_stops", pa.int64()),
        ("opt_first_pit_lap", pa.int64()), ("times", pa.list_(pa.float32())), ("mean_s", pa.float64()),
        ("sd_s", pa.float64()), ("cvar_90_s", pa.float64()), ("pit_now_opt", pa.bool_()),
        ("runner_up_plan", pa.string()), ("runner_up_mean_s", pa.float64()), ("runner_up_se_s", pa.float64()),
    ])


def _record(ref: Reference, snapshot) -> dict:
    ok = ref.plan is not None
    pits = ref.plan.pit_laps(snapshot) if ok else []
    return {"scenario_id": ref.scenario_id, "status": ref.status, "reason": ref.reason,
            "plan": ref.plan.model_dump_json() if ok else None,
            "opt_plan": json.dumps(ref.plan.as_pairs()) if ok else None,
            "opt_num_stops": ref.plan.num_stops if ok else None, "opt_first_pit_lap": pits[0] if pits else None,
            "times": ref.times.tolist() if ref.times is not None else None,
            "mean_s": ref.mean_s, "sd_s": ref.sd_s, "cvar_90_s": ref.cvar_90_s, "pit_now_opt": ref.pit_now_opt,
            "runner_up_plan": ref.runner_up_plan, "runner_up_mean_s": ref.runner_up_mean_s,
            "runner_up_se_s": ref.runner_up_se_s}


def _nan(v):
    return np.nan if v is None else float(v)


def _from_record(r: dict) -> Reference:
    return Reference(
        r["scenario_id"], status=r["status"], reason=r["reason"],
        plan=Plan.model_validate_json(r["plan"]) if r["plan"] is not None else None,
        times=np.asarray(r["times"], dtype=np.float32) if r["times"] is not None else None,
        mean_s=_nan(r["mean_s"]), sd_s=_nan(r["sd_s"]), cvar_90_s=_nan(r["cvar_90_s"]),
        pit_now_opt=r["pit_now_opt"], runner_up_plan=r["runner_up_plan"],
        runner_up_mean_s=_nan(r["runner_up_mean_s"]), runner_up_se_s=_nan(r["runner_up_se_s"]))


def reference_path(root, set_version: str, key: str) -> Path:
    return Path(root) / set_version / f"{key}.parquet"


def write_references(refs, snapshots, inputs: dict, set_version: str, root=REFERENCE_DIR) -> Path:
    """Write one key's references (with `snapshots` by scenario_id) and its json sidecar."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    key = reference_key(**inputs)
    path = reference_path(root, set_version, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_record(r, snapshots[r.scenario_id]) for r in refs], schema=_schema())
    pq.write_table(table, path)
    sidecar = {**inputs, "key": key, "scenario_set": set_version, "n_scenarios": len(refs)}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return path


def load_references(path) -> dict[str, Reference]:
    """scenario_id -> Reference from one cached table."""
    import pyarrow.parquet as pq
    return {r["scenario_id"]: _from_record(r) for r in pq.read_table(path).to_pylist()}


def build_references(scenario_set, sim_for, *, seed: int = 0, n_samples: int | None = None,
                     max_stops: int = 2, hindsight: bool = False, root=REFERENCE_DIR):
    """Compute and write the references of every scenario. Scenarios are grouped by key (one
    group unless races differ in world config). Returns [(path, [Reference, ...]), ...]."""
    groups = {}
    for scenario in scenario_set.items:
        world = sim_for(scenario.race_id, "world")
        inputs = reference_inputs(scenario_set, world, seed=seed, n_samples=n_samples, max_stops=max_stops,
                                  hindsight=hindsight)
        group = groups.setdefault(reference_key(**inputs), (inputs, [], {}))
        group[1].append(compute_reference(scenario, world, seed=seed, n_samples=n_samples,
                                          max_stops=max_stops, hindsight=hindsight))
        group[2][scenario.scenario_id] = scenario.snapshot
    return [(write_references(refs, snaps, inputs, scenario_set.version, root), refs)
            for inputs, refs, snaps in groups.values()]


def stale_references(root=REFERENCE_DIR, code_version: str | None = None) -> list[tuple[Path, dict]]:
    """(parquet path, sidecar) of every cached reference whose sim_code_version is not
    `code_version` (default: the current SIM_CODE_VERSION)."""
    code_version = code_version or calibration.SIM_CODE_VERSION
    out = []
    for side in sorted(Path(root).glob("*/*.json")):
        meta = json.loads(side.read_text())
        if meta.get("sim_code_version") != code_version:
            out.append((side.with_suffix(".parquet"), meta))
    return out


def summarize_references(refs) -> dict:
    """The reference-vs-runner-up precision of one run: how far the runner-up's mean is above
    the reference's, the paired standard error of that gap, and the share of scenarios where
    the gap is within 2 standard errors (the two plans are not told apart at this N)."""
    ok = [r for r in refs if r.status == "ok"]
    gap = np.array([r.runner_up_mean_s - r.mean_s for r in ok if np.isfinite(r.runner_up_mean_s)])
    se = np.array([r.runner_up_se_s for r in ok if np.isfinite(r.runner_up_mean_s)])

    def stat(f, xs):
        return float(f(xs)) if len(xs) else np.nan
    return {"scenarios": len(refs), "no_reference": len(refs) - len(ok),
            "n_samples": len(ok[0].times) if ok else 0, "with_runner_up": len(gap),
            "mean_runner_up_gap_s": stat(np.mean, gap), "median_runner_up_gap_s": stat(np.median, gap),
            "mean_runner_up_se_s": stat(np.mean, se), "median_runner_up_se_s": stat(np.median, se),
            "max_runner_up_se_s": stat(np.max, se),
            "share_runner_up_within_mc": stat(np.mean, gap <= 2 * se) if len(gap) else np.nan}


class ReferenceStore:
    """score's view of the cache under `root`: a matching reference or None. `require` makes
    a miss raise ReferenceMissing; otherwise the first miss of a run prints MISS_NOTICE."""

    def __init__(self, root=REFERENCE_DIR, require: bool = False):
        self.root, self.require = Path(root), require
        self.missed = False
        self._tables: dict[Path, dict[str, Reference] | None] = {}

    def get(self, scenario_set, scenario, world, *, seed: int = 0, n_samples: int | None = None,
            max_stops: int = 2) -> Reference | None:
        inputs = reference_inputs(scenario_set, world, seed=seed, n_samples=n_samples, max_stops=max_stops)
        path = reference_path(self.root, scenario_set.version, reference_key(**inputs))
        if path not in self._tables:
            self._tables[path] = load_references(path) if path.exists() else None
        table = self._tables[path]
        ref = table.get(scenario.scenario_id) if table is not None else None
        if ref is None:
            if self.require:
                raise ReferenceMissing(f"no cached reference for {scenario.scenario_id} at {path} "
                                       f"(inputs {inputs}); run `python -m src.eval reference` first")
            if not self.missed:
                print(MISS_NOTICE)
                self.missed = True
        return ref
