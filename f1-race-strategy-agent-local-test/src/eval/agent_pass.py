"""The rerun's agent stage (src/eval/rerun.py, docs/eval.md "Rerunning the evaluation"): the live
agent over a scenario set, `jobs` worker processes, resumable, with a sequential timing sample.

Scenarios are independent (one conversation each, on its own snapshot), so the pass runs them in
`jobs` spawned worker processes that take scenario ids from one shared queue in the set's order.
Each worker builds its own simulators (one race's at a time, a race's data loaded once per worker
while its scenarios last), its own backend and its own AgentPolicy, BLAS at one thread (set in the
environment the worker starts with, before it imports numpy). One scenario is one trajectory file,
trajectories/<scenario id, ":" written "_">.jsonl (the set's ids are checked to map to distinct
names), written by the one worker that runs it. Each worker appends one record per finished scenario
to its own file, parts/agent/rows_w<k>.jsonl; the parent merges the rows in the set's scenario order
into agent.csv, so the file's rows and columns are the same for any `jobs` apart from the timing
fields (latency_s). jobs 1 runs the same code in this process.

Safety: a scenario with a record and a trajectory is skipped by --resume, never rerun (an error
record with --rerun-errors is); a finished agent.csv is never overwritten without --force; a
scenario that fails (an API error the agent loop reports, an exception while scoring, a worker
process that dies) is an `error` row with the message, and the others go on; a transient HTTP error
(429, 5xx, a timeout or a dropped connection) is retried on the same turn, at most TRIES tries with
BACKOFF_S between them, and the row records the retries (api_retries); an authentication error
(401, 403) stops the pass: its scenario gets no record, so --resume reruns it once the key is fixed.

The timing sample (--timing-sample K): before the pass, K scenarios, a seeded sample stratified
across races and early/mid/late laps, run sequentially in this process, one job, rows to
timing_sample.csv and trajectories to timing_sample/trajectories/. They are for timing only: the
reported decision time comes from them, measured without concurrent episodes from this pass, and
the full pass runs every scenario again (those K included) under `jobs`, which gives the results.
agent_run.json records both passes (jobs, start, end, wall time) and which latency is the headline.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TRIES = 3                  # tries per model turn on a transient HTTP error (the first one included)
BACKOFF_S = (5.0, 20.0)    # the wait before the 2nd and the 3rd try
REQUEST_TIMEOUT_S = 180.0  # an OpenAI-compatible client's per-request timeout (its own default is 600 s)
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
PHASES = ("early", "mid", "late")
TIMING_FIELDS = ("latency_s",)
DEFAULT_BACKEND = "src.llm:create_backend"
HEADLINE = ("the decision time reported is the isolated timing sample's (timing_sample.csv: K episodes run "
            "sequentially, one job, before the pass); the full pass's per-episode latency_s was measured with "
            "agent_jobs episodes running concurrently and is reported beside it, not as the decision time")


# --- transient and fatal API errors ----------------------------------------------------------------
class FatalBackendError(RuntimeError):
    """An authentication error: no scenario can succeed, the pass stops."""


def classify(exc) -> str | None:
    """'auth' (401, 403), 'transient' (429, 5xx, a timeout, a dropped connection) or None (anything
    else: the agent loop's own error, reported on the row as it is). Reads the status code and the
    OpenAI client's exception names, so a stub exception classifies the same way."""
    status = getattr(exc, "status_code", None)
    names = {c.__name__ for c in type(exc).__mro__}
    if status in (401, 403) or names & {"AuthenticationError", "PermissionDeniedError"}:
        return "auth"
    if status == 429 or (isinstance(status, int) and status >= 500) or isinstance(exc, TimeoutError) or \
            names & {"RateLimitError", "InternalServerError", "APITimeoutError", "APIConnectionError"}:
        return "transient"
    return None


def _label(exc) -> str:
    status = getattr(exc, "status_code", None)
    return f"{status}" if status is not None else type(exc).__name__


_SECRETISH = re.compile(r"(sk-)[A-Za-z0-9_\-*.]{8,}")


def redact(text) -> str | None:
    """`text` with the configured keys and anything key-shaped masked: an error message can echo
    part of a key, and rows and logs are kept."""
    if text is None:
        return None
    text = str(text)
    for name in ("VOYAGER_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 6:
            text = text.replace(value, "<redacted>")
    return _SECRETISH.sub(lambda m: m.group(1) + "<redacted>", text)


class RetryingBackend:
    """A model backend with a bounded retry of one turn on a transient HTTP error (classify): the same
    request is sent again after BACKOFF_S, at most `tries` tries; `retries` counts the retries since the
    last reset and `events` their labels (status code or exception name). An auth error sets `fatal`
    and is raised as it is (AgentPolicy reports it on the row; the worker then stops the pass). The
    agent loop sees the backend's own name and attributes. A backend with an OpenAI-compatible
    `client` gets that client's own retries off and REQUEST_TIMEOUT_S, so every retry is counted here."""

    def __init__(self, inner, *, tries: int = TRIES, backoff=BACKOFF_S, sleep=time.sleep):
        self.inner, self.tries, self.backoff, self.sleep = inner, tries, tuple(backoff), sleep
        self.retries, self.events, self.fatal = 0, [], None
        client = getattr(inner, "client", None)
        self.client_retries_off = False
        if client is not None and hasattr(client, "with_options"):
            inner.client = client.with_options(max_retries=0, timeout=REQUEST_TIMEOUT_S)
            self.client_retries_off = True

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def name(self):
        return self.inner.name

    def reset(self):
        self.retries, self.events = 0, []

    def run_turn(self, **kwargs):
        for attempt in range(self.tries):
            try:
                return self.inner.run_turn(**kwargs)
            except Exception as exc:  # noqa: BLE001  classified below; anything else goes on unchanged
                kind = classify(exc)
                if kind == "auth":
                    self.fatal = redact(f"{type(exc).__name__}: {exc}")
                    raise
                if kind != "transient" or attempt == self.tries - 1:
                    raise
                self.retries += 1
                self.events.append(_label(exc))
                self.sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
        raise AssertionError("unreachable")


# --- the timing sample -------------------------------------------------------------------------------
def phase_of(scenario) -> str:
    """early, mid or late: the decision lap's third of the race (laps completed / race laps)."""
    frac = scenario.lap / max(scenario.snapshot.race.total_laps, 1)
    return PHASES[min(int(frac * 3), 2)]


def timing_sample(items, k: int, seed: int = 0) -> list:
    """A seeded sample of `k` of `items`, stratified across races and early/mid/late laps: the phases
    in turn, each from the next race (in a seeded order) that has a scenario left in it, races before
    repeats. Returned in the items' order."""
    import numpy as np
    k = min(int(k), len(items))
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    pools = {}
    for sc in items:
        pools.setdefault((sc.race_id, phase_of(sc)), []).append(sc)
    for key in sorted(pools):
        pools[key] = [pools[key][i] for i in rng.permutation(len(pools[key]))]
    races = [str(r) for r in rng.permutation(sorted({sc.race_id for sc in items}))]
    out, r, i = [], 0, 0
    while len(out) < k:
        phase = PHASES[i % len(PHASES)]
        i += 1
        for _ in range(len(races)):
            race = races[r % len(races)]
            r += 1
            if pools.get((race, phase)):
                out.append(pools[race, phase].pop())
                break
    chosen = {sc.scenario_id for sc in out}
    return [sc for sc in items if sc.scenario_id in chosen]


# --- one worker ----------------------------------------------------------------------------------------
def _import(spec: str):
    import importlib
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def _jsonable(v):
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"not JSON serialisable: {type(v).__name__}")


def _base_row(sc) -> dict:
    return {"scenario_id": sc.scenario_id, "race_id": sc.race_id, "driver": sc.driver, "lap": sc.lap,
            "policy": "agent", "status": "error"}


class Runner:
    """One process's agent: the set, the references, a RetryingBackend around `backend` (a
    "module:function" spec), an AgentPolicy logging to `traj_dir`, and the simulators of the race in
    hand (race_simulators per race, the last `keep` races kept)."""

    def __init__(self, cfg: dict, keep: int = 2):
        from ..sim import SimConfig
        from ..sim.compute import load_profile
        from .policies import AgentPolicy
        from .reference import ReferenceStore
        from .scenarios import ScenarioSet
        self.cfg, self.keep = cfg, keep
        self.set = ScenarioSet.load(cfg["scenarios"])
        self.by_id = {sc.scenario_id: sc for sc in self.set.items}
        self.profile = load_profile(cfg["profile"])
        self.belief = SimConfig.from_file(cfg["belief"]) if cfg["belief"] else None
        self.world = SimConfig.from_file(cfg["world"]) if cfg["world"] else None
        self.references = ReferenceStore(cfg["references"], require=True)
        self.backend = RetryingBackend(_import(cfg.get("backend") or DEFAULT_BACKEND)(),
                                       backoff=cfg.get("backoff", BACKOFF_S))
        self.policy = AgentPolicy(set_name=".", profile=self.profile, log_dir=cfg["traj_dir"], backend=self.backend)
        self._sims = {}

    def sim_for(self, race_id):
        from .score import race_simulators
        if race_id not in self._sims:
            while len(self._sims) >= self.keep:
                self._sims.pop(next(iter(self._sims)))
            self._sims[race_id] = race_simulators(self.profile, self.belief, self.world)
        return self._sims[race_id]

    def run(self, sid: str) -> dict:
        """{scenario_id, rows, retries, retry_events, fatal}: the scenario's rows (an error row when
        scoring raised), `fatal` the auth error that stops the pass (no rows then)."""
        from .score import score_set
        sc = self.by_id[sid]
        self.backend.reset()
        hook = getattr(self.backend.inner, "on_scenario", None)   # tests: a scripted backend's script
        if hook is not None:
            hook(sid)
        try:
            frame = score_set(self.set, self.sim_for(sc.race_id), {"agent": self.policy},
                              n_samples=self.profile.sim.n_samples, max_stops=self.profile.sim.max_stops,
                              items=[sc], warn=False, references=self.references)
            rows = frame.to_dict("records")
        except Exception as exc:  # noqa: BLE001  one scenario's failure is its row, the pass goes on
            rows = [{**_base_row(sc), "reason": f"scoring: {type(exc).__name__}: {exc}"}]
        if self.backend.fatal:
            self.policy.log_path(sc).unlink(missing_ok=True)
            return {"scenario_id": sid, "rows": [], "fatal": self.backend.fatal}
        for row in rows:
            if row.get("reason") is not None and row.get("status") == "error":
                row["reason"] = redact(row["reason"])
            row["api_retries"] = self.backend.retries
        return {"scenario_id": sid, "rows": rows, "retries": self.backend.retries,
                "retry_events": list(self.backend.events), "fatal": None}


def append_record(path: Path, record: dict, worker) -> None:
    record = {**record, "worker": worker, "finished_at": time.time()}
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_jsonable) + "\n")
        f.flush()


def _worker_main(k, cfg, tasks, results):
    """A spawned worker: scenario ids from `tasks` until None, each record appended to its own rows
    file before the parent hears of it."""
    try:
        runner = Runner(cfg)
    except BaseException as exc:  # noqa: BLE001  the parent counts a worker that cannot start
        results.put(("broken", k, None, redact(f"{type(exc).__name__}: {exc}")))
        return
    rows_file = Path(cfg["parts"]) / f"rows_w{k}.jsonl"
    while True:
        sid = tasks.get()
        if sid is None:
            results.put(("exit", k, None, None))
            return
        results.put(("start", k, sid, None))
        rec = runner.run(sid)
        if rec["fatal"]:
            results.put(("fatal", k, sid, rec["fatal"]))
            return
        append_record(rows_file, rec, k)
        results.put(("done", k, sid, _summary(rec)))


def _summary(rec) -> dict:
    return {"errors": sum(r.get("status") == "error" for r in rec["rows"]), "retries": rec.get("retries", 0),
            "retry_events": rec.get("retry_events", [])}


# --- the records on disk ------------------------------------------------------------------------------
def read_records(parts: Path) -> dict:
    """scenario_id -> its latest record over every rows file under `parts`."""
    out = {}
    for path in sorted(Path(parts).glob("rows_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:   # a line cut off by a killed process: that scenario reruns
                continue
            old = out.get(rec["scenario_id"])
            if old is None or rec.get("finished_at", 0) >= old.get("finished_at", 0):
                out[rec["scenario_id"]] = rec
    return out


def traj_name(scenario_id: str) -> str:
    return f"{scenario_id.replace(':', '_')}.jsonl"


def is_done(rec, traj_dir: Path, rerun_errors: bool) -> bool:
    """A scenario is finished when it has a record and a trajectory; an error record also counts
    unless `rerun_errors`."""
    if rec is None or not (Path(traj_dir) / traj_name(rec["scenario_id"])).exists():
        return False
    errored = any(r.get("status") == "error" for r in rec["rows"])
    return not (errored and rerun_errors)


def merge_rows(items, records: dict):
    """The rows of every scenario of `items`, in the items' order, as one DataFrame (the order and
    columns a single sequential score_set gives)."""
    import pandas as pd
    rows = [r for sc in items for r in records[sc.scenario_id]["rows"]]
    return pd.DataFrame(rows)


# --- the pass ------------------------------------------------------------------------------------------
class PassStopped(SystemExit):
    pass


def _fmt_eta(done, total, elapsed):
    return f"{elapsed / done * (total - done):.0f} s" if done else "?"


def run_pass(items, *, cfg: dict, jobs: int, traj_dir: Path, parts: Path, label: str, rerun_errors: bool = False,
             progress_every: int = 10, log=print) -> dict:
    """Run every scenario of `items` that is not finished (is_done) with `jobs` workers (in this
    process when jobs is 1); returns {jobs, started, ended, wall_s, ran, skipped, errors, retries,
    retry_events, crashed}. Raises PassStopped on an auth error, or when workers keep dying."""
    traj_dir, parts = Path(traj_dir), Path(parts)
    traj_dir.mkdir(parents=True, exist_ok=True)
    parts.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "traj_dir": str(traj_dir), "parts": str(parts)}
    records = read_records(parts)
    pending = [sc.scenario_id for sc in items if not is_done(records.get(sc.scenario_id), traj_dir, rerun_errors)]
    total, skipped = len(pending), len(items) - len(pending)
    started, t0 = datetime.now(timezone.utc), time.perf_counter()
    stats = {"done": 0, "errors": 0, "retries": 0, "retry_events": [], "crashed": 0}
    log(f"{label}: {total} to run, {skipped} finished already (skipped), jobs {jobs}")

    def finished(summary):
        stats["done"] += 1
        stats["errors"] += summary["errors"]
        stats["retries"] += summary["retries"]
        stats["retry_events"] += summary["retry_events"]
        if stats["done"] % max(progress_every, 1) == 0 or stats["done"] == total:
            el = time.perf_counter() - t0
            log(f"{label} {stats['done']}/{total} errors {stats['errors']} retries {stats['retries']} "
                f"elapsed {el:.0f} s eta {_fmt_eta(stats['done'], total, el)}")

    def info(**extra):
        events = stats["retry_events"]
        return {"label": label, "jobs": jobs, "started": started.isoformat(),
                "ended": datetime.now(timezone.utc).isoformat(), "wall_s": round(time.perf_counter() - t0, 1),
                "to_run": total, "ran": stats["done"], "skipped": skipped, "errors": stats["errors"],
                "retries": stats["retries"], "retry_event_counts": {e: events.count(e) for e in sorted(set(events))},
                "crashed": stats["crashed"], **extra}

    try:
        if total and jobs <= 1:
            runner = Runner(cfg)
            for sid in pending:
                rec = runner.run(sid)
                if rec["fatal"]:
                    raise PassStopped(f"{label}: authentication failed on {sid}, the pass stopped: {rec['fatal']}")
                append_record(parts / "rows_w0.jsonl", rec, 0)
                finished(_summary(rec))
        elif total:
            _run_workers(pending, cfg, min(jobs, total), parts, finished, stats, label, log)
    except (PassStopped, KeyboardInterrupt) as exc:
        exc.info = info(stopped=str(exc) or type(exc).__name__)
        raise
    return info()


def _run_workers(pending, cfg, jobs, parts, finished, stats, label, log):
    import multiprocessing as mp
    by_id = None
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for sid in pending:
        tasks.put(sid)
    for _ in range(jobs):
        tasks.put(None)
    procs, inflight, exited = {}, {}, set()
    respawns = [0]

    def start(k):
        saved = {v: os.environ.get(v) for v in BLAS_ENV}
        os.environ.update({v: "1" for v in BLAS_ENV})   # the worker starts with BLAS at one thread
        try:
            p = ctx.Process(target=_worker_main, args=(k, cfg, tasks, results), daemon=True)
            p.start()
        finally:
            for v, val in saved.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val
        procs[k], inflight[k] = p, None

    def stop_all():
        for p in procs.values():
            if p.is_alive():
                p.terminate()
        for p in procs.values():
            p.join(5)

    for k in range(jobs):
        start(k)
    remaining = len(pending)
    try:
        while remaining:
            try:
                kind, k, sid, payload = results.get(timeout=2.0)
            except queue.Empty:
                kind = None
            if kind == "start":
                inflight[k] = sid
            elif kind == "done":
                inflight[k] = None
                remaining -= 1
                finished(payload)
            elif kind == "fatal":
                raise PassStopped(f"{label}: authentication failed on {sid}, the pass stopped: {payload}")
            elif kind == "exit":
                exited.add(k)
            elif kind == "broken":
                raise PassStopped(f"{label}: worker {k} could not start: {payload}")
            # a worker that died: its scenario in hand is an error row, a new worker takes its place
            for k, p in list(procs.items()):
                if k in exited or p.is_alive():
                    continue
                p.join(0.1)
                exited.add(k)
                if p.exitcode == 0:   # a clean exit whose "exit" message is still in the queue
                    continue
                sid = inflight.pop(k, None)
                if sid is not None:
                    if by_id is None:
                        from .scenarios import ScenarioSet
                        by_id = {sc.scenario_id: sc for sc in ScenarioSet.load(cfg["scenarios"]).items}
                    row = {**_base_row(by_id[sid]), "reason": f"worker crashed (exit code {p.exitcode})",
                           "api_retries": None}
                    append_record(parts / "rows_parent.jsonl", {"scenario_id": sid, "rows": [row]}, "parent")
                    stats["crashed"] += 1
                    remaining -= 1
                    finished({"errors": 1, "retries": 0, "retry_events": []})
                respawns[0] += 1
                if respawns[0] > 2 * jobs:
                    raise PassStopped(f"{label}: workers keep dying ({respawns[0]} so far); stopped")
                if remaining:
                    start(max(procs) + 1)   # takes the stop marker the dead worker never reached
            if remaining and all(k in exited for k in procs):
                raise PassStopped(f"{label}: every worker exited with {remaining} scenarios left")
    finally:
        stop_all()


def check_names(items):
    """Trajectory names must be distinct: two ids that differ only in ':' against '_' would share one."""
    names = [traj_name(sc.scenario_id) for sc in items]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"scenario ids map to the same trajectory file: {dup[:5]}")


def clear(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def main_guard_env():
    """The loop settings an agent pass must run with their defaults: abort when one is set."""
    return [v for v in ("AGENT_TURN_STATUS", "AGENT_REPEAT_MODE", "AGENT_SUBMIT_GRACE_STEPS", "VOYAGER_TEMPERATURE")
            if v in os.environ]


if __name__ == "__main__":   # pragma: no cover
    sys.exit("run it through python -m src.eval.rerun agent")
