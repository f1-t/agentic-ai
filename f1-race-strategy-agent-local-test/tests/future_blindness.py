"""The future-blindness harness behind tests/test_future_blindness.py.

A policy decides at a snapshot: a race, a driver, after lap L. It may see what is known at
the end of lap L and nothing later. The harness checks that mechanically: it builds a race
twice, once as recorded and once with everything after lap L altered, and compares every
quantity a policy sees at the snapshot. Anything that differs reached the policy from the
future.

Where the alteration enters. A race reaches the eval through sim.jobs.load_race, which reads
one FastF1 race session (session.laps, session.total_laps, session.results) and derives
everything else from it: data.lap_frame's columns (the robust pace filter, gaps), the race
constants (calibrate._race_setup: the compound list; calibrate._pit_loss: the observed pit
loss) and the results. `build` runs those same functions with calibrate.load_session
answering the race session from a RaceInput, so the alteration is applied to the raw laps
before anything is derived from them. test_harness_builds_what_load_race_builds pins that
`build` on the recorded laps is jobs.load_race. Practice and qualifying come before the race
and are passed through untouched.

The past is defined by time. The snapshot moment T is when the snapshot driver crossed the
line to complete lap L (its lap-L row's Time; for L = 0, its lap-1 LapStartTime, the start).
A row is past when its lap ENDED at or before T (Time, else LapStartTime + LapTime; a row
with neither falls back to its lap number against L), per driver the prefix of such rows. So
a faster car's lap L + 1 that ended before T is past, and a slower car's lap L that ended
after T is future. The first future row of each car is its lap in progress at T: its
start-of-lap fields (LapStartTime, Stint, Compound, TyreLife, PitOutTime) are known at T, the
rest is not. Alterations keep every past row as it is, bit for bit and in order:

  truncate    the race stops at T: each car's lap in progress keeps only its start-of-lap
              fields, every later row is dropped, classified positions are unknown.
  scramble    everything after T is re-raced: every car's current stint ends on a different
              lap (longer or shorter), later stints take other compounds (HARD where the race
              ran none), sets start new or used, lap times follow another tyre model,
              pit-lane times change, the SC and VSC move, positions follow the new clock,
              every car but the winner gains or loses a lap (lapped or unlapped) and one car
              retires. The classification is shuffled. Seeded per (race, L).

test_snapshot_builds_from_the_past_alone checks the stricter cut, every row after T gone
(no lap in progress either), on which the snapshot builder must still work.
  aggregates  the race frame is untouched; the whole-race constants passed beside it are
              shifted: RaceParams.pit_loss_s + 6.5 s (the compound list is the nominated three since its fix), the
              classification reversed.

Policy-visible surfaces (observe), per snapshot:

  snapshot    the scenario's contracts.Snapshot (BUILDER, calibration.snapshot_as_of, the scenario
              builder), the agent's RaceSnapshot view (what run_agent shows the model) and
              agent_tools.contract_of(view)
  belief      the belief simulator's fit at the snapshot (score.race_simulators, belief.yaml):
              coefficients, covariances, sds, sources, flags and priors in fit_report, the
              hazard, ref_lap_s, the calibrator's output and practice prior, the modelled
              compounds, the R8 limits and their sources; its model report; simulated times of
              a fixed plan set
  rules       the belief's rule verdicts (every error string) on a fixed script of plans: a
              single compound, too-short stints, too-long stints on every dry compound, and
              sweeps of one- and two-stop plans
  tools       every src/tools.py tool, as the agent calls it, on a fixed script (all compounds,
              illegal plans and actions included), plus agent_tools.race_state and
              simulate_plan on the illegal plans
  decisions   the belief_optimizer plan and the continuation of PIT onto each dry compound
              and of STAY_OUT 3 (eval.decisions, as score_scenario expands them)
  baselines   fixed_one_stop, fixed_two_stop, greedy_crossover, run_to_flag and
              one_stop_total through the eval wrapper (eval.policies): plan and meta.
              actual_strategy is left out: it is the real future by definition.
  traffic     the belief with traffic on (configs/sim/belief_traffic.yaml, the projected
              field, src/sim/traffic.py): the projected ghost table of every variant, own pace
              and dirty air, a plan set's times and positions, get_model_report, compare_plans
              and evaluate_strategy as shown, the belief optimum and run_to_flag planned with it;
              and the same belief with the as-of passing difficulty (configs/sim/
              belief_difficulty.yaml): its threshold and source races, a plan set's times and
              positions, get_model_report and evaluate_strategy as shown

The scoring world and the hindsight reference are allowed to see the whole race and are
not observed.

Known leaks. Each entry of KNOWN_LEAKS names an upstream quantity that carries the future to
a policy and a `neutralise` that restores the recorded race's value of that quantity in the
altered race. test_future_blind runs every alteration with every known leak neutralised, so
it passes today and fails on any new leak. test_leak_<name> neutralises every other known leak
and is an expected failure (xfail strict): when the fix lands it passes, pytest reports the
unexpected pass, and the fix removes the xfail and the KNOWN_LEAKS entry.

What it cannot see: anything that does not move when this race's laps after L move. Tables
and constants fitted or tuned with hindsight (configs/sim/compound_steps.json,
reports/calibration/midrace_summary.csv behind calibration_error_s, belief.yaml's SC rate and
pre_race_base_delta_s taken from 2021 Spain, cross-race and event-rate tables built on later
seasons, any choice developers tuned on these races); the lap L + 1 row's start-of-lap fields,
which both frame alterations keep as known; how scenarios are chosen (decision laps beside SC
periods, finishers only), which conditions the set on the future without reaching a policy;
frozen scenario sets, which keep the snapshots they were built with; what an LLM already knows
about a real race; surfaces not observed (weather is absent belief-side).
A dependence that happens to leave every observed output unchanged at the sampled snapshots
is missed too: a pass is evidence, not proof.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from src import tools
from src.contracts import Snapshot
from src.eval import decisions, policies, score
from src.eval.scenarios import Scenario
from src.sim import agent_tools, calibrate, calibration, engine, jobs, rules
from src.sim.compute import load_profile
from src.sim.config import SimConfig
from src.sim.data import session_results
from src.sim.race_ids import parse_race_id
from src.snapshot import RaceSnapshot

RAW_COLUMNS = ["Driver", "LapNumber", "Stint", "Compound", "TyreLife", "LapTime", "TrackStatus", "PitInTime",
               "PitOutTime", "LapStartTime", "Time", "Team", "Position"]
RESULT_COLUMNS = ["Abbreviation", "GridPosition", "ClassifiedPosition"]
DRY = ("SOFT", "MEDIUM", "HARD")
MAX_STOPS = 2
N_SAMPLES = 8
LOCAL = load_profile("local")
PROFILE = LOCAL.model_copy(update={"sim": LOCAL.sim.model_copy(update={"n_samples": N_SAMPLES,
                                                                           "compare_n_samples": None,
                                                                           "max_stops": MAX_STOPS})})
REL_TOL, ABS_TOL = 1e-9, 1e-9
ALTERATIONS = ("truncate", "scramble", "aggregates")
SURFACES = ("snapshot", "belief", "rules", "tools", "decisions", "baselines", "traffic")
OBSERVED_BASELINES = ("fixed_one_stop", "fixed_two_stop", "greedy_crossover", "run_to_flag", "one_stop_total")
# the scenario builder new sets use (src/eval/scenarios.py, rule as_of); calibration.snapshot_at
# is the legacy one (the race_length and gap_behind leaks, and it needs a lap L + 1 row)
BUILDER = calibration.snapshot_as_of


# --- race input ---------------------------------------------------------------------------
@dataclass
class RaceInput:
    """One race as it enters jobs.load_race: the race session's raw laps (RAW_COLUMNS), its
    scheduled length and results (RESULT_COLUMNS), and the weekend's practice and qualifying
    frames, which are before the race and never altered."""
    race_id: str
    laps: pd.DataFrame
    total_laps: int
    results: pd.DataFrame
    practice: pd.DataFrame | None
    qualifying: pd.DataFrame | None


def real_input(race_id: str) -> RaceInput:
    """The committed cache's race, read as jobs.load_race reads it."""
    from src.sim.data import load_session
    season, rnd, _ = parse_race_id(race_id)
    session = load_session(season, rnd)
    laps = pd.DataFrame({c: session.laps[c].to_numpy() for c in RAW_COLUMNS})
    results = pd.DataFrame({c: session.results[c].to_numpy() for c in RESULT_COLUMNS})
    cfg = SimConfig(degradation={"name": SimConfig().degradation.name})   # calibrate._config, as load_race builds it
    return RaceInput(race_id, laps, int(session.total_laps), results, calibrate._practice(season, rnd, cfg.long_run),
                     calibrate._qualifying(season, rnd))


def raw_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw laps (RAW_COLUMNS) from a frame in data.lap_frame's base columns, e.g. a synthetic
    race from tests/test_sim_position.race_frame: the inverse of data._columns."""
    def td(x):
        return pd.to_timedelta(pd.Series(np.asarray(x, dtype=float)), unit="s")
    pit_in, pit_out = df["pit_in"].to_numpy(dtype=bool), df["pit_out"].to_numpy(dtype=bool)
    end, start = td(df["end_s"]), td(df["start_s"])
    return pd.DataFrame({
        "Driver": df["driver"].to_numpy(), "LapNumber": df["lap"].to_numpy(dtype=float),
        "Stint": df["stint"].to_numpy(dtype=float), "Compound": df["compound"].to_numpy(),
        "TyreLife": df["tyre_age"].to_numpy(dtype=float), "LapTime": td(df["lap_time_s"]),
        "TrackStatus": df["track_status"].astype(str).to_numpy(),
        "PitInTime": end.where(pit_in), "PitOutTime": start.where(pit_out),
        "LapStartTime": start, "Time": end, "Team": df["team"].to_numpy(),
        "Position": df["position"].to_numpy(dtype=float)})


def synthetic_input(race_id="2099_01_synthetic", seed=0) -> RaceInput:
    """A seven-car, 50-lap synthetic race from the test suite's builders
    (test_sim_position.race_frame on test_sim.TRUE, test_sim.synthetic_practice): one- and
    two-stop cars on SOFT and MEDIUM, a backmarker (GGG) lapped by the end, lap noise, an SC
    on laps 31-33, positions from the clock. No car runs HARD, so the compound proxy prices it from practice and the
    cross-race step."""
    from test_sim import synthetic_practice
    from test_sim_position import race_frame
    df = race_frame(("AAA", "T1", [("SOFT", 15), ("MEDIUM", 35)]),
                    ("BBB", "T1", [("SOFT", 10), ("MEDIUM", 40)], 0.2),
                    ("CCC", "T2", [("SOFT", 25), ("MEDIUM", 25)], 0.4),
                    ("DDD", "T2", [("MEDIUM", 30), ("SOFT", 20)], 0.1),
                    ("EEE", "T3", [("MEDIUM", 20), ("SOFT", 12), ("SOFT", 18)], 0.3),
                    ("FFF", "T3", [("SOFT", 12), ("MEDIUM", 20), ("SOFT", 18)], 0.5),
                    ("GGG", "T4", [("MEDIUM", 25), ("SOFT", 25)], 2.0))
    df = df[~((df["driver"] == "GGG") & (df["lap"] == 50))]   # a backmarker, lapped by the end
    rng = np.random.default_rng(seed)
    df["lap_time_s"] = df["lap_time_s"] + rng.normal(0.0, 0.3, len(df))
    sc = df["lap"].between(31, 33)
    df.loc[sc, "track_status"] = "4"
    df.loc[sc, "lap_time_s"] *= 1.35
    df = _reclock(df.sort_values(["driver", "lap"]).reset_index(drop=True))
    results = (df.sort_values("lap").groupby("driver").tail(1).sort_values(["lap", "end_s"], ascending=[False, True]))
    res = pd.DataFrame({"Abbreviation": results["driver"].to_numpy(), "GridPosition": np.arange(1, len(results) + 1.0),
                        "ClassifiedPosition": [str(i) for i in range(1, len(results) + 1)]})
    return RaceInput(race_id, raw_from_frame(df), 50, res, synthetic_practice(drivers=("AAA", "CCC", "EEE")), None)


def _reclock(df):
    """start_s / end_s cumulative from the lap times per driver, positions by the clock per lap."""
    df = df.copy()
    df["end_s"] = df.groupby("driver")["lap_time_s"].cumsum()
    df["start_s"] = df["end_s"] - df["lap_time_s"]
    df["position"] = df.groupby("lap")["end_s"].rank(method="first")
    return df


# --- building a race the way jobs.load_race does ------------------------------------------
@contextlib.contextmanager
def _race_session(inp: RaceInput):
    """calibrate.load_session answers the race session from `inp`; any other session is an error."""
    session = SimpleNamespace(laps=inp.laps, total_laps=inp.total_laps, results=inp.results)

    def load_session(season, gp, kind="R", *a, **k):
        if kind != "R":
            raise AssertionError(f"the harness passes practice and qualifying through, asked for {kind}")
        return session
    with mock.patch.object(calibrate, "load_session", load_session):
        yield session


def build(inp: RaceInput, profile=PROFILE, *, altered=False) -> jobs.RaceData:
    """jobs.load_race's body on `inp`: the frame (data.lap_frame), the race constants
    (calibrate._race_setup, the observed pit loss of calibrate._pit_loss), the results;
    practice and qualifying passed through. With `altered`, RuntimeWarnings are silenced:
    the pit loss replay's summary statistics (calibration.run's rank correlations) warn on a
    truncated race, which has too few distinct finishing times; nothing in the recorded race
    is silenced."""
    season, rnd, _ = parse_race_id(inp.race_id)
    args = SimpleNamespace(season=season, gp=rnd, pit_loss="observed", degradation=SimConfig().degradation.name,
                           config=None, min_stint=jobs.race_min_stint(profile))
    with _race_session(inp), warnings.catch_warnings():
        if altered:
            warnings.simplefilter("ignore", RuntimeWarning)
        session, frame, cfg, race = calibrate._race_setup(args)
        try:
            pit_loss, _ = calibrate._pit_loss(args, frame, cfg, race)
        except ValueError:
            # a race truncated at the start has no lap to replay: the pit loss is not
            # observable, so the race carries the pipeline's first-pass value
            if not altered:
                raise
            pit_loss = calibrate.DEFAULT_PIT_LOSS_S
    race = race.model_copy(update={"pit_loss_s": pit_loss, "race_id": inp.race_id})
    return jobs.RaceData(inp.race_id, frame, race, inp.practice, session_results(session), cfg, inp.qualifying)


# --- alterations -----------------------------------------------------------------------------
def _td(x):
    return pd.to_timedelta(x, unit="s")


TIME_COLUMNS = ["LapTime", "PitInTime", "PitOutTime", "LapStartTime", "Time"]


def _typed(laps: pd.DataFrame) -> pd.DataFrame:
    """Raw laps with the recorded dtypes back (row-wise edits leave object columns)."""
    laps = laps.copy()
    for c in TIME_COLUMNS:
        laps[c] = pd.to_timedelta(laps[c])
    for c in ("LapNumber", "Stint", "TyreLife", "Position"):
        laps[c] = pd.to_numeric(laps[c]).astype(float)
    return laps


def _concat(frames) -> pd.DataFrame:
    """pd.concat of raw laps with the time columns carried as int64 nanoseconds, so all-NaT
    blocks concatenate without pandas' generic-unit NaT (a DeprecationWarning) and every value
    comes back bit for bit."""
    def ns(f):
        f = _typed(f)
        for c in TIME_COLUMNS:
            f[c] = f[c].to_numpy(dtype="timedelta64[ns]").view("i8")
        return f
    out = pd.concat([ns(f) for f in frames])
    for c in TIME_COLUMNS:
        out[c] = pd.to_timedelta(out[c].to_numpy(dtype="i8").view("m8[ns]"))
    return out


def moment(laps: pd.DataFrame, driver: str, lap: int) -> pd.Timedelta:
    """The snapshot moment T: when `driver` crossed the line completing lap `lap` (its lap-L
    row's Time), or for lap 0 its lap-1 LapStartTime, the start."""
    g = laps[laps["Driver"] == driver]
    if lap == 0:
        t = g.loc[g["LapNumber"] == 1, "LapStartTime"]
        t = t if len(t) and pd.notna(t.iloc[0]) else laps.loc[laps["LapNumber"] == 1, "LapStartTime"].dropna()
        return t.min()
    row = g[g["LapNumber"] == lap].iloc[0]
    return row["Time"] if pd.notna(row["Time"]) else row["LapStartTime"] + row["LapTime"]


def is_past(laps: pd.DataFrame, driver: str, lap: int) -> pd.Series:
    """Rows whose lap ended at or before the snapshot moment (module docstring): per car the
    prefix, in lap order, of rows that ended by T. A row with no end time falls back to its lap
    number against `lap`."""
    t = moment(laps, driver, lap)
    end = laps["Time"].fillna(laps["LapStartTime"] + laps["LapTime"])
    # no end time: a lap with a start is in progress (not ended); with neither, the lap number decides
    unknown = (laps["LapNumber"] <= lap).where(laps["LapStartTime"].isna(), False)
    ended = (end <= t).where(end.notna(), unknown).astype(bool)
    order = laps.sort_values(["Driver", "LapNumber"], kind="mergesort")
    prefix = (~ended.loc[order.index]).groupby(order["Driver"]).cumsum() == 0
    return prefix.reindex(laps.index)


def _current(laps: pd.DataFrame, past: pd.Series) -> pd.Series:
    """Each car's first future row: its lap in progress at the snapshot moment."""
    future = laps[~past].sort_values(["Driver", "LapNumber"], kind="mergesort")
    first = future.groupby("Driver").head(1).index
    return pd.Series(laps.index.isin(first), index=laps.index)


def _start_of_lap(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows reduced to what is known when the lap starts (module docstring)."""
    rows = rows.copy()
    for c in ("LapTime", "Time", "PitInTime"):
        rows[c] = pd.Series(pd.NaT, index=rows.index, dtype="timedelta64[ns]")
    rows["Position"] = np.nan
    rows["TrackStatus"] = ""
    return rows


def truncate(inp: RaceInput, driver: str, lap: int, seed=None) -> RaceInput:
    raw = inp.laps
    past = is_past(raw, driver, lap)
    now = _current(raw, past)
    laps = _concat([raw[past], _start_of_lap(raw[now])]).sort_index().reset_index(drop=True)
    results = inp.results.copy()
    results["ClassifiedPosition"] = None
    return dataclasses.replace(inp, laps=laps, results=results)


def cut(inp: RaceInput, driver: str, lap: int, seed=None) -> RaceInput:
    """The strict cut: only rows that ended by the snapshot moment, no lap in progress."""
    raw = inp.laps
    laps = _concat([raw[is_past(raw, driver, lap)]]).reset_index(drop=True)
    results = inp.results.copy()
    results["ClassifiedPosition"] = None
    return dataclasses.replace(inp, laps=laps, results=results)


def scramble(inp: RaceInput, driver: str, lap: int, seed=0) -> RaceInput:
    """Everything after the snapshot moment re-raced (module docstring). Past rows are kept as
    they are; each car's lap in progress keeps its start-of-lap state."""
    rng = np.random.default_rng([seed, lap, sum(map(ord, inp.race_id))])
    raw = inp.laps.copy()
    past = is_past(raw, driver, lap)
    now = _current(raw, past)
    moment_t = moment(raw, driver, lap)
    total = inp.total_laps
    counts = raw.groupby("Driver")["LapNumber"].max()
    finish = raw[raw["LapNumber"] == raw.groupby("Driver")["LapNumber"].transform("max")]
    winner = finish.sort_values(["LapNumber", "Time"], ascending=[False, True])["Driver"].iloc[0]
    # the re-raced tyre model: other bases, slopes and fuel, and other pit-lane times
    base = {c: 80.0 + rng.uniform(0.0, 12.0) for c in DRY}
    slope = {c: rng.uniform(0.02, 0.25) for c in DRY}
    fuel = rng.uniform(-0.09, -0.01)
    offset = {d: rng.normal(0.0, 0.6) for d in counts.index}
    pit_in_s, pit_out_s = rng.uniform(8.0, 25.0), rng.uniform(8.0, 20.0)
    sc_start = int(rng.integers(lap + 2, max(lap + 3, total - 4)))
    vsc_start = int(rng.integers(lap + 2, max(lap + 3, total - 3)))
    status = {x: "1" for x in range(0, total + 2)}
    status.update({x: "6" for x in range(vsc_start, vsc_start + 2)})
    status.update({x: "4" for x in range(sc_start, sc_start + 3)})
    unraced = [c for c in DRY if c not in set(raw.loc[past, "Compound"].dropna())]
    starts = raw[now].set_index("Driver")["LapNumber"].astype(int)
    alive = [d for d in sorted(starts.index) if counts[d] > starts[d] + 3 and d != winner]
    retiree = alive[int(rng.integers(len(alive)))] if alive else None
    parts = [raw[past]]
    for d, g in raw.groupby("Driver", sort=True):
        g = g.sort_values("LapNumber")
        first = g[now.loc[g.index]]
        if first.empty:
            continue
        s = int(first["LapNumber"].iloc[0])   # this car's lap in progress at the moment
        n = int(counts[d])
        if d != winner:   # lapped or unlapped in the future
            n = n - 1 if n >= total else n + 1
        if d == retiree:
            n = int(rng.integers(s + 1, max(s + 2, n)))
        n = max(n, s + 1)
        start = first.iloc[0]
        stint, comp, age = start["Stint"], start["Compound"], start["TyreLife"]
        if pd.isna(age):
            age = 1.0
        current_left = int(rng.integers(1, max(2, n - s + 1)))   # laps of the current stint from lap s
        clock = start["LapStartTime"]
        if pd.isna(clock):
            before = g.loc[past.loc[g.index], "Time"].dropna()
            clock = before.iloc[-1] if len(before) else _td(0.0)
        rows, left_in_stint = [], current_left
        for x in range(s, n + 1):
            out_lap = x > s and left_in_stint == 0
            if out_lap:   # a stop at the end of the previous lap: a new stint
                stint = (stint if not pd.isna(stint) else 1.0) + 1.0
                comp = (unraced + list(DRY))[int(rng.integers(len(unraced) + len(DRY)))] if unraced else \
                    DRY[int(rng.integers(len(DRY)))]
                age = 1.0 if rng.uniform() < 0.7 else float(rng.integers(2, 5))
                left_in_stint = int(rng.integers(4, max(5, n - x + 2)))
            in_lap = left_in_stint == 1 and x < n
            st = status.get(x, "1")
            t = base.get(comp, 85.0) + slope.get(comp, 0.1) * age + fuel * x + offset[d] + rng.normal(0.0, 0.4)
            if st == "4":
                t *= 1.4
            elif st == "6":
                t *= 1.3
            t += (pit_in_s if in_lap else 0.0) + (pit_out_s if out_lap else 0.0)
            lap_start, lap_end = clock, clock + _td(t)
            if x == s and lap_end <= moment_t:   # a lap in progress at the moment ends after it
                lap_end = moment_t + _td(rng.uniform(0.05, 2.0))
                t = (lap_end - clock).total_seconds()
            if x == s:   # the state at the start of the lap in progress is known at the moment
                row = start.copy()
                row["LapTime"], row["Time"], row["TrackStatus"] = _td(t), lap_end, st
                row["PitInTime"] = lap_end if in_lap else pd.NaT
            else:
                row = start.copy()
                row["LapNumber"], row["Stint"], row["Compound"], row["TyreLife"] = float(x), stint, comp, age
                row["LapTime"], row["Time"], row["LapStartTime"], row["TrackStatus"] = _td(t), lap_end, lap_start, st
                row["PitInTime"] = lap_end if in_lap else pd.NaT
                row["PitOutTime"] = lap_start if out_lap else pd.NaT
            rows.append(row)
            clock, age, left_in_stint = lap_end, age + 1.0, left_in_stint - 1
        parts.append(pd.DataFrame(rows))
    laps = _concat(parts).reset_index(drop=True)
    later = np.ones(len(laps), dtype=bool)
    later[:int(past.sum())] = False   # parts[0] is the past
    ranks = laps.groupby("LapNumber")["Time"].rank(method="first")
    laps.loc[later, "Position"] = ranks[later].to_numpy(dtype=float)
    # the recorded row order (drivers as first seen, then laps), so the rows up to `lap` reach
    # every fit in the same order and sum bit for bit as they do in the recorded race
    rank = {d: i for i, d in enumerate(pd.unique(raw["Driver"]))}
    laps = (laps.assign(_rank=laps["Driver"].map(rank)).sort_values(["_rank", "LapNumber"], kind="mergesort")
            .drop(columns="_rank").reset_index(drop=True))
    results = inp.results.copy()
    results["ClassifiedPosition"] = rng.permutation(results["ClassifiedPosition"].to_numpy())
    return dataclasses.replace(inp, laps=laps, results=results)


def past_rows(laps: pd.DataFrame, driver: str, lap: int) -> pd.DataFrame:
    """The past rows (is_past), in order, index reset: what an alteration keeps."""
    return laps[is_past(laps, driver, lap)].reset_index(drop=True)


def shift_aggregates(data: jobs.RaceData) -> jobs.RaceData:
    """`data` with the whole-race constants passed beside the frame shifted (module docstring).
    RaceParams.compounds is no longer one: since the compound_list fix it is the weekend's
    nominated dry compounds (calibrate.race_compounds), a regulation constant."""
    race = data.race.model_copy(update={"pit_loss_s": data.race.pit_loss_s + 6.5})
    results = data.results.copy()
    results["classified"] = results["classified"].to_numpy()[::-1]
    return dataclasses.replace(data, race=race, results=results)


FRAME_ALTERATIONS = {"truncate": truncate, "scramble": scramble}


# --- known leaks: the upstream quantity, and how to restore the recorded race's value -----------
@dataclass
class Leak:
    """An upstream quantity that carries the future to a policy. `data` restores it on the
    altered RaceData, `snapshot` on the altered snapshot, `patches` while the altered race is
    observed. `alteration` is the one test_leak_<name> runs; `where` is the code path."""
    name: str
    what: str
    where: str
    alteration: str
    data: object = None       # fn(original RaceData, altered RaceData, lap) -> RaceData
    snapshot: object = None   # fn(original Snapshot, altered Snapshot) -> Snapshot
    patches: object = None    # fn(original RaceData) -> list of context managers


def _restore_frame_columns(cols):
    def restore(orig, alt, lap):
        frame = alt.frame.copy()
        key = ["driver", "lap"]
        mine = frame[frame["lap"] <= lap][key].reset_index()
        src = orig.frame[orig.frame["lap"] <= lap].set_index(key)[cols]
        got = src.reindex(pd.MultiIndex.from_frame(mine[key]))
        for c in cols:
            frame.loc[mine["index"].to_numpy(), c] = got[c].to_numpy()
        return dataclasses.replace(alt, frame=frame)
    return restore


FILTER_COLUMNS = ["clean", "pace_resid_s", "driver_offset_s"]


def _restore_lap_window(orig, alt, lap):
    """Every recorded row with lap <= L back in the frame (rows of other cars that end after
    the moment included), in recorded order, the filter columns (frame_filter) as the altered
    build has them where it has the row."""
    key = ["driver", "lap"]
    rec = orig.frame[orig.frame["lap"] <= lap].copy()
    a = alt.frame
    mine = a[a["lap"] <= lap].set_index(key)[FILTER_COLUMNS]
    idx = pd.MultiIndex.from_frame(rec[key])
    present = idx.isin(mine.index)
    got = mine.reindex(idx)
    for c in FILTER_COLUMNS:
        rec[c] = np.where(present, got[c].to_numpy(dtype=object), rec[c].to_numpy(dtype=object))
        rec[c] = rec[c].astype(orig.frame[c].dtype)
    frame = pd.concat([rec, a[a["lap"] > lap]], ignore_index=True)
    return dataclasses.replace(alt, frame=frame)


def _restore_gap_behind(orig_snap, alt_snap):
    """The recorded car-behind fields: gap_behind_s and the nearby entry behind (gap_s >= 0)."""
    nearby = [c for c in alt_snap.field.nearby if c.gap_s < 0] + [c for c in orig_snap.field.nearby if c.gap_s >= 0]
    return alt_snap.model_copy(update={"gap_behind_s": orig_snap.gap_behind_s,
                                       "field": alt_snap.field.model_copy(update={"nearby": nearby})})


def _restore_race(field_name):
    def restore(orig, alt, lap):
        return dataclasses.replace(alt, race=alt.race.model_copy(update={field_name: getattr(orig.race, field_name)}))
    return restore


def _restore_total_laps(orig_snap, alt_snap):
    return alt_snap.model_copy(update={"race": alt_snap.race.model_copy(update={"total_laps": orig_snap.race.total_laps})})


KNOWN_LEAKS: dict[str, Leak] = {}
# leaks whose fix has landed: never neutralised; test_leak_<name> stays as the regression check
FIXED_LEAKS: dict[str, Leak] = {}


def register(leak: Leak, fixed: bool = False):
    (FIXED_LEAKS if fixed else KNOWN_LEAKS)[leak.name] = leak
    return leak


# Most severe first (by how far they move what the belief and the policies plan).
register(Leak(
    "frame_filter", "data.lap_frame's clean flag (with pace_resid_s and driver_offset_s) comes from one robust "
    "pace fit over every lap of the race, so which laps before the snapshot count as clean depends on later laps",
    "src/sim/data.py:316 lap_frame -> robust_filter; read by the to_snapshot fit (src/sim/degradation.py:553) "
    "and ref_lap_s (src/sim/engine.py:277)", "scramble",
    data=_restore_frame_columns(["clean", "pace_resid_s", "driver_offset_s"])),
    fixed=True)   # data.as_of_columns on the visible rows (SimConfig.policy_view.clean)
# Fixed: "pit_loss" (RaceParams.pit_loss_s, the observed pit loss of a full-race replay). Snapshots
# carry the pre-race prior (calibration.pre_race_pit_loss), the belief and the tools the as-of
# estimate (src/sim/pit_loss.py); only the world gets the whole-race value. test_leak_pit_loss
# keeps it as a regression test.
register(Leak(
    "pit_loss", "RaceParams.pit_loss_s is the observed pit loss of a full-race replay",
    "src/sim/jobs.py:98 load_race -> src/sim/calibrate.py:137 _pit_loss (calibration.run observed_pit_loss_s) -> "
    "snapshot.race; src/sim/engine.py:500, src/sim/agent_tools.py:383", "scramble", data=_restore_race("pit_loss_s")),
    fixed=True)
register(Leak(
    "stint_limit", "R8's stint limit was the oldest tyre age any car ran on the compound in the whole race "
    "(and, under the compound proxy, whether a compound counts as raced at all)",
    "retired (docs/rules.md R8): src/sim/rules.py stint_limits and proxy.unraced_limits run only under the "
    "legacy setting; tyre life is the wear cliff, the belief's edge from what is visible at the snapshot "
    "(src/sim/cliff.py as_of_cliffs, Simulator.cliffs_for)", "scramble"),
    fixed=True)
register(Leak(
    "race_length", "the snapshot's race length is the laps the driver drove in the whole race, so a driver "
    "who will be lapped (or will retire) is told fewer laps remain",
    "src/sim/calibration.py:591 snapshot_at: race.model_copy(update={'total_laps': len(g)})", "scramble",
    snapshot=_restore_total_laps), fixed=True)   # calibration.snapshot_as_of: the scheduled distance
register(Leak(
    "lap_window", "the belief's to_snapshot window (and the calibrator's laps so far) takes race laps by lap "
    "number, lap <= L, so a slower car's lap L that ends after the snapshot moment is in the fit",
    "src/sim/engine.py:374 _fit_window race[race['lap'] <= k0]; src/sim/engine.py:340 _calibrate", "scramble",
    data=_restore_lap_window), fixed=True)   # Simulator.window_rows: cut by time (policy_view.window_cut)
register(Leak(
    "gap_behind", "the snapshot's gap to the car behind (and that car in field.nearby) is the car behind's line "
    "crossing, which happens after the snapshot moment",
    "src/sim/calibration.py:585 snapshot_at _gaps(g, k + 1) gap_behind_s; :611 _nearby car behind", "scramble",
    snapshot=_restore_gap_behind), fixed=True)   # snapshot_as_of: the car behind at the previous crossing
register(Leak(
    "compound_list", "RaceParams.compounds is every compound in the whole race frame",
    "src/sim/calibrate.py:123 _race_setup compounds=sorted(frame['compound'].unique()) -> src/snapshot.py:84 "
    "(view available_compounds), src/sim/agent_tools.py:255 legal_compounds, src/tools.py:101, "
    "src/sim/model_report.py:98", "scramble", data=_restore_race("compounds")),
    fixed=True)   # calibrate.race_compounds: the nominated dry three


# --- one altered race, with known leaks neutralised ---------------------------------------------
def altered(inp: RaceInput, orig: jobs.RaceData, alteration: str, driver: str, lap: int, neutralise=()):
    """(RaceData, snapshot_fix(original snapshot) -> fn, patches) for `alteration` at the
    moment `driver` completes `lap`, with the KNOWN_LEAKS named in `neutralise` restored to the
    recorded race's values, in KNOWN_LEAKS order."""
    if alteration == "aggregates":
        data = shift_aggregates(orig)
    else:
        data = build({**FRAME_ALTERATIONS, "cut": cut}[alteration](inp, driver, lap), altered=True)
    leaks = [lk for n, lk in KNOWN_LEAKS.items() if n in set(neutralise)]
    for lk in leaks:
        if lk.data is not None:
            data = lk.data(orig, data, lap)
    patches = [cm for lk in leaks if lk.patches is not None for cm in lk.patches(orig)]
    fixes = [lk.snapshot for lk in leaks if lk.snapshot is not None]

    def snapshot_fix(orig_snap):
        def fix(snap):
            for f in fixes:
                snap = f(orig_snap, snap)
            return snap
        return fix if fixes else None
    return data, snapshot_fix, patches


# --- observing what a policy sees ------------------------------------------------------------
def plain(x, depth=0, seen=None):
    """A comparable plain value: pydantic models dumped, numpy to lists, objects to their
    public attributes, NaN as the string "nan", callables and private attributes left out."""
    seen = set() if seen is None else seen
    if depth > 12:
        return f"<depth {type(x).__name__}>"
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if hasattr(x, "model_dump"):
        return plain(x.model_dump(), depth + 1, seen)
    if isinstance(x, dict):
        return {str(k): plain(v, depth + 1, seen) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = sorted(x, key=str) if isinstance(x, (set, frozenset)) else x
        return [plain(v, depth + 1, seen) for v in items]
    if isinstance(x, np.ndarray):
        return plain(x.tolist(), depth + 1, seen)
    if isinstance(x, pd.DataFrame):
        return plain({"columns": list(map(str, x.columns)), "index": [str(i) for i in x.index],
                      "values": x.astype(object).to_numpy().tolist()}, depth + 1, seen)
    if isinstance(x, pd.Series):
        return plain(x.to_dict(), depth + 1, seen)
    if callable(x) and not hasattr(x, "__dict__"):
        return f"<callable {type(x).__name__}>"
    if id(x) in seen:
        return f"<seen {type(x).__name__}>"
    if dataclasses.is_dataclass(x):
        seen = seen | {id(x)}
        return {"__type__": type(x).__name__,
                **{f.name: plain(getattr(x, f.name), depth + 1, seen) for f in dataclasses.fields(x)}}
    if hasattr(x, "__dict__"):
        seen = seen | {id(x)}
        attrs = {k: v for k, v in vars(x).items() if not k.startswith("_") and not callable(v)}
        return {"__type__": type(x).__name__, **{k: plain(v, depth + 1, seen) for k, v in sorted(attrs.items())}}
    return repr(x)


def _err(exc):
    return {"raised": type(exc).__name__, "message": str(exc)}


def _call(fn, *a, **k):
    try:
        return plain(fn(*a, **k))
    except Exception as exc:   # noqa: BLE001  an error is an output like any other
        return _err(exc)


def _other(compound):
    return next(c for c in DRY if c != compound)


def plan_script(snap: Snapshot) -> dict:
    """The fixed plans the rules and the tools are asked about, from the snapshot alone."""
    cur, rest = snap.compound, snap.remaining_laps
    other = _other(cur)
    out = {"single_compound": [[cur, rest]]}
    if rest >= 2:
        out["too_short_first"] = [[cur, 1], [other, rest - 1]]
        out["too_short_last"] = [[cur, rest - 1], [other, 1]]
    if rest > 6:
        out["too_long_first"] = [[cur, rest - 3], [other, 3]]
        for c in DRY:
            out[f"too_long_{c}"] = [[cur, 3], [c, rest - 3]]
        half = rest // 2
        for c in DRY:
            out[f"one_stop_half_{c}"] = [[cur, half], [c, rest - half]]
        third = rest // 3
        out["two_stop_thirds"] = [[cur, third], [other, third], [cur if cur != other else _other(other), rest - 2 * third]]
    return out


def _rules_surface(belief, snap):
    script = plan_script(snap)
    out = {f"script/{k}": _call(belief.validate, snap, _plan(p)) for k, p in script.items()}
    rest, cur = snap.remaining_laps, snap.compound
    sweep1 = {}
    for c in DRY:
        for a in range(1, rest):
            sweep1[f"{c}:{a}"] = _call(belief.validate, snap, _plan([[cur, a], [c, rest - a]]))
    sweep2 = {}
    step = max(1, rest // 12)
    for c1 in DRY:
        for c2 in DRY:
            for a in range(1, rest - 1, step):
                for b in range(1, rest - a, step):
                    sweep2[f"{c1}:{c2}:{a}:{b}"] = _call(belief.validate, snap,
                                                        _plan([[cur, a], [c1, b], [c2, rest - a - b]]))
    out["sweep_one_stop"], out["sweep_two_stop"] = sweep1, sweep2
    return out


def _plan(pairs):
    from src.contracts import Plan
    return Plan.of(pairs)


def _belief_surface(belief, snap):
    fit = belief.fitted_for(snap)
    plans = [p for k, p in plan_script(snap).items() if k.startswith(("one_stop", "two_stop", "single"))]
    batch = belief.simulate_batch(snap, plans, seed=agent_tools.SEED, n_samples=N_SAMPLES)
    return {
        "modelled": plain(sorted(fit.modelled)),
        "degradation": plain(fit.degradation),
        "hazard": plain(fit.hazard),
        "ref_lap_s": plain(fit.ref_lap_s),
        "calibration": plain(fit.calibration),
        "stint_limits": plain(belief.stint_limits),
        "stint_limit_sources": plain(belief.stint_limit_sources),
        "model_report": _call(belief.model_report, snap, {c: 40 for c in DRY}),
        "simulate_batch": {"plans": plain(batch.plans), "errors": plain(batch.errors), "times": plain(batch.times),
                           "support_warnings": plain(batch.support_warnings)},
    }


def _shown(fn, *a):
    """A src/tools.py output (or the error it raised) as the model is shown it: through
    tools.present, as agent._dispatch passes every tool output and error string."""
    try:
        return plain(tools.present(plain(fn(*a))))
    except Exception as exc:   # noqa: BLE001  an error is an output like any other
        return plain(tools.present(_err(exc)))


# entries of the tools surface that are agent_tools internals, not a tool the model calls
MODEL_UNSEEN_TOOLS = ("race_state", "simulate_plan/")


def _tools_surface(snap):
    """src/tools.py as the agent calls it (and is shown it, _shown), then agent_tools' race_state
    and simulate_plan, which no model sees (MODEL_UNSEEN_TOOLS)."""
    tools.set_snapshot(snap)
    cur, rest = snap.compound, snap.remaining_laps
    script = plan_script(snap)
    out = {"get_race_snapshot": _shown(tools.get_race_snapshot),
           "estimate_pit_loss": _shown(tools.estimate_pit_loss),
           "get_model_report": _shown(tools.get_model_report, {c: 40 for c in DRY})}
    for c in (*DRY, "INTERMEDIATE"):
        out[f"estimate_tyre_degradation/{c}"] = _shown(tools.estimate_tyre_degradation, c)
    cands = [script[k] for k in script if not k.startswith("two_stop")]
    cands += [{"action": "pit", "compound": c} for c in DRY] + [{"action": "stay_out", "laps": 3}]
    out["compare_plans"] = _shown(tools.compare_plans, cands)
    for c in DRY:
        out[f"evaluate_strategy/pit_{c}"] = _shown(tools.evaluate_strategy, {"action": "pit", "compound": c})
    for n in (1, 3):
        out[f"evaluate_strategy/stay_out_{n}"] = _shown(tools.evaluate_strategy, {"action": "stay_out", "laps": n})
    out["race_state"] = _call(agent_tools.race_state, snap)
    for k, p in script.items():
        if k.startswith(("too_", "single")):
            out[f"simulate_plan/{k}"] = _call(agent_tools.simulate_plan, snap, p)
    return out


def _decisions_surface(belief, snap):
    out = {"belief_optimizer": _call(lambda: decisions.belief_optimum(snap, belief, MAX_STOPS, seed=0,
                                                                     n_samples=N_SAMPLES).as_pairs())}
    subs = {f"PIT_{c}": {"decision": "PIT", "compound": c, "confidence": 0.6} for c in DRY}
    subs["STAY_OUT_3"] = {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6}
    for k, sub in subs.items():
        out[k] = _call(lambda s=sub: decisions.to_plan(snap, s, belief, MAX_STOPS, seed=0, n_samples=N_SAMPLES).as_pairs())
    return out


def _traffic_surface(data, snap, scenario):
    """The belief with traffic on (configs/sim/belief_traffic.yaml, the projected field): its
    traffic state (the projected ghost table of every variant, our own pace, dirty air), a
    fixed plan set's times and positions, the traffic tools as the model is shown them, the
    belief optimum and one baseline, all planned with traffic."""
    from src.sim import traffic
    cfg = SimConfig.from_file(score.BELIEF_TRAFFIC_CONFIG)
    belief = score.race_simulators(PROFILE, belief_config=cfg)(data.race_id, "belief")
    fit = belief.fitted_for(snap)
    laps = np.arange(snap.laps_completed + 1, snap.race.total_laps + 1)
    state = fit.traffic.prepare(snap, laps, fit.laps, snap.driver, ref_lap_s=fit.ref_lap_s, model=fit.degradation)
    out = {"state": None if state is None else {
        "anchor": plain(state.anchor), "own_offset_s": plain(state.own_offset_s),
        "dirty_air_s": plain(state.dirty_air_s), "final_idx": plain(state.final_idx),
        "variants": [{"ghosts": list(v.table.ghosts), "G": plain(v.table.G), "G_prev": plain(v.table.G_prev),
                      "in_lap": plain(v.table.in_lap), "out_lap": plain(v.table.out_lap)}
                     for v in state.variants]}}
    plans = [p for k, p in plan_script(snap).items() if k.startswith(("one_stop", "single"))]
    b = belief.simulate_batch(snap, plans, seed=agent_tools.SEED, n_samples=N_SAMPLES)
    out["simulate_batch"] = {"times": plain(b.times), "traffic_s": plain(b.traffic_s),
                             "final_pos": plain(b.final_pos), "rejoin_pos": plain(b.rejoin_pos),
                             "pit_window_cars": plain(b.pit_window_cars),
                             "support_warnings": plain(b.support_warnings)}
    with mock.patch.dict(agent_tools._BELIEF, {"config": cfg}), \
            mock.patch.object(agent_tools, "_SIMS", {}):
        tools.set_snapshot(snap)
        out["get_model_report"] = _shown(tools.get_model_report, None)
        out["compare_plans"] = _shown(tools.compare_plans, [{"action": "pit", "compound": c} for c in DRY]
                                      + [{"action": "stay_out", "laps": 3}])
        out["evaluate_strategy"] = _shown(tools.evaluate_strategy, {"action": "pit", "compound": _other(snap.compound)})
    out["belief_optimizer"] = _call(lambda: decisions.belief_optimum(snap, belief, MAX_STOPS, seed=0,
                                                                    n_samples=N_SAMPLES).as_pairs())
    out["run_to_flag"] = _call(lambda: plain(policies.RULE_POLICIES["run_to_flag"](scenario, belief, MAX_STOPS).plan))
    assert traffic.GhostTraffic(**cfg.traffic.params).reads_future is False
    out.update(_difficulty_surface(data, snap, plans))
    return out


BELIEF_DIFFICULTY_CONFIG = score.BELIEF_TRAFFIC_CONFIG.parent / "belief_difficulty.yaml"


def _difficulty_surface(data, snap, plans):
    """The belief with traffic and the as-of passing difficulty (configs/sim/belief_difficulty.yaml,
    src/sim/overtaking.py): the threshold it plays with and the races it read, a plan set's times and
    positions, and the passing words the tools show."""
    from src.sim import traffic
    cfg = SimConfig.from_file(BELIEF_DIFFICULTY_CONFIG)
    belief = score.race_simulators(PROFILE, belief_config=cfg)(data.race_id, "belief")
    fit = belief.fitted_for(snap)
    laps = np.arange(snap.laps_completed + 1, snap.race.total_laps + 1)
    state = fit.traffic.prepare(snap, laps, fit.laps, snap.driver, ref_lap_s=fit.ref_lap_s, model=fit.degradation)
    out = {"difficulty_state": None if state is None else {
        "theta_s": plain(state.pass_theta_s), "slope": plain(state.pass_slope),
        "sources": list(state.difficulty.sources) if state.difficulty is not None else None}}
    b = belief.simulate_batch(snap, plans, seed=agent_tools.SEED, n_samples=N_SAMPLES)
    out["difficulty_batch"] = {"times": plain(b.times), "traffic_s": plain(b.traffic_s),
                               "final_pos": plain(b.final_pos), "rejoin_pos": plain(b.rejoin_pos)}
    with mock.patch.dict(agent_tools._BELIEF, {"config": cfg}), \
            mock.patch.object(agent_tools, "_SIMS", {}):
        tools.set_snapshot(snap)
        out["difficulty_model_report"] = _shown(tools.get_model_report, None)
        out["difficulty_evaluate_strategy"] = _shown(tools.evaluate_strategy, {"action": "pit", "compound": _other(snap.compound)})
    assert traffic.GhostTraffic(**cfg.traffic.params).reads_future is False
    return out


def _baselines_surface(belief, scenario):
    pols = {**policies.BASELINE_POLICIES, **policies.RULE_POLICIES}
    out = {}
    for name in OBSERVED_BASELINES:
        def run(policy=pols[name]):
            d = policy(scenario, belief, MAX_STOPS)
            plan = d.plan.as_pairs() if hasattr(d.plan, "as_pairs") else d.plan
            return {"plan": plan, "meta": d.meta}
        out[name] = _call(run)
    return out


@contextlib.contextmanager
def serving(data: jobs.RaceData, extra=()):
    """While open, jobs.load_race returns `data` (the belief and the tools build from it),
    the tools' caches start empty and every profile lookup gets PROFILE."""
    def load_race(race_id, profile=None):
        return data
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(jobs, "load_race", load_race))
        stack.enter_context(mock.patch.object(agent_tools, "_RACES", {}))
        stack.enter_context(mock.patch.object(agent_tools, "_SIMS", {}))
        stack.enter_context(mock.patch.object(agent_tools, "load_profile", lambda *a, **k: PROFILE))
        for cm in extra:
            stack.enter_context(cm)
        yield


def observe(data: jobs.RaceData, driver: str, lap: int, *, snapshot_fix=None, patches=()) -> dict:
    """Every policy-visible surface (SURFACES) at (driver, lap) of `data`. `snapshot_fix`
    rewrites the built snapshot first (a known leak's neutraliser)."""
    snap = BUILDER(data.frame, data.race, driver, lap)
    if snapshot_fix is not None:
        snap = snapshot_fix(snap)
    view = RaceSnapshot.from_contract(snap)
    shown = tools.set_snapshot(snap)   # what the model is shown (anonymised unless AGENT_SHOW_IDENTITY)
    scenario = Scenario(scenario_id=f"{data.race_id}:{driver}:{lap:02d}", race_id=data.race_id, driver=driver,
                        lap=lap, snapshot=snap, actual_stints=[])
    with serving(data, patches):
        belief = score.race_simulators(PROFILE)(data.race_id, "belief")
        out = {"snapshot": {"contract": plain(snap), "view": plain(shown.to_dict()),
                            "contract_of_view": _call(agent_tools.contract_of, view)}}
        out["belief"] = _belief_surface(belief, snap)
        out["rules"] = _rules_surface(belief, snap)
        out["tools"] = _tools_surface(snap)
        out["decisions"] = _decisions_surface(belief, snap)
        out["baselines"] = _baselines_surface(belief, scenario)
        out["traffic"] = _traffic_surface(data, snap, scenario)   # raises: an error here is a bug, not an output
    return out


# --- comparing ---------------------------------------------------------------------------------
def diff(a, b, path=""):
    """[(path, a value, b value)] of every leaf that differs; floats within REL_TOL / ABS_TOL."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", a.get(k, "<absent>"), b.get(k, "<absent>")))
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(f"{path}[len]", len(a), len(b))] + diff(a[:min(len(a), len(b))], b[:min(len(a), len(b))], path)
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float) or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                                       and not isinstance(a, bool) and not isinstance(b, bool)
                                                       and (isinstance(a, float) or isinstance(b, float))):
        if abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b)):
            return []
        return [(path, a, b)]
    return [] if a == b else [(path, a, b)]


@dataclass
class Difference:
    point: str
    surface: str
    n: int
    first: tuple
    paths: list = field(default_factory=list)

    def __str__(self):
        p, a, b = self.first
        return f"{self.point} {self.surface}: {self.n} field(s) differ; first {p}: recorded {_short(a)} vs altered {_short(b)}"


def _short(v, n=160):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def compare(orig: dict, alt: dict, point: str) -> list[Difference]:
    out = []
    for s in SURFACES:
        d = diff(orig[s], alt[s], f"{s}")
        if d:
            out.append(Difference(point, s, len(d), d[0], [x[0] for x in d]))
    return out


# --- checks the perturbations cannot make ---------------------------------------------------------
def _strings(x, path=""):
    """(path, text) of every string leaf and dict key in a plain value."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield f"{path}/{k}", str(k)
            yield from _strings(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(x, str):
        yield path, x


def identity_tokens(race_id: str, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(names matched case-insensitively as substrings, driver codes matched as whole words):
    the season, the race id's event slug and its words, the cached schedule's event name,
    country and location, and the race's driver codes."""
    from src.sim.data import season_schedule
    season, rnd, gp = parse_race_id(race_id)
    names = {str(season), gp.replace("_", " "), *[w for w in gp.split("_") if len(w) >= 4]}
    sched = season_schedule(season)
    if sched is not None:
        row = sched[sched["RoundNumber"] == rnd]
        if len(row):
            names |= {str(row.iloc[0][c]) for c in ("EventName", "Country", "Location")}
    return sorted(n for n in names if n), sorted(frame["driver"].dropna().unique())


def identity_hits(value, race_id: str, frame: pd.DataFrame) -> list[tuple[str, str]]:
    """(path, token) of every string in `value` that names the real race or a real driver."""
    import re
    names, codes = identity_tokens(race_id, frame)
    code_re = re.compile(r"\b(" + "|".join(map(re.escape, codes)) + r")\b") if codes else None
    out = []
    for path, text in _strings(value):
        low = text.lower()
        out += [(path, n) for n in names if n.lower() in low]
        if code_re is not None:
            out += [(path, m) for m in code_re.findall(text)]
    return out


# The policy-side artifacts are the project's registry (src/sim/provenance.py): each records the
# races it was fitted on (recorded) and, per race, the races pricing that race reads (races_for);
# kind "a_priori" reads none, "not_as_of" (the learned calibrator) is off by default and refused
# for a race not after all its training races.
from src.sim import provenance as _provenance   # noqa: E402

POLICY_SIDE_ARTIFACTS = _provenance.POLICY_SIDE


def provenance(name: str):
    """The races `name` records it was fitted on ([] for an a priori constant); None when it
    records none (no registry entry)."""
    if name not in POLICY_SIDE_ARTIFACTS:
        return None
    return _provenance.recorded(name)


def tyre_history_held_out() -> set:
    """The races data/eval_set.json holds out: none may feed a policy-side table."""
    from src.sim.tyre_history import held_out
    return set(held_out())


def _before(race_id, other) -> bool:
    try:
        return parse_race_id(other)[:2] < parse_race_id(race_id)[:2]
    except ValueError:
        return False


def artifacts_not_before(race_id: str) -> dict:
    """{artifact: why} for every policy-side artifact that, pricing `race_id`, reads a race not
    strictly before it, or records nothing."""
    out = {}
    for name in POLICY_SIDE_ARTIFACTS:
        if provenance(name) is None:
            out[name] = "records no races"
            continue
        used = _provenance.races_for(name, race_id)
        later = sorted(r for r in used if not _before(race_id, r))
        if later:
            out[name] = f"reads {later}"
    return out


def later_table_hits(obs: dict) -> list[tuple[str, object]]:
    """(path, value) of what an observation took from a cross-race table: a source naming the
    cross-race step or a life ratio, and a calibration_error_s with a value."""
    out = [(p, t) for p, t in _strings(obs) if "cross_race_step" in t or "life_ratio" in t]

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "calibration_error_s" and isinstance(v, float):
                    out.append((f"{path}/{k}", v))
                walk(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(obs)
    return out


def check(inp: RaceInput, orig: jobs.RaceData, points, alteration: str, neutralise=(), recorded=None) -> list[Difference]:
    """Differences, recorded race against altered, at every (driver, lap) of `points`: the
    past ends at each point's own moment, so each point gets its own altered race.
    `recorded` caches the recorded race's observations by point."""
    recorded = {} if recorded is None else recorded
    out = []
    for driver, lap in points:
        data, snapshot_fix, patches = altered(inp, orig, alteration, driver, lap, neutralise)
        if (driver, lap) not in recorded:
            recorded[driver, lap] = observe(orig, driver, lap)
        snap = BUILDER(orig.frame, orig.race, driver, lap)
        alt = observe(data, driver, lap, snapshot_fix=snapshot_fix(snap), patches=patches)
        out += compare(recorded[driver, lap], alt, f"{orig.race_id}:{driver}:{lap} [{alteration}]")
    return out
