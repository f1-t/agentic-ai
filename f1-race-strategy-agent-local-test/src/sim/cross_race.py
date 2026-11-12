"""Cross-race calibration report (SPEC 0016 with its amendment).

    python -m src.sim.calibrate cross run --races {auto|2021_04_spain ...} [--allow-held-out] [--synthetic 2]
                                          [--profile local] [--out DIR]
    python -m src.sim.calibrate cross report [--out DIR] [--race-features data/race_features.csv]

Held-out races: a race data/eval_set.json lists is refused (guard_held_out, before anything
about it is loaded) unless --allow-held-out; `--races auto` (every cached race,
real_run.cached_races) skips them. The report's "Held-out races" section names them.

`run` does, resumably (every unit is a jobs.run_task, skipped when its output is current):
  A  calibration.eligibility per race, written to eligibility.json; only eligible races go on,
     the others are listed with their reasons and kept for later.
  B  per eligible race (jobs cross_race): calibration.run (replay, by_kind with the clean
     split, observed factors incl. the pack penalty), observed_periods, the pace gate
     (pace.gate_table, simulated order), G (jobs.g_rows, with the SPEC 0017 intervals at the
     profile's n_samples), hindsight targets and the eligibility record, as one parquet under
     races/<race_id>/.
  C  leave-one-race-out per arm (jobs h_fold --arm): each held-out race replayed on G's
     windows under the arm's config through calibrators.loro, with the factor errors and
     the stop-gain rows (calibration.stop_gain), under h/arms/<arm>/. Three stages: the
     committed arms (configs/sim/arms/*.yaml, ARMS); then the prior-sd fix (SPEC 0017 G,
     degradation param divide_data_by_s2) alone on the to_snapshot choice; then the tau grid
     (TAU_MULTIPLIERS of base, wear, fuel and, for a hinge model, hinge) on it.
`report` writes D factors.csv and E summary.md, with selection.csv, g_by_arm.csv,
stop_gain.csv (gitignored), stop_gain_summary.csv, stop_gain_by_race.csv (real and predicted
gain, signed bias and mae per window, race and arm), stop_counts.csv (the stop-count check:
the field's stop counts against the reference's from lap 0 under each committed arm's world,
B's field_stops against C's reference_stops), coverage_summary.csv, tau_grid.csv and arms.csv
beside them, from what `run` left.

Arms are overlays on each race's config, frozen before the run; arms.csv records each arm's
file hash and its config hash on SimConfig() (a cached race's config). A derived arm is named
<arm>__s2 (the prior-sd fix) or <arm>[__s2]__tau_<b>_<w>_<f>_<h> (tau multipliers).

Selection, stated before any cross-race data (SPEC 0016 C, amendment 3, review of #44), per
window, every point estimate race-weighted (the mean over races of the per-race value, the
statistic the race-cluster CIs resample; the row-pooled value is reported beside it), and
every comparison incumbent-first: the incumbent is the arm the config carries now
(INCUMBENTS, the report's "Current defaults"), and a change in either direction must pass:
  to_snapshot (the BELIEF), in two stages: first the bound, the baseline (no bound) as the
      challenger against the incumbent wear_nonneg (belief.yaml, adopted a priori), so the
      bound stays unless removing it passes; then every other candidate against the stage-1
      result. A challenger replaces its comparator when the 95 percent CI of the paired
      stop-gain mae difference (challenger less comparator) is below 0 AND the upper end of
      the paired G mae_offset CI at horizons 10-49 is under +NONINFERIORITY_S.
  full (the WORLD) and pre_race, against the SimConfig default: when the paired G
      mae_offset difference at horizons 10-49 is below 0 AND its 95 percent CI excludes 0.
      In the full window an arm that differs only in its calibrator or a to_snapshot-only
      option is the baseline by construction: n/a.
Among arms that pass, the lowest primary metric wins; ties go to the simpler arm
(SIMPLICITY). "posterior" is reported for coverage only. The tau grid is ranked by the belief
rule against its x1 taus. With fewer than MIN_RACES eligible races the report says the
selection is provisional and the defaults stay. All G and stop-gain numbers are held out:
each race's rows come from its own fold. The pace gate's practice comes from
pace.load_practice (wet sessions left out). A run or report with a synthetic race refuses
to write into CROSS_DIR (guard_committed).

Synthetic races (synthetic_race, ids <season>_<round>_synthetic...) are generated from the
id, so a worker process rebuilds the same race; they exist to build and test the pipeline.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import warnings
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import RaceParams
from . import calibration, calibrators, jobs, pace
from .compute import load_profile
from .config import EventModel, SimConfig
from .engine import Simulator
from .race_ids import canonical_race_id, parse_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
ARMS_DIR = ROOT / "configs" / "sim" / "arms"
CROSS_DIR = ROOT / "reports" / "calibration" / "cross_race"
RACE_FEATURES = ROOT / "data" / "race_features.csv"   # the data team's per-race covariates, when present
LOADER = "src.sim.cross_race:load_race"

ARMS = ("baseline", "none", "hinge", "free_air", "hinge_free", "posterior", "wear_nonneg", "collinear_hold",
        "collinear_hold_hinge", "collinear_hold_nonneg", "collinear_hold_nonneg_hinge",
        # wear identification: one pace offset per driver in the degradation fit (degradation.py)
        "driver_effects", "driver_effects_nonneg", "driver_effects_hinge")
BASELINE = "baseline"
COVERAGE_ONLY = ("posterior",)
# simplest first: the tie-break among arms that pass
SIMPLICITY = ("none", "baseline", "wear_nonneg", "collinear_hold", "collinear_hold_nonneg", "free_air",
              "driver_effects", "driver_effects_nonneg", "hinge", "collinear_hold_hinge", "collinear_hold_nonneg_hinge",
              "driver_effects_hinge", "hinge_free")
MIN_RACES = 8               # below this the selection is provisional and the defaults stay
G_HORIZONS = (10, 49)       # selection horizons, inclusive
TAU_HORIZONS = (20, 40)     # tau grid horizons, inclusive
NONINFERIORITY_S = 2.0      # belief rule: upper end of the paired G mae_offset CI, a priori
TAU_MULTIPLIERS = (0.5, 1.0, 2.0)
TAU_KEYS = ("base_s", "wear_s_per_lap", "fuel_s_per_lap", "hinge_s_per_lap")
S2 = "s2"
BELIEF_WINDOW = "to_snapshot"
N_BOOT = 2000


# --- arms -----------------------------------------------------------------------------
def parse_arm(name: str) -> tuple[str, bool, tuple | None]:
    """(committed arm, prior-sd fix on, tau multipliers or None) of an arm name."""
    parts = name.split("__")
    base, s2, tau = parts[0], False, None
    if base not in ARMS:
        raise KeyError(f"unknown arm {base!r}; arms: {list(ARMS)}")
    for p in parts[1:]:
        if p == S2:
            s2 = True
        elif p.startswith("tau_"):
            tau = tuple(float(x) for x in p[len("tau_"):].split("_"))
            if len(tau) != len(TAU_KEYS):
                raise ValueError(f"{name}: tau needs {len(TAU_KEYS)} multipliers")
        else:
            raise ValueError(f"{name}: unknown arm suffix {p!r}")
    return base, s2, tau


def tau_arm(arm: str, mults) -> str:
    return f"{arm}__tau_" + "_".join(f"{float(m):g}" for m in mults)


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def arm_overlay(name: str) -> dict:
    """The arm's overlay: its committed yaml, plus the derived parts its name carries."""
    import yaml
    base, s2, tau = parse_arm(name)
    over = yaml.safe_load((ARMS_DIR / f"{base}.yaml").read_text()) or {}
    if s2:
        _merge(over, {"degradation": {"params": {"divide_data_by_s2": True}}})
    if tau is not None:
        default = SimConfig().shrinkage
        _merge(over, {"shrinkage": {k: default[k] * m for k, m in zip(TAU_KEYS, tau)}})
    return over


def arm_config(name: str, base: SimConfig | None = None) -> SimConfig:
    """`base` (a race's config, SimConfig() when None) with the arm's overlay merged in."""
    base = SimConfig() if base is None else base
    return SimConfig.model_validate(_merge(base.model_dump(), arm_overlay(name)))


def _sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def arms_table(names=ARMS) -> pd.DataFrame:
    """Per arm: its file, the file's sha256 (12 hex), its config_hash on SimConfig(), its role."""
    rows = []
    for n in names:
        base = parse_arm(n)[0]
        path = ARMS_DIR / f"{base}.yaml"
        cfg = arm_config(n)
        rows.append({"arm": n, "file": f"configs/sim/arms/{base}.yaml", "file_sha256": _sha12(path.read_bytes()),
                     "config_hash": calibration.config_hash(cfg), "degradation": cfg.degradation.name,
                     "calibrator": cfg.calibrator.name,
                     "role": "coverage only" if base in COVERAGE_ONLY else "baseline" if n == BASELINE else "candidate"})
    return pd.DataFrame(rows)


def _is_hinge(arm: str) -> bool:
    return arm_config(arm).degradation.name == "linear_fuel_hinge"


def _holds_prior(arm: str) -> bool:
    return arm_config(arm).calibrator.name == "none"


def tau_combos(arm: str, multipliers=TAU_MULTIPLIERS) -> list[str]:
    """The tau grid on `arm`: every combination of `multipliers` on base, wear, fuel and, for a
    hinge model, hinge (81 at three multipliers; 27 without a hinge, whose tau does nothing)."""
    keys = len(TAU_KEYS) if _is_hinge(arm) else len(TAU_KEYS) - 1
    return [tau_arm(arm, m + (1.0,) * (len(TAU_KEYS) - keys)) for m in product(multipliers, repeat=keys)]


# --- races ----------------------------------------------------------------------------
_SYNTHETIC = re.compile(r"^\d{4}_\d{2}_synthetic(?:_[a-z0-9]+)*$")
SYN_TRUE = {"SOFT": (80.0, 0.08), "MEDIUM": (80.6, 0.05), "HARD": (81.2, 0.03)}
SYN_FUEL = -0.05
SYN_PIT_LOSS_S = 21.0
SYN_NOISE_S = 0.25
SYN_SC_FACTOR = 1.36


def is_synthetic(race_id: str) -> bool:
    return bool(_SYNTHETIC.match(race_id))


_LOADED: dict = {}
LOAD_CACHE_SIZE = 4


def load_race(race_id: str, profile) -> jobs.RaceData:
    """jobs.load_race for a cached race, synthetic_race for a synthetic id: the run's loader.
    The last LOAD_CACHE_SIZE races loaded are kept per process (by id and R2's minimum
    stint, the profile's one input to the load); callers copy the RaceData with replace()
    and never modify its frames."""
    race_id = canonical_race_id(race_id)
    key = (race_id, jobs.race_min_stint(profile))
    if key not in _LOADED:
        if len(_LOADED) >= LOAD_CACHE_SIZE:
            del _LOADED[next(iter(_LOADED))]
        _LOADED[key] = synthetic_race(race_id) if is_synthetic(race_id) else jobs.load_race(race_id, profile)
    return _LOADED[key]


def race_messages(race_id: str):
    """(race control messages or None, note or None). Synthetic races have none."""
    if is_synthetic(race_id):
        return None, None
    from .data import load_session
    season, rnd, _ = parse_race_id(race_id)
    try:
        return load_session(season, rnd, messages=True).race_control_messages, None
    except Exception as exc:   # not in the offline cache: eligibility from the lap frame alone
        return None, f"no race control messages ({type(exc).__name__})"


def _strategy(rng, laps):
    """A one- or two-stop plan [(compound, laps)] over `laps` laps, two dry compounds at least."""
    kind = rng.integers(3)
    if kind == 0:
        a = int(rng.integers(int(0.35 * laps), int(0.5 * laps) + 1))
        return [("SOFT", a), ("MEDIUM", laps - a)]
    if kind == 1:
        a = int(rng.integers(int(0.3 * laps), int(0.45 * laps) + 1))
        return [("MEDIUM", a), ("HARD", laps - a)]
    a = int(rng.integers(int(0.22 * laps), int(0.32 * laps) + 1))
    b = int(rng.integers(int(0.3 * laps), int(0.4 * laps) + 1))
    return [("SOFT", a), ("MEDIUM", b), ("SOFT", laps - a - b)]


def synthetic_race(race_id: str, n_drivers: int = 16, total_laps: int = 50, sc: bool = True, fuel: float = SYN_FUEL,
                   practice_fuel: float | None = None, evolution_s_per_min: float | None = None) -> jobs.RaceData:
    """A dry race drawn from a known model, seeded by `race_id`: lap time = base_c + wear_c x age
    + fuel x lap + the driver's pace + N(0, SYN_NOISE_S^2), lap 1 slower by the config's
    lap1_extra_s, the pit loss split over the in and out laps, and (with `sc`) one 3-lap SC
    at SYN_SC_FACTOR. Practice: an FP2 and an FP3 long run per driver, 1 s quicker in base,
    with `practice_fuel` (default `fuel`) per lap in the run. Every driver finishes on the
    lead lap, so it is eligible.

    `fuel` is the race-pace slope (fuel burn plus track evolution), SYN_FUEL by default.
    `evolution_s_per_min` (SPEC 0020) adds track evolution to practice: each long run starts
    at a stint_start_min drawn in [5, 55] from a second stream seeded by the id, and its laps
    are evolution_s_per_min x stint_start_min slower. None, the default, draws nothing and
    leaves the practice frame as before."""
    rng = np.random.default_rng(int(hashlib.sha256(race_id.encode()).hexdigest()[:8], 16))
    practice_fuel = fuel if practice_fuel is None else practice_fuel
    cfg = SimConfig()
    drivers = [f"D{i:02d}" for i in range(n_drivers)]
    team = {d: f"T{i // 2}" for i, d in enumerate(drivers)}
    pace_s = dict(zip(drivers, rng.normal(0.0, 0.3, n_drivers)))
    sc_start = int(rng.integers(12, total_laps - 15)) if sc else None
    sc_laps = set(range(sc_start, sc_start + 3)) if sc else set()
    ref = SYN_TRUE["MEDIUM"][0]
    rows = []
    for d in drivers:
        plan, lap, t = _strategy(rng, total_laps), 0, 0.0
        for i, (comp, n) in enumerate(plan):
            base, wear = SYN_TRUE[comp]
            for age in range(1, n + 1):
                lap += 1
                pit_in, pit_out = age == n and i < len(plan) - 1, age == 1 and i > 0
                under_sc = lap in sc_laps
                lt = base + wear * age + fuel * lap + pace_s[d] + rng.normal(0.0, SYN_NOISE_S)
                if under_sc:
                    lt = SYN_SC_FACTOR * (ref + fuel * lap) + rng.normal(0.0, SYN_NOISE_S)
                lt += cfg.effects.lap1_extra_s if lap == 1 else 0.0
                lt += 0.45 * SYN_PIT_LOSS_S if pit_in else 0.55 * SYN_PIT_LOSS_S if pit_out else 0.0
                rows.append({"driver": d, "lap": lap, "stint": i + 1, "compound": comp, "tyre_age": float(age),
                             "lap_time_s": lt, "track_status": "4" if under_sc else "1", "pit_in": pit_in,
                             "pit_out": pit_out, "start_s": t, "end_s": t + lt, "team": team[d],
                             "clean": not (pit_in or pit_out or under_sc or lap == 1)})
                t += lt
    frame = pd.DataFrame(rows)
    frame["position"] = frame.groupby("lap")["end_s"].rank(method="first").astype(float)
    order = frame.sort_values(["lap", "start_s"])
    gap = order.groupby("lap")["start_s"].diff()
    frame["free_air"] = (gap.isna() | (gap >= 1.5)).reindex(frame.index)
    practice = []
    evo_rng = (np.random.default_rng(int(hashlib.sha256(f"{race_id}:evolution".encode()).hexdigest()[:8], 16))
               if evolution_s_per_min is not None else None)
    for k, d in enumerate(drivers):
        for session, comp in (("FP2", "SOFT"), ("FP3", "HARD" if k % 2 else "MEDIUM")):
            base, wear = SYN_TRUE[comp]
            start = float(evo_rng.uniform(5.0, 55.0)) if evo_rng is not None else None
            for age in range(1, 13):
                row = {"driver": d, "team": team[d], "session": session, "stint": 1, "compound": comp,
                       "tyre_age": float(age), "lap": np.nan, "lap_in_stint": float(age),
                       "lap_time_s": base - cfg.pre_race_base_delta_s + wear * age + practice_fuel * age
                       + pace_s[d] + rng.normal(0.0, SYN_NOISE_S),
                       "clean": True, "run_class": "long_run"}
                if start is not None:
                    row["lap_time_s"] += evolution_s_per_min * start
                    row["stint_start_min"] = start
                practice.append(row)
    last = frame[frame["lap"] == total_laps].set_index("driver")["position"]
    results = pd.DataFrame({"grid": pd.Series(pace_s).rank(method="first").to_numpy(),
                            "classified": last.reindex(drivers).to_numpy()},
                           index=pd.Index(drivers, name="driver"))
    race = RaceParams(race_id=race_id, total_laps=total_laps, pit_loss_s=SYN_PIT_LOSS_S,
                      compounds=sorted(frame["compound"].unique()))
    return jobs.RaceData(race_id, frame, race, pd.DataFrame(practice), results, cfg)


# --- packing: one parquet holds several tables ------------------------------------------
def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if v is pd.NA or v is pd.NaT:
        return None
    raise TypeError(f"not serialisable: {type(v)}")


def pack(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tables as one frame: a `table` column and a `record` column with each row as JSON."""
    rows = [{"table": name, "record": json.dumps(rec, default=_jsonable)}
            for name, t in tables.items() for rec in t.to_dict("records")]
    return pd.DataFrame(rows, columns=["table", "record"])


def unpack(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame([json.loads(s) for s in part["record"]])
            for name, part in df.groupby("table", sort=False)}


def _stamped(df, data, cfg, profile, **extra):
    return df.assign(race_id=data.race_id, pit_loss_s=data.race.pit_loss_s, config_hash=calibration.config_hash(cfg),
                     sim_code_version=calibration.SIM_CODE_VERSION, profile=profile.name, **extra)


# --- B: per race -------------------------------------------------------------------------
def free_air_share(frame) -> float:
    """Share of the clean laps in free air (data.lap_frame free_air), NaN without the column."""
    if "free_air" not in frame:
        return float("nan")
    clean = frame[frame["clean"].astype(bool)]
    return float(clean["free_air"].eq(True).mean()) if len(clean) else float("nan")


def pace_practice(data: jobs.RaceData):
    """(practice frame or None, notes) for the pace gate: pace.load_practice for a cached race,
    which leaves out a wet FP2 or FP3; a synthetic race's own FP2 and FP3 (always dry)."""
    if not is_synthetic(data.race_id):
        season, rnd, _ = parse_race_id(data.race_id)
        return pace.load_practice(season, rnd, long_run=data.config.long_run)
    p = data.practice
    if p is not None and "session" in p:
        p = p[p["session"].isin(pace.PRACTICE_SESSIONS)]
    return (p if p is not None and len(p) else None), []


# --- the stop-count check: the field's stop counts against the reference's from lap 0 -----------
FIELD_WITHIN_LAPS = 2   # a car counts for the field when it ran at least total_laps - this
FIELD_STOP_COLUMNS = ["race_id", "driver", "laps", "stops", "start_compound"]
REFERENCE_STOP_COLUMNS = ["race_id", "driver", "start_compound", "ref_stops", "ref_plan", "ref_mean_s",
                          "best_1stop_s", "best_2stop_s", "two_less_one_s"]


def field_drivers(frame, total_laps: int, within=FIELD_WITHIN_LAPS) -> list[str]:
    """Drivers whose last lap is within `within` laps of the race distance, sorted."""
    last = frame.groupby("driver")["lap"].max()
    return sorted(last.index[last >= total_laps - within])


def field_stops(frame, race, within=FIELD_WITHIN_LAPS) -> pd.DataFrame:
    """One row per field car (field_drivers): laps run, stops (its pit-in laps before its last
    lap: a stop needs a lap after it) and starting compound."""
    rows = []
    for d in field_drivers(frame, race.total_laps, within):
        g = frame[frame["driver"] == d].sort_values("lap")
        last = int(g["lap"].max())
        stops = int((g["pit_in"].astype(bool) & (g["lap"] < last)).sum())
        rows.append({"race_id": race.race_id, "driver": d, "laps": last, "stops": stops,
                     "start_compound": str(g["compound"].iloc[0])})
    return pd.DataFrame(rows, columns=FIELD_STOP_COLUMNS)


def reference_stops(data: jobs.RaceData, cfg: SimConfig, max_stops: int, within=FIELD_WITHIN_LAPS) -> pd.DataFrame:
    """The reference at lap 0 under `cfg`'s world: optimizer_bridge.optimize_batch on the full
    window, one sample (seed 0), up to `max_stops`, at each field car's lap-0 snapshot
    (calibration.snapshot_at). Per car: the reference's stops, plan and mean, the best one-stop
    and best two-stop means among the searched plans, and two_less_one_s (two-stop less
    one-stop: above 0 when one stop is quicker). NaN where no plan of that count is legal."""
    from ..baselines.domain import NoFeasiblePlan
    from .optimizer_bridge import optimize_batch
    world = Simulator(data.frame, cfg.model_copy(update={"fit_window": "full", "n_samples": 1}), data.practice)
    rows = []
    for d in field_drivers(data.frame, data.race.total_laps, within):
        snap = calibration.snapshot_at(data.frame, data.race, d, 0)
        row = {"race_id": data.race_id, "driver": d, "start_compound": snap.compound}
        try:
            res = optimize_batch(snap, world, seed=0, n_samples=1, max_stops=max_stops, return_times=True)
        except NoFeasiblePlan:
            rows.append({**row, **dict.fromkeys(REFERENCE_STOP_COLUMNS[3:], np.nan)})
            continue
        searched = res.diagnostics["searched"]
        stops = np.array([len(p.stints) - 1 for p in searched["plans"]])
        means = np.asarray(searched["mean_time_s"], dtype=float)
        best = {n: float(means[stops == n].min()) if (stops == n).any() else np.nan for n in (1, 2)}
        rows.append({**row, "ref_stops": int(len(res.plan.stints) - 1),
                     "ref_plan": " ".join(f"{s.compound}:{s.laps}" for s in res.plan.stints),
                     "ref_mean_s": float(res.evaluation.mean_time_s), "best_1stop_s": best[1],
                     "best_2stop_s": best[2], "two_less_one_s": best[2] - best[1]})
    return pd.DataFrame(rows, columns=REFERENCE_STOP_COLUMNS)


def race_tables(data: jobs.RaceData, profile, messages=None, practice=None, notes=()) -> dict[str, pd.DataFrame]:
    """SPEC 0016 B for one race under its own config: replay, summary, by_kind, periods, gate, g,
    hindsight, eligibility and field_stops (the stop-count check's field) tables. `practice`
    feeds the pace gate (pace_practice); its notes go in the summary row."""
    frame, race, cfg = data.frame, data.race, data.config
    full = Simulator(frame, cfg, data.practice)
    table, summary, by_kind = calibration.run(full, frame, race)
    paces, _ = pace.offsets(full, frame, race, practice)
    gate = pace.gate_table(full, frame, race, data.results, paces)
    gate = gate[gate["mode"] == "simulated_order"].reset_index(drop=True)
    g = jobs.g_rows(data, cfg.model_copy(update={"n_samples": profile.sim.n_samples}), profile)
    hind = calibrators.hindsight_targets(data)
    elig = calibration.eligibility(frame, race, messages)
    periods = calibration.observed_periods(frame)
    return {
        "replay": table.reset_index(),
        "summary": pd.DataFrame([{**summary, "free_air_share": free_air_share(frame),
                                  "total_laps": race.total_laps, "pace_practice_notes": "; ".join(notes)}]),
        "by_kind": by_kind.reset_index(),
        "periods": pd.DataFrame(periods, columns=["kind", "start_lap", "end_lap"]),
        "gate": gate,
        "g": g,
        "hindsight": pd.DataFrame({"factor": list(hind), "value": list(hind.values())}),
        "eligibility": pd.DataFrame([{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                      for k, v in elig.items()}]),
        "field_stops": field_stops(frame, race),
    }


def race_job(unit: str, profile, loader=load_race) -> pd.DataFrame:
    """jobs cross_race: the packed race_tables of one race, stamped."""
    data = loader(unit, profile)
    messages, _ = race_messages(data.race_id)
    practice, notes = pace_practice(data)
    return _stamped(pack(race_tables(data, profile, messages, practice, notes)), data, data.config, profile)


# --- C: one fold of one arm ---------------------------------------------------------------
def _grid_laps(frame, ks, horizons=TAU_HORIZONS, min_laps_frac=0.9):
    """The snapshot laps where some finisher's horizon falls in `horizons`."""
    laps = frame.groupby("driver")["lap"].max()
    laps = laps[laps >= min_laps_frac * laps.max()]
    return [k for k in ks if ((laps - k >= horizons[0]) & (laps - k <= horizons[1])).any()]


def fold_tables(data: jobs.RaceData, arm: str, profile, loader=load_race, pool=None) -> dict[str, pd.DataFrame]:
    """The held-out race `data` under `arm`: G rows (calibrators.loro on the profile's windows,
    intervals at the profile's n_samples), factor errors, and stop-gain rows per window
    against the arm's own full-window fit. A tau-grid arm runs to_snapshot only, one replay
    per row, on the laps whose horizons reach TAU_HORIZONS (its stop-gain rows on every
    profile lap: a fit per k, no sampling). `loader` and
    `pool` give the training races of a learned calibrator."""
    base, _, tau = parse_arm(arm)
    grid = tau is not None
    unit = data.race_id
    cfg = arm_config(arm, data.config)
    data = replace(data, config=cfg)
    all_ks = ks = jobs.laps_k_for(data.frame, data.race.total_laps, profile)
    if grid:
        ks = _grid_laps(data.frame, ks)
    races = [data]
    if calibrators.CALIBRATORS.get(cfg.calibrator.name).learned:   # trains on the races outside the fold
        races += [replace(r, config=arm_config(arm, r.config)) for r in
                  (loader(x, profile) for x in (pool or []) if canonical_race_id(x) != canonical_race_id(unit))]
    dataset = calibrators.build_dataset(
        races, laps_k=lambda d: ks if d.race_id == data.race_id else jobs.laps_k_for(d.frame, d.race.total_laps, profile))
    windows = [BELIEF_WINDOW] if grid else list(profile.g.windows)
    n = 1 if grid else profile.sim.n_samples
    res = calibrators.loro(dataset, cfg.calibrator.name, data.race_id, cadences=(1,), windows=windows,
                           n_samples=None if grid else n, calibrator_params=cfg.calibrator.params or None,
                           midrace_samples=n)
    stops = []
    if base not in COVERAGE_ONLY:   # a grid arm too: a fit at each k, no sampling
        hind = Simulator(data.frame, cfg.model_copy(update={"fit_window": "full"}), data.practice).degradation
        for w in windows:
            sim = Simulator(data.frame, cfg.model_copy(update={"fit_window": w}), data.practice)
            stops.append(calibration.stop_gain(sim, data.frame, data.race, hind, all_ks))
    stop = pd.concat(stops, ignore_index=True) if stops else pd.DataFrame(columns=calibration.STOP_GAIN_COLUMNS)
    # the stop-count check: a committed arm's world at lap 0 (a derived arm changes to_snapshot only)
    ref = (reference_stops(data, cfg, profile.sim.max_stops) if base not in COVERAGE_ONLY and arm == base
           else pd.DataFrame(columns=REFERENCE_STOP_COLUMNS))
    return {"g": res.rows.drop(columns=["calibration_every"], errors="ignore"), "factors": res.factor_errors,
            "stop_gain": stop, "reference_stops": ref}


FOLD_ERROR_COLUMNS = ["race_id", "arm", "error"]


def fold_job(unit: str, arm: str, profile, loader=load_race, pool=None) -> pd.DataFrame:
    """jobs h_fold --arm: the packed fold_tables, stamped with the arm and its config hash.

    A fold the arm cannot replay (calibration's "replay plan rejected": the arm's world does not
    model a compound the race ran, e.g. free_air on a compound with too few free-air laps) is
    kept as a fold_error row instead of stopping the run; the arm is then compared on the
    other races, and the report lists the fold."""
    data = loader(unit, profile)
    cfg = arm_config(arm, data.config)
    try:
        tables = fold_tables(data, arm, profile, loader, pool)
    except ValueError as exc:
        if "replay plan rejected" not in str(exc):
            raise
        tables = {"fold_error": pd.DataFrame([{"race_id": data.race_id, "arm": arm, "error": str(exc)}],
                                             columns=FOLD_ERROR_COLUMNS)}
    return _stamped(pack(tables), data, cfg, profile, arm=arm)


# --- running ------------------------------------------------------------------------------
def _args(races, **kw) -> Namespace:
    """The jobs task arguments for a cross-race unit: the run's loader, `kw` (arm, profile) on top."""
    return Namespace(**{"races": races, "pool": races, "calibrator": "shrinkage", "race_loader": LOADER, "arm": None,
                        "wandb": False, **kw})


PENDING = "pending"   # a runner's status for a unit left to a scheduler (real_run's Slurm runner)


def item_label(item) -> str:
    """'job[/arm] unit' of a run item, as the run log prints it."""
    return f"{item[0]}{'/' + item[4].arm if item[4].arm else ''} {item[1]}"


def _run_items(items, n_jobs: int):
    """jobs._run_one over the items, in a pool of n_jobs; [(job[/arm] unit, status)]."""
    if n_jobs > 1 and len(items) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            done = list(ex.map(jobs._run_one, items))
    else:
        done = [jobs._run_one(i) for i in items]
    return [(item_label(i), st) for i, (_, st) in zip(items, done)]


def _fold_items(arms, races, profile, out):
    return [("h_fold", r, profile, str(out), _args(races, arm=a, profile=profile.name)) for a in arms for r in races]


def guard_committed(out, races):
    """Refuse a run or report with a synthetic race id that would write into the committed
    report directory (CROSS_DIR): synthetic races belong in a scratch or tmp directory."""
    synthetic = [r for r in races if is_synthetic(canonical_race_id(r))]
    if synthetic and Path(out).resolve() == Path(CROSS_DIR).resolve():
        raise ValueError(f"synthetic races {synthetic} would go into the committed report {CROSS_DIR}; "
                         "pass --out a scratch directory")


EVAL_SET = ROOT / "data" / "eval_set.json"   # the races held out for the final evaluation


def held_out_races(path=EVAL_SET) -> list[str]:
    """The canonical race ids `path` (data/eval_set.json, {"races": [...]}) holds out; none when
    the file is absent. Model choices must not rest on them, so the cross-race runner leaves
    them out."""
    p = Path(path)
    if not p.is_file():
        return []
    return sorted({canonical_race_id(r) for r in json.loads(p.read_text()).get("races", [])})


def split_held_out(races, path=EVAL_SET) -> tuple[list[str], list[str]]:
    """(the races not held out, the held-out ones), canonical ids in the given order. Reads
    `path` only: nothing about a held-out race is loaded."""
    held = set(held_out_races(path))
    ids = [canonical_race_id(r) for r in races]
    return [r for r in ids if r not in held], [r for r in ids if r in held]


def guard_held_out(races, path=EVAL_SET, allow=False) -> list[str]:
    """Refuse races `path` holds out unless `allow` (--allow-held-out); returns the held-out
    races allowed in (none without `allow`)."""
    _, held = split_held_out(races, path)
    if held and not allow:
        raise ValueError(f"{held} held out in {path}: the cross-race runner leaves held-out races out (model "
                         "choices must not rest on them); drop them or pass --allow-held-out")
    return held


def _shown(path) -> str:
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def eligibility_records(races, profile, log=print) -> list[dict]:
    """A: calibration.eligibility per race (canonical ids or aliases), with the race control
    messages when cached; each record gains race_id, and a note when the messages are missing."""
    elig = []
    for r in (canonical_race_id(r) for r in races):
        data = load_race(r, profile)
        messages, note = race_messages(r)
        e = calibration.eligibility(data.frame, data.race, messages)
        e["race_id"] = r
        e["notes"] = e["notes"] + ([note] if note else [])
        elig.append(e)
        log(f"eligibility {r}: {'eligible' if e['eligible'] else 'not eligible: ' + '; '.join(e['reasons'])}")
    return elig


def run(races, profile, out=CROSS_DIR, tau_multipliers=TAU_MULTIPLIERS, tau_grid=True, n_jobs=None,
        log=print, elig=None, runner=None, eval_set=EVAL_SET, allow_held_out=False, held_out_skipped=()) -> dict:
    """A, B and the three C stages over `races` (canonical ids or aliases; synthetic ids
    allowed outside CROSS_DIR, guard_committed). Writes eligibility.json and run.json under
    `out`; returns the run record.

    A race `eval_set` (data/eval_set.json) holds out is refused before anything is loaded
    (guard_held_out) unless `allow_held_out`. The record's "held_out" says what the file lists,
    which of them `--races auto` skipped (`held_out_skipped`) and which were let in.

    `elig`: eligibility_records of `races` already computed (real_run), else computed here.
    `runner(items, phase)` runs one phase's units (phase races, arms, s2 or tau_grid) and
    returns [(label, status)]; the default runs them in-process with n_jobs workers. When a
    runner leaves units PENDING (a scheduler will run them), the run stops after that phase:
    run.json is removed, so nothing reads a half-finished run, and the record returned has
    `pending` (the labels) and `phase`. Rerunning resumes, as every unit is a jobs.run_task."""
    guard_committed(out, races)
    allowed = guard_held_out(races, eval_set, allow_held_out)
    held_out = {"eval_set": _shown(eval_set), "listed": held_out_races(eval_set),
                "skipped": [canonical_race_id(r) for r in held_out_skipped], "included": allowed}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = profile.parallel.n_jobs if n_jobs is None else n_jobs
    races = [canonical_race_id(r) for r in races]
    elig = eligibility_records(races, profile, log) if elig is None else elig
    (out / "eligibility.json").write_text(json.dumps(elig, indent=1))
    eligible = [e["race_id"] for e in elig if e["eligible"]]
    runner = runner or (lambda items, phase: _run_items(items, n_jobs))
    status = []

    def step(items, phase):
        done = runner(items, phase)
        status.extend(done)
        return [u for u, st in done if st == PENDING]

    def halt(phase, pending):
        for unit, st in status:
            log(f"  {unit}: {st}")
        (out / "run.json").unlink(missing_ok=True)
        return {"races": races, "eligible": eligible, "phase": phase, "pending": pending}

    pending = step([("cross_race", r, profile, str(out), _args(eligible, profile=profile.name)) for r in eligible],
                   "races")
    pending += step(_fold_items(ARMS, eligible, profile, out), "arms")
    if pending:
        return halt("races and arms", pending)
    sel = selection_tables(out, eligible, profile)
    chosen = sel["chosen"].get(BELIEF_WINDOW, BASELINE)
    record = {"races": races, "eligible": eligible, "arms": list(ARMS), "profile": profile.name,
              "profile_hash": profile.hash, "sim_code_version": calibration.SIM_CODE_VERSION,
              "belief_bound": sel["bound"], "belief_choice": chosen, "s2_arm": None, "grid_arm": None, "grid": [],
              "notes": [], "held_out": held_out}
    if _holds_prior(chosen):
        record["notes"].append(f"{chosen} holds the prior: no shrinkage fit, so no prior-sd fix and no tau grid")
    else:
        record["s2_arm"] = f"{chosen}__{S2}"
        if pending := step(_fold_items([record["s2_arm"]], eligible, profile, out), "s2"):
            return halt("s2", pending)
        s2 = s2_table(out, eligible, profile, chosen)
        grid_on = record["s2_arm"] if bool(s2["passes"].iloc[0]) else chosen
        if tau_grid:
            record["grid_arm"], record["grid"] = grid_on, tau_combos(grid_on, tau_multipliers)
            if pending := step(_fold_items(record["grid"], eligible, profile, out), "tau_grid"):
                return halt("tau_grid", pending)
    for unit, st in status:
        log(f"  {unit}: {st}")
    (out / "run.json").write_text(json.dumps(record, indent=1))
    return record


# --- reading outputs ----------------------------------------------------------------------
def _read(job, units, profile, out, args) -> dict[str, dict[str, pd.DataFrame]]:
    found = {}
    for u in units:
        path = job.output(out, u, args)
        if jobs.is_done(path, profile):
            found[u] = unpack(pd.read_parquet(path))
    return found


def race_outputs(out, races, profile) -> dict[str, dict[str, pd.DataFrame]]:
    return _read(jobs.JOBS["cross_race"], races, profile, out, _args(races))


def arm_outputs(out, arm, races, profile) -> dict[str, pd.DataFrame]:
    """One arm's folds, concatenated: {"g", "factors", "stop_gain", "reference_stops",
    "fold_error"}, empty frames when absent."""
    folds = _read(jobs.JOBS["h_fold"], races, profile, out, _args(races, arm=arm))
    cols = {"g": calibration.MIDRACE_COLUMNS, "factors": calibrators.FACTOR_ERROR_COLUMNS,
            "stop_gain": calibration.STOP_GAIN_COLUMNS, "reference_stops": REFERENCE_STOP_COLUMNS,
            "fold_error": FOLD_ERROR_COLUMNS}
    return {t: (pd.concat([f[t] for f in folds.values() if t in f], ignore_index=True)
                if any(t in f for f in folds.values()) else pd.DataFrame(columns=c)) for t, c in cols.items()}


# --- selection ------------------------------------------------------------------------------
def _boot_mean(values: pd.Series, n_boot=N_BOOT, seed=0) -> tuple[float, float]:
    """95 percent CI of the mean of per-race values, resampling races. NaN with fewer than 2."""
    v = values.dropna().to_numpy(dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


PAIRED_KEYS = ("mae", "mae_base", "mae_lo", "mae_hi", "bias", "diff", "diff_lo", "diff_hi", "mae_pooled",
               "mae_base_pooled", "diff_pooled")


def paired(arm_rows: pd.DataFrame, base_rows: pd.DataFrame, keys, value, n_boot=N_BOOT, seed=0) -> dict:
    """Arm against its comparator on the rows both have (matched on `keys`, race_id among them).

    The point estimates are race-weighted, the statistic the race-cluster CIs bootstrap, so a
    race with many rows does not dominate: mae and mae_base, the mean over races of each
    race's mean |value|, with mae's 95 percent CI; diff, the mean over races of the per-race
    paired difference (arm less comparator), with its CI; bias, the mean of per-race signed
    means. The row-pooled values (mae_pooled, mae_base_pooled, diff_pooled) are secondary."""
    a = arm_rows[keys + [value]].dropna(subset=[value])
    b = base_rows[keys + [value]].dropna(subset=[value])
    m = a.merge(b, on=keys, suffixes=("_arm", "_base"))
    out = {"n_rows": int(len(m)), "n_races": int(m["race_id"].nunique()) if len(m) else 0}
    if not len(m):
        return {**out, **dict.fromkeys(PAIRED_KEYS, np.nan)}
    ea, eb, race = m[f"{value}_arm"].abs(), m[f"{value}_base"].abs(), m["race_id"]
    ra, rb = ea.groupby(race).mean(), eb.groupby(race).mean()
    lo, hi = _boot_mean(ra - rb, n_boot, seed)
    mlo, mhi = _boot_mean(ra, n_boot, seed)
    return {**out, "mae": float(ra.mean()), "mae_base": float(rb.mean()), "mae_lo": mlo, "mae_hi": mhi,
            "bias": float(m[f"{value}_arm"].groupby(race).mean().mean()), "diff": float((ra - rb).mean()),
            "diff_lo": lo, "diff_hi": hi, "mae_pooled": float(ea.mean()), "mae_base_pooled": float(eb.mean()),
            "diff_pooled": float(ea.mean() - eb.mean())}


G_KEYS = ["race_id", "window", "driver", "k"]
SG_KEYS = ["race_id", "window", "k", "stop"]


def _g_window(rows, window, horizons=G_HORIZONS):
    return rows[(rows["window"] == window) & rows["horizon"].between(*horizons)] if len(rows) else rows


def passes_g(row) -> bool:
    """WORLD and pre_race rule: G mae lower (race-weighted paired difference below 0) AND its
    95 percent CI excludes 0 (NaN fails)."""
    return bool(row["g_diff"] < 0 and (row["g_diff_hi"] < 0 or row["g_diff_lo"] > 0))


def passes_belief(row) -> bool:
    """BELIEF rule: the paired stop-gain CI below 0 AND the paired G CI's upper end under the margin."""
    return bool(row["sg_diff_hi"] < 0 and row["g_diff_hi"] < NONINFERIORITY_S)


def rule_for(window: str) -> str:
    return "belief: stop-gain, G non-inferiority" if window == BELIEF_WINDOW else "G mae_offset 10-49"


def choose(table: pd.DataFrame, window: str, comparator=BASELINE) -> str:
    """The window's arm: among those that pass its rule, the lowest race-weighted primary metric
    (stop-gain mae for the belief, G mae_offset otherwise); ties to the simpler; else the comparator."""
    t = table[(table["window"] == window) & table["passes"]]
    if t.empty:
        return comparator
    metric = "sg_mae" if window == BELIEF_WINDOW else "g_mae"
    rank = {a: i for i, a in enumerate(SIMPLICITY)}
    t = t.assign(_m=t[metric].round(6), _s=[rank.get(parse_arm(a)[0], len(rank)) for a in t["arm"]])
    return str(t.sort_values(["_m", "_s"]).iloc[0]["arm"])


# the full window fits every race lap with no calibrator and no shrinkage fit, so these do nothing there
NA_ROLE = "n/a (same by construction)"
_TO_SNAPSHOT_ONLY_PARAMS = ("collinear_hold", "divide_data_by_s2")


def same_by_construction(arm: str, comparator: str, window: str) -> bool:
    """Whether `arm` is `comparator` in `window` by construction: in the full window an arm that
    differs only in its calibrator (none), its taus or a to_snapshot-only degradation option."""
    if window != "full" or arm == comparator:
        return False

    def world(name):
        cfg = arm_config(name)
        params = {k: v for k, v in cfg.degradation.params.items() if k not in _TO_SNAPSHOT_ONLY_PARAMS}
        default = SimConfig()
        return cfg.model_copy(update={"calibrator": default.calibrator, "shrinkage": default.shrinkage,
                                      "degradation": cfg.degradation.model_copy(update={"params": params})})
    return world(arm) == world(comparator)


def compare_arms(arm_data: dict[str, dict], comparator: str, windows, n_boot=N_BOOT, stage="arms",
                 horizons=G_HORIZONS) -> pd.DataFrame:
    """One row per (window, arm) against `comparator`: G (at `horizons`) and stop-gain mae with
    CIs, the paired differences with CIs (paired), and whether the window's rule passes. An arm
    the same as the comparator by construction (same_by_construction) has role NA_ROLE (shown as n/a)."""
    rows = []
    base = arm_data[comparator]
    for window in windows:
        for arm, d in arm_data.items():
            g = paired(_g_window(d["g"], window, horizons), _g_window(base["g"], window, horizons), G_KEYS,
                       "err_offset_s", n_boot)
            sw = d["stop_gain"][d["stop_gain"]["window"] == window] if len(d["stop_gain"]) else d["stop_gain"]
            bw = base["stop_gain"][base["stop_gain"]["window"] == window] if len(base["stop_gain"]) else base["stop_gain"]
            sg = paired(sw, bw, SG_KEYS, "err_s", n_boot)
            role = ("comparator" if arm == comparator else "coverage only" if parse_arm(arm)[0] in COVERAGE_ONLY
                    else NA_ROLE if same_by_construction(arm, comparator, window) else "candidate")
            row = {"stage": stage, "window": window, "arm": arm, "comparator": comparator, "rule": rule_for(window),
                   "role": role, "n_races": g["n_races"], "g_n": g["n_rows"],
                   **{f"g_{k}": v for k, v in g.items() if k not in ("n_rows", "n_races")},
                   "sg_n": sg["n_rows"], **{f"sg_{k}": v for k, v in sg.items() if k not in ("n_rows", "n_races")}}
            row["passes"] = role == "candidate" and (passes_belief(row) if window == BELIEF_WINDOW else passes_g(row))
            rows.append(row)
    return pd.DataFrame(rows)


BOUND_ARM = "wear_nonneg"
# Every comparison is incumbent-first: the incumbent is the arm the config carries now, and a
# change in either direction has to pass the window's rule. (arm, source, set by) per window.
INCUMBENTS = {
    "to_snapshot": (BOUND_ARM, "a priori",
                    "configs/sim/belief.yaml: wear bound 0 (SPEC 0017b, #40: the physical argument and the "
                    "stop-gain check)"),
    "full": (BASELINE, "default", "SimConfig defaults (configs/sim/world.yaml adds traffic only)"),
    "pre_race": (BASELINE, "default", "SimConfig defaults"),
}


def incumbent(window: str) -> str:
    return INCUMBENTS.get(window, (BASELINE,))[0]


def _physics(cfg: SimConfig) -> tuple:
    """What an arm sets: degradation model and params (draw left out), calibrator, free-air fit."""
    params = {k: v for k, v in cfg.degradation.params.items() if k not in ("draw", "clip_wear_at_zero")}
    return cfg.degradation.name, params, cfg.calibrator.name, cfg.fit_free_air_only, cfg.shrinkage


def current_defaults() -> pd.DataFrame:
    """Per window its incumbent arm, the arm's config hash, its source and what set it, and whether
    the config that carries it (belief.yaml for to_snapshot, SimConfig() otherwise) has the same
    physics, so the report and the config never disagree."""
    belief = SimConfig.from_file(ROOT / "configs" / "sim" / "belief.yaml")
    rows = []
    for w, (arm, source, set_by) in INCUMBENTS.items():
        cfg = arm_config(arm)
        carried = belief if w == BELIEF_WINDOW else SimConfig()
        rows.append({"window": w, "incumbent": arm, "config_hash": calibration.config_hash(cfg), "source": source,
                     "set by": set_by, "matches the config": _physics(cfg) == _physics(carried)})
    return pd.DataFrame(rows)


def belief_selection(arm_data, n_boot=N_BOOT) -> tuple[pd.DataFrame, str, str]:
    """The belief (to_snapshot) in two stages, incumbent-first. Stage 1 is the bound: the baseline
    (no bound) is the challenger against the incumbent wear_nonneg, so the bound stays unless
    removing it passes the belief rule. Stage 2 compares every other candidate against the
    stage-1 result, the belief default a candidate would replace (while provisional, the
    incumbent). Returns (both stages' rows, the stage-1 result, the choice)."""
    inc = incumbent(BELIEF_WINDOW)
    challenger = BASELINE if inc == BOUND_ARM else BOUND_ARM
    s1 = compare_arms({a: arm_data[a] for a in (inc, challenger)}, inc, [BELIEF_WINDOW], n_boot, "bound")
    bound = choose(s1, BELIEF_WINDOW, inc)
    others = [a for a in arm_data if a not in (BASELINE, BOUND_ARM)]
    s2 = compare_arms({bound: arm_data[bound], **{a: arm_data[a] for a in others}}, bound, [BELIEF_WINDOW], n_boot,
                      "candidates")
    return pd.concat([s1, s2], ignore_index=True), bound, choose(s2, BELIEF_WINDOW, bound)


def selection_tables(out, races, profile, n_boot=N_BOOT) -> dict:
    """The selection over the committed arms: {"table", "chosen": {window: arm}, "bound", "arm_data"}.
    The belief window in two stages (belief_selection), the others against their incumbent."""
    arm_data = {a: arm_outputs(out, a, races, profile) for a in ARMS}
    windows = list(profile.g.windows)
    parts, chosen, bound = [], {}, incumbent(BELIEF_WINDOW)
    for w in windows:
        if w == BELIEF_WINDOW:
            t, bound, chosen[w] = belief_selection(arm_data, n_boot)
        else:
            t = compare_arms(arm_data, incumbent(w), [w], n_boot)
            chosen[w] = choose(t, w, incumbent(w))
        parts.append(t)
    return {"table": pd.concat(parts, ignore_index=True), "chosen": chosen, "bound": bound, "arm_data": arm_data}


def s2_table(out, races, profile, chosen, n_boot=N_BOOT) -> pd.DataFrame:
    """The prior-sd fix alone: <chosen>__s2 against <chosen> under the belief rule."""
    data = {chosen: arm_outputs(out, chosen, races, profile), f"{chosen}__{S2}": arm_outputs(out, f"{chosen}__{S2}",
                                                                                           races, profile)}
    t = compare_arms(data, chosen, [BELIEF_WINDOW], n_boot, "prior-sd fix")
    return t[t["arm"] != chosen].reset_index(drop=True)


def tau_table(out, races, profile, arms, n_boot=N_BOOT) -> pd.DataFrame:
    """The tau grid ranked by the belief rule against its x1 combination (the current taus):
    rows that pass it first, then those non-inferior on G at TAU_HORIZONS (paired CI upper end
    under +NONINFERIORITY_S), each by race-weighted stop-gain mae."""
    if not arms:
        return pd.DataFrame()
    base_arm = parse_arm(arms[0])
    ones = tau_arm(base_arm[0] + (f"__{S2}" if base_arm[1] else ""), (1.0,) * len(TAU_KEYS))
    data = {a: arm_outputs(out, a, races, profile) for a in arms}
    t = compare_arms(data, ones, [BELIEF_WINDOW], n_boot, "tau grid", TAU_HORIZONS)
    t["passes"] = [passes_belief(r) for _, r in t.iterrows()]
    t["non_inferior"] = t["g_diff_hi"] < NONINFERIORITY_S
    for k, i in zip(TAU_KEYS, range(len(TAU_KEYS))):
        t[f"x_{k}"] = [parse_arm(a)[2][i] for a in t["arm"]]
    t = t.sort_values(["passes", "non_inferior", "sg_mae"], ascending=[False, False, True], kind="stable")
    return t.reset_index(drop=True)


# --- D: factors ------------------------------------------------------------------------------
FACTOR_KEYS = ["observed_pit_loss_s", "observed_sc_factor", "observed_sc_factor_median", "observed_vsc_factor",
               "observed_lap1_extra_s", "observed_sc_pit_loss_factor", "n_sc_stops", "observed_pack_penalty_s",
               "observed_pack_penalty_median_s", "free_air_share", "sc_count", "sc_laps", "vsc_count", "vsc_laps"]


def factors_table(outputs: dict[str, dict], features_path=RACE_FEATURES) -> pd.DataFrame:
    """Per race: every observed factor, SC/VSC periods (count, laps, start laps), free-air share,
    pack penalty, the practice-to-race base delta and fuel slope of the hindsight fit, pit loss,
    and the race_features.csv covariates (by race_id) when that file exists."""
    rows = []
    for rid, t in outputs.items():
        s = t["summary"].iloc[0]
        periods = t.get("periods", pd.DataFrame(columns=["kind", "start_lap", "end_lap"]))
        hind = dict(zip(t["hindsight"]["factor"], t["hindsight"]["value"])) if "hindsight" in t else {}
        row = {"race_id": rid, "total_laps": int(s["total_laps"]),
               "pit_loss_s": float(t["g"]["pit_loss_s"].iloc[0]) if "g" in t and len(t["g"]) else np.nan,
               **{k: s.get(k, np.nan) for k in FACTOR_KEYS}}
        for kind in ("sc", "vsc"):
            row[f"{kind}_start_laps"] = " ".join(str(int(x)) for x in periods.loc[periods["kind"] == kind, "start_lap"])
        row["hindsight_fuel_s_per_lap"] = hind.get("fuel_s_per_lap", np.nan)
        for name, v in sorted(hind.items()):
            if name.startswith("base_delta_s["):
                row[name] = v
        rows.append(row)
    table = pd.DataFrame(rows)
    path = Path(features_path) if features_path else None
    if path is not None and path.is_file() and len(table):
        feats = pd.read_csv(path)
        if "race_id" in feats:
            feats = feats.assign(race_id=feats["race_id"].map(canonical_race_id))
            table = table.merge(feats.rename(columns={c: f"feature:{c}" for c in feats if c != "race_id"}),
                                on="race_id", how="left")
    return table


def write_factors(outputs, out, features_path=RACE_FEATURES) -> Path:
    path = Path(out) / "factors.csv"
    factors_table(outputs, features_path).to_csv(path, index=False)
    return path


# --- E: report -------------------------------------------------------------------------------
def _fmt(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits=2) -> str:
    if df is None or not len(df):
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _ci(v, lo, hi):
    return f"{_fmt(v)} [{_fmt(lo)}, {_fmt(hi)}]"


def _gate_pairs(arm_data, windows) -> pd.DataFrame:
    rows = []
    for arm, d in arm_data.items():
        g = d["g"]
        for w in windows:
            t = g[g["window"] == w] if len(g) else g
            if not len(t):
                continue
            m = calibration._metrics(t)
            pairs = calibration.midrace_pairs(t)
            counted = (pairs["n_pairs"] - pairs["n_ties"]) if len(pairs) else pd.Series(dtype=float)
            rate = (float((pairs["sign_agree_rate"].fillna(0) * counted).sum() / counted.sum())
                    if len(counted) and counted.sum() > 0 else np.nan)
            rows.append({"arm": arm, "window": w, "within_2_rate_sim": m["within_2_rate_sim"],
                         "mae_pos_sim": m["mae_pos_sim"], "n_pairs": int(pairs["n_pairs"].sum()) if len(pairs) else 0,
                         "sign_agree_rate": rate})
    return pd.DataFrame(rows)


def _coverage(arm_data, arms, n_boot) -> pd.DataFrame:
    parts = []
    for arm in arms:
        g = arm_data.get(arm, {}).get("g")
        if g is None or not len(g) or "pred_p05_s" not in g:
            continue
        agg = calibration.aggregate(g, n_boot=n_boot).reset_index()
        parts.append(agg.assign(arm=arm)[["arm", "window", "horizon_bucket", "n", "coverage_68", "coverage_90",
                                          "mean_width_90_s", "mae_offset_s"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stop_gain_summary(arm_data, n_boot=500) -> pd.DataFrame:
    """Stop-gain error by window, arm and k bucket: n, bias, mae, the race-cluster 95% CI of the
    mae (NaN with fewer than 2 races) and the mean of each decomposition term."""
    rows = []
    for arm, d in arm_data.items():
        s = d["stop_gain"]
        if not len(s):
            continue
        for (window, kb), t in s.groupby(["window", "k_bucket"], sort=True):
            per_race = t.assign(a=t["err_s"].abs()).groupby("race_id")["a"].mean()
            lo, hi = _boot_mean(per_race, n_boot)
            rows.append({"window": window, "arm": arm, "k_bucket": int(kb), "n": int(len(t)),
                         "n_races": int(t["race_id"].nunique()), "bias_s": float(t["err_s"].mean()),
                         "mae_s": float(t["err_s"].abs().mean()), "mae_ci_lo": lo, "mae_ci_hi": hi,
                         **{c: float(t[c].mean()) for c in calibration.STOP_GAIN_TERMS}})
    return pd.DataFrame(rows)


STOP_GAIN_RACE_COLUMNS = ["window", "race_id", "arm", "n", "n_stops", "real_gain_s", "pred_gain_s", "bias_s", "mae_s"]


def stop_gain_by_race(arm_data) -> pd.DataFrame:
    """Stop-gain per window, race and arm, over the stop-gain rows (one per stop and k): n rows,
    distinct stops, mean real and predicted gain, the signed bias (mean pred - real, err_s) and
    the mae. The selection's sg_mae and sg_bias are the means over races of mae_s and bias_s."""
    rows = []
    for arm, d in arm_data.items():
        s = d["stop_gain"]
        if not len(s):
            continue
        for (window, race), t in s.groupby(["window", "race_id"], sort=True):
            rows.append({"window": window, "race_id": race, "arm": arm, "n": int(len(t)),
                         "n_stops": int(t["stop"].nunique()), "real_gain_s": float(t["real_gain_s"].mean()),
                         "pred_gain_s": float(t["pred_gain_s"].mean()), "bias_s": float(t["err_s"].mean()),
                         "mae_s": float(t["err_s"].abs().mean())})
    return pd.DataFrame(rows, columns=STOP_GAIN_RACE_COLUMNS)


def _counts(values) -> str:
    """'1:3 2:16': each stop count and how many cars, ascending."""
    v = pd.Series(values).dropna().astype(int)
    return " ".join(f"{k}:{n}" for k, n in v.value_counts().sort_index().items()) or "-"


def stop_mode(values):
    """The most common stop count, the smaller one on a tie; None when there are none."""
    v = pd.Series(values).dropna().astype(int)
    if not len(v):
        return None
    c = v.value_counts()
    return int(min(c.index[c == c.max()]))


STOP_COUNT_COLUMNS = ["race_id", "field_cars", "field_stops", "field_mode", "arm", "ref_cars", "ref_stops",
                      "ref_mode", "mode_match", "two_less_one_s_median", "two_less_one_s_min", "two_less_one_s_max"]


def stop_count_table(outputs: dict[str, dict], arm_data: dict[str, dict]) -> pd.DataFrame:
    """The stop-count check, per race and arm: the field's stop counts (cars within
    FIELD_WITHIN_LAPS laps of the distance, field_stops) and their mode, against the reference's
    stop counts from lap 0 under the arm's world (reference_stops, the same cars) and their mode;
    mode_match when the modes agree; two_less_one_s (best two-stop less best one-stop, above 0
    when one stop is quicker) over the cars: median, min and max."""
    rows = []
    for rid, t in outputs.items():
        field = t.get("field_stops", pd.DataFrame(columns=FIELD_STOP_COLUMNS))
        fmode = stop_mode(field["stops"])
        for arm, d in arm_data.items():
            ref = d.get("reference_stops", pd.DataFrame(columns=REFERENCE_STOP_COLUMNS))
            ref = ref[ref["race_id"] == rid] if len(ref) else ref
            if not len(ref):
                continue
            gap = pd.to_numeric(ref["two_less_one_s"], errors="coerce").dropna()
            rmode = stop_mode(ref["ref_stops"])
            rows.append({"race_id": rid, "field_cars": int(len(field)), "field_stops": _counts(field["stops"]),
                         "field_mode": fmode, "arm": arm, "ref_cars": int(len(ref)),
                         "ref_stops": _counts(ref["ref_stops"]), "ref_mode": rmode,
                         "mode_match": fmode is not None and rmode == fmode,
                         "two_less_one_s_median": float(gap.median()) if len(gap) else np.nan,
                         "two_less_one_s_min": float(gap.min()) if len(gap) else np.nan,
                         "two_less_one_s_max": float(gap.max()) if len(gap) else np.nan})
    return pd.DataFrame(rows, columns=STOP_COUNT_COLUMNS)


def coverage_summary(arm_data, n_boot=500) -> pd.DataFrame:
    """Interval coverage by window, arm and horizon bucket, for the pace-corrected target
    (coverage_68/90) and the raw one (coverage_raw_68/90), with the mean 90% width."""
    parts = []
    for arm, d in arm_data.items():
        g = d["g"]
        if not len(g) or "pred_p05_s" not in g:
            continue
        agg = calibration.aggregate(g, n_boot=n_boot).reset_index().assign(arm=arm)
        parts.append(agg[["window", "arm", "horizon_bucket", "n", "coverage_68", "coverage_90",
                          "coverage_raw_68", "coverage_raw_90", "mean_width_90_s"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _decomposition(arm_data, window=BELIEF_WINDOW) -> pd.DataFrame:
    rows = []
    for arm, d in arm_data.items():
        s = d["stop_gain"]
        s = s[s["window"] == window] if len(s) else s
        if not len(s):
            continue
        rows.append({"arm": arm, "n": int(len(s)), "err_s": float(s["err_s"].mean()),
                     **{t: float(s[t].mean()) for t in calibration.STOP_GAIN_TERMS}})
    return pd.DataFrame(rows)


SPAIN = "2021_04_spain"


def changed_from_spain(factors: pd.DataFrame) -> pd.DataFrame:
    """The Spain-derived constants against what the eligible races measure."""
    cfg, ev = SimConfig(), EventModel()
    if not len(factors):
        return pd.DataFrame()
    green = (factors["total_laps"] - factors["sc_laps"].fillna(0) - factors["vsc_laps"].fillna(0)).clip(lower=1)
    base = factors[[c for c in factors if c.startswith("base_delta_s[")]]
    per_race = {
        "sc_lap_factor": factors["observed_sc_factor"],
        "lap1_extra_s": factors["observed_lap1_extra_s"],
        "sc_pit_loss_factor": factors["observed_sc_pit_loss_factor"],
        "pre_race_base_delta_s": base.mean(axis=1) if base.shape[1] else pd.Series(np.nan, index=factors.index),
        "pre_race_fuel_slope_s_per_lap": factors["hindsight_fuel_s_per_lap"],
        "p_sc (per green lap)": factors["sc_count"].fillna(0) / green,
        "d_sc (laps)": factors["sc_laps"] / factors["sc_count"].where(factors["sc_count"] > 0),
    }
    spain_values = {"sc_lap_factor": cfg.effects.sc_lap_factor, "lap1_extra_s": cfg.effects.lap1_extra_s,
                    "sc_pit_loss_factor": cfg.effects.sc_pit_loss_factor,
                    "pre_race_base_delta_s": cfg.pre_race_base_delta_s,
                    "pre_race_fuel_slope_s_per_lap": cfg.pre_race_fuel_slope_s_per_lap,
                    "p_sc (per green lap)": ev.p_sc, "d_sc (laps)": ev.d_sc}
    spain = factors["race_id"] == SPAIN
    rows = []
    for name, v in per_race.items():
        v = pd.to_numeric(v, errors="coerce")
        ok = v.dropna()
        rows.append({"constant": name, "config (from Spain)": spain_values[name],
                     "Spain measured": float(v[spain].iloc[0]) if spain.any() and pd.notna(v[spain].iloc[0]) else np.nan,
                     "races": int(len(ok)), "mean": float(ok.mean()) if len(ok) else np.nan,
                     "sd": float(ok.std()) if len(ok) > 1 else np.nan,
                     "min": float(ok.min()) if len(ok) else np.nan, "max": float(ok.max()) if len(ok) else np.nan})
    return pd.DataFrame(rows)


def _selection_md(t: pd.DataFrame) -> str:
    """One stage's rows: race-weighted G and SG mae and paired differences [95% CIs], the
    row-pooled differences beside them; an arm the same by construction shows n/a."""
    na = t["role"] == NA_ROLE

    def col(f):
        return ["n/a" if n else f(r) for n, (_, r) in zip(na, t.iterrows())]
    return md_table(pd.DataFrame({
        "arm": t["arm"].to_numpy(), "role": t["role"].to_numpy(), "races": t["n_races"].to_numpy(),
        "G mae": col(lambda r: _ci(r["g_mae"], r["g_mae_lo"], r["g_mae_hi"])),
        "G bias": col(lambda r: _fmt(r["g_bias"])),
        "G diff": col(lambda r: _ci(r["g_diff"], r["g_diff_lo"], r["g_diff_hi"])),
        "G diff pooled": col(lambda r: _fmt(r["g_diff_pooled"])),
        "SG mae": col(lambda r: _ci(r["sg_mae"], r["sg_mae_lo"], r["sg_mae_hi"])),
        "SG bias": col(lambda r: _fmt(r["sg_bias"])),
        "SG diff": col(lambda r: _ci(r["sg_diff"], r["sg_diff_lo"], r["sg_diff_hi"])),
        "SG diff pooled": col(lambda r: _fmt(r["sg_diff_pooled"])),
        "passes": col(lambda r: _fmt(bool(r["passes"])))}))


def held_out_md(held) -> str:
    """The report's held-out line from run.json's "held_out" (None for a run before the guard):
    what eval_set.json lists, which of them this run left out for that reason (and which
    --races auto skipped), and any let in with --allow-held-out."""
    if held is None:
        return "Not recorded: this run.json predates the held-out guard."
    excluded = [r for r in held["listed"] if r not in held["included"]]
    return (f"{held['eval_set']} holds out {', '.join(held['listed']) or 'no race'}. Excluded from this run for "
            f"that reason: {', '.join(excluded) or 'none'}"
            + (f" (--races auto skipped {', '.join(held['skipped'])})" if held["skipped"] else "") + "."
            + (f" Run anyway with --allow-held-out: {', '.join(held['included'])}." if held["included"] else ""))


def report(out=CROSS_DIR, profile=None, features_path=RACE_FEATURES, n_boot=N_BOOT) -> Path:
    """D and E from a run's outputs under `out` (run.json names the races and the profile).
    Refuses a run with a synthetic race when `out` is CROSS_DIR (guard_committed)."""
    out = Path(out)
    record = json.loads((out / "run.json").read_text())
    guard_committed(out, record["races"])
    elig = json.loads((out / "eligibility.json").read_text())
    profile = load_profile(record["profile"]) if profile is None else profile
    eligible = record["eligible"]
    outputs = race_outputs(out, eligible, profile)
    factors = factors_table(outputs, features_path)
    factors.to_csv(out / "factors.csv", index=False)
    arms_table(ARMS).to_csv(out / "arms.csv", index=False)
    sel = selection_tables(out, eligible, profile, n_boot)
    table, chosen = sel["table"], sel["chosen"]
    s2 = s2_table(out, eligible, profile, record["belief_choice"], n_boot) if record.get("s2_arm") else pd.DataFrame()
    provisional = len(eligible) < MIN_RACES
    selection = pd.concat([table, s2], ignore_index=True).assign(provisional=provisional)
    selection.to_csv(out / "selection.csv", index=False)
    grid = tau_table(out, eligible, profile, record.get("grid") or [], n_boot)
    grid.to_csv(out / "tau_grid.csv", index=False)
    g_parts = [calibration.aggregate(d["g"], n_boot=min(n_boot, 500)).reset_index().assign(arm=a)
               for a, d in sel["arm_data"].items() if len(d["g"])]
    g_by_arm = pd.concat(g_parts, ignore_index=True) if g_parts else pd.DataFrame()
    g_by_arm.to_csv(out / "g_by_arm.csv", index=False)
    sg_parts = [d["stop_gain"].assign(arm=a) for a, d in sel["arm_data"].items() if len(d["stop_gain"])]
    stop_gain = pd.concat(sg_parts, ignore_index=True) if sg_parts else pd.DataFrame()
    stop_gain.to_csv(out / "stop_gain.csv", index=False)
    stop_gain_summary(sel["arm_data"]).to_csv(out / "stop_gain_summary.csv", index=False)
    sg_race = stop_gain_by_race(sel["arm_data"])
    sg_race.to_csv(out / "stop_gain_by_race.csv", index=False)
    stop_counts = stop_count_table(outputs, sel["arm_data"])
    stop_counts.to_csv(out / "stop_counts.csv", index=False)
    coverage_summary(sel["arm_data"]).to_csv(out / "coverage_summary.csv", index=False)
    windows = list(profile.g.windows)

    md = ["# Cross-race calibration report (SPEC 0016)", ""]
    md.append(f"sim_code_version `{record['sim_code_version']}`, profile `{record['profile']}` "
              f"(`{record['profile_hash']}`), {len(record['races'])} race(s) considered, {len(eligible)} eligible.")
    md.append("")
    if provisional:
        md += [f"**PROVISIONAL: {len(eligible)} eligible race(s), fewer than {MIN_RACES}. The selection below is "
               "provisional and the defaults stay** (belief.yaml, world.yaml and the SimConfig defaults are unchanged).",
               ""]
    synthetic = [r for r in record["races"] if is_synthetic(r)]
    if synthetic:
        md += [f"Synthetic races in this run (pipeline test only, not evidence): {', '.join(synthetic)}.", ""]
    md += ["All G and stop-gain numbers are held out: each race's rows come from its own leave-one-race-out "
           "fold, and Spain is one race among the rest.", ""]

    md += ["## Held-out races", "", held_out_md(record.get("held_out")), ""]

    md += ["## Eligibility", "", "Drivers: eval.scenarios.eligible_drivers (laps 1..n with no gap, at most two laps "
           "down), the eval sampler's set.", ""]
    et = pd.DataFrame([{"race": e["race_id"], "eligible": e["eligible"], "drivers": e["n_finishers"],
                        "wet share": e["wet_laps_share"], "red flags": " ".join(map(str, e["red_flags"])) or "-",
                        "SC": " ".join(f"{a}-{b}" for a, b in e["sc_periods"]) or "-",
                        "VSC": " ".join(f"{a}-{b}" for a, b in e["vsc_periods"]) or "-",
                        "dry compounds": " ".join(e["dry_compounds"]),
                        "reasons": "; ".join(e["reasons"]) or "-"} for e in elig])
    md += [md_table(et), ""]
    notes = [f"{e['race_id']}: {n}" for e in elig for n in e.get("notes", [])]
    if notes:
        md += ["Notes: " + " ".join(notes), ""]

    md += ["## Arms", "", md_table(arms_table(ARMS)[["arm", "file", "file_sha256", "config_hash", "role"]]), ""]

    md += ["## Current defaults", "", "Each window's incumbent: the arm its config carries now. Every comparison "
           "is incumbent-first, so a change in either direction has to pass the window's rule.", "",
           md_table(current_defaults()), ""]

    md += ["## Selection", "",
           f"G: mae_offset_s at horizons {G_HORIZONS[0]}-{G_HORIZONS[1]}; SG: stop-gain mae_s at real pit stops, "
           "and SG bias its signed mean (predicted less real gain: below 0 when the model under-predicts the gain "
           "from fresh tyres), race-weighted. "
           "mae and diff are race-weighted (the mean over races of each race's value, the statistic the "
           "race-cluster 95% CI in brackets resamples); diff is the per-race paired difference, challenger less "
           "the stage's comparator (the incumbent); 'pooled' is the row-pooled difference, secondary. "
           f"Belief rule (to_snapshot): SG diff CI below 0 and G diff CI upper end under +{NONINFERIORITY_S:g} s, "
           f"in two stages: the bound (the baseline, no bound, challenging the incumbent {BOUND_ARM}; the bound "
           "stays unless removing it passes), then every other candidate against the stage-1 result. World "
           "(full) and pre_race: G diff below 0 with its CI excluding 0, against the SimConfig default. Ties to "
           "the simpler arm. n/a: the same as the comparator in that window by construction (in the full window "
           "there is no calibrator and no shrinkage fit).", ""]
    errors = [d["fold_error"] for d in sel["arm_data"].values() if len(d.get("fold_error", ()))]
    if errors:
        md += ["Folds not run (the arm cannot replay that race; its rows pair on the other races, so its "
               "races column is lower):", "", md_table(pd.concat(errors, ignore_index=True)[FOLD_ERROR_COLUMNS]), ""]
    for w in windows:
        t = table[table["window"] == w]
        inc = incumbent(w)
        md += [f"### {w} ({rule_for(w)}; incumbent {inc})", ""]
        for stage, part in t.groupby("stage", sort=False):
            if w == BELIEF_WINDOW:
                label = (f"1, the bound: {BASELINE} (no bound) challenging the incumbent {inc}" if stage == "bound"
                         else f"2, against the stage-1 result {part['comparator'].iloc[0]}")
                md += [f"Stage {label}:", ""]
            md += [_selection_md(part), ""]
        pick = chosen[w]
        if w == BELIEF_WINDOW:
            kept = sel["bound"] == inc
            md += [f"Stage 1 result: {sel['bound']} ("
                   + ("the incumbent stays" if kept else "the challenger passes") + (", provisional)." if provisional
                                                                                      else ")."), ""]
        verdict = (f"provisional choice {pick}; the defaults stay" if provisional else
                   f"{pick} replaces the incumbent {inc}" if pick != inc else f"the incumbent {inc} stays")
        md += [f"Decision for {w}: **{verdict}**.", ""]
    md += ["### Prior-sd fix (SPEC 0017 G), measured alone", ""]
    if len(s2):
        r = s2.iloc[0]
        md += [f"{r['arm']} against {r['comparator']} under the belief rule: G diff "
               f"{_ci(r['g_diff'], r['g_diff_lo'], r['g_diff_hi'])}, SG diff "
               f"{_ci(r['sg_diff'], r['sg_diff_lo'], r['sg_diff_hi'])}; passes: {_fmt(bool(r['passes']))}"
               + (" (provisional)." if provisional else "."), ""]
    else:
        md += ["Not run. " + " ".join(record.get("notes", [])), ""]
    md += [f"### Tau grid on {record.get('grid_arm') or '-'} (belief rule against the x1 taus: stop-gain mae "
           f"primary, G at horizons {TAU_HORIZONS[0]}-{TAU_HORIZONS[1]} non-inferior within +{NONINFERIORITY_S:g} s)",
           ""]
    if len(grid):
        show = grid.assign(**{
            "SG mae": [_fmt(v) for v in grid["sg_mae"]],
            "SG bias": [_fmt(v) for v in grid["sg_bias"]],
            "SG diff": [_ci(a, b, c) for a, b, c in zip(grid["sg_diff"], grid["sg_diff_lo"], grid["sg_diff_hi"])],
            "G mae 20-40": [_fmt(v) for v in grid["g_mae"]],
            "G diff": [_ci(a, b, c) for a, b, c in zip(grid["g_diff"], grid["g_diff_lo"], grid["g_diff_hi"])]})
        cols = ["arm", *[f"x_{k}" for k in TAU_KEYS], "SG mae", "SG bias", "SG diff", "G mae 20-40", "G diff", "non_inferior",
                "passes"]
        one = grid[grid[[f"x_{k}" for k in TAU_KEYS]].eq(1.0).all(axis=1)]
        rank = int(one.index[0]) + 1 if len(one) else None
        md += [md_table(show[cols].head(5)), f"{len(grid)} combinations; the current taus (x1) rank {rank or '-'}"
               + ("; provisional, nothing adopted." if provisional else "."), ""]
    else:
        md += ["Not run.", ""]

    md += ["## Stop-gain per race", "", "Per window, race and arm, over the held-out stop-gain rows (one per real "
           "stop and snapshot k before it): mean real and predicted gain (s/lap, the last two clean laps before the "
           "stop less the first two after it), the signed bias (predicted less real) and the mae. "
           "stop_gain_by_race.csv holds the rows.", ""]
    for w in windows:
        part = sg_race[sg_race["window"] == w]
        if len(part):
            md += [f"### {w}", "", md_table(part.drop(columns=["window"]).rename(columns={
                "race_id": "race", "n_stops": "stops", "real_gain_s": "real", "pred_gain_s": "pred",
                "bias_s": "SG bias", "mae_s": "SG mae"})), ""]

    md += ["## Stop-count check (lap 0)", "", f"Per race: the field's stop counts (pit-in laps of the cars within "
           f"{FIELD_WITHIN_LAPS} laps of the distance, 'stops:cars') and their mode, against the reference's from "
           "lap 0 for the same cars (optimizer_bridge.optimize_batch on the arm's world: full window, one sample, "
           f"up to {profile.sim.max_stops} stops) and its mode; 2-1 is the best two-stop less the best one-stop "
           "remaining time over the cars (s; above 0 when one stop is quicker), median [min, max]. "
           "stop_counts.csv holds the rows.", "",
           f"Note: the reference search covers at most max_stops = {profile.sim.max_stops} stops (the profile's "
           "sim.max_stops), so a field mode above that can never be matched; a car that stopped more often counts "
           "in the field but has no reference plan of its count.", ""]
    if len(stop_counts):
        md += [md_table(pd.DataFrame({
            "race": stop_counts["race_id"], "field (cars)": stop_counts["field_cars"],
            "field stops": stop_counts["field_stops"], "field mode": stop_counts["field_mode"],
            "arm": stop_counts["arm"], "reference stops": stop_counts["ref_stops"],
            "reference mode": stop_counts["ref_mode"], "match": stop_counts["mode_match"],
            "2-1 (s)": [_ci(a, b, c) for a, b, c in zip(stop_counts["two_less_one_s_median"],
                                                         stop_counts["two_less_one_s_min"],
                                                         stop_counts["two_less_one_s_max"])]})), ""]
    else:
        md += ["Not run.", ""]

    md += ["## Coverage (intervals at the profile's n_samples; only the posterior arm has a width)", ""]
    cov = _coverage(sel["arm_data"], [BASELINE, *COVERAGE_ONLY], min(n_boot, 500))
    md += [md_table(cov), ""]

    md += ["## Gate and pairs", "", "Per arm and window, over every held-out G row: the finishing-position gate "
           "(simulated order) and team-mate sign agreement on remaining time.", "",
           md_table(_gate_pairs(sel["arm_data"], windows)), ""]
    gates = [t["gate"].assign(race=r) for r, t in outputs.items() if "gate" in t]
    if gates:
        gt = pd.concat(gates, ignore_index=True)
        md += ["Pace gate per race (calibration.positions, simulated order, baseline config):", "",
               md_table(gt[["race", "pace", "n_drivers", "within_2_rate", "mae_pos", "exact_rate"]]), ""]

    md += ["## Stop-gain decomposition (to_snapshot, mean per stop and k)", "",
           "err_s = pred - real gain; the terms sum to it against each arm's own full-window fit (t_floor is "
           "that fit's own error, which includes the behavioural part no lap-time model of tyre age captures: "
           "drivers manage the last laps of a stint and push the first laps of the next).", "",
           md_table(_decomposition(sel["arm_data"]), 3), ""]

    md += ["## Factor summary", "", "factors.csv holds every race's row; the per-circuit fuel prior and the "
           "0011b training table come from it.", ""]
    text = ["race_id", "sc_start_laps", "vsc_start_laps"]
    num = (factors.drop(columns=[c for c in text if c in factors]).apply(pd.to_numeric, errors="coerce")
           .dropna(axis=1, how="all"))
    if len(num):
        fs = pd.DataFrame({"factor": num.columns, "races": num.notna().sum().to_numpy(),
                           "mean": num.mean().to_numpy(),
                           "sd": num.std().to_numpy() if len(num) > 1 else np.nan,
                           "min": num.min().to_numpy(), "max": num.max().to_numpy()})
        md += [md_table(fs, 3), ""]
    if not (features_path and Path(features_path).is_file()):
        md += [f"No race features file ({Path(features_path).name if features_path else 'none given'}): "
               "no circuit covariates yet.", ""]

    md += ["## What changed from Spain", "", "The constants set from 2021 Spain against what the eligible races "
           "measure (Spain's own value included in the mean).", "", md_table(changed_from_spain(factors), 3), ""]
    if len(eligible) <= 1:
        md += ["Only one eligible race: nothing has changed from Spain yet; every number here is Spain's.", ""]
    path = out / "summary.md"
    path.write_text("\n".join(md))
    return path


# --- cli ----------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m src.sim.calibrate cross", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="eligibility, per-race tables and LORO per arm (resumable)")
    r.add_argument("--races", nargs="+", default=list(jobs.DEFAULT_RACES),
                   help="race ids, or auto: every cached race (real_run.cached_races) less the held-out ones")
    r.add_argument("--allow-held-out", action="store_true",
                   help="run races data/eval_set.json holds out (refused without it; auto skips them)")
    r.add_argument("--eval-set", default=str(EVAL_SET), help="the held-out race list (default data/eval_set.json)")
    r.add_argument("--synthetic", type=int, default=0, help="add this many synthetic races (pipeline test)")
    r.add_argument("--tau-multipliers", nargs="+", type=float, default=list(TAU_MULTIPLIERS))
    r.add_argument("--no-tau-grid", action="store_true")
    r.add_argument("--n-jobs", type=int, help="default: the profile's parallel.n_jobs")
    for p in (r, sub.add_parser("report", help="factors.csv and summary.md from a run")):
        p.add_argument("--profile", help="compute profile; report: the run's, from run.json")
        p.add_argument("--out", default=str(CROSS_DIR))
        p.add_argument("--race-features", default=str(RACE_FEATURES))
    args = ap.parse_args(argv)
    if args.command == "run":
        if "auto" in args.races and len(args.races) > 1:
            ap.error("--races takes auto or race ids, not both")
        skipped = []
        if args.races == ["auto"]:
            from .real_run import cached_races
            races, notes = cached_races()
            for n in notes:
                print(f"auto: {n}")
            if not args.allow_held_out:
                races, skipped = split_held_out(races, args.eval_set)
                for r in skipped:
                    print(f"auto: {r} held out in {args.eval_set}, skipped")
        else:
            races = list(args.races)
        races += [f"2099_{i + 1:02d}_synthetic" for i in range(args.synthetic)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            run(races, load_profile(args.profile), args.out, tuple(args.tau_multipliers), not args.no_tau_grid,
                args.n_jobs, eval_set=args.eval_set, allow_held_out=args.allow_held_out, held_out_skipped=skipped)
    path = report(args.out, load_profile(args.profile) if args.profile else None, args.race_features)
    print(f"written to {path}")
    return 0
