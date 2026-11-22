"""Per-race observations of the calibration races, read as of each race on the policy side.

    python -m src.sim.race_history build [--races ...]

docs/eval.md "Information contract": what a policy reads that was estimated from other races
must record the races it was fitted on and, pricing race R, use only races that started before
R (race_ids.as_of); with no earlier race it falls back to a documented a priori constant with a
wide sd, never to a later race. This module holds two committed tables, built from the
calibration races (compound_steps.CALIBRATION_RACES; a race data/eval_set.json holds out is
refused before it is loaded):

  configs/sim/race_history.csv        one row per race: its hindsight factors
                                      (calibrators.hindsight_targets: base_delta_s per compound and
                                      their mean, sc_lap_factor, lap1_extra_s) and its fuel target
                                      (fuel_prior.race_target, the free-air full-window fuel slope)
                                      with the race's circuit
  configs/sim/neutralised_history.csv the race's SC and VSC periods (calibration.observed_periods),
                                      in event_rates' neutralised_periods format

and the as-of readers:

  as_of_factor(race_id, name)   base_delta_s, sc_lap_factor, lap1_extra_s: the median of the
                                earlier races, sd their spread floored at SD_FLOOR (A_PRIORI's sd
                                with one earlier race); A_PRIORI with none
  as_of_fuel(race, ...)         fuel_prior.circuit_history_prediction on the earlier races, else the
                                physics prior (fuel_prior.practice_longrun_prediction, a priori)
  as_of_event_rates(race_id)    event_rates.fit on the earlier races' periods, "global" level,
                                under the a priori EVENT_PRIOR

The "as_of_history" calibrator (below) puts the first two into the belief's prior and its event
factors; RateEvents(as_of=True) (events.py) draws periods with the third. The world (the judge)
does not read these tables.

A PRIORI constants (no earlier race): engineering priors stated here, not fitted on any race.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import CalibrationOutput
from .calibrators import CALIBRATORS, ShrinkageCalibrator, split_factor
from .config import EventModel
from .race_ids import as_of, canonical_race_id, parse_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_FILE = ROOT / "configs" / "sim" / "race_history.csv"
PERIODS_FILE = ROOT / "configs" / "sim" / "neutralised_history.csv"
FACTORS = ("base_delta_s", "sc_lap_factor", "lap1_extra_s")
# (mean, sd) where no earlier race exists: a priori, not fitted. base delta: race laps are a few
# tenths slower than practice long runs (track, fuel and engine modes), the sign and size unsure;
# an SC lap about 40 percent slower than a green one; a standing start costs several seconds on lap 1
A_PRIORI = {"base_delta_s": (0.5, 1.0), "sc_lap_factor": (1.4, 0.15), "lap1_extra_s": (8.0, 4.0)}
# the smallest sd an as-of median carries (a few races agreeing is not certainty), a priori
SD_FLOOR = {"base_delta_s": 0.3, "sc_lap_factor": 0.05, "lap1_extra_s": 1.0}
# the a priori SC/VSC rates event_rates shrinks the earlier races toward: about one SC start per
# race distance (1 / 60 per green lap) lasting 3 to 4 laps; the VSC values the EventModel placeholders
EVENT_PRIOR = EventModel(p_sc=1 / 60, d_sc=3.5, p_vsc=0.02, d_vsc=2.0)
EVENT_LEVEL = "global"


# --- the tables ---------------------------------------------------------------------------------
_TABLES: dict = {}


def load_history(path=HISTORY_FILE) -> pd.DataFrame:
    key = ("history", str(path))
    if key not in _TABLES:
        _TABLES[key] = pd.read_csv(path) if Path(path).is_file() else pd.DataFrame(columns=["race_id"])
    return _TABLES[key]


def load_periods(path=PERIODS_FILE) -> pd.DataFrame:
    key = ("periods", str(path))
    if key not in _TABLES:
        _TABLES[key] = (pd.read_csv(path, keep_default_na=False, na_values={"start_lap": [""], "end_lap": [""]})
                        if Path(path).is_file() else pd.DataFrame(columns=["race_id"]))
    return _TABLES[key]


def recorded_races(path=HISTORY_FILE) -> list[str]:
    """Every race the history tables record."""
    races = set(load_history(path)["race_id"])
    if Path(PERIODS_FILE).is_file():
        races |= set(load_periods()["race_id"])
    return sorted(map(str, races))


def earlier_races(race_id, races) -> list[str]:
    """race_ids.as_of, with an id that has no date (a synthetic "syn") having no earlier race."""
    try:
        return as_of(canonical_race_id(str(race_id)), list(races))
    except ValueError:
        return []


# --- as-of readers --------------------------------------------------------------------------------
def as_of_factor(race_id, name, history: pd.DataFrame | None = None) -> dict:
    """{mean, sd, source, races} of `name` (FACTORS) for `race_id` from the races before it."""
    h = load_history() if history is None else history
    col = "base_delta_s" if name == "base_delta_s" else name
    earlier = set(earlier_races(race_id, h["race_id"]))
    rows = h[h["race_id"].isin(earlier)] if col in h else h.iloc[0:0]
    v = pd.to_numeric(rows[col], errors="coerce").dropna() if col in rows else pd.Series(dtype=float)
    races = sorted(rows.loc[v.index, "race_id"]) if len(v) else []
    mean0, sd0 = A_PRIORI[name]
    if not len(v):
        return {"mean": mean0, "sd": sd0, "source": "a priori (no earlier race)", "races": []}
    sd = max(float(v.std(ddof=1)), SD_FLOOR[name]) if len(v) >= 2 else sd0
    return {"mean": float(v.median()), "sd": sd, "source": f"as of {len(v)} earlier race(s)", "races": races}


def as_of_fuel(race, history: pd.DataFrame | None = None, practice_model=None, practice=None) -> dict:
    """{mean, sd, source, races}: the fuel prior for `race` (RaceParams), the circuit history of
    the earlier races (fuel_prior.circuit_history_prediction), else the physics prior."""
    from . import fuel_prior
    h = load_history() if history is None else history
    earlier = set(earlier_races(race.race_id, h["race_id"]))
    train = h[h["race_id"].isin(earlier) & h["target"].notna()] if "target" in h else h.iloc[0:0]
    if len(train) >= 2:
        circuit = _circuit(race.race_id)
        p = fuel_prior.circuit_history_prediction(train, circuit)
        if p is not None and not (math.isnan(p["mean"]) or math.isnan(p["sd"])):
            return {"mean": p["mean"], "sd": max(float(p["sd"]), fuel_prior.MIN_SD), "source": p["source"],
                    "races": sorted(train["race_id"])}
    p = fuel_prior.practice_longrun_prediction(race, practice_model, practice)
    return {"mean": p["mean"], "sd": p["sd"], "source": f"a priori {p['source']}", "races": []}


def _circuit(race_id) -> str:
    from .event_rates import circuit_info
    try:
        parse_race_id(canonical_race_id(race_id))
    except ValueError:
        return str(race_id)
    if "synthetic" in str(race_id):
        return str(race_id)
    return str(circuit_info([canonical_race_id(race_id)]).iloc[0]["circuit"])


_RATES: dict = {}


def as_of_event_rates(race_id, periods: pd.DataFrame | None = None) -> dict:
    """{p_sc, d_sc, p_vsc, d_vsc, source, races}: event_rates' global posterior on the dry races
    before `race_id`, shrunk toward EVENT_PRIOR (its means with no earlier race)."""
    from . import event_rates
    p = load_periods() if periods is None else periods
    before = tuple(earlier_races(race_id, sorted(set(p["race_id"])))) if len(p) else ()
    key = (len(p), tuple(sorted(set(map(str, p["race_id"])))) if len(p) else (), before)
    if key not in _RATES:
        stats = event_rates.stats_from_periods(p[p["race_id"].isin(before)]) if before else \
            pd.DataFrame(columns=event_rates.STAT_COLUMNS)
        stats = event_rates.dry(stats)
        fit = event_rates.fit(stats, prior=EVENT_PRIOR)
        row = {"race_id": str(race_id), "era": "", "circuit": event_rates.UNKNOWN, "circuit_type": event_rates.UNKNOWN}
        r = fit.rates(row, EVENT_LEVEL)
        _RATES[key] = {"p_sc": r["p_sc"], "d_sc": r["d_sc"], "p_vsc": r["p_vsc"], "d_vsc": r["d_vsc"],
                       "source": (f"as of {len(stats)} earlier dry race(s), global level" if len(stats)
                                  else "a priori (no earlier race)"), "races": sorted(stats["race_id"])}
    return _RATES[key]


# --- the calibrator --------------------------------------------------------------------------------
def _register(cls):
    """Register once: `python -m src.sim.race_history` runs this file as __main__ besides the
    package module that calibrators.py imports."""
    return cls if cls.name in CALIBRATORS else CALIBRATORS.register(cls.name)(cls)


@_register
class AsOfHistoryCalibrator(ShrinkageCalibrator):
    """The shrinkage calibrator with every number estimated from other races taken as of the race
    (module docstring): the practice-to-race base delta (as_of_factor, its sd the base tau), the
    fuel prior (as_of_fuel, its sd the fuel tau) and the event factors sc_lap_factor and
    lap1_extra_s (drawn per sample with their sd). The wear tau and the hinge tau stay
    SimConfig.shrinkage's, engineering priors. The diagnostics name each source and its races."""
    name = "as_of_history"

    def calibrate(self, ctx):
        out = super().calibrate(ctx)
        rid = ctx.race.race_id
        factors, sources = dict(out.factors), {}
        bd = as_of_factor(rid, "base_delta_s")
        for k in list(factors):
            if split_factor(k)[0] == "base_delta_s":
                factors[k] = (bd["mean"], bd["sd"])
        fuel = as_of_fuel(ctx.race, practice_model=ctx.practice_model, practice=ctx.practice)
        factors["fuel_s_per_lap"] = (fuel["mean"], fuel["sd"])
        sources["base_delta_s"], sources["fuel_s_per_lap"] = bd, fuel
        for name in ("sc_lap_factor", "lap1_extra_s"):
            p = as_of_factor(rid, name)
            factors[name] = (p["mean"], p["sd"])
            sources[name] = p
        taus = {**out.diagnostics["taus"], "base_s": bd["sd"], "fuel_s_per_lap": fuel["sd"]}
        return CalibrationOutput(factors=factors, valid_until_lap=out.valid_until_lap, source=self.name,
                                 diagnostics={"prior": self.name, "taus": taus,
                                              "as_of": {k: {"source": v["source"], "races": v["races"]}
                                                        for k, v in sources.items()}})


# --- building the tables -------------------------------------------------------------------------
def race_row(data) -> tuple[dict, list[dict]]:
    """(history row, period rows) of one race (jobs.RaceData)."""
    from . import calibration, calibrators, fuel_prior
    rid = canonical_race_id(data.race_id)
    t = calibrators.hindsight_targets(data)
    bases = {k: v for k, v in t.items() if k.startswith("base_delta_s[")}
    target, target_sd = fuel_prior.race_target(data)
    row = {"race_id": rid, "circuit": _circuit(rid), "total_laps": int(data.race.total_laps),
           "base_delta_s": float(np.mean(list(bases.values()))) if bases else np.nan, **bases,
           "sc_lap_factor": t.get("sc_lap_factor", np.nan), "lap1_extra_s": t.get("lap1_extra_s", np.nan),
           "target": target, "target_sd": target_sd}
    wet = bool(data.frame["compound"].isin(["INTERMEDIATE", "WET"]).any())
    periods = [{"race_id": rid, "kind": k, "start_lap": a, "end_lap": b, "total_laps": int(data.race.total_laps),
                "wet": wet} for k, a, b in calibration.observed_periods(data.frame)]
    if not periods:
        periods = [{"race_id": rid, "kind": "", "start_lap": "", "end_lap": "", "total_laps": int(data.race.total_laps),
                    "wet": wet}]
    return row, periods


def build(races, history_path=HISTORY_FILE, periods_path=PERIODS_FILE):
    from . import cross_race, jobs
    from .compute import load_profile
    cross_race.guard_held_out(races)
    rows, periods = [], []
    for rid in sorted(races, key=lambda r: parse_race_id(r)[:2]):
        try:
            row, per = race_row(jobs.load_race(rid, load_profile()))
        except Exception as exc:   # noqa: BLE001  a race the loader cannot build is left out, said so
            print(f"{rid}: left out ({type(exc).__name__}: {exc})", flush=True)
            continue
        rows.append(row)
        periods += per
        print(f"{rid}: {row}", flush=True)
    h = pd.DataFrame(rows)
    h.to_csv(history_path, index=False)
    pd.DataFrame(periods).to_csv(periods_path, index=False)
    _TABLES.clear()
    return h


def main(argv=None):
    from .compound_steps import CALIBRATION_RACES
    ap = argparse.ArgumentParser(prog="race_history", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--races", nargs="+", default=list(CALIBRATION_RACES))
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(build(args.races).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
