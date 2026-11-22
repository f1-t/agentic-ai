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
                                      with the race's circuit and its median clean practice lap
                                      (practice_lap_s, pre-race: the anchored fuel prior's scale)
  configs/sim/neutralised_history.csv the race's SC and VSC periods (calibration.observed_periods),
                                      in event_rates' neutralised_periods format

and the as-of readers:

  as_of_factor(race_id, name)   base_delta_s, sc_lap_factor, lap1_extra_s: the median of the
                                earlier races, sd their spread floored at SD_FLOOR (A_PRIORI's sd
                                with one earlier race); A_PRIORI with none
  as_of_fuel(race, ...)         fuel_prior.circuit_history_prediction on the earlier races, else the
                                physics prior (fuel_prior.practice_longrun_prediction, a priori);
                                mode "anchored": fuel_prior.anchored_prediction, the lap-scaled
                                physics slope times the earlier races' ratio target / physics
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
from .calibrators import CALIBRATORS, ShrinkageCalibrator, factor_name, split_factor
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


# per-compound base delta (AsOfHistoryCalibrator param base_delta "per_compound"): the a priori
# prior strength, in races, a compound's own offset is shrunk with toward the common delta (as
# tyre_history's shrinkage.prior_strength_races; stated, not fitted)
COMPOUND_DELTA_PRIOR_RACES = 2.0
BASE_DELTA_MODES = ("common", "per_compound")
PRACTICE_ANCHOR_MODES = ("off", "pace", "pace_wear")


def as_of_compound_delta(race_id, compound, history: pd.DataFrame | None = None) -> dict:
    """{mean, sd, source, races, offset, n}: the practice-to-race base delta of `compound` for
    `race_id` from the races before it. The common delta (as_of_factor "base_delta_s") plus the
    compound's own offset from it: per earlier race with a base_delta_s[compound] value, that value
    minus the race's mean over its compounds (base_delta_s); the median over those races, shrunk
    by n / (n + COMPOUND_DELTA_PRIOR_RACES). Practice long runs on one compound come from one or
    two cars in a fit without driver or session terms, so a compound's practice pace can sit
    seconds off the others' in the same direction race after race (HARD in 2019-2022); the common
    delta cannot see that. The sd is the common delta's. No earlier race with the compound: the
    common delta alone (offset 0)."""
    h = load_history() if history is None else history
    common = as_of_factor(race_id, "base_delta_s", h)
    col = f"base_delta_s[{compound}]"
    earlier = set(earlier_races(race_id, h["race_id"]))
    rows = h[h["race_id"].isin(earlier)] if col in h and "base_delta_s" in h else h.iloc[0:0]
    rel = (pd.to_numeric(rows[col], errors="coerce") - pd.to_numeric(rows["base_delta_s"], errors="coerce")).dropna() \
        if len(rows) else pd.Series(dtype=float)
    n = len(rel)
    offset = float(rel.median()) * n / (n + COMPOUND_DELTA_PRIOR_RACES) if n else 0.0
    races = sorted(rows.loc[rel.index, "race_id"]) if n else []
    return {"mean": common["mean"] + offset, "sd": common["sd"], "offset": offset, "n": n,
            "source": f"{common['source']}; {compound} offset as of {n} earlier race(s)", "races": races}


def anchored_gap_offsets(practice_model, config, race_id, quali=None, kinds=("pace",)) -> dict:
    """{compound: {"pace": s, "wear": s/lap, "ref", "gap_practice", "gap_anchored", "sources"}}: how
    far each practice compound's fresh-pace gap (and, with "wear" in `kinds`, its wear difference)
    to the reference compound moves when the practice step is blended with the other sources the
    compound proxy uses (proxy._blend: qualifying for pace, the as-of cross-race neighbour step,
    with the proxy's as-of weights, proxy.as_of_config, races strictly before `race_id`). The
    reference is the practice compound with the most practice rows; it moves by 0. Practice long
    runs on a compound often come from one or two cars, so its practice gap carries their pace;
    the blend gives it the weight its validation error earns. {} without a practice fit, with
    fewer than two practice compounds, or with the compound proxy off."""
    from . import proxy
    if practice_model is None or not config.compound_proxy.enabled:
        return {}
    coefs = {c: v for c, v in practice_model.coefs.items() if c in proxy.SOFTNESS}
    if len(coefs) < 2:
        return {}
    n_rows = (practice_model.fit_report or {}).get("n_rows") or {}
    ref = max(coefs, key=lambda c: (n_rows.get(c, 0), -proxy.SOFTNESS.index(c)))
    px = proxy.as_of_config(config.compound_proxy, race_id)
    out = {}
    for c in coefs:
        if c == ref:
            continue
        e = {"ref": ref, "pace": 0.0, "wear": 0.0}
        for kind in kinds:
            j = 0 if kind == "pace" else 1
            srcs = list(proxy.WEIGHT_SOURCES[kind])
            b = proxy._blend(kind, c, [ref], coefs, practice_model, quali, px, {}, srcs)
            if b is None:
                continue
            e[kind] = float(b["value"] - coefs[c][j])
            e[f"{kind}_sources"] = b["source"]
        out[c] = e
    return out


# --- the practice-to-race map per compound (AsOfHistoryCalibrator param practice_map "as_of") -------
MAP_FILE = ROOT / "configs" / "sim" / "practice_map_history.csv"
PRACTICE_MAP_MODES = ("off", "as_of")
MAP_COMPOUNDS = ("HARD", "MEDIUM", "SOFT")


def load_map(path=MAP_FILE) -> pd.DataFrame:
    key = ("map", str(path))
    if key not in _TABLES:
        _TABLES[key] = pd.read_csv(path) if Path(path).is_file() else pd.DataFrame(columns=["race_id"])
    return _TABLES[key]


def as_of_practice_map(race_id, compound, history: pd.DataFrame | None = None) -> dict:
    """{wear, base, n, races}: how the belief's practice prior of `compound` moves at `race_id`,
    from the races before it in the practice-map table (one row per race: per compound the
    belief's practice fit, practice_base / practice_wear, and the judge's full-race fit,
    judge_base / judge_wear, a compound the judge prices by proxy left out).
      wear: the median over earlier races of judge_wear - practice_wear, shrunk by
            n / (n + COMPOUND_DELTA_PRIOR_RACES): added to this weekend's practice wear;
      base: the median over earlier races of the compound's practice-to-race base change less that
            race's mean change over its compounds, shrunk the same way: added to the compound's
            base delta (the common level stays the as-of base delta's).
    Practice long runs on a compound come from a handful of cars and short runs, so their wear
    slope and fresh-tyre intercept are poorly separated, and race after race in the same
    direction; the two move together here because they were measured together. No earlier race
    with the compound: 0 and 0."""
    h = load_map() if history is None else history
    need = [f"practice_wear[{compound}]", f"judge_wear[{compound}]", f"practice_base[{compound}]",
            f"judge_base[{compound}]"]
    earlier = set(earlier_races(race_id, h["race_id"]))
    if not all(c in h for c in need):
        return {"wear": 0.0, "base": 0.0, "n": 0, "races": []}
    rows = h[h["race_id"].isin(earlier)].dropna(subset=need)
    n = len(rows)
    if not n:
        return {"wear": 0.0, "base": 0.0, "n": 0, "races": []}
    shrink = n / (n + COMPOUND_DELTA_PRIOR_RACES)
    wear = float((rows[need[1]] - rows[need[0]]).median()) * shrink
    deltas = {c: rows[f"judge_base[{c}]"] - rows[f"practice_base[{c}]"] for c in MAP_COMPOUNDS
              if f"judge_base[{c}]" in rows and f"practice_base[{c}]" in rows}
    mean = pd.concat(deltas.values(), axis=1).mean(axis=1, skipna=True)
    base = float((deltas[compound] - mean).median()) * shrink
    return {"wear": wear, "base": base, "n": n, "races": sorted(map(str, rows["race_id"]))}


def practice_map_row(data, belief_config) -> dict:
    """One race's practice-map row (as_of_practice_map): the belief's practice fit (Simulator
    physics_prior under `belief_config`, this race's fuel anchoring) and the judge's full-race
    fit (the eval's default world), per compound."""
    from ..eval.score import default_world_config
    from . import config as _config, proxy
    from .engine import Simulator
    rid = canonical_race_id(data.race_id)
    belief = Simulator(data.frame, belief_config, data.practice, qualifying=getattr(data, "qualifying", None))
    raw = belief.physics_prior(None, data.race)
    world_cfg = default_world_config(getattr(_config, "world_default", lambda c: c)(data.config))
    world = Simulator(data.frame, world_cfg, data.practice, qualifying=getattr(data, "qualifying", None))
    jdeg = world.degradation
    borrowed = proxy.proxied(jdeg.fit_report)
    row = {"race_id": rid}
    n_rows = ((raw.fit_report or {}).get("n_rows") or {}) if raw is not None else {}
    for c in MAP_COMPOUNDS:
        if raw is not None and c in raw.coefs:
            row[f"practice_base[{c}]"], row[f"practice_wear[{c}]"] = map(float, raw.coefs[c])
            row[f"practice_rows[{c}]"] = int(n_rows.get(c, 0))
        if c in jdeg.coefs and c not in borrowed:
            row[f"judge_base[{c}]"], row[f"judge_wear[{c}]"] = map(float, jdeg.coefs[c])
    return row


def build_map(races, path=MAP_FILE, belief_file=None):
    """The practice-map table over `races` (held-out races refused before anything is loaded)."""
    import yaml
    from . import cross_race, jobs
    from .compute import load_profile
    from .config import SimConfig
    cross_race.guard_held_out(races)
    belief_file = belief_file or ROOT / "configs" / "sim" / "belief.yaml"
    cfg = SimConfig.model_validate(yaml.safe_load(Path(belief_file).read_text()))
    rows = []
    for rid in sorted(races, key=lambda r: parse_race_id(r)[:2]):
        try:
            rows.append(practice_map_row(jobs.load_race(rid, load_profile()), cfg))
        except Exception as exc:   # noqa: BLE001  a race the loader cannot build is left out, said so
            print(f"{rid}: left out ({type(exc).__name__}: {exc})", flush=True)
            continue
        print(f"{rid}: {rows[-1]}", flush=True)
    h = pd.DataFrame(rows)
    h.to_csv(path, index=False)
    _TABLES.clear()
    return h


FUEL_MODES = ("circuit_history", "anchored")


def as_of_fuel(race, history: pd.DataFrame | None = None, practice_model=None, practice=None,
               mode: str = "circuit_history") -> dict:
    """{mean, sd, source, races}: the fuel prior for `race` (RaceParams).

    mode "circuit_history" (#102): the circuit history of the earlier races
    (fuel_prior.circuit_history_prediction), else the physics prior. mode "anchored": the
    lap-scaled physics slope times the ratio target / physics of the earlier races
    (fuel_prior.anchored_prediction), this weekend's practice lap the scale; the a priori physics
    prediction with no earlier race."""
    from . import fuel_prior
    if mode not in FUEL_MODES:
        raise ValueError(f"fuel mode must be one of {FUEL_MODES}, got {mode!r}")
    h = load_history() if history is None else history
    earlier = set(earlier_races(race.race_id, h["race_id"]))
    train = h[h["race_id"].isin(earlier) & h["target"].notna()] if "target" in h else h.iloc[0:0]
    if mode == "anchored":
        p = fuel_prior.anchored_prediction(race.total_laps, fuel_prior.practice_lap_s(practice), train)
        races = sorted(train["race_id"]) if p["n_history"] else []
        return {"mean": p["mean"], "sd": p["sd"], "source": p["source"], "races": races}
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


def pooled_event_rates(periods: pd.DataFrame | None = None) -> dict:
    """as_of_event_rates over every dry race the table records (the judge's later periods under
    RateEvents pooled: a world that may read the calibration set)."""
    from . import event_rates
    p = load_periods() if periods is None else periods
    key = ("pooled", len(p), tuple(sorted(set(map(str, p["race_id"])))) if len(p) else ())
    if key not in _RATES:
        stats = event_rates.dry(event_rates.stats_from_periods(p)) if len(p) else pd.DataFrame(columns=event_rates.STAT_COLUMNS)
        fit = event_rates.fit(stats, prior=EVENT_PRIOR)
        row = {"race_id": "pooled", "era": "", "circuit": event_rates.UNKNOWN, "circuit_type": event_rates.UNKNOWN}
        r = fit.rates(row, EVENT_LEVEL)
        _RATES[key] = {"p_sc": r["p_sc"], "d_sc": r["d_sc"], "p_vsc": r["p_vsc"], "d_vsc": r["d_vsc"],
                       "source": f"pooled over {len(stats)} dry race(s), global level", "races": sorted(stats["race_id"])}
    return _RATES[key]


# Period lengths (events.RateEvents duration "history"): the lengths of the neutralised periods of
# the dry races, as of the race (earlier races only) or pooled over the table, plus an a priori
# prior worth LENGTH_PRIOR_PERIODS periods: a geometric length with EVENT_PRIOR's mean, on 1..MAX_PERIOD_LAPS.
MAX_PERIOD_LAPS = 12
LENGTH_PRIOR_PERIODS = 2.0
_LENGTHS: dict = {}


def _prior_pmf(mean: float) -> np.ndarray:
    p = 1.0 / mean
    k = np.arange(1, MAX_PERIOD_LAPS + 1)
    pmf = p * (1 - p) ** (k - 1)
    return pmf / pmf.sum()


def period_lengths(race_id, kind: str, scope: str = "as_of", periods: pd.DataFrame | None = None) -> dict:
    """{pmf (lengths 1..MAX_PERIOD_LAPS), n, source, races}: the length distribution of `kind`
    ("sc" / "vsc") periods, from the dry races before `race_id` (scope "as_of", the belief's) or
    from every dry race in the table (scope "pooled", the judge's), shrunk toward the a priori
    geometric of EVENT_PRIOR (worth LENGTH_PRIOR_PERIODS periods)."""
    p = load_periods() if periods is None else periods
    if scope == "as_of":
        races = tuple(earlier_races(race_id, sorted(set(p["race_id"])))) if len(p) else ()
    else:
        races = tuple(sorted(set(map(str, p["race_id"])))) if len(p) else ()
    key = (kind, scope, len(p), races)
    if key not in _LENGTHS:
        rows = p[p["race_id"].isin(races) & (p["kind"] == kind)] if len(p) else p
        if len(rows) and "wet" in rows:
            rows = rows[~rows["wet"].astype(str).str.lower().isin(["true", "1"])]
        L = (rows["end_lap"].astype(float) - rows["start_lap"].astype(float) + 1).to_numpy() if len(rows) else np.array([])
        L = np.clip(L[np.isfinite(L)].astype(int), 1, MAX_PERIOD_LAPS)
        counts = np.bincount(L - 1, minlength=MAX_PERIOD_LAPS).astype(float)[:MAX_PERIOD_LAPS]
        prior = _prior_pmf(EVENT_PRIOR.d_sc if kind == "sc" else EVENT_PRIOR.d_vsc)
        pmf = counts + LENGTH_PRIOR_PERIODS * prior
        _LENGTHS[key] = {"pmf": pmf / pmf.sum(), "n": int(len(L)),
                         "source": f"{len(L)} {kind} period(s) of {'earlier' if scope == 'as_of' else 'all'} dry races "
                                   f"+ a priori geometric (mean {EVENT_PRIOR.d_sc if kind == 'sc' else EVENT_PRIOR.d_vsc})",
                         "races": sorted(set(map(str, rows["race_id"]))) if len(rows) else []}
    return _LENGTHS[key]


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
    SimConfig.shrinkage's, engineering priors. The diagnostics name each source and its races.
    The fuel prior's mode (as_of_fuel) is SimConfig.fuel_anchor.prior, "circuit_history" (the
    default, #102) or "anchored"; a param `fuel` overrides it. The param `base_delta`
    "common" (the default) gives every practice compound the one as-of delta; "per_compound"
    gives each its own (as_of_compound_delta: the common delta plus the compound's shrunk as-of
    offset from it)."""
    name = "as_of_history"

    def __init__(self, **params):
        super().__init__(**params)
        if self.params.get("base_delta", "common") not in BASE_DELTA_MODES:
            raise ValueError(f"as_of_history base_delta must be one of {BASE_DELTA_MODES}, got {params['base_delta']!r}")
        if self.params.get("practice_map", "off") not in PRACTICE_MAP_MODES:
            raise ValueError(f"as_of_history practice_map must be one of {PRACTICE_MAP_MODES}, got {params['practice_map']!r}")
        if self.params.get("practice_anchor", "off") not in PRACTICE_ANCHOR_MODES:
            raise ValueError(f"as_of_history practice_anchor must be one of {PRACTICE_ANCHOR_MODES}, "
                             f"got {params['practice_anchor']!r}")

    def calibrate(self, ctx):
        out = super().calibrate(ctx)
        rid = ctx.race.race_id
        factors, sources = dict(out.factors), {}
        bd = as_of_factor(rid, "base_delta_s")
        mode = self.params.get("base_delta", "common")
        anchor = self.params.get("practice_anchor", "off")
        moved = (anchored_gap_offsets(ctx.practice_model, ctx.config, rid, ctx.qualifying_gaps,
                                      ("pace",) if anchor == "pace" else ("pace", "wear"))
                 if anchor != "off" else {})
        for k in list(factors):
            name, comp = split_factor(k)
            if name == "base_delta_s":
                if mode == "per_compound" and comp is not None:
                    got = as_of_compound_delta(rid, comp)
                    factors[k] = (got["mean"], got["sd"])
                    sources[k] = got
                else:
                    factors[k] = (bd["mean"], bd["sd"])
                if comp in moved:
                    factors[k] = (factors[k][0] + moved[comp]["pace"], factors[k][1])
        if anchor == "pace_wear" and ctx.practice_model is not None:
            for comp, e in moved.items():
                w = float(ctx.practice_model.coefs[comp][1]) + e["wear"]
                floor = ctx.config.degradation.params.get("wear_lower_bound")
                key = factor_name("wear_s_per_lap", comp)
                sd = factors.get(key, (None, 0.0))[1]
                factors[key] = (max(w, float(floor)) if floor is not None else w, sd)
        if self.params.get("practice_map", "off") == "as_of" and ctx.practice_model is not None:
            floor = ctx.config.degradation.params.get("wear_lower_bound")
            for comp, (_, w0) in ctx.practice_model.coefs.items():
                if comp not in MAP_COMPOUNDS:
                    continue
                got = as_of_practice_map(rid, comp)
                if not got["n"]:
                    continue
                kb, kw = factor_name("base_delta_s", comp), factor_name("wear_s_per_lap", comp)
                if kb in factors:
                    factors[kb] = (factors[kb][0] + got["base"], factors[kb][1])
                w = float(w0) + got["wear"]
                factors[kw] = (max(w, float(floor)) if floor is not None else w, factors.get(kw, (None, 0.0))[1])
                sources[f"practice_map[{comp}]"] = {"source": f"as of {got['n']} earlier race(s)", "races": got["races"]}
        if moved:
            sources["practice_anchor"] = {"source": f"proxy blend ({anchor})", "races": [],
                                          **{c: {kk: v for kk, v in e.items()} for c, e in moved.items()}}
        fuel = as_of_fuel(ctx.race, practice_model=ctx.practice_model, practice=ctx.practice,
                          mode=self.params.get("fuel", ctx.config.fuel_anchor.prior))
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
           "target": target, "target_sd": target_sd, "practice_lap_s": fuel_prior.practice_lap_s(data.practice)}
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
    m = sub.add_parser("build-map", help="the practice-map table (as_of_practice_map)")
    m.add_argument("--races", nargs="+", default=list(CALIBRATION_RACES))
    m.add_argument("--belief", default=None, help="the belief config whose practice fit is recorded (default belief.yaml)")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(build(args.races).to_string(index=False))
    elif args.cmd == "build-map":
        print(build_map(args.races, belief_file=args.belief).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
