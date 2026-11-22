"""Pit loss without hindsight, and the reduced pit loss under SC and VSC (docs/eval.md
"Pit loss and track status").

The world (the scorer) prices a green stop at the race's whole-race pit loss (jobs.load_race,
calibrate._pit_loss: the in- plus out-lap residual of every finisher's green stop against a
full-window fit). A policy may not know that number at lap L: it is measured from stops that
have not happened yet. A belief with SimConfig.pit_loss.source "asof" instead uses `estimate`:

  prior    the fallback chain (below, "priors"): earlier seasons at the same circuit, else
           earlier races at any circuit with a wider sd, else a documented constant; never the
           race itself nor anything after it.
  update   with this race's green stops seen by the snapshot: in- and out-lap both green and
           the out-lap ended by the time the snapshot driver ended lap L (session time, not lap
           number). Each stop's loss is its in- plus out-lap time less the belief fit's green
           lap times for those laps (the whole-race definition, per stop). A stop outside
           plausible x the prior mean (a repair, a penalty, a stop on an unflagged slow lap) is
           left out. Normal-normal: per-stop sd max(MAD, stop_sd_floor_s) over the stops seen.
  interval the posterior sd plus model_sd_s in quadrature (the per-stop estimand differs from
           the whole-race one, which also counts green out-laps of stops made under SC).

The SC and VSC pit-loss factors (the share of a green stop paid when the in-lap is under SC or
VSC) follow the same pattern: a prior (factor_prior: the world's a priori factor, or one pooled
from the table's stops of that kind), worth factor_prior_stops stops, updated with this race's
stops of that kind seen by the snapshot. A stop's factor is its in- plus out-lap time less the
reference lap times, over the green pit loss; the reference for a neutralised lap is the field's median lap time on it
(non-pit laps of that status ended by then), for a green lap the fit's.

`build_prior` writes the table from whole races (the calibration races only, explicit ids);
`asof_error` reports the as-of estimate against the whole-race value by lap.

    python -m src.sim.pit_loss build-prior --races 2021_04_spain 2023_07_spain ...
    python -m src.sim.pit_loss report --races 2021_04_spain ... [--out reports/calibration/pit_loss]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from .data import visible_rows
from .race_ids import as_of, canonical_race_id, parse_race_id

PRIOR_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim", "pit_loss_prior.json")
# the run-time pool's entries (race_entry) of the cached races outside PRIOR_FILE, precomputed so a
# process prices a prior without loading those races (build-entries); outside SimConfig, so it
# changes no config_hash, and it never holds a held-out race
ENTRIES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim", "pit_loss_entries.json")
GREEN_STATUS = ("1", "12", "21")   # calibration._by_lap_type's green in/out laps
MIN_PRIOR_RACES = 2
KINDS = ("sc", "vsc")


def load_prior(path=PRIOR_FILE) -> dict:
    """{"races": {race_id: {...}}} from the committed table, {} when absent."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def circuit_key(race_id: str) -> str:
    """The circuit a prior is pooled by: the race id's gp slug. A race id does not name a
    circuit in general (race_ids), but within the table every slug is one circuit (Spain is
    Catalunya in 2021 and 2023); a country with two circuits in a season gets event slugs."""
    return parse_race_id(race_id)[2]


def _status_kind(status) -> str:
    s = str(status)
    return "sc" if "4" in s else "vsc" if ("6" in s or "7" in s) else "green" if s in GREEN_STATUS else "other"


def stop_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per stop (in-lap with pit_in, next lap with pit_out): driver, in_lap, kind (the
    in-lap's status: sc, vsc; green when the in- and out-lap are both green; else other),
    known_s (session time the out-lap ended), and the two laps' compound, age, time, status."""
    cols = ["driver", "in_lap", "kind", "known_s", "c_in", "age_in", "t_in", "s_in", "c_out", "age_out",
            "t_out", "s_out", "team", "lane_s"]
    rows = []
    f = frame.sort_values(["driver", "lap"])
    for d, g in f.groupby("driver", sort=True):
        g = g.set_index("lap")
        for lap in g.index[g["pit_in"].astype(bool)]:
            if lap + 1 not in g.index or not bool(g.loc[lap + 1, "pit_out"]):
                continue
            a, b = g.loc[lap], g.loc[lap + 1]
            k_in, k_out = _status_kind(a["track_status"]), _status_kind(b["track_status"])
            kind = k_in if k_in in KINDS else ("green" if k_in == "green" and k_out == "green" else "other")
            # pit-lane time, entry to exit (data.lap_frame's pit_in_time_s / pit_out_time_s), NaN without them
            lane = (float(b["pit_out_time_s"]) - float(a["pit_in_time_s"])
                    if "pit_in_time_s" in g and pd.notna(a["pit_in_time_s"]) and pd.notna(b["pit_out_time_s"])
                    else float("nan"))
            rows.append([d, int(lap), kind, float(b["end_s"]), a["compound"], a["tyre_age"], a["lap_time_s"],
                         k_in, b["compound"], b["tyre_age"], b["lap_time_s"], k_out,
                         a["team"] if "team" in g else None, lane])
    return pd.DataFrame(rows, columns=cols)


# --- actual stop durations (SimConfig.pit_stops, docs/eval.md "Pit-stop durations") ------------
def lane_deltas(stops: pd.DataFrame) -> pd.Series:
    """Per stop (stop_table rows): its pit-lane time less the median lane time of the table's green
    stops. The pit loss is a whole-race median; a stop's own duration differs from it by its lane
    time's difference (stationary time, a slow wheel, a double stack): what a replay of the real
    strategy should charge on top. NaN where the lane time is missing."""
    if not len(stops) or "lane_s" not in stops:
        return pd.Series(np.nan, index=stops.index, dtype=float)
    green = stops.loc[stops["kind"] == "green", "lane_s"].dropna()
    med = float(green.median()) if len(green) else float(stops["lane_s"].median())
    return (stops["lane_s"] - med).astype(float)


def team_delta(stops: pd.DataFrame, team, prior_stops: float) -> tuple[float, int]:
    """(delta, n): the team's typical stop against the field's, the median lane-time difference of its
    green stops in `stops` shrunk toward 0 by n / (n + prior_stops); (0, 0) with none."""
    if team is None or not len(stops) or "team" not in stops:
        return 0.0, 0
    dev = lane_deltas(stops)
    sel = (stops["team"] == team) & (stops["kind"] == "green") & dev.notna()
    n = int(sel.sum())
    if n == 0:
        return 0.0, 0
    return float(n / (n + prior_stops) * dev[sel].median()), n


def _field_medians(frame: pd.DataFrame, upto_s: float | None = None) -> dict:
    """{(lap, kind): median lap time} of the field's non-pit laps under SC or VSC, laps ended by
    `upto_s` (None: all)."""
    f = frame[~frame["pit_in"].astype(bool) & ~frame["pit_out"].astype(bool) & frame["lap_time_s"].notna()]
    if upto_s is not None:
        f = f[f["end_s"] <= upto_s]
    kind = f["track_status"].map(_status_kind)
    f = f[kind.isin(KINDS)].assign(kind=kind[kind.isin(KINDS)])
    return {(int(lap), k): float(v) for (lap, k), v in f.groupby(["lap", "kind"])["lap_time_s"].median().items()}


def _green_pred(model, compound, age, lap) -> float:
    if model is None or compound not in model.compounds or age is None or pd.isna(age):
        return float("nan")
    return float(model.lap_times(compound, np.array([float(age)]), np.array([float(lap)]))[0])


def _ref(model, medians, compound, age, lap, status_kind) -> float:
    if status_kind in KINDS:
        return medians.get((int(lap), status_kind), float("nan"))
    return _green_pred(model, compound, age, lap)


def stop_losses(stops: pd.DataFrame, model, medians: dict) -> pd.Series:
    """Per stop: in- plus out-lap time less the reference lap times (_ref), NaN where either is missing."""
    out = []
    for r in stops.itertuples():
        ref = _ref(model, medians, r.c_in, r.age_in, r.in_lap, r.s_in) + \
            _ref(model, medians, r.c_out, r.age_out, r.in_lap + 1, r.s_out)
        out.append(float(r.t_in) + float(r.t_out) - ref if pd.notna(r.t_in) and pd.notna(r.t_out) else float("nan"))
    return pd.Series(out, index=stops.index, dtype=float)


# --- priors --------------------------------------------------------------------------------
# The fallback chain (pit_loss_prior), every tier as of the race's date, never the race itself:
#   1 same_circuit_asof   earlier seasons at the same circuit (sd same_circuit_sd_s)
#   2 cross_circuit_asof  at least MIN_PRIOR_RACES earlier races at any circuit (sd their spread,
#                         at least min_cross_sd_s: wider than tier 1)
#   3 constant            default_s with default_sd_s (the widest), when nothing earlier exists
# The pool is the committed table (configs/sim/pit_loss_prior.json, precomputed) plus, under
# cache_lookup, every race in the FastF1 cache at run time (real_run.cached_races). A cached
# race's entry is read from ENTRIES_FILE (race_entry, precomputed by build-entries), else built on
# first use from its whole race (race_entry) and kept per process: more cached seasons widen the
# pool with no code change, and the committed entries keep a process from loading them. A held-out race (the registered split's file) enters the
# pool only when the race being priced is itself held out (the held-out run), so a development
# run never loads one; it is still used only when earlier in time.
_ENTRIES: dict = {}     # (race id, the loader) -> run-time entry (race_entry), per process
_CACHED: dict = {}      # cache dir -> cached race ids
_COMMITTED: dict = {}   # (ENTRIES_FILE, mtime, size) -> its races


def committed_entries(path=ENTRIES_FILE) -> dict:
    """ENTRIES_FILE's races ({race id: entry, or {"left_out": reason} for a race the run-time
    build leaves out}), read once per process; {} without the file."""
    if not os.path.isfile(path):
        return {}
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _COMMITTED:
        with open(path) as f:
            _COMMITTED[key] = json.load(f).get("races", {})
    return _COMMITTED[key]


def _real_loader(fn) -> bool:
    """jobs.load_race itself, not a test's replacement: only then do committed entries stand in for it."""
    return getattr(fn, "__module__", None) == "src.sim.jobs" and getattr(fn, "__qualname__", None) == "load_race"


def _cached_ids() -> list[str]:
    from .data import CACHE_DIR
    key = os.path.abspath(CACHE_DIR)
    if key not in _CACHED:
        from .real_run import cached_races
        _CACHED[key] = cached_races()[0]
    return _CACHED[key]


def _held_out() -> set:
    from .cross_race import held_out_races
    return set(held_out_races())


def prior_pool(race_id: str, cfg, cached=None, held=None) -> list[str]:
    """The race ids the prior for `race_id` may draw on, before the as-of cut: the table's, plus
    the cached ones under cfg.cache_lookup (`cached` overrides the discovery), less the race
    itself, less every held-out race (`held`, default the registered split) unless `race_id`
    is one."""
    rid = canonical_race_id(race_id)
    ids = set((cfg.prior or {}).get("races", {}))
    if cfg.cache_lookup:
        ids |= set(_cached_ids() if cached is None else cached)
    held = _held_out() if held is None else set(held)
    if rid not in held:
        ids -= held
    ids.discard(rid)
    return sorted(ids)


def _entry(race_id: str, cfg) -> dict | None:
    """The race's prior entry: the table's, else built from the cache once per process (None
    when it cannot be loaded)."""
    table = (cfg.prior or {}).get("races", {})
    if race_id in table:
        return table[race_id]
    from . import jobs
    key = (race_id, jobs.load_race)   # a replaced loader (a test's) never shares entries with the real one
    committed = committed_entries() if _real_loader(jobs.load_race) else {}
    if key not in _ENTRIES and race_id in committed:
        got = committed[race_id]
        if "left_out" in got:
            import warnings
            warnings.warn(f"pit-loss prior: {race_id} left out ({got['left_out']})")
            got = None
        _ENTRIES[key] = got
    if key not in _ENTRIES:
        try:
            from .config import SimConfig
            from .engine import Simulator
            data = _load(race_id)
            if canonical_race_id(data.race_id) != race_id:   # a loader that serves another race (a test's)
                raise ValueError(f"the loader returned {data.race_id}")
            full = Simulator(data.frame, SimConfig(), data.practice, qualifying=data.qualifying)
            _ENTRIES[key] = race_entry(data, full.degradation)
        except Exception as exc:   # an unloadable cached race is left out of the pool, never fatal
            import warnings
            warnings.warn(f"pit-loss prior: {race_id} left out ({exc})")
            _ENTRIES[key] = None
    return _ENTRIES[key]


def _usable(entry) -> bool:
    """A pool entry the prior may use: loaded, with a pit loss, and a dry race (a wet race's
    stops mix tyre changes for the weather with slow laps; 2022 Imola's median green stop is 38 s)."""
    return bool(entry) and entry.get("pit_loss_s") is not None and not entry.get("wet", False)


def pit_loss_prior(race_id: str, cfg, cached=None, held=None) -> dict:
    """{mean_s, sd_s, source, races}: the green pit loss prior for `race_id`, the first tier of
    the fallback chain above that has data. Never fails: a race id with no time order (a
    synthetic one) gets the constant."""
    rid = canonical_race_id(race_id)
    try:
        earlier = as_of(rid, prior_pool(rid, cfg, cached, held))
        here = circuit_key(rid)
    except ValueError:   # a synthetic or unparseable id: no time order, no circuit
        earlier, here = [], None
    same = [r for r in earlier if circuit_key(r) == here]
    vals = {r: e["pit_loss_s"] for r in same if _usable(e := _entry(r, cfg))}
    if vals:
        return {"mean_s": float(np.mean(list(vals.values()))), "sd_s": float(cfg.same_circuit_sd_s),
                "source": "same_circuit_asof", "races": sorted(vals)}
    vals = {r: e["pit_loss_s"] for r in earlier if _usable(e := _entry(r, cfg))}
    if len(vals) >= MIN_PRIOR_RACES:
        v = np.array(list(vals.values()), dtype=float)
        return {"mean_s": float(v.mean()), "sd_s": max(float(v.std(ddof=1)), float(cfg.min_cross_sd_s)),
                "source": "cross_circuit_asof", "races": sorted(vals)}
    return {"mean_s": float(cfg.default_s), "sd_s": float(cfg.default_sd_s), "source": "constant", "races": []}


def factor_prior(race_id: str, kind: str, cfg) -> dict:
    """{mean, sd, n, source, races}: the `kind` pit-loss factor prior for `race_id`, defined for
    every race. Under factor_prior "config" (the default) the a priori factor (default_sc_factor,
    default_vsc_factor, the world's), sd factor_sd, source "constant". Under "pooled" the mean
    of the stops of that kind in the earlier races of the pool (prior_pool, as-of) when they
    hold at least min_factor_stops, else the constant. The pooled factor lost to the a priori
    one out of sample on the calibration races (reports/calibration/neutralised), so it is off."""
    default = {"sc": cfg.default_sc_factor, "vsc": cfg.default_vsc_factor}[kind]
    constant = {"mean": float(default), "sd": float(cfg.factor_sd), "n": 0, "source": "constant", "races": []}
    if cfg.factor_prior == "config":
        return constant
    try:
        earlier = as_of(canonical_race_id(race_id), prior_pool(race_id, cfg))
    except ValueError:
        return constant
    got = {r: (e.get(kind) or {}).get("factors") or [] for r in earlier if (e := _entry(r, cfg))}
    x = np.array([f for v in got.values() for f in v], dtype=float)
    if len(x) < cfg.min_factor_stops:
        return constant
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n": int(len(x)), "source": "pooled_asof",
            "races": sorted(r for r, v in got.items() if v)}


# --- as-of estimate --------------------------------------------------------------------------
def snapshot_time(frame: pd.DataFrame, driver: str, lap: int) -> float | None:
    """Session time at which `driver` ended lap `lap`; at lap 0 the race start (their lap-1
    start); None when the frame does not have it."""
    g = frame[frame["driver"] == driver]
    if lap == 0:
        row = g[g["lap"] == 1]
        return float(row["start_s"].iloc[0]) if len(row) and pd.notna(row["start_s"].iloc[0]) else None
    row = g[g["lap"] == lap]
    return float(row["end_s"].iloc[0]) if len(row) and pd.notna(row["end_s"].iloc[0]) else None


def _normal_update(m0, s0, x, floor, center=np.mean):
    """Posterior (mean, sd) of a normal location under N(m0, s0^2) given observations x summarised
    by `center` (the median for green stops), with a per-observation sd max(MAD, floor); the
    median's larger variance (pi / 2 of the mean's) is not added, MAD already widens it."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float(m0), float(s0)
    mad = 1.4826 * float(np.median(np.abs(x - np.median(x)))) if len(x) >= 3 else 0.0
    se2 = max(mad, floor) ** 2 / len(x)
    prec = 1.0 / s0 ** 2 + 1.0 / se2
    return float((m0 / s0 ** 2 + float(center(x)) / se2) / prec), float(math.sqrt(1.0 / prec))


def estimate(frame: pd.DataFrame, race_id: str, driver: str, lap: int, model, cfg, stops=None) -> dict:
    """The policy-side pit loss at (race, driver, lap L): green pit loss and the SC and VSC
    factors, each with its prior, the stops seen and an sd (module doc). Reads only rows of
    `frame` ended by the time `driver` ended lap L (by lap number when the frame has no time
    for it), and `model` (the belief's fit at the snapshot)."""
    stops = stop_table(frame) if stops is None else stops
    t = snapshot_time(frame, driver, lap)
    seen = stops[stops["known_s"] <= t] if t is not None else stops[stops["in_lap"] + 1 <= lap]
    rows = visible_rows(frame, t) if t is not None else frame[frame["lap"] <= lap]   # data.visible_rows: the shared cut
    medians = _field_medians(rows)
    prior = pit_loss_prior(race_id, cfg)
    lo, hi = cfg.plausible
    green = seen[seen["kind"] == "green"]
    loss = stop_losses(green, model, medians).dropna()
    used = loss[(loss >= lo * prior["mean_s"]) & (loss <= hi * prior["mean_s"])]
    mean, sd = _normal_update(prior["mean_s"], prior["sd_s"], used.to_numpy(), cfg.stop_sd_floor_s, np.median)
    sd_total = math.sqrt(sd ** 2 + cfg.model_sd_s ** 2)
    out = {"pit_loss_s": mean, "sd_s": sd_total, "n_green_stops": int(len(used)),
           "n_green_stops_dropped": int(len(loss) - len(used)), "prior_s": prior["mean_s"],
           "prior_sd_s": prior["sd_s"], "prior_source": prior["source"], "prior_races": prior["races"]}
    for kind in KINDS:
        fp = factor_prior(race_id, kind, cfg)
        k = seen[seen["kind"] == kind]
        x = (stop_losses(k, model, medians) / mean).dropna().clip(0.0, 1.5) if len(k) else pd.Series(dtype=float)
        # a factor prior worth factor_prior_stops stops: its sd over sqrt of that
        s0 = max(fp["sd"], 1e-6) / math.sqrt(cfg.factor_prior_stops)
        fm, fsd = _normal_update(fp["mean"], s0, x.to_numpy(), cfg.factor_sd_floor)
        out[f"{kind}_factor"] = float(min(max(fm, 0.0), 1.0))
        out[f"{kind}_factor_sd"] = fsd
        out[f"{kind}_factor_prior"] = fp["mean"]
        out[f"{kind}_factor_prior_source"] = fp["source"]
        out[f"n_{kind}_stops"] = int(len(x))
    return out


# --- the world's whole-race value -----------------------------------------------------------
def whole_race(frame: pd.DataFrame, model, race_id: str, cfg) -> dict:
    """The world's pit loss for a race (the scorer may use every stop): the median of every green
    stop's loss against the full-window `model` (the per-stop estimand of `estimate`, so policy
    and world measure the same thing), when the race has at least cfg.min_green_stops of them;
    else the prior (pit_loss_prior). Robust to slow stops and to a race whose stops were mostly
    neutralised, where the replay mean (calibrate._pit_loss) is not (2022 Saudi Arabia: 15.05 s
    from 5 green stops, against a median of 24.4 s). Stops made under SC or VSC are not pooled
    in: divided by the a priori factor, Saudi's (most made on the deployment lap) give 31 s."""
    stops = stop_table(frame)
    green = stop_losses(stops[stops["kind"] == "green"], model, _field_medians(frame)).dropna()
    if len(green) >= cfg.min_green_stops:
        return {"pit_loss_s": float(green.median()), "source": "green_median", "n_green_stops": int(len(green))}
    return {"pit_loss_s": pit_loss_prior(race_id, cfg)["mean_s"], "source": "prior", "n_green_stops": int(len(green))}


# --- the table and the report -----------------------------------------------------------------
def race_entry(data, model, cfg=None) -> dict:
    """One race's prior-table entry from its whole race: the world's whole-race pit loss
    (whole_race on the full-window `model`; the replay mean beside it) and each SC / VSC stop's
    factor against it, the reference being `model` and the field medians."""
    from .config import PitLossModel
    cfg = cfg or PitLossModel()
    f = data.frame
    stops = stop_table(f)
    medians = _field_medians(f)
    world = whole_race(f, model, data.race_id, cfg)
    pl = float(world["pit_loss_s"])
    out = {"pit_loss_s": round(pl, 3), "source": world["source"], "replay_mean_s": float(data.race.pit_loss_s),
           "circuit": circuit_key(data.race_id), "total_laps": int(data.race.total_laps),
           "n_green_stops": world["n_green_stops"],
           "wet": bool(f["compound"].astype(str).isin(["INTERMEDIATE", "WET"]).any())}
    for kind in KINDS:
        k = stops[stops["kind"] == kind]
        x = (stop_losses(k, model, medians) / pl).dropna() if len(k) else pd.Series(dtype=float)
        out[kind] = {"factors": [round(float(v), 4) for v in x], "stops": [f"{d}:{lap}" for d, lap in
                                                                          zip(k.loc[x.index, "driver"], k.loc[x.index, "in_lap"])]}
    return out


def _load(race_id):
    from .compute import load_profile
    from .jobs import load_race
    return load_race(race_id, load_profile())


def build_prior(race_ids, path=PRIOR_FILE) -> dict:
    from .config import SimConfig
    from .engine import Simulator
    races = {}
    for rid in race_ids:
        data = _load(rid)
        full = Simulator(data.frame, SimConfig(), data.practice, qualifying=data.qualifying)
        races[canonical_race_id(rid)] = race_entry(data, full.degradation)
    table = {"note": ("Whole-race pit loss (jobs.load_race) and the factor of every stop whose in-lap was under SC or "
                      "VSC, per calibration race. Built by `python -m src.sim.pit_loss build-prior`; read by "
                      "SimConfig.pit_loss.prior, so config_hash covers it. Never holds a held-out race."),
             "races": races}
    with open(path, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    return table


def build_entries(path=ENTRIES_FILE, cached=None, held=None) -> dict:
    """ENTRIES_FILE: race_entry of every cached race (real_run.cached_races) outside PRIOR_FILE and
    the held-out split, built as _entry builds it at run time (a race that fails to build is
    recorded with the reason, as _entry leaves it out). Held-out races are never loaded."""
    from .config import SimConfig
    from .engine import Simulator
    table = load_prior()
    held = _held_out() if held is None else set(held)
    ids = sorted(set(_cached_ids() if cached is None else cached) - set(table.get("races", {})) - held)
    races = {}
    for rid in ids:
        try:
            data = _load(rid)
            full = Simulator(data.frame, SimConfig(), data.practice, qualifying=data.qualifying)
            races[rid] = race_entry(data, full.degradation)
        except Exception as exc:   # noqa: BLE001  as _entry: left out, with the reason
            races[rid] = {"left_out": str(exc)}
    doc = {"note": ("race_entry of each cached race outside pit_loss_prior.json and the held-out split, as "
                    "pit_loss._entry would build it from the cache at run time. Built by `python -m "
                    "src.sim.pit_loss build-entries`; rebuild it when the cache, race_entry or the default "
                    "SimConfig's fit changes. Never holds a held-out race."),
           "races": races}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    return doc


def asof_error(race_ids, laps_every=5, belief_config=None) -> pd.DataFrame:
    """Per race and lap L (every laps_every, through total_laps - 5): the as-of estimate for
    the driver at the median position after lap L against the whole-race value."""
    from .calibration import snapshot_at
    from .config import SimConfig
    from .engine import Simulator
    from ..eval.score import BELIEF_CONFIG
    cfg = belief_config or SimConfig.from_file(BELIEF_CONFIG)
    rows = []
    for rid in race_ids:
        data = _load(rid)
        sim = Simulator(data.frame, cfg, data.practice, qualifying=data.qualifying)
        f = data.frame
        full = Simulator(f, SimConfig(), data.practice, qualifying=data.qualifying)
        truth = whole_race(f, full.degradation, data.race_id, cfg.pit_loss)["pit_loss_s"]
        stops = stop_table(f)
        for lap in range(0, data.race.total_laps - 4, laps_every):
            at = f[f["lap"] == max(lap, 1)].sort_values("position")
            if at.empty:
                continue
            driver = str(at["driver"].iloc[len(at) // 2])
            snap = snapshot_at(f, data.race, driver, lap)
            est = estimate(f, data.race_id, driver, lap, sim.fitted_for(snap).degradation, cfg.pit_loss, stops)
            rows.append({"race_id": data.race_id, "lap": lap, "driver": driver, "whole_race_s": truth,
                         "replay_mean_s": float(data.race.pit_loss_s),
                         "asof_s": est["pit_loss_s"], "sd_s": est["sd_s"], "err_s": est["pit_loss_s"] - truth,
                         "covered_90": abs(est["pit_loss_s"] - truth) <= 1.645 * est["sd_s"],
                         "prior_s": est["prior_s"], "prior_source": est["prior_source"],
                         "n_green_stops": est["n_green_stops"], "sc_factor": est["sc_factor"],
                         "vsc_factor": est["vsc_factor"], "n_sc_stops": est["n_sc_stops"],
                         "n_vsc_stops": est["n_vsc_stops"]})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.pit_loss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-prior")
    b.add_argument("--races", nargs="+", required=True)
    b.add_argument("--out", default=PRIOR_FILE)
    e = sub.add_parser("build-entries")
    e.add_argument("--out", default=ENTRIES_FILE)
    r = sub.add_parser("report")
    r.add_argument("--races", nargs="+", required=True)
    r.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "reports", "calibration",
                                                 "pit_loss"))
    args = ap.parse_args(argv)
    if args.cmd == "build-prior":
        t = build_prior(args.races, args.out)
        print(json.dumps({k: {"pit_loss_s": v["pit_loss_s"], "sc": len(v["sc"]["factors"]),
                              "vsc": len(v["vsc"]["factors"])} for k, v in t["races"].items()}, indent=1))
        return 0
    if args.cmd == "build-entries":
        t = build_entries(args.out)
        print(json.dumps({k: v.get("pit_loss_s", v.get("left_out")) for k, v in t["races"].items()}, indent=1))
        return 0
    df = asof_error(args.races)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "asof_error.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
