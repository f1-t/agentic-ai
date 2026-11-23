"""The historical tyre-life prior: where a compound's tyre life runs out at a race, from earlier races.

    python -m src.sim.tyre_history build    [--races ...] [--allow-held-out]
    python -m src.sim.tyre_history validate [--out-dir reports/calibration/tyre_history]

tyre_life_prior(race_id, role) estimates the tyre age at which the compound in `role` (HARD,
MEDIUM or SOFT) at `race_id` starts to fall off (the simulator's support edge), with its
uncertainty, from races that started strictly BEFORE `race_id` (race_ids.as_of, and by date when
both dates are known), never from a held-out race (data/eval_set.json plus the config's
held_out_extra) unless `allow_held_out` is passed AND that race is itself earlier. It reads two
committed tables and never loads a race at run time:

  data/derived/tyre_history.csv          one row per (race, role): the per-race observation
                                         (observed_tyre_life), built by `build`
  data/derived/tyre_history_stints.csv   the stints behind it (validation only)
  data/derived/tyre_nominations.csv      the Pirelli nomination per race weekend (role -> C-number),
                                         public pre-race information, plus the scheduled distance
  data/derived/compound_scale.csv        (scale, C-number) -> common hardness index

The observation of one compound in one race (observed_tyre_life): its stints are split by how
they ended (a green-flag pit stop, the chequered flag, a stop under SC / VSC / red flag, a
red-flag tyre change, a retirement); over the stints ended by a green-flag stop or by the flag,
the `quantile` (0.9) of the tyre ages at which they ended is the edge, in laps and as a share of
the race distance. Stops under a neutralisation and retirements say nothing about tyre life and
are left out; a stint run to the flag shows the tyre lasted at least that long, so it counts. A
high quantile rather than the maximum: one car's gamble does not set the edge. A pace check
(the late-stint excess over each stint's own early trend) is reported next to it, never used.

The prior (tyre_life_prior) shrinks a same-circuit estimate toward a pooled one:
  pooled   every usable earlier observation, each treated as another circuit's and moved onto
           the target: as a share of race distance (so a circuit with shorter laps runs more
           laps on the same rubber), times life_ratio ** steps for a compound `steps`
           common-index steps harder (positive) or softer than the target's, plus the severity
           step (the difference in the mean index of the nominated trios). Weighted by the
           category (other circuits, same compound > other circuits, other compound), times
           recency_per_season per season back, other_era across the 2022 regulation change,
           wet_race for a wet race.
  local    the same circuit's earlier observations, not moved by severity, weighted same
           compound > other compound with the same recency and era factors; a race flagged
           abnormal (a red flag, wet or damp: known once that race is over) weighs
           shrinkage.abnormal_same_circuit (0: left out).
  combined w_s * local + (1 - w_s) * pooled, w_s = n / (n + k0): n the circuit's effective
           number of dry, normal races (each race counts its heaviest observation's weight
           relative to a same-compound race of the same season, 1.0), k0 the stated prior
           strength shrinkage.prior_strength_races. One noisy race at a circuit (a red-flag
           Monaco) cannot override the pooled estimate.
Its tier is the most specific source that contributed: 1 same circuit and compound, 2 same
circuit, other compound, 3 pooled only, 4 no earlier race at all (a constant share of race
distance per role). The sd is the mixture's: each part's weighted spread (Kish n_effective),
plus w_s (1 - w_s) times their squared disagreement, in quadrature with a floor blended the same
way between the tier's and tier 3's.

A missing nomination (the target's or an observation's) never fails: the compounds are then
compared by role (HARD 0, MEDIUM 1, SOFT 2 steps), never counted as the same compound.

The module is NON_PHYSICS (calibration.NON_PHYSICS): nothing in the simulator calls it yet, so it
cannot move a simulated time. The change that places the support edge with it (fix/tyre-life-no-
hindsight) must move it to PHYSICS_MODULES, or carry its numbers into SimConfig, since from then
on this code, its config and its tables move lap times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import date as _date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .race_ids import as_of, canonical_race_id, parse_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED = ROOT / "data" / "derived"
NOMINATIONS = DERIVED / "tyre_nominations.csv"
SCALE = DERIVED / "compound_scale.csv"
HISTORY = DERIVED / "tyre_history.csv"
STINTS = DERIVED / "tyre_history_stints.csv"
MANIFEST = DERIVED / "tyre_history_manifest.json"
CONFIG = ROOT / "configs" / "tyre_history.yaml"   # not under configs/sim/, whose every yaml is read as a SimConfig
REPORT_DIR = ROOT / "reports" / "calibration" / "tyre_history"
EVAL_SET = ROOT / "data" / "eval_set.json"
CACHE_DIR = ROOT / "cache"
COMMAND = "python -m src.sim.tyre_history build"

ROLES = ("HARD", "MEDIUM", "SOFT")
ROLE_RANK = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}   # softer is larger, like the common index
WET = ("INTERMEDIATE", "WET")
SAME_C_SAME_K, SAME_C_OTHER_K, OTHER_C_SAME_K, OTHER_C_OTHER_K = (
    "same_circuit_same_compound", "same_circuit_other_compound",
    "other_circuit_same_compound", "other_circuit_other_compound")
CATEGORIES = (SAME_C_SAME_K, SAME_C_OTHER_K, OTHER_C_SAME_K, OTHER_C_OTHER_K)
END_KINDS = ("pitted", "flag", "neutralised", "red_flag_change", "retired")
USABLE_ENDS = ("pitted", "flag")
HISTORY_COLUMNS = [
    "race_id", "season", "round", "date", "era", "circuit", "total_laps", "role", "compound", "scale",
    "common_index", "trio_index", "n_stints", "n_pitted", "n_flag", "n_neutralised", "n_retired", "n_usable",
    "edge_laps", "edge_frac", "max_age", "median_end_age", "pace_n_stints", "pace_late_excess_s",
    "pace_falloff_share", "wet_share", "red_flag", "excluded"]
OBS_COLUMNS = ["race_id", "total_laps", "role", "n_stints", "n_pitted", "n_flag", "n_neutralised", "n_retired",
               "n_usable", "edge_laps", "edge_frac", "max_age", "median_end_age", "pace_n_stints",
               "pace_late_excess_s", "pace_falloff_share", "wet_share", "red_flag"]


@dataclass(frozen=True)
class TyreLifePrior:
    """The prior for one compound role at one race. edge_laps: the tyre age (laps on the set)
    at which fall-off begins; sd_laps its standard deviation; n_effective the Kish effective
    number of observations behind it (0 at tier 4); sources the earlier races used, heaviest
    first, as (race_id, role, weight); tier 1-4 (module docstring)."""
    edge_laps: float
    sd_laps: float
    n_effective: float
    sources: tuple
    tier: int
    edge_frac: float = float("nan")
    sd_frac: float = float("nan")
    total_laps: int = 0
    compound: str | None = None          # the role's C-number at this race; None when not nominated
    life_ratio: float = float("nan")      # per common-index step, used to move other compounds
    notes: tuple = field(default_factory=tuple)

    def interval(self, level: float = 0.68) -> tuple[float, float]:
        """The central `level` interval in laps (normal), e.g. 0.68 or 0.9."""
        from statistics import NormalDist
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return self.edge_laps - z * self.sd_laps, self.edge_laps + z * self.sd_laps


# --- inputs ---------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _config_cached(path: str) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def load_config(path=CONFIG) -> dict:
    return _config_cached(str(path))


def era(season: int) -> str:
    from .event_rates import era as _era
    return _era(int(season))


def held_out(config=None, path=EVAL_SET) -> set[str]:
    """data/eval_set.json's races plus the config's held_out_extra, canonical ids. Reads the
    list only; nothing about those races is loaded."""
    cfg = config or load_config()
    p = Path(path)
    races = json.loads(p.read_text()).get("races", []) if p.is_file() else []
    return {canonical_race_id(r) for r in [*races, *(cfg.get("held_out_extra") or [])]}


def load_scale(path=SCALE) -> dict[tuple[str, str], float]:
    """(scale, C-number) -> common hardness index (larger is softer)."""
    if not Path(path).is_file():
        return {}
    t = pd.read_csv(path, comment="#", dtype=str)
    return {(r.scale.strip(), r.compound.strip().upper()): float(r.common_index) for r in t.itertuples()}


# Per process: the committed tables read once per file version (path, mtime, size), and life_ratio
# once per history it is given. Pure functions of their inputs, so the values are the same; a
# caller gets its own copy of a table.
_TABLES: dict = {}
_RATIOS: dict = {}
_RATIOS_MAX = 4096


def _file_key(path) -> tuple:
    p = Path(path)
    st = p.stat()
    return str(p.resolve()), st.st_mtime_ns, st.st_size


def load_nominations(path=NOMINATIONS, scale=None) -> pd.DataFrame:
    """The nomination table indexed by race_id, with hard_index, medium_index, soft_index added
    (NaN where the nomination or its scale mapping is missing)."""
    if not Path(path).is_file():
        return pd.DataFrame(columns=["season", "round", "circuit", "hard", "medium", "soft", "scale"]).rename_axis("race_id")
    if scale is not None:
        return _read_nominations(path, scale)
    key = ("nominations", _file_key(path), _file_key(SCALE) if Path(SCALE).is_file() else None)
    if key not in _TABLES:
        _TABLES[key] = _read_nominations(path, None)
    return _TABLES[key].copy()


def _read_nominations(path, scale) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={"hard": str, "medium": str, "soft": str, "scale": str, "date": str})
    t["race_id"] = t["race_id"].map(canonical_race_id)
    scale = load_scale() if scale is None else scale
    for role in ROLES:
        col = role.lower()
        t[col] = t[col].where(t[col].notna(), None)
        t[f"{col}_index"] = [scale.get((str(s).strip(), str(c).strip().upper()), np.nan) if isinstance(c, str) else np.nan
                             for s, c in zip(t["scale"], t[col])]
    return t.set_index("race_id")


def load_history(path=HISTORY) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    key = ("history", _file_key(path))
    if key not in _TABLES:
        _TABLES[key] = pd.read_csv(path, dtype={"compound": str, "scale": str, "date": str, "excluded": str})
    return _TABLES[key].copy()


def _date_of(value) -> _date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() in ("", "nan"):
        return None
    return _date.fromisoformat(str(value)[:10])


@dataclass
class _Race:
    race_id: str
    season: int
    round: int
    date: _date | None
    circuit: str
    scheduled_laps: float | None
    compound: dict
    index: dict

    @property
    def trio(self) -> float:
        """The mean common index of the nominated trio: the circuit's severity as Pirelli judged
        it before the race (a harder trio for a harsher circuit); NaN unless all three are known."""
        v = [self.index.get(k, np.nan) for k in ROLES]
        return float(np.mean(v)) if all(np.isfinite(x) for x in v) else float("nan")


def race_info(race_id: str, nominations: pd.DataFrame | None = None) -> _Race:
    """What is known about a race before it starts: season, round, date, circuit, scheduled
    distance and the nomination per role, from the nomination table, else the race id and the
    cached schedule (event_rates.circuit_info) with no nomination."""
    rid = canonical_race_id(race_id)
    season, rnd, slug = parse_race_id(rid)
    noms = load_nominations() if nominations is None else nominations
    if rid in noms.index:
        r = noms.loc[rid]
        laps = pd.to_numeric(r.get("scheduled_laps"), errors="coerce") if "scheduled_laps" in r else np.nan
        return _Race(rid, season, rnd, _date_of(r.get("date")), str(r.get("circuit") or slug),
                     None if pd.isna(laps) else float(laps),
                     {k: (r.get(k.lower()) if isinstance(r.get(k.lower()), str) else None) for k in ROLES},
                     {k: float(r.get(f"{k.lower()}_index", np.nan)) for k in ROLES})
    circuit = slug
    if (CACHE_DIR / str(season)).is_dir():   # never reach for a schedule the cache does not hold
        try:
            from .event_rates import circuit_info
            circuit = str(circuit_info([rid]).iloc[0]["circuit"])
        except Exception:   # schedule unreadable: the slug is its own circuit
            pass
    return _Race(rid, season, rnd, None, circuit, None, {k: None for k in ROLES}, {k: np.nan for k in ROLES})


# --- the per-race observation -----------------------------------------------------------------
def stint_table(frame: pd.DataFrame, config=None) -> pd.DataFrame:
    """One row per (driver, stint) of a race lap frame (data._columns): compound, first and last
    lap, laps, end_age (the tyre age on the stint's last lap) and `end`, how it ended:
      pitted           a pit stop at its end, track green on the in-lap
      neutralised      a pit stop on an in-lap run under SC, VSC or red flag (opportunistic)
      red_flag_change  the next stint starts without a pit stop (a tyre change under a red flag)
      flag             the driver's last stint and he took the chequered flag
      retired          the driver's last stint and he did not."""
    cfg = config or load_config()
    codes = tuple(cfg["observation"]["neutralised_codes"])
    f = frame[frame["stint"].notna()].sort_values(["driver", "lap"])
    if not len(f):
        return pd.DataFrame(columns=["driver", "stint", "compound", "first_lap", "last_lap", "laps", "end_age", "end"])
    total = int(f["lap"].max())
    finish = f.loc[f["lap"] == total, "end_s"].min() if "end_s" in f else np.nan
    rows = []
    for drv, g in f.groupby("driver", sort=True):
        stints = list(g.groupby("stint", sort=True))
        last_lap = int(g["lap"].max())
        end_s = g.loc[g["lap"] == last_lap, "end_s"].max() if "end_s" in g else np.nan
        took_flag = (bool(np.isfinite(end_s)) and bool(np.isfinite(finish)) and end_s >= finish - 1e-6) or \
            (not np.isfinite(end_s) and last_lap >= total - 1)
        for i, (sid, s) in enumerate(stints):
            last = s.iloc[-1]
            ages = pd.to_numeric(s["tyre_age"], errors="coerce")
            if i + 1 < len(stints):
                if bool(last["pit_in"]):
                    status = str(last["track_status"])
                    end = "neutralised" if any(c in status for c in codes) else "pitted"
                else:
                    end = "red_flag_change"
            else:
                end = "flag" if took_flag else "retired"
            rows.append({"driver": drv, "stint": int(sid), "compound": str(s["compound"].iloc[0]).upper(),
                         "first_lap": int(s["lap"].iloc[0]), "last_lap": int(last["lap"]), "laps": len(s),
                         "end_age": float(ages.max()) if ages.notna().any() else np.nan, "end": end})
    return pd.DataFrame(rows)


def _pace_excess(frame: pd.DataFrame, stints: pd.DataFrame, cfg) -> dict:
    """compound -> (n stints checked, median late excess in s, share of them above pace_falloff_s)."""
    o = cfg["observation"]
    out = {}
    green = (frame["track_status"].astype(str) == "1") & ~frame["pit_in"].astype(bool) & \
        ~frame["pit_out"].astype(bool) & frame["lap_time_s"].notna() & (frame["lap"] > 1)
    g_all = frame[green]
    by_key = {k: v for k, v in g_all.groupby(["driver", "stint"])}
    for comp, sc in stints[stints["end"].isin(USABLE_ENDS)].groupby("compound"):
        excess = []
        for r in sc.itertuples():
            laps = by_key.get((r.driver, float(r.stint)), by_key.get((r.driver, r.stint)))
            if laps is None or len(laps) < o["pace_min_green_laps"]:
                continue
            laps = laps.sort_values("lap")
            n_fit = max(4, int(len(laps) * o["pace_fit_share"]))
            x, y = laps["lap"].to_numpy(float), laps["lap_time_s"].to_numpy(float)
            slope, icpt = np.polyfit(x[:n_fit], y[:n_fit], 1)
            late = y[-o["pace_late_laps"]:] - (icpt + slope * x[-o["pace_late_laps"]:])
            excess.append(float(np.median(late)))
        if excess:
            out[comp] = (len(excess), float(np.median(excess)), float(np.mean(np.array(excess) > o["pace_falloff_s"])))
    return out


def observed_tyre_life(race, *, race_id: str | None = None, config=None, allow_held_out=False,
                       return_stints=False):
    """The per-race observation, one row per dry role raced (module docstring): race_id,
    total_laps, role, n_stints, n_pitted, n_flag, n_neutralised, n_retired, n_usable, edge_laps
    (the `quantile` of the end ages of the usable stints; NaN under min_stints usable stints),
    edge_frac (edge_laps / total_laps), max_age, median_end_age, pace_n_stints,
    pace_late_excess_s, pace_falloff_share, wet_share.

    `race` is a race lap frame (data._columns; then pass `race_id`) or a race id, whose race
    session is loaded from the offline cache; a held-out race id is refused before anything
    about it is loaded unless `allow_held_out`. With `return_stints`, (observation, stints)."""
    cfg = config or load_config()
    if isinstance(race, str):
        race_id = canonical_race_id(race)
        if race_id in held_out(cfg) and not allow_held_out:
            raise ValueError(f"{race_id} is held out (data/eval_set.json / held_out_extra): its race data is never loaded")
        frame = _load_frame(race_id)
    else:
        frame = race
    o = cfg["observation"]
    stints = stint_table(frame, cfg)
    total = int(frame["lap"].max()) if len(frame) else 0
    comp = frame["compound"].astype(str).str.upper()
    wet_share = float(comp.isin(WET).mean()) if len(frame) else 0.0
    red_flag = bool(frame["track_status"].astype(str).str.contains("5").any()) if len(frame) else False
    pace = _pace_excess(frame, stints, cfg) if len(stints) else {}
    rows = []
    for role in ROLES:
        s = stints[stints["compound"] == role]
        if not len(s):
            continue
        usable = s[s["end"].isin(USABLE_ENDS)]["end_age"].dropna().to_numpy(float)
        enough = len(usable) >= o["min_stints"]
        edge = float(np.quantile(usable, o["quantile"])) if enough else np.nan
        pn, pex, pshare = pace.get(role, (0, np.nan, np.nan))
        rows.append({"race_id": race_id, "total_laps": total, "role": role, "n_stints": len(s),
                     "n_pitted": int((s["end"] == "pitted").sum()), "n_flag": int((s["end"] == "flag").sum()),
                     "n_neutralised": int(s["end"].isin(("neutralised", "red_flag_change")).sum()),
                     "n_retired": int((s["end"] == "retired").sum()), "n_usable": len(usable),
                     "edge_laps": edge, "edge_frac": edge / total if enough and total else np.nan,
                     "max_age": float(usable.max()) if len(usable) else np.nan,
                     "median_end_age": float(np.median(usable)) if len(usable) else np.nan,
                     "pace_n_stints": pn, "pace_late_excess_s": pex, "pace_falloff_share": pshare,
                     "wet_share": wet_share, "red_flag": red_flag})
    obs = pd.DataFrame(rows, columns=OBS_COLUMNS)
    if return_stints:
        return obs, stints.assign(race_id=race_id)
    return obs


def _load_frame(race_id: str) -> pd.DataFrame:
    from .data import _columns, load_session
    season, rnd, _ = parse_race_id(race_id)
    session = load_session(season, rnd, "R", cache_dir=str(CACHE_DIR))
    return _columns(session.laps)


# --- the history table --------------------------------------------------------------------------
def cached_races(cache_dir=CACHE_DIR) -> list[tuple[str, str]]:
    """(race_id, date) of every race weekend in the offline cache, from the weekend folders
    (<season>/<date>_<Event_Name>) and the cached schedule; read from folder names only."""
    from .data import season_schedule
    from .race_ids import event_slug, make_race_id_from_schedule
    out = []
    for d in sorted(Path(cache_dir).glob("20[0-9][0-9]/*/")):
        day, _, name = d.name.partition("_")
        season = int(d.parent.name)
        sched = season_schedule(season, str(cache_dir))
        if sched is None:
            continue
        hit = sched[sched["EventName"].map(event_slug) == event_slug(name.replace("_", " "))]
        if len(hit):
            out.append((make_race_id_from_schedule(hit.iloc[0], sched, season), day))
    return sorted(set(out), key=lambda r: parse_race_id(r[0])[:2])


def build(races=None, *, out=HISTORY, stints_out=STINTS, manifest=MANIFEST, allow_held_out=False,
          config=None, log=print) -> pd.DataFrame:
    """The history table from `races` (default: every cached race weekend not held out):
    observed_tyre_life per race, with its date, era, circuit and nomination. Held-out races are
    refused (or, by default, left out of the default list) before anything about them is read."""
    cfg = config or load_config()
    held = held_out(cfg)
    dates = dict(cached_races())
    if races is None:
        races = [r for r in dates if r not in held]
    races = [canonical_race_id(r) for r in races]
    bad = [r for r in races if r in held]
    if bad and not allow_held_out:
        raise ValueError(f"{bad} held out: never built into the tyre history (pass --allow-held-out to override)")
    noms = load_nominations()
    excluded = cfg.get("excluded_observations") or {}
    obs_rows, stint_rows, manifest_races = [], [], []
    for rid in sorted(races, key=lambda r: parse_race_id(r)[:2]):
        log(f"  {rid}")
        obs, st = observed_tyre_life(rid, config=cfg, allow_held_out=allow_held_out, return_stints=True)
        info = race_info(rid, noms)
        nom_date = info.date.isoformat() if info.date else None
        obs = obs.assign(season=info.season, round=info.round, date=dates.get(rid) or nom_date,
                         era=era(info.season), circuit=info.circuit,
                         compound=obs["role"].map(info.compound), common_index=obs["role"].map(info.index),
                         trio_index=info.trio,
                         scale=(noms.loc[rid, "scale"] if rid in noms.index else None),
                         excluded=excluded.get(rid))
        obs_rows.append(obs)
        stint_rows.append(st)
        manifest_races.append({"race_id": rid, "date": dates.get(rid) or nom_date, "circuit": info.circuit})
    obs_rows = [o for o in obs_rows if len(o)]
    hist = pd.concat(obs_rows, ignore_index=True)[HISTORY_COLUMNS] if obs_rows else pd.DataFrame(columns=HISTORY_COLUMNS)
    stints = pd.concat(stint_rows, ignore_index=True) if stint_rows else pd.DataFrame()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out, index=False, float_format="%.6g")
    if len(stints):
        stints[["race_id", *[c for c in stints.columns if c != "race_id"]]].to_csv(stints_out, index=False)
    Path(manifest).write_text(json.dumps({
        "command": COMMAND,
        "description": "Per-race tyre-life observations (src/sim/tyre_history.py observed_tyre_life) of the races below; "
                       "read at run time by tyre_life_prior, which uses only those strictly earlier than its race.",
        "races": manifest_races,
        "held_out_never_loaded": sorted(held),
        "excluded_observations": excluded,
        "observation": cfg["observation"],
    }, indent=1) + "\n")
    return hist


# --- the prior ------------------------------------------------------------------------------------
def _usable(history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    h["race_id"] = h["race_id"].map(canonical_race_id)
    ex = h["excluded"] if "excluded" in h else pd.Series(None, index=h.index)
    ok = pd.to_numeric(h["edge_frac"], errors="coerce").gt(0) & (ex.isna() | (ex.astype(str).str.strip() == ""))
    return h[ok]


def life_ratio(history: pd.DataFrame, config=None) -> tuple[float, int]:
    """(per-step life ratio, pairs used): the geometric mean over the races of `history` (already
    as-of) of (edge_softer / edge_harder) ** (1 / index steps) for every pair of its indexed
    compounds; the config's default under min_pairs pairs. Clipped to life_ratio.clip."""
    cfg = config or load_config()
    lr = cfg["life_ratio"]
    cols = [c for c in ("race_id", "common_index", "edge_frac") if c in history]
    key = (hashlib.sha256(pd.util.hash_pandas_object(history[cols], index=False).to_numpy().tobytes()).hexdigest(),
           tuple(cols), json.dumps(lr, sort_keys=True, default=str))
    if key not in _RATIOS:
        if len(_RATIOS) >= _RATIOS_MAX:
            _RATIOS.pop(next(iter(_RATIOS)))
        _RATIOS[key] = _life_ratio(history, lr)
    return _RATIOS[key]


def _life_ratio(history: pd.DataFrame, lr: dict) -> tuple[float, int]:
    logs = []
    for _, g in history.groupby("race_id"):
        g = g[pd.to_numeric(g["common_index"], errors="coerce").notna()]
        recs = list(zip(g["common_index"].astype(float), g["edge_frac"].astype(float)))
        for i in range(len(recs)):
            for j in range(len(recs)):
                d = recs[i][0] - recs[j][0]
                if d > 0:   # i softer than j
                    logs.append(math.log(recs[i][1] / recs[j][1]) / d)
    if len(logs) < lr["min_pairs"]:
        return float(lr["default_per_step"]), len(logs)
    lo, hi = lr["clip"]
    return float(min(hi, max(lo, math.exp(float(np.mean(logs)))))), len(logs)


def tyre_life_prior(race_id: str, role: str, *, races=None, total_laps: float | None = None,
                    history: pd.DataFrame | None = None, nominations: pd.DataFrame | None = None,
                    allow_held_out: bool = False, config: dict | None = None,
                    use_compound: bool = True, use_circuit: bool = True,
                    use_severity: bool | None = None, ratings: str = "off",
                    guidance: str = "off") -> TyreLifePrior:
    """The tyre-life prior of `role` (HARD / MEDIUM / SOFT) at `race_id` (module docstring).

    races:          restrict the history to these race ids (still as-of and held-out filtered)
    total_laps:     the race distance in laps (default: the nomination table's scheduled_laps,
                    else the same circuit's latest earlier race, else config default_race_laps)
    history, nominations: tables to use instead of the committed ones (tests, validation)
    allow_held_out: let an EARLIER held-out race feed the prior (default off: never)
    use_compound, use_circuit: False ignores the C-number (compounds compared by role) or the
                    circuit (every race counts as another circuit); for the validation's ablation
    use_severity:   move another circuit's observation by the difference in severity, read off
                    the nominated trios (default: the config's scaling.severity_from_nomination)
    ratings:        the supplier's circuit ratings as covariates of tyre life (src/sim/prerace.py,
                    SimConfig.prerace.tyre_life_ratings): "off" (as before); "replace": a ridge fit on
                    the earlier races moves each pooled observation instead of the trio shift;
                    "supplement": the trio shift, then the fit on what it leaves. No fit (too few
                    earlier rated races) or no ratings for a race: the trio shift alone.
    guidance:       combine, by precision, with the supplier's pre-race guidance (the longest stint
                    its expected strategies imply on the role, prerace.guidance_life), its bias and
                    sd from the earlier races' guidance against their observed edges
    Always returns; never reads a race that started on or after `race_id`."""
    role = str(role).upper()
    if role not in ROLES:
        raise ValueError(f"role {role!r} is not one of {ROLES}")
    cfg = config or load_config()
    w = cfg["weights"]
    rid = canonical_race_id(race_id)
    noms = load_nominations() if nominations is None else nominations
    target = race_info(rid, noms)
    h = _usable(load_history() if history is None else history)
    earlier = set(as_of(rid, list(h["race_id"].unique())))
    if target.date is not None and "date" in h:
        # strictly before by date too, where the history row has a date
        late = {r for r, d in zip(h["race_id"], h["date"]) if _date_of(d) is not None and _date_of(d) >= target.date}
        earlier -= late
    if races is not None:
        earlier &= {canonical_race_id(r) for r in races}
    if not allow_held_out:
        earlier -= held_out(cfg)
    h = h[h["race_id"].isin(earlier)]
    notes = []

    laps = total_laps or target.scheduled_laps
    if not laps:
        same = h[h["circuit"] == target.circuit].sort_values(["season", "round"])
        laps = float(same["total_laps"].iloc[-1]) if len(same) else float(cfg["default_race_laps"])
        notes.append(f"race distance {laps:g} laps from {'the circuit history' if len(same) else 'config default'}")
    laps = float(laps)
    compound = target.compound.get(role)
    idx_t = target.index.get(role, np.nan)
    if use_compound and not np.isfinite(idx_t):
        notes.append(f"no nomination for {role} at {rid}: compounds compared by role")

    ratio, n_pairs = life_ratio(h, cfg)
    floors = {int(k): float(v) for k, v in cfg["sd_floor_frac"].items()}
    if use_severity is None:
        use_severity = bool((cfg.get("scaling") or {}).get("severity_from_nomination", False))
    shrink = cfg["shrinkage"]
    trio_t = target.trio
    rfit = None
    if ratings != "off" and use_compound:
        rfit = _ratings_fit(h, ratio, ratings == "supplement" and use_severity)
    pooled, local = [], []     # (race_id, role, category, weight, moved edge_frac)
    for r in h.itertuples():
        idx_h = float(r.common_index) if pd.notna(r.common_index) else np.nan
        if use_compound and np.isfinite(idx_t) and np.isfinite(idx_h):
            steps = idx_t - idx_h
            same_k = abs(steps) <= w["same_compound_tolerance"]
        else:
            steps = float(ROLE_RANK[role] - ROLE_RANK[r.role])
            same_k = (not use_compound) and steps == 0
        base = w["recency_per_season"] ** max(0, target.season - int(r.season))
        if era(int(r.season)) != era(target.season):
            base *= w["other_era"]
        decay = 1.0 if same_k else w["compound_step_decay"] ** max(0.0, abs(steps) - 1.0)
        edge = float(r.edge_frac)
        # the pooled estimate: every earlier race as "another circuit", moved by its severity
        moved = steps
        trio_h = float(getattr(r, "trio_index", np.nan))
        if use_compound and use_severity and np.isfinite(idx_t) and np.isfinite(idx_h) \
                and np.isfinite(trio_t) and np.isfinite(trio_h):
            # a circuit given a softer trio is gentler: its tyres last longer than they would at
            # the target, by the same per-step ratio
            moved = steps + trio_h - trio_t
        shift = 1.0
        if rfit is not None and np.isfinite(idx_t) and np.isfinite(idx_h):
            from .prerace import ratings_shift
            if ratings == "replace" and _rated(rid, r.race_id):
                moved = steps
            shift = math.exp(ratings_shift(rfit, rid, r.race_id))
        cat = OTHER_C_SAME_K if same_k else OTHER_C_OTHER_K
        wet = float(r.wet_share or 0.0) > cfg["observation"]["wet_lap_share"]
        pooled.append((r.race_id, r.role, cat, w[cat] * decay * base * (w["wet_race"] if wet else 1.0),
                       edge * ratio ** moved * shift))
        # the same-circuit estimate: this circuit's own earlier races, not moved by severity;
        # a race flagged abnormal (red flag, wet or damp) weighs abnormal_same_circuit (0: left out)
        if use_circuit and r.circuit == target.circuit:
            cat = SAME_C_SAME_K if same_k else SAME_C_OTHER_K
            wt = w[cat] * decay * base * (shrink["abnormal_same_circuit"] if abnormal(r, cfg) else 1.0)
            if wt > 0:
                local.append((r.race_id, r.role, cat, wt, edge * ratio ** steps))

    if not pooled:
        frac, sd = float(cfg["tier4_frac"][role]), floors[4]
        notes.append("no usable earlier race: constant prior")
        return TyreLifePrior(frac * laps, sd * laps, 0.0, (), 4, frac, sd, int(round(laps)), compound, ratio, tuple(notes))
    m_pool, v_pool = _weighted(pooled)
    # the same circuit counts as its effective number of dry, normal races: each race its heaviest
    # observation's weight relative to a same-compound race of the same season and era (1.0)
    per_race = {}
    for rid_h, _, _, wt, _ in local:
        per_race[rid_h] = max(per_race.get(rid_h, 0.0), wt / w[SAME_C_SAME_K])
    n_local = float(sum(per_race.values()))
    k0 = float(shrink["prior_strength_races"])
    w_s = n_local / (n_local + k0) if local else 0.0
    w_p = 1.0 - w_s
    if local:
        m_loc, v_loc = _weighted(local)
        tier = 1 if any(c == SAME_C_SAME_K for _, _, c, _, _ in local) else 2
    else:
        m_loc, v_loc, tier = 0.0, 0.0, 3
    mean = w_p * m_pool + w_s * m_loc
    # a two-part mixture: each part's spread, plus their disagreement
    var = w_p * v_pool + w_s * v_loc + w_p * w_s * (m_loc - m_pool) ** 2
    floor2 = w_s * floors[tier] ** 2 + w_p * floors[3] ** 2
    sd = math.sqrt(var + floor2)
    share = {}
    for part, wt_part in ((pooled, w_p), (local, w_s)):
        tot = sum(x[3] for x in part)
        for rid_h, role_h, _, wt, _ in part:
            if tot > 0 and wt_part > 0:
                share[(rid_h, role_h)] = share.get((rid_h, role_h), 0.0) + wt_part * wt / tot
    wts = np.array(list(share.values()))
    n_eff = float(wts.sum() ** 2 / np.sum(wts ** 2))
    sources = tuple((k[0], k[1], round(float(v), 6)) for k, v in sorted(share.items(), key=lambda kv: -kv[1]))
    if local:
        notes.append(f"same circuit: {n_local:.2f} effective races, weight {w_s:.2f} against the pooled estimate")
    if n_pairs < cfg["life_ratio"]["min_pairs"]:
        notes.append(f"life ratio {ratio:g} per step: config default ({n_pairs} earlier pairs)")
    if rfit is not None:
        notes.append(f"circuit ratings ({ratings}) fitted on {rfit['races']} earlier races")
    if guidance and guidance != "off":
        mean, sd = _with_guidance(rid, role, laps, h, mean, sd, notes, "windows" if guidance is True else guidance)
    return TyreLifePrior(mean * laps, sd * laps, n_eff, sources, tier, mean, sd, int(round(laps)), compound, ratio,
                         tuple(notes))


def _rated(*race_ids) -> bool:
    from .prerace import LIFE_COVARIATES, ratings_of
    return all(ratings_of(r, LIFE_COVARIATES) is not None for r in race_ids)


def _ratings_fit(h: pd.DataFrame, ratio: float, after_trio: bool):
    """prerace.life_severity_fit on the earlier usable rows `h` (already as-of): y = log edge share
    moved to common index 0 by the life ratio (and, after_trio, to a trio of 0 by the trio shift)."""
    from .prerace import life_severity_fit
    lr = math.log(ratio)
    rows = []
    for r in h.itertuples():
        idx, trio = float(r.common_index), float(getattr(r, "trio_index", np.nan))
        if not np.isfinite(idx) or (after_trio and not np.isfinite(trio)):
            continue
        y = math.log(float(r.edge_frac)) - idx * lr + (trio * lr if after_trio else 0.0)
        rows.append((r.race_id, y))
    return life_severity_fit(rows)


def _with_guidance(rid, role, laps, h, mean, sd, notes, mode="windows") -> tuple[float, float]:
    """(mean, sd) as shares of distance, combined by precision with the guidance's stint on `role`."""
    from .prerace import guidance_error_as_of, guidance_life
    g = guidance_life(rid, laps, mode).get(role)
    if not g:
        return mean, sd
    bias, lsd, n = guidance_error_as_of(h, mode)
    g_frac = g * math.exp(bias) / laps
    g_sd = g_frac * lsd
    w_h, w_g = 1.0 / sd ** 2, 1.0 / g_sd ** 2
    m = (w_h * mean + w_g * g_frac) / (w_h + w_g)
    notes.append(f"pre-race guidance {g:g} laps (bias {bias:+.2f} log, sd {lsd:.2f} log, {n} earlier races) "
                 f"weight {w_g / (w_h + w_g):.2f}")
    return m, math.sqrt(1.0 / (w_h + w_g))


def _weighted(rows) -> tuple[float, float]:
    """(weighted mean, weighted variance with the Kish n_effective correction) of the moved edges."""
    wts = np.array([r[3] for r in rows])
    x = np.array([r[4] for r in rows])
    mean = float(np.sum(wts * x) / np.sum(wts))
    n_eff = float(np.sum(wts) ** 2 / np.sum(wts ** 2))
    var = float(np.sum(wts * (x - mean) ** 2) / np.sum(wts))
    return mean, var * n_eff / (n_eff - 1.0) if n_eff > 1.0 else var


def abnormal(row, config=None) -> bool:
    """A history race flagged abnormal: a red flag, or more than damp_lap_share of its laps on
    intermediates or wets. Both are known once that race is over, so before any later race."""
    cfg = config or load_config()
    red = getattr(row, "red_flag", False)
    red = bool(red) if isinstance(red, (bool, np.bool_)) else str(red).strip().lower() in ("true", "1")
    return red or float(getattr(row, "wet_share", 0.0) or 0.0) > cfg["observation"]["damp_lap_share"]


# --- validation ----------------------------------------------------------------------------------
VARIANTS = {"full": {}, "no_severity": {"use_severity": False}, "no_compound": {"use_compound": False},
            "no_circuit": {"use_circuit": False}, "neither": {"use_compound": False, "use_circuit": False}}


def validate(history: pd.DataFrame | None = None, stints: pd.DataFrame | None = None,
             nominations: pd.DataFrame | None = None, config=None, margin_laps: float = 3.0) -> pd.DataFrame:
    """As-of (hence leave-one-race-out) on every usable history observation of a race not held
    out: the prior from the races strictly before it against what the race showed. One row per
    (variant, race, role): prior edge and sd, observed edge, error (prior - observed), within
    the 68% and 90% intervals, and the share of the race's usable stints on that role ending
    beyond edge + margin_laps."""
    cfg = config or load_config()
    hist = load_history() if history is None else history
    st = (pd.read_csv(STINTS) if Path(STINTS).is_file() else pd.DataFrame()) if stints is None else stints
    held = held_out(cfg)
    target_rows = _usable(hist)
    target_rows = target_rows[~target_rows["race_id"].isin(held)]
    out = []
    for name, kw in VARIANTS.items():
        for r in target_rows.itertuples():
            p = tyre_life_prior(r.race_id, r.role, total_laps=float(r.total_laps), history=hist,
                                nominations=nominations, config=cfg, **kw)
            lo68, hi68 = p.interval(0.68)
            lo90, hi90 = p.interval(0.90)
            s = st[(st["race_id"] == r.race_id) & (st["compound"] == r.role) & st["end"].isin(USABLE_ENDS)] \
                if len(st) else pd.DataFrame(columns=["end_age"])
            ages = pd.to_numeric(s["end_age"], errors="coerce").dropna()
            out.append({"variant": name, "race_id": r.race_id, "era": era(int(r.season)), "role": r.role,
                        "abnormal_race": abnormal(r, cfg),
                        "compound": r.compound, "tier": p.tier, "n_effective": p.n_effective,
                        "prior_edge": p.edge_laps, "prior_sd": p.sd_laps, "observed_edge": float(r.edge_laps),
                        "error": p.edge_laps - float(r.edge_laps), "abs_error": abs(p.edge_laps - float(r.edge_laps)),
                        "in68": lo68 <= r.edge_laps <= hi68, "in90": lo90 <= r.edge_laps <= hi90,
                        "n_stints": len(ages), "n_beyond": int((ages > p.edge_laps + margin_laps).sum()),
                        # the same count against the race's own observed edge: the floor any prior can reach
                        "n_beyond_observed": int((ages > float(r.edge_laps) + margin_laps).sum())})
    return pd.DataFrame(out)


def summarise(v: pd.DataFrame) -> pd.DataFrame:
    """Per variant x (tier, era, all): n, mean signed error, MAE, 68 / 90 coverage, share of
    stints beyond edge + margin."""
    def agg(g):
        return pd.Series({"n": len(g), "mean_error": g["error"].mean(), "mae": g["abs_error"].mean(),
                          "cover68": g["in68"].mean(), "cover90": g["in90"].mean(),
                          "stints": int(g["n_stints"].sum()),
                          "beyond_share": g["n_beyond"].sum() / g["n_stints"].sum() if g["n_stints"].sum() else np.nan,
                          "beyond_share_observed": (g["n_beyond_observed"].sum() / g["n_stints"].sum()
                                                    if g["n_stints"].sum() else np.nan)})
    parts = []
    for name, g in v.groupby("variant", sort=False):
        parts.append(agg(g).to_frame().T.assign(variant=name, group="all"))
        for t, gt in g.groupby("tier"):
            parts.append(agg(gt).to_frame().T.assign(variant=name, group=f"tier {t}"))
        for e, ge in g.groupby("era"):
            parts.append(agg(ge).to_frame().T.assign(variant=name, group=f"era {e}"))
        if "abnormal_race" in g:
            # the race being priced was itself red-flagged, wet or damp: no prior can know that
            normal = g[~g["abnormal_race"].astype(bool)]
            if len(normal):
                parts.append(agg(normal).to_frame().T.assign(variant=name, group="normal races"))
    s = pd.concat(parts, ignore_index=True)
    return s[["variant", "group", "n", "mean_error", "mae", "cover68", "cover90", "stints", "beyond_share",
              "beyond_share_observed"]]


# --- command line ------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.tyre_history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the history table from the cached races not held out")
    b.add_argument("--races", nargs="+", default=None)
    b.add_argument("--allow-held-out", action="store_true")
    v = sub.add_parser("validate", help="as-of validation of the prior, with the weighting ablations")
    v.add_argument("--out-dir", default=str(REPORT_DIR))
    args = ap.parse_args(argv)
    if args.cmd == "build":
        hist = build(args.races, allow_held_out=args.allow_held_out)
        print(f"wrote {len(hist)} rows over {hist['race_id'].nunique()} races to {HISTORY.relative_to(ROOT)}")
        return 0
    val = validate()
    summ = summarise(val)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val.to_csv(out / "validation_rows.csv", index=False, float_format="%.4g")
    summ.to_csv(out / "validation_summary.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
