"""Circuit-specific overtaking difficulty: on-track passes counted from the lap data, per race, and
an as-of per-circuit table built from them (docs/eval.md "Overtaking difficulty").

    python -m src.sim.overtaking build     [--races ...]       # data/derived/overtaking.csv (+ _pairs.csv)
    python -m src.sim.overtaking validate  [--out DIR]         # as-of / leave-one-race-out against a pooled constant

The observation of one race (observe_race), from the timing rows only:

  close lap  a car F starts lap L (the line at the end of L - 1) within CLOSE_GAP_S of the car A
             directly ahead on the same lap count (data.gap_columns), lap L >= 2, both cars have a
             lap-L row with an end time, neither is on an in-lap or out-lap on L, and lap L is
             green for both (track status "1"); A's L-1 row green too (no SC restart bunching).
  pass       on such a lap F crosses the line at the end of L before A does. Pit stops, lap 1, SC /
             VSC laps, retirements (A has no lap-L end) and lapping (same lap count only) never count.
  rate       passes / close laps: how often a car that is in passing range completes the pass on
             the lap, so a processional race at a wide track (few cars in range) is not read as hard.
  all passes every order change between two cars on the same lap count between the line at
             L - 1 and the line at L under the same exclusions (not only the car directly ahead),
             per green racing lap: the "overtakes per race" count.
  pace       a free-air pace fit per race (lap time ~ driver-stint intercept + compound age slope +
             a common lap slope, on green clean laps with >= FREE_AIR_FIT_S to the car ahead); the
             follower's pace advantage on a close lap is pred(A) - pred(F) (positive: F faster).
             theta: the pace advantage at which half the close laps end in a pass (a logistic
             with a pooled slope); the ghost model's DRS pass threshold is in the same units.
  dirty air  the mean residual over the pace fit of close laps (gap < DIRTY_RANGE_S) less that of
             free-air laps.
  sectors    the boundary (S1, S2, line) at which F first runs ahead of A on a pass lap, from the
             cumulative sector session times, where the timing has them.

The as-of table (difficulty_prior) shrinks a same-circuit estimate toward the pooled rate of the
earlier races (strictly earlier, race_ids.as_of; never a held-out race), in log-odds, with
w = n_close_circuit / (n_close_circuit + k0); with no earlier race at all a fixed prior with a
wide uncertainty (tier 3). The judge's version (difficulty_judge) may also read the race itself
(hindsight: listed in fidelity's ledger).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .race_ids import as_of, canonical_race_id, parse_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED = ROOT / "data" / "derived"
TABLE = DERIVED / "overtaking.csv"
PAIRS = DERIVED / "overtaking_pairs.csv"
MANIFEST = DERIVED / "overtaking_manifest.json"
CACHE_DIR = ROOT / "cache"
EVAL_SET = ROOT / "data" / "eval_set.json"
COMMAND = "python -m src.sim.overtaking build"

CLOSE_GAP_S = 1.0        # passing range at the line (DRS detection gap)
DIRTY_RANGE_S = 1.5      # config.FREE_AIR_GAP_S: dirty air measured on laps under this gap
FREE_AIR_FIT_S = 2.0     # the pace fit reads laps at least this far behind the car ahead
MIN_STINT_FIT_LAPS = 4   # a driver-stint intercept needs this many free-air laps
ADV_CLIP_S = 3.0         # pace advantages are clipped to +-this in the threshold fit
MIN_RACING_LAPS = 10     # a race with fewer green racing laps (a red-flagged wash-out) is left out
SECTORS = ("S1", "S2", "S3")
# a priori, fitted on no race: the tier-3 prior (no earlier race) in log-odds of the per-close-lap
# pass rate, and the prior strength in close laps of a circuit's own evidence
PRIOR_RATE = 0.06
PRIOR_LOGIT_SD = 1.0
K0_CLOSE_LAPS = 150.0
# the ghost model's earned-pass geometry from a held car at the follow gap: 1.0 + 0.2 + 0.4 s
PRIOR_THETA_S = 1.6
PRIOR_SLOPE = 1.0
PRIOR_DIRTY_AIR_S = 0.4
# a priori DRS-train effect (no earlier train laps): a train's car ahead has DRS too
PRIOR_TRAIN_SHIFT_S = 1.5
PRIOR_TRAIN_SHARE = 0.4
NEVER = frozenset({"2019_06_monaco", "2020_16_sakhir", "2025_11_austria", "2024_11_austria"})


def held_out() -> set[str]:
    races = json.loads(EVAL_SET.read_text()).get("races", []) if EVAL_SET.is_file() else []
    return {canonical_race_id(r) for r in races} | set(NEVER)


def _refuse(race_id):
    if canonical_race_id(race_id) in held_out():
        raise ValueError(f"{race_id} is held out: never loaded by the overtaking table")


def _load(race_id: str) -> pd.DataFrame:
    """The race's lap rows (data._columns) with the cumulative sector session times."""
    _refuse(race_id)
    from .data import _columns, load_session
    season, rnd, _ = parse_race_id(race_id)
    laps = load_session(season, rnd, "R", cache_dir=str(CACHE_DIR)).laps
    df = _columns(laps)
    for k, col in (("s1_s", "Sector1SessionTime"), ("s2_s", "Sector2SessionTime")):
        df[k] = laps[col].dt.total_seconds().to_numpy() if col in laps else np.nan
    return df


def _pace_fit(df: pd.DataFrame) -> pd.Series:
    """Predicted free-air lap time per row (NaN where the driver-stint has too few free-air laps)."""
    ok = (df["track_status"] == "1") & ~df["pit_in"] & ~df["pit_out"] & df["lap_time_s"].notna() \
        & df["tyre_age"].notna() & (df["lap"] >= 2)
    med = df.loc[ok, "lap_time_s"].median()
    ok &= df["lap_time_s"] < 1.07 * med
    free = ok & (df["gap_ahead_s"].isna() | (df["gap_ahead_s"] >= FREE_AIR_FIT_S))
    key = df["driver"].astype(str) + "|" + df["stint"].astype(str)
    n = free.groupby(key).transform("sum")
    fit = free & (n >= MIN_STINT_FIT_LAPS)
    if fit.sum() < 30:
        return pd.Series(np.nan, index=df.index)
    keys = sorted(key[fit].unique())
    comps = sorted(df.loc[fit, "compound"].astype(str).unique())
    kidx = {k: i for i, k in enumerate(keys)}
    cidx = {c: i for i, c in enumerate(comps)}

    def design(rows):
        X = np.zeros((len(rows), len(keys) + len(comps) + 1))
        ki = key[rows.index].map(kidx).to_numpy()
        ci = rows["compound"].astype(str).map(cidx).to_numpy()
        X[np.arange(len(rows)), ki] = 1.0
        X[np.arange(len(rows)), len(keys) + ci] = rows["tyre_age"].to_numpy(float)
        X[:, -1] = rows["lap"].to_numpy(float)
        return X

    rows = df[fit]
    y = rows["lap_time_s"].to_numpy(float)
    X = design(rows)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(2):   # one robust pass: drop residuals beyond 2.5 MAD-sigma
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        keep = np.abs(r) < 2.5 * max(s, 0.05)
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    can = key.isin(kidx.keys()) & df["compound"].astype(str).isin(cidx.keys()) & df["tyre_age"].notna()
    out = pd.Series(np.nan, index=df.index)
    out[can] = design(df[can]) @ beta
    return out


def observe_race(race_id: str, df: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """(summary, close-lap rows) of one race; module docstring."""
    from .data import gap_columns
    race_id = canonical_race_id(race_id)
    df = _load(race_id) if df is None else df
    df = df.drop_duplicates(["driver", "lap"]).reset_index(drop=True)
    gap_columns(df)
    df["pred_s"] = _pace_fit(df)
    df["resid_s"] = df["lap_time_s"] - df["pred_s"]
    idx = df.set_index(["driver", "lap"])
    green = df["track_status"] == "1"
    clean = green & ~df["pit_in"] & ~df["pit_out"] & df["end_s"].notna() & (df["lap"] >= 2)

    def at(col, drivers, laps):
        return idx[col].reindex(pd.MultiIndex.from_arrays([drivers, laps])).to_numpy()

    # close laps: F within CLOSE_GAP_S of the car directly ahead, both clean on L, A green on L-1
    c = df[clean & df["car_ahead"].notna() & (df["gap_ahead_s"] <= CLOSE_GAP_S)].copy()
    a_drv, L = c["car_ahead"].to_numpy(), c["lap"].to_numpy()
    a_status = at("track_status", a_drv, L)
    a_prev_status = at("track_status", a_drv, L - 1)
    a_in, a_out = at("pit_in", a_drv, L), at("pit_out", a_drv, L)
    a_end = at("end_s", a_drv, L)
    ok = (a_status == "1") & (a_prev_status == "1") & (a_in == False) & (a_out == False) & np.isfinite(  # noqa: E712
        a_end.astype(float))
    c = c[ok].copy()
    a_drv, L = c["car_ahead"].to_numpy(), c["lap"].to_numpy()
    c["a_end_s"] = at("end_s", a_drv, L).astype(float)
    c["passed"] = c["end_s"].to_numpy(float) < c["a_end_s"].to_numpy()
    c["a_pred_s"] = at("pred_s", a_drv, L).astype(float)
    c["a_lap_s"] = at("lap_time_s", a_drv, L).astype(float)
    c["adv_s"] = c["a_pred_s"] - c["pred_s"]           # follower's free-air pace advantage
    # a DRS train: the car ahead is itself within CLOSE_GAP_S of its own car ahead
    c["a_gap_ahead_s"] = at("gap_ahead_s", a_drv, L).astype(float)
    c["train"] = c["a_gap_ahead_s"] <= CLOSE_GAP_S
    # sectors: where F first runs ahead of A on a pass lap
    sector = np.full(len(c), None, dtype=object)
    for col_f, name in (("s1_s", "S1"), ("s2_s", "S2")):
        mine = c[col_f].to_numpy(float)
        theirs = at(col_f, a_drv, L).astype(float)
        hit = c["passed"].to_numpy() & (sector == None) & np.isfinite(mine) & np.isfinite(theirs) & (mine < theirs)  # noqa: E711
        sector[hit] = name
    has_sec = np.isfinite(c["s1_s"].to_numpy(float)) & np.isfinite(at("s1_s", a_drv, L).astype(float)) \
        & np.isfinite(c["s2_s"].to_numpy(float)) & np.isfinite(at("s2_s", a_drv, L).astype(float))
    sector[c["passed"].to_numpy() & (sector == None) & has_sec] = "S3"  # noqa: E711
    c["sector"] = sector
    c["race_id"] = race_id

    # every order change on the same lap count, line L-1 to line L, both cars clean on L
    end = df[clean].pivot_table(index="lap", columns="driver", values="end_s", aggfunc="first")
    prev = df.pivot_table(index="lap", columns="driver", values="end_s", aggfunc="first")
    all_passes = 0
    racing_laps = 0
    neutral_field = df.groupby("lap")["track_status"].agg(lambda s: (s != "1").mean() > 0.5)
    for L_ in end.index:
        if L_ < 2 or L_ - 1 not in prev.index or neutral_field.get(L_, True) or neutral_field.get(L_ - 1, True):
            continue
        racing_laps += 1
        e1 = end.loc[L_].dropna()
        e0 = prev.loc[L_ - 1].reindex(e1.index).dropna()
        e1 = e1.reindex(e0.index)
        r0, r1 = e0.rank().to_numpy(), e1.rank().to_numpy()
        # pairs whose order flips
        all_passes += int(((r0[:, None] < r0[None, :]) & (r1[:, None] > r1[None, :])).sum())

    # dirty air: residual on close laps (gap < DIRTY_RANGE_S) less free-air laps
    base = clean & df["resid_s"].notna() & (df["resid_s"].abs() < 5)
    near = base & (df["gap_ahead_s"] < DIRTY_RANGE_S)
    far = base & (df["gap_ahead_s"] >= FREE_AIR_FIT_S)
    dirty = float(df.loc[near, "resid_s"].mean() - df.loc[far, "resid_s"].mean()) if near.sum() >= 20 and far.sum() >= 20 \
        else float("nan")
    n_close, n_pass = int(len(c)), int(c["passed"].sum())
    total_laps = int(df["lap"].max())
    secs = pd.Series(c.loc[c["passed"], "sector"]).value_counts()
    summary = {
        "race_id": race_id, "total_laps": total_laps, "racing_laps": racing_laps,
        "n_close": n_close, "n_pass": n_pass, "rate": n_pass / n_close if n_close else float("nan"),
        "all_passes": all_passes, "passes_per_lap": all_passes / racing_laps if racing_laps else float("nan"),
        "close_share": n_close / max(1, int(clean.sum())),
        "dirty_air_s": dirty, "n_dirty": int(near.sum()),
        "adv_median_s": float(c["adv_s"].median()) if c["adv_s"].notna().any() else float("nan"),
        **{f"pass_{s}": int(secs.get(s, 0)) for s in SECTORS},
    }
    cols = ["race_id", "driver", "lap", "car_ahead", "gap_ahead_s", "adv_s", "passed", "sector", "resid_s",
            "a_gap_ahead_s", "train"]
    return summary, c[cols].reset_index(drop=True)


# --- the logistic threshold ----------------------------------------------------------------------
def fit_theta(adv, passed, slope=None, prior_theta=None, prior_sd=None):
    """(theta, slope, se_theta) of P(pass) = sigmoid(slope * (adv - theta)), maximum likelihood
    (with an optional normal prior on theta); slope fixed when given. adv is clipped to
    [-ADV_CLIP_S, ADV_CLIP_S] (a pace fit's tail is not a pace advantage)."""
    from scipy.optimize import minimize
    x = np.asarray(adv, float)
    y = np.asarray(passed, float)
    m = np.isfinite(x)
    x, y = np.clip(x[m], -ADV_CLIP_S, ADV_CLIP_S), y[m]
    if len(x) < 10 or y.sum() < 2:
        return (float("nan"), float(slope) if slope else float("nan"), float("nan"))

    def nll(v):
        th = v[0]
        b = slope if slope is not None else math.exp(v[1])
        z = b * (x - th)
        out = np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
        if prior_theta is not None and prior_sd:
            out += 0.5 * ((th - prior_theta) / prior_sd) ** 2
        return out

    v0 = [1.0] if slope is not None else [1.0, math.log(2.0)]
    r = minimize(nll, v0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000})
    th = float(r.x[0])
    b = float(slope if slope is not None else math.exp(r.x[1]))
    p = 1.0 / (1.0 + np.exp(-b * (x - th)))
    info = float(np.sum(p * (1 - p))) * b * b + ((1.0 / prior_sd ** 2) if (prior_theta is not None and prior_sd) else 0.0)
    return th, b, float(1.0 / math.sqrt(info)) if info > 0 else float("nan")


# --- the history table ----------------------------------------------------------------------------
def cached_races() -> list[str]:
    from .tyre_history import cached_races as _cr
    held = held_out()
    return [r for r, _ in _cr() if r not in held]


def build(races=None, log=print) -> pd.DataFrame:
    from .tyre_history import race_info
    races = [canonical_race_id(r) for r in (races or cached_races())]
    for r in races:
        _refuse(r)
    rows, pairs = [], []
    for rid in sorted(races, key=lambda r: parse_race_id(r)[:2]):
        log(f"  {rid}")
        try:
            s, c = observe_race(rid)
        except Exception as e:   # noqa: BLE001 - a race that does not load is reported, not fatal
            log(f"    skipped: {e}")
            continue
        if s["racing_laps"] < MIN_RACING_LAPS:
            log(f"    left out: {s['racing_laps']} racing laps")
            continue
        info = race_info(rid)
        s.update(season=info.season, round=info.round, circuit=info.circuit)
        rows.append(s)
        pairs.append(c)
    table = pd.DataFrame(rows)
    pr = pd.concat(pairs, ignore_index=True)
    # the pooled slope, then theta per race with that slope (no prior)
    _, slope, _ = fit_theta(pr["adv_s"], pr["passed"])
    th = []
    for rid in table["race_id"]:
        sub = pr[pr["race_id"] == rid]
        t, _, se = fit_theta(sub["adv_s"], sub["passed"], slope=slope)
        th.append((t, se))
    table["theta_s"] = [t for t, _ in th]
    table["theta_se_s"] = [s for _, s in th]
    table["pooled_slope"] = slope
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False, float_format="%.6g")
    pr.to_csv(PAIRS, index=False, float_format="%.5g")
    MANIFEST.write_text(json.dumps({
        "command": COMMAND,
        "description": "Per-race on-track passing observations (src/sim/overtaking.py observe_race); read at run "
                       "time by difficulty_prior (strictly earlier races only) and difficulty_judge (the race too).",
        "races": list(table["race_id"]), "held_out_never_loaded": sorted(held_out()),
        "close_gap_s": CLOSE_GAP_S, "pooled_slope": slope,
    }, indent=1) + "\n")
    return table


@lru_cache(maxsize=1)
def load_table() -> pd.DataFrame:
    t = pd.read_csv(TABLE)
    t["race_id"] = t["race_id"].map(canonical_race_id)
    return t


@lru_cache(maxsize=1)
def load_pairs() -> pd.DataFrame:
    return pd.read_csv(PAIRS)


def _circuit(race_id):
    from .tyre_history import race_info
    return race_info(race_id).circuit


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _expit(z):
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class Difficulty:
    """The passing difficulty at one race: rate (per close lap pass probability), theta_s (pace
    advantage at which half the close laps end in a pass), dirty_air_s, their sds, tier (1 same
    circuit, 2 pooled only, 3 no earlier race), sources (race ids read)."""
    rate: float
    rate_lo: float
    rate_hi: float
    theta_s: float
    theta_sd_s: float
    dirty_air_s: float
    tier: int
    sources: tuple
    n_close_circuit: int
    slope: float = PRIOR_SLOPE
    # theta is the average over close laps; a car ahead in a DRS train raises it by train_shift_s
    # (train_share of close laps are in a train): free theta - shift * share, train + shift * (1 - share)
    train_shift_s: float = 0.0
    train_share: float = 0.0


@lru_cache(maxsize=256)
def _pooled_fit(race_ids: tuple) -> tuple[float, float]:
    """(theta, slope) of the logistic over the close laps of `race_ids` (the a priori pair with
    fewer than 200 usable close laps)."""
    pr = load_pairs()
    sub = pr[pr["race_id"].isin(race_ids)]
    if sub["adv_s"].notna().sum() < 200:
        return PRIOR_THETA_S, PRIOR_SLOPE
    th, b, _ = fit_theta(sub["adv_s"], sub["passed"])
    return th, b


@lru_cache(maxsize=256)
def _train_fit(race_ids: tuple) -> tuple[float, float]:
    """(shift, share): how much higher theta is when the car ahead is itself in a DRS train (within
    CLOSE_GAP_S of its own car ahead) than when it is not, at the pooled slope of these races, and
    the share of close laps in a train. The a priori pair with too few train laps."""
    pr = load_pairs()
    sub = pr[pr["race_id"].isin(race_ids) & pr["adv_s"].notna()]
    if "train" not in sub or (sub["train"] == True).sum() < 100:   # noqa: E712
        return PRIOR_TRAIN_SHIFT_S, PRIOR_TRAIN_SHARE
    _, slope = _pooled_fit(race_ids)
    tr = sub["train"].astype(bool)
    th_free = fit_theta(sub.loc[~tr, "adv_s"], sub.loc[~tr, "passed"], slope=slope)[0]
    th_train = fit_theta(sub.loc[tr, "adv_s"], sub.loc[tr, "passed"], slope=slope)[0]
    if not (np.isfinite(th_free) and np.isfinite(th_train)):
        return PRIOR_TRAIN_SHIFT_S, PRIOR_TRAIN_SHARE
    return float(th_train - th_free), float(tr.mean())


def _combine(rows: pd.DataFrame, circuit: str, k0=K0_CLOSE_LAPS) -> Difficulty:
    if not len(rows):
        z, sd = _logit(PRIOR_RATE), PRIOR_LOGIT_SD
        return Difficulty(PRIOR_RATE, _expit(z - 1.645 * sd), _expit(z + 1.645 * sd), PRIOR_THETA_S,
                          PRIOR_LOGIT_SD / PRIOR_SLOPE, PRIOR_DIRTY_AIR_S, 3, (), 0, PRIOR_SLOPE,
                          PRIOR_TRAIN_SHIFT_S, PRIOR_TRAIN_SHARE)
    pooled_rate = rows["n_pass"].sum() / max(1, rows["n_close"].sum())
    # race-level spread of the log-odds around the pooled rate (between-race sd)
    zr = np.array([_logit((p + 0.5) / (n + 1)) for p, n in zip(rows["n_pass"], rows["n_close"])])
    z_pool = _logit(pooled_rate)
    sd_between = float(np.std(zr, ddof=1)) if len(zr) > 2 else PRIOR_LOGIT_SD
    loc = rows[rows["circuit"] == circuit]
    n_loc = int(loc["n_close"].sum())
    w = n_loc / (n_loc + k0)
    if n_loc:
        z_loc = _logit((loc["n_pass"].sum() + 0.5) / (n_loc + 1))
        z = w * z_loc + (1 - w) * z_pool
        sd = math.sqrt((1 - w) * sd_between ** 2 + w * (1.0 / max(1.0, loc["n_pass"].sum() + 0.5)))
        tier = 1
    else:
        z, sd, tier = z_pool, sd_between, 2
    # the threshold from the rate: the pooled logistic of these races' close laps (theta_pool, slope),
    # moved by the circuit's log-odds against the pooled: theta = theta_pool + (z_pool - z) / slope
    th_pool, slope = _pooled_fit(tuple(sorted(rows["race_id"])))
    theta = th_pool + (z_pool - z) / slope
    th_sd = sd / slope
    d_pool = float(np.nanmedian(rows["dirty_air_s"]))
    d_loc = float(np.nanmean(loc["dirty_air_s"])) if n_loc and loc["dirty_air_s"].notna().any() else d_pool
    dirty = w * d_loc + (1 - w) * d_pool
    shift, share = _train_fit(tuple(sorted(rows["race_id"])))
    return Difficulty(_expit(z), _expit(z - 1.645 * sd), _expit(z + 1.645 * sd), theta, th_sd, dirty, tier,
                      tuple(rows["race_id"]), n_loc, slope, shift, share)


def difficulty_prior(race_id: str, table: pd.DataFrame | None = None) -> Difficulty:
    """As of race_id: only races strictly earlier, never a held-out one."""
    t = load_table() if table is None else table
    held = held_out()
    earlier = set(as_of(race_id, [r for r in t["race_id"] if r not in held]))
    return _combine(t[t["race_id"].isin(earlier)], _circuit(race_id))


def difficulty_judge(race_id: str, table: pd.DataFrame | None = None) -> Difficulty:
    """The judge's: every table race but the held-out ones, the race itself included (hindsight)."""
    t = load_table() if table is None else table
    held = held_out()
    return _combine(t[~t["race_id"].isin(held)], _circuit(race_id))


def difficulty_race(race_id: str, table: pd.DataFrame | None = None) -> Difficulty:
    """The judge's, from the race's own count only (shrunk toward the pooled one)."""
    t = load_table() if table is None else table
    t = t[~t["race_id"].isin(held_out())].copy()
    t["circuit"] = t["race_id"]
    return _combine(t, canonical_race_id(race_id))


@lru_cache(maxsize=256)
def difficulty_for(race_id, mode: str) -> Difficulty:
    """traffic.GhostParams.difficulty's lookup: 'asof' (policy side), 'circuit' or 'race' (judge).
    A race without an id takes the tier-3 prior. The table never holds a held-out race (build
    refuses them), so pricing one reads only the other races: as of it ('asof'), or the circuit's
    other races ('circuit'); 'race' then has no own count and takes the pooled value."""
    try:
        race_id = canonical_race_id(race_id)
        parse_race_id(race_id)
    except (TypeError, ValueError, AttributeError):   # no id, or not a season_round_slug id
        return _combine(pd.DataFrame(), "")
    if mode == "asof":
        return difficulty_prior(race_id)
    if mode == "circuit":
        return difficulty_judge(race_id)
    if mode == "race":
        return difficulty_race(race_id)
    raise ValueError(f"unknown difficulty mode {mode!r}")


def races_for(race_id: str) -> list[str]:
    """What difficulty_prior reads when pricing race_id (provenance)."""
    if not TABLE.is_file():
        return []
    t = load_table()
    return as_of(race_id, [r for r in t["race_id"] if r not in held_out()])


def recorded() -> list[str]:
    return list(load_table()["race_id"]) if TABLE.is_file() else []


# --- validation -------------------------------------------------------------------------------------
def validate(out_dir=None, log=print) -> pd.DataFrame:
    """Does the as-of / leave-one-race-out circuit value predict a race's own pass rate better than a
    pooled constant? Binomial log-loss per close lap and absolute log-odds error, per race."""
    t = load_table()
    rows = []
    for rid in t["race_id"]:
        me = t[t["race_id"] == rid].iloc[0]
        circ = me["circuit"]
        rest = t[t["race_id"] != rid]
        earlier = t[t["race_id"].isin(as_of(rid, list(t["race_id"])))]
        preds = {
            "pooled_loro": rest["n_pass"].sum() / rest["n_close"].sum(),
            "circuit_loro": _combine(rest, circ).rate,
            "pooled_asof": (earlier["n_pass"].sum() / earlier["n_close"].sum()) if len(earlier) else PRIOR_RATE,
            "circuit_asof": _combine(earlier, circ).rate,
        }
        has_loc_loro = bool((rest["circuit"] == circ).any())
        has_loc_asof = bool((earlier["circuit"] == circ).any())
        k, n = int(me["n_pass"]), int(me["n_close"])
        r = {"race_id": rid, "circuit": circ, "n_close": n, "n_pass": k, "rate": k / n if n else np.nan,
             "has_circuit_loro": has_loc_loro, "has_circuit_asof": has_loc_asof}
        for name, p in preds.items():
            p = min(max(p, 1e-4), 1 - 1e-4)
            r[f"{name}_rate"] = p
            r[f"{name}_logloss"] = -(k * math.log(p) + (n - k) * math.log(1 - p)) / max(n, 1)
            r[f"{name}_abs_logit_err"] = abs(_logit((k + 0.5) / (n + 1)) - _logit(p))
        rows.append(r)
    v = pd.DataFrame(rows)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        v.to_csv(Path(out_dir) / "validation.csv", index=False, float_format="%.5g")
    for scope, col in (("loro", "has_circuit_loro"), ("asof", "has_circuit_asof")):
        sub = v[v[col]]
        log(f"{scope}: races with an earlier/other same-circuit race {len(sub)}; "
            f"logloss pooled {sub[f'pooled_{scope}_logloss'].mean():.4f} circuit {sub[f'circuit_{scope}_logloss'].mean():.4f}; "
            f"|logit err| pooled {sub[f'pooled_{scope}_abs_logit_err'].mean():.3f} "
            f"circuit {sub[f'circuit_{scope}_abs_logit_err'].mean():.3f}")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.sim.overtaking")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--races", nargs="*")
    v = sub.add_parser("validate")
    v.add_argument("--out", default=str(ROOT / "reports" / "calibration" / "overtaking"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        t = build(a.races)
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(t.drop(columns=["pooled_slope"]).to_string(index=False))
    else:
        validate(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
