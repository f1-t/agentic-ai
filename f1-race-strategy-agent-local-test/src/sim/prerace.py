"""The tyre supplier's pre-race publications, read by race id (docs/eval.md "Information contract").

Two committed tables, one row per race weekend, both published BEFORE the race (data/derived/README.md):

  data/derived/circuit_ratings.csv    the supplier's circuit ratings from its race-week preview: 1-5 for
                                      traction, braking, lateral, tyre stress, asphalt grip, asphalt
                                      abrasion, track evolution, downforce; minimum starting pressures
                                      and camber limits where given
  data/derived/prerace_guidance.csv   the supplier's pre-race strategy guidance (after qualifying):
                                      the expected fastest strategies in order (stops, compound
                                      sequence, pit-window laps), stated pace gaps, a degradation note

They are the same kind of information as the tyre nominations (data/derived/tyre_nominations.csv):
public before the race, saying nothing about what happened in it, so a row exists for every
weekend, held-out ones too. A blank row (nothing found) is normal; every reader returns None then,
and every caller falls back to what it did without the tables.

What reads them (each a switch, SimConfig.prerace, off by default):
  tyre_life_prior(..., ratings=...)   tyre_history: the circuit ratings as covariates of tyre life,
                                      fitted as of the race on earlier races (life_severity_fit)
  tyre_life_prior(..., guidance=...)  tyre_history: the guidance's implied stint lengths as a
                                      second tyre-life estimate, combined by precision with the
                                      historical one (guidance_life, guidance_error_as_of)
  agent_tools.prerace_block           the ratings in words and the expected strategies, in the
                                      agent's race_picture under AGENT_PRERACE (words kept there, in a
                                      non-physics module)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .race_ids import canonical_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED = ROOT / "data" / "derived"
RATINGS = DERIVED / "circuit_ratings.csv"
GUIDANCE = DERIVED / "prerace_guidance.csv"

RATING_KEYS = ("traction", "braking", "lateral", "tyre_stress", "asphalt_grip", "asphalt_abrasion",
               "track_evolution", "downforce")
# the covariates of tyre life (stated a priori: the categories that describe load on the tyre)
LIFE_COVARIATES = ("asphalt_abrasion", "tyre_stress", "lateral", "traction")
ROLES = ("HARD", "MEDIUM", "SOFT")
MAX_STRATEGIES = 3

# a priori constants of the ratings fit and the guidance prior (stated, not fitted)
RIDGE_LAMBDA = 4.0            # ridge penalty, in observations, on the covariates (rating points)
MIN_RATED_RACES = 6           # fewer earlier races with ratings: no ratings fit (the trio shift alone)
GUIDANCE_LOG_SD = 0.25        # sd of log(observed edge / guidance stint) with too few earlier races
GUIDANCE_MIN_RACES = 3        # earlier races with both guidance and an observed edge to estimate it
GUIDANCE_LOG_SD_FLOOR = 0.10

_TABLES: dict = {}


def _key(path) -> tuple:
    p = Path(path)
    st = p.stat()
    return str(p.resolve()), st.st_mtime_ns, st.st_size


def load_ratings(path=RATINGS) -> pd.DataFrame:
    """The ratings table indexed by canonical race id (empty when the file is absent)."""
    if not Path(path).is_file():
        return pd.DataFrame(columns=list(RATING_KEYS)).rename_axis("race_id")
    k = ("ratings", _key(path))
    if k not in _TABLES:
        t = pd.read_csv(path, dtype={"race_id": str})
        t["race_id"] = t["race_id"].map(canonical_race_id)
        for c in RATING_KEYS:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        _TABLES[k] = t.set_index("race_id")
    return _TABLES[k]


def load_guidance(path=GUIDANCE) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame().rename_axis("race_id")
    k = ("guidance", _key(path))
    if k not in _TABLES:
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
        t["race_id"] = t["race_id"].map(canonical_race_id)
        _TABLES[k] = t.set_index("race_id")
    return _TABLES[k]


def ratings_of(race_id, keys=RATING_KEYS) -> dict | None:
    """{category: rating} for `race_id`, None unless every one of `keys` is known."""
    t = load_ratings()
    rid = _rid(race_id)
    if rid is None or rid not in t.index:
        return None
    r = t.loc[rid]
    out = {c: float(r[c]) for c in keys}
    return out if all(np.isfinite(v) for v in out.values()) else None


def _rid(race_id):
    try:
        return canonical_race_id(race_id)
    except (ValueError, TypeError):
        return None


def _windows(text: str) -> list[tuple[float, float]] | None:
    """'18-24;40-46' -> [(18, 24), (40, 46)]; a single lap '25' -> (25, 25); None if any is blank."""
    out = []
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            return None
        lo, _, hi = part.partition("-")
        try:
            a, b = float(lo), float(hi or lo)
        except ValueError:
            return None
        out.append((min(a, b), max(a, b)))
    return out


def strategies_of(race_id) -> list[dict]:
    """The guidance's expected strategies for `race_id`, fastest first: {rank, stops, sequence
    (list of roles), windows (list of (lo, hi) per stop, or None when no laps were stated)}; []
    when the race has no guidance."""
    t = load_guidance()
    rid = _rid(race_id)
    if rid is None or rid not in t.index:
        return []
    r = t.loc[rid]
    out = []
    for i in range(1, MAX_STRATEGIES + 1):
        seq = str(r.get(f"s{i}_sequence", "") or "").strip().upper()
        roles = [s.strip() for s in seq.split("-") if s.strip()]
        if not roles or any(x not in ROLES for x in roles):
            roles = None      # not stated in full (blank, "MEDIUM/HARD mix", "?-?-SOFT")
        try:
            stops = int(float(r.get(f"s{i}_stops")))
        except (TypeError, ValueError):
            stops = len(roles) - 1 if roles else None
        if stops is None:
            continue
        if roles is not None and len(roles) != stops + 1:
            roles = None
        w = _windows(r.get(f"s{i}_windows", "")) if roles else None
        if w is not None and len(w) != len(roles) - 1:
            w = None
        out.append({"rank": i, "stops": stops, "sequence": roles, "windows": w})
    return out


def expected_stops(race_id) -> int | None:
    """The stop count of the strategy the guidance calls fastest (its first), None without one."""
    s = strategies_of(race_id)
    return s[0]["stops"] if s else None


# the share of a stated strategy's distance each role runs when no laps are stated ("stated" mode):
# proportional to a role's life weight, one life-ratio step (0.75, configs/tyre_history.yaml's
# default_per_step) per role step; stated a priori
ROLE_LIFE_WEIGHT = {"HARD": 1.0, "MEDIUM": 0.75, "SOFT": 0.5625}


def guidance_life(race_id, total_laps: float, mode: str = "windows") -> dict[str, float]:
    """role -> the longest stint the guidance expects on it, in laps.
    windows  per strategy with stated windows: a first stint to the upper end of the first window,
             a middle stint from the lower end of its opening window to the upper end of its closing
             one, the last stint from the lower end of the last window to the flag (the upper end of
             a window is the supplier's estimate of how long the compound runs)
    stated   also the strategy called fastest when it states a full compound sequence but no laps:
             the distance shared among its stints by ROLE_LIFE_WEIGHT (a one-stop MEDIUM-HARD gives
             the HARD 57% of the race), a priori
    {} with no usable guidance."""
    out: dict[str, float] = {}
    strategies = strategies_of(race_id)
    for s in strategies:
        w, roles = s["windows"], s["sequence"]
        if not roles:
            continue
        if w:
            for j, role in enumerate(roles):
                start = 0.0 if j == 0 else w[j - 1][0]
                end = float(total_laps) if j == len(roles) - 1 else w[j][1]
                if end > start:
                    out[role] = max(out.get(role, 0.0), end - start)
        elif mode == "stated" and s["rank"] == 1:
            tot = sum(ROLE_LIFE_WEIGHT[x] for x in roles)
            for role in roles:
                out[role] = max(out.get(role, 0.0), float(total_laps) * ROLE_LIFE_WEIGHT[role] / tot)
    return out


def guidance_error_as_of(earlier_obs: pd.DataFrame, mode: str = "windows") -> tuple[float, float, int]:
    """(bias, sd, races) of log(observed edge / guidance stint) over `earlier_obs` (usable tyre
    history rows of races strictly before the target, already as-of): the mean and sd over the
    races holding both; (0, GUIDANCE_LOG_SD, n) under GUIDANCE_MIN_RACES races."""
    logs, races = [], set()
    for r in earlier_obs.itertuples():
        g = guidance_life(r.race_id, float(r.total_laps), mode).get(r.role)
        e = float(r.edge_laps) if pd.notna(r.edge_laps) else float("nan")
        if g and g > 0 and np.isfinite(e) and e > 0:
            logs.append(math.log(e / g))
            races.add(r.race_id)
    if len(races) < GUIDANCE_MIN_RACES:
        return 0.0, GUIDANCE_LOG_SD, len(races)
    x = np.array(logs)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else GUIDANCE_LOG_SD
    # shrink the bias toward 0 by the races behind it (a priori: worth GUIDANCE_MIN_RACES races)
    n = len(races)
    bias = float(np.mean(x)) * n / (n + GUIDANCE_MIN_RACES)
    return bias, max(GUIDANCE_LOG_SD_FLOOR, sd), n


def life_severity_fit(rows: list[tuple[str, float]], lam: float = RIDGE_LAMBDA,
                      min_races: int = MIN_RATED_RACES) -> dict | None:
    """A ridge fit of a compound-normalised log tyre life on the circuit ratings: rows are
    (race_id, y) for earlier observations (y = log edge share moved to a reference compound by the
    life ratio; under 'supplement' also by the trio shift). Returns {beta: {cov: coef}, races: n}
    (log life per rating point, centred on the rows' means) or None under min_races rated races."""
    xs, ys, races = [], [], set()
    for rid, y in rows:
        x = ratings_of(rid, LIFE_COVARIATES)
        if x is None or not np.isfinite(y):
            continue
        xs.append([x[c] for c in LIFE_COVARIATES])
        ys.append(y)
        races.add(rid)
    if len(races) < min_races:
        return None
    X, Y = np.array(xs), np.array(ys)
    Xc, Yc = X - X.mean(0), Y - Y.mean()
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    return {"beta": dict(zip(LIFE_COVARIATES, map(float, beta))), "races": len(races)}


def ratings_shift(fit: dict | None, target_id, obs_id) -> float:
    """log-life shift moving an observation at `obs_id` onto `target_id` under `fit`; 0 when
    either race lacks ratings or there is no fit."""
    if not fit:
        return 0.0
    xt, xh = ratings_of(target_id, LIFE_COVARIATES), ratings_of(obs_id, LIFE_COVARIATES)
    if xt is None or xh is None:
        return 0.0
    return float(sum(fit["beta"][c] * (xt[c] - xh[c]) for c in LIFE_COVARIATES))
