"""SC and VSC rates by circuit and era (SPEC 0011b), the "event_rates" calibrator.

    python -m src.sim.calibrate rates [--periods PATH] [--factors PATH] [--race-features PATH] [--out DIR]
    python -m src.sim.calibrate rates --synthetic --out /tmp/dir     # the synthetic test report

Input: data/derived/neutralised_periods.csv (race_id, kind sc/vsc/red_flag, start_lap,
end_lap, total_laps, wet; a race with no period has one row with an empty kind) when it
exists, else the SC/VSC periods of 0016's factors.csv (reports/calibration/cross_race/,
dry races by construction). Wet races are left out of the dry rates (include_wet=False) and
counted apart. Circuit and circuit type come from race_features.csv (columns circuit,
circuit_type) when it exists, else CIRCUITS, keyed by the race id's GP slug.

Model, per kind (sc, vsc): starts ~ Poisson(p x laps_at_risk) per race, laps_at_risk the
total laps less the neutralised ones (any kind, red flags included), and period lengths
geometric with mean d. p is shrunk empirical-Bayes (Gamma-Poisson) along
global -> era (2018-2021, 2022+) -> circuit type (street / permanent) -> circuit. A node's
estimate is (N + E0 m) / (L + E0): N starts over L laps at risk, m its parent's estimate,
E0 = k x (mean laps at risk per race), k the level's prior strength in races. k comes from the
method of moments over the level's nodes (between-node variance of N/L around the parent
less the Poisson part), floored a priori at K0 = 5 races; no excess variance is full pooling.
The global level is the pooled MLE; with no data it is the EventModel prior. d is pooled per
kind and era only (global when the era has no complete period, the prior with none at all).

Durations: a period that ends on the race's last lap is right-censored; it says only that
the length was at least what was seen, so it adds P(D >= l) to the likelihood and its laps
but not a completed period to the d estimate (MLE d = 1 + (laps - periods) / completed).
factors.csv has no period ends, so its periods count as complete. Back-to-back periods of
one kind (one ending and the next starting on the following lap) cannot be told apart in
the lap data and are taken as one period; this is ignored.

Lap-1 term (candidate circuit_lap1): p_sc_lap1 / p_vsc_lap1 (config.EventModel) are the share
of races with a period of that kind starting on lap 1; p is then fitted on laps 2 onward.

As-of: EventRatesCalibrator gives a race the rates fitted on races dated before it
((season, round) of the race id), never itself.

Selection, stated before any data, on as-of folds (forward chaining): each dry race from
FORWARD_FROM_SEASON on is scored under the rates fitted on the dry races dated strictly
before it, as the calibrator is used. Leave-one-race-out (every other race) is reported
beside it and does not select. The score is the held-out negative log-likelihood of the
race's period starts (a Poisson process per lap at risk; lap 1 a Bernoulli under the lap-1
term) and of each period's length (geometric under the candidate's d, censored as above).
Candidates global, era, circuit_type, circuit, circuit_lap1; a candidate replaces global when
the race-cluster 95 percent CI of the paired NLL difference (candidate less global) is below
0. Among those that pass the lowest mean difference wins; ties go to the simpler
(CANDIDATES order).

Output: rates.csv (each race's as-of rates per candidate), forward_nll.csv, loro_nll.csv,
selection.csv (both schemes, the forward one decides), strengths.csv and summary.md under
OUT_DIR. A report with a synthetic race id refuses any
directory under reports/calibration/ (guard_committed).
"""
import argparse
import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import RaceParams, Snapshot
from .calibrators import CALIBRATORS, Calibrator
from .config import EventModel
from .race_ids import canonical_race_id, parse_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
NEUTRALISED_PERIODS = ROOT / "data" / "derived" / "neutralised_periods.csv"
FACTORS = ROOT / "reports" / "calibration" / "cross_race" / "factors.csv"
RACE_FEATURES = ROOT / "data" / "race_features.csv"     # same file cross_race reads
REPORTS_DIR = ROOT / "reports" / "calibration"
OUT_DIR = REPORTS_DIR / "event_rates"

KINDS = ("sc", "vsc")
PERIOD_KINDS = ("sc", "vsc", "red_flag")
ERAS = ("2018-2021", "2022+")
LEVELS = ("global", "era", "circuit_type", "circuit")
CANDIDATES = ("global", "era", "circuit_type", "circuit", "circuit_lap1")   # simplest first
LAP1 = "circuit_lap1"
K0 = 5.0              # a priori floor on each level's prior strength, in races
N_BOOT = 2000
UNKNOWN = "unknown"

STREET, PERMANENT = "street", "permanent"
# GP slug of the race id (race_ids.slug of the --gp name) -> (circuit, circuit type), for the
# circuits of the 2018 onward calendars. Street: a temporary or public-road circuit.
CIRCUITS = {
    "australia": ("albert_park", STREET), "melbourne": ("albert_park", STREET),
    "bahrain": ("sakhir", PERMANENT), "sakhir": ("sakhir_outer", PERMANENT),
    "china": ("shanghai", PERMANENT),
    "azerbaijan": ("baku", STREET), "baku": ("baku", STREET),
    "spain": ("catalunya", PERMANENT), "barcelona": ("catalunya", PERMANENT),
    "monaco": ("monaco", STREET),
    "canada": ("villeneuve", STREET), "montreal": ("villeneuve", STREET),
    "france": ("paul_ricard", PERMANENT),
    "austria": ("red_bull_ring", PERMANENT), "styria": ("red_bull_ring", PERMANENT),
    "great_britain": ("silverstone", PERMANENT), "british": ("silverstone", PERMANENT),
    "britain": ("silverstone", PERMANENT), "70th_anniversary": ("silverstone", PERMANENT),
    "germany": ("hockenheim", PERMANENT), "eifel": ("nurburgring", PERMANENT),
    "hungary": ("hungaroring", PERMANENT),
    "belgium": ("spa", PERMANENT),
    "netherlands": ("zandvoort", PERMANENT), "dutch": ("zandvoort", PERMANENT),
    "italy": ("monza", PERMANENT), "monza": ("monza", PERMANENT),
    "tuscany": ("mugello", PERMANENT),
    "emilia_romagna": ("imola", PERMANENT), "imola": ("imola", PERMANENT),
    "portugal": ("portimao", PERMANENT), "turkey": ("istanbul_park", PERMANENT),
    "russia": ("sochi", PERMANENT),
    "singapore": ("marina_bay", STREET),
    "japan": ("suzuka", PERMANENT), "suzuka": ("suzuka", PERMANENT),
    "qatar": ("lusail", PERMANENT),
    "united_states": ("cota", PERMANENT), "usa": ("cota", PERMANENT), "austin": ("cota", PERMANENT),
    "mexico": ("rodriguez", PERMANENT), "mexico_city": ("rodriguez", PERMANENT),
    "brazil": ("interlagos", PERMANENT), "sao_paulo": ("interlagos", PERMANENT),
    "saudi_arabia": ("jeddah", STREET), "jeddah": ("jeddah", STREET),
    "miami": ("miami", STREET),
    "las_vegas": ("las_vegas", STREET),
    "abu_dhabi": ("yas_marina", PERMANENT),
}

STAT_COLUMNS = ["race_id", "season", "round", "era", "circuit", "circuit_type", "total_laps", "wet",
                "neutralised_laps", "at_risk", "n_red_flag",
                *[f"{c}_{k}" for k in KINDS for c in ("n", "laps", "lap1", "cens")]]
FORWARD_FROM_SEASON = 2019   # forward chaining scores races from this season on


def era(season: int) -> str:
    return ERAS[1] if int(season) >= 2022 else ERAS[0]


def _date_key(race_id: str) -> tuple[int, int]:
    season, rnd, _ = parse_race_id(race_id)
    return season, rnd


# --- input ---------------------------------------------------------------------------------
def circuit_info(race_ids, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """race_id, circuit, circuit_type per race: from `features` (race_features.csv columns
    circuit and circuit_type) where it has them, else CIRCUITS by GP slug, else the slug as
    the circuit with type UNKNOWN."""
    rows = []
    for rid in race_ids:
        slug = parse_race_id(rid)[2]
        circuit, ctype = CIRCUITS.get(slug, (slug, UNKNOWN))
        rows.append({"race_id": rid, "circuit": circuit, "circuit_type": ctype})
    out = pd.DataFrame(rows, columns=["race_id", "circuit", "circuit_type"])
    if features is not None and len(features) and "race_id" in features:
        f = features.assign(race_id=features["race_id"].map(canonical_race_id)).drop_duplicates("race_id")
        f = f.set_index("race_id")
        for col in ("circuit", "circuit_type"):
            if col in f:
                given = out["race_id"].map(f[col])
                ok = given.notna() & (given.astype(str).str.strip() != "")
                out.loc[ok, col] = given[ok].astype(str).str.strip().str.lower()
    return out


def _truthy(v) -> bool:
    """A wet flag as read from csv: bool, 0/1, or true/yes/wet strings; blank is dry."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "wet")
    return bool(v)


def _stats_frame(rows, features) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=STAT_COLUMNS)
    t = pd.DataFrame(rows)
    keys = [_date_key(r) for r in t["race_id"]]
    t["season"], t["round"] = [k[0] for k in keys], [k[1] for k in keys]
    t["era"] = t["season"].map(era)
    t = t.merge(circuit_info(list(t["race_id"]), features), on="race_id", how="left")
    t["at_risk"] = t["total_laps"] - t["neutralised_laps"]
    order = np.lexsort((t["round"].to_numpy(), t["season"].to_numpy()))
    return t.iloc[order][STAT_COLUMNS].reset_index(drop=True)


def stats_from_periods(periods: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-race stats (STAT_COLUMNS) from a neutralised_periods table: one row per period
    (race_id, kind, start_lap, end_lap, total_laps, wet), a race with no period as one row
    with an empty kind. Laps inclusive."""
    need = {"race_id", "kind", "start_lap", "end_lap", "total_laps"}
    missing = need - set(periods.columns)
    if missing:
        raise ValueError(f"neutralised periods table lacks columns {sorted(missing)}")
    p = periods.assign(race_id=periods["race_id"].map(canonical_race_id))
    kinds = p["kind"].astype("string").str.strip().str.lower().fillna("")
    bad = sorted(set(kinds) - {"", *PERIOD_KINDS})
    if bad:
        raise ValueError(f"unknown period kinds {bad}; expected {list(PERIOD_KINDS)}")
    p = p.assign(kind=kinds.to_numpy())
    rows = []
    for rid, g in p.groupby("race_id", sort=False):
        total = int(g["total_laps"].iloc[0])
        wet = any(_truthy(v) for v in g["wet"]) if "wet" in g else False
        real = g[g["kind"] != ""]
        laps = set()
        row = {"race_id": rid, "total_laps": total, "wet": wet,
               "n_red_flag": int((real["kind"] == "red_flag").sum())}
        for kind in KINDS:
            k = real[real["kind"] == kind]
            row[f"n_{kind}"] = int(len(k))
            row[f"laps_{kind}"] = int((k["end_lap"] - k["start_lap"] + 1).sum()) if len(k) else 0
            row[f"lap1_{kind}"] = int((k["start_lap"] == 1).any())
            row[f"cens_{kind}"] = int((k["end_lap"] >= total).sum())   # still running at the flag
        for s, e in zip(real["start_lap"], real["end_lap"]):
            laps |= set(range(max(1, int(s)), min(total, int(e)) + 1))
        row["neutralised_laps"] = len(laps)
        rows.append(row)
    return _stats_frame(rows, features)


def stats_from_factors(factors: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-race stats from 0016's factors.csv (sc_count, sc_laps, vsc_count, vsc_laps,
    sc_start_laps, vsc_start_laps). Its races are dry (cross-race eligibility) unless a wet or
    feature:wet column says otherwise; it has no red flags."""
    rows = []
    wet_col = next((c for c in ("wet", "feature:wet") if c in factors), None)
    for _, r in factors.iterrows():
        row = {"race_id": canonical_race_id(r["race_id"]), "total_laps": int(r["total_laps"]), "n_red_flag": 0,
               "wet": _truthy(r[wet_col]) if wet_col else False}
        for kind in KINDS:
            n = r.get(f"{kind}_count", 0)
            laps = r.get(f"{kind}_laps", 0)
            starts = r.get(f"{kind}_start_laps", "")
            starts = [] if not isinstance(starts, str) else [int(float(x)) for x in starts.split()]
            row[f"n_{kind}"] = int(n) if pd.notna(n) else 0
            row[f"laps_{kind}"] = int(laps) if pd.notna(laps) else 0
            row[f"lap1_{kind}"] = int(1 in starts)
            row[f"cens_{kind}"] = 0   # no period ends in factors.csv: every period counts as complete
        row["neutralised_laps"] = row["laps_sc"] + row["laps_vsc"]
        rows.append(row)
    return _stats_frame(rows, features)


def load_stats(periods_path=NEUTRALISED_PERIODS, factors_path=FACTORS, features_path=RACE_FEATURES):
    """(stats, source): from the neutralised periods table when it exists, else factors.csv,
    else no races."""
    features = pd.read_csv(features_path) if features_path and Path(features_path).is_file() else None
    if periods_path and Path(periods_path).is_file():
        return stats_from_periods(pd.read_csv(periods_path), features), str(periods_path)
    if factors_path and Path(factors_path).is_file():
        return stats_from_factors(pd.read_csv(factors_path), features), str(factors_path)
    return pd.DataFrame(columns=STAT_COLUMNS), "none"


# --- fit -----------------------------------------------------------------------------------
def _prior(prior: EventModel | None):
    return prior if prior is not None else EventModel()


@dataclass
class KindFit:
    """One kind's shrunk start rates per node and its pooled durations."""
    kind: str
    rates: dict = field(default_factory=dict)       # level -> {node key: p}
    strength: dict = field(default_factory=dict)    # level -> k in races (inf: full pooling)
    d_global: float = float("nan")
    d_era: dict = field(default_factory=dict)
    lap1: float | None = None                       # share of races with a lap-1 start
    n_races: int = 0

    def p(self, level: str, row) -> float:
        """The node's rate at `level` for a race (its era, circuit type, circuit), falling
        back to the nearest ancestor the training data has."""
        keys = _node_keys(row)
        out = self.rates["global"][()]
        for lv in LEVELS[1:LEVELS.index(level) + 1]:
            out = self.rates[lv].get(keys[lv], out)
        return out

    def d(self, level: str, row) -> float:
        return self.d_global if level == "global" else self.d_era.get(row["era"], self.d_global)


def _node_keys(row) -> dict:
    e, t, c = row["era"], row["circuit_type"], row["circuit"]
    return {"global": (), "era": (e,), "circuit_type": (e, t), "circuit": (e, t, c)}


def _strength(groups: pd.DataFrame, mean_laps: float, k0: float) -> float:
    """Prior strength in races at one level: method of moments over its nodes (columns N, L,
    parent), floored at k0; inf when the nodes vary no more than Poisson noise."""
    g = groups[groups["L"] > 0]
    if not len(g) or mean_laps <= 0:
        return math.inf
    r = g["N"] / g["L"]
    excess = float(((r - g["parent"]) ** 2 - g["parent"] / g["L"]).mean())
    m = float(g["parent"].mean())
    if excess <= 0 or m <= 0:
        return math.inf
    return max(k0, (m / excess) / mean_laps)


def _d_mle(laps: float, n: float, complete: float, fallback):
    """Geometric mean length from n periods over `laps` laps, `complete` of them not censored:
    1 + (laps - n) / complete; `fallback` without a complete period."""
    return 1.0 + float(laps - n) / float(complete) if complete > 0 else fallback


def fit_kind(train: pd.DataFrame, kind: str, prior: EventModel | None = None, k0: float = K0,
             lap1: bool = False) -> KindFit:
    """Shrunk rates of one kind on `train` (stats rows). With `lap1`, p is fitted on laps 2
    onward (a lap-1 start not counted, lap 1 not at risk) and the lap-1 share is kept."""
    prior = _prior(prior)
    p0, d0 = (prior.p_sc, prior.d_sc) if kind == "sc" else (prior.p_vsc, prior.d_vsc)
    out = KindFit(kind=kind, n_races=len(train))
    t = pd.DataFrame({"era": train["era"].to_numpy(), "circuit_type": train["circuit_type"].to_numpy(),
                      "circuit": train["circuit"].to_numpy(),
                      "N": (train[f"n_{kind}"] - (train[f"lap1_{kind}"] if lap1 else 0)).to_numpy(dtype=float),
                      "L": (train["at_risk"] - (1 if lap1 else 0)).to_numpy(dtype=float),
                      "D": train[f"laps_{kind}"].to_numpy(dtype=float),
                      "n": train[f"n_{kind}"].to_numpy(dtype=float),
                      "u": (train[f"n_{kind}"] - train[f"cens_{kind}"]).to_numpy(dtype=float)})
    total_l = float(t["L"].sum())
    m0 = float(t["N"].sum()) / total_l if len(t) and total_l > 0 else p0
    out.rates["global"] = {(): m0}
    out.strength["global"] = math.nan
    mean_laps = total_l / len(t) if len(t) else 0.0
    parents = {(): m0}
    cols = []
    for level in LEVELS[1:]:
        cols = cols + [level]
        g = t.groupby(cols, sort=True)[["N", "L"]].sum().reset_index()
        keys = [tuple(r) for r in g[cols].itertuples(index=False, name=None)]
        g["parent"] = [parents[k[:-1]] for k in keys]
        k = _strength(g, mean_laps, k0)
        out.strength[level] = k
        e0 = k * mean_laps
        rates = {}
        for key, n, lap, par in zip(keys, g["N"], g["L"], g["parent"]):
            rates[key] = par if math.isinf(e0) else (n + e0 * par) / (lap + e0)
        out.rates[level] = rates
        parents = rates
    out.d_global = _d_mle(float(t["D"].sum()), float(t["n"].sum()), float(t["u"].sum()), d0)
    by_era = t.groupby("era")[["D", "n", "u"]].sum()
    out.d_era = {e: _d_mle(r["D"], r["n"], r["u"], None) for e, r in by_era.iterrows() if r["u"] > 0}
    if lap1 and len(train):
        out.lap1 = float(train[f"lap1_{kind}"].astype(bool).mean())
    return out


@dataclass
class RateFit:
    """Both kinds, plain (every lap on p) and with the lap-1 term."""
    plain: dict
    lap1: dict
    n_races: int

    def rates(self, row, candidate: str) -> dict:
        """{p_sc, d_sc, p_vsc, d_vsc, p_sc_lap1, p_vsc_lap1} for a race under a candidate."""
        level = "circuit" if candidate == LAP1 else candidate
        fits = self.lap1 if candidate == LAP1 else self.plain
        out = {}
        for kind in KINDS:
            f = fits[kind]
            out[f"p_{kind}"], out[f"d_{kind}"] = f.p(level, row), f.d(level, row)
            out[f"p_{kind}_lap1"] = f.lap1 if candidate == LAP1 else None
        return out


def fit(train: pd.DataFrame, prior: EventModel | None = None, k0: float = K0) -> RateFit:
    return RateFit({k: fit_kind(train, k, prior, k0) for k in KINDS},
                   {k: fit_kind(train, k, prior, k0, lap1=True) for k in KINDS}, len(train))


# --- likelihood ------------------------------------------------------------------------------
def _xlog(x: float, y: float) -> float:
    """x log y with 0 log 0 = 0 and x log 0 = -inf for x > 0."""
    if x == 0:
        return 0.0
    return -math.inf if y <= 0 else x * math.log(y)


def starts_nll(n: float, laps: float, p: float) -> float:
    """Poisson process with rate p per lap at risk: p L - n log p."""
    return p * laps - _xlog(n, p)


def lap1_nll(x: int, p: float) -> float:
    return -(_xlog(x, p) + _xlog(1 - x, 1 - p))


def durations_nll(n: float, laps: float, d: float, censored: float = 0) -> float:
    """n geometric lengths (support 1, 2, ...) with mean d summing to `laps`, `censored` of them
    still running at the race's end. A complete period of length l adds log q + (l - 1) log(1 - q),
    a censored one log P(D >= l) = (l - 1) log(1 - q), with q = 1 / d."""
    if n == 0:
        return 0.0
    q = 1.0 / d
    return -(_xlog(n - censored, q) + _xlog(laps - n, 1.0 - q))


def race_nll(row, rates: dict) -> float:
    """Held-out NLL of one race's period starts and durations, both kinds, under `rates`
    (RateFit.rates). A lap-1 share of None leaves lap 1 on p."""
    total = 0.0
    for kind in KINDS:
        n, at_risk, lap1 = float(row[f"n_{kind}"]), float(row["at_risk"]), int(row[f"lap1_{kind}"])
        p, d, p1 = rates[f"p_{kind}"], rates[f"d_{kind}"], rates[f"p_{kind}_lap1"]
        if p1 is None:
            total += starts_nll(n, at_risk, p)
        else:
            total += lap1_nll(lap1, p1) + starts_nll(n - lap1, at_risk - 1, p)
        total += durations_nll(n, float(row[f"laps_{kind}"]), d, float(row[f"cens_{kind}"]))
    return total


# --- selection -------------------------------------------------------------------------------
def dry(stats: pd.DataFrame, include_wet: bool = False) -> pd.DataFrame:
    return stats if include_wet else stats[~stats["wet"].astype(bool)]


def loro(stats: pd.DataFrame, candidates=CANDIDATES, prior=None, k0=K0, include_wet=False) -> pd.DataFrame:
    """Per held-out race and candidate: its NLL under the rates fitted on every other race
    (race_id, candidate, nll)."""
    s = dry(stats, include_wet).reset_index(drop=True)
    rows = []
    for i, row in s.iterrows():
        f = fit(s.drop(index=i), prior, k0)
        for c in candidates:
            rows.append({"race_id": row["race_id"], "candidate": c, "nll": race_nll(row, f.rates(row, c))})
    return pd.DataFrame(rows, columns=["race_id", "candidate", "nll"])


def forward(stats: pd.DataFrame, candidates=CANDIDATES, prior=None, k0=K0, include_wet=False,
            from_season: int = FORWARD_FROM_SEASON) -> pd.DataFrame:
    """As-of folds (forward chaining): per dry race from `from_season` on and candidate, its NLL
    under the rates fitted on the dry races dated strictly before it (race_id, candidate, nll)."""
    s = dry(stats, include_wet).reset_index(drop=True)
    dates = [_date_key(r) for r in s["race_id"]]
    rows, fits = [], {}
    for i, row in s.iterrows():
        if dates[i][0] < from_season:
            continue
        before = [d < dates[i] for d in dates]
        cut = sum(before)   # stats are in date order, so the earlier races are a prefix
        if cut not in fits:
            fits[cut] = fit(s[before], prior, k0)
        for c in candidates:
            rows.append({"race_id": row["race_id"], "candidate": c, "nll": race_nll(row, fits[cut].rates(row, c))})
    return pd.DataFrame(rows, columns=["race_id", "candidate", "nll"])


def both_schemes(stats: pd.DataFrame, n_boot: int = N_BOOT, k0: float = K0, include_wet: bool = False):
    """(table, chosen, forward_nll, loro_nll): the selection table under forward chaining (it
    decides) and leave-one-race-out (reported beside), one row per scheme and candidate."""
    fw, lo = forward(stats, k0=k0, include_wet=include_wet), loro(stats, k0=k0, include_wet=include_wet)
    parts, chosen = [], "global"
    for scheme, nll in (("forward", fw), ("loro", lo)):
        if not len(nll):
            continue
        t, c = selection(nll, n_boot)
        parts.append(t.assign(scheme=scheme))
        if scheme == "forward":
            chosen = c
    cols = ["scheme", *SELECTION_COLUMNS, "chosen"]
    table = pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
    return table, chosen, fw, lo


def _boot_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    if not np.isfinite(values).all():
        return math.nan, math.nan
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


SELECTION_COLUMNS = ["candidate", "n_races", "n_dropped", "nll_per_race", "diff", "diff_lo", "diff_hi", "passes"]


def selection(nll: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """(table, chosen): per candidate the mean held-out NLL per race, the paired difference
    against global with its race-cluster 95 percent CI, whether it passes (CI below 0; a
    non-finite difference never passes) and the choice. A race whose NLL is infinite under every
    candidate (an early fold whose estimate rules out what the race shows, e.g. d = 1 from one
    1-lap period against a 3-lap one) cannot tell them apart and is dropped, counted in
    n_dropped; a race infinite under some candidates only stays and fails them."""
    wide = nll.pivot(index="race_id", columns="candidate", values="nll")
    void = ~np.isfinite(wide.to_numpy(dtype=float)).any(axis=1)
    n_dropped = int(void.sum())
    wide = wide[~void]
    rows = []
    for c in [c for c in CANDIDATES if c in wide]:
        diff = (wide[c] - wide["global"]).to_numpy(dtype=float)
        lo, hi = _boot_ci(diff, n_boot, seed) if c != "global" else (math.nan, math.nan)
        finite = bool(np.isfinite(diff).all()) and len(diff) > 0
        rows.append({"candidate": c, "n_races": int(len(diff)), "n_dropped": n_dropped,
                     "nll_per_race": float(wide[c].mean()),
                     "diff": float(diff.mean()) if finite else math.nan, "diff_lo": lo, "diff_hi": hi,
                     "passes": c == "global" or bool(finite and hi < 0)})
    t = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    chosen = "global"
    passing = t[(t["candidate"] != "global") & t["passes"]]
    if len(passing):
        order = sorted(passing.itertuples(), key=lambda r: (round(r.diff, 9), CANDIDATES.index(r.candidate)))
        chosen = order[0].candidate
    t["chosen"] = t["candidate"] == chosen
    return t, chosen


# --- the calibrator --------------------------------------------------------------------------
@CALIBRATORS.register("event_rates")
class EventRatesCalibrator(Calibrator):
    """Pre-race SC/VSC rates per race, as-of: fit(stats) takes per-race stats (load_stats,
    stats_from_periods, stats_from_factors), race_params(race) fills RaceParams
    sc_rate_per_lap, sc_mean_laps, vsc_rate_per_lap and vsc_mean_laps with the `level`
    candidate's rates fitted on the races dated before it. Under circuit_lap1, event_params
    gives the EventModel p_sc_lap1 / p_vsc_lap1 for it. It fills race constants and has no
    lap-time factors, so it is not a SimConfig.calibrator: calibrate raises."""
    name = "event_rates"

    def __init__(self, level: str = "global", k0: float = K0, include_wet: bool = False, **params):
        if level not in CANDIDATES:
            raise ValueError(f"unknown event_rates level {level!r}; candidates: {list(CANDIDATES)}")
        super().__init__(level=level, k0=k0, include_wet=include_wet, **params)
        self.level, self.k0, self.include_wet = level, float(k0), bool(include_wet)
        self.stats = pd.DataFrame(columns=STAT_COLUMNS)
        self.fitted = False
        self._fits = {}

    def fit(self, train) -> "EventRatesCalibrator":
        self.stats = dry(train, self.include_wet).reset_index(drop=True)
        self._fits = {}
        self.fitted = True
        return self

    def _race_row(self, race_id: str, circuit=None, circuit_type=None) -> dict:
        rid = canonical_race_id(race_id)
        known = self.stats[self.stats["race_id"] == rid]
        if len(known):
            row = known.iloc[0].to_dict()
        else:
            row = {"race_id": rid, "era": era(_date_key(rid)[0]), **circuit_info([rid]).iloc[0].to_dict()}
        if circuit is not None:
            row["circuit"] = circuit
        if circuit_type is not None:
            row["circuit_type"] = circuit_type
        return row

    def fit_as_of(self, race_id: str) -> RateFit:
        """The fit on the training races dated strictly before `race_id`."""
        key = _date_key(canonical_race_id(race_id))
        dates = [_date_key(r) for r in self.stats["race_id"]]
        before = [d < key for d in dates]
        cut = sum(before)
        if cut not in self._fits:
            self._fits[cut] = fit(self.stats[before], k0=self.k0)
        return self._fits[cut]

    def rates_for(self, race_id: str, circuit=None, circuit_type=None) -> dict:
        if not self.fitted:
            raise RuntimeError("event_rates: call fit(stats) first")
        row = self._race_row(race_id, circuit, circuit_type)
        return self.fit_as_of(row["race_id"]).rates(row, self.level)

    def race_params(self, race: RaceParams, circuit=None, circuit_type=None) -> RaceParams:
        r = self.rates_for(race.race_id, circuit, circuit_type)
        return race.model_copy(update={"sc_rate_per_lap": r["p_sc"], "sc_mean_laps": r["d_sc"],
                                       "vsc_rate_per_lap": r["p_vsc"], "vsc_mean_laps": r["d_vsc"]})

    def event_params(self, race_id: str) -> dict:
        """EventModel params for the lap-1 term ({} unless the level is circuit_lap1)."""
        if self.level != LAP1:
            return {}
        r = self.rates_for(race_id)
        return {k: r[k] for k in ("p_sc_lap1", "p_vsc_lap1") if r[k] is not None}

    def calibrate(self, ctx):
        raise TypeError("calibrator 'event_rates' fills RaceParams pre-race (race_params); "
                        "it is not a lap-time calibrator for SimConfig.calibrator")

    def report(self) -> dict:
        return {**super().report(), "n_races": int(len(self.stats))}


# --- synthetic world -------------------------------------------------------------------------
# circuit -> (type, SC multiplier, VSC multiplier), on SYN_BASE[era]; street circuits x SYN_STREET
SYN_CIRCUITS = {
    "street1": (STREET, 1.5, 1.0), "street2": (STREET, 0.8, 1.3), "street3": (STREET, 1.1, 0.7),
    "perm1": (PERMANENT, 0.5, 1.0), "perm2": (PERMANENT, 1.0, 1.0), "perm3": (PERMANENT, 1.6, 1.4),
    "perm4": (PERMANENT, 0.7, 0.6), "perm5": (PERMANENT, 1.2, 1.0),
}
SYN_BASE = {"2018-2021": {"sc": 0.012, "vsc": 0.010}, "2022+": {"sc": 0.018, "vsc": 0.016}}
SYN_STREET = {"sc": 2.0, "vsc": 1.2}
SYN_D = {"2018-2021": {"sc": 3.5, "vsc": 1.8}, "2022+": {"sc": 4.5, "vsc": 2.2}}
SYN_LAP1 = {"sc": 0.12, "vsc": 0.03}
SYN_WET_SHARE = 0.1
SYN_WET_FACTOR = 3.0


def synthetic_truth(circuits=None) -> pd.DataFrame:
    """The planted p per (era, circuit, kind), with its circuit type."""
    circuits = SYN_CIRCUITS if circuits is None else circuits
    rows = []
    for e in ERAS:
        for c, (ctype, m_sc, m_vsc) in circuits.items():
            for kind, m in (("sc", m_sc), ("vsc", m_vsc)):
                p = SYN_BASE[e][kind] * m * (SYN_STREET[kind] if ctype == STREET else 1.0)
                rows.append({"era": e, "circuit": c, "circuit_type": ctype, "kind": kind, "p": p,
                             "d": SYN_D[e][kind]})
    return pd.DataFrame(rows)


def _runs(laps):
    out, start, prev = [], None, None
    for lap in sorted(laps):
        if start is None:
            start = prev = lap
        elif lap == prev + 1:
            prev = lap
        else:
            out.append((start, prev))
            start = prev = lap
    if start is not None:
        out.append((start, prev))
    return out


def synthetic_periods(seasons=range(2018, 2026), circuits=None, repeats: int = 1, lap1=None, wet_share=SYN_WET_SHARE,
                      seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(periods, features): a neutralised_periods table drawn by the simulator's own "rate"
    model (events.RateEvents, a snapshot at lap 0) from synthetic_truth, the lap-1 shares
    `lap1` (SYN_LAP1 by default; {} for none), wet races at SYN_WET_FACTOR the rates, and the
    features table (race_id, circuit, circuit_type). Race ids <season>_<round>_synthetic_<circuit>,
    `repeats` rounds per circuit and season."""
    from .events import RateEvents
    circuits = SYN_CIRCUITS if circuits is None else circuits
    lap1 = SYN_LAP1 if lap1 is None else lap1
    truth = synthetic_truth(circuits).set_index(["era", "circuit", "kind"])
    names = list(circuits)
    total_laps = {c: 50 + 3 * i for i, c in enumerate(names)}
    periods, feats = [], []
    for season in seasons:
        e = era(season)
        rnd = 0
        for rep in range(repeats):
            for c in names:
                rnd += 1
                rid = f"{season}_{rnd:02d}_synthetic_{c}"
                rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()[:8], 16))
                wet = bool(rng.uniform() < wet_share)
                scale = SYN_WET_FACTOR if wet else 1.0
                race = RaceParams(race_id=rid, total_laps=total_laps[c], pit_loss_s=20.0,
                                  sc_rate_per_lap=min(1.0, truth.loc[(e, c, "sc"), "p"] * scale),
                                  sc_mean_laps=SYN_D[e]["sc"],
                                  vsc_rate_per_lap=min(1.0, truth.loc[(e, c, "vsc"), "p"] * scale),
                                  vsc_mean_laps=SYN_D[e]["vsc"])
                params = {f"p_{k}_lap1": v for k, v in lap1.items()}
                scen = RateEvents(**params).draw(Snapshot(race=race, driver="X", compound="SOFT"), rng)
                found = [(k, s, t) for k, laps in (("sc", scen.sc_laps), ("vsc", scen.vsc_laps))
                         for s, t in _runs(laps)]
                base = {"race_id": rid, "total_laps": race.total_laps, "wet": wet}
                if found:
                    periods += [{**base, "kind": k, "start_lap": s, "end_lap": t} for k, s, t in found]
                else:
                    periods.append({**base, "kind": "", "start_lap": np.nan, "end_lap": np.nan})
                feats.append({"race_id": rid, "circuit": c, "circuit_type": circuits[c][0]})
    cols = ["race_id", "kind", "start_lap", "end_lap", "total_laps", "wet"]
    return pd.DataFrame(periods)[cols], pd.DataFrame(feats)


def recovery_table(stats: pd.DataFrame, truth: pd.DataFrame, k0: float = K0) -> pd.DataFrame:
    """Planted against recovered p per (era, circuit, kind) on every dry race: the circuit MLE,
    the shrunk circuit estimate, its circuit-type parent and the global rate. The planted p
    applies from lap 2 (lap 1 has its own share), so the estimates are those of the lap-1
    fit, on laps 2 onward."""
    s = dry(stats)
    f = fit(s, k0=k0)
    rows = []
    for _, t in truth.iterrows():
        kind = t["kind"]
        part = s[(s["era"] == t["era"]) & (s["circuit"] == t["circuit"])]
        lap = float((part["at_risk"] - 1).sum())
        row = {"era": t["era"], "circuit": t["circuit"], "circuit_type": t["circuit_type"]}
        k = f.lap1[kind]
        rows.append({**row, "kind": kind, "races": int(len(part)), "planted": float(t["p"]),
                     "mle": float((part[f"n_{kind}"] - part[f"lap1_{kind}"]).sum()) / lap if lap > 0 else math.nan,
                     "shrunk": k.p("circuit", row), "type_level": k.p("circuit_type", row),
                     "global": k.p("global", row)})
    return pd.DataFrame(rows)


def recovery_d_lap1(stats: pd.DataFrame, lap1=None, k0: float = K0) -> pd.DataFrame:
    """Planted against recovered mean period length per (kind, era) and lap-1 share per kind."""
    lap1 = SYN_LAP1 if lap1 is None else lap1
    f = fit(dry(stats), k0=k0)
    rows = [{"kind": k, "quantity": f"d ({e})", "planted": SYN_D[e][k], "recovered": f.plain[k].d_era.get(e, math.nan)}
            for k in KINDS for e in ERAS]
    rows += [{"kind": k, "quantity": "lap-1 share", "planted": lap1.get(k, math.nan),
              "recovered": f.lap1[k].lap1 if f.lap1[k].lap1 is not None else math.nan} for k in KINDS]
    return pd.DataFrame(rows)


# --- report ----------------------------------------------------------------------------------
def guard_committed(out, race_ids):
    """Refuse a report with a synthetic race id anywhere under reports/calibration/."""
    from .cross_race import is_synthetic
    synthetic = [r for r in race_ids if is_synthetic(canonical_race_id(r))]
    out, root = Path(out).resolve(), REPORTS_DIR.resolve()
    if synthetic and (out == root or root in out.parents):
        raise ValueError(f"{len(synthetic)} synthetic race(s) (e.g. {synthetic[0]}) would go into the committed "
                         f"reports under {REPORTS_DIR}; pass --out a scratch or tmp directory")


def as_of_rates(stats: pd.DataFrame, k0: float = K0, include_wet: bool = False) -> pd.DataFrame:
    """Each race's rates per candidate, fitted on the races dated before it."""
    cal = EventRatesCalibrator(k0=k0, include_wet=include_wet).fit(stats)
    rows = []
    for _, r in stats.iterrows():
        f = cal.fit_as_of(r["race_id"])
        for c in CANDIDATES:
            rows.append({"race_id": r["race_id"], "candidate": c, "circuit": r["circuit"],
                         "circuit_type": r["circuit_type"], "era": r["era"], "wet": bool(r["wet"]),
                         "n_train": f.n_races, **f.rates(r, c)})
    return pd.DataFrame(rows)


def strengths_table(stats: pd.DataFrame, k0: float = K0) -> pd.DataFrame:
    f = fit(dry(stats), k0=k0)
    return pd.DataFrame([{"kind": k, "level": lv, "k_races": f.plain[k].strength[lv],
                          "k_races_lap1": f.lap1[k].strength[lv]} for k in KINDS for lv in LEVELS[1:]])


def report(stats: pd.DataFrame, out=OUT_DIR, source: str = "", n_boot: int = N_BOOT, k0: float = K0,
           truth: pd.DataFrame | None = None) -> Path:
    """rates.csv, forward_nll.csv, loro_nll.csv, selection.csv, strengths.csv (and recovery.csv
    with `truth`) and summary.md under `out`."""
    from .cross_race import md_table
    guard_committed(out, list(stats["race_id"]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = dry(stats)
    table, chosen, fw, lo = both_schemes(stats, n_boot, k0)
    fw.to_csv(out / "forward_nll.csv", index=False)
    lo.to_csv(out / "loro_nll.csv", index=False)
    table.to_csv(out / "selection.csv", index=False)
    as_of_rates(stats, k0).to_csv(out / "rates.csv", index=False)
    strengths = strengths_table(stats, k0)
    strengths.to_csv(out / "strengths.csv", index=False)
    wet = stats[stats["wet"].astype(bool)]
    md = ["# SC and VSC rates by circuit and era (SPEC 0011b)", "",
          f"Input: {source or 'given table'}. {len(stats)} race(s): {len(d)} dry, {len(wet)} wet (left out of the "
          f"dry rates, counted apart), {int(stats['n_red_flag'].sum())} red flag period(s).", ""]
    if len(wet):
        md += ["Wet races, counted apart:", "", md_table(pd.DataFrame({
            "races": [len(wet)], "sc starts": [int(wet["n_sc"].sum())], "vsc starts": [int(wet["n_vsc"].sum())],
            "laps at risk": [int(wet["at_risk"].sum())]})), ""]
    md += ["## Selection (held-out NLL of period starts and lengths, per race)", "",
           f"forward: each dry race from {FORWARD_FROM_SEASON} on, fitted on the races dated before it (as the "
           "calibrator is used); it decides. loro: fitted on every other race, reported beside. A candidate "
           "replaces global when the race-cluster 95 percent CI of the paired NLL difference (candidate less "
           "global) is below 0; among those that pass the lowest mean difference wins, ties to the simpler. "
           "`chosen` marks each scheme's pick.", "", md_table(table, 3), "",
           f"Chosen (forward): **{chosen}**.", ""]
    md += ["## Prior strengths (all dry races, races-equivalent; floor "
           f"{k0:g}, inf = full pooling)", "", md_table(strengths, 2), ""]
    if truth is not None:
        rec = recovery_table(stats, truth, k0)
        rec.to_csv(out / "recovery.csv", index=False)
        err = rec.assign(shrunk_err=(rec["shrunk"] - rec["planted"]).abs(), mle_err=(rec["mle"] - rec["planted"]).abs(),
                         global_err=(rec["global"] - rec["planted"]).abs())
        md += ["## Planted against recovered (synthetic)", "", "Mean absolute error of p against the planted value, "
               "per kind:", "", md_table(err.groupby("kind")[["shrunk_err", "mle_err", "global_err"]].mean()
                                         .reset_index(), 4), "",
               md_table(recovery_d_lap1(stats, k0=k0), 3), "", md_table(rec, 4), ""]
    path = out / "summary.md"
    path.write_text("\n".join(md))
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m src.sim.calibrate rates", description=__doc__.split("\n\n")[0])
    ap.add_argument("--periods", default=str(NEUTRALISED_PERIODS))
    ap.add_argument("--factors", default=str(FACTORS))
    ap.add_argument("--race-features", default=str(RACE_FEATURES))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--synthetic", action="store_true", help="the synthetic world (synthetic_periods) instead")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2018, 2025], help="--synthetic: first and last season")
    ap.add_argument("--repeats", type=int, default=1, help="--synthetic: races per circuit and season")
    ap.add_argument("--seed", type=int, default=0, help="--synthetic")
    args = ap.parse_args(argv)
    truth = None
    if args.synthetic:
        periods, feats = synthetic_periods(range(args.seasons[0], args.seasons[1] + 1), repeats=args.repeats,
                                           seed=args.seed)
        stats, source, truth = stats_from_periods(periods, feats), "synthetic_periods()", synthetic_truth()
    else:
        stats, source = load_stats(args.periods, args.factors, args.race_features)
    path = report(stats, args.out, source, args.n_boot, truth=truth)
    print(f"written to {path}")
    return 0
