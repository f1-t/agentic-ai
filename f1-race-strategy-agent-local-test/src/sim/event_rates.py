"""SC and VSC rates by circuit and era (SPEC 0011b), the "event_rates" calibrator.

    python -m src.sim.calibrate rates [--periods PATH] [--factors PATH] [--race-features PATH] [--out DIR]
    python -m src.sim.calibrate rates --synthetic --out /tmp/dir     # the synthetic test report

Input: data/derived/neutralised_periods.csv (race_id, kind sc/vsc/red_flag, start_lap,
end_lap, total_laps, wet; a race with no period has one row with an empty kind) when it
exists, else the SC/VSC periods of 0016's factors.csv (reports/calibration/cross_race/,
dry races by construction). Wet races are left out of the dry rates (include_wet=False) and
counted apart. Circuit and circuit type come from race_features.csv (circuit or F1DB's
circuitId, circuit_type or circuitType) when it names them, else from the cached event
schedule's Location through CIRCUITS, else the race id's slug as its own circuit of unknown
type (circuit_info).

Model, per kind (sc, vsc): starts ~ Poisson(p x laps_at_risk) per race, laps_at_risk the
total laps less the neutralised ones (any kind, red flags included), and period lengths
geometric with mean d = 1 / q. p is shrunk empirical-Bayes (Gamma-Poisson) along
global -> era (2018-2021, 2022+) -> circuit type (street / permanent) -> circuit. A node's
posterior is Gamma(N + E0 m, L + E0): N starts over L laps at risk, m its parent's posterior
mean, E0 = k x (mean laps at risk per race), k the level's prior strength in races. k comes
from the method of moments over the level's nodes (between-node variance of N/L around the
parent less the Poisson part), floored a priori at K0 = 5 races; no excess variance is full
pooling. The global level has an a priori Gamma prior: the EventModel rate as its mean and
K0 races of exposure. q is pooled per kind and era only (global when the era has no
period), under an a priori Beta prior of mean 1 / d0 (EventModel d) worth D_PRIOR_PERIODS
periods. The calibrator writes posterior means; with no data they are the EventModel values.

Durations: a period that ends on the race's last lap is right-censored; it says only that
the length was at least what was seen, so it adds the survival P(D >= l) to the likelihood
and its continuations, but no end, to the q posterior. factors.csv has no period ends, so
its periods count as complete. Back-to-back periods of one kind (one ending and the next
starting on the following lap) cannot be told apart in the lap data and are taken as one
period; this is ignored.

Lap-1 term (candidate circuit_lap1): p_sc_lap1 / p_vsc_lap1 (config.EventModel) are the
posterior mean share of races with a period of that kind starting on lap 1, under a Beta
prior with the global per-lap rate as its mean (no lap-1 excess a priori) worth
LAP1_PRIOR_RACES races; p is then fitted on laps 2 onward. A circuit-scaled lap-1 term (the
circuit's rate times a global lap-1 multiplier) is a later candidate, worth adding only if the
real periods table shows lap-1 risk tracking circuit type.

As-of: EventRatesCalibrator gives a race the rates fitted on the races before it,
race_ids.as_of (strictly earlier (season, round)), never itself.

Selection, stated before any data, on as-of folds (forward chaining): each dry race from
FORWARD_FROM_SEASON on is scored under the posteriors fitted on the dry races before it
(race_ids.as_of), as the calibrator is used. Leave-one-race-out (every other race) is
reported beside it and does not select. The score is the held-out negative log-likelihood
under the posterior predictive, so no held-out race can have probability 0: its period
starts negative binomial (Gamma-Poisson; lap 1 beta-Bernoulli under the lap-1 term) and its
periods' lengths beta-geometric (censored as above).
Candidates global, era, circuit_type, circuit, circuit_lap1; a candidate replaces global when
the race-cluster 95 percent CI of the paired NLL difference (candidate less global) is below
0. Among those that pass the lowest mean difference wins; ties go to the simpler
(CANDIDATES order).

Output: rates.csv (each race's as-of rates per candidate), forward_nll.csv, loro_nll.csv,
selection.csv (both schemes, the forward one decides), strengths.csv and summary.md under
OUT_DIR. A report with a synthetic race id refuses any directory under reports/calibration/
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
from .race_ids import as_of, canonical_race_id, parse_race_id

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
# The circuit of a race comes from race_features.csv when it names one, else from the cached
# event schedule's Location (location_key) through CIRCUITS, never from the race id: a country
# can host several circuits in one season (Imola, Mugello and Monza in 2020; Miami, Austin
# and Las Vegas from 2023). Location -> (circuit, circuit type), for the 2018 onward calendars,
# with the spellings the schedules use. Street: a temporary or public-road circuit.
CIRCUITS = {
    "melbourne": ("albert_park", STREET), "albert_park": ("albert_park", STREET),
    "sakhir": ("sakhir", PERMANENT), "bahrain": ("sakhir", PERMANENT),
    "shanghai": ("shanghai", PERMANENT),
    "baku": ("baku", STREET),
    "barcelona": ("catalunya", PERMANENT), "montmelo": ("catalunya", PERMANENT),
    "catalunya": ("catalunya", PERMANENT),
    "monte_carlo": ("monaco", STREET), "monaco": ("monaco", STREET),
    "montreal": ("villeneuve", STREET),
    "le_castellet": ("paul_ricard", PERMANENT), "paul_ricard": ("paul_ricard", PERMANENT),
    "spielberg": ("red_bull_ring", PERMANENT),
    "silverstone": ("silverstone", PERMANENT),
    "hockenheim": ("hockenheim", PERMANENT),
    "nurburgring": ("nurburgring", PERMANENT), "nurburg": ("nurburgring", PERMANENT),
    "budapest": ("hungaroring", PERMANENT), "mogyorod": ("hungaroring", PERMANENT),
    "spa_francorchamps": ("spa", PERMANENT), "spa": ("spa", PERMANENT), "stavelot": ("spa", PERMANENT),
    "zandvoort": ("zandvoort", PERMANENT),
    "monza": ("monza", PERMANENT),
    "mugello": ("mugello", PERMANENT), "scarperia": ("mugello", PERMANENT),
    "imola": ("imola", PERMANENT),
    "portimao": ("portimao", PERMANENT),
    "istanbul": ("istanbul_park", PERMANENT),
    "sochi": ("sochi", PERMANENT),
    "marina_bay": ("marina_bay", STREET), "singapore": ("marina_bay", STREET),
    "suzuka": ("suzuka", PERMANENT),
    "lusail": ("lusail", PERMANENT), "doha": ("lusail", PERMANENT),
    "austin": ("cota", PERMANENT),
    "mexico_city": ("rodriguez", PERMANENT),
    "sao_paulo": ("interlagos", PERMANENT), "interlagos": ("interlagos", PERMANENT),
    "jeddah": ("jeddah", STREET),
    "miami": ("miami", STREET), "miami_gardens": ("miami", STREET),
    "las_vegas": ("las_vegas", STREET),
    "yas_island": ("yas_marina", PERMANENT), "yas_marina": ("yas_marina", PERMANENT),
    "abu_dhabi": ("yas_marina", PERMANENT),
}
# events held at a location on another layout than its usual one: event_slug -> circuit
EVENT_CIRCUITS = {"sakhir": ("sakhir_outer", PERMANENT)}
# F1DB's circuitType in race_features.csv -> circuit type
F1DB_CIRCUIT_TYPES = {"race": PERMANENT, "street": STREET, "road": STREET}

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
def location_key(location) -> str:
    """A schedule Location as a CIRCUITS key: accents dropped, slugged ('São Paulo' -> 'sao_paulo')."""
    from .race_ids import _ascii, slug
    return slug(_ascii(location))


def schedule_circuit(event) -> tuple[str, str] | None:
    """(circuit, circuit type) of one schedule event (EventName, Location), None when its
    location is not in CIRCUITS."""
    from .race_ids import event_slug
    hit = EVENT_CIRCUITS.get(event_slug(event["EventName"]))
    return hit or CIRCUITS.get(location_key(event["Location"]))


def circuit_info(race_ids, features: pd.DataFrame | None = None, schedule=None) -> pd.DataFrame:
    """race_id, circuit, circuit_type per race, never read from the race id's slug when anything
    else knows it:
      1. `features` (race_features.csv): circuit (or F1DB's circuitId) and circuit_type (or
         F1DB's circuitType: RACE permanent, STREET and ROAD street), where they are given;
      2. else the race's event in its season's schedule, `schedule(season)` (default
         data.season_schedule, the cached FastF1 schedule), through schedule_circuit;
      3. else the race id's slug as its own circuit, type UNKNOWN.
    Synthetic race ids skip the schedule."""
    from .cross_race import is_synthetic
    from .data import season_schedule
    schedule = season_schedule if schedule is None else schedule
    rows = []
    for rid in race_ids:
        season, rnd, slug = parse_race_id(rid)
        circuit, ctype = slug, UNKNOWN
        sched = None if is_synthetic(canonical_race_id(rid)) else schedule(season)
        if sched is not None:
            event = sched[sched["RoundNumber"] == rnd]
            hit = schedule_circuit(event.iloc[0]) if len(event) else None
            if hit is not None:
                circuit, ctype = hit
        rows.append({"race_id": rid, "circuit": circuit, "circuit_type": ctype})
    out = pd.DataFrame(rows, columns=["race_id", "circuit", "circuit_type"])
    if features is not None and len(features) and "race_id" in features:
        f = features.assign(race_id=features["race_id"].map(canonical_race_id)).drop_duplicates("race_id")
        f = f.set_index("race_id")
        for col, names, values in (("circuit", ("circuit", "circuitId", "circuit_id"), None),
                                   ("circuit_type", ("circuit_type", "circuitType"), F1DB_CIRCUIT_TYPES)):
            src = next((n for n in names if n in f), None)
            if src is None:
                continue
            given = out["race_id"].map(f[src])
            ok = given.notna() & (given.astype(str).str.strip() != "")
            text = given[ok].astype(str).str.strip().str.lower()
            out.loc[ok, col] = text.map(lambda v: values.get(v, v)) if values else text
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
PRIOR_RACE_LAPS = 60.0     # laps at risk per race behind the global prior strength when no race is known
D_PRIOR_PERIODS = 2.0      # strength of the a priori Beta on the per-lap ending probability, in periods
LAP1_PRIOR_RACES = 2.0     # strength of the a priori Beta on the lap-1 start probability, in races


def _prior(prior: EventModel | None):
    return prior if prior is not None else EventModel()


@dataclass
class KindFit:
    """One kind's posteriors: Gamma (shape, rate) of the start rate per node, Beta (a, b) of the
    per-lap ending probability globally and per era, and (lap-1 fit only) Beta (a, b) of the
    lap-1 start probability. Point estimates are posterior means: p = shape / rate,
    d = 1 / E[q] = (a + b) / a, the lap-1 share a / (a + b)."""
    kind: str
    gamma: dict = field(default_factory=dict)       # level -> {node key: (shape, rate)}
    strength: dict = field(default_factory=dict)    # level -> k in races (inf: full pooling)
    beta_global: tuple = (1.0, 0.0)
    beta_era: dict = field(default_factory=dict)
    lap1_beta: tuple | None = None
    n_races: int = 0

    def gamma_of(self, level: str, row) -> tuple:
        """The node's posterior at `level` for a race (its era, circuit type, circuit), falling
        back to the nearest ancestor the training data has."""
        keys = _node_keys(row)
        out = self.gamma["global"][()]
        for lv in LEVELS[1:LEVELS.index(level) + 1]:
            out = self.gamma[lv].get(keys[lv], out)
        return out

    def p(self, level: str, row) -> float:
        shape, rate = self.gamma_of(level, row)
        return shape / rate

    def beta_of(self, level: str, row) -> tuple:
        return self.beta_global if level == "global" else self.beta_era.get(row["era"], self.beta_global)

    def d(self, level: str, row) -> float:
        a, b = self.beta_of(level, row)
        return (a + b) / a

    @property
    def lap1(self) -> float | None:
        return None if self.lap1_beta is None else self.lap1_beta[0] / sum(self.lap1_beta)

    # kept for the report and tests: the posterior mean rate per node
    @property
    def rates(self) -> dict:
        return {lv: {k: s / r for k, (s, r) in nodes.items()} for lv, nodes in self.gamma.items()}

    @property
    def d_global(self) -> float:
        a, b = self.beta_global
        return (a + b) / a

    @property
    def d_era(self) -> dict:
        return {e: (a + b) / a for e, (a, b) in self.beta_era.items()}


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


def fit_kind(train: pd.DataFrame, kind: str, prior: EventModel | None = None, k0: float = K0,
             lap1: bool = False) -> KindFit:
    """Posteriors of one kind on `train` (stats rows). With `lap1`, p is fitted on laps 2
    onward (a lap-1 start not counted, lap 1 not at risk) and the lap-1 Beta is kept.

    Global start rate: Gamma prior with the EventModel rate as mean and k0 races of laps at risk
    (the training mean, PRIOR_RACE_LAPS without races) as its exposure. Each lower node: the
    parent's posterior mean as prior mean with the level's strength (method of moments, floor
    k0); full pooling takes the parent's posterior. Ending probability q = 1 / d: Beta prior
    with mean 1 / d0 (EventModel) and D_PRIOR_PERIODS periods; a complete period adds an end
    and l - 1 continuations, a censored one only the continuations. An era with no period takes
    the global posterior. Lap 1: Beta prior with mean the global per-lap rate (no lap-1 excess a
    priori) and LAP1_PRIOR_RACES races, updated by the races with and without a lap-1 start."""
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
    mean_laps = total_l / len(t) if len(t) else 0.0
    e0g = k0 * (mean_laps if mean_laps > 0 else PRIOR_RACE_LAPS - (1 if lap1 else 0))
    out.gamma["global"] = {(): (float(t["N"].sum()) + e0g * p0, total_l + e0g)}
    out.strength["global"] = k0
    parents = out.gamma["global"]
    cols = []
    for level in LEVELS[1:]:
        cols = cols + [level]
        g = t.groupby(cols, sort=True)[["N", "L"]].sum().reset_index()
        keys = [tuple(r) for r in g[cols].itertuples(index=False, name=None)]
        par = [parents[k[:-1]] for k in keys]
        g["parent"] = [s / r for s, r in par]
        k = _strength(g, mean_laps, k0)
        out.strength[level] = k
        e0 = k * mean_laps
        nodes = {}
        for key, n, lap, pg, m in zip(keys, g["N"], g["L"], par, g["parent"]):
            nodes[key] = pg if math.isinf(e0) else (n + e0 * m, lap + e0)
        out.gamma[level] = nodes
        parents = nodes
    a0, b0 = D_PRIOR_PERIODS / d0, D_PRIOR_PERIODS * (1.0 - 1.0 / d0)
    out.beta_global = (a0 + float(t["u"].sum()), b0 + float(t["D"].sum() - t["n"].sum()))
    by_era = t.groupby("era")[["D", "n", "u"]].sum()
    out.beta_era = {e: (a0 + float(r["u"]), b0 + float(r["D"] - r["n"])) for e, r in by_era.iterrows() if r["n"] > 0}
    if lap1 and len(train):
        m = out.p("global", {"era": None, "circuit_type": None, "circuit": None})
        x = float(train[f"lap1_{kind}"].astype(bool).sum())
        out.lap1_beta = (LAP1_PRIOR_RACES * m + x, LAP1_PRIOR_RACES * (1.0 - m) + len(train) - x)
    return out


@dataclass
class RateFit:
    """Both kinds, plain (every lap on p) and with the lap-1 term."""
    plain: dict
    lap1: dict
    n_races: int

    def _pick(self, candidate):
        return ("circuit" if candidate == LAP1 else candidate), (self.lap1 if candidate == LAP1 else self.plain)

    def rates(self, row, candidate: str) -> dict:
        """{p_sc, d_sc, p_vsc, d_vsc, p_sc_lap1, p_vsc_lap1} for a race under a candidate:
        posterior means."""
        level, fits = self._pick(candidate)
        out = {}
        for kind in KINDS:
            f = fits[kind]
            out[f"p_{kind}"], out[f"d_{kind}"] = f.p(level, row), f.d(level, row)
            out[f"p_{kind}_lap1"] = f.lap1 if candidate == LAP1 else None
        return out

    def predictive(self, row, candidate: str) -> dict:
        """Per kind the posteriors race_nll scores with: {kind: (gamma, beta, lap1_beta or None)}."""
        level, fits = self._pick(candidate)
        return {k: (fits[k].gamma_of(level, row), fits[k].beta_of(level, row),
                    fits[k].lap1_beta if candidate == LAP1 else None) for k in KINDS}


def fit(train: pd.DataFrame, prior: EventModel | None = None, k0: float = K0) -> RateFit:
    return RateFit({k: fit_kind(train, k, prior, k0) for k in KINDS},
                   {k: fit_kind(train, k, prior, k0, lap1=True) for k in KINDS}, len(train))


# --- likelihood (posterior predictive) --------------------------------------------------------
def _lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def starts_nll(n: float, laps: float, shape: float, rate: float) -> float:
    """Gamma-Poisson: the start laps of a held-out race with n starts over `laps` laps at risk,
    the rate integrated over its Gamma(shape, rate) posterior. This is the negative binomial
    count less log(n! / L^n), a term the same under every candidate."""
    return -(math.lgamma(shape + n) - math.lgamma(shape) + shape * math.log(rate) - (shape + n) * math.log(rate + laps))


def durations_nll(n: float, laps: float, a: float, b: float, censored: float = 0) -> float:
    """Beta-geometric: a held-out race's n periods of one kind, `laps` laps in all, `censored` of
    them still running at the race's end, q = 1 / d integrated over its Beta(a, b) posterior. A
    complete period of length l is an end after l - 1 continuations, a censored one only the
    continuations (the survival function P(D >= l))."""
    if n == 0:
        return 0.0
    ends, cont = n - censored, laps - n
    return -(_lbeta(a + ends, b + cont) - _lbeta(a, b))


def lap1_nll(x: int, a: float, b: float) -> float:
    """Beta-Bernoulli: a lap-1 start (x = 1) or none."""
    return -math.log((a if x else b) / (a + b))


def race_nll(row, pred: dict) -> float:
    """Held-out NLL of one race's period starts and lengths, both kinds, under the posterior
    predictive `pred` (RateFit.predictive). A lap-1 Beta of None leaves lap 1 on p. Always finite."""
    total = 0.0
    for kind in KINDS:
        n, at_risk, lap1 = float(row[f"n_{kind}"]), float(row["at_risk"]), int(row[f"lap1_{kind}"])
        (shape, rate), (a, b), l1 = pred[kind]
        if l1 is None:
            total += starts_nll(n, at_risk, shape, rate)
        else:
            total += lap1_nll(lap1, *l1) + starts_nll(n - lap1, at_risk - 1, shape, rate)
        total += durations_nll(n, float(row[f"laps_{kind}"]), a, b, float(row[f"cens_{kind}"]))
    if not math.isfinite(total):
        raise AssertionError(f"held-out NLL of {row.get('race_id', '?')} is not finite: {total}")
    return total


# --- selection -------------------------------------------------------------------------------
def dry(stats: pd.DataFrame, include_wet: bool = False) -> pd.DataFrame:
    return stats if include_wet else stats[~stats["wet"].astype(bool)]


def _score(row, f: RateFit, candidates) -> list[dict]:
    return [{"race_id": row["race_id"], "candidate": c, "nll": race_nll(row, f.predictive(row, c))}
            for c in candidates]


def loro(stats: pd.DataFrame, candidates=CANDIDATES, prior=None, k0=K0, include_wet=False) -> pd.DataFrame:
    """Per held-out race and candidate: its NLL under the posteriors fitted on every other race
    (race_id, candidate, nll)."""
    s = dry(stats, include_wet).reset_index(drop=True)
    rows = []
    for i, row in s.iterrows():
        rows += _score(row, fit(s.drop(index=i), prior, k0), candidates)
    return pd.DataFrame(rows, columns=["race_id", "candidate", "nll"])


def forward(stats: pd.DataFrame, candidates=CANDIDATES, prior=None, k0=K0, include_wet=False,
            from_season: int = FORWARD_FROM_SEASON) -> pd.DataFrame:
    """As-of folds (forward chaining): per dry race from `from_season` on and candidate, its NLL
    under the posteriors fitted on the dry races before it (race_ids.as_of)."""
    s = dry(stats, include_wet).reset_index(drop=True)
    ids = list(s["race_id"])
    rows, fits = [], {}
    for _, row in s.iterrows():
        if _date_key(row["race_id"])[0] < from_season:
            continue
        before = tuple(as_of(row["race_id"], ids))
        if before not in fits:
            fits[before] = fit(s[s["race_id"].isin(before)], prior, k0)
        rows += _score(row, fits[before], candidates)
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
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


SELECTION_COLUMNS = ["candidate", "n_races", "nll_per_race", "diff", "diff_lo", "diff_hi", "passes"]


def selection(nll: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """(table, chosen): per candidate the mean held-out NLL per race, the paired difference
    against global with its race-cluster 95 percent CI, whether it passes (CI below 0) and the
    choice. Every held-out NLL is finite (posterior predictive scoring); a table with a
    non-finite one is refused."""
    wide = nll.pivot(index="race_id", columns="candidate", values="nll")
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError("selection: a held-out NLL is not finite")
    rows = []
    for c in [c for c in CANDIDATES if c in wide]:
        diff = (wide[c] - wide["global"]).to_numpy(dtype=float)
        lo, hi = _boot_ci(diff, n_boot, seed) if c != "global" else (math.nan, math.nan)
        rows.append({"candidate": c, "n_races": int(len(diff)), "nll_per_race": float(wide[c].mean()),
                     "diff": float(diff.mean()) if len(diff) else math.nan, "diff_lo": lo, "diff_hi": hi,
                     "passes": c == "global" or bool(len(diff) and hi < 0)})
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
        """The fit on the training races before `race_id` (race_ids.as_of)."""
        before = tuple(as_of(canonical_race_id(race_id), list(self.stats["race_id"])))
        if before not in self._fits:
            self._fits[before] = fit(self.stats[self.stats["race_id"].isin(before)], k0=self.k0)
        return self._fits[before]

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
