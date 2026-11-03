"""Per-circuit race-pace prior for the fuel term: fuel burn plus track evolution (SPEC 0020).

    python -m src.sim.calibrate fuel --races 2021_04_spain ... [--race-features PATH] [--out DIR]
    python -m src.sim.calibrate fuel --synthetic --out /tmp/dir     # the synthetic test report

The config fuel prior (SimConfig.pre_race_fuel_slope_s_per_lap, -0.045 s/lap, tau 0.008) is
one number for every race. The race-pace slope is fuel burn plus track evolution, both
circuit properties, so other seasons at the same circuit should predict it.

A  Target, per race: the hindsight race-pace slope, the full-window linear_fuel fuel slope on
   free-air rows (#29: fit_free_air_only), with its fit_report sd (race_target).
B  Candidates, calibrators in the #23 registry. Each is the shrinkage calibrator with
   fuel_s_per_lap (mean, sd) replaced by the candidate's prediction; the sd is then the
   to_snapshot fuel tau (calibrators.degradation_prior passes factor sds as taus):
     fuel_config            -0.045, sd 0.008: the shrinkage calibrator's own output, the reference.
     fuel_circuit_history   the mean target at the race's circuit in earlier seasons of the
                            training races, shrunk toward their grand mean with weight
                            n / (n + K0), K0 = 1 a priori; sd^2 = s_c^2 + (1 - w) s_g^2, s_c
                            the between-race sd at the circuit (the pooled within-circuit sd
                            when it has fewer than 2 races), s_g the between-race sd of all
                            the as-of training races. No race at the circuit: the grand mean,
                            sd s_g.
     fuel_circuit_features  ridge regression of the target on the race_features.csv
                            covariates (FEATURE_COLUMNS), standardised, the intercept free,
                            alpha from ALPHAS by inner leave-one-race-out; sd the inner LORO
                            rmse at that alpha. Only with MIN_FEATURE_RACES as-of training
                            races that have every covariate.
     fuel_practice_longrun  a physics fuel term, -FUEL_EFFECT_S_PER_KG x FUEL_START_KG /
                            total laps, plus the practice fit's session-evolution slopes (#13,
                            fit_report evolution_s_per_min, mean over sessions) read as
                            evolution per lap: times the median practice long-run lap in
                            minutes. The physics constants are a priori rules of thumb,
                            fixed below before any run: FUEL_START_KG 110 kg (not fitted)
                            and FUEL_EFFECT_S_PER_KG 0.03 s/kg (not fitted).
   As-of: a learned candidate sees only races strictly before the held-out race by (season,
   round) (race_ids.as_of, the rule of event_rates too): for the grand mean and the features
   that admits the same season's earlier rounds; never the race itself or a later one. With no such race, it gives the config prior (source
   "config (no history)").
C  Selection, leave-one-race-out over the eligible races (calibration.eligibility): per
   candidate the held-out target mae and the 68 percent interval coverage (|err| <= sd), the
   downstream G effect (pre_race and to_snapshot bias and mae_offset by horizon, race-weighted,
   cross_race.paired), and the to_snapshot stop-gain mae (calibration.stop_gain against the
   race's own full-window fit). Rule, stated before any data: a candidate replaces config only
   if its held-out target mae is lower with a race-cluster 95 percent CI on the paired
   difference (candidate less config) below 0; among those that pass the lowest mae wins, and
   on a tie the lower stop-gain mae.
D  If a candidate wins, the to_snapshot fuel tau becomes its predictive sd per race: that is
   what its calibrator outputs, so adopting it is setting SimConfig.calibrator to it. Nothing
   here changes a default; the report only names the choice.
   For the real run: report pre_race and to_snapshot G twice, with the tau from the predictive
   sd and with it floored at the config 0.008, to see whether a tighter tau helps or hurts on
   real races (on synthetic histories a tighter sd from a larger grand mean trusted a prior
   still off for some races).
E  tests/test_sim_fuel_prior.py.
F  report(): f_table.csv (candidate, held-out mae, coverage, G bias at 20-40 and 50-60 per
   window, stop-gain mae), and prior_share.csv: how much of the absolute G error was the
   race-pace prior, from the hindsight row (each race's own target as the prior, a bound and
   not a candidate). The sentence for the simulator section is written only on real races:
   on synthetic histories the share holds by construction, so it is a pipeline check and
   stays out of the report text.

A report with a synthetic race id refuses any directory under reports/calibration/
(event_rates.guard_committed).
"""
import argparse
import hashlib
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import CalibrationOutput, RaceParams
from .calibrators import CALIBRATORS, CalibrationDataset, ShrinkageCalibrator
from .config import ComponentSpec, SimConfig
from .event_rates import PERMANENT, REPORTS_DIR, STREET, circuit_info, era, guard_committed
from . import race_ids
from .race_ids import canonical_race_id, parse_race_id

ROOT = Path(__file__).resolve().parent.parent.parent
RACE_FEATURES = ROOT / "data" / "race_features.csv"     # the data team's per-race covariates, when present
OUT_DIR = REPORTS_DIR / "fuel_prior"

CANDIDATES = ("config", "circuit_history", "circuit_features", "practice_longrun")   # the reference first
REFERENCE = "config"
ORACLE = "hindsight"          # each race's own target: a bound on the prior's share, not a candidate
WINDOWS = ("pre_race", "to_snapshot")
SG_WINDOW = "to_snapshot"
G_RANGES = {"20-40": (20, 40), "50-60": (50, 60)}   # horizons, inclusive
N_BOOT = 2000

K0 = 1.0                      # circuit_history prior strength in races, a priori
MIN_FEATURE_RACES = 12        # circuit_features needs this many as-of training races with every covariate
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
# covariate -> the race_features.csv columns that may carry it, first found wins. street and
# era_2022 are derived (circuit type, season), total_laps comes from the race itself.
FEATURE_ALIASES = {"lap_length_km": ("lap_length_km", "track_length_km", "circuit_length_km"),
                   "turns": ("turns", "n_turns", "corners")}
FEATURE_COLUMNS = ("lap_length_km", "turns", "street", "total_laps", "era_2022")
MIN_SD = 1e-3                 # floor on a predictive sd, s/lap: a 0 would hold the fuel term alone

# practice_longrun physics, fixed a priori (SPEC 0020 B), not tuned on any race:
FUEL_START_KG = 110.0         # a priori rule of thumb, not fitted: the race fuel limit (2019 on), cars start near it
FUEL_EFFECT_S_PER_KG = 0.03   # a priori rule of thumb, not fitted: about 0.3 s per 10 kg of fuel
FUEL_EFFECT_SD_S_PER_KG = 0.005   # its uncertainty, about one sixth of it
EVOLUTION_SD_S_PER_LAP = 0.005    # the uncertainty of practice evolution read as race evolution

PREDICTION_COLUMNS = ["race_id", "candidate", "season", "circuit", "mean", "sd", "source", "n_history", "target",
                      "target_sd", "err", "covered"]


# --- A: the target -----------------------------------------------------------------------
def target_config(config: SimConfig) -> SimConfig:
    """The target's config: the race's, with the full window, linear_fuel and free-air rows only."""
    return config.model_copy(update={"fit_window": "full", "exclude_snapshot_driver": False, "fit_free_air_only": True,
                                     "degradation": ComponentSpec(name="linear_fuel")})


_TARGETS: dict = {}


def race_target(data) -> tuple[float, float]:
    """(slope, sd) of one race (jobs.RaceData or calibrators.RaceInput): the full-window
    linear_fuel fuel slope on free-air rows and its fit_report sd. NaN when the slope was not
    fitted (held at the prior). Cached per race id, config and lap frame."""
    from . import calibration
    from .engine import Simulator
    cfg = target_config(data.config)
    key = (canonical_race_id(data.race_id), calibration.config_hash(cfg), len(data.frame),
           float(data.frame["lap_time_s"].sum()))
    if key not in _TARGETS:
        deg = Simulator(data.frame, cfg, data.practice).degradation
        rep = deg.fit_report
        if rep.get("fuel_slope_source") != "fitted":
            _TARGETS[key] = (math.nan, math.nan)
        else:
            _TARGETS[key] = (float(deg.fuel_slope), float(rep["sd"]["fuel_s_per_lap"]))
    return _TARGETS[key]


def _season_round(race_id: str) -> tuple[int, int]:
    season, rnd, _ = parse_race_id(race_id)
    return season, rnd


def _feature_frame(features) -> pd.DataFrame | None:
    """race_features as a frame with canonical race ids, None without one."""
    if features is None:
        return None
    if not isinstance(features, pd.DataFrame):
        path = Path(features)
        if not path.is_file():
            return None
        features = pd.read_csv(path)
    if not len(features) or "race_id" not in features:
        return None
    return features.assign(race_id=features["race_id"].map(canonical_race_id)).drop_duplicates("race_id")


def race_rows(race_ids, total_laps: dict, features=None) -> pd.DataFrame:
    """Per race: race_id, season, round, era, circuit, circuit_type (event_rates.circuit_info:
    race_features.csv, else the cached schedule's location, else the race id's slug of unknown
    type) and FEATURE_COLUMNS (NaN where unknown)."""
    feats = _feature_frame(features)
    ids = [canonical_race_id(r) for r in race_ids]
    t = circuit_info(ids, feats)
    sr = [_season_round(r) for r in ids]
    t["season"], t["round"] = [s for s, _ in sr], [r for _, r in sr]
    t["era"] = t["season"].map(era)
    t["total_laps"] = [float(total_laps.get(r, np.nan)) for r in ids]
    t["street"] = t["circuit_type"].map({STREET: 1.0, PERMANENT: 0.0})
    t["era_2022"] = (t["season"] >= 2022).astype(float)
    f = feats.set_index("race_id") if feats is not None else None
    for col, names in FEATURE_ALIASES.items():
        src = next((n for n in names if f is not None and n in f), None)
        t[col] = pd.to_numeric(t["race_id"].map(f[src]), errors="coerce") if src else np.nan
    return t[["race_id", "season", "round", "era", "circuit", "circuit_type", *FEATURE_COLUMNS]]


def history_table(targets: dict, total_laps: dict, features=None) -> pd.DataFrame:
    """The training history: race_rows plus target and target_sd, from {race_id: (slope, sd)}."""
    t = race_rows(list(targets), total_laps, features)
    t["target"] = [float(targets[r][0]) for r in t["race_id"]]
    t["target_sd"] = [float(targets[r][1]) for r in t["race_id"]]
    return t.sort_values(["season", "round"], kind="stable").reset_index(drop=True)


def history_from_races(races, features=None) -> pd.DataFrame:
    """history_table over races (jobs.RaceData, calibrators.RaceInput), targets by race_target."""
    races = list(races)
    targets = {canonical_race_id(r.race_id): race_target(r) for r in races}
    return history_table(targets, {canonical_race_id(r.race_id): r.race.total_laps for r in races}, features)


def as_of(history: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """The as-of training set: the rows of races strictly before `race_id` by (season, round)
    (race_ids.as_of, the rule event_rates uses too), with a target."""
    keep = set(race_ids.as_of(race_id, list(history["race_id"])))
    h = history[history["race_id"].isin(keep)]
    return h[h["target"].notna()]


# --- B: the predictions ------------------------------------------------------------------------
def _sd(values) -> float:
    v = pd.Series(values, dtype=float).dropna()
    return float(v.std(ddof=1)) if len(v) >= 2 else math.nan


def _config_prediction(config: SimConfig, source: str) -> dict:
    return {"mean": float(config.pre_race_fuel_slope_s_per_lap), "sd": float(config.shrinkage["fuel_s_per_lap"]),
            "source": source, "n_history": 0}


def circuit_history_prediction(train: pd.DataFrame, circuit: str, k0: float = K0) -> dict | None:
    """The shrunk circuit mean on `train` (as-of rows with a target), None with no rows."""
    if not len(train):
        return None
    y = train["target"].astype(float)
    grand, s_g = float(y.mean()), _sd(y)
    at = train[train["circuit"] == circuit]
    n = int(len(at))
    if n == 0:
        return {"mean": grand, "sd": s_g, "source": "grand mean (no circuit history)", "n_history": 0,
                "weight": 0.0, "circuit_mean": math.nan, "grand_mean": grand}
    w = n / (n + k0)
    cm = float(at["target"].mean())
    within = [g["target"].astype(float) - g["target"].mean() for _, g in train.groupby("circuit") if len(g) >= 2]
    dof = sum(len(x) - 1 for x in within)
    s_w = math.sqrt(sum(float((x ** 2).sum()) for x in within) / dof) if dof > 0 else s_g
    s_c = _sd(at["target"]) if n >= 2 else s_w
    sd = math.sqrt(s_c ** 2 + (1 - w) * s_g ** 2) if not (math.isnan(s_c) or math.isnan(s_g)) else math.nan
    return {"mean": w * cm + (1 - w) * grand, "sd": sd, "source": "circuit history", "n_history": n, "weight": w,
            "circuit_mean": cm, "grand_mean": grand}


def _ridge(X, y, alpha):
    mu = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mu) / scale
    ym = float(y.mean())
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return mu, scale, ym, beta


def _ridge_predict(model, X):
    mu, scale, ym, beta = model
    return ym + ((X - mu) / scale) @ beta


def ridge_inner_loro(X, y, alphas=ALPHAS) -> tuple[float, float]:
    """(alpha, rmse): the alpha with the lowest leave-one-race-out mse on (X, y), ties to the
    larger alpha, and that rmse."""
    best = None
    for a in sorted(alphas, reverse=True):
        err = [float(y[i] - _ridge_predict(_ridge(np.delete(X, i, 0), np.delete(y, i), a), X[i:i + 1])[0])
               for i in range(len(y))]
        mse = float(np.mean(np.square(err)))
        if best is None or mse < best[1] - 1e-15:
            best = (a, mse)
    return best[0], math.sqrt(best[1])


def circuit_features_prediction(train: pd.DataFrame, row: dict, alphas=ALPHAS,
                                min_races: int = MIN_FEATURE_RACES) -> dict | None:
    """Ridge on FEATURE_COLUMNS over the as-of rows with every covariate; None with fewer than
    `min_races` of them or a held-out race missing a covariate."""
    cols = list(FEATURE_COLUMNS)
    t = train.dropna(subset=cols + ["target"])
    x = np.array([[float(row.get(c, np.nan)) for c in cols]])
    if len(t) < min_races or np.isnan(x).any():
        return None
    X, y = t[cols].to_numpy(dtype=float), t["target"].to_numpy(dtype=float)
    alpha, rmse = ridge_inner_loro(X, y, alphas)
    mean = float(_ridge_predict(_ridge(X, y, alpha), x)[0])
    return {"mean": mean, "sd": rmse, "source": "circuit features", "n_history": int(len(t)), "alpha": alpha}


def practice_evolution_per_lap(practice_model, practice) -> tuple[float, int]:
    """(evolution s/lap, sessions): the mean of the practice fit's evolution_s_per_min times
    the median clean practice lap in minutes; (0, 0) without either."""
    rep = getattr(practice_model, "fit_report", None) or {}
    evo = {s: v for s, v in (rep.get("evolution_s_per_min") or {}).items() if v is not None and not math.isnan(v)}
    if not evo or practice is None or not len(practice):
        return 0.0, 0
    clean = practice[practice["clean"].astype(bool)] if "clean" in practice else practice
    lap_min = float(clean["lap_time_s"].median()) / 60.0
    if math.isnan(lap_min):
        return 0.0, 0
    return float(np.mean(list(evo.values()))) * lap_min, len(evo)


def practice_longrun_prediction(race: RaceParams, practice_model=None, practice=None) -> dict:
    """Physics fuel term plus practice evolution per lap, with the a priori sd."""
    per_lap_kg = FUEL_START_KG / race.total_laps
    evo, n = practice_evolution_per_lap(practice_model, practice)
    sd = math.sqrt((FUEL_EFFECT_SD_S_PER_KG * per_lap_kg) ** 2 + EVOLUTION_SD_S_PER_LAP ** 2)
    return {"mean": -FUEL_EFFECT_S_PER_KG * per_lap_kg + evo, "sd": sd, "n_history": 0,
            "source": f"physics + evolution ({n} session(s))" if n else "physics (no practice evolution)",
            "physics_s_per_lap": -FUEL_EFFECT_S_PER_KG * per_lap_kg, "evolution_s_per_lap": evo}


# --- B: the calibrators --------------------------------------------------------------------------
class FuelPriorCalibrator(ShrinkageCalibrator):
    """The shrinkage calibrator with fuel_s_per_lap (mean, sd) from prediction(): the mean is
    the to_snapshot fit's fuel prior, the sd its tau, and pre_race holds the mean. fit(train)
    takes a CalibrationDataset (its races' targets by race_target) or a history_table frame;
    `features_path` the race_features.csv (default RACE_FEATURES; a frame is taken as is), for
    the circuit and the covariates of a race, the held-out one included (they are pre-race)."""
    name = "fuel_base"
    candidate = None

    def __init__(self, features_path=None, **params):
        path = isinstance(features_path, (str, Path))
        super().__init__(**({"features_path": str(features_path)} if path else {}), **params)
        self.features = _feature_frame(RACE_FEATURES if features_path is None else features_path)
        self.history = history_table({}, {})
        self.fitted = True

    def fit(self, train) -> "FuelPriorCalibrator":
        if isinstance(train, CalibrationDataset):
            train = history_from_races(train.races.values(), self.features)
        self.history = train.reset_index(drop=True)
        self.fitted = True
        return self

    def race_row(self, race: RaceParams) -> dict:
        rid = canonical_race_id(race.race_id)
        known = self.history[self.history["race_id"] == rid]
        if len(known):
            return known.iloc[0].to_dict()
        return race_rows([rid], {rid: race.total_laps}, self.features).iloc[0].to_dict()

    def predict(self, race: RaceParams, config: SimConfig, practice_model=None, practice=None) -> dict:
        raise NotImplementedError

    def prediction(self, race: RaceParams, config: SimConfig | None = None, practice_model=None, practice=None) -> dict:
        """{mean, sd, source, n_history, ...}: the candidate's fuel prior for `race`, the config's
        when the candidate has nothing (no as-of history, too few feature races, an id that is
        not <season>_<round>_<slug>). sd floored at MIN_SD."""
        config = SimConfig() if config is None else config
        try:
            parse_race_id(canonical_race_id(race.race_id))
        except ValueError:
            return _config_prediction(config, "config (race id has no season)")
        p = self.predict(race, config, practice_model, practice)
        if p is None or p["sd"] is None or math.isnan(p["sd"]) or math.isnan(p["mean"]):
            return _config_prediction(config, "config (no history)")
        return {**p, "sd": max(float(p["sd"]), MIN_SD)}

    def calibrate(self, ctx):
        out = super().calibrate(ctx)
        p = self.prediction(ctx.race, ctx.config, ctx.practice_model, ctx.practice)
        taus = {**out.diagnostics["taus"], "fuel_s_per_lap": p["sd"]}
        return CalibrationOutput(factors={**out.factors, "fuel_s_per_lap": (p["mean"], p["sd"])},
                                 valid_until_lap=out.valid_until_lap, source=self.name,
                                 diagnostics={"prior": self.name, "taus": taus,
                                              "fuel_prior": {k: v for k, v in p.items()
                                                             if isinstance(v, (int, float, str))}})

    def report(self) -> dict:
        return {**super().report(), "n_races": int(len(self.history))}


@CALIBRATORS.register("fuel_config")
class FuelConfigCalibrator(ShrinkageCalibrator):
    """SPEC 0020 reference: the config fuel prior (-0.045, sd 0.008), the shrinkage output as is."""
    name = "fuel_config"
    candidate = "config"

    def __init__(self, features_path=None, **params):
        super().__init__(**params)

    def fit(self, train):
        self.fitted = True
        return self

    def prediction(self, race, config=None, practice_model=None, practice=None) -> dict:
        return _config_prediction(SimConfig() if config is None else config, "config")


@CALIBRATORS.register("fuel_circuit_history")
class CircuitHistoryCalibrator(FuelPriorCalibrator):
    """The race's circuit in earlier seasons, shrunk toward the grand mean (K0 = 1)."""
    name = "fuel_circuit_history"
    candidate = "circuit_history"
    learned = True

    def __init__(self, k0: float = K0, features_path=None, **params):
        super().__init__(features_path=features_path, k0=float(k0), **params)
        self.k0 = float(k0)

    def predict(self, race, config, practice_model=None, practice=None):
        return circuit_history_prediction(as_of(self.history, race.race_id), self.race_row(race)["circuit"], self.k0)


@CALIBRATORS.register("fuel_circuit_features")
class CircuitFeaturesCalibrator(FuelPriorCalibrator):
    """Ridge on the race_features.csv covariates, alpha by inner LORO; MIN_FEATURE_RACES races."""
    name = "fuel_circuit_features"
    candidate = "circuit_features"
    learned = True

    def predict(self, race, config, practice_model=None, practice=None):
        return circuit_features_prediction(as_of(self.history, race.race_id), self.race_row(race))


@CALIBRATORS.register("fuel_practice_longrun")
class PracticeLongrunCalibrator(FuelPriorCalibrator):
    """Physics fuel term plus practice evolution per lap; trains on nothing."""
    name = "fuel_practice_longrun"
    candidate = "practice_longrun"

    def predict(self, race, config, practice_model=None, practice=None):
        return practice_longrun_prediction(race, practice_model, practice)


class HindsightFuel(FuelPriorCalibrator):
    """Each race's own target as its fuel prior (its fit sd as the tau): the bound the report's
    sentence reads, not a candidate (it sees the held-out race) and not registered."""
    name = "fuel_hindsight"
    candidate = ORACLE

    def predict(self, race, config, practice_model=None, practice=None):
        row = self.history[self.history["race_id"] == canonical_race_id(race.race_id)]
        if not len(row) or pd.isna(row["target"].iloc[0]):
            return None
        return {"mean": float(row["target"].iloc[0]), "sd": float(row["target_sd"].iloc[0]),
                "source": "hindsight", "n_history": 1}


CALIBRATOR_OF = {"config": "fuel_config", "circuit_history": "fuel_circuit_history",
                 "circuit_features": "fuel_circuit_features", "practice_longrun": "fuel_practice_longrun"}


def make_calibrator(candidate: str, features_path=None):
    if candidate == ORACLE:
        return HindsightFuel(features_path=features_path)
    return CALIBRATORS.get(CALIBRATOR_OF[candidate])(features_path=features_path)


# --- C: leave-one-race-out ---------------------------------------------------------------------------
def _physics_prior(data):
    from .engine import Simulator
    if data.practice is None or data.practice.empty:
        return None
    return Simulator(data.frame, data.config.model_copy(update={"fit_window": "pre_race"}), data.practice).physics_prior()


def _fold_calibrator(candidate, history, race_id, features_path):
    cal = make_calibrator(candidate, features_path)
    train = history if candidate == ORACLE else history[history["race_id"] != race_id]
    return cal.fit(train)


def predictions(races: dict, history: pd.DataFrame, candidates=CANDIDATES, features_path=None) -> pd.DataFrame:
    """Per held-out race and candidate (PREDICTION_COLUMNS): the pre-race fuel prior (mean, sd)
    fitted on the other races, as-of, against the race's target."""
    rows = []
    for rid, data in races.items():
        row = history[history["race_id"] == rid].iloc[0]
        raw = _physics_prior(data)
        practice = data.practice.assign(lap=np.nan) if data.practice is not None and len(data.practice) else None
        for c in candidates:
            cal = _fold_calibrator(c, history, rid, features_path)
            p = cal.prediction(data.race, data.config, raw, practice)
            err = p["mean"] - row["target"]
            rows.append({"race_id": rid, "candidate": c, "season": int(row["season"]), "circuit": row["circuit"],
                         "mean": p["mean"], "sd": p["sd"], "source": p["source"], "n_history": p["n_history"],
                         "target": row["target"], "target_sd": row["target_sd"], "err": err,
                         "covered": bool(abs(err) <= p["sd"])})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def fold_tables(data, cal, laps_k, windows=WINDOWS, n_samples=1) -> dict[str, pd.DataFrame]:
    """The held-out race under one fitted calibrator: G rows (calibration.midrace) per window
    and stop-gain rows (calibration.stop_gain against the race's own full-window fit)."""
    from . import calibration
    from .engine import Simulator
    cfg = data.config.model_copy(update={"calibrator": ComponentSpec(name=cal.name)})

    def factory(window):
        return Simulator(data.frame, cfg.model_copy(update={"fit_window": window}), data.practice, calibrator=cal)

    g = calibration.midrace(factory, data.frame, data.race, laps_k=laps_k, windows=list(windows),
                            results=data.results, n_samples=n_samples)
    hind = Simulator(data.frame, data.config.model_copy(update={"fit_window": "full"}), data.practice).degradation
    sg = pd.concat([calibration.stop_gain(factory(w), data.frame, data.race, hind, laps_k) for w in windows],
                   ignore_index=True)
    return {"g": g, "stop_gain": sg}


def downstream(races: dict, history: pd.DataFrame, laps_k, candidates=CANDIDATES, features_path=None,
               n_samples=1, log=None) -> dict[str, dict[str, pd.DataFrame]]:
    """{candidate: {"g", "stop_gain"}} over every held-out race (each from its own fold)."""
    out = {}
    for c in candidates:
        parts = {"g": [], "stop_gain": []}
        for rid, data in races.items():
            cal = _fold_calibrator(c, history, rid, features_path)
            t = fold_tables(data, cal, laps_k(data), n_samples=n_samples)
            for k in parts:
                parts[k].append(t[k].assign(candidate=c))
            if log:
                log(f"  {c} {rid}")
        out[c] = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    return out


def target_selection(pred: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per candidate: held-out target mae (one value per race, so race-weighted) with its CI,
    the paired difference against config with its race-cluster 95 percent CI, 68 percent
    coverage, and whether it passes (difference below 0 with the CI's upper end below 0)."""
    from .cross_race import _boot_mean
    wide = pred.assign(abs_err=pred["err"].abs()).pivot(index="race_id", columns="candidate", values="abs_err")
    rows = []
    for c in [c for c in (*CANDIDATES, ORACLE) if c in wide]:
        p = pred[pred["candidate"] == c]
        diff = wide[c] - wide[REFERENCE]
        lo, hi = _boot_mean(diff, n_boot) if c != REFERENCE else (math.nan, math.nan)
        mlo, mhi = _boot_mean(wide[c], n_boot)
        role = "reference" if c == REFERENCE else "bound (not a candidate)" if c == ORACLE else "candidate"
        fallback = float(p["source"].str.startswith("config").mean()) if c != REFERENCE else math.nan
        rows.append({"candidate": c, "role": role, "n_races": int(wide[c].notna().sum()),
                     "mae": float(wide[c].mean()), "mae_lo": mlo, "mae_hi": mhi, "bias": float(p["err"].mean()),
                     "diff": float(diff.mean()) if c != REFERENCE else math.nan, "diff_lo": lo, "diff_hi": hi,
                     "coverage_68": float(p["covered"].mean()), "mean_sd": float(p["sd"].mean()),
                     "config_fallback_share": fallback,
                     "passes": role == "candidate" and bool(diff.mean() < 0 and hi < 0)})
    return pd.DataFrame(rows)


def _g_rows(d, window, lo, hi):
    g = d["g"]
    return g[(g["window"] == window) & g["horizon"].between(lo, hi)]


def _race_weighted(rows, value):
    v = rows[["race_id", value]].dropna()
    if not len(v):
        return math.nan, math.nan
    return float(v.groupby("race_id")[value].mean().mean()), float(v[value].abs().groupby(v["race_id"]).mean().mean())


def f_table(sel: pd.DataFrame, down: dict | None, available: dict) -> pd.DataFrame:
    """SPEC 0020 F: candidate, held-out mae, coverage, G bias at 20-40 and 50-60 per window
    (race-weighted mean err_offset_s), stop-gain mae (to_snapshot, race-weighted)."""
    rows = []
    for _, r in sel.iterrows():
        c = r["candidate"]
        row = {"candidate": c, "role": r["role"], "available": available.get(c, True), "held_out_mae": r["mae"],
               "mae_diff": r["diff"], "mae_diff_lo": r["diff_lo"], "mae_diff_hi": r["diff_hi"],
               "coverage_68": r["coverage_68"], "passes": r["passes"]}
        d = (down or {}).get(c)
        for w in WINDOWS:
            for name, (lo, hi) in G_RANGES.items():
                row[f"g_bias_{name}_{w}"] = _race_weighted(_g_rows(d, w, lo, hi), "err_offset_s")[0] if d else math.nan
        if d:
            s = d["stop_gain"]
            row["stop_gain_mae"] = _race_weighted(s[s["window"] == SG_WINDOW], "err_s")[1]
        else:
            row["stop_gain_mae"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def by_horizon(down: dict) -> pd.DataFrame:
    """Per candidate, window and 10-lap horizon bucket: race-weighted G bias and mae_offset."""
    from .calibration import BUCKET_LAPS
    rows = []
    for c, d in down.items():
        g = d["g"].assign(bucket=(d["g"]["horizon"] // BUCKET_LAPS * BUCKET_LAPS).astype(int))
        for (w, b), part in g.groupby(["window", "bucket"]):
            bias, mae = _race_weighted(part, "err_offset_s")
            rows.append({"candidate": c, "window": w, "horizon_bucket": int(b), "n": int(len(part)),
                         "n_races": int(part["race_id"].nunique()), "bias_offset_s": bias, "mae_offset_s": mae})
    return pd.DataFrame(rows)


def choose(f: pd.DataFrame) -> str:
    """Among the candidates that pass, the lowest held-out mae; a tie to the lower stop-gain
    mae, then the CANDIDATES order; else config."""
    t = f[f["passes"].astype(bool) & (f["role"] == "candidate")]
    if not len(t):
        return REFERENCE
    t = t.assign(_m=t["held_out_mae"].round(6), _s=t["stop_gain_mae"].fillna(math.inf).round(6),
                 _o=[CANDIDATES.index(c) for c in t["candidate"]])
    return str(t.sort_values(["_m", "_s", "_o"]).iloc[0]["candidate"])


def prior_share(f: pd.DataFrame, window="pre_race", horizons="50-60", best=None) -> dict:
    """How much of the config's absolute G bias the race-pace prior accounts for: the share the
    hindsight prior removes, and the share `best` removes."""
    col = f"g_bias_{horizons}_{window}"
    get = dict(zip(f["candidate"], f[col]))
    base = abs(get.get(REFERENCE, math.nan))
    out = {"window": window, "horizons": horizons, "config_abs_bias_s": base,
           "hindsight_abs_bias_s": abs(get.get(ORACLE, math.nan)), "best": best,
           "best_abs_bias_s": abs(get.get(best, math.nan)) if best else math.nan}
    for k in ("hindsight", "best"):
        v = out[f"{k}_abs_bias_s"]
        out[f"{k}_share"] = (base - v) / base if base and not math.isnan(base) and not math.isnan(v) else math.nan
    return out


# --- synthetic histories -----------------------------------------------------------------------------
# circuit -> (type, laps, lap length km, turns, circuit effect s/lap). The planted race-pace
# slope of a race is SYN_BURN_S / laps (fuel burn: the burn sensitivity in s per race, spread
# over the laps) + SYN_EVOLUTION[type] + the circuit effect + N(0, SYN_RACE_SD^2) per race.
# Practice carries the evolution part as s/min (over SYN_LAP_MIN), the race the whole slope.
SYN_CIRCUITS = {
    "c1": (STREET, 58, 5.3, 19, -0.008), "c2": (STREET, 70, 4.4, 14, 0.006),
    "c3": (PERMANENT, 66, 4.7, 16, 0.0), "c4": (PERMANENT, 52, 5.9, 18, -0.012),
    "c5": (PERMANENT, 56, 5.5, 15, 0.010), "c6": (PERMANENT, 62, 4.9, 12, 0.004),
}
SYN_BURN_S = -3.4
SYN_EVOLUTION = {STREET: -0.012, PERMANENT: -0.004}
SYN_RACE_SD = 0.003
SYN_LAP_MIN = 80.0 / 60.0
SYN_SEASONS = range(2018, 2023)


def synthetic_id(season: int, circuit: str, circuits=None) -> str:
    circuits = SYN_CIRCUITS if circuits is None else circuits
    return f"{season}_{list(circuits).index(circuit) + 1:02d}_synthetic_{circuit}"


def _synthetic_circuit(race_id: str) -> str:
    return parse_race_id(race_id)[2].split("_", 1)[1]


def planted(race_id: str, circuits=None) -> dict:
    """The planted race-pace slope of a synthetic history race and its parts, from its id."""
    circuits = SYN_CIRCUITS if circuits is None else circuits
    c = _synthetic_circuit(race_id)
    ctype, laps, _, _, effect = circuits[c]
    rng = np.random.default_rng(int(hashlib.sha256(f"fuel:{race_id}".encode()).hexdigest()[:8], 16))
    burn, evo = SYN_BURN_S / laps, SYN_EVOLUTION[ctype]
    circuit_slope = burn + evo + effect
    return {"race_id": race_id, "circuit": c, "laps": laps, "burn": burn, "evolution": evo, "effect": effect,
            "circuit_slope": circuit_slope, "slope": circuit_slope + float(rng.normal(0.0, SYN_RACE_SD))}


def synthetic_history_race(race_id: str, circuits=None):
    """cross_race.synthetic_race with the planted slope, laps and practice evolution of the id."""
    from .cross_race import synthetic_race
    p = planted(canonical_race_id(race_id), circuits)
    return synthetic_race(canonical_race_id(race_id), total_laps=p["laps"], fuel=p["slope"], practice_fuel=p["burn"],
                          evolution_s_per_min=p["evolution"] / SYN_LAP_MIN)


def synthetic_features(race_ids, circuits=None) -> pd.DataFrame:
    """race_features rows for synthetic history races: circuit_type, lap_length_km, turns."""
    circuits = SYN_CIRCUITS if circuits is None else circuits
    rows = []
    for rid in race_ids:
        ctype, _, length, turns, _ = circuits[_synthetic_circuit(rid)]
        rows.append({"race_id": rid, "circuit_type": ctype, "lap_length_km": length, "turns": turns})
    return pd.DataFrame(rows)


def synthetic_history(seasons=SYN_SEASONS, circuits=None) -> pd.DataFrame:
    """A history table straight from the planted slopes (target = planted slope, target_sd
    SYN_RACE_SD), no lap frames: for the calibrators' own tests."""
    circuits = SYN_CIRCUITS if circuits is None else circuits
    ids = [synthetic_id(s, c, circuits) for s in seasons for c in circuits]
    p = {r: planted(r, circuits) for r in ids}
    return history_table({r: (p[r]["slope"], SYN_RACE_SD) for r in ids}, {r: p[r]["laps"] for r in ids},
                         synthetic_features(ids, circuits))


def recovery_table(history: pd.DataFrame, races=None, k0: float = K0, circuits=None) -> pd.DataFrame:
    """Planted against recovered circuit effects: per circuit the planted circuit slope and its
    effect (less the mean over circuits), the mean measured target, and circuit_history's
    prediction for a race the season after the last, with its effect (less its grand mean)."""
    circuits = SYN_CIRCUITS if circuits is None else circuits
    nxt = int(history["season"].max()) + 1
    rows = []
    for c in circuits:
        p = planted(synthetic_id(nxt, c, circuits), circuits)
        h = history[history["circuit"] == f"synthetic_{c}"]
        pred = circuit_history_prediction(as_of(history, synthetic_id(nxt, c, circuits)), f"synthetic_{c}", k0)
        rows.append({"circuit": c, "races": int(len(h)), "planted_slope": p["circuit_slope"],
                     "measured_mean": float(h["target"].mean()), "recovered_slope": pred["mean"],
                     "recovered_sd": pred["sd"], "grand_mean": pred["grand_mean"]})
    t = pd.DataFrame(rows)
    t["planted_effect"] = t["planted_slope"] - t["planted_slope"].mean()
    t["recovered_effect"] = t["recovered_slope"] - t["grand_mean"]
    t["error"] = t["recovered_slope"] - t["planted_slope"]
    return t


# --- running and the report --------------------------------------------------------------------------
def load_races(race_ids, loader, profile) -> tuple[dict, list]:
    """({race_id: RaceData} of the eligible races, eligibility records)."""
    from . import calibration
    races, elig = {}, []
    for r in race_ids:
        rid = canonical_race_id(r)
        data = loader(rid, profile)
        e = calibration.eligibility(data.frame, data.race)
        elig.append({"race_id": rid, "eligible": e["eligible"], "reasons": "; ".join(e["reasons"])})
        if e["eligible"]:
            races[rid] = data
    return races, elig


def run(race_ids, out=OUT_DIR, profile=None, loader=None, features_path=RACE_FEATURES, downstream_on=True,
        n_boot=N_BOOT, n_samples=1, log=print, truth=None) -> Path:
    """Targets, LORO predictions, the downstream G and stop-gain folds, and the report under
    `out`. Refuses a synthetic race id under reports/calibration/ (guard_committed)."""
    from .compute import load_profile
    from .cross_race import is_synthetic, md_table
    from . import jobs
    guard_committed(out, list(race_ids))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    profile = load_profile("local") if profile is None else profile
    if loader is None:
        from .cross_race import load_race as loader
    races, elig = load_races(race_ids, loader, profile)
    features = _feature_frame(features_path)
    history = history_from_races(races.values(), features)
    history.to_csv(out / "history.csv", index=False)
    with_feats = int(history.dropna(subset=list(FEATURE_COLUMNS) + ["target"]).shape[0])
    available = {c: True for c in CANDIDATES}
    available["circuit_features"] = with_feats >= MIN_FEATURE_RACES
    cands = [c for c in CANDIDATES if available[c]]
    pred = predictions(races, history, (*cands, ORACLE), features_path)
    pred.to_csv(out / "predictions.csv", index=False)
    sel = target_selection(pred, n_boot)
    down = None
    if downstream_on:
        down = downstream(races, history, lambda d: jobs.laps_k_for(d.frame, d.race.total_laps, profile),
                          (*cands, ORACLE), features_path, n_samples, log)
        pd.concat([d["g"] for d in down.values()], ignore_index=True).to_csv(out / "g_rows.csv", index=False)
        pd.concat([d["stop_gain"] for d in down.values()], ignore_index=True).to_csv(out / "stop_gain.csv", index=False)
        by_horizon(down).to_csv(out / "g_by_horizon.csv", index=False)
    f = f_table(sel, down, available)
    chosen = choose(f)
    f.to_csv(out / "f_table.csv", index=False)
    sel.assign(chosen=sel["candidate"] == chosen).to_csv(out / "selection.csv", index=False)
    best = next((c for c in f.sort_values("held_out_mae")["candidate"] if c not in (REFERENCE, ORACLE)), None)
    share = prior_share(f, best=best)

    synthetic = [r for r in races if is_synthetic(r)]
    md = ["# Per-circuit race-pace prior for the fuel term (SPEC 0020)", "",
          f"{len(elig)} race(s) considered, {len(races)} eligible; seasons "
          f"{int(history['season'].min())}-{int(history['season'].max())}, {history['circuit'].nunique()} circuit(s).", ""]
    if synthetic:
        md += [f"**Synthetic histories ({len(synthetic)} races): a pipeline test, not evidence. The defaults stay.**", ""]
    md += ["## Target (A)", "", "Full-window linear_fuel fuel slope on free-air rows, per race, with its fit sd.", "",
           md_table(history[["race_id", "circuit", "total_laps", "target", "target_sd"]], 4), ""]
    md += ["## Candidates (B)", "",
           f"circuit_history: K0 = {K0:g}. circuit_features: "
           + ("available" if available["circuit_features"] else
              f"**unavailable**: {with_feats} eligible race(s) with every covariate ({', '.join(FEATURE_COLUMNS)}), "
              f"fewer than {MIN_FEATURE_RACES}")
           + f". practice_longrun: -{FUEL_EFFECT_S_PER_KG:g} s/kg x {FUEL_START_KG:g} kg / laps plus practice "
           f"evolution per lap, sd from {FUEL_EFFECT_SD_S_PER_KG:g} s/kg and {EVOLUTION_SD_S_PER_LAP:g} s/lap. "
           "Learned candidates see only races strictly before the held-out race by (season, round).", ""]
    if truth is not None:
        md += ["## Planted against recovered circuit effect (synthetic)", "",
               "circuit_history's prediction for each circuit the season after the last, on every race "
               "(planted effect: the circuit's planted slope less the mean over circuits; recovered: the "
               "prediction less its grand mean).", "", md_table(truth, 4), ""]
    md += ["## F: selection (C)", "",
           "Held-out mae: |predicted fuel prior - target| per race at k 0 (one value per race, so race-weighted); "
           "coverage: share with |err| <= the predicted sd. G bias: race-weighted mean err_offset_s at the "
           "horizons, per window. Stop-gain mae: to_snapshot, race-weighted. Rule: a candidate replaces config "
           "only if its held-out mae is lower with the race-cluster 95% CI of the paired difference below 0; "
           "among those that pass the lowest mae, ties to the lower stop-gain mae. hindsight is each race's own "
           "target, a bound, not a candidate.", ""]
    show = f.assign(mae_diff=[f"{d:.4f} [{a:.4f}, {b:.4f}]" if not math.isnan(d) else "-"
                              for d, a, b in zip(f["mae_diff"], f["mae_diff_lo"], f["mae_diff_hi"])])
    md += [md_table(show.drop(columns=["mae_diff_lo", "mae_diff_hi"]), 4), ""]
    md += ["Target selection detail:", "", md_table(sel, 4), ""]
    verdict = (f"**{chosen} replaces config**: its predictive sd becomes the to_snapshot fuel tau per race (D)"
               if chosen != REFERENCE else "**config stays** (no candidate passes)")
    if synthetic:
        verdict += "; synthetic, so nothing is adopted"
    md += [f"Decision: {verdict}.", ""]
    if down:
        md += ["## G by horizon (race-weighted bias and mae_offset, s)", "", md_table(by_horizon(down), 2), ""]
        pd.DataFrame([share]).to_csv(out / "prior_share.csv", index=False)
    if down and not synthetic:   # on synthetic histories the share holds by construction: no sentence
        sentence = (f"Simulator section: on the eligible races "
                    f"the config fuel prior leaves a {share['window']} G bias of {share['config_abs_bias_s']:.1f} s at "
                    f"horizons {share['horizons']}; each race's own race-pace slope as the prior brings it to "
                    f"{share['hindsight_abs_bias_s']:.1f} s, so about {100 * share['hindsight_share']:.0f} percent of "
                    f"that absolute error was the race-pace prior, and the best candidate ({best}) removes "
                    f"{100 * share['best_share']:.0f} percent.")
        md += ["## Sentence (F)", "", sentence, ""]
        (out / "sentence.txt").write_text(sentence + "\n")
    elif down:
        md += ["## Sentence (F)", "", "Written on the real run only: on synthetic histories the race-pace "
               "prior's share of the absolute error holds by construction (prior_share.csv keeps it as a "
               "pipeline check).", ""]
    md += ["## Eligibility", "", md_table(pd.DataFrame(elig)), ""]
    path = out / "summary.md"
    path.write_text("\n".join(md))
    return path


def run_synthetic(out, seasons=SYN_SEASONS, circuits=None, profile=None, **kw) -> Path:
    """run() on synthetic history races (synthetic_history_race), their features in a csv under
    `out`, and the planted-against-recovered table."""
    circuits = SYN_CIRCUITS if circuits is None else circuits
    out = Path(out)
    ids = [synthetic_id(s, c, circuits) for s in seasons for c in circuits]
    guard_committed(out, ids)
    out.mkdir(parents=True, exist_ok=True)
    feats = out / "race_features.csv"
    synthetic_features(ids, circuits).to_csv(feats, index=False)
    races = {r: synthetic_history_race(r, circuits) for r in ids}
    history = history_from_races(races.values(), feats)
    truth = recovery_table(history, circuits=circuits)
    truth.to_csv(out / "recovery.csv", index=False)
    return run(ids, out, profile, loader=lambda r, _p: races[r], features_path=feats, truth=truth, **kw)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m src.sim.calibrate fuel", description=__doc__.split("\n\n")[0])
    ap.add_argument("--races", nargs="+", default=["2021_04_spain"])
    ap.add_argument("--race-features", default=str(RACE_FEATURES))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--synthetic", action="store_true", help="synthetic history races instead (pipeline test)")
    ap.add_argument("--no-downstream", action="store_true", help="targets and predictions only, no G or stop gain")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    from .compute import load_profile
    profile = load_profile(args.profile) if args.profile else None
    kw = {"downstream_on": not args.no_downstream, "n_boot": args.n_boot}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if args.synthetic:
            path = run_synthetic(args.out, profile=profile, **kw)
        else:
            path = run(args.races, args.out, profile, features_path=args.race_features, **kw)
    print(f"written to {path}")
    return 0
