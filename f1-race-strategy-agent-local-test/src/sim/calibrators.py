"""Calibrators: physics, shrinkage, and learned candidates (SPEC H v2).

A calibrator maps what is known at a calibration point k to the simulator's factors, each
as (mean, sd), plus an optional correction table (contracts.CalibrationOutput). One
interface serves the pre-race predictor (k = 0) and the online calibrator (k > 0):

    cal = CALIBRATORS.get(name)(**params)
    cal.fit(train: CalibrationDataset) -> cal
    cal.calibrate(context: CalibrationContext) -> CalibrationOutput
    cal.save(path); Calibrator.load(path); cal.report() -> dict

Simulator.fitted_for builds the context from the snapshot and the frames, never from laps
after k, and plugs the output in:
  (a) degradation: base_delta_s[c] (added to the practice base), wear_s_per_lap[c] and
      fuel_s_per_lap are the prior means, their sds the prior taus, of the to_snapshot
      shrinkage fit on race laps up to k; so is hinge_s_per_lap[c] under linear_fuel_hinge
      (the practice hinge, 0 where practice dropped it). When every degradation factor has sd 0 the prior
      is held as is: no fit, and hazard and ref_lap_s from practice, as under pre_race.
      The pre_race window always holds it.
  (b) pit_loss_s, sc_lap_factor, vsc_lap_factor, sc_pit_loss_factor, lap1_extra_s: one draw
      per sample when their sd is above 0 (clipped at physical bounds), drawn after the
      events and before the degradation draw, shared by every plan; their mean when the sd
      is 0. A factor the output leaves out takes the config or race registry value.
  (c) the correction table, if any, is added to green lap times by (compound, floor(age), lap).
The full window (the world) never calls a calibrator.

Registry:
  "none"       physics plus config values: the practice fit with the config base delta
               and fuel prior, held at every k. No race lap moves it.
  "shrinkage"  the to_snapshot fit of #13: the same prior with the SimConfig.shrinkage taus,
               penalised least squares on race laps up to k. The default, and the baseline
               a learned calibrator must beat.
  "ridge", "gbm", "mlp"
               learned candidates, registered stubs until the multi-race dataset exists:
               fit raises NotImplementedError.

Cadence (SimConfig.calibration_every = N): the calibration point for a snapshot at lap k
is the largest multiple of N at or below k, moved up to the latest refresh event since
then (SC or VSC start or end, rain onset, a track temperature change above
recalibrate_temp_delta_c). N = None is pre-race only: k0 = 0 and no event refreshes.

Validation: build_dataset() makes one row per (race, k) with features known at k and the
race's hindsight factors as targets; loro(dataset, name) runs leave-one-race-out and
leave-one-season-out on SPEC G's pre_race and to_snapshot columns, with per factor rmse and
68 percent interval coverage. loro(dataset, name, fold) runs one fold (jobs.py h_fold).
"""
import copy
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import (COMPOUND_FACTORS, RACE_FACTORS, CalibrationOutput, CorrectionTable, RaceParams,  # noqa: F401
                         Snapshot, factor_name, split_factor)
from .config import HINGE_TAU_S_PER_LAP, ComponentSpec, SimConfig
from .events import Scenario
from .race_ids import canonical_race_id, race_season
from .registry import Registry

CALIBRATORS = Registry("calibrator")
NEEDS_DATA = "needs the multi-race dataset"
REFERENCE = ("none", "shrinkage")     # always reported beside any learned class
DEGRADATION_FACTORS = ("base_delta_s", "wear_s_per_lap", "fuel_s_per_lap", "hinge_s_per_lap")
TAU_KEYS = {"base_delta_s": "base_s", "wear_s_per_lap": "wear_s_per_lap", "fuel_s_per_lap": "fuel_s_per_lap",
            "hinge_s_per_lap": "hinge_s_per_lap"}
# factors drawn per sample, with their physical bounds
EVENT_BOUNDS = {"pit_loss_s": (0.0, np.inf), "sc_lap_factor": (1.0, np.inf), "vsc_lap_factor": (1.0, np.inf),
                "sc_pit_loss_factor": (0.0, 1.0), "lap1_extra_s": (0.0, np.inf)}


# --- context and interface ---------------------------------------------------------
@dataclass
class CalibrationContext:
    """What is known at calibration point k. Built by Simulator.fitted_for; nothing after lap k.

    laps_so_far: race frame rows with lap <= k, every driver (less the snapshot driver
    under exclude_snapshot_driver). residuals_so_far: lap time less the config prior's
    prediction on green non-pit laps after lap 1 up to k (lap, driver, compound, tyre_age,
    resid_s). practice_model: the practice fit with no base delta (None without practice).
    pre_race_output: this race's lap-0 output (None when k is 0). previous_output: the
    output at the latest earlier calibration point computed, else pre_race_output.
    """
    k: int
    config: SimConfig
    race: RaceParams
    valid_until_lap: int
    laps_so_far: pd.DataFrame
    practice: pd.DataFrame | None = None
    practice_model: object = None
    residuals_so_far: pd.DataFrame | None = None
    weather_so_far: pd.DataFrame | None = None
    sc_history: Scenario = field(default_factory=Scenario)
    race_features: dict | None = None
    pre_race_output: CalibrationOutput | None = None
    previous_output: CalibrationOutput | None = None


class Calibrator:
    name = "base"
    learned = False

    def __init__(self, **params):
        self.params = params
        self.fitted = not self.learned

    def fit(self, train: "CalibrationDataset") -> "Calibrator":
        self.fitted = True
        return self

    def calibrate(self, context: CalibrationContext) -> CalibrationOutput:
        raise NotImplementedError

    def report(self) -> dict:
        return {"name": self.name, "learned": self.learned, "fitted": self.fitted, "params": dict(self.params)}

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"name": self.name, "params": self.params}, f, indent=1)

    @staticmethod
    def load(path) -> "Calibrator":
        with open(path) as f:
            data = json.load(f)
        return CALIBRATORS.get(data["name"])(**data.get("params", {}))


def physics_means(ctx: CalibrationContext) -> dict[str, float]:
    """The config prior: per practice compound the config base delta and the practice wear
    (and hinge, for a model with one: 0 where practice dropped it), and the config fuel
    slope. Without practice only the fuel slope."""
    cfg, raw = ctx.config, ctx.practice_model
    out = {}
    if raw is not None:
        hinges = getattr(raw, "hinges", None)
        for c, (_, wear) in raw.coefs.items():
            out[factor_name("base_delta_s", c)] = float(cfg.pre_race_base_delta_s)
            out[factor_name("wear_s_per_lap", c)] = float(wear)
            if hinges is not None:
                out[factor_name("hinge_s_per_lap", c)] = float(hinges[c])
    out["fuel_s_per_lap"] = float(cfg.pre_race_fuel_slope_s_per_lap)
    return out


@CALIBRATORS.register("none")
class NoneCalibrator(Calibrator):
    """Physics plus config values: the config prior with sd 0, so it is held at every k."""
    name = "none"

    def calibrate(self, ctx):
        if ctx.practice_model is None:
            raise ValueError("calibrator 'none' holds the practice fit and needs practice laps")
        return CalibrationOutput(factors={k: (v, 0.0) for k, v in physics_means(ctx).items()},
                                 valid_until_lap=ctx.valid_until_lap, source=self.name)


@CALIBRATORS.register("shrinkage")
class ShrinkageCalibrator(Calibrator):
    """The #13 fit: the config prior with the SimConfig.shrinkage taus as sds. The simulator
    then fits race laps up to k toward it (plug-in (a)); there is no correction table. A
    hinge prior (linear_fuel_hinge) takes tau shrinkage["hinge_s_per_lap"], whose key the
    diagnostics carry only then."""
    name = "shrinkage"

    def calibrate(self, ctx):
        tau = {"hinge_s_per_lap": HINGE_TAU_S_PER_LAP, **ctx.config.shrinkage}
        means = physics_means(ctx)
        factors = {k: (v, float(tau[TAU_KEYS[split_factor(k)[0]]])) for k, v in means.items()}
        hinged = any(split_factor(k)[0] == "hinge_s_per_lap" for k in means)
        taus = {k: v for k, v in tau.items() if hinged or k != "hinge_s_per_lap"}
        return CalibrationOutput(factors=factors, valid_until_lap=ctx.valid_until_lap, source=self.name,
                                 diagnostics={"prior": "config", "taus": taus})


class _LearnedStub(Calibrator):
    """A learned class, registered so tables and jobs can name it; not trainable yet."""
    learned = True

    def fit(self, train):
        raise NotImplementedError(f"calibrator {self.name!r} {NEEDS_DATA}")

    def calibrate(self, ctx):
        raise NotImplementedError(f"calibrator {self.name!r} {NEEDS_DATA}")


@CALIBRATORS.register("ridge")
class RidgeCalibrator(_LearnedStub):
    """Per-target ridge with predictive sd. The first learned candidate (pipeline proof at 20 races)."""
    name = "ridge"


@CALIBRATORS.register("gbm")
class GbmCalibrator(_LearnedStub):
    """Per-target gradient boosting with quantile or NGBoost-style sd."""
    name = "gbm"


@CALIBRATORS.register("mlp")
class MlpCalibrator(_LearnedStub):
    """Gaussian-NLL multi-output net, seed ensemble."""
    name = "mlp"


# --- cadence -----------------------------------------------------------------------
def _run_edges(laps, k):
    """Starts (first lap of a run) and ends (first lap after a run, if <= k) of the runs in `laps`."""
    s = {int(x) for x in laps if x <= k}
    return sorted({x for x in s if x - 1 not in s} | {x + 1 for x in s if x + 1 not in s and x + 1 <= k})


def refresh_laps(snapshot: Snapshot, weather=None, temp_delta_c=3.0, since=0) -> list[int]:
    """Laps in (since, k] whose events force a recalibration: SC or VSC start or end, rain
    onset, and a track temperature (from a lap_weather frame) more than temp_delta_c away
    from the value at the last calibration point."""
    k = snapshot.laps_completed
    rain = snapshot.weather.rain_laps
    if weather is not None and "rain" in weather:
        rain = sorted(set(rain) | {int(x) for x in weather.index[weather["rain"].astype(bool)] if x <= k})
    events = set(_run_edges(snapshot.sc_laps, k)) | set(_run_edges(snapshot.vsc_laps, k))
    events |= {x for x in _run_edges(rain, k) if x in set(rain)}   # onsets only
    if weather is not None and "track_temp_c" in weather and k > since:
        temp = weather["track_temp_c"].astype(float)
        seen = temp[temp.index <= since].dropna()
        ref = float(seen.iloc[-1]) if len(seen) else np.nan
        for lap, t in temp[(temp.index > since) & (temp.index <= k)].items():
            if np.isnan(t):
                continue
            if np.isnan(ref):
                ref = t
            elif abs(t - ref) > temp_delta_c:
                events.add(int(lap))
                ref = t
    return sorted(x for x in events if since < x <= k)


def calibration_point(snapshot: Snapshot, every: int | None, weather=None, temp_delta_c=3.0) -> int:
    """k0 for a snapshot at lap k: the largest multiple of `every` at or below k, or the latest
    refresh event after it. `every` None: 0, always."""
    k = snapshot.laps_completed
    if every is None:
        return 0
    base = k // every * every
    return max([base, *refresh_laps(snapshot, weather, temp_delta_c, since=base)])


def valid_until(k0: int, every: int | None, total_laps: int) -> int:
    return total_laps if every is None else min(total_laps, (k0 // every + 1) * every - 1)


# --- plug-in helpers (engine side) ---------------------------------------------------
def prior_model(raw, factors: dict, config: SimConfig):
    """The degradation prior from a practice fit with no base delta and the output factors.

    Practice-only compounds get base + base_delta_s[c] (config value when left out) and
    wear_s_per_lap[c]; the fuel slope is fuel_s_per_lap, and a hinge model's hinges
    hinge_s_per_lap[c]. With the config values this is exactly the fit with base_delta =
    pre_race_base_delta_s, report included."""
    m = copy.copy(raw)
    rep = raw.fit_report
    practice = [c for c in raw.coefs if rep["n_race_rows_per_compound"].get(c, 0) == 0]
    deltas = {c: float(factors.get(factor_name("base_delta_s", c), (config.pre_race_base_delta_s,))[0])
              for c in practice}
    m.coefs = {c: (b + deltas[c] if c in deltas else b, float(factors.get(factor_name("wear_s_per_lap", c), (w,))[0]))
               for c, (b, w) in raw.coefs.items()}
    m.fit_report = {**rep, "base_delta_s": deltas, "base_delta_source": "prior" if practice else None}
    if hasattr(m, "fuel_slope") and "fuel_s_per_lap" in factors:
        m.fuel_slope = float(factors["fuel_s_per_lap"][0])
    if getattr(raw, "hinges", None) is not None:
        m.hinges = {c: float(factors.get(factor_name("hinge_s_per_lap", c), (h,))[0]) for c, h in raw.hinges.items()}
        m.fit_report["hinge"] = {c: {**e, "value": m.hinges[c]} for c, e in rep["hinge"].items()}
    return m


def degradation_prior(out: CalibrationOutput, raw, config: SimConfig):
    """(prior model or None, shrinkage taus, fuel prior, held) for plug-in (a)."""
    sds = {k: sd for k, (_, sd) in out.factors.items() if split_factor(k)[0] in DEGRADATION_FACTORS}
    held = bool(sds) and all(sd == 0 for sd in sds.values())
    if not held and any(sd == 0 for sd in sds.values()):
        raise NotImplementedError("a mix of held (sd 0) and free degradation factors")
    taus = dict(config.shrinkage)
    for name, key in TAU_KEYS.items():
        vals = {sd for k, sd in sds.items() if split_factor(k)[0] == name}
        if len(vals) > 1:
            raise NotImplementedError(f"per compound prior sd for {name}")
        if vals:
            taus[key] = vals.pop()
    fuel = float(out.factors.get("fuel_s_per_lap", (config.pre_race_fuel_slope_s_per_lap,))[0])
    prior = prior_model(raw, out.factors, config) if raw is not None else None
    return prior, taus, fuel, held


class EventDraws:
    """Plug-in (b): per sample values of the event factors. `fixed` holds the values used
    when nothing is drawn: the output's mean where its sd is 0, else the config or registry."""

    def __init__(self, out: CalibrationOutput | None, fixed: dict):
        self.fixed = dict(fixed)
        self.names = []
        for name in EVENT_BOUNDS:
            if out is not None and name in out.factors:
                mean, sd = out.factors[name]
                if sd > 0:
                    self.names.append(name)
                else:
                    self.fixed[name] = float(mean)
        self.mean = np.array([out.factors[n][0] for n in self.names]) if self.names else None
        self.sd = np.array([out.factors[n][1] for n in self.names]) if self.names else None
        self.lo = np.array([EVENT_BOUNDS[n][0] for n in self.names])
        self.hi = np.array([EVENT_BOUNDS[n][1] for n in self.names])

    def sample(self, rng) -> dict:
        if not self.names:
            return self.fixed
        v = np.clip(self.mean + self.sd * rng.standard_normal(len(self.names)), self.lo, self.hi)
        return {**self.fixed, **{n: float(x) for n, x in zip(self.names, v)}}


class CorrectionLookup:
    """Plug-in (c): table values by (compound code, floor(age), lap) on the batch layout."""

    def __init__(self, table: CorrectionTable, names, codes, laps):
        self.mean = np.asarray(table.mean, dtype=float)
        self.sd = np.asarray(table.sd, dtype=float)
        rows = np.array([table.compounds.index(c) if c in table.compounds else -1 for c in names])[codes]
        self.ok, self.row = rows >= 0, np.maximum(rows, 0)
        self.lap = np.clip(np.broadcast_to(laps, codes.shape) - 1, 0, self.mean.shape[2] - 1)
        self.has_sd = bool((self.sd > 0).any())

    def add(self, ages, z=0.0):
        a = np.clip(np.floor(ages).astype(int), 0, self.mean.shape[1] - 1)
        v = self.mean[self.row, a, self.lap]
        if self.has_sd:
            v = v + z * self.sd[self.row, a, self.lap]
        return np.where(self.ok, v, 0.0)


@dataclass
class CalibrationState:
    """What a windowed Simulator was calibrated with."""
    output: CalibrationOutput
    k0: int
    held: bool          # degradation held at the prior, no race-lap fit
    practice_model: object = None


def effective_factors(sim, race: RaceParams | None = None) -> dict[str, tuple[float, float]]:
    """The factors a windowed Simulator uses, (mean, sd): degradation from its fitted model
    (sd from fit_report, or the output's sd when held), event factors from the output or
    the config and `race` registry values (sd 0)."""
    st = sim.calibration
    if st is None:
        raise ValueError("effective_factors needs a pre_race or to_snapshot instance from fitted_for")
    deg, raw, out = sim.degradation, st.practice_model, st.output.factors
    sd_rep = (deg.fit_report or {}).get("sd") or {}

    def sd_of(key, rep_name, comp=None):
        if st.held:
            return float(out.get(key, (0.0, 0.0))[1])
        v = sd_rep.get(rep_name)
        return float((v or {}).get(comp, np.nan) if comp is not None else (np.nan if v is None else v))

    eff = {}
    for c, (b, w) in deg.coefs.items():
        if raw is not None and c in raw.coefs:
            eff[factor_name("base_delta_s", c)] = (b - raw.coefs[c][0], sd_of(factor_name("base_delta_s", c), "base_s", c))
        eff[factor_name("wear_s_per_lap", c)] = (w, sd_of(factor_name("wear_s_per_lap", c), "wear_s_per_lap", c))
        if getattr(deg, "hinges", None) is not None:
            key = factor_name("hinge_s_per_lap", c)
            sd = float(out.get(key, (0.0, 0.0))[1]) if st.held else deg.fit_report["hinge"][c]["sd"]
            eff[key] = (float(deg.hinges[c]), sd)
    if hasattr(deg, "fuel_slope"):
        eff["fuel_s_per_lap"] = (float(deg.fuel_slope), sd_of("fuel_s_per_lap", "fuel_s_per_lap"))
    fx = sim.config.effects
    defaults = {"sc_lap_factor": fx.sc_lap_factor, "vsc_lap_factor": fx.vsc_lap_factor,
                "sc_pit_loss_factor": fx.sc_pit_loss_factor, "lap1_extra_s": fx.lap1_extra_s}
    if race is not None:
        defaults["pit_loss_s"] = race.pit_loss_s
    for name, v in defaults.items():
        eff[name] = tuple(map(float, out.get(name, (v, 0.0))))
    return eff


# --- dataset -------------------------------------------------------------------------
@dataclass
class RaceInput:
    """One race for the dataset: the frames G replays plus optional lap weather and features.
    jobs.RaceData has the same fields and can be passed as is."""
    race_id: str
    frame: pd.DataFrame
    race: RaceParams
    practice: pd.DataFrame | None = None
    results: pd.DataFrame | None = None
    config: SimConfig = field(default_factory=SimConfig)
    weather: pd.DataFrame | None = None     # weather.lap_weather, indexed by lap
    features: dict | None = None            # pre-race columns (the data team's per-race row)


@dataclass
class CalibrationDataset:
    """One row per (race, k): race_id, season, k, feature columns, `target:<factor>` columns."""
    rows: pd.DataFrame
    races: dict

    @property
    def feature_columns(self):
        return [c for c in self.rows.columns if c not in ("race_id", "season", "k") and not c.startswith("target:")]

    @property
    def target_columns(self):
        return [c for c in self.rows.columns if c.startswith("target:")]

    def laps_k(self, race_id) -> list[int]:
        return sorted(int(k) for k in self.rows.loc[self.rows["race_id"] == race_id, "k"])

    def subset(self, race_ids) -> "CalibrationDataset":
        keep = set(race_ids)
        return CalibrationDataset(self.rows[self.rows["race_id"].isin(keep)].reset_index(drop=True),
                                  {r: d for r, d in self.races.items() if r in keep})


def _known_green(frame, k):
    """Race laps known at k usable for pace: green, no pit in or out, after lap 1, timed.
    Not the frame's clean flag, whose filter saw the whole race. A missing status or pit
    column counts as green and no pit."""
    f = frame[frame["lap"] <= k]

    def col(name, default):
        return f[name] if name in f else pd.Series(default, index=f.index)

    return f[(f["lap"] > 1) & (col("track_status", "1").astype(str) == "1") & ~col("pit_in", False).astype(bool)
             & ~col("pit_out", False).astype(bool) & f["lap_time_s"].notna() & f["tyre_age"].notna()]


def residuals(model, frame, k) -> pd.DataFrame:
    """Lap time less `model`'s prediction on the known green laps up to k."""
    g = _known_green(frame, k)
    g = g[g["compound"].isin(model.coefs)] if model is not None else g.iloc[:0]
    pred = np.full(len(g), np.nan)
    for comp, idx in g.groupby("compound").indices.items():
        sub = g.iloc[idx]
        pred[idx] = model.lap_times(comp, sub["tyre_age"].to_numpy(dtype=float), sub["lap"].to_numpy(dtype=float))
    cols = ["lap", "driver", "compound", "tyre_age"]
    return g[[c for c in cols if c in g]].assign(resid_s=g["lap_time_s"].to_numpy(dtype=float) - pred).reset_index(drop=True)


def features_at(r: RaceInput, k: int, prior=None) -> dict:
    """Pre-race columns plus in-race summaries from laps <= k only."""
    weather = getattr(r, "weather", None)
    row = {"pre_total_laps": r.race.total_laps, "pre_pit_loss_s": r.race.pit_loss_s,
           **{f"pre_{n}": v for n, v in (getattr(r, "features", None) or {}).items()}}
    res = residuals(prior, r.frame, k)
    known = _known_green(r.frame, k)
    for c in sorted(r.race.compounds):
        rc = res.loc[res["compound"] == c, "resid_s"]
        row[f"resid_mean_s[{c}]"] = float(rc.mean()) if len(rc) else np.nan
        row[f"laps_so_far[{c}]"] = int((known["compound"] == c).sum())
    seen = r.frame[r.frame["lap"] <= k]
    status = seen.groupby("lap")["track_status"].agg(lambda s: "".join(map(str, s)))
    row["sc_laps_so_far"] = int(status.str.contains("4").sum())
    row["vsc_laps_so_far"] = int(status.str.contains("[67]").sum())
    trend = np.nan
    if weather is not None and "track_temp_c" in weather:
        t = weather.loc[(weather.index >= 1) & (weather.index <= k), "track_temp_c"].astype(float).dropna()
        if len(t) >= 2 and t.index.nunique() >= 2:
            trend = float(np.polyfit(t.index.to_numpy(dtype=float), t.to_numpy(), 1)[0])
    row["temp_trend_c_per_lap"] = trend
    return row


def hindsight_targets(r: RaceInput) -> dict[str, float]:
    """The race's factors measured with the whole race: the full-window fit (base over the
    practice fit with no base delta, wear, hinge for a model with one, fuel) and the
    calibration.run observations."""
    from . import calibration
    from .engine import Simulator
    cfg = r.config.model_copy(update={"fit_window": "full", "exclude_snapshot_driver": False})
    full = Simulator(r.frame, cfg, r.practice)
    raw = full.physics_prior()
    deg = full.degradation
    out = {}
    for c, (b, w) in deg.coefs.items():
        if raw is not None and c in raw.coefs:
            out[factor_name("base_delta_s", c)] = b - raw.coefs[c][0]
        out[factor_name("wear_s_per_lap", c)] = w
        if getattr(deg, "hinges", None) is not None:
            out[factor_name("hinge_s_per_lap", c)] = float(deg.hinges[c])
    if hasattr(deg, "fuel_slope"):
        out["fuel_s_per_lap"] = float(deg.fuel_slope)
    summary = calibration.run(full, r.frame, r.race)[1]
    for name, key in (("pit_loss_s", "observed_pit_loss_s"), ("sc_lap_factor", "observed_sc_factor"),
                      ("vsc_lap_factor", "observed_vsc_factor"), ("sc_pit_loss_factor", "observed_sc_pit_loss_factor"),
                      ("lap1_extra_s", "observed_lap1_extra_s")):
        out[name] = float(summary[key])
    return {k: v for k, v in out.items() if not (isinstance(v, float) and math.isnan(v))}


def build_dataset(races, laps_k=None) -> CalibrationDataset:
    """The CalibrationDataset over `races` (RaceInput or jobs.RaceData).

    laps_k: None (calibration.default_laps_k), a list, or fn(race_input) -> list. Features
    at k come from laps <= k only; targets are the race's hindsight factors.
    """
    from . import calibration
    from .engine import Simulator
    rows, by_id = [], {}
    for r in races:
        rid = canonical_race_id(r.race_id)
        by_id[rid] = r
        ks = (calibration.default_laps_k(r.frame, r.race.total_laps) if laps_k is None
              else laps_k(r) if callable(laps_k) else list(laps_k))
        prior = None
        if r.practice is not None and not r.practice.empty:
            pre = Simulator(r.frame, r.config.model_copy(update={"fit_window": "pre_race"}), r.practice)
            prior = prior_model(pre.physics_prior(), {}, r.config)
        targets = {f"target:{n}": v for n, v in hindsight_targets(r).items()}
        for k in ks:
            rows.append({"race_id": rid, "season": race_season(rid), "k": int(k), **features_at(r, k, prior), **targets})
    return CalibrationDataset(pd.DataFrame(rows), by_id)


# --- validation ----------------------------------------------------------------------
def cadence_label(n) -> str:
    return "none" if n is None else str(n)


def folds(dataset: CalibrationDataset, protocol="race") -> dict[str, list[str]]:
    """{fold: held-out race ids}: one per race, or one per season as season:YYYY."""
    ids = sorted(dataset.races)
    if protocol == "race":
        return {r: [r] for r in ids}
    if protocol == "season":
        return {f"season:{s}": [r for r in ids if race_season(r) == s] for s in sorted({race_season(r) for r in ids})}
    raise ValueError(f"unknown protocol {protocol!r}")


def held_out(dataset: CalibrationDataset, fold: str) -> list[str]:
    if fold.startswith("season:"):
        held = [r for r in sorted(dataset.races) if race_season(r) == int(fold.split(":", 1)[1])]
    else:
        held = [canonical_race_id(fold)] if canonical_race_id(fold) in dataset.races else []
    if not held:
        raise KeyError(f"fold {fold!r} matches no race in the dataset {sorted(dataset.races)}")
    return held


FACTOR_ERROR_COLUMNS = ["fold", "race_id", "calibration_every", "window", "k", "factor", "mean", "sd", "target",
                        "err", "covered"]


@dataclass
class LoroResult:
    """One protocol's held-out results for one calibrator.

    rows: SPEC G midrace rows of the held-out races, with fold, calibrator and
    calibration_every columns. factor_errors: per (race, k, window, factor) the effective
    factor (mean, sd) against the hindsight target. folds: fold -> held-out and training races.
    """
    calibrator: str
    protocol: str
    rows: pd.DataFrame
    factor_errors: pd.DataFrame
    folds: pd.DataFrame

    def targets(self) -> pd.DataFrame:
        """Per (calibration_every, window, factor): n, rmse, sse, bias, mean sd, 68 percent coverage."""
        e = self.factor_errors
        if e.empty:
            return pd.DataFrame(columns=["n", "rmse", "sse", "bias", "mean_sd", "coverage_68"])
        g = e.groupby(["calibration_every", "window", "factor"])
        t = g.agg(n=("err", "size"), sse=("err", lambda x: float((x ** 2).sum())), bias=("err", "mean"),
                  mean_sd=("sd", "mean"), coverage_68=("covered", "mean"))
        t["rmse"] = np.sqrt(t["sse"] / t["n"])
        return t[["n", "rmse", "sse", "bias", "mean_sd", "coverage_68"]]

    def midrace(self, n_boot=None, seed=0) -> pd.DataFrame:
        """calibration.aggregate per cadence, indexed (calibration_every, window, horizon_bucket).
        The race-bootstrap CI is NaN with one held-out race."""
        from . import calibration
        n_boot = calibration.N_BOOT if n_boot is None else n_boot
        parts = [calibration.aggregate(part.drop(columns=["calibration_every"]), n_boot=n_boot, seed=seed)
                 .assign(calibration_every=ce).set_index("calibration_every", append=True)
                 for ce, part in self.rows.groupby("calibration_every", sort=False)]
        return pd.concat(parts).reorder_levels(["calibration_every", "window", "horizon_bucket"])


def _probe_driver(frame, k, min_laps_frac=0.9):
    """A finisher with a lap-k row (any one: without exclusion the fit does not depend on the driver)."""
    from . import calibration
    for d in calibration._finisher_ids(frame, min_laps_frac):
        g = frame[frame["driver"] == d]
        if k < len(g) and (k == 0 or (g["lap"] == k).any()):
            return d
    return None


def _run_folds(dataset, name, fold_map, protocol, cadences, windows, n_samples, wandb_profile, calibrator_params):
    from . import calibration
    from .engine import Simulator
    rows, errs, fold_rows = [], [], []
    for fold, held in fold_map.items():
        train = dataset.subset([r for r in dataset.races if r not in held])
        fold_rows.append({"fold": fold, "held_out": " ".join(held), "train": " ".join(sorted(train.races))})
        cal = CALIBRATORS.get(name)(**(calibrator_params or {}))
        if wandb_profile is not None and cal.learned:
            from .compute import wandb_run
            with wandb_run(wandb_profile, "h_fold", {"calibrator": name, "protocol": protocol, "fold": fold,
                                                    "held_out": held, "train": sorted(train.races)}):
                cal.fit(train)
        else:
            cal.fit(train)
        for rid in held:
            r = dataset.races[rid]
            race = r.race.model_copy(update={"race_id": rid})
            ks = dataset.laps_k(rid)
            for n in cadences:
                update = {"calibrator": ComponentSpec(name=name, params=calibrator_params or {}),
                          "calibration_every": n, **({"n_samples": n_samples} if n_samples else {})}
                cfg = r.config.model_copy(update=update)

                def factory(window, cfg=cfg, r=r):
                    return Simulator(r.frame, cfg.model_copy(update={"fit_window": window}), r.practice,
                                     weather=getattr(r, "weather", None), race_features=getattr(r, "features", None),
                                     calibrator=cal)

                g = calibration.midrace(factory, r.frame, race, laps_k=ks, windows=list(windows), results=r.results)
                rows.append(g.assign(calibrator=name, calibration_every=cadence_label(n), fold=fold))
                targets = {c.split(":", 1)[1]: v for c, v in
                           dataset.rows.loc[dataset.rows["race_id"] == rid, dataset.target_columns].iloc[0].items()
                           if pd.notna(v)}
                for window in windows:
                    sim = factory(window)
                    for k in ks:
                        d = _probe_driver(r.frame, k)
                        if d is None:
                            continue
                        eff = effective_factors(sim.fitted_for(calibration.snapshot_at(r.frame, race, d, k)), race)
                        for f, target in targets.items():
                            if f in eff:
                                mean, sd = eff[f]
                                err = mean - target
                                errs.append({"fold": fold, "race_id": rid, "calibration_every": cadence_label(n),
                                             "window": window, "k": k, "factor": f, "mean": mean, "sd": sd,
                                             "target": target, "err": err, "covered": bool(abs(err) <= sd)})
    return LoroResult(name, protocol, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
                      pd.DataFrame(errs, columns=FACTOR_ERROR_COLUMNS), pd.DataFrame(fold_rows))


def loro(dataset: CalibrationDataset, name: str, fold: str | None = None, *, cadences=(1,),
         windows=("pre_race", "to_snapshot"), n_samples=None, wandb_profile=None, calibrator_params=None):
    """Leave-one-race-out and leave-one-season-out validation of calibrator `name` on SPEC G.

    With `fold` None, returns {"loro": LoroResult, "loso": LoroResult}. With a fold (a race
    id, alias accepted, or season:YYYY) returns that fold's LoroResult; this is the h_fold
    job's runner. The calibrator is fitted on the races outside the fold (wandb_profile, a
    compute profile, logs learned fits with the fold assignment), then each held-out race
    is replayed at every cadence in `cadences`. With one race the training set is empty and
    the CIs are NaN: none and shrinkage need no training, learned classes raise.
    """
    args = (cadences, windows, n_samples, wandb_profile, calibrator_params)
    if fold is not None:
        protocol = "season" if fold.startswith("season:") else "race"
        key = fold if protocol == "season" else canonical_race_id(fold)
        return _run_folds(dataset, name, {key: held_out(dataset, fold)}, protocol, *args)
    return {"loro": _run_folds(dataset, name, folds(dataset, "race"), "race", *args),
            "loso": _run_folds(dataset, name, folds(dataset, "season"), "season", *args)}


def selection_table(results: dict, window="to_snapshot", horizons=(20, 40), calibration_every="1",
                    n_boot=None, seed=0) -> pd.DataFrame:
    """none, shrinkage and each learned class side by side, from {name: LoroResult}.

    G mae_offset_s over horizons in [horizons[0], horizons[1]] with its race-bootstrap 95
    percent CI, the per factor rmse and coverage, and the adoption decision: a learned class
    is adopted only when its CI upper bound is below shrinkage's CI lower bound; ties,
    overlapping or NaN CIs keep shrinkage.
    """
    from . import calibration
    n_boot = calibration.N_BOOT if n_boot is None else n_boot
    out = {}
    for name, res in results.items():
        r = res.rows
        r = r[(r["window"] == window) & (r["calibration_every"] == calibration_every)
              & r["horizon"].between(*horizons)]
        lo, hi = calibration._bootstrap_ci(r, n_boot, seed) if len(r) else (np.nan, np.nan)
        row = {"kind": "reference" if name in REFERENCE else "learned", "n": int(r["err_offset_s"].notna().sum()),
               "n_races": int(r["race_id"].nunique()), "mae_offset_s": float(r["err_offset_s"].abs().mean()),
               "mae_offset_ci_lo": lo, "mae_offset_ci_hi": hi}
        t = res.targets()
        if not t.empty and (calibration_every, window) in t.index.droplevel("factor"):
            for f, v in t.loc[(calibration_every, window)].iterrows():
                row[f"rmse:{f}"], row[f"coverage_68:{f}"] = v["rmse"], v["coverage_68"]
        out[name] = row
    table = pd.DataFrame.from_dict(out, orient="index")
    ref_lo = table.loc["shrinkage", "mae_offset_ci_lo"] if "shrinkage" in table.index else np.nan
    table["decision"] = [
        "reference" if k == "reference" else ("adopt" if hi < ref_lo else "keep shrinkage")
        for k, hi in zip(table["kind"], table["mae_offset_ci_hi"])]
    return table
