"""Strategy simulator: a snapshot and plans in, lap time summaries out."""
import copy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..contracts import (Compound, CompareResult, ModelReport, Plan, PlanStint, SimResult, Snapshot,
                         StintRecord, TimeSummary, TrafficEstimate)
from . import calibrators as _cal
from . import cliff as _cliff
from . import model_report as _report
from . import proxy as _proxy
from . import sc_queue as _queue
from .config import SimConfig, class_names, compound_class
from .data import as_of_columns, visible_rows
from .degradation import DEGRADATION
from .events import EVENTS, Scenario
from .hazard import HAZARD
from .rules import RuleContext, apply_inventory, check, stint_limits
from .track import TRACK
from .traffic import TRAFFIC


def summarise(totals) -> TimeSummary:
    totals = np.asarray(totals, dtype=float)
    n = len(totals)
    return TimeSummary(
        mean_s=float(totals.mean()), std_s=float(totals.std()),
        p05_s=float(np.percentile(totals, 5)), p50_s=float(np.percentile(totals, 50)),
        p95_s=float(np.percentile(totals, 95)), n_samples=n,
        cvar_90_s=float(np.sort(totals)[-max(1, int(np.ceil(0.1 * n))):].mean()),
    )


def _messages(exc: ValidationError) -> list[str]:
    return [e["msg"] for e in exc.errors()]


class _Action(BaseModel):
    """A decision at the start of lap laps_completed + 1: stay out for `laps` laps, or pit at
    the end of this lap (a pit takes laps 1)."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["stay", "pit"]
    compound: Compound | None = None
    set_id: str | None = None
    start_age: int | None = Field(default=None, ge=0)
    laps: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check(self):
        if self.action == "pit" and self.compound is None:
            raise ValueError("pit action needs a compound")
        if self.action == "pit" and self.laps != 1:
            raise ValueError("a pit action is taken at the end of this lap; stay out first for later stops")
        return self


def _advance(snapshot: Snapshot, laps: int = 1, **update) -> Snapshot:
    """The snapshot `laps` laps on. Nested models are passed through, not re-dumped."""
    data = {name: getattr(snapshot, name) for name in Snapshot.model_fields}
    data.update(laps_completed=snapshot.laps_completed + laps, elapsed_s=None, **update)
    return Snapshot.model_validate(data)


def _matrix(values, mask):
    return [[float(v) if m else None for v, m in zip(row, ok)] for row, ok in zip(values, mask)]


@dataclass
class BatchResult:
    """Many plans scored under the same draws. Rows follow the input plan order.

    With a hazard model on (SimConfig.hazard), `times` include the hazard penalties: a
    DNF sample keeps the remaining laps as if the car had continued and adds
    dnf_penalty_s once, a big loss sample pays big_loss_s on the event lap. The
    `lap_times` of sample 0 carry the same penalties. The mean time then depends on
    dnf_penalty_s, an arbitrary time-metric convention for pricing a retirement; p_dnf
    and cvar_90_s are the honest views of the risk, and the eval should report them
    beside the mean. A position metric should rank on `dnf_lap` instead (a DNF behind
    every finisher, an earlier retirement worse), not on the penalised time.
    `dnf` and `big_loss` are exclusive (at most one hazard event per sample), so
    p_dnf + p_big_loss is the probability of any hazard event.
    """

    plans: list            # Plan, or None when the input did not parse
    errors: list           # list[str] per plan, empty when valid
    valid: np.ndarray = field(init=False)
    times: np.ndarray = None          # (n_plans, n_samples) remaining seconds incl. pit loss, NaN if invalid
    pit_loss: np.ndarray = None       # (n_plans, n_samples)
    stint_seconds: list = None        # per plan (n_stints, n_samples), pit loss excluded, None if invalid
    lap_times: np.ndarray | None = None  # (n_plans, n_laps) sample 0 lap times when requested
    dnf: np.ndarray = None            # (n_plans, n_samples) bool, a hazard event ended the race
    big_loss: np.ndarray = None       # (n_plans, n_samples) bool, a hazard event the car survived
    dnf_lap: np.ndarray = None        # (n_plans, n_samples) race lap of retirement, NaN without a DNF
    elapsed_s: float | None = None
    seed: int = 0
    support_warnings: list = None     # list[str] per plan, empty when invalid
    # traffic (traffic.py): time added by the ghosts and overtakes charged, both included in
    # times and the stint sums; zeros when traffic is off or could not be placed
    traffic_s: np.ndarray = None      # (n_plans, n_samples) float, green laps
    passes: np.ndarray = None         # (n_plans, n_samples) int
    bunching_s: np.ndarray = None     # (n_plans, n_samples) float, SC laps (the SC queue)
    # (n_plans, n_laps) sample 0 with return_laps: the field's SC lap less the engine's on each
    # bunched SC lap (traffic.py), NaN elsewhere; None without return_laps
    sc_field_delta: np.ndarray | None = None
    traffic_on: bool = False          # a traffic state was placed for this snapshot
    # (n_plans, n_samples) with traffic placed, else None (traffic.py): the position at the
    # flag, after the first out-lap (NaN without a stop), and the cars that come by during the
    # first stop (within its pit loss behind at the in-lap's line; NaN without a stop)
    final_pos: np.ndarray | None = None
    rejoin_pos: np.ndarray | None = None
    pit_window_cars: np.ndarray | None = None

    def __post_init__(self):
        self.valid = np.array([p is not None and not e for p, e in zip(self.plans, self.errors)], dtype=bool)

    @property
    def race_times(self):
        return self.times + (self.elapsed_s or 0.0)

    def result(self, i, snapshot: Snapshot) -> SimResult:
        plan = self.plans[i]
        if not self.valid[i]:
            return SimResult(valid=False, errors=self.errors[i], plan=plan)
        return SimResult(
            valid=True, plan=plan, num_stops=plan.num_stops, pit_laps=plan.pit_laps(snapshot),
            remaining=summarise(self.times[i]), elapsed_s=snapshot.elapsed_s,
            stint_seconds=self.stint_seconds[i].mean(axis=1).tolist(),
            pit_loss_total_s=float(self.pit_loss[i].mean()),
            lap_times_s=self.lap_times[i].tolist() if self.lap_times is not None else None,
            p_dnf=float(self.dnf[i].mean()), p_big_loss=float(self.big_loss[i].mean()),
            support_warnings=self.support_warnings[i] if self.support_warnings else [],
            traffic_s=float(self.traffic_s[i].mean()) if self.traffic_on else None,
            passes_mean=float(self.passes[i].mean()) if self.traffic_on else None,
            bunching_s=float(self.bunching_s[i].mean()) if self.traffic_on else None,
            sc_field_delta_s=(self.sc_field_delta[i].tolist()
                              if self.traffic_on and self.sc_field_delta is not None else None),
            traffic=self.traffic_estimate(i),
        )

    def traffic_estimate(self, i) -> TrafficEstimate | None:
        """Plan i's TrafficEstimate over the samples, None when traffic was not placed or the
        plan is invalid."""
        if not self.traffic_on or self.final_pos is None or not self.valid[i]:
            return None

        def band(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if not len(x):
                return None
            lo, med, hi = np.percentile(x, [5, 50, 95])
            return {"low": float(lo), "median": float(med), "high": float(hi)}
        window = self.pit_window_cars[i]
        return TrafficEstimate(
            time_lost_s=band(self.traffic_s[i] + self.bunching_s[i]), rejoin_position=band(self.rejoin_pos[i]),
            cars_in_pit_window=float(np.nanmean(window)) if np.isfinite(window).any() else None,
            final_position=band(self.final_pos[i]), passes_mean=float(self.passes[i].mean()))


# support warning on every plan when recorded ghosts run under events other than the record's
TRAFFIC_DRAWN_EVENTS = "traffic_recorded_with_drawn_events"
FIT_CACHE_SIZE = 64  # windowed fits kept per Simulator, oldest dropped first
# events draw from their own substream, default_rng([seed, k, EVENT_STREAM]), so the main
# stream never sees an event draw: with events on, every other draw matches the run without.
# The degradation parameter draw and the residual lap noise come from one generator per
# stream per batch call, default_rng([seed, STREAM_KEY, stream]), sample k taking row k
# (batch_normals). Turning either on leaves the hazard, track and event draws as they were.
EVENT_STREAM = 1
DEGRADATION_STREAM = 2
LAP_NOISE_STREAM = 3
# a posterior draw under a wear bound takes further proposals from this batch stream when
# its first (DEGRADATION_STREAM's row) is rejected, so an accepted first proposal is the
# unbounded run's draw
DEGRADATION_RETRY_STREAM = 4
# the per-sample draw of an as-of pit loss (SimConfig.pit_loss.source "asof"); unused otherwise
PIT_LOSS_STREAM = 5
# the belief's wear-cliff age spread (sim/cliff.py, Cliff.sample_ages): one
# (n, len(config.compound_classes)) normal matrix per batch call, column per compound in sorted
# order, so turning it on leaves every other draw as it was
CLIFF_STREAM = 6
# lap_noise "persistent": per sample one offset per plan stint (column j stint j), an
# (n, max stints) normal matrix per batch call; the lap-to-lap part still draws from
# LAP_NOISE_STREAM
LAP_OFFSET_STREAM = 7
# never a sample index, so [seed, STREAM_KEY, s] differs from every main [seed, k] and event
# [seed, k, 1] key
STREAM_KEY = 2 ** 32 - 1


def batch_normals(seed: int, stream: int, n: int, width: int, sd: float = 1.0) -> np.ndarray:
    """(n, width) normals N(0, sd^2) for `stream`, row k for sample k. Rows are drawn in order,
    so row k depends on (seed, k, width) only, never on n: sample k is the same in a batch of
    10 or 200 (R6)."""
    return np.random.default_rng([seed, STREAM_KEY, stream]).normal(0.0, sd, (n, width))


def _race_key(snapshot: Snapshot) -> tuple:
    """The race constants a cached fit or wear-cliff edge reads from snapshot.race: the race, its
    length (the tyre-life prior and the fuel prior scale with total_laps) and its compounds. Every
    per-instance cache built from a snapshot keys on it, so what a scenario is scored with never
    depends on which scenario built the cache first (a snapshot_at set varies total_laps by
    driver; an as-of set's compounds grow after a wet start). pit_loss_s is left out: no default
    fit reads it (the learned calibrator's feature row does, and policy_scenario sets it per
    snapshot, so keying on it would refit the belief for every scenario)."""
    return snapshot.race.race_id, int(snapshot.race.total_laps), tuple(sorted(snapshot.race.compounds))


class Simulator:
    """Fitted on one race's lap frame. Cheap to call repeatedly.

    `practice` is a frame of practice long-run laps (data.practice_frame: FP1-FP3
    concatenated, with a "session" column), needed by the pre_race and early
    to_snapshot fit windows. The instance itself carries the configured window's fit
    at lap 0: every race lap for "full", and the practice fit for "pre_race" and
    "to_snapshot" (every race lap when there is no practice). `fitted_for(snapshot)`
    gives the instance fitted on the window `config.fit_window` picks for that snapshot.

    The pre_race and to_snapshot windows go through the calibrator (calibrators.py):
    `calibrator` is an instance (a fitted learned class, say), else the one
    config.calibrator names. `weather` (weather.lap_weather) feeds the cadence's rain and
    temperature refreshes and the context, `race_features` the context.

    `pace` (driver -> offset_s, a dict or Series such as pace.practice_pace(...)["offset_s"])
    adds the snapshot driver's offset to every green lap, lap 1 included, never to an SC or
    VSC lap (the field's lap, or a delta every car drives to). Drivers it does not list,
    and None, get 0. `with_pace` gives a copy with another pace.

    `qualifying` (data.qualifying_frame of this weekend's qualifying) feeds the tier-3 pace of
    an unraced compound (proxy.py); with config.compound_proxy.enabled every fit then prices
    HARD, MEDIUM and SOFT.

    The wear cliff (config.wear_cliff, sim/cliff.py) prices tyres past their life: under the
    full window from the whole race (the scorer), under any other window only from what is
    known at the snapshot (cliffs_for). The retired R8 stint limit (config.stint_limit_margin_laps,
    legacy only) is computed from the full race frame only when it is set.
    """

    def __init__(self, laps: pd.DataFrame, config: SimConfig | None = None,
                 practice: pd.DataFrame | None = None, pace=None, *, weather: pd.DataFrame | None = None,
                 race_features: dict | None = None, calibrator=None, qualifying: pd.DataFrame | None = None,
                 pit_loss_s: float | None = None):
        self.config = config or SimConfig()
        # the world's hindsight pit loss (eval.score.race_simulators): when set it replaces the
        # snapshot's on every simulation; a belief never gets it (docs/eval.md "Pit loss")
        self.race_pit_loss_s = pit_loss_s
        self._pit_loss_cache = {}   # pit_loss.estimate per (driver, lap, window), shared with copies
        self._pit_stops = {}        # pit_loss.stop_table of `laps`, built on first use
        self.laps, self.practice, self.pace = laps, practice, pace
        self.qualifying, self._quali_gaps = qualifying, None
        # retired R8 (legacy setting only), from the full race frame; shared by fitted_for copies.
        # Never computed under the default config, so nothing a policy sees carries it.
        self.stint_limits, self.stint_limit_sources = {}, {}
        if self.config.stint_limit_margin_laps is not None:
            self.stint_limits = stint_limits(laps, self.config.stint_limit_rank)
        if self.config.stint_limit_margin_laps is not None and self.config.compound_proxy.enabled:
            self.stint_limits, self.stint_limit_sources = _proxy.unraced_limits(
                self.stint_limits, practice, self.config.compound_proxy.steps)
        self.weather, self.race_features = weather, race_features
        spec = self.config.calibrator
        self.calibrator = calibrator if calibrator is not None else _cal.CALIBRATORS.get(spec.name)(**spec.params)
        self.calibration = None   # CalibrationState on a pre_race or to_snapshot instance
        self.window = None   # fitted_for cache key on a windowed instance, None on the root
        self._race_id = None   # the snapshot race of a windowed instance (as-of cross-race tables)
        self._race = None      # its RaceParams (the anchored practice fuel, config.fuel_anchor)
        self._judge_anchor = None   # the full fit's fuel anchor (config.fuel_anchor.full_fit), once
        self._fits, self._cliffs = {}, {}
        self._outputs, self._physics = {}, {}   # shared with the windowed copies
        self._windows = {}   # window_rows, shared with the windowed copies
        ev = self.config.events
        self.events = EVENTS.get(ev.name)(**ev.params)
        if hasattr(self.events, "bind"):   # "recorded" / "hybrid": the race's own neutralised laps
            self.events.bind(laps)
        tr = self.config.track
        self.track = TRACK.get(tr.name)(**tr.params)
        tf = self.config.traffic
        self.traffic = TRAFFIC.get(tf.name)(**tf.params)
        if self.config.fit_window != "full" and self._has_practice:
            self._fit(self.practice.assign(lap=np.nan))
        else:
            self._fit(laps)

    @property
    def modelled(self):
        return self.degradation.compounds

    def with_pace(self, pace) -> "Simulator":
        """A shallow copy with `pace` set. The fits are shared, the windowed copies are not
        (they carry the pace they were built with)."""
        out = copy.copy(self)
        out.pace, out._fits, out._cliffs = pace, {}, {}
        return out

    def _pace_offset(self, driver) -> float:
        if self.pace is None:
            return 0.0
        offset = float(self.pace.get(driver, 0.0))
        return 0.0 if np.isnan(offset) else offset

    @property
    def _has_practice(self):
        return self.practice is not None and not self.practice.empty

    def _model(self, frame, **extra):
        spec, cfg = self.config.degradation, self.config
        kw = dict(min_samples=cfg.min_samples, fuel_prior=cfg.pre_race_fuel_slope_s_per_lap,
                  base_delta=cfg.pre_race_base_delta_s, fuel_prior_sd=cfg.shrinkage["fuel_s_per_lap"],
                  hinge_knots=cfg.hinge_knots, hinge_knot_default=cfg.hinge_knot_default, **spec.params)
        if cfg.fuel_anchor.full_fit and cfg.fit_window == "full" and "shrinkage" not in extra:
            kw["fuel_anchor"] = self.judge_fuel_anchor()
        kw.update(extra)
        return DEGRADATION.get(spec.name).fit(frame, **kw)

    def judge_fuel_anchor(self) -> tuple[float, float]:
        """(mean, sd) the full fit's fuel slope is shrunk toward under config.fuel_anchor.full_fit:
        the a priori physics prior (fuel_prior.physics_prediction) on the race's laps and this
        weekend's practice lap (the race's median clean lap without practice; the judge may read
        the whole race). No other race is read."""
        if self._judge_anchor is None:
            from . import fuel_prior
            lap_s = fuel_prior.practice_lap_s(self.practice) if self._has_practice else float("nan")
            if not np.isfinite(lap_s):
                lap_s = fuel_prior.practice_lap_s(self.laps)
            p = fuel_prior.physics_prediction(int(self.laps["lap"].max()), lap_s)
            self._judge_anchor = (float(p["mean"]), float(p["sd"]))
        return self._judge_anchor

    def _free_air(self, frame):
        """(rows the fit may use, clean rows dropped). Every row unless config.fit_free_air_only;
        then the race rows with free_air True and every practice row (lap NaN)."""
        if not self.config.fit_free_air_only:
            return frame, 0
        practice = frame["lap"].isna().to_numpy()
        if "free_air" not in frame:
            if not practice.all():
                raise ValueError("fit_free_air_only needs the free_air column (data.lap_frame)")
            return frame, 0
        keep = practice | frame["free_air"].eq(True).to_numpy()
        return frame[keep], int((frame["clean"].to_numpy(dtype=bool) & ~keep).sum())

    def _thin_dropped(self, frame):
        """`frame` without the race rows of a compound with fewer than
        config.min_race_rows_to_fit clean race rows (None: every compound keeps its rows), so the
        compound proxy prices it (proxy.py) instead of an extrapolated fit of a handful of laps.
        Practice rows (lap NaN) stay."""
        n = self.config.min_race_rows_to_fit
        if n is None or frame is None or not len(frame) or "lap" not in frame:
            return frame
        race = frame["lap"].notna().to_numpy()
        clean = frame["clean"].astype(bool).to_numpy() if "clean" in frame else np.ones(len(frame), dtype=bool)
        counts = frame[race & clean].groupby("compound").size()
        thin = set(counts.index[counts < n]) | (set(frame.loc[race, "compound"].dropna()) - set(counts.index))
        if not thin:
            return frame
        return frame[~(race & frame["compound"].isin(thin).to_numpy())]

    def _fit(self, frame, ref=None, model=None, **extra):
        """Fit the degradation model on `frame` (or use `model`); the hazard model and ref_lap_s
        (the median clean lap) on `ref`, which adds practice to the race laps under to_snapshot.
        Under fit_free_air_only both keep only their free-air rows (_free_air), and the
        degradation fit_report gains n_pack_rows, the clean rows of `frame` dropped."""
        frame, n_pack = self._free_air(frame)
        ref = frame if ref is None else self._free_air(ref)[0]
        if model is None:
            frame = self._thin_dropped(frame)
        self.degradation = model if model is not None else self._model(frame, **extra)
        if self.config.fit_free_air_only and getattr(self.degradation, "fit_report", None) is not None:
            self.degradation = copy.copy(self.degradation)   # `model` may be a shared prior
            self.degradation.fit_report = {**self.degradation.fit_report, "n_pack_rows": n_pack}
        if self.config.compound_proxy.enabled:
            self.degradation = self._proxied(self.degradation, extra.get("prior"))
        hz = self.config.hazard
        self.hazard = HAZARD.get(hz.name).fit(ref, min_samples=self.config.min_samples, **hz.params)
        self.ref_lap_s = float(ref.loc[ref["clean"].astype(bool), "lap_time_s"].median())

    def quali_gaps(self):
        """proxy.qualifying_gaps of `qualifying`, computed once; None without qualifying."""
        if self._quali_gaps is None and self.qualifying is not None:
            self._quali_gaps = _proxy.qualifying_gaps(self.qualifying) or {}
        return self._quali_gaps or None

    def _proxied(self, model, prior=None):
        """proxy.apply on a fitted model: the tier-2 steps from `prior` (the to_snapshot fit's
        prior) when given, else from the practice fit (physics_prior, the snapshot driver left
        out under exclude_snapshot_driver)."""
        cfg = self.config
        if prior is None and self._has_practice:
            prior = self.physics_prior(self.window[2] if self.window is not None else None)
        px = cfg.compound_proxy
        if px.as_of and self.window is not None and cfg.fit_window != "full":
            # a policy's fit: the tier-5 inputs from the races before this one (proxy.as_of_config)
            px = _proxy.as_of_config(px, self._race_id or "")
        return _proxy.apply(model, px, min_samples=cfg.min_samples, practice=prior,
                            quali=self.quali_gaps(), knot=lambda c: cfg.hinge_knots.get(c, cfg.hinge_knot_default),
                            wear_lower_bound=cfg.degradation.params.get("wear_lower_bound"))

    def fitted_for(self, snapshot: Snapshot) -> "Simulator":
        """The instance fitted on this snapshot's fit window. Built once per key, then cached.

        Under to_snapshot the key's lap is the calibration point k0 (calibrators.calibration_point),
        so with calibration_every N the fit at k0 serves every k up to the next point.
        """
        cfg = self.config
        if self.window is not None or (cfg.fit_window == "full" and not cfg.exclude_snapshot_driver):
            if cfg.fit_window == "full" and cfg.wear_guard.enabled:
                self._wear_guarded(snapshot)
            return self
        k0 = (_cal.calibration_point(snapshot, cfg.calibration_every, self.weather, cfg.recalibrate_temp_delta_c)
              if cfg.fit_window == "to_snapshot" else None)
        key = (cfg.fit_window, k0, snapshot.driver if cfg.exclude_snapshot_driver else None)
        t = self.cut_time(snapshot.driver, k0 or 0)
        if t is not None:   # cut by session time: one fit per cut, the time the driver completed k0
            key = (*key, t)
        # the cache also keys on the race constants the fit reads (_fit_window keeps snapshot.race:
        # the fuel prior and the calibrator scale with total_laps), so no fit built for one race
        # length serves another; the fit's `window` stays the window alone
        cache_key = (*key, _race_key(snapshot))
        if cache_key not in self._fits:
            if len(self._fits) >= FIT_CACHE_SIZE:
                del self._fits[next(iter(self._fits))]
            fit = copy.copy(self)
            fit.window, fit._fits, fit._cliffs = key, {}, {}
            fit._guard_race = None
            fit._fit_window(snapshot, k0 or 0)
            if cfg.fit_window == "full" and cfg.wear_guard.enabled:
                fit._wear_guarded(snapshot)
            self._fits[cache_key] = fit
        return self._fits[cache_key]

    def _wear_guarded(self, snapshot):
        """The thin-fit wear guard on this whole-race fit, once per race (wear_guard.guarded):
        SimConfig.wear_guard, the judge only. The race id comes from the snapshot, which is why
        it is applied here rather than in _fit."""
        rid = snapshot.race.race_id
        if getattr(self, "_guard_race", None) == rid:
            return
        from . import wear_guard as _wg
        self.degradation, _ = _wg.guarded(self.degradation, self.laps, rid, self.config.wear_guard)
        self._guard_race = rid

    def point_estimate(self) -> "Simulator":
        """A shallow copy that simulates the point estimate: no posterior parameter draw and no
        lap noise; events, pit-loss and calibrated factor draws are kept (SimConfig.plan_on
        "point"). Call it on a fitted instance (fitted_for)."""
        out = copy.copy(self)
        out.config = self.config.model_copy(update={"lap_noise": "none"})
        if getattr(self.degradation, "param_draw", "none") != "none":
            m = copy.copy(self.degradation)
            m.param_draw = "none"
            out.degradation = m
        return out

    def _keep(self, drop):
        def keep(f):
            return f[f["driver"] != drop] if drop is not None and "driver" in f else f
        return keep

    def physics_prior(self, drop=None, race=None):
        """The practice fit with no base delta (calibrators.prior_model adds the factors), None
        without practice. Cached per excluded driver (and fuel). Under config.fuel_anchor.practice
        "anchored" with a race (`race`, else the windowed instance's), the practice in-stint fuel
        column is fixed at the race's anchored fuel prior (race_history.as_of_fuel "anchored")
        instead of pre_race_fuel_slope_s_per_lap, so the practice wear is net of it."""
        race = race if race is not None else self._race
        practice = self._keep(drop)(self.practice.assign(lap=np.nan)) if self._has_practice else None
        fuel = None
        if self.config.fuel_anchor.practice == "anchored" and race is not None and practice is not None:
            from .race_history import as_of_fuel
            fuel = float(as_of_fuel(race, practice=practice, mode="anchored")["mean"])
        key = drop if fuel is None else (drop, fuel)
        if key not in self._physics:
            extra = {} if fuel is None else {"fuel_prior": fuel}
            self._physics[key] = self._model(practice, base_delta=0.0, **extra) if practice is not None else None
        return self._physics[key]

    def cut_time(self, driver, k0: int) -> float | None:
        """The session time a windowed fit at calibration point k0 is cut at under
        policy_view.window_cut "time": the end_s of `driver`'s lap k0, else (no such row, a
        made-up snapshot) the earliest end_s of any car's lap k0, which shows no more. None when
        the cut is by lap: window_cut "lap", k0 0 (no race lap is visible), or a frame without
        driver and end_s columns (a hand-built one)."""
        if (self.config.policy_view.window_cut != "time" or k0 <= 0
                or "end_s" not in self.laps or "driver" not in self.laps):
            return None
        at = self.laps[self.laps["lap"] == k0]
        end = pd.to_numeric(at["end_s"], errors="coerce")
        own = end[(at["driver"] == driver).to_numpy()].dropna()
        if len(own):
            return float(own.iloc[0])
        return float(end.min()) if end.notna().any() else None

    def window_rows(self, snapshot: Snapshot, k0: int, drop=None) -> pd.DataFrame:
        """The race rows a windowed fit at calibration point k0 may read (SimConfig.policy_view):
        cut by session time at cut_time (else by lap number, lap <= k0), with the derived
        columns recomputed on those rows (data.as_of_columns) under clean "as_of"; then the
        excluded driver `drop` taken out. Cached per (k0, cut time, drop)."""
        t = self.cut_time(snapshot.driver, k0)
        key = (k0, t, drop)
        if key not in self._windows:
            race = self.laps
            race = race[race["lap"] <= k0] if t is None else visible_rows(race, t)
            if self.config.policy_view.clean == "as_of" and len(race):
                race = as_of_columns(race)
            if len(self._windows) >= FIT_CACHE_SIZE:
                del self._windows[next(iter(self._windows))]
            self._windows[key] = self._keep(drop)(race)
        return self._windows[key]

    def _calibrate(self, snapshot: Snapshot, k0: int, drop):
        """The calibrator's output at k0, cached per (calibrator, k0, exclusion, cut time)."""
        cfg, every = self.config, self.config.calibration_every
        t = self.cut_time(snapshot.driver, k0)
        rk = _race_key(snapshot)   # the calibrator reads snapshot.race (total_laps): never shared across lengths
        key = (self.calibrator.name, k0, drop, t, rk)
        if key in self._outputs:
            return self._outputs[key]
        pre = self._calibrate(snapshot, 0, drop) if k0 > 0 else None
        # an earlier point that saw no more than this one (by time when the window is cut by time)
        earlier = [(k, s) for (n, k, d, s, r) in self._outputs
                   if n == key[0] and d == drop and r == rk and k < k0 and (t is None or s is None or s <= t)]
        if earlier:
            k_prev, t_prev = max(earlier, key=lambda e: (e[0], -1.0 if e[1] is None else e[1]))
            previous = self._outputs[(key[0], k_prev, drop, t_prev, rk)]
        else:
            previous = pre
        race = self.window_rows(snapshot, k0, drop)
        practice = self._keep(drop)(self.practice.assign(lap=np.nan)) if self._has_practice else None
        raw = self.physics_prior(drop, snapshot.race)
        ctx = _cal.CalibrationContext(
            k=k0, config=cfg, race=snapshot.race, valid_until_lap=_cal.valid_until(k0, every, snapshot.race.total_laps),
            laps_so_far=race, practice=practice, practice_model=raw,
            residuals_so_far=_cal.residuals(_cal.prior_model(raw, {}, cfg) if raw is not None else None, race, k0),
            weather_so_far=self.weather[self.weather.index <= k0] if self.weather is not None else None,
            sc_history=Scenario(sc_laps=[x for x in snapshot.sc_laps if x <= k0],
                                vsc_laps=[x for x in snapshot.vsc_laps if x <= k0]),
            race_features=self.race_features, pre_race_output=pre, previous_output=previous)
        out = self.calibrator.calibrate(ctx)
        if len(self._outputs) >= FIT_CACHE_SIZE:
            del self._outputs[next(iter(self._outputs))]
        self._outputs[key] = out
        return out

    def _fit_window(self, snapshot: Snapshot, k0: int = 0):
        """Refit this instance on the snapshot's window.

        full: every race lap. pre_race: the calibrator's lap-0 prior, held (for "none" and
        "shrinkage" the practice fit plus the config base delta). to_snapshot: every car's
        race laps visible at the calibration point k0 (window_rows: completed by the time the
        snapshot driver completed lap k0, their clean flags their own), shrunk toward the prior
        with the calibrator's sds as taus; practice enters only through the prior, and
        ref_lap_s is the median of both. When the sds are all 0 ("none") the prior is held
        and the instance is the pre_race one. Exclusion drops the snapshot driver from
        whichever frames apply.
        """
        cfg = self.config
        drop = snapshot.driver if cfg.exclude_snapshot_driver else None
        self._race_id, self._race = snapshot.race.race_id, snapshot.race
        keep = self._keep(drop)
        if cfg.fit_window == "full":
            return self._fit(keep(self.laps))
        race = self.window_rows(snapshot, k0 if cfg.fit_window == "to_snapshot" else 0, drop)
        practice = keep(self.practice.assign(lap=np.nan)) if self._has_practice else None
        if practice is None and race.empty:
            raise ValueError("fit window needs practice laps")
        out = self._calibrate(snapshot, k0 if cfg.fit_window == "to_snapshot" else 0, drop)
        raw = self.physics_prior(drop, snapshot.race)
        prior, taus, fuel, held = _cal.degradation_prior(out, raw, cfg)
        if cfg.fit_window == "pre_race":
            self.calibration = _cal.CalibrationState(out, 0, True, raw)
            return self._fit(practice, model=_cal.held_prior(prior, out))
        self.calibration = _cal.CalibrationState(out, k0, held, raw)
        if held:   # nothing from the race: the pre_race fit, hazard and ref_lap_s included
            if prior is None:
                raise ValueError(f"calibrator {out.source!r} holds the prior, which needs practice laps")
            return self._fit(practice, model=_cal.held_prior(prior, out))
        ref = race if practice is None else pd.concat([race, practice], ignore_index=True)
        self._fit(race, ref=ref, prior=prior, shrinkage=taus, fuel_prior=fuel)
        if cfg.belief_guard.enabled and cfg.fit_window == "to_snapshot":
            # the belief's thin-fit guard (sim/belief_guard.py): visible rows, as-of tables only
            from . import belief_guard as _bg
            self.degradation, _ = _bg.guarded(
                self.degradation, race[race["lap"].notna()] if "lap" in race else race, snapshot.race.race_id,
                cfg.belief_guard, proxy_cfg=_proxy.as_of_config(cfg.compound_proxy, snapshot.race.race_id),
                practice=prior, quali=self.quali_gaps(), wear_lower_bound=cfg.degradation.params.get("wear_lower_bound"))

    def cliffs_for(self, snapshot: Snapshot) -> dict:
        """compound -> cliff.Cliff on this instance's fit at `snapshot` ({} with the cliff off).
        The full window is the scorer's (cliff.world_cliffs: the historical prior extended by
        this race's laps, or the race alone under wear_cliff.world_edge "race");
        every other window reads only what is known at the snapshot
        (cliff.as_of_cliffs: practice, race laps up to the fit's calibration point, earlier races).
        Cached per instance and information horizon."""
        wc = self.config.wear_cliff
        if not wc.enabled:
            return {}
        fit = self.fitted_for(snapshot)
        if fit is not self:
            return fit.cliffs_for(snapshot)
        if self.config.fit_window == "full":
            # keyed by the race constants world_cliffs reads (the prior scales with total_laps,
            # which a set built by snapshot_at varies per driver): a cache keyed without them kept
            # the first snapshot's edges, so a plan's score depended on the scenarios scored before
            key = ("world", *_race_key(snapshot))
            if key not in self._cliffs:
                if len(self._cliffs) >= FIT_CACHE_SIZE:
                    del self._cliffs[next(iter(self._cliffs))]
                steps = self.config.compound_proxy.steps if self.config.compound_proxy.enabled else None
                self._cliffs[key] = _cliff.world_cliffs(self.laps, self.practice, snapshot.race, wc, steps)
                rep = getattr(self.degradation, "fit_report", None)
                if rep is not None:   # where each edge came from, for the fidelity tool's hindsight ledger
                    self.degradation = copy.copy(self.degradation)
                    self.degradation.fit_report = {**rep, "support_edge": {
                        "world_edge": wc.world_edge, **_cliff.world_ledger(self._cliffs[key])}}
            return self._cliffs[key]
        # the information horizon is the fit's, never the snapshot's: a plan's continuation is
        # scored at advanced snapshots, and their laps are the plan's hypothetical future
        # and the rows are the fit window's (window_rows: cut by session time at the calibration
        # point under policy_view.window_cut "time"), the same cut every as-of quantity uses
        horizon = (self.window[1] or 0) if self.config.fit_window == "to_snapshot" and self.window else 0
        key = (snapshot.race.race_id, snapshot.race.total_laps, horizon, self.cut_time(snapshot.driver, horizon))
        if key not in self._cliffs:
            if len(self._cliffs) >= FIT_CACHE_SIZE:
                del self._cliffs[next(iter(self._cliffs))]
            rows = self.window_rows(snapshot, horizon) if horizon > 0 else self.laps.iloc[0:0]
            self._cliffs[key] = _cliff.as_of_cliffs(rows, self.practice, snapshot.race, wc, self.config.prerace)
        return self._cliffs[key]

    def cliff_ledger(self, snapshot: Snapshot) -> dict:
        """compound -> {edge, source, whole_race_input} of cliffs_for(snapshot) (cliff.world_ledger):
        where each edge came from, and whether it used this race's laps."""
        return _cliff.world_ledger(self.cliffs_for(snapshot))

    def validate(self, snapshot: Snapshot, plan: Plan, rules=None):
        names = self.config.rules if rules is None else rules
        fit = self.fitted_for(snapshot)
        return check(names, RuleContext(snapshot, plan, fit.modelled, self.config, self.stint_limits,
                                        self.stint_limit_sources))

    # --- public API ----------------------------------------------------------
    def model_report(self, snapshot: Snapshot, ages=None,
                     calibration_summary=_report.CALIBRATION_SUMMARY) -> ModelReport:
        """The fit behind this snapshot's window, and what it cannot tell (model_report.build).

        `ages` ({compound: tyre age}) adds requested ages to the extrapolation check, which
        always covers the current set's next lap. `calibration_summary` is the midrace
        summary csv (or frame) calibration_error_s is read from; None gives NaN.
        """
        fit = self.fitted_for(snapshot)
        rep = _report.build(fit.degradation, self.config.fit_window, snapshot, ages, calibration_summary,
                            hazard=fit.hazard, min_samples=self.config.min_samples,
                            as_of=self.config.cross_race_as_of and self.config.fit_window != "full")
        sources = fit.calibration.output.diagnostics.get("sources") if fit.calibration is not None else None
        return rep.model_copy(update={"prior_sources": dict(sources)}) if sources else rep

    def simulate(self, snapshot: Snapshot, plan, *, seed=0, n_samples=None,
                 scenario: Scenario | None = None, rules=None, return_laps=False) -> SimResult:
        return self.fitted_for(snapshot).simulate_batch(
            snapshot, [plan], seed=seed, n_samples=n_samples, scenario=scenario,
            rules=rules, return_laps=return_laps).result(0, snapshot)

    def simulate_batch(self, snapshot: Snapshot, plans, *, seed=0, n_samples=None,
                       scenario: Scenario | None = None, rules=None, return_laps=False) -> BatchResult:
        if (fit := self.fitted_for(snapshot)) is not self:
            return fit.simulate_batch(snapshot, plans, seed=seed, n_samples=n_samples, scenario=scenario,
                                      rules=rules, return_laps=return_laps)
        parsed, errors = [], []
        for raw in plans:
            try:
                plan = Plan.of(raw)
            except ValidationError as exc:
                parsed.append(None)
                errors.append(_messages(exc))
                continue
            errs = self.validate(snapshot, plan, rules)
            parsed.append(plan if errs else apply_inventory(snapshot, plan, self.config))
            errors.append(errs)
        out = BatchResult(plans=parsed, errors=errors, elapsed_s=snapshot.elapsed_s, seed=seed)
        out.support_warnings = [_report.support_warnings(self.degradation, snapshot, p) if ok else []
                                for p, ok in zip(parsed, out.valid)]

        n = n_samples or self.config.n_samples
        laps = np.arange(snapshot.laps_completed + 1, snapshot.race.total_laps + 1)
        n_plans, n_laps = len(parsed), len(laps)
        out.times = np.full((n_plans, n), np.nan)
        out.pit_loss = np.full((n_plans, n), np.nan)
        out.stint_seconds = [None] * n_plans
        out.lap_times = np.full((n_plans, n_laps), np.nan) if return_laps else None
        out.dnf = np.zeros((n_plans, n), dtype=bool)
        out.big_loss = np.zeros((n_plans, n), dtype=bool)
        out.dnf_lap = np.full((n_plans, n), np.nan)
        out.traffic_s = np.zeros((n_plans, n))
        out.passes = np.zeros((n_plans, n), dtype=int)
        out.bunching_s = np.zeros((n_plans, n))
        out.sc_field_delta = np.full((n_plans, n_laps), np.nan) if return_laps else None
        idx = np.flatnonzero(out.valid)
        if len(idx) == 0:
            return out
        # traffic: placed once per call, plan- and sample-independent, draws nothing
        tstate = self.traffic.prepare(snapshot, laps, self.laps, snapshot.driver, ref_lap_s=self.ref_lap_s,
                                      model=self.degradation)
        out.traffic_on = tstate is not None
        if out.traffic_on:
            out.final_pos, out.rejoin_pos, out.pit_window_cars = (np.full((n_plans, n), np.nan) for _ in range(3))
        tw = self.traffic.support_warnings(snapshot, tstate)
        # a projected field (traffic.py) reads nothing after the snapshot; _events_drawn reads
        # the recorded race, so only the recorded ghosts are checked against it
        if tstate is not None and getattr(self.traffic, "reads_future", True) and self._events_drawn(snapshot, scenario):
            tw = [*tw, TRAFFIC_DRAWN_EVENTS]
        if tw:
            for i in idx:
                out.support_warnings[i] = out.support_warnings[i] + tw
        offset = self._pace_offset(snapshot.driver)
        team_pit = self.team_pit_delta(snapshot)   # 0.0 unless config.pit_stops.team
        fq = self.field_queue(snapshot)   # None unless config.sc_queue is on in mode "single"

        # layout: plan-dependent, sample-independent, built once. `ages` are the green-flag
        # ages; under SC/VSC the effective age is sample-dependent (see the sample loop)
        valid = [parsed[i] for i in idx]
        names = sorted(self.modelled)
        code_of = {c: k for k, c in enumerate(names)}
        codes = np.empty((len(valid), n_laps), dtype=int)
        class_of = {c: k for k, c in enumerate(class_names(self.config))}
        class_codes = np.empty((len(valid), n_laps), dtype=int)   # tyre class, see track.py
        ages = np.empty((len(valid), n_laps))
        age0 = np.empty((len(valid), n_laps))                  # the lap's stint starting age
        stint_start = np.empty((len(valid), n_laps), dtype=int)  # index of the stint's first lap
        stint_no = np.empty((len(valid), n_laps), dtype=int)     # the lap's stint, 0 the current one
        stops = np.zeros((len(valid), n_laps))
        n_stints = np.array([len(p.stints) for p in valid])
        bounds = np.full((len(valid), n_stints.max() + 1), n_laps, dtype=int)
        for r, p in enumerate(valid):
            pos = 0
            for j, s in enumerate(p.stints):
                a0 = snapshot.tyre_age if j == 0 else s.start_age
                bounds[r, j] = pos
                codes[r, pos:pos + s.laps] = code_of[s.compound]
                class_codes[r, pos:pos + s.laps] = class_of[compound_class(self.config, s.compound)]
                ages[r, pos:pos + s.laps] = a0 + np.arange(1, s.laps + 1)
                age0[r, pos:pos + s.laps] = a0
                stint_start[r, pos:pos + s.laps] = pos
                stint_no[r, pos:pos + s.laps] = j
                pos += s.laps
                if j < len(p.stints) - 1:
                    stops[r, pos - 1] = 1.0
        lapnums = np.broadcast_to(laps, codes.shape)
        groups = [(name, m, ages[m], lapnums[m]) for name, code in code_of.items()
                  if (m := codes == code).any()]
        padded = np.arange(bounds.shape[1] - 1) >= n_stints[:, None]

        fx, hx = self.config.effects, self.config.hazard_effects
        cal = self.calibration.output if self.calibration is not None else None
        pl = self.pit_loss_inputs(snapshot)
        factors = _cal.EventDraws(cal, {
            "pit_loss_s": pl["pit_loss_s"], "sc_lap_factor": fx.sc_lap_factor,
            "vsc_lap_factor": fx.vsc_lap_factor, "sc_pit_loss_factor": pl["sc_factor"],
            "lap1_extra_s": fx.lap1_extra_s})
        # an as-of pit loss carries its sd: one draw per sample from its own stream, unless the
        # calibrator already draws pit_loss_s
        z_pit = (batch_normals(seed, PIT_LOSS_STREAM, n, 1, pl["sd_s"])[:, 0]
                 if pl["sd_s"] > 0 and "pit_loss_s" not in factors.names else None)
        corr = (_cal.CorrectionLookup(cal.correction, names, codes, laps)
                if cal is not None and cal.correction is not None else None)
        lap1_extra = snapshot.laps_completed == 0 and snapshot.standing_start
        noise_sd = self._lap_noise_sd()
        # "persistent": the offsets on top of the lap-to-lap noise (_lap_noise_offsets)
        off_sd = self._lap_noise_offsets()
        z_off = (batch_normals(seed, LAP_OFFSET_STREAM, n, int(n_stints.max()), off_sd)
                 if off_sd is not None and off_sd > 0 else None)
        # parameter draws: the linear models take row k of one (n, n_params) normal matrix
        # (draw_z); another model's draw(rng) gets the per-sample substream [seed, k, 2]. A
        # model whose draw is "none" returns itself and draws nothing, so nothing is built.
        # Under a wear bound row k of an (n, (draw_tries - 1) n_params) DEGRADATION_RETRY_STREAM
        # matrix follows, the rejection proposals after the first.
        param_draw = getattr(self.degradation, "param_draw", None) != "none"
        z_par = (batch_normals(seed, DEGRADATION_STREAM, n, self.degradation.n_params)
                 if param_draw and hasattr(self.degradation, "draw_z") else None)
        tries = getattr(self.degradation, "draw_tries", 1) if z_par is not None else 1
        z_retry = (batch_normals(seed, DEGRADATION_RETRY_STREAM, n, (tries - 1) * self.degradation.n_params)
                   if tries > 1 else None)
        eps = batch_normals(seed, LAP_NOISE_STREAM, n, n_laps, noise_sd) if noise_sd > 0 else None
        # wear cliff (sim/cliff.py): per compound in the plans, (slope, age per sample)
        cliffs = {name: cl for name, cl in self.cliffs_for(snapshot).items()
                  if name in code_of and cl.kappa > 0 and cl.cap > 0}
        cliff_age = {}
        if cliffs:
            cols = sorted(set(self.config.compound_classes) | set(cliffs))
            z_cl = (batch_normals(seed, CLIFF_STREAM, n, len(cols))
                    if any(cl.prior_sd > 0 for cl in cliffs.values()) and self.config.belief_cliff_edge == "draw"
                    else np.zeros((n, len(cols))))
            cliff_age = {name: cl.sample_ages(z_cl[:, cols.index(name)]) for name, cl in cliffs.items()}
        seg = np.empty((len(valid), bounds.shape[1] - 1, n))
        green_p = None   # hazard probabilities at the green-flag ages, sample-independent
        t_cache = None   # the last traffic call's inputs and outputs (reused for an identical sample)
        for k in range(n):
            # main stream draw order: calibrated factors, correction z, hazard, track: do not
            # reorder, seeds depend on it (the hazard uniform is drawn here, ahead of the track
            # state; its use comes later). The factor and z draws happen only when the
            # calibrator gives a factor sd or a correction sd above 0. Events, the degradation
            # parameters and the lap noise draw from their own streams (EVENT_STREAM per
            # sample, DEGRADATION_STREAM and LAP_NOISE_STREAM per batch), never from rng.
            rng = np.random.default_rng([seed, k])
            scen = scenario or self.events.draw(snapshot, np.random.default_rng([seed, k, EVENT_STREAM]))
            v = factors.sample(rng)
            if z_pit is not None:
                v = {**v, "pit_loss_s": max(0.0, v["pit_loss_s"] + z_pit[k])}
            z = rng.standard_normal() if corr is not None and corr.has_sd else 0.0
            if z_par is not None:
                model = self.degradation.draw_z(z_par[k] if z_retry is None else np.concatenate([z_par[k], z_retry[k]]))
            elif param_draw:
                model = self.degradation.draw(np.random.default_rng([seed, k, DEGRADATION_STREAM]))
            else:
                model = self.degradation
            u = rng.uniform(size=(2, n_laps)) if self.hazard.draws else None
            # u is shared by every plan (common random numbers)
            track = self.track.draw(snapshot, rng)
            sc = np.isin(laps, scen.sc_laps)
            vsc = np.isin(laps, scen.vsc_laps)
            green = np.empty(codes.shape)
            if sc.any() or vsc.any():
                # effective age: neutralised laps age the tyre by sc/vsc_age_increment, and
                # only within the current stint. Ages may be fractional, so a degradation
                # model's lap_times must accept float ages (linear ones do).
                inc = np.where(sc, fx.sc_age_increment, np.where(vsc, fx.vsc_age_increment, 1.0))
                cum = inc.cumsum()
                cum_before = np.concatenate([[0.0], cum[:-1]])
                eff = age0 + cum - cum_before[stint_start]
                for name, m, _, ln in groups:
                    green[m] = model.lap_times(name, eff[m], ln)
            else:
                eff = ages
                for name, m, a, ln in groups:
                    green[m] = model.lap_times(name, a, ln)
            for name, m, _, _ in groups:
                if name in cliffs:   # the cost past the support edge
                    green[m] += cliffs[name].extra(eff[m], cliff_age[name][k])
            if corr is not None:
                green += corr.add(eff, z)
            green *= self.track.factor(track, class_codes, lapnums)
            lap_s = np.where(sc, v["sc_lap_factor"] * self.ref_lap_s,
                             np.where(vsc, v["vsc_lap_factor"] * green, green))
            if scen.lap_time_s:   # recorded neutralised lap times ("recorded" / "hybrid" events), where given
                rec = np.array([scen.lap_time_s.get(int(x), np.nan) for x in laps])
                lap_s = np.where((sc | vsc) & ~np.isnan(rec), rec, lap_s)
            if offset:   # green laps only; skipped at 0 so a run without pace is bit for bit unchanged
                lap_s = np.where(sc | vsc, lap_s, lap_s + offset)
            if eps is not None:   # one vector per sample on the green laps, shared by every plan
                lap_s += np.where(sc | vsc, 0.0, eps[k])
            if z_off is not None:   # persistent: each plan stint's own offset, held over the stint
                lap_s += np.where(sc | vsc, 0.0, z_off[k][stint_no])
            if lap1_extra:
                lap_s[:, 0] += v["lap1_extra_s"]
            scale = np.where(sc, v["sc_pit_loss_factor"], np.where(vsc, pl["vsc_factor"], 1.0))
            if self.hazard.draws:
                if eff is ages:
                    if green_p is None:
                        green_p = self.hazard.per_lap_probability(codes, ages, names)
                    p = green_p
                else:
                    p = self.hazard.per_lap_probability(codes, eff, names)
                hit = p > u[0]
                event = hit.any(axis=1)
                lap = hit.argmax(axis=1)            # the first event lap; one event per sample
                dnf = event & (u[1, lap] < hx.dnf_share)
                big = event & ~dnf
                lap_s[big, lap[big]] += hx.big_loss_s
                # a DNF keeps the remaining laps and stops as if the car had continued and
                # adds dnf_penalty_s once, booked on the lap after the event (on the event
                # lap when it is the last), so a DNF always costs dnf_penalty_s more than
                # finishing. Dropping the laps after the event would make a DNF faster.
                rows = np.flatnonzero(dnf)
                lap_s[rows, np.minimum(lap[rows] + 1, n_laps - 1)] += hx.dnf_penalty_s
                out.dnf[idx, k], out.big_loss[idx, k] = dnf, big
                out.dnf_lap[idx, k] = np.where(dnf, laps[lap], np.nan)
            if fq is not None and sc.any():   # the safety-car queue (sim/sc_queue.py), mode "single"
                pit_q = v["pit_loss_s"] * stops * scale
                for S, E in _queue.runs(laps[sc]):
                    lap_s[:, E - int(laps[0])] += fq.shift(S, E, lap_s, pit_q, int(laps[0]), snapshot.elapsed_s or 0.0,
                                                           v["pit_loss_s"], v["sc_pit_loss_factor"])
            if tstate is not None:
                # the full pit loss is booked at the end of the in-lap, so the out-lap starts
                # from the rejoin position; it stays out of lap_s, as below
                pit_lap = v["pit_loss_s"] * stops * scale + team_pit * stops
                # traffic draws nothing, so a sample whose inputs equal the previous one's (a world
                # that draws no events, no noise and no parameters: every sample) reuses its
                # result; a projected field cycles its variants per sample, so it never does
                same = (t_cache is not None and not getattr(tstate, "variants", None)
                        and np.array_equal(lap_s, t_cache[0]) and np.array_equal(pit_lap, t_cache[1])
                        and np.array_equal(sc, t_cache[2]) and np.array_equal(vsc, t_cache[3])
                        and scen.sc_laps == t_cache[4])
                if same:
                    lap_s, extra, tstate.positions = t_cache[5], t_cache[6], t_cache[7]
                else:
                    key = (lap_s.copy(), pit_lap, sc, vsc, list(scen.sc_laps))
                    lap_s, extra = self.traffic.apply(tstate, lap_s, pit_lap, sc, vsc, scen)
                    t_cache = (*key, lap_s, extra, getattr(tstate, "positions", None))
                out.traffic_s[idx, k], out.passes[idx, k], out.bunching_s[idx, k], field_delta = extra
                pos = getattr(tstate, "positions", None)
                if pos is not None:   # positions at the first rejoin and at the flag
                    out.final_pos[idx, k], out.rejoin_pos[idx, k] = pos["final_pos"], pos["rejoin_pos"]
                    out.pit_window_cars[idx, k] = pos["pit_window_cars"]
                if k == 0 and return_laps:
                    out.sc_field_delta[idx] = field_delta
            pit_s = v["pit_loss_s"] * (stops * scale).sum(axis=1)
            if team_pit:   # the team's typical stop (config.pit_stops.team), on every stop
                pit_s = pit_s + team_pit * stops.sum(axis=1)
            out.times[idx, k] = lap_s.sum(axis=1) + pit_s
            out.pit_loss[idx, k] = pit_s
            cs = np.concatenate([np.zeros((len(valid), 1)), lap_s.cumsum(axis=1)], axis=1)
            sums = np.take_along_axis(cs, bounds[:, 1:], 1) - np.take_along_axis(cs, bounds[:, :-1], 1)
            seg[:, :, k] = np.where(padded, 0.0, sums)
            if k == 0 and return_laps:
                out.lap_times[idx] = lap_s

        for r, i in enumerate(idx):
            out.stint_seconds[i] = seg[r, :n_stints[r]]
        return out

    def team_pit_delta(self, snapshot: Snapshot) -> float:
        """The snapshot car's team's typical stop less the field's (pit_loss.team_delta) under
        config.pit_stops.team, else 0. The full window (the judge) reads every stop of the race; a
        windowed fit the stops completed when the car completed its fit's calibration point."""
        k0 = None if self.config.fit_window == "full" else (
            (self.window[1] or 0) if self.window is not None and self.config.fit_window == "to_snapshot" else 0)
        return self.team_pit_delta_for(snapshot.driver, k0)

    def team_pit_delta_for(self, driver, k0) -> float:
        """team_pit_delta for `driver` from the stops completed when it completed lap k0 (None: every
        stop of the race, the judge's reading)."""
        ps = self.config.pit_stops
        if not ps.team or self.laps is None or "team" not in self.laps:
            return 0.0
        key = ("team_pit", driver, k0)
        if key not in self._pit_loss_cache:
            from . import pit_loss as _pl
            if "stops" not in self._pit_stops:
                self._pit_stops["stops"] = _pl.stop_table(self.laps)
            stops = self._pit_stops["stops"]
            if k0 is not None:
                t = _pl.snapshot_time(self.laps, driver, k0) if k0 > 0 else None
                stops = stops[stops["known_s"] <= t] if t is not None else stops.iloc[0:0]
            team = self.laps.loc[self.laps["driver"] == driver, "team"]
            if len(self._pit_loss_cache) >= FIT_CACHE_SIZE:
                del self._pit_loss_cache[next(iter(self._pit_loss_cache))]
            self._pit_loss_cache[key] = _pl.team_delta(stops, team.iloc[0] if len(team) else None,
                                                       ps.team_prior_stops)[0]
        return self._pit_loss_cache[key]

    def field_queue(self, snapshot: Snapshot):
        """sc_queue.FieldQueue of this snapshot under config.sc_queue (mode "single"), else None.
        The full window (the judge) reads the whole race and the recorded periods' lengths; a
        windowed fit only its window's rows (window_rows at its calibration point, the snapshot's
        own lap k on a pre_race fit reading nothing after lap 0), projected beyond them. Cached."""
        q = self.config.sc_queue
        if not q.enabled or q.mode != "single" or self.laps is None or "end_s" not in self.laps:
            return None
        key = ("sc_queue", self.window, snapshot.driver, snapshot.laps_completed, self.pace is None)
        if key not in self._pit_loss_cache:
            lapped, own_lapped = (), False
            if self.config.fit_window == "full":
                rows, horizon, recorded = self.laps, int(self.laps["lap"].max()), True
                # the recorded classification: cars that finished laps down stayed out of the queue
                last = rows.groupby("driver")["lap"].max()
                lapped = set(last.index[last < last.max()])
                own_lapped = snapshot.race.total_laps < last.max()
            else:
                k0 = (self.window[1] or 0) if self.window is not None and self.config.fit_window == "to_snapshot" else 0
                rows = self.window_rows(snapshot, k0) if k0 > 0 else self.laps.iloc[0:0]
                horizon, recorded = k0, False
            if len(self._pit_loss_cache) >= FIT_CACHE_SIZE:
                del self._pit_loss_cache[next(iter(self._pit_loss_cache))]
            # the car's own pace places it in the queue when no pace term is in its lap times
            pace = (_queue.own_pace(self.degradation, rows[rows["driver"] == snapshot.driver])
                    if self.pace is None and len(rows) else 0.0)
            self._pit_loss_cache[key] = (_queue.FieldQueue(rows, snapshot.driver, horizon, q.spacing_s, self.ref_lap_s,
                                                           recorded, pace, lapped, own_lapped, q.held_behind_laps,
                                                           q.placement)
                                           if len(rows) else None)
        return self._pit_loss_cache[key]

    def pit_loss_inputs(self, snapshot: Snapshot) -> dict:
        """{pit_loss_s, sd_s, sc_factor, vsc_factor, source, ...} this simulation prices a stop
        with. The world's hindsight value when the Simulator was given one (pit_loss_s=...,
        source "race"); under config.pit_loss.source "asof" pit_loss.estimate at the snapshot
        on this instance's fit (source "asof", with its sd, the SC/VSC factors and their
        provenance); else snapshot.race.pit_loss_s and the effects' factors (source "snapshot")."""
        fx, cfg = self.config.effects, self.config.pit_loss
        if self.race_pit_loss_s is not None:
            return {"pit_loss_s": float(self.race_pit_loss_s), "sd_s": 0.0, "sc_factor": fx.sc_pit_loss_factor,
                    "vsc_factor": fx.vsc_pit_loss_factor, "source": "race"}
        if cfg.source != "asof":
            return {"pit_loss_s": float(snapshot.race.pit_loss_s), "sd_s": 0.0, "sc_factor": fx.sc_pit_loss_factor,
                    "vsc_factor": fx.vsc_pit_loss_factor, "source": "snapshot"}
        from . import pit_loss as _pl
        # the information cut is the fit's: on a windowed instance its lap k0, so a continuation's
        # advanced snapshot (laps_completed + n) never reads the stops of the laps it skips ahead
        lap = (self.window[1] or 0) if self.window is not None else snapshot.laps_completed
        key = (snapshot.race.race_id, snapshot.driver, lap, self.window)
        if key not in self._pit_loss_cache:
            if len(self._pit_loss_cache) >= FIT_CACHE_SIZE:
                del self._pit_loss_cache[next(iter(self._pit_loss_cache))]
            if "stops" not in self._pit_stops:
                self._pit_stops["stops"] = _pl.stop_table(self.laps)
            est = _pl.estimate(self.laps, snapshot.race.race_id, snapshot.driver, lap, self.degradation, cfg,
                               self._pit_stops["stops"])
            self._pit_loss_cache[key] = {**est, "source": "asof"}
        return self._pit_loss_cache[key]

    def _lap_noise_sd(self) -> float:
        """SimConfig.lap_noise: 0 for "none", else the fit's residual_s (0 when the fit
        report has none, or it is NaN). Under "persistent" with the fit's residual_components,
        their lap-to-lap part only (the offsets are _lap_noise_offsets')."""
        if self.config.lap_noise == "none":
            return 0.0
        rep = getattr(self.degradation, "fit_report", None) or {}
        comp = rep.get("residual_components")
        if self.config.lap_noise == "persistent" and comp:
            return float(comp["lap_s"])
        s = float((rep.get("sd") or {}).get("residual_s", float("nan")))
        return 0.0 if np.isnan(s) else s

    def _lap_noise_offsets(self) -> float | None:
        """The stint-offset sd of lap_noise "persistent" (docs/eval.md "Honest uncertainty"): the
        driver-stint offset sd of the fit's residual_components, sqrt(driver_s^2 + stint_s^2)
        (degradation._residual_components on the rows the fit saw, so as of the snapshot under
        to_snapshot; early in a race every car is on its first stint and the two cannot be told
        apart). Each sample draws one offset per plan stint, held on that stint's green laps; the
        lap-to-lap part is the iid noise. None otherwise (no offsets drawn)."""
        if self.config.lap_noise != "persistent":
            return None
        comp = (getattr(self.degradation, "fit_report", None) or {}).get("residual_components")
        return float(np.hypot(comp["driver_s"], comp["stint_s"])) if comp else None

    def _events_drawn(self, snapshot: Snapshot, scenario: Scenario | None) -> bool:
        """Whether the samples' SC and VSC laps may differ from the recorded race's: an event
        model other than "none", or a fixed scenario (the `scenario` argument or the "fixed"
        model) other than the record over the remaining laps. Recorded ghosts are valid only
        under the recorded events: under a drawn SC the field ran green, so positions and
        traffic are approximate there."""
        if scenario is None:
            if self.config.events.name in ("none", "recorded"):   # "recorded" is the record itself
                return False
            if self.config.events.name != "fixed":
                return True
            scenario = self.events.scenario
        first, last = snapshot.laps_completed + 1, snapshot.race.total_laps
        g = self.laps[self.laps["driver"] == snapshot.driver] if "driver" in self.laps else self.laps.iloc[0:0]
        g = g[(g["lap"] >= first) & (g["lap"] <= last)]
        status = g["track_status"].astype(str) if "track_status" in g else pd.Series(dtype=str)

        def laps(xs):
            return {int(x) for x in xs if first <= int(x) <= last}
        recorded = (laps(g.loc[status.str.contains("4"), "lap"]), laps(g.loc[status.str.contains("[67]"), "lap"]))
        return (laps(scenario.sc_laps), laps(scenario.vsc_laps)) != recorded

    def compare(self, snapshot: Snapshot, candidates, *, seed=0, n_samples=None,
                scenario: Scenario | None = None, rules=None, continuation=None) -> CompareResult:
        """Score plans and stay/pit actions against each other under one set of draws.

        `continuation(snapshot_after)` returns the plan for the laps after an action.
        """
        if (fit := self.fitted_for(snapshot)) is not self:
            return fit.compare(snapshot, candidates, seed=seed, n_samples=n_samples, scenario=scenario,
                               rules=rules, continuation=continuation)
        plans, errors = [], []
        for raw in candidates:
            if isinstance(raw, dict) and "action" in raw:
                plan, errs = self.expand_action(snapshot, raw, continuation)
            else:
                plan, errs = raw, []
            plans.append(plan)
            errors.append(errs)
        todo = [i for i, e in enumerate(errors) if not e]
        batch = self.simulate_batch(snapshot, [plans[i] for i in todo], seed=seed, n_samples=n_samples,
                                    scenario=scenario, rules=rules)
        n_cand, n = len(plans), batch.times.shape[1]
        times = np.full((n_cand, n), np.nan)
        times[todo] = batch.times
        valid = np.zeros(n_cand, dtype=bool)
        valid[todo] = batch.valid
        warnings = [[] for _ in plans]
        traffic = [None] * len(plans) if batch.traffic_on else None
        for r, i in enumerate(todo):
            plans[i], errors[i], warnings[i] = batch.plans[r], batch.errors[r], batch.support_warnings[r]
            if traffic is not None:
                traffic[i] = batch.traffic_estimate(r)

        idx = np.flatnonzero(valid)
        summaries = [summarise(times[i]) if valid[i] else None for i in range(n_cand)]
        p_best = [None] * n_cand
        if len(idx):
            t = times[idx]
            win = t == t.min(axis=0)
            for i, share in zip(idx, (win / win.sum(axis=0)).mean(axis=1)):
                p_best[i] = float(share)
        ti, tj = times[:, None, :], times[None, :, :]
        beats = (ti < tj).mean(axis=2) + 0.5 * (ti == tj).mean(axis=2)
        gap = ti - tj
        pair = valid[:, None] & valid[None, :]
        mean = {i: summaries[i].mean_s for i in idx}
        cvar = {i: summaries[i].cvar_90_s for i in idx}
        dominated = [bool(valid[i]) and any(
            mean[j] <= mean[i] and cvar[j] <= cvar[i] and (mean[j] < mean[i] or cvar[j] < cvar[i])
            for j in idx if j != i) for i in range(n_cand)]
        return CompareResult(
            candidates=plans, valid=valid.tolist(), errors=errors, summaries=summaries, p_best=p_best,
            p_beats=_matrix(beats, pair & ~np.eye(n_cand, dtype=bool)),
            gap_mean_s=_matrix(gap.mean(axis=2), pair), gap_std_s=_matrix(gap.std(axis=2), pair),
            best=int(idx[np.argmin([mean[i] for i in idx])]) if len(idx) else None,
            dominated=dominated, n_samples=n, seed=seed, support_warnings=warnings, traffic=traffic,
        )

    def expand_action(self, snapshot: Snapshot, raw, continuation):
        """Turn a stay or pit action into a full plan from this snapshot. Returns (plan, errors).

        `raw` is {"action": "stay", "laps": n} (n defaults to 1) or {"action": "pit",
        "compound": c} with optional set_id and start_age. `continuation(snapshot_after)`
        returns the plan for the laps after the action: n laps on after a stay, one lap on
        after a pit. A stay must leave at least one lap for the continuation.

        A stay taken under SC or VSC is not advanced: _advance would keep the period on at the
        reassess lap with certainty. When the continuation has `stay_plan(snapshot, n)`
        (optimizer_bridge.as_continuation) the plan is that, the best full plan from this
        snapshot with no stop before the end of lap laps_completed + n + 1, on which the period
        in progress runs on as the event model draws it per sample. A green stay, and a plain
        continuation callable, advance the snapshot as before."""
        try:
            act = _Action.model_validate(raw)
        except ValidationError as exc:
            return None, _messages(exc)
        if continuation is None:
            return None, ["continuation required for action candidates"]
        stay_plan = getattr(continuation, "stay_plan", None)
        if act.action == "stay" and snapshot.track_status != "green" and stay_plan is not None:
            try:
                plan = Plan.of(stay_plan(snapshot, act.laps))
            except ValidationError as exc:
                return None, _messages(exc)
            if plan.stints[0].compound != snapshot.compound or plan.stints[0].laps <= act.laps:
                return None, ["a stay's plan must run the current set past the stay"]
            return plan, []
        chosen = None
        try:
            if act.action == "stay":
                advanced = _advance(snapshot, act.laps, tyre_age=snapshot.tyre_age + act.laps)
            else:
                pool = [t for t in snapshot.sets_available or [] if t.source in self.config.inventory_sources]
                age = act.start_age or 0
                chosen = next((t for t in pool if (t.set_id == act.set_id if act.set_id is not None
                               else t.compound == act.compound and t.laps_run == age)), None)
                if chosen is not None and act.set_id is not None and age not in (0, chosen.laps_run):
                    return None, [f"start_age {act.start_age} conflicts with set '{chosen.set_id}' "
                                  f"({chosen.laps_run} laps run)"]
                # laps on the fitted set in this race, from the stint history when it adds up
                in_race = min(snapshot.tyre_age, snapshot.laps_completed - sum(s.laps for s in snapshot.stints))
                done = StintRecord(compound=snapshot.compound, laps=in_race + 1,
                                   start_lap=snapshot.laps_completed - in_race + 1,
                                   start_age=snapshot.tyre_age - in_race)
                update = {"compound": act.compound, "tyre_age": chosen.laps_run if chosen else age,
                          "stints": [*snapshot.stints, done]}
                if chosen is not None:
                    update["sets_available"] = [t for t in snapshot.sets_available if t is not chosen]
                advanced = _advance(snapshot, **update)
        except ValidationError as exc:
            return None, _messages(exc)
        try:
            cont = Plan.of(continuation(advanced))
        except ValidationError as exc:
            return None, _messages(exc)
        first = cont.stints[0]
        if first.compound != advanced.compound or (act.action == "stay" and first.start_age != 0):
            return None, ["continuation must continue the current set"]
        if act.action == "stay":
            stints = [PlanStint(compound=snapshot.compound, laps=act.laps + first.laps), *cont.stints[1:]]
        else:
            carried = first.model_copy(update={"set_id": chosen.set_id if chosen else act.set_id,
                                               "start_age": advanced.tyre_age})
            stints = [PlanStint(compound=snapshot.compound, laps=1), carried, *cont.stints[1:]]
        return Plan(stints=stints), []

    _expand = expand_action   # the name before it was public
