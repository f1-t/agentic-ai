"""Strategy simulator: a snapshot and plans in, lap time summaries out."""
import copy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..contracts import (Compound, CompareResult, ModelReport, Plan, PlanStint, SimResult, Snapshot,
                         StintRecord, TimeSummary)
from . import model_report as _report
from .config import SimConfig
from .degradation import DEGRADATION
from .events import EVENTS, Scenario
from .rules import RuleContext, apply_inventory, check


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
    """A decision at the start of lap laps_completed + 1: stay out, or pit at its end."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["stay", "pit"]
    compound: Compound | None = None
    set_id: str | None = None
    start_age: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if self.action == "pit" and self.compound is None:
            raise ValueError("pit action needs a compound")
        return self


def _advance(snapshot: Snapshot, **update) -> Snapshot:
    """The snapshot one lap on. Nested models are passed through, not re-dumped."""
    data = {name: getattr(snapshot, name) for name in Snapshot.model_fields}
    data.update(laps_completed=snapshot.laps_completed + 1, elapsed_s=None, **update)
    return Snapshot.model_validate(data)


def _matrix(values, mask):
    return [[float(v) if m else None for v, m in zip(row, ok)] for row, ok in zip(values, mask)]


@dataclass
class BatchResult:
    """Many plans scored under the same draws. Rows follow the input plan order."""

    plans: list            # Plan, or None when the input did not parse
    errors: list           # list[str] per plan, empty when valid
    valid: np.ndarray = field(init=False)
    times: np.ndarray = None          # (n_plans, n_samples) remaining seconds incl. pit loss, NaN if invalid
    pit_loss: np.ndarray = None       # (n_plans, n_samples)
    stint_seconds: list = None        # per plan (n_stints, n_samples), pit loss excluded, None if invalid
    lap_times: np.ndarray | None = None  # (n_plans, n_laps) sample 0 lap times when requested
    elapsed_s: float | None = None
    seed: int = 0
    support_warnings: list = None     # list[str] per plan, empty when invalid

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
            support_warnings=self.support_warnings[i] if self.support_warnings else [],
        )


FIT_CACHE_SIZE = 64  # windowed fits kept per Simulator, oldest dropped first


class Simulator:
    """Fitted on one race's lap frame. Cheap to call repeatedly.

    `practice` is a frame of practice long-run laps (data.practice_frame: FP1-FP3
    concatenated, with a "session" column), needed by the pre_race and early
    to_snapshot fit windows. The instance itself carries the configured window's fit
    at lap 0: every race lap for "full", and the practice fit for "pre_race" and
    "to_snapshot" (every race lap when there is no practice). `fitted_for(snapshot)`
    gives the instance fitted on the window `config.fit_window` picks for that snapshot.
    """

    def __init__(self, laps: pd.DataFrame, config: SimConfig | None = None,
                 practice: pd.DataFrame | None = None):
        self.config = config or SimConfig()
        self.laps, self.practice = laps, practice
        self.window = None   # fitted_for cache key on a windowed instance, None on the root
        self._fits = {}
        ev = self.config.events
        self.events = EVENTS.get(ev.name)(**ev.params)
        if self.config.fit_window != "full" and self._has_practice:
            self._fit(self.practice.assign(lap=np.nan))
        else:
            self._fit(laps)

    @property
    def modelled(self):
        return self.degradation.compounds

    @property
    def _has_practice(self):
        return self.practice is not None and not self.practice.empty

    def _model(self, frame, **extra):
        spec, cfg = self.config.degradation, self.config
        return DEGRADATION.get(spec.name).fit(
            frame, min_samples=cfg.min_samples, fuel_prior=cfg.pre_race_fuel_slope_s_per_lap,
            base_delta=cfg.pre_race_base_delta_s, **spec.params, **extra)

    def _fit(self, frame, ref=None, **extra):
        """Fit the degradation model on `frame`; ref_lap_s is the median clean lap of `ref`."""
        ref = frame if ref is None else ref
        self.degradation = self._model(frame, **extra)
        self.ref_lap_s = float(ref.loc[ref["clean"].astype(bool), "lap_time_s"].median())

    def fitted_for(self, snapshot: Snapshot) -> "Simulator":
        """The instance fitted on this snapshot's fit window. Built once per key, then cached."""
        cfg = self.config
        if self.window is not None or (cfg.fit_window == "full" and not cfg.exclude_snapshot_driver):
            return self
        key = (cfg.fit_window, snapshot.laps_completed if cfg.fit_window == "to_snapshot" else None,
               snapshot.driver if cfg.exclude_snapshot_driver else None)
        if key not in self._fits:
            if len(self._fits) >= FIT_CACHE_SIZE:
                del self._fits[next(iter(self._fits))]
            fit = copy.copy(self)
            fit.window, fit._fits = key, {}
            fit._fit_window(snapshot)
            self._fits[key] = fit
        return self._fits[key]

    def _fit_window(self, snapshot: Snapshot):
        """Refit this instance on the snapshot's window.

        full: every race lap. pre_race: practice only. to_snapshot: every car's race laps
        up to laps_completed (they are public), shrunk toward the pre_race fit, which is
        the only way practice enters; ref_lap_s is the median of both. Exclusion drops
        the snapshot driver from whichever frames apply.
        """
        cfg = self.config
        drop = snapshot.driver if cfg.exclude_snapshot_driver else None

        def keep(f):
            return f[f["driver"] != drop] if drop is not None and "driver" in f else f

        race = keep(self.laps)
        if cfg.fit_window == "full":
            return self._fit(race)
        race = race[race["lap"] <= (snapshot.laps_completed if cfg.fit_window == "to_snapshot" else 0)]
        practice = keep(self.practice.assign(lap=np.nan)) if self._has_practice else None
        if practice is None and race.empty:
            raise ValueError("fit window needs practice laps")
        if cfg.fit_window == "pre_race":
            return self._fit(practice)
        prior = self._model(practice) if practice is not None else None
        ref = race if practice is None else pd.concat([race, practice], ignore_index=True)
        self._fit(race, ref=ref, prior=prior, shrinkage=cfg.shrinkage)

    def validate(self, snapshot: Snapshot, plan: Plan, rules=None):
        names = self.config.rules if rules is None else rules
        fit = self.fitted_for(snapshot)
        return check(names, RuleContext(snapshot, plan, fit.modelled, self.config))

    # --- public API ----------------------------------------------------------
    def model_report(self, snapshot: Snapshot, ages=None,
                     calibration_summary=_report.CALIBRATION_SUMMARY) -> ModelReport:
        """The fit behind this snapshot's window, and what it cannot tell (model_report.build).

        `ages` ({compound: tyre age}) adds requested ages to the extrapolation check, which
        always covers the current set's next lap. `calibration_summary` is the midrace
        summary csv (or frame) calibration_error_s is read from; None gives NaN.
        """
        fit = self.fitted_for(snapshot)
        return _report.build(fit.degradation, self.config.fit_window, snapshot, ages, calibration_summary,
                             hazard=getattr(fit, "hazard", None))

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
        idx = np.flatnonzero(out.valid)
        if len(idx) == 0:
            return out

        # layout: plan-dependent, sample-independent, built once
        valid = [parsed[i] for i in idx]
        code_of = {c: k for k, c in enumerate(sorted(self.modelled))}
        codes = np.empty((len(valid), n_laps), dtype=int)
        ages = np.empty((len(valid), n_laps))
        stops = np.zeros((len(valid), n_laps))
        n_stints = np.array([len(p.stints) for p in valid])
        bounds = np.full((len(valid), n_stints.max() + 1), n_laps, dtype=int)
        for r, p in enumerate(valid):
            pos = 0
            for j, s in enumerate(p.stints):
                age0 = snapshot.tyre_age if j == 0 else s.start_age
                bounds[r, j] = pos
                codes[r, pos:pos + s.laps] = code_of[s.compound]
                ages[r, pos:pos + s.laps] = age0 + np.arange(1, s.laps + 1)
                pos += s.laps
                if j < len(p.stints) - 1:
                    stops[r, pos - 1] = 1.0
        lapnums = np.broadcast_to(laps, codes.shape)
        groups = [(name, m, ages[m], lapnums[m]) for name, code in code_of.items()
                  if (m := codes == code).any()]
        padded = np.arange(bounds.shape[1] - 1) >= n_stints[:, None]

        fx = self.config.effects
        pit_loss_s = snapshot.race.pit_loss_s
        lap1_extra = snapshot.laps_completed == 0 and snapshot.standing_start
        seg = np.empty((len(valid), bounds.shape[1] - 1, n))
        for k in range(n):
            rng = np.random.default_rng([seed, k])
            scen = scenario or self.events.draw(snapshot, rng)
            model = self.degradation.draw(rng)
            green = np.empty(codes.shape)
            for name, m, a, ln in groups:
                green[m] = model.lap_times(name, a, ln)
            sc = np.isin(laps, scen.sc_laps)
            vsc = np.isin(laps, scen.vsc_laps)
            lap_s = np.where(sc, fx.sc_lap_factor * self.ref_lap_s, np.where(vsc, fx.vsc_lap_factor * green, green))
            if lap1_extra:
                lap_s[:, 0] += fx.lap1_extra_s
            scale = np.where(sc, fx.sc_pit_loss_factor, np.where(vsc, fx.vsc_pit_loss_factor, 1.0))
            pit_s = pit_loss_s * (stops * scale).sum(axis=1)
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
                plan, errs = self._expand(snapshot, raw, continuation)
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
        for r, i in enumerate(todo):
            plans[i], errors[i], warnings[i] = batch.plans[r], batch.errors[r], batch.support_warnings[r]

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
            dominated=dominated, n_samples=n, seed=seed, support_warnings=warnings,
        )

    def _expand(self, snapshot: Snapshot, raw, continuation):
        """Turn a stay or pit action into a full plan from this snapshot. Returns (plan, errors)."""
        try:
            act = _Action.model_validate(raw)
        except ValidationError as exc:
            return None, _messages(exc)
        if continuation is None:
            return None, ["continuation required for action candidates"]
        chosen = None
        try:
            if act.action == "stay":
                advanced = _advance(snapshot, tyre_age=snapshot.tyre_age + 1)
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
            stints = [PlanStint(compound=snapshot.compound, laps=1 + first.laps), *cont.stints[1:]]
        else:
            carried = first.model_copy(update={"set_id": chosen.set_id if chosen else act.set_id,
                                               "start_age": advanced.tyre_age})
            stints = [PlanStint(compound=snapshot.compound, laps=1), carried, *cont.stints[1:]]
        return Plan(stints=stints), []
