"""Strategy simulator: a snapshot and plans in, lap time summaries out."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..contracts import Plan, SimResult, Snapshot, TimeSummary
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
        )


class Simulator:
    """Fitted on one race's lap frame. Cheap to call repeatedly."""

    def __init__(self, laps: pd.DataFrame, config: SimConfig | None = None):
        self.config = config or SimConfig()
        spec = self.config.degradation
        self.degradation = DEGRADATION.get(spec.name).fit(
            laps, min_samples=self.config.min_samples, **spec.params)
        ev = self.config.events
        self.events = EVENTS.get(ev.name)(**ev.params)
        self.ref_lap_s = float(laps.loc[laps["clean"], "lap_time_s"].median())

    @property
    def modelled(self):
        return self.degradation.compounds

    def validate(self, snapshot: Snapshot, plan: Plan, rules=None):
        names = self.config.rules if rules is None else rules
        return check(names, RuleContext(snapshot, plan, self.modelled, self.config))

    # --- public API ----------------------------------------------------------
    def simulate(self, snapshot: Snapshot, plan, *, seed=0, n_samples=None,
                 scenario: Scenario | None = None, rules=None, return_laps=False) -> SimResult:
        return self.simulate_batch(snapshot, [plan], seed=seed, n_samples=n_samples, scenario=scenario,
                                   rules=rules, return_laps=return_laps).result(0, snapshot)

    def simulate_batch(self, snapshot: Snapshot, plans, *, seed=0, n_samples=None,
                       scenario: Scenario | None = None, rules=None, return_laps=False) -> BatchResult:
        parsed, errors = [], []
        for raw in plans:
            try:
                plan = Plan.of(raw)
            except ValidationError as exc:
                parsed.append(None)
                errors.append([e["msg"] for e in exc.errors()])
                continue
            errs = self.validate(snapshot, plan, rules)
            parsed.append(plan if errs else apply_inventory(snapshot, plan, self.config))
            errors.append(errs)
        out = BatchResult(plans=parsed, errors=errors, elapsed_s=snapshot.elapsed_s, seed=seed)

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
