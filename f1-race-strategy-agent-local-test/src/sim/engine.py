"""Strategy simulator: a snapshot and a plan in, a lap time summary out."""
import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..contracts import Plan, SimResult, Snapshot, TimeSummary
from .config import SimConfig
from .degradation import DEGRADATION
from .events import EVENTS, Scenario
from .rules import RuleContext, check


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
        return check(names, RuleContext(snapshot, plan, self.modelled))

    # --- public API ----------------------------------------------------------
    def simulate(self, snapshot: Snapshot, plan, *, seed=0, n_samples=None,
                 scenario: Scenario | None = None, rules=None, return_laps=False) -> SimResult:
        try:
            plan = Plan.of(plan)
        except ValidationError as exc:
            return SimResult(valid=False, errors=[e["msg"] for e in exc.errors()])
        errors = self.validate(snapshot, plan, rules)
        if errors:
            return SimResult(valid=False, errors=errors, plan=plan)

        n = n_samples or self.config.n_samples
        runs = [self._run(snapshot, plan, np.random.default_rng([seed, i]), scenario) for i in range(n)]
        totals = np.array([r["total"] for r in runs])
        summary = TimeSummary(
            mean_s=float(totals.mean()), std_s=float(totals.std()),
            p05_s=float(np.percentile(totals, 5)), p50_s=float(np.percentile(totals, 50)),
            p95_s=float(np.percentile(totals, 95)), n_samples=n,
            cvar_90_s=float(np.sort(totals)[-max(1, int(np.ceil(0.1 * n))):].mean()),
        )
        return SimResult(
            valid=True, plan=plan, num_stops=plan.num_stops, pit_laps=plan.pit_laps(snapshot),
            remaining=summary, elapsed_s=snapshot.elapsed_s,
            stint_seconds=np.mean([r["stints"] for r in runs], axis=0).tolist(),
            pit_loss_total_s=float(np.mean([r["pit"] for r in runs])),
            lap_times_s=runs[0]["laps"].tolist() if return_laps else None,
        )

    # --- internals -----------------------------------------------------------
    def _run(self, snapshot, plan, rng, scenario):
        fx = self.config.effects
        scen = scenario or self.events.draw(snapshot, rng)
        model = self.degradation.draw(rng)
        race = snapshot.race

        laps_all = np.arange(snapshot.laps_completed + 1, race.total_laps + 1)
        ages, comps, ends = [], [], []
        pos = 0
        for i, s in enumerate(plan.stints):
            age0 = snapshot.tyre_age if i == 0 else s.start_age
            ages.append(age0 + np.arange(1, s.laps + 1))
            comps.append(s.compound)
            pos += s.laps
            ends.append(pos)

        green = np.concatenate([
            model.lap_times(c, a, laps_all[e - len(a):e]) for c, a, e in zip(comps, ages, ends)])
        sc = np.isin(laps_all, scen.sc_laps)
        vsc = np.isin(laps_all, scen.vsc_laps)
        laps = np.where(sc, fx.sc_lap_factor * self.ref_lap_s, np.where(vsc, fx.vsc_lap_factor * green, green))

        stop_idx = np.array(ends[:-1], dtype=int) - 1  # in-lap index of each stop
        pit_scale = np.where(sc[stop_idx], fx.sc_pit_loss_factor,
                             np.where(vsc[stop_idx], fx.vsc_pit_loss_factor, 1.0))
        pit = float(race.pit_loss_s * pit_scale.sum())

        bounds = [0] + ends
        stints = [float(laps[a:b].sum()) for a, b in zip(bounds[:-1], bounds[1:])]
        return {"laps": laps, "stints": stints, "pit": pit, "total": float(laps.sum()) + pit}
