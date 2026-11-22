"""Safety car and virtual safety car scenarios for the remaining laps.

An event model draws a `Scenario` per sample. Drawing depends on the snapshot and
rng only, never on the plan, so every plan sees the same events for a given seed.
"""
from pydantic import BaseModel, Field

from ..contracts import Snapshot
from .config import EventModel
from .registry import Registry

EVENTS = Registry("event model")


class Scenario(BaseModel):
    sc_laps: list[int] = Field(default_factory=list)   # absolute lap numbers
    vsc_laps: list[int] = Field(default_factory=list)
    # recorded lap time of a neutralised lap (events "recorded" with lap_times "recorded"): the
    # engine uses it instead of the sc/vsc lap factor on that lap; empty for every other model
    lap_time_s: dict[int, float] = Field(default_factory=dict)


@EVENTS.register("none")
class NoEvents:
    def __init__(self, **_):
        pass

    def draw(self, snapshot: Snapshot, rng) -> Scenario:
        return Scenario()


@EVENTS.register("fixed")
class FixedEvents:
    """Same neutralised laps on every sample. Used for replays and tests."""

    def __init__(self, sc_laps=(), vsc_laps=(), **_):
        self.scenario = Scenario(sc_laps=list(sc_laps), vsc_laps=list(vsc_laps))

    def draw(self, snapshot: Snapshot, rng) -> Scenario:
        return self.scenario


@EVENTS.register("recorded")
class RecordedEvents:
    """The race's own neutralised laps, for the world (the scorer may know them; docs/eval.md
    "Pit loss and track status"). Each driver gets the SC and VSC laps of their own recorded
    track status (4 = SC, 6/7 = VSC, as calibration._scenario reads it for the replays), every
    sample alike; a driver the frame does not have gets the race's majority periods
    (calibration.observed_periods). With lap_times "recorded" each such lap also carries the
    field's median recorded lap time on it (non-pit laps of that status), which the engine uses
    instead of sc_lap_factor x ref_lap_s or vsc_lap_factor x green; "factor" keeps the factors.
    Simulator binds the race frame (bind) before the first draw."""

    def __init__(self, lap_times: str = "recorded", **_):
        if lap_times not in ("recorded", "factor"):
            raise ValueError(f"recorded events: lap_times must be 'recorded' or 'factor', got {lap_times!r}")
        self.lap_times = lap_times
        self.by_driver, self.race_laps, self.medians = {}, Scenario(), {"sc": {}, "vsc": {}}

    def bind(self, laps):
        import pandas as pd
        if laps is None or "track_status" not in laps or laps.empty:
            return self
        status = laps["track_status"].astype(str)
        sc, vsc = status.str.contains("4"), status.str.contains("[67]") & ~status.str.contains("4")
        for d, g in laps.assign(sc=sc, vsc=vsc).groupby("driver"):
            self.by_driver[str(d)] = (sorted(int(x) for x in g.loc[g["sc"], "lap"]),
                                      sorted(int(x) for x in g.loc[g["vsc"], "lap"]))
        share = laps[["lap"]].assign(sc=sc, vsc=vsc).groupby("lap")[["sc", "vsc"]].mean()
        self.race_laps = Scenario(sc_laps=[int(x) for x in share.index[share["sc"] > 0.5]],
                                  vsc_laps=[int(x) for x in share.index[(share["vsc"] > 0.5) & ~(share["sc"] > 0.5)]])
        if self.lap_times == "recorded":
            plain = ~laps["pit_in"].astype(bool) & ~laps["pit_out"].astype(bool) & laps["lap_time_s"].notna()
            for kind, mask in (("sc", sc), ("vsc", vsc)):
                med = laps[plain & mask].groupby("lap")["lap_time_s"].median()
                self.medians[kind] = {int(k): float(v) for k, v in med.items() if pd.notna(v)}
        return self

    def draw(self, snapshot: Snapshot, rng) -> Scenario:
        hit = self.by_driver.get(snapshot.driver)
        sc, vsc = hit if hit is not None else (self.race_laps.sc_laps, self.race_laps.vsc_laps)
        first = snapshot.laps_completed + 1
        sc, vsc = [x for x in sc if x >= first], [x for x in vsc if x >= first]
        times = {**{x: self.medians["sc"][x] for x in sc if x in self.medians["sc"]},
                 **{x: self.medians["vsc"][x] for x in vsc if x in self.medians["vsc"]}}
        return Scenario(sc_laps=sc, vsc_laps=vsc, lap_time_s=times)


@EVENTS.register("rate")
class RateEvents:
    """SC and VSC periods started at a constant rate per green lap (params: config.EventModel).

    The walk covers laps laps_completed + 1 to the end; the history is never redrawn. On a
    lap with no period active an SC period starts with probability p_sc, else a VSC
    period with probability p_vsc (one uniform per lap, SC checked first). A period lasts
    a geometric number of laps with mean d (at least 1) and at most one is active. A
    snapshot taken under SC or VSC starts inside that period with a freshly drawn
    remaining length, which the geometric's lack of memory makes the right law.

    Rates come from the race when its RaceParams carries them (sc_rate_per_lap,
    sc_mean_laps, vsc_rate_per_lap, vsc_mean_laps), else from these params. The engine
    passes the events substream (engine.EVENT_STREAM), so these draws never shift the
    degradation, hazard or track draws. With both rates zero and the snapshot green,
    nothing is drawn.

    The optional lap-1 term (params p_sc_lap1, p_vsc_lap1, SPEC 0011b) replaces p_sc and
    p_vsc on lap 1 when the walk starts there (laps_completed 0); the same uniform decides
    it, so the stream is the same with or without the term. None, the default, keeps p on
    lap 1 like every lap.
    """

    # duration "geometric" (the default, as before): every period, and the rest of one in
    # progress, a geometric number of laps with mean d (memoryless, at least one more lap).
    # "history": a period's length is drawn from the lengths of the neutralised periods of the
    # history table (race_history.period_lengths: as of the race for the belief, pooled for the
    # judge with `pooled`), and a period in progress for e laps continues by L - e laps with L
    # drawn given L >= e (0: it ends at the snapshot). On the calibration races the geometric
    # overstated the rest of an SC in progress by 1.8 laps on average (3.9 predicted at every
    # elapsed lap against 3.7, 2.7, 1.7, 1.7, 1.0, 0.2 real after 1..6 laps) and of a VSC by 1.7.
    # `pooled` (the judge's later periods): the rates and lengths pooled over every dry race in
    # the table (race_history.pooled_event_rates) instead of the params.
    def __init__(self, as_of: bool = False, duration: str = "geometric", pooled: bool = False, **params):
        if duration not in ("geometric", "history"):
            raise ValueError(f"events duration must be 'geometric' or 'history', not {duration!r}")
        self.as_of = bool(as_of)
        self.duration, self.pooled = duration, bool(pooled)
        self.params = EventModel(**params)

    def defaults(self, race) -> tuple:
        """(p_sc, d_sc, p_vsc, d_vsc) where the race carries no circuit rates: the params, or
        with `as_of` (configs/sim/belief.yaml) the rates fitted on the races before this one
        (race_history.as_of_event_rates, an a priori prior with none): what a policy may know."""
        if self.as_of:
            from .race_history import as_of_event_rates
            r = as_of_event_rates(race.race_id)
            return r["p_sc"], r["d_sc"], r["p_vsc"], r["d_vsc"]
        if self.pooled:
            from .race_history import pooled_event_rates
            r = pooled_event_rates()
            return r["p_sc"], r["d_sc"], r["p_vsc"], r["d_vsc"]
        cfg = self.params
        return cfg.p_sc, cfg.d_sc, cfg.p_vsc, cfg.d_vsc

    def rates(self, race):
        """(p_sc, d_sc, p_vsc, d_vsc) for this race."""
        p_sc, d_sc, p_vsc, d_vsc = self.defaults(race)

        def pick(override, default):
            return default if override is None else override

        out = (pick(race.sc_rate_per_lap, p_sc), pick(race.sc_mean_laps, d_sc),
               pick(race.vsc_rate_per_lap, p_vsc), pick(race.vsc_mean_laps, d_vsc))
        # RaceParams checks each field alone, not the sum, so check it after the overrides
        if out[0] + out[2] > 1:
            raise ValueError(f"race {race.race_id}: p_sc + p_vsc = {out[0] + out[2]} exceeds 1")
        return out

    def lap1_rates(self, race):
        """(p_sc, p_vsc) on lap 1 of a walk that starts there: the lap-1 params where set,
        else this race's p."""
        p_sc, _, p_vsc, _ = self.rates(race)
        cfg = self.params
        out = (p_sc if cfg.p_sc_lap1 is None else cfg.p_sc_lap1, p_vsc if cfg.p_vsc_lap1 is None else cfg.p_vsc_lap1)
        if out[0] + out[1] > 1:
            raise ValueError(f"race {race.race_id}: lap-1 p_sc + p_vsc = {out[0] + out[1]} exceeds 1")
        return out

    def length_pmf(self, race, kind):
        """The period length distribution of duration "history" (race_history.period_lengths)."""
        from .race_history import period_lengths
        return period_lengths(race.race_id, kind, "pooled" if self.pooled or not self.as_of else "as_of")["pmf"]

    def _length(self, race, kind, rng, at_least=1) -> int:
        """A period length drawn from length_pmf given at least `at_least` laps."""
        pmf = self.length_pmf(race, kind)[at_least - 1:] if at_least <= len(self.length_pmf(race, kind)) else []
        if not len(pmf) or pmf.sum() <= 0:
            return at_least
        return at_least + int(rng.choice(len(pmf), p=pmf / pmf.sum()))

    @staticmethod
    def elapsed(snapshot: Snapshot) -> int:
        """Laps of the snapshot's SC or VSC so far: the run of its laps ending at laps_completed."""
        laps = set(snapshot.sc_laps if snapshot.track_status == "sc" else snapshot.vsc_laps)
        e, lap = 0, snapshot.laps_completed
        while lap in laps:
            e, lap = e + 1, lap - 1
        return max(e, 1)

    def draw(self, snapshot: Snapshot, rng) -> Scenario:
        p_sc, d_sc, p_vsc, d_vsc = self.rates(snapshot.race)
        status = snapshot.track_status
        first, last = snapshot.laps_completed + 1, snapshot.race.total_laps
        lap1 = None
        if first == 1 and (self.params.p_sc_lap1 is not None or self.params.p_vsc_lap1 is not None):
            lap1 = self.lap1_rates(snapshot.race)
        if p_sc + p_vsc == 0 and status == "green" and (lap1 is None or sum(lap1) == 0):
            return Scenario()
        mean = {"sc": d_sc, "vsc": d_vsc}
        hist = self.duration == "history"
        active, left = None, 0
        if status != "green":
            if hist:
                e = self.elapsed(snapshot)
                left = self._length(snapshot.race, status, rng, e) - e
                active = status if left > 0 else None
            else:
                active, left = status, int(rng.geometric(1.0 / mean[status]))
        u = rng.uniform(size=last - first + 1)
        out = {"sc": [], "vsc": []}
        for lap, x in zip(range(first, last + 1), u):
            if active is None:
                a, b = lap1 if lap1 is not None and lap == 1 else (p_sc, p_vsc)
                active = "sc" if x < a else "vsc" if x < a + b else None
                if active is not None:
                    left = (self._length(snapshot.race, active, rng) if hist
                            else int(rng.geometric(1.0 / mean[active])))
            if active is not None:
                out[active].append(lap)
                left -= 1
                if left == 0:
                    active = None
        return Scenario(sc_laps=out["sc"], vsc_laps=out["vsc"])


@EVENTS.register("hybrid")
class HybridEvents(RecordedEvents):
    """The scoring world's events (docs/eval.md "Pit loss and track status"): the SC or VSC in
    progress at the snapshot ends where it really ended (the driver's recorded laps of that kind
    from the next lap on, with the field's recorded lap times under lap_times "recorded"), and
    every later period is drawn by the rate model (params: config.EventModel, with the race's
    rates when RaceParams carries them) from the lap after it, on the events substream, so every
    plan and the reference see the same draws for a seed. A green snapshot draws exactly as
    "rate" does. The record is not used past the period in progress: a reference that waits for
    a safety car no policy could know about would score luck. Simulator binds the race frame."""

    def __init__(self, lap_times: str = "recorded", **params):
        super().__init__(lap_times=lap_times)
        self.rate = RateEvents(**params)

    def in_progress(self, snapshot: Snapshot) -> list[int]:
        """The recorded laps from laps_completed + 1 that continue the snapshot's SC or VSC."""
        kind = snapshot.track_status
        if kind not in ("sc", "vsc"):
            return []
        hit = self.by_driver.get(snapshot.driver)
        laps = set((hit if hit is not None else (self.race_laps.sc_laps, self.race_laps.vsc_laps))[kind == "vsc"])
        out, lap = [], snapshot.laps_completed + 1
        while lap in laps and lap <= snapshot.race.total_laps:
            out.append(lap)
            lap += 1
        return out

    def draw(self, snapshot: Snapshot, rng) -> Scenario:
        kind = snapshot.track_status
        if kind not in ("sc", "vsc"):
            return self.rate.draw(snapshot, rng)
        now = self.in_progress(snapshot)
        end = snapshot.laps_completed + len(now)
        later = Scenario()
        if end < snapshot.race.total_laps:
            after = snapshot.model_copy(update={"laps_completed": end, "track_status": "green"})
            later = self.rate.draw(after, rng)
        times = {x: self.medians[kind][x] for x in now if x in self.medians[kind]}
        return Scenario(sc_laps=(now if kind == "sc" else []) + later.sc_laps,
                        vsc_laps=(now if kind == "vsc" else []) + later.vsc_laps, lap_time_s=times)
