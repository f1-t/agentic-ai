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

    def __init__(self, **params):
        self.params = EventModel(**params)

    def rates(self, race):
        """(p_sc, d_sc, p_vsc, d_vsc) for this race."""
        cfg = self.params

        def pick(override, default):
            return default if override is None else override

        out = (pick(race.sc_rate_per_lap, cfg.p_sc), pick(race.sc_mean_laps, cfg.d_sc),
               pick(race.vsc_rate_per_lap, cfg.p_vsc), pick(race.vsc_mean_laps, cfg.d_vsc))
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
        active, left = None, 0
        if status != "green":
            active, left = status, int(rng.geometric(1.0 / mean[status]))
        u = rng.uniform(size=last - first + 1)
        out = {"sc": [], "vsc": []}
        for lap, x in zip(range(first, last + 1), u):
            if active is None:
                a, b = lap1 if lap1 is not None and lap == 1 else (p_sc, p_vsc)
                active = "sc" if x < a else "vsc" if x < a + b else None
                if active is not None:
                    left = int(rng.geometric(1.0 / mean[active]))
            if active is not None:
                out[active].append(lap)
                left -= 1
                if left == 0:
                    active = None
        return Scenario(sc_laps=out["sc"], vsc_laps=out["vsc"])
