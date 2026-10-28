"""Safety car and virtual safety car scenarios for the remaining laps.

An event model draws a `Scenario` per sample. Drawing depends on the snapshot and
rng only, never on the plan, so every plan sees the same events for a given seed.
"""
from pydantic import BaseModel, Field

from ..contracts import Snapshot
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
