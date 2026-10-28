"""Shared data contracts between the data layer, simulator, optimizer and agent.

DRAFT: field names and semantics are proposals to be agreed in sprint 1, week one.
See docs/sim_interfaces.md for the open questions.

Conventions
- `laps_completed` is the number of laps finished at the decision point. The
  decision is made at the start of lap `laps_completed + 1`.
- `tyre_age` is the number of laps already run on the current set. A fresh set
  has age 0 and runs its first lap at age 1.
- A `Plan` covers exactly the remaining laps. Its first stint continues the
  current set, later stints each start with a pit stop at the end of the
  previous one.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _norm_compound(value: str) -> str:
    return str(value).strip().upper()


Compound = Annotated[str, AfterValidator(_norm_compound)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Race level --------------------------------------------------------------
class RaceParams(_Contract):
    """Per-race constants. Owned by the race registry."""

    race_id: str
    total_laps: int = Field(gt=0)
    pit_loss_s: float = Field(ge=0)
    min_stint_laps: int = Field(default=10, ge=1)
    compounds: list[Compound] = Field(default_factory=list)  # available to run


# --- Snapshot ----------------------------------------------------------------
class StintRecord(_Contract):
    """A completed stint."""

    compound: Compound
    laps: int = Field(ge=1)
    start_lap: int = Field(ge=1)
    start_age: int = Field(default=0, ge=0)


class Weather(_Contract):
    """Observed weather at the decision lap plus rain history. Belief-side only."""

    rainfall_now: bool = False
    track_temp_c: float | None = None
    air_temp_c: float | None = None
    wind_kph: float | None = None
    rain_laps: list[int] = Field(default_factory=list)   # laps with rain so far
    wetness_estimate: float | None = Field(default=None, ge=0, le=1)


class FieldState(_Contract):
    """What the timing screens show: who is on which tyre class, and how fast."""

    class_counts: dict[str, int] = Field(default_factory=dict)    # SLICK / INTERMEDIATE / WET
    class_pace_s: dict[str, float] = Field(default_factory=dict)  # median last-lap time per class


class Snapshot(_Contract):
    """Race state for one driver at one lap. Built by the snapshot builder.

    Everything after `position` is optional and defaults to "unknown or unconstrained",
    so a dry snapshot without weather or gap data is unchanged.
    """

    race: RaceParams
    driver: str
    laps_completed: int = Field(default=0, ge=0)
    compound: Compound                       # current set
    tyre_age: int = Field(default=0, ge=0)   # laps on the current set
    stints: list[StintRecord] = Field(default_factory=list)  # completed only
    sc_laps: list[int] = Field(default_factory=list)         # history
    vsc_laps: list[int] = Field(default_factory=list)        # history
    track_status: Literal["green", "sc", "vsc"] = "green"    # at the decision lap
    elapsed_s: float | None = None           # race time so far
    position: int | None = None
    gap_ahead_s: float | None = None
    gap_behind_s: float | None = None
    car_ahead: str | None = None
    sets_available: dict[Compound, int] | None = None  # fresh sets left; None = unconstrained
    weather: Weather = Field(default_factory=Weather)
    field: FieldState = Field(default_factory=FieldState)
    red_flag_laps: list[int] = Field(default_factory=list)   # history
    standing_start: bool = True

    @model_validator(mode="after")
    def _check(self):
        if self.laps_completed >= self.race.total_laps:
            raise ValueError("snapshot must leave at least one lap to run")
        if sum(s.laps for s in self.stints) > self.laps_completed:
            raise ValueError("completed stints exceed laps_completed")
        history = {
            "neutralised": self.sc_laps + self.vsc_laps,
            "red flag": self.red_flag_laps,
            "rain": self.weather.rain_laps,
        }
        for name, laps in history.items():
            late = [lap for lap in laps if lap > self.laps_completed]
            if late:
                raise ValueError(f"{name} laps {late} are after the snapshot")
        return self

    @property
    def remaining_laps(self) -> int:
        return self.race.total_laps - self.laps_completed

    @property
    def stops_made(self) -> int:
        return len(self.stints)

    @property
    def compounds_used(self) -> set[str]:
        return {s.compound for s in self.stints} | {self.compound}

    @classmethod
    def at_start(cls, race: RaceParams, driver: str, compound: str, tyre_age: int = 0):
        return cls(race=race, driver=driver, compound=compound, tyre_age=tyre_age)


# --- Plan --------------------------------------------------------------------
class PlanStint(_Contract):
    compound: Compound
    laps: int = Field(ge=1)
    start_age: int = Field(default=0, ge=0)  # used sets; ignored for the first stint

    @model_validator(mode="before")
    @classmethod
    def _from_sequence(cls, value):
        # accepts [compound, laps] or [compound, laps, start_age]
        if isinstance(value, (list, tuple)):
            if not 2 <= len(value) <= 3:
                raise ValueError(f"stint must be [compound, laps], got {value!r}")
            return dict(zip(("compound", "laps", "start_age"), value))
        return value


class Plan(_Contract):
    """Stints covering the remaining laps. First stint continues the current set."""

    stints: list[PlanStint] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value):
        return {"stints": value} if isinstance(value, (list, tuple)) else value

    @classmethod
    def of(cls, value) -> "Plan":
        return value if isinstance(value, cls) else cls.model_validate(value)

    @property
    def total_laps(self) -> int:
        return sum(s.laps for s in self.stints)

    @property
    def num_stops(self) -> int:
        return len(self.stints) - 1

    def pit_laps(self, snapshot: Snapshot) -> list[int]:
        """Absolute laps on which each planned stop is taken (in-lap)."""
        out, lap = [], snapshot.laps_completed
        for s in self.stints[:-1]:
            lap += s.laps
            out.append(lap)
        return out

    def as_pairs(self) -> list[list]:
        return [[s.compound, s.laps] for s in self.stints]


# --- Simulator output --------------------------------------------------------
class TimeSummary(_Contract):
    """Summary of the remaining-race time over samples. std is 0 when deterministic."""

    mean_s: float
    std_s: float = 0.0
    p05_s: float
    p50_s: float
    p95_s: float
    n_samples: int = 1
    cvar_90_s: float | None = None  # mean of the worst 10 percent of samples


class SimResult(_Contract):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    plan: Plan | None = None
    num_stops: int | None = None
    pit_laps: list[int] = Field(default_factory=list)
    remaining: TimeSummary | None = None
    elapsed_s: float | None = None
    stint_seconds: list[float] = Field(default_factory=list)  # mean per stint, pit loss excluded
    pit_loss_total_s: float | None = None                     # mean over samples
    lap_times_s: list[float] | None = None                    # sample 0, pit loss excluded
    p_dnf: float = Field(default=0.0, ge=0, le=1)             # share of samples ending the race
    p_big_loss: float = Field(default=0.0, ge=0, le=1)        # share with at least one hazard event
    support_warnings: list[str] = Field(default_factory=list)  # e.g. stint longer than observed tyre age

    @property
    def race_time_s(self) -> float | None:
        if not self.valid:
            return None
        return self.remaining.mean_s + (self.elapsed_s or 0.0)
