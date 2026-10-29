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
- `sets_available` lists the sets in the garage, not the fitted one. `[]`
  means nothing is in the garage, so every pit is rejected; `None` means
  unconstrained. A plan stint may name a set with `set_id`; its
  `start_age` is then the set's `laps_run`.
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
    # circuit SC/VSC rates for the "rate" event model: chance per green lap that a period
    # starts, and its mean length in laps. None takes the SimConfig events params.
    sc_rate_per_lap: float | None = Field(default=None, ge=0, le=1)
    sc_mean_laps: float | None = Field(default=None, ge=1)
    vsc_rate_per_lap: float | None = Field(default=None, ge=0, le=1)
    vsc_mean_laps: float | None = Field(default=None, ge=1)


# --- Snapshot ----------------------------------------------------------------
class StintRecord(_Contract):
    """A completed stint."""

    compound: Compound
    laps: int = Field(ge=1)
    start_lap: int = Field(ge=1)
    start_age: int = Field(default=0, ge=0)


class TyreSet(_Contract):
    """One set in the driver's allocation that is not currently fitted."""

    compound: Compound
    laps_run: int = Field(default=0, ge=0)
    source: Literal["observed", "inferred", "assumed"] = "observed"
    set_id: str | None = None
    is_new: bool | None = None   # derived from laps_run when omitted

    @model_validator(mode="after")
    def _derive_is_new(self) -> "TyreSet":
        if self.is_new is None:
            self.is_new = self.laps_run == 0
        elif self.is_new != (self.laps_run == 0):
            raise ValueError("is_new conflicts with laps_run")
        return self


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
    sets_available: list[TyreSet] | None = None   # garage sets; None = unconstrained
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
        ids = [t.set_id for t in self.sets_available or [] if t.set_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("set_id values in sets_available must be unique")
        return self

    def find_set(self, set_id: str) -> "TyreSet | None":
        return next((t for t in self.sets_available or [] if t.set_id == set_id), None)

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
    set_id: str | None = None                 # a set in Snapshot.sets_available

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
    """One plan's simulated outcome. p_dnf and p_big_loss are exclusive (a sample has at
    most one hazard event), so p_dnf + p_big_loss is the probability of any hazard event."""

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
    p_big_loss: float = Field(default=0.0, ge=0, le=1)        # share with a hazard event the car survived
    support_warnings: list[str] = Field(default_factory=list)  # e.g. stint longer than observed tyre age

    @property
    def race_time_s(self) -> float | None:
        if not self.valid:
            return None
        return self.remaining.mean_s + (self.elapsed_s or 0.0)


class CompareResult(_Contract):
    """Candidates scored against each other under shared draws.

    Lists follow the candidate order. Matrices are indexed [i][j]; any entry that
    involves an invalid candidate is None. Times are remaining-race seconds.
    """

    candidates: list[Plan | None]       # full plans, actions expanded
    valid: list[bool]
    errors: list[list[str]]
    summaries: list[TimeSummary | None]
    p_best: list[float | None]          # share of samples where i is strictly fastest, ties split
    p_beats: list[list[float | None]]   # share with t_i < t_j, ties 0.5; diagonal None
    gap_mean_s: list[list[float | None]]  # t_i - t_j over samples
    gap_std_s: list[list[float | None]]
    best: int | None                    # lowest mean among valid
    dominated: list[bool]               # some valid j is no worse on mean and cvar_90, better on one
    n_samples: int
    seed: int
    support_warnings: list[list[str]] = Field(default_factory=list)  # per candidate, as SimResult


# --- Model report --------------------------------------------------------------
class CoefficientReport(_Contract):
    """One fitted coefficient: its value, posterior sd, and how much the data moved it.

    source is "fitted" (shrinkage_weight >= 0.9), "prior" (weight <= 0.1) or "shrunk" (in
    between). A coefficient held at its prior (weight 0, a fixed fuel slope included)
    reports the prior tau as sd. sd is NaN where it cannot be estimated.
    """

    value: float
    sd: float
    source: Literal["fitted", "shrunk", "prior"]
    shrinkage_weight: float = Field(ge=0, le=1)


class CompoundReport(_Contract):
    """What the degradation fit knows about one compound. Unmodelled compounds cannot be planned."""

    modelled: bool
    reason: str | None = None              # why not modelled, None when modelled
    n_rows: int = Field(default=0, ge=0)   # rows behind the line, practice support included
    n_race_rows: int = Field(default=0, ge=0)
    max_age_observed: float | None = None  # oldest tyre behind the line
    base_s: CoefficientReport | None = None
    wear_s_per_lap: CoefficientReport | None = None


class ModelReport(_Contract):
    """The simulator's view of its own uncertainty for one snapshot (Simulator.model_report).

    flags:
      wear_fuel_poorly_identified(fuel_weight=w)  fuel weight below 0.5, or at most one compound
                                   with min_samples race rows; w to two decimals
      prior_dominated:<compound>   wear weight below 0.5
      extrapolation:<compound>     a requested tyre age beyond max_age_observed
      beyond_knot_unsupported:<compound>  a requested tyre age past the hinge knot of a
                                   compound whose hinge was dropped (linear_fuel_hinge only)
      no_hazard_support:<compound> the hazard model has no observed support for this modelled
                                   compound, so its tyre risk is not known (only with a hazard
                                   model that has per compound support)
    calibration_error_s is the offset-adjusted mean absolute error of the remaining race
    time for this fit window at the snapshot's horizon bucket, from the mid-race
    calibration summary; NaN when there is no table or no row for it.
    """

    fit_window: str
    compounds: dict[str, CompoundReport]
    fuel_s_per_lap: CoefficientReport | None = None   # None for models without a fuel term
    # linear_fuel_hinge: {compound: {value, sd, source, weight, knot, n_past_knot}} from the
    # fit_report, source "fitted", "shrunk", "prior" or "dropped"; None for models without a hinge
    hinge: dict | None = None
    condition_number: float
    flags: list[str] = Field(default_factory=list)
    calibration_error_s: float = float("nan")

    @property
    def modelled(self) -> list[str]:
        return sorted(c for c, r in self.compounds.items() if r.modelled)


# --- Calibration ---------------------------------------------------------------
# Factor names a calibrator may output. Per compound factors are written name[COMPOUND],
# e.g. base_delta_s[SOFT]. A factor the output leaves out takes the config or race
# registry value.
COMPOUND_FACTORS = ("base_delta_s", "wear_s_per_lap", "hinge_s_per_lap")
RACE_FACTORS = ("fuel_s_per_lap", "pit_loss_s", "sc_lap_factor", "vsc_lap_factor", "sc_pit_loss_factor",
                "lap1_extra_s", "sc_rate_per_lap")


def factor_name(name: str, compound: str | None = None) -> str:
    return name if compound is None else f"{name}[{_norm_compound(compound)}]"


def split_factor(key: str) -> tuple[str, str | None]:
    """'base_delta_s[SOFT]' -> ('base_delta_s', 'SOFT'); 'pit_loss_s' -> ('pit_loss_s', None)."""
    if key.endswith("]") and "[" in key:
        name, _, comp = key[:-1].partition("[")
        return name, comp
    return key, None


class CorrectionTable(_Contract):
    """Additive green lap time correction by (compound, tyre age 0..max_age, race lap 1..total).

    mean and sd are nested lists of shape (len(compounds), max_age + 1, total_laps). The
    simulator adds mean + sd * z, with one standard normal z per sample shared by every plan.
    """

    compounds: list[Compound]
    mean: list[list[list[float]]]
    sd: list[list[list[float]]]

    @model_validator(mode="after")
    def _check(self):
        def shape(a):
            return (len(a), len(a[0]) if a else 0, len(a[0][0]) if a and a[0] else 0)
        s = shape(self.mean)
        if s[0] != len(self.compounds) or 0 in s:
            raise ValueError(f"mean shape {s} does not match {len(self.compounds)} compounds")
        if shape(self.sd) != s or any(len(r) != s[1] or any(len(c) != s[2] for c in r)
                                      for r in self.mean + self.sd):
            raise ValueError("mean and sd must be full arrays of one shape")
        if any(v < 0 for r in self.sd for c in r for v in c):
            raise ValueError("sd must be non-negative")
        return self


class CalibrationOutput(_Contract):
    """What a calibrator says at one calibration point (see src/sim/calibrators.py).

    factors maps a factor name (COMPOUND_FACTORS per compound, RACE_FACTORS) to (mean, sd).
    An sd of 0 holds the factor at its mean. valid_until_lap is the last lap this output
    is reused for under the calibration cadence.
    """

    factors: dict[str, tuple[float, float]] = Field(default_factory=dict)
    correction: CorrectionTable | None = None
    valid_until_lap: int = Field(ge=0)
    source: str
    diagnostics: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        for key, (_, sd) in self.factors.items():
            name, comp = split_factor(key)
            known = name in COMPOUND_FACTORS if comp is not None else name in RACE_FACTORS
            if not known:
                raise ValueError(f"unknown factor {key!r}")
            if not sd >= 0:
                raise ValueError(f"factor {key!r}: sd must be >= 0, got {sd}")
        return self
