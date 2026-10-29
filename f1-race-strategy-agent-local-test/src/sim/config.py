"""Simulator configuration. Components are chosen by registry key plus params."""
import json
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


log = logging.getLogger(__name__)

DEFAULT_COMPOUND_CLASSES = {
    **dict.fromkeys(["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"], "SLICK"),
    "INTERMEDIATE": "INTERMEDIATE", "WET": "WET",
}


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Cfg):
    name: str
    params: dict = Field(default_factory=dict)


class EventEffects(_Cfg):
    """How neutralised laps change lap time and pit loss. Provisional defaults."""

    sc_lap_factor: float = 1.36       # provisional, 2021 Spain observed mean (1.361, v2 filter); fitted per race in sprint 2
    vsc_lap_factor: float = 1.3       # VSC lap time as a multiple of the green lap
    sc_pit_loss_factor: float = 0.5   # share of the normal pit loss paid under SC
    vsc_pit_loss_factor: float = 0.6
    lap1_extra_s: float = 9.0         # provisional: standing start extra on lap 1, 2021 Spain median +8.7 to +10 s
    # tyre age gained per neutralised lap (1 per green lap), so a set run under SC/VSC is
    # younger afterwards. Provisional, fitted in sprint 2 from annotated races.
    sc_age_increment: float = 0.3
    vsc_age_increment: float = 0.5


class HazardEffects(_Cfg):
    """What a tyre hazard event costs (hazard.py gives the per lap probability). Provisional."""

    dnf_share: float = Field(default=0.5, ge=0, le=1)  # share of events that end the race
    # added once to a DNF sample, on top of its remaining laps run as if it had continued.
    # An arbitrary time-metric convention that only keeps a DNF slower than finishing; a
    # position metric uses BatchResult.dnf_lap (the lap of retirement) instead.
    dnf_penalty_s: float = 300.0
    big_loss_s: float = 20.0          # added on the event lap when the car carries on


class LongRun(_Cfg):
    """When a practice stint is a long run (data.practice_frame). Timed laps are green,
    non pit laps with a lap time, tyre age and compound, so the out and in laps never count."""

    min_timed_laps: int = 6           # qualifying runs are short
    # consistent: at least consistent_share of the timed laps within consistent_band_s of the
    # stint median. A push/cool programme has about half its timed laps off-pace; a long run
    # with a traffic lap or two has at most one in five; 0.8 encodes "at most one lap in five
    # off-pace". Frozen on a priori grounds, Spain only; selected across races by
    # leave-one-race-out later.
    consistent_share: float = 0.8
    consistent_band_s: float = 2.0
    max_pace_gap_s: float = 2.5       # stint median within this of the session's long-run reference


class SimConfig(_Cfg):
    degradation: ComponentSpec = ComponentSpec(name="linear_fuel")
    events: ComponentSpec = ComponentSpec(name="none")
    effects: EventEffects = Field(default_factory=EventEffects)
    hazard: ComponentSpec = ComponentSpec(name="none")
    hazard_effects: HazardEffects = Field(default_factory=HazardEffects)
    rules: list[str] = Field(default_factory=lambda: [
        "first_stint_matches", "lap_sum", "known_compounds", "min_stint", "two_compounds", "inventory",
    ])
    inventory_sources: list[str] = Field(default_factory=lambda: ["observed", "inferred", "assumed"])
    min_samples: int = 5      # clean laps needed before a compound is modelled
    n_samples: int = 1
    # Which laps the degradation model is fitted on, per snapshot (Simulator.fitted_for):
    #   full         every race lap (hindsight, the world)
    #   to_snapshot  race laps up to snapshot.laps_completed, shrunk toward the pre_race fit
    #   pre_race     practice long runs only
    fit_window: Literal["full", "to_snapshot", "pre_race"] = "full"
    exclude_snapshot_driver: bool = False   # leave the snapshot driver's laps out of the fit
    # practice cannot identify the race fuel term; linear_fuel uses this when no race laps fit
    # it, and shrinks toward it under to_snapshot
    pre_race_fuel_slope_s_per_lap: float = -0.045
    # practice-to-race base level, s/lap, added to every practice-derived compound base in the
    # pre_race prior (Spain: full-fit base 85.22 less practice 84.18, rounded). A config prior
    # for now (fit_report base_delta_source "prior"); the first factor SPEC H's calibrator
    # will predict per race. to_snapshot shrinks it with the base_s tau.
    pre_race_base_delta_s: float = 1.0
    # to_snapshot prior standard deviations around the pre_race fit (see degradation.py). Fuel
    # is tight: burn is near a physical constant per circuit, and the wear/fuel split is not
    # identified while every car is in its first stint.
    shrinkage: dict[str, float] = Field(default_factory=lambda: {
        "base_s": 0.5, "wear_s_per_lap": 0.03, "fuel_s_per_lap": 0.008,
    })
    long_run: LongRun = Field(default_factory=LongRun)   # practice long-run test, for data.practice_frame
    # tyre class per compound (SLICK, INTERMEDIATE, WET). Machinery for wet-weather rule
    # waivers and track-state factors; no wet behaviour is modelled.
    compound_classes: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_COMPOUND_CLASSES))
    track: ComponentSpec = ComponentSpec(name="dry")   # track-state transform, see track.py

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            if str(path).endswith((".yaml", ".yml")):
                import yaml
                return cls.model_validate(yaml.safe_load(f))
            return cls.model_validate(json.load(f))


def compound_class(config, compound) -> str:
    """The tyre class of a compound. An unknown compound counts as SLICK (with a warning);
    known_compounds still rejects compounds the degradation model has not seen."""
    classes = DEFAULT_COMPOUND_CLASSES if config is None else config.compound_classes
    if compound not in classes:
        log.warning("compound %r has no class, treating it as SLICK", compound)
        return "SLICK"
    return classes[compound]


def class_names(config) -> list[str]:
    """Class labels in a fixed order (SLICK first), so engine class codes depend on the
    config only and never on the plans scored."""
    classes = DEFAULT_COMPOUND_CLASSES if config is None else config.compound_classes
    return ["SLICK", *sorted(set(classes.values()) - {"SLICK"})]
