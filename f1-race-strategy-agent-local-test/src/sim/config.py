"""Simulator configuration. Components are chosen by registry key plus params."""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class LongRun(_Cfg):
    """When a practice stint is a long run (data.practice_frame). Timed laps are green,
    non pit laps with a lap time, tyre age and compound, so the out and in laps never count."""

    min_timed_laps: int = 6           # qualifying runs are short
    max_median_step_s: float = 1.0    # median |lap-to-lap step| of the timed laps; rejects push/cool alternation
    max_pace_gap_s: float = 2.5       # stint median within this of the session's long-run reference


class SimConfig(_Cfg):
    degradation: ComponentSpec = ComponentSpec(name="linear_fuel")
    events: ComponentSpec = ComponentSpec(name="none")
    effects: EventEffects = Field(default_factory=EventEffects)
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
    # to_snapshot prior standard deviations around the pre_race fit (see degradation.py)
    shrinkage: dict[str, float] = Field(default_factory=lambda: {
        "base_s": 0.5, "wear_s_per_lap": 0.03, "fuel_s_per_lap": 0.015,
    })
    long_run: LongRun = Field(default_factory=LongRun)   # practice long-run test, for data.practice_frame

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            if str(path).endswith((".yaml", ".yml")):
                import yaml
                return cls.model_validate(yaml.safe_load(f))
            return cls.model_validate(json.load(f))
