"""Simulator configuration. Components are chosen by registry key plus params."""
import json

from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Cfg):
    name: str
    params: dict = Field(default_factory=dict)


class EventEffects(_Cfg):
    """How neutralised laps change lap time and pit loss. Provisional defaults."""

    sc_lap_factor: float = 1.6        # SC lap time as a multiple of the reference lap
    vsc_lap_factor: float = 1.3       # VSC lap time as a multiple of the green lap
    sc_pit_loss_factor: float = 0.5   # share of the normal pit loss paid under SC
    vsc_pit_loss_factor: float = 0.6


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

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            if str(path).endswith((".yaml", ".yml")):
                import yaml
                return cls.model_validate(yaml.safe_load(f))
            return cls.model_validate(json.load(f))
