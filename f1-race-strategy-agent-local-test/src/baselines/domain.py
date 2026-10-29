from dataclasses import dataclass
from math import isfinite


class NoFeasiblePlan(ValueError):
    pass


class PolicyUnavailable(ValueError):
    pass


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _compound(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A compound must be a nonempty string.")
    return value.strip().upper()


@dataclass(frozen=True, slots=True)
class Snapshot:
    total_laps: int
    completed_laps: int
    current_compound: str
    tyre_age: int
    current_stint_laps: int
    used_compounds: tuple[str, ...]
    available_compounds: tuple[str, ...]
    min_stint_laps: int = 10
    pit_loss_s: float = 22.0
    require_two_compounds: bool = True

    def __post_init__(self):
        for name in ("total_laps", "completed_laps", "tyre_age", "current_stint_laps"):
            _integer(getattr(self, name), name, minimum=int(name == "total_laps"))
        _integer(self.min_stint_laps, "min_stint_laps", minimum=1)
        if not 0 <= self.current_stint_laps <= self.completed_laps <= self.total_laps:
            raise ValueError(
                "Completed laps and current stint length are inconsistent."
            )
        if self.tyre_age < self.current_stint_laps:
            raise ValueError("Tyre age cannot be shorter than the current stint.")
        if type(self.require_two_compounds) is not bool:
            raise ValueError("require_two_compounds must be a boolean.")
        if isinstance(self.pit_loss_s, bool) or not isinstance(
            self.pit_loss_s, (int, float)
        ):
            raise TypeError("Pit loss must be a finite nonnegative number.")
        if not isfinite(self.pit_loss_s) or self.pit_loss_s < 0:
            raise ValueError("Pit loss must be a finite nonnegative number.")
        current = _compound(self.current_compound)
        compounds = tuple(_compound(c) for c in self.available_compounds)
        if (
            not compounds
            or len(set(compounds)) != len(compounds)
            or current not in compounds
        ):
            raise ValueError(
                "Available compounds must be unique and include the current tyre."
            )
        used = tuple(sorted({_compound(c) for c in self.used_compounds} | {current}))
        object.__setattr__(self, "current_compound", current)
        object.__setattr__(self, "available_compounds", compounds)
        object.__setattr__(self, "used_compounds", used)

    @property
    def remaining_laps(self):
        return self.total_laps - self.completed_laps


@dataclass(frozen=True, slots=True)
class Stint:
    compound: str
    laps: int

    def __post_init__(self):
        _integer(self.laps, "stint laps")
        object.__setattr__(self, "compound", _compound(self.compound))


@dataclass(frozen=True, slots=True)
class Plan:
    stints: tuple[Stint, ...]

    def __post_init__(self):
        stints = tuple(self.stints)
        if not stints or any(not isinstance(s, Stint) for s in stints):
            raise ValueError("A plan must contain at least one Stint.")
        object.__setattr__(self, "stints", stints)

    @property
    def num_stops(self):
        return len(self.stints) - 1

    def pit_laps(self, snapshot):
        completed = snapshot.completed_laps
        result = []
        for stint in self.stints[:-1]:
            completed += stint.laps
            result.append(completed)
        return tuple(result)

    def to_list(self):
        return [[stint.compound, stint.laps] for stint in self.stints]


def validate_plan(snapshot, plan):
    first, *future = plan.stints
    if first.compound != snapshot.current_compound:
        raise ValueError("The first stint must continue the current tyre.")
    if sum(s.laps for s in plan.stints) != snapshot.remaining_laps:
        raise ValueError("Stint lengths must sum to the remaining race laps.")
    if any(s.compound not in snapshot.available_compounds for s in plan.stints):
        raise ValueError("The plan uses an unavailable compound.")
    if not snapshot.remaining_laps:
        if future:
            raise ValueError("A finished race cannot have future stops.")
        return
    if future and snapshot.current_stint_laps + first.laps < snapshot.min_stint_laps:
        raise ValueError("The current stint must reach the minimum before a pit stop.")
    if any(s.laps < snapshot.min_stint_laps for s in future):
        raise ValueError("New stints must reach the minimum stint length.")
    used = set(snapshot.used_compounds) | {s.compound for s in plan.stints if s.laps}
    if snapshot.require_two_compounds and len(used) < 2:
        raise ValueError("The complete dry race must use at least two compounds.")
