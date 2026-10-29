from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Real

from .domain import Plan, PolicyUnavailable, Snapshot, Stint, validate_plan
from .search import _stop_limit

PaceModel = Callable[[str, int], float]


def _legal(snapshot, stints):
    plan = Plan(tuple(stints))
    try:
        validate_plan(snapshot, plan)
    except ValueError as error:
        raise PolicyUnavailable(str(error)) from error
    return plan


def actual_strategy(snapshot: Snapshot, actual_stints: Sequence[Stint]) -> Plan:
    actual_stints = tuple(actual_stints)
    if not actual_stints or any(s.laps <= 0 for s in actual_stints):
        raise PolicyUnavailable(
            "Actual stints must contain complete positive-length stints."
        )
    if sum(s.laps for s in actual_stints) != snapshot.total_laps:
        raise PolicyUnavailable("Actual strategy does not cover the entire race.")
    start = 0
    used = set()
    for index, stint in enumerate(actual_stints):
        end = start + stint.laps
        used.add(stint.compound)
        if (
            start <= snapshot.completed_laps <= end
            and stint.compound == snapshot.current_compound
            and snapshot.current_stint_laps == snapshot.completed_laps - start
            and used == set(snapshot.used_compounds)
        ):
            return _legal(
                snapshot,
                (
                    Stint(stint.compound, end - snapshot.completed_laps),
                    *actual_stints[index + 1 :],
                ),
            )
        start = end
    raise PolicyUnavailable(
        "Snapshot history does not match the driver's actual strategy."
    )


def _fixed(snapshot, stops, preference):
    remaining = snapshot.remaining_laps
    lower = max(0, snapshot.min_stint_laps - snapshot.current_stint_laps)
    upper = remaining - stops * snapshot.min_stint_laps
    if lower > upper or remaining == 0:
        raise PolicyUnavailable(
            f"There are too few laps for exactly {stops} future stops."
        )
    first = min(max(remaining // (stops + 1), lower), upper)
    rest = remaining - first
    lengths = [first]
    for count in range(stops, 0, -1):
        length = rest // count
        lengths.append(length)
        rest -= length
    available = snapshot.available_compounds
    alternatives = [
        c for c in preference if c in available and c != snapshot.current_compound
    ]
    alternatives += sorted(
        c for c in available if c != snapshot.current_compound and c not in alternatives
    )
    next_compound = alternatives[0] if alternatives else snapshot.current_compound
    compounds = (snapshot.current_compound, next_compound)
    if stops == 2:
        compounds += (snapshot.current_compound,)
    return _legal(snapshot, (Stint(c, n) for c, n in zip(compounds, lengths)))


def fixed_one_stop(snapshot: Snapshot) -> Plan:
    return _fixed(snapshot, 1, ("HARD", "MEDIUM", "SOFT"))


def fixed_two_stop(snapshot: Snapshot) -> Plan:
    return _fixed(snapshot, 2, ("MEDIUM", "HARD", "SOFT"))


def greedy_crossover(
    snapshot: Snapshot,
    pace: PaceModel,
    max_stops: int = 2,
    crossover_margin_s: float = 0.0,
) -> Plan:
    _stop_limit(max_stops)
    if not isfinite(crossover_margin_s) or crossover_margin_s < 0:
        raise ValueError("The crossover margin must be finite and nonnegative.")

    def predicted(compound, age):
        value = pace(compound, age)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("The pace model must return a finite positive lap time.")
        return value

    stints = []
    compound = snapshot.current_compound
    age = snapshot.tyre_age
    stint_laps = snapshot.current_stint_laps
    segment_laps = 0
    used = set(snapshot.used_compounds)
    for offset in range(snapshot.remaining_laps):
        remaining = snapshot.remaining_laps - offset
        can_pit = (
            len(stints) < max_stops
            and stint_laps >= snapshot.min_stint_laps
            and remaining >= snapshot.min_stint_laps
        )
        needs_compound = snapshot.require_two_compounds and len(used) < 2
        if can_pit:
            alternatives = [c for c in snapshot.available_compounds if c != compound]
            if needs_compound and len(stints) + 1 == max_stops:
                alternatives = [c for c in alternatives if c not in used]
            if alternatives:
                replacement = min(alternatives, key=lambda c: (predicted(c, 1), c))
                crosses = (
                    predicted(compound, age + 1)
                    > predicted(replacement, 1) + crossover_margin_s
                )
                mandatory = needs_compound and remaining == snapshot.min_stint_laps
                if crosses or mandatory:
                    stints.append(Stint(compound, segment_laps))
                    compound, age, stint_laps, segment_laps = replacement, 0, 0, 0
                    used.add(compound)
        age += 1
        stint_laps += 1
        segment_laps += 1
    return _legal(snapshot, (*stints, Stint(compound, segment_laps)))
