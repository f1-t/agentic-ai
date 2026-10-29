from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product
from math import isfinite, sqrt
from numbers import Real
from statistics import fmean, pstdev, stdev

from .domain import NoFeasiblePlan, Plan, Snapshot, Stint, validate_plan

Simulator = Callable[[Snapshot, Plan, int], float]


def _stop_limit(max_stops):
    if type(max_stops) is not int or not 0 <= max_stops <= 2:
        raise ValueError("max_stops must be 0, 1, or 2 future stops.")


def _seeds(seeds):
    values = tuple(seeds)
    if not values or any(type(s) is not int or s < 0 for s in values):
        raise ValueError("Seeds must be a nonempty sequence of nonnegative integers.")
    if len(set(values)) != len(values):
        raise ValueError("Seeds must be unique so each scenario has equal weight.")
    return values


def candidate_plans(snapshot: Snapshot, max_stops: int = 2) -> Iterator[Plan]:
    _stop_limit(max_stops)
    remaining = snapshot.remaining_laps
    minimum = snapshot.min_stint_laps
    first_minimum = max(0, minimum - snapshot.current_stint_laps)
    for stops in range(max_stops + 1):
        if stops == 0:
            lengths = [(remaining,)]
        elif stops == 1:
            lengths = (
                (n, remaining - n)
                for n in range(first_minimum, remaining - minimum + 1)
            )
        else:
            lengths = (
                (n, m, remaining - n - m)
                for n in range(first_minimum, remaining - 2 * minimum + 1)
                for m in range(minimum, remaining - n - minimum + 1)
            )
        for split in lengths:
            for replacements in product(
                sorted(snapshot.available_compounds), repeat=stops
            ):
                compounds = (snapshot.current_compound, *replacements)
                plan = Plan(tuple(Stint(c, n) for c, n in zip(compounds, split)))
                try:
                    validate_plan(snapshot, plan)
                except ValueError:
                    continue
                yield plan


@dataclass(frozen=True, slots=True)
class Evaluation:
    plan: Plan
    seeds: tuple[int, ...]
    seed_times_s: tuple[float, ...]

    @property
    def mean_time_s(self):
        return fmean(self.seed_times_s)

    @property
    def stddev_time_s(self):
        return pstdev(self.seed_times_s)

    @property
    def standard_error_s(self):
        return (
            stdev(self.seed_times_s) / sqrt(len(self.seeds))
            if len(self.seeds) > 1
            else 0.0
        )

    def to_dict(self, snapshot):
        pits = self.plan.pit_laps(snapshot)
        return {
            "stint_plan": self.plan.to_list(),
            "num_stops": self.plan.num_stops,
            "pit_laps": list(pits),
            "first_pit_lap": pits[0] if pits else None,
            "expected_remaining_time_s": self.mean_time_s,
            "stddev_time_s": self.stddev_time_s,
            "standard_error_s": self.standard_error_s,
            "seeds": list(self.seeds),
            "seed_times_s": list(self.seed_times_s),
        }


def _evaluate(snapshot, plan, simulator, seeds):
    times = []
    for seed in seeds:
        value = simulator(snapshot, plan, seed)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"The simulator returned an invalid remaining time for seed {seed}: {value!r}"
            )
        times.append(float(value))
    return Evaluation(plan, seeds, tuple(times))


def score_plan(
    snapshot: Snapshot, plan: Plan, simulator: Simulator, seeds=(0,)
) -> Evaluation:
    validate_plan(snapshot, plan)
    return _evaluate(snapshot, plan, simulator, _seeds(seeds))


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    evaluation: Evaluation
    candidates_evaluated: int


def optimize(
    snapshot: Snapshot, simulator: Simulator, seeds=(0,), max_stops: int = 2
) -> OptimizationResult:
    seeds = _seeds(seeds)
    best = None
    best_key = None
    count = 0
    for plan in candidate_plans(snapshot, max_stops):
        evaluation = _evaluate(snapshot, plan, simulator, seeds)
        key = (
            evaluation.mean_time_s,
            plan.num_stops,
            plan.pit_laps(snapshot),
            tuple(s.compound for s in plan.stints),
        )
        if best_key is None or key < best_key:
            best, best_key = evaluation, key
        count += 1
    if best is None:
        raise NoFeasiblePlan(
            "No legal remaining plan exists within the future-stop limit."
        )
    return OptimizationResult(best, count)
