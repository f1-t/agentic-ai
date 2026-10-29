from dataclasses import FrozenInstanceError, replace
from itertools import combinations_with_replacement, pairwise, product

import pytest

from src.baselines import (
    NoFeasiblePlan,
    Plan,
    PolicyUnavailable,
    Snapshot,
    Stint,
    actual_strategy,
    candidate_plans,
    fixed_one_stop,
    fixed_two_stop,
    greedy_crossover,
    optimize,
    score_plan,
    validate_plan,
)


def snapshot(**overrides):
    values = {
        "total_laps": 9,
        "completed_laps": 0,
        "current_compound": "SOFT",
        "tyre_age": 0,
        "current_stint_laps": 0,
        "used_compounds": ("SOFT",),
        "available_compounds": ("SOFT", "MEDIUM", "HARD"),
        "min_stint_laps": 2,
        "pit_loss_s": 3.0,
    }
    return Snapshot(**(values | overrides))


def independent_plans(state, max_stops):
    plans = set()
    for stops in range(max_stops + 1):
        for pits in combinations_with_replacement(
            range(state.completed_laps, state.total_laps), stops
        ):
            boundaries = (state.completed_laps, *pits, state.total_laps)
            lengths = tuple(b - a for a, b in pairwise(boundaries))
            for replacements in product(state.available_compounds, repeat=stops):
                compounds = (state.current_compound, *replacements)
                if (
                    stops
                    and lengths[0] + state.current_stint_laps < state.min_stint_laps
                ):
                    continue
                if any(length < state.min_stint_laps for length in lengths[1:]):
                    continue
                used = set(state.used_compounds) | set(compounds)
                if (
                    state.require_two_compounds
                    and state.remaining_laps
                    and len(used) < 2
                ):
                    continue
                plans.add(Plan(tuple(Stint(c, n) for c, n in zip(compounds, lengths))))
    return plans


def quadratic_time(state, plan, seed):
    bases = {"SOFT": 8.0, "MEDIUM": 9.0, "HARD": 10.0}
    degradation = {"SOFT": 0.7, "MEDIUM": 0.3, "HARD": 0.1}
    result = state.pit_loss_s * plan.num_stops
    for index, stint in enumerate(plan.stints):
        age = state.tyre_age if index == 0 else 0
        result += sum(
            bases[stint.compound] + degradation[stint.compound] * lap_age
            for lap_age in range(age + 1, age + stint.laps + 1)
        )
    return result + seed * 0.01


@pytest.mark.parametrize("max_stops", [0, 1, 2])
@pytest.mark.parametrize(
    "state",
    [
        snapshot(),
        snapshot(completed_laps=3, current_stint_laps=3, tyre_age=7),
        snapshot(
            completed_laps=7,
            current_stint_laps=1,
            tyre_age=1,
            used_compounds=("SOFT", "MEDIUM"),
        ),
        snapshot(require_two_compounds=False),
    ],
)
def test_enumeration_matches_independent_absolute_pit_search(state, max_stops):
    actual = list(candidate_plans(state, max_stops=max_stops))
    assert len(actual) == len(set(actual))
    assert set(actual) == independent_plans(state, max_stops)


def test_optimizer_matches_independent_minimum():
    state = snapshot(completed_laps=3, current_stint_laps=3, tyre_age=7)
    expected = min(
        quadratic_time(state, plan, 0) for plan in independent_plans(state, 2)
    )
    result = optimize(state, quadratic_time, seeds=(0, 1, 2))
    assert result.evaluation.mean_time_s == pytest.approx(expected + 0.01)
    assert result.candidates_evaluated == len(independent_plans(state, 2))


def test_stochastic_search_commits_to_one_plan_before_observing_seeds():
    state = snapshot(total_laps=6, available_compounds=("SOFT", "HARD"))
    risky = Plan((Stint("SOFT", 2), Stint("HARD", 4)))
    stable = Plan((Stint("SOFT", 3), Stint("HARD", 3)))
    calls = []

    def simulate(state, plan, seed):
        calls.append((plan, seed))
        if plan == risky:
            return 1.0 if seed == 10 else 199.0
        return 80.0 if plan == stable else 200.0

    result = optimize(state, simulate, seeds=(10, 20), max_stops=1)
    assert result.evaluation.plan == stable
    assert result.evaluation.seed_times_s == (80.0, 80.0)
    for plan in candidate_plans(state, max_stops=1):
        assert [(p, s) for p, s in calls if p == plan] == [(plan, 10), (plan, 20)]


def test_immediate_stop_is_considered_and_current_age_is_preserved():
    state = snapshot(completed_laps=3, current_stint_laps=3, tyre_age=30)
    result = optimize(state, quadratic_time)
    assert result.evaluation.plan.stints[0].laps == 0
    assert result.evaluation.plan.pit_laps(state)[0] == 3


def test_history_allows_finishing_without_another_stop():
    state = snapshot(
        completed_laps=8,
        current_stint_laps=1,
        tyre_age=1,
        used_compounds=("SOFT", "MEDIUM"),
    )
    result = optimize(state, quadratic_time)
    assert result.evaluation.plan == Plan((Stint("SOFT", 1),))


def test_no_legal_plan_is_explicit():
    state = snapshot(total_laps=3)
    with pytest.raises(NoFeasiblePlan):
        optimize(state, quadratic_time)


def test_finished_snapshot_returns_zero_stops():
    state = snapshot(completed_laps=9, current_stint_laps=9, tyre_age=9)
    result = optimize(state, quadratic_time)
    assert result.evaluation.plan == Plan((Stint("SOFT", 0),))
    assert result.evaluation.mean_time_s == 0


def test_equal_times_choose_fewer_stops_then_earlier_pit_laps():
    state = snapshot()
    result = optimize(state, lambda state, plan, seed: 100)
    assert result.evaluation.plan.num_stops == 1
    assert result.evaluation.plan.pit_laps(state) == (2,)
    assert result.evaluation.plan.stints[1].compound == "HARD"


@pytest.mark.parametrize("seeds", [(), (1, 1), (True,), (-1,), (1.5,)])
def test_invalid_seed_sets_fail(seeds):
    with pytest.raises(ValueError):
        optimize(snapshot(), quadratic_time, seeds=seeds)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "100"])
def test_invalid_simulator_outputs_fail_instead_of_being_silently_skipped(value):
    with pytest.raises(ValueError, match="simulator"):
        optimize(snapshot(), lambda state, plan, seed: value)


@pytest.mark.parametrize("max_stops", [-1, 3, True, 1.5])
def test_unsupported_stop_limits_fail(max_stops):
    with pytest.raises(ValueError):
        list(candidate_plans(snapshot(), max_stops=max_stops))


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_laps": -1},
        {"completed_laps": 10},
        {"completed_laps": True},
        {"tyre_age": -1},
        {"current_stint_laps": 1},
        {"current_compound": "WET"},
        {"available_compounds": ()},
        {"available_compounds": ("SOFT", "SOFT")},
        {"min_stint_laps": 0},
        {"pit_loss_s": float("nan")},
    ],
)
def test_invalid_snapshots_are_rejected(changes):
    with pytest.raises(ValueError):
        snapshot(**changes)


def test_domain_records_are_immutable_and_normalize_compounds():
    state = snapshot(current_compound="soft", available_compounds=["soft", "hard"])
    assert state.available_compounds == ("SOFT", "HARD")
    with pytest.raises(FrozenInstanceError):
        state.tyre_age = 4


def test_actual_strategy_is_cropped_at_absolute_lap_boundaries():
    history = (Stint("SOFT", 3), Stint("HARD", 6))
    state = snapshot(completed_laps=3, current_stint_laps=3, tyre_age=3)
    plan = actual_strategy(state, history)
    assert plan == Plan((Stint("SOFT", 0), Stint("HARD", 6)))
    after_pit = replace(
        state,
        current_compound="HARD",
        current_stint_laps=0,
        tyre_age=0,
        used_compounds=("SOFT", "HARD"),
    )
    assert actual_strategy(after_pit, history) == Plan((Stint("HARD", 6),))


def test_actual_strategy_reports_incomplete_or_incompatible_history():
    with pytest.raises(PolicyUnavailable):
        actual_strategy(snapshot(), (Stint("SOFT", 3), Stint("HARD", 5)))
    with pytest.raises(PolicyUnavailable):
        actual_strategy(
            snapshot(current_compound="HARD"), (Stint("SOFT", 3), Stint("HARD", 6))
        )


def test_fixed_policies_have_fixed_compounds_and_equal_remaining_stints():
    state = snapshot(total_laps=12)
    assert fixed_one_stop(state) == Plan((Stint("SOFT", 6), Stint("HARD", 6)))
    assert fixed_two_stop(state) == Plan(
        (Stint("SOFT", 4), Stint("MEDIUM", 4), Stint("SOFT", 4))
    )
    for policy in (fixed_one_stop, fixed_two_stop):
        validate_plan(state, policy(state))


def test_fixed_policies_do_not_silently_change_stop_count_when_race_is_too_short():
    state = snapshot(total_laps=3)
    for policy in (fixed_one_stop, fixed_two_stop):
        with pytest.raises(PolicyUnavailable):
            policy(state)


def test_greedy_policy_pits_at_degradation_crossover():
    state = snapshot(total_laps=10, available_compounds=("SOFT", "HARD"))

    def pace(compound, age):
        return 8 + age if compound == "SOFT" else 12

    plan = greedy_crossover(state, pace, max_stops=1)
    assert plan == Plan((Stint("SOFT", 4), Stint("HARD", 6)))


def test_greedy_policy_uses_existing_tyre_age_and_enforces_required_compound():
    state = snapshot(completed_laps=3, current_stint_laps=3, tyre_age=12)
    plan = greedy_crossover(
        state, lambda compound, age: 8 + age if compound == "SOFT" else 12, max_stops=1
    )
    assert plan.stints[0].laps == 0
    validate_plan(state, plan)
    late_switch = greedy_crossover(
        snapshot(), lambda compound, age: 8 if compound == "SOFT" else 20, max_stops=1
    )
    assert late_switch.pit_laps(snapshot()) == (7,)


def test_scoring_reports_unrounded_mean_and_uncertainty():
    state = snapshot()
    plan = fixed_one_stop(state)
    result = score_plan(state, plan, lambda state, plan, seed: 10 + seed, seeds=(0, 2))
    assert result.mean_time_s == 11
    assert result.stddev_time_s == 1
    assert result.standard_error_s == pytest.approx(1)
