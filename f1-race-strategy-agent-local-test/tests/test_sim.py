"""Offline simulator tests. Synthetic data only, no model and no network."""
import time

import numpy as np
import pandas as pd
import pytest

from src.contracts import Plan, RaceParams, Snapshot
from src.sim import SimConfig, Simulator
from src.sim.config import EventEffects
from src.sim.degradation import DEGRADATION
from src.sim.events import Scenario
from src.sim.registry import Registry

TRUE = {"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.03)}
FUEL = -0.04
RACE = RaceParams(race_id="syn", total_laps=50, pit_loss_s=20.0, min_stint_laps=8,
                  compounds=["SOFT", "MEDIUM"])


def synthetic_frame(fuel=0.0):
    rows = []
    for comp, (base, slope) in TRUE.items():
        for start in range(0, 30, 3):
            for age in range(1, 21):
                lap = start + age
                rows.append({"compound": comp, "tyre_age": age, "lap": lap, "clean": True,
                             "lap_time_s": base + slope * age + fuel * lap})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sim():
    return Simulator(synthetic_frame())


def start(compound="SOFT", age=0):
    return Snapshot.at_start(RACE, "X", compound, tyre_age=age)


def hand_time(plan, age0=0):
    total, first = 0.0, True
    for comp, n in plan:
        base, slope = TRUE[comp]
        a0 = age0 if first else 0
        total += sum(base + slope * (a0 + k) for k in range(1, n + 1))
        first = False
    return total + RACE.pit_loss_s * (len(plan) - 1)


# --- fitting -----------------------------------------------------------------
def test_linear_fit_recovers_coefficients(sim):
    for comp, (base, slope) in TRUE.items():
        got = sim.degradation.coefs[comp]
        assert got == pytest.approx((base, slope), abs=1e-6)


def test_compound_with_too_few_laps_is_not_modelled():
    df = synthetic_frame()
    df = pd.concat([df, pd.DataFrame([{"compound": "HARD", "tyre_age": 1, "lap": 1,
                                       "clean": True, "lap_time_s": 85.0}] * 3)])
    s = Simulator(df)
    assert s.modelled == {"SOFT", "MEDIUM"}
    res = s.simulate(start(), [["SOFT", 20], ["HARD", 30]])
    assert not res.valid and "Unknown compound 'HARD'" in res.errors[0]


def test_fuel_model_recovers_fuel_slope():
    s = Simulator(synthetic_frame(fuel=FUEL), SimConfig(degradation={"name": "linear_fuel"}))
    assert s.degradation.fuel_slope == pytest.approx(FUEL, abs=1e-6)
    for comp, (base, slope) in TRUE.items():
        assert s.degradation.coefs[comp] == pytest.approx((base, slope), abs=1e-6)


# --- simulation --------------------------------------------------------------
@pytest.mark.parametrize("plan", [
    [("SOFT", 20), ("MEDIUM", 30)],
    [("SOFT", 15), ("MEDIUM", 20), ("SOFT", 15)],
])
def test_matches_hand_computation(sim, plan):
    res = sim.simulate(start(), [list(p) for p in plan])
    assert res.valid
    assert res.race_time_s == pytest.approx(hand_time(plan))
    assert res.num_stops == len(plan) - 1
    assert res.remaining.std_s == 0.0 and res.remaining.n_samples == 1


def test_mid_race_snapshot_continues_tyre_age(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20,
                    elapsed_s=1650.0, stints=[])
    plan = [("SOFT", 5), ("MEDIUM", 25)]
    res = sim.simulate(snap, [list(p) for p in plan])
    assert res.valid and res.pit_laps == [25]
    base, slope = TRUE["SOFT"]
    soft = sum(base + slope * (20 + k) for k in range(1, 6))
    assert res.stint_seconds[0] == pytest.approx(soft)
    assert res.race_time_s == pytest.approx(1650.0 + soft + sum(
        TRUE["MEDIUM"][0] + TRUE["MEDIUM"][1] * k for k in range(1, 26)) + 20.0)


def test_used_set_start_age_is_applied(sim):
    res = sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30, 6]])
    assert res.race_time_s == pytest.approx(hand_time([("SOFT", 20), ("MEDIUM", 30)]) + sum(
        TRUE["MEDIUM"][1] * 6 for _ in range(30)))


def test_return_laps_sums_to_remaining_less_pit_loss(sim):
    res = sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30]], return_laps=True)
    assert len(res.lap_times_s) == 50
    assert sum(res.lap_times_s) + res.pit_loss_total_s == pytest.approx(res.remaining.mean_s)


# --- rules -------------------------------------------------------------------
@pytest.mark.parametrize("plan, fragment", [
    ([["SOFT", 20], ["MEDIUM", 29]], "sum to 49"),
    ([["SOFT", 50]], "two different compounds"),
    ([["SOFT", 20], ["MEDIUM", 5], ["SOFT", 25]], "below the 8-lap minimum"),
    ([["MEDIUM", 20], ["SOFT", 30]], "continues the current set"),
    ([["SOFT", "x"], ["MEDIUM", 30]], "valid integer"),
])
def test_invalid_plans_report_errors(sim, plan, fragment):
    res = sim.simulate(start(), plan)
    assert not res.valid and res.race_time_s is None
    assert any(fragment in e for e in res.errors)


def test_single_compound_plan_is_valid_after_a_stop_on_another(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="MEDIUM", tyre_age=3,
                    stints=[{"compound": "SOFT", "laps": 20, "start_lap": 1}])
    assert sim.simulate(snap, [["MEDIUM", 30]]).valid


def test_rules_are_configurable():
    s = Simulator(synthetic_frame(), SimConfig(rules=["lap_sum"]))
    assert s.simulate(start(), [["SOFT", 50]]).valid


# --- neutralised laps --------------------------------------------------------
def test_safety_car_laps_and_pit_loss(sim):
    fx = EventEffects()
    plan = [["SOFT", 20], ["MEDIUM", 30]]
    base = sim.simulate(start(), plan, return_laps=True)
    sc = sim.simulate(start(), plan, scenario=Scenario(sc_laps=[20, 21]), return_laps=True)
    delta = fx.sc_lap_factor * sim.ref_lap_s
    expected = base.race_time_s - sum(base.lap_times_s[19:21]) + 2 * delta \
        - (1 - fx.sc_pit_loss_factor) * RACE.pit_loss_s
    assert sc.race_time_s == pytest.approx(expected)


def test_vsc_scales_green_lap(sim):
    plan = [["SOFT", 20], ["MEDIUM", 30]]
    base = sim.simulate(start(), plan, return_laps=True)
    vsc = sim.simulate(start(), plan, scenario=Scenario(vsc_laps=[5]), return_laps=True)
    assert vsc.lap_times_s[4] == pytest.approx(EventEffects().vsc_lap_factor * base.lap_times_s[4])


def test_fixed_event_model_from_config():
    cfg = SimConfig(events={"name": "fixed", "params": {"sc_laps": [10]}})
    s = Simulator(synthetic_frame(), cfg)
    plain = Simulator(synthetic_frame())
    plan = [["SOFT", 20], ["MEDIUM", 30]]
    assert s.simulate(start(), plan).race_time_s != plain.simulate(start(), plan).race_time_s


# --- sampling ----------------------------------------------------------------
@DEGRADATION.register("test_noisy")
class NoisyModel:
    """Stand-in for an uncertainty aware model: a random offset per draw."""

    def __init__(self, inner, offset=0.0):
        self.inner, self.offset, self.compounds = inner, offset, inner.compounds

    @classmethod
    def fit(cls, data, **kw):
        return cls(DEGRADATION.get("linear").fit(data, **kw))

    def draw(self, rng):
        return NoisyModel(self.inner, rng.normal(0, 0.2))

    def lap_times(self, compound, ages, laps):
        return self.inner.lap_times(compound, ages, laps) + self.offset


def test_samples_are_reproducible():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_noisy"}, n_samples=50))
    plan = [["SOFT", 20], ["MEDIUM", 30]]
    a, b, c = (s.simulate(start(), plan, seed=k).remaining for k in (3, 3, 4))
    assert a == b and a != c
    assert a.std_s > 0 and a.p05_s < a.p50_s < a.p95_s


def test_cvar_is_mean_of_worst_tail():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_noisy"}, n_samples=50))
    r = s.simulate(start(), [["SOFT", 20], ["MEDIUM", 30]], seed=7).remaining
    assert r.cvar_90_s >= r.p95_s >= r.mean_s
    det = Simulator(synthetic_frame()).simulate(start(), [["SOFT", 20], ["MEDIUM", 30]]).remaining
    assert det.cvar_90_s == det.mean_s


def test_plans_share_random_draws_for_a_seed():
    # the offset adds the same amount to every plan of this length, so with shared draws
    # the spread is identical and the mean gap equals the deterministic gap
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_noisy"}, n_samples=200))
    det = Simulator(synthetic_frame())
    p1, p2 = [["SOFT", 15], ["MEDIUM", 35]], [["SOFT", 25], ["MEDIUM", 25]]
    r1, r2 = (s.simulate(start(), p, seed=1).remaining for p in (p1, p2))
    assert r1.std_s == pytest.approx(r2.std_s)
    gap = det.simulate(start(), p1).race_time_s - det.simulate(start(), p2).race_time_s
    assert r1.mean_s - r2.mean_s == pytest.approx(gap)


def test_registry_rejects_duplicates_and_unknown_keys():
    reg = Registry("thing")
    reg.register("a")(object)
    with pytest.raises(ValueError):
        reg.register("a")(object)
    with pytest.raises(KeyError):
        reg.get("b")


# --- tyre set inventory ------------------------------------------------------
def inv_snapshot(sets):
    return Snapshot(race=RACE, driver="X", compound="SOFT", sets_available=sets)


def test_unconstrained_snapshot_ignores_inventory(sim):
    assert sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30, 6]]).valid


def test_plan_draws_each_stint_from_a_distinct_set(sim):
    snap = inv_snapshot([{"compound": "MEDIUM"}, {"compound": "SOFT", "laps_run": 5}])
    assert sim.simulate(snap, [["SOFT", 20], ["MEDIUM", 30]]).valid
    res = sim.simulate(snap, [["SOFT", 15], ["MEDIUM", 15], ["MEDIUM", 20]])
    assert not res.valid and "no MEDIUM set" in res.errors[0]


def test_new_set_requested_but_only_used_available(sim):
    snap = inv_snapshot([{"compound": "MEDIUM", "laps_run": 7}])
    res = sim.simulate(snap, [["SOFT", 20], ["MEDIUM", 30]])
    assert not res.valid and "[7] laps run" in res.errors[0]
    assert sim.simulate(snap, [["SOFT", 20], ["MEDIUM", 30, 7]]).valid


def test_named_set_supplies_start_age(sim):
    snap = inv_snapshot([{"compound": "MEDIUM", "laps_run": 6, "set_id": "m2"}])
    named = sim.simulate(snap, [["SOFT", 20], {"compound": "MEDIUM", "laps": 30, "set_id": "m2"}])
    explicit = sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30, 6]])
    assert named.valid and named.race_time_s == pytest.approx(explicit.race_time_s)
    assert named.plan.stints[1].start_age == 6


@pytest.mark.parametrize("stint, fragment", [
    ({"compound": "MEDIUM", "laps": 30, "set_id": "nope"}, "not available"),
    ({"compound": "SOFT", "laps": 30, "set_id": "m2"}, "is MEDIUM"),
    ({"compound": "MEDIUM", "laps": 30, "set_id": "m2", "start_age": 2}, "conflicts"),
])
def test_named_set_errors(sim, stint, fragment):
    snap = inv_snapshot([{"compound": "MEDIUM", "laps_run": 6, "set_id": "m2"}])
    res = sim.simulate(snap, [["SOFT", 20], stint])
    assert not res.valid and any(fragment in e for e in res.errors)


def test_assumed_sets_can_be_excluded_by_config():
    sets = [{"compound": "MEDIUM", "source": "assumed"}, {"compound": "MEDIUM", "laps_run": 4, "source": "inferred"}]
    snap = inv_snapshot(sets)
    strict = Simulator(synthetic_frame(), SimConfig(inventory_sources=["observed", "inferred"]))
    assert not strict.simulate(snap, [["SOFT", 20], ["MEDIUM", 30]]).valid
    assert strict.simulate(snap, [["SOFT", 20], ["MEDIUM", 30, 4]]).valid
    assert Simulator(synthetic_frame()).simulate(snap, [["SOFT", 20], ["MEDIUM", 30]]).valid


# --- batch -------------------------------------------------------------------
def test_batch_matches_single_simulation(sim):
    plans = [[["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]],
             [["SOFT", 50]], [["SOFT", 20], ["MEDIUM", 30, 6]]]
    batch = sim.simulate_batch(start(), plans, return_laps=True)
    assert batch.valid.tolist() == [True, True, False, True]
    assert np.isnan(batch.times[2]).all()
    for i, plan in enumerate(plans):
        got, want = batch.result(i, start()), sim.simulate(start(), plan, return_laps=True)
        assert got.valid == want.valid
        if want.valid:
            assert got.race_time_s == pytest.approx(want.race_time_s)
            assert got.stint_seconds == pytest.approx(want.stint_seconds)
            assert got.lap_times_s == pytest.approx(want.lap_times_s)
        else:
            assert got.errors == want.errors


def test_batch_shares_draws_across_plans():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_noisy"}, n_samples=40))
    det = Simulator(synthetic_frame())
    p1, p2 = [["SOFT", 15], ["MEDIUM", 35]], [["SOFT", 25], ["MEDIUM", 25]]
    batch = s.simulate_batch(start(), [p1, p2], seed=2)
    assert batch.times.shape == (2, 40)
    gap = det.simulate(start(), p1).race_time_s - det.simulate(start(), p2).race_time_s
    assert batch.times[0] - batch.times[1] == pytest.approx(np.full(40, gap))


def test_batch_with_scenario_and_mid_race_snapshot(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20,
                    elapsed_s=1650.0, stints=[])
    plans = [[["SOFT", 5], ["MEDIUM", 25]], [["SOFT", 10], ["MEDIUM", 20]]]
    scen = Scenario(sc_laps=[25, 26])
    batch = sim.simulate_batch(snap, plans, scenario=scen)
    for i, plan in enumerate(plans):
        assert batch.race_times[i, 0] == pytest.approx(sim.simulate(snap, plan, scenario=scen).race_time_s)
    assert batch.pit_loss[0, 0] == pytest.approx(RACE.pit_loss_s * EventEffects().sc_pit_loss_factor)
    assert batch.pit_loss[1, 0] == pytest.approx(RACE.pit_loss_s)


def test_batch_all_invalid_returns_nan_without_error(sim):
    batch = sim.simulate_batch(start(), [[["SOFT", 50]], [["SOFT", "x"], ["MEDIUM", 30]]])
    assert not batch.valid.any()
    assert batch.times.shape == (2, 1) and np.isnan(batch.times).all()
    assert batch.lap_times is None
    assert batch.plans[1] is None and batch.errors[1]


def test_batch_is_fast_enough_for_brute_force(sim):
    plans = [[["SOFT", a], ["MEDIUM", b], ["SOFT", 50 - a - b]]
             for a in range(8, 35) for b in range(8, 43 - a)]
    assert len(plans) == 378
    t0 = time.perf_counter()
    batch = sim.simulate_batch(start(), plans, n_samples=50)
    elapsed = time.perf_counter() - t0
    print(f"simulate_batch: {len(plans)} plans x 50 samples in {elapsed:.3f} s")
    assert batch.valid.all() and batch.times.shape == (378, 50)
    assert elapsed < 5.0
