"""Offline simulator tests. Synthetic data only, no model and no network."""
import time

import numpy as np
import pandas as pd
import pytest

from src.contracts import CompareResult, Plan, RaceParams, Snapshot
from src.sim import SimConfig, Simulator
from src.sim.config import EventEffects
from src.sim.degradation import DEGRADATION
from src.sim.engine import FIT_CACHE_SIZE
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
    return Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))


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
    assert res.race_time_s == pytest.approx(hand_time(plan) + EventEffects().lap1_extra_s)
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
        TRUE["MEDIUM"][1] * 6 for _ in range(30)) + EventEffects().lap1_extra_s)


def test_return_laps_sums_to_remaining_less_pit_loss(sim):
    res = sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30]], return_laps=True)
    assert len(res.lap_times_s) == 50
    assert sum(res.lap_times_s) + res.pit_loss_total_s == pytest.approx(res.remaining.mean_s)


def test_min_stint_applies_to_first_stint_at_race_start(sim):
    res = sim.simulate(start(), [["SOFT", 5], ["MEDIUM", 45]])
    assert not res.valid and any("below the 8-lap minimum" in e for e in res.errors)


def test_min_stint_exempts_stint_already_underway(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20, stints=[])
    assert sim.simulate(snap, [["SOFT", 5], ["MEDIUM", 25]]).valid


def test_min_stint_applies_to_first_stint_just_after_a_stop(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="MEDIUM", tyre_age=0,
                    stints=[{"compound": "SOFT", "laps": 20, "start_lap": 1}])
    res = sim.simulate(snap, [["MEDIUM", 7], ["SOFT", 23]])
    assert not res.valid and any("below the 8-lap minimum" in e for e in res.errors)


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


# --- compare -----------------------------------------------------------------
def other(compound):
    return "MEDIUM" if compound == "SOFT" else "SOFT"


def stub(first_laps=12):
    """Continuation that runs the fitted set for first_laps, then the other compound."""
    def cont(adv):
        return [[adv.compound, first_laps], [other(adv.compound), adv.remaining_laps - first_laps]]
    return cont


def mid_race(**kw):
    return Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10, **kw)


def test_compare_pairwise_probabilities_are_complementary():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_noisy"}, n_samples=30))
    plans = [[["SOFT", 15], ["MEDIUM", 35]], [["SOFT", 25], ["MEDIUM", 25]],
             [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]]
    res = s.compare(start(), plans, seed=4)
    for i in range(3):
        assert res.p_beats[i][i] is None
        for j in range(3):
            if i != j:
                assert res.p_beats[i][j] + res.p_beats[j][i] == pytest.approx(1.0)
    assert sum(res.p_best) == pytest.approx(1.0)
    assert res.n_samples == 30 and res.seed == 4


def test_compare_deterministic_probabilities_and_identical_plans(sim):
    fast, slow = [["SOFT", 25], ["MEDIUM", 25]], [["SOFT", 20], ["MEDIUM", 30]]
    res = sim.compare(start(), [fast, slow, fast])
    assert res.p_beats[0][1] == 1.0 and res.p_beats[1][0] == 0.0
    assert res.p_beats[0][2] == res.p_beats[2][0] == 0.5
    assert res.p_best == [0.5, 0.0, 0.5]
    assert res.best == 0
    gap = sim.simulate(start(), fast).race_time_s - sim.simulate(start(), slow).race_time_s
    assert res.gap_mean_s[0][1] == pytest.approx(gap) and res.gap_mean_s[1][0] == pytest.approx(-gap)


def test_compare_shared_offset_cancels_in_gaps():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_noisy"}, n_samples=100))
    res = s.compare(start(), [[["SOFT", 15], ["MEDIUM", 35]], [["SOFT", 25], ["MEDIUM", 25]]], seed=1)
    assert res.summaries[0].std_s > 0
    assert res.gap_std_s[0][1] == pytest.approx(0.0, abs=1e-6)
    assert res.p_best == [0.0, 1.0]


def test_compare_actions_match_hand_built_plans(sim):
    snap = mid_race()
    res = sim.compare(snap, [{"action": "stay"}, {"action": "pit", "compound": "MEDIUM"},
                             [["SOFT", 11], ["MEDIUM", 29]]], continuation=stub())
    assert all(res.valid)
    hand = [[["SOFT", 13], ["MEDIUM", 27]], [["SOFT", 1], ["MEDIUM", 12], ["SOFT", 27]]]
    for i, plan in enumerate(hand):
        assert res.candidates[i].as_pairs() == plan
        assert res.summaries[i].mean_s == pytest.approx(sim.simulate(snap, plan).remaining.mean_s)


def test_compare_stay_merges_into_first_stint(sim):
    seen = []

    def cont(adv):
        seen.append(adv)
        return stub(7)(adv)

    res = sim.compare(mid_race(), [{"action": "stay"}], continuation=cont)
    adv = seen[0]
    assert (adv.laps_completed, adv.compound, adv.tyre_age, adv.elapsed_s, adv.stints) == (11, "SOFT", 11, None, [])
    assert res.valid == [True] and res.candidates[0].stints[0].laps == 1 + 7


def test_compare_stay_rejects_continuation_on_another_compound(sim):
    res = sim.compare(mid_race(), [{"action": "stay"}],
                      continuation=lambda adv: [["MEDIUM", 20], ["SOFT", adv.remaining_laps - 20]])
    assert res.valid == [False] and res.errors[0] == ["continuation must continue the current set"]


def test_compare_pit_consumes_the_chosen_set(sim):
    snap = mid_race(sets_available=[{"compound": "MEDIUM", "laps_run": 6, "set_id": "m2"},
                                    {"compound": "SOFT", "laps_run": 3}])
    seen = []

    def cont(adv):
        seen.append(adv)
        return [["MEDIUM", 20], ["SOFT", adv.remaining_laps - 20, 3]]

    res = sim.compare(snap, [{"action": "pit", "compound": "MEDIUM", "set_id": "m2"}], continuation=cont)
    adv = seen[0]
    assert (adv.laps_completed, adv.compound, adv.tyre_age) == (11, "MEDIUM", 6)
    assert [t.compound for t in adv.sets_available] == ["SOFT"]
    assert adv.stints[0].model_dump() == {"compound": "SOFT", "laps": 11, "start_lap": 1, "start_age": 0}
    assert res.valid == [True]
    second = res.candidates[0].stints[1]
    assert (second.set_id, second.start_age) == ("m2", 6)

    # a later stop back onto the set fitted at this stop is invalid
    again = sim.compare(snap, [{"action": "pit", "compound": "MEDIUM", "set_id": "m2"}],
                        continuation=lambda adv: [["MEDIUM", 15], {"compound": "MEDIUM", "laps": 24, "set_id": "m2"}])
    assert again.valid == [False] and any("'m2' is not available" in e for e in again.errors[0])


def test_compare_actions_without_continuation_are_error_entries(sim):
    res = sim.compare(mid_race(), [{"action": "stay"}, [["SOFT", 11], ["MEDIUM", 29]],
                                   {"action": "pit", "compound": "MEDIUM"}])
    assert res.valid == [False, True, False]
    assert res.errors[0] == res.errors[2] == ["continuation required for action candidates"]
    assert res.candidates[0] is None and res.best == 1


@DEGRADATION.register("test_soft_risk")
class SoftRiskModel(NoisyModel):
    """Per-draw offset on SOFT laps only, so plans with more SOFT laps have a fatter tail."""

    @classmethod
    def fit(cls, data, **kw):
        return cls(DEGRADATION.get("linear").fit(data, **kw))

    def draw(self, rng):
        return SoftRiskModel(self.inner, rng.normal(0, 0.5))

    def lap_times(self, compound, ages, laps):
        return self.inner.lap_times(compound, ages, laps) + (self.offset if compound == "SOFT" else 0.0)


def test_compare_dominated_needs_mean_and_tail():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_soft_risk"}, n_samples=200))
    plans = [[["MEDIUM", 20], ["SOFT", 30]],   # fastest on mean, widest tail
             [["MEDIUM", 35], ["SOFT", 15]],   # slower, tighter tail
             [["MEDIUM", 40], ["SOFT", 10]]]   # slower again and a worse tail than the second
    res = s.compare(start("MEDIUM"), plans, seed=0)
    m = [r.mean_s for r in res.summaries]
    c = [r.cvar_90_s for r in res.summaries]
    assert m[0] < m[1] < m[2] and c[1] < c[0] and c[1] < c[2]
    assert res.dominated == [False, False, True] and res.best == 0


def test_compare_invalid_candidate_is_excluded(sim):
    res = sim.compare(start(), [[["SOFT", 50]], [["SOFT", 20], ["MEDIUM", 30]], [["SOFT", "x"]]])
    assert res.valid == [False, True, False] and res.best == 1
    assert res.candidates[2] is None and res.errors[2]
    for bad in (0, 2):
        assert res.summaries[bad] is None and res.p_best[bad] is None and not res.dominated[bad]
        assert res.p_beats[bad] == res.gap_mean_s[bad] == res.gap_std_s[bad] == [None] * 3
        assert all(row[bad] is None for row in res.p_beats + res.gap_mean_s + res.gap_std_s)
    assert res.p_best[1] == 1.0
    assert CompareResult.model_validate_json(res.model_dump_json()) == res


def test_compare_pit_rejects_start_age_conflicting_with_named_set(sim):
    snap = mid_race(sets_available=[{"compound": "MEDIUM", "laps_run": 6, "set_id": "m2"},
                                    {"compound": "SOFT"}])
    act = {"action": "pit", "compound": "MEDIUM", "set_id": "m2", "start_age": 2}
    res = sim.compare(snap, [act], continuation=lambda adv: [["MEDIUM", 20], ["SOFT", adv.remaining_laps - 20]])
    assert res.valid == [False]
    assert res.errors[0] == ["start_age 2 conflicts with set 'm2' (6 laps run)"]
    ok = dict(act, start_age=6)
    assert sim.compare(snap, [ok], continuation=lambda adv: [["MEDIUM", 20], ["SOFT", adv.remaining_laps - 20]]).valid == [True]


# --- fit windows -------------------------------------------------------------
PRIOR = SimConfig().pre_race_fuel_slope_s_per_lap
PLANS = [[["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 12], ["MEDIUM", 38]], [["SOFT", 30], ["MEDIUM", 20]],
         [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]], [["SOFT", 10], ["SOFT", 10], ["MEDIUM", 30]]]


def synthetic_practice(fuel=0.0, coefs=TRUE, drivers=("A", "B", "C")):
    """One 10-lap long run per compound per driver. Practice has no race lap, so lap is NaN;
    `fuel` is the per-lap fuel gain within the run, which practice cannot tell from wear."""
    rows = []
    for k, drv in enumerate(drivers):
        for comp, (base, slope) in coefs.items():
            for age in range(1, 11):
                rows.append({"driver": drv, "session": f"FP{k % 3 + 1}", "compound": comp, "tyre_age": age,
                             "lap": np.nan, "clean": True, "lap_time_s": base + (slope + fuel) * age})
    return pd.DataFrame(rows)


def window(fit_window, **kw):
    return SimConfig(fit_window=fit_window, **kw)


def at_lap(n, driver="X"):
    return Snapshot(race=RACE, driver=driver, laps_completed=n, compound="SOFT", tyre_age=n)


def coefs_close(a, b, abs=1e-6):
    return set(a) == set(b) and all(a[c] == pytest.approx(b[c], abs=abs) for c in a)


@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
def test_world_equals_full_when_window_is_whole_race(name):
    # practice on the true coefficients and race fuel at the prior: the prior agrees with
    # the race, so the penalty vanishes at the full fit
    frame = synthetic_frame(fuel=PRIOR if name == "linear_fuel" else 0.0)
    full = Simulator(frame, SimConfig(degradation={"name": name}))
    last = RACE.total_laps - 1
    s = Simulator(frame, window("to_snapshot", degradation={"name": name}), synthetic_practice())
    late = s.fitted_for(at_lap(last))
    assert late.window == ("to_snapshot", last, None)
    assert coefs_close(late.degradation.coefs, full.degradation.coefs)
    assert getattr(late.degradation, "fuel_slope", 0.0) == pytest.approx(getattr(full.degradation, "fuel_slope", 0.0))


@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
def test_whole_race_window_without_practice_is_close_to_full(name):
    # no practice: base and wear unpenalised, only the fuel slope is pulled toward the prior
    frame = synthetic_frame(fuel=FUEL)
    full = Simulator(frame, SimConfig(degradation={"name": name}))
    late = Simulator(frame, window("to_snapshot", degradation={"name": name})).fitted_for(at_lap(RACE.total_laps - 1))
    assert coefs_close(late.degradation.coefs, full.degradation.coefs, abs=1e-9 if name == "linear" else 0.02)
    if name == "linear_fuel":
        assert FUEL > late.degradation.fuel_slope > FUEL - 1e-3  # pulled a little toward the -0.045 prior
    assert late.ref_lap_s == full.ref_lap_s


def test_pre_race_recovers_practice_coefficients():
    practice = synthetic_practice()
    s = Simulator(synthetic_frame(fuel=FUEL), window("pre_race", degradation={"name": "linear_fuel"}), practice)
    fit = s.fitted_for(start())
    for comp, (base, slope) in TRUE.items():
        assert fit.degradation.coefs[comp] == pytest.approx((base, slope), abs=1e-6)
    assert fit.degradation.fuel_slope == PRIOR
    assert fit.degradation.fit_report["fuel_slope_source"] == "prior"
    assert fit.degradation.fit_report["n_race_rows"] == 0
    assert fit.ref_lap_s == practice["lap_time_s"].median()
    assert s.fitted_for(at_lap(30)) is fit  # pre_race does not depend on the lap


@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
def test_to_snapshot_grows_with_the_race(name):
    practice = synthetic_practice(coefs={"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.10)})
    s = Simulator(synthetic_frame(), window("to_snapshot", degradation={"name": name}), practice)
    early, late = (s.fitted_for(at_lap(n)).degradation.coefs["MEDIUM"][1] for n in (5, 40))
    race_wear, practice_wear = TRUE["MEDIUM"][1], 0.10
    assert race_wear < late < early < practice_wear
    assert abs(early - practice_wear) < abs(early - race_wear)
    assert abs(late - race_wear) < abs(late - practice_wear)


@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
@pytest.mark.parametrize("practice", [None, synthetic_practice(),
                                      synthetic_practice(coefs={"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.10)})])
def test_to_snapshot_wear_is_positive_by_lap_30(name, practice):
    fit = Simulator(synthetic_frame(), window("to_snapshot", degradation={"name": name}), practice)
    coefs = fit.fitted_for(at_lap(30)).degradation.coefs
    assert set(coefs) == set(TRUE) and all(wear > 0 for _, wear in coefs.values())


def test_to_snapshot_at_lap_zero_is_pre_race():
    practice = synthetic_practice()
    cfg = dict(degradation={"name": "linear_fuel"})
    zero = Simulator(synthetic_frame(fuel=FUEL), window("to_snapshot", **cfg), practice).fitted_for(start())
    pre = Simulator(synthetic_frame(fuel=FUEL), window("pre_race", **cfg), practice).fitted_for(start())
    assert coefs_close(zero.degradation.coefs, pre.degradation.coefs, abs=1e-9)
    assert zero.degradation.fuel_slope == pytest.approx(PRIOR) and pre.degradation.fuel_slope == PRIOR
    assert zero.degradation.fit_report["fuel_slope_source"] == "prior"
    assert zero.ref_lap_s == pre.ref_lap_s


def test_shrinkage_weight_is_the_data_share_of_precision():
    tau = SimConfig().shrinkage
    practice = synthetic_practice(coefs={"SOFT": TRUE["SOFT"]})  # MEDIUM is not in practice
    s = Simulator(synthetic_frame(fuel=FUEL), window("to_snapshot", degradation={"name": "linear_fuel"}), practice)
    rep = s.fitted_for(at_lap(30)).degradation.fit_report
    w, n = rep["shrinkage_weight"], rep["n_race_rows_per_compound"]
    race = synthetic_frame()[lambda d: d["lap"] <= 30]
    assert n == race["compound"].value_counts().to_dict() and rep["n_race_rows"] == len(race)
    assert w["base_s"]["SOFT"] == pytest.approx(n["SOFT"] / (n["SOFT"] + tau["base_s"] ** -2))
    assert w["base_s"]["MEDIUM"] == pytest.approx(n["MEDIUM"] / (n["MEDIUM"] + (2 * tau["base_s"]) ** -2))
    sq = float((race.loc[race["compound"] == "MEDIUM", "tyre_age"] ** 2).sum())
    assert w["wear_s_per_lap"]["MEDIUM"] == pytest.approx(sq / (sq + (2 * tau["wear_s_per_lap"]) ** -2))
    lsq = float((race["lap"] ** 2).sum())
    assert w["fuel_s_per_lap"] == pytest.approx(lsq / (lsq + tau["fuel_s_per_lap"] ** -2))
    zero = s.fitted_for(start()).degradation.fit_report["shrinkage_weight"]
    assert zero["fuel_s_per_lap"] == 0.0 and set(zero["base_s"].values()) == {0.0}
    full = Simulator(synthetic_frame(fuel=FUEL)).degradation.fit_report["shrinkage_weight"]
    assert full["fuel_s_per_lap"] == 1.0 and set(full["wear_s_per_lap"].values()) == {1.0}


def test_compound_absent_from_practice_shrinks_to_the_pooled_practice_mean():
    from src.sim.degradation import LinearDegradation
    practice = synthetic_practice(coefs={"SOFT": (80.0, 0.10), "HARD": (84.0, 0.02)})
    prior = LinearDegradation.fit(practice)
    race = synthetic_frame()[lambda d: d["lap"] <= 10]
    tight = LinearDegradation.fit(race, prior=prior, shrinkage={"base_s": 1e-6, "wear_s_per_lap": 1e-6})
    assert tight.coefs["MEDIUM"] == pytest.approx((82.0, 0.06), abs=1e-4)
    assert tight.coefs["HARD"] == pytest.approx((84.0, 0.02), abs=1e-6)  # no race rows: the prior
    assert tight.fit_report["n_rows"]["HARD"] == 30 and tight.fit_report["max_tyre_age"]["HARD"] == 10


def test_to_snapshot_without_practice_penalises_only_the_fuel_slope():
    s = Simulator(synthetic_frame(fuel=FUEL), window("to_snapshot", degradation={"name": "linear_fuel"}))
    w = s.fitted_for(at_lap(30)).degradation.fit_report["shrinkage_weight"]
    assert set(w["base_s"].values()) == set(w["wear_s_per_lap"].values()) == {1.0}
    assert 0.0 < w["fuel_s_per_lap"] < 1.0


def test_base_instance_carries_the_lap_zero_window():
    practice = synthetic_practice(coefs={"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.10)})
    full = Simulator(synthetic_frame())
    for fit_window in ("pre_race", "to_snapshot"):
        s = Simulator(synthetic_frame(), window(fit_window), practice)
        assert coefs_close(s.degradation.coefs, {"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.10)})
        assert s.ref_lap_s == practice["lap_time_s"].median()
        assert coefs_close(s.fitted_for(start()).degradation.coefs, s.degradation.coefs, abs=1e-9)
        bare = Simulator(synthetic_frame(), window(fit_window))  # no practice: the full fit
        assert bare.degradation.coefs == full.degradation.coefs


@pytest.mark.parametrize("fit_window", ["pre_race", "to_snapshot"])
def test_window_without_practice_raises_at_fit_time(fit_window):
    s = Simulator(synthetic_frame(), window(fit_window))  # construction is fine
    with pytest.raises(ValueError, match="fit window needs practice laps"):
        s.simulate(start(), [["SOFT", 20], ["MEDIUM", 30]])


def test_to_snapshot_without_practice_fits_race_laps_once_there_are_some():
    s = Simulator(synthetic_frame(), window("to_snapshot"))
    assert s.simulate(at_lap(30), [["SOFT", 5], ["MEDIUM", 15]]).valid


def test_narrow_window_drops_compound_and_practice_only_compound_is_modelled():
    practice = synthetic_practice(coefs={"SOFT": TRUE["SOFT"], "HARD": (83.0, 0.02)})
    s = Simulator(synthetic_frame(), window("to_snapshot"), practice)
    snap = Snapshot(race=RACE, driver="X", laps_completed=3, compound="SOFT", tyre_age=3)
    fit = s.fitted_for(snap)  # 3 race laps on MEDIUM, under min_samples, none in practice
    assert fit.modelled == {"SOFT", "HARD"}
    assert fit.degradation.coefs["HARD"] == pytest.approx((83.0, 0.02), abs=1e-6)
    res = s.simulate(snap, [["SOFT", 20], ["MEDIUM", 27]])
    assert not res.valid and "Unknown compound 'MEDIUM'" in res.errors[0]


def test_exclude_driver_removes_that_driver():
    others = synthetic_frame().assign(driver="A")
    slow = synthetic_frame().assign(driver="X", lap_time_s=lambda d: d["lap_time_s"] + 1.0)
    frame = pd.concat([others, slow], ignore_index=True)
    pooled = Simulator(frame)
    assert pooled.degradation.coefs["SOFT"][0] == pytest.approx(TRUE["SOFT"][0] + 0.5)
    s = Simulator(frame, SimConfig(exclude_snapshot_driver=True))
    out = s.fitted_for(start())  # driver X
    for comp, (base, slope) in TRUE.items():
        assert out.degradation.coefs[comp] == pytest.approx((base, slope), abs=1e-6)
    absent = s.fitted_for(Snapshot.at_start(RACE, "Z", "SOFT"))
    assert absent.degradation.coefs == pooled.degradation.coefs


def test_fitted_for_is_cached():
    root = Simulator(synthetic_frame())
    assert root.fitted_for(at_lap(10)) is root  # full without exclusion
    s = Simulator(synthetic_frame(), window("to_snapshot"), synthetic_practice())
    a = s.fitted_for(at_lap(10))
    assert s.fitted_for(at_lap(10)) is a and a.fitted_for(at_lap(20)) is a
    assert s.fitted_for(at_lap(11)) is not a
    ex = Simulator(synthetic_frame().assign(driver="D0"), SimConfig(exclude_snapshot_driver=True))
    first = ex.fitted_for(Snapshot.at_start(RACE, "D0", "SOFT"))
    for k in range(FIT_CACHE_SIZE + 6):
        ex.fitted_for(Snapshot.at_start(RACE, f"D{k}", "SOFT"))
    assert len(ex._fits) == FIT_CACHE_SIZE
    assert ex.fitted_for(Snapshot.at_start(RACE, "D0", "SOFT")) is not first  # oldest went first


@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
def test_fit_report_fields_present(name):
    s = Simulator(synthetic_frame(fuel=FUEL), window("to_snapshot", degradation={"name": name}),
                  synthetic_practice())
    keys = {"n_rows", "n_race_rows", "n_race_rows_per_compound", "max_tyre_age", "fuel_slope_source",
            "condition_number", "shrinkage_weight"}
    for snap in (start(), at_lap(30)):
        rep = s.fitted_for(snap).degradation.fit_report
        assert set(rep) == keys
        assert set(rep["n_rows"]) == set(rep["n_race_rows_per_compound"]) == set(rep["max_tyre_age"]) == set(TRUE)
        assert sum(rep["n_race_rows_per_compound"].values()) == rep["n_race_rows"]
        assert set(rep["shrinkage_weight"]["base_s"]) == set(rep["shrinkage_weight"]["wear_s_per_lap"]) == set(TRUE)
    assert np.isfinite(rep["condition_number"]) and rep["condition_number"] >= 1.0
    full = Simulator(synthetic_frame(fuel=FUEL), SimConfig(degradation={"name": name})).degradation.fit_report
    assert set(full) == keys and np.isfinite(full["condition_number"])
    assert full["n_rows"] == {"SOFT": 200, "MEDIUM": 200} and full["n_race_rows"] == 400
    assert full["n_race_rows_per_compound"] == {"SOFT": 200, "MEDIUM": 200}
    assert full["max_tyre_age"] == {"SOFT": 20, "MEDIUM": 20}
    assert full["fuel_slope_source"] == ("fitted" if name == "linear_fuel" else None)
    zero = s.fitted_for(start()).degradation.fit_report
    assert zero["n_race_rows"] == 0 and zero["max_tyre_age"] == {"SOFT": 10, "MEDIUM": 10}
    late = s.fitted_for(at_lap(30)).degradation.fit_report
    assert late["n_race_rows"] > 0 and sum(late["n_rows"].values()) == late["n_race_rows"] + 60
    assert late["max_tyre_age"] == {"SOFT": 20, "MEDIUM": 20}
    if name == "linear_fuel":
        assert late["fuel_slope_source"] == "shrunk" and zero["fuel_slope_source"] == "prior"
    else:
        assert late["fuel_slope_source"] is None and late["shrinkage_weight"]["fuel_s_per_lap"] is None


@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
def test_full_ranking_is_unchanged_by_a_constant_lap_offset(name):
    cfg = SimConfig(degradation={"name": name})
    frame = synthetic_frame(fuel=FUEL)
    base = Simulator(frame, cfg).simulate_batch(start(), PLANS).times[:, 0]
    shifted = Simulator(frame.assign(lap_time_s=frame["lap_time_s"] + 3.7), cfg).simulate_batch(start(), PLANS)
    assert list(np.argsort(base)) == list(np.argsort(shifted.times[:, 0]))
    assert shifted.times[:, 0] - base == pytest.approx(np.full(len(PLANS), 3.7 * RACE.total_laps))


def test_pre_race_argmin_does_not_depend_on_the_fuel_prior():
    best = set()
    for prior in (-0.1, PRIOR, 0.0, 0.05):
        cfg = window("pre_race", degradation={"name": "linear_fuel"}, pre_race_fuel_slope_s_per_lap=prior)
        s = Simulator(synthetic_frame(fuel=FUEL), cfg, synthetic_practice(fuel=0.01))
        res = s.simulate_batch(start(), PLANS)
        assert res.valid.all()
        best.add(int(np.argmin(res.times[:, 0])))
    assert len(best) == 1


def test_compare_runs_on_the_windowed_fit():
    s = Simulator(synthetic_frame(), window("pre_race"),
                  synthetic_practice(coefs={"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.10)}))
    res = s.compare(start(), PLANS[:2])
    fit = s.fitted_for(start())
    assert res.summaries[0].mean_s == fit.simulate(start(), PLANS[0]).remaining.mean_s
    assert res.summaries[0].mean_s != Simulator(synthetic_frame()).simulate(start(), PLANS[0]).remaining.mean_s


# --- lap 1 term ----------------------------------------------------------------
LAP1_PLANS = [[["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 10], ["MEDIUM", 40]]]


def sim_with_lap1(extra):
    return Simulator(synthetic_frame(), SimConfig(effects=EventEffects(lap1_extra_s=extra)))


def test_start_snapshot_pays_lap1_extra_exactly():
    on, off = sim_with_lap1(9.0), sim_with_lap1(0.0)
    for plan in LAP1_PLANS:
        assert on.simulate(start(), plan).race_time_s == pytest.approx(off.simulate(start(), plan).race_time_s + 9.0)


def test_mid_race_snapshot_has_no_lap1_extra():
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20,
                    elapsed_s=1650.0, stints=[])
    plan = [["SOFT", 5], ["MEDIUM", 25]]
    assert sim_with_lap1(9.0).simulate(snap, plan).race_time_s == pytest.approx(
        sim_with_lap1(0.0).simulate(snap, plan).race_time_s)


def test_lap1_extra_does_not_change_plan_ranking():
    on, off = sim_with_lap1(9.0), sim_with_lap1(0.0)
    t_on = [on.simulate(start(), p).race_time_s for p in LAP1_PLANS]
    t_off = [off.simulate(start(), p).race_time_s for p in LAP1_PLANS]
    assert np.argsort(t_on).tolist() == np.argsort(t_off).tolist()
    assert t_on[0] - t_on[1] == pytest.approx(t_off[0] - t_off[1])
