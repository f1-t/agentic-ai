"""The optimizer bridge (src/sim/optimizer_bridge.py) and the rule set both packages share
(docs/rules.md). Synthetic data, plus the committed Spain cache where marked."""
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict

import numpy as np
import pytest

from src.baselines import NoFeasiblePlan, Plan, Stint, candidate_plans, greedy_crossover, optimize
from src.baselines.domain import validate_plan
from src.contracts import Snapshot, StintRecord, TyreSet
from src.sim import SimConfig, Simulator
from src.sim.data import CACHE_DIR
from src.sim.degradation import DEGRADATION
from src.sim.optimizer_bridge import (
    BatchOptimizationResult, as_continuation, batch_comparison, from_contract, legacy_snapshot,
    optimize_batch, pace_for_crossover, score_batch, to_contract,
)

from test_sim import RACE, TRUE, start, synthetic_frame

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))


@DEGRADATION.register("test_bridge_noisy")
class CompoundNoise:
    """A random per-compound offset and wear scale per draw, so the ranking of plans
    depends on the draw."""

    def __init__(self, inner, offsets=None, wear=None):
        self.inner, self.compounds = inner, inner.compounds
        self.offsets, self.wear = offsets or {}, wear or {}

    @classmethod
    def fit(cls, data, **kw):
        return cls(DEGRADATION.get("linear").fit(data, **kw))

    def draw(self, rng):
        names = sorted(self.compounds)
        off, wear = rng.normal(0, 0.3, len(names)), rng.normal(1, 0.3, len(names))
        return CompoundNoise(self.inner, dict(zip(names, off)), dict(zip(names, wear)))

    def lap_times(self, compound, ages, laps):
        base, slope = self.inner.coefs[compound]
        ages = np.asarray(ages, dtype=float)
        return base + self.offsets.get(compound, 0.0) + slope * self.wear.get(compound, 1.0) * ages


@pytest.fixture(scope="module")
def sim():
    return Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))


@pytest.fixture(scope="module")
def noisy():
    return Simulator(synthetic_frame(), SimConfig(degradation={"name": "test_bridge_noisy"}))


def mid_race(**kw):
    """Three laps on a MEDIUM set fitted at the end of lap 20."""
    return Snapshot(race=RACE, driver="X", laps_completed=23, compound="MEDIUM", tyre_age=3,
                    stints=[StintRecord(compound="SOFT", laps=20, start_lap=1)], **kw)


# --- from_contract / to_contract -------------------------------------------------
def test_from_contract_at_start(sim):
    q = from_contract(start(), sim)
    assert (q.total_laps, q.completed_laps, q.current_compound, q.tyre_age, q.current_stint_laps) == (
        50, 0, "SOFT", 0, 0)
    assert q.used_compounds == ("SOFT",) and q.available_compounds == ("MEDIUM", "SOFT")
    assert (q.min_stint_laps, q.pit_loss_s, q.require_two_compounds) == (8, 20.0, True)


def test_from_contract_mid_race(sim):
    q = from_contract(mid_race(), sim)
    assert (q.completed_laps, q.current_compound, q.tyre_age, q.current_stint_laps) == (23, "MEDIUM", 3, 3)
    assert q.used_compounds == ("MEDIUM", "SOFT") and q.remaining_laps == 27


def test_from_contract_waives_two_compounds_after_a_non_slick_stint(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=0,
                    stints=[StintRecord(compound="INTERMEDIATE", laps=10, start_lap=1)])
    assert from_contract(snap, sim).require_two_compounds is False


@pytest.mark.parametrize("snap", [start(), mid_race()], ids=["start", "mid_race"])
def test_round_trip_and_rule_agreement(sim, snap):
    q = from_contract(snap, sim)
    plans = [p for p in candidate_plans(q, 2) if p.stints[0].laps > 0]
    assert plans
    for p in plans:
        c = to_contract(p, snap)
        assert c.as_pairs() == p.to_list()
        assert all(s.start_age == 0 and s.set_id is None for s in c.stints)
    assert all(e == [] for e in sim.simulate_batch(snap, [to_contract(p, snap) for p in plans]).errors)
    # every one-stop split, legal or not, gets the same verdict from both packages (R2, R3)
    for n in range(1, q.remaining_laps):
        for c2 in q.available_compounds:
            p = Plan((Stint(q.current_compound, n), Stint(c2, q.remaining_laps - n)))
            try:
                validate_plan(q, p)
                ok = True
            except ValueError:
                ok = False
            assert ok == (sim.validate(snap, to_contract(p, snap)) == []), p


def test_zero_lap_first_stint_is_rejected(sim):
    p = Plan((Stint("MEDIUM", 0), Stint("SOFT", 27)))
    with pytest.raises(ValueError, match="take the snapshot one lap earlier"):
        to_contract(p, mid_race())


def test_to_contract_assigns_new_sets_first_then_the_least_used():
    sets = [TyreSet(compound="MEDIUM", laps_run=6, set_id="m6"), TyreSet(compound="MEDIUM", laps_run=2, set_id="m2"),
            TyreSet(compound="MEDIUM", laps_run=0, set_id="m0"), TyreSet(compound="SOFT", laps_run=0)]
    snap = start().model_copy(update={"sets_available": sets})
    p = Plan((Stint("SOFT", 10), Stint("MEDIUM", 10), Stint("MEDIUM", 10), Stint("MEDIUM", 10), Stint("SOFT", 10)))
    c = to_contract(p, snap)
    assert [(s.set_id, s.start_age) for s in c.stints[1:]] == [("m0", 0), ("m2", 2), ("m6", 6), (None, 0)]
    # no set of that compound left: the plan goes through unassigned and the simulator rejects it
    none_left = snap.model_copy(update={"sets_available": sets[:3]})
    assert to_contract(p, none_left).stints[4].set_id is None


# --- optimize_batch ----------------------------------------------------------------
def adapter(sim, snap, n):
    """Quan's simulator callback over our simulator: (snapshot, plan, seed) -> sample
    `seed` of seed 0, from a one-plan batch (R6). Cached per plan across seeds."""
    seen = {}

    def callback(_state, plan, seed):
        if plan not in seen:
            seen[plan] = sim.simulate_batch(snap, [to_contract(plan, snap)], n_samples=n).times[0]
        return float(seen[plan][seed])
    return callback


@pytest.mark.parametrize("max_stops", [1, 2])
@pytest.mark.parametrize("snap", [start(), mid_race()], ids=["start", "mid_race"])
def test_optimize_batch_equals_callback_optimize(noisy, snap, max_stops):
    n = 6
    ours = optimize_batch(snap, noisy, n_samples=n, max_stops=max_stops)
    theirs = optimize(from_contract(snap, noisy), adapter(noisy, snap, n), tuple(range(n)), max_stops)
    assert isinstance(ours, BatchOptimizationResult)
    assert ours.evaluation.plan == theirs.evaluation.plan
    assert ours.evaluation.seeds == theirs.evaluation.seeds == tuple(range(n))
    assert ours.evaluation.mean_time_s == pytest.approx(theirs.evaluation.mean_time_s, abs=1e-9)
    assert ours.candidates_evaluated == theirs.candidates_evaluated and ours.invalid == 0
    assert ours.plan.as_pairs() == ours.evaluation.plan.to_list()
    assert np.std(ours.evaluation.seed_times_s) > 0


def test_optimize_batch_seed_maps_to_sample_draws(noisy):
    a = optimize_batch(start(), noisy, seed=3, n_samples=5)
    batch = noisy.simulate_batch(start(), [a.plan], seed=3, n_samples=5)
    assert a.evaluation.seed_times_s == tuple(batch.times[0])
    assert a.diagnostics["seed"] == 3 and a.diagnostics["n_samples"] == 5


def test_zero_lap_candidates_are_filtered(sim):
    # 20 laps run on the set: candidate_plans offers an immediate stop, the bridge drops it
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20)
    zero = [p for p in candidate_plans(from_contract(snap, sim), 2) if p.stints[0].laps == 0]
    assert zero
    res = optimize_batch(snap, sim)
    assert res.diagnostics["zero_lap_first_stints"] == len(zero) and res.invalid == 0
    assert res.evaluation.plan.stints[0].laps >= 1


def test_one_used_medium_is_carried_into_the_best_plan(sim):
    snap = start().model_copy(update={"sets_available": [
        TyreSet(compound="MEDIUM", laps_run=5, set_id="m1"),
        TyreSet(compound="SOFT", laps_run=0, set_id="s1"), TyreSet(compound="SOFT", laps_run=0, set_id="s2")]})
    res = optimize_batch(snap, sim)
    medium = [s for s in res.plan.stints[1:] if s.compound == "MEDIUM"]
    assert len(medium) == 1 and (medium[0].set_id, medium[0].start_age) == ("m1", 5)
    assert {"stint": 1 + res.plan.stints.index(medium[0]), "compound": "MEDIUM", "set_id": "m1",
            "start_age": 5, "substituted": True} in res.diagnostics["set_assignment"]
    assert res.invalid > 0   # plans with two MEDIUM stints have no second set
    # the time is the used set's: 5 laps older on every MEDIUM lap
    fresh = sim.simulate(start(), res.plan.model_copy(update={"stints": [
        s.model_copy(update={"start_age": 0, "set_id": None}) for s in res.plan.stints]}))
    assert res.evaluation.mean_time_s == pytest.approx(
        fresh.remaining.mean_s + 5 * TRUE["MEDIUM"][1] * medium[0].laps)


def test_no_feasible_plan(sim):
    with pytest.raises(NoFeasiblePlan):
        optimize_batch(start().model_copy(update={"sets_available": []}), sim)


# --- as_continuation / pace_for_crossover -----------------------------------------
def test_as_continuation_satisfies_compare(sim):
    snap = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10,
                    sets_available=[TyreSet(compound="MEDIUM", laps_run=4, set_id="m4"),
                                    TyreSet(compound="SOFT", laps_run=0, set_id="s0")])
    cont = as_continuation(sim, max_stops=1)
    seen = []

    def spy(adv):
        seen.append((adv, cont(adv)))
        return seen[-1][1]

    res = sim.compare(snap, [{"action": "stay"}, {"action": "pit", "compound": "MEDIUM", "set_id": "m4"}],
                      continuation=spy)
    assert res.valid == [True, True], res.errors
    (stay_adv, stay_plan), (_, pit_plan) = seen
    assert stay_plan.stints[0].start_age == 0
    assert res.candidates[0].stints[0].laps == 1 + stay_plan.stints[0].laps
    assert res.candidates[0].stints[1:] == stay_plan.stints[1:]
    assert stay_plan == optimize_batch(stay_adv, sim, max_stops=1).plan
    assert (pit_plan.stints[0].compound, pit_plan.stints[0].start_age) == ("MEDIUM", 4)
    carried = res.candidates[1].stints[1]
    assert (carried.compound, carried.set_id, carried.start_age) == ("MEDIUM", "m4", 4)
    assert res.candidates[1].stints[0].laps == 1


def test_pace_for_crossover_reads_the_fit(sim):
    pace = pace_for_crossover(sim, start())
    base, slope = TRUE["SOFT"]
    assert pace("SOFT", 4) == pytest.approx(base + 4 * slope)
    plan = greedy_crossover(from_contract(start(), sim), pace, 2)
    assert sim.validate(start(), to_contract(plan, start())) == []


# --- the legacy path and Spain (committed cache) ----------------------------------
@pytest.fixture(scope="module")
def legacy():
    from src import tools
    return tools._get_simulator(), tools._race_params()


@pytest.mark.skipif(not HAS_CACHE, reason="no cache")
def test_legacy_frame_reproduces_ground_truth(legacy):
    sim, race = legacy
    snap = Snapshot.at_start(race, "HAM", "SOFT")
    res = optimize_batch(snap, sim)
    assert round(res.evaluation.mean_time_s, 1) == 5510.1
    # S28/S28/M10 and S28/M10/S28 tie exactly; the tie-break takes the earlier second stop
    truth = Plan((Stint("SOFT", 28), Stint("SOFT", 28), Stint("MEDIUM", 10)))
    assert Counter(res.evaluation.plan.stints) == Counter(truth.stints)
    assert res.evaluation.plan.to_list() == [["SOFT", 28], ["MEDIUM", 10], ["SOFT", 28]]
    (ev,), _, _, _ = score_batch(snap, [truth], sim)
    assert ev.mean_time_s == pytest.approx(res.evaluation.mean_time_s, abs=1e-9)


@pytest.mark.skipif(not HAS_CACHE, reason="no cache")
def test_spain_two_stop_search_is_fast():
    from src import tools
    from src.sim.data import lap_frame, load_session
    sim = Simulator(lap_frame(load_session(2021, "Spain").laps))
    snap = Snapshot.at_start(tools._race_params(), "HAM", "SOFT")
    optimize_batch(snap, sim, n_samples=50)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        res = optimize_batch(snap, sim, n_samples=50, max_stops=2)
        best = min(best, time.perf_counter() - t0)
    print(f"optimize_batch: {res.candidates_evaluated} plans x 50 samples in {best:.3f} s")
    assert res.candidates_evaluated > 2000 and best < 0.5


@pytest.mark.skipif(not HAS_CACHE, reason="no cache")
@pytest.mark.parametrize("lap", [0, 30])
def test_batch_comparison_has_the_callback_report_shape(lap):
    from src import tools
    from src.baselines.__main__ import comparison
    from src.baselines.legacy import load_legacy_context

    def shape(value):
        if isinstance(value, dict):
            return {k: shape(v) for k, v in value.items()}
        return type(value).__name__ if not isinstance(value, list) else "list"

    context = load_legacy_context(completed_laps=lap)
    ours = batch_comparison(context, (0, 1), 2)
    theirs = comparison(context, (0, 1), 2)
    assert shape(ours) == shape(theirs)
    assert ours["snapshot"] == asdict(context.snapshot)
    assert ours["optimizer"]["expected_remaining_time_s"] == pytest.approx(
        theirs["optimizer"]["expected_remaining_time_s"], abs=0.2)
    snap = legacy_snapshot(context, tools._race_params())
    assert snap.compounds_used == set(context.snapshot.used_compounds)
    assert sum(s.laps for s in snap.stints) == lap - context.snapshot.current_stint_laps


@pytest.mark.skipif(not HAS_CACHE, reason="no cache")
def test_batch_comparison_needs_contiguous_seeds():
    from src.baselines.legacy import load_legacy_context
    with pytest.raises(ValueError, match="0 1 ... n-1"):
        batch_comparison(load_legacy_context(), (3, 5), 2)


@pytest.mark.skipif(not HAS_CACHE, reason="no cache")
def test_cli_batch_scorer():
    run = [sys.executable, "-m", "src.baselines", "--scorer", "batch", "--lap", "30"]
    out = subprocess.run([*run, "--seeds", "0", "1"], capture_output=True, text=True, check=True)
    report = json.loads(out.stdout)
    assert report["simulator"] == "sim_batch" and report["optimizer"]["seeds"] == [0, 1]
    assert set(report["baselines"]) == {"actual_strategy", "fixed_one_stop", "fixed_two_stop", "greedy_crossover"}
    bad = subprocess.run([*run, "--seeds", "4"], capture_output=True, text=True)
    assert bad.returncode == 2 and "0 1 ... n-1" in bad.stderr
