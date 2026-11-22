"""A STAY_OUT taken under SC or VSC (docs/eval.md "Decisions"): its plan is the belief's best
full plan from the decision snapshot with no stop before the end of lap laps_completed + k + 1,
not a continuation planned from a snapshot advanced k laps with the period still on. Synthetic
race (test_sim)."""
import pytest
from test_sim import mid_race, synthetic_frame

from src.eval import decisions
from src.sim import SimConfig, Simulator, agent_tools
from src.sim.optimizer_bridge import as_continuation, optimize_batch

N = 40
EVENTS = {"name": "rate", "params": {"p_sc": 0.0, "p_vsc": 0.0, "d_sc": 2.0, "d_vsc": 1.5}}
NEUTRALISED = {"sc": {"sc_laps": [9, 10]}, "vsc": {"vsc_laps": [10]}}


@pytest.fixture(scope="module")
def sim():
    return Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}, events=EVENTS, n_samples=N))


def _stay(sim, snap, k, cont):
    plan, errors = sim.expand_action(snap, {"action": "stay", "laps": k}, cont)
    assert errors == []
    return plan


@pytest.mark.parametrize("status", ["sc", "vsc"])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_a_stay_under_neutralisation_is_the_constrained_optimum(sim, status, k):
    snap = mid_race(track_status=status, **NEUTRALISED[status])
    cont = as_continuation(sim, seed=3, n_samples=N, max_stops=2)
    plan = _stay(sim, snap, k, cont)
    constrained = optimize_batch(snap, sim, seed=3, n_samples=N, max_stops=2, min_first_laps=k + 1).plan
    assert plan == constrained
    assert plan.pit_laps(snap)[0] > snap.laps_completed + k
    # the advanced snapshot (a plain continuation callable) keeps the period on at the reassess
    # lap, so its continuation stops at once there: what the stay no longer does
    advanced = _stay(sim, snap, k, lambda adv: cont(adv))
    assert advanced.pit_laps(snap)[0] == snap.laps_completed + k + 1
    assert plan.pit_laps(snap)[0] != advanced.pit_laps(snap)[0]


@pytest.mark.parametrize("k", [1, 2, 4, 8, 15])
def test_a_green_stay_still_advances_the_snapshot(sim, k):
    snap = mid_race()
    cont = as_continuation(sim, seed=3, n_samples=N, max_stops=2)
    assert _stay(sim, snap, k, cont) == _stay(sim, snap, k, lambda adv: cont(adv))


@pytest.mark.parametrize("status", ["green", "sc", "vsc"])
def test_scorer_and_tools_expand_a_stay_alike_and_never_stop_inside_it(sim, status):
    snap = mid_race(track_status=status, **NEUTRALISED.get(status, {}))
    tools_cont = agent_tools._continuation(sim, N, 2)
    for k in (1, 2, 4, 8, 15):
        plan = decisions.to_plan(snap, {"decision": "STAY_OUT", "reassess_after_laps": k}, sim, 2, seed=0, n_samples=N)
        shown = sim.compare(snap, [{"action": "stay", "laps": k}], seed=agent_tools.SEED, n_samples=N,
                            continuation=tools_cont).candidates[0]
        assert shown == plan
        assert all(lap > snap.laps_completed + k for lap in plan.pit_laps(snap))
        if status != "green":
            assert plan == optimize_batch(snap, sim, seed=agent_tools.SEED, n_samples=N, max_stops=2,
                                          min_first_laps=k + 1).plan


def test_a_pit_under_neutralisation_is_unchanged(sim):
    snap = mid_race(track_status="sc", **NEUTRALISED["sc"])
    cont = as_continuation(sim, seed=3, n_samples=N, max_stops=1)
    pit = {"action": "pit", "compound": "MEDIUM"}
    assert sim.expand_action(snap, pit, cont) == sim.expand_action(snap, pit, lambda adv: cont(adv))


def test_min_first_laps_one_is_the_unconstrained_optimizer(sim):
    snap = mid_race(track_status="sc", **NEUTRALISED["sc"])
    a = optimize_batch(snap, sim, seed=3, n_samples=N, max_stops=2, return_times=True)
    b = optimize_batch(snap, sim, seed=3, n_samples=N, max_stops=2, return_times=True, min_first_laps=1)
    assert a.plan == b.plan and (a.diagnostics["times"] == b.diagnostics["times"]).all()
