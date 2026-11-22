"""The rule baselines run_to_flag and one_stop_total (src/eval/policies.py, docs/eval.md
"Rule baselines"): synthetic races, plus the committed 2021 Spain cache where marked."""
from types import SimpleNamespace

import pandas as pd
import pytest
from test_eval import HAS_CACHE, frame, items, race_frame  # noqa: F401  (fixtures)
from test_optimizer_bridge import limit_sim
from test_sim import RACE, start

from src.contracts import Snapshot, StintRecord
from src.eval import ScenarioSet, resolve, sample_race, score_scenario
from src.eval.policies import BASELINE_POLICIES, one_stop_total, run_to_flag
from src.eval.score import race_simulators, score_set
from src.sim import SimConfig, Simulator

# synthetic two-compound races (SOFT, MEDIUM), pinned without the compound proxy (tests/conftest.py)
pytestmark = pytest.mark.compound_proxy_off

RULES = ("run_to_flag", "one_stop_total")


def _scenario(snap, scenario_id="syn:X:00"):
    return SimpleNamespace(snapshot=snap, actual_stints=[], scenario_id=scenario_id, race_id="syn", driver="X",
                           lap=snap.laps_completed)


def stopped(laps_completed=20, compound="MEDIUM", tyre_age=5, first=("SOFT", 15)):
    """A driver who stopped once: `first` run, then `tyre_age` laps on `compound`."""
    return Snapshot(race=RACE, driver="X", laps_completed=laps_completed, compound=compound, tyre_age=tyre_age,
                    stints=[StintRecord(compound=first[0], laps=first[1], start_lap=1)])


def test_default_baselines_group_is_unchanged_and_the_rules_are_named_policies():
    assert list(resolve(["baselines"])) == ["actual_strategy", "fixed_one_stop", "fixed_two_stop",
                                            "greedy_crossover"]
    assert list(resolve(["baselines", *RULES])) == ["actual_strategy", "fixed_one_stop", "fixed_two_stop",
                                                    "greedy_crossover", *RULES]
    assert resolve(list(RULES)) == {"run_to_flag": run_to_flag, "one_stop_total": one_stop_total}
    with pytest.raises(KeyError, match="run_to_flag"):
        resolve(["no_such_policy"])


@pytest.mark.parametrize("margin", [None, 3])
def test_already_stopped_inside_every_rule_runs_to_the_flag(margin):
    """Two compounds used and the set inside its limit at the flag: no further stop."""
    sim = limit_sim(margin)   # R8 limit 20 + 3 on both compounds
    snap = stopped(laps_completed=30, tyre_age=2, first=("SOFT", 28))   # MEDIUM at 2, 20 left: ends at 22
    for policy in (run_to_flag, one_stop_total):
        d = policy(_scenario(snap), sim, 2)
        assert d.plan.as_pairs() == [["MEDIUM", 20]] and sim.validate(snap, d.plan) == []
    assert run_to_flag(_scenario(snap), sim, 2).meta == {"forced_stops": 0}
    # fixed_one_stop still makes one more stop here, the case these baselines were added for
    assert BASELINE_POLICIES["fixed_one_stop"](_scenario(snap), sim, 2).plan.num_stops == 1


def test_already_stopped_past_the_limit_takes_one_forced_stop_as_late_as_allowed():
    """MEDIUM at age 5 with 30 laps left would end at 35, past 23: one stop, on the last lap
    the set may run (age 23, 18 laps), onto SOFT, the softest compound that keeps it legal."""
    sim = limit_sim(3)
    snap = stopped()
    d = run_to_flag(_scenario(snap), sim, 2)
    assert d.plan.as_pairs() == [["MEDIUM", 18], ["SOFT", 12]] and d.meta == {"forced_stops": 1}
    assert sim.validate(snap, d.plan) == []
    assert one_stop_total(_scenario(snap), sim, 2).plan == d.plan   # stopped: one_stop_total is run_to_flag
    # R8 off: the same snapshot runs to the flag
    assert run_to_flag(_scenario(snap), limit_sim(None), 2).plan.as_pairs() == [["MEDIUM", 30]]


def test_two_compound_rule_forces_the_latest_stop_onto_the_softest_new_compound():
    """At the start on SOFT the flag is illegal (one compound, R3); the latest legal stop leaves
    one minimum stint (8 laps), on MEDIUM, the only other compound."""
    off = limit_sim(None)
    d = run_to_flag(_scenario(start()), off, 2)
    assert d.plan.as_pairs() == [["SOFT", 42], ["MEDIUM", 8]] and d.meta == {"forced_stops": 1}
    assert off.validate(start(), d.plan) == []
    # under R8 (23 on each compound) 50 laps need two stops: the first at age 23, the second
    # as late as the middle stint's limit and the last stint's minimum allow, SOFT first
    on = limit_sim(3)
    two = run_to_flag(_scenario(start()), on, 2)
    assert two.plan.as_pairs() == [["SOFT", 23], ["SOFT", 19], ["MEDIUM", 8]] and two.meta == {"forced_stops": 2}
    assert on.validate(start(), two.plan) == []
    # with one stop allowed no plan is legal: the flag plan goes to the world, an invalid row
    one = run_to_flag(_scenario(start()), on, 1)
    assert one.plan.as_pairs() == [["SOFT", 50]] and one.meta == {"forced_stops": None}
    # (a world without R8 has a one-stop reference; the flag plan breaks R3 in any world)
    (row,) = score_scenario(_scenario(start()), off, {"run_to_flag": run_to_flag}, belief=on, n_samples=1,
                            max_stops=1)
    assert row["status"] == "invalid" and "two different compounds" in row["reason"]


def test_not_yet_stopped_is_fixed_one_stop_through_the_wrapper():
    """No completed stint: one_stop_total is fixed_one_stop's decision, its limit-aware move
    included (SOFT at 15 may run 15 more laps under a 30-lap limit, so the stop moves from
    lap 20 to 15)."""
    fresh = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=15)
    for sim in (limit_sim(None), limit_sim(10)):
        for snap in (start(), fresh):
            got = one_stop_total(_scenario(snap), sim, 2)
            want = BASELINE_POLICIES["fixed_one_stop"](_scenario(snap), sim, 2)
            assert got.plan == want.plan and got.meta == want.meta and got.plan.num_stops == 1
    assert one_stop_total(_scenario(fresh), limit_sim(10), 2).plan.as_pairs() == [["SOFT", 15], ["MEDIUM", 25]]
    # run_to_flag on the same snapshot stops once, as late as allowed (age 30 is the limit)
    assert run_to_flag(_scenario(fresh), limit_sim(10), 2).plan.as_pairs() == [["SOFT", 15], ["MEDIUM", 25]]
    assert run_to_flag(_scenario(fresh), limit_sim(None), 2).plan.as_pairs() == [["SOFT", 32], ["MEDIUM", 8]]


def test_synthetic_legal_deterministic_nonnegative_regret(items):
    """On the synthetic race of test_eval, both plan inside the reference's search space (at
    most max_stops stops), so no row may beat the reference; runs repeat row for row."""
    frame = race_frame()
    world = Simulator(frame, SimConfig(degradation={"name": "linear"}))
    belief = Simulator(frame, SimConfig(degradation={"name": "linear"}, exclude_snapshot_driver=True))
    pol = resolve(list(RULES))
    runs = [pd.DataFrame([r for s in items for r in score_scenario(s, world, pol, belief=belief, n_samples=2)])
            for _ in range(2)]
    a, b = (r.drop(columns="latency_s") for r in runs)
    assert a.equals(b)
    assert set(a["status"]) <= {"ok", "unavailable"}
    ok = a[a.status == "ok"]
    assert len(ok) and (ok["regret_s"] >= 0).all()
    assert (ok["num_stops"] <= 2).all()
    stopped_ids = {s.scenario_id for s in items if s.snapshot.stints}
    flag = ok[(ok.policy == "run_to_flag") & ok.scenario_id.isin(stopped_ids)]
    assert len(flag) and (flag["num_stops"] == 0).all()   # two compounds used, no limit: to the flag
    fresh = ok[(ok.policy == "one_stop_total") & ~ok.scenario_id.isin(stopped_ids)]
    assert len(fresh) and (fresh["num_stops"] == 1).all()


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_spain_legal_deterministic_branches_and_nonnegative_regret():
    """2021 Spain (calibration race): every row is scored (legal), the two branches of
    one_stop_total are fixed_one_stop's and run_to_flag's decisions, and no regret is negative."""
    from src.sim import jobs
    from src.sim.compute import load_profile

    profile = load_profile("local")
    data = jobs.load_race("2021_04_spain", profile)
    found, _ = sample_race(data.frame, data.race, drivers=["HAM", "VER", "BOT"], max_per_race=8)
    s = ScenarioSet(version="t", items=found)
    pol = resolve([*RULES, "fixed_one_stop"])
    sims = race_simulators(profile)
    first = score_set(s, sims, pol, n_samples=1)
    second = score_set(s, race_simulators(profile), pol, n_samples=1)
    assert first.drop(columns="latency_s").equals(second.drop(columns="latency_s"))
    rules = first[first.policy.isin(RULES)]
    assert set(rules["status"]) == {"ok"}
    assert (rules["regret_s"] >= 0).all() and (rules["num_stops"] <= 2).all()
    plans = first.set_index(["scenario_id", "policy"])["plan"]
    stopped_ids = {x.scenario_id for x in found if x.snapshot.stints}
    assert stopped_ids and set(s.scenario_id for s in found) - stopped_ids   # both branches are covered
    for x in found:
        branch = "run_to_flag" if x.scenario_id in stopped_ids else "fixed_one_stop"
        assert plans[x.scenario_id, "one_stop_total"] == plans[x.scenario_id, branch]
    belief = sims("2021_04_spain", "belief")
    for x in found:   # the plans pass every rule on the belief too, and the branches stop as stated
        d = run_to_flag(x, belief, 2)
        assert belief.validate(x.snapshot, d.plan) == []
        if x.scenario_id not in stopped_ids:
            assert d.meta["forced_stops"] >= 1   # one compound so far: the flag is illegal (R3)
