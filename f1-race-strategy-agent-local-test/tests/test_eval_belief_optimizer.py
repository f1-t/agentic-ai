"""The belief_optimizer policy (src/eval/policies.py): the optimizer on the belief, no model."""
import pytest
from test_eval import HAS_CACHE, frame, items, race_frame, sim  # noqa: F401  (fixtures)

from src.eval import ScenarioSet, resolve, sample_race, score_scenario
from src.eval.score import race_simulators, score_set
from src.sim import SimConfig, Simulator


def test_default_baselines_group_is_unchanged():
    assert list(resolve(["baselines"])) == ["actual_strategy", "fixed_one_stop", "fixed_two_stop",
                                            "greedy_crossover"]
    assert list(resolve(["baselines", "belief_optimizer"])) == [
        "actual_strategy", "fixed_one_stop", "fixed_two_stop", "greedy_crossover", "belief_optimizer"]


def test_synthetic_belief_equal_to_world_has_zero_regret(items, sim):
    pol = resolve(["belief_optimizer"])
    seen = 0
    for scenario in items[:6]:
        (row,) = score_scenario(scenario, sim, pol, belief=sim, n_samples=1)
        if row["status"] == "ok":
            seen += 1
            assert row["regret_s"] == pytest.approx(0.0, abs=1e-6)
            assert row["policy"] == "belief_optimizer"
    assert seen


def test_synthetic_legal_deterministic_nonnegative_regret(items):
    frame = race_frame()
    world = Simulator(frame, SimConfig(degradation={"name": "linear"}))
    belief = Simulator(frame, SimConfig(degradation={"name": "linear"}, exclude_snapshot_driver=True))
    pol = resolve(["belief_optimizer"])
    runs = [[score_scenario(s, world, pol, belief=belief, n_samples=2)[0] for s in items[:6]] for _ in range(2)]
    import pandas as pd
    strip = [pd.DataFrame(run).drop(columns="latency_s") for run in runs]
    assert strip[0].equals(strip[1])
    ok = [r for r in runs[0] if r["status"] == "ok"]
    assert ok and all(r["regret_s"] >= 0 for r in ok)
    assert all(r["num_stops"] is not None and r["plan"] and r["first_pit_lap"] is not None or r["num_stops"] == 0
               for r in ok)
    assert all(r["reason"] for r in runs[0] if r["status"] == "invalid")


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_spain_belief_optimizer_legal_deterministic_nonnegative():
    from src.sim import jobs
    from src.sim.compute import load_profile

    profile = load_profile("local")
    data = jobs.load_race("2021_04_spain", profile)
    found, _ = sample_race(data.frame, data.race, drivers=["HAM", "VER"], max_per_race=4)
    s = ScenarioSet(version="t", items=found)
    pol = resolve(["belief_optimizer"])
    first = score_set(s, race_simulators(profile), pol, n_samples=1)
    second = score_set(s, race_simulators(profile), pol, n_samples=1)
    assert first.drop(columns="latency_s").equals(second.drop(columns="latency_s"))
    assert set(first["status"]) <= {"ok", "invalid"}
    ok = first[first.status == "ok"]
    assert len(ok) and (ok["regret_s"] >= 0).all()
    assert not ({"decision", "p_pit"} & set(first.columns))   # agent-only columns stay absent
    assert {"stint_limit_rank", "stint_limit_margin_laps", "belief_window"} <= set(first.columns)
    assert "reason" not in first or first.loc[first.status == "invalid", "reason"].notna().all()
