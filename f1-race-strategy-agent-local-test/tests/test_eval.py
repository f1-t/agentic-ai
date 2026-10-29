"""The eval harness (src/eval): scenario sampling and freezing, regret scoring, result tables.
Synthetic race data, plus the committed Spain cache where marked. No model is called."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from test_sim import TRUE

from src.baselines.domain import PolicyUnavailable
from src.contracts import RaceParams
from src.eval import Decision, ScenarioSet, recorded_policy, resolve, sample_race, score_scenario, summarize
from src.eval.policies import agent_policy
from src.eval.scenarios import eligible_drivers
from src.eval.score import score_set, to_markdown
from src.sim import SimConfig, Simulator
from src.sim.data import CACHE_DIR

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))
SCENARIOS_V1 = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios", "v1.json")
RACE = RaceParams(race_id="2099_01_syn", total_laps=40, pit_loss_s=20.0, min_stint_laps=8,
                  compounds=["SOFT", "MEDIUM"])
# driver -> stints (compound, laps); laps of the race the driver drove
STRATEGIES = {
    "AAA": [("SOFT", 14), ("MEDIUM", 26)],
    "BBB": [("MEDIUM", 22), ("SOFT", 18)],
    "CCC": [("SOFT", 12), ("MEDIUM", 27)],   # one lap down
    "DNF": [("SOFT", 9)],
}
SC_LAPS = {20, 21}


def race_frame(gap_driver=None):
    rows = []
    for pos, (driver, stints) in enumerate(STRATEGIES.items(), start=1):
        lap, t = 0, 0.0
        for stint, (comp, n) in enumerate(stints, start=1):
            base, slope = TRUE[comp]
            for age in range(1, n + 1):
                lap += 1
                if driver == gap_driver and lap == 5:
                    continue
                lt = base + slope * age
                rows.append({"driver": driver, "lap": lap, "stint": stint, "compound": comp, "tyre_age": age,
                             "lap_time_s": lt, "track_status": "4" if lap in SC_LAPS else "1",
                             "start_s": t, "end_s": t + lt, "position": pos, "clean": lap not in SC_LAPS,
                             "pit_in": age == n, "pit_out": age == 1 and stint > 1})
                t += lt
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame():
    return race_frame()


@pytest.fixture(scope="module")
def sim(frame):
    return Simulator(frame, SimConfig(degradation={"name": "linear"}))


@pytest.fixture(scope="module")
def items(frame):
    return sample_race(frame, RACE)[0]


# --- scenario sampling ---------------------------------------------------------
def test_eligible_drivers_drop_dnfs_and_gaps():
    assert eligible_drivers(race_frame(), RACE.total_laps) == ["AAA", "BBB", "CCC"]
    assert eligible_drivers(race_frame(), RACE.total_laps, max_laps_down=0) == ["AAA", "BBB"]
    assert eligible_drivers(race_frame(gap_driver="AAA"), RACE.total_laps) == ["BBB", "CCC"]


def test_sampled_laps_leave_a_stint_and_include_sc_neighbours(items):
    laps = {s.lap for s in items if s.driver == "AAA"}
    assert {0, 5, 10, 15, 19, 22} <= laps                 # every 5th, plus either side of the SC
    assert max(laps) <= RACE.total_laps - RACE.min_stint_laps
    assert max(s.lap for s in items if s.driver == "CCC") <= 39 - RACE.min_stint_laps


def test_scenarios_carry_snapshot_and_actual_stints(items):
    s = next(s for s in items if s.scenario_id == "2099_01_syn:AAA:15")
    assert (s.driver, s.lap, s.snapshot.laps_completed) == ("AAA", 15, 15)
    assert s.snapshot.compound == "MEDIUM" and [x.compound for x in s.snapshot.stints] == ["SOFT"]
    assert s.actual_stints == [("SOFT", 14), ("MEDIUM", 26)]


def test_subsample_is_seeded_and_ordered(frame):
    a, _ = sample_race(frame, RACE, max_per_race=7, seed=3)
    b, _ = sample_race(frame, RACE, max_per_race=7, seed=3)
    c, _ = sample_race(frame, RACE, max_per_race=7, seed=4)
    ids = [s.scenario_id for s in a]
    assert len(a) == 7 and ids == [s.scenario_id for s in b] and ids != [s.scenario_id for s in c]
    assert [(s.driver, s.lap) for s in a] == sorted((s.driver, s.lap) for s in a)


def test_builder_failures_are_recorded_not_raised(frame):
    def builder(f, race, driver, k):
        raise ValueError("no row")
    items, skipped = sample_race(frame, RACE, builder=builder, drivers=["AAA"])
    assert items == [] and skipped and skipped[0]["reason"] == "no row"


def test_scenario_set_is_frozen_by_its_digest(items, tmp_path):
    path = ScenarioSet(version="t1", items=items).save(tmp_path / "t1.json")
    loaded = ScenarioSet.load(path)
    assert loaded.digest == loaded.compute_digest() and len(loaded.items) == len(items)
    data = json.loads(path.read_text())
    data["items"][0]["lap"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen"):
        ScenarioSet.load(path)


def test_chunks_cover_the_set_once(items):
    s = ScenarioSet(version="t", items=items)
    got = [x.scenario_id for i in range(5) for x in s.chunk(i, 10)]
    assert got == [x.scenario_id for x in items]


# --- regret scoring ------------------------------------------------------------
def _fixed(plan):
    return lambda scenario, sim, max_stops: Decision(plan)


def test_the_optimizer_plan_has_zero_regret(items, sim):
    scenario = items[0]
    (row,) = score_scenario(scenario, sim, {"probe": _fixed([["SOFT", 1], ["MEDIUM", 39]])})
    opt = json.loads(row["opt_plan"])
    (best,) = score_scenario(scenario, sim, {"opt": _fixed(opt)})
    assert best["status"] == "ok" and best["regret_s"] == 0.0 and best["stop_agree"]
    assert best["first_pit_error"] == 0 and best["time_s"] == pytest.approx(best["opt_time_s"])


def test_regret_is_the_gap_to_the_optimizer_under_shared_draws(items, sim):
    rows = score_scenario(items[0], sim, resolve(["baselines"]))
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok
    for r in ok:
        assert r["regret_s"] == pytest.approx(r["time_s"] - r["opt_time_s"], abs=1e-6)
        assert r["regret_s"] >= 0
        assert r["regret_pct"] == pytest.approx(100 * r["regret_s"] / r["opt_time_s"])


def test_statuses(items, sim):
    def unavailable(scenario, sim, max_stops):
        raise PolicyUnavailable("not here")

    pols = {
        "broken_rule": _fixed([["SOFT", 39], ["MEDIUM", 1]]),         # last stint below the minimum
        "wrong_length": _fixed([["SOFT", 10], ["MEDIUM", 10]]),
        "no_plan": _fixed(None),
        "garbage": _fixed("pit whenever"),
        "unavailable": unavailable,
        "api_down": lambda scenario, sim, m: Decision(None, error="TimeoutError"),
        "bare_plan": lambda scenario, sim, m: [["SOFT", 20], ["MEDIUM", 20]],
    }
    rows = {r["policy"]: r for r in score_scenario(items[0], sim, pols)}
    assert {p: rows[p]["status"] for p in pols} == {
        "broken_rule": "invalid", "wrong_length": "invalid", "no_plan": "invalid", "garbage": "invalid",
        "unavailable": "unavailable", "api_down": "error", "bare_plan": "ok"}
    assert rows["no_plan"]["reason"] == "no plan submitted"
    assert rows["bare_plan"]["first_pit_lap"] == 20


def test_first_pit_error_and_stop_agreement(items, sim):
    (row,) = score_scenario(items[0], sim, {"p": _fixed([["SOFT", 20], ["MEDIUM", 20]])})
    opt = json.loads(row["opt_plan"])
    assert row["stop_agree"] == (len(opt) == 2)
    assert row["first_pit_error"] == abs(20 - row["opt_first_pit_lap"])


def test_score_set_stamps_versions(items, sim):
    s = ScenarioSet(version="t2", items=items[:3])
    s.digest = s.compute_digest()
    rows = score_set(s, lambda race_id: sim, {"p": _fixed(None)})
    assert len(rows) == 3
    assert set(rows["scenario_set"]) == {"t2"} and set(rows["scenario_digest"]) == {s.digest[:12]}
    assert rows["config_hash"].notna().all() and rows["sim_code_version"].notna().all()


# --- tables ----------------------------------------------------------------------
def test_summarize_counts_and_rates():
    rows = pd.DataFrame([
        {"race_id": "r1", "policy": "a", "status": "ok", "regret_s": 1.0, "regret_pct": 0.1,
         "stop_agree": True, "first_pit_error": 2, "latency_s": 1.0},
        {"race_id": "r1", "policy": "a", "status": "ok", "regret_s": 3.0, "regret_pct": 0.3,
         "stop_agree": False, "first_pit_error": np.nan, "latency_s": 3.0},
        {"race_id": "r2", "policy": "a", "status": "invalid", "latency_s": 2.0},
        {"race_id": "r2", "policy": "a", "status": "unavailable"},
        {"race_id": "r2", "policy": "a", "status": "error", "latency_s": 2.0},
    ])
    table = summarize(rows)
    overall = table[(table.race_id == "ALL") & (table.policy == "a")].iloc[0]
    assert (overall.scenarios, overall.scored, overall.unavailable, overall.errors) == (5, 2, 1, 1)
    assert overall.invalid_rate == pytest.approx(1 / 3)      # invalid / (ok + invalid)
    assert overall.mean_regret_s == 2.0 and overall.stop_agreement == 0.5
    assert overall.mean_first_pit_error == 2.0 and overall.mean_latency_s == 2.0
    r2 = table[(table.race_id == "r2")].iloc[0]
    assert r2.scored == 0 and np.isnan(r2.mean_regret_s) and r2.invalid_rate == 1.0
    md = to_markdown(table)
    assert md.splitlines()[0].startswith("| race_id | policy |") and "| ALL | a |" in md


# --- policies --------------------------------------------------------------------
def test_recorded_policy_replays_saved_decisions(items, sim, tmp_path):
    target = items[0].scenario_id
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({"name": "rec", "decisions": {target: {
        "stint_plan": [["SOFT", 20], ["MEDIUM", 20]], "tool_calls": 7}}}))
    name, policy = recorded_policy(path)
    assert name == "rec"
    rows = {r["scenario_id"]: r for i in items[:2] for r in score_scenario(i, sim, {name: policy})}
    assert rows[target]["status"] == "ok" and rows[target]["tool_calls"] == 7
    assert rows[items[1].scenario_id]["status"] == "unavailable"


def test_resolve_expands_baselines_and_rejects_unknown_names():
    assert sorted(resolve(["baselines"])) == ["actual_strategy", "fixed_one_stop", "fixed_two_stop",
                                              "greedy_crossover"]
    assert list(resolve(["fixed_one_stop", "agent"])) == ["fixed_one_stop", "agent"]
    with pytest.raises(KeyError):
        resolve(["oracle"])


def test_agent_is_unavailable_mid_race_without_calling_a_model(items, sim):
    mid = next(s for s in items if s.lap > 0)
    with pytest.raises(PolicyUnavailable, match="mid-race"):
        agent_policy(mid, sim, 2)


# --- committed Spain cache -------------------------------------------------------
@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_committed_v1_set_loads_and_matches_its_digest():
    s = ScenarioSet.load(SCENARIOS_V1)
    assert s.version == "v1" and len(s.items) > 200
    assert len({x.scenario_id for x in s.items}) == len(s.items)
    assert {x.race_id for x in s.items} == {"2021_04_spain"}


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_spain_baselines_score_end_to_end(tmp_path):
    from src.eval.score import race_simulators
    from src.sim import jobs
    from src.sim.compute import load_profile

    profile = load_profile("local")
    data = jobs.load_race("2021_04_spain", profile)
    items, _ = sample_race(data.frame, data.race, drivers=["HAM", "VER"], max_per_race=6)
    s = ScenarioSet(version="t", items=items)
    rows = score_set(s, race_simulators(profile), resolve(["baselines"]), n_samples=1)
    assert set(rows["status"]) <= {"ok", "unavailable"}
    assert (rows.loc[rows.status == "ok", "regret_s"] >= 0).all()
    assert (summarize(rows).query("race_id == 'ALL'")["scored"] > 0).any()


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_eval_chunk_job_runs_through_the_installed_runner(tmp_path):
    from src.eval.score import install_job_runner
    from src.sim import jobs
    from src.sim.compute import load_profile

    profile = load_profile("local")
    full = ScenarioSet.load(SCENARIOS_V1)
    small = ScenarioSet(version="t", items=full.items[:3])
    path = small.save(tmp_path / "t.json")
    try:
        assert install_job_runner(path, ["fixed_one_stop"]) == 3
        assert jobs.run_task(jobs.JOBS["eval_chunk"], "chunk_0000", profile, tmp_path) == jobs.DONE
        out = pd.read_parquet(tmp_path / "eval" / "chunk_0000.parquet")
        assert len(out) == 3 and set(out["policy"]) == {"fixed_one_stop"}
    finally:
        jobs.set_eval_runner(None)
