"""SPEC 0014 D to J: position regret, the drawn-events warning, the model-disagreement flag,
the registered eval split, run-all with the eval runner, the agent on the snapshot loop, the
snapshot contract fields and the minimum-stint rule. Synthetic races, plus the committed
Spain cache where marked. No model is called."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_eval import RACE as SAMPLE_RACE
from test_eval import race_frame as sample_frame
from test_eval_integration import BELIEF_TRUE, frame_for, scenario
from test_sim import mid_race, synthetic_frame
from test_sim_traffic import OUR, car, frame as traffic_frame, ghost, ghost_sim, perturb_after, snap as traffic_snap

from src import grade, legacy_tools
from src.agent import run_agent
from src.contracts import NearbyCar, Plan, RaceParams, Snapshot, TyreSet, Weather, FieldState
from src.eval import Decision, ScenarioSet, resolve, summarize
from src.eval import scenarios as scen_mod
from src.eval import score
from src.eval.policies import AgentPolicy
from src.eval.score import (NoStopReferenceWarning, finishing_positions, references_agree, score_scenario,
                            score_set, summarize_agreeing)
from src.llm.base import ModelBackend, ModelTurn, ToolCall
from src.sim import SimConfig, Simulator, calibration, jobs, traffic
from src.sim.data import CACHE_DIR, gap_columns
from src.sim.engine import TRAFFIC_DRAWN_EVENTS
from src.sim.events import Scenario as Events
from src.sim.optimizer_bridge import optimize_batch
from src.snapshot import RaceSnapshot

HAS_CACHE = os.path.isdir(os.path.join(CACHE_DIR, "2021"))
ROOT = Path(__file__).resolve().parents[1]
LIN = {"name": "linear"}


class Scripted(ModelBackend):
    """Replays model turns; each turn's raw carries a usage dict."""
    name = "test"
    model_id = "scripted"

    def __init__(self, *calls, usage=(100, 20)):
        self.turns = iter(calls)
        self.usage = usage

    def run_turn(self, **kw):
        name, args = next(self.turns)
        raw = {"usage": {"input_tokens": self.usage[0], "output_tokens": self.usage[1]}} if self.usage else None
        return ModelTurn(tool_calls=[ToolCall("1", name, args)], raw=raw)


class Broken(ModelBackend):
    name = "test"

    def run_turn(self, **kw):
        raise ConnectionError("backend down")


def submit(decision="STAY_OUT", **kw):
    args = {"decision": decision, "reassess_after_laps": 3, "confidence": 0.7, "rationale": "r", "key_risks": []}
    return "submit_recommendation", {**args, **kw}


# --- D: position regret ------------------------------------------------------------------
def position_frame():
    """XXX at 80 s a lap; AAA faster, BBB slower, both finish; CCC retires after 10, DDD after 15."""
    return pd.DataFrame(car(OUR, 80.0) + car("AAA", 79.0) + car("BBB", 81.0)
                        + car("CCC", 80.5, n=10) + car("DDD", 80.2, n=15))


def test_place_is_public_and_the_calibration_uses_it():
    assert not hasattr(calibration, "_place")
    df = position_frame()
    assert traffic.place(df, OUR, 20, 1590.0)[0] == 2        # AAA finished at 1580, BBB at 1620
    assert traffic.place(df, OUR, 20, 1570.0)[0] == 1
    assert traffic.place(df, OUR, 20, 1630.0) == (3, pytest.approx(10.0))


def test_finishing_positions_rank_among_ghosts_and_order_dnfs():
    df, s = position_frame(), traffic_snap(5)     # anchor 400 s, 15 laps left
    times = np.array([1170.0, 1190.0, 1230.0, 1190.0, 1190.0, 1190.0])
    dnf = np.array([np.nan, np.nan, np.nan, 12, 16, 10])
    pos = finishing_positions(df, s, times, dnf)
    # finishers by traffic.place; a DNF is behind both finishers and every ghost that retired
    # no earlier: DDD reached 15 laps, CCC 10
    assert pos[:3].tolist() == [traffic.place(df, OUR, 20, 400 + t)[0] for t in times[:3]] == [1, 2, 3]
    assert pos[3:].tolist() == [4, 3, 5]


@pytest.fixture(scope="module")
def traffic_world():
    df = pd.DataFrame(car(OUR, 80.0) + ghost("AAA", 80.5, 3.0) + ghost("BBB", 80.4, -6.0)
                      + ghost("CCC", 79.0, 30.0))
    return df, ghost_sim(df)


def test_position_regret_is_zero_for_the_reference_plan(traffic_world):
    df, world = traffic_world
    s = traffic_snap(5)
    opt = optimize_batch(s, world, max_stops=1)
    pols = {"ref": lambda sc, sim, m: Decision(opt.plan),
            "late": lambda sc, sim, m: Decision([["SOFT", 13], ["MEDIUM", 2]])}
    rows = {r["policy"]: r for r in score_scenario(scenario(s, "syn:XXX:05"), world, pols, max_stops=1)}
    assert rows["ref"]["regret_s"] == 0.0 and rows["ref"]["position_regret"] == 0.0
    assert rows["ref"]["final_position_mean"] == rows["ref"]["opt_final_position_mean"]
    assert 1 <= rows["late"]["final_position_mean"] <= 4
    assert np.sign(rows["late"]["position_regret"]) in (0.0, 1.0)
    table = summarize(pd.DataFrame(rows.values())).set_index(["race_id", "policy"])
    assert table.loc[("syn_traffic", "ref"), "mean_position_regret"] == 0.0


def test_position_columns_are_nan_with_world_traffic_off():
    world = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    (row,) = score_scenario(scenario(mid_race()), world, resolve(["fixed_one_stop"]))
    assert np.isnan(row["position_regret"]) and np.isnan(row["final_position_mean"])
    assert np.isnan(row["opt_final_position_mean"])
    assert "mean_position_regret" not in summarize(pd.DataFrame([row]))


def warnings_of(sim, s, scenario=None):
    batch = sim.simulate_batch(s, [[["SOFT", 7], ["MEDIUM", 8]], [["SOFT", 15]]], scenario=scenario,
                               rules=["lap_sum", "known_compounds"], n_samples=3)
    return [TRAFFIC_DRAWN_EVENTS in w for w in batch.support_warnings]


@pytest.mark.parametrize("events, scenario, warned", [
    ({"name": "none"}, None, False),
    ({"name": "rate"}, None, True),
    ({"name": "fixed", "params": {"sc_laps": []}}, None, False),        # the record: all green
    ({"name": "fixed", "params": {"sc_laps": [9, 10]}}, None, True),
    ({"name": "none"}, Events(), False),                                # a replay of the record
    ({"name": "none"}, Events(vsc_laps=[12]), True),
])
def test_recorded_ghosts_under_drawn_events_warn_on_every_plan(events, scenario, warned):
    df = traffic_frame(ghost("AAA", 80.5, 3.0))
    on = Simulator(df, SimConfig(degradation=LIN, traffic={"name": "ghost"}, events=events))
    assert warnings_of(on, traffic_snap(5), scenario) == [warned, warned]
    off = Simulator(df, SimConfig(degradation=LIN, events=events))
    assert warnings_of(off, traffic_snap(5), scenario) == [False, False]


def test_recorded_sc_laps_are_the_record():
    df = traffic_frame(ghost("AAA", 80.5, 3.0))
    df.loc[df["lap"].isin([9, 10]), "track_status"] = "14"
    on = Simulator(df, SimConfig(degradation=LIN, traffic={"name": "ghost"}))
    assert warnings_of(on, traffic_snap(5), Events(sc_laps=[9, 10])) == [False, False]
    assert warnings_of(on, traffic_snap(5), Events()) == [True, True]


# --- E: model-disagreement flag (optional) ----------------------------------------------------
def test_references_agree_on_stops_and_first_pit_within_two_laps():
    s = mid_race()
    a = Plan.of([["SOFT", 10], ["MEDIUM", 30]])
    assert references_agree(a, Plan.of([["SOFT", 12], ["MEDIUM", 28]]), s)
    assert not references_agree(a, Plan.of([["SOFT", 13], ["MEDIUM", 27]]), s)
    assert not references_agree(a, Plan.of([["SOFT", 10], ["MEDIUM", 15], ["SOFT", 15]]), s)
    assert references_agree(Plan.of([["SOFT", 40]]), Plan.of([["SOFT", 40]]), s)


def test_rows_carry_ref_models_agree_and_the_agreeing_summary():
    world = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    other = Simulator(frame_for(BELIEF_TRUE), SimConfig(degradation=LIN, min_samples=6))
    pols = resolve(["fixed_one_stop"])
    same = score_scenario(scenario(mid_race()), world, pols, ref_worlds=[world])
    differ = score_scenario(scenario(mid_race(), "syn:X:11"), world, pols, ref_worlds=[other])
    assert same[0]["ref_models_agree"] is True
    alt = optimize_batch(mid_race(), other).plan
    assert differ[0]["ref_models_agree"] == references_agree(optimize_batch(mid_race(), world).plan, alt, mid_race())
    rows = pd.DataFrame(same + differ)
    agreeing = summarize_agreeing(rows).query("race_id == 'ALL'")
    assert int(agreeing["scenarios"].iloc[0]) == int(rows["ref_models_agree"].sum())
    assert summarize_agreeing(rows.drop(columns="ref_models_agree")) is None
    assert "ref_models_agree" not in score_scenario(scenario(mid_race()), world, pols)[0]


def test_score_set_stamps_the_reference_world_hashes():
    world = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    other = Simulator(synthetic_frame(), SimConfig(degradation=LIN, n_samples=2))
    rows = score_set(ScenarioSet(version="t", items=[scenario(mid_race())]), lambda r, role: world,
                     resolve(["fixed_one_stop"]), ref_sims=[lambda r, role: other])
    assert rows["ref_world_config_hashes"].iloc[0] == calibration.config_hash(other.config)


# --- F: the registered eval split --------------------------------------------------------------
@pytest.fixture
def sample_load(monkeypatch):
    data = jobs.RaceData(SAMPLE_RACE.race_id, sample_frame(), SAMPLE_RACE, None, None, SimConfig())
    monkeypatch.setattr(jobs, "load_race", lambda race_id, profile: data)


def test_without_an_eval_set_the_set_is_a_development_set(sample_load, tmp_path):
    out = scen_mod.build_set([SAMPLE_RACE.race_id], "d", eval_set=tmp_path / "missing.json")
    assert out.development and "eval_set_digest" not in out.sampling and out.items


@pytest.mark.parametrize("form", ["dict", "list"])
def test_an_eval_set_refuses_other_races_and_stamps_its_digest(sample_load, tmp_path, form):
    path = tmp_path / "eval_set.json"
    races = [SAMPLE_RACE.race_id, "2022_05_miami"]
    path.write_text(json.dumps({"races": races, "registered_by": "test"} if form == "dict" else races))
    out = scen_mod.build_set([SAMPLE_RACE.race_id], "r", eval_set=path)
    assert not out.development
    assert out.sampling["eval_set_digest"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(scen_mod.OutsideEvalSet, match="2021_04_spain"):
        scen_mod.build_set([SAMPLE_RACE.race_id, "2021_Spain"], "r", eval_set=path)


def test_default_eval_set_path_and_v1_is_a_development_set():
    assert scen_mod.EVAL_SET == ROOT / "data" / "eval_set.json"
    assert ScenarioSet.load(ROOT / "data" / "scenarios" / "v1.json").development


# --- G: run-all with the eval runner ------------------------------------------------------------
def _stub_installer(tag):
    jobs.set_eval_runner(lambda chunk, prof: pd.DataFrame({"chunk": [chunk], "tag": [tag]}),
                         spec={"installer": "test_eval_part2:_stub_installer", "tag": tag})


def test_an_eval_chunk_task_installs_the_runner_from_its_payload(tmp_path):
    from types import SimpleNamespace
    from src.sim.compute import load_profile

    spec = {"installer": "test_eval_part2:_stub_installer", "tag": "from-payload"}
    try:
        jobs.set_eval_runner(None)
        out = jobs.JOBS["eval_chunk"].run("chunk_0000", load_profile("local"), tmp_path,
                                          SimpleNamespace(eval_runner=spec))
        assert out["tag"].tolist() == ["from-payload"] and jobs._EVAL_SPEC[0] == spec
    finally:
        jobs.set_eval_runner(None)


@pytest.mark.skipif(not HAS_CACHE, reason="committed race cache not present")
def test_run_all_scores_a_two_chunk_set_with_two_workers(tmp_path, monkeypatch):
    """Spawned workers start without the parent's installed runner, so each one installs it
    from the task payload."""
    import functools
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from src.eval.__main__ import main

    monkeypatch.setattr(jobs, "ProcessPoolExecutor", functools.partial(
        ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")))
    full = ScenarioSet.load(ROOT / "data" / "scenarios" / "v1.json")
    path = ScenarioSet(version="t2", items=full.items[:2]).save(tmp_path / "t2.json")
    prof = yaml.safe_load((ROOT / "configs" / "compute" / "local.yaml").read_text())
    prof.update(name="t2")
    prof["sim"]["n_samples"] = 1
    prof["parallel"].update(n_jobs=2, eval_chunk_size=1)
    (tmp_path / "t2.yaml").write_text(yaml.safe_dump(prof))
    (tmp_path / "manifest.json").write_text(json.dumps(["chunk_0000", "chunk_0001"]))
    try:
        assert main(["jobs", "--scenarios", str(path), "--policies", "fixed_one_stop", "--",
                     "run-all", "eval_chunk", "--profile", str(tmp_path / "t2.yaml"),
                     "--manifest", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "out")]) == 0
    finally:
        jobs.set_eval_runner(None)
    parts = [pd.read_parquet(tmp_path / "out" / "eval" / f"chunk_000{i}.parquet") for i in (0, 1)]
    assert [p["scenario_id"].tolist() for p in parts] == [[full.items[0].scenario_id], [full.items[1].scenario_id]]
    assert all(set(p["policy"]) == {"fixed_one_stop"} and (p["n_samples"] == 1).all() for p in parts)


# --- H: the agent on the snapshot loop ------------------------------------------------------------
def test_run_agent_takes_a_contract_snapshot_and_reports_usage():
    stats = {}
    rec, traj = run_agent(mid_race(), backend=Scripted(("get_race_snapshot", {}), submit()), verbose=False,
                          stats=stats)
    assert rec["decision"] == "STAY_OUT" and len(traj) == 2
    assert traj[0]["output"]["lap"] == 10 and traj[0]["output"]["min_stint_laps"] == 8
    assert stats == {"steps": 2, "input_tokens": 200, "output_tokens": 40}
    stats = {}
    run_agent(mid_race(), backend=Scripted(submit(), usage=None), verbose=False, stats=stats)
    assert stats == {"steps": 1, "input_tokens": None, "output_tokens": None}


def test_agent_policy_rows_are_decisions_with_trajectories(tmp_path):
    world = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    sc = scenario(mid_race(), "syn:X:10")
    pols = {"agent": AgentPolicy(set_name="v9", max_steps=4, log_dir=tmp_path,
                                 backend=Scripted(("estimate_pit_loss", {}), submit("PIT", compound="MEDIUM"))),
            "broken": AgentPolicy(set_name="v9", max_steps=4, log_dir=tmp_path / "other", backend=Broken())}
    rows = {r["policy"]: r for r in score_scenario(sc, world, pols)}
    agent = rows["agent"]
    assert agent["status"] == "ok" and agent["decision"] == "PIT" and agent["p_pit"] == 0.7
    assert (agent["tool_calls"], agent["input_tokens"], agent["output_tokens"]) == (2, 200, 40)
    assert agent["first_pit_lap"] == 11
    log = tmp_path / "v9" / "syn_X_10.jsonl"
    assert [json.loads(x)["tool"] for x in log.read_text().splitlines()] == ["estimate_pit_loss",
                                                                              "submit_recommendation"]
    assert rows["broken"]["status"] == "error" and "backend down" in rows["broken"]["reason"]


def test_agent_policy_budget_comes_from_the_profile(tmp_path):
    from src.sim.compute import load_profile

    profile = load_profile("local")
    assert profile.agent.max_steps == 10
    small = profile.model_copy(update={"agent": profile.agent.model_copy(update={"max_steps": 1})})
    turns = Scripted(("get_race_snapshot", {}), submit())
    d = AgentPolicy(set_name="b", profile=small, log_dir=tmp_path, backend=turns)(scenario(mid_race()), None, 2)
    assert d.submission is None and d.tool_calls == 1 and d.meta["steps"] == 1   # out of steps
    assert resolve(["agent"], set_name="v1", profile=small)["agent"].set_name == "v1"


def test_grade_decision_reports_validity_and_regret():
    world = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    sims = lambda race_id, role: world   # noqa: E731
    s = mid_race()
    g = grade.grade_decision(s, {"decision": "STAY_OUT", "reassess_after_laps": 2, "confidence": 0.5},
                             sim_for=sims, n_samples=1, max_stops=2)
    assert g["valid"] and g["regret_s"] >= 0 and g["opt_time_s"] > 0 and g["decision"] == "STAY_OUT"
    bad = grade.grade_decision(s, {"decision": "PIT"}, sim_for=sims, n_samples=1, max_stops=2)
    assert not bad["valid"] and "compound" in bad["reason"]


def test_max_stops_zero_with_decisions_warns_once():
    world = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    one = ScenarioSet(version="t", items=[scenario(mid_race())])
    agentish = AgentPolicy(backend=Broken())
    with pytest.warns(NoStopReferenceWarning, match="no-stop reference") as caught:
        score_set(one, lambda r, role: world, {"agent": agentish}, max_stops=0)
    assert len([w for w in caught if w.category is NoStopReferenceWarning]) == 1
    assert not score.warn_no_stop_reference(resolve(["fixed_one_stop"]), 0)
    assert not score.warn_no_stop_reference({"agent": agentish}, 1)


def test_recorded_policies_count_as_decisions(tmp_path):
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({"name": "rec", "decisions": {}}))
    assert resolve([], [path])["rec"].agent_decisions


# --- I: the snapshot contract and the agent's view ---------------------------------------------------
def full_snapshot():
    race = RaceParams(race_id="2021_04_spain", total_laps=66, pit_loss_s=21.5, min_stint_laps=3,
                      compounds=["SOFT", "MEDIUM", "HARD"])
    return Snapshot(race=race, driver="HAM", laps_completed=20, compound="MEDIUM", tyre_age=5, position=2,
                    gap_ahead_s=1.4, gap_behind_s=3.2, track_status="sc",
                    weather=Weather(rain_probability=0.3, air_temp_c=24.0, track_temp_c=40.0),
                    field=FieldState(nearby=[NearbyCar(driver="VER", position=1, gap_s=-1.4),
                                             NearbyCar(driver="BOT", position=3, gap_s=3.2)]),
                    recent_lap_times_s=[81.2, 81.0, 80.9], context={"note": "undercut threat"},
                    sets_available=[TyreSet(compound="HARD"), TyreSet(compound="SOFT", laps_run=3)],
                    standing_start=False)


def test_contract_fields_are_optional_and_strict():
    s = Snapshot.at_start(RaceParams(race_id="x", total_laps=10, pit_loss_s=20), "A", "SOFT")
    assert (s.recent_lap_times_s, s.context, s.field.nearby, s.weather.rain_probability) == ([], {}, [], None)
    with pytest.raises(Exception):
        NearbyCar(driver="A", gap_s=1.0, colour="red")
    with pytest.raises(Exception):
        Weather(rain_probability=1.5)


def test_from_contract_and_back():
    s = full_snapshot()
    view = RaceSnapshot.from_contract(s)
    assert (view.season, view.event, view.lap, view.total_laps) == (2021, "spain", 20, 66)
    assert (view.track_status, view.rain_probability, view.min_stint_laps) == ("SC", 0.3, 3)
    assert view.available_compounds == ["HARD", "SOFT"]           # the sets, not the race list
    assert view.nearby_cars[0] == {"driver": "VER", "position": 1, "gap_s": -1.4}
    assert view.recent_laps == [{"lap": 18, "lap_time_s": 81.2}, {"lap": 19, "lap_time_s": 81.0},
                                {"lap": 20, "lap_time_s": 80.9}]
    back = view.to_contract(race=s.race)
    carried = ["driver", "laps_completed", "compound", "tyre_age", "position", "gap_ahead_s", "gap_behind_s",
               "track_status", "field", "recent_lap_times_s", "context", "standing_start"]
    assert {k: getattr(back, k) for k in carried} == {k: getattr(s, k) for k in carried}
    assert back.weather.rain_probability == 0.3 and back.sets_available is None
    assert RaceSnapshot.from_contract(back).available_compounds == ["SOFT", "MEDIUM", "HARD"]
    assert RaceSnapshot.from_contract(view.to_contract(race=s.race)) == RaceSnapshot.from_contract(back)
    assert view.to_contract().race.min_stint_laps == 3


def nearby_frame():
    df = pd.DataFrame(car(OUR, 80.0, position=2.0) + ghost("AAA", 80.5, 3.0, position=1.0)
                      + ghost("BBB", 80.4, -2.0, position=3.0))
    return df


def test_snapshot_at_fills_nearby_and_recent_laps():
    from test_sim_traffic import RACE as T_RACE

    df = gap_columns(nearby_frame())
    s = calibration.snapshot_at(df, T_RACE, OUR, 5)
    assert [(c.driver, c.position) for c in s.field.nearby] == [("AAA", 1), ("BBB", 3)]
    assert [c.gap_s for c in s.field.nearby] == pytest.approx([-3.0, 2.0])
    assert s.recent_lap_times_s == [80.0] * 5
    s0 = calibration.snapshot_at(df, T_RACE, OUR, 0)
    assert s0.field.nearby == [] and s0.recent_lap_times_s == []


def test_information_boundary_canary_for_the_agent_view():
    """The agent's view at lap k is unchanged when every car's laps after k are perturbed."""
    from test_sim_traffic import RACE as T_RACE

    df = nearby_frame()
    a = calibration.snapshot_at(gap_columns(df), T_RACE, OUR, 5)
    b = calibration.snapshot_at(gap_columns(perturb_after(df, 5, lap_times=True)), T_RACE, OUR, 5)
    assert a == b and RaceSnapshot.from_contract(a) == RaceSnapshot.from_contract(b)


def test_published_digest_survives_the_added_fields():
    v1 = ScenarioSet.load(ROOT / "data" / "scenarios" / "v1.json")
    item = v1.items[0]
    filled = item.model_copy(update={"snapshot": item.snapshot.model_copy(update={"recent_lap_times_s": [80.0]})})
    assert ScenarioSet(version="x", items=[filled]).compute_digest() != ScenarioSet(
        version="x", items=[item]).compute_digest()


# --- J: one minimum-stint rule ---------------------------------------------------------------
def test_min_stint_default_is_three_and_the_legacy_pins_ten():
    assert RaceParams(race_id="x", total_laps=10, pit_loss_s=20).min_stint_laps == 3
    assert legacy_tools.MIN_STINT_LAPS == 10
    from src.sim import calibrate
    assert calibrate.MIN_STINT_DEFAULT == 3


def test_v1_keeps_ten_inside_its_snapshots():
    v1 = ScenarioSet.load(ROOT / "data" / "scenarios" / "v1.json")
    assert {s.snapshot.race.min_stint_laps for s in v1.items} == {10}


def test_rule_r2_names_the_value():
    text = (ROOT / "docs" / "rules.md").read_text()
    r2 = " ".join(text[text.index("**R2"):text.index("**R3")].split())
    for phrase in ("RaceParams.min_stint_laps", "default 3", "per race", "legacy_tools.py", "pins 10",
                   "tools read it from the snapshot", "v1 carries 10"):
        assert phrase in r2, phrase
