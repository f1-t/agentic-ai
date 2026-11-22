import json

import pytest

from src.baselines.__main__ import main as baselines_main
from src.baselines.runner import format_table, load_snapshot, run_pipeline
from src.baselines.runner import main as runner_main


def test_load_snapshot_at_start():
    contract_snap, actual_stints, view = load_snapshot(
        race_id="2021_04_spain", driver="HAM", lap=0
    )
    assert contract_snap.driver == "HAM"
    assert contract_snap.laps_completed == 0
    assert contract_snap.compound == "SOFT"
    assert len(actual_stints) == 3
    assert view.lap == 0


def test_load_snapshot_midrace():
    contract_snap, actual_stints, view = load_snapshot(
        race_id="2021_04_spain", driver="HAM", lap=27
    )
    assert contract_snap.driver == "HAM"
    assert contract_snap.laps_completed == 27
    assert contract_snap.compound == "SOFT"
    assert contract_snap.tyre_age == 31
    assert len(actual_stints) == 3
    assert view.lap == 27


# the R8 wrapper's moves on 2021 Spain, pinned without the compound proxy: with it on, fixed_one_stop
# takes the proxied HARD, whose practice cap (18 + 3) no one-stop from these snapshots fits
@pytest.mark.compound_proxy_off
def test_run_pipeline_offline_lap0():
    res = run_pipeline(
        race_id="2021_04_spain",
        driver="HAM",
        lap=0,
        run_llm_agent=False,
        max_stops=2,
        seed=0,
    )
    assert res["scenario"] == "2021_04_spain:HAM:00"
    assert res["optimizer_reference"]["plan"] is not None
    assert res["optimizer_reference"]["time_s"] > 5000.0
    rows = {r["policy"]: r for r in res["results"]}
    assert "actual_strategy" in rows
    assert "fixed_one_stop" in rows
    assert "fixed_two_stop" in rows
    assert "greedy_crossover" in rows
    assert rows["actual_strategy"]["status"] == "ok"
    assert rows["fixed_two_stop"]["status"] == "ok"
    # HAM starts on a 4-lap-old SOFT: S33 > M33 ends it at 37. R8 (retired, docs/rules.md) moved
    # the stop to lap 31 with the whole race's limit; now the wrapper moves it inside the belief's
    # as-of tyre-life estimate (about 29 laps for SOFT at lap 0, from earlier races)
    assert rows["fixed_one_stop"]["status"] == "ok"
    assert json.loads(rows["fixed_one_stop"]["plan"]) == [["SOFT", 25], ["MEDIUM", 41]]
    assert rows["actual_strategy"]["regret_s"] >= 0.0


@pytest.mark.compound_proxy_off
def test_run_pipeline_offline_midrace():
    res = run_pipeline(
        race_id="2021_04_spain",
        driver="HAM",
        lap=27,
        run_llm_agent=False,
        max_stops=2,
        seed=0,
    )
    assert res["scenario"] == "2021_04_spain:HAM:27"
    assert res["optimizer_reference"]["plan"] is not None
    rows = {r["policy"]: r for r in res["results"]}
    assert rows["actual_strategy"]["status"] == "ok"
    # SOFT at age 31. Under R8 (retired, docs/rules.md) the one stop was pulled to age 35 and the
    # two stop was invalid; now the set is past the belief's as-of tyre-life estimate, so the one
    # stop pits at once, and no plan is invalid for tyre age
    assert rows["fixed_one_stop"]["status"] == "ok"
    assert json.loads(rows["fixed_one_stop"]["plan"]) == [["SOFT", 1], ["MEDIUM", 38]]
    assert rows["fixed_two_stop"]["status"] == "ok"
    assert rows["fixed_one_stop"]["regret_s"] > 0


def test_run_pipeline_agent_row_carries_submitted_in_grace(monkeypatch):
    from src.baselines import runner

    rec = {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6, "rationale": "r", "key_risks": []}
    trajectory = [{"tool": "get_race_snapshot", "phase": "evidence"},
                  {"tool": "submit_recommendation", "phase": "grace", "output": {"accepted": True}}]
    def fake_run_agent(*args, stats, **kwargs):
        stats.update(evidence_calls=1, repeat_calls=2)
        return rec, trajectory
    monkeypatch.setattr(runner, "run_agent", fake_run_agent)
    res = run_pipeline(race_id="2021_04_spain", driver="HAM", lap=27, verbose=False, max_stops=2, seed=0)
    rows = {r["policy"]: r for r in res["results"]}
    assert rows["agent"]["submitted_in_grace"] is True and rows["agent"]["tool_calls"] == 2
    assert (rows["agent"]["evidence_calls"], rows["agent"]["repeat_calls"]) == (1, 2)
    assert rows["fixed_one_stop"]["submitted_in_grace"] is None and rows["fixed_one_stop"]["evidence_calls"] is None


def test_format_table_renders():
    rows = [
        {
            "policy": "actual_strategy",
            "status": "ok",
            "num_stops": 2,
            "first_pit_lap": 28,
            "plan": [["SOFT", 28], ["MEDIUM", 14], ["MEDIUM", 24]],
            "time_s": 5587.32,
            "regret_s": 8.91,
        }
    ]
    opt_ref = {
        "plan": [["SOFT", 41], ["MEDIUM", 25]],
        "time_s": 5578.41,
        "num_stops": 1,
        "first_pit_lap": 41,
    }
    table = format_table(rows, opt_ref)
    assert "STRATEGY EVALUATION & COMPARISON SUMMARY" in table
    assert "actual_strategy" in table
    assert "optimizer (ref)" in table


def test_main_cli_offline_json(capsys):
    runner_main(["--no-agent", "--json", "--driver", "HAM", "--lap", "0"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["scenario"] == "2021_04_spain:HAM:00"
    assert len(data["results"]) == 4


def test_main_cli_from_snapshot_file(tmp_path, capsys):
    snap_file = tmp_path / "snap.json"
    snap_data = {
        "season": 2021,
        "event": "spain",
        "lap": 0,
        "total_laps": 66,
        "driver": "HAM",
        "position": 1,
        "compound": "SOFT",
        "tyre_age_laps": 4,
        "gap_ahead_s": None,
        "gap_behind_s": 1.5,
        "track_status": "GREEN",
        "rain_probability": None,
        "nearby_cars": [],
    }
    snap_file.write_text(json.dumps(snap_data))
    runner_main([str(snap_file), "--no-agent", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["snapshot"]["driver"] == "HAM"
    assert len(data["results"]) == 4


def test_baselines_main_pipeline_delegation(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["src.baselines", "--pipeline", "--driver", "HAM", "--lap", "0"],
    )
    baselines_main()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["scenario"] == "2021_04_spain:HAM:00"
    assert len(data["results"]) == 4
