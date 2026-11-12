import json

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
    # HAM starts on a 3-lap-old SOFT: half the race on it passes the R8 stint limit (docs/rules.md)
    assert rows["fixed_one_stop"]["status"] == "unavailable"
    assert rows["fixed_one_stop"]["reason"].startswith("R8 stint limit: Stint 1 (SOFT)")
    assert rows["actual_strategy"]["regret_s"] >= 0.0


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
    # SOFT at age 31: one stop puts it past the R8 stint limit (docs/rules.md)
    assert rows["fixed_one_stop"]["status"] == "unavailable"
    assert rows["fixed_one_stop"]["reason"].startswith("R8 stint limit")


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
