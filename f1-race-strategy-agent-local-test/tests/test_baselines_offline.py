import json
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src import tools
from src.baselines import Plan, Stint, actual_strategy, candidate_plans, optimize
from src.baselines.__main__ import comparison
from src.baselines.legacy import load_legacy_context
from src.sim.degradation import LinearDegradation


@pytest.fixture(scope="module", autouse=True)
def forbid_network():
    attempted = []

    def forbidden(connection, address):
        attempted.append(address)
        raise AssertionError("Offline verification attempted a network connection.")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", forbidden)
        yield
    assert attempted == []


@pytest.fixture(scope="module")
def context():
    return load_legacy_context()


def test_legacy_adapter_preserves_stint_rounding_with_extracted_simulator(
    context,
    monkeypatch,
):
    state = replace(context.snapshot, tyre_age=0)
    degradation = {c: (base, slope) for c, base, slope in context.simulator.degradation}
    monkeypatch.setattr(
        tools._get_simulator(), "degradation", LinearDegradation(degradation)
    )
    for plan in candidate_plans(state):
        expected = tools.simulate_strategy(plan.to_list())
        assert "error" not in expected
        legacy_time = round(
            sum(expected["per_stint_seconds"]) + expected["pit_loss_total_s"], 1
        )
        assert context.simulator(state, plan, 0) == legacy_time


def test_new_optimizer_reproduces_frozen_ground_truth(context):
    state = replace(context.snapshot, tyre_age=0)
    truth = json.loads(
        (Path(__file__).parents[1] / "data" / "ground_truth.json").read_text()
    )
    result = optimize(state, context.simulator)
    assert result.evaluation.mean_time_s == truth["optimal_total_time_s"]


@pytest.mark.parametrize("lap", [0, 15, 28, 30, 42, 60, 65, 66])
def test_cached_data_workflow_at_start_mid_race_pit_boundaries_and_finish(lap):
    context = load_legacy_context(completed_laps=lap)
    report = comparison(context, seeds=(0, 1, 2), max_stops=2)
    assert report["snapshot"]["completed_laps"] == lap
    for result in report["baselines"].values():
        if result["status"] == "ok":
            assert result["regret_s"] >= 0
            assert len(set(result["seed_times_s"])) == 1
    replay = report["baselines"]["actual_strategy"]
    assert replay["status"] == "ok"
    assert sum(n for c, n in replay["stint_plan"]) == 66 - lap


def test_legacy_simulator_charges_only_future_pits_and_keeps_old_tyre_age(context):
    state = replace(
        context.snapshot,
        completed_laps=60,
        current_stint_laps=18,
        tyre_age=21,
        current_compound="MEDIUM",
        used_compounds=("SOFT", "MEDIUM"),
    )
    plan = Plan((Stint("MEDIUM", 6),))
    expected = round(
        sum(context.simulator.pace("MEDIUM", age) for age in range(22, 28)), 1
    )
    assert context.simulator(state, plan, 0) == expected


def test_actual_replay_retains_recorded_compounds_even_when_same_compound_is_refitted(
    context,
):
    plan = actual_strategy(context.snapshot, context.actual_stints)
    assert plan.stints == context.actual_stints


def test_cli_emits_parseable_json_from_a_fresh_offline_process():
    script = """import runpy, socket, sys
def forbidden(*args):
    raise AssertionError('Unexpected network request')
socket.socket.connect = forbidden
sys.argv = ['src.baselines', '--lap', '30', '--seeds', '0', '1', '2']
runpy.run_module('src.baselines', run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["snapshot"]["completed_laps"] == 30
    assert report["optimizer"]["seeds"] == [0, 1, 2]
    assert set(report["baselines"]) == {
        "actual_strategy",
        "fixed_one_stop",
        "fixed_two_stop",
        "greedy_crossover",
    }
    assert "Unexpected network request" not in completed.stderr


def test_cli_invalid_lap_fails_with_a_clear_error():
    result = subprocess.run(
        [sys.executable, "-m", "src.baselines", "--lap", "-1"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "snapshot lap must be" in result.stderr
    assert result.stdout == ""
