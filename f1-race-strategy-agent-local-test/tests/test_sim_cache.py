"""Checks against the committed 2021 Spanish GP cache. Offline, no model."""
import json
import os

import pytest

from src.contracts import RaceParams
from src.sim import Simulator, calibration
from src.sim.data import CACHE_DIR, WEATHER_COLUMNS, lap_frame, load_session, session_weather

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(CACHE_DIR, "2021")), reason="no cache")


@pytest.fixture(scope="module")
def session():
    return load_session(2021, "Spain")


@pytest.fixture(scope="module")
def frame(session):
    return lap_frame(session.laps)


def test_baseline_fit_and_ground_truth_are_reproduced(frame):
    from src import tools
    gt = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.json")))
    res = tools.simulate_strategy(gt["optimal_strategy"])
    assert res["total_time_s"] == pytest.approx(gt["optimal_total_time_s"], abs=0.05)
    soft = Simulator(frame).degradation.coefs["SOFT"]
    assert soft == pytest.approx((84.1554, -0.00576), abs=1e-3)


def test_tool_errors_keep_the_original_contract():
    from src import tools
    assert "sum to 65" in tools.simulate_strategy([["SOFT", 28], ["MEDIUM", 37]])["error"]
    assert "Unknown compound" in tools.simulate_strategy([["SOFT", 28], ["HARD", 38]])["error"]


def test_replay_of_actual_strategy(session, frame):
    race = RaceParams(race_id="2021_ESP", total_laps=int(session.total_laps), pit_loss_s=22.0)
    table, summary, by_kind = calibration.run(Simulator(frame), frame, race)
    assert summary["n_drivers"] == 19                       # one retirement excluded
    assert table.loc["HAM", "stops"] == 2
    assert table.loc["HAM", "real_s"] == pytest.approx(5587.8, abs=0.1)
    assert {"lap1", "sc", "in_lap", "out_lap", "clean"} <= set(by_kind.index)


def test_weather_is_empty_by_default_and_loads_offline(session):
    assert list(session_weather(session).columns) == WEATHER_COLUMNS
    assert session_weather(session).empty
    assert "team" in lap_frame(session.laps).columns
    full = load_session(2021, "Spain", weather=True, messages=True)
    w = session_weather(full)
    if w.empty:
        pytest.skip("cache lacks weather")
    assert list(w.columns) == WEATHER_COLUMNS
    assert w["rainfall"].dtype == bool
    assert w["time_s"].is_monotonic_increasing
