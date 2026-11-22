"""docs/eval.md "Information contract": the cross-race tables a policy reads, as of the race.
Synthetic tables only; tests/test_future_blindness.py checks the committed ones."""
import math

import numpy as np
import pandas as pd
import pytest

from src.contracts import CalibrationOutput, RaceParams
from src.sim import calibrators, proxy, race_history
from src.sim.config import CompoundProxy
from src.sim.model_report import calibration_error

STEP_ROWS = [{"race_id": rid, "pair": "SOFT-MEDIUM", "thin": False, "pace_frac": f, "wear_ratio": 0.5, "life_ratio": 0.7}
             for rid, f in (("2020_03_a", 0.01), ("2021_02_b", 0.02), ("2022_05_c", 0.03))]
VAL_ROWS = [{"race_id": rid, "compound": "SOFT", "case": case, "available": True, "pace_err_s": e,
             "wear_err_s_per_lap": e / 10}
            for rid in ("2020_03_a", "2021_02_b", "2022_05_c") for case, e in
            (("practice", 1.0), ("qualifying", 0.5), ("cross_race_step", 0.25), ("blend", 0.4))]


def px():
    return CompoundProxy(per_race=STEP_ROWS, validation=VAL_ROWS * 3)


def test_the_tier5_inputs_use_the_races_before_the_race_only():
    got = proxy.as_of_config(px(), "2021_10_z")
    assert got.steps["SOFT-MEDIUM"]["pace_frac"]["races"] == ["2020_03_a", "2021_02_b"]
    assert got.steps["SOFT-MEDIUM"]["pace_frac"]["value"] == pytest.approx(0.015)
    assert got.weights["pace"] == pytest.approx({"practice": 1.0, "qualifying": 4.0, "cross_race_step": 16.0})
    assert got.sd_floor["pace"] == pytest.approx(0.4)
    # the race itself and a later one never enter; one earlier race: its step, no spread
    assert proxy.as_of_config(px(), "2021_02_b").steps["SOFT-MEDIUM"]["pace_frac"]["races"] == ["2020_03_a"]


@pytest.mark.parametrize("race_id", ["2019_01_first", "syn", ""])
def test_no_earlier_race_means_no_step_equal_weights_and_the_a_priori_floor(race_id):
    got = proxy.as_of_config(px(), race_id)
    assert got.steps == {} and got.weights == {"pace": {}, "wear": {}} and got.sd_floor == proxy.PRIOR_SD_FLOOR


HISTORY = pd.DataFrame({"race_id": ["2020_03_a", "2021_02_b", "2022_05_c"], "circuit": ["x", "y", "x"],
                        "base_delta_s": [0.2, 0.6, 9.0], "sc_lap_factor": [1.3, np.nan, 1.5],
                        "lap1_extra_s": [6.0, 8.0, 99.0], "target": [-0.05, -0.07, -0.5], "target_sd": [0.003] * 3})


def test_as_of_factors_are_the_median_of_earlier_races_else_a_priori():
    got = race_history.as_of_factor("2022_01_q", "base_delta_s", HISTORY)
    assert got["mean"] == pytest.approx(0.4) and got["races"] == ["2020_03_a", "2021_02_b"]
    assert got["sd"] == pytest.approx(max(np.std([0.2, 0.6], ddof=1), race_history.SD_FLOOR["base_delta_s"]))
    one = race_history.as_of_factor("2022_01_q", "sc_lap_factor", HISTORY)   # the NaN race does not count
    assert one["mean"] == 1.3 and one["sd"] == race_history.A_PRIORI["sc_lap_factor"][1]
    none = race_history.as_of_factor("2019_01_q", "lap1_extra_s", HISTORY)
    assert (none["mean"], none["sd"], none["races"]) == (*race_history.A_PRIORI["lap1_extra_s"], [])


def test_the_fuel_prior_falls_back_to_the_physics_prior_without_history():
    race = RaceParams(race_id="2019_01_q", total_laps=55, pit_loss_s=20.0)
    got = race_history.as_of_fuel(race, HISTORY)
    assert got["races"] == [] and got["source"].startswith("a priori") and got["mean"] == pytest.approx(-0.03 * 110 / 55)
    later = race_history.as_of_fuel(race.model_copy(update={"race_id": "2022_01_q"}), HISTORY)
    assert later["races"] == ["2020_03_a", "2021_02_b"] and -0.08 < later["mean"] < -0.04


PERIODS = pd.DataFrame({"race_id": ["2020_03_a", "2021_02_b", "2022_05_c"], "kind": ["sc", "", "sc"],
                        "start_lap": [5, np.nan, 1], "end_lap": [7, np.nan, 40], "total_laps": [50, 50, 50],
                        "wet": [False, False, False]})


def test_event_rates_shrink_earlier_races_toward_the_a_priori_prior():
    none = race_history.as_of_event_rates("2019_01_q", PERIODS)
    prior = race_history.EVENT_PRIOR
    assert (none["p_sc"], none["d_sc"], none["races"]) == (pytest.approx(prior.p_sc), pytest.approx(prior.d_sc), [])
    got = race_history.as_of_event_rates("2021_09_q", PERIODS)
    assert got["races"] == ["2020_03_a", "2021_02_b"] and got["p_sc"] != pytest.approx(prior.p_sc)


def test_calibration_error_is_read_as_of_the_race():
    table = pd.DataFrame({"window": ["to_snapshot"], "horizon_bucket": [30], "mae_offset_s": [12.0]})
    assert calibration_error("to_snapshot", 35, table) == 12.0                         # no race: as before
    assert calibration_error("to_snapshot", 35, table, race_id="2022_02_saudi_arabia") == 12.0
    for rid in ("2021_04_spain", "2019_06_monaco", "1990_01_synthetic", "syn"):
        assert math.isnan(calibration_error("to_snapshot", 35, table, race_id=rid)), rid


class _Model:
    train_races = ["2021_04_spain", "2023_07_spain"]

    def calibrate(self, ctx):
        return CalibrationOutput(factors={}, valid_until_lap=0, source="ridge")


@pytest.mark.parametrize("race_id, refused", [("2022_02_saudi_arabia", True), ("2024_07_emilia_romagna", False)])
def test_the_learned_calibrator_is_refused_for_a_race_before_its_training_races(race_id, refused, tmp_path, monkeypatch):
    cal = calibrators.TrainedCalibrator(artifacts=str(tmp_path))
    monkeypatch.setattr(cal, "artifact_for", lambda rid: tmp_path / "all.json")
    monkeypatch.setattr(cal, "model_for", lambda rid: _Model())
    monkeypatch.setattr(cal, "manifest", lambda: {"calibrator": "ridge"})

    class Ctx:
        race = RaceParams(race_id=race_id, total_laps=50, pit_loss_s=20.0)
    if refused:
        with pytest.raises(ValueError, match="not as of the race"):
            cal.calibrate(Ctx)
        forced = calibrators.TrainedCalibrator(artifacts=str(tmp_path), force_not_as_of=True)
        for name in ("artifact_for", "model_for", "manifest"):
            monkeypatch.setattr(forced, name, getattr(cal, name))
        assert forced.calibrate(Ctx).source == "trained"
    else:
        assert cal.calibrate(Ctx).source == "trained"
