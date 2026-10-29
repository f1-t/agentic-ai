"""ModelReport, posterior sd and support warnings. Synthetic data only."""
import math

import numpy as np
import pandas as pd
import pytest

from src.contracts import ModelReport, RaceParams, Snapshot
from src.sim import SimConfig, Simulator

from test_sim import FUEL, RACE, TRUE, at_lap, hand_time, start, synthetic_frame, synthetic_practice, window

TAU = SimConfig().shrinkage


def noisy(frame, sd=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return frame.assign(lap_time_s=frame["lap_time_s"] + rng.normal(0.0, sd, len(frame)))


def first_stint_race(drivers=("A", "B", "C"), pit=20, total=50):
    """Every car on SOFT for laps 1..pit, then MEDIUM to the flag."""
    rows = []
    for drv in drivers:
        for lap in range(1, total + 1):
            comp, age = ("SOFT", lap) if lap <= pit else ("MEDIUM", lap - pit)
            base, wear = TRUE[comp]
            rows.append({"driver": drv, "compound": comp, "tyre_age": age, "lap": lap, "clean": True,
                         "lap_time_s": base + wear * age + FUEL * lap})
    return noisy(pd.DataFrame(rows))


# --- fit_report sd -------------------------------------------------------------
def test_sd_is_the_ols_standard_error_on_an_unpenalised_fit():
    frame = noisy(synthetic_frame())
    rep = Simulator(frame, SimConfig(degradation={"name": "linear"})).degradation.fit_report
    design, rss = {}, 0.0
    for comp in TRUE:
        d = frame[frame["compound"] == comp]
        X = np.column_stack([np.ones(len(d)), d["tyre_age"].to_numpy(dtype=float)])
        y = d["lap_time_s"].to_numpy()
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        design[comp], rss = X, rss + float(r @ r)
    s2 = rss / (len(frame) - 4)  # residual variance pooled over both lines, 4 coefficients
    for comp, X in design.items():
        se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
        assert rep["sd"]["base_s"][comp] == pytest.approx(se[0])
        assert rep["sd"]["wear_s_per_lap"][comp] == pytest.approx(se[1])
    assert rep["sd"]["fuel_s_per_lap"] is None
    assert rep["sd"]["residual_s"] == pytest.approx(0.3, rel=0.15)


def test_sd_with_no_race_rows_is_the_prior_residual_times_tau():
    practice = synthetic_practice().pipe(noisy)
    s = Simulator(synthetic_frame(fuel=FUEL), window("to_snapshot"), practice)
    prior = Simulator(synthetic_frame(fuel=FUEL), window("pre_race"), practice).fitted_for(start())
    sigma = prior.degradation.fit_report["sd"]["residual_s"]
    assert prior.degradation.fit_report["sd"]["fuel_s_per_lap"] != prior.degradation.fit_report["sd"]["fuel_s_per_lap"]
    zero = s.fitted_for(start()).degradation.fit_report["sd"]
    assert zero["residual_s"] == pytest.approx(sigma)
    for comp in TRUE:
        assert zero["base_s"][comp] == pytest.approx(sigma * TAU["base_s"])
        assert zero["wear_s_per_lap"][comp] == pytest.approx(sigma * TAU["wear_s_per_lap"])
    assert zero["fuel_s_per_lap"] == pytest.approx(sigma * TAU["fuel_s_per_lap"])
    late = s.fitted_for(at_lap(30)).degradation.fit_report["sd"]
    assert all(late["wear_s_per_lap"][c] < zero["wear_s_per_lap"][c] for c in TRUE)  # data tightens it


# --- ModelReport ---------------------------------------------------------------
@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
def test_report_fields_on_the_synthetic_frame(name):
    s = Simulator(noisy(synthetic_frame(fuel=FUEL)), window("to_snapshot", degradation={"name": name}),
                  synthetic_practice())
    race = RACE.model_copy(update={"compounds": ["SOFT", "MEDIUM", "HARD"]})
    snap = Snapshot(race=race, driver="X", laps_completed=30, compound="SOFT", tyre_age=5)
    rep = s.model_report(snap, calibration_summary=None)
    assert isinstance(rep, ModelReport) and rep.fit_window == "to_snapshot"
    fit = s.fitted_for(snap).degradation
    fr = fit.fit_report
    assert rep.modelled == ["MEDIUM", "SOFT"]
    assert rep.compounds["HARD"].modelled is False and rep.compounds["HARD"].reason == "no laps in the fit window"
    assert rep.compounds["HARD"].base_s is None
    for comp in TRUE:
        c = rep.compounds[comp]
        assert c.modelled and c.reason is None
        assert c.n_rows == fr["n_rows"][comp] and c.n_race_rows == fr["n_race_rows_per_compound"][comp] > 0
        assert c.max_age_observed == fr["max_tyre_age"][comp] == 20
        assert (c.base_s.value, c.wear_s_per_lap.value) == fit.coefs[comp]
        assert c.base_s.sd == fr["sd"]["base_s"][comp] and c.wear_s_per_lap.sd == fr["sd"]["wear_s_per_lap"][comp]
        assert c.wear_s_per_lap.shrinkage_weight == fr["shrinkage_weight"]["wear_s_per_lap"][comp]
        assert c.base_s.source == c.wear_s_per_lap.source == "shrunk"
        assert 0 < c.wear_s_per_lap.sd < TAU["wear_s_per_lap"]
    assert rep.condition_number == fr["condition_number"]
    assert rep.hinge is None and math.isnan(rep.calibration_error_s)
    if name == "linear_fuel":
        assert rep.fuel_s_per_lap.value == fit.fuel_slope and rep.fuel_s_per_lap.source == "shrunk"
        assert rep.fuel_s_per_lap.shrinkage_weight == fr["shrinkage_weight"]["fuel_s_per_lap"]
        assert rep.fuel_s_per_lap.sd == fr["sd"]["fuel_s_per_lap"] > 0
    else:
        assert rep.fuel_s_per_lap is None
    assert not any(f.startswith(("prior_dominated", "extrapolation")) for f in rep.flags)

    zero = s.model_report(start(), calibration_summary=None)
    assert {c.wear_s_per_lap.source for c in zero.compounds.values()} == {"prior"}
    assert {"prior_dominated:MEDIUM", "prior_dominated:SOFT"} <= set(zero.flags)
    assert all(c.n_race_rows == 0 and c.max_age_observed == 10 for c in zero.compounds.values())

    full = Simulator(noisy(synthetic_frame(fuel=FUEL)), SimConfig(degradation={"name": name}))
    frep = full.model_report(start(), calibration_summary=None)
    assert frep.fit_window == "full" and frep.flags == []
    assert {c.wear_s_per_lap.source for c in frep.compounds.values()} == {"fitted"}
    if name == "linear_fuel":
        assert frep.fuel_s_per_lap.source == "fitted" and frep.fuel_s_per_lap.shrinkage_weight == 1.0


def test_unmodelled_compound_carries_the_fit_reason():
    df = pd.concat([synthetic_frame(), pd.DataFrame([{"compound": "HARD", "tyre_age": 1, "lap": 1,
                                                      "clean": True, "lap_time_s": 85.0}] * 3)])
    rep = Simulator(df).model_report(start(), calibration_summary=None)
    assert rep.compounds["HARD"].modelled is False
    assert rep.compounds["HARD"].reason == "3 clean laps, under min_samples 5"
    assert rep.compounds["HARD"].n_rows == 0 and rep.compounds["HARD"].max_age_observed is None


def test_wear_fuel_flag_fires_on_a_one_compound_first_stint_and_clears_with_a_second():
    s = Simulator(first_stint_race(), window("to_snapshot"), synthetic_practice())

    def flags(k):
        snap = Snapshot(race=RACE, driver="A", laps_completed=k, compound="SOFT" if k < 20 else "MEDIUM",
                        tyre_age=k if k < 20 else k - 20)
        return s.model_report(snap, calibration_summary=None).flags

    rep10 = s.model_report(at_lap(10, "A"), calibration_summary=None)
    assert rep10.compounds["SOFT"].n_race_rows > 0 and rep10.compounds["MEDIUM"].n_race_rows == 0
    assert rep10.fuel_s_per_lap.shrinkage_weight < 0.5
    assert "wear_fuel_poorly_identified" in flags(10)
    assert "wear_fuel_poorly_identified" in flags(19)
    assert "wear_fuel_poorly_identified" not in flags(30)  # MEDIUM rows from lap 21
    assert "prior_dominated:MEDIUM" in flags(10)
    linear = Simulator(first_stint_race(), window("to_snapshot", degradation={"name": "linear"}), synthetic_practice())
    assert "wear_fuel_poorly_identified" not in linear.model_report(at_lap(10, "A"), calibration_summary=None).flags


def test_extrapolation_flag_beyond_the_observed_age():
    s = Simulator(synthetic_frame())  # every stint observed to age 20
    young = Snapshot.at_start(RACE, "X", "SOFT", tyre_age=5)
    assert s.model_report(young, calibration_summary=None).flags == []
    old = Snapshot.at_start(RACE, "X", "SOFT", tyre_age=20)  # next lap is age 21
    assert s.model_report(old, calibration_summary=None).flags == ["extrapolation:SOFT"]
    assert s.model_report(young, ages={"medium": 25}, calibration_summary=None).flags == ["extrapolation:MEDIUM"]
    assert s.model_report(young, ages={"MEDIUM": 20, "HARD": 40}, calibration_summary=None).flags == []


# --- support warnings ----------------------------------------------------------
def test_support_warning_per_stint_beyond_the_synthetic_support():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))
    assert s.simulate(start(), [["SOFT", 20], ["MEDIUM", 20], ["SOFT", 10]]).support_warnings == []
    res = s.simulate(start(), [["SOFT", 25], ["MEDIUM", 25]])
    assert res.valid and res.support_warnings == ["SOFT to age 25, observed up to 20",
                                                  "MEDIUM to age 25, observed up to 20"]
    mid = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10)
    assert s.simulate(mid, [["SOFT", 12], ["MEDIUM", 28]]).support_warnings == [
        "SOFT to age 22, observed up to 20", "MEDIUM to age 28, observed up to 20"]
    used = s.simulate(start(), [["SOFT", 20], ["MEDIUM", 15, 10], ["SOFT", 15]])
    assert used.support_warnings == ["MEDIUM to age 25, observed up to 20"]
    assert s.simulate(start(), [["SOFT", 49]]).support_warnings == []  # invalid: none


def test_support_warnings_are_carried_per_candidate_by_compare():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))
    plans = [[["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]], [["SOFT", 50]]]
    res = s.compare(start(), plans)
    assert res.valid == [True, True, False]
    assert res.support_warnings == [["MEDIUM to age 30, observed up to 20"], [], []]
    batch = s.simulate_batch(start(), plans)
    assert batch.support_warnings == res.support_warnings


def test_support_warnings_leave_the_numbers_unchanged():
    s = Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))
    plan = [["SOFT", 25], ["MEDIUM", 25]]
    res = s.simulate(start(), plan)
    assert res.support_warnings
    assert res.remaining.mean_s == pytest.approx(hand_time(plan) + SimConfig().effects.lap1_extra_s)


# --- calibration_error_s -------------------------------------------------------
def test_calibration_error_is_read_from_a_summary_csv(tmp_path):
    stub = pd.DataFrame({"window": ["to_snapshot"] * 3 + ["full"], "horizon_bucket": [20, 30, 40, 40],
                         "n": 1, "mae_s": 9.0, "mae_offset_s": [3.5, 4.5, 6.25, 1.0]})
    path = tmp_path / "midrace_summary.csv"
    stub.to_csv(path, index=False)
    s = Simulator(synthetic_frame(), window("to_snapshot"), synthetic_practice())
    k10 = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10)  # horizon 40
    assert s.model_report(k10, calibration_summary=path).calibration_error_s == 6.25
    k21 = Snapshot(race=RACE, driver="X", laps_completed=21, compound="SOFT", tyre_age=21)  # horizon 29
    assert s.model_report(k21, calibration_summary=str(path)).calibration_error_s == 3.5
    indexed = stub.set_index(["window", "horizon_bucket"])
    assert s.model_report(k10, calibration_summary=indexed).calibration_error_s == 6.25
    assert math.isnan(s.model_report(start(), calibration_summary=path).calibration_error_s)  # bucket 50
    assert math.isnan(s.model_report(k10, calibration_summary=tmp_path / "missing.csv").calibration_error_s)
    pre = Simulator(synthetic_frame(), window("pre_race"), synthetic_practice())
    assert math.isnan(pre.model_report(k10, calibration_summary=path).calibration_error_s)


def test_default_summary_is_the_committed_midrace_table():
    from src.sim.model_report import CALIBRATION_SUMMARY
    table = pd.read_csv(CALIBRATION_SUMMARY)
    want = float(table[(table["window"] == "to_snapshot") & (table["horizon_bucket"] == 40)]["mae_offset_s"].iloc[0])
    s = Simulator(synthetic_frame(), window("to_snapshot"), synthetic_practice())
    k10 = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10)
    assert s.model_report(k10).calibration_error_s == want


def test_race_without_the_compound_list_reports_only_fit_compounds():
    bare = RaceParams(race_id="syn", total_laps=50, pit_loss_s=20.0)
    rep = Simulator(synthetic_frame()).model_report(Snapshot.at_start(bare, "X", "SOFT"), calibration_summary=None)
    assert sorted(rep.compounds) == ["MEDIUM", "SOFT"]


# --- hazard support --------------------------------------------------------------
class StubHazard:
    draws = True

    def __init__(self, a0):
        self.a0 = a0


def test_no_hazard_support_flag_hook():
    from src.sim.model_report import hazard_support
    s = Simulator(synthetic_frame())
    assert not any(f.startswith("no_hazard_support") for f in s.model_report(start(), calibration_summary=None).flags)
    s.hazard = StubHazard({"SOFT": 23.0})
    flags = s.model_report(start(), calibration_summary=None).flags
    assert "no_hazard_support:MEDIUM" in flags and "no_hazard_support:SOFT" not in flags
    off = StubHazard({})
    off.draws = False  # a hazard model that draws nothing says nothing about tyre risk
    assert hazard_support(off) is None and hazard_support(None) is None
    assert hazard_support(StubHazard({"SOFT": 23.0, "MEDIUM": 23.0})) == {"SOFT", "MEDIUM"}


def test_no_hazard_support_flag_with_the_fitted_hazard_model():
    from src.sim.hazard import AgeLogisticHazard
    s = Simulator(synthetic_frame(), SimConfig(hazard={"name": "age_logistic"}))
    assert isinstance(s.hazard, AgeLogisticHazard) and set(s.hazard.a0) == set(TRUE)
    assert not any(f.startswith("no_hazard_support") for f in s.model_report(start(), calibration_summary=None).flags)
    s.hazard.a0.pop("MEDIUM")  # MEDIUM now borrows the mean a0: no support of its own
    assert "no_hazard_support:MEDIUM" in s.model_report(start(), calibration_summary=None).flags
    # under to_snapshot the hazard is the one fitted for the window
    w = Simulator(synthetic_frame(), window("to_snapshot", hazard={"name": "age_logistic"}), synthetic_practice())
    fit = w.fitted_for(at_lap(30))
    fit.hazard.a0.pop("SOFT")
    assert "no_hazard_support:SOFT" in w.model_report(at_lap(30), calibration_summary=None).flags
    none = Simulator(synthetic_frame())  # default hazard "none": draws nothing, no flag
    assert not any(f.startswith("no_hazard_support") for f in none.model_report(start(), calibration_summary=None).flags)
