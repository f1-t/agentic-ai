"""linear_fuel_hinge (SPEC 0010): the hinge degradation model. Synthetic data only.

The cliff frame wears 0.03 s/lap below tyre age 25 and 0.03 + 0.15 s/lap above it.
"""
import numpy as np
import pandas as pd
import pytest

from src.contracts import RaceParams, Snapshot
from src.sim import SimConfig, Simulator
from src.sim import calibrators as cal
from src.sim.config import HINGE_KNOT
from src.sim.degradation import DEGRADATION
from src.sim.optimizer_bridge import optimize_batch

from test_sim import at_lap, start, window

KNOT, HINGE, FUEL = 25, 0.15, -0.04
TRUE = {"SOFT": (80.0, 0.03), "MEDIUM": (80.4, 0.03)}
HINGED = {"degradation": {"name": "linear_fuel_hinge"}}


def cliff_frame(max_age=40, fuel=FUEL):
    """Ten stints per compound, starting every third lap, each run to `max_age`."""
    rows = []
    for comp, (base, wear) in TRUE.items():
        for first in range(0, 30, 3):
            for age in range(1, max_age + 1):
                lap = first + age
                rows.append({"driver": "A", "compound": comp, "tyre_age": age, "lap": lap, "clean": True,
                             "lap_time_s": base + wear * age + HINGE * max(0, age - KNOT) + fuel * lap})
    return pd.DataFrame(rows)


def cliff_practice(max_age=30, drivers=("A", "B", "C")):
    """One long run per compound per driver on the cliff, no race lap (practice)."""
    rows = []
    for k, drv in enumerate(drivers):
        for comp, (base, wear) in TRUE.items():
            for age in range(1, max_age + 1):
                rows.append({"driver": drv, "session": f"FP{k % 3 + 1}", "compound": comp, "tyre_age": age,
                             "lap": np.nan, "clean": True,
                             "lap_time_s": base + wear * age + HINGE * max(0, age - KNOT)})
    return pd.DataFrame(rows)


def test_registered_beside_linear_fuel_which_stays_the_default():
    assert "linear_fuel_hinge" in DEGRADATION and "linear_fuel" in DEGRADATION
    cfg = SimConfig()
    assert cfg.degradation.name == "linear_fuel"
    assert cfg.hinge_knots == {} and cfg.hinge_knot_default == HINGE_KNOT == 25
    assert cfg.shrinkage["hinge_s_per_lap"] == 0.05


def test_full_fit_recovers_the_hinge():
    m = Simulator(cliff_frame(), SimConfig(**HINGED)).degradation
    for comp, (base, wear) in TRUE.items():
        assert m.coefs[comp] == pytest.approx((base, wear), abs=1e-6)
        assert m.hinges[comp] == pytest.approx(HINGE, abs=1e-6)
        h = m.fit_report["hinge"][comp]
        assert h["value"] == m.hinges[comp] and h["source"] == "fitted" and h["weight"] == 1.0
        assert h["knot"] == KNOT and h["n_past_knot"] == 10 * 15 and h["sd"] == pytest.approx(0.0, abs=1e-6)
    assert m.fuel_slope == pytest.approx(FUEL, abs=1e-6)
    assert m.knots == {"SOFT": KNOT, "MEDIUM": KNOT}
    # fractional ages, as the engine passes on neutralised laps
    base, wear = m.coefs["SOFT"]
    got = m.lap_times("SOFT", [24.5, 25.5], [30, 30])
    assert got == pytest.approx([base + wear * 24.5 + FUEL * 30, base + wear * 25.5 + HINGE * 0.5 + FUEL * 30],
                                abs=1e-6)


def test_per_compound_knots_come_from_the_config():
    m = Simulator(cliff_frame(), SimConfig(**HINGED, hinge_knots={"SOFT": 20})).degradation
    assert m.knots == {"SOFT": 20, "MEDIUM": KNOT}
    assert m.fit_report["hinge"]["SOFT"]["knot"] == 20
    assert m.hinges["MEDIUM"] == pytest.approx(HINGE, abs=1e-6)


def test_hinge_dropped_below_the_knot_is_linear_fuel_and_flagged():
    frame = cliff_frame(max_age=20)  # no row past the knot
    hinge = Simulator(frame, SimConfig(**HINGED))
    plain = Simulator(frame, SimConfig(degradation={"name": "linear_fuel"}))
    m, ref = hinge.degradation, plain.degradation
    assert m.hinges == {"SOFT": 0.0, "MEDIUM": 0.0}
    for comp in TRUE:
        h = m.fit_report["hinge"][comp]
        assert h["source"] == "dropped" and h["value"] == 0.0 and h["weight"] == 0.0 and np.isnan(h["sd"])
        assert h["n_past_knot"] == 0
    # with no hinge column the fit is linear_fuel's, bit for bit
    assert m.coefs == ref.coefs and m.fuel_slope == ref.fuel_slope
    assert {k: v for k, v in m.fit_report.items() if k != "hinge"} == ref.fit_report
    rep = hinge.model_report(start(), ages={"MEDIUM": 30}, calibration_summary=None)
    assert "beyond_knot_unsupported:MEDIUM" in rep.flags and "beyond_knot_unsupported:SOFT" not in rep.flags
    assert rep.hinge["MEDIUM"]["source"] == "dropped"
    ok = hinge.model_report(start(), ages={"MEDIUM": KNOT}, calibration_summary=None)  # at the knot is fine
    assert not any(f.startswith("beyond_knot_unsupported") for f in ok.flags)
    fitted = Simulator(cliff_frame(), SimConfig(**HINGED)).model_report(start(), ages={"MEDIUM": 40},
                                                                         calibration_summary=None)
    assert not any(f.startswith("beyond_knot_unsupported") for f in fitted.flags)
    assert fitted.hinge["MEDIUM"]["value"] == pytest.approx(HINGE, abs=1e-6)


def test_hinge_needs_min_samples_rows_past_the_knot():
    frame = cliff_frame(max_age=40)
    few = frame[(frame["tyre_age"] <= KNOT) | ((frame["compound"] == "SOFT") & (frame["lap"] <= 30)
                                               & (frame["tyre_age"] <= 27))]
    m = Simulator(few, SimConfig(**HINGED, min_samples=5)).degradation
    soft = m.fit_report["hinge"]["SOFT"]
    assert soft["n_past_knot"] == 4 and soft["source"] == "dropped"  # ages 26 and 27 of the first two stints
    assert m.fit_report["hinge"]["MEDIUM"]["n_past_knot"] == 0


def test_linear_fuel_ignores_the_hinge_config():
    frame = cliff_frame()
    a = Simulator(frame, SimConfig(degradation={"name": "linear_fuel"})).degradation
    b = Simulator(frame, SimConfig(degradation={"name": "linear_fuel"}, hinge_knots={"SOFT": 10},
                                   hinge_knot_default=5)).degradation
    assert a.coefs == b.coefs and a.fuel_slope == b.fuel_slope and a.fit_report == b.fit_report
    assert "hinge" not in a.fit_report and not hasattr(a, "hinges")
    assert Simulator(frame).model_report(start(), calibration_summary=None).hinge is None


def test_pre_race_carries_the_practice_hinge():
    s = Simulator(cliff_frame(), window("pre_race", **HINGED), cliff_practice())
    raw = s.physics_prior()
    assert raw.hinges["MEDIUM"] == pytest.approx(HINGE, abs=1e-6)
    assert raw.fit_report["hinge"]["MEDIUM"]["source"] == "fitted"
    pre = s.fitted_for(start())
    assert pre.degradation.hinges == pytest.approx(raw.hinges, abs=1e-12)
    out = pre.calibration.output.factors
    assert out["hinge_s_per_lap[MEDIUM]"] == pytest.approx((HINGE, 0.05), abs=1e-6)
    rep = s.model_report(start(), ages={"MEDIUM": 35}, calibration_summary=None)
    assert rep.hinge["MEDIUM"]["value"] == pytest.approx(HINGE, abs=1e-6)
    assert not any(f.startswith("beyond_knot_unsupported") for f in rep.flags)
    eff = cal.effective_factors(pre)
    assert eff["hinge_s_per_lap[MEDIUM]"] == pytest.approx((HINGE, 0.05), abs=1e-6)  # held: the output's sd
    none = Simulator(cliff_frame(), window("to_snapshot", calibrator={"name": "none"}, **HINGED), cliff_practice())
    held = none.fitted_for(at_lap(30))
    assert held.calibration.held and held.calibration.output.factors["hinge_s_per_lap[SOFT]"][1] == 0.0
    assert held.degradation.hinges == pytest.approx(raw.hinges, abs=1e-12)


def test_to_snapshot_shrinks_the_hinge_toward_the_practice_prior():
    tau = SimConfig().shrinkage["hinge_s_per_lap"]
    s = Simulator(cliff_frame(), window("to_snapshot", **HINGED), cliff_practice())
    prior = s.physics_prior().hinges
    late = s.fitted_for(at_lap(30)).degradation
    race = cliff_frame()[lambda d: d["lap"] <= 30]
    for comp in TRUE:
        h = late.fit_report["hinge"][comp]
        rows = race[race["compound"] == comp]
        sq = float((np.maximum(0.0, rows["tyre_age"] - KNOT) ** 2).sum())
        assert h["source"] == "shrunk" and h["n_past_knot"] == 7 + 3 * 5  # race rows past the knot, plus practice
        assert h["weight"] == pytest.approx(sq / (sq + tau ** -2))
        assert np.isfinite(h["sd"]) and 0.0 < h["sd"] < tau
        assert late.hinges[comp] == pytest.approx(HINGE, abs=1e-4)
    early = s.fitted_for(at_lap(20)).degradation  # no race row past the knot yet: the prior, at its tau
    for comp in TRUE:
        h = early.fit_report["hinge"][comp]
        assert h["source"] == "prior" and h["weight"] == 0.0 and h["sd"] == pytest.approx(tau)
        assert early.hinges[comp] == pytest.approx(prior[comp], abs=1e-12)
    eff = cal.effective_factors(s.fitted_for(at_lap(30)))
    assert eff["hinge_s_per_lap[MEDIUM]"] == (late.hinges["MEDIUM"], late.fit_report["hinge"]["MEDIUM"]["sd"])


def test_to_snapshot_without_a_practice_hinge_shrinks_toward_zero():
    tau = SimConfig().shrinkage["hinge_s_per_lap"]
    practice = cliff_practice(max_age=20)  # practice never passes the knot: prior 0
    s = Simulator(cliff_frame(), window("to_snapshot", **HINGED), practice)
    assert s.physics_prior().fit_report["hinge"]["SOFT"]["source"] == "dropped"
    assert s.fitted_for(start()).calibration.output.factors["hinge_s_per_lap[SOFT]"] == (0.0, tau)
    m = s.fitted_for(at_lap(30)).degradation
    rows = cliff_frame()[lambda d: (d["lap"] <= 30) & (d["compound"] == "SOFT")]
    h = np.maximum(0.0, rows["tyre_age"].to_numpy(dtype=float) - KNOT)
    assert m.fit_report["hinge"]["SOFT"]["weight"] == pytest.approx(float(h @ h) / (float(h @ h) + tau ** -2))
    assert 0.0 < m.hinges["SOFT"] < HINGE   # pulled toward 0 by the prior
    # no hinge column at lap 20 (nothing past the knot anywhere): bit for bit linear_fuel's shrunk fit
    hinge = s.fitted_for(at_lap(20)).degradation
    plain = Simulator(cliff_frame(), window("to_snapshot", degradation={"name": "linear_fuel"}),
                      practice).fitted_for(at_lap(20)).degradation
    assert hinge.coefs == plain.coefs and hinge.fuel_slope == plain.fuel_slope
    assert {k: v for k, v in hinge.fit_report.items() if k != "hinge"} == plain.fit_report
    assert {h["source"] for h in hinge.fit_report["hinge"].values()} == {"dropped"}


def test_shrinkage_calibrator_carries_no_hinge_for_linear_fuel():
    s = Simulator(cliff_frame(), window("to_snapshot", degradation={"name": "linear_fuel"}), cliff_practice())
    out = s.fitted_for(at_lap(30)).calibration.output
    assert not any(k.startswith("hinge_s_per_lap") for k in out.factors)
    assert "hinge_s_per_lap" not in out.diagnostics["taus"]
    hinged = Simulator(cliff_frame(), window("to_snapshot", **HINGED), cliff_practice())
    assert hinged.fitted_for(at_lap(30)).calibration.output.diagnostics["taus"]["hinge_s_per_lap"] == 0.05


def test_brute_force_optimum_stays_off_the_cliff():
    # the data stop at age 28, so a straight line barely sees the cliff and runs a long stint into it
    race = RaceParams(race_id="cliff", total_laps=66, pit_loss_s=18.0, min_stint_laps=8, compounds=["SOFT", "MEDIUM"])
    snap = Snapshot.at_start(race, "X", "SOFT")
    frame = cliff_frame(max_age=28)
    ends = {}
    for name in ("linear_fuel", "linear_fuel_hinge"):
        best = optimize_batch(snap, Simulator(frame, SimConfig(degradation={"name": name})))
        ends[name] = max(s.laps + s.start_age for s in best.plan.stints)
    assert ends["linear_fuel"] > KNOT
    assert ends["linear_fuel_hinge"] <= KNOT
