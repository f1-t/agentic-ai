"""Driver effects in the degradation fit (degradation.params driver_effects): a field where the
faster cars run the older tyres flattens a pooled wear fit; per-driver offsets that sum to zero
take the car out of the tyre. Synthetic, offline."""
import json

import numpy as np
import pandas as pd
import pytest

from src.contracts import Snapshot
from src.sim import SimConfig, Simulator
from src.sim.degradation import DEGRADATION, _sum_to_zero_basis
from test_sim import RACE, synthetic_practice

TRUE = {"SOFT": (80.0, 0.08), "MEDIUM": (80.8, 0.05)}
FUEL = -0.05
N_DRIVERS = 12


def pace_of(n=N_DRIVERS):
    """Driver pace offsets, fastest first: -0.6 s to +0.6 s."""
    return {f"D{i:02d}": float(p) for i, p in enumerate(np.linspace(-0.6, 0.6, n))}


def confounded_frame(seed=0, noise=0.05, laps=50, used=12):
    """Every car runs SOFT then MEDIUM, stopping between laps 15 and 37 (an order unrelated to
    pace), both stints on sets that start `used` laps old for the fastest car down to new for the
    slowest: faster cars run older tyres, so tyre age and driver pace are negatively correlated."""
    rng = np.random.default_rng(seed)
    pace = pace_of()
    stops = np.random.default_rng(1).permutation(np.linspace(15, 37, len(pace)).round().astype(int))
    rows = []
    for rank, (d, p) in enumerate(pace.items()):
        start = round(used * (1 - rank / (len(pace) - 1)))
        for lap in range(1, laps + 1):
            comp = "SOFT" if lap <= stops[rank] else "MEDIUM"
            base, wear = TRUE[comp]
            age = float(start + (lap if comp == "SOFT" else lap - stops[rank]))
            rows.append({"driver": d, "compound": comp, "tyre_age": age, "lap": float(lap), "clean": lap > 1,
                         "lap_time_s": base + wear * age + FUEL * lap + p + rng.normal(0.0, noise)})
    return pd.DataFrame(rows)


def fit(frame, name="linear_fuel", **kw):
    return DEGRADATION.get(name).fit(frame, **kw)


def wear(model):
    return {c: w for c, (_, w) in model.coefs.items()}


# --- the confounding and its fix ------------------------------------------------------------------
def test_faster_cars_on_older_tyres_flatten_the_pooled_wear_and_driver_effects_recover_it():
    frame = confounded_frame()
    clean = frame[frame["clean"]]
    offset = clean["driver"].map(pace_of())
    assert np.corrcoef(offset, clean["tyre_age"])[0, 1] < -0.3   # the premise: fast cars, old tyres
    pooled, de = wear(fit(frame)), wear(fit(frame, driver_effects=True))
    for c, (_, w) in TRUE.items():
        assert pooled[c] < w - 0.01            # pooling credits the old tyre with the fast car's pace
        assert de[c] == pytest.approx(w, abs=0.003)
    assert fit(frame, driver_effects=True).fuel_slope == pytest.approx(FUEL, abs=0.003)


@pytest.mark.parametrize("name", ["linear_fuel", "linear_fuel_hinge"])
def test_the_shrunk_fit_with_driver_effects_moves_wear_toward_the_truth(name):
    frame = confounded_frame()
    shrink = SimConfig().shrinkage
    pooled = fit(frame, name, prior=None, shrinkage=shrink, fuel_prior=FUEL)
    de = fit(frame, name, prior=None, shrinkage=shrink, fuel_prior=FUEL, driver_effects=True)
    for c, (_, w) in TRUE.items():
        assert abs(wear(de)[c] - w) < abs(wear(pooled)[c] - w)
    rep = de.fit_report["driver_effects"]
    assert rep["tau"] == shrink["base_s"] and rep["n_drivers"] == N_DRIVERS
    # a tau of its own when the shrinkage dict carries one
    tight = fit(frame, name, prior=None, shrinkage={**shrink, "driver_s": 0.01}, fuel_prior=FUEL, driver_effects=True)
    assert tight.fit_report["driver_effects"]["tau"] == 0.01
    spread = lambda m: np.ptp(list(m.fit_report["driver_effects"]["value"].values()))  # noqa: E731
    assert spread(tight) < spread(de)


# --- the constraint -------------------------------------------------------------------------------
def test_the_basis_is_orthonormal_and_sums_to_zero():
    for m in (1, 2, 5, 20):
        Q = _sum_to_zero_basis(m)
        assert Q.shape == (m, max(m - 1, 0))
        np.testing.assert_allclose(Q.T @ Q, np.eye(m - 1), atol=1e-12)
        np.testing.assert_allclose(Q.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("name, params", [("linear_fuel", {}), ("linear_fuel", {"wear_lower_bound": 0.0}),
                                          ("linear_fuel_hinge", {"hinge_knot_default": 20})])
def test_driver_effects_sum_to_zero_and_are_the_centred_pace(name, params):
    frame = confounded_frame()
    for kw in ({}, {"prior": None, "shrinkage": SimConfig().shrinkage, "fuel_prior": FUEL}):
        rep = fit(frame, name, driver_effects=True, **params, **kw).fit_report["driver_effects"]
        value = rep["value"]
        assert sorted(value) == sorted(pace_of()) and abs(sum(value.values())) < 1e-9
        assert set(rep["sd"]) == set(value) and all(s > 0 for s in rep["sd"].values())
        if not kw:   # unpenalised: the offsets are the pace about the field mean
            for d, p in pace_of().items():
                assert value[d] == pytest.approx(p, abs=0.02)


def test_each_compound_base_stays_the_field_mean():
    """Without confounding (every car on new sets) the bases are the field mean whatever the pace."""
    frame = confounded_frame(used=0, noise=0.0)
    model = fit(frame, driver_effects=True)
    for c, (b, w) in TRUE.items():
        assert model.coefs[c][0] == pytest.approx(b + np.mean(list(pace_of().values())), abs=1e-6)
        assert model.coefs[c][1] == pytest.approx(w, abs=1e-6)


def test_drivers_are_nuisance_parameters_left_out_of_the_cov_and_the_draws():
    frame = confounded_frame()
    plain, de = fit(frame), fit(frame, driver_effects=True, draw="posterior")
    assert de.fit_report["cov"]["names"] == plain.fit_report["cov"]["names"]
    assert de.n_params == plain.n_params
    drawn = de.draw(np.random.default_rng(0))
    assert set(drawn.coefs) == set(TRUE) and drawn.coefs != de.coefs
    # the engine's lap time has no driver term
    np.testing.assert_allclose(de.lap_times("SOFT", [5.0], [10.0]),
                               de.coefs["SOFT"][0] + 5 * de.coefs["SOFT"][1] + 10 * de.fuel_slope)


def test_the_practice_and_to_snapshot_windows_take_driver_effects():
    practice = synthetic_practice(coefs=TRUE, in_stint=True)
    cfg = SimConfig(fit_window="to_snapshot", degradation={"name": "linear_fuel", "params": {"driver_effects": True}})
    sim = Simulator(confounded_frame(), cfg, practice)
    snap = Snapshot(race=RACE, driver="D00", laps_completed=30, compound="MEDIUM", tyre_age=5)
    rep = sim.fitted_for(snap).degradation.fit_report["driver_effects"]
    assert rep["n_drivers"] == N_DRIVERS and abs(sum(rep["value"].values())) < 1e-9
    pre = Simulator(confounded_frame(), cfg.model_copy(update={"fit_window": "pre_race"}), practice)
    rep = pre.fitted_for(snap).degradation.fit_report["driver_effects"]
    assert rep["n_drivers"] == 3 and rep["tau"] is None   # practice: drivers A, B, C, unpenalised


def test_driver_effects_need_a_driver_column_and_a_fuel_model():
    with pytest.raises(ValueError, match="driver column"):
        fit(confounded_frame().drop(columns=["driver"]), driver_effects=True)
    with pytest.raises(ValueError, match="linear_fuel"):
        fit(confounded_frame(), "linear", driver_effects=True)


# --- off by default -------------------------------------------------------------------------------
def _dump(model):
    return json.dumps({"coefs": model.coefs, "fuel": model.fuel_slope, "rep": model.fit_report},
                      sort_keys=True, default=str)


@pytest.mark.parametrize("name", ["linear_fuel", "linear_fuel_hinge"])
def test_off_by_default_and_off_is_byte_identical(name):
    frame = confounded_frame()
    assert "driver_effects" not in SimConfig().degradation.params
    shrink = {"prior": None, "shrinkage": SimConfig().shrinkage, "fuel_prior": FUEL}
    for kw in ({}, shrink, {"wear_lower_bound": 0.0}):
        default, off = fit(frame, name, **kw), fit(frame, name, driver_effects=False, **kw)
        assert "driver_effects" not in default.fit_report
        assert _dump(default) == _dump(off)
    # through the simulator: a default config and one with the option off score the same bytes
    snap = Snapshot(race=RACE, driver="D00", laps_completed=10, compound="SOFT", tyre_age=10)
    plans = [[["SOFT", 20], ["MEDIUM", 20]], [["SOFT", 10], ["MEDIUM", 15], ["SOFT", 15]]]
    for window in ("full", "to_snapshot"):
        a = SimConfig(fit_window=window, degradation={"name": name})
        b = SimConfig(fit_window=window, degradation={"name": name, "params": {"driver_effects": False}})
        ta = Simulator(confounded_frame(), a, synthetic_practice(coefs=TRUE)).simulate_batch(snap, plans, n_samples=3)
        tb = Simulator(confounded_frame(), b, synthetic_practice(coefs=TRUE)).simulate_batch(snap, plans, n_samples=3)
        assert ta.times.tobytes() == tb.times.tobytes()
