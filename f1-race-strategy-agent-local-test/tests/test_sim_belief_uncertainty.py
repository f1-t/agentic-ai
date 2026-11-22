"""docs/eval.md "Honest uncertainty": the widened parameter sds, the persistent lap noise and the
plan_on switch. Synthetic races only."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import Plan, RaceParams, Snapshot
from src.sim import agent_tools
from src.sim.config import ComponentSpec, SimConfig
from src.sim.degradation import LinearFuelDegradation, _residual_components, _scale_sd
from src.sim.engine import Simulator


def race_rows(seed=0, n_drivers=6, laps=45, stint_sd=0.4):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        pit = 15 + 3 * d
        for st, (a, b) in enumerate(((1, pit), (pit + 1, laps))):
            off = rng.normal(0, stint_sd)
            for lap in range(a, b + 1):
                rows.append({"driver": f"D{d}", "stint": st, "compound": ("SOFT", "MEDIUM")[(st + d) % 2], "lap": lap,
                             "tyre_age": lap - a + 1, "clean": True,
                             "lap_time_s": 90 + off + 0.05 * (lap - a + 1) - 0.06 * lap + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def prior_model(data):
    return LinearFuelDegradation.fit(data[data["lap"] <= 10])


TAUS = {"base_s": 0.5, "wear_s_per_lap": 0.03, "fuel_s_per_lap": 0.01}


def test_sd_scale_widens_every_sd_and_the_cov_but_keeps_the_point_estimate():
    data = race_rows()
    prior = prior_model(data)
    plain = LinearFuelDegradation.fit(data, prior=prior, shrinkage=TAUS, fuel_prior=-0.05)
    wide = LinearFuelDegradation.fit(data, prior=prior, shrinkage=TAUS, fuel_prior=-0.05,
                                     sd_scale={"wear_s_per_lap": 2.0, "fuel_s_per_lap": 3.0})
    assert wide.coefs == plain.coefs and wide.fuel_slope == plain.fuel_slope
    assert wide.fit_report["sd"]["fuel_s_per_lap"] == pytest.approx(3 * plain.fit_report["sd"]["fuel_s_per_lap"])
    for c in plain.coefs:
        assert wide.fit_report["sd"]["wear_s_per_lap"][c] == pytest.approx(2 * plain.fit_report["sd"]["wear_s_per_lap"][c])
        assert wide.fit_report["sd"]["base_s"][c] == pytest.approx(plain.fit_report["sd"]["base_s"][c])
    names = plain.fit_report["cov"]["names"]
    a, b = (np.asarray(m.fit_report["cov"]["matrix"]) for m in (plain, wide))
    s = np.array([{"wear_s_per_lap": 2.0, "fuel_s_per_lap": 3.0}.get(n.split("[")[0], 1.0) for n in names])
    assert b == pytest.approx(a * np.outer(s, s))   # correlations kept
    with pytest.raises(ValueError):
        _scale_sd({"sd": {}}, {"nope": 2.0})


def test_residual_components_split_stint_offsets_from_lap_noise():
    data = race_rows(stint_sd=0.6)
    m = LinearFuelDegradation.fit(data, prior=prior_model(data), shrinkage=TAUS, fuel_prior=-0.06)
    comp = m.fit_report["residual_components"]
    assert comp["lap_s"] == pytest.approx(0.2, abs=0.05)
    assert comp["stint_s"] + comp["driver_s"] > 0.3
    assert _residual_components(pd.DataFrame({"x": [1, 2]}), np.array([0.1, -0.1]))["n_groups"] == 0


def snapshot(k=10, laps=45):
    race = RaceParams(race_id="2099_01_synthetic", total_laps=laps, pit_loss_s=20.0)
    return Snapshot(race=race, driver="D0", laps_completed=k, compound="SOFT", tyre_age=k)


def config(draw="posterior", **kw):
    return SimConfig(fit_window="to_snapshot", calibrator=ComponentSpec(name="shrinkage"),
                     degradation=ComponentSpec(name="linear_fuel", params={"draw": draw}), **kw)


def test_persistent_noise_is_reproducible_and_wider_than_iid_over_a_stint():
    data = race_rows(stint_sd=0.6)
    practice = data[data["lap"] <= 12].assign(lap=np.nan, lap_in_stint=lambda f: f["tyre_age"])
    snap = snapshot()
    plan = Plan.of([("SOFT", 5), ("MEDIUM", 30)])
    out = {}
    quiet = Simulator(data, config("none"), practice).simulate_batch(snap, [plan], seed=3, n_samples=300).times[0]
    for noise in ("residual", "persistent"):   # no parameter draw; the noise is the difference from none
        sim = Simulator(data, config("none", lap_noise=noise), practice)
        a = sim.simulate_batch(snap, [plan], seed=3, n_samples=300).times[0]
        b = sim.simulate_batch(snap, [plan], seed=3, n_samples=300).times[0]
        assert np.array_equal(a, b)
        out[noise] = (a - quiet).std()
    assert out["persistent"] > 2 * out["residual"]


def test_plan_on_point_ranks_on_the_point_estimate():
    from src.sim.optimizer_bridge import optimize_batch
    data = race_rows()
    practice = data[data["lap"] <= 12].assign(lap=np.nan, lap_in_stint=lambda f: f["tyre_age"])
    snap = snapshot()
    sim = Simulator(data, config(lap_noise="residual", plan_on="point"), practice)
    fit = sim.fitted_for(snap)
    pt = fit.point_estimate()
    assert pt.degradation.param_draw == "none" and pt.config.lap_noise == "none" and fit.degradation.param_draw == "posterior"
    point = optimize_batch(snap, sim, seed=1, n_samples=50)
    draws = optimize_batch(snap, Simulator(data, config(lap_noise="residual"), practice), seed=1, n_samples=50)
    # the parameter and lap-noise spread is gone; what is left is drawn elsewhere (pit loss, factors)
    assert np.std(point.evaluation.seed_times_s) < np.std(draws.evaluation.seed_times_s)


def test_the_tools_say_what_their_intervals_mean():
    data = race_rows()
    prior = prior_model(data)
    wide = LinearFuelDegradation.fit(data, prior=prior, shrinkage=TAUS, fuel_prior=-0.05, sd_scale={"fuel_s_per_lap": 2})
    plain = LinearFuelDegradation.fit(data, prior=prior, shrinkage=TAUS, fuel_prior=-0.05)
    assert agent_tools.interval_meaning(wide) == agent_tools.INTERVAL_MEANING_WIDENED
    assert agent_tools.interval_meaning(plain) == agent_tools.INTERVAL_MEANING_RAW
