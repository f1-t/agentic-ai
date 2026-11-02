"""SPEC 0017: the posterior covariance in the fit report, parameter draws and residual lap
noise in the samples, and interval coverage in the mid-race table. Synthetic, offline."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import Snapshot
from src.sim import SimConfig, Simulator, calibration
from src.sim.config import EventEffects
from src.sim.degradation import DEGRADATION, NaNCovarianceWarning
from src.sim.engine import DEGRADATION_STREAM, EVENT_STREAM, LAP_NOISE_STREAM
from test_sim import FUEL, RACE, TRUE, synthetic_frame, synthetic_practice
from test_sim_position import race_frame

POST = {"name": "linear_fuel", "params": {"draw": "posterior"}}
ONE, TWO = [["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]


def noisy_frame(sd=0.5, seed=0, fuel=FUEL):
    """synthetic_frame's stints with N(0, sd^2) on every lap."""
    f = synthetic_frame(fuel=fuel)
    return f.assign(lap_time_s=f["lap_time_s"] + np.random.default_rng(seed).normal(0, sd, len(f)))


def cov_of(model):
    c = model.fit_report["cov"]
    return c["names"], np.array(c["matrix"])


def draws(model, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = model.draw(rng)
        out.append(m.param_vector()[1])
    return np.array(out)


# --- (1) defaults -------------------------------------------------------------------
def test_defaults_draw_nothing_and_add_only_cov():
    cfg = SimConfig()
    assert cfg.lap_noise == "none" and "draw" not in cfg.degradation.params
    m = Simulator(noisy_frame(), cfg).degradation
    assert m.param_draw == "none" and m.draw(np.random.default_rng(0)) is m
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    m.draw(rng)
    assert rng.bit_generator.state == state   # "none" consumes nothing
    assert (EVENT_STREAM, DEGRADATION_STREAM, LAP_NOISE_STREAM) == (1, 2, 3)
    with pytest.raises(ValueError, match="draw"):
        Simulator(noisy_frame(), SimConfig(degradation={"name": "linear_fuel", "params": {"draw": "bayes"}}))


@pytest.mark.parametrize("name", ["linear", "linear_fuel", "linear_fuel_hinge"])
@pytest.mark.parametrize("fit_window", ["full", "to_snapshot", "pre_race"])
def test_cov_names_order_and_sd_agree(name, fit_window):
    s = Simulator(noisy_frame(), SimConfig(degradation={"name": name}, fit_window=fit_window),
                  synthetic_practice())
    fit = s.fitted_for(Snapshot(race=RACE, driver="X", laps_completed=30, compound="SOFT", tyre_age=10))
    names, m = cov_of(fit.degradation)
    comps = sorted(fit.degradation.coefs)
    head = [f"{p}[{c}]" for c in comps for p in ("base_s", "wear_s_per_lap")]
    assert names[:len(head)] == head
    assert ("fuel_s_per_lap" in names) == (name != "linear")
    assert m.shape == (len(names), len(names)) and np.allclose(m, m.T, equal_nan=True)
    if fit_window != "pre_race":   # a fitted or shrunk model: the diagonal is the reported sd squared
        sd = fit.degradation.fit_report["sd"]
        for c in comps:
            assert np.sqrt(m[names.index(f"wear_s_per_lap[{c}]"), names.index(f"wear_s_per_lap[{c}]")]) == \
                pytest.approx(sd["wear_s_per_lap"][c], rel=1e-9)


def test_fixed_fuel_on_practice_gets_the_prior_variance():
    m = DEGRADATION.get("linear_fuel").fit(synthetic_practice().assign(lap_time_s=lambda d: d["lap_time_s"]
                                                                       + np.random.default_rng(0).normal(0, .3, len(d))),
                                           fuel_prior_sd=0.008)
    names, cov = cov_of(m)
    i = names.index("fuel_s_per_lap")
    assert cov[i, i] == pytest.approx(0.008 ** 2)
    assert not np.any(np.delete(cov[i], i))


def test_weight_zero_shrunk_coefficient_gets_tau_squared():
    """At lap 0 nothing from the race: every shrunk coefficient is held at its prior tau."""
    s = Simulator(noisy_frame(), SimConfig(fit_window="to_snapshot", calibration_every=None), synthetic_practice())
    taus = SimConfig().shrinkage
    # cadence None keeps k0 = 0: the race adds nothing, the shrinkage calibrator is not held
    fit = s.fitted_for(Snapshot(race=RACE, driver="X", laps_completed=0, compound="SOFT", tyre_age=0))
    names, cov = cov_of(fit.degradation)
    for n in names:
        key = n.split("[")[0]
        assert cov[names.index(n), names.index(n)] == pytest.approx(taus[key] ** 2)
    assert not np.any(cov - np.diag(np.diag(cov)))


# --- (2) the draws match the covariance ------------------------------------------------
@pytest.mark.parametrize("fit_window", ["full", "to_snapshot"])
def test_draw_covariance_matches_fit_report(fit_window):
    s = Simulator(noisy_frame(sd=1.0), SimConfig(degradation=POST, fit_window=fit_window), synthetic_practice())
    fit = s.fitted_for(Snapshot(race=RACE, driver="X", laps_completed=25, compound="SOFT", tyre_age=10))
    names, cov = cov_of(fit.degradation)
    x = draws(fit.degradation)
    assert np.diag(np.cov(x.T)) == pytest.approx(np.diag(cov), rel=0.10)
    assert x.mean(axis=0) == pytest.approx(fit.degradation.param_vector()[1], abs=4 * np.sqrt(np.diag(cov).max()))


def test_first_stint_only_gives_a_strongly_negative_fuel_wear_correlation():
    """Every car on its first set: tyre age and race lap move together, so the fit cannot
    tell wear from fuel, and the two slopes trade off."""
    rows = []
    rng = np.random.default_rng(3)
    for i, a0 in enumerate([0, 0, 1, 2, 3, 0, 4, 1]):
        comp = "SOFT" if i % 2 == 0 else "MEDIUM"
        base, wear = TRUE[comp]
        for lap in range(1, 21):
            age = a0 + lap
            rows.append({"compound": comp, "tyre_age": age, "lap": lap, "clean": True,
                         "lap_time_s": base + wear * age + FUEL * lap + rng.normal(0, 0.3)})
    s = Simulator(pd.DataFrame(rows), SimConfig())
    rep = s.model_report(Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20),
                         calibration_summary=None)
    assert rep.fuel_wear_corr is not None and set(rep.fuel_wear_corr) == {"SOFT", "MEDIUM"}
    assert all(v < -0.8 for v in rep.fuel_wear_corr.values())
    assert Simulator(noisy_frame(), SimConfig(degradation={"name": "linear"})).model_report(
        Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20),
        calibration_summary=None).fuel_wear_corr is None


# --- (3) a held prior moves the practice bases together -------------------------------
def test_held_pre_race_prior_moves_practice_bases_together():
    tau = SimConfig().shrinkage
    s = Simulator(noisy_frame(), SimConfig(degradation=POST, fit_window="pre_race"), synthetic_practice())
    fit = s.fitted_for(Snapshot.at_start(RACE, "X", "SOFT"))
    names, cov = cov_of(fit.degradation)
    i, j = names.index("base_s[MEDIUM]"), names.index("base_s[SOFT]")
    assert cov[i, j] == pytest.approx(tau["base_s"] ** 2) and cov[i, i] == pytest.approx(tau["base_s"] ** 2)
    w = names.index("wear_s_per_lap[SOFT]")
    assert cov[w, w] == pytest.approx(tau["wear_s_per_lap"] ** 2) and cov[w, i] == 0.0
    x = draws(fit.degradation, n=200)
    mean = fit.degradation.param_vector()[1]
    assert (x[:, i] - mean[i]) == pytest.approx(x[:, j] - mean[j], abs=1e-9)   # one common delta
    # the none calibrator holds the prior with sd 0: nothing moves
    held = Simulator(noisy_frame(), SimConfig(degradation=POST, fit_window="to_snapshot",
                                              calibrator={"name": "none"}), synthetic_practice())
    m = held.fitted_for(Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20)).degradation
    assert np.all(draws(m, n=20) == m.param_vector()[1])


# --- (4) substreams ------------------------------------------------------------------
def _recorded_events(sim, snap, plans, **kw):
    seen = []
    draw = sim.events.draw

    def spy(snapshot, rng):
        sc = draw(snapshot, rng)
        seen.append((tuple(sc.sc_laps), tuple(sc.vsc_laps)))
        return sc
    sim.events.draw = spy
    b = sim.simulate_batch(snap, plans, **kw)
    return b, seen


@pytest.mark.parametrize("arm", [{"degradation": POST}, {"lap_noise": "residual"},
                                 {"degradation": POST, "lap_noise": "residual"}])
def test_draws_leave_hazard_and_event_draws_as_they_were(arm):
    base = dict(events={"name": "rate", "params": {"p_sc": 0.05, "p_vsc": 0.05}},
                hazard={"name": "age_logistic", "params": {"p_max": 0.2}}, n_samples=200)
    frame, snap = noisy_frame(), Snapshot.at_start(RACE, "X", "SOFT")
    off, ev_off = _recorded_events(Simulator(frame, SimConfig(**base)), snap, [ONE, TWO], seed=4)
    on, ev_on = _recorded_events(Simulator(frame, SimConfig(**base, **arm)), snap, [ONE, TWO], seed=4)
    assert ev_on == ev_off and len({e for e in ev_on}) > 1
    assert off.dnf.any() and (on.dnf == off.dnf).all() and (on.big_loss == off.big_loss).all()
    assert np.array_equal(on.dnf_lap, off.dnf_lap, equal_nan=True)
    assert not np.array_equal(on.times, off.times)


# --- (5) common random numbers ---------------------------------------------------------
def test_plans_sharing_a_first_stint_share_its_lap_times_with_both_draws_on():
    s = Simulator(noisy_frame(), SimConfig(degradation=POST, lap_noise="residual"))
    plans = [ONE, [["SOFT", 20], ["MEDIUM", 15], ["SOFT", 15]], [["SOFT", 20], ["SOFT", 10], ["MEDIUM", 20]]]
    first = []
    for seed in range(5):
        b = s.simulate_batch(Snapshot.at_start(RACE, "X", "SOFT"), plans, seed=seed, return_laps=True, n_samples=3)
        assert (b.lap_times[:, :20] == b.lap_times[0, :20]).all()
        assert (b.stint_seconds[0][0] == b.stint_seconds[1][0]).all()
        first.append(b.lap_times[0, :20].sum())
    assert len(set(first)) == 5   # and they vary from sample to sample


# --- (6) compare sees the parameter uncertainty ----------------------------------------
def test_large_wear_sd_widens_the_gap_and_pulls_p_beats_toward_half():
    f = noisy_frame(sd=3.0, seed=2)
    f = f[f["lap"] <= 12]   # few laps, noisy: a wide wear posterior
    snap = Snapshot.at_start(RACE, "X", "SOFT")
    out = {}
    for name, deg in (("none", {"name": "linear_fuel"}), ("posterior", POST)):
        c = Simulator(f, SimConfig(degradation=deg)).compare(snap, [ONE, TWO], n_samples=400)
        out[name] = (c.gap_std_s[0][1], c.p_beats[0][1])
    (std0, p0), (std1, p1) = out["none"], out["posterior"]
    assert std0 == 0.0 and p0 in (0.0, 1.0)
    assert std1 > 1.0 and abs(p1 - 0.5) < abs(p0 - 0.5)


# --- NaN covariance and clipping --------------------------------------------------------
def test_nan_covariance_rows_are_held_with_one_warning():
    m = Simulator(noisy_frame(), SimConfig(degradation=POST)).degradation
    names, cov = cov_of(m)
    cov[names.index("fuel_s_per_lap")] = np.nan
    m.fit_report = {**m.fit_report, "cov": {"names": names, "matrix": cov.tolist()}}
    with pytest.warns(NaNCovarianceWarning, match="fuel_s_per_lap") as rec:
        x = draws(m, n=50)
    assert len(rec) == 1
    assert np.all(x[:, names.index("fuel_s_per_lap")] == m.fuel_slope)
    assert x[:, names.index("wear_s_per_lap[SOFT]")].std() > 0


def test_clip_wear_at_zero_floors_the_drawn_wear():
    f = noisy_frame(sd=3.0, seed=2)
    f = f[f["lap"] <= 10]
    deg = {"name": "linear_fuel", "params": {"draw": "posterior", "clip_wear_at_zero": True}}
    m = Simulator(f, SimConfig(degradation=deg)).degradation
    names = m.fit_report["cov"]["names"]
    x = draws(m, n=500)
    wear = [names.index(n) for n in names if n.startswith("wear_s_per_lap")]
    assert (x[:, wear] >= 0).all() and (x[:, wear] == 0).any()
    unclipped = Simulator(f, SimConfig(degradation=POST)).degradation
    assert (draws(unclipped, n=500)[:, wear] < 0).any()


# --- (7) coverage on a race simulated from the model itself ----------------------------
def test_coverage_90_on_a_race_from_the_model_itself():
    """200 replications: race laps from the true model plus N(0, s2), fitted on the laps up
    to k, the remaining time of one car predicted with both draws on. Its 90 percent
    interval should hold the replicated remaining time about 90 percent of the time."""
    sd, k, total = 0.6, 20, 50
    race = RACE.model_copy(update={"total_laps": total})
    true = {"SOFT": (80.0, 0.10, 0.02), "MEDIUM": (82.0, 0.03, 0.0)}
    cfg = SimConfig(degradation=POST, lap_noise="residual", rules=["lap_sum", "known_compounds"],
                    effects=EventEffects(lap1_extra_s=0.0))
    starts = [0, 2, 4, 1, 3, 0]
    hits = []
    for rep in range(200):
        rng = np.random.default_rng([17, rep])
        rows = []
        for i, a0 in enumerate(starts):
            comp = "SOFT" if i % 2 == 0 else "MEDIUM"
            base, wear, _ = true[comp]
            for lap in range(1, k + 1):
                rows.append({"compound": comp, "tyre_age": a0 + lap, "lap": lap, "clean": True,
                             "lap_time_s": base + wear * (a0 + lap) + FUEL * lap + rng.normal(0, sd)})
        base, wear, _ = true["MEDIUM"]
        future = np.arange(k + 1, total + 1)
        real = float((base + wear * future + FUEL * future + rng.normal(0, sd, len(future))).sum())
        s = Simulator(pd.DataFrame(rows), cfg)
        snap = Snapshot(race=race, driver="X", laps_completed=k, compound="MEDIUM", tyre_age=k)
        b = s.simulate_batch(snap, [[["MEDIUM", total - k]]], n_samples=300, seed=rep)
        lo, hi = np.percentile(b.times[0], [5, 95])
        hits.append(lo <= real <= hi)
    assert 0.80 <= np.mean(hits) <= 0.97


# --- (D) intervals in the mid-race table -----------------------------------------------
def test_midrace_intervals_and_aggregate_coverage():
    frame = race_frame(("AAA", "T", [("SOFT", 20), ("MEDIUM", 30)]),
                       ("BBB", "T", [("SOFT", 10), ("MEDIUM", 25), ("SOFT", 15)]))
    frame = frame.assign(lap_time_s=frame["lap_time_s"] + np.random.default_rng(0).normal(0, 0.3, len(frame)))

    def make(window, **kw):
        return Simulator(frame, SimConfig(fit_window=window, pre_race_base_delta_s=0.0,
                                          effects=EventEffects(lap1_extra_s=0.0), **kw), synthetic_practice())
    plain = calibration.midrace(make, frame, RACE, laps_k=[10, 20], windows=["full"])
    assert list(plain.columns) == calibration.MIDRACE_COLUMNS
    agg = calibration.aggregate(plain, n_boot=10)
    assert not set(calibration.COVERAGE_COLUMNS) & set(agg.columns)

    rows = calibration.midrace(lambda w: make(w, degradation=POST, lap_noise="residual"), frame, RACE,
                               laps_k=[10, 20], windows=["full"], n_samples=100)
    assert list(rows.columns) == calibration.MIDRACE_COLUMNS + calibration.INTERVAL_COLUMNS
    q = rows[calibration.INTERVAL_COLUMNS].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (q[:, -1] > q[:, 0]).all()
    agg = calibration.aggregate(rows, n_boot=10)
    assert set(calibration.COVERAGE_COLUMNS) <= set(agg.columns)
    real = rows["real_remaining_s"] - rows["pace_offset_s"] * rows["horizon"]
    for (w, b), r in agg.iterrows():
        part = rows[(rows["horizon"] // 10 * 10 == b) & real.notna()]
        rr = real[part.index]
        assert r["coverage_90"] == pytest.approx(((part["pred_p05_s"] <= rr) & (rr <= part["pred_p95_s"])).mean())
        assert r["mean_width_90_s"] == pytest.approx((part["pred_p95_s"] - part["pred_p05_s"]).mean())
