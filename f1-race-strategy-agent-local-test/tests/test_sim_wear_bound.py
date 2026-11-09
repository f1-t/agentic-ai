"""SPEC 0017b: the physical lower bound on tyre wear (degradation.params wear_lower_bound).
Synthetic, offline."""
import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.contracts import Snapshot
from src.sim import SimConfig, Simulator
from src.sim import calibrators as _cal
from src.sim.degradation import DEGRADATION, DRAW_TRIES, _active_set
from src.sim.engine import DEGRADATION_RETRY_STREAM, DEGRADATION_STREAM, LAP_NOISE_STREAM, batch_normals
from test_sim import RACE, synthetic_practice
from test_sim_posterior import noisy_frame

SHRINK = SimConfig().shrinkage
PRIOR_FUEL, TRUE_FUEL = -0.045, -0.09
FLAT = {"SOFT": (80.0, 0.0), "MEDIUM": (82.0, 0.0)}
ONE, TWO = [["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]


def first_stints(k=20, wear=FLAT, fuel=TRUE_FUEL, sd=0.3, seed=0, starts=(0, 0, 1, 2, 3, 0, 4, 1)):
    """Every car on its first set up to lap k: tyre age and race lap move together."""
    rng = np.random.default_rng(seed)
    rows = []
    comps = list(wear)
    for i, a0 in enumerate(starts):
        comp = comps[i % len(comps)]
        base, w = wear[comp]
        for lap in range(1, k + 1):
            rows.append({"compound": comp, "tyre_age": a0 + lap, "lap": lap, "clean": True,
                         "lap_time_s": base + w * (a0 + lap) + fuel * lap + rng.normal(0, sd)})
    return pd.DataFrame(rows)


def flat_prior(wear=FLAT):
    """A practice fit whose wear is the truth (0): the prior is right about wear."""
    return DEGRADATION.get("linear_fuel").fit(synthetic_practice(coefs=wear), fuel_prior=PRIOR_FUEL,
                                              fuel_prior_sd=SHRINK["fuel_s_per_lap"])


def shrunk(bound=None, data=None, draw="none", name="linear_fuel", **kw):
    data = first_stints() if data is None else data
    return DEGRADATION.get(name).fit(data, prior=flat_prior(), shrinkage=SHRINK, fuel_prior=PRIOR_FUEL,
                                     wear_lower_bound=bound, draw=draw, **kw)


def wear_of(m):
    return {c: w for c, (_, w) in m.coefs.items()}


def draws(model, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([model.draw(rng).param_vector()[1] for _ in range(n)])


# --- A. the constrained fit -------------------------------------------------------------
def test_flat_wear_with_an_understated_fuel_prior_sits_on_the_bound():
    """True wear 0, a race speed-up the prior under-states: unconstrained, the speed-up the
    tight fuel tau will not take goes to wear, which turns negative; the bound puts it at 0
    exactly and the fuel slope moves toward the truth."""
    free, bounded = shrunk(), shrunk(0.0)
    assert all(w < 0 for w in wear_of(free).values())
    assert all(w == 0.0 for w in wear_of(bounded).values())
    assert TRUE_FUEL < bounded.fuel_slope < free.fuel_slope
    assert bounded.fit_report["bounded"] == {"MEDIUM": True, "SOFT": True}
    assert "bounded" not in free.fit_report and "unconstrained" not in free.fit_report


@pytest.mark.parametrize("seed", range(40))
def test_active_set_matches_brute_force_on_random_problems(seed):
    """Three wear coordinates among correlated columns, ridge or plain, against the best
    feasible fixed set over all 8 subsets."""
    rng = np.random.default_rng(seed)
    n, p = 40, 8
    X = rng.normal(size=(n, p))
    X[:, 1::2][:, :3] += 0.8 * X[:, [6]]   # the wear columns share a direction with column 6
    y = X @ rng.normal(0, 1, p) + rng.normal(0, 0.5, n)
    prec = np.where(rng.random(p) < 0.5, rng.uniform(0.1, 5.0, p), 0.0)
    b0 = rng.normal(0, 1, p) * (prec > 0)
    wear = np.zeros(p, dtype=bool)
    wear[[1, 3, 5]] = True
    bound = float(rng.normal(0, 0.5))
    H, rhs = X.T @ X + np.diag(prec), X.T @ y + prec * b0
    beta = np.linalg.solve(H, rhs)
    b, fixed = _active_set(X, y, prec, b0, beta, wear, bound, ridge=False)

    def obj(bb):
        r = y - X @ bb
        return r @ r + np.sum(prec * (bb - b0) ** 2)
    best = None
    for s in itertools.product([False, True], repeat=3):
        f = np.zeros(p, dtype=bool)
        f[np.flatnonzero(wear)[list(s)]] = True
        bb = np.full(p, bound)
        free = ~f
        bb[free] = np.linalg.solve(H[np.ix_(free, free)], rhs[free] - H[np.ix_(free, f)] @ bb[f])
        if np.all(bb[wear] >= bound - 1e-12) and (best is None or obj(bb) < obj(best)):
            best = bb
    assert np.all(b[wear] >= bound)
    assert b == pytest.approx(best, abs=1e-8)
    assert np.all(b[fixed] == bound)


@pytest.mark.parametrize("penalised", [False, True])
def test_three_compound_fit_matches_brute_force(penalised):
    """linear_fuel on SOFT / MEDIUM / HARD, unpenalised (full) and shrunk toward a prior:
    the bounded fit against every subset of wears fixed at 0."""
    wear = {"SOFT": (80.0, -0.04), "MEDIUM": (81.0, 0.01), "HARD": (82.0, -0.01)}
    data = first_stints(k=25, wear=wear, fuel=-0.06, sd=0.4, seed=3)
    comps = sorted(wear)
    d = data
    X = np.column_stack([c for comp in comps for c in ((d["compound"] == comp).to_numpy(float),
                                                       (d["compound"] == comp).to_numpy(float) * d["tyre_age"])]
                        + [d["lap"].to_numpy(float)])
    y = d["lap_time_s"].to_numpy(float)
    if penalised:
        prior = DEGRADATION.get("linear_fuel").fit(synthetic_practice(coefs={c: (b, 0.0) for c, (b, _) in wear.items()}),
                                                   fuel_prior=PRIOR_FUEL)
        b0 = np.array([v for c in comps for v in prior.coefs[c]] + [PRIOR_FUEL])
        prec = np.array([SHRINK["base_s"] ** -2, SHRINK["wear_s_per_lap"] ** -2] * 3 + [SHRINK["fuel_s_per_lap"] ** -2])
        m = DEGRADATION.get("linear_fuel").fit(data, prior=prior, shrinkage=SHRINK, fuel_prior=PRIOR_FUEL,
                                               wear_lower_bound=0.0)
    else:
        b0, prec = np.zeros(7), np.zeros(7)
        m = DEGRADATION.get("linear_fuel").fit(data, wear_lower_bound=0.0)
    H, rhs = X.T @ X + np.diag(prec), X.T @ y + prec * b0
    best, best_obj = None, np.inf
    for s in itertools.product([False, True], repeat=3):
        f = np.zeros(7, dtype=bool)
        f[[2 * i + 1 for i in range(3) if s[i]]] = True
        bb = np.zeros(7)
        bb[~f] = np.linalg.solve(H[np.ix_(~f, ~f)], rhs[~f])
        r = y - X @ bb
        o = r @ r + np.sum(prec * (bb - b0) ** 2)
        if np.all(bb[1::2][:3] >= -1e-12) and o < best_obj:
            best, best_obj = bb, o
    got = np.array([v for c in comps for v in m.coefs[c]] + [m.fuel_slope])
    assert got == pytest.approx(best, abs=1e-7)
    assert any(m.fit_report["bounded"].values())


@pytest.mark.parametrize("name", ["linear", "linear_fuel", "linear_fuel_hinge"])
@pytest.mark.parametrize("fit_window", ["full", "to_snapshot", "pre_race"])
def test_positive_wear_is_unchanged_bit_for_bit(name, fit_window):
    snap = Snapshot(race=RACE, driver="X", laps_completed=30, compound="SOFT", tyre_age=10)
    fits = []
    for params in ({}, {"wear_lower_bound": 0.0}):
        cfg = SimConfig(degradation={"name": name, "params": params}, fit_window=fit_window)
        fits.append(Simulator(noisy_frame(fuel=0.0), cfg, synthetic_practice()).fitted_for(snap).degradation)
    free, bounded = fits
    assert free.coefs == bounded.coefs and getattr(free, "fuel_slope", None) == getattr(bounded, "fuel_slope", None)
    assert getattr(free, "hinges", None) == getattr(bounded, "hinges", None)
    extra = {"bounded"}
    assert set(bounded.fit_report) - set(free.fit_report) == extra
    assert not any(bounded.fit_report["bounded"].values())
    same = {k: v for k, v in bounded.fit_report.items() if k not in extra}
    assert json.dumps(same, sort_keys=True, default=str) == json.dumps(free.fit_report, sort_keys=True, default=str)


@pytest.mark.parametrize("name", ["linear", "linear_fuel", "linear_fuel_hinge"])
def test_every_model_and_path_respects_the_bound(name):
    """Negative true SOFT wear: full, practice and to_snapshot fits all hold it at 0."""
    neg = {"SOFT": (80.0, -0.05), "MEDIUM": (82.0, 0.03)}
    race = first_stints(k=40, wear=neg, fuel=0.0 if name == "linear" else -0.04, sd=0.2, seed=1)
    full = DEGRADATION.get(name).fit(race, wear_lower_bound=0.0)
    practice = DEGRADATION.get(name).fit(synthetic_practice(coefs=neg), wear_lower_bound=0.0)
    to_snap = DEGRADATION.get(name).fit(race, prior=practice, shrinkage=SHRINK, fuel_prior=PRIOR_FUEL,
                                        wear_lower_bound=0.0)
    for m in (full, practice, to_snap):
        assert m.coefs["SOFT"][1] == 0.0 and m.fit_report["bounded"]["SOFT"]
        assert m.coefs["MEDIUM"][1] > 0 and not m.fit_report["bounded"]["MEDIUM"]
        assert DEGRADATION.get(name).fit(race if m is not practice else synthetic_practice(coefs=neg)).coefs["SOFT"][1] < 0


# --- B. the report ----------------------------------------------------------------------
def test_conditional_covariance_at_the_bound():
    """The free block is s2 (X'X + D)^-1 with the bounded rows and columns dropped from the
    precision; the bounded wear keeps its unconstrained variance, no covariance."""
    free, bounded = shrunk(), shrunk(0.0)
    names = bounded.fit_report["cov"]["names"]
    cov = np.array(bounded.fit_report["cov"]["matrix"])
    unc = bounded.fit_report["unconstrained"]
    cu = np.array(unc["matrix"])
    assert unc["names"] == names
    assert cu == pytest.approx(np.array(free.fit_report["cov"]["matrix"]), rel=1e-12)
    assert unc["mean"] == pytest.approx(free.param_vector()[1], rel=1e-12)
    w = [names.index(f"wear_s_per_lap[{c}]") for c in ("MEDIUM", "SOFT")]
    u = [i for i in range(len(names)) if i not in w]
    for j in w:
        assert cov[j, j] == cu[j, j] and not np.any(np.delete(cov[j], j))
    s2u, s2c = free.fit_report["sd"]["residual_s"] ** 2, bounded.fit_report["sd"]["residual_s"] ** 2
    prec_u = np.linalg.inv(cu)[np.ix_(u, u)] * s2u
    assert cov[np.ix_(u, u)] == pytest.approx(s2c * np.linalg.inv(prec_u), rel=1e-8)
    sd = bounded.fit_report["sd"]
    assert sd["wear_s_per_lap"]["SOFT"] == pytest.approx(np.sqrt(cu[w[1], w[1]]))
    f = names.index("fuel_s_per_lap")
    assert sd["fuel_s_per_lap"] == pytest.approx(np.sqrt(cov[f, f]))


def test_model_report_flags_wear_at_bound():
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20)
    frame = first_stints(wear={"SOFT": (80.0, -0.05), "MEDIUM": (82.0, 0.03)}, fuel=-0.04)
    flags = {}
    for params in ({}, {"wear_lower_bound": 0.0}):
        s = Simulator(frame, SimConfig(degradation={"name": "linear_fuel", "params": params}))
        flags[bool(params)] = s.model_report(snap, calibration_summary=None).flags
    assert "wear_at_bound:SOFT" in flags[True] and "wear_at_bound:MEDIUM" not in flags[True]
    assert not any(f.startswith("wear_at_bound") for f in flags[False])


# --- C. draws under the bound -------------------------------------------------------------
def test_draws_respect_the_bound_and_carry_steeper_fuel():
    """A noisier, shorter frame than the point-estimate test: the wears sit within about one
    sd below 0, so a fair share of the proposals is accepted."""
    data = first_stints(k=15, sd=1.5, seed=0)
    free, bounded = shrunk(data=data, draw="posterior"), shrunk(0.0, data=data, draw="posterior")
    assert all(bounded.fit_report["bounded"].values())
    names = bounded.fit_report["cov"]["names"]
    w = [names.index(n) for n in names if n.startswith("wear_s_per_lap[")]
    f = names.index("fuel_s_per_lap")
    xf, xb = draws(free), draws(bounded)
    assert (xf[:, w] < 0).any() and (xb[:, w] >= 0).all()
    assert xb[:, f].mean() < xf[:, f].mean()
    st = bounded.fit_report["draw_stats"]
    assert st["n_samples"] == 2000 and st["n_proposals"] >= 2000
    assert st["acceptance_rate"] == pytest.approx((st["n_samples"] - st["n_clipped"]) / st["n_proposals"])
    assert 0 < st["acceptance_rate"] < 1


def test_all_rejected_falls_back_to_the_clipped_last_proposal():
    m = shrunk(0.0, draw="posterior")
    names = m.fit_report["cov"]["names"]
    i = names.index("wear_s_per_lap[SOFT]")
    L = np.linalg.cholesky(np.array(m.fit_report["unconstrained"]["matrix"]))
    z = -50.0 * L[i] / np.linalg.norm(L[i])   # pushes SOFT wear far below 0
    out = m.draw_z(np.tile(z, DRAW_TRIES))
    assert out.coefs["SOFT"][1] == 0.0
    st = m.fit_report["draw_stats"]
    assert (st["n_samples"], st["n_proposals"], st["n_clipped"]) == (1, DRAW_TRIES, 1)
    assert st["acceptance_rate"] == 0.0


def test_an_accepted_first_proposal_is_the_unbounded_draw():
    """Positive wear, nothing on the bound: the same Gaussian, and the first proposal is
    DEGRADATION_STREAM's row, so a sample accepted at once draws what it draws unbounded."""
    snap = Snapshot.at_start(RACE, "X", "SOFT")
    out = {}
    for params in ({"draw": "posterior"}, {"draw": "posterior", "wear_lower_bound": 0.0}):
        f = noisy_frame(sd=3.0, seed=3)
        s = Simulator(f[f["lap"] <= 10], SimConfig(degradation={"name": "linear_fuel", "params": params}))
        out[len(params)] = (s, s.simulate_batch(snap, [ONE, TWO], seed=3, n_samples=200).times)
    (free_sim, a), (sim, b) = out[1], out[2]
    z = batch_normals(3, DEGRADATION_STREAM, 200, sim.degradation.n_params)
    names = sim.degradation.fit_report["cov"]["names"]
    w = [names.index(n) for n in names if n.startswith("wear_s_per_lap[")]
    L = np.linalg.cholesky(np.array(sim.degradation.fit_report["cov"]["matrix"]))
    first_ok = np.all((sim.degradation.param_vector()[1] + z @ L.T)[:, w] >= 0, axis=1)
    assert first_ok.any() and not first_ok.all()
    assert np.array_equal(a[:, first_ok], b[:, first_ok])
    assert not np.array_equal(a[:, ~first_ok], b[:, ~first_ok])
    st = sim.degradation.fit_report["draw_stats"]
    assert st["n_samples"] == 200 and st["n_proposals"] > 200


def test_bounded_batch_rows_do_not_depend_on_the_batch_size():
    assert DEGRADATION_RETRY_STREAM not in (DEGRADATION_STREAM, LAP_NOISE_STREAM, 1)
    s = Simulator(first_stints(), SimConfig(degradation={"name": "linear_fuel",
                                                         "params": {"draw": "posterior", "wear_lower_bound": 0.0}},
                                            lap_noise="residual"))
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20)
    a, b = (s.simulate_batch(snap, [[["SOFT", 10], ["MEDIUM", 20]]], seed=7, n_samples=n).times for n in (10, 200))
    assert np.array_equal(a, b[:, :10])


def test_held_prior_drops_the_unconstrained_posterior_and_counts_its_own_draws():
    neg = {"SOFT": (80.0, -0.05), "MEDIUM": (82.0, 0.03)}
    practice = DEGRADATION.get("linear_fuel").fit(synthetic_practice(coefs=neg), draw="posterior",
                                                  wear_lower_bound=0.0, fuel_prior_sd=0.008)
    assert "unconstrained" in practice.fit_report
    s = Simulator(first_stints(), SimConfig(fit_window="pre_race", degradation={
        "name": "linear_fuel", "params": {"draw": "posterior", "wear_lower_bound": 0.0}}),
        synthetic_practice(coefs=neg))
    fit = s.fitted_for(Snapshot.at_start(RACE, "X", "SOFT"))
    m = fit.degradation
    assert "unconstrained" not in m.fit_report and m.fit_report["bounded"]["SOFT"]
    assert m.fit_report["draw_stats"]["n_samples"] == 0
    x = draws(m, n=300)
    names = m.fit_report["cov"]["names"]
    assert (x[:, names.index("wear_s_per_lap[SOFT]")] >= 0).all()
    assert m.fit_report["draw_stats"]["n_samples"] == 300
    held = _cal.held_prior(practice, fit.calibration.output)
    assert held.fit_report["draw_stats"] is not practice.fit_report["draw_stats"]


# --- defaults -----------------------------------------------------------------------------
def test_default_none_is_the_unbounded_fit_and_draw():
    assert "wear_lower_bound" not in SimConfig().degradation.params
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20)
    frame = first_stints()
    out = []
    for params in ({"draw": "posterior"}, {"draw": "posterior", "wear_lower_bound": None}):
        s = Simulator(frame, SimConfig(degradation={"name": "linear_fuel", "params": params}, lap_noise="residual",
                                       fit_window="to_snapshot"), synthetic_practice(coefs=FLAT))
        fit = s.fitted_for(snap)
        assert fit.degradation.draw_tries == 1 and fit.degradation.wear_lower_bound is None
        assert not {"bounded", "unconstrained", "draw_stats"} & set(fit.degradation.fit_report)
        out.append((json.dumps(fit.degradation.fit_report, sort_keys=True, default=str),
                    s.simulate_batch(snap, [[["SOFT", 10], ["MEDIUM", 20]]], seed=2, n_samples=50).times))
    assert out[0][0] == out[1][0] and np.array_equal(out[0][1], out[1][1])


def test_belief_config_sets_the_bound():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = SimConfig.from_file(os.path.join(root, "configs", "sim", "belief.yaml"))
    assert cfg.degradation.params["wear_lower_bound"] == 0.0
    assert cfg.degradation.params["draw"] == "posterior" and cfg.lap_noise == "residual"
