"""The per-circuit race-pace prior for the fuel term (SPEC 0020): the target, the four
candidates as calibrators, as-of discipline, recovery of a planted circuit effect, the
no-history fallback, the factor reaching the fit as prior mean and tau, the selection rule,
and the synthetic report. Offline, synthetic histories only."""
import math

import numpy as np
import pandas as pd
import pytest

from src.contracts import RaceParams
from src.sim import SimConfig, Simulator, calibration, calibrators, cross_race, fuel_prior as fp, race_ids
from src.sim.calibrators import CALIBRATORS
from src.sim.compute import load_profile

SEASONS = range(2018, 2025)


@pytest.fixture(scope="module")
def history():
    return fp.synthetic_history(SEASONS)


@pytest.fixture(scope="module")
def features(history):
    return fp.synthetic_features(list(history["race_id"]))


def race(rid, laps=60):
    return RaceParams(race_id=rid, total_laps=laps, pit_loss_s=21.0)


def predict(candidate, train, rid, laps=60, **kw):
    return fp.make_calibrator(candidate, **kw).fit(train).prediction(race(rid, laps))


# --- B: registry -----------------------------------------------------------------------------------
def test_the_candidates_are_registered_calibrators():
    for c, name in fp.CALIBRATOR_OF.items():
        cal = CALIBRATORS.get(name)()
        assert cal.name == name and cal.candidate == c
    assert CALIBRATORS.get("fuel_circuit_history").learned and not CALIBRATORS.get("fuel_practice_longrun").learned
    assert SimConfig().calibrator.name == "shrinkage"   # defaults unchanged


def test_config_candidate_is_the_shrinkage_output(history):
    data = fp.synthetic_history_race(fp.synthetic_id(2020, "c3"))
    k = 20
    snap = calibration.snapshot_at(data.frame, data.race, "D03", k)
    base = data.config.model_copy(update={"fit_window": "to_snapshot"})
    a = Simulator(data.frame, base, data.practice).fitted_for(snap)
    b = Simulator(data.frame, base, data.practice, calibrator=CALIBRATORS.get("fuel_config")()).fitted_for(snap)
    assert a.calibration.output.factors == b.calibration.output.factors
    assert b.calibration.output.factors["fuel_s_per_lap"] == (-0.045, 0.008)
    assert a.degradation.coefs == b.degradation.coefs and a.degradation.fuel_slope == b.degradation.fuel_slope


# --- A: the target ---------------------------------------------------------------------------------
def test_target_is_the_free_air_full_window_fuel_slope():
    rid = fp.synthetic_id(2019, "c4")
    data = fp.synthetic_history_race(rid)
    slope, sd = fp.race_target(data)
    cfg = data.config.model_copy(update={"fit_window": "full", "fit_free_air_only": True})
    deg = Simulator(data.frame, cfg, data.practice).degradation
    assert slope == deg.fuel_slope and sd == deg.fit_report["sd"]["fuel_s_per_lap"] and 0 < sd < 0.01
    assert slope == pytest.approx(fp.planted(rid)["slope"], abs=0.004)


def test_synthetic_race_defaults_are_unchanged_and_evolution_is_opt_in():
    a = cross_race.synthetic_race("2099_01_synthetic")
    b = cross_race.synthetic_race("2099_01_synthetic", fuel=cross_race.SYN_FUEL, evolution_s_per_min=None)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    pd.testing.assert_frame_equal(a.practice, b.practice)
    assert "stint_start_min" not in a.practice
    c = cross_race.synthetic_race("2099_01_synthetic", evolution_s_per_min=-0.01)
    pd.testing.assert_frame_equal(a.frame, c.frame)
    assert c.practice["stint_start_min"].between(5, 55).all()
    np.testing.assert_allclose(c.practice["lap_time_s"] - a.practice["lap_time_s"], -0.01 * c.practice["stint_start_min"])


# --- E: as-of discipline ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate", ["circuit_history", "circuit_features"])
def test_a_held_out_2022_race_never_sees_its_own_or_later_races(history, features, candidate):
    rid = fp.synthetic_id(2022, "c3")   # round 3: 2022 rounds 1 and 2 are known on the day
    seen = fp.as_of(history, rid)
    assert set(seen["season"]) == {2018, 2019, 2020, 2021, 2022}
    assert set(seen.loc[seen["season"] == 2022, "round"]) == {1, 2}
    assert list(seen["race_id"]) == race_ids.as_of(rid, list(history["race_id"]))
    later = (history["season"] > 2022) | ((history["season"] == 2022) & (history["round"] >= 3))
    want = predict(candidate, history[~later], rid, 66, features_path=features)
    # the held-out race, its season's later rounds and every later season, with absurd targets: nothing moves
    poisoned = history.assign(target=np.where(later, 5.0, history["target"]))
    got = predict(candidate, poisoned, rid, 66, features_path=features)
    assert got == want and got["source"] in ("circuit history", "circuit features")
    assert abs(got["mean"]) < 0.2
    # the same season's earlier rounds do enter the grand mean
    assert predict(candidate, history.assign(target=np.where(history["race_id"] == fp.synthetic_id(2022, "c1"), 5.0,
                                                             history["target"])), rid, 66, features_path=features) != want


def test_circuit_history_shrinks_toward_the_grand_mean(history):
    rid = fp.synthetic_id(2021, "c1")
    train = fp.as_of(history, rid)
    p = predict("circuit_history", history, rid, 58)
    at = train[train["circuit"] == "synthetic_c1"]["target"]
    n, grand = len(at), train["target"].mean()
    w = n / (n + fp.K0)
    assert n == 3 and p["n_history"] == 3 and p["weight"] == pytest.approx(w)
    assert p["mean"] == pytest.approx(w * at.mean() + (1 - w) * grand)
    assert p["sd"] == pytest.approx(math.sqrt(at.std() ** 2 + (1 - w) * train["target"].std() ** 2))


# --- E: recovery -----------------------------------------------------------------------------------
def test_a_planted_circuit_effect_is_recovered(history):
    t = fp.recovery_table(history)
    assert len(t) == len(fp.SYN_CIRCUITS) and (t["races"] == len(SEASONS)).all()
    assert (t["recovered_effect"] - t["planted_effect"]).abs().max() < 0.004
    assert np.corrcoef(t["planted_effect"], t["recovered_effect"])[0, 1] > 0.98
    # shrinkage: the recovered effect is pulled toward 0 by n / (n + K0)
    assert (t["recovered_effect"].abs() < t["planted_effect"].abs() + 0.003).all()


def test_the_planted_effect_is_recovered_from_simulated_races():
    races = [fp.synthetic_history_race(fp.synthetic_id(s, c)) for s in (2018, 2019, 2020) for c in ("c1", "c5")]
    h = fp.history_from_races(races, fp.synthetic_features([r.race_id for r in races]))
    for c in ("c1", "c5"):
        got = fp.circuit_history_prediction(fp.as_of(h, fp.synthetic_id(2021, c)), f"synthetic_{c}")
        assert got["n_history"] == 3
        assert got["mean"] == pytest.approx(fp.planted(fp.synthetic_id(2021, c))["circuit_slope"], abs=0.005)
    assert h.loc[h["circuit"] == "synthetic_c1", "target"].mean() < h.loc[h["circuit"] == "synthetic_c5", "target"].mean()


# --- E: the no-history fallback --------------------------------------------------------------------
def test_no_history_falls_back(history):
    # a new circuit: the fold's grand mean, sd its between-race spread
    p = predict("circuit_history", history, "2023_09_new_circuit", 55)
    train = fp.as_of(history, "2023_09_new_circuit")
    assert p["source"].startswith("grand mean") and p["n_history"] == 0
    assert p["mean"] == pytest.approx(train["target"].mean()) and p["sd"] == pytest.approx(train["target"].std())
    # no earlier race at all (round 1 of the first season), an empty history, or an id without a season: the config prior
    for train, rid in ((history, fp.synthetic_id(2018, "c1")), (history.iloc[:0], fp.synthetic_id(2022, "c1")),
                       (history, "syn")):
        for c in ("circuit_history", "circuit_features"):
            q = predict(c, train, rid)
            assert (q["mean"], q["sd"]) == (-0.045, 0.008) and q["source"].startswith("config")
    # an unfitted calibrator has no history either
    q = CALIBRATORS.get("fuel_circuit_history")().prediction(race(fp.synthetic_id(2022, "c1")))
    assert (q["mean"], q["sd"]) == (-0.045, 0.008)


def test_circuit_features_needs_twelve_races_with_every_covariate(history):
    rid = fp.synthetic_id(2020, "c1")   # round 1 of 2020: the 12 races of 2018-2019 before it
    p = predict("circuit_features", history, rid, 58)
    assert p["source"] == "circuit features" and p["n_history"] == 12 and p["alpha"] in fp.ALPHAS
    assert -0.1 < p["mean"] < -0.03 and p["sd"] > 0
    q = predict("circuit_features", history[history["race_id"] != fp.synthetic_id(2019, "c6")], rid, 58)
    assert q["source"].startswith("config")
    nofeat = history.assign(turns=np.nan)
    assert predict("circuit_features", nofeat, rid, 58)["source"].startswith("config")


def test_ridge_inner_loro_picks_the_lowest_mse_alpha():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([0.01, -0.02, 0.0]) + rng.normal(0, 0.001, 20)
    alpha, rmse = fp.ridge_inner_loro(X, y)
    assert alpha == min(fp.ALPHAS) and rmse < 0.003
    alpha, _ = fp.ridge_inner_loro(X, rng.normal(0, 0.01, 20))   # pure noise: the most shrinkage
    assert alpha == max(fp.ALPHAS)


def test_practice_longrun_constants_and_evolution():
    r = race("2021_04_spain", 66)
    p = fp.practice_longrun_prediction(r)
    assert p["mean"] == pytest.approx(-0.03 * 110 / 66) and p["evolution_s_per_lap"] == 0.0
    assert p["sd"] == pytest.approx(math.hypot(0.005 * 110 / 66, 0.005))
    data = fp.synthetic_history_race(fp.synthetic_id(2020, "c2"))
    raw = fp._physics_prior(data)
    evo = raw.fit_report["evolution_s_per_min"]
    q = fp.practice_longrun_prediction(data.race, raw, data.practice)
    lap_min = data.practice["lap_time_s"].median() / 60
    assert set(evo) == {"FP2", "FP3"}
    assert q["evolution_s_per_lap"] == pytest.approx(np.mean(list(evo.values())) * lap_min)
    assert q["mean"] == pytest.approx(-0.03 * 110 / 70 + q["evolution_s_per_lap"])


# --- E: the factor reaches the fit as prior mean and tau ---------------------------------------------
class FixedFuel(fp.FuelPriorCalibrator):
    name = "fuel_fixed_test"

    def __init__(self, mean, sd):
        super().__init__()
        self.value = (mean, sd)

    def predict(self, race, config, practice_model=None, practice=None):
        return {"mean": self.value[0], "sd": self.value[1], "source": "fixed", "n_history": 0}


def test_the_factor_reaches_the_fit_as_prior_mean_and_tau():
    data = fp.synthetic_history_race(fp.synthetic_id(2020, "c4"))
    snap = calibration.snapshot_at(data.frame, data.race, "D05", 12)
    cfg = data.config.model_copy(update={"fit_window": "to_snapshot"})
    tight = Simulator(data.frame, cfg, data.practice, calibrator=FixedFuel(-0.07, 1e-3)).fitted_for(snap)
    out = tight.calibration.output
    assert out.factors["fuel_s_per_lap"] == (-0.07, 1e-3) and out.diagnostics["taus"]["fuel_s_per_lap"] == 1e-3
    prior, taus, fuel, held = calibrators.degradation_prior(out, tight.calibration.practice_model, cfg)
    assert fuel == -0.07 and taus["fuel_s_per_lap"] == 1e-3 and not held
    assert tight.degradation.fuel_slope == pytest.approx(-0.07, abs=2e-3)
    # an sd under MIN_SD is floored there: a 0 would hold the fuel term alone
    assert FixedFuel(-0.07, 0.0).prediction(data.race)["sd"] == fp.MIN_SD
    assert tight.degradation.fit_report["fuel_slope_source"] == "shrunk"
    # the same fit by hand: the prior model, the taus and the fuel prior the calibrator gave
    race_rows = data.frame[data.frame["lap"] <= 12]
    by_hand = tight._model(race_rows, prior=prior, shrinkage=taus, fuel_prior=fuel)
    assert by_hand.fuel_slope == tight.degradation.fuel_slope and by_hand.coefs == tight.degradation.coefs
    # a loose tau lets the race move it; pre_race holds the mean
    loose = Simulator(data.frame, cfg, data.practice, calibrator=FixedFuel(-0.07, 0.5)).fitted_for(snap)
    assert abs(loose.degradation.fuel_slope + 0.07) > abs(tight.degradation.fuel_slope + 0.07)
    pre = Simulator(data.frame, cfg.model_copy(update={"fit_window": "pre_race"}), data.practice,
                    calibrator=FixedFuel(-0.07, 0.002)).fitted_for(calibration.snapshot_at(data.frame, data.race,
                                                                                            "D05", 0))
    assert pre.degradation.fuel_slope == -0.07
    assert calibrators.effective_factors(pre, data.race)["fuel_s_per_lap"] == (-0.07, 0.002)


def test_a_fuel_calibrator_runs_through_the_registry_loro():
    races = [fp.synthetic_history_race(fp.synthetic_id(s, "c6")) for s in (2018, 2019, 2020)]
    ds = calibrators.build_dataset(races, laps_k=[0, 20])
    held = fp.synthetic_id(2020, "c6")
    res = calibrators.loro(ds, "fuel_circuit_history", held, windows=("pre_race",))
    fuel = res.factor_errors[res.factor_errors["factor"] == "fuel_s_per_lap"]
    want = fp.circuit_history_prediction(fp.history_from_races(races[:2]), "synthetic_c6")
    assert len(fuel) and fuel["mean"].eq(want["mean"]).all() and fuel["sd"].eq(want["sd"]).all()


# --- C: the selection rule -----------------------------------------------------------------------------
def _pred(errs: dict, sd=0.01):
    rows = [{"race_id": f"r{i}", "candidate": c, "err": e, "sd": sd, "covered": abs(e) <= sd,
             "source": "config" if c == "config" else c} for c, es in errs.items() for i, e in enumerate(es)]
    return pd.DataFrame(rows)


def test_a_candidate_replaces_config_only_with_its_ci_below_zero():
    rng = np.random.default_rng(1)
    base = rng.normal(0.015, 0.004, 12)
    pred = _pred({"config": base, "circuit_history": base * 0.3, "practice_longrun": base + rng.normal(0, 0.01, 12),
                  "circuit_features": base * 0.3})
    sel = fp.target_selection(pred, n_boot=500)
    s = sel.set_index("candidate")
    assert bool(s.loc["circuit_history", "passes"]) and s.loc["circuit_history", "diff_hi"] < 0
    assert not bool(s.loc["practice_longrun", "passes"]) and not bool(s.loc["config", "passes"])
    f = sel.rename(columns={"mae": "held_out_mae"}).assign(stop_gain_mae=[1.0, 0.9, 0.8, 1.0])
    assert fp.choose(f) == "circuit_features"   # a tie on mae: the lower stop-gain mae
    assert fp.choose(f.assign(stop_gain_mae=1.0)) == "circuit_history"   # then the CANDIDATES order
    assert fp.choose(f.assign(passes=False)) == "config"


# --- F: the report ----------------------------------------------------------------------------------------
def small_profile():
    p = load_profile("local")
    return p.model_copy(update={"g": p.g.model_copy(update={"laps_every": 20, "sc_adjacent": False})})


def test_synthetic_report_end_to_end(tmp_path):
    circuits = {k: fp.SYN_CIRCUITS[k] for k in ("c1", "c3")}
    path = fp.run_synthetic(tmp_path, seasons=range(2018, 2021), circuits=circuits, profile=small_profile(),
                            n_boot=50, log=None)
    md = path.read_text()
    for name in ("history.csv", "predictions.csv", "selection.csv", "f_table.csv", "g_rows.csv", "stop_gain.csv",
                 "g_by_horizon.csv", "recovery.csv", "prior_share.csv"):
        assert (tmp_path / name).is_file()
    # the share of error held by construction on synthetic histories stays out of the report text
    assert not (tmp_path / "sentence.txt").exists() and "percent of that absolute error" not in md
    f = pd.read_csv(tmp_path / "f_table.csv")
    assert list(f["candidate"]) == ["config", "circuit_history", "practice_longrun", "hindsight"]
    assert "circuit_features: **unavailable**" in md and "Synthetic histories" in md
    assert {"g_bias_20-40_pre_race", "g_bias_50-60_to_snapshot", "stop_gain_mae", "coverage_68"} <= set(f.columns)
    g = pd.read_csv(tmp_path / "g_rows.csv")
    assert set(g["window"]) == {"pre_race", "to_snapshot"} and g["race_id"].nunique() == 6
    pred = pd.read_csv(tmp_path / "predictions.csv")
    hist = pred[pred["candidate"] == "circuit_history"].set_index("race_id")["source"]
    assert hist[fp.synthetic_id(2018, "c1", circuits)].startswith("config")   # nothing before round 1 of 2018
    # one earlier race gives a mean but no spread: still the config prior
    assert hist[fp.synthetic_id(2018, "c3", circuits)].startswith("config")
    assert hist[fp.synthetic_id(2019, "c1", circuits)] == "circuit history"


def test_synthetic_reports_never_go_under_reports_calibration(tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        fp.run_synthetic(fp.OUT_DIR, seasons=range(2018, 2019))
    with pytest.raises(ValueError, match="synthetic"):
        fp.run([fp.synthetic_id(2018, "c1")], fp.REPORTS_DIR / "elsewhere")
    assert not fp.OUT_DIR.exists()
