"""RidgeCalibrator (SPEC 0024 step 2): unit checks on a known linear map, the leakage check
through loro, the training log's mode, and the cross-race pipeline with a ridge arm. Synthetic
races only, offline; WANDB_MODE is disabled for every test (conftest.py)."""
import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import yaml

from src.contracts import split_factor
from src.sim import SimConfig, Simulator, calibration, calibrators, compute, cross_race
from src.sim.calibrators import CALIBRATORS, CalibrationDataset, RidgeCalibrator
from src.sim.compute import load_profile

TRUE_A, TRUE_B = 21.0, 1.5      # factor = a + b * feature + noise
NOISE_SD = 0.4
SYN = [f"2099_{i:02d}_synthetic" for i in range(1, 7)]


def linear_rows(n_races, rng, b=TRUE_B, rows_per_race=1, extra=None):
    """One target per race (a hindsight factor is per race), `rows_per_race` rows of it."""
    x = rng.normal(0.0, 1.0, n_races)
    y = TRUE_A + b * x + rng.normal(0.0, NOISE_SD, n_races)
    rows = []
    for i in range(n_races):
        for k in range(rows_per_race):
            row = {"race_id": f"2021_{i + 1:03d}_r", "season": 2021, "k": 10 * k, "feature": x[i],
                   "target:pit_loss_s": y[i]}
            if extra is not None:
                row.update(extra(rng))
            rows.append(row)
    return pd.DataFrame(rows)


def fit(rows, **kw) -> RidgeCalibrator:
    return CALIBRATORS.get("ridge")(**kw).fit(CalibrationDataset(rows, {}))


# --- unit: a known linear factor ---------------------------------------------------------------
def test_ridge_recovers_a_known_linear_coefficient():
    rng = np.random.default_rng(1)
    cal = fit(linear_rows(200, rng, rows_per_race=2))
    rep = cal.report()
    b = rep["coef"]["pit_loss_s"][0]
    f = rep["factors"]["pit_loss_s"]
    assert rep["factors"]["pit_loss_s"]["design"] == ["feature"]
    assert b == pytest.approx(TRUE_B, abs=0.08)
    assert f["intercept"] == pytest.approx(TRUE_A, abs=0.08)
    assert f["sigma"] == pytest.approx(NOISE_SD, rel=0.15)
    assert f["alpha"] in calibrators.RIDGE_ALPHAS and f["n_races"] == 200 and f["n_rows"] == 400


def test_ridge_68_percent_intervals_cover_about_68_percent_of_held_out_rows():
    rng = np.random.default_rng(2)
    cal = fit(linear_rows(150, rng))
    held = linear_rows(4000, np.random.default_rng(3))
    mean, sd = cal.predict(held)["pit_loss_s"]
    covered = np.abs(held["target:pit_loss_s"].to_numpy() - mean) <= sd
    assert covered.mean() == pytest.approx(0.68, abs=0.04)
    assert (sd > NOISE_SD * 0.8).all() and (sd < NOISE_SD * 1.3).all()


def test_ridge_with_a_pure_noise_feature_predicts_the_training_mean():
    """The feature carries nothing: the inner LORO picks heavy shrinkage, so predictions sit at
    the training mean, much closer than least squares'. Over 10 draws, as one draw can find a
    spurious correlation the inner LORO believes."""
    heavy, spread, ols_spread, off = 0, [], [], []
    for seed in range(10):
        rows = linear_rows(150, np.random.default_rng(seed), b=0.0)
        held = linear_rows(1000, np.random.default_rng(100 + seed), b=0.0)
        cal, ols = fit(rows), fit(rows, alphas=(1e-8,))
        heavy += cal.report()["factors"]["pit_loss_s"]["alpha"] >= 100.0
        mean, _ = cal.predict(held)["pit_loss_s"]
        assert abs(cal.report()["coef"]["pit_loss_s"][0]) <= abs(ols.report()["coef"]["pit_loss_s"][0])
        spread.append(mean.std())
        ols_spread.append(ols.predict(held)["pit_loss_s"][0].std())
        off.append(abs(mean.mean() - rows["target:pit_loss_s"].mean()))
    assert heavy >= 7
    assert np.median(spread) < 0.03 * NOISE_SD and np.median(spread) < 0.25 * np.median(ols_spread)
    assert max(off) < 0.05 * NOISE_SD


def test_missing_values_are_median_imputed_with_an_indicator_from_the_training_rows():
    rng = np.random.default_rng(6)
    rows = linear_rows(60, rng)
    rows.loc[rows.index[:10], "feature"] = np.nan
    rows["all_missing"] = np.nan                     # no value anywhere: dropped
    rows["text"] = "street"                          # not numeric: missing everywhere, dropped
    cal = fit(rows)
    rep = cal.report()["factors"]["pit_loss_s"]
    assert rep["design"] == ["feature", "missing:feature"]
    m = cal.models["pit_loss_s"]
    assert m["median"][0] == pytest.approx(rows["feature"].median())
    # a held-out row lacking the column is imputed, not dropped
    mean, sd = cal.predict(pd.DataFrame({"other": [1.0]}))["pit_loss_s"]
    assert np.isfinite(mean).all() and np.isfinite(sd).all()


def test_a_factor_with_under_5_training_races_falls_back_and_is_reported():
    rng = np.random.default_rng(7)
    rows = linear_rows(12, rng)
    rows["target:sc_lap_factor"] = np.where(rows["race_id"].isin(rows["race_id"].unique()[:4]), 1.35, np.nan)
    cal = fit(rows)
    assert set(cal.models) == {"pit_loss_s"}
    assert cal.fallback == {"sc_lap_factor": "4 training race(s), under 5"}
    table = {r["factor"]: r for r in cal.fit_table()}
    assert table["sc_lap_factor"]["status"] == "fallback" and table["pit_loss_s"]["status"] == "ridge"
    assert cal.metrics()["n_fallback"] == 1
    with pytest.raises(ValueError, match="training set is empty"):
        fit(rows.iloc[:0])


def test_save_and_load_round_trip_the_fit(tmp_path):
    rng = np.random.default_rng(8)
    cal = fit(linear_rows(40, rng), alphas=(0.1, 10.0))
    cal.save(tmp_path / "ridge.json")
    saved = json.loads((tmp_path / "ridge.json").read_text())
    assert saved["name"] == "ridge" and saved["params"]["alphas"] == [0.1, 10.0]
    assert {"features", "models", "fallback"} <= set(saved["state"])
    assert {"beta", "scale", "mean", "median", "alpha", "sigma2", "a_inv"} <= set(saved["state"]["models"]["pit_loss_s"])
    back = calibrators.Calibrator.load(tmp_path / "ridge.json")
    assert isinstance(back, RidgeCalibrator) and back.fitted and back.alphas == (0.1, 10.0)
    held = linear_rows(50, np.random.default_rng(9))
    for a, b in zip(cal.predict(held)["pit_loss_s"], back.predict(held)["pit_loss_s"]):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    assert back.report() == cal.report()


# --- calibrate: the output contract on synthetic races -------------------------------------------
@pytest.fixture(scope="module")
def syn_dataset():
    races = [cross_race.synthetic_race(r) for r in SYN]
    return calibrators.build_dataset(races, laps_k=[0, 15, 30])


def test_calibrate_gives_shrinkage_contract_with_floored_uniform_sds(syn_dataset):
    held = SYN[0]
    cal = CALIBRATORS.get("ridge")().fit(syn_dataset.subset(SYN[1:]))
    r = syn_dataset.races[held]
    for window, k in (("pre_race", 0), ("to_snapshot", 15)):
        sim = Simulator(r.frame, r.config.model_copy(update={"fit_window": window}), r.practice, calibrator=cal)
        fitted = sim.fitted_for(calibration.snapshot_at(r.frame, r.race, "D00", k))
        out = fitted.calibration.output          # the engine took it: degradation_prior read the sds as taus
        assert out.source == "ridge" and out.valid_until_lap >= k
        tau = {"hinge_s_per_lap": calibrators.HINGE_TAU_S_PER_LAP, **r.config.shrinkage}
        for kind in calibrators.DEGRADATION_FACTORS:
            sds = {sd for f, (_, sd) in out.factors.items() if split_factor(f)[0] == kind}
            assert len(sds) <= 1                                     # one prior sd per factor type
            if sds:
                assert sds.pop() >= tau[calibrators.TAU_KEYS[kind]]  # never more certain than shrinkage
        assert set(out.diagnostics["ridge"]) >= {"wear_s_per_lap[SOFT]", "fuel_s_per_lap", "pit_loss_s"}
        assert out.factors["pit_loss_s"][1] > 0          # an event factor ridge fitted: drawn per sample
        shrink = CALIBRATORS.get("shrinkage")()
        ref = Simulator(r.frame, r.config.model_copy(update={"fit_window": window}), r.practice, calibrator=shrink)
        ref_out = ref.fitted_for(calibration.snapshot_at(r.frame, r.race, "D00", k)).calibration.output
        assert set(ref_out.factors) <= set(out.factors)  # every factor shrinkage gives, ridge gives


def test_unfitted_ridge_refuses_to_calibrate():
    r = cross_race.synthetic_race(SYN[0])
    sim = Simulator(r.frame, r.config.model_copy(update={"fit_window": "pre_race"}), r.practice,
                    calibrator=CALIBRATORS.get("ridge")())
    with pytest.raises(RuntimeError, match="not fitted"):
        sim.fitted_for(calibration.snapshot_at(r.frame, r.race, "D00", 0))


# --- leakage: no held-out race row reaches any fit -----------------------------------------------
def test_no_held_out_race_row_reaches_any_fit_including_the_alpha_search(syn_dataset, monkeypatch):
    seen = []
    real = calibrators._ridge_fit

    def spy(x, y, races, alpha):
        seen.append(set(races))
        return real(x, y, races, alpha)

    monkeypatch.setattr(calibrators, "_ridge_fit", spy)
    held = SYN[2]
    res = calibrators.loro(syn_dataset, "ridge", held, windows=("pre_race", "to_snapshot"))
    fitted = res.fits[res.fits["status"] == "ridge"]
    train = len(SYN) - 1
    # per fitted factor: every alpha times every inner held-out race, then the final fit
    assert len(seen) == len(fitted) * (len(calibrators.RIDGE_ALPHAS) * train + 1)
    assert all(held not in s for s in seen)
    assert set().union(*seen) == set(SYN) - {held}
    assert max(len(s) for s in seen) == train and min(len(s) for s in seen) == train - 1
    assert set(res.factor_errors["race_id"]) == {held} and len(res.factor_errors)


def test_the_fit_does_not_depend_on_the_held_out_race(syn_dataset):
    held = SYN[3]
    rows = syn_dataset.rows.copy()
    bad = rows["race_id"] == held
    for c in syn_dataset.feature_columns + syn_dataset.target_columns:
        rows.loc[bad, c] = 1e6                       # absurd values on the held-out race only
    poisoned = CalibrationDataset(rows, syn_dataset.races)
    a = CALIBRATORS.get("ridge")().fit(syn_dataset.subset([r for r in SYN if r != held]))
    b = CALIBRATORS.get("ridge")().fit(poisoned.subset([r for r in SYN if r != held]))
    assert json.dumps(a.state(), sort_keys=True) == json.dumps(b.state(), sort_keys=True)


# --- the training log ---------------------------------------------------------------------------
def test_training_mode_respects_wandb_mode_and_falls_back_offline(tmp_path):
    no_netrc = tmp_path / "netrc"
    assert compute.training_mode({"WANDB_MODE": "disabled"}, no_netrc) == ("disabled", None)
    assert compute.training_mode({"WANDB_MODE": "offline"}, no_netrc) == ("offline", None)
    assert compute.training_mode({"WANDB_MODE": "dryrun"}, no_netrc) == ("offline", None)
    mode, note = compute.training_mode({}, no_netrc)
    assert mode == "offline" and "no login or API key" in note
    assert compute.training_mode({"WANDB_API_KEY": "x" * 40}, no_netrc) == ("online", None)
    no_netrc.write_text("machine api.wandb.ai\n  login user\n  password " + "y" * 40 + "\n")
    assert compute.training_mode({}, no_netrc) == ("online", None)
    assert compute.training_mode({"WANDB_MODE": "bogus"}, no_netrc)[0] == "offline"


def test_training_run_is_a_no_op_when_disabled_or_wandb_is_missing(monkeypatch, capsys):
    monkeypatch.setitem(__import__("sys").modules, "wandb", None)   # import wandb raises ImportError
    with compute.training_run("calibrator_fit", {"a": 1}) as run:   # conftest: WANDB_MODE=disabled
        assert run is None
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(compute, "_NOTED", set())
    for _ in range(2):
        with compute.training_run("calibrator_fit", {"a": 1}) as run:
            assert run is None
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["wandb is not installed (requirements-train.txt): calibrator training is not logged"]


def test_a_learned_fit_logs_its_split_by_default(syn_dataset, monkeypatch):
    logged = []

    class Run:
        def log(self, d):
            logged.append(("log", d))

    @contextmanager
    def fake(job, config=None):
        logged.append(("init", job, config))
        yield Run()

    monkeypatch.setattr(compute, "training_run", fake)
    calibrators.loro(syn_dataset, "ridge", SYN[0], windows=("pre_race",))
    calibrators.loro(syn_dataset, "shrinkage", SYN[0], windows=("pre_race",))   # not learned: no log
    assert [e[0] for e in logged] == ["init", "log"]
    _, job, config = logged[0]
    assert job == "calibrator_fit" and config["calibrator"] == "ridge"
    assert (config["fold"], config["held_out"], config["train"], config["fold_scheme"]) == (
        SYN[0], [SYN[0]], SYN[1:], "leave-one-race-out")
    assert logged[1][1]["n_train_races"] == 5 and "pit_loss_s/alpha" in logged[1][1]


# --- pipeline: cross run --synthetic 4 with the ridge arm ---------------------------------------
def test_ridge_arm_is_frozen_and_learned():
    assert "ridge" not in cross_race.ARMS and cross_race.LEARNED_ARMS == ("ridge",)
    assert cross_race.arm_config("ridge").calibrator.name == "ridge"
    assert cross_race.arm_config("ridge").model_copy(update={"calibrator": SimConfig().calibrator}) == SimConfig()
    t = cross_race.arms_table(["baseline", "ridge"]).set_index("arm")
    assert t.loc["ridge", "role"] == cross_race.LEARNED_ROLE and t.loc["ridge", "file"] == "configs/sim/arms/ridge.yaml"
    with pytest.raises(KeyError, match="unknown learned arm"):
        cross_race.run(["2099_01_synthetic"], load_profile("local"), "unused", learned=("gbm",))


@pytest.fixture(scope="module")
def ridge_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cross_ridge")
    p = load_profile("local")
    small = p.model_copy(update={"g": p.g.model_copy(update={"laps_every": 20, "n_boot": 20}),
                                 "sim": p.sim.model_copy(update={"n_samples": 5}),
                                 "parallel": p.parallel.model_copy(update={"n_jobs": 1})})
    prof = out / "small.yaml"
    prof.write_text(yaml.safe_dump(small.model_dump(mode="json")))
    assert cross_race.main(["run", "--races", "--synthetic", "4", "--learned", "ridge", "--tau-multipliers", "1",
                            "--profile", str(prof), "--out", str(out)]) == 0
    return out


def test_cross_run_with_synthetic_races_and_a_ridge_arm_completes(ridge_run):
    out = ridge_run
    record = json.loads((out / "run.json").read_text())
    races = [f"2099_{i:02d}_synthetic" for i in range(1, 5)]
    assert record["races"] == races and record["eligible"] == races
    assert record["arms"] == [*cross_race.ARMS, "ridge"] and record["belief_choice"] in cross_race.ARMS
    for r in races:
        assert (out / "h" / "arms" / "ridge" / f"{r}.parquet").is_file()
    assert not (cross_race.CROSS_DIR / "h" / "arms" / "ridge").exists()
    sel = pd.read_csv(out / "selection.csv")
    for w in ("pre_race", "to_snapshot"):
        part = sel[sel["window"] == w]
        assert {"none", "baseline", "ridge"} <= set(part["arm"])
        ridge = part[part["arm"] == "ridge"]
        assert (ridge["role"] == cross_race.LEARNED_ROLE).all() and not ridge["passes"].any()
    full = sel[(sel["window"] == "full") & (sel["arm"] == "ridge")]
    assert (full["role"] == cross_race.NA_ROLE).all()
    learned = pd.read_csv(out / "learned.csv")
    for w in ("pre_race", "to_snapshot"):
        t = learned[learned["window"] == w].set_index("calibrator")
        assert list(t.index) == ["none", "shrinkage", "ridge"]
        assert t.loc["shrinkage", "arm"] == "baseline" and t.loc["ridge", "kind"] == "learned"
        assert (t["n_races"] == 4).all() and t["provisional"].all()
    md = (out / "summary.md").read_text()
    assert "## Learned calibrators (SPEC 0024)" in md and "nothing is adopted" in md


def test_with_3_training_races_every_ridge_factor_falls_back_to_shrinkage(ridge_run):
    out = ridge_run
    fits = pd.read_csv(out / "learned_fits.csv")
    assert len(fits) and (fits["ridge"] == 0).all() and (fits["folds"] == 4).all()
    reasons = {x for r in fits["fallback_reasons"] for x in r.split("; ")}
    # 3 races train each fold; a factor some race lacks (no SC pit stop, say) has fewer
    assert "3 training race(s), under 5" in reasons and reasons <= {f"{n} training race(s), under 5" for n in (1, 2, 3)}
    # so its G rows are shrinkage's
    races = [f"2099_{i:02d}_synthetic" for i in range(1, 5)]
    prof = load_profile(str(out / "small.yaml"))
    ridge = cross_race.arm_outputs(out, "ridge", races, prof)["g"]
    base = cross_race.arm_outputs(out, "baseline", races, prof)["g"]
    keys = ["race_id", "window", "driver", "k"]
    m = ridge.merge(base, on=keys, suffixes=("_r", "_b"))
    m = m[m["window"].isin(["pre_race", "to_snapshot"])]
    assert len(m) and np.allclose(m["err_offset_s_r"], m["err_offset_s_b"], equal_nan=True)
