"""Calibrators (SPEC H v2): the interface, "none" and "shrinkage", cadence, plug-ins, dataset
and leave-one-race-out validation. Synthetic data only, offline."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from src.contracts import CalibrationOutput, CorrectionTable, Snapshot, Weather
from src.sim import SimConfig, Simulator, calibration, calibrators
from src.sim.calibrators import CALIBRATORS, LoroResult
from src.sim.config import EventEffects
from src.sim.degradation import DEGRADATION
from src.sim.race_ids import canonical_race_id, make_race_id, parse_race_id, race_season
from test_sim import FUEL, RACE, TRUE, synthetic_frame, synthetic_practice
from test_sim_position import race_frame

OFF = {"SOFT": (79.0, 0.06), "MEDIUM": (81.5, 0.05)}   # practice that disagrees with the race
A_PLAN = [("SOFT", 20), ("MEDIUM", 30)]
B_PLAN = [("SOFT", 10), ("MEDIUM", 25), ("SOFT", 15)]


def to_snap(**kw):
    return SimConfig(fit_window="to_snapshot", **kw)


def at(k, driver="X", **kw):
    return Snapshot(race=RACE, driver=driver, laps_completed=k, compound="SOFT", tyre_age=k, **kw)


def report_json(rep):
    return json.dumps(rep, sort_keys=True, default=str)


class Fixed(calibrators.Calibrator):
    """Shrinkage's output with some factors replaced, and an optional correction table."""
    name = "fixed"

    def __init__(self, factors=None, correction=None, base="shrinkage"):
        super().__init__()
        self.extra, self.correction, self.base = factors or {}, correction, base

    def calibrate(self, ctx):
        out = CALIBRATORS.get(self.base)().calibrate(ctx)
        return out.model_copy(update={"factors": {**out.factors, **self.extra}, "correction": self.correction,
                                      "source": self.name})


# --- registry and contract -------------------------------------------------------------
def test_registry_and_defaults():
    assert CALIBRATORS.keys() == ["gbm", "mlp", "none", "ridge", "shrinkage"]
    cfg = SimConfig()
    assert (cfg.calibrator.name, cfg.calibration_every, cfg.recalibrate_temp_delta_c) == ("shrinkage", 1, 3.0)
    assert calibration.config_hash(SimConfig(calibrator={"name": "none"})) != calibration.config_hash(cfg)
    assert calibration.config_hash(SimConfig(calibration_every=None)) != calibration.config_hash(cfg)
    with pytest.raises(KeyError, match="unknown calibrator"):
        Simulator(synthetic_frame(), SimConfig(calibrator={"name": "nope"}))


@pytest.mark.parametrize("name", ["ridge", "gbm", "mlp"])
def test_learned_classes_are_stubs_until_the_dataset_exists(name):
    cal = CALIBRATORS.get(name)()
    assert cal.learned and not cal.fitted and cal.report()["learned"]
    with pytest.raises(NotImplementedError, match="needs the multi-race dataset"):
        cal.fit(calibrators.CalibrationDataset(pd.DataFrame(), {}))
    s = Simulator(synthetic_frame(), to_snap(calibrator={"name": name}), synthetic_practice())
    with pytest.raises(NotImplementedError, match="needs the multi-race dataset"):
        s.fitted_for(at(10))


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="ridge is a stub until the multi-race dataset")
def test_ridge_recovers_a_linear_synthetic_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    rows = pd.DataFrame({"race_id": [f"2021_{i % 20 + 1:02d}_r{i % 20}" for i in range(200)], "season": 2021,
                         "k": 0, "f1": x[:, 0], "f2": x[:, 1], "target:pit_loss_s": 20 + 2 * x[:, 0] - x[:, 1]})
    cal = CALIBRATORS.get("ridge")().fit(calibrators.CalibrationDataset(rows, {}))
    assert cal.report()["coef"]["pit_loss_s"] == pytest.approx([2.0, -1.0], abs=0.05)


def test_save_load_and_report(tmp_path):
    cal = CALIBRATORS.get("none")(note="x")
    cal.save(tmp_path / "c.json")
    back = calibrators.Calibrator.load(tmp_path / "c.json")
    assert type(back) is type(cal) and back.params == {"note": "x"}
    assert back.report() == {"name": "none", "learned": False, "fitted": True, "params": {"note": "x"}}


def test_output_contract_validates_factors_and_tables():
    CalibrationOutput(factors={"base_delta_s[soft]": (1.0, 0.5), "pit_loss_s": (22.0, 0.0)}, valid_until_lap=3,
                      source="t")
    for bad in ({"base_delta_s": (1.0, 0.1)}, {"pit_loss_s[SOFT]": (1.0, 0.1)}, {"wing": (1.0, 0.1)},
                {"pit_loss_s": (22.0, -1.0)}):
        with pytest.raises(ValueError):
            CalibrationOutput(factors=bad, valid_until_lap=0, source="t")
    ok = [[[0.0, 0.0]]]
    CorrectionTable(compounds=["SOFT"], mean=ok, sd=ok)
    with pytest.raises(ValueError):
        CorrectionTable(compounds=["SOFT", "MEDIUM"], mean=ok, sd=ok)
    with pytest.raises(ValueError):
        CorrectionTable(compounds=["SOFT"], mean=ok, sd=[[[0.0]]])
    with pytest.raises(ValueError):
        CorrectionTable(compounds=["SOFT"], mean=ok, sd=[[[0.0, -1.0]]])


# --- "shrinkage" is #13, "none" is the held prior -----------------------------------------
@pytest.mark.parametrize("name", ["linear", "linear_fuel"])
@pytest.mark.parametrize("exclude", [False, True])
def test_shrinkage_reproduces_the_13_fit_exactly(name, exclude):
    frame = synthetic_frame(fuel=FUEL).assign(driver=lambda d: np.where(d["lap"] % 2, "X", "Y"))
    practice = synthetic_practice(coefs=OFF)
    cfg = to_snap(degradation={"name": name}, exclude_snapshot_driver=exclude)
    s = Simulator(frame, cfg, practice)
    model = DEGRADATION.get(name)
    kw = dict(min_samples=cfg.min_samples, fuel_prior=cfg.pre_race_fuel_slope_s_per_lap,
              base_delta=cfg.pre_race_base_delta_s, fuel_prior_sd=cfg.shrinkage["fuel_s_per_lap"])
    for k in (0, 5, 30, 49):
        fit = s.fitted_for(at(k))
        race = frame[frame["lap"] <= k]
        prac = practice.assign(lap=np.nan)
        if exclude:
            race, prac = race[race["driver"] != "X"], prac[prac["driver"] != "X"]
        prior = model.fit(prac, **kw)
        hand = model.fit(race, prior=prior, shrinkage=cfg.shrinkage, **kw)
        assert fit.degradation.coefs == hand.coefs
        assert getattr(fit.degradation, "fuel_slope", None) == getattr(hand, "fuel_slope", None)
        assert report_json(fit.degradation.fit_report) == report_json(hand.fit_report)
        assert fit.calibration.output.source == "shrinkage" and not fit.calibration.held


def test_shrinkage_without_practice_is_the_13_fuel_only_penalty():
    frame = synthetic_frame(fuel=FUEL)
    cfg = to_snap()
    fit = Simulator(frame, cfg).fitted_for(at(30))
    hand = DEGRADATION.get("linear_fuel").fit(frame[frame["lap"] <= 30], min_samples=5, prior=None,
                                              shrinkage=cfg.shrinkage, fuel_prior=cfg.pre_race_fuel_slope_s_per_lap)
    assert fit.degradation.coefs == hand.coefs and fit.degradation.fuel_slope == hand.fuel_slope
    assert set(fit.calibration.output.factors) == {"fuel_s_per_lap"}


def test_none_holds_the_pre_race_fit_at_every_k():
    frame, practice = synthetic_frame(fuel=FUEL), synthetic_practice(coefs=OFF)
    pre = Simulator(frame, SimConfig(fit_window="pre_race"), practice).fitted_for(at(0))
    none = Simulator(frame, to_snap(calibrator={"name": "none"}), practice)
    shrink = Simulator(frame, to_snap(), practice)
    for k in (0, 10, 40):
        fit = none.fitted_for(at(k))
        assert fit.calibration.held and fit.calibration.output.source == "none"
        assert fit.degradation.coefs == pre.degradation.coefs and fit.degradation.fuel_slope == pre.degradation.fuel_slope
        assert fit.ref_lap_s == pre.ref_lap_s
        assert report_json(fit.degradation.fit_report) == report_json(pre.degradation.fit_report)
    # the race moves shrinkage toward the truth and leaves none where practice put it
    assert shrink.fitted_for(at(40)).degradation.coefs["SOFT"][1] == pytest.approx(TRUE["SOFT"][1], abs=0.01)
    assert none.fitted_for(at(40)).degradation.coefs["SOFT"][1] == pytest.approx(OFF["SOFT"][1])
    with pytest.raises(ValueError, match="needs practice"):
        Simulator(frame, to_snap(calibrator={"name": "none"})).fitted_for(at(10))


def test_none_and_shrinkage_agree_on_the_pre_race_window():
    frame, practice = synthetic_frame(fuel=FUEL), synthetic_practice(coefs=OFF)
    a, b = (Simulator(frame, SimConfig(fit_window="pre_race", calibrator={"name": n}), practice).fitted_for(at(20))
            for n in ("none", "shrinkage"))
    assert a.degradation.coefs == b.degradation.coefs
    eff_a, eff_b = calibrators.effective_factors(a, RACE), calibrators.effective_factors(b, RACE)
    assert {k: v[0] for k, v in eff_a.items()} == {k: v[0] for k, v in eff_b.items()}
    tau = SimConfig().shrinkage
    assert eff_a["wear_s_per_lap[SOFT]"][1] == 0.0 and eff_b["wear_s_per_lap[SOFT]"][1] == tau["wear_s_per_lap"]
    assert eff_b["base_delta_s[SOFT]"] == (pytest.approx(1.0), tau["base_s"])
    assert eff_b["pit_loss_s"] == (RACE.pit_loss_s, 0.0)


def test_effective_factors_of_a_shrunk_fit_use_the_posterior_sd():
    s = Simulator(synthetic_frame(fuel=FUEL), to_snap(), synthetic_practice(coefs=OFF)).fitted_for(at(30))
    eff, rep = calibrators.effective_factors(s), s.degradation.fit_report
    assert eff["wear_s_per_lap[SOFT]"] == (s.degradation.coefs["SOFT"][1], rep["sd"]["wear_s_per_lap"]["SOFT"])
    assert eff["base_delta_s[SOFT]"][0] == pytest.approx(s.degradation.coefs["SOFT"][0] - OFF["SOFT"][0])
    with pytest.raises(ValueError):
        calibrators.effective_factors(Simulator(synthetic_frame()))


def test_mixed_held_and_free_degradation_factors_are_refused():
    s = Simulator(synthetic_frame(), to_snap(), synthetic_practice(),
                  calibrator=Fixed({"wear_s_per_lap[SOFT]": (0.1, 0.0)}))
    with pytest.raises(NotImplementedError, match="mix"):
        s.fitted_for(at(10))


def test_calibrated_factor_means_move_the_prior():
    """A wear and base delta prior given with a tiny sd is where the fit ends up."""
    out = {"wear_s_per_lap[SOFT]": (0.2, 1e-6), "base_delta_s[SOFT]": (2.0, 1e-6), "wear_s_per_lap[MEDIUM]": (0.03, 1e-6),
           "base_delta_s[MEDIUM]": (0.0, 1e-6), "fuel_s_per_lap": (-0.05, 1e-6)}
    s = Simulator(synthetic_frame(), to_snap(), synthetic_practice(), calibrator=Fixed(out)).fitted_for(at(30))
    assert s.degradation.coefs["SOFT"] == pytest.approx((TRUE["SOFT"][0] + 2.0, 0.2), abs=1e-4)
    assert s.degradation.fuel_slope == pytest.approx(-0.05, abs=1e-4)


# --- cadence -----------------------------------------------------------------------------
def test_cadence_reuses_the_last_calibration_point_and_events_force_a_refresh():
    s = Simulator(synthetic_frame(), to_snap(calibration_every=5), synthetic_practice())
    five = s.fitted_for(at(5))
    assert s.fitted_for(at(7)) is five and five.window == ("to_snapshot", 5, None)
    assert five.calibration.output.valid_until_lap == 9 and five.calibration.k0 == 5
    assert s.fitted_for(at(6, sc_laps=[6])).window == ("to_snapshot", 6, None)        # SC starts at 6
    assert s.fitted_for(at(7, sc_laps=[6, 7])) is s.fitted_for(at(6, sc_laps=[6]))
    assert s.fitted_for(at(9, sc_laps=[6, 7])).window[1] == 8                           # SC ends: lap 8 green
    assert s.fitted_for(at(12, sc_laps=[6, 7])).window[1] == 10
    assert s.fitted_for(at(13, vsc_laps=[12])).window[1] == 13                           # VSC end
    # the k = 5 fit sees laps <= 5 only: it equals the every-lap fit at 5
    every = Simulator(synthetic_frame(), to_snap(), synthetic_practice()).fitted_for(at(5))
    assert five.degradation.coefs == every.degradation.coefs


def test_pre_race_only_cadence_never_refreshes():
    s = Simulator(synthetic_frame(), to_snap(calibration_every=None), synthetic_practice())
    zero = s.fitted_for(at(0))
    assert s.fitted_for(at(30, sc_laps=[25, 26])) is zero and zero.calibration.output.valid_until_lap == RACE.total_laps


def test_refresh_on_rain_onset_and_track_temperature():
    lw = pd.DataFrame({"track_temp_c": [30.0] * 13 + [34.0] * 8, "rain": [False] * 21}, index=range(0, 21))
    assert calibrators.refresh_laps(at(15), lw, 3.0, since=10) == [13]
    assert calibrators.calibration_point(at(15), 10, lw, 3.0) == 13
    assert calibrators.calibration_point(at(15), 10, lw, 5.0) == 10
    wet = at(15, weather=Weather(rain_laps=[12, 13]))
    assert calibrators.refresh_laps(wet, since=10) == [12]                                # onset only
    assert calibrators.calibration_point(at(15), 1) == 15
    s = Simulator(synthetic_frame(), to_snap(calibration_every=10), synthetic_practice(), weather=lw)
    assert s.fitted_for(at(15)).window[1] == 13


def test_outputs_are_cached_per_calibration_point_with_the_pre_race_output_in_the_context():
    seen = []

    class Spy(calibrators.ShrinkageCalibrator):
        def calibrate(self, ctx):
            seen.append((ctx.k, ctx.pre_race_output is not None, ctx.previous_output is not None,
                         int(ctx.laps_so_far["lap"].max()) if len(ctx.laps_so_far) else 0,
                         bool(ctx.residuals_so_far["lap"].le(ctx.k).all())))
            return super().calibrate(ctx)

    s = Simulator(synthetic_frame(), to_snap(calibration_every=5), synthetic_practice(), calibrator=Spy())
    for k in (5, 6, 7, 10, 11):
        s.fitted_for(at(k))
    assert seen == [(0, False, False, 0, True), (5, True, True, 5, True), (10, True, True, 10, True)]


# --- plug-ins ----------------------------------------------------------------------------
def test_stub_pit_loss_distribution_is_drawn_per_sample_and_shared_across_plans():
    s = Simulator(synthetic_frame(), to_snap(), synthetic_practice(), calibrator=Fixed({"pit_loss_s": (25.0, 2.0)}))
    snap = at(10)
    one, two = [["SOFT", 10], ["MEDIUM", 30]], [["SOFT", 10], ["MEDIUM", 15], ["SOFT", 15]]
    b = s.simulate_batch(snap, [one, two], n_samples=500, rules=calibration.REPLAY_RULES)
    assert b.valid.all()
    assert b.pit_loss[0].mean() == pytest.approx(25.0, abs=0.3) and b.pit_loss[0].std() == pytest.approx(2.0, rel=0.1)
    np.testing.assert_allclose(b.pit_loss[1], 2 * b.pit_loss[0])                      # one draw, both plans
    again = s.simulate_batch(snap, [two], n_samples=500, rules=calibration.REPLAY_RULES)
    np.testing.assert_array_equal(again.pit_loss[0], b.pit_loss[1])                   # same seed, same draws


def test_event_factor_draws_are_clipped_and_held_at_sd_zero():
    s = Simulator(synthetic_frame(), to_snap(effects=EventEffects(lap1_extra_s=0.0)), synthetic_practice(),
                  calibrator=Fixed({"sc_pit_loss_factor": (0.9, 1.0), "pit_loss_s": (30.0, 0.0)}))
    plan = [[["SOFT", 10], ["MEDIUM", 30]]]   # stops at the end of lap 20
    green = s.simulate_batch(at(10), plan, n_samples=200, scenario=calibrators.Scenario(sc_laps=[30]),
                             rules=calibration.REPLAY_RULES)
    assert set(np.unique(green.pit_loss[0])) == {30.0}             # sd 0: the output's mean, not the race's 20
    sc = s.simulate_batch(at(10), plan, n_samples=200, scenario=calibrators.Scenario(sc_laps=[20]),
                          rules=calibration.REPLAY_RULES).pit_loss[0]
    # stop under SC: 30 x a factor drawn around 0.9 with sd 1, clipped to [0, 1]
    assert sc.min() >= 0.0 and sc.max() == 30.0 and (sc == 30.0).mean() > 0.3 and (sc == 0.0).mean() > 0.1


def table(fn, sd=0.0, max_age=45):
    comps = ["MEDIUM", "SOFT"]
    mean = [[[fn(c, a, lap) for lap in range(1, RACE.total_laps + 1)] for a in range(max_age + 1)] for c in comps]
    return CorrectionTable(compounds=comps, mean=mean,
                           sd=[[[sd] * RACE.total_laps for _ in range(max_age + 1)] for _ in comps])


def test_correction_table_lookup_adds_exactly_the_table_values():
    def f(c, a, lap):
        return (0.5 if c == "SOFT" else -0.25) + 0.01 * a + 0.001 * lap

    frame, practice = synthetic_frame(), synthetic_practice()
    plan = [["SOFT", 10], ["MEDIUM", 30]]
    base = Simulator(frame, to_snap(), practice, calibrator=Fixed(base="none"))
    corr = Simulator(frame, to_snap(), practice, calibrator=Fixed(correction=table(f), base="none"))
    snap = at(10)
    a = base.simulate(snap, plan, return_laps=True, rules=calibration.REPLAY_RULES)
    b = corr.simulate(snap, plan, return_laps=True, rules=calibration.REPLAY_RULES)
    # laps 11-20 on the SOFT set from age 11, laps 21-50 on a new MEDIUM from age 1
    expect = [f("SOFT", 10 + i, 10 + i) for i in range(1, 11)] + [f("MEDIUM", i, 20 + i) for i in range(1, 31)]
    np.testing.assert_allclose(np.subtract(b.lap_times_s, a.lap_times_s), expect, atol=1e-9)


def test_correction_sd_is_one_z_per_sample_shared_across_plans_and_laps():
    frame, practice = synthetic_frame(), synthetic_practice()
    plans = [[["SOFT", 10], ["MEDIUM", 30]], [["SOFT", 20], ["MEDIUM", 20]]]
    base = Simulator(frame, to_snap(), practice, calibrator=Fixed(base="none"))
    corr = Simulator(frame, to_snap(), practice, calibrator=Fixed(correction=table(lambda *_: 0.0, sd=1.0), base="none"))
    a = base.simulate_batch(at(10), plans, n_samples=400, return_laps=True, rules=calibration.REPLAY_RULES)
    b = corr.simulate_batch(at(10), plans, n_samples=400, return_laps=True, rules=calibration.REPLAY_RULES)
    z = (b.times - a.times) / 40.0
    np.testing.assert_allclose(z[0], z[1])
    d = b.lap_times - a.lap_times
    np.testing.assert_allclose(d, d[0, 0])
    assert z[0].mean() == pytest.approx(0.0, abs=0.2) and z[0].std() == pytest.approx(1.0, rel=0.15)


def test_no_correction_and_no_factor_sd_draws_nothing():
    """The reference calibrators leave the sample stream as it was: same draws as no calibrator."""
    frame, practice = synthetic_frame(), synthetic_practice()
    cfg = to_snap(hazard={"name": "age_logistic"})
    plans = [[["SOFT", 15], ["MEDIUM", 25]], [["SOFT", 25], ["MEDIUM", 15]]]
    a = Simulator(frame, cfg, practice).simulate_batch(at(10), plans, n_samples=50, seed=3)
    b = Simulator(frame, cfg, practice, calibrator=Fixed()).simulate_batch(at(10), plans, n_samples=50, seed=3)
    np.testing.assert_array_equal(a.times, b.times)
    np.testing.assert_array_equal(a.dnf, b.dnf)


# --- dataset, loro, selection ------------------------------------------------------------
def race_input(race_id, frame=None, **kw):
    frame = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)) if frame is None else frame
    cfg = SimConfig(effects=EventEffects(lap1_extra_s=0.0), pre_race_base_delta_s=0.0)
    return calibrators.RaceInput(race_id, frame, RACE.model_copy(update={"race_id": race_id}),
                                 synthetic_practice(coefs=OFF), None, cfg, **kw)


def test_dataset_never_includes_a_lap_beyond_k():
    a = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN))
    b = a.copy()
    late = b["lap"] > 20
    b.loc[late, "lap_time_s"] += 3.0
    b.loc[b["lap"].between(30, 32), "track_status"] = "4"
    temp = pd.DataFrame({"track_temp_c": np.r_[np.linspace(30, 32, 21), np.full(30, 45.0)]}, index=range(0, 51))
    temp_b = temp.copy()
    temp_b.loc[temp_b.index > 20, "track_temp_c"] = 10.0
    ds = calibrators.build_dataset([race_input("2021_01_a", a, weather=temp), race_input("2021_02_b", b, weather=temp_b)],
                                   laps_k=[0, 10, 20])
    rows = ds.rows.set_index(["race_id", "k"])
    for k in (0, 10, 20):
        pd.testing.assert_series_equal(rows.loc[("2021_01_a", k), ds.feature_columns],
                                       rows.loc[("2021_02_b", k), ds.feature_columns], check_names=False)
    assert rows.loc[("2021_01_a", 20), "temp_trend_c_per_lap"] == pytest.approx(0.1)
    assert rows.loc[("2021_01_a", 20), "laps_so_far[SOFT]"] > rows.loc[("2021_01_a", 10), "laps_so_far[SOFT]"]
    assert rows.loc[("2021_01_a", 0), "target:fuel_s_per_lap"] != rows.loc[("2021_02_b", 0), "target:fuel_s_per_lap"]
    assert set(ds.rows["season"]) == {2021} and ds.laps_k("2021_01_a") == [0, 10, 20]


def test_loro_on_one_race_runs_with_nan_cis():
    ds = calibrators.build_dataset([race_input("2021_01_syn")], laps_k=[0, 10, 20])
    out = calibrators.loro(ds, "shrinkage", cadences=(1, None))
    assert set(out) == {"loro", "loso"}
    res = out["loro"]
    assert isinstance(res, LoroResult) and res.protocol == "race"
    assert res.folds.to_dict("records") == [{"fold": "2021_01_syn", "held_out": "2021_01_syn", "train": ""}]
    assert out["loso"].folds["fold"].tolist() == ["season:2021"]
    assert set(res.rows["window"]) == {"pre_race", "to_snapshot"} and set(res.rows["calibration_every"]) == {"1", "none"}
    assert set(res.rows["calibrator"]) == {"shrinkage"} and set(res.rows.columns) >= set(calibration.MIDRACE_COLUMNS)
    t = res.targets()
    assert t.index.names == ["calibration_every", "window", "factor"]
    assert {"wear_s_per_lap[SOFT]", "fuel_s_per_lap", "pit_loss_s"} <= set(t.index.get_level_values("factor"))
    assert (t["n"] == 3).all() and t["coverage_68"].between(0, 1).all()
    m = res.midrace(n_boot=20)
    assert m.index.names == ["calibration_every", "window", "horizon_bucket"]
    assert m["mae_offset_ci_lo"].isna().all() and m["mae_offset_ci_hi"].isna().all()
    one = calibrators.loro(ds, "none", "2021_01_syn")
    assert isinstance(one, LoroResult) and set(one.rows["calibration_every"]) == {"1"}
    # none's to_snapshot is its pre_race: the same rows
    w = one.rows.set_index(["driver", "k"]).sort_index()
    pd.testing.assert_series_equal(w.loc[w["window"] == "pre_race", "pred_remaining_s"],
                                   w.loc[w["window"] == "to_snapshot", "pred_remaining_s"])
    with pytest.raises(NotImplementedError, match="needs the multi-race dataset"):
        calibrators.loro(ds, "ridge")
    with pytest.raises(KeyError, match="matches no race"):
        calibrators.loro(ds, "none", "2021_09_other")


def test_loro_does_not_need_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    ds = calibrators.build_dataset([race_input("2021_01_syn")], laps_k=[0, 10])
    assert len(calibrators.loro(ds, "shrinkage", "season:2021").rows)


def test_folds_split_by_race_and_by_season():
    ds = calibrators.CalibrationDataset(pd.DataFrame(), {r: None for r in ("2021_01_a", "2021_02_b", "2022_01_c")})
    assert calibrators.folds(ds, "race") == {"2021_01_a": ["2021_01_a"], "2021_02_b": ["2021_02_b"],
                                             "2022_01_c": ["2022_01_c"]}
    assert calibrators.folds(ds, "season") == {"season:2021": ["2021_01_a", "2021_02_b"], "season:2022": ["2022_01_c"]}
    assert calibrators.held_out(ds, "season:2022") == ["2022_01_c"]


def fake_result(name, err_by_race):
    rows = pd.DataFrame([{"race_id": r, "window": "to_snapshot", "calibration_every": "1", "horizon": h,
                          "err_offset_s": e + 0.1 * h} for r, e in err_by_race.items() for h in (20, 25, 30, 40, 45)])
    return LoroResult(name, "race", rows, pd.DataFrame(columns=calibrators.FACTOR_ERROR_COLUMNS), pd.DataFrame())


def test_selection_adopts_a_learned_class_only_beyond_the_ci():
    races = [f"2021_{i:02d}_r" for i in range(1, 7)]
    res = {"none": fake_result("none", dict(zip(races, [12, 14, 13, 15, 12, 13]))),
           "shrinkage": fake_result("shrinkage", dict(zip(races, [10, 11, 10, 12, 11, 10]))),
           "ridge": fake_result("ridge", dict(zip(races, [2, 3, 2, 2, 3, 2]))),
           "gbm": fake_result("gbm", dict(zip(races, [9, 11, 10, 12, 11, 10])))}
    t = calibrators.selection_table(res, n_boot=200)
    assert t.loc["none", "decision"] == t.loc["shrinkage", "decision"] == "reference"
    assert t.loc["ridge", "decision"] == "adopt" and t.loc["gbm", "decision"] == "keep shrinkage"
    assert t.loc["shrinkage", "n"] == 6 * 4                                            # horizons 20 to 40
    one = calibrators.selection_table({"shrinkage": res["shrinkage"], "ridge": fake_result("ridge", {races[0]: 1})})
    assert one.loc["ridge", "decision"] == "keep shrinkage"                            # NaN CI keeps shrinkage


# --- race ids ----------------------------------------------------------------------------
def test_race_ids():
    assert make_race_id(2021, 4, "Spain") == "2021_04_spain" and make_race_id(2024, 7, "Emilia Romagna") == "2024_07_emilia_romagna"
    assert canonical_race_id("2021_Spain") == "2021_04_spain" and canonical_race_id("x") == "x"
    assert parse_race_id("2021_Spain") == parse_race_id("2021_04_spain") == (2021, 4, "spain")
    assert race_season("2021_Spain") == 2021
    with pytest.raises(ValueError, match="season"):
        parse_race_id("Spain")
