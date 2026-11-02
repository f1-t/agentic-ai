"""Cross-race calibration report (SPEC 0016 and its amendment): eligibility, the selection
rules, the collinear-hold and prior-sd fit options, the arms, and the end-to-end run on Spain
plus two synthetic races. Offline: synthetic races and the committed Spain cache."""
import json

import numpy as np
import pandas as pd
import pytest

from src.sim import Simulator, SimConfig, calibration, cross_race, jobs
from src.sim.compute import load_profile
from src.sim.cross_race import ARMS, BASELINE
from src.contracts import Snapshot
from test_sim import RACE, synthetic_practice

SYN = "2099_01_synthetic"


@pytest.fixture(scope="module")
def syn():
    return cross_race.synthetic_race(SYN)


# --- A: eligibility -----------------------------------------------------------------------
def test_synthetic_race_is_eligible_and_deterministic(syn):
    e = calibration.eligibility(syn.frame, syn.race)
    assert e["eligible"] and e["reasons"] == []
    assert e["n_finishers"] == 16 and e["wet_laps_share"] == 0.0 and e["red_flags"] == []
    assert len(e["sc_periods"]) == 1 and e["vsc_periods"] == [] and e["dry_compounds"] == ["HARD", "MEDIUM", "SOFT"]
    again = cross_race.synthetic_race(SYN)
    pd.testing.assert_frame_equal(again.frame, syn.frame)


def test_a_wet_lap_makes_a_race_ineligible(syn):
    f = syn.frame.copy()
    f.loc[(f["driver"] == "D03") & (f["lap"] == 30), "compound"] = "INTERMEDIATE"
    e = calibration.eligibility(f, syn.race)
    assert not e["eligible"] and e["wet_laps_share"] > 0
    assert any(r.startswith("wet: 1 laps on INTERMEDIATE") for r in e["reasons"])


@pytest.mark.parametrize("lap", [1, 30])
def test_red_flags_at_lap_1_and_30_make_a_race_ineligible(syn, lap):
    f = syn.frame.copy()
    f.loc[f["lap"] == lap, "track_status"] = "5"
    e = calibration.eligibility(f, syn.race)
    assert not e["eligible"] and e["red_flags"] == [lap]
    reason = next(r for r in e["reasons"] if r.startswith("red flag"))
    assert f"lap {lap}" in reason
    assert ("restart, excluded too" in reason) == (lap == 1)


def test_a_red_flag_in_the_race_control_messages_counts(syn):
    msgs = pd.DataFrame({"Time": pd.to_timedelta([1, 2], unit="s"), "Category": ["Flag", "Flag"],
                         "Message": ["GREEN LIGHT", "RED FLAG"], "Status": [None, None], "Flag": ["GREEN", "RED"],
                         "Lap": [1, 30]})
    e = calibration.eligibility(syn.frame, syn.race, msgs)
    assert not e["eligible"] and e["red_flags"] == [30]


def test_too_few_finishers_and_too_few_dry_compounds_or_clean_laps():
    small = cross_race.synthetic_race("2099_03_synthetic", n_drivers=10)
    e = calibration.eligibility(small.frame, small.race)
    assert not e["eligible"] and e["n_finishers"] == 10
    assert any("10 classified finishers with full laps, under 12" in r for r in e["reasons"])
    f = small.frame.copy()
    f["compound"] = "SOFT"
    e = calibration.eligibility(f, small.race)
    assert any(r.startswith("1 dry compound(s) raced") for r in e["reasons"])
    f = small.frame.copy()
    f.loc[f["compound"] == "MEDIUM", "clean"] = False   # every strategy runs MEDIUM
    e = calibration.eligibility(f, small.race)
    assert any(r.startswith("MEDIUM: used by") and "0 clean laps" in r for r in e["reasons"])


def test_a_retired_driver_or_a_gap_in_the_laps_is_not_a_full_lap_finisher(syn):
    f = syn.frame[~((syn.frame["driver"] == "D00") & (syn.frame["lap"] == 20))]
    f = f[~((f["driver"] == "D01") & (f["lap"] > 30))]
    assert calibration.eligibility(f, syn.race)["n_finishers"] == 14


def test_spain_is_eligible():
    data = cross_race.load_race("2021_04_spain", load_profile("local"))
    messages, note = cross_race.race_messages("2021_04_spain")
    e = calibration.eligibility(data.frame, data.race, messages)
    assert e["eligible"], e["reasons"]
    assert (e["n_finishers"], e["red_flags"], e["sc_periods"], e["vsc_periods"]) == (19, [], [[8, 10]], [])
    assert e["dry_compounds"] == ["MEDIUM", "SOFT"] and e["notes"] == []


# --- the fit options (off by default) ---------------------------------------------------------
PRIOR = {"SOFT": (79.0, 0.06), "MEDIUM": (81.0, 0.04)}


K = 25


def first_stints(noise=0.3, seed=0, n=10):
    """Race laps 2 to K of first stints only, n cars per compound: tyre age and race lap move
    together, so the to_snapshot design cannot tell wear from fuel (|corr| about 0.62 here)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, comp in enumerate(["SOFT"] * n + ["MEDIUM"] * n):
        base, wear = {"SOFT": (80.0, 0.1), "MEDIUM": (82.0, 0.05)}[comp]
        for lap in range(2, K + 1):
            rows.append({"driver": f"D{d}", "compound": comp, "tyre_age": float(lap), "lap": float(lap),
                         "clean": True, "lap_time_s": base + wear * lap - 0.05 * lap + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def fitted(params, k=K, frame=None):
    cfg = SimConfig(fit_window="to_snapshot", pre_race_base_delta_s=0.0,
                    degradation={"name": "linear_fuel", "params": params})
    sim = Simulator(first_stints() if frame is None else frame, cfg, synthetic_practice(coefs=PRIOR, in_stint=True))
    return sim.fitted_for(Snapshot(race=RACE, driver="X", laps_completed=k, compound="SOFT", tyre_age=k)).degradation


def test_options_off_by_default_leave_the_fit_report():
    rep = fitted({}).fit_report
    assert "collinear_hold" not in rep and "data_term_s2" not in rep


def test_collinear_hold_holds_wear_at_the_prior_while_the_correlation_is_high():
    plain, held = fitted({}), fitted({"collinear_hold": 0.5})
    corr = held.fit_report["collinear_hold"]["corr"]
    assert all(abs(c) > 0.5 for c in corr.values())   # first stints: wear and fuel collinear
    assert held.fit_report["collinear_hold"]["held"] == {"MEDIUM": True, "SOFT": True}
    prior_wear = {c: w for c, (_, w) in fitted({}, k=0).coefs.items()}
    for c in PRIOR:
        assert held.coefs[c][1] == pytest.approx(prior_wear[c], abs=1e-12)
        assert held.fit_report["shrinkage_weight"]["wear_s_per_lap"][c] == 0.0
        assert held.fit_report["sd"]["wear_s_per_lap"][c] == pytest.approx(SimConfig().shrinkage["wear_s_per_lap"])
    assert held.coefs["SOFT"][0] != plain.coefs["SOFT"][0] and held.fuel_slope != plain.fuel_slope
    names = held.fit_report["cov"]["names"]
    m = np.array(held.fit_report["cov"]["matrix"])
    j = names.index("wear_s_per_lap[SOFT]")
    assert m[j, j] == pytest.approx(SimConfig().shrinkage["wear_s_per_lap"] ** 2)
    assert np.count_nonzero(m[j]) == 1


def test_collinear_hold_above_every_correlation_is_the_ordinary_fit():
    plain, off = fitted({}), fitted({"collinear_hold": 0.999})
    assert off.fit_report["collinear_hold"]["held"] == {"MEDIUM": False, "SOFT": False}
    assert off.coefs == plain.coefs and off.fuel_slope == plain.fuel_slope
    with pytest.raises(NotImplementedError, match="wear_lower_bound"):
        fitted({"collinear_hold": 0.5, "wear_lower_bound": 0.0})


def test_prior_sd_fix_is_the_fit_with_taus_over_the_residual_sd():
    plain, fix = fitted({}), fitted({"divide_data_by_s2": True})
    s2 = fix.fit_report["data_term_s2"]
    assert s2 == pytest.approx(plain.fit_report["sd"]["residual_s"] ** 2)
    taus = {k: v / np.sqrt(s2) for k, v in SimConfig().shrinkage.items()}
    cfg = SimConfig(fit_window="to_snapshot", pre_race_base_delta_s=0.0, shrinkage=taus)
    sim = Simulator(first_stints(), cfg, synthetic_practice(coefs=PRIOR, in_stint=True))
    same = sim.fitted_for(Snapshot(race=RACE, driver="X", laps_completed=K, compound="SOFT", tyre_age=K)).degradation
    for c in PRIOR:
        assert fix.coefs[c] == pytest.approx(same.coefs[c], rel=1e-9)
    assert fix.fuel_slope == pytest.approx(same.fuel_slope, rel=1e-9)
    # the data's share of the precision falls as s2 rises
    w = fix.fit_report["shrinkage_weight"]["wear_s_per_lap"]["SOFT"]
    assert (w < plain.fit_report["shrinkage_weight"]["wear_s_per_lap"]["SOFT"]) == (s2 > 1)


# --- arms ----------------------------------------------------------------------------------------
def test_committed_arms_load_and_are_distinct():
    t = cross_race.arms_table()
    assert list(t["arm"]) == list(ARMS)
    assert t["config_hash"].is_unique and t["file_sha256"].str.len().eq(12).all()
    assert cross_race.arm_config(BASELINE) == SimConfig()
    assert cross_race.arm_config("collinear_hold_hinge").degradation.model_dump() == {
        "name": "linear_fuel_hinge", "params": {"collinear_hold": 0.5}}
    assert t.set_index("arm").loc["posterior", "role"] == "coverage only"
    assert cross_race.arm_config("wear_nonneg").degradation.params == {"wear_lower_bound": 0.0}


def test_derived_arm_names():
    cfg = cross_race.arm_config("hinge__s2__tau_0.5_1_2_1")
    assert cfg.degradation.params == {"divide_data_by_s2": True} and cfg.degradation.name == "linear_fuel_hinge"
    d = SimConfig().shrinkage
    assert cfg.shrinkage == {"base_s": d["base_s"] * 0.5, "wear_s_per_lap": d["wear_s_per_lap"],
                             "fuel_s_per_lap": d["fuel_s_per_lap"] * 2, "hinge_s_per_lap": d["hinge_s_per_lap"]}
    assert len(cross_race.tau_combos("hinge")) == 81 and len(cross_race.tau_combos(BASELINE)) == 27
    assert cross_race.tau_arm(BASELINE, (1, 1, 1, 1)) in cross_race.tau_combos(BASELINE)
    with pytest.raises(KeyError):
        cross_race.parse_arm("nope")
    with pytest.raises(ValueError):
        cross_race.parse_arm("baseline__what")


def test_pack_round_trip():
    t = {"a": pd.DataFrame({"x": [1, 2], "y": [0.1, float("nan")], "s": ["p", None]}),
         "b": pd.DataFrame({"flag": pd.array([True, pd.NA], dtype="boolean"), "v": np.array([1.5, 2.5])})}
    back = cross_race.unpack(cross_race.pack(t))
    assert back["a"]["x"].tolist() == [1, 2] and back["a"]["y"].iloc[0] == 0.1 and np.isnan(back["a"]["y"].iloc[1])
    assert back["b"]["flag"].tolist() == [True, None] and back["b"]["v"].tolist() == [1.5, 2.5]


def test_stop_gain_terms_sum_to_the_error(syn):
    cfg = cross_race.arm_config("hinge", syn.config)
    hind = Simulator(syn.frame, cfg, syn.practice).degradation
    sim = Simulator(syn.frame, cfg.model_copy(update={"fit_window": "to_snapshot"}), syn.practice)
    rows = calibration.stop_gain(sim, syn.frame, syn.race, hind, laps_k=[0, 10, 20])
    assert len(rows) > 20 and set(rows["window"]) == {"to_snapshot"}
    assert np.allclose(rows[calibration.STOP_GAIN_TERMS].sum(axis=1), rows["err_s"], atol=1e-9)
    floor = calibration.stop_gain(Simulator(syn.frame, cfg, syn.practice), syn.frame, syn.race, hind, laps_k=[0])
    assert np.allclose(floor["err_s"], floor["t_floor"])


# --- C: the selection rules ----------------------------------------------------------------------
def g_rows(race_errs: dict, window="to_snapshot"):
    """G rows: per race a list of err_offset_s at horizon 20."""
    return pd.DataFrame([{"race_id": r, "window": window, "driver": f"D{i}", "k": 10, "horizon": 20,
                          "err_offset_s": e} for r, errs in race_errs.items() for i, e in enumerate(errs)],
                        columns=["race_id", "window", "driver", "k", "horizon", "err_offset_s"])


def sg_rows(race_errs: dict, window="to_snapshot"):
    return pd.DataFrame([{"race_id": r, "window": window, "k": 10, "stop": f"D{i}:20", "err_s": e}
                         for r, errs in race_errs.items() for i, e in enumerate(errs)],
                        columns=["race_id", "window", "k", "stop", "err_s"])


RACES = [f"2099_{i:02d}_synthetic" for i in range(1, 9)]


def arm(g, sg):
    return {"g": g_rows(g), "stop_gain": sg_rows(sg), "factors": pd.DataFrame()}


def test_an_arm_better_on_average_with_a_ci_crossing_0_is_not_selected():
    base = {r: [10.0, 10.0] for r in RACES}
    # better by 6 s on one race, worse by 0.5 s on the seven others: lower on average, CI crosses 0
    lucky = {r: ([4.0, 4.0] if i == 0 else [10.5, 10.5]) for i, r in enumerate(RACES)}
    steady = {r: [9.0, 9.0] for r in RACES}
    data = {BASELINE: arm(base, {r: [1.0] for r in RACES}),
            "hinge": arm(lucky, {r: ([0.2] if i == 0 else [1.05]) for i, r in enumerate(RACES)}),
            "free_air": arm(steady, {r: [0.8] for r in RACES})}
    t = cross_race.compare_arms(data, BASELINE, ["to_snapshot", "full"], n_boot=500)
    for w in ("to_snapshot",):
        row = t[(t["window"] == w) & (t["arm"] == "hinge")].iloc[0]
        assert row["g_mae"] < row["g_mae_base"] and row["sg_mae"] < row["sg_mae_base"]
        assert row["sg_diff_lo"] < 0 < row["sg_diff_hi"] and not row["passes"]
        assert t[(t["window"] == w) & (t["arm"] == "free_air")].iloc[0]["passes"]
    assert cross_race.choose(t, "to_snapshot") == "free_air"


def test_the_world_rule_needs_the_g_ci_to_exclude_0():
    base = {r: [10.0, 10.0] for r in RACES}
    lucky = {r: ([4.0, 4.0] if i == 0 else [10.5, 10.5]) for i, r in enumerate(RACES)}
    data = {BASELINE: {"g": g_rows(base, "full"), "stop_gain": sg_rows({}, "full"), "factors": pd.DataFrame()},
            "hinge": {"g": g_rows(lucky, "full"), "stop_gain": sg_rows({}, "full"), "factors": pd.DataFrame()}}
    t = cross_race.compare_arms(data, BASELINE, ["full"], n_boot=500)
    row = t[t["arm"] == "hinge"].iloc[0]
    assert row["g_mae"] < row["g_mae_base"] and row["g_diff_lo"] < 0 < row["g_diff_hi"]
    assert not row["passes"] and cross_race.choose(t, "full") == BASELINE


def test_belief_non_inferiority_margin_and_ties_to_the_simpler_arm():
    base = {r: [10.0] for r in RACES}
    data = {BASELINE: arm(base, {r: [1.0] for r in RACES}),
            # better stop gain, G worse by 3 s on every race: fails the +2 s margin
            "hinge": arm({r: [13.0] for r in RACES}, {r: [0.5] for r in RACES}),
            # the same improvement, two arms: the simpler one wins the tie
            "hinge_free": arm({r: [10.5] for r in RACES}, {r: [0.7] for r in RACES}),
            "wear_nonneg": arm({r: [10.5] for r in RACES}, {r: [0.7] for r in RACES}),
            "posterior": arm({r: [5.0] for r in RACES}, {r: [0.1] for r in RACES})}
    t = cross_race.compare_arms(data, BASELINE, ["to_snapshot"], n_boot=500).set_index("arm")
    assert not t.loc["hinge", "passes"] and t.loc["hinge", "sg_diff_hi"] < 0
    assert t.loc["hinge_free", "passes"] and t.loc["wear_nonneg", "passes"]
    assert not t.loc["posterior", "passes"]            # coverage only
    assert cross_race.choose(t.reset_index(), "to_snapshot") == "wear_nonneg"


def test_one_race_never_selects():
    data = {BASELINE: arm({RACES[0]: [10.0]}, {RACES[0]: [1.0]}),
            "hinge": arm({RACES[0]: [1.0]}, {RACES[0]: [0.1]})}
    t = cross_race.compare_arms(data, BASELINE, ["to_snapshot", "full"], n_boot=100)
    assert not t["passes"].any() and t["g_diff_lo"].isna().all()


# --- end to end and resumability ------------------------------------------------------------------
RUN_RACES = ["2021_04_spain", "2099_01_synthetic", "2099_02_synthetic"]


def small_profile():
    p = load_profile("local")
    return p.model_copy(update={"g": p.g.model_copy(update={"laps_every": 20, "n_boot": 20}),
                                "sim": p.sim.model_copy(update={"n_samples": 5}),
                                "parallel": p.parallel.model_copy(update={"n_jobs": 1})})


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("cross")
    log = []
    record = cross_race.run(RUN_RACES, small_profile(), out, tau_multipliers=(1.0,), log=log.append)
    cross_race.report(out, small_profile(), n_boot=50)
    return out, record, log


def test_end_to_end_on_spain_and_two_synthetic_races(run_dir):
    out, record, log = run_dir
    assert record["eligible"] == RUN_RACES and record["belief_choice"] in ARMS
    assert record["s2_arm"] == f"{record['belief_choice']}__s2" and len(record["grid"]) == 1
    for r in RUN_RACES:
        assert (out / "races" / r / f"{r}.parquet").is_file()
        for a in ARMS:
            assert (out / "h" / "arms" / a / f"{r}.parquet").is_file()
    tables = cross_race.unpack(pd.read_parquet(out / "races" / "2021_04_spain" / "2021_04_spain.parquet"))
    assert {"replay", "summary", "by_kind", "periods", "gate", "g", "hindsight", "eligibility"} <= set(tables)
    assert {"clean_free", "clean_pack"} <= set(tables["by_kind"]["kind"])
    assert set(calibration.INTERVAL_COLUMNS) <= set(tables["g"].columns)
    assert set(tables["gate"]["mode"]) == {"simulated_order"}
    fold = cross_race.unpack(pd.read_parquet(out / "h" / "arms" / "posterior" / "2021_04_spain.parquet"))
    assert set(fold["g"]["window"]) == {"full", "to_snapshot", "pre_race"}
    assert (fold["g"]["pred_p95_s"] - fold["g"]["pred_p05_s"]).max() > 0
    factors = pd.read_csv(out / "factors.csv", dtype={"sc_start_laps": str})
    assert list(factors["race_id"]) == RUN_RACES
    spain = factors.set_index("race_id").loc["2021_04_spain"]
    assert spain["sc_count"] == 1 and str(spain["sc_start_laps"]) == "8"
    assert {"observed_pack_penalty_s", "free_air_share", "hindsight_fuel_s_per_lap", "base_delta_s[SOFT]"} <= set(factors)
    sel = pd.read_csv(out / "selection.csv")
    assert set(sel.loc[sel["stage"] == "arms", "arm"]) == set(ARMS) and sel["provisional"].all()
    assert (sel.loc[sel["stage"] == "arms"].groupby("window").size() == len(ARMS)).all()
    md = (out / "summary.md").read_text()
    for heading in ("## Eligibility", "## Selection", "### Prior-sd fix", "### Tau grid", "## Coverage",
                    "## Gate and pairs", "## Stop-gain decomposition", "## Factor summary", "## What changed from Spain"):
        assert heading in md
    assert "PROVISIONAL: 3 eligible race(s), fewer than 8" in md and "the defaults stay" in md
    assert "Synthetic races in this run" in md and "No race features file" in md


def test_rerun_skips_everything_and_a_missing_unit_reruns_alone(run_dir):
    out, record, _ = run_dir
    path = out / "h" / "arms" / "hinge" / "2099_01_synthetic.parquet"
    before, table = path.read_bytes(), pd.read_parquet(path)
    log = []
    again = cross_race.run(RUN_RACES, small_profile(), out, tau_multipliers=(1.0,), log=log.append)
    status = [x for x in log if x.startswith("  ")]
    assert status and all(x.endswith(": skipped") for x in status)
    assert again == record and path.read_bytes() == before
    jobs._meta_path(path).unlink()   # an interrupted task: the output without its meta
    log = []
    cross_race.run(RUN_RACES, small_profile(), out, tau_multipliers=(1.0,), log=log.append)
    done = [x for x in log if x.endswith(": done")]
    assert done == ["  h_fold/hinge 2099_01_synthetic: done"]
    pd.testing.assert_frame_equal(pd.read_parquet(path), table)   # the rerun gives the same rows


def test_race_features_join_when_present(run_dir, tmp_path):
    out, _, _ = run_dir
    feats = tmp_path / "race_features.csv"
    pd.DataFrame({"race_id": ["2021_Spain"], "track_length_km": [4.675]}).to_csv(feats, index=False)
    outputs = cross_race.race_outputs(out, RUN_RACES, small_profile())
    t = cross_race.factors_table(outputs, feats).set_index("race_id")
    assert t.loc["2021_04_spain", "feature:track_length_km"] == 4.675
    assert np.isnan(t.loc["2099_01_synthetic", "feature:track_length_km"])
    assert "feature:track_length_km" not in cross_race.factors_table(outputs, tmp_path / "missing.csv")


def test_eligibility_json_lists_every_race(run_dir):
    out, _, _ = run_dir
    elig = json.loads((out / "eligibility.json").read_text())
    assert [e["race_id"] for e in elig] == RUN_RACES and all(e["eligible"] for e in elig)
