"""Cross-race calibration report (SPEC 0016 and its amendment): eligibility, the selection
rules, the collinear-hold and prior-sd fit options, the arms, and the end-to-end run on Spain
plus two synthetic races. Offline: synthetic races and the committed Spain cache."""
import json
from pathlib import Path

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
    assert any("10 eligible drivers (laps 1..n, at most 2 down), under 12" in r for r in e["reasons"])
    f = small.frame.copy()
    f["compound"] = "SOFT"
    e = calibration.eligibility(f, small.race)
    assert any(r.startswith("1 dry compound(s) raced") for r in e["reasons"])
    f = small.frame.copy()
    f.loc[f["compound"] == "MEDIUM", "clean"] = False   # every strategy runs MEDIUM
    e = calibration.eligibility(f, small.race)
    assert any(r.startswith("MEDIUM: used by") and "0 clean laps" in r for r in e["reasons"])


def test_drivers_are_the_eval_samplers_a_gap_or_three_laps_down_is_out(syn):
    f = syn.frame[~((syn.frame["driver"] == "D00") & (syn.frame["lap"] == 20))]   # a gap
    f = f[~((f["driver"] == "D01") & (f["lap"] > 48))]   # two laps down: in
    f = f[~((f["driver"] == "D02") & (f["lap"] > 47))]   # three laps down: out
    e = calibration.eligibility(f, syn.race)
    assert e["n_finishers"] == 14 and "D01" in e["drivers"] and not {"D00", "D02"} & set(e["drivers"])
    from src.eval.scenarios import eligible_drivers
    assert e["drivers"] == eligible_drivers(f, syn.race.total_laps)


def test_spain_is_eligible_with_v1s_drivers():
    data = cross_race.load_race("2021_04_spain", load_profile("local"))
    messages, note = cross_race.race_messages("2021_04_spain")
    e = calibration.eligibility(data.frame, data.race, messages)
    assert e["eligible"], e["reasons"]
    assert (e["n_finishers"], e["red_flags"], e["sc_periods"], e["vsc_periods"]) == (19, [], [[8, 10]], [])
    assert e["dry_compounds"] == ["MEDIUM", "SOFT"] and e["notes"] == []
    v1 = json.loads((Path(__file__).parent.parent / "data" / "scenarios" / "v1.json").read_text())
    assert e["drivers"] == sorted({i["driver"] for i in v1["items"] if i["race_id"] == "2021_04_spain"})


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
    bound_off = fitted({"collinear_hold": 0.999, "wear_lower_bound": 0.0})
    assert bound_off.coefs == fitted({"wear_lower_bound": 0.0}).coefs


def held_and_released(seed=1):
    """SOFT in first stints (collinear, held) and MEDIUM on staggered later stints whose wear
    is negative in the data (released, so the bound binds)."""
    rng = np.random.default_rng(seed)
    soft = first_stints()
    rows = [{"driver": f"M{d}", "compound": "MEDIUM", "tyre_age": float(lap - 2 - 3 * d + 1), "lap": float(lap),
             "clean": True, "lap_time_s": 82.0 - 0.08 * (lap - 2 - 3 * d + 1) - 0.05 * lap + rng.normal(0, 0.3)}
            for d in range(10) for lap in range(2 + 3 * d, 17 + 3 * d)]
    return pd.concat([soft[soft["compound"] == "SOFT"], pd.DataFrame(rows)], ignore_index=True)


def test_collinear_hold_with_the_bound_holds_while_collinear_and_bounds_once_released():
    f = held_and_released()
    held = fitted({"collinear_hold": 0.5}, k=45, frame=f)
    both = fitted({"collinear_hold": 0.5, "wear_lower_bound": 0.0}, k=45, frame=f)
    assert both.fit_report["collinear_hold"]["held"] == {"MEDIUM": False, "SOFT": True}
    assert held.coefs["MEDIUM"][1] < 0 and both.coefs["MEDIUM"][1] == 0.0
    assert both.fit_report["bounded"] == {"MEDIUM": True, "SOFT": False}
    assert both.coefs["SOFT"][1] == pytest.approx(held.coefs["SOFT"][1], abs=1e-12)   # the prior's wear
    names = both.fit_report["cov"]["names"]
    m = np.array(both.fit_report["cov"]["matrix"])
    j = names.index("wear_s_per_lap[SOFT]")
    assert m[j, j] == pytest.approx(SimConfig().shrinkage["wear_s_per_lap"] ** 2) and np.count_nonzero(m[j]) == 1
    unc = both.fit_report["unconstrained"]
    assert unc["names"] == names and np.array(unc["matrix"])[j, j] == pytest.approx(m[j, j])
    draws = both.with_draw("posterior", wear_lower_bound=0.0)
    rng = np.random.default_rng(0)
    assert all(draws.draw(rng).coefs["MEDIUM"][1] >= 0.0 for _ in range(50))


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


def arm(g, sg, window="to_snapshot"):
    return {"g": g_rows(g, window), "stop_gain": sg_rows(sg, window), "factors": pd.DataFrame()}


def each(v, n=1):
    return {r: [v] * n for r in RACES}


def test_an_arm_better_on_average_with_a_ci_crossing_0_is_not_selected():
    # better by 6 s on one race, worse by 0.5 s on the seven others: lower on average, CI crosses 0
    lucky = {r: ([4.0, 4.0] if i == 0 else [10.5, 10.5]) for i, r in enumerate(RACES)}
    data = {BASELINE: arm(each(10.0, 2), each(1.0)),
            "hinge": arm(lucky, {r: ([0.2] if i == 0 else [1.05]) for i, r in enumerate(RACES)}),
            "free_air": arm(each(9.0, 2), each(0.8))}
    t = cross_race.compare_arms(data, BASELINE, ["to_snapshot"], n_boot=500).set_index("arm")
    row = t.loc["hinge"]
    assert row["g_diff"] < 0 and row["sg_diff"] < 0 and row["sg_mae"] < row["sg_mae_base"]
    assert row["sg_diff_lo"] < 0 < row["sg_diff_hi"] and not row["passes"]
    assert t.loc["free_air", "passes"] and t.loc[BASELINE, "role"] == "comparator"
    assert cross_race.choose(t.reset_index(), "to_snapshot") == "free_air"


def test_the_world_rule_needs_the_g_ci_to_exclude_0():
    lucky = {r: ([4.0, 4.0] if i == 0 else [10.5, 10.5]) for i, r in enumerate(RACES)}
    data = {BASELINE: arm(each(10.0, 2), {}, "full"), "hinge": arm(lucky, {}, "full")}
    t = cross_race.compare_arms(data, BASELINE, ["full"], n_boot=500)
    row = t[t["arm"] == "hinge"].iloc[0]
    assert row["g_diff"] < 0 and row["g_diff_lo"] < 0 < row["g_diff_hi"]
    assert not row["passes"] and cross_race.choose(t, "full") == BASELINE


def test_the_point_estimate_is_race_weighted_not_row_pooled():
    # one race with 20 rows 1 s better, seven races with one row 0.5 s worse: pooled lower, race-weighted higher
    base = {r: ([10.0] * 20 if i == 0 else [10.0]) for i, r in enumerate(RACES)}
    heavy = {r: ([9.0] * 20 if i == 0 else [10.5]) for i, r in enumerate(RACES)}
    t = cross_race.compare_arms({BASELINE: arm(base, {}, "full"), "hinge": arm(heavy, {}, "full")}, BASELINE,
                                ["full"], n_boot=200).set_index("arm")
    r = t.loc["hinge"]
    assert r["g_diff_pooled"] < 0 < r["g_diff"]
    assert r["g_diff"] == pytest.approx((-1.0 + 7 * 0.5) / 8)
    assert r["g_mae"] == pytest.approx((9.0 + 7 * 10.5) / 8) and not r["passes"]


def test_belief_non_inferiority_margin_and_ties_to_the_simpler_arm():
    data = {BASELINE: arm(each(10.0), each(1.0)),
            "hinge": arm(each(13.0), each(0.5)),          # better stop gain, G worse by 3 s: fails the margin
            "hinge_free": arm(each(10.5), each(0.7)),     # the same improvement twice: the simpler wins
            "collinear_hold": arm(each(10.5), each(0.7)),
            "posterior": arm(each(5.0), each(0.1))}       # coverage only
    t = cross_race.compare_arms(data, BASELINE, ["to_snapshot"], n_boot=500).set_index("arm")
    assert not t.loc["hinge", "passes"] and t.loc["hinge", "sg_diff_hi"] < 0
    assert t.loc["hinge_free", "passes"] and t.loc["collinear_hold", "passes"]
    assert not t.loc["posterior", "passes"]
    assert cross_race.choose(t.reset_index(), "to_snapshot") == "collinear_hold"


def all_arms(**over):
    """Every committed arm at the baseline's numbers, with `over` replacing some."""
    return {a: over.get(a, arm(each(10.0), each(1.0))) for a in ARMS}


def test_the_belief_is_incumbent_first_the_bound_stays_unless_removing_it_passes():
    # the baseline (no bound) is no better than the incumbent wear_nonneg: the bound stays, even
    # where the bound itself would not have passed against the baseline (a tie)
    t, bound, pick = cross_race.belief_selection(all_arms(), n_boot=200)
    s1 = t[t["stage"] == "bound"]
    assert list(s1["arm"]) == ["wear_nonneg", BASELINE] and s1["role"].tolist() == ["comparator", "candidate"]
    assert (bound, pick) == ("wear_nonneg", "wear_nonneg") and not s1["passes"].any()
    # stage 2 runs against the stage-1 result: collinear_hold beats the baseline's numbers but
    # not the bound's, collinear_hold_nonneg beats the bound
    data = all_arms(wear_nonneg=arm(each(10.0), each(0.6)), collinear_hold=arm(each(10.0), each(0.8)),
                    collinear_hold_nonneg=arm(each(10.0), each(0.4)))
    t, bound, pick = cross_race.belief_selection(data, n_boot=200)
    assert bound == "wear_nonneg" and pick == "collinear_hold_nonneg"
    s2 = t[t["stage"] == "candidates"].set_index("arm")
    assert (s2["comparator"] == "wear_nonneg").all() and BASELINE not in s2.index
    assert not s2.loc["collinear_hold", "passes"] and s2.loc["collinear_hold", "sg_diff"] > 0
    # removing the bound passes the rule: stage 2 then runs against the baseline
    data = all_arms(wear_nonneg=arm(each(10.0), each(1.4)), collinear_hold=arm(each(10.0), each(0.8)))
    t, bound, pick = cross_race.belief_selection(data, n_boot=200)
    assert t[t["stage"] == "bound"].set_index("arm").loc[BASELINE, "passes"]
    assert (bound, pick) == (BASELINE, "collinear_hold")


def test_current_defaults_name_each_windows_incumbent_and_match_the_config():
    t = cross_race.current_defaults().set_index("window")
    assert t.loc["to_snapshot", "incumbent"] == "wear_nonneg" and t.loc["to_snapshot", "source"] == "a priori"
    assert "belief.yaml" in t.loc["to_snapshot", "set by"]
    assert (t.loc[["full", "pre_race"], "incumbent"] == BASELINE).all()
    assert (t.loc[["full", "pre_race"], "source"] == "default").all() and t["matches the config"].all()
    assert t.loc["to_snapshot", "config_hash"] == calibration.config_hash(cross_race.arm_config("wear_nonneg"))


def test_calibrator_arms_are_na_in_the_full_window():
    assert cross_race.same_by_construction("none", BASELINE, "full")
    assert cross_race.same_by_construction("collinear_hold", BASELINE, "full")
    assert not cross_race.same_by_construction("hinge", BASELINE, "full")
    assert not cross_race.same_by_construction("wear_nonneg", BASELINE, "full")
    assert not cross_race.same_by_construction("none", BASELINE, "to_snapshot")
    data = {BASELINE: arm(each(10.0), {}, "full"), "none": arm(each(9.0), {}, "full")}
    t = cross_race.compare_arms(data, BASELINE, ["full"], n_boot=100).set_index("arm")
    assert t.loc["none", "role"] == cross_race.NA_ROLE and not t.loc["none", "passes"]
    assert "| none | n/a (same by construction) | 8 | n/a |" in cross_race._selection_md(t.reset_index())


def test_one_race_never_selects():
    data = {BASELINE: arm({RACES[0]: [10.0]}, {RACES[0]: [1.0]}),
            "hinge": arm({RACES[0]: [1.0]}, {RACES[0]: [0.1]})}
    t = cross_race.compare_arms(data, BASELINE, ["to_snapshot", "full"], n_boot=100)
    assert not t["passes"].any() and t["g_diff_lo"].isna().all()


def test_synthetic_races_never_go_into_the_committed_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cross_race, "CROSS_DIR", tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"races": ["2021_04_spain", SYN]}))
    with pytest.raises(ValueError, match="committed report"):
        cross_race.report(tmp_path)
    with pytest.raises(ValueError, match="committed report"):
        cross_race.run([SYN], load_profile("local"), tmp_path)
    cross_race.guard_committed(tmp_path / "elsewhere", [SYN])   # anywhere else is fine
    cross_race.guard_committed(tmp_path, ["2021_04_spain"])


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
    assert {"replay", "summary", "by_kind", "periods", "gate", "g", "hindsight", "wear_check",
            "eligibility"} <= set(tables)
    assert list(tables["wear_check"]["compound"]) == ["MEDIUM", "SOFT"]
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
    assert sel["provisional"].all()
    for w in ("full", "pre_race"):
        assert set(sel.loc[(sel["stage"] == "arms") & (sel["window"] == w), "arm"]) == set(ARMS)
    s1 = sel[sel["stage"] == "bound"]
    assert list(s1["arm"]) == ["wear_nonneg", BASELINE] and (s1["window"] == "to_snapshot").all()
    s2 = sel[sel["stage"] == "candidates"]
    assert len(s2) == len(ARMS) - 1 and (s2["comparator"] == record["belief_bound"]).all()
    na = sel.loc[(sel["window"] == "full") & sel["arm"].isin(["none", "collinear_hold"]), "role"]
    assert set(na) == {cross_race.NA_ROLE}
    assert set(sel.loc[sel["stage"] == "prior-sd fix", "arm"]) == {record["s2_arm"]}
    grid = pd.read_csv(out / "tau_grid.csv")
    assert len(grid) == 1 and grid["sg_n"].iloc[0] > 0 and {"non_inferior", "passes", "g_diff_hi"} <= set(grid)
    gfold = cross_race.unpack(pd.read_parquet(out / "h" / "arms" / record["grid"][0] / "2021_04_spain.parquet"))
    assert set(gfold["stop_gain"]["window"]) == {"to_snapshot"} and set(gfold["g"]["window"]) == {"to_snapshot"}
    md = (out / "summary.md").read_text()
    for heading in ("## Eligibility", "## Selection", "### Prior-sd fix", "### Tau grid", "## Coverage",
                    "## Gate and pairs", "## Current defaults", "## Stop-gain decomposition", "## Factor summary", "## What changed from Spain"):
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


def test_report_writes_stop_gain_and_coverage_summaries(run_dir):
    out, _, _ = run_dir
    sg = pd.read_csv(out / "stop_gain_summary.csv")
    assert {"window", "arm", "k_bucket", "n", "bias_s", "mae_s", "mae_ci_lo", "mae_ci_hi",
            *calibration.STOP_GAIN_TERMS} <= set(sg)
    assert set(sg["arm"]) == set(ARMS) - {"posterior"} and sg["n"].min() > 0
    cov = pd.read_csv(out / "coverage_summary.csv")
    assert {"coverage_68", "coverage_90", "coverage_raw_68", "coverage_raw_90", "mean_width_90_s"} <= set(cov)
    assert "posterior" in set(cov["arm"])


def test_eligibility_json_lists_every_race(run_dir):
    out, _, _ = run_dir
    elig = json.loads((out / "eligibility.json").read_text())
    assert [e["race_id"] for e in elig] == RUN_RACES and all(e["eligible"] for e in elig)


def test_wear_check_reads_within_stint_wear_net_of_fuel():
    """SPEC 0023 A: per (driver, stint) of 8+ clean laps, the slope on tyre age less the fuel
    slope; short stints and unclean laps stay out."""
    from types import SimpleNamespace
    rows = []
    for d, (stint, n, start) in enumerate([(1, 12, 0), (2, 10, 15), (1, 5, 0)]):
        for age in range(1, n + 1):
            lap = start + age
            rows.append({"driver": f"D{d}", "stint": stint, "compound": "SOFT", "tyre_age": age, "lap": lap,
                         "clean": True, "lap_time_s": 80.0 + (0.05 + 0.01 * d) * age - 0.04 * lap})
    rows.append({**rows[0], "lap_time_s": 99.0, "clean": False})
    model = SimpleNamespace(coefs={"SOFT": (80.0, 0.03), "MEDIUM": (81.0, 0.02)}, fuel_slope=-0.04)
    t = cross_race.wear_check(pd.DataFrame(rows), model).set_index("compound")
    assert t.loc["SOFT", "fitted_wear_s_per_lap"] == pytest.approx(0.03)
    assert t.loc["SOFT", "within_stint_wear_s_per_lap"] == pytest.approx(0.055)   # median of 0.05, 0.06
    assert t.loc["SOFT", "n_stints"] == 2 and t.loc["MEDIUM", "n_stints"] == 0
    assert np.isnan(t.loc["MEDIUM", "within_stint_wear_s_per_lap"])
