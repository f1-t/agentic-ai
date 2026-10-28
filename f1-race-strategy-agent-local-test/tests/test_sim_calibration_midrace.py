"""Mid-race replay calibration on synthetic races with a known model. Offline, no model."""
import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, Simulator, calibration
from src.sim.config import EventEffects
from test_sim import RACE, synthetic_practice
from test_sim_position import driver_rows, race_frame

A_PLAN = [("SOFT", 20), ("MEDIUM", 30)]
B_PLAN = [("SOFT", 10), ("MEDIUM", 25), ("SOFT", 15)]
WINDOWS = ("full", "to_snapshot", "pre_race")


@pytest.fixture(scope="module")
def frame():
    return race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN))


def cfg(**kw):
    """The synthetic races have no standing start penalty on lap 1, so the lap 1 term is off."""
    return SimConfig(effects=EventEffects(lap1_extra_s=0.0), **kw)


def factory(frame, name="linear", practice=None):
    """Fitted on the race itself; by default practice follows the same model, so every window is exact."""
    practice = synthetic_practice() if practice is None else practice

    def make(window):
        return Simulator(frame, cfg(degradation={"name": name}, fit_window=window), practice)
    return make


# --- snapshot and plan ---------------------------------------------------------
def hand_frame():
    """BBB on a set three laps old at the start, SC on laps 12-13, VSC on lap 30, stops on laps 10 and 35."""
    rows = driver_rows("BBB", "T", B_PLAN)
    for r in rows:
        if r["stint"] == 1:
            r["tyre_age"] += 3
        if r["lap"] in (12, 13):
            r["track_status"] = "4"
        if r["lap"] == 30:
            r["track_status"] = "6"
    return pd.DataFrame(rows)


@pytest.mark.parametrize("k,compound,age,stints", [
    (0, "SOFT", 3, []),
    (4, "SOFT", 7, []),
    (10, "MEDIUM", 0, [("SOFT", 10, 1, 3)]),          # pitted at the end of lap 10
    (11, "MEDIUM", 1, [("SOFT", 10, 1, 3)]),
    (40, "SOFT", 5, [("SOFT", 10, 1, 3), ("MEDIUM", 25, 11, 0)]),
])
def test_snapshot_at_reads_the_lap_k_state(k, compound, age, stints):
    f = hand_frame()
    snap = calibration.snapshot_at(f, RACE, "BBB", k)
    assert snap.laps_completed == k
    assert (snap.compound, snap.tyre_age) == (compound, age)
    assert [(s.compound, s.laps, s.start_lap, s.start_age) for s in snap.stints] == stints
    g = f.set_index("lap")
    assert snap.elapsed_s == (0.0 if k == 0 else pytest.approx(g.loc[k, "end_s"] - g.loc[1, "start_s"]))
    assert snap.standing_start == (k == 0)
    assert snap.sets_available is None


def test_snapshot_history_and_status():
    f = hand_frame()
    assert calibration.snapshot_at(f, RACE, "BBB", 12).track_status == "sc"
    assert calibration.snapshot_at(f, RACE, "BBB", 12).sc_laps == [12]
    late = calibration.snapshot_at(f, RACE, "BBB", 35)
    assert (late.sc_laps, late.vsc_laps, late.track_status) == ([12, 13], [30], "green")
    assert calibration.snapshot_at(f, RACE, "BBB", 0).sc_laps == []


@pytest.mark.parametrize("k", [0, 1, 9, 10, 11, 34, 35, 36, 49])
def test_remaining_plan_covers_the_rest_and_continues_the_set(k):
    f = hand_frame()
    snap = calibration.snapshot_at(f, RACE, "BBB", k)
    plan = calibration.remaining_plan(f, "BBB", k)
    assert plan.total_laps == snap.remaining_laps == 50 - k
    assert plan.stints[0].compound == snap.compound
    assert plan.stints[0].start_age == snap.tyre_age
    assert plan.num_stops == sum(1 for lap in (10, 35) if lap > k)


def test_in_lap_at_k_pays_its_pit_loss_once(frame):
    """k = 10 is BBB's in lap: the real in lap, pit loss and all, is in elapsed_s and the
    remaining plan has only the lap 35 stop."""
    sim = factory(frame)("full")
    snap = calibration.snapshot_at(frame, RACE, "BBB", 10)
    res = sim.simulate(snap, calibration.remaining_plan(frame, "BBB", 10), rules=calibration.REPLAY_RULES)
    assert res.pit_laps == [35]
    assert res.pit_loss_total_s == pytest.approx(RACE.pit_loss_s)
    g = frame[frame["driver"] == "BBB"].set_index("lap")
    assert snap.elapsed_s == pytest.approx(g.loc[1:10, "lap_time_s"].sum())   # lap 10 includes the pit loss
    assert res.race_time_s == pytest.approx(g["lap_time_s"].sum())          # elapsed plus remaining, two stops


def test_default_laps_k_adds_laps_around_sc_and_vsc_periods():
    f = hand_frame()
    assert calibration.default_laps_k(f, 50) == [0, 5, 10, 11, 14, 15, 20, 25, 29, 30, 31, 35, 40]


# --- rows ----------------------------------------------------------------------
def test_k0_full_rows_equal_run(frame):
    sim = Simulator(frame)  # default config, lap 1 term included on both sides
    table, _, _ = calibration.run(sim, frame, RACE)
    pos, _ = calibration.positions(sim, frame, RACE)
    rows = calibration.midrace(lambda w: sim, frame, RACE, laps_k=[0], windows=["full"]).set_index("driver")
    assert sorted(rows.index) == sorted(table.index)
    for d in table.index:
        assert rows.loc[d, "real_remaining_s"] == pytest.approx(table.loc[d, "real_s"], abs=1e-9)
        assert rows.loc[d, "pred_remaining_s"] == pytest.approx(table.loc[d, "pred_s"], abs=1e-9)
        assert rows.loc[d, "err_s"] == pytest.approx(table.loc[d, "err_s"], abs=1e-9)
        assert rows.loc[d, "pred_pos_sim"] == pos.loc[d, "pred_pos_simulated_order"]
        assert rows.loc[d, "pred_pos_real"] == pos.loc[d, "pred_pos_vs_real"]
        assert rows.loc[d, "real_pos"] == pos.loc[d, "real_pos"]
        assert rows.loc[d, "horizon"] == table.loc[d, "laps"]
    assert rows["err_offset_s"].isna().all()


def test_k0_full_rows_equal_run_on_the_spain_shape():
    """Lapped drivers and pace offsets, against a sim fitted elsewhere."""
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)))
    f = race_frame(("CCC", "U", A_PLAN, 0.2), ("DDD", "U", [("SOFT", 15), ("MEDIUM", 34)], -0.3))
    table, _, _ = calibration.run(sim, f, RACE)
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[0], windows=["full"]).set_index("driver")
    pd.testing.assert_series_equal(rows["pred_remaining_s"], table["pred_s"], check_names=False)
    pd.testing.assert_series_equal(rows["err_s"], table["err_s"], check_names=False)


def test_pred_remaining_exact_on_noise_free_data_at_every_k(frame):
    rows = calibration.midrace(factory(frame), frame, RACE, laps_k=list(range(0, 50)), windows=WINDOWS)
    assert len(rows) == 3 * 2 * 50
    assert rows["err_s"].abs().max() < 1e-6
    later = rows[rows["k"] >= 6]
    assert later["err_offset_s"].abs().max() < 1e-6
    assert later["pace_offset_s"].notna().all()
    assert rows.loc[rows["k"] == 0, "err_offset_s"].isna().all()
    assert (rows["horizon"] == 50 - rows["k"]).all()
    assert (rows["pred_pos_sim"] == rows["pred_pos_real"]).all()


def test_to_snapshot_at_last_lap_equals_full(frame):
    rows = calibration.midrace(factory(frame), frame, RACE, laps_k=[RACE.total_laps - 1],
                               windows=["full", "to_snapshot"]).set_index(["window", "driver"])
    for d in ("AAA", "BBB"):
        assert rows.loc[("to_snapshot", d), "pred_remaining_s"] == pytest.approx(
            rows.loc[("full", d), "pred_remaining_s"], abs=1e-6)


def test_to_snapshot_moves_from_pre_race_to_full_when_practice_disagrees(frame):
    """to_snapshot is shrunk toward the pre_race fit, so with practice off the race model it
    starts at pre_race and ends next to full."""
    practice = synthetic_practice(coefs={"SOFT": (80.6, 0.13), "MEDIUM": (82.5, 0.06)})
    rows = calibration.midrace(factory(frame, practice=practice), frame, RACE,
                               laps_k=[0, 10, 49]).set_index(["window", "driver", "k"])
    for d in ("AAA", "BBB"):
        err = {(w, k): abs(rows.loc[(w, d, k), "err_s"]) for w in WINDOWS for k in (0, 10, 49)}
        assert err[("full", 0)] < 1e-6 and err[("full", 49)] < 1e-6
        assert err[("to_snapshot", 0)] == pytest.approx(err[("pre_race", 0)])
        assert err[("to_snapshot", 10)] < err[("pre_race", 10)]
        assert err[("to_snapshot", 49)] < 0.25 * err[("pre_race", 49)]


def test_pace_offset_removes_a_driver_offset():
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)), cfg(degradation={"name": "linear"}))
    f = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN, -0.4))
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[0, 20], windows=["full"]).set_index(["k", "driver"])
    assert rows.loc[(20, "BBB"), "err_s"] == pytest.approx(0.4 * 30, abs=1e-6)
    assert rows.loc[(20, "BBB"), "pace_offset_s"] == pytest.approx(-0.4, abs=1e-6)
    assert rows.loc[(20, "BBB"), "err_offset_s"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(rows.loc[(0, "BBB"), "err_offset_s"])


def test_pace_offset_uses_laps_up_to_k_only_and_not_the_clean_flag():
    """Laps after k never enter; a lap the whole-race filter cut still counts if it is green and
    not a pit lap; neutralised and pit laps do not count."""
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)), cfg(degradation={"name": "linear"}))
    f = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN))
    bbb = f["driver"] == "BBB"
    f.loc[bbb & f["lap"].between(2, 7), "lap_time_s"] += 1.0   # 6 slow green laps
    f.loc[bbb & f["lap"].between(2, 7), "clean"] = False       # which the whole-race filter cut
    f.loc[bbb & (f["lap"] == 8), "track_status"] = "4"          # an SC lap, ignored
    f.loc[bbb & (f["lap"] > 20), "lap_time_s"] -= 3.0           # the future, ignored
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[6, 9], windows=["full"]).set_index(["k", "driver"])
    assert rows.loc[(6, "BBB"), "pace_offset_s"] == pytest.approx(1.0)   # laps 2-6, all slow
    assert rows.loc[(9, "BBB"), "pace_offset_s"] == pytest.approx(1.0)   # laps 2-7 slow, 9 not: median 1.0
    assert rows.loc[(6, "AAA"), "pace_offset_s"] == pytest.approx(0.0, abs=1e-9)


def test_too_few_green_laps_gives_no_offset(frame):
    rows = calibration.midrace(factory(frame), frame, RACE, laps_k=[3, 5, 6], windows=["full"]).set_index(["k", "driver"])
    assert rows.loc[3, "pace_offset_s"].isna().all() and rows.loc[5, "pace_offset_s"].isna().all()
    assert rows.loc[6, "pace_offset_s"].notna().all()   # laps 2-6


def test_positions_in_both_modes():
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)), cfg())
    f = race_frame(("AAA", "T", A_PLAN, 0.0, (1, 2)), ("BBB", "T", B_PLAN, -0.5, (2, 1)))
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[0, 45], windows=["full"]).set_index(["k", "driver"])
    # BBB is half a second a lap quicker than predicted: against AAA's real total it is second at the start,
    # among the predicted totals it is second too (its plan is 2.3 s slower on paper)
    assert rows.loc[(0, "BBB"), "pred_pos_real"] == 2
    assert rows.loc[(0, "BBB"), "pred_pos_sim"] == 2
    assert rows.loc[(0, "BBB"), "real_pos"] == 1
    assert rows.loc[(0, "BBB"), "abs_pos_err_real"] == 1
    assert rows.loc[(45, "BBB"), "pred_pos_real"] == 1   # five laps out the error is too small to matter
    assert rows.loc[(45, "BBB"), "pred_pos_sim"] == 1


def test_simulated_order_cancels_a_common_bias():
    """Every driver 1 s a lap slower than the model: against real totals each is predicted to win,
    while the order of the predicted totals is the real order."""
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)), cfg())
    # real totals: CCC 4162.25, AAA 4164.95, BBB 4167.25
    f = race_frame(("AAA", "T", A_PLAN, 1.0, (2, 2)), ("BBB", "T", B_PLAN, 1.0, (3, 3)),
                   ("CCC", "U", [("SOFT", 25), ("MEDIUM", 25)], 1.0, (1, 1)))
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[0], windows=["full"]).set_index("driver")
    assert (rows["pred_pos_real"] == 1).all()
    assert (rows["pred_pos_sim"] == rows["real_pos"]).all()
    assert rows.loc[["CCC", "AAA", "BBB"], "pred_pos_sim"].tolist() == [1, 2, 3]


def test_real_pos_comes_from_results_when_they_cover_every_finisher(frame):
    results = pd.DataFrame({"grid": [1.0, 2.0], "classified": [2.0, 1.0]}, index=pd.Index(["AAA", "BBB"]))
    rows = calibration.midrace(factory(frame), frame, RACE, laps_k=[0], windows=["full"], results=results)
    assert dict(zip(rows["driver"], rows["real_pos"])) == {"AAA": 2.0, "BBB": 1.0}
    partial = results.drop(index="BBB")
    rows = calibration.midrace(factory(frame), frame, RACE, laps_k=[0], windows=["full"], results=partial)
    assert dict(zip(rows["driver"], rows["real_pos"])) == {"AAA": 1.0, "BBB": 1.0}   # last-lap Position


def test_drivers_without_a_lap_k_row_are_skipped(frame):
    f = frame[~((frame["driver"] == "BBB") & (frame["lap"] == 25))]
    rows = calibration.midrace(factory(frame), f, RACE, laps_k=[20, 25], windows=["full"])
    assert sorted(zip(rows["k"], rows["driver"])) == [(20, "AAA"), (20, "BBB"), (25, "AAA")]


def test_unmodellable_plan_is_nan_and_counted(frame):
    """A window that knows only SOFT cannot replay a MEDIUM stint."""
    soft_only = synthetic_practice(coefs={"SOFT": (80.0, 0.10)})
    rows = calibration.midrace(factory(frame, practice=soft_only), frame, RACE, laps_k=[0], windows=["pre_race"])
    assert rows["pred_remaining_s"].isna().all() and rows["pred_pos_sim"].isna().all()
    s = calibration.aggregate(rows).loc[("pre_race", 50)]
    assert (s["n"], s["n_nan"]) == (0, 2)


# --- aggregate -----------------------------------------------------------------
def three_races(frame):
    rows = calibration.midrace(factory(frame), frame, RACE, windows=WINDOWS)
    rng = np.random.default_rng(1)
    out = []
    for i in range(3):
        r = rows.copy()
        r["race_id"] = f"race{i}"
        noise = rng.normal(0, 5, len(r))
        r["err_s"] += noise
        r["err_offset_s"] = r["err_offset_s"] + noise
        out.append(r)
    return rows, pd.concat(out, ignore_index=True)


def test_aggregate_shapes_with_one_and_three_races(frame):
    one, three = three_races(frame)
    buckets = sorted({h // 10 * 10 for h in one["horizon"]})
    for rows, n_races in ((one, 1), (three, 3)):
        s = calibration.aggregate(rows)
        assert s.index.names == ["window", "horizon_bucket"]
        assert list(s.columns) == calibration.SUMMARY_COLUMNS
        assert sorted(s.index) == sorted((w, b) for w in WINDOWS for b in buckets)
        assert (s["n_races"] == n_races).all()
        assert s["n"].sum() == len(rows)
        assert (s["n_nan"] == 0).all()
    s1 = calibration.aggregate(one)
    assert s1["mae_offset_ci_lo"].isna().all() and s1["mae_offset_ci_hi"].isna().all()
    s3 = calibration.aggregate(three)
    ok = s3["mae_offset_s"].notna()
    assert s3.loc[ok, "mae_offset_ci_lo"].notna().all()
    assert (s3.loc[ok, "mae_offset_ci_lo"] <= s3.loc[ok, "mae_offset_s"] + 1e-9).all()
    assert (s3.loc[ok, "mae_offset_s"] <= s3.loc[ok, "mae_offset_ci_hi"] + 1e-9).all()
    assert (s3["within_2_rate_sim"] == 1.0).all() and (s3["within_2_rate_real"] == 1.0).all()


def test_aggregate_metrics_by_hand():
    rows = pd.DataFrame({
        "race_id": "r", "window": "full", "horizon": [5, 8, 15, 9],
        "pred_remaining_s": [1.0, 1.0, 1.0, np.nan], "err_s": [1.0, -3.0, 2.0, np.nan],
        "err_offset_s": [np.nan, 1.0, -2.0, np.nan], "abs_pos_err_sim": [0.0, 1.0, 1.0, np.nan],
        "abs_pos_err_real": [0.0, 3.0, 1.0, np.nan]})
    s = calibration.aggregate(rows)
    a = s.loc[("full", 0)]
    assert (a["n"], a["n_nan"], a["mae_s"], a["bias_s"], a["mae_offset_s"]) == (2, 1, 2.0, -1.0, 1.0)
    assert (a["within_2_rate_sim"], a["mae_pos_sim"]) == (1.0, 0.5)
    assert (a["within_2_rate_real"], a["mae_pos_real"]) == (0.5, 1.5)
    assert s.loc[("full", 10), "n"] == 1


def test_appendix_tables(frame):
    _, three = three_races(frame)
    pr = calibration.per_race(three)
    assert pr.index.names == ["race_id", "window", "horizon_bucket"]
    assert set(pr.index.get_level_values("race_id")) == {"race0", "race1", "race2"}
    bd = calibration.by_driver(three)
    assert bd.index.names == ["race_id", "window", "driver", "horizon_bucket"]
    assert bd["n"].sum() == len(three)
    one = calibration.midrace(factory(frame), frame, RACE, windows=WINDOWS)
    pairs = calibration.midrace_pairs(one)
    assert pairs.index.names == ["window", "horizon_bucket"]
    decided = pairs[pairs["n_pairs"] > pairs["n_ties"]]
    assert (decided["sign_agree_rate"] == 1.0).all()   # exact predictions agree on every sign
    assert pairs["n_pairs"].sum() == len(one) // 2


@pytest.mark.parametrize("arg,used", [("observed", 20.0), ("25", 25.0)])
def test_midrace_pit_loss_is_observed_unless_given(frame, arg, used):
    """The synthetic in laps carry exactly RACE.pit_loss_s (20 s), so that is what is observed."""
    from types import SimpleNamespace

    from src.sim import calibrate
    args = SimpleNamespace(pit_loss=arg)
    race = RACE.model_copy(update={"pit_loss_s": 99.0})
    got, observed = calibrate._pit_loss(args, frame, cfg(degradation={"name": "linear"}), race)
    assert observed == pytest.approx(20.0, abs=0.01)
    assert got == pytest.approx(used, abs=0.01)


def test_sim_version_is_a_stable_config_hash():
    a, b = SimConfig(), SimConfig()
    assert calibration.sim_version(a) == calibration.sim_version(b)
    assert len(calibration.sim_version(a)) == 12
    assert calibration.sim_version(SimConfig(fit_window="pre_race")) == calibration.sim_version(a)
    assert calibration.sim_version(cfg()) != calibration.sim_version(a)   # lap1_extra_s differs
    assert calibration.sim_version(SimConfig(degradation={"name": "linear"})) != calibration.sim_version(a)


def race_csvs(frame, tmp_path, versions):
    """One stamped midrace csv per race, as calibrate midrace writes them."""
    from src.sim import calibrate
    _, three = three_races(frame)
    files = []
    for (race_id, r), version in zip(three.groupby("race_id"), versions):
        files.append(str(tmp_path / f"{race_id}_midrace.csv"))
        stamp = {"pit_loss_s": 23.81, "pit_loss_source": "observed", "sim_version": version}
        calibrate._stamp(r, stamp).to_csv(files[-1], index=False)
    return three, files


def test_aggregate_cli_reads_race_csvs_and_writes_the_summary(frame, tmp_path, capsys):
    from src.sim import calibrate
    three, files = race_csvs(frame, tmp_path, ["abc"] * 3)
    head = pd.read_csv(files[0], nrows=1)
    assert {"pit_loss_s", "pit_loss_source", "sim_version"} <= set(head.columns)
    assert open(files[0]).readline().startswith("race_id,")   # a plain csv, no comment lines
    out = tmp_path / "midrace_summary.csv"
    calibrate.main(["midrace-aggregate", *files, "--out", str(out)])
    got = pd.read_csv(out, index_col=["window", "horizon_bucket"])
    pd.testing.assert_frame_equal(got, calibration.aggregate(three), check_dtype=False)
    assert "mae_offset_s (horizon bucket x window)" in capsys.readouterr().out


def test_aggregate_cli_refuses_mixed_sim_versions(frame, tmp_path, capsys):
    from src.sim import calibrate
    _, files = race_csvs(frame, tmp_path, ["abc", "abc", "def"])
    out = tmp_path / "midrace_summary.csv"
    with pytest.raises(SystemExit) as exc:
        calibrate.main(["midrace-aggregate", *files, "--out", str(out)])
    assert exc.value.code == 2
    assert "2 sim_version values ['abc', 'def']" in capsys.readouterr().err
    assert not out.exists()


def test_aggregate_cli_allow_mixed(frame, tmp_path):
    from src.sim import calibrate
    three, files = race_csvs(frame, tmp_path, ["abc", "abc", "def"])
    out = tmp_path / "midrace_summary.csv"
    calibrate.main(["midrace-aggregate", *files, "--out", str(out), "--allow-mixed"])
    got = pd.read_csv(out, index_col=["window", "horizon_bucket"])
    pd.testing.assert_frame_equal(got, calibration.aggregate(three), check_dtype=False)
