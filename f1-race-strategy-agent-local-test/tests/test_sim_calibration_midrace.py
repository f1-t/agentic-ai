"""Mid-race replay calibration on synthetic races with a known model. Offline, no model."""
import numpy as np
import pandas as pd
import pytest

from src.sim import SimConfig, Simulator, calibration
from test_sim import RACE, synthetic_practice
from test_sim_position import driver_rows, race_frame

A_PLAN = [("SOFT", 20), ("MEDIUM", 30)]
B_PLAN = [("SOFT", 10), ("MEDIUM", 25), ("SOFT", 15)]
WINDOWS = ("full", "to_snapshot", "pre_race")


@pytest.fixture(scope="module")
def frame():
    return race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN))


def factory(frame, name="linear"):
    """Fitted on the race itself; practice follows the same model, so every window is exact."""
    def make(window):
        return Simulator(frame, SimConfig(degradation={"name": name}, fit_window=window), synthetic_practice())
    return make


# --- snapshot and plan ---------------------------------------------------------
def hand_frame():
    """BBB with tyre ages that do not start at zero, an SC period on laps 12-13 and a stop on lap 10."""
    rows = driver_rows("BBB", "T", B_PLAN)
    for r in rows:
        if r["stint"] == 1:
            r["tyre_age"] += 3       # a used set, three laps old at the start
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


def test_default_laps_k_adds_laps_around_sc_periods():
    f = hand_frame()
    assert calibration.default_laps_k(f, 50) == [0, 5, 10, 11, 14, 15, 20, 25, 30, 35, 40]


# --- rows ----------------------------------------------------------------------
def test_k0_full_rows_equal_run(frame):
    sim = Simulator(frame)  # default model, full window
    table, _, _ = calibration.run(sim, frame, RACE)
    pos, _ = calibration.positions(sim, frame, RACE)
    rows = calibration.midrace(lambda w: sim, frame, RACE, laps_k=[0], windows=["full"]).set_index("driver")
    assert sorted(rows.index) == sorted(table.index)
    for d in table.index:
        assert rows.loc[d, "real_remaining_s"] == pytest.approx(table.loc[d, "real_s"], abs=1e-9)
        assert rows.loc[d, "pred_remaining_s"] == pytest.approx(table.loc[d, "pred_s"], abs=1e-9)
        assert rows.loc[d, "err_s"] == pytest.approx(table.loc[d, "err_s"], abs=1e-9)
        assert rows.loc[d, "pred_pos"] == pos.loc[d, "pred_pos"]
        assert rows.loc[d, "horizon"] == table.loc[d, "laps"]
    assert rows["err_offset_s"].isna().all()


def test_k0_full_rows_equal_run_on_the_spain_shape():
    """Lapped drivers and a used first set, against a sim fitted elsewhere."""
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

    def make(window):
        return Simulator(frame, SimConfig(degradation={"name": "linear"}, fit_window=window), practice)
    rows = calibration.midrace(make, frame, RACE, laps_k=[0, 10, 49]).set_index(["window", "driver", "k"])
    for d in ("AAA", "BBB"):
        err = {(w, k): abs(rows.loc[(w, d, k), "err_s"]) for w in WINDOWS for k in (0, 10, 49)}
        assert err[("full", 0)] < 1e-6 and err[("full", 49)] < 1e-6
        assert err[("to_snapshot", 0)] == pytest.approx(err[("pre_race", 0)])
        assert err[("to_snapshot", 10)] < err[("pre_race", 10)]
        assert err[("to_snapshot", 49)] < 0.25 * err[("pre_race", 49)]


def test_pace_offset_removes_a_driver_offset():
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)), SimConfig(degradation={"name": "linear"}))
    f = race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN, -0.4))
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[0, 20], windows=["full"]).set_index(["k", "driver"])
    assert rows.loc[(20, "BBB"), "err_s"] == pytest.approx(0.4 * 30, abs=1e-6)
    assert rows.loc[(20, "BBB"), "pace_offset_s"] == pytest.approx(-0.4, abs=1e-6)
    assert rows.loc[(20, "BBB"), "err_offset_s"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(rows.loc[(0, "BBB"), "err_offset_s"])


def test_too_few_clean_laps_gives_no_offset(frame):
    rows = calibration.midrace(factory(frame), frame, RACE, laps_k=[3], windows=["full"])
    assert rows["pace_offset_s"].isna().all()


def test_positions_rank_against_other_finishers_real_totals():
    sim = Simulator(race_frame(("AAA", "T", A_PLAN), ("BBB", "T", B_PLAN)))
    f = race_frame(("AAA", "T", A_PLAN, 0.0, (1, 2)), ("BBB", "T", B_PLAN, -0.5, (2, 1)))
    rows = calibration.midrace(lambda w: sim, f, RACE, laps_k=[0, 45], windows=["full"]).set_index(["k", "driver"])
    # BBB is half a second a lap quicker than predicted, so at the start it is placed second
    assert rows.loc[(0, "BBB"), "pred_pos"] == 2
    assert rows.loc[(0, "BBB"), "real_pos"] == 1
    assert rows.loc[(0, "BBB"), "abs_pos_err"] == 1
    assert rows.loc[(45, "BBB"), "pred_pos"] == 1   # five laps out the error is too small to matter


def test_drivers_without_a_lap_k_row_are_skipped(frame):
    f = frame[~((frame["driver"] == "BBB") & (frame["lap"] == 25))]
    rows = calibration.midrace(factory(frame), f, RACE, laps_k=[20, 25], windows=["full"])
    assert sorted(zip(rows["k"], rows["driver"])) == [(20, "AAA"), (20, "BBB"), (25, "AAA")]


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
    s1 = calibration.aggregate(one)
    assert s1["mae_offset_ci_lo"].isna().all() and s1["mae_offset_ci_hi"].isna().all()
    s3 = calibration.aggregate(three)
    ok = s3["mae_offset_s"].notna()
    assert s3.loc[ok, "mae_offset_ci_lo"].notna().all()
    assert (s3.loc[ok, "mae_offset_ci_lo"] <= s3.loc[ok, "mae_offset_s"] + 1e-9).all()
    assert (s3.loc[ok, "mae_offset_s"] <= s3.loc[ok, "mae_offset_ci_hi"] + 1e-9).all()
    assert (s3["within_2_rate"] == 1.0).all()   # the noise is on time only


def test_aggregate_metrics_by_hand():
    rows = pd.DataFrame({
        "race_id": "r", "window": "full", "horizon": [5, 8, 15], "err_s": [1.0, -3.0, 2.0],
        "err_offset_s": [np.nan, 1.0, -2.0], "abs_pos_err": [0.0, 3.0, 1.0]})
    s = calibration.aggregate(rows)
    a = s.loc[("full", 0)]
    assert (a["n"], a["mae_s"], a["bias_s"], a["mae_offset_s"]) == (2, 2.0, -1.0, 1.0)
    assert (a["within_2_rate"], a["mae_pos"]) == (0.5, 1.5)
    assert s.loc[("full", 10), "n"] == 1


def test_per_race_and_pairs_tables(frame):
    _, three = three_races(frame)
    pr = calibration.per_race(three)
    assert pr.index.names == ["race_id", "window", "horizon_bucket"]
    assert set(pr.index.get_level_values("race_id")) == {"race0", "race1", "race2"}
    one = calibration.midrace(factory(frame), frame, RACE, windows=WINDOWS)
    pairs = calibration.midrace_pairs(one)
    assert pairs.index.names == ["window", "horizon_bucket"]
    decided = pairs[pairs["n_pairs"] > pairs["n_ties"]]
    assert (decided["sign_agree_rate"] == 1.0).all()   # exact predictions agree on every sign
    assert pairs["n_pairs"].sum() == len(one) // 2


def test_aggregate_cli_reads_race_csvs_and_writes_the_summary(frame, tmp_path, capsys):
    from src.sim import calibrate
    _, three = three_races(frame)
    files = []
    for race_id, r in three.groupby("race_id"):
        files.append(str(tmp_path / f"{race_id}_midrace.csv"))
        r.to_csv(files[-1], index=False)
    out = tmp_path / "midrace_summary.csv"
    calibrate.main(["midrace-aggregate", *files, "--out", str(out)])
    got = pd.read_csv(out, index_col=["window", "horizon_bucket"])
    pd.testing.assert_frame_equal(got, calibration.aggregate(three), check_dtype=False)
    assert "mae_offset_s (horizon bucket x window)" in capsys.readouterr().out
