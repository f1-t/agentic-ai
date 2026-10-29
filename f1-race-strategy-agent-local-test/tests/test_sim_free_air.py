"""SPEC 0010b: lap-frame gap columns and the opt-in free-air filter on the fit. Synthetic, no cache."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.contracts import RaceParams, Snapshot
from src.sim import SimConfig, Simulator, calibrate, calibration
from src.sim.config import FREE_AIR_GAP_S
from src.sim.data import GAP_COLUMNS, gap_columns, lap_frame, practice_frame

from test_sim import FUEL, RACE, TRUE, at_lap, synthetic_frame, synthetic_practice, window
from test_sim_filter import practice_laps

PACK_SLOW_S = 0.5


def cars(ends):
    """A frame of {driver: {lap: end_s}}, one row per (driver, lap) given."""
    return pd.DataFrame([{"driver": d, "lap": lap, "end_s": t} for d, laps in ends.items() for lap, t in laps.items()])


def at(f, driver, lap):
    return f.loc[(f["driver"] == driver) & (f["lap"] == lap)].iloc[0]


# --- gap columns ---------------------------------------------------------------
def test_gap_columns_on_three_cars():
    # C has no lap 2 row: at the start of lap 3 it is skipped, not taken as ahead of B on its
    # lap 1 crossing, and its own lap 3 row has no gap
    f = gap_columns(cars({"A": {1: 100.0, 2: 200.0, 3: 300.0},
                          "B": {1: 101.0, 2: 202.0, 3: 303.0},
                          "C": {1: 103.0, 3: 306.0}}))
    assert list(f.columns[-4:]) == GAP_COLUMNS
    lap1 = f[f["lap"] == 1]
    assert lap1["gap_ahead_s"].isna().all() and lap1["gap_behind_s"].isna().all()
    assert lap1["car_ahead"].isna().all() and lap1["free_air"].all()
    a, b = at(f, "A", 2), at(f, "B", 2)
    assert np.isnan(a["gap_ahead_s"]) and a["car_ahead"] is None and a["gap_behind_s"] == pytest.approx(1.0)
    assert b["gap_ahead_s"] == pytest.approx(1.0) and b["car_ahead"] == "A" and b["gap_behind_s"] == pytest.approx(2.0)
    a, b, c = at(f, "A", 3), at(f, "B", 3), at(f, "C", 3)
    assert np.isnan(a["gap_ahead_s"]) and a["gap_behind_s"] == pytest.approx(2.0)   # leader on lap 3
    assert b["car_ahead"] == "A" and b["gap_ahead_s"] == pytest.approx(2.0) and np.isnan(b["gap_behind_s"])
    assert np.isnan(c["gap_ahead_s"]) and c["car_ahead"] is None and np.isnan(c["gap_behind_s"])


def test_missing_end_s_is_skipped_and_has_no_gap():
    f = gap_columns(cars({"A": {1: 100.0, 2: 200.0}, "B": {1: np.nan, 2: 201.0}, "C": {1: 104.0, 2: 205.0}}))
    assert at(f, "C", 2)["car_ahead"] == "A" and at(f, "C", 2)["gap_ahead_s"] == pytest.approx(4.0)
    b = at(f, "B", 2)
    assert np.isnan(b["gap_ahead_s"]) and b["car_ahead"] is None and b["free_air"]


def test_free_air_threshold():
    assert FREE_AIR_GAP_S == 1.5
    f = gap_columns(cars({"A": {1: 100.0, 2: 0.0}, "B": {1: 101.5, 2: 0.0}, "C": {1: 102.99, 2: 0.0}}))
    gap = {d: at(f, d, 2)["gap_ahead_s"] for d in "ABC"}
    assert gap["B"] == 1.5 and gap["C"] == pytest.approx(1.49) and np.isnan(gap["A"])
    assert [bool(at(f, d, 2)["free_air"]) for d in "ABC"] == [True, True, False]   # NaN, 1.50, 1.49
    assert not gap_columns(cars({"A": {1: 100.0, 2: 0.0}, "B": {1: 101.5, 2: 0.0}}), 2.0)["free_air"].iloc[3]


def race_laps(drivers=("A", "B", "C"), n=12, pace=(0.0, 0.8, 2.5), pits=(6, 5, 7)):
    """FastF1-shaped race Laps with a running clock, one stop per car, SOFT then MEDIUM."""
    rows = []
    for d, off, pit in zip(drivers, pace, pits):
        clock = 0.2 * len(rows) / n
        for lap in range(1, n + 1):
            comp, age = ("SOFT", lap) if lap <= pit else ("MEDIUM", lap - pit)
            t = 80.0 + off + 0.1 * age + (5.0 if lap == 1 else 0.0)
            rows.append({
                "Driver": d, "LapNumber": float(lap), "Compound": comp,
                "Stint": 1.0 if lap <= pit else 2.0,
                "TyreLife": float(age), "LapTime": pd.Timedelta(t, unit="s") if lap > 1 else pd.NaT,
                "TrackStatus": "1", "PitInTime": pd.Timedelta(1, unit="s") if lap == pit else pd.NaT,
                "PitOutTime": pd.Timedelta(1, unit="s") if lap == pit + 1 else pd.NaT,
                "LapStartTime": pd.Timedelta(clock, unit="s"), "Time": pd.Timedelta(clock + t, unit="s"),
                "Team": "T", "Position": float(drivers.index(d) + 1)})
            clock += t
    return pd.DataFrame(rows)


def test_lap_frame_carries_the_gap_columns_on_both_paths():
    for f in (lap_frame(race_laps()), lap_frame(race_laps(), outlier_factor=1.07)):
        assert set(GAP_COLUMNS) <= set(f.columns)
        b = at(f, "B", 5)
        assert b["car_ahead"] == "A"
        assert b["gap_ahead_s"] == pytest.approx(at(f, "B", 4)["end_s"] - at(f, "A", 4)["end_s"])


def test_practice_rows_have_no_gap_and_are_free_air():
    f = practice_frame({"FP2": practice_laps([("SOFT", 10)])})
    assert f["gap_ahead_s"].isna().all() and f["gap_behind_s"].isna().all()
    assert f["car_ahead"].isna().all() and f["free_air"].all()


# --- default off ---------------------------------------------------------------
# MAIN_DIGEST is batch_digest(...)[1] run on origin/main 940f61d, on the frames without the gap
# columns (which main does not have)
MAIN_DIGEST = "f0b05641cbb14fb859a5a40d2890022c62e4c5770a00180084fab7ba4b3cf5cb"


def with_columns(frame, seed=0):
    """`frame` plus gap columns, about a third of the rows pack-bound (free_air False)."""
    rng = np.random.default_rng(seed)
    gap = rng.uniform(0.2, 4.0, len(frame))
    return frame.assign(gap_ahead_s=gap, car_ahead="Y", gap_behind_s=gap, free_air=gap >= FREE_AIR_GAP_S)


def batch_digest(make):
    """sha256 over fits, model reports and batches under each window; `make(frame)` adapts the
    race frame (identity, or with_columns)."""
    h = hashlib.sha256()
    race = make(synthetic_frame(fuel=FUEL))
    plans = [[["SOFT", x], ["MEDIUM", 50 - x]] for x in range(8, 43, 3)]
    for fw in ("full", "to_snapshot", "pre_race"):
        for model in ("linear_fuel", "linear_fuel_hinge"):
            cfg = window(fw, degradation={"name": model}, hazard={"name": "age_logistic"})
            sim = Simulator(race, cfg, synthetic_practice())
            for k in (0, 15, 30):
                snap = at_lap(k) if k else Snapshot.at_start(RACE, "X", "SOFT")
                fit = sim.fitted_for(snap)
                h.update(repr(sorted(fit.degradation.fit_report.items())).encode())
                h.update(np.array([fit.ref_lap_s]).tobytes())
                h.update(sim.model_report(snap, calibration_summary=None).model_dump_json().encode())
                b = sim.simulate_batch(snap, [[["SOFT", 50 - k]]] if k >= 30 else plans, seed=3, n_samples=20)
                for a in (b.times, b.pit_loss, b.dnf, b.big_loss):
                    h.update(np.ascontiguousarray(a).tobytes())
    return len(plans), h.hexdigest()


def test_default_off_fit_is_identical_with_and_without_the_columns():
    n, plain = batch_digest(lambda f: f)
    assert n == 12
    assert batch_digest(with_columns)[1] == plain


def test_default_off_digest_equals_main():
    assert batch_digest(with_columns)[1] == MAIN_DIGEST


def test_the_filter_changes_the_config_hash():
    # the hash keeps one meaning (every field), so the new field moved the default stamp too
    assert calibration.config_hash(SimConfig()) != "94a5373a6a20"   # origin/main 940f61d
    assert calibration.config_hash(SimConfig(fit_free_air_only=True)) != calibration.config_hash(SimConfig())


# --- filter on -----------------------------------------------------------------
def pack_frame(seed=1):
    """synthetic_frame with a random third of its rows pack-bound and PACK_SLOW_S slower."""
    f = with_columns(synthetic_frame(fuel=FUEL), seed)
    return f.assign(lap_time_s=f["lap_time_s"] + np.where(f["free_air"], 0.0, PACK_SLOW_S))


def coefs_off(sim):
    return max(abs(sim.degradation.coefs[c][0] - TRUE[c][0]) for c in TRUE)


def test_filtered_fit_recovers_the_true_base_and_the_unfiltered_does_not():
    f = pack_frame()
    on = Simulator(f, SimConfig(fit_free_air_only=True))
    off = Simulator(f, SimConfig())
    for c, (base, wear) in TRUE.items():
        assert on.degradation.coefs[c] == pytest.approx((base, wear), abs=1e-6)
    assert on.degradation.fuel_slope == pytest.approx(FUEL, abs=1e-6)
    assert coefs_off(off) > 0.05
    n_pack = int((~f["free_air"]).sum())
    assert 0 < n_pack < len(f)
    assert on.degradation.fit_report["n_pack_rows"] == n_pack
    assert "n_pack_rows" not in off.degradation.fit_report
    assert on.ref_lap_s == pytest.approx(f.loc[f["free_air"], "lap_time_s"].median())
    flags = on.model_report(Snapshot.at_start(RACE, "X", "SOFT"), calibration_summary=None).flags
    assert f"free_air_filter:{n_pack}" in flags
    assert not any(x.startswith("free_air_filter") for x in
                   off.model_report(Snapshot.at_start(RACE, "X", "SOFT"), calibration_summary=None).flags)


def test_n_pack_rows_counts_clean_rows_only():
    f = pack_frame()
    f.loc[f.index[:10], "clean"] = False
    on = Simulator(f, SimConfig(fit_free_air_only=True))
    assert on.degradation.fit_report["n_pack_rows"] == int((f["clean"] & ~f["free_air"]).sum())


def test_hazard_is_fitted_on_the_free_air_rows():
    f = pack_frame()
    oldest = f["compound"].eq("SOFT") & f["tyre_age"].eq(20)
    f.loc[oldest, "free_air"] = False
    cfg = {"hazard": {"name": "age_logistic"}}
    on = Simulator(f, SimConfig(fit_free_air_only=True, **cfg))
    off = Simulator(f, SimConfig(**cfg))
    assert off.hazard.a0["SOFT"] == 20 + 3.0
    assert on.hazard.a0["SOFT"] == 19 + 3.0


@pytest.mark.parametrize("fw", ["to_snapshot", "pre_race"])
def test_practice_rows_pass_the_filter(fw):
    """The race frame has no pack rows here, so the filter must change nothing: practice
    (lap NaN, with or without a free_air column) is never dropped."""
    race = with_columns(synthetic_frame(fuel=FUEL)).assign(free_air=True)
    for practice in (synthetic_practice(), synthetic_practice().assign(free_air=True)):
        on = Simulator(race, window(fw, fit_free_air_only=True), practice)
        off = Simulator(race, window(fw), practice)
        for snap in (Snapshot.at_start(RACE, "X", "SOFT"), at_lap(20)):
            a, b = on.fitted_for(snap), off.fitted_for(snap)
            assert a.degradation.coefs == b.degradation.coefs and a.ref_lap_s == b.ref_lap_s
            assert a.degradation.fit_report["n_pack_rows"] == 0


def test_to_snapshot_drops_pack_rows_from_the_race_window():
    race = pack_frame()
    on = Simulator(race, window("to_snapshot", fit_free_air_only=True), synthetic_practice())
    fit = on.fitted_for(at_lap(20))
    early = race[race["lap"] <= 20]
    assert fit.degradation.fit_report["n_pack_rows"] == int((~early["free_air"]).sum())
    assert fit.degradation.fit_report["n_race_rows"] == int(early["free_air"].sum())


def test_filter_on_without_the_column_is_an_error():
    with pytest.raises(ValueError, match="free_air"):
        Simulator(synthetic_frame(), SimConfig(fit_free_air_only=True))


def test_calibrate_free_air_flag_maps_to_the_config():
    args = SimpleNamespace(config=None, degradation="linear_fuel", free_air=True)
    assert calibrate._config(args).fit_free_air_only
    assert not calibrate._config(SimpleNamespace(config=None, degradation="linear_fuel")).fit_free_air_only


# --- snapshot and by_kind ------------------------------------------------------
def race_params(frame):
    return RaceParams(race_id="syn_race", total_laps=int(frame["lap"].max()), pit_loss_s=20.0,
                      min_stint_laps=3, compounds=["SOFT", "MEDIUM"])


def test_snapshot_at_fills_the_gaps_at_k_above_0():
    f = lap_frame(race_laps())
    race = race_params(f)
    s0 = calibration.snapshot_at(f, race, "B", 0)
    assert s0.gap_ahead_s is None and s0.gap_behind_s is None and s0.car_ahead is None
    s = calibration.snapshot_at(f, race, "B", 4)
    assert s.car_ahead == "A"
    assert s.gap_ahead_s == pytest.approx(at(f, "B", 4)["end_s"] - at(f, "A", 4)["end_s"])
    assert s.gap_behind_s == pytest.approx(at(f, "C", 4)["end_s"] - at(f, "B", 4)["end_s"])
    lead = calibration.snapshot_at(f, race, "A", 4)
    assert lead.gap_ahead_s is None and lead.car_ahead is None and lead.gap_behind_s is not None


def test_by_kind_splits_clean_into_free_and_pack():
    f = lap_frame(race_laps(pace=(0.0, 0.3, 2.5)))
    race = race_params(f)
    sim = Simulator(f, SimConfig(degradation={"name": "linear"}))
    by_kind = calibration.run(sim, f, race)[2]
    old = calibration.run(sim, f.drop(columns=GAP_COLUMNS), race)[2]
    assert by_kind.loc["clean_pack", "count"] > 0 and by_kind.loc["clean_free", "count"] > 0
    assert (by_kind.loc["clean_free", "count"] + by_kind.loc["clean_pack", "count"]
            == old.loc["clean_free", "count"] == int(f.loc[f["clean"], "lap"].size))
    assert "clean_pack" not in old.index
    other = [k for k in by_kind.index if not k.startswith("clean")]
    pd.testing.assert_frame_equal(by_kind.loc[other], old.loc[other])


def test_pack_penalty_is_in_the_summary():
    f = lap_frame(race_laps(pace=(0.0, 0.3, 2.5)))
    race = race_params(f)
    sim = Simulator(f, SimConfig(degradation={"name": "linear"}))
    table, summary, by_kind = calibration.run(sim, f, race)
    pack = by_kind.loc["clean_pack"]
    free = by_kind.loc["clean_free"]
    assert summary["observed_pack_penalty_s"] == pytest.approx(pack["mean"] - free["mean"], abs=2e-3)
    assert summary["observed_pack_penalty_median_s"] == pytest.approx(pack["median"] - free["median"], abs=2e-3)
    old = calibration.run(sim, f.drop(columns=GAP_COLUMNS), race)[1]
    assert np.isnan(old["observed_pack_penalty_s"]) and np.isnan(old["observed_pack_penalty_median_s"])


# --- pace offset under the filter ------------------------------------------------
def offset_frame(pack_laps):
    """Three cars; B's laps in `pack_laps` are pack-bound and 1 s slow, its free-air laps 0.2 s slow."""
    f = lap_frame(race_laps(pace=(0.0, 3.0, 6.0)))
    b = f["driver"] == "B"
    f.loc[b & f["lap"].gt(1), "lap_time_s"] += 0.2
    pack = b & f["lap"].isin(pack_laps)
    f.loc[pack, "lap_time_s"] += 0.8
    f["free_air"] = ~pack
    return f


def test_pace_offset_uses_free_air_laps_when_the_filter_is_on():
    f = offset_frame(pack_laps=[2, 3, 4])   # B pits at the end of lap 5: free-air green laps 7-12
    on = Simulator(f, SimConfig(degradation={"name": "linear"}, fit_free_air_only=True))
    g = f[f["driver"] == "B"].sort_values("lap")
    flags = []
    got = calibration._pace_offset(on, g, 12, flags)
    assert flags == []
    c = g[g["lap"].between(2, 12) & ~g["pit_in"] & ~g["pit_out"]]
    resid = c["lap_time_s"].to_numpy() - np.array([
        on.degradation.lap_times(r.compound, np.array([float(r.tyre_age)]), np.array([float(r.lap)]))[0]
        for r in c.itertuples()])
    assert got == pytest.approx(float(np.median(resid[c["free_air"].to_numpy()])))
    assert got < float(np.median(resid))   # the all-laps median absorbs B's own traffic


def test_pace_offset_falls_back_to_every_green_lap_with_a_flag():
    f = offset_frame(pack_laps=list(range(2, 9)))
    on = Simulator(f, SimConfig(degradation={"name": "linear"}, fit_free_air_only=True))
    g = f[f["driver"] == "B"].sort_values("lap")
    flags = []
    got = calibration._pace_offset(on, g, 9, flags)
    assert flags == [calibration.PACE_OFFSET_ALL_LAPS]
    plain = Simulator(f, SimConfig(degradation={"name": "linear"}))
    plain.degradation = on.degradation   # the same model, the filter off: every green lap
    assert got == pytest.approx(calibration._pace_offset(plain, g, 9))
    rows = calibration.midrace(lambda w: on, f, race_params(f), laps_k=[9], windows=["full"])
    assert rows.set_index("driver").loc["B", "pace_offset_flag"] == calibration.PACE_OFFSET_ALL_LAPS
    assert "pace_offset_flag" not in calibration.midrace(lambda w: plain, f, race_params(f), laps_k=[9],
                                                         windows=["full"]).columns


# --- pace offset under traffic (0013 ruling) --------------------------------------
GHOST = {"name": "ghost"}


def test_free_air_offset_rule():
    assert not calibration.free_air_offset(SimConfig())
    assert calibration.free_air_offset(SimConfig(fit_free_air_only=True))
    assert calibration.free_air_offset(SimConfig(traffic=GHOST))
    assert [calibration.pace_offset_definition(c) for c in (SimConfig(), SimConfig(traffic=GHOST))] == [
        "all_laps", "free_air"]


def test_pace_offset_uses_free_air_laps_when_traffic_is_on():
    # the free-air laps whenever traffic is on, the filter off; the fallback holds as well
    f = offset_frame(pack_laps=[2, 3, 4])
    g = f[f["driver"] == "B"].sort_values("lap")
    plain = Simulator(f, SimConfig(degradation={"name": "linear"}))
    traffic = Simulator(f, SimConfig(degradation={"name": "linear"}, traffic=GHOST))
    filt = Simulator(f, SimConfig(degradation={"name": "linear"}, fit_free_air_only=True))
    traffic.degradation = plain.degradation = filt.degradation   # one model, so only the lap choice differs
    assert calibration._pace_offset(traffic, g, 12) == pytest.approx(calibration._pace_offset(filt, g, 12))
    assert calibration._pace_offset(traffic, g, 12) < calibration._pace_offset(plain, g, 12)
    flags = []
    f = offset_frame(pack_laps=list(range(2, 9)))
    traffic = Simulator(f, SimConfig(degradation={"name": "linear"}, traffic=GHOST))
    calibration._pace_offset(traffic, f[f["driver"] == "B"].sort_values("lap"), 9, flags)
    assert flags == [calibration.PACE_OFFSET_ALL_LAPS]


def test_run_offset_uses_clean_free_air_laps_when_traffic_is_on():
    f = offset_frame(pack_laps=[2, 3, 4, 8])
    race = race_params(f)
    sim = Simulator(f, SimConfig(degradation={"name": "linear"}, traffic=GHOST))
    table = calibration.run(sim, f, race)[0]
    g, res = calibration.replay_driver(sim, f, race, "B")
    resid = g["lap_time_s"].to_numpy() - np.array(res.lap_times_s)
    use = g["clean"].to_numpy() & g["free_air"].to_numpy()
    assert table.loc["B", "pace_offset_s"] == pytest.approx(float(np.median(resid[use])))
    assert table.loc["B", "pace_offset_s"] < float(np.median(resid[g["clean"].to_numpy()]))
    assert table.loc["B", "pace_offset_flag"] == ""
    default = calibration.run(Simulator(f, SimConfig(degradation={"name": "linear"})), f, race)[0]
    assert "pace_offset_flag" not in default.columns
    # too few clean free-air laps: every clean lap, flagged
    f = offset_frame(pack_laps=list(range(2, 12)))
    table = calibration.run(Simulator(f, SimConfig(degradation={"name": "linear"}, traffic=GHOST)), f, race)[0]
    assert table.loc["B", "pace_offset_flag"] == calibration.PACE_OFFSET_ALL_LAPS
