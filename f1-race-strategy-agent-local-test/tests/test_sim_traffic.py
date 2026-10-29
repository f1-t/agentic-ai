"""Ghost-car traffic (traffic.py) and the engine pace term. Synthetic frames, no cache, no model.

Our car XXX laps at a constant 80 s; ghosts lap at constant recorded paces with end_s
cumulative on the session clock. Lap times are multiples of 0.1 s or 0.5 s so the clock
arithmetic is exact where a test sits on a boundary.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.contracts import RaceParams, Snapshot
from src.sim import SimConfig, Simulator, calibration
from src.sim.config import EventEffects
from src.sim.events import Scenario
from src.sim.traffic import TRAFFIC, GhostTraffic, ghost_table
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_frame, synthetic_practice
from test_sim_position import race_frame

OUR, F, N = "XXX", 80.0, 20
RACE = RaceParams(race_id="syn_traffic", total_laps=N, pit_loss_s=20.0, min_stint_laps=1,
                  compounds=["SOFT", "MEDIUM"])
RULES = ["lap_sum", "known_compounds"]
NO_DRS = {"pass_delta_drs_s": 0.8}   # DRS gives no extra help: pass_delta 0.8 either way


def car(driver, lap_s, start=0.0, compound="SOFT", pit_in=(), times=None, n=N, position=1.0):
    """Constant recorded pace from `start` on the session clock; `times` overrides laps."""
    rows, t = [], start
    for lap in range(1, n + 1):
        lt = (times or {}).get(lap, lap_s)
        rows.append({"driver": driver, "team": driver, "lap": lap, "stint": 1 + sum(lap > p for p in pit_in),
                     "compound": compound, "tyre_age": lap - max([p for p in pit_in if p < lap], default=0),
                     "lap_time_s": lt, "track_status": "1", "pit_in": lap in pit_in, "pit_out": lap - 1 in pit_in,
                     "start_s": t, "end_s": t + lt, "clean": not (lap in pit_in or lap - 1 in pit_in),
                     "position": position})
        t += lt
    return rows


def ghost(name, lap_s, ahead, k=5, **kw):
    """A ghost `ahead` seconds in front of our car (at F per lap from 0) at the end of lap k."""
    return car(name, lap_s, start=k * F - ahead - k * lap_s, compound="MEDIUM", **kw)


def frame(*ghosts):
    return pd.DataFrame(car(OUR, F) + [r for g in ghosts for r in g])


def snap(k=5, elapsed=None, **kw):
    kw.setdefault("standing_start", False)
    return Snapshot(race=RACE, driver=OUR, laps_completed=k, compound="SOFT", tyre_age=k,
                    elapsed_s=k * F if elapsed is None else elapsed, **kw)


def prepare(df, k=5, **params):
    model = GhostTraffic(**params)
    s = snap(k)
    laps = np.arange(k + 1, N + 1)
    return model, model.prepare(s, laps, df, OUR), laps


def apply(model, state, laps, lap_s=F, pit=None, sc=(), vsc=(), n_plans=1):
    """Returns (increment per lap of plan 0, lap numbers, traffic_s, passes)."""
    lap_s = np.broadcast_to(np.asarray(lap_s, dtype=float), (n_plans, len(laps))).copy()
    pit = np.zeros_like(lap_s) if pit is None else pit
    neutral = np.isin(laps, [*sc, *vsc])
    out, (traffic_s, passes) = model.apply(state, lap_s, pit, neutral, Scenario(sc_laps=list(sc), vsc_laps=list(vsc)))
    return out - lap_s, traffic_s, passes


def by_lap(inc, laps):
    return dict(zip(laps.tolist(), np.round(inc[0], 9).tolist()))


# --- engine: byte identity and pace -----------------------------------------------
# 1. default off, pace None: hashes equal to main. MAIN_DIGEST was computed by running
# batch_digest on origin/main 7a53507 (before traffic and pace existed).
MAIN_DIGEST = "5f5159d7d8c8e48ce68e8365fb561329b8750589eaedda4ecf79e9a45bf3cb14"


def batch_digest(sim):
    """sha256 over a 50-plan x 20-sample batch on test_sim's synthetic race with rate events."""
    plans = [[["SOFT", x], ["MEDIUM", 50 - x]] for x in range(8, 43)]
    plans += [[["SOFT", x], ["MEDIUM", 20], ["SOFT", 30 - x]] for x in range(8, 23)]
    s = Snapshot(race=SYN_RACE, driver="X", laps_completed=0, compound="SOFT")
    b = sim.simulate_batch(s, plans, seed=3, n_samples=20, return_laps=True)
    h = hashlib.sha256()
    for a in (b.times, b.pit_loss, b.lap_times, *b.stint_seconds):
        h.update(np.ascontiguousarray(a).tobytes())
    return len(plans), h.hexdigest()


def syn_sim(**kw):
    return Simulator(synthetic_frame(), SimConfig(events={"name": "rate", "params": {"p_sc": 0.05}}), **kw)


def test_default_off_and_no_pace_hash_equal_to_main():
    n, digest = batch_digest(syn_sim())
    assert n == 50
    assert digest == MAIN_DIGEST
    assert batch_digest(syn_sim(pace=None))[1] == digest
    assert batch_digest(syn_sim(pace={"OTHER": 0.5}))[1] == digest   # not the snapshot driver


def test_default_config_hash_is_unchanged_and_traffic_changes_it():
    assert calibration.config_hash(SimConfig()) == "ffd423d35d0c"   # origin/main 7a53507
    assert calibration.config_hash(SimConfig(traffic={"name": "ghost"})) != "ffd423d35d0c"


@pytest.fixture(scope="module")
def solo():
    return frame()


@pytest.mark.parametrize("pace", [{OUR: 0.5}, pd.Series({OUR: 0.5, "YYY": -1.0})])
def test_pace_adds_offset_to_green_laps_only(solo, pace):
    # 2. +0.5 on every green lap, lap 1 included, nothing on SC or VSC laps
    s = Snapshot(race=RACE, driver=OUR, compound="SOFT", elapsed_s=0.0)
    plan, scen = [["SOFT", N]], Scenario(sc_laps=[5, 6], vsc_laps=[12])
    base = Simulator(solo).simulate(s, plan, scenario=scen, rules=RULES, return_laps=True)
    paced = Simulator(solo, pace=pace).simulate(s, plan, scenario=scen, rules=RULES, return_laps=True)
    diff = np.array(paced.lap_times_s) - np.array(base.lap_times_s)
    green = ~np.isin(np.arange(1, N + 1), [5, 6, 12])
    assert diff[green] == pytest.approx(0.5)
    assert np.all(diff[~green] == 0.0)
    assert diff[0] == pytest.approx(0.5)
    assert paced.remaining.mean_s - base.remaining.mean_s == pytest.approx(0.5 * green.sum())
    other = Simulator(solo, pace={"YYY": 0.5}).simulate(s, plan, scenario=scen, rules=RULES, return_laps=True)
    assert other.lap_times_s == base.lap_times_s


def test_with_pace_and_run_pace_put_the_term_in_the_engine(solo):
    sim = Simulator(solo)
    paced = sim.with_pace({OUR: 0.5})
    assert paced.pace == {OUR: 0.5} and sim.pace is None and paced.degradation is sim.degradation
    t0 = calibration.run(sim, solo, RACE)[0]
    t1 = calibration.run(sim, solo, RACE, pace={OUR: 0.5})[0]
    assert t1.loc[OUR, "pred_s"] - t0.loc[OUR, "pred_s"] == pytest.approx(0.5 * N)
    assert t1.loc[OUR, "pace_offset_s"] == pytest.approx(t0.loc[OUR, "pace_offset_s"] - 0.5)


# --- traffic rule ------------------------------------------------------------------
def test_registry_and_none_model():
    assert {"none", "ghost"} <= set(TRAFFIC.keys())
    none = TRAFFIC.get("none")()
    assert none.prepare(snap(), np.arange(6, N + 1), frame(), OUR) is None
    with pytest.raises(NotImplementedError):
        GhostTraffic(source="projected")


def test_ghost_table_arrays():
    df = frame(ghost("AAA", 80.5, 3.0, pit_in=(8,)))
    t = ghost_table(df, OUR, np.arange(6, N + 1))
    assert t.ghosts == ["AAA"] and t.G.shape == (1, N - 5)
    assert t.G_prev[0, 0] == pytest.approx(5 * F - 3.0)            # the lap-k row
    assert np.allclose(t.g, 80.5) and t.in_lap[0].tolist() == [lap == 8 for lap in range(6, N + 1)]
    assert np.isnan(ghost_table(df, OUR, np.arange(1, N + 1)).G_prev[0, 0])   # no lap-0 row


def test_held_behind_a_slower_car():
    # 3. 0.5 s/lap faster than a ghost 3 s ahead: close at 0.5 s/lap, then lap at its pace
    model, state, laps = prepare(frame(ghost("AAA", 80.5, 3.0)), **NO_DRS)
    inc, traffic_s, passes = apply(model, state, laps)
    got = by_lap(inc, laps)
    assert all(got[lap] == 0.0 for lap in range(6, 10))             # gap 3.0, 2.5, 2.0, 1.5
    assert all(got[lap] == pytest.approx(0.5) for lap in range(10, N + 1))   # lap_s = 80.5
    assert traffic_s[0] == pytest.approx(inc.sum()) == pytest.approx(0.5 * 11)
    assert passes[0] == 0


def test_a_clean_pass_pays_pass_cost_once():
    # 4. 1.0 s/lap faster, caught within pass reach: one lap pays pass_cost_s, then free air.
    # The spec's 3 s start does not give this under the rule, see the next test.
    df = frame(ghost("AAA", 81.0, 0.5))
    model, state, laps = prepare(df)
    inc, traffic_s, passes = apply(model, state, laps)
    assert by_lap(inc, laps)[6] == pytest.approx(0.4)
    assert np.all(inc[0, 1:] == 0.0)
    assert passes[0] == 1 and traffic_s[0] == pytest.approx(0.4)
    end6 = 6 * F + 0.4
    assert end6 - state.table.G[0, 0] < 0                             # next lap d < 0


def test_a_pass_that_ends_behind_the_ghost_is_charged_again():
    # the rule as written: from 3 s at 1.0 s/lap the catch lap starts 1.0 s behind, the pass
    # costs 0.4 and ends 0.4 s behind the ghost's line, so the next lap passes it again
    model, state, laps = prepare(frame(ghost("AAA", 81.0, 3.0)))
    inc, traffic_s, passes = apply(model, state, laps)
    got = by_lap(inc, laps)
    assert got[8] == pytest.approx(0.4) and got[9] == pytest.approx(0.4)
    assert passes[0] == 2 and traffic_s[0] == pytest.approx(0.8)


def first_pass_lap(inc, laps, cost=0.4):
    hit = np.isclose(inc[0], cost)
    return int(laps[hit.argmax()]) if hit.any() else None


def test_drs_off_on_the_first_laps():
    # 5. 0.5 s/lap faster, 0.8 s behind at the start of lap 2 (lap 1 has no lap-0 row, so no
    # traffic): held on lap 2 (DRS off), pass charged on lap 3 (DRS on, d = follow gap)
    df = frame(car("AAA", 80.5, start=F - 0.8 - 80.5, compound="MEDIUM"))
    model = GhostTraffic()
    laps = np.arange(1, N + 1)
    state = model.prepare(snap(0, 0.0), laps, df, OUR)
    inc, _, passes = apply(model, state, laps)
    got = by_lap(inc, laps)
    assert got[1] == 0.0 and got[2] == pytest.approx(0.7)            # 0.3 s behind its line, held to G + 1
    assert first_pass_lap(inc, laps) == 3 and passes[0] >= 1


def test_drs_off_for_two_laps_after_an_sc_period():
    # 5. same car held at the follow gap from lap 5; SC on laps 5-6, ghost's laps there as ours
    times = {5: 110.0, 6: 110.0}
    df = frame(ghost("AAA", 80.5, 1.0, k=4, times=times))
    model = GhostTraffic()
    laps = np.arange(5, N + 1)
    state = model.prepare(snap(4, 4 * F), laps, df, OUR)
    lap_s = np.where(np.isin(laps, [5, 6]), 110.0, F)
    inc, _, _ = apply(model, state, laps, lap_s=lap_s, sc=(5, 6))
    got = by_lap(inc, laps)
    assert got[5] == got[6] == 0.0
    assert got[7] == pytest.approx(0.5) and got[8] == pytest.approx(0.5)   # held, DRS off
    assert first_pass_lap(inc, laps) == 9
    # the snapshot's SC history counts for the first remaining laps
    df = frame(ghost("AAA", 80.5, 1.0, k=6))
    laps = np.arange(7, N + 1)
    s = Snapshot(race=RACE, driver=OUR, laps_completed=6, compound="SOFT", tyre_age=6, sc_laps=[5, 6],
                 elapsed_s=6 * F, standing_start=False)
    state = model.prepare(s, laps, df, OUR)
    inc, _, _ = apply(model, state, laps)
    assert first_pass_lap(inc, laps) == 9
    assert by_lap(inc, laps)[7] == pytest.approx(0.5)


def test_ghost_in_lap_never_clamps_its_out_lap_does():
    # 6. held at the follow gap; the ghost's in-lap (8) frees us, its out-lap (9) holds us again
    df = frame(ghost("AAA", 80.5, 1.0, pit_in=(8,)))
    model, state, laps = prepare(df, **NO_DRS)
    assert not state.eligible[laps.tolist().index(8)].any() and state.eligible[laps.tolist().index(9)].all()
    got = by_lap(apply(model, state, laps)[0], laps)
    assert got[6] == pytest.approx(0.5) and got[7] == pytest.approx(0.5)
    assert got[8] == 0.0
    assert got[9] == pytest.approx(1.0)                               # 0.5 s behind, held to G + 1


def test_neutral_laps_get_no_increment():
    # 7. catch would hold on the SC lap 7 and the VSC lap 9
    model, state, laps = prepare(frame(ghost("AAA", 80.5, 1.0)), **NO_DRS)
    got = by_lap(apply(model, state, laps, sc=(7,), vsc=(9,))[0], laps)
    assert got[7] == 0.0 and got[9] == 0.0
    assert got[6] == pytest.approx(0.5) and got[8] > 0 and got[10] > 0


def test_max_passes_per_lap_caps_the_charge():
    # 12. three cars 2 s/lap slower just ahead, all passed on lap 6
    ghosts = [ghost(name, 82.0, gap) for name, gap in (("AAA", 0.2), ("BBB", 0.3), ("CCC", 0.4))]
    for cap, charged in ((3, 3), (2, 2)):
        model, state, laps = prepare(frame(*ghosts), max_passes_per_lap=cap)
        inc, traffic_s, passes = apply(model, state, laps)
        assert passes[0] == charged
        assert by_lap(inc, laps)[6] == pytest.approx(0.4 * charged)
        assert np.all(inc[0, 1:] == 0.0)


def test_dirty_air_is_off_by_default_and_charged_within_range():
    df = frame(ghost("AAA", 80.0, 1.5))   # same pace, 1.5 s ahead: never caught
    model, state, laps = prepare(df)
    assert np.all(apply(model, state, laps)[0] == 0.0)
    model, state, laps = prepare(df, dirty_air_s=0.2)
    inc = apply(model, state, laps)[0]
    assert inc[0, :3] == pytest.approx([0.2] * 3) and np.all(inc[0, 3:] == 0.0)   # until 2 s behind


def test_anchor_mismatch_warning():
    # 11. the anchor implies 3.0 s to AAA
    df = frame(ghost("AAA", 80.5, 3.0))
    model = GhostTraffic()
    laps = np.arange(6, N + 1)
    for gap, warned in ((4.5, True), (3.5, False), (1.5, True), (2.5, False)):
        state = model.prepare(snap(car_ahead="AAA", gap_ahead_s=gap), laps, df, OUR)
        w = model.support_warnings(snap(), state)
        assert w == ([f"traffic_anchor_mismatch:{3.0 - gap:.1f}"] if warned else [])


# --- engine hook ---------------------------------------------------------------------
LIN = {"name": "linear"}   # our laps have tyre age = lap, so keep the fuel term out of the fit


def off_sim(df):
    return Simulator(df, SimConfig(degradation=LIN))


def ghost_sim(df, **params):
    return Simulator(df, SimConfig(degradation=LIN, traffic={"name": "ghost", "params": params}))


def test_elapsed_none_places_no_traffic():
    # 10. state None, a warning on every plan, lap times unchanged
    df = frame(ghost("AAA", 80.5, 1.0))
    s = snap(elapsed=None).model_copy(update={"elapsed_s": None})
    plans = [[["SOFT", 15]], [["SOFT", 7], ["SOFT", 8]]]
    on = ghost_sim(df).simulate_batch(s, plans, rules=RULES, return_laps=True)
    off = off_sim(df).simulate_batch(s, plans, rules=RULES, return_laps=True)
    assert not on.traffic_on and np.array_equal(on.lap_times, off.lap_times)
    assert all("traffic_unavailable:no_elapsed_s" in w for w in on.support_warnings)
    assert on.result(0, s).traffic_s is None and off.result(0, s).traffic_s is None


def test_pit_loss_is_booked_at_the_end_of_the_in_lap():
    # 8. 15.5 s ahead of a car 0.5 s/lap slower; the stop on lap 12 drops us 1 s behind it,
    # where we are held for the rest of the race. lap_s excludes the pit loss.
    df = frame(car("BBB", 80.5, start=5 * F + 15.5 - 5 * 80.5, compound="MEDIUM"))
    s = snap()
    plan = [["SOFT", 7], ["SOFT", 8]]
    on = ghost_sim(df, **NO_DRS).simulate_batch(s, [plan], rules=RULES, return_laps=True)
    off = off_sim(df).simulate_batch(s, [plan], rules=RULES, return_laps=True)
    inc = on.lap_times[0] - off.lap_times[0]
    assert np.allclose(inc[:7], 0.0)                                   # ahead of BBB until the stop
    assert np.allclose(inc[7:], 0.5)                                   # out-lap 13 on: behind it
    assert on.lap_times[0, 6] == off.lap_times[0, 6]                   # the in-lap has no pit loss in it
    assert on.times[0, 0] == pytest.approx(on.lap_times[0].sum() + RACE.pit_loss_s)
    assert on.pit_loss[0, 0] == off.pit_loss[0, 0] == RACE.pit_loss_s
    assert on.stint_seconds[0][:, 0] == pytest.approx([on.lap_times[0, :7].sum(), on.lap_times[0, 7:].sum()])
    assert on.traffic_s[0, 0] == pytest.approx(0.5 * 8) and on.passes[0, 0] == 0
    res = on.result(0, s)
    assert res.traffic_s == pytest.approx(4.0) and res.passes_mean == 0.0
    assert off.result(0, s).traffic_s is None and off.result(0, s).passes_mean is None


def test_common_random_numbers_across_plans():
    # 9. same draws and ghosts for every plan: identical plans get identical traffic, plans
    # sharing the first stint get identical increments up to the first pit
    df = frame(ghost("AAA", 80.5, 3.0), ghost("BBB", 81.0, 9.0), car("CCC", 80.5, start=30.0, compound="MEDIUM"))
    cfg = SimConfig(degradation=LIN, traffic={"name": "ghost"}, events={"name": "rate", "params": {"p_sc": 0.1, "p_vsc": 0.1}})
    plans = [[["SOFT", 7], ["SOFT", 8]], [["SOFT", 7], ["SOFT", 8]], [["SOFT", 7], ["MEDIUM", 8]],
             [["SOFT", 10], ["SOFT", 5]]]
    s = snap()
    on = Simulator(df, cfg).simulate_batch(s, plans, rules=RULES, n_samples=20, return_laps=True, seed=4)
    off = Simulator(df, cfg.model_copy(update={"traffic": SimConfig().traffic})).simulate_batch(
        s, plans, rules=RULES, n_samples=20, return_laps=True, seed=4)
    assert np.array_equal(on.traffic_s[0], on.traffic_s[1]) and np.array_equal(on.passes[0], on.passes[1])
    inc = on.lap_times - off.lap_times
    assert np.array_equal(inc[0, :7], inc[2, :7]) and np.array_equal(inc[0, :7], inc[3, :7])
    assert on.traffic_s.any()
    assert np.array_equal(on.pit_loss, off.pit_loss)                  # events unchanged by traffic


def test_traffic_flows_into_compare_and_times():
    df = frame(ghost("AAA", 80.5, 3.0))
    sim = ghost_sim(df, **NO_DRS)
    s = snap()
    c = sim.compare(s, [[["SOFT", 15]]], rules=RULES)
    b = sim.simulate_batch(s, [[["SOFT", 15]]], rules=RULES)
    off = off_sim(df).simulate_batch(s, [[["SOFT", 15]]], rules=RULES)
    assert c.summaries[0].mean_s == pytest.approx(b.times[0, 0])
    assert b.times[0, 0] - off.times[0, 0] == pytest.approx(b.traffic_s[0, 0]) == pytest.approx(5.5)


# --- rejoin ----------------------------------------------------------------------------
def rejoin_frame():
    """XXX pits on lap 10 (in-lap 80 + 20 s), rejoins 0.7 s behind AAA (80.5 s/lap) and is held
    on the out-lap to 1 s behind it, as recorded (out-lap 80.8 s)."""
    ours = car(OUR, F, pit_in=(10,), times={10: F + 20.0, 11: 80.8}, position=2.0)
    aaa = car("AAA", 80.5, start=738.8 - 9 * 80.5, compound="MEDIUM")
    return pd.DataFrame(ours + aaa)


def test_rejoin_matches_the_construction():
    # 13. traffic on reproduces the held out-lap; off is the pure pit-loss view
    df = rejoin_frame()
    on = calibration.rejoin(ghost_sim(df, **NO_DRS), df, RACE)
    off = calibration.rejoin(off_sim(df), df, RACE)
    for t in (on, off):
        assert t[["driver", "in_lap", "k", "out_lap"]].values.tolist() == [[OUR, 10, 9, 11]]
        assert t["pred_pos"].tolist() == [2] and t["real_pos"].tolist() == [2.0] and t["clock_pos"].tolist() == [2]
        assert t["real_gap_ahead_s"].iloc[0] == pytest.approx(1.0)
    assert on["pred_gap_ahead_s"].iloc[0] == pytest.approx(1.0) and on["traffic_s"].iloc[0] == pytest.approx(0.8)
    assert off["pred_gap_ahead_s"].iloc[0] == pytest.approx(0.2) and off["traffic_s"].isna().all()
    assert calibration.rejoin_summary(on) == {"n_stops": 1, "mae_pos": 0.0, "within_1_rate": 1.0,
                                              "gap_bias_s": 0.0, "n_gaps": 1}
    assert calibration.rejoin_summary(off)["gap_bias_s"] == pytest.approx(-0.8)


def test_placement_ranks_the_replay_among_the_ghosts():
    df = rejoin_frame()
    t = calibration.placement(ghost_sim(df, **NO_DRS), df, RACE, lap=12).set_index("driver").loc[[OUR]]
    assert t["pred_pos"].tolist() == [2] and t["real_pos"].tolist() == [2.0] and t["abs_err"].tolist() == [0.0]


def test_traffic_report_has_every_table_for_both_arms():
    from src.sim import calibrate
    df = rejoin_frame().assign(team="T")   # one team, so the pairs table has a row
    off, on = calibrate.traffic_arms(SimConfig(degradation=LIN))
    assert (off.traffic.name, on.traffic.name) == ("none", "ghost")
    assert calibrate.traffic_arms(SimConfig(traffic={"name": "ghost", "params": NO_DRS}))[1].traffic.params == NO_DRS
    table, summaries = calibrate.traffic_report(df, RACE, SimConfig(degradation=LIN), {OUR: 0.0}, None)
    got = table.groupby(["table", "traffic"]).size()
    assert set(got.index) == {(t, a) for t in calibrate.TRAFFIC_TABLES for a in ("none", "ghost")}
    assert set(summaries) == {"none", "ghost"}
    assert table.groupby("traffic")["config_hash"].nunique().tolist() == [1, 1]
    assert summaries["ghost"]["rejoin"]["n_stops"] == 1


# --- information boundary --------------------------------------------------------------
def perturb_after(df, k, lap_times=False):
    """The frame with every car's session clock (and optionally lap times) changed after lap k."""
    df = df.copy()
    late = df["lap"] > k
    for col in ("start_s", "end_s"):
        df.loc[late, col] += 37.0
    df.loc[late, "position"] = 99.0
    if lap_times:
        df.loc[late, "lap_time_s"] += 1.3
    return df


def test_information_boundary_canary_default_config():
    """A default-config snapshot result is unchanged when the field's laps after k are
    perturbed (traffic is off by default, so the future session clock is never read); with
    traffic on it changes, which is why traffic stays world-model only."""
    df = frame(ghost("AAA", 80.5, 3.0), ghost("BBB", 81.0, 9.0))
    s, plan = snap(), [[["SOFT", 7], ["SOFT", 8]]]
    for sim_of in (Simulator, lambda d: ghost_sim(d)):
        a = sim_of(df).simulate_batch(s, plan, rules=RULES, n_samples=5, return_laps=True)
        b = sim_of(perturb_after(df, 5)).simulate_batch(s, plan, rules=RULES, n_samples=5, return_laps=True)
        same = np.array_equal(a.times, b.times) and np.array_equal(a.lap_times, b.lap_times)
        assert same == (sim_of is Simulator)


def test_information_boundary_canary_to_snapshot_window():
    """Under the to_snapshot window lap times after k are invisible as well."""
    df = race_frame(("AAA", "T", [("SOFT", 20), ("MEDIUM", 30)]), ("BBB", "T", [("SOFT", 10), ("MEDIUM", 40)]))
    cfg = SimConfig(effects=EventEffects(lap1_extra_s=0.0), fit_window="to_snapshot", pre_race_base_delta_s=0.0)
    s = calibration.snapshot_at(df, SYN_RACE, "AAA", 15)
    plan = calibration.remaining_plan(df, "AAA", 15)
    a = Simulator(df, cfg, synthetic_practice()).simulate(s, plan, rules=RULES, return_laps=True)
    b = Simulator(perturb_after(df, 15, lap_times=True), cfg, synthetic_practice()).simulate(
        s, plan, rules=RULES, return_laps=True)
    assert a.model_dump_json() == b.model_dump_json()
