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
from src.sim.config import FREE_AIR_GAP_S, ComponentSpec, EventEffects
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


def apply(model, state, laps, lap_s=F, pit=None, sc=(), vsc=(), n_plans=1, bunching=False):
    """Returns (increment per lap, traffic_s, passes), each per plan; with `bunching` also bunching_s."""
    lap_s = np.broadcast_to(np.asarray(lap_s, dtype=float), (n_plans, len(laps))).copy()
    pit = np.zeros_like(lap_s) if pit is None else pit
    out, (traffic_s, passes, bunching_s, _) = model.apply(state, lap_s, pit, np.isin(laps, sc), np.isin(laps, vsc),
                                                       Scenario(sc_laps=list(sc), vsc_laps=list(vsc)))
    return (out - lap_s, traffic_s, passes) + ((bunching_s,) if bunching else ())


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


def test_traffic_is_part_of_the_config_hash():
    # no special case for the default: adding the field moved the default hash off main's
    # 0778d5ef02f6 (eeea1e3), as any new config field does
    default, ghost_on = calibration.config_hash(SimConfig()), calibration.config_hash(SimConfig(traffic={"name": "ghost"}))
    assert default == "eccb59070c14" and ghost_on == "a50205e6c507" and default != "0778d5ef02f6"


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
    assert t.out_lap[0].tolist() == [lap == 9 for lap in range(6, N + 1)]
    assert np.isnan(ghost_table(df, OUR, np.arange(1, N + 1)).G_prev[0, 0])   # no lap-0 row


def test_ghost_table_and_gap_columns_share_the_crossings():
    # one home for the end_s(L - 1) arithmetic (data.line_crossings): the lap frame's
    # gap_ahead_s to AAA is our crossing less AAA's G_prev on every lap AAA is ahead
    from src.sim.data import gap_columns
    df = gap_columns(frame(ghost("AAA", 80.5, 3.0)))
    laps = np.arange(2, N + 1)
    t = ghost_table(df, OUR, laps)
    ours = df[df["driver"] == OUR].set_index("lap")
    ahead = ours.loc[laps, "car_ahead"].eq("AAA").to_numpy()   # the recorded XXX passes AAA on lap 7
    assert ahead.sum() == 10
    ours_start = ours.loc[laps - 1, "end_s"].to_numpy()
    assert ours.loc[laps, "gap_ahead_s"].to_numpy()[ahead] == pytest.approx((ours_start - t.G_prev[0])[ahead])
    assert np.all((ours_start - t.G_prev[0])[~ahead] <= 0)   # level or ahead of it: not the car ahead


def test_held_behind_a_slower_car():
    # 3. 0.5 s/lap faster than a ghost 3 s ahead: close at 0.5 s/lap, then lap at its pace
    model, state, laps = prepare(frame(ghost("AAA", 80.5, 3.0)), **NO_DRS)
    inc, traffic_s, passes = apply(model, state, laps)
    got = by_lap(inc, laps)
    assert all(got[lap] == 0.0 for lap in range(6, 10))             # gap 3.0, 2.5, 2.0, 1.5
    assert all(got[lap] == pytest.approx(0.5) for lap in range(10, N + 1))   # lap_s = 80.5
    assert traffic_s[0] == pytest.approx(inc.sum()) == pytest.approx(0.5 * 11)
    assert passes[0] == 0


def test_a_pass_is_a_position_swap():
    # 4. 1.0 s/lap faster than a ghost 3 s ahead, pass_delta 0.8: caught on the third lap
    # (lap 8, 1.0 s behind its line at the start), passed once, ending pass_gap_s ahead of
    # it; arrive + pass_cost_s would be 0.4 s behind, so the swap binds: increment -0.2
    model, state, laps = prepare(frame(ghost("AAA", 81.0, 3.0)), **NO_DRS)
    inc, traffic_s, passes = apply(model, state, laps)
    got = by_lap(inc, laps)
    assert got[6] == got[7] == 0.0
    assert got[8] == pytest.approx(-0.2)
    assert all(got[lap] == 0.0 for lap in range(9, N + 1))           # no further charge
    assert passes[0] == 1 and traffic_s[0] == pytest.approx(-0.2)
    end8 = 8 * F - 0.2
    G8 = state.table.G[0, laps.tolist().index(8)]
    assert end8 == pytest.approx(G8 - 0.2) and end8 < G8               # d < 0 from lap 9 on


def test_held_until_drs_then_one_pass():
    # 0.5 s/lap faster, 3 s behind after lap 1: free laps 2-5 close to 1.0 s at the start of
    # lap 6, where DRS (0.3) lets the pass through; one charge, then free air
    model = GhostTraffic()
    laps = np.arange(2, N + 1)
    state = model.prepare(snap(1, F), laps, frame(ghost("AAA", 80.5, 3.0, k=1)), OUR)
    inc, traffic_s, passes = apply(model, state, laps)
    got = by_lap(inc, laps)
    assert all(got[lap] == 0.0 for lap in range(2, 6))
    assert got[6] == pytest.approx(-0.7)                               # G - pass_gap_s, from G + 0.5
    assert all(got[lap] == 0.0 for lap in range(7, N + 1))
    assert passes[0] == 1


def first_pass_lap(inc, laps):
    """The first lap whose increment is negative: a pass from close behind, where the swap binds."""
    hit = inc[0] < 0
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
    assert first_pass_lap(inc, laps) == 3 and got[3] == pytest.approx(-0.7) and passes[0] == 1


def test_drs_off_for_two_laps_after_an_sc_period():
    # 5. same car held at the follow gap from lap 5; SC on laps 5-6, ghost's laps there as ours
    # (bunching off: under it the queue interval would put us 1.2 s back, see the 0013 tests)
    times = {5: 110.0, 6: 110.0}
    df = frame(ghost("AAA", 80.5, 1.0, k=4, times=times))
    model = GhostTraffic(sc_bunching=False)
    laps = np.arange(5, N + 1)
    state = model.prepare(snap(4, 4 * F), laps, df, OUR)
    lap_s = np.where(np.isin(laps, [5, 6]), 110.0, F)
    inc, _, passes = apply(model, state, laps, lap_s=lap_s, sc=(5, 6))
    got = by_lap(inc, laps)
    assert got[5] == got[6] == 0.0
    assert got[7] == pytest.approx(0.5) and got[8] == pytest.approx(0.5)   # held, DRS off
    assert first_pass_lap(inc, laps) == 9 and passes[0] == 1
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
    # 6. held at the follow gap; the ghost's in-lap (8) frees us (0.5 s behind after it), its
    # out-lap (9, 80.2 s) has it rejoin 0.3 s ahead of our line: held to G + 1
    df = frame(ghost("AAA", 80.5, 1.0, pit_in=(8,), times={9: 80.2}))
    model, state, laps = prepare(df, **NO_DRS)
    assert not state.eligible[laps.tolist().index(8)].any() and state.eligible[laps.tolist().index(9)].all()
    got = by_lap(apply(model, state, laps)[0], laps)
    assert got[6] == pytest.approx(0.5) and got[7] == pytest.approx(0.5)
    assert got[8] == 0.0
    assert got[9] == pytest.approx(0.7)


def out_lap_ghost(gap_after_line):
    """AAA pits on lap 7 and runs its out-lap 8 in 82 s, 2 s/lap slower than us, so it would
    be passable on track; we finish lap 8 `gap_after_line` seconds after its line."""
    return frame(car("AAA", F, start=-2.0 - gap_after_line, compound="MEDIUM", pit_in=(7,), times={8: 82.0}))


@pytest.mark.parametrize("on_out_lap", [True, False])
def test_out_lap_ghost_is_never_passed(on_out_lap):
    model = GhostTraffic()
    laps = np.arange(8, N + 1)
    df = out_lap_ghost(0.3)
    if not on_out_lap:
        df.loc[df["driver"] == "AAA", ["pit_in", "pit_out"]] = False
    state = model.prepare(snap(7), laps, df, OUR)
    inc, _, passes = apply(model, state, laps)
    if on_out_lap:   # it rejoined 0.3 s ahead of us: a clamp, not a pass
        assert by_lap(inc, laps)[8] == pytest.approx(0.7) and passes[0] == 0
    else:            # the same car on track is passed
        assert by_lap(inc, laps)[8] == pytest.approx(-0.5) and passes[0] == 1
    # rejoining behind us: nothing, no cost and no swap
    state = model.prepare(snap(7), laps, out_lap_ghost(-0.3), OUR)
    inc, traffic_s, passes = apply(model, state, laps)
    assert np.all(inc == 0.0) and passes[0] == 0


def test_neutral_laps_get_no_increment():
    # 7. catch would hold on the SC lap 7 and the VSC lap 9 (bunching off: #27's SC branch)
    model, state, laps = prepare(frame(ghost("AAA", 80.5, 1.0)), sc_bunching=False, **NO_DRS)
    got = by_lap(apply(model, state, laps, sc=(7,), vsc=(9,))[0], laps)
    assert got[7] == 0.0 and got[9] == 0.0
    assert got[6] == pytest.approx(0.5) and got[8] > 0 and got[10] > 0


def test_three_passes_in_one_lap():
    # three cars 2 s/lap slower just ahead, all passed on lap 6: free (arrive + 1.2) is
    # ahead of the swap bound, so it binds
    ghosts = [ghost(name, 82.0, gap) for name, gap in (("AAA", 0.2), ("BBB", 0.3), ("CCC", 0.4))]
    model, state, laps = prepare(frame(*ghosts))
    inc, traffic_s, passes = apply(model, state, laps)
    assert passes[0] == 3 and by_lap(inc, laps)[6] == pytest.approx(1.2)
    assert np.all(inc[0, 1:] == 0.0)


def test_max_passes_per_lap_keeps_the_closest():
    # 12. four passable cars, cap 3: the three closest (largest G_prev, 0.2-0.4 s ahead) are
    # passed and charged, DDD (1.7 s ahead) holds us at G + 1
    ghosts = [ghost(name, 82.0, gap) for name, gap in (("AAA", 0.2), ("BBB", 0.3), ("CCC", 0.4), ("DDD", 1.7))]
    df, lap6 = frame(*ghosts), np.array([6])

    def one_lap(**params):
        model = GhostTraffic(**params)
        state = model.prepare(snap(), lap6, df, OUR)
        inc, _, passes = apply(model, state, lap6)
        return state, 6 * F + inc[0, 0], passes[0]

    state, end, passes = one_lap()
    G = dict(zip(state.table.ghosts, state.table.G[:, 0]))
    assert passes == 3
    assert end == pytest.approx(G["DDD"] + 1.0)                       # behind DDD, ahead of the three
    assert end < min(G["AAA"], G["BBB"], G["CCC"]) - 0.2
    # over the race DDD is passed on lap 7, from 1 s behind with DRS: 4 passes in all
    model, state, laps = prepare(df)
    inc, _, passes = apply(model, state, laps)
    assert passes[0] == 4 and by_lap(inc, laps)[7] == pytest.approx(0.4)
    # cap 1: AAA is kept, but BBB (ahead of AAA on the line) is now held, and a kept pass
    # never ends past the car holding us, so nothing is passed
    _, end, passes = one_lap(max_passes_per_lap=1)
    assert passes == 0 and end == pytest.approx(G["BBB"] + 1.0)
    _, end, passes = one_lap(max_passes_per_lap=0)
    assert passes == 0 and end == pytest.approx(G["AAA"] + 1.0)        # held behind the last on the line


def test_no_swap_past_the_car_that_holds_us():
    # held by AAA (0.5 s/lap slower, no DRS help); BBB is passable but ahead of AAA on the
    # line, so passing it would put us past the car holding us: no swap
    a, b = ghost("AAA", 80.5, 0.5), ghost("BBB", 82.0, 2.5)
    model, state, laps = prepare(frame(a, b), **NO_DRS)
    inc, _, passes = apply(model, state, laps)
    assert passes[0] == 0 and by_lap(inc, laps)[6] == pytest.approx(1.0)   # held at G_AAA + 1
    model, state, laps = prepare(frame(b), **NO_DRS)                        # without AAA it swaps
    inc, _, passes = apply(model, state, laps)
    assert passes[0] == 1 and by_lap(inc, laps)[6] == pytest.approx(-0.7)


def test_dirty_air_is_off_by_default_and_charged_within_range():
    df = frame(ghost("AAA", 80.0, 1.5))   # same pace, 1.5 s ahead: never caught
    model, state, laps = prepare(df)
    assert np.all(apply(model, state, laps)[0] == 0.0)
    # the window defaults to FREE_AIR_GAP_S, the gap dirty_air_s is measured under: 1.5 s,
    # then 1.7 s behind, so one lap; overridden to 2.0 s, three laps (1.5, 1.7, 1.9)
    assert GhostTraffic().params.dirty_air_range_s == FREE_AIR_GAP_S == 1.5
    model, state, laps = prepare(df, dirty_air_s=0.2)
    inc = apply(model, state, laps)[0]
    assert inc[0, 0] == pytest.approx(0.2) and np.all(inc[0, 1:] == 0.0)
    model, state, laps = prepare(df, dirty_air_s=0.2, dirty_air_range_s=2.0)
    inc = apply(model, state, laps)[0]
    assert inc[0, :3] == pytest.approx([0.2] * 3) and np.all(inc[0, 3:] == 0.0)


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


# --- SC bunching (SPEC 0013) ----------------------------------------------------------
# SC on laps 5-6, snapshot at k = 4: our SC laps and the ghosts' are 110 s, so every gap at
# the end of lap 4 is still the gap at the line on laps 5 and 6 unless the queue moves us.
SC_LAPS, SC_LAP_S = (5, 6), 110.0


def sc_ghost(name, ahead, lap_s=80.0, sc_lap_s=SC_LAP_S, **kw):
    times = {lap: sc_lap_s for lap in SC_LAPS}
    times.update(kw.pop("times", {}))
    return ghost(name, lap_s, ahead, k=4, times=times, **kw)


def sc_apply(ghosts, f_extra=0.0, pit=None, vsc=False, ref_lap_s=F, **params):
    """(model, state, laps, increments, traffic_s, passes, bunching_s) over laps 5-20, bunching on."""
    df = frame(*ghosts)
    model = GhostTraffic(**{"sc_bunching": True, **params})
    laps = np.arange(5, N + 1)
    state = model.prepare(snap(4, 4 * F), laps, df, OUR, ref_lap_s=ref_lap_s)
    extra = np.atleast_1d(np.asarray(f_extra, dtype=float))   # per plan, on lap 5 only
    lap_s = np.tile(np.where(np.isin(laps, SC_LAPS), SC_LAP_S, F), (len(extra), 1))
    lap_s[:, 0] += extra
    kw = {"vsc": SC_LAPS} if vsc else {"sc": SC_LAPS}
    out = apply(model, state, laps, lap_s=lap_s, pit=pit, n_plans=len(lap_s), bunching=True, **kw)
    return (model, state, laps) + out


def line(state, name, lap, laps):
    """Ghost `name`'s crossing at the end of `lap`."""
    return state.G[list(laps).index(lap), state.table.ghosts.index(name)]


def our_end(state, laps, inc, lap, lap_s=SC_LAP_S, pit=0.0):
    """Our clock at the end of `lap` (laps 5.. from the anchor), increments and pit loss in."""
    upto = list(laps).index(lap) + 1
    base = np.where(np.isin(laps[:upto], SC_LAPS), lap_s, F).sum()
    return state.anchor + base + inc[:, :upto].sum(axis=1) + pit


def test_sc_bunching_closes_a_20_s_gap_to_the_queue_interval():
    # 1. 20 s behind AAA, BBB 10 s behind us: end of lap 5 at G_A + min(1.2, 30 / 2) = G_A + 1.2
    model, state, laps, inc, traffic_s, passes, bunching_s = sc_apply([sc_ghost("AAA", 20.0), sc_ghost("BBB", -10.0)])
    got = by_lap(inc, laps)
    assert got[5] == pytest.approx(-18.8) and got[6] == pytest.approx(0.0, abs=1e-9)
    assert our_end(state, laps, inc, 6)[0] == pytest.approx(line(state, "AAA", 6, laps) + 1.2)
    assert bunching_s[0] == pytest.approx(-18.8) and traffic_s[0] == 0.0 and passes[0] == 0
    assert np.all(inc[0, 2:] == 0.0)   # AAA laps at our pace from lap 7: 1.2 s back, never caught


def test_sc_bunching_leader_keeps_the_sc_lap_time_exactly():
    # 2. no ghost ahead at the line: end = e, the field's SC lap (here equal to ours) bit for bit
    model, state, laps, inc, traffic_s, passes, bunching_s = sc_apply([sc_ghost("AAA", -5.0), sc_ghost("BBB", -15.0)])
    assert np.all(inc[0, :2] == 0.0) and bunching_s[0] == 0.0


def field_apply(ghosts, sc_lap_s=110.0, ref_lap_s=F):
    """sc_apply with the engine's SC lap at sc_lap_s; returns (state, laps, lap_s out, field delta, bunching_s)."""
    df = frame(*ghosts)
    model = GhostTraffic(sc_bunching=True)
    laps = np.arange(5, N + 1)
    state = model.prepare(snap(4, 4 * F), laps, df, OUR, ref_lap_s=ref_lap_s)
    lap_s = np.where(np.isin(laps, SC_LAPS), sc_lap_s, F)[None, :]
    out, (_, _, bunching_s, delta) = model.apply(state, lap_s, np.zeros_like(lap_s), np.isin(laps, SC_LAPS),
                                                 np.zeros(len(laps), dtype=bool), Scenario(sc_laps=list(SC_LAPS)))
    return state, laps, out, delta, bunching_s


def test_sc_lap_is_the_field_recorded_sc_lap():
    # the queue ran 130 s and 135 s; the engine gives 110 s. The leader laps at the field's
    # median, and the per-lap field delta is recorded; a car 20 s back still queues at G_A + 1.2
    times = {5: 130.0, 6: 135.0}
    lead = [sc_ghost("AAA", -5.0, times=times), sc_ghost("BBB", -15.0, times={5: 131.0, 6: 136.0}),
            sc_ghost("CCC", -25.0, times={5: 129.0, 6: 134.0})]
    state, laps, out, delta, bunching_s = field_apply(lead)
    assert out[0, :2].tolist() == [130.0, 135.0]
    assert delta[0, :2].tolist() == [20.0, 25.0] and np.all(np.isnan(delta[0, 2:]))
    assert bunching_s[0] == pytest.approx(45.0)
    state, laps, out, delta, _ = field_apply([sc_ghost("AAA", 20.0, times=times), sc_ghost("BBB", -10.0, times=times)])
    end6 = state.anchor + out[0, :2].sum()
    assert end6 == pytest.approx(line(state, "AAA", 6, laps) + 1.2)
    # no ghost in the queue: the engine's lap is the fallback, nothing recorded
    state, laps, out, delta, bunching_s = field_apply([sc_ghost("AAA", -60.0, times=times)])
    assert out[0, :2].tolist() == [110.0, 110.0] and np.all(np.isnan(delta)) and bunching_s[0] == 0.0


def test_sc_queue_half_lap_guard_drops_cars_a_lap_away():
    # AAA 50 s ahead is more than half a reference lap (40 s) away: not in the queue, so we
    # lead and keep e; inside the guard (ref 120 s, 60 s) it is the car ahead
    state, laps, out, delta, _ = field_apply([sc_ghost("AAA", 50.0)])
    assert out[0, :2].tolist() == [110.0, 110.0]
    state, laps, out, delta, _ = field_apply([sc_ghost("AAA", 50.0)], ref_lap_s=120.0)
    assert state.anchor + out[0, 0] == pytest.approx(line(state, "AAA", 5, laps) + 1.2)


def test_sc_queue_drops_in_lap_ghosts_and_keeps_out_lap_ghosts():
    # AAA pits on lap 5 (its in-lap, 3 s ahead): it leaves the queue, so BBB (8 s ahead) is A
    pit = {"pit_in": (5,), "times": {5: 125.0, 6: 110.0}}
    state, laps, out, delta, _ = field_apply([sc_ghost("AAA", 3.0, **pit), sc_ghost("BBB", 8.0)])
    assert state.anchor + out[0, 0] == pytest.approx(line(state, "BBB", 5, laps) + 1.2)
    assert delta[0, 0] == 0.0   # the median is BBB's 110 s alone, not AAA's in-lap
    # lap 6 is AAA's out-lap (91.5 s, rejoining 1.5 s behind BBB): it stays in the queue as B,
    # so half its gap to BBB binds (0.75 s), but its lap is not timed (the median stays 110 s)
    state, laps, out, delta, _ = field_apply([sc_ghost("AAA", 3.0, **{"pit_in": (5,), "times": {5: 125.0, 6: 91.5}}),
                                              sc_ghost("BBB", 8.0)])
    assert delta[0, 1] == 0.0
    assert line(state, "AAA", 6, laps) == pytest.approx(line(state, "BBB", 6, laps) + 1.5)
    assert state.anchor + out[0, :2].sum() == pytest.approx(line(state, "BBB", 6, laps) + 0.75)


def test_sc_bunching_pushes_a_close_car_back_to_the_interval():
    # 3. 0.3 s behind AAA, nobody behind: pushed back to G_A + 1.2
    model, state, laps, inc, *_, bunching_s = sc_apply([sc_ghost("AAA", 0.3)])
    assert by_lap(inc, laps)[5] == pytest.approx(0.9) and bunching_s[0] == pytest.approx(0.9)
    assert our_end(state, laps, inc, 5)[0] == pytest.approx(line(state, "AAA", 5, laps) + 1.2)
    # BBB 1.0 s behind AAA: half the gap, 0.5 s, binds before the interval
    model, state, laps, inc, *_ = sc_apply([sc_ghost("AAA", 0.3), sc_ghost("BBB", -0.7)])
    assert our_end(state, laps, inc, 5)[0] == pytest.approx(line(state, "AAA", 5, laps) + 0.5)
    assert by_lap(inc, laps)[5] == pytest.approx(0.2)


def test_sc_bunching_places_a_pit_under_the_sc_behind_the_right_ghost():
    # 4. AAA 5 s, BBB 15 s and CCC 30 s behind us; a stop on SC lap 5 at 12 s puts e between
    # AAA and BBB (queued 1.2 s behind AAA), at 20 s between BBB and CCC (1.2 s behind BBB).
    # The pit loss stays out of lap_s.
    ghosts = [sc_ghost("AAA", -5.0), sc_ghost("BBB", -15.0), sc_ghost("CCC", -30.0)]
    pit = np.zeros((2, N - 4))
    pit[:, 0] = [12.0, 20.0]
    model, state, laps, inc, traffic_s, passes, bunching_s = sc_apply(ghosts, f_extra=[0.0, 0.0], pit=pit)
    assert inc[:, 0] == pytest.approx([1.2 + 5.0 - 12.0, 1.2 + 15.0 - 20.0])
    for row, behind in ((0, "AAA"), (1, "BBB")):
        assert our_end(state, laps, inc, 6, pit=pit[row, 0])[row] == pytest.approx(line(state, behind, 6, laps) + 1.2)
    assert bunching_s == pytest.approx([-5.8, -3.8]) and np.all(passes == 0)


def test_vsc_laps_keep_their_gaps_under_sc_bunching():
    # 5. the same 20 s gap under a VSC: no increment
    model, state, laps, inc, traffic_s, passes, bunching_s = sc_apply(
        [sc_ghost("AAA", 20.0), sc_ghost("BBB", -10.0)], vsc=True)
    assert np.all(inc[0, :2] == 0.0) and bunching_s[0] == 0.0


def test_sc_bunching_off_is_the_old_sc_branch():
    # 6. on the synthetic batches: no increment on SC laps, as #27
    model, state, laps, inc, traffic_s, passes, bunching_s = sc_apply(
        [sc_ghost("AAA", 20.0), sc_ghost("BBB", -10.0)], sc_bunching=False)
    assert np.all(inc[0] == 0.0) and bunching_s[0] == 0.0


def test_sc_bunching_never_ends_past_the_car_behind():
    # 8. many provisional ends across a field: the order against every ghost is kept, and
    # the end is never past the ghost directly behind e
    # the lap time is the field's, so a pit loss under the SC moves e over the queue
    ghosts = [sc_ghost(n, a) for n, a in (("AAA", -1.0), ("BBB", -1.5), ("CCC", -7.0), ("DDD", -10.0),
                                           ("EEE", -10.8), ("FFF", -22.0))]
    pit_s = np.linspace(0.0, 28.0, 113)
    pit = np.zeros((len(pit_s), N - 4))
    pit[:, 0] = pit_s
    model, state, laps, inc, *_ = sc_apply(ghosts, f_extra=np.zeros(len(pit_s)), pit=pit)
    G = state.G[0]
    e = state.anchor + SC_LAP_S + pit_s
    end = e + inc[:, 0]
    assert np.array_equal(G[None, :] <= e[:, None], G[None, :] <= end[:, None])
    b = np.where(G[None, :] > e[:, None], G, np.inf).min(axis=1)
    assert np.all(end < b)
    assert (inc[:, 0] < 0).any() and (inc[:, 0] > 0).any() and (inc[:, 0] == 0).any()   # closed, pushed, led


def ghost_digest(traffic):
    """sha256 over a 13-plan x 40-sample batch with drawn SC and VSC on a five-ghost field."""
    df = frame(ghost("AAA", 80.5, 3.0), ghost("BBB", 81.0, 9.0), ghost("CCC", 79.5, -4.0),
               ghost("DDD", 80.2, 20.0, pit_in=(9,), times={9: 100.0}), car("EEE", 80.5, start=30.0, compound="MEDIUM"))
    plans = [[["SOFT", x], ["MEDIUM", 15 - x]] for x in range(2, 14)] + [[["SOFT", 15]]]
    cfg = SimConfig(degradation=LIN, traffic=traffic, events={"name": "rate", "params": {"p_sc": 0.15, "p_vsc": 0.1}})
    b = Simulator(df, cfg).simulate_batch(snap(), plans, rules=RULES, n_samples=40, return_laps=True, seed=7)
    h = hashlib.sha256()
    for a in (b.times, b.pit_loss, b.lap_times, b.traffic_s, b.passes, *b.stint_seconds):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# computed with ghost_digest on origin/main 3c8b1a8 (#27, before bunching)
GHOST_MAIN_DIGEST = "ee500eacb2ddca1e597cff06f9add58ef490e975a6b69899b651fa9e5c4e375e"
NONE_MAIN_DIGEST = "9b70c51f5437003c63911c138eae1e57e7a30372f18d2557ca086e281c2081d5"


def test_sc_bunching_off_and_traffic_none_digests_equal_main():
    # 6 and 7
    assert ghost_digest({"name": "ghost", "params": {"sc_bunching": False}}) == GHOST_MAIN_DIGEST
    assert ghost_digest({"name": "none"}) == NONE_MAIN_DIGEST
    assert ghost_digest({"name": "ghost"}) == GHOST_MAIN_DIGEST   # bunching is off by default
    assert ghost_digest({"name": "ghost", "params": {"sc_bunching": True}}) != GHOST_MAIN_DIGEST
    assert GhostTraffic().params.sc_bunching is False


def test_bunching_flows_into_the_batch_and_the_result():
    df = frame(sc_ghost("AAA", 20.0), sc_ghost("BBB", -10.0))
    s = snap(4, 4 * F)
    cfg = SimConfig(degradation=LIN, events={"name": "fixed", "params": {"sc_laps": list(SC_LAPS)}})
    plans = [[["SOFT", 16]]]
    sim = Simulator(df, cfg.model_copy(update={"traffic": ComponentSpec(name="ghost", params={"sc_bunching": True})}))
    on = sim.simulate_batch(s, plans, rules=RULES, return_laps=True)
    off = Simulator(df, cfg).simulate_batch(s, plans, rules=RULES, return_laps=True)
    assert on.bunching_s[0, 0] < -10.0   # about 18 s behind AAA closed to the queue
    assert on.times[0, 0] - off.times[0, 0] == pytest.approx(on.traffic_s[0, 0] + on.bunching_s[0, 0])
    assert on.result(0, s).bunching_s == pytest.approx(on.bunching_s[0].mean())
    assert off.result(0, s).bunching_s is None and np.all(off.bunching_s == 0.0)
    # the SC laps run at the field's 110 s: the delta to the engine's SC lap, per SC lap
    engine_sc = off.lap_times[0, :2]
    assert on.sc_field_delta[0, :2] == pytest.approx(110.0 - engine_sc) and np.all(np.isnan(on.sc_field_delta[0, 2:]))
    assert on.result(0, s).sc_field_delta_s[:2] == pytest.approx((110.0 - engine_sc).tolist())
    assert off.result(0, s).sc_field_delta_s is None
    assert sim.simulate_batch(s, plans, rules=RULES).sc_field_delta is None   # sample 0 only with return_laps


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
    table, summaries = calibrate.traffic_report(df, RACE, SimConfig(degradation=LIN), {OUR: 0.0}, None,
                                                dirty_air=False, bunching=False)
    got = table.groupby(["table", "traffic"]).size()
    assert set(got.index) == {(t, a) for t in calibrate.TRAFFIC_TABLES for a in ("none", "ghost")}
    assert set(summaries) == {"none", "ghost"}
    assert table.groupby("traffic")["config_hash"].nunique().tolist() == [1, 1]
    assert summaries["ghost"]["rejoin"]["n_stops"] == 1


def test_traffic_report_dirty_air_ablation_uses_the_clean_pack_residual():
    from src.sim import calibrate
    from src.sim.data import gap_columns
    df = gap_columns(rejoin_frame().assign(team="T"))   # AAA runs close: some clean laps in the pack
    table, summaries = calibrate.traffic_report(df, RACE, SimConfig(degradation=LIN), {OUR: 0.0}, None)
    assert set(summaries) == {"none", "ghost", "ghost_bunching", "ghost_dirty"}
    pack = summaries["ghost"]["by_kind_clean_pack_mean_s"]
    assert summaries["ghost_dirty"]["dirty_air_s"] == (round(pack, 3) if pack > 0 else 0.0)
    assert summaries["ghost"]["dirty_air_s"] == 0.0
    assert table.groupby("traffic")["config_hash"].nunique().to_dict() == {
        "ghost": 1, "ghost_bunching": 1, "ghost_dirty": 1, "none": 1}
    assert summaries["ghost"]["sc_bunching"] is False and summaries["ghost_bunching"]["sc_bunching"] is True
    # traffic arms measure the drivers-table offset on free-air laps, and every row says so
    assert table.groupby("traffic")["pace_offset_def"].unique().map(list).to_dict() == {
        "none": ["all_laps"], "ghost": ["free_air"], "ghost_bunching": ["free_air"], "ghost_dirty": ["free_air"]}
    assert {k: summaries[k]["pace_offset_def"] for k in summaries} == {
        "none": "all_laps", "ghost": "free_air", "ghost_bunching": "free_air", "ghost_dirty": "free_air"}


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
