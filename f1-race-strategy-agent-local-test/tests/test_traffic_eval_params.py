"""The traffic params the eval turns on (traffic.py): earned passes, neutral alignment, own
pace, positions. Synthetic frames from test_sim_traffic, no cache."""
import numpy as np

from test_sim_traffic import F, apply, frame, ghost, prepare


def test_earned_passes_never_gain_time_and_swap_does():
    # a ghost 0.9 s ahead and 1 s a lap slower: swap passes it on the line and gains time
    df = frame(ghost("AAA", 81.0, 0.9))
    m, s, laps = prepare(df)
    inc, _, passes = apply(m, s, laps)
    assert inc.min() < 0 and passes[0] == 1
    m, s, laps = prepare(df, pass_mode="earned")
    inc, _, _ = apply(m, s, laps)
    assert inc.min() >= 0   # it drifts past at its own pace instead


def test_earned_attack_pays_dirty_air_until_the_pass():
    df = frame(ghost("AAA", 80.5, 0.8))
    m, s, laps = prepare(df, pass_mode="earned", dirty_air_s=0.3)
    inc, traffic_s, _ = apply(m, s, laps)
    assert traffic_s[0] > 0 and inc.min() >= 0   # dirty air while attacking, never a gain


def test_align_neutral_keeps_the_gap_through_a_recorded_sc():
    # the ghost's record has an SC on laps 8-9 (130 s laps); our sample runs them green
    sc = {8: 130.0, 9: 130.0}
    df = frame(ghost("AAA", 80.0, 3.0, times=sc))
    df.loc[df["lap"].isin([8, 9]), "track_status"] = "4"   # the record ran an SC there
    m, s, laps = prepare(df, align_neutral=True)
    inc, traffic_s, _ = apply(m, s, laps)
    assert np.all(inc == 0)   # the gap stays 3 s: never caught
    m, s, laps = prepare(df)
    inc, traffic_s, _ = apply(m, s, laps)
    assert traffic_s[0] > 0   # without alignment we run into the SC-slowed ghost


def test_own_pace_moves_catching_not_free_lap_times():
    df = frame(ghost("AAA", 80.0, 3.0))
    m, s, laps = prepare(df, own_pace=True)
    s.own_offset_s = -1.0   # a second a lap quicker than the model: we catch AAA
    inc, traffic_s, _ = apply(m, s, laps)
    assert traffic_s[0] != 0
    s.own_offset_s = 1.0    # slower: we never catch it, and the lap times are the model's
    inc, _, _ = apply(m, s, laps)
    assert np.all(inc == 0)


def test_positions_after_the_out_lap_and_at_the_flag():
    df = frame(ghost("AAA", 80.0, 3.0), ghost("BBB", 80.0, 30.0))
    m, s, laps = prepare(df)
    pit = np.zeros((1, len(laps)))
    pit[0, 4] = 20.0   # stop at the end of lap 10: AAA (3 s ahead) stays ahead, BBB (30 s) too
    apply(m, s, laps, pit=pit)
    assert s.positions["rejoin_pos"][0] == 3 and s.positions["final_pos"][0] == 3
    assert s.positions["pit_window_cars"][0] == 0
    apply(m, s, laps)
    assert np.isnan(s.positions["rejoin_pos"][0]) and s.positions["final_pos"][0] == 3
    assert F == 80.0
