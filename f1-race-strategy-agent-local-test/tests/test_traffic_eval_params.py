"""The traffic params the eval turns on (traffic.py): earned passes, neutral alignment, own
pace, positions; the traffic switch's configs; the flag classification. Synthetic frames from
test_sim_traffic, no cache."""
import numpy as np

from src.contracts import Snapshot
from src.eval import score
from src.sim import SimConfig
from src.sim.config import ComponentSpec, world_default
from src.sim.traffic import flag_classification
from test_sim_traffic import F, RACE, apply, car, frame, ghost, prepare


def test_the_traffic_switch_is_the_default_plus_traffic_only():
    belief = SimConfig.from_file(score.BELIEF_CONFIG)
    belief_on = SimConfig.from_file(score.BELIEF_TRAFFIC_CONFIG)
    assert belief.traffic.name == "none" and belief_on.traffic.params["source"] == "projected"
    assert belief_on.model_copy(update={"traffic": ComponentSpec(name="none")}) == belief
    world = score.default_world_config(world_default(SimConfig()))
    world_on = SimConfig.from_file(score.WORLD_TRAFFIC_CONFIG)
    assert world.traffic.name == "none" and world_on.traffic.params.get("source", "recorded") == "recorded"
    assert world_on.model_copy(update={"traffic": ComponentSpec(name="none")}) == world


def test_a_lapped_finisher_is_classified_and_a_retirement_is_not():
    # laps 1-3 of the table; the leader crosses lap 3 at 300, a lapped car ends lap 2 at 310
    # (after the flag), a retirement ends lap 2 at 150
    G = np.array([[100.0, 200.0, 300.0], [110.0, 310.0, np.nan], [100.0, 150.0, np.nan]])
    assert flag_classification(G, 10.0).tolist() == [2, 1, -1]


def test_finishing_positions_rank_a_lapped_finisher_by_its_own_last_lap():
    # BBB a lap down finishes on lap 19 at 1620; our car finishes lap 20 at 400 + 1200 = 1600,
    # so it crossed lap 19 at about 1540: ahead of BBB. AAA finishes lap 20 at 1580: ahead of us
    df = frame(car("AAA", 79.0), car("BBB", 85.0, n=19))
    s = Snapshot(race=RACE, driver="XXX", laps_completed=5, compound="SOFT", tyre_age=5, elapsed_s=400.0,
                 standing_start=False)
    pos = score.finishing_positions(df, s, np.array([1200.0, 1100.0]), np.array([np.nan, np.nan]))
    assert pos.tolist() == [2, 1]
    dnf = score.finishing_positions(df, s, np.array([1200.0]), np.array([10.0]))
    assert dnf.tolist() == [3]   # behind both classified cars, BBB included


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
