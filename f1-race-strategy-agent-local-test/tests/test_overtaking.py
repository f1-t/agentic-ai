"""Circuit-specific passing (src/sim/overtaking.py) and the ghost traffic's hazard pass rule
(traffic.GhostParams.difficulty). Synthetic frames from test_sim_traffic and the committed table;
no cache."""
import numpy as np
import pandas as pd
import pytest

from src.sim import overtaking
from src.sim.overtaking import Difficulty, difficulty_for, difficulty_prior
from src.sim.race_ids import parse_race_id
from src.sim.traffic import GhostTraffic
from test_sim_traffic import apply, frame, ghost, prepare


def test_the_difficulty_configs_are_the_traffic_configs_plus_difficulty_only():
    from src.sim import SimConfig
    from src.sim.config import ComponentSpec
    root = overtaking.ROOT / "configs" / "sim"
    for on, off, mode in (("world_difficulty.yaml", "world.yaml", "circuit"),
                          ("belief_difficulty.yaml", "belief_traffic.yaml", "asof")):
        a, b = SimConfig.from_file(root / on), SimConfig.from_file(root / off)
        params = dict(a.traffic.params)
        assert params.pop("difficulty") == mode
        assert a.model_copy(update={"traffic": ComponentSpec(name=a.traffic.name, params=params)}) == b


def test_expert_regret_is_the_real_plan_less_the_judges_best_legal_plan():
    from src.sim import fidelity
    b = pd.DataFrame({"driver": ["A"] * 3, "k": [5] * 3, "plan": ["real", "stay", "1:3:SOFT"], "valid": [True] * 3,
                      "legal_w": [True, True, False], "pred_w_s": [100.0, 90.0, 50.0],
                      "err_s": [0.0] * 3, "err_green_s": [0.0] * 3})
    assert fidelity.expert_regrets(b).tolist() == [10.0]
    stats = fidelity.race_stats({"b_rows": b})
    assert stats["E_real_regret_s"] == (10.0, 1) and stats["E_real_over_20s"] == (0.0, 1)


def _fixed(theta, slope=1.2):
    return Difficulty(0.1, 0.05, 0.2, theta, 0.3, 0.4, 1, (), 0, slope)


@pytest.fixture
def theta(monkeypatch):
    """Set the table's threshold for the synthetic race."""
    def set_(value):
        monkeypatch.setattr(overtaking, "difficulty_for", lambda rid, mode: _fixed(value))
    return set_


def test_difficulty_none_is_the_earned_rule_bit_for_bit():
    df = frame(ghost("AAA", 80.6, 0.9), ghost("BBB", 81.5, 6.0))
    m0, s0, laps = prepare(df, pass_mode="earned", dirty_air_s=0.3)
    m1, s1, _ = prepare(df, pass_mode="earned", dirty_air_s=0.3, difficulty="none")
    a, b = apply(m0, s0, laps), apply(m1, s1, laps)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_a_hard_circuit_holds_a_faster_car_longer_and_never_gains_time(theta):
    # a ghost 0.9 s ahead, 0.6 s a lap slower than us
    df = frame(ghost("AAA", 80.6, 0.9))
    out = {}
    for th in (0.3, 1.0, 6.0):
        theta(th)
        m, s, laps = prepare(df, pass_mode="earned", difficulty="circuit")
        inc, traffic_s, passes = apply(m, s, laps)
        assert inc.min() >= 0   # no swap: a pass never conjures time
        out[th] = (float(traffic_s[0]), int(passes[0]))
    assert out[0.3][0] < out[1.0][0] < out[6.0][0]
    assert out[0.3][1] == 1 and out[6.0][1] == 0   # stuck to the flag where passing is that hard


def test_the_projected_source_refuses_a_difficulty_read_from_the_race():
    for mode in ("circuit", "race"):
        with pytest.raises(ValueError):
            GhostTraffic(source="projected", difficulty=mode)
    assert not GhostTraffic(source="projected", difficulty="asof").reads_future
    assert GhostTraffic(source="recorded", difficulty="none").reads_future


def test_the_table_never_holds_a_held_out_race():
    t = overtaking.load_table()
    assert not set(t["race_id"]) & overtaking.held_out()


def test_the_as_of_prior_reads_only_earlier_races():
    t = overtaking.load_table()
    for rid in ("2021_04_spain", "2022_08_azerbaijan", "2024_07_emilia_romagna"):
        d = difficulty_prior(rid)
        key = parse_race_id(rid)[:2]
        assert d.sources and all(parse_race_id(r)[:2] < key for r in d.sources)
        assert set(overtaking.races_for(rid)) == set(d.sources)
    assert difficulty_prior(t["race_id"].min()).tier == 3 or len(difficulty_prior(t["race_id"].min()).sources) == 0


def test_an_unknown_race_takes_the_wide_a_priori_value():
    d = difficulty_for("syn_traffic", "asof")
    assert d.tier == 3 and d.theta_s == overtaking.PRIOR_THETA_S and d.rate_hi > 3 * d.rate_lo


def test_a_held_out_race_is_priced_from_other_races_only():
    # never loaded: the table has no row for it, so 'circuit' reads the circuit's other races
    d = difficulty_for("2019_06_monaco", "circuit")
    assert "2019_06_monaco" not in d.sources and d.tier == 1


def _race(rows):
    df = pd.DataFrame(rows)
    df["s1_s"] = np.nan
    df["s2_s"] = np.nan
    return df


def test_a_pass_counts_and_a_pit_stop_does_not():
    # BBB starts 0.5 s behind AAA every lap and is 1 s faster on lap 4 (a pass); on lap 7 AAA
    # pits (in-lap): BBB goes by, which is not an on-track pass
    from test_sim_traffic import car
    a = car("AAA", 90.0, start=0.0, pit_in=(7,), n=10)
    b = car("BBB", 90.0, start=0.5, n=10, times={4: 88.0})
    for r in a:
        if r["lap"] == 7:
            r["lap_time_s"] += 20.0
    # rebuild AAA's clock with the in-lap
    t = 0.0
    for r in a:
        r["start_s"], r["end_s"] = t, t + r["lap_time_s"]
        t = r["end_s"]
    df = _race(a + b)
    s, c = overtaking.observe_race("2021_04_spain", df=df)
    assert s["n_pass"] == 1 and c.loc[c["passed"], "lap"].tolist() == [4]
    assert 7 not in c["lap"].tolist()
