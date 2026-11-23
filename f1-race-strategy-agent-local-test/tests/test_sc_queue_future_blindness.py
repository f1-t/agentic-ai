"""No future information in the belief's safety-car queue (sim/sc_queue.py, mode "single").

The belief places its car in the queue among the field visible at the snapshot (Simulator.window_rows,
cut by the clock), projected beyond it with the gaps held. Every other car's row that ended after
the snapshot driver ended lap K is altered (tests/test_traffic_future_blindness.alter_others_after),
and the belief's times under a safety car in progress at K and under a later one must come out
identical; the judge, which reads the recorded race, must not (the canary).
"""
import numpy as np
import pytest

from src.sim import SimConfig, Simulator, calibration
from src.sim.events import Scenario
from test_sim import RACE as SYN_RACE
from test_sim import synthetic_practice
from test_traffic_future_blindness import DRIVER, K, alter_others_after, field

QUEUE = {"sc_queue": {"enabled": True, "mode": "single"}}
BELIEF = SimConfig.from_file("configs/sim/belief.yaml").model_copy(update={"sc_queue": SimConfig.model_validate(QUEUE).sc_queue})
JUDGE = SimConfig.model_validate(QUEUE)
N = 4


@pytest.fixture(scope="module")
def pair():
    df = field()
    snap = calibration.snapshot_at(df, SYN_RACE, DRIVER, K)
    return df, alter_others_after(df), snap


def _plans(df, snap):
    rest = snap.remaining_laps
    return [calibration.remaining_plan(df, DRIVER, K), [["SOFT", 3], ["MEDIUM", rest - 3]],
            [["SOFT", 12], ["MEDIUM", rest - 12]]]


@pytest.mark.parametrize("sc", [[K - 1, K, K + 1, K + 2], [K + 8, K + 9, K + 10]])
def test_belief_queue_ignores_the_other_cars_after_the_cut(pair, sc):
    df, alt, snap = pair
    scen = Scenario(sc_laps=sc, vsc_laps=[])
    a, b = (Simulator(f, BELIEF, synthetic_practice()).simulate_batch(snap, _plans(df, snap), scenario=scen,
                                                                       n_samples=N, seed=3) for f in (df, alt))
    assert np.array_equal(a.times, b.times, equal_nan=True)


def test_the_queue_moves_the_belief(pair):
    df, _, snap = pair
    scen = Scenario(sc_laps=[K + 8, K + 9, K + 10], vsc_laps=[])
    off = BELIEF.model_copy(update={"sc_queue": SimConfig().sc_queue})
    a, b = (Simulator(df, c, synthetic_practice()).simulate_batch(snap, _plans(df, snap), scenario=scen,
                                                                   n_samples=N, seed=3) for c in (BELIEF, off))
    assert not np.allclose(a.times, b.times)


def test_the_judge_reads_the_recorded_field(pair):
    # the canary: the judge's queue reads the other cars' recorded race after the cut
    df, alt, snap = pair
    scen = Scenario(sc_laps=[K + 8, K + 9, K + 10], vsc_laps=[])
    a, b = (Simulator(f, JUDGE).simulate_batch(snap, _plans(df, snap), scenario=scen, n_samples=1)
            for f in (df, alt))
    assert not np.allclose(a.times, b.times)


HELD = BELIEF.model_copy(update={"sc_queue": SimConfig.model_validate(
    {"sc_queue": {"enabled": True, "mode": "single", "held_behind_laps": 1.25}}).sc_queue})


@pytest.mark.parametrize("sc", [[K - 1, K, K + 1, K + 2], [K + 8, K + 9, K + 10]])
def test_near_lapped_rule_ignores_the_other_cars_after_the_cut(pair, sc):
    # the near-lapped rule reads the deployment from the visible rows only (sc_queue.deployment_behind)
    df, alt, snap = pair
    scen = Scenario(sc_laps=sc, vsc_laps=[])
    a, b = (Simulator(f, HELD, synthetic_practice()).simulate_batch(snap, _plans(df, snap), scenario=scen,
                                                                     n_samples=N, seed=3) for f in (df, alt))
    assert np.array_equal(a.times, b.times, equal_nan=True)
