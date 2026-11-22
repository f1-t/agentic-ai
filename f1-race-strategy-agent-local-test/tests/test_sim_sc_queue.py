"""The safety-car queue (src/sim/sc_queue.py) on hand-built fields."""
import numpy as np
import pandas as pd
import pytest

from src.sim import sc_queue


def _cum(times_per_lap: dict, laps=10):
    return {d: pd.Series(np.cumsum(np.full(laps, t)), index=np.arange(1, laps + 1)) for d, t in times_per_lap.items()}


def _nopit(cum):
    return {d: pd.Series(0.0, index=s.index) for d, s in cum.items()}


def test_periods_from_majority_status():
    f = pd.DataFrame({"lap": [1, 1, 2, 2, 3, 3, 4, 4, 6, 6], "driver": list("ABABABABAB"),
                      "track_status": ["1", "1", "4", "4", "41", "4", "1", "4", "4", "4"]})
    assert sc_queue.sc_periods(f) == [(2, 3), (6, 6)]


def test_queue_closes_gaps_keeps_order_and_shifts_later_laps():
    cum = _cum({"A": 90.0, "B": 91.0, "C": 95.0})
    out = sc_queue.apply(cum, _nopit(cum), [(4, 5)], spacing_s=0.5, lap_ref_s=90.0)
    at5 = {d: out[d][5] for d in "ABC"}
    assert at5["A"] == pytest.approx(450.0)
    assert at5["B"] - at5["A"] == pytest.approx(0.5)   # 3 s gap at lap 3, closed to the spacing
    assert at5["C"] - at5["B"] == pytest.approx(0.5)
    assert out["B"][3] == pytest.approx(cum["B"][3])     # before the period untouched
    assert out["C"][10] - out["C"][5] == pytest.approx(cum["C"][10] - cum["C"][5])   # racing resumes
    assert cum["C"][5] == pytest.approx(475.0)           # the input is not modified


def test_a_stop_under_the_safety_car_costs_places_not_seconds():
    cum = _cum({"A": 90.0, "B": 90.5, "C": 91.0})
    pit = _nopit(cum)
    pit["A"].loc[4] = 10.0                                # A pits under it: 10 s behind the order of lap 3
    cum["A"] = cum["A"] + pit["A"].cumsum()
    out = sc_queue.apply(cum, pit, [(4, 5)], spacing_s=0.5, lap_ref_s=90.0)
    order = sorted("ABC", key=lambda d: out[d][5])
    assert order == ["B", "C", "A"]
    assert out["A"][5] - out["B"][5] == pytest.approx(1.0)   # two places, two spacings


def test_recorded_length_sets_the_leader_lapped_cars_unlap_unless_they_stayed_down():
    cum = _cum({"A": 90.0, "B": 91.0, "L": 120.0, "M": 121.0})   # L and M are a lap down by lap 3
    out = sc_queue.apply(cum, _nopit(cum), [(4, 5)], spacing_s=0.5, lengths={(4, 5): 250.0}, lapped={"M"})
    assert out["A"][5] == pytest.approx(270.0 + 250.0)
    assert out["B"][5] - out["A"][5] == pytest.approx(0.5)
    assert out["L"][5] - out["B"][5] == pytest.approx(0.5)   # waved past: joins the queue
    assert out["M"][5] == pytest.approx(cum["M"][5])          # finished laps down: untouched


def test_single_car_queue_against_a_recorded_field():
    """FieldQueue: the car is placed among the other cars' recorded times by its own time before
    the period plus its stop under it, and lands spacing_s per car ahead behind the first."""
    laps = np.arange(1, 9)
    rows = []
    for d, t in (("A", 90.0), ("B", 91.0), ("C", 92.0)):
        for L in laps:
            sc = L in (4, 5)
            rows.append({"driver": d, "lap": L, "start_s": (L - 1) * t, "end_s": L * t, "pit_in": False,
                         "track_status": "4" if sc else "1"})
    q = sc_queue.FieldQueue(pd.DataFrame(rows), "C", horizon=8, spacing_s=0.5, lap_ref_s=90.0, recorded=True)
    # the car C from lap 0 at 92 s a lap, one plan staying out, one pitting on lap 4 (10 s under the SC)
    lap_s = np.full((2, 8), 92.0)
    pit = np.zeros((2, 8))
    pit[1, 3] = 10.0
    sh = q.shift(4, 5, lap_s, pit, 1, 0.0, 20.0, 0.5)
    end5 = np.cumsum(lap_s + pit, axis=1)[:, 4] + sh
    assert end5[0] == pytest.approx(270.0 + 180.0 + 1.0)   # behind A and B: A's lap 3 end + recorded length + 2 x 0.5
    assert end5[1] == pytest.approx(end5[0])                 # the stop under the SC cost no place here (last anyway)
