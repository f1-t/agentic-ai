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


# --- the near-lapped rule (ScQueue.held_behind_laps) ---------------------------------------------
def _field(paces, sc=(4, 5), laps=8, behind=None):
    """Rows of cars lapping at constant pace, `behind[d]` seconds after the start; status 4 on the laps `sc`."""
    rows = []
    for d, t in paces.items():
        o = (behind or {}).get(d, 0.0)
        for L in range(1, laps + 1):
            rows.append({"driver": d, "lap": L, "start_s": o + (L - 1) * t, "end_s": o + L * t, "lap_time_s": t,
                         "pit_in": False, "pit_out": False, "track_status": "4" if L in sc else "1"})
    return pd.DataFrame(rows)


def test_deployment_behind_reads_the_crossings_before_the_safety_car():
    # A and C 90 s a lap, C 135 s (1.5 laps) behind; B 100 s a lap (30 s behind A after lap 3)
    f = _field({"A": 90.0, "B": 100.0, "C": 90.0}, behind={"C": 135.0})
    b = sc_queue.deployment_behind(f, 4, 90.0)
    assert b["A"] == pytest.approx(0.0)
    assert b["B"] == pytest.approx(30.0 / 90.0)
    assert b["C"] == pytest.approx(1.5)
    assert sc_queue.deployment_behind(f[f["lap"] <= 1], 4, 90.0) is None   # the rows end before it


def test_held_car_keeps_its_gap_to_the_train_and_is_not_queued():
    cum = _cum({"A": 90.0, "B": 91.0, "L": 120.0})   # L: 90 s behind at lap 3, one lap: not held
    kw = dict(spacing_s=0.5, lap_ref_s=90.0, lengths={(4, 5): 250.0})
    old = sc_queue.apply(cum, _nopit(cum), [(4, 5)], **kw)
    new = sc_queue.apply(cum, _nopit(cum), [(4, 5)], held_laps=1.25, **kw)
    assert old["L"][5] - old["B"][5] == pytest.approx(0.5)
    assert new["L"][5] == pytest.approx(old["L"][5])
    far = _cum({"A": 90.0, "B": 91.0, "H": 130.0})   # H: 120 s behind at lap 3, 1.33 laps: held
    old = sc_queue.apply(far, _nopit(far), [(4, 5)], **kw)
    new = sc_queue.apply(far, _nopit(far), [(4, 5)], held_laps=1.25, **kw)
    assert old["H"][5] - old["B"][5] == pytest.approx(0.5)   # the old rule pulls H into the train
    # held: its gap at lap 3 to B, the train's last car (390 - 273 = 117 s), kept behind B's new time
    assert new["H"][5] - new["B"][5] == pytest.approx(117.0)
    assert new["B"][5] == pytest.approx(old["B"][5])
    rec = sc_queue.apply(cum, _nopit(cum), [(4, 5)], held_laps=1.25,
                         behind={(4, 5): pd.Series({"A": 0.0, "B": 1.4, "L": 0.9})}, **kw)
    assert rec["B"][5] - rec["A"][5] == pytest.approx(3.0)   # the recorded distance decides: B held
    assert rec["L"][5] - rec["A"][5] == pytest.approx(0.5)


def test_single_car_queue_holds_a_car_lapped_at_the_deployment():
    f = _field({"A": 90.0, "B": 91.0, "C": 90.0}, behind={"C": 135.0})
    pit = np.zeros((1, 8))
    lap_c = np.full((1, 8), 90.0)
    q = {h: sc_queue.FieldQueue(f, "C", horizon=8, spacing_s=0.5, lap_ref_s=90.0, recorded=True, held_laps=h)
         for h in (None, 1.25)}
    # 1.5 laps behind at the deployment: held, its gap to B (the train's last car) kept. Keys at lap 3
    # (green times from the crossings before the deployment): A 270, B 273, C 135 + 270 = 405
    end5 = {h: np.cumsum(lap_c, axis=1)[0, 4] + 135.0 + q[h].shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5)[0] for h in q}
    length = 450.0 - 270.0                                           # A's recorded laps 4 and 5
    assert end5[None] == pytest.approx(270.0 + length + 1.0)         # queued behind A and B
    assert end5[1.25] == pytest.approx(270.0 + length + 0.5 + (405.0 - 273.0))
    # B, 0.03 laps behind, is queued as before; the held C was never a car ahead of it
    lap_b = np.full((1, 8), 91.0)
    qb = {h: sc_queue.FieldQueue(f, "B", horizon=8, spacing_s=0.5, lap_ref_s=90.0, recorded=True, held_laps=h)
          for h in (None, 1.25)}
    assert qb[1.25].shift(4, 5, lap_b, pit, 1, 0.0, 20.0, 0.5) == pytest.approx(
        qb[None].shift(4, 5, lap_b, pit, 1, 0.0, 20.0, 0.5))
    # the belief: rows cut before the deployment, so the distance comes from the plan's time at lap 3
    w = f[f["lap"] <= 2]
    qw = {h: sc_queue.FieldQueue(w, "C", horizon=2, spacing_s=0.5, lap_ref_s=90.0, held_laps=h) for h in (None, 1.25)}
    sw = {h: qw[h].shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5)[0] for h in qw}
    # B projected to lap 3 from its last shown lap (182 + 90 = 272): C keeps 405 - 272 behind it, one
    # spacing less than the queued place behind A and B
    assert sw[1.25] - sw[None] == pytest.approx((405.0 - 272.0) - 0.5)


def test_the_queue_orders_the_others_by_their_place_at_the_deployment():
    """X was 54 s behind A when the safety car came out, but its recorded crossing of lap 2 was made
    under it (264 s, 30 s slow). The simulated car D, 70 s behind at green pace, is behind X in the
    queue under the near-lapped rule (X's green time at lap 2: 144 + 90 = 234), ahead of it before."""
    def row(d, L, start, end, st="1"):
        return {"driver": d, "lap": L, "start_s": start, "end_s": end, "lap_time_s": end - start,
                "pit_in": False, "pit_out": False, "track_status": st}
    rows = [row("A", 1, 0, 90), row("A", 2, 90, 180), row("A", 3, 180, 300, "4"), row("A", 4, 300, 420, "4"),
            row("X", 1, 54, 144), row("X", 2, 144, 264, "14"), row("X", 3, 264, 384, "4"), row("X", 4, 384, 424, "4"),
            row("D", 1, 0, 160), row("D", 2, 160, 250), row("D", 3, 250, 370, "4"), row("D", 4, 370, 425, "4")]
    f = pd.DataFrame(rows)
    lap_d = np.array([[160.0, 90.0, 120.0, 55.0, 90.0, 90.0]])
    pit = np.zeros((1, 6))
    sh = {h: sc_queue.FieldQueue(f, "D", horizon=6, spacing_s=0.5, lap_ref_s=90.0, recorded=True,
                                 held_laps=h).shift(3, 4, lap_d, pit, 1, 0.0, 20.0, 0.5)[0] for h in (None, 1.25)}
    assert sh[1.25] - sh[None] == pytest.approx(0.5)   # one more car ahead of it in the queue


def test_a_car_that_retired_is_not_in_the_queue():
    """R stopped after lap 1; the rule leaves it out of the field instead of projecting it forward at
    lap_ref_s (which would make it the first car)."""
    f = _field({"A": 90.0, "B": 91.0, "C": 92.0})
    r = _field({"R": 80.0}).query("lap == 1")
    lap_c = np.full((1, 8), 92.0)
    pit = np.zeros((1, 8))
    with_r = sc_queue.FieldQueue(pd.concat([f, r]), "C", horizon=8, spacing_s=0.5, lap_ref_s=70.0, recorded=True,
                                 held_laps=1.25).shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5)
    without = sc_queue.FieldQueue(f, "C", horizon=8, spacing_s=0.5, lap_ref_s=70.0, recorded=True,
                                  held_laps=1.25).shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5)
    assert with_r == pytest.approx(without)


def test_placement_by_the_record_removes_the_drift_of_the_simulated_clock():
    """C really ran 92 s laps (third at lap 3: 276 s behind A's 270, B's 273). Its simulated clock
    drifted to 89 s laps (267 s at lap 3): by the clock it leads the train; by the record (placement
    "record") it is third, and a plan pitting at lap 2 moves by its time difference to the real one."""
    f = _field({"A": 90.0, "B": 91.0, "C": 92.0})
    lap_c = np.full((2, 8), 89.0)
    pit = np.zeros((2, 8))
    pit[1, 1] = 20.0                                   # the second plan stops at the end of lap 2
    q = {p: sc_queue.FieldQueue(f, "C", horizon=8, spacing_s=0.5, lap_ref_s=90.0, recorded=True, held_laps=1.25,
                                placement=p) for p in ("clock", "record")}
    end5 = {p: np.cumsum(lap_c + pit, axis=1)[:, 4] + q[p].shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5) for p in q}
    length = 450.0 - 270.0
    assert end5["clock"][0] == pytest.approx(267.0 + length)              # first by its clock
    assert end5["record"][0] == pytest.approx(270.0 + length + 1.0)       # third by the record
    assert end5["record"][1] == pytest.approx(270.0 + length + 1.0)       # 296 s at lap 3: still third, last
    # the belief (recorded False) is never placed by the record
    qb = sc_queue.FieldQueue(f, "C", horizon=8, spacing_s=0.5, lap_ref_s=90.0, held_laps=1.25, placement="record")
    qc = sc_queue.FieldQueue(f, "C", horizon=8, spacing_s=0.5, lap_ref_s=90.0, held_laps=1.25)
    assert qb.shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5) == pytest.approx(qc.shift(4, 5, lap_c, pit, 1, 0.0, 20.0, 0.5))
