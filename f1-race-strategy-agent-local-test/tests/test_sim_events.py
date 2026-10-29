"""The "rate" SC/VSC event model and observed neutralised periods. Offline."""
import os

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.contracts import RaceParams, Snapshot
from src.sim import SimConfig, Simulator, calibration
from src.sim.config import EventEffects, EventModel
from src.sim.data import CACHE_DIR
from src.sim.degradation import DEGRADATION
from src.sim.events import EVENTS

RACE = RaceParams(race_id="syn", total_laps=50, pit_loss_s=20.0, min_stint_laps=8,
                  compounds=["SOFT", "MEDIUM"])
TRUE = {"SOFT": (80.0, 0.10), "MEDIUM": (82.0, 0.03)}
PLANS = [[["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]]


@DEGRADATION.register("test_events_noisy")
class NoisyModel:
    """A random offset per draw, so any shift in the rng stream shows in the times."""

    def __init__(self, inner, offset=0.0):
        self.inner, self.offset, self.compounds = inner, offset, inner.compounds

    @classmethod
    def fit(cls, data, **kw):
        return cls(DEGRADATION.get("linear").fit(data, **kw))

    def draw(self, rng):
        return NoisyModel(self.inner, float(rng.normal()))

    def lap_times(self, compound, ages, laps):
        return self.inner.lap_times(compound, ages, laps) + self.offset


def synthetic_frame():
    rows = []
    for comp, (base, slope) in TRUE.items():
        for start in range(0, 30, 3):
            for age in range(1, 21):
                rows.append({"compound": comp, "tyre_age": age, "lap": start + age, "clean": True,
                             "lap_time_s": base + slope * age})
    return pd.DataFrame(rows)


def make_sim(events=None, **extra):
    cfg = {"degradation": {"name": "test_events_noisy"}, "hazard": {"name": "age_logistic"}, **extra}
    if events is not None:
        cfg["events"] = events
    return Simulator(synthetic_frame(), SimConfig(**cfg))


def draws(model, snap, n, seed=0):
    return [model.draw(snap, np.random.default_rng([seed, k])) for k in range(n)]


def start(race=RACE):
    return Snapshot.at_start(race, "X", "SOFT")


# --- model -----------------------------------------------------------------------
def test_zero_rates_are_byte_identical_to_none():
    snaps = [start(), Snapshot(race=RACE, driver="X", laps_completed=12, compound="SOFT", tyre_age=12,
                               sc_laps=[11, 12], standing_start=False)]
    plans = {0: PLANS, 12: [[["SOFT", 8], ["MEDIUM", 30]], [["SOFT", 18], ["MEDIUM", 20]]]}
    none = make_sim()
    zero = make_sim({"name": "rate", "params": {"p_sc": 0.0, "p_vsc": 0.0}})
    for snap in snaps:
        a = none.simulate_batch(snap, plans[snap.laps_completed], seed=3, n_samples=40, return_laps=True)
        b = zero.simulate_batch(snap, plans[snap.laps_completed], seed=3, n_samples=40, return_laps=True)
        for name in ("times", "pit_loss", "lap_times", "dnf", "big_loss", "dnf_lap"):
            assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


def test_rate_model_changes_the_draws_when_on():
    none = make_sim().simulate_batch(start(), PLANS, n_samples=40)
    on = make_sim({"name": "rate", "params": {"p_sc": 0.05}}).simulate_batch(start(), PLANS, n_samples=40)
    assert not np.array_equal(none.times, on.times)


def test_expected_sc_laps_match_rate_times_duration():
    model = EVENTS.get("rate")(p_sc=0.015, d_sc=3, p_vsc=0.0)
    got = np.mean([len(s.sc_laps) for s in draws(model, start(), 2000)])
    assert got == pytest.approx(0.015 * 3 * RACE.total_laps, rel=0.15)
    vsc = EVENTS.get("rate")(p_sc=0.0, p_vsc=0.02, d_vsc=2)
    got = np.mean([len(s.vsc_laps) for s in draws(vsc, start(), 2000)])
    assert got == pytest.approx(0.02 * 2 * RACE.total_laps, rel=0.15)


def test_periods_never_overlap_and_stay_in_the_remaining_laps():
    snap = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10,
                    sc_laps=[4, 5], standing_start=False)
    model = EVENTS.get("rate")(p_sc=0.2, d_sc=3, p_vsc=0.2, d_vsc=2)
    seen_sc = seen_vsc = False
    for s in draws(model, snap, 300):
        sc, vsc = set(s.sc_laps), set(s.vsc_laps)
        assert len(sc) == len(s.sc_laps) and len(vsc) == len(s.vsc_laps)
        assert not sc & vsc
        assert all(11 <= lap <= RACE.total_laps for lap in sc | vsc)   # history is not redrawn
        seen_sc, seen_vsc = seen_sc or bool(sc), seen_vsc or bool(vsc)
    assert seen_sc and seen_vsc


def test_mean_period_length_is_d():
    model = EVENTS.get("rate")(p_sc=0.01, d_sc=4, p_vsc=0.0)
    lengths = []
    for s in draws(model, start(), 3000):
        laps = s.sc_laps
        if laps and laps[-1] < RACE.total_laps:   # periods cut by the flag are shorter
            run = 1
            for a, b in zip(laps, laps[1:]):
                if b == a + 1:
                    run += 1
                else:
                    lengths.append(run)
                    run = 1
            lengths.append(run)
    assert min(lengths) >= 1
    assert np.mean(lengths) == pytest.approx(4, rel=0.15)


def test_snapshot_inside_a_period_continues_it():
    for status, field in (("sc", "sc_laps"), ("vsc", "vsc_laps")):
        snap = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10,
                        track_status=status, standing_start=False, **{field: [9, 10]})
        model = EVENTS.get("rate")(p_sc=0.0, p_vsc=0.0, d_sc=3, d_vsc=2)
        scens = draws(model, snap, 2000)
        laps = [getattr(s, field) for s in scens]
        assert all(ls[0] == 11 and ls == list(range(11, 11 + len(ls))) for ls in laps)
        assert np.mean([len(ls) for ls in laps]) == pytest.approx(3 if status == "sc" else 2, rel=0.1)
        other = "vsc_laps" if field == "sc_laps" else "sc_laps"
        assert all(not getattr(s, other) for s in scens)


def test_race_rates_override_the_config():
    model = EVENTS.get("rate")(p_sc=0.3, p_vsc=0.3)
    quiet = RACE.model_copy(update={"sc_rate_per_lap": 0.0, "vsc_rate_per_lap": 0.0})
    assert all(not s.sc_laps and not s.vsc_laps for s in draws(model, start(quiet), 50))
    busy = RACE.model_copy(update={"sc_rate_per_lap": 0.2, "sc_mean_laps": 1.0, "vsc_rate_per_lap": 0.0})
    assert model.rates(busy) == (0.2, 1.0, 0.0, model.params.d_vsc)
    assert model.rates(RACE) == (0.3, 3.0, 0.3, 2.0)
    assert all(not s.vsc_laps for s in draws(model, start(busy), 50))


def test_event_model_params_are_checked():
    assert EventModel() == EventModel(p_sc=0.015, d_sc=3, p_vsc=0.02, d_vsc=2)
    for bad in ({"p_sc": 0.6, "p_vsc": 0.5}, {"d_sc": 0.5}, {"p_vsc": -0.1}, {"q": 1}):
        with pytest.raises(ValidationError):
            EventModel(**bad)
    with pytest.raises(ValidationError):
        RaceParams(race_id="x", total_laps=10, pit_loss_s=20, sc_mean_laps=0.5)


def test_common_random_numbers_two_plans_see_the_same_scenario():
    cfg = {"name": "rate", "params": {"p_sc": 0.08, "p_vsc": 0.08}}
    sim = make_sim(cfg, hazard={"name": "none"}, degradation={"name": "linear"})
    fx = EventEffects()
    sc_s = fx.sc_lap_factor * sim.ref_lap_s
    for seed in range(20):
        scen = sim.events.draw(start(), np.random.default_rng([seed, 0]))
        batch = sim.simulate_batch(start(), PLANS, seed=seed, return_laps=True)
        for row in batch.lap_times:
            assert np.flatnonzero(np.isclose(row, sc_s)).tolist() == [lap - 1 for lap in scen.sc_laps]
        # VSC laps: the 1.3 factor on the green lap, the same lap set for both plans
        green = make_sim(None, hazard={"name": "none"}, degradation={"name": "linear"}).simulate_batch(
            start(), PLANS, seed=seed, return_laps=True)
        for row, base in zip(batch.lap_times, green.lap_times):
            changed = set(np.flatnonzero(~np.isclose(row, base)).tolist())
            assert {lap - 1 for lap in scen.sc_laps + scen.vsc_laps} <= changed


# --- observed periods ------------------------------------------------------------
def _frame(status_by_lap, n_cars=4, odd=None):
    rows = [{"driver": f"D{c}", "lap": lap, "track_status": ts}
            for lap, ts in status_by_lap.items() for c in range(n_cars)]
    for lap, ts in (odd or {}).items():   # one car's lap carries a different status
        rows.append({"driver": "LATE", "lap": lap, "track_status": ts})
    return pd.DataFrame(rows)


def _messages(rows):
    return pd.DataFrame([{"Time": i, "Lap": lap,   # message order; FastF1 has timestamps
                          "Category": cat, "Status": status, "Message": msg}
                         for i, (lap, cat, status, msg) in enumerate(rows)])


def test_observed_periods_from_track_status():
    status = {1: "1", 2: "1", 3: "14", 4: "4", 5: "41", 6: "1", 7: "6", 8: "67", 9: "1", 10: "1"}
    assert calibration.observed_periods(_frame(status, odd={2: "124"})) == [("sc", 3, 5), ("vsc", 7, 8)]
    assert calibration.observed_periods(_frame({1: "1", 2: "1"})) == []


def test_observed_periods_from_messages_and_cross_check():
    msgs = _messages([(3, "SafetyCar", "DEPLOYED", "SAFETY CAR DEPLOYED"),
                      (5, "SafetyCar", "IN THIS LAP", "SAFETY CAR IN THIS LAP"),
                      (6, "Flag", None, "TRACK CLEAR"),
                      (7, "SafetyCar", "DEPLOYED", "VIRTUAL SAFETY CAR DEPLOYED"),
                      (8, "SafetyCar", "ENDING", "VIRTUAL SAFETY CAR ENDING"),
                      (10, "Flag", "None", "CHEQUERED FLAG")])
    assert calibration.observed_periods(messages=msgs) == [("sc", 3, 5), ("vsc", 7, 8)]
    status = {1: "1", 2: "1", 3: "4", 4: "4", 5: "4", 6: "1", 7: "6", 8: "7", 9: "1", 10: "1"}
    assert calibration.observed_periods(_frame(status), msgs) == [("sc", 3, 5), ("vsc", 7, 8)]
    status[9] = "7"
    with pytest.warns(UserWarning, match="differ from race control"):
        assert calibration.observed_periods(_frame(status), msgs)[-1] == ("vsc", 7, 9)
    still_out = _messages([(4, "SafetyCar", "DEPLOYED", "SAFETY CAR DEPLOYED"), (9, "Flag", None, "CHEQUERED")])
    assert calibration.observed_periods(messages=still_out) == [("sc", 4, 9)]
    with pytest.raises(ValueError):
        calibration.observed_periods()


@pytest.mark.skipif(not os.path.isdir(os.path.join(CACHE_DIR, "2021")), reason="no cache")
def test_observed_periods_on_spain():
    from src.sim.data import lap_frame, load_session
    session = load_session(2021, "Spain", messages=True)
    frame = lap_frame(session.laps)
    assert calibration.observed_periods(frame) == [("sc", 8, 10)]
    assert calibration.observed_periods(messages=session.race_control_messages) == [("sc", 8, 10)]
    assert calibration.observed_periods(frame, session.race_control_messages) == [("sc", 8, 10)]
