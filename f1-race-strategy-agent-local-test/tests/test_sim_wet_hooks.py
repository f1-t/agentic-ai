"""Wet-weather hooks: compound classes, rule waivers, the track-state seam. Machinery only."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import Plan, Snapshot, StintRecord
from src.sim import SimConfig, Simulator
from src.sim.config import DEFAULT_COMPOUND_CLASSES, class_names, compound_class
from src.sim.rules import RuleContext, check
from src.sim.track import TRACK

from test_sim import RACE, start, synthetic_frame

LIN = {"name": "linear"}


@TRACK.register("test_double")
class DoubleTrack:
    """Test-only: doubles the lap time on the configured race laps."""

    def __init__(self, laps=()):
        self.laps = list(laps)

    def draw(self, snapshot, rng):
        return None

    def factor(self, state, class_codes, laps):
        return np.where(np.isin(laps, self.laps), 2.0, 1.0)


SEEN = {}


@TRACK.register("test_spy")
class SpyTrack:
    def draw(self, snapshot, rng):
        return None

    def factor(self, state, class_codes, laps):
        SEEN["codes"], SEEN["laps"] = class_codes.copy(), np.array(laps)
        return 1.0


def frame_with(compound="INTERMEDIATE"):
    rows = [{"compound": compound, "tyre_age": a, "lap": a + s, "clean": True, "lap_time_s": 95.0 + 0.05 * a}
            for s in range(0, 30, 3) for a in range(1, 21)]
    return pd.concat([synthetic_frame(), pd.DataFrame(rows)], ignore_index=True)


def ctx(plan, snap=None, cfg=None):
    return RuleContext(snap or start(), Plan.of(plan), {"SOFT", "MEDIUM", "INTERMEDIATE", "WET"}, cfg or SimConfig())


def test_class_defaults_and_override():
    cfg = SimConfig()
    assert cfg.compound_classes == DEFAULT_COMPOUND_CLASSES
    assert {compound_class(cfg, c) for c in ("SOFT", "MEDIUM", "HARD", "HYPERSOFT")} == {"SLICK"}
    assert compound_class(cfg, "INTERMEDIATE") == "INTERMEDIATE" and compound_class(cfg, "WET") == "WET"
    assert class_names(cfg)[0] == "SLICK"
    over = SimConfig(compound_classes={"SOFT": "SLICK", "X": "WET"})
    assert compound_class(over, "X") == "WET"


def test_unknown_compound_is_slick_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert compound_class(SimConfig(), "MYSTERY") == "SLICK"
    assert "MYSTERY" in caplog.text


def test_two_compounds_waived_with_intermediate_in_history():
    snap = Snapshot(race=RACE, driver="X", compound="SOFT", laps_completed=20, tyre_age=10,
                    stints=[StintRecord(compound="INTERMEDIATE", laps=10, start_lap=1)])
    assert check(["two_compounds"], ctx([["SOFT", 30]], snap)) == []


def test_two_compounds_waived_with_wet_stint_in_plan_and_fitted():
    assert check(["two_compounds"], ctx([["SOFT", 20], ["WET", 30]])) == []
    assert check(["two_compounds"], ctx([["INTERMEDIATE", 50]], start("INTERMEDIATE"))) == []


def test_two_compounds_still_enforced_for_slicks():
    assert check(["two_compounds"], ctx([["SOFT", 50]]))
    assert check(["two_compounds"], ctx([["SOFT", 25], ["SOFT", 25]]))


def test_min_stint_class_waiver():
    snap = start("INTERMEDIATE")
    assert check(["min_stint"], ctx([["INTERMEDIATE", 3], ["SOFT", 47]], snap)) == []
    assert check(["min_stint"], ctx([["SOFT", 3], ["MEDIUM", 47]])) != []
    assert check(["min_stint"], ctx([["SOFT", 30], ["INTERMEDIATE", 3], ["SOFT", 17]])) == []


def test_min_stint_waiver_end_to_end():
    sim = Simulator(frame_with(), SimConfig(degradation=LIN))
    res = sim.simulate(start("INTERMEDIATE"), [["INTERMEDIATE", 3], ["SOFT", 47]])
    assert res.valid, res.errors


def test_dry_track_changes_nothing():
    plans = [[["SOFT", 20], ["MEDIUM", 30]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]]
    kw = dict(seed=7, n_samples=20)
    base = SimConfig(degradation=LIN, hazard={"name": "age_logistic"})
    a = Simulator(synthetic_frame(), base).simulate_batch(start(), plans, **kw)
    b = Simulator(synthetic_frame(), SimConfig(
        degradation=LIN, hazard={"name": "age_logistic"}, track={"name": "dry"})).simulate_batch(start(), plans, **kw)
    assert a.times.tobytes() == b.times.tobytes()


def test_track_stub_doubles_exactly_the_configured_laps():
    plan = [["SOFT", 20], ["MEDIUM", 30]]
    base = Simulator(synthetic_frame(), SimConfig(degradation=LIN)).simulate(start(), plan, return_laps=True)
    dbl = Simulator(synthetic_frame(), SimConfig(
        degradation=LIN, track={"name": "test_double", "params": {"laps": [5, 30]}}))
    res = dbl.simulate(start(), plan, return_laps=True)
    want = np.array(base.lap_times_s)
    for lap in (5, 30):
        want[lap - 1] *= 2
    assert np.array(res.lap_times_s) == pytest.approx(want)


def test_track_factor_sees_class_codes():
    sim = Simulator(frame_with(), SimConfig(degradation=LIN, track={"name": "test_spy"}))
    sim.simulate(start("INTERMEDIATE"), [["INTERMEDIATE", 10], ["SOFT", 40]])
    names = class_names(sim.config)
    codes = SEEN["codes"]
    assert codes.dtype.kind == "i" and codes.shape == (1, 50)
    assert {names[c] for c in codes[0, :10]} == {"INTERMEDIATE"}
    assert {names[c] for c in codes[0, 10:]} == {"SLICK"}


def test_intermediate_plan_rejected_without_modelled_laps():
    sim = Simulator(synthetic_frame(), SimConfig(degradation=LIN))
    res = sim.simulate(start(), [["SOFT", 20], ["INTERMEDIATE", 30]])
    assert not res.valid and "Unknown compound 'INTERMEDIATE'" in res.errors[0]
