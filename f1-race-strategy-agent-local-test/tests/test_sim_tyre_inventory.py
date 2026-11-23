"""SimConfig.tyre_inventory: the as-of tyre sets at a snapshot (src/sim/tyre_inventory.py)."""
import pandas as pd
import pytest

from src.sim import SimConfig, calibration, tyre_inventory
from src.sim.config import world_default


def test_switch_is_off_by_default_and_keeps_the_hashes():
    assert SimConfig().tyre_inventory is False
    on = SimConfig(tyre_inventory=True)
    assert calibration.config_hash(SimConfig()) == calibration.config_hash(SimConfig().model_copy())
    assert calibration.config_hash(on) != calibration.config_hash(SimConfig())
    assert not tyre_inventory.on(SimConfig(), world_default(SimConfig()), None)
    assert tyre_inventory.on(None, on)


def test_phantom_sets_are_counted_per_compound():
    notes = ["FP1: HARD at TyreLife 11 matches no held set, created HARD-1",
             "FP2: SOFT at TyreLife 5 matches no held set, created SOFT-3",
             "FP1: HARD at TyreLife 10 matches no held set, created HARD-2",
             "SOFT: 8 sets seen, allocation 6"]
    assert tyre_inventory.phantoms(notes) == {"HARD": 2, "SOFT": 1}


def _laps(rows):
    df = pd.DataFrame(rows, columns=["Driver", "LapNumber", "Time"])
    df["Time"] = pd.to_timedelta(df["Time"], unit="s")
    return df


def test_visible_race_laps_end_at_the_drivers_lap():
    race = _laps([("A", 1, 100), ("A", 2, 190), ("A", 3, 280), ("B", 1, 101), ("B", 2, 192), ("B", 3, 283)])
    got = tyre_inventory.visible_race_laps(race, "A", 2)
    assert sorted(zip(got["Driver"], got["LapNumber"])) == [("A", 1), ("A", 2), ("B", 1)]
    start = tyre_inventory.visible_race_laps(race, "A", 0)
    assert sorted(start["LapNumber"]) == [1, 1]


def test_2019_stays_unconstrained():
    assert tyre_inventory.weekend("2019_10_united_kingdom") is None
    assert tyre_inventory.sets_at("2019_10_united_kingdom", "HAM", 10) is None
