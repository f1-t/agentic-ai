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
    assert not tyre_inventory.on(SimConfig(), None) and tyre_inventory.on(world_default(SimConfig()))
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


def test_2019_is_open_on_new_sets_but_holds_its_used_ones():
    sessions, race, allocation = tyre_inventory.weekend("2019_10_united_kingdom")
    assert allocation == tyre_inventory.ALLOCATION_OPEN
    sets = tyre_inventory.sets_at("2019_10_united_kingdom", "HAM", 10)
    assert sum(s["laps_run"] == 0 for s in sets) >= 20 and any(s["laps_run"] > 0 for s in sets)


def test_a_synthetic_race_stays_unconstrained():
    assert tyre_inventory.weekend("syn") is None and tyre_inventory.weekend("2099_01_synthetic") is None


def test_the_agent_prompt_states_the_sets_where_the_inventory_is_on():
    from src import agent
    assert agent.PRERACE_ANCHOR in agent.SYSTEM_PROMPT_V2FREE
    assert "sets_left" in agent.TYRE_SETS_SENTENCE and "starts at its age" in agent.TYRE_SETS_SENTENCE


def test_a_recorded_strategy_may_refit_a_set_it_took_off():
    from src.contracts import Plan, RaceParams, Snapshot
    from src.sim.rules import assign_sets
    race = RaceParams(race_id="syn", total_laps=50, pit_loss_s=20.0, min_stint_laps=3)
    snap = Snapshot(race=race, driver="X", compound="SOFT", sets_available=[{"compound": "HARD"}])
    plan = Plan.of([["SOFT", 12], ["HARD", 3], ["HARD", 17, 3]])
    sets, errors = assign_sets(snap, plan, best=True)
    assert not errors and [t.laps_run if t else None for t in sets] == [None, 0, 3]
    assert assign_sets(snap, Plan.of([["SOFT", 12], ["HARD", 3], ["HARD", 17]]), best=True)[1]
