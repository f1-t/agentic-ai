"""Laps with no recorded compound (2022 Austria: whole stints with Compound None): lap_frame labels
them data.UNKNOWN_COMPOUND, logged; they keep their timing but no fit, replay or snapshot uses them."""
import logging

import numpy as np
import pandas as pd
import pytest

from src.contracts import RaceParams
from src.sim import SimConfig, Simulator, calibration
from src.sim.data import UNKNOWN_COMPOUND, derive_columns, label_unknown_compounds
from test_sim_position import race_frame

LEAN = SimConfig(wear_cliff={"enabled": False}, compound_proxy={"enabled": False})   # no cross-race tables
RACE = RaceParams(race_id="syn",   # no date: the pit-loss prior reads no cached race
                  total_laps=50, pit_loss_s=20.0, min_stint_laps=8,
                  compounds=["HARD", "MEDIUM", "SOFT"])


def frame():
    df = race_frame(("AAA", "T1", [("SOFT", 15), ("MEDIUM", 35)]), ("BBB", "T2", [("SOFT", 20), ("MEDIUM", 30)], 0.3),
                    ("CCC", "T3", [("MEDIUM", 25), ("SOFT", 25)], 0.5))
    df["lap_time_s"] += np.random.default_rng(0).normal(0.0, 0.2, len(df))
    late = (df["driver"] == "BBB") & (df["stint"] == 2)
    df.loc[late, "compound"] = None                               # as FastF1 gives it
    df.loc[(df["driver"] == "CCC") & (df["lap"] > 40), "compound"] = "None"   # or as a string
    return df


def test_unlabelled_laps_are_labelled_logged_and_never_clean(caplog):
    with caplog.at_level(logging.WARNING, logger="src.sim.data"):
        df = derive_columns(label_unknown_compounds(frame().drop(columns=["clean"]).assign(pace_resid_s=np.nan)))
    assert "40 laps with no recorded compound" in caplog.text and "BBB: 30" in caplog.text and "CCC: 10" in caplog.text
    unknown = df["compound"] == UNKNOWN_COMPOUND
    assert unknown.sum() == 40 and not df.loc[unknown, "clean"].any()
    assert df.loc[unknown, "end_s"].notna().all()            # the timing stays: gaps and positions need it
    world = Simulator(df, LEAN)
    assert set(world.degradation.coefs) <= {"SOFT", "MEDIUM", "HARD"}


def test_a_driver_with_unlabelled_laps_is_not_replayed_and_gets_no_snapshot_there():
    df = derive_columns(label_unknown_compounds(frame().drop(columns=["clean"]).assign(pace_resid_s=np.nan)))
    assert calibration._finisher_ids(df, 0.9) == ["AAA"]
    table, summary, _ = calibration.run(Simulator(df, LEAN), df, RACE)
    assert list(table.index) == ["AAA"]
    assert calibration.snapshot_as_of(df, RACE, "BBB", 15).compound == "SOFT"   # before the unlabelled stint
    with pytest.raises(ValueError, match="no recorded compound"):
        calibration.snapshot_as_of(df, RACE, "BBB", 30)
    snap = calibration.snapshot_as_of(df, RACE, "AAA", 45)
    assert UNKNOWN_COMPOUND not in snap.race.compounds
