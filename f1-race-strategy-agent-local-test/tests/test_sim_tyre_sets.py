"""Tyre-set inventory tests. Synthetic Laps frames only, no cache."""
import numpy as np
import pandas as pd
import pytest

from src.sim.tyre_sets import weekend_sets

ALLOC = {"HARD": 2, "MEDIUM": 3, "SOFT": 8}


def laps(stints, driver="AAA", first_lap=1):
    """Laps frame from [(compound, fresh, tyre_life_at_start, n_laps), ...]."""
    rows, lap = [], first_lap
    for i, (comp, fresh, life0, n) in enumerate(stints, start=1):
        for k in range(n):
            rows.append({"Driver": driver, "LapNumber": lap, "Stint": float(i), "Compound": comp,
                         "TyreLife": float(life0 + k), "FreshTyre": fresh})
            lap += 1
    return pd.DataFrame(rows)


def by_id(sets):
    return {s["set_id"]: s for s in sets}


def test_new_sets_are_detected():
    fp1 = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 5), ("MEDIUM", True, 1, 8)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {}))
    assert sets == {
        "S1": {"compound": "SOFT", "laps_run": 3, "source": "observed", "set_id": "S1"},
        "S2": {"compound": "SOFT", "laps_run": 5, "source": "observed", "set_id": "S2"},
        "M1": {"compound": "MEDIUM", "laps_run": 8, "source": "observed", "set_id": "M1"},
    }


def test_reappearance_continues_tyre_life_across_sessions():
    fp1 = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 3)])
    fp2 = laps([("SOFT", False, 4, 2)])  # 3 laps already on it: one of the two S3 sets
    sets = by_id(weekend_sets({"FP1": fp1, "FP2": fp2}, "AAA", {"SOFT": 2}, {}))
    assert {k: v["laps_run"] for k, v in sets.items()} == {"S1": 3, "S2": 5}  # most recent matched
    assert all(s["source"] == "observed" for s in sets.values())


def test_reappearance_within_a_session_and_unmatched_is_inferred():
    fp1 = laps([("MEDIUM", True, 1, 4), ("MEDIUM", False, 5, 3), ("MEDIUM", False, 10, 2)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"MEDIUM": 2}, {}))
    assert sets["M1"]["laps_run"] == 7 and sets["M1"]["source"] == "observed"
    assert sets["M2"]["laps_run"] == 11 and sets["M2"]["source"] == "inferred"


def test_returns_remove_the_most_used_sets():
    fp1 = laps([("SOFT", True, 1, 5), ("SOFT", True, 1, 2), ("MEDIUM", True, 1, 8)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {"FP1": 1}))
    assert set(sets) == {"S1", "S2"}
    assert all(s["source"] == "observed" for s in sets.values())


def test_return_tie_removes_earliest_and_marks_survivors_inferred():
    fp1 = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 3), ("MEDIUM", True, 1, 1)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {"FP1": 1}))
    assert set(sets) == {"S2", "M1"}
    assert sets["S2"]["source"] == "inferred" and sets["M1"]["source"] == "inferred"


def test_allocation_gap_yields_assumed_new_sets():
    fp1 = laps([("SOFT", True, 1, 3)])
    sets = weekend_sets({"FP1": fp1}, "AAA", ALLOC, {})
    assumed = [s for s in sets if s["source"] == "assumed"]
    assert len(sets) == 13 and len(assumed) == 12
    assert all(s["laps_run"] == 0 for s in assumed)
    assert {c: sum(s["compound"] == c for s in sets) for c in ALLOC} == ALLOC


def test_missing_session_still_applies_its_returns():
    fp2 = laps([("SOFT", True, 1, 3)])
    sets = weekend_sets({"FP2": fp2}, "AAA", ALLOC, {"FP1": 2})
    assert len(sets) == 11
    assert sum(s["source"] == "assumed" for s in sets) == 10


def test_race_snapshot_excludes_fitted_and_consumed_sets():
    q = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 2)])
    race = laps([("SOFT", False, 4, 20), ("MEDIUM", True, 1, 25), ("SOFT", False, 3, 21)])
    alloc = {"MEDIUM": 1, "SOFT": 2}

    at_start = by_id(weekend_sets({"Q": q}, "AAA", alloc, {}, race_laps=race))
    assert set(at_start) == {"S2", "M1"}  # S1 (3 laps) was fitted on the grid

    lap30 = by_id(weekend_sets({"Q": q}, "AAA", alloc, {}, race_laps=race, upto_lap=30))
    assert set(lap30) == {"S2"}  # the medium went on at lap 21

    lap50 = weekend_sets({"Q": q}, "AAA", alloc, {}, race_laps=race, upto_lap=50)
    assert lap50 == []  # S2 (2 laps) went on at lap 46 with TyreLife 3

    no_race = by_id(weekend_sets({"Q": q}, "AAA", alloc, {}))
    assert set(no_race) == {"S1", "S2", "M1"}  # start set unknown, nothing removed


def test_upto_lap_needs_race_laps():
    with pytest.raises(ValueError):
        weekend_sets({}, "AAA", ALLOC, {}, upto_lap=10)


def test_set_ids_unique():
    fp1 = laps([("HARD", True, 1, 9), ("SOFT", True, 1, 3), ("SOFT", False, 4, 3)])
    fp2 = laps([("MEDIUM", True, 1, 7), ("SOFT", True, 1, 3), ("SOFT", False, 9, 2)])
    q = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 3), ("SOFT", True, 1, 3)])
    fp3 = laps([("HYPERSOFT", True, 1, 2)])  # shares "H" with HARD
    sets = weekend_sets({"FP1": fp1, "FP2": fp2, "FP3": fp3, "Q": q}, "AAA", ALLOC,
                        {"FP1": 1, "FP2": 1})
    ids = [s["set_id"] for s in sets]
    assert len(ids) == len(set(ids))


def test_messy_rows_and_other_drivers():
    fp1 = laps([("SOFT", True, 1, 4), ("nan", True, 1, 2), ("INTERMEDIATE", True, 1, 3)])
    fp1.loc[1, "TyreLife"] = np.nan                   # NaN TyreLife: lap skipped, stint kept
    fp1.loc[5, "Compound"] = np.nan                   # NaN compound on one lap
    other = laps([("SOFT", True, 1, 10)], driver="BBB")
    sets = by_id(weekend_sets({"FP1": pd.concat([fp1, other]), "FP2": other}, "AAA",
                              {"SOFT": 1}, {}))
    assert sets["S1"]["laps_run"] == 4
    assert sets["I1"]["compound"] == "INTERMEDIATE"  # wet sets are kept
    assert set(sets) == {"S1", "I1"}                 # "nan" stint dropped, BBB ignored
