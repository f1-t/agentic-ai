"""Tyre-set inventory tests. Synthetic Laps frames only, no cache."""
import numpy as np
import pandas as pd
import pytest

from src.sim.tyre_sets import split_fitted, weekend_report, weekend_sets

ALLOC = {"HARD": 2, "MEDIUM": 3, "SOFT": 8}


def laps(stints, driver="AAA", first_lap=1, number="1", team="T1"):
    """Laps frame from [(compound, fresh, tyre_life_at_start, n_laps[, lap_time_s]), ...]."""
    rows, lap = [], first_lap
    for i, (comp, fresh, life0, n, *t) in enumerate(stints, start=1):
        for k in range(n):
            rows.append({"Driver": driver, "DriverNumber": number, "Team": team,
                         "LapNumber": lap, "Stint": float(i), "Compound": comp,
                         "TyreLife": float(life0 + k), "FreshTyre": fresh,
                         "LapTime": pd.Timedelta(seconds=t[0] if t else 90.0)})
            lap += 1
    return pd.DataFrame(rows)


def by_id(sets):
    return {s["set_id"]: s for s in sets}


class SplitLaps(pd.DataFrame):
    """A frame with FastF1's Laps.split_qualifying_sessions(), returning `parts`."""
    _metadata = ["parts"]

    def split_qualifying_sessions(self):
        return self.parts


def test_new_sets_are_detected():
    fp1 = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 5), ("MEDIUM", True, 1, 8)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {}))
    assert sets == {
        "SOFT-1": {"compound": "SOFT", "laps_run": 3, "source": "observed", "set_id": "SOFT-1"},
        "SOFT-2": {"compound": "SOFT", "laps_run": 5, "source": "observed", "set_id": "SOFT-2"},
        "MEDIUM-1": {"compound": "MEDIUM", "laps_run": 8, "source": "observed",
                     "set_id": "MEDIUM-1"},
    }


def test_reappearance_continues_tyre_life_across_sessions():
    fp1 = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 3)])
    fp2 = laps([("SOFT", False, 4, 2)])  # 3 laps already on it: one of the two S3 sets
    sets = by_id(weekend_sets({"FP1": fp1, "FP2": fp2}, "AAA", {"SOFT": 2}, {}))
    # the most recently fitted of the two is matched
    assert {k: v["laps_run"] for k, v in sets.items()} == {"SOFT-1": 3, "SOFT-2": 5}
    assert all(s["source"] == "observed" for s in sets.values())


def test_reappearance_within_a_session_and_unmatched_is_inferred():
    fp1 = laps([("MEDIUM", True, 1, 4), ("MEDIUM", False, 5, 3), ("MEDIUM", False, 10, 2)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"MEDIUM": 2}, {}))
    assert sets["MEDIUM-1"]["laps_run"] == 7 and sets["MEDIUM-1"]["source"] == "observed"
    assert sets["MEDIUM-2"]["laps_run"] == 11 and sets["MEDIUM-2"]["source"] == "inferred"


def test_returns_remove_the_most_used_sets():
    fp1 = laps([("SOFT", True, 1, 5), ("SOFT", True, 1, 2), ("MEDIUM", True, 1, 8)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {"FP1": 1}))
    assert set(sets) == {"SOFT-1", "SOFT-2"}
    assert all(s["source"] == "observed" for s in sets.values())


def test_return_tie_removes_earliest_and_marks_survivors_inferred():
    fp1 = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 3), ("MEDIUM", True, 1, 1)])
    sets = by_id(weekend_sets({"FP1": fp1}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {"FP1": 1}))
    assert set(sets) == {"SOFT-2", "MEDIUM-1"}
    assert sets["SOFT-2"]["source"] == "inferred" and sets["MEDIUM-1"]["source"] == "inferred"


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
    assert set(at_start) == {"SOFT-2", "MEDIUM-1"}  # SOFT-1 (3 laps) was fitted on the grid

    lap30 = by_id(weekend_sets({"Q": q}, "AAA", alloc, {}, race_laps=race, upto_lap=30))
    assert set(lap30) == {"SOFT-2"}  # the medium went on at lap 21

    lap50 = weekend_sets({"Q": q}, "AAA", alloc, {}, race_laps=race, upto_lap=50)
    assert lap50 == []  # SOFT-2 (2 laps) went on at lap 46 with TyreLife 3

    no_race = by_id(weekend_sets({"Q": q}, "AAA", alloc, {}))
    assert set(no_race) == {"SOFT-1", "SOFT-2", "MEDIUM-1"}  # start set unknown, nothing removed


def test_upto_lap_needs_race_laps():
    with pytest.raises(ValueError):
        weekend_sets({}, "AAA", ALLOC, {}, upto_lap=10)


def test_set_ids_unique():
    fp1 = laps([("HARD", True, 1, 9), ("SOFT", True, 1, 3), ("SOFT", False, 4, 3)])
    fp2 = laps([("MEDIUM", True, 1, 7), ("SOFT", True, 1, 3), ("SOFT", False, 9, 2)])
    q = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 3), ("SOFT", True, 1, 3)])
    fp3 = laps([("HYPERSOFT", True, 1, 2)])  # same initial as HARD
    sets = weekend_sets({"FP1": fp1, "FP2": fp2, "FP3": fp3, "Q": q}, "AAA", ALLOC,
                        {"FP1": 1, "FP2": 1})
    ids = [s["set_id"] for s in sets]
    assert len(ids) == len(set(ids))
    assert {"HARD-2", "HYPERSOFT-1"} <= set(ids)  # HARD-1 went back after FP1


def test_messy_rows_and_other_drivers():
    fp1 = laps([("SOFT", True, 1, 4), ("nan", True, 1, 2), ("INTERMEDIATE", True, 1, 3)])
    fp1.loc[1, "TyreLife"] = np.nan                   # NaN TyreLife: lap skipped, stint kept
    fp1.loc[5, "Compound"] = np.nan                   # NaN compound on one lap
    other = laps([("SOFT", True, 1, 10)], driver="BBB", number="2")
    sets = by_id(weekend_sets({"FP1": pd.concat([fp1, other]), "FP2": other}, "AAA",
                              {"SOFT": 1}, {}))
    assert sets["SOFT-1"]["laps_run"] == 4
    assert sets["INTERMEDIATE-1"]["compound"] == "INTERMEDIATE"  # wet sets are kept
    assert set(sets) == {"SOFT-1", "INTERMEDIATE-1"}  # "nan" stint dropped, BBB ignored


def test_refit_fallback_order_and_count():
    # FP2 refits at TyreLife 8 on a 7-lap set (exact), FP3 at TyreLife 6 on a 6-lap set
    # (refit recorded without the increment), Q at TyreLife 8 on a 6-lap set (one
    # extra lap counted).
    fp1 = laps([("SOFT", True, 1, 7), ("MEDIUM", True, 1, 6), ("HARD", True, 1, 6)])
    fp2 = laps([("SOFT", False, 8, 2)])
    fp3 = laps([("MEDIUM", False, 6, 1)])
    q = laps([("HARD", False, 8, 1)])
    rep = weekend_report({"FP1": fp1, "FP2": fp2, "FP3": fp3, "Q": q}, "AAA",
                         {"SOFT": 1, "MEDIUM": 1, "HARD": 1}, {})
    sets = by_id(rep["sets"])
    assert set(sets) == {"SOFT-1", "MEDIUM-1", "HARD-1"}  # no phantom sets
    assert sets["SOFT-1"] == {"compound": "SOFT", "laps_run": 9, "source": "observed",
                              "set_id": "SOFT-1"}
    assert sets["MEDIUM-1"]["laps_run"] == 6 and sets["MEDIUM-1"]["source"] == "inferred"
    assert sets["HARD-1"]["laps_run"] == 8 and sets["HARD-1"]["source"] == "inferred"
    assert rep["fallbacks"] == 2


def test_refit_prefers_k_over_k_minus_2():
    fp1 = laps([("SOFT", True, 1, 4), ("SOFT", True, 1, 2)])
    fp2 = laps([("SOFT", False, 4, 3)])  # no 3-lap set: the 4-lap one wins over the 2-lap one
    rep = weekend_report({"FP1": fp1, "FP2": fp2}, "AAA", {"SOFT": 2}, {})
    assert {s["set_id"]: s["laps_run"] for s in rep["sets"]} == {"SOFT-1": 6, "SOFT-2": 2}
    assert rep["fallbacks"] == 1


def test_race_stint_uses_the_fallback_too():
    q = laps([("SOFT", True, 1, 2), ("SOFT", True, 1, 3)])
    race = laps([("SOFT", False, 4, 20), ("SOFT", False, 2, 30)])  # 2-lap set at TyreLife 2
    rep = weekend_report({"Q": q}, "AAA", {"SOFT": 2}, {}, race_laps=race, upto_lap=66)
    assert rep["sets"] == [] and rep["fitted"]["set_id"] == "SOFT-2"
    assert rep["fallbacks"] == 1


def test_compound_constrained_return():
    # The int form returns the 6-lap medium; the Q3 return is one soft.
    q = laps([("MEDIUM", True, 1, 6), ("SOFT", True, 1, 3), ("SOFT", True, 1, 4)])
    sets = by_id(weekend_sets({"Q": q}, "AAA", {"MEDIUM": 1, "SOFT": 2}, {"Q": {"SOFT": 1}}))
    assert set(sets) == {"MEDIUM-1", "SOFT-1"}
    any_compound = by_id(weekend_sets({"Q": q}, "AAA", {"MEDIUM": 1, "SOFT": 2}, {"Q": 1}))
    assert set(any_compound) == {"SOFT-1", "SOFT-2"}


def test_compound_return_with_no_seen_set_comes_from_that_compound():
    fp1 = laps([("MEDIUM", True, 1, 6)])
    sets = weekend_sets({"FP1": fp1}, "AAA", {"MEDIUM": 2, "SOFT": 2}, {"Q": {"SOFT": 1}})
    assert sorted((s["compound"], s["source"]) for s in sets) == [
        ("MEDIUM", "assumed"), ("MEDIUM", "observed"), ("SOFT", "assumed")]


def test_hindsight_keeps_back_sets_the_race_runs():
    fp3 = laps([("MEDIUM", True, 1, 6), ("MEDIUM", True, 1, 4)])
    race = laps([("SOFT", True, 1, 30), ("MEDIUM", False, 7, 36)])  # races the 6-lap medium
    args = ({"FP3": fp3}, "AAA", {"MEDIUM": 2, "SOFT": 1}, {"FP3": 1})
    belief = weekend_report(*args, race_laps=race, upto_lap=66)
    assert belief["returned"] == [("FP3", "MEDIUM", 6)]
    hind = weekend_report(*args, race_laps=race, upto_lap=66, hindsight=True)
    assert hind["returned"] == [("FP3", "MEDIUM", 4)]
    assert hind["sets"] == []
    with pytest.raises(ValueError):
        weekend_sets(*args, hindsight=True)


def _team_weekend(reserve_team="T1"):
    """FP1 run by reserve RES (own car number) for team T1 and by AAA's teammate BBB."""
    fp1 = pd.concat([laps([("HARD", True, 1, 9), ("SOFT", True, 1, 3)], driver="RES",
                          number="45", team=reserve_team),
                     laps([("SOFT", True, 1, 5)], driver="BBB", number="2")])
    fp2 = laps([("SOFT", True, 1, 4)])
    race = pd.concat([laps([("SOFT", True, 1, 30)]),
                      laps([("SOFT", True, 1, 30)], driver="BBB", number="2")])
    return {"FP1": fp1, "FP2": fp2}, race


def test_reserve_driver_of_the_team_counts_as_the_nominated_driver():
    sessions, race = _team_weekend()
    rep = weekend_report(sessions, "AAA", ALLOC, {"FP1": 1}, race_laps=race)
    seen = sorted((s["compound"], s["laps_run"], s["source"]) for s in rep["sets"]
                  if s["source"] != "assumed")
    assert seen == [("SOFT", 3, "observed"), ("SOFT", 4, "observed")]  # RES's hard went back
    assert rep["fitted"]["laps_run"] == 0 and len(rep["sets"]) == 11
    assert "FP1: no AAA laps, took RES laps (T1)" in rep["notes"]
    # BBB ran FP1 himself, so RES is not his
    bbb = weekend_report(sessions, "BBB", ALLOC, {}, race_laps=race)
    assert not any("RES" in n for n in bbb["notes"])


def test_reserve_driver_needs_race_laps_or_an_explicit_mapping():
    sessions, race = _team_weekend()
    alone = weekend_sets(sessions, "AAA", ALLOC, {"FP1": 2})
    assert sum(s["source"] == "observed" for s in alone) == 1  # only AAA's FP2 soft
    rep = weekend_report(sessions, "AAA", ALLOC, {"FP1": 1}, reserves={"FP1": {"RES": "AAA"}})
    assert "FP1: no AAA laps, took RES laps (reserves)" in rep["notes"]
    assert sum(s["source"] == "observed" for s in rep["sets"]) == 2
    with pytest.raises(ValueError):
        weekend_sets(sessions, "AAA", ALLOC, {}, reserves={"FP4": {"RES": "AAA"}})


def test_reserve_driver_of_another_team_or_ambiguous_is_not_attributed():
    sessions, race = _team_weekend(reserve_team="T2")
    sets = weekend_sets(sessions, "AAA", ALLOC, {"FP1": 2}, race_laps=race)
    assert sum(s["source"] == "observed" for s in sets) == 1  # only AAA's FP2 soft
    sessions, race = _team_weekend()
    sessions["FP1"] = pd.concat([sessions["FP1"],
                                 laps([("MEDIUM", True, 1, 4)], driver="ZZZ", number="46")])
    rep = weekend_report(sessions, "AAA", ALLOC, {"FP1": 2}, race_laps=race)
    assert not any("took" in n for n in rep["notes"])  # two non-race codes ran for T1


def _q3_weekend():
    """Q split: the 1-lap Q1 soft is refitted in Q2 for the fastest Q2 lap (4 laps)."""
    q1 = laps([("SOFT", True, 1, 1, 80.0), ("SOFT", True, 1, 3, 79.0)])
    q2 = laps([("SOFT", False, 2, 3, 77.0), ("SOFT", True, 1, 3, 78.0)])
    q3 = laps([("SOFT", True, 1, 3, 76.0), ("SOFT", True, 1, 3, 76.5)])
    return {"Q1": q1, "Q2": q2, "Q3": q3}


def test_q2_start_rule_protects_the_fastest_q2_set():
    alloc, ret = {"SOFT": 6, "MEDIUM": 1}, {"Q": {"SOFT": 1}}
    rep = weekend_report(_q3_weekend(), "AAA", alloc, ret, q2_start_rule=True)
    assert rep["fitted"] == {"compound": "SOFT", "laps_run": 4, "source": "observed",
                             "set_id": "SOFT-1"}
    assert "SOFT-1" not in by_id(rep["sets"])  # known start set, out of the garage
    assert rep["returned"] == [("Q", "SOFT", 3)]  # not the 4-lap set, the most used
    assert len(rep["sets"]) == 5

    race = laps([("SOFT", False, 5, 25), ("MEDIUM", True, 1, 41)])
    with_race = weekend_report(_q3_weekend(), "AAA", alloc, ret, race_laps=race,
                               q2_start_rule=True)
    assert with_race["fitted"] == rep["fitted"] and with_race["sets"] == rep["sets"]
    assert with_race["fallbacks"] == 0 and not any("mismatch" in n for n in with_race["notes"])

    # Without the rule the 4-lap set is handed back and the start is unknown.
    q = laps([("SOFT", True, 1, 1), ("SOFT", True, 1, 3), ("SOFT", False, 2, 3),
              ("SOFT", True, 1, 3), ("SOFT", True, 1, 3), ("SOFT", True, 1, 3)])
    off = weekend_report({"Q": q}, "AAA", alloc, ret)
    assert off["fitted"] is None and off["returned"] == [("Q", "SOFT", 4)]


def test_q2_start_rule_mismatch_keeps_the_race_choice():
    race = laps([("MEDIUM", True, 1, 30), ("SOFT", False, 5, 36)])
    rep = weekend_report(_q3_weekend(), "AAA", {"SOFT": 6, "MEDIUM": 1}, {"Q": {"SOFT": 1}},
                         race_laps=race, q2_start_rule=True)
    assert rep["fitted"]["compound"] == "MEDIUM"
    assert "SOFT-1" in by_id(rep["sets"])  # the protected set stays in the garage
    assert any(n.startswith("R: Q2 start set mismatch") for n in rep["notes"])


def test_q2_start_rule_non_q3_driver_start_set_is_unknown():
    sessions = _q3_weekend()
    sessions["Q3"] = laps([("SOFT", True, 1, 3)], driver="BBB", number="2")
    rep = weekend_report(sessions, "AAA", {"SOFT": 6, "MEDIUM": 1}, {}, q2_start_rule=True)
    assert rep["fitted"] is None and len(rep["sets"]) == 7


def test_q2_start_rule_splits_fastf1_laps():
    parts = list(_q3_weekend().values())
    q = SplitLaps(pd.concat(parts))
    q.parts = parts
    args = ("AAA", {"SOFT": 6, "MEDIUM": 1}, {"Q": {"SOFT": 1}})
    from_laps = weekend_report({"Q": q}, *args, q2_start_rule=True)
    assert from_laps == weekend_report(_q3_weekend(), *args, q2_start_rule=True)


def test_q2_start_rule_inputs():
    sessions = _q3_weekend()
    with pytest.raises(ValueError):  # Q1/Q2/Q3 keys only under the rule
        weekend_sets(sessions, "AAA", ALLOC, {})
    with pytest.raises(ValueError):
        weekend_sets(dict(sessions, Q=sessions["Q2"]), "AAA", ALLOC, {}, q2_start_rule=True)
    plain = weekend_report({"Q": sessions["Q2"]}, "AAA", ALLOC, {}, q2_start_rule=True)
    assert "Q: not split into Q1/Q2/Q3, q2_start_rule not applied" in plain["notes"]
    assert plain["fitted"] is None


def test_allocation_overrun_is_noted():
    fp1 = laps([("MEDIUM", True, 1, 3), ("MEDIUM", True, 1, 3), ("SOFT", True, 1, 2)])
    rep = weekend_report({"FP1": fp1}, "AAA", {"MEDIUM": 1, "SOFT": 2}, {})
    assert "MEDIUM: 2 sets seen, allocation 1" in rep["notes"]
    assert not any(n.startswith("SOFT:") for n in rep["notes"])
    assert len(rep["sets"]) == 4  # 2 MEDIUM seen + 1 SOFT seen + 1 SOFT assumed


def test_sessions_run_in_canonical_order_whatever_the_dict_order():
    fp1 = laps([("SOFT", True, 1, 3)])
    q = laps([("SOFT", False, 4, 2)])  # carries on from the FP1 set
    sets = weekend_sets({"Q": q, "FP1": fp1}, "AAA", {"SOFT": 1}, {})
    assert sets == [{"compound": "SOFT", "laps_run": 5, "source": "observed", "set_id": "SOFT-1"}]
    flipped = weekend_sets({"Q": q, "FP1": fp1}, "AAA", {"SOFT": 1}, {}, order=["Q", "FP1"])
    assert [s["source"] for s in flipped] == ["inferred", "observed"]
    with pytest.raises(ValueError):
        weekend_sets({"FP4": fp1}, "AAA", {"SOFT": 1}, {})


def test_split_fitted_takes_the_start_set_out():
    q = laps([("SOFT", True, 1, 3), ("SOFT", True, 1, 2)])
    sets = weekend_sets({"Q": q}, "AAA", {"SOFT": 2, "MEDIUM": 1}, {})
    fitted, garage = split_fitted(sets, "SOFT", 3)
    assert fitted["set_id"] == "SOFT-1"
    assert [s["set_id"] for s in garage] == ["SOFT-2", "MEDIUM-1"]
    assert len(sets) == 3  # input untouched
    assert split_fitted(sets, "HARD", 0) == (None, sets)
