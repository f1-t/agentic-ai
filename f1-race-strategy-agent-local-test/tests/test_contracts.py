import pytest
from pydantic import ValidationError

from src.contracts import Plan, RaceParams, Snapshot

RACE = RaceParams(race_id="t", total_laps=50, pit_loss_s=20.0, compounds=["soft", "medium"])


def test_compounds_are_normalised():
    assert RACE.compounds == ["SOFT", "MEDIUM"]
    assert Plan.of([["soft", 10]]).stints[0].compound == "SOFT"


def test_plan_accepts_pairs_and_triples():
    plan = Plan.of([["SOFT", 20], ["MEDIUM", 30, 4]])
    assert plan.total_laps == 50 and plan.num_stops == 1
    assert plan.stints[1].start_age == 4
    assert plan.as_pairs() == [["SOFT", 20], ["MEDIUM", 30]]


def test_plan_rejects_bad_stints():
    with pytest.raises(ValidationError):
        Plan.of([["SOFT"]])
    with pytest.raises(ValidationError):
        Plan.of([])


def test_pit_laps_are_absolute():
    snap = Snapshot(race=RACE, driver="X", laps_completed=10, compound="SOFT", tyre_age=10)
    assert Plan.of([["SOFT", 15], ["MEDIUM", 15], ["SOFT", 10]]).pit_laps(snap) == [25, 40]


def test_snapshot_properties():
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="MEDIUM", tyre_age=5,
                    stints=[{"compound": "SOFT", "laps": 15, "start_lap": 1}])
    assert snap.remaining_laps == 30 and snap.stops_made == 1
    assert snap.compounds_used == {"SOFT", "MEDIUM"}


@pytest.mark.parametrize("kw", [
    {"laps_completed": 50},
    {"laps_completed": 5, "stints": [{"compound": "SOFT", "laps": 9, "start_lap": 1}]},
    {"laps_completed": 5, "sc_laps": [7]},
])
def test_snapshot_rejects_inconsistent_state(kw):
    with pytest.raises(ValidationError):
        Snapshot(race=RACE, driver="X", compound="SOFT", **kw)


def test_optional_fields_default_to_unknown():
    snap = Snapshot(race=RACE, driver="X", compound="SOFT")
    assert snap.weather.rainfall_now is False and snap.weather.rain_laps == []
    assert snap.field.class_counts == {} and snap.sets_available is None
    assert snap.gap_ahead_s is None and snap.red_flag_laps == [] and snap.standing_start


def test_weather_and_field_round_trip():
    snap = Snapshot(
        race=RACE, driver="X", laps_completed=12, compound="INTERMEDIATE", tyre_age=12,
        weather={"rainfall_now": True, "track_temp_c": 18.5, "rain_laps": [9, 10, 11, 12],
                 "wetness_estimate": 0.7},
        field={"class_counts": {"INTERMEDIATE": 17, "SLICK": 3},
               "class_pace_s": {"INTERMEDIATE": 95.2, "SLICK": 101.0}},
        sets_available=[{"compound": "soft"}, {"compound": "medium", "laps_run": 3, "source": "inferred"}],
        gap_ahead_s=1.4, car_ahead="VER",
    )
    assert snap.weather.wetness_estimate == 0.7
    assert [t.compound for t in snap.sets_available] == ["SOFT", "MEDIUM"]
    assert snap.sets_available[1].is_new is False
    assert snap.field.class_counts["INTERMEDIATE"] == 17


@pytest.mark.parametrize("kw", [
    {"red_flag_laps": [8]},
    {"weather": {"rain_laps": [6]}},
    {"weather": {"wetness_estimate": 1.5}},
])
def test_history_after_snapshot_and_bad_wetness_rejected(kw):
    with pytest.raises(ValidationError):
        Snapshot(race=RACE, driver="X", laps_completed=5, compound="SOFT", **kw)


def test_wet_compounds_pass_the_plan_contract():
    plan = Plan.of([["intermediate", 20], ["SOFT", 30]])
    assert plan.stints[0].compound == "INTERMEDIATE"


def test_tyre_set_is_new_is_derived_and_ids_unique():
    from src.contracts import TyreSet
    assert TyreSet(compound="soft").is_new and TyreSet(compound="soft").model_dump()["is_new"]
    assert not TyreSet(compound="soft", laps_run=3, source="inferred").is_new
    with pytest.raises(ValidationError):
        Snapshot(race=RACE, driver="X", compound="SOFT",
                 sets_available=[{"compound": "SOFT", "set_id": "a"}, {"compound": "HARD", "set_id": "a"}])
    snap = Snapshot(race=RACE, driver="X", compound="SOFT",
                    sets_available=[{"compound": "MEDIUM", "laps_run": 4, "set_id": "m1"}])
    assert snap.find_set("m1").laps_run == 4 and snap.find_set("zz") is None


def test_tyre_set_round_trips_through_model_dump():
    from src.contracts import TyreSet
    ts = TyreSet(compound="SOFT", laps_run=3)
    assert ts.is_new is False
    again = TyreSet.model_validate(ts.model_dump())
    assert again == ts and again.is_new is False


def test_snapshot_with_sets_available_round_trips():
    snap = Snapshot(race=RACE, driver="X", compound="SOFT",
                    sets_available=[{"compound": "MEDIUM", "laps_run": 4, "set_id": "m1"},
                                    {"compound": "HARD"}])
    assert Snapshot.model_validate(snap.model_dump()) == snap
    assert Snapshot.model_validate_json(snap.model_dump_json()) == snap


def test_tyre_set_rejects_conflicting_is_new():
    from src.contracts import TyreSet
    with pytest.raises(ValidationError):
        TyreSet(compound="SOFT", laps_run=3, is_new=True)
