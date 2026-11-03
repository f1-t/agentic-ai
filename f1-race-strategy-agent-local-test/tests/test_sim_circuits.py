"""Race ids from the season schedule and circuit identity from the schedule's location
(review of #49, 2a and 2b). The schedules are the cached FastF1 ones where the cache holds
them (2021) and otherwise the committed fixture tests/fixtures/schedules_2018_2025.csv, which
is fastf1.get_event_schedule for 2018-2025 as fetched (its header line gives the version and
date). Nothing is fetched here."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.scenarios import check_schedule_ids
from src.sim import data, event_rates as er
from src.sim.race_ids import make_race_id_from_schedule

FIXTURE = Path(__file__).parent / "fixtures" / "schedules_2018_2025.csv"
SEASONS = range(2018, 2026)
ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["RoundNumber", "EventName", "Country", "Location"]

# The circuit of every event name of 2018-2025, written from the calendars and independent of
# the Location map under test (each of these names was held at one circuit in those seasons).
EXPECTED = {
    "Australian Grand Prix": "albert_park", "Bahrain Grand Prix": "sakhir", "Sakhir Grand Prix": "sakhir_outer",
    "Chinese Grand Prix": "shanghai", "Azerbaijan Grand Prix": "baku", "Spanish Grand Prix": "catalunya",
    "Monaco Grand Prix": "monaco", "Canadian Grand Prix": "villeneuve", "French Grand Prix": "paul_ricard",
    "Austrian Grand Prix": "red_bull_ring", "Styrian Grand Prix": "red_bull_ring",
    "British Grand Prix": "silverstone", "70th Anniversary Grand Prix": "silverstone",
    "German Grand Prix": "hockenheim", "Eifel Grand Prix": "nurburgring", "Hungarian Grand Prix": "hungaroring",
    "Belgian Grand Prix": "spa", "Dutch Grand Prix": "zandvoort", "Italian Grand Prix": "monza",
    "Tuscan Grand Prix": "mugello", "Emilia Romagna Grand Prix": "imola", "Portuguese Grand Prix": "portimao",
    "Turkish Grand Prix": "istanbul_park", "Russian Grand Prix": "sochi", "Singapore Grand Prix": "marina_bay",
    "Japanese Grand Prix": "suzuka", "Qatar Grand Prix": "lusail", "United States Grand Prix": "cota",
    "Miami Grand Prix": "miami", "Las Vegas Grand Prix": "las_vegas", "Mexican Grand Prix": "rodriguez",
    "Mexico City Grand Prix": "rodriguez", "Brazilian Grand Prix": "interlagos",
    "São Paulo Grand Prix": "interlagos", "Saudi Arabian Grand Prix": "jeddah",
    "Abu Dhabi Grand Prix": "yas_marina",
}


def fixture() -> pd.DataFrame:
    f = pd.read_csv(FIXTURE, comment="#", encoding="utf-8")
    return f.assign(season=f["EventDate"].str[:4].astype(int))


def fixture_schedule(season) -> pd.DataFrame:
    f = fixture()
    return f[f["season"] == season][COLUMNS].reset_index(drop=True)


def schedule(season) -> pd.DataFrame:
    """The cached schedule when there is one, else the fixture's."""
    cached = data.season_schedule(season)
    return cached if cached is not None else fixture_schedule(season)


def ids(season) -> dict[int, str]:
    s = schedule(season)
    return {int(r["RoundNumber"]): make_race_id_from_schedule(r, s, season) for _, r in s.iterrows()}


def test_the_fixture_is_the_fetched_2018_2025_calendar():
    f = fixture()
    assert FIXTURE.read_text(encoding="utf-8").startswith("# fastf1.get_event_schedule")
    assert f.groupby("season").size().to_dict() == {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22,
                                                     2024: 24, 2025: 24}
    assert set(f["EventName"]) == set(EXPECTED)
    for season, g in f.groupby("season"):   # rounds 1..n, renumbered after a cancellation (Imola 2023)
        assert list(g["RoundNumber"]) == list(range(1, len(g) + 1)), season


def test_the_cache_holds_the_2021_schedule_and_it_agrees_with_the_fixture():
    cached = data.season_schedule(2021)
    assert cached is not None
    assert cached[COLUMNS].astype(str).equals(fixture_schedule(2021).astype(str))


def test_race_ids_take_the_event_name_when_a_country_hosts_more_than_one_race():
    got = {s: ids(s) for s in SEASONS}
    for season, rnd, rid in [(2021, 4, "2021_04_spain"), (2021, 8, "2021_08_styrian"), (2021, 9, "2021_09_austrian"),
                             (2022, 5, "2022_05_miami"), (2022, 19, "2022_19_united_states"),
                             (2023, 21, "2023_21_las_vegas"), (2021, 2, "2021_02_emilia_romagna"),
                             (2021, 14, "2021_14_italian"), (2020, 9, "2020_09_tuscan"),
                             (2020, 5, "2020_05_70th_anniversary"), (2020, 16, "2020_16_sakhir"),
                             (2023, 14, "2023_14_italy"), (2023, 6, "2023_06_monaco"),
                             (2019, 10, "2019_10_united_kingdom"), (2024, 12, "2024_12_united_kingdom"),
                             (2021, 19, "2021_19_brazil"), (2018, 21, "2018_21_united_arab_emirates"),
                             (2021, 22, "2021_22_united_arab_emirates"),
                             (2024, 24, "2024_24_united_arab_emirates")]:
        assert got[season][rnd] == rid
    for season, by_round in got.items():
        slugs = [r.split("_", 2)[2] for r in by_round.values()]
        assert len(set(slugs)) == len(slugs), season   # no two races of a season share a slug


def test_a_circuit_that_is_its_countrys_only_race_keeps_one_slug_in_every_season():
    """Country spellings drift between seasons (COUNTRY_ALIASES): per circuit, the seasons in
    which it is its country's only race all give it the same slug."""
    from src.sim.race_ids import country_name
    slugs = {}
    for season in SEASONS:
        s = schedule(season)
        by_round = ids(season)
        hosts = s["Country"].map(country_name).value_counts()
        for _, r in s.iterrows():
            if hosts[country_name(r["Country"])] == 1:
                circuit = er.CIRCUITS[er.location_key(r["Location"])][0]
                slugs.setdefault(circuit, set()).add(by_round[int(r["RoundNumber"])].split("_", 2)[2])
    drift = {c: v for c, v in slugs.items() if len(v) > 1}
    assert not drift and len(slugs) == 27
    assert slugs["yas_marina"] == {"united_arab_emirates"} and slugs["silverstone"] == {"united_kingdom"}
    assert slugs["catalunya"] == {"spain"} and slugs["cota"] == {"united_states"}


def test_v1s_frozen_scenario_ids_are_unchanged():
    v1 = json.loads((ROOT / "data" / "scenarios" / "v1.json").read_text())
    races = sorted({i["race_id"] for i in v1["items"]})
    assert races == ["2021_04_spain"] and data.schedule_race_id(2021, 4) == "2021_04_spain"
    check_schedule_ids(races + ["2021_Spain"])
    with pytest.raises(ValueError, match="2021_08_styrian"):
        check_schedule_ids(["2021_08_austria"])


def test_every_2018_2025_race_maps_to_its_circuit():
    f = fixture()
    wrong, n = [], 0
    for season in SEASONS:
        s = schedule(season)
        names = dict(zip(s["RoundNumber"].astype(int), s["EventName"]))
        for rnd, rid in ids(season).items():
            c = er.circuit_info([rid], schedule=lambda _season, s=s: s).iloc[0]
            n += 1
            if c["circuit"] != EXPECTED[names[rnd]] or c["circuit_type"] == er.UNKNOWN:
                wrong.append((rid, c["circuit"], c["circuit_type"], EXPECTED[names[rnd]]))
    assert not wrong and n == len(f) == 173


def test_every_fetched_location_is_a_circuits_key():
    keys = {er.location_key(loc) for loc in fixture()["Location"]}
    assert keys <= set(er.CIRCUITS), sorted(keys - set(er.CIRCUITS))


@pytest.mark.parametrize("season, country, circuits", [
    (2020, "Austria", ["red_bull_ring", "red_bull_ring"]),
    (2021, "Austria", ["red_bull_ring", "red_bull_ring"]),
    (2020, "Great Britain", ["silverstone", "silverstone"]),
    (2020, "Bahrain", ["sakhir", "sakhir_outer"]),
    (2020, "Italy", ["monza", "mugello", "imola"]),
    (2021, "Italy", ["imola", "monza"]),
    (2022, "Italy", ["imola", "monza"]),
    (2024, "Italy", ["imola", "monza"]),
    (2022, "United States", ["miami", "cota"]),
    (2023, "United States", ["miami", "cota", "las_vegas"]),
    (2025, "United States", ["miami", "cota", "las_vegas"]),
])
def test_duplicate_country_seasons_get_each_races_own_circuit(season, country, circuits):
    s = schedule(season)
    rows = s[s["Country"] == country]
    rids = [make_race_id_from_schedule(r, s, season) for _, r in rows.iterrows()]
    got = er.circuit_info(rids, schedule=lambda _season: s)
    assert list(got["circuit"]) == circuits and len(set(rids)) == len(rids)


def test_without_a_schedule_the_slug_is_its_own_circuit_of_unknown_type():
    got = er.circuit_info(["2023_05_united_states", "2024_02_atlantis"], schedule=lambda _season: None)
    assert [tuple(r) for r in got[["circuit", "circuit_type"]].to_numpy()] == [
        ("united_states", er.UNKNOWN), ("atlantis", er.UNKNOWN)]   # not cota: the id names a race only


def test_race_features_f1db_circuit_ids_come_first():
    s = fixture_schedule(2022)
    feats = pd.DataFrame({"race_id": ["2022_05_miami", "2022_19_united_states"],
                          "circuitId": ["miami", "austin"], "circuitType": ["STREET", "RACE"]})
    got = er.circuit_info(["2022_05_miami", "2022_19_united_states", "2022_01_bahrain"], feats,
                          schedule=lambda _season: s).set_index("race_id")
    assert tuple(got.loc["2022_19_united_states"]) == ("austin", er.PERMANENT)
    assert tuple(got.loc["2022_05_miami"]) == ("miami", er.STREET)
    assert tuple(got.loc["2022_01_bahrain"]) == ("sakhir", er.PERMANENT)    # not in the features: the schedule
