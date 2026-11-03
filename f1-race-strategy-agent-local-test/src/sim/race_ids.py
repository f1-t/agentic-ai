"""Race ids follow the data layout: <season>_<round:02d>_<gp_slug>, e.g. 2021_04_spain.

The gp slug comes from the season's event schedule (make_race_id_from_schedule): the
country, unless that country hosts more than one race that season, in which case every one of
its races takes its event name instead (2021_08_styrian, 2021_09_austrian, 2022_05_miami).
A race id names a race; it never says which circuit the race was on (event_rates.circuit_info).

Until a race registry exists, RACE_ID_ALIASES maps the older <season>_<GP> ids (the
mid-race csvs of SPEC G use 2021_Spain) to the canonical id.
"""
import re
import unicodedata

RACE_ID_ALIASES = {"2021_Spain": "2021_04_spain"}
_CANONICAL = re.compile(r"^(\d{4})_(\d{2})_([a-z0-9]+(?:_[a-z0-9]+)*)$")


def slug(name: str) -> str:
    """'Spain' -> 'spain', 'Emilia Romagna' -> 'emilia_romagna'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def make_race_id(season: int, round_number: int, gp: str) -> str:
    return f"{int(season)}_{int(round_number):02d}_{slug(gp)}"


def _ascii(name) -> str:
    """'São Paulo' -> 'Sao Paulo': accents dropped, so the slug keeps every letter."""
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()


def event_slug(event_name: str) -> str:
    """'Emilia Romagna Grand Prix' -> 'emilia_romagna', '70th Anniversary Grand Prix' -> '70th_anniversary'."""
    return slug(re.sub(r"\bgrand\s+prix\b", " ", _ascii(event_name), flags=re.IGNORECASE))


def make_race_id_from_schedule(row, season_schedule, season: int | None = None) -> str:
    """The race id of one event of a season's schedule (columns RoundNumber, EventName,
    Country; a FastF1 EventSchedule or any frame with them): slug(Country) when that country
    hosts one race in `season_schedule`, else the event name without "Grand Prix" for every race
    of that country that season (2021_02_emilia_romagna and 2021_14_italian; 2022_05_miami
    and 2022_19_united_states). `season` defaults to the schedule's year (FastF1's
    EventSchedule.year), else the row's Season or EventDate."""
    if season is None:
        season = getattr(season_schedule, "year", None)
    if season is None:
        season = row["Season"] if "Season" in row else row["EventDate"].year
    country = str(row["Country"])
    hosts = int((season_schedule["Country"].astype(str) == country).sum())
    gp = _ascii(country) if hosts <= 1 else event_slug(row["EventName"])
    return make_race_id(season, int(row["RoundNumber"]), gp)


def canonical_race_id(race_id: str) -> str:
    """The canonical id for an alias; any other id is returned unchanged."""
    return RACE_ID_ALIASES.get(race_id, race_id)


def parse_race_id(race_id: str) -> tuple[int, int, str]:
    """(season, round, gp slug) of a canonical id or an alias. Raises ValueError otherwise."""
    m = _CANONICAL.match(canonical_race_id(race_id))
    if m is None:
        raise ValueError(f"race id {race_id!r} is not <season>_<round:02d>_<gp_slug> "
                         f"(e.g. 2021_04_spain) or a known alias {sorted(RACE_ID_ALIASES)}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def race_season(race_id: str) -> int:
    """The season of any id that starts with <season>_."""
    return int(canonical_race_id(race_id).split("_", 1)[0])


def as_of(race_id: str, races) -> list[str]:
    """The races in `races` strictly before `race_id` by (season, round), in their given order:
    what is known on the day of `race_id`. Never the race itself, nor another race of the same
    season and round. Every id must parse (parse_race_id); aliases compare as canonical."""
    key = parse_race_id(race_id)[:2]
    return [r for r in races if parse_race_id(r)[:2] < key]
