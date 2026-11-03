"""Race ids follow the data layout: <season>_<round:02d>_<gp_slug>, e.g. 2021_04_spain.

Until a race registry exists, RACE_ID_ALIASES maps the older <season>_<GP> ids (the
mid-race csvs of SPEC G use 2021_Spain) to the canonical id.
"""
import re

RACE_ID_ALIASES = {"2021_Spain": "2021_04_spain"}
_CANONICAL = re.compile(r"^(\d{4})_(\d{2})_([a-z0-9]+(?:_[a-z0-9]+)*)$")


def slug(name: str) -> str:
    """'Spain' -> 'spain', 'Emilia Romagna' -> 'emilia_romagna'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def make_race_id(season: int, round_number: int, gp: str) -> str:
    return f"{int(season)}_{int(round_number):02d}_{slug(gp)}"


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
