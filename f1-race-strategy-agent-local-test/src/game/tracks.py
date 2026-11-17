"""Load FastF1-generated circuit geometries from ``data/tracks``."""
from pathlib import Path

from .engine import TrackGeometry

TRACK_DIR = Path(__file__).resolve().parents[2] / "data" / "tracks"


def load_track(circuit_id: str, directory: str | Path = TRACK_DIR, *, season: int | None = None) -> TrackGeometry:
    """Load a generated track by circuit id and optional season (e.g. ``catalunya``, 2024)."""
    directory = Path(directory)
    path = directory / f"{circuit_id}_{season}.json" if season is not None else directory / f"{circuit_id}.json"
    if season is None and not path.is_file():
        versions = sorted(directory.glob(f"{circuit_id}_[0-9][0-9][0-9][0-9].json"))
        if versions:
            path = versions[-1]
    if not path.is_file():
        raise FileNotFoundError(
            f"track geometry {circuit_id!r} is not installed at {path}; "
            "run `python scripts/build_track_geometries.py` from the project directory"
        )
    return TrackGeometry.load(path)


def load_race_track(race_id: str, directory: str | Path = TRACK_DIR) -> TrackGeometry:
    """Resolve a canonical project race id through its schedule to a circuit geometry."""
    from ..sim.event_rates import circuit_info
    from ..sim.race_ids import race_season
    circuit = str(circuit_info([race_id]).iloc[0]["circuit"])
    if circuit == "unknown":
        raise ValueError(f"no known circuit mapping for race {race_id!r}")
    try:
        return load_track(circuit, directory, season=race_season(race_id))
    except FileNotFoundError:
        return load_track(circuit, directory)
