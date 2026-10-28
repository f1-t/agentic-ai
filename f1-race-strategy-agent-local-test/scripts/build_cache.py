"""Populate the local FastF1 cache for the one weekend the baseline uses.

Run this with a network connection. It downloads lap timing, weather and
race-control messages (no telemetry -> small cache) for every session of the
weekend, plus the event schedule and race results, into ./cache so that
everything else in this project runs fully offline via
fastf1.Cache.offline_mode(True).

    python scripts/build_cache.py            # add to the existing cache
    python scripts/build_cache.py --fresh    # rebuild from an empty cache
    python scripts/build_cache.py --season 2022 --gp Monaco --sessions R Q

--fresh builds in a temp dir so the sqlite holds only what these loads need,
prunes it, VACUUMs it, then replaces ./cache only after the build has
succeeded. --prune drops the raw livetiming responses from the sqlite (the
parsed .ff1pkl pickles already hold that data and offline mode only needs the
schedule and results); it is on by default under --fresh, --no-prune disables.
"""
import argparse
import os
import shutil
import sqlite3
import tempfile
import fastf1
from requests_cache.backends.sqlite import SQLiteCache

# Defaults: the weekend the baseline is built around. Barcelona 2021 was a
# classic 1-stop vs 2-stop / undercut battle, so tyre strategy actually mattered.
SEASON = 2021
GP_NAME = "Spain"
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
HTTP_CACHE = "fastf1_http_cache.sqlite"


def build(cache_dir, season, gp, sessions, prune):
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.configure(cache_dir=cache_dir)

    print(f"Fetching {season} {gp} GP schedule ...")
    event = fastf1.get_event(season, gp)
    print(f"Event: {event['EventName']}")

    for name in sessions:
        print(f"Loading {season} {gp} GP ({name}) ...")
        session = fastf1.get_session(season, gp, name)
        # No telemetry keeps the committed cache small.
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        print(f"  {len(session.laps)} laps, {len(session.weather_data)} weather rows, "
              f"{len(session.race_control_messages)} messages")
        if name == "R":
            # Race results (Ergast) come with the race load.
            print(f"  results: {len(session.results)} drivers")

    if prune:
        prune_raw_responses(os.path.join(cache_dir, HTTP_CACHE))
    vacuum(os.path.join(cache_dir, HTTP_CACHE))


def prune_raw_responses(path):
    cache = SQLiteCache(db_path=path)
    raw = [key for key, response in cache.responses.items()
           if "/static/" in response.url and response.url.endswith(".jsonStream")]
    for key in raw:
        del cache.responses[key]
    cache.close()
    print(f"Pruned {len(raw)} raw livetiming responses.")


def vacuum(path):
    conn = sqlite3.connect(path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true",
                        help="build from an empty cache dir and replace ./cache")
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument("--gp", default=GP_NAME)
    parser.add_argument("--sessions", nargs="+", default=SESSIONS)
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, default=None,
                        help="drop raw livetiming responses from the sqlite (default: on with --fresh)")
    args = parser.parse_args()
    prune = args.fresh if args.prune is None else args.prune
    spec = (args.season, args.gp, args.sessions, prune)

    if not args.fresh:
        build(CACHE_DIR, *spec)
        print("Cache populated at:", os.path.abspath(CACHE_DIR))
        return

    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, *spec)
        # Build succeeded: only now replace the committed cache contents.
        for entry in os.listdir(CACHE_DIR):
            path = os.path.join(CACHE_DIR, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif entry != ".gitignore":
                os.remove(path)
        shutil.copytree(tmp, CACHE_DIR, dirs_exist_ok=True)
    print("Cache rebuilt at:", os.path.abspath(CACHE_DIR))


if __name__ == "__main__":
    main()
