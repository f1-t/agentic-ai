"""Populate the local FastF1 cache for the one weekend the baseline uses.

Run this with a network connection. It downloads lap timing, weather and
race-control messages (no telemetry -> small cache) for every session of the
weekend, plus the event schedule and race results, into ./cache so that
everything else in this project runs fully offline via
fastf1.Cache.offline_mode(True).

    python scripts/build_cache.py            # add to the existing cache
    python scripts/build_cache.py --fresh    # rebuild from an empty cache

--fresh builds in a temp dir so the sqlite holds only what these loads need,
VACUUMs it, then replaces ./cache only after the build has succeeded.
"""
import argparse
import os
import shutil
import sqlite3
import tempfile
import fastf1

# The race the baseline is built around. Barcelona 2021 was a classic
# 1-stop vs 2-stop / undercut battle, so tyre strategy actually mattered.
SEASON = 2021
GP_NAME = "Spain"
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
HTTP_CACHE = "fastf1_http_cache.sqlite"


def build(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.configure(cache_dir=cache_dir)

    print(f"Fetching {SEASON} {GP_NAME} GP schedule ...")
    event = fastf1.get_event(SEASON, GP_NAME)
    print(f"Event: {event['EventName']}")

    for name in SESSIONS:
        print(f"Loading {SEASON} {GP_NAME} GP ({name}) ...")
        session = fastf1.get_session(SEASON, GP_NAME, name)
        # No telemetry keeps the committed cache small.
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        print(f"  {len(session.laps)} laps, {len(session.weather_data)} weather rows, "
              f"{len(session.race_control_messages)} messages")
        if name == "R":
            # Race results (Ergast) come with the race load.
            print(f"  results: {len(session.results)} drivers")

    vacuum(os.path.join(cache_dir, HTTP_CACHE))


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
    args = parser.parse_args()

    if not args.fresh:
        build(CACHE_DIR)
        print("Cache populated at:", os.path.abspath(CACHE_DIR))
        return

    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
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
