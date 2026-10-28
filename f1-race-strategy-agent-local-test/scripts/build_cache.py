"""Populate the local FastF1 cache for the one race the baseline uses.

Run this ONCE with a network connection. It downloads lap-level timing data
(no telemetry, no weather -> small cache) into ./cache so that everything else
in this project runs fully offline via fastf1.Cache.offline_mode(True).

    python scripts/build_cache.py
"""
import os
import fastf1

# The race the baseline is built around. Barcelona 2021 was a classic
# 1-stop vs 2-stop / undercut battle, so tyre strategy actually mattered.
SEASON = 2021
GP_NAME = "Spain"
SESSION = "R"  # Race

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.configure(cache_dir=CACHE_DIR)

    print(f"Loading {SEASON} {GP_NAME} GP ({SESSION}) ...")
    session = fastf1.get_session(SEASON, GP_NAME, SESSION)
    # Laps only keeps the committed cache small.
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps
    print(f"Loaded {len(laps)} laps for {session.event['EventName']}.")
    print("Drivers:", sorted(laps["Driver"].unique().tolist()))
    print("Cache populated at:", os.path.abspath(CACHE_DIR))


if __name__ == "__main__":
    main()
