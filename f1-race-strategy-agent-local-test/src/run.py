"""CLI for a snapshot-based F1 strategy decision."""
import json
import sys

from .agent import run_agent
from .snapshot import RaceSnapshot


def demo_snapshot():
    return RaceSnapshot(
        season=2021, event="Spanish Grand Prix", lap=27, total_laps=66,
        driver="HAM", position=2, compound="SOFT", tyre_age_laps=27,
        gap_ahead_s=1.4, gap_behind_s=3.8, track_status="GREEN",
        rain_probability=0.05,
        nearby_cars=[{"driver": "VER", "position": 1, "gap_s": -1.4}, {"driver": "BOT", "position": 3, "gap_s": 3.8}],
    )


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            snapshot = RaceSnapshot.from_dict(json.load(f))
    else:
        snapshot = demo_snapshot()
    print("Snapshot:\n" + json.dumps(snapshot.to_dict(), indent=2))
    print("\nAgent working (snapshot -> evidence -> uncertainty-aware decision):\n")
    rec, trajectory = run_agent(snapshot, verbose=True)
    print("\n" + "=" * 60)
    if rec is None:
        print("Agent did not submit a recommendation.")
        sys.exit(1)
    print("FINAL RECOMMENDATION")
    print(json.dumps(rec, indent=2))
    print(f"\n({len(trajectory)} logged tool calls.)")


if __name__ == "__main__":
    main()
