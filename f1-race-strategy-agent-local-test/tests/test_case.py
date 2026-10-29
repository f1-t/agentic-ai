"""Concrete test case for the F1 strategy agent (proposal Section 4), on the snapshot loop.

Builds Hamilton's lap-0 snapshot of the 2021 Spanish GP from the committed cache, runs the
agent on it, and grades the submitted decision with src.grade.grade_decision: the PIT or
STAY_OUT decision is expanded into an action plus an optimizer continuation and scored by
the eval's world simulator against the optimizer's plan from the same snapshot.

  PASS if the agent submits a valid decision within its step and tool-call budget.

The regret against the lap-0 optimizer is reported, not asserted (Rishabh's call whether
the graded test should assert more). Select a model backend in .env. This file is not
collected by pytest (pytest.ini python_files); run it directly:

    python tests/test_case.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import run_agent                  # noqa: E402
from src.grade import grade_decision             # noqa: E402
from src.sim import calibration, jobs            # noqa: E402
from src.sim.compute import load_profile         # noqa: E402

RACE_ID, DRIVER, LAP = "2021_04_spain", "HAM", 0
QUESTION = ("Hamilton starts the 2021 Spanish Grand Prix on the SOFT tyre. Decide the immediate action "
            "from this snapshot: pit at the end of this lap (and onto which compound) or stay out, and for "
            "how many laps before reassessing. Justify it with the tools' estimates.")


def main():
    profile = load_profile()
    data = jobs.load_race(RACE_ID, profile)
    snapshot = calibration.snapshot_at(data.frame, data.race, DRIVER, LAP)

    print("=" * 64)
    print("TEST CASE: 2021 Spanish GP, Hamilton, lap-0 decision")
    print("=" * 64)
    print(f"INPUT (question to agent):\n  {QUESTION}\n")
    print(f"SNAPSHOT: lap {snapshot.laps_completed}/{snapshot.race.total_laps}, {snapshot.compound} "
          f"age {snapshot.tyre_age}\n")
    print("EXPECTED:\n  a valid PIT or STAY_OUT submission within the agent's budget.\n")

    print("AGENT TRAJECTORY:")
    rec, trajectory = run_agent(snapshot, QUESTION, verbose=True, max_steps=profile.agent.max_steps)

    print("\nACTUAL (agent decision):")
    if rec is None:
        print("  <no decision submitted within the budget>\nRESULT: FAIL")
        sys.exit(1)
    for key in ("decision", "compound", "reassess_after_laps", "confidence", "rationale"):
        print(f"  {key:<20}= {rec.get(key)}")

    g = grade_decision(snapshot, rec, profile=profile)
    print("\nGRADE:")
    if not g["valid"]:
        print(f"  invalid decision: {g['reason']}\nRESULT: FAIL")
        sys.exit(1)
    print(f"  expanded plan   = {g['plan']}  ({g['num_stops']}-stop, first pit lap {g['first_pit_lap']})")
    print(f"  optimizer plan  = {g['opt_plan']}  ({g['opt_num_stops']}-stop, first pit lap {g['opt_first_pit_lap']})")
    print(f"  regret          = {g['regret_s']:.2f} s ({g['regret_pct']:.3f}%), reported, not asserted")
    print(f"  ({len(trajectory)} tool calls)")
    print("\nRESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
