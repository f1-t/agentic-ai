"""Concrete test case for the F1 strategy agent (proposal Section 4).

Runs the agent on one question, then checks its recommendation against the
deterministic optimum (data/ground_truth.json):

  PASS if the recommended strategy is valid AND its simulated race time is
  within tolerance_pct of the optimum.

Select a model backend in .env. Race data is read from the committed offline cache.

    python tests/test_case.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import run_agent      # noqa: E402
from src.grade import grade          # noqa: E402

QUESTION = (
    "Hamilton starts the 2021 Spanish Grand Prix on the SOFT tyre. "
    "Should he run a one-stop or a two-stop strategy, and how should the "
    "stints be split, to minimise total race time? Justify it with simulated times."
)


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.json")) as f:
        gt = json.load(f)

    print("=" * 64)
    print("TEST CASE: 2021 Spanish GP — Hamilton strategy")
    print("=" * 64)
    print(f"INPUT (question to agent):\n  {QUESTION}\n")
    print("EXPECTED:")
    print(f"  a valid strategy within {gt['tolerance_pct']}% of the optimum "
          f"({gt['optimal_total_time_s']}s).\n")

    print("AGENT TRAJECTORY:")
    rec, trajectory = run_agent(QUESTION, verbose=True)

    print("\nACTUAL (agent recommendation):")
    if rec is None:
        print("  <no recommendation submitted>\nRESULT: FAIL")
        sys.exit(1)
    print(f"  num_stops     = {rec.get('num_stops')}")
    print(f"  first_pit_lap = {rec.get('first_pit_lap')}")
    print(f"  stint_plan    = {rec.get('stint_plan')}")
    print(f"  rationale     = {rec.get('rationale')}")

    g = grade(rec, gt)
    print("\nGRADE:")
    if not g["valid"]:
        print(f"  invalid plan: {g['reason']}\nRESULT: FAIL")
        sys.exit(1)
    print(f"  agent sim time = {g['agent_time_s']}s   optimum = {g['optimal_time_s']}s")
    print(f"  gap            = {g['gap_pct']}%   (tolerance {g['tolerance_pct']}%)")
    print(f"  ({g['agent_num_stops']}-stop, first pit lap {g['agent_first_pit_lap']}, {len(trajectory)} tool calls)")
    print("\nRESULT:", "PASS" if g["passed"] else "FAIL")
    sys.exit(0 if g["passed"] else 1)


if __name__ == "__main__":
    main()
