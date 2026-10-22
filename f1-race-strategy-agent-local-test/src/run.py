"""CLI entry point for the F1 strategy agent.

    python -m src.run "Hamilton starts on SOFT. One-stop or two-stop, and when should he first pit?"

If no question is given, a default one about Hamilton in the 2021 Spanish GP is used.
Selects Anthropic or a local Hugging Face model via .env (see .env.example).
"""
import sys

from .agent import run_agent

DEFAULT_Q = (
    "Hamilton starts the 2021 Spanish Grand Prix on the SOFT tyre. "
    "Should he run a one-stop or a two-stop strategy, and on which lap "
    "should he make his first pit stop? Justify it with simulated race times."
)


def main():
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_Q
    print(f"Question: {question}\n")
    print("Agent working (plan -> call tools -> observe -> refine):\n")
    rec, trajectory = run_agent(question, verbose=True)

    print("\n" + "=" * 60)
    if rec is None:
        print("Agent did not submit a recommendation.")
        sys.exit(1)
    print("FINAL RECOMMENDATION")
    print(f"  Stops:         {rec.get('num_stops')}")
    print(f"  First pit lap: {rec.get('first_pit_lap')}")
    print(f"  Stint plan:    {rec.get('stint_plan')}")
    if rec.get("total_time_s") is not None:
        print(f"  Sim time (s):  {rec.get('total_time_s')}")
    print(f"  Rationale:     {rec.get('rationale')}")
    print("=" * 60)
    print(f"\n({len(trajectory)} tool calls made.)")


if __name__ == "__main__":
    main()
