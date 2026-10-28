import argparse
import json
from dataclasses import asdict

from . import (
    PolicyUnavailable,
    actual_strategy,
    fixed_one_stop,
    fixed_two_stop,
    greedy_crossover,
    optimize,
    score_plan,
)
from .legacy import load_legacy_context


def comparison(context, seeds, max_stops):
    snapshot, simulator = context.snapshot, context.simulator
    optimum = optimize(snapshot, simulator, seeds, max_stops)
    reference = optimum.evaluation
    policies = {
        "actual_strategy": lambda: actual_strategy(snapshot, context.actual_stints),
        "fixed_one_stop": lambda: fixed_one_stop(snapshot),
        "fixed_two_stop": lambda: fixed_two_stop(snapshot),
        "greedy_crossover": lambda: greedy_crossover(
            snapshot, simulator.pace, max_stops
        ),
    }
    rows = {}
    for name, policy in policies.items():
        try:
            plan = policy()
            if plan.num_stops > max_stops:
                raise PolicyUnavailable(
                    "This policy exceeds the optimizer's future-stop limit."
                )
        except PolicyUnavailable as error:
            rows[name] = {"status": "unavailable", "reason": str(error)}
            continue
        result = score_plan(snapshot, plan, simulator, seeds)
        rows[name] = {
            "status": "ok",
            **result.to_dict(snapshot),
            "regret_s": result.mean_time_s - reference.mean_time_s,
        }
    return {
        "schema_version": 1,
        "race": context.race,
        "driver": context.driver,
        "simulator": "legacy_deterministic_linear",
        "objective": "minimum_mean_remaining_time_over_supplied_seeds",
        "max_future_stops": max_stops,
        "snapshot": asdict(snapshot),
        "optimizer": {
            **reference.to_dict(snapshot),
            "candidates_evaluated": optimum.candidates_evaluated,
        },
        "baselines": rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference F1 policies using the offline race cache."
    )
    parser.add_argument("--driver", default="HAM")
    parser.add_argument(
        "--lap",
        type=int,
        default=0,
        help="Completed laps before the next pit decision.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-stops", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    try:
        context = load_legacy_context(args.driver, args.lap)
        result = comparison(context, tuple(args.seeds), args.max_stops)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Baseline comparison failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
