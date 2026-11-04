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
    # added for the simulator bridge (src/sim/optimizer_bridge.py, docs/rules.md)
    parser.add_argument(
        "--scorer",
        choices=("callback", "batch"),
        default="callback",
        help="callback: the legacy simulator per plan and seed; "
        "batch: every plan in one Simulator.simulate_batch call (seeds must be 0..n-1).",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run live agent along with the optimizer and baselines on world simulator.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run integrated evaluation pipeline on the calibrated world simulator.",
    )
    parser.add_argument(
        "--snapshot",
        help="Path to snapshot JSON file (for pipeline / agent comparison).",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Render results as formatted table instead of JSON.",
    )
    args = parser.parse_args()
    if args.agent or args.pipeline or args.snapshot:
        from .runner import main as run_pipeline_main

        run_args = [
            "--driver",
            args.driver,
            "--lap",
            str(args.lap),
            "--max-stops",
            str(args.max_stops),
            "--seed",
            str(args.seeds[0] if args.seeds else 0),
        ]
        if not args.agent:
            run_args.append("--no-agent")
        if args.snapshot:
            run_args.extend(["--snapshot", args.snapshot])
        if not args.table:
            run_args.append("--json")
        run_pipeline_main(run_args)
        return
    try:
        context = load_legacy_context(args.driver, args.lap)
        if args.scorer == "batch":
            from ..sim.optimizer_bridge import batch_comparison

            result = batch_comparison(context, tuple(args.seeds), args.max_stops)
        else:
            result = comparison(context, tuple(args.seeds), args.max_stops)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Baseline comparison failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
