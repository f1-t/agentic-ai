import argparse
import json
from pathlib import Path
from typing import Any

from ..agent import run_agent
from ..contracts import Snapshot as ContractSnapshot
from ..eval.policies import BASELINE_POLICIES, Decision
from ..eval.scenarios import Scenario
from ..eval.score import race_simulators, score_scenario
from ..sim import agent_tools, calibration, jobs
from ..sim.compute import ComputeProfile, load_profile
from ..sim.data import race_stints
from ..snapshot import RaceSnapshot


def load_snapshot(
    race_id: str,
    driver: str,
    lap: int,
    snapshot_path: str | Path | None = None,
    profile: ComputeProfile | None = None,
) -> tuple[ContractSnapshot, list[tuple[str, int]], RaceSnapshot]:
    if profile is None:
        profile = load_profile("local")

    if snapshot_path:
        raw = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        if "race" in raw and "laps_completed" in raw:
            contract_snap = ContractSnapshot.model_validate(raw)
            view = RaceSnapshot.from_contract(contract_snap)
        else:
            view = RaceSnapshot.from_dict(raw)
            contract_snap = agent_tools.contract_of(view)
        actual_stints = [(contract_snap.compound, contract_snap.laps_completed)]
        try:
            data = jobs.load_race(contract_snap.race.race_id, profile)
            actual_stints = [
                (c, laps)
                for c, laps, *_ in race_stints(data.frame, contract_snap.driver)
            ]
        except (KeyError, ValueError, OSError):
            actual_stints = [(contract_snap.compound, contract_snap.laps_completed)]
        return contract_snap, actual_stints, view

    data = jobs.load_race(race_id, profile)
    contract_snap = calibration.snapshot_at(data.frame, data.race, driver, lap)
    actual_stints = [(c, laps) for c, laps, *_ in race_stints(data.frame, driver)]
    view = RaceSnapshot.from_contract(contract_snap)
    return contract_snap, actual_stints, view


def format_table(rows: list[dict[str, Any]], opt_ref: dict[str, Any]) -> str:
    lines = [
        "=" * 105,
        "STRATEGY EVALUATION & COMPARISON SUMMARY",
        "=" * 105,
        f"Optimizer Reference Plan: {opt_ref.get('plan')} | Time: {opt_ref.get('time_s', 0.0):.2f}s | Stops: {opt_ref.get('num_stops')}",
        "-" * 105,
        f"{'Policy':<22} {'Status':<8} {'Stops':<6} {'First Pit':<10} {'Plan':<32} {'Time (s)':<11} {'Regret (s)':<10}",
        "-" * 105,
    ]
    ref_row = {
        "policy": "optimizer (ref)",
        "status": "ok",
        "num_stops": str(opt_ref.get("num_stops", "-")),
        "first_pit_lap": str(opt_ref.get("first_pit_lap", "-")),
        "plan": str(opt_ref.get("plan", "-")),
        "time_s": f"{opt_ref.get('time_s', 0.0):.2f}"
        if opt_ref.get("time_s") is not None
        else "-",
        "regret_s": "0.00",
    }
    all_rows = [ref_row] + [
        {
            "policy": r.get("policy", "-"),
            "status": r.get("status", "-"),
            "num_stops": str(r.get("num_stops", "-")),
            "first_pit_lap": str(r.get("first_pit_lap", "-")),
            "plan": str(r.get("plan", "-")),
            "time_s": f"{r.get('time_s', 0.0):.2f}"
            if r.get("time_s") is not None
            else "-",
            "regret_s": f"{r.get('regret_s', 0.0):+.2f}"
            if r.get("regret_s") is not None
            else "-",
        }
        for r in rows
    ]
    for row in all_rows:
        lines.append(
            f"{row['policy']:<22} {row['status']:<8} {row['num_stops']:<6} {row['first_pit_lap']:<10} {row['plan']:<32} {row['time_s']:<11} {row['regret_s']:<10}"
        )
    lines.append("=" * 105)
    return "\n".join(lines)


def run_pipeline(
    race_id: str = "2021_04_spain",
    driver: str = "HAM",
    lap: int = 0,
    snapshot_path: str | Path | None = None,
    question: str | None = None,
    run_llm_agent: bool = True,
    max_steps: int | None = None,
    max_stops: int = 2,
    seed: int = 0,
    n_samples: int | None = None,
    verbose: bool = True,
    profile_name: str | None = None,
) -> dict[str, Any]:
    profile = load_profile(profile_name)
    contract_snap, actual_stints, view = load_snapshot(
        race_id=race_id,
        driver=driver,
        lap=lap,
        snapshot_path=snapshot_path,
        profile=profile,
    )

    scenario_id = f"{contract_snap.race.race_id}:{contract_snap.driver}:{contract_snap.laps_completed:02d}"
    scenario = Scenario(
        scenario_id=scenario_id,
        race_id=contract_snap.race.race_id,
        driver=contract_snap.driver,
        lap=contract_snap.laps_completed,
        snapshot=contract_snap,
        actual_stints=actual_stints,
    )

    sim_for = race_simulators(profile)
    world = sim_for(scenario.race_id, "world")
    belief = sim_for(scenario.race_id, "belief")

    rec, trajectory = None, []
    if run_llm_agent:
        steps = max_steps or profile.agent.max_steps
        rec, trajectory = run_agent(
            view, question=question, verbose=verbose, max_steps=steps
        )

    policies = dict(BASELINE_POLICIES)
    if run_llm_agent and rec is not None:

        def make_agent_policy(submission, tool_count):
            def policy(sc, sim, stops_limit):
                dec = Decision(None, submission=submission, tool_calls=tool_count)
                dec.agent_decisions = True
                return dec

            policy.agent_decisions = True
            return policy

        policies = {
            "agent": make_agent_policy(rec, len(trajectory)),
            **BASELINE_POLICIES,
        }

    scored_rows = score_scenario(
        scenario,
        world,
        policies,
        belief=belief,
        seed=seed,
        n_samples=n_samples,
        max_stops=max_stops,
    )

    first_row = scored_rows[0] if scored_rows else {}
    opt_ref = {
        "plan": json.loads(first_row.get("opt_plan", "[]"))
        if first_row.get("opt_plan")
        else None,
        "num_stops": first_row.get("opt_num_stops"),
        "first_pit_lap": first_row.get("opt_first_pit_lap"),
        "time_s": first_row.get("opt_time_s"),
    }

    return {
        "scenario": scenario_id,
        "snapshot": contract_snap.model_dump(mode="json"),
        "view": view.to_dict(),
        "agent_recommendation": rec,
        "trajectory_steps": len(trajectory),
        "optimizer_reference": opt_ref,
        "results": scored_rows,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run end-to-end strategy evaluation across agent, baselines, and optimizer."
    )
    parser.add_argument(
        "positional_target",
        nargs="?",
        default=None,
        help="Optional snapshot file path or question text.",
    )
    parser.add_argument(
        "--race", default="2021_04_spain", help="Race identifier in cache."
    )
    parser.add_argument("--driver", default="HAM", help="Three-letter driver code.")
    parser.add_argument(
        "--lap", type=int, default=0, help="Laps completed at decision point."
    )
    parser.add_argument("--snapshot", help="Path to snapshot JSON file.")
    parser.add_argument("--question", help="Custom strategic question for agent.")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip agent and evaluate baselines only.",
    )
    parser.add_argument("--max-steps", type=int, help="Max tool steps for agent.")
    parser.add_argument(
        "--max-stops",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Max future pit stops.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for simulation draws."
    )
    parser.add_argument("--n-samples", type=int, help="Number of Monte Carlo samples.")
    parser.add_argument("--profile", help="Compute profile (local or sol).")
    parser.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose agent trajectory logging.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    snapshot_path = args.snapshot
    question = args.question

    if args.positional_target:
        if Path(args.positional_target).is_file():
            snapshot_path = args.positional_target
        elif not question:
            question = args.positional_target

    result = run_pipeline(
        race_id=args.race,
        driver=args.driver,
        lap=args.lap,
        snapshot_path=snapshot_path,
        question=question,
        run_llm_agent=not args.no_agent,
        max_steps=args.max_steps,
        max_stops=args.max_stops,
        seed=args.seed,
        n_samples=args.n_samples,
        verbose=not args.quiet,
        profile_name=args.profile,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    opt = result["optimizer_reference"]
    table_str = format_table(result["results"], opt)

    if result.get("agent_recommendation"):
        rec = result["agent_recommendation"]
        print("\n" + "=" * 60)
        print("AGENT DECISION SUMMARY")
        print("=" * 60)
        print(
            f"Action:       {rec.get('decision')} (reassess after {rec.get('reassess_after_laps')} laps)"
        )
        if rec.get("compound"):
            print(f"Compound:     {rec.get('compound')}")
        print(f"Confidence:   {rec.get('confidence')}")
        print(f"Rationale:    {rec.get('rationale')}")
        if rec.get("key_risks"):
            print(f"Key Risks:    {'; '.join(rec.get('key_risks'))}")

    print("\n" + table_str)


if __name__ == "__main__":
    main()
