"""python -m src.eval {scenarios, score, report, jobs}. See docs/eval.md."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .policies import resolve
from .scenarios import ScenarioSet, build_set
from .score import install_job_runner, race_simulators, score_set, summarize, to_markdown


def _config(path):
    if path is None:
        return None
    from ..sim import SimConfig
    return SimConfig.from_file(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.eval", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scenarios", help="sample and write a versioned scenario set")
    p.add_argument("--races", nargs="+", default=["2021_04_spain"])
    p.add_argument("--version", required=True, help="e.g. v1; never reuse a published version")
    p.add_argument("--out", required=True)
    p.add_argument("--drivers", nargs="+", help="limit to these driver codes")
    p.add_argument("--max-laps-down", type=int, default=2)
    p.add_argument("--max-per-race", type=int, help="seeded subsample of each race's decision points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile")

    p = sub.add_parser("score", help="score policies against the optimizer on a scenario set")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--policies", nargs="*", default=["baselines"],
                   help="baselines, agent, or single baseline names")
    p.add_argument("--recorded", nargs="*", default=[], help="json files of saved decisions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, help="default: the profile's sim.n_samples")
    p.add_argument("--max-stops", type=int, choices=(0, 1, 2), help="default: the profile's sim.max_stops")
    p.add_argument("--config", help="SimConfig yaml or json for the scoring simulator")
    p.add_argument("--profile")
    p.add_argument("--out", required=True, help="per-row csv; the summary goes beside it")

    p = sub.add_parser("report", help="print the result tables of a scored csv")
    p.add_argument("rows")

    p = sub.add_parser("jobs", help="run sim.jobs with the eval runner installed: "
                                    "jobs --scenarios PATH [...] -- <sim.jobs arguments>")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--policies", nargs="*", default=["baselines"])
    p.add_argument("--recorded", nargs="*", default=[])
    p.add_argument("--config")

    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    args = ap.parse_args(argv)
    if args.command == "jobs" and not rest:
        ap.error("jobs needs the sim.jobs arguments after --, e.g. -- eval_chunk --manifest M --task-id 0 --out O")

    if args.command == "scenarios":
        from ..sim.compute import load_profile
        out = build_set(args.races, args.version, profile=load_profile(args.profile), drivers=args.drivers,
                        max_laps_down=args.max_laps_down, max_per_race=args.max_per_race, seed=args.seed)
        path = out.save(args.out)
        print(f"{out.version}: {len(out.items)} scenarios, {len(out.skipped)} skipped, digest {out.digest[:12]} "
              f"-> {path}")
        return 0

    if args.command == "score":
        from ..sim.compute import load_profile
        profile = load_profile(args.profile)
        scenario_set = ScenarioSet.load(args.scenarios)
        rows = score_set(scenario_set, race_simulators(profile, _config(args.config)),
                         resolve(args.policies, args.recorded), seed=args.seed,
                         n_samples=args.n_samples or profile.sim.n_samples,
                         max_stops=profile.sim.max_stops if args.max_stops is None else args.max_stops)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        table = summarize(rows)
        table.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
        print(to_markdown(table))
        return 0

    if args.command == "report":
        print(to_markdown(summarize(pd.read_csv(args.rows))))
        return 0

    from ..sim import jobs
    install_job_runner(args.scenarios, args.policies, args.recorded, config=_config(args.config))
    return jobs.main(rest)


if __name__ == "__main__":
    sys.exit(main())
