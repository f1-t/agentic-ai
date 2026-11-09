"""python -m src.eval {scenarios, reference, score, report, jobs}. See docs/eval.md."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .policies import resolve
from .reference import REFERENCE_DIR, ReferenceMissing, ReferenceStore, build_references, stale_references, \
    summarize_references
from .scenarios import ScenarioSet, build_set
from .score import install_job_runner, race_simulators, score_set, summarize, summarize_agreeing, \
    summarize_near_ties, to_markdown


def _config(path):
    if path is None:
        return None
    from ..sim import SimConfig
    return SimConfig.from_file(path)


def _sim_configs(p):
    p.add_argument("--belief-config", help="SimConfig yaml or json the policies plan on "
                                           "(default configs/sim/belief.yaml)")
    p.add_argument("--world-config", "--config", dest="world_config",
                   help="SimConfig yaml or json for the reference and the scoring (default: the "
                        "profile's; configs/sim/world.yaml turns ghost traffic on)")
    p.add_argument("--world-configs", nargs="+",
                   help="optional (model-disagreement flag): two or more world configs; the first "
                        "scores, every one plans a reference, and rows gain ref_models_agree")


def _worlds(ap, args):
    """(world config path, alternative world config paths) from --world-config / --world-configs."""
    if args.world_configs is None:
        return args.world_config, []
    if args.world_config is not None:
        ap.error("give --world-config or --world-configs, not both")
    if len(args.world_configs) < 2:
        ap.error("--world-configs needs at least two configs")
    return args.world_configs[0], args.world_configs[1:]


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
    p.add_argument("--eval-set", help="the registered eval split (default data/eval_set.json): when it "
                                      "exists, races outside it are refused and the set is a report set")
    p.add_argument("--development", action="store_true",
                   help="sample a development (debug) set even when the eval split exists: no race is "
                        "refused, the split's digest is still recorded; never a source of report numbers")
    p.add_argument("--profile")

    p = sub.add_parser("reference", help="compute and cache the world optimizer's reference per scenario")
    p.add_argument("--scenarios", help="scenario set json (not needed with --check)")
    p.add_argument("--world-config", "--config", dest="world_config",
                   help="SimConfig yaml or json of the world (default: the profile's)")
    p.add_argument("--profile")
    p.add_argument("--hindsight", action="store_true",
                   help="fix every draw's SC and VSC laps to the driver's recorded ones")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, help="default: the profile's sim.n_samples")
    p.add_argument("--max-stops", type=int, choices=(0, 1, 2), help="default: the profile's sim.max_stops")
    p.add_argument("--reference-dir", default=str(REFERENCE_DIR), help="default data/references")
    p.add_argument("--check", action="store_true",
                   help="list cached references whose sim_code_version differs from the current one "
                        "or whose format is older than the current one")

    p = sub.add_parser("score", help="score policies against the optimizer on a scenario set")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--policies", nargs="*", default=["baselines"],
                   help="baselines, agent, or single baseline names")
    p.add_argument("--recorded", nargs="*", default=[], help="json files of saved decisions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, help="default: the profile's sim.n_samples")
    p.add_argument("--max-stops", type=int, choices=(0, 1, 2), help="default: the profile's sim.max_stops")
    _sim_configs(p)
    p.add_argument("--profile")
    p.add_argument("--out", required=True, help="per-row csv; the summary goes beside it")
    p.add_argument("--reference-dir", default=str(REFERENCE_DIR),
                   help="where cached references are looked up (default data/references)")
    p.add_argument("--require-reference", action="store_true",
                   help="a scenario without a cached reference is an error (report runs)")

    p = sub.add_parser("report", help="print the result tables of a scored csv")
    p.add_argument("rows")

    p = sub.add_parser("jobs", help="run sim.jobs with the eval runner installed: "
                                    "jobs --scenarios PATH [...] -- <sim.jobs arguments>")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--policies", nargs="*", default=["baselines"])
    p.add_argument("--recorded", nargs="*", default=[])
    _sim_configs(p)
    p.add_argument("--reference-dir", default=str(REFERENCE_DIR),
                   help="where the runner looks up cached references (default data/references)")
    p.add_argument("--require-reference", action="store_true",
                   help="a scenario without a cached reference fails its chunk (report runs)")

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
                        max_laps_down=args.max_laps_down, max_per_race=args.max_per_race, seed=args.seed,
                        eval_set=args.eval_set, development=args.development)
        path = out.save(args.out)
        kind = "development set" if out.development else "report set"
        print(f"{out.version} ({kind}): {len(out.items)} scenarios, {len(out.skipped)} skipped, "
              f"digest {out.digest[:12]} -> {path}")
        return 0

    if args.command == "reference":
        if args.check:
            stale = stale_references(args.reference_dir)
            from ..sim.calibration import SIM_CODE_VERSION
            from .reference import REFERENCE_FORMAT
            print(f"{len(stale)} cached references with a sim_code_version other than {SIM_CODE_VERSION} "
                  f"or a format older than {REFERENCE_FORMAT}" + (":" if stale else ""))
            for path, meta in stale:
                print(f"  {path}  sim_code_version {meta.get('sim_code_version')}  format {meta.get('format', 1)}")
            return 0
        if args.scenarios is None:
            ap.error("reference needs --scenarios (or --check)")
        from ..sim.compute import load_profile
        profile = load_profile(args.profile)
        scenario_set = ScenarioSet.load(args.scenarios)
        written = build_references(scenario_set, race_simulators(profile, None, _config(args.world_config)),
                                   seed=args.seed, n_samples=args.n_samples or profile.sim.n_samples,
                                   max_stops=profile.sim.max_stops if args.max_stops is None else args.max_stops,
                                   hindsight=args.hindsight, root=args.reference_dir)
        for path, refs in written:
            print(f"{scenario_set.version}: {len(refs)} references -> {path}")
            print(to_markdown(pd.DataFrame([summarize_references(refs)]), digits=3))
        return 0

    if args.command == "score":
        from ..sim.compute import load_profile
        profile = load_profile(args.profile)
        scenario_set = ScenarioSet.load(args.scenarios)
        print(f"scoring {scenario_set.version}, a {scenario_set.kind}"
              + (": not a source of report numbers" if scenario_set.development else ": report numbers"))
        world, alts = _worlds(ap, args)
        belief = _config(args.belief_config)
        sims = race_simulators(profile, belief, _config(world))
        store = ReferenceStore(args.reference_dir, require=args.require_reference)
        try:
            rows = score_set(scenario_set, sims,
                             resolve(args.policies, args.recorded, set_name=scenario_set.version, profile=profile),
                             seed=args.seed, n_samples=args.n_samples or profile.sim.n_samples,
                             max_stops=profile.sim.max_stops if args.max_stops is None else args.max_stops,
                             ref_sims=[race_simulators(profile, belief, _config(a)) for a in alts],
                             references=store)
        except ReferenceMissing as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        table = summarize(rows)
        table.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
        print(to_markdown(table))
        agreeing = summarize_agreeing(rows)
        if agreeing is not None:
            agreeing.to_csv(out.with_name(out.stem + "_summary_agree.csv"), index=False)
            print("\nRows where the reference models agree:\n" + to_markdown(agreeing))
        return 0

    if args.command == "report":
        rows = pd.read_csv(args.rows)
        print(to_markdown(summarize(rows)))
        agreeing = summarize_agreeing(rows)
        if agreeing is not None:
            print("\nRows where the reference models agree:\n" + to_markdown(agreeing))
        ties = summarize_near_ties(rows)
        if ties is not None:
            print("\nNear-tie view (information only; optimal set: plans within 2 SE of the reference):\n"
                  + to_markdown(ties))
        return 0

    from ..sim import jobs
    world, alts = _worlds(ap, args)
    install_job_runner(args.scenarios, args.policies, args.recorded, belief_config=args.belief_config,
                       world_config=world, ref_world_configs=alts, reference_dir=args.reference_dir,
                       require_reference=args.require_reference)
    return jobs.main(rest)


if __name__ == "__main__":
    sys.exit(main())
