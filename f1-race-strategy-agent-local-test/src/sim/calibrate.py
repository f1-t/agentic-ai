"""CLI for the calibration check.

    python -m src.sim.calibrate --season 2021 --gp Spain --pit-loss 22      # a number, or observed (default)
    python -m src.sim.calibrate --degradation linear

Also prints the team-mate and finishing position checks and writes them to
reports/calibration/<race_id>_pairs.csv and <race_id>_positions.csv.

Mid-race replay, per race and then over every race replayed so far:

    python -m src.sim.calibrate midrace --season 2021 --gp Spain [--windows ...] [--laps ...]
    python -m src.sim.calibrate midrace-aggregate reports/calibration/*_midrace.csv

The first writes reports/calibration/<race_id>_midrace.csv and the per driver appendix
<race_id>_midrace_by_driver.csv. Its pit loss is the observed one, measured as the plain
command does, unless --pit-loss gives a number; both values head each csv as `# key: value`
lines (read with comment="#"). The second prints the summary pivot and writes
reports/calibration/midrace_summary.csv.
"""
import argparse
import json
import os
import sys

import pandas as pd

from ..contracts import RaceParams
from . import calibration
from .config import SimConfig
from .data import lap_frame, load_session, practice_frame, session_results
from .engine import Simulator

DEFAULT_PIT_LOSS_S = 22.0
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "calibration")
PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")
PIVOT_METRICS = ["n", "n_nan", "mae_s", "bias_s", "mae_offset_s", "within_2_rate_sim", "mae_pos_sim",
                 "within_2_rate_real", "mae_pos_real"]


def _race_args(ap):
    ap.add_argument("--season", type=int, default=2021)
    ap.add_argument("--gp", default="Spain")
    ap.add_argument("--pit-loss", default="observed",
                    help=f'seconds, or "observed" to measure it from a first pass at {DEFAULT_PIT_LOSS_S}')
    ap.add_argument("--min-stint", type=int, default=10)
    ap.add_argument("--degradation", default=SimConfig().degradation.name)
    ap.add_argument("--config", help="json or yaml SimConfig file, overrides --degradation")
    ap.add_argument("--report-dir", default=REPORT_DIR, help="where the csvs go")


def _race_setup(args):
    """Session, frame, config and race. The race's pit loss is args.pit_loss, or DEFAULT_PIT_LOSS_S
    when that is "observed"; _pit_loss then measures the observed value."""
    session = load_session(args.season, args.gp)
    frame = lap_frame(session.laps)
    cfg = SimConfig.from_file(args.config) if args.config else SimConfig(degradation={"name": args.degradation})
    pit_loss = DEFAULT_PIT_LOSS_S if args.pit_loss == "observed" else float(args.pit_loss)
    race = RaceParams(
        race_id=f"{args.season}_{args.gp}", total_laps=int(session.total_laps), pit_loss_s=pit_loss,
        min_stint_laps=args.min_stint, compounds=sorted(frame["compound"].dropna().unique()))
    return session, frame, cfg, race


def _pit_loss(args, frame, cfg, race):
    """(pit loss to use, observed pit loss). The observed value comes from a first full-window
    run() at DEFAULT_PIT_LOSS_S, as the calibrate command does; a number in --pit-loss overrides it."""
    first = race.model_copy(update={"pit_loss_s": DEFAULT_PIT_LOSS_S})
    observed = calibration.run(Simulator(frame, cfg), frame, first)[1]["observed_pit_loss_s"]
    return (observed if args.pit_loss == "observed" else float(args.pit_loss)), observed


def _write_csv(df, path, header, **kw):
    """CSV with `# key: value` lines before the table; read it back with pd.read_csv(path, comment="#")."""
    with open(path, "w") as f:
        f.writelines(f"# {k}: {v}\n" for k, v in header.items())
        df.to_csv(f, **kw)


def _practice(season, gp, long_run=None):
    """FP1-FP3 long-run laps, for the pre_race and to_snapshot fit windows."""
    return practice_frame({kind: load_session(season, gp, kind).laps for kind in PRACTICE_SESSIONS},
                          long_run=long_run)


def midrace_main(argv):
    ap = argparse.ArgumentParser(prog="calibrate midrace", description="mid-race replay calibration")
    _race_args(ap)
    ap.add_argument("--windows", nargs="+", default=list(calibration.MIDRACE_WINDOWS),
                    choices=list(calibration.MIDRACE_WINDOWS))
    ap.add_argument("--laps", nargs="+", type=int, help="snapshot laps k, default calibration.default_laps_k")
    args = ap.parse_args(argv)

    session, frame, cfg, race = _race_setup(args)
    used, observed = _pit_loss(args, frame, cfg, race)
    race = race.model_copy(update={"pit_loss_s": used})
    header = {"pit_loss_s": used, "observed_pit_loss_s": observed,
              "pit_loss_source": "observed" if args.pit_loss == "observed" else "--pit-loss"}
    print(f"pit loss used: {used} ({header['pit_loss_source']}), observed: {observed}")
    practice = _practice(args.season, args.gp, cfg.long_run) if set(args.windows) - {"full"} else None

    def factory(window):
        return Simulator(frame, cfg.model_copy(update={"fit_window": window}), practice)

    rows = calibration.midrace(factory, frame, race, laps_k=args.laps, windows=args.windows,
                               results=session_results(session))
    os.makedirs(args.report_dir, exist_ok=True)
    path = os.path.normpath(os.path.join(args.report_dir, f"{race.race_id}_midrace.csv"))
    _write_csv(rows, path, header, index=False)
    print(f"{len(rows)} rows, k = {sorted(rows['k'].unique().tolist())}, written to {path}")
    appendix = os.path.normpath(os.path.join(args.report_dir, f"{race.race_id}_midrace_by_driver.csv"))
    _write_csv(calibration.by_driver(rows), appendix, header)
    print(f"per driver appendix written to {appendix}")
    _print_pivot(calibration.aggregate(rows))


def _print_pivot(summary):
    pivot = summary[PIVOT_METRICS].unstack("window")
    for metric in PIVOT_METRICS:
        print(f"\n{metric} (horizon bucket x window):\n", pivot[metric].round(3).to_string())


def midrace_aggregate_main(argv):
    ap = argparse.ArgumentParser(prog="calibrate midrace-aggregate", description="summary over midrace csvs")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=os.path.join(REPORT_DIR, "midrace_summary.csv"))
    args = ap.parse_args(argv)

    rows = pd.concat([pd.read_csv(f, comment="#") for f in args.files], ignore_index=True)
    summary = calibration.aggregate(rows)
    print(f"{len(rows)} rows from {rows['race_id'].nunique()} race(s)")
    _print_pivot(summary)
    print("\nmae_offset_s with race-cluster bootstrap 95% CI:\n",
          summary[["n_races", "mae_offset_s", "mae_offset_ci_lo", "mae_offset_ci_hi"]].round(2).to_string())
    print("\nteam-mate sign agreement on remaining time:\n", calibration.midrace_pairs(rows).round(3).to_string())
    print("\nper race:\n", calibration.per_race(rows).round(3).to_string())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    summary.to_csv(args.out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "midrace":
        return midrace_main(argv[1:])
    if argv and argv[0] == "midrace-aggregate":
        return midrace_aggregate_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2021)
    ap.add_argument("--gp", default="Spain")
    ap.add_argument("--pit-loss", default="observed",
                    help='seconds, or "observed" to measure it from a first pass at 22.0 and rerun with it')
    ap.add_argument("--min-stint", type=int, default=10)
    ap.add_argument("--degradation", default=SimConfig().degradation.name)
    ap.add_argument("--config", help="json or yaml SimConfig file, overrides --degradation")
    ap.add_argument("--out", help="write per driver csv here")
    ap.add_argument("--report-dir", default=REPORT_DIR, help="where the pairs and positions csvs go")
    args = ap.parse_args(argv)

    session = load_session(args.season, args.gp)
    frame = lap_frame(session.laps)
    cfg = SimConfig.from_file(args.config) if args.config else SimConfig(degradation={"name": args.degradation})

    def make_race(pit_loss_s):
        return RaceParams(
            race_id=f"{args.season}_{args.gp}", total_laps=int(session.total_laps), pit_loss_s=pit_loss_s,
            min_stint_laps=args.min_stint, compounds=sorted(frame["compound"].dropna().unique()))

    sim = Simulator(frame, cfg)
    if args.pit_loss == "observed":
        race = make_race(DEFAULT_PIT_LOSS_S)
        first = calibration.run(sim, frame, race)[1]
        measured = first["observed_pit_loss_s"]
        print(f"first pass at pit loss {DEFAULT_PIT_LOSS_S}: summary {json.dumps(first)}")
        print(f"observed pit loss {measured}, rerunning with it")
        race = make_race(measured)
    else:
        race = make_race(float(args.pit_loss))
    table, summary, by_kind = calibration.run(sim, frame, race)
    print(f"pit loss used: {race.pit_loss_s}")

    print(table.round(1).to_string())
    print("\nsummary:", json.dumps(summary, indent=2))
    print("\nresidual (actual - predicted) by lap type:\n", by_kind.to_string())
    print("observed pit loss (s):", summary["observed_pit_loss_s"])
    print("observed sc factor:", summary["observed_sc_factor"], " vsc factor:", summary["observed_vsc_factor"])
    if args.out:
        table.to_csv(args.out)

    pairs, pair_summary = calibration.pairwise(sim, frame, race)
    pos, pos_summary = calibration.positions(sim, frame, race, session_results(session))
    print("\nteam-mate pairs (gap = first driver minus second):\n", pairs.round(2).to_string())
    print("\npair summary:", json.dumps(pair_summary, indent=2))
    print("\nfinishing positions:\n", pos.round(1).to_string())
    print("\nposition summary:", json.dumps(pos_summary, indent=2))
    os.makedirs(args.report_dir, exist_ok=True)
    pairs.to_csv(os.path.join(args.report_dir, f"{race.race_id}_pairs.csv"))
    pos.to_csv(os.path.join(args.report_dir, f"{race.race_id}_positions.csv"))


if __name__ == "__main__":
    main()
