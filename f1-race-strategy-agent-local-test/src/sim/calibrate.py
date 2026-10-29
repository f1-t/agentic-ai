"""CLI for the calibration check.

    python -m src.sim.calibrate --season 2021 --gp Spain --pit-loss 22      # a number, or observed (default)
    python -m src.sim.calibrate --degradation linear

Also prints the team-mate and finishing position checks and writes them to
reports/calibration/<race_id>_pairs.csv and <race_id>_positions.csv.

    python -m src.sim.calibrate --pace practice      # positions with a pace term (pace.py)
    python -m src.sim.calibrate --pace-table         # the gate under every pace term, both orders

--pace-table also writes <race_id>_pace_gate.csv and <race_id>_pace_offsets.csv, every row
with config_hash and sim_code_version as below.

Mid-race replay, per race and then over every race replayed so far:

    python -m src.sim.calibrate midrace --season 2021 --gp Spain [--windows ...] [--laps ...]
    python -m src.sim.calibrate midrace-aggregate reports/calibration/*_midrace.csv

The first writes reports/calibration/<race_id>_midrace.csv and the per driver appendix
<race_id>_midrace_by_driver.csv. Its pit loss is the observed one, measured as the plain
command does, unless --pit-loss gives a number. Every row of both csvs carries pit_loss_s,
pit_loss_source ("observed" or "--pit-loss"), config_hash (calibration.config_hash, a hash
of the physics config) and sim_code_version (calibration.SIM_CODE_VERSION, a hash of the
src/sim and src/contracts.py source). The second prints the summary pivot and writes
reports/calibration/midrace_summary.csv, every row with the same four columns taken from
the input rows (calibrate._provenance); it refuses rows with more than one value of
either version column unless --allow-mixed is given.

Ghost-car traffic (traffic.py), off against on, with the pace term in the engine:

    python -m src.sim.calibrate traffic --season 2021 --gp Spain --pace practice

writes reports/calibration/<race_id>_traffic.csv, one csv with a `table` column (by_kind,
drivers, pairs, gate, rejoin, placement) and a `traffic` column (none, ghost), stamped
like the midrace csvs plus `pace`; config_hash is the arm's own.

Calibrators side by side (calibrators.py), per cadence N:

    python -m src.sim.calibrate calibrators --season 2021 --gp Spain [--calibrators none shrinkage] [--every 1 none]

writes reports/calibration/<race_id>_calibrators.csv, <race_id>_calibrator_factors.csv and
<race_id>_selection.csv (calibrators.selection_table per cadence, with --note on every row),
named by the canonical race id (2021_04_spain).
"""
import argparse
import json
import os
import sys

import pandas as pd

from ..contracts import RaceParams
from . import calibration, pace
from .config import SimConfig
from .data import lap_frame, load_session, practice_frame, session_results
from .engine import Simulator

DEFAULT_PIT_LOSS_S = 22.0
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "calibration")
PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")
VERSION_COLUMNS = ("config_hash", "sim_code_version")
PROVENANCE_COLUMNS = ("pit_loss_s", "pit_loss_source", *VERSION_COLUMNS)
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


def _stamp(df, stamp):
    """`df` with the run's pit_loss_s, pit_loss_source, config_hash and sim_code_version as plain columns."""
    return df.assign(**stamp)


def _provenance(rows):
    """The stamp columns of stamped `rows` for a summary over them: the value when the rows
    share one, the sorted distinct values joined by a space when they do not (pit loss differs
    per race; the version columns only under --allow-mixed), "missing" for a column the rows
    lack."""
    out = {}
    for col in PROVENANCE_COLUMNS:
        values = sorted(set(rows[col].dropna().astype(str))) if col in rows else []
        if col in rows and len(values) == 1:
            out[col] = rows[col].dropna().iloc[0]
        else:
            out[col] = " ".join(values) if values else "missing"
    return out


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
    stamp = {"pit_loss_s": used, "pit_loss_source": "observed" if args.pit_loss == "observed" else "--pit-loss",
             "config_hash": calibration.config_hash(cfg), "sim_code_version": calibration.SIM_CODE_VERSION}
    print(f"pit loss used: {used} ({stamp['pit_loss_source']}), observed: {observed}; "
          f"config_hash {stamp['config_hash']}, sim_code_version {stamp['sim_code_version']}")
    practice = _practice(args.season, args.gp, cfg.long_run) if set(args.windows) - {"full"} else None

    def factory(window):
        return Simulator(frame, cfg.model_copy(update={"fit_window": window}), practice)

    rows = calibration.midrace(factory, frame, race, laps_k=args.laps, windows=args.windows,
                               results=session_results(session))
    os.makedirs(args.report_dir, exist_ok=True)
    path = os.path.normpath(os.path.join(args.report_dir, f"{race.race_id}_midrace.csv"))
    _stamp(rows, stamp).to_csv(path, index=False)
    print(f"{len(rows)} rows, k = {sorted(rows['k'].unique().tolist())}, written to {path}")
    appendix = os.path.normpath(os.path.join(args.report_dir, f"{race.race_id}_midrace_by_driver.csv"))
    _stamp(calibration.by_driver(rows), stamp).to_csv(appendix)
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
    ap.add_argument("--allow-mixed", action="store_true",
                    help="aggregate rows from different config_hash or sim_code_version values "
                         "(refused by default)")
    args = ap.parse_args(argv)

    rows = pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True)
    for col in VERSION_COLUMNS:
        # a csv without the column counts as one version; its rows read as NaN when mixed with newer csvs
        versions = sorted(map(str, rows[col].unique())) if col in rows else ["missing"]
        if len(versions) > 1 and not args.allow_mixed:
            ap.error(f"rows come from {len(versions)} {col} values {versions}; "
                     "rerun them on one config and code version or pass --allow-mixed")
        print(f"{col}: {', '.join(versions)}")
    summary = _stamp(calibration.aggregate(rows), _provenance(rows))
    print(f"{len(rows)} rows from {rows['race_id'].nunique()} race(s)")
    _print_pivot(summary)
    print("\nmae_offset_s with race-cluster bootstrap 95% CI:\n",
          summary[["n_races", "mae_offset_s", "mae_offset_ci_lo", "mae_offset_ci_hi"]].round(2).to_string())
    print("\nteam-mate sign agreement on remaining time:\n", calibration.midrace_pairs(rows).round(3).to_string())
    print("\nper race:\n", calibration.per_race(rows).round(3).to_string())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    summary.to_csv(args.out)


CALIBRATOR_METRICS = ["n", "mae_s", "bias_s", "mae_offset_s", "mae_offset_ci_lo", "mae_offset_ci_hi"]


def _cadence(value):
    return None if value.lower() == "none" else int(value)


def calibrators_main(argv):
    """SPEC G per calibrator and cadence: calibrators.loro on this race's fold.

    With one race the fold trains on nothing (none and shrinkage need no training) and the
    race-bootstrap CIs are NaN. Writes <race_id>_calibrators.csv (the G summary per
    calibrator, cadence, window and horizon bucket), <race_id>_calibrator_factors.csv
    (per factor rmse and 68 percent coverage) and <race_id>_selection.csv (selection_table,
    to_snapshot at horizons 20 to 40, per cadence), all stamped like the midrace csvs.
    """
    from . import calibrators
    from .race_ids import make_race_id
    ap = argparse.ArgumentParser(prog="calibrate calibrators", description="G pivot per calibrator and cadence")
    _race_args(ap)
    ap.add_argument("--calibrators", nargs="+", default=list(calibrators.REFERENCE))
    ap.add_argument("--every", nargs="+", type=_cadence, default=[1, None], help="cadences N, 'none' = pre-race only")
    ap.add_argument("--laps", nargs="+", type=int, help="snapshot laps k, default calibration.default_laps_k")
    ap.add_argument("--note", default="", help="free text stored on every row of the selection csv")
    args = ap.parse_args(argv)

    session, frame, cfg, race = _race_setup(args)
    used, _ = _pit_loss(args, frame, cfg, race)
    race_id = make_race_id(args.season, int(session.event["RoundNumber"]), args.gp)
    race = race.model_copy(update={"pit_loss_s": used, "race_id": race_id})
    data = calibrators.RaceInput(race_id, frame, race, _practice(args.season, args.gp, cfg.long_run),
                                 session_results(session), cfg)
    dataset = calibrators.build_dataset([data], laps_k=args.laps)
    stamp = {"pit_loss_s": used, "pit_loss_source": "observed" if args.pit_loss == "observed" else "--pit-loss",
             "config_hash": calibration.config_hash(cfg), "sim_code_version": calibration.SIM_CODE_VERSION}
    summaries, factors, results = [], [], {}
    for name in args.calibrators:
        res = results[name] = calibrators.loro(dataset, name, race_id, cadences=args.every)
        summaries.append(res.midrace().assign(calibrator=name).set_index("calibrator", append=True))
        factors.append(res.targets().assign(calibrator=name).set_index("calibrator", append=True))
    summary = pd.concat(summaries).reorder_levels(["calibrator", "calibration_every", "window", "horizon_bucket"])
    factor_table = pd.concat(factors).reorder_levels(["calibrator", "calibration_every", "window", "factor"])
    one_race = len(dataset.races) < 2
    selection = pd.concat([
        calibrators.selection_table(results, calibration_every=calibrators.cadence_label(n))
        .assign(calibration_every=calibrators.cadence_label(n)) for n in args.every])
    selection.index.name = "calibrator"
    selection = selection.assign(races=" ".join(sorted(dataset.races)), window="to_snapshot", horizons="20-40",
                                 scope="in-sample, one race, no CI" if one_race else "held out", note=args.note)
    os.makedirs(args.report_dir, exist_ok=True)
    for table, suffix in ((summary, "calibrators"), (factor_table, "calibrator_factors"), (selection, "selection")):
        path = os.path.normpath(os.path.join(args.report_dir, f"{race_id}_{suffix}.csv"))
        _stamp(table, stamp).to_csv(path)
        print(f"written to {path}")
    for metric in ("mae_offset_s", "mae_s"):
        pivot = summary[metric].unstack(["calibrator", "calibration_every"])
        print(f"\n{metric} (window, horizon bucket) x (calibrator, N):\n", pivot.round(2).to_string())
    print("\nfactors (rmse, 68% coverage):\n",
          factor_table[["n", "rmse", "coverage_68"]].unstack("calibrator").round(4).to_string())


TRAFFIC_TABLES = ("by_kind", "drivers", "pairs", "gate", "rejoin", "placement")
PLACEMENT_LAP = 12   # Spain's SC is laps 8-10: the post-SC placement for the 0013 decision


def traffic_arms(cfg):
    """(off, on) configs: `cfg` with traffic none, and `cfg` with its traffic, or "ghost" at
    the default params when `cfg` has none."""
    from .config import ComponentSpec
    none = ComponentSpec(name="none")
    on = cfg.traffic if cfg.traffic != none else ComponentSpec(name="ghost")
    return cfg.model_copy(update={"traffic": none}), cfg.model_copy(update={"traffic": on})


def traffic_report(frame, race, cfg, pace_series, results, placement_lap=PLACEMENT_LAP):
    """The SPEC 0012 tables for traffic off and on, the pace in the engine in both arms.

    Returns (one frame with `table` and `traffic` columns plus each table's own columns,
    {arm: summary dict}). Tables: by_kind and drivers (calibration.run), pairs
    (calibration.pairwise), gate (calibration.positions, both orders), rejoin and placement
    at `placement_lap` (calibration.rejoin, .placement).
    """
    parts, summaries = [], {}
    for arm_cfg in traffic_arms(cfg):
        arm = arm_cfg.traffic.name
        sim = Simulator(frame, arm_cfg).with_pace(pace_series)
        table, summary, by_kind = calibration.run(sim, frame, race)
        pairs, pair_summary = calibration.pairwise(sim, frame, race)
        _, pos_summary = calibration.positions(sim, frame, race, results)
        rejoin = calibration.rejoin(sim, frame, race)
        place = calibration.placement(sim, frame, race, placement_lap)
        gate = pd.DataFrame([{"mode": m, **pos_summary["by_mode"][m]} for m in ("simulated_order", "vs_real")])
        early = rejoin["in_lap"] < placement_lap
        summaries[arm] = {
            "run": summary, "pairs": pair_summary, "gate": pos_summary["by_mode"],
            "rejoin": calibration.rejoin_summary(rejoin),
            f"rejoin_before_{placement_lap}": calibration.rejoin_summary(rejoin[early]),
            f"rejoin_from_{placement_lap}": calibration.rejoin_summary(rejoin[~early]),
            "placement": {"n": int(len(place)), "mae_pos": round(float(place["abs_err"].mean()), 2),
                          "within_1_rate": round(float((place["abs_err"] <= 1).mean()), 3)},
        }
        for name, t in (("by_kind", by_kind.reset_index()), ("drivers", table.reset_index()),
                        ("pairs", pairs.reset_index()), ("gate", gate), ("rejoin", rejoin), ("placement", place)):
            # all-NaN columns (traffic_s with traffic off) are dropped; the concat restores them as NaN
            parts.append(t.dropna(axis=1, how="all").assign(
                table=name, traffic=arm, config_hash=calibration.config_hash(arm_cfg)))
    out = pd.concat(parts, ignore_index=True)
    first = ["table", "traffic"]
    return out[first + [c for c in out.columns if c not in first]], summaries


def traffic_main(argv):
    """SPEC 0012 acceptance: the traffic tables, off and on, written to <race_id>_traffic.csv."""
    from .race_ids import make_race_id
    ap = argparse.ArgumentParser(prog="calibrate traffic", description="ghost-car traffic, off against on")
    _race_args(ap)
    ap.add_argument("--pace", choices=pace.PACE_KINDS, default="practice", help="pace term in the engine, both arms")
    ap.add_argument("--pace-sessions", default=",".join(pace.PRACTICE_SESSIONS))
    args = ap.parse_args(argv)

    session, frame, cfg, race = _race_setup(args)
    off_cfg, _ = traffic_arms(cfg)
    used, observed = _pit_loss(args, frame, off_cfg, race)
    race_id = make_race_id(args.season, int(session.event["RoundNumber"]), args.gp)
    race = race.model_copy(update={"pit_loss_s": used, "race_id": race_id})
    results = session_results(session)
    pace_series = None
    if args.pace != "none":
        practice, notes = pace.load_practice(args.season, args.gp, args.pace_sessions.split(","), cfg.long_run)
        for note in notes:
            print("practice:", note)
        paces, _ = pace.offsets(Simulator(frame, off_cfg), frame, race, practice)
        if args.pace not in paces:
            ap.error(f"--pace {args.pace} needs practice laps")
        pace_series = paces[args.pace]
    table, summaries = traffic_report(frame, race, cfg, pace_series, results)
    # the committed gate's path for reference: traffic off, pace added post hoc
    summaries["posthoc_gate"] = calibration.positions(Simulator(frame, off_cfg), frame, race, results,
                                                      pace=pace_series)[1]["by_mode"]
    stamp = {"pace": args.pace, "pit_loss_s": used,
             "pit_loss_source": "observed" if args.pit_loss == "observed" else "--pit-loss",
             "sim_code_version": calibration.SIM_CODE_VERSION}
    print(f"pit loss used: {used} ({stamp['pit_loss_source']}), observed: {observed}; pace {args.pace}; "
          f"sim_code_version {stamp['sim_code_version']}")
    print(json.dumps(summaries, indent=2, default=str))
    os.makedirs(args.report_dir, exist_ok=True)
    path = os.path.normpath(os.path.join(args.report_dir, f"{race_id}_traffic.csv"))
    _stamp(table, stamp).to_csv(path, index=False)
    print(f"written to {path}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "traffic":
        return traffic_main(argv[1:])
    if argv and argv[0] == "calibrators":
        return calibrators_main(argv[1:])
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
    ap.add_argument("--pace", choices=pace.PACE_KINDS, default="none", help="pace term in the position check")
    ap.add_argument("--pace-table", action="store_true", help="the position gate under every pace term")
    ap.add_argument("--pace-sessions", default=",".join(pace.PRACTICE_SESSIONS),
                    help="practice sessions for the practice and team pace terms")
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
    results = session_results(session)
    paces, practice_table = {"none": None}, None
    if args.pace != "none" or args.pace_table:
        practice, notes = pace.load_practice(args.season, args.gp, args.pace_sessions.split(","), cfg.long_run)
        paces, practice_table = pace.offsets(sim, frame, race, practice)
        for note in notes + (practice_table.attrs["notes"] if practice_table is not None else []):
            print("practice:", note)
    if args.pace not in paces:
        ap.error(f"--pace {args.pace} needs practice laps")
    pos, pos_summary = calibration.positions(sim, frame, race, results, pace=paces[args.pace])
    print("\nteam-mate pairs (gap = first driver minus second):\n", pairs.round(2).to_string())
    print("\npair summary:", json.dumps(pair_summary, indent=2))
    print("\nfinishing positions:\n", pos.round(1).to_string())
    print("\nposition summary:", json.dumps(pos_summary, indent=2))
    os.makedirs(args.report_dir, exist_ok=True)
    pairs.to_csv(os.path.join(args.report_dir, f"{race.race_id}_pairs.csv"))
    pos.to_csv(os.path.join(args.report_dir, f"{race.race_id}_positions.csv"))
    if args.pace_table:
        stamp = {"config_hash": calibration.config_hash(cfg), "sim_code_version": calibration.SIM_CODE_VERSION}
        gate = pace.gate_table(sim, frame, race, results, paces)
        print("\nposition gate by pace term:\n", gate.to_string(index=False))
        _stamp(gate, stamp).to_csv(os.path.join(args.report_dir, f"{race.race_id}_pace_gate.csv"), index=False)
        if practice_table is not None:
            table, corr = pace.offsets_table(practice_table, paces["insample"])
            print("\npace offsets (s/lap):\n", table.round(3).to_string())
            print("\npractice vs in-sample:", json.dumps(corr))
            _stamp(table, stamp).to_csv(os.path.join(args.report_dir, f"{race.race_id}_pace_offsets.csv"))


if __name__ == "__main__":
    main()
