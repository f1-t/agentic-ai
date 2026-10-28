"""CLI for the calibration check.

    python -m src.sim.calibrate --season 2021 --gp Spain --pit-loss 22      # a number, or observed (default)
    python -m src.sim.calibrate --degradation linear_fuel

Also prints the team-mate and finishing position checks and writes them to
reports/calibration/<race_id>_pairs.csv and <race_id>_positions.csv.
"""
import argparse
import json
import os

from ..contracts import RaceParams
from . import calibration
from .config import SimConfig
from .data import lap_frame, load_session, session_results
from .engine import Simulator

DEFAULT_PIT_LOSS_S = 22.0
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "calibration")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2021)
    ap.add_argument("--gp", default="Spain")
    ap.add_argument("--pit-loss", default="observed",
                    help='seconds, or "observed" to measure it from a first pass at 22.0 and rerun with it')
    ap.add_argument("--min-stint", type=int, default=10)
    ap.add_argument("--degradation", default="linear")
    ap.add_argument("--config", help="json or yaml SimConfig file, overrides --degradation")
    ap.add_argument("--out", help="write per driver csv here")
    ap.add_argument("--report-dir", default=REPORT_DIR, help="where the pairs and positions csvs go")
    args = ap.parse_args()

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
