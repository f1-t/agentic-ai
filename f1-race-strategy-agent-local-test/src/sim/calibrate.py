"""CLI for the calibration check.

    python -m src.sim.calibrate --season 2021 --gp Spain --pit-loss 22
    python -m src.sim.calibrate --degradation linear_fuel
"""
import argparse
import json

from ..contracts import RaceParams
from . import calibration
from .config import SimConfig
from .data import lap_frame, load_session
from .engine import Simulator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2021)
    ap.add_argument("--gp", default="Spain")
    ap.add_argument("--pit-loss", type=float, default=22.0)
    ap.add_argument("--min-stint", type=int, default=10)
    ap.add_argument("--degradation", default="linear")
    ap.add_argument("--config", help="json or yaml SimConfig file, overrides --degradation")
    ap.add_argument("--out", help="write per driver csv here")
    args = ap.parse_args()

    session = load_session(args.season, args.gp)
    frame = lap_frame(session.laps)
    cfg = SimConfig.from_file(args.config) if args.config else SimConfig(degradation={"name": args.degradation})
    race = RaceParams(
        race_id=f"{args.season}_{args.gp}", total_laps=int(session.total_laps), pit_loss_s=args.pit_loss,
        min_stint_laps=args.min_stint, compounds=sorted(frame["compound"].dropna().unique()))
    table, summary, by_kind = calibration.run(Simulator(frame, cfg), frame, race)

    print(table.round(1).to_string())
    print("\nsummary:", json.dumps(summary, indent=2))
    print("\nresidual (actual - predicted) by lap type:\n", by_kind.to_string())
    print("observed pit loss (s):", round(by_kind.attrs["observed_pit_loss_s"], 2))
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
