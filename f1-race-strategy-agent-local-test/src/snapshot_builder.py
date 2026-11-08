
import argparse
import re
from pathlib import Path
from pprint import pprint

from src.contracts import Snapshot
from src.contracts import RaceParams, Snapshot
from src.sim import calibration, calibrate
from src.sim.config import SimConfig
from src.sim.data import CACHE_DIR, lap_frame, load_session
from src.sim.data import CACHE_DIR, load_session
from src.sim.race_ids import make_race_id


# Output path relative to the project, not the terminal's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshots"


def get_red_flag_laps(frame, k):
    """Find red-flag laps up to the snapshot decision point."""
    red_flag_laps = set()

    for _, row in frame.iterrows():
        lap = int(row["lap"])
        status = str(row["track_status"])

        if 1 <= lap <= k and "5" in status:
            red_flag_laps.add(lap)

    return sorted(red_flag_laps)



def _build_snapshot(frame, race, driver, k, messages=None) -> Snapshot:
    """Build a snapshot, optionally cross-checking race-control messages."""

    snapshot = calibration.snapshot_at(frame, race, driver, k)

    periods = calibration.observed_periods(frame, messages)

    sc_laps = []
    vsc_laps = []

    for kind, start, end in periods:
        first = max(1, start)
        last = min(k, end)

        if first > last:
            continue

        laps = range(first, last + 1)

        if kind == "sc":
            sc_laps.extend(laps)
        elif kind == "vsc":
            vsc_laps.extend(laps)

    return Snapshot.model_validate({
        **snapshot.model_dump(),
        "sc_laps": sorted(set(sc_laps)),
        "vsc_laps": sorted(set(vsc_laps)),
        "red_flag_laps": get_red_flag_laps(frame, k),
    })


def build_snapshot(frame, race, driver, k) -> Snapshot:
    """Evaluator-compatible snapshot builder."""
    return _build_snapshot(frame, race, driver, k)



def save_snapshot(snapshot, season, gp, driver, lap_number):
    """Save a validated snapshot with an automatic filename."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    race_slug = re.sub(
        r"[^a-z0-9]+", "_", gp.lower()
    ).strip("_")

    race_slug = race_slug.removesuffix("_grand_prix")

    filename = (
        f"{season}_{race_slug}_"
        f"{driver.upper()}_lap{lap_number}.json"
    )

    output_file = OUTPUT_DIR / filename

    output_file.write_text(
        snapshot.model_dump_json(indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"\nSnapshot saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate an F1 race snapshot"
    )

    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument(
        "--gp", type=str, default="Monaco Grand Prix"
    )
    parser.add_argument("--driver", type=str, default="LEC")
    parser.add_argument("--lap", type=int, default=20)

    args = parser.parse_args()

    driver = args.driver.upper()
    lap_number = args.lap

    print(f"Using shared cache: {CACHE_DIR}")
    print(f"Loading {args.season} {args.gp}...")

    session = load_session(
    args.season,
    args.gp,
    kind="R",
    messages=True,
    )

    # Resolve the selected event to the repository's race ID.
    round_number = int(session.event["RoundNumber"])

    race_id = make_race_id(
        args.season,
        round_number,
        args.gp.removesuffix(" Grand Prix"),
    )

    
    # Convert the loaded race laps to the simulator's format.
    frame = lap_frame(session.laps)

    # Create the race parameters.
    config = SimConfig()

    race = RaceParams(
        race_id=race_id,
        total_laps=int(session.total_laps),
        pit_loss_s=calibrate.DEFAULT_PIT_LOSS_S,
        min_stint_laps=RaceParams.model_fields[
            "min_stint_laps"
        ].default,
        compounds=sorted(
            frame["compound"].dropna().unique().tolist()
        ),
    )

    # Use the same observed pit-loss calculation as the simulator.
    from src.sim.engine import Simulator

    observed_pit_loss = calibration.run(
        Simulator(frame, config),
        frame,
        race,
    )[1]["observed_pit_loss_s"]

    race = race.model_copy(
        update={"pit_loss_s": observed_pit_loss}
    )

    print(f"Race loaded: {race_id}")
    print(f"Observed pit loss: {observed_pit_loss:.2f}s")


    # Validate the requested driver and decision lap.
    driver_laps = frame.loc[
        frame["driver"] == driver, "lap"
    ]

    if driver_laps.empty:
        raise ValueError(
            f"Driver {driver} not found in {race_id}"
        )

    if not 0 <= lap_number < int(driver_laps.max()):
        raise ValueError(
            f"Lap must be between 0 and "
            f"{int(driver_laps.max()) - 1}"
        )

    # Generate the snapshot using the shared contract.
    snapshot = _build_snapshot(
    frame,
    race,
    driver,
    lap_number,
    messages=session.race_control_messages,
)

    print("\nGenerated Snapshot:")
    pprint(snapshot.model_dump())

    save_snapshot(
        snapshot,
        args.season,
        args.gp,
        driver,
        lap_number,
    )


if __name__ == "__main__":
    main()
