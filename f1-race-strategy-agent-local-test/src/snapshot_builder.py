
import fastf1
import pandas as pd
import json
import re
from pathlib import Path
from pprint import pprint


def build_snapshot(session, driver, lap_number):
    # ----------------------------------
    # Get lap data for selected driver
    # ----------------------------------
    laps = session.laps.pick_drivers(driver)

    completed = laps[
        laps["LapNumber"] <= lap_number
    ].sort_values("LapNumber")

    if completed.empty:
        raise ValueError("No completed laps found")

    current = completed.iloc[-1]

    # ----------------------------------
    # Stint history
    # ----------------------------------
    stint_history = []

    for stint_number, stint_laps in completed.groupby(
        "Stint", sort=True
    ):
        if pd.isna(stint_number):
            continue

        stint_history.append({
            "stint": int(stint_number),
            "compound": stint_laps.iloc[0]["Compound"],
            "start_lap": int(stint_laps["LapNumber"].min()),
            "end_lap": int(stint_laps["LapNumber"].max()),
            "laps_completed": len(stint_laps)
        })

    # ----------------------------------
    # Safety car / red flag / VSC history
    # ----------------------------------
    safety_car_history = []

    messages = session.race_control_messages

    if messages is not None:
        messages = messages.copy()

        # Include events up to selected lap
        messages = messages[
            messages["Lap"].notna()
            & (messages["Lap"] <= lap_number)
        ]

        event_types = {
            "RED FLAG": "RED_FLAG",
            "SAFETY CAR DEPLOYED": "SC",
            "VIRTUAL SAFETY CAR DEPLOYED": "VSC"
        }

        for _, event in messages.iterrows():
            message = str(event["Message"]).upper().strip()

            if message not in event_types:
                continue

            safety_car_history.append({
                "type": event_types[message],
                "lap": int(event["Lap"]),
                "message": message
            })

    # ----------------------------------
    # Build snapshot
    # ----------------------------------
    snapshot = {
        "race": (
            f"{session.event['EventDate'].year} "
            f"{session.event['EventName']}"
        ),
        "driver": driver,
        "lap_completed": int(current["LapNumber"]),
        "total_laps": int(session.total_laps),
        "remaining_laps": (
            int(session.total_laps)
            - int(current["LapNumber"])
        ),
        "current_compound": current["Compound"],
        "tyre_age": (
            int(current["TyreLife"])
            if pd.notna(current["TyreLife"])
            else None
        ),
        "current_stint": (
            int(current["Stint"])
            if pd.notna(current["Stint"])
            else None
        ),
        "stint_history": stint_history,
        "safety_car_history": safety_car_history,
        "compounds_used": (
            completed["Compound"]
            .dropna()
            .unique()
            .tolist()
        )
    }

    return snapshot


def save_snapshot(snapshot, session, driver, lap_number):
    # ----------------------------------
    # Create output directory
    # ----------------------------------
    output_dir = Path("data/snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # Automatically generate filename
    # ----------------------------------
    season = session.event["EventDate"].year
    race_name = session.event["EventName"]

    race_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        race_name.lower()
    ).strip("_")

    # Remove duplicate Grand Prix suffix
    race_slug = race_slug.removesuffix("_grand_prix")

    filename = (
        f"{season}_{race_slug}_"
        f"{driver.upper()}_lap{lap_number}.json"
    )

    output_file = output_dir / filename

    # ----------------------------------
    # Save snapshot as JSON
    # ----------------------------------
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=4)

    print(f"\nSnapshot saved to: {output_file}")

    return output_file


def main():
    # ----------------------------------
    # Configure FastF1 cache
    # ----------------------------------
    fastf1.Cache.enable_cache("cache")
    fastf1.Cache.offline_mode(True)

    # ----------------------------------
    # Select race, driver, and lap
    # ----------------------------------
    season = 2024
    gp = "Sao Paolo"
    driver = "HAM"
    lap_number = 40

    # ----------------------------------
    # Load race data
    # ----------------------------------
    session = fastf1.get_session(
        season, gp, "R"
    )

    session.load(
        telemetry=False,
        weather=False
    )

    # ----------------------------------
    # Generate snapshot
    # ----------------------------------
    snapshot = build_snapshot(
        session, driver, lap_number
    )

    # ----------------------------------
    # Print snapshot
    # ----------------------------------
    print("\nGenerated Snapshot:")
    pprint(snapshot)

    # ----------------------------------
    # Save snapshot to JSON
    # ----------------------------------
    save_snapshot(
        snapshot, session, driver, lap_number
    )


if __name__ == "__main__":
    main()
