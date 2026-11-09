
import json
from pathlib import Path

import requests

BASE_URL = "https://api.openf1.org/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "openf1"



def fetch_openf1(endpoint, session_key):
    """Fetch and cache one OpenF1 dataset."""

    output_dir = CACHE_DIR / str(session_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{endpoint}.json"

    if output_file.exists():
        return json.loads(output_file.read_text(encoding="utf-8"))

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params={"session_key": session_key},
        timeout=60,
    )

    if response.status_code == 404 and endpoint == "pit":
        print(
            f"WARNING: OpenF1 pit data unavailable "
            f"for session {session_key} (HTTP 404)"
        )
        return None

    response.raise_for_status()

    data = response.json()

    output_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return data



def load_race_data(session_key):
    """Load datasets useful for race strategy validation."""

    endpoints = [
        "drivers",
        "laps",
        "stints",
        "pit",
        "position",
        "intervals",
        "race_control",
        "weather",
    ]

    
    data = {}

    for endpoint in endpoints:
        records = fetch_openf1(endpoint, session_key)
        data[endpoint] = records

        if records is None:
            print(f"{endpoint}: unavailable")
        else:
            print(f"{endpoint}: {len(records)} records")

    return data

