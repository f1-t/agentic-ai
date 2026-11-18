#!/usr/bin/env python3
"""Build GPS-derived circuit centerlines from FastF1 sessions (network required).

Writes one JSON geometry per unique circuit/layout in seasons 2021-2025. The centerline is
made from the fastest available race lap's car position telemetry, not a hand-drawn outline.
Run from this directory: ``python scripts/build_track_geometries.py``.

Output is read by ``scripts/run_game_demo.py`` and offered in the demo's circuit selector.
Files already present are skipped; delete one to re-download it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fastf1

from src.sim import event_rates

OUTPUT = ROOT / "data" / "tracks"
SEASONS = range(2021, 2026)
POINTS = 500


def resample(points, count=POINTS):
    """Uniformly sample an ordered path by cumulative distance."""
    import numpy as np
    pts = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.05]
    pts = pts[keep]
    if len(pts) < 3:
        raise ValueError("telemetry contains fewer than three distinct GPS points")
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    target = np.linspace(0.0, dist[-1], count, endpoint=False)
    return np.column_stack([np.interp(target, dist, pts[:, axis]) for axis in (0, 1)])


def extract(year, round_number, event):
    session = fastf1.get_session(year, int(round_number), "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    laps = session.laps.pick_quicklaps()
    if laps.empty:
        # Wet, interrupted, or short races may have no laps meeting FastF1's quick-lap
        # filter. Still derive a centerline from the fastest recorded lap with telemetry.
        laps = session.laps
    if laps.empty:
        raise ValueError(f"no race laps for {year} {event['EventName']}")
    lap = laps.pick_fastest()
    telemetry = lap.get_telemetry().dropna(subset=["X", "Y"])
    points = resample(telemetry[["X", "Y"]].to_numpy())
    length = float(telemetry["Distance"].max()) if "Distance" in telemetry else None
    circuit = event_rates.schedule_circuit(event)
    if circuit is None:
        raise ValueError(f"unmapped circuit: {event['Location']}")
    circuit_id, _ = circuit
    return circuit_id, points, length, lap


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    saved, errors = set(), []
    for year in SEASONS:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            name = event["EventName"]
            try:
                circuit, points, telemetry_length, lap = extract(year, event["RoundNumber"], event)
                key = f"{circuit}_{year}"
                if key in saved:
                    continue
                path = OUTPUT / f"{key}.json"
                if path.exists():
                    saved.add(key)
                    continue
                doc = {
                    "name": name,
                    "circuit_id": circuit,
                    "season": year,
                    "points": points.round(2).tolist(),
                    "width_m": 12.0,
                    "length_m": round(telemetry_length, 2) if telemetry_length else None,
                    "source": "FastF1 race-lap position telemetry",
                    "source_url": "https://docs.fastf1.dev/",
                    "reference": {"season": year, "event": name,
                                  "driver": str(lap["Driver"]), "lap": int(lap["LapNumber"])},
                }
                path.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
                saved.add(key)
                print(f"saved {path.relative_to(ROOT)} ({len(points)} points)")
            except Exception as exc:  # keep building the rest of the catalog
                errors.append(f"{year} {name}: {type(exc).__name__}: {exc}")
                print(f"skipped {year} {name}: {exc}")
    manifest = {"seasons": list(SEASONS), "circuits_saved": sorted(saved),
                "source": "FastF1 race-lap GPS position telemetry", "errors": errors}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(saved)} circuit geometries to {OUTPUT}")
    if errors:
        print(f"{len(errors)} event(s) unavailable; see manifest.json")


if __name__ == "__main__":
    main()
