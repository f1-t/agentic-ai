
from src.contracts import RaceParams, Snapshot
from src.snapshot_builder import build_snapshot
from src.sim import calibration
import pandas as pd
from src.snapshot_builder import get_red_flag_laps


def test_red_flag_detection():
    frame = pd.DataFrame({
        "lap": [1, 2, 3, 4],
        "track_status": ["1254", "14", "1", "5"],
    })

    assert get_red_flag_laps(frame, 2) == [1]
    assert get_red_flag_laps(frame, 4) == [1, 4]
    assert get_red_flag_laps(frame, 0) == []


def test_builder_returns_snapshot(monkeypatch):
    race = RaceParams(
        race_id="test_race",
        total_laps=50,
        pit_loss_s=20.0,
    )

    base = Snapshot(
        race=race,
        driver="HAM",
        laps_completed=20,
        compound="HARD",
        tyre_age=5,
    )

    monkeypatch.setattr(
        calibration,
        "snapshot_at",
        lambda frame, race, driver, k: base,
    )

    monkeypatch.setattr(
    calibration,
    "observed_periods",
    lambda frame, messages=None: [
        ("sc", 10, 13),
        ("vsc", 19, 22),
    ],
    )
    frame = pd.DataFrame({
        "lap": [1, 2, 3, 4],
        "track_status": ["1254", "14", "1", "1"],
    })

    result = build_snapshot(frame, race, "HAM", 20)

    assert isinstance(result, Snapshot)
    assert result.sc_laps == [10, 11, 12, 13]
    assert result.vsc_laps == [19, 20]
    assert result.red_flag_laps == [1]
    assert result.laps_completed == 20
    assert result.compound == "HARD"
    
