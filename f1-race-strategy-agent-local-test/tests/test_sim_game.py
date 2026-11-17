import json

import pytest

from src.game.engine import Action, GameEngine, TrackGeometry
from src.game.tracks import load_track


def oval():
    return TrackGeometry("test oval", ((0, 0), (100, 0), (100, 50), (0, 50)), width_m=14)


def test_track_closes_and_samples_wrapped_distance():
    track = oval()
    assert track.points[0] == track.points[-1]
    assert track.length_m == pytest.approx(300)
    x0, y0, _ = track.at(0)
    x1, y1, _ = track.at(track.length_m)
    assert (x0, y0) == pytest.approx((x1, y1))
    assert track.at(-track.length_m / 4)[:2] == pytest.approx(track.at(track.length_m * 3 / 4)[:2])


def test_game_engine_advances_car_and_reports_renderer_telemetry():
    game = GameEngine(oval(), dt=0.1, max_speed_mps=20, acceleration_mps2=10)
    car = game.add_car("VER")
    for _ in range(20):
        game.step({"VER": Action(throttle=1)})
    assert car.speed_mps > 0
    data = game.telemetry()["VER"]
    assert set(("x", "y", "heading_rad", "speed_mps", "lap", "off_track")) <= set(data)


def test_car_completes_lap_and_engine_stops_at_race_distance():
    game = GameEngine(oval(), laps=1, dt=0.1, max_speed_mps=25, acceleration_mps2=25)
    car = game.add_car("HAM", speed_mps=20)
    for _ in range(1000):
        game.step()
        if game.state.finished:
            break
    assert car.lap >= 1
    assert game.state.finished


def test_geometry_rejects_bad_data_and_loads_json(tmp_path):
    with pytest.raises(ValueError, match="at least three"):
        TrackGeometry("bad", ((0, 0), (1, 1)))
    path = tmp_path / "track.json"
    path.write_text(json.dumps({"name": "test", "points": [[0, 0], [1, 0], [1, 1]]}))
    assert TrackGeometry.load(path).name == "test"


def test_track_catalog_missing_message_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_track_geometries.py"):
        load_track("not-installed", tmp_path)
