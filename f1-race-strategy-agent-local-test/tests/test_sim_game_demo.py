import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_game_demo.py"
SPEC = importlib.util.spec_from_file_location("run_game_demo", SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture
def server():
    httpd = demo.ThreadingHTTPServer(("127.0.0.1", 0), demo.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def request(base, path, *, method="GET", body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_only_allowlisted_get_routes_are_served(server):
    assert request(server, "/")[0] == 200
    assert request(server, "/api/tracks")[0] == 200
    for path in ("/.env", "/.env.example", "/requirements.txt", "/cache/fastf1_http_cache.sqlite",
                 "/../.env", "/%2e%2e/.env", "/favicon.ico"):
        assert request(server, path)[0] == 404


def test_track_endpoint_rejects_traversal_and_unknown_files(server):
    for path in ("/api/track/../../../.env", "/api/track/%2e%2e%2f%2e%2e%2f.env",
                 "/api/track/manifest.json"):
        assert request(server, path)[0] in (400, 404)
    assert request(server, "/api/track/not-installed")[0] == 404


def test_snapshot_list_contains_only_project_contract_snapshots(server):
    status, body = request(server, "/api/snapshots")
    assert status == 200
    data = json.loads(body)
    assert data
    assert all({"id", "label", "race_id", "driver", "lap"} <= set(row) for row in data)
    assert any(row["id"] == "2024_monaco_LEC_lap20" for row in data)


def test_strategist_route_calls_project_agent_and_returns_submission(server, monkeypatch):
    calls = []

    def fake_agent(snapshot, question):
        calls.append((snapshot, question))
        return ({"decision": "PIT", "compound": "MEDIUM", "confidence": 0.76,
                 "rationale": "The simulated option has the best balance.",
                 "key_risks": ["traffic after rejoining"]},
                [{"tool": "compare_plans"}, {"tool": "submit_recommendation"}])

    monkeypatch.setattr(demo, "run_project_agent", fake_agent)
    status, body = request(server, "/api/strategist", method="POST",
                           headers={"Content-Type": "application/json"},
                           body={"snapshot_id": "2024_monaco_LEC_lap20"})
    result = json.loads(body)
    assert status == 200
    assert len(calls) == 1
    assert calls[0][0].race.race_id == "2024_08_monaco"
    assert "immediate strategy" in calls[0][1]
    assert result["recommendation"]["decision"] == "PIT"
    assert [row["tool"] for row in result["trajectory"]] == ["compare_plans", "submit_recommendation"]


def test_strategist_rejects_unknown_snapshot_and_cross_origin(server):
    status, _ = request(server, "/api/strategist", method="POST",
                         body={"snapshot_id": "../../.env"})
    assert status == 400
    status, _ = request(server, "/api/strategist", method="POST",
                         headers={"Content-Type": "application/json", "Origin": "http://evil.invalid"},
                         body={"snapshot_id": "2024_monaco_LEC_lap20"})
    assert status == 403
