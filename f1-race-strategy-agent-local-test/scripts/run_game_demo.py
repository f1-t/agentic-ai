#!/usr/bin/env python3
"""Serve the bird's-eye race visualization and the project's strategy agent on localhost.

Only the demo page and explicit JSON APIs are served. The project directory holds ``.env``
and the FastF1 cache, so this module intentionally does not use a static-file handler.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
TRACK_DIR = (ROOT / "data" / "tracks").resolve()
SNAPSHOT_DIR = (ROOT / "data" / "snapshots").resolve()
HTML = ROOT / "src" / "game" / "demo.html"
MAX_REQUEST_BYTES = 16_384
log = logging.getLogger("game-demo")
# src.tools holds the active snapshot globally, so only one live agent episode may run at a time.
AGENT_LOCK = threading.Lock()


def run_project_agent(snapshot: dict, question: str):
    """Invoke the same provider-neutral agent loop used by the project CLI."""
    from src.agent import run_agent
    return run_agent(snapshot, question=question, verbose=False)


class Handler(BaseHTTPRequestHandler):
    """Request handler for the demo's fixed allowlist of routes."""

    def do_GET(self):  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/tracks":
            return self._send(200, json.dumps(self._tracks()))
        if path == "/api/snapshots":
            return self._send(200, json.dumps(self._snapshots()))
        if path.startswith("/api/track/"):
            return self._track(path.removeprefix("/api/track/"))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):  # noqa: N802 - http.server API
        if urlparse(self.path).path != "/api/strategist":
            return self._send(404, json.dumps({"error": "not found"}))
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
            if parsed.hostname is None or parsed.hostname.lower() != host:
                return self._send(403, json.dumps({"error": "cross-origin request rejected"}))
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send(400, json.dumps({"error": "invalid content length"}))
        if size < 1 or size > MAX_REQUEST_BYTES:
            return self._send(413, json.dumps({"error": "request body must be 1–16384 bytes"}))
        try:
            payload = json.loads(self.rfile.read(size))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send(400, json.dumps({"error": "invalid JSON body"}))
        if not isinstance(payload, dict):
            return self._send(400, json.dumps({"error": "expected a JSON object"}))
        snapshot_id = payload.get("snapshot_id")
        question = payload.get("question", "Recommend the best immediate strategy action for this race snapshot. Use the project tools and submit a recommendation.")
        if not isinstance(snapshot_id, str) or not snapshot_id or Path(snapshot_id).name != snapshot_id:
            return self._send(400, json.dumps({"error": "select a valid bundled race snapshot"}))
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            return self._send(400, json.dumps({"error": "question must contain 1–2000 characters"}))
        snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}.json"
        if snapshot_path.resolve().parent != SNAPSHOT_DIR or not snapshot_path.is_file():
            return self._send(404, json.dumps({"error": "race snapshot not found"}))
        try:
            from src.contracts import Snapshot
            snapshot = Snapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return self._send(422, json.dumps({"error": "race snapshot is not a valid project Snapshot"}))

        try:
            # The project agent's tool module carries a process-global active snapshot.
            with AGENT_LOCK:
                recommendation, trajectory = run_project_agent(snapshot, question.strip())
        except Exception:
            log.exception("project strategy agent failed for snapshot %s", snapshot_id)
            return self._send(502, json.dumps({"error": "strategy agent failed; see local server output"}))
        if recommendation is None:
            return self._send(502, json.dumps({
                "error": "agent did not submit a recommendation",
                "trajectory": _json_safe(trajectory),
            }))
        return self._send(200, json.dumps({
            "snapshot_id": snapshot_id,
            "recommendation": _json_safe(recommendation),
            "trajectory": _json_safe(trajectory),
        }))

    def _tracks(self):
        """Circuit ids available under ``data/tracks``."""
        tracks = []
        if TRACK_DIR.is_dir():
            for file in sorted(TRACK_DIR.glob("*.json")):
                if file.name == "manifest.json":
                    continue
                try:
                    resolved = file.resolve()
                    if resolved.parent != TRACK_DIR:
                        continue
                    geometry = json.loads(resolved.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                circuit = str(geometry.get("circuit_id", file.stem))
                season = geometry.get("season")
                label = f"{circuit.replace('_', ' ').title()} · {season}" if season else circuit
                tracks.append({"id": file.stem, "label": label})
        return tracks

    def _snapshots(self):
        """List only native snapshots accepted by the shared Snapshot contract."""
        from src.contracts import Snapshot
        snapshots = []
        if SNAPSHOT_DIR.is_dir():
            for file in sorted(SNAPSHOT_DIR.glob("*.json")):
                try:
                    resolved = file.resolve()
                    if resolved.parent != SNAPSHOT_DIR:
                        continue
                    snap = Snapshot.model_validate_json(resolved.read_text(encoding="utf-8"))
                except Exception:
                    # Some older files use a display-only legacy snapshot format.
                    continue
                snapshots.append({
                    "id": file.stem,
                    "label": f"{snap.race.race_id} · {snap.driver} · lap {snap.laps_completed}",
                    "race_id": snap.race.race_id,
                    "driver": snap.driver,
                    "lap": snap.laps_completed,
                    "total_laps": snap.race.total_laps,
                })
        return snapshots

    def _track(self, raw_identifier):
        """Geometry for one circuit id, resolved inside ``data/tracks`` only."""
        identifier = unquote(raw_identifier)
        if not identifier or identifier != Path(identifier).name or identifier == "manifest.json":
            return self._send(400, json.dumps({"error": "invalid track id"}))
        file = TRACK_DIR / f"{identifier}.json"
        resolved = file.resolve()
        if resolved.parent != TRACK_DIR or not resolved.is_file():
            return self._send(404, json.dumps({"error": "track geometry not installed"}))
        try:
            geometry = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._send(500, json.dumps({"error": f"unreadable geometry: {exc}"}))
        return self._send(200, json.dumps({
            "name": geometry.get("name", identifier),
            "points": geometry.get("points", []),
            "width_m": geometry.get("width_m", 13),
            "source": geometry.get("source", "unknown source"),
        }))

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def _json_safe(value):
    """Convert agent/model scalar types to JSON-native values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback address only; the strategy endpoint uses configured model credentials")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host.lower() == "localhost"
    if not loopback:
        raise SystemExit("For safety the demo binds to localhost only; non-loopback hosts are not allowed.")
    if not HTML.is_file():
        raise SystemExit(f"demo page missing: {HTML}")
    if not TRACK_DIR.is_dir():
        print("note: data/tracks not found; the demo will only offer the built-in "
              "illustrative loop (run scripts/build_track_geometries.py to add real geometry)")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Gridline strategy demo: http://{args.host}:{args.port}")
    print("Routes: /, /api/tracks, /api/track/<id>, /api/snapshots, POST /api/strategist")
    print("The strategist endpoint runs the configured project LLM and may incur provider usage.")
    print("Loopback-only binding protects local model credentials and API quota.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
