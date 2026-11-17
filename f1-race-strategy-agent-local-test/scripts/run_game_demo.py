#!/usr/bin/env python3
"""Serve the autonomous bird's-eye race demo on localhost."""
from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
TRACK_DIR = ROOT / "data" / "tracks"
HTML = ROOT / "src" / "game" / "demo.html"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/tracks":
            tracks = []
            if TRACK_DIR.is_dir():
                for file in sorted(TRACK_DIR.glob("*.json")):
                    if file.name == "manifest.json":
                        continue
                    try:
                        geometry = json.loads(file.read_text(encoding="utf-8"))
                        circuit = str(geometry.get("circuit_id", file.stem))
                        season = geometry.get("season")
                        identifier = file.stem
                        label = f"{circuit.replace('_', ' ').title()} · {season}" if season else circuit
                        tracks.append({"id": identifier, "label": label})
                    except (OSError, json.JSONDecodeError):
                        continue
            return self._send(200, json.dumps(tracks))
        if path.startswith("/api/track/"):
            identifier = unquote(path.removeprefix("/api/track/"))
            # IDs are filename stems only; path separators are rejected.
            if not identifier or any(x in identifier for x in ("/", "\\", "..")):
                return self._send(400, json.dumps({"error": "invalid track id"}))
            file = TRACK_DIR / f"{identifier}.json"
            if not file.is_file():
                return self._send(404, json.dumps({"error": "track geometry not installed"}))
            try:
                geometry = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return self._send(500, json.dumps({"error": str(exc)}))
            return self._send(200, json.dumps({
                "name": geometry.get("name", identifier),
                "points": geometry.get("points", []),
                "width_m": geometry.get("width_m", 13),
                "source": geometry.get("source", "unknown source"),
            }))
        if path == "/src/game/demo.html":
            return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        return super().do_GET()

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Gridline race demo: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop. Screen capture is available inside the page.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
