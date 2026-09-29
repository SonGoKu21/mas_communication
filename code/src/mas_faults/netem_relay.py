from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/relay":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            padding_bytes = max(0, min(int(request.get("padding_bytes", 0)), 2_097_152))
            response = {
                "message_id": request.get("message_id"),
                "payload": request.get("payload"),
                "padding": "x" * padding_bytes,
            }
            self._send_json(200, response)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": type(exc).__name__})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Echo structured inter-agent messages over a real TCP path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RelayHandler)
    print(f"NETEM_RELAY_READY host={args.host} port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
