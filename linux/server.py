"""Phone-friendly web page: take a photo, classify it with DINOv2, show the top-5 labels."""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from classify import classify

PAGE = Path(__file__).with_name("index.html").read_bytes()
MAX_UPLOAD = 20 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send(200, "text/html; charset=utf-8", PAGE)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if self.path != "/classify" or not 0 < size <= MAX_UPLOAD:
            return self.send(400, "application/json", b'{"error": "POST a photo to /classify"}')
        try:
            result = classify(io.BytesIO(self.rfile.read(size)), backend=self.server.backend)
            self.send(200, "application/json", json.dumps(result).encode())
        except Exception as exc:
            self.send(500, "application/json", json.dumps({"error": str(exc)}).encode())

    def send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu", action="store_true", help="run on the laptop CPU instead of the NPU")
    args = parser.parse_args()
    # Single-threaded on purpose: one photo at a time on the NPU.
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.backend = "cpu" if args.cpu else "npu"
    print(f"Open http://<this machine's IP>:{args.port} ({server.backend.upper()})")
    server.serve_forever()


if __name__ == "__main__":
    main()
