"""Serve the web UI (web/dist), classify uploaded photos with DINOv2, and keep recent shots for the big screen."""

import argparse
import collections
import functools
import io
import itertools
import json
import socket
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from PIL import Image, ImageOps

import mcu
from classify import classify, load_cpu_model

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
MAX_UPLOAD = 20 * 1024 * 1024
HISTORY_SIZE = 12


def thumbnail(photo):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(photo))).convert("RGB")
    img.thumbnail((640, 640))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue()


class Session:
    """The last few shots for the big screen, plus running totals since the server started."""

    def __init__(self):
        self.recent = collections.deque(maxlen=HISTORY_SIZE)  # (shot, jpeg), newest first
        self.ids = itertools.count(1)
        self.count = 0
        self.model_ms = 0.0
        self.total_ms = 0.0

    def add(self, photo, result):
        shot_id = next(self.ids)
        shot = {"id": shot_id, "time": time.time(), "image": f"/shots/{shot_id}.jpg", "result": result}
        self.recent.appendleft((shot, thumbnail(photo)))
        self.count += 1
        self.model_ms += result["model_ms"]
        self.total_ms += result["total_ms"]
        return shot

    def image(self, shot_id):
        return next((jpeg for shot, jpeg in self.recent if shot["id"] == shot_id), None)

    def summary(self):
        n = max(self.count, 1)
        return {
            "count": self.count,
            "avg_model_ms": self.model_ms / n,
            "avg_total_ms": self.total_ms / n,
            "shots": [shot for shot, _ in self.recent],
        }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/info":
            return self.send_json(200, {"backend": self.server.backend, "url": self.server.url})
        if path == "/shots":
            return self.send_json(200, self.server.session.summary())
        if path.startswith("/shots/") and path.endswith(".jpg"):
            shot_id = path[len("/shots/"):-len(".jpg")]
            jpeg = self.server.session.image(int(shot_id)) if shot_id.isdigit() else None
            if jpeg is None:
                return self.send_json(404, {"error": "unknown shot"})
            return self.send_body(200, "image/jpeg", jpeg)
        if path == "/screen":
            self.path = "/"  # same page; the app switches to the big-screen view
        super().do_GET()  # static files from web/dist

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if self.path != "/classify" or not 0 < size <= MAX_UPLOAD:
            return self.send_json(400, {"error": "POST a photo to /classify"})
        photo = self.rfile.read(size)
        mcu.call("show_thinking")
        try:
            result = classify(io.BytesIO(photo), backend=self.server.backend)
        except Exception as exc:
            mcu.call("show_error")
            return self.send_json(500, {"error": str(exc)})
        # Send the raw label and confidence; the sketch decides what to show (see mcu/sketch/sketch.ino).
        label, prob = result["top"][0]
        result["action_ms"] = mcu.call("show_result", label, round(prob * 100))
        self.send_json(200, self.server.session.add(photo, result))

    def send_json(self, status, payload):
        self.send_body(status, "application/json", json.dumps(payload).encode())

    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/shots", "/info")):  # the UI polls these constantly
            super().log_message(fmt, *args)


def local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # sends nothing; just picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu", action="store_true", help="run on the laptop CPU instead of the NPU")
    args = parser.parse_args()
    if not (WEB_DIST / "index.html").exists():
        print(f"warning: {WEB_DIST} is missing; run `npm run build` in web/")
    if args.cpu:
        load_cpu_model()  # load once up front so the first photo isn't slow
    # Single-threaded on purpose: one photo at a time on the NPU.
    server = HTTPServer(("0.0.0.0", args.port), functools.partial(Handler, directory=str(WEB_DIST)))
    server.backend = "cpu" if args.cpu else "npu"
    server.url = f"http://{local_ip()}:{args.port}"
    server.session = Session()
    print(f"Phone: {server.url}   Big screen: {server.url}/screen   ({server.backend.upper()})")
    server.serve_forever()


if __name__ == "__main__":
    main()
