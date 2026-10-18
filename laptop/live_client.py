"""Laptop-side live demo view. Shows the webcam at full frame rate; a background thread sends the
newest frame to live_server.py on the board and overlays the latest prediction.

Setup:  adb forward tcp:5000 tcp:5000   (or pass --host <board-ip>)
Run:    python live_client.py
Keys:   q = quit
"""
import argparse
import socket
import struct
import threading
import time

import cv2

VALID_MIN, VALID_MAX = 50, 1200   # sensor readings outside this range count as "covered"


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Board disconnected")
        buf.extend(chunk)
    return bytes(buf)


def put(img, text, y, scale, colour, thick=2):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


class Predictor(threading.Thread):
    """Sends the newest frame, waits for the reply, repeats. Never queues old frames."""

    def __init__(self, host, port, width):
        super().__init__(daemon=True)
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.width = width
        self.lock = threading.Lock()
        self.frame = None
        self.result = None          # (sensor_mm, pred_mm, infer_ms, round_trip_ms)
        self.running = True
        self.error = None

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def run(self):
        try:
            while self.running:
                with self.lock:
                    frame, self.frame = self.frame, None
                if frame is None:
                    time.sleep(0.005)
                    continue
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (self.width, int(h * self.width / w)))
                ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
                data = jpg.tobytes()
                t0 = time.time()
                self.sock.sendall(struct.pack("!dI", t0, len(data)) + data)
                sensor, pred, infer_ms = struct.unpack("!iff", recv_exact(self.sock, 12))
                with self.lock:
                    self.result = (sensor, pred, infer_ms, (time.time() - t0) * 1000)
        except Exception as e:
            self.error = e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=320, help="must match the width used for collection")
    ap.add_argument("--smooth", type=float, default=0.3, help="0 = no smoothing, closer to 1 = smoother")
    args = ap.parse_args()

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cam.isOpened():
        raise SystemExit("Could not open camera (check macOS camera permission for your terminal).")

    worker = Predictor(args.host, args.port, args.width)
    worker.start()

    shown, last_pred, preds_per_s, last_t = None, None, 0.0, time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        worker.submit(frame)
        if worker.error:
            raise SystemExit(f"Connection to board lost: {worker.error}")

        h, w = frame.shape[:2]
        view = cv2.resize(frame, (960, int(h * 960 / w)))
        vh, vw = view.shape[:2]
        cv2.drawMarker(view, (vw // 2, vh // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)

        with worker.lock:
            res = worker.result
        if res is None:
            put(view, "Waiting for the board...", 56, 1.0, (200, 200, 200))
        else:
            sensor, pred, infer_ms, rtt = res
            if pred != last_pred:                     # a new prediction arrived
                now = time.time()
                preds_per_s = 0.7 * preds_per_s + 0.3 / max(now - last_t, 1e-3)
                last_t, last_pred = now, pred
                shown = pred if shown is None else args.smooth * shown + (1 - args.smooth) * pred
            covered = not (VALID_MIN <= sensor <= VALID_MAX)
            put(view, f"Camera: {shown / 10:5.1f} cm", 56, 1.6, (255, 255, 255), 3)
            if covered:
                put(view, "Sensor: covered", 108, 1.1, (0, 165, 255))
            else:
                put(view, f"Sensor: {sensor / 10:5.1f} cm", 108, 1.1, (200, 200, 200))
                put(view, f"Error: {abs(shown - sensor) / 10:4.1f} cm", 150, 0.9, (200, 200, 200))
            put(view, f"model {infer_ms:.0f} ms on board CPU (ExecuTorch) | {preds_per_s:.1f} predictions/s",
                vh - 20, 0.6, (200, 200, 200), 1)

        cv2.imshow("sensor teaches camera", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    worker.running = False
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
