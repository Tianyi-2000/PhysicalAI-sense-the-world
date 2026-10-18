"""Board-side live demo server. Put next to collector_server.py in ~/ (needs ~/ml/depth_ml.py
and a trained model in ~/ml/model/).

For every frame from the laptop: read the distance sensor, run DINOv2 + head, reply with both.
Logs every frame to ~/depth_data/live_log.csv so the live run can also be scored.

Usage:  python live_server.py            (add --led once the sketch provides set_led(r, g, b))
"""
import argparse
import csv
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.expanduser("~/ml"))
from depth_ml import DepthPredictor, MIN_MM, MAX_MM   # noqa: E402
from collector_server import RouterClient, recv_exact  # noqa: E402


def led_colour(mm):
    if mm < 300:
        return (255, 0, 0)        # near: red
    if mm < 600:
        return (255, 160, 0)      # mid: amber
    return (0, 200, 0)            # far: green


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--led", action="store_true", help="call set_led(r, g, b) on the MCU")
    args = ap.parse_args()

    print("Loading model...")
    model = DepthPredictor()
    router = RouterClient()
    use_led = args.led

    log_path = os.path.expanduser("~/depth_data/live_log.csv")
    new = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new:
        writer.writerow(["t", "sensor_mm", "pred_mm", "infer_ms"])

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"Live server ready on port {args.port}")

    while True:
        conn, addr = srv.accept()
        print("Laptop connected:", addr)
        try:
            while True:
                _, length = struct.unpack("!dI", recv_exact(conn, 12))
                jpg = recv_exact(conn, length)

                sensor = int(router.call("get_distance"))
                t0 = time.perf_counter()
                pred = model.predict_jpeg(jpg)
                infer_ms = (time.perf_counter() - t0) * 1000

                if use_led:
                    try:
                        router.call("set_led", *led_colour(pred))
                    except Exception as e:
                        print("set_led failed, disabling LED output:", e)
                        use_led = False

                writer.writerow([f"{time.time():.3f}", sensor, f"{pred:.1f}", f"{infer_ms:.1f}"])
                conn.sendall(struct.pack("!iff", sensor, pred, infer_ms))
        except ConnectionError as e:
            log.flush()
            print(e, "- waiting for reconnect...")


if __name__ == "__main__":
    main()
