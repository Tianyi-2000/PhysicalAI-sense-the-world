"""Board-side collector. Run on the VENTUNO Q (in ~/venv) while the App Lab
sketch exposing "get_distance" is running.

Receives JPEG frames from the laptop over TCP, reads the distance sensor the
moment each frame arrives, and saves (frame, distance) pairs.

Usage:  python collector_server.py --session train1
Output: ~/depth_data/<session>/frames/*.jpg  and  ~/depth_data/<session>/labels.csv
"""
import argparse
import csv
import os
import socket
import struct
import time

import msgpack

ROUTER_SOCK = "/var/run/arduino-router.sock"


class RouterClient:
    """Minimal MessagePack-RPC client for the Arduino Router."""

    def __init__(self, path=ROUTER_SOCK, timeout=1.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.unpacker = msgpack.Unpacker(raw=False)
        self.msgid = 0

    def call(self, method, *args):
        self.msgid += 1
        msgid = self.msgid
        self.sock.sendall(msgpack.packb([0, msgid, method, list(args)]))
        while True:
            for msg in self.unpacker:
                if isinstance(msg, list) and len(msg) == 4 and msg[0] == 1 and msg[1] == msgid:
                    if msg[2] is not None:
                        raise RuntimeError(f"RPC error: {msg[2]}")
                    return msg[3]
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Router closed the connection")
            self.unpacker.feed(chunk)


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Laptop disconnected")
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="name for this recording, e.g. train1, test_mug")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    out_dir = os.path.expanduser(f"~/depth_data/{args.session}")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "labels.csv")
    new_file = not os.path.exists(csv_path)

    router = RouterClient()
    print("Sensor check:", router.call("get_distance"), "mm")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"Waiting for laptop on port {args.port} (session '{args.session}')...")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["idx", "file", "t_capture_laptop", "t_recv_board", "distance_mm", "rpc_ms"])
        idx = sum(1 for _ in os.scandir(frames_dir))

        while True:
            conn, addr = srv.accept()
            print("Laptop connected:", addr)
            try:
                while True:
                    # Header: capture timestamp (double) + JPEG length (uint32)
                    t_capture, length = struct.unpack("!dI", recv_exact(conn, 12))
                    jpg = recv_exact(conn, length)

                    t_recv = time.time()
                    t0 = time.perf_counter()
                    dist = router.call("get_distance")
                    rpc_ms = (time.perf_counter() - t0) * 1000

                    name = f"{idx:06d}.jpg"
                    with open(os.path.join(frames_dir, name), "wb") as img:
                        img.write(jpg)
                    writer.writerow([idx, name, f"{t_capture:.4f}", f"{t_recv:.4f}", dist, f"{rpc_ms:.2f}"])
                    idx += 1
                    if idx % 50 == 0:
                        f.flush()
                        print(f"{idx} pairs saved, last distance {dist} mm")

                    conn.sendall(struct.pack("!i", int(dist)))
            except ConnectionError as e:
                f.flush()
                print(e, f"({idx} pairs total). Waiting for reconnect...")


if __name__ == "__main__":
    main()
