"""Laptop-side capture. Streams webcam frames to the board's collector_server.py
and shows the board's distance reading live on the video.

Setup (laptop):  pip install opencv-python
Connect over USB: adb forward tcp:5000 tcp:5000
Run:             python capture_client.py
Keys:            SPACE = start/stop recording, q = quit
"""
import argparse
import socket
import struct
import time

import cv2


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Board disconnected")
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 with adb forward, or the board's IP")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--fps", type=float, default=10.0, help="max frames per second sent while recording")
    args = ap.parse_args()

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit("Could not open camera. On macOS, allow camera access for your terminal app.")

    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Connected. SPACE to start/stop recording, q to quit.")

    recording, sent, last_send, dist = False, 0, 0.0, None
    interval = 1.0 / args.fps
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (args.width, int(h * args.width / w)))

        now = time.time()
        if recording and now - last_send >= interval:
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            data = jpg.tobytes()
            sock.sendall(struct.pack("!dI", now, len(data)) + data)
            dist = struct.unpack("!i", recv_exact(sock, 4))[0]
            sent += 1
            last_send = now

        view = small.copy()
        sh, sw = view.shape[:2]
        cv2.drawMarker(view, (sw // 2, sh // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        status = "REC" if recording else "paused"
        label = f"{status}  sent={sent}  dist={dist if dist is not None else '-'} mm"
        cv2.putText(view, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if recording else (200, 200, 200), 1)
        cv2.imshow("depth capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    sock.close()
    print(f"Done. {sent} frames sent.")


if __name__ == "__main__":
    main()
