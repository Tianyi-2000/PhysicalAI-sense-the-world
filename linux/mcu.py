"""Call the MCU sketch's Bridge handlers through the Arduino Router socket.

server.py runs outside App Lab, so it talks MessagePack-RPC to the Router directly (same protocol
as msgpack_test.py). If the Router or the sketch isn't there, e.g. on a laptop, calls fail fast and
the web demo keeps working.
"""

import itertools
import socket
import time

import msgpack

ROUTER_SOCKET = "/var/run/arduino-router.sock"
_ids = itertools.count(1)


def call(method, *args, timeout=1.0):
    """Call an MCU handler. Return the round-trip time in ms, or None if the MCU didn't answer."""
    start = time.perf_counter()
    msg_id = next(_ids)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(ROUTER_SOCKET)
            client.sendall(msgpack.packb([0, msg_id, method, list(args)], use_bin_type=True))
            unpacker = msgpack.Unpacker(raw=False, max_buffer_size=65536)
            while True:
                data = client.recv(4096)
                if not data:
                    return None
                unpacker.feed(data)
                for msg in unpacker:
                    if isinstance(msg, list) and len(msg) == 4 and msg[0] == 1 and msg[1] == msg_id:
                        return None if msg[2] is not None else (time.perf_counter() - start) * 1000
    except (OSError, ValueError, msgpack.UnpackException):
        return None
