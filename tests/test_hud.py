"""HUD transport. Mostly: a browser going away is not an error.

An SSE stream is held open for the whole call, so *every* session ends with the
client vanishing -- tab closed, page reloaded, laptop slept. socketserver's
default behaviour is to print a full traceback for each one, which buries the
actual log under noise and makes a working tool look broken.
"""

from __future__ import annotations

import io
import json
import socket
import struct
import sys
import threading
import time
import urllib.request
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalamus.hud.server import Broadcaster, make_server


def _serve(port: int):
    bus = Broadcaster()
    server = make_server(bus, port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return bus, server


def _frame(t: float = 1.0) -> dict:
    return {"t": t, "tick": 1, "cards": [], "traces": {}, "meter": {}}


def test_a_client_vanishing_mid_stream_prints_nothing():
    """The reported bug: ConnectionResetError dumped a traceback per tab close."""
    bus, server = _serve(8823)
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            for _ in range(3):
                s = socket.create_connection(("127.0.0.1", 8823), timeout=3)
                s.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
                bus.publish(_frame())
                time.sleep(0.12)
                # SO_LINGER with a zero timeout makes close() send RST rather
                # than FIN -- exactly what a hard tab close looks like.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                             struct.pack("ii", 1, 0))
                s.close()
                time.sleep(0.25)
    finally:
        server.shutdown()
    noise = err.getvalue()
    assert "Traceback" not in noise, f"still printing tracebacks:\n{noise[:600]}"
    assert "ConnectionResetError" not in noise, noise[:600]


def test_the_server_survives_those_disconnects():
    """Silencing the noise must not mean silently dying."""
    bus, server = _serve(8824)
    try:
        for _ in range(3):
            s = socket.create_connection(("127.0.0.1", 8824), timeout=3)
            s.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
            bus.publish(_frame())
            time.sleep(0.1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            s.close()
        time.sleep(0.2)
        body = urllib.request.urlopen("http://127.0.0.1:8824/", timeout=3).read()
        # The brand renders as THAL<span>A</span>MUS, so match the <title>.
        assert b"<title>Thalamus</title>" in body, "server stopped serving after resets"
    finally:
        server.shutdown()


def test_session_history_survives_a_reload():
    """Reloading mid-call must not wipe the cards already fired."""
    bus, server = _serve(8825)
    try:
        bus.publish({**_frame(2.0), "cards": [{"rule": "a", "title": "One", "t": 2.0}]})
        bus.publish({**_frame(5.0), "cards": [{"rule": "b", "title": "Two", "t": 5.0}]})
        bus.publish(_frame(6.0))  # a quiet tick must not erase anything
        got = json.loads(
            urllib.request.urlopen("http://127.0.0.1:8825/session", timeout=3).read()
        )
        assert [c["title"] for c in got["cards"]] == ["One", "Two"]
    finally:
        server.shutdown()


def test_a_slow_tab_never_back_pressures_the_decision_loop():
    """A browser that stops reading must not stall the engine's tick."""
    bus = Broadcaster(maxsize=2)
    q = bus.subscribe()
    started = time.monotonic()
    for i in range(50):
        bus.publish(_frame(float(i)))       # far more than the queue holds
    assert time.monotonic() - started < 1.0, "publish blocked on a full queue"
    assert q.qsize() <= 2
    # The subscriber keeps the newest frames, not the stalest.
    assert q.get_nowait()["t"] >= 48.0


def test_unknown_paths_404():
    bus, server = _serve(8826)
    try:
        try:
            urllib.request.urlopen("http://127.0.0.1:8826/nope", timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("expected 404")
    finally:
        server.shutdown()


if __name__ == "__main__":
    import urllib.error  # noqa: F401  (used above)

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
