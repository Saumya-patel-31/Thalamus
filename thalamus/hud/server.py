"""HUD transport: Server-Sent Events over stdlib http.server.

SSE rather than websockets because it is one-directional, survives reconnects
by itself, and needs no dependency. Frames carry current values only; the
browser keeps its own ring buffers and draws the scrolling traces, which keeps
each frame small enough to push several times a second without thinking about it.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX = Path(__file__).with_name("index.html")


class Broadcaster:
    def __init__(self, maxsize: int = 8) -> None:
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self.latest: dict | None = None
        self.history: list[dict] = []
        """Cards fired this session. A browser that reloads mid-call must not
        lose what already happened, so history is served separately from the
        live stream rather than repeated in every frame."""

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._clients.append(q)
        if self.latest is not None:
            try:
                q.put_nowait(self.latest)
            except queue.Full:
                pass
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, payload: dict) -> None:
        self.latest = payload
        new = payload.get("cards") or []
        if new:
            self.history.extend(new)
            del self.history[:-200]
        with self._lock:
            targets = list(self._clients)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # A slow tab must never back-pressure the decision loop.
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except queue.Empty:
                    pass


def make_server(bus: Broadcaster, host: str = "127.0.0.1", port: int = 8777):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the terminal clean
            pass

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/stream"):
                return self._stream()
            if self.path.startswith("/session"):
                return self._session()
            if self.path in ("/", "/index.html"):
                return self._index()
            self.send_error(404)

        def _session(self):
            body = json.dumps({"cards": bus.history}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _index(self):
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = bus.subscribe()
            try:
                while True:
                    try:
                        payload = q.get(timeout=10.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    chunk = f"data: {json.dumps(payload)}\n\n".encode()
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                bus.unsubscribe(q)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
