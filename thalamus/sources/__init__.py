"""Transcript sources. Each yields (Utterance, oracle_or_None) asynchronously."""

from __future__ import annotations

__all__ = ["replay_source", "stdin_source", "dual_source", "flux_stream",
           "list_devices", "detect_loopback"]


def replay_source(*a, **kw):
    from .replay import replay_source as f

    return f(*a, **kw)


def stdin_source(*a, **kw):
    from .stdin import stdin_source as f

    return f(*a, **kw)


def dual_source(*a, **kw):
    from .deepgram import dual_source as f

    return f(*a, **kw)


def flux_stream(*a, **kw):
    from .deepgram import flux_stream as f

    return f(*a, **kw)


def list_devices(*a, **kw):
    from .audio import list_devices as f

    return f(*a, **kw)


def detect_loopback(*a, **kw):
    from .audio import detect_loopback as f

    return f(*a, **kw)
