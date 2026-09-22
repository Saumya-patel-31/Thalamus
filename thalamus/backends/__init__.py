"""Decision backends. One real, one honest fake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["Verdict", "Backend", "get_backend"]


@dataclass
class Verdict:
    answers: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"
    error: str | None = None


class Backend(Protocol):
    name: str

    async def evaluate(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        oracle: dict[str, Any] | None = None,
    ) -> Verdict:
        """Answer every question against shared state, in one round trip.

        ``oracle`` carries demo ground truth. Real backends must ignore it; it
        never crosses the network.
        """
        ...


def get_backend(kind: str, **kw) -> Backend:
    if kind == "jev":
        from .jev import JevBackend

        return JevBackend(**kw)
    if kind == "mock":
        from .mock import MockBackend

        return MockBackend(**kw)
    raise ValueError(f"unknown backend {kind!r}; expected 'jev' or 'mock'")
