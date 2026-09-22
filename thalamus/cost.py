"""Live cost and rate accounting.

A continuous loop that silently bills you is not a product, so the meter is a
first-class component and it is on screen the whole time. It also watches the
documented rate ceilings (250,000 tokens/sec, 1,200 requests/min) so you find
out you are near a limit from the HUD rather than from a 429.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

PRICE_PER_MTOK = 0.042
TOKENS_PER_SEC_LIMIT = 250_000
REQUESTS_PER_MIN_LIMIT = 1_200


@dataclass
class Meter:
    window_s: float = 20.0
    events: deque[tuple[float, int]] = field(default_factory=deque)
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    total_tokens: int = 0
    total_calls: int = 0
    gate_calls: int = 0
    bank_calls: int = 0
    gate_tokens: int = 0
    bank_tokens: int = 0
    errors: int = 0
    clock: Callable[[], float] = time.monotonic
    """Shared with the engine. A virtual-clock replay must report the rates
    the same run would produce live, not rates against wall time."""
    started: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.started:
            self.started = self.clock()

    def record(self, tokens: int, latency_ms: float, *, bank: bool) -> None:
        now = self.clock()
        self.events.append((now, tokens))
        self.latencies.append(latency_ms)
        self.total_tokens += tokens
        self.total_calls += 1
        if bank:
            self.bank_calls += 1
            self.bank_tokens += tokens
        else:
            self.gate_calls += 1
            self.gate_tokens += tokens
        self._trim(now)

    def _trim(self, now: float) -> None:
        while self.events and now - self.events[0][0] > self.window_s:
            self.events.popleft()

    @property
    def tokens_per_sec(self) -> float:
        if not self.events:
            return 0.0
        now = self.clock()
        self._trim(now)
        span = max(now - self.events[0][0], 1e-3)
        return sum(n for _, n in self.events) / span

    @property
    def requests_per_min(self) -> float:
        if not self.events:
            return 0.0
        now = self.clock()
        span = max(now - self.events[0][0], 1e-3)
        return len(self.events) / span * 60.0

    @property
    def dollars_per_hour(self) -> float:
        return self.tokens_per_sec * 3600 / 1_000_000 * PRICE_PER_MTOK

    @property
    def spent(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_MTOK

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        xs = sorted(self.latencies)
        k = min(int(p / 100 * len(xs)), len(xs) - 1)
        return xs[k]

    @property
    def gate_saving(self) -> float:
        """Fraction of tokens the two-stage split actually saved.

        Negative means the gate is *costing* you money -- which happens whenever
        the gate opens often and the state dominates the token count, because
        the state then gets billed twice on every open tick. Reported rather
        than assumed, because the intuition that a cheap pre-filter must save
        money is wrong for a large enough share of operating points.
        """
        if not self.bank_calls or not self.gate_calls:
            return 0.0
        avg_bank = self.bank_tokens / self.bank_calls
        single_stage = avg_bank * self.gate_calls   # full bank on every tick
        actual = self.gate_tokens + self.bank_tokens
        return (single_stage - actual) / single_stage if single_stage else 0.0

    @property
    def gate_open_rate(self) -> float:
        return self.bank_calls / self.gate_calls if self.gate_calls else 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "requests_per_min": round(self.requests_per_min, 1),
            "dollars_per_hour": round(self.dollars_per_hour, 4),
            "spent": round(self.spent, 6),
            "p50_ms": round(self.percentile(50), 1),
            "p95_ms": round(self.percentile(95), 1),
            "calls": self.total_calls,
            "gate_open_rate": round(self.gate_open_rate, 3),
            "gate_saving": round(self.gate_saving, 3),
            "errors": self.errors,
            "uptime_s": round(self.clock() - self.started, 1),
            "tok_limit_pct": round(self.tokens_per_sec / TOKENS_PER_SEC_LIMIT * 100, 2),
            "req_limit_pct": round(self.requests_per_min / REQUESTS_PER_MIN_LIMIT * 100, 2),
        }
