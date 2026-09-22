"""The tick loop.

Two stages, for one reason: money. A full read of the bank every tick at 5Hz is
roughly five dollars an hour. A cheap three-question gate every tick, opening
the full bank only when the conversation actually moved, is roughly one. Same
responsiveness, a fifth of the bill, and the gate itself is three of the most
useful channels so nothing is wasted.

Exactly one request is ever in flight. If a tick comes due while the previous
call is still out, it is dropped rather than queued -- a backlog of stale
judgments about a conversation that has moved on is worse than no judgment.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .backends import Backend, Verdict
from .channels import (
    BANK, Channel, Tier, build_questions, gate_channels, parse_answers, trace_specs,
)
from .calibration import Calibration
from .cost import Meter
from .dsp import Trace
from .record import Recorder
from .rules import Card, Rule, Signals, default_rules
from .state import RollingState, Utterance

__all__ = ["Engine", "Frame"]


@dataclass
class Frame:
    t: float
    tick: int
    gate_open: bool
    traces: dict[str, dict[str, Any]]
    cards: list[dict[str, Any]]
    transcript: list[str]
    meter: dict[str, float]
    backend: str
    model: str
    calibrated: bool = False
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "tick": self.tick,
            "gate_open": self.gate_open,
            "traces": self.traces,
            "cards": self.cards,
            "transcript": self.transcript,
            "meter": self.meter,
            "backend": self.backend,
            "model": self.model,
            "calibrated": self.calibrated,
            "error": self.error,
        }


class Engine:
    def __init__(
        self,
        backend: Backend,
        *,
        rate_hz: float = 5.0,
        rules: list[Rule] | None = None,
        state: RollingState | None = None,
        calibration: Calibration | None = None,
        gate_threshold: float = 0.62,
        bank_max_gap_s: float = 8.0,
        gate_window_s: float = 15.0,
        gate_max_chars: int = 700,
        on_frame: Callable[[Frame], None] | None = None,
        recorder: Recorder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.rate_hz = rate_hz
        # Rules carry refractory state, so never share instances between engines.
        self.rules = [copy.deepcopy(r) for r in rules] if rules is not None else default_rules()
        for r in self.rules:
            r.reset()
        self.state = state or RollingState()
        self.calibration = calibration or Calibration.identity()
        self.gate_threshold = gate_threshold
        self.bank_max_gap_s = bank_max_gap_s
        self.gate_window_s = gate_window_s
        self.gate_max_chars = gate_max_chars
        self.on_frame = on_frame
        self.recorder = recorder
        self.clock = clock

        self.meter = Meter(clock=clock)
        self.tick_count = 0
        self.cards: list[Card] = []
        self.oracle: dict[str, Any] = {}
        self.last_error: str | None = None
        self.model = "-"
        self._busy = False
        self._last_bank_t: float | None = None
        self._dropped_ticks = 0
        self._t0 = self.clock()

        self._specs = trace_specs(rate_hz)
        self.traces: dict[str, Trace] = {
            key: Trace(key, spec) for key, (_, _, spec) in self._specs.items()
        }
        self._gate_qs = build_questions(gate_channels())
        self._bank_only = tuple(c for c in BANK if c.tier is not Tier.GATE)
        self._bank_qs = build_questions(self._bank_only)

    # -- ingestion -----------------------------------------------------------

    def ingest(self, u: Utterance, oracle: dict[str, Any] | None = None) -> None:
        self.state.add(u)
        if oracle:
            self.oracle.update(oracle)

    # -- one tick ------------------------------------------------------------

    async def tick(self) -> Frame | None:
        if self._busy:
            self._dropped_ticks += 1
            return None
        self._busy = True
        try:
            return await self._tick_inner()
        finally:
            self._busy = False

    async def _tick_inner(self) -> Frame:
        now = self.clock() - self._t0
        wall = self.clock()
        self.tick_count += 1

        # The gate sees a deliberately narrow slice; the bank sees the window.
        gate_payload = self.state.build(
            wall, window_s=self.gate_window_s, max_chars=self.gate_max_chars
        )
        gate = await self.backend.evaluate(gate_payload, self._gate_qs, oracle=self.oracle)
        self._record(gate, bank=False)
        answers = dict(gate.answers)

        open_gate = self._should_open(gate, now)
        if open_gate:
            payload = self.state.build(wall)
            bank = await self.backend.evaluate(payload, self._bank_qs, oracle=self.oracle)
            self._record(bank, bank=True)
            answers.update(bank.answers)
            self._last_bank_t = now

        fired = self._absorb(now, answers)
        cards = self._fire_rules(now, fired)
        return self._frame(now, open_gate, cards)

    def _record(self, v: Verdict, *, bank: bool) -> None:
        if v.error:
            self.last_error = v.error
            self.meter.errors += 1
        else:
            self.last_error = None
            self.model = v.model
        self.meter.record(v.input_tokens, v.latency_ms, bank=bank)

    def _should_open(self, gate: Verdict, now: float) -> bool:
        """Open the bank when the tick looks interesting, or when the bank's own
        traces are going stale. Warm up fully open so nothing starts cold."""
        if now < 3.0:
            return True
        if self._last_bank_t is None or (now - self._last_bank_t) > self.bank_max_gap_s:
            return True
        ev = (gate.answers.get("eventful") or {}).get("noul")
        return ev is not None and float(ev) >= self.gate_threshold

    def _absorb(self, now: float, answers: dict[str, Any]) -> set[str]:
        """Calibrate, push into traces, collect rising edges."""
        parsed = parse_answers(answers, BANK)
        fired: set[str] = set()
        context = self.state.transcript()[-3:] if self.recorder else []
        for key, (raw, alt) in parsed.items():
            tr = self.traces.get(key)
            if tr is None:
                continue
            value = self.calibration.apply(key, raw)
            alt_c = self.calibration.apply(key, alt) if alt is not None else None
            pushed = tr.push(now, value, raw=raw, alt=alt_c)
            if self.recorder is not None:
                # Record the *raw* value: calibration is what we are trying to
                # fit, so labelling its own output would be circular.
                self.recorder.observe(
                    key=key, raw=raw, alt=alt, ema=tr.ema, t=now,
                    tick=self.tick_count, context=context,
                    instructions=self._question_for(key),
                )
            if pushed:
                ch, _, _ = self._specs[key]
                if not ch.silent:
                    fired.add(key)
        return fired

    def _question_for(self, key: str) -> str:
        """The question a human is being asked to answer, in plain words."""
        ch, opt, _ = self._specs[key]
        if opt is not None:
            label = (ch.criteria or {}).get(opt, opt)
            return f"{ch.instructions} — is it {opt!r} ({label})?"
        return ch.instructions

    def _fire_rules(self, now: float, fired: set[str]) -> list[Card]:
        sig = Signals(
            self.traces,
            now,
            fired,
            silence_for=self.state.silence_for(self.clock()),
            last_question=self._last_question(),
        )
        out: list[Card] = []
        for rule in self.rules:
            card = rule.check(sig)
            if card is not None:
                card.tick = self.tick_count
                out.append(card)
                self.cards.append(card)
        del self.cards[:-60]
        return out

    def _last_question(self) -> str:
        for u in reversed(self.state.utterances):
            if "?" in u.text:
                frag = u.text.strip()
                return frag if len(frag) <= 90 else frag[:87] + "..."
        return ""

    def _frame(self, now: float, gate_open: bool, cards: list[Card]) -> Frame:
        traces: dict[str, Any] = {}
        for key, tr in self.traces.items():
            ch, opt, spec = self._specs[key]
            last = tr.samples[-1] if tr.samples else None
            traces[key] = {
                "label": key.replace(":", " · ").replace("_", " "),
                "channel": ch.id,
                "tier": ch.tier.value,
                "why": ch.why,
                "silent": ch.silent,
                "ema": None if tr.ema is None else round(tr.ema, 4),
                "raw": None if last is None else round(last.raw, 4),
                "alt": None if last is None or last.alt is None else round(last.alt, 4),
                "disagreement": None if tr.disagreement is None else round(tr.disagreement, 4),
                "fire": spec.fire,
                "release": spec.release,
                "polarity": spec.polarity.value,
                "tau": spec.ema_tau_s,
                "dwell": spec.dwell_s,
                "accuracy": ch.accuracy,
                "latched": tr.latched,
                "warm": tr.warm,
                "unreliable": tr.unreliable,
                "fire_count": tr.fire_count,
                "slope": round(tr.signed_slope, 4),
            }
        frame = Frame(
            t=now,
            tick=self.tick_count,
            gate_open=gate_open,
            traces=traces,
            cards=[c.to_json() for c in cards],
            transcript=self.state.transcript()[-8:],
            meter={**self.meter.snapshot(), "dropped_ticks": self._dropped_ticks},
            backend=self.backend.name,
            model=self.model,
            calibrated=self.calibration.fitted,
            error=self.last_error,
        )
        if self.on_frame:
            self.on_frame(frame)
        return frame

    # -- driving -------------------------------------------------------------

    async def run(self, source, *, until: float | None = None) -> None:
        """Consume a transcript source and tick at rate_hz until it ends."""
        stop = asyncio.Event()

        async def pump() -> None:
            try:
                async for item in source:
                    u, oracle = item if isinstance(item, tuple) else (item, None)
                    self.ingest(u, oracle)
            finally:
                stop.set()

        async def ticker() -> None:
            period = 1.0 / self.rate_hz
            while not stop.is_set():
                started = self.clock()
                await self.tick()
                if until is not None and (self.clock() - self._t0) > until:
                    stop.set()
                    break
                elapsed = self.clock() - started
                await asyncio.sleep(max(period - elapsed, 0.0))

        await asyncio.gather(pump(), ticker())
