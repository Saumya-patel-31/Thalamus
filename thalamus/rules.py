"""Cross-channel rules: the part that decides what to actually say.

Two deliberate choices here.

**Rules read combinations, not single channels.** "Confusion is high" is weak.
"Confusion is climbing while interest falls" is a situation. Individual channels
mostly exist to be combined, which is why several are marked ``silent``.

**Card bodies are written by a human, in advance.** There is no language model
in this path. Jev decides *which* of a fixed set of situations you are in and
*when* you are in it; the wording was decided last Tuesday. That is what makes
the reaction arrive in 300ms instead of 3 seconds, and it is also why the
system cannot say anything surprising or unsafe. Escalate to a real model only
for the rare card that genuinely needs prose about *this* conversation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

from .dsp import Trace

__all__ = ["Card", "Rule", "Signals", "DEFAULT_RULES", "default_rules"]


@dataclass
class Card:
    rule: str
    title: str
    body: str
    severity: str
    t: float
    evidence: dict[str, float] = field(default_factory=dict)
    tick: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "title": self.title,
            "body": self.body,
            "severity": self.severity,
            "t": round(self.t, 2),
            "tick": self.tick,
            "evidence": {k: round(v, 3) for k, v in self.evidence.items()},
        }


class Signals:
    """Read-only accessor handed to every rule predicate.

    Every temporal question -- how long, how much of the last thirty seconds,
    is it still rising -- is answered here, in Python, from stored samples. None
    of it is ever asked of the model.
    """

    def __init__(self, traces: dict[str, Trace], now: float, fired: set[str],
                 silence_for: float = 0.0, last_question: str = "") -> None:
        self._t = traces
        self.now = now
        self._fired = fired
        self.silence_for = silence_for
        self.last_question = last_question

    def fired(self, key: str) -> bool:
        return key in self._fired

    def ema(self, key: str, default: float = 0.0) -> float:
        tr = self._t.get(key)
        return default if tr is None or tr.ema is None else tr.ema

    def slope(self, key: str) -> float:
        tr = self._t.get(key)
        return 0.0 if tr is None else tr.slope()

    def in_band_for(self, key: str) -> float:
        tr = self._t.get(key)
        return 0.0 if tr is None else tr.in_band_for()

    def unreliable(self, key: str) -> bool:
        tr = self._t.get(key)
        return bool(tr and tr.unreliable)

    def warm(self, key: str) -> bool:
        tr = self._t.get(key)
        return bool(tr and tr.warm)

    def fire_level(self, key: str, default: float = 0.6) -> float:
        """This channel's own fire threshold.

        Rules compare against this rather than a literal, so a threshold means
        "inside the firing band on average" in whatever units the channel
        actually produces. A Choice and a Noul do not live on the same scale.
        """
        tr = self._t.get(key)
        return default if tr is None else tr.spec.fire

    def integral(self, key: str, seconds: float) -> float:
        """Time-average of a trace over a window, in [0,1].

        This is the monologue detector. ``integral("floor:me", 30) > 0.78``
        means you have held the floor for most of the last half minute -- a fact
        about counting and duration, computed where counting is reliable.
        """
        tr = self._t.get(key)
        if tr is None:
            return 0.0
        win = tr.window(seconds)
        if len(win) < 2:
            return 0.0
        num = den = 0.0
        for a, b in zip(win, win[1:]):
            dt = b.t - a.t
            if dt <= 0:
                continue
            num += dt * (a.ema + b.ema) / 2.0
            den += dt
        return num / den if den else 0.0


@dataclass
class Rule:
    id: str
    title: str
    body: str
    when: Callable[[Signals], bool]
    severity: str = "nudge"
    refractory_s: float = 45.0
    evidence: tuple[str, ...] = ()
    _last_t: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._last_t = None

    def check(self, s: Signals) -> Card | None:
        if self._last_t is not None and (s.now - self._last_t) < self.refractory_s:
            return None
        try:
            hit = self.when(s)
        except Exception:  # a broken predicate must not take the loop down
            return None
        if not hit:
            return None
        self._last_t = s.now
        return Card(
            rule=self.id,
            title=self.title,
            body=self.body.format(q=s.last_question or "that question"),
            severity=self.severity,
            t=s.now,
            evidence={k: s.ema(k) for k in self.evidence},
        )


# ---------------------------------------------------------------------------
# The card deck. Every body pre-written; no generation in this path.
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[Rule] = [
    Rule(
        id="losing_them",
        title="You're losing them",
        body="Confusion is climbing while engagement drops. Stop, and ask them "
             "to say back what they understood so far.",
        when=lambda s: s.fired("confusion") and s.ema("interest", 0.5) < 0.5,
        severity="alert",
        evidence=("confusion", "interest"),
        refractory_s=40.0,
    ),
    Rule(
        id="confusion_only",
        title="Back up one step",
        body="They started tracking worse about a sentence ago. Re-anchor on the "
             "last thing they agreed with.",
        when=lambda s: s.fired("confusion") and s.ema("interest", 0.5) >= 0.5,
        evidence=("confusion",),
        refractory_s=45.0,
    ),
    Rule(
        id="hanging_question",
        title="Unanswered question",
        body="{q} — still hanging. Answer it or explicitly park it.",
        when=lambda s: s.fired("question_answered"),
        severity="alert",
        evidence=("question_answered",),
        refractory_s=50.0,
    ),
    Rule(
        id="vague_commitment",
        title="Pin that down",
        body="Someone just committed to something with no owner and no date. "
             "Ask: who, and by when?",
        when=lambda s: s.fired("commitment") and s.ema("commitment_bound", 1.0) < 0.45,
        severity="alert",
        evidence=("commitment", "commitment_bound"),
        refractory_s=25.0,
    ),
    Rule(
        id="clean_commitment",
        title="Action item captured",
        body="Owner and date both present — drafted, nothing needed from you.",
        when=lambda s: s.fired("commitment") and s.ema("commitment_bound", 0.0) >= 0.45,
        severity="info",
        evidence=("commitment", "commitment_bound"),
        refractory_s=15.0,
    ),
    Rule(
        id="objection_price",
        title="Price objection",
        body="Move off list price and onto cost-of-delay: what does another "
             "quarter of the status quo cost them?",
        when=lambda s: s.fired("objection:price"),
        severity="alert",
        evidence=("objection:price",),
    ),
    Rule(
        id="objection_timing",
        title="Timing objection",
        body="Don't fight the calendar. Ask what has to be true by then, and "
             "offer to start the part that has no dependency.",
        when=lambda s: s.fired("objection:timing"),
        evidence=("objection:timing",),
    ),
    Rule(
        id="objection_authority",
        title="Not the decision maker",
        body="Stop selling. Ask who else has to nod, and offer to put together "
             "the thing that gets sent to them.",
        when=lambda s: s.fired("objection:authority"),
        evidence=("objection:authority",),
    ),
    Rule(
        id="objection_trust",
        title="Trust objection",
        body="Proof, not argument. Reach for the closest reference customer and "
             "let them talk to a human.",
        when=lambda s: s.fired("objection:trust"),
        severity="alert",
        evidence=("objection:trust",),
    ),
    Rule(
        id="monologue",
        title="You've been talking a while",
        body="You've held the floor for most of the last half minute. Hand it "
             "over with a question.",
        # Not a question anyone asked the model. This is floor:me integrated.
        when=lambda s: (
            s.integral("floor:me", 30.0) > s.fire_level("floor:me")
            and s.ema("floor:me") > s.fire_level("floor:me")
            and s.warm("floor:me")
        ),
        evidence=("floor:me",),
        refractory_s=60.0,
    ),
    Rule(
        id="awkward_silence",
        title="They're waiting on you",
        body="It's gone quiet and the last thing on the table was yours to "
             "answer.",
        # Semantic silence, not a gap between transcript lines: a streaming
        # STT emits words continuously, so wall-clock gaps mean nothing.
        when=lambda s: (
            s.ema("floor:silence") > s.fire_level("floor:silence")
            and s.ema("question_answered", 1.0) < 0.5
        ),
        severity="alert",
        evidence=("question_answered",),
        refractory_s=30.0,
    ),
    Rule(
        id="talking_over",
        title="You're talking over them",
        body="Overlapping speech. Yield — they started first.",
        when=lambda s: s.fired("floor:overlap"),
        evidence=("floor:overlap",),
        refractory_s=20.0,
    ),
    Rule(
        id="repeating",
        title="Already covered",
        body="You've made this point earlier in the call. Move on.",
        when=lambda s: s.fired("repetition"),
        evidence=("repetition",),
        refractory_s=90.0,
    ),
    Rule(
        id="drifted",
        title="Off agenda",
        body="This has drifted from the agenda item for a while now. Park it or "
             "rename the meeting.",
        when=lambda s: s.fired("off_agenda"),
        evidence=("off_agenda",),
        refractory_s=120.0,
    ),
    Rule(
        id="check_that",
        title="Checkable claim",
        body="A specific figure just went on the record. Pulling the source.",
        when=lambda s: s.fired("claim_checkable"),
        severity="info",
        evidence=("claim_checkable",),
        refractory_s=25.0,
    ),
    Rule(
        id="wrap_up",
        title="You could close this",
        body="Everything needed for a decision is on the table and engagement "
             "is past its peak. Ask for the decision.",
        when=lambda s: s.fired("decision_ready") and s.slope("interest") <= 0.0,
        severity="alert",
        evidence=("decision_ready", "interest"),
        refractory_s=150.0,
    ),
]


def default_rules() -> list[Rule]:
    """A fresh, independent copy of the deck.

    ``DEFAULT_RULES`` holds *stateful* objects -- each rule remembers when it
    last fired so it can honour its own refractory period. Handing the same
    instances to two engines silently couples them: the second engine inherits
    the first's cooldowns and goes quiet for no visible reason. Always take a
    copy, which is what Engine does.
    """
    return [copy.deepcopy(r) for r in DEFAULT_RULES]
