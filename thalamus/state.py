"""What Jev is allowed to see.

Accuracy is documented to fall as irrelevant state is added -- "context rot" --
so the rolling window is built narrowly and deliberately in code. The model
gets the last few seconds of conversation plus a small, code-maintained brief.
It does not get the whole transcript, and it never gets anything it would have
to count or date-sort.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

__all__ = ["Utterance", "RollingState"]


@dataclass
class Utterance:
    t: float
    speaker: str
    text: str
    final: bool = True


@dataclass
class RollingState:
    """A bounded, self-pruning view of the conversation."""

    window_s: float = 45.0
    max_chars: int = 3000
    agenda: str = ""
    brief: dict[str, str] = field(default_factory=dict)
    utterances: deque[Utterance] = field(default_factory=lambda: deque(maxlen=400))

    def add(self, u: Utterance) -> None:
        self.utterances.append(u)

    def prune(self, now: float) -> None:
        while self.utterances and now - self.utterances[0].t > self.window_s:
            self.utterances.popleft()

    @property
    def last_speaker(self) -> str | None:
        return self.utterances[-1].speaker if self.utterances else None

    def silence_for(self, now: float) -> float:
        return now - self.utterances[-1].t if self.utterances else 0.0

    def transcript(self) -> list[str]:
        """Speaker-tagged lines. ME is the microphone wearer, which is what the
        ``floor`` channel keys off."""
        return [f"{u.speaker}: {u.text}" for u in self.utterances]

    def build(
        self,
        now: float,
        *,
        window_s: float | None = None,
        max_chars: int | None = None,
    ) -> dict[str, object]:
        """The payload. Kept to a handful of named fields, newest-last.

        Note what is absent: no timestamps, no turn counts, no durations. Those
        exist in this process but are withheld, because Jev is unreliable on
        ordering and arithmetic and would be given the chance to get them wrong.
        """
        self.prune(now)
        window_s = self.window_s if window_s is None else window_s
        budget = self.max_chars if max_chars is None else max_chars
        lines = [f"{u.speaker}: {u.text}" for u in self.utterances
                 if now - u.t <= window_s]
        # Trim from the front until under the character budget.
        while lines and sum(len(x) + 1 for x in lines) > budget:
            lines.pop(0)

        state: dict[str, object] = {"recent_conversation": lines}
        if self.agenda:
            state["agenda"] = self.agenda
        if self.brief:
            state["brief"] = self.brief
        return state
