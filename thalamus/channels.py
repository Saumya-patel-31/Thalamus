"""The sensor bank.

A channel is one semantic question asked of the world, repeatedly, forever.

Three rules govern every channel here, all of them consequences of how Jev
actually behaves rather than matters of taste:

1. **Phrase positively.** Jev reads literally and negations land at face value,
   so "the question has NOT been answered" is a trap. Ask whether it *has* been
   answered and watch for the probability collapsing (``Polarity.LOW``).

2. **Never ask about time, counts or arithmetic.** Those are documented weak
   spots. A channel reports only what is true *at this instant*; dsp.py owns
   every integral, duration and tally. "Have I been monologuing" is not a
   question -- it is ``floor`` integrated over thirty seconds, in Python.

3. **Ask everything twice.** Output tokens are free and questions evaluate in
   parallel, so every channel carries a second, differently-worded phrasing.
   Agreement costs nothing; disagreement is a free measurement that the
   *question* is ill-posed, which is not obtainable from a model you only
   ask once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .dsp import Polarity, TriggerSpec, suggest_band

__all__ = ["Tier", "Channel", "BANK", "gate_channels", "bank_channels",
           "build_questions", "parse_answers", "trace_specs"]


class Tier(str, Enum):
    """Which stage of the two-stage loop a channel belongs to.

    GATE channels run on every tick -- a handful of cheap questions whose only
    job is to decide whether the tick is interesting. BANK channels run only
    when the gate opens. This is what keeps a continuously-running loop at
    about a dollar an hour instead of five.
    """

    GATE = "gate"
    BANK = "bank"


@dataclass(frozen=True)
class Channel:
    id: str
    kind: Literal["noul", "choice", "score"]
    instructions: str
    why: str
    """Why this channel exists. If you cannot write this line, delete the channel."""
    alt: str | None = None
    criteria: Any = None
    tier: Tier = Tier.BANK
    accuracy: float = 0.78
    """Measured hit rate. 0.78 is a placeholder standing between TypeSafe's own
    workflow evals (~0.68) and independent public-benchmark numbers (~0.72-0.77).
    Replace it with your own measurement -- scripts/fit_calibration.py prints
    per-channel accuracy, and accuracy buys response latency quadratically."""
    watch: tuple[str, ...] = ()
    """For choice channels: which options get their own trace. One Jev question,
    several signals, no extra latency."""
    polarity: Polarity = Polarity.HIGH
    dwell_s: float = 0.0
    refractory_s: float = 25.0
    min_slope: float | None = None
    fire_at: float = 0.65
    p_hit: float | None = None
    p_miss: float | None = None
    """Where this channel's probability mass actually lands when it is right
    and when it is wrong. Left as None these are derived from ``kind``: a Noul
    that is confident sits near 0.90, but a Choice must divide its mass across
    N options and so peaks far lower. Feeding a Noul's 0.90 into the band
    derivation for a five-way Choice puts the fire line *above* the channel's
    own true attractor, and it then never fires -- which is exactly the bug
    this field exists to prevent."""
    silent: bool = False
    """Traced and displayed, but never fires on its own. For channels that exist
    only to feed cross-channel rules -- ``floor`` is the archetype."""

    @property
    def alt_id(self) -> str:
        return f"{self.id}~alt"

    def mass(self) -> tuple[float, float]:
        """(p_hit, p_miss) for this channel, derived from its shape."""
        if self.p_hit is not None and self.p_miss is not None:
            return self.p_hit, self.p_miss
        if self.kind == "choice":
            n = max(len(self.criteria or {}), 2)
            hit = 0.74                      # typical winning mass, not 0.90
            return hit, (1.0 - hit) / (n - 1)
        if self.kind == "score":
            # Normalised score position: less peaked than a Noul, no hard floor.
            return 0.82, 0.18
        return 0.90, 0.10

    def trace_keys(self) -> tuple[str, ...]:
        if self.kind == "choice":
            return tuple(f"{self.id}:{opt}" for opt in self.watch)
        return (self.id,)


# ---------------------------------------------------------------------------
# The bank. Twelve channels, which is the v1 line -- enough to be uncanny,
# few enough that every one can be justified in a sentence.
# ---------------------------------------------------------------------------

BANK: tuple[Channel, ...] = (
    # -- gate tier: cheap, every tick, decides if the tick matters -----------
    Channel(
        id="eventful",
        kind="noul",
        tier=Tier.GATE,
        instructions="Something just happened in the last few seconds that a "
                     "listener might need to respond to",
        alt="The most recent turn contains a substantive development rather "
            "than filler or small talk",
        why="The gate. Opens the expensive bank only when the conversation "
            "actually moved.",
        accuracy=0.80,
        silent=True,
    ),
    Channel(
        id="floor",
        kind="choice",
        tier=Tier.GATE,
        instructions="Who is speaking at the end of this transcript",
        alt="Whose voice occupies the final moments of this transcript",
        criteria={
            "me": "The person wearing the microphone, labelled ME",
            "them": "Any other participant",
            "silence": "Nobody is speaking; the transcript has gone quiet",
            "overlap": "Two or more people talking at once",
        },
        watch=("me", "silence", "overlap"),
        why="Never asked 'am I talking too much'. Code integrates this over "
            "30s to get monologue detection, because Jev cannot count.",
        accuracy=0.86,
        silent=True,
    ),
    Channel(
        id="question_answered",
        kind="noul",
        tier=Tier.GATE,
        instructions="The most recent question in this transcript has received "
                     "a substantive answer",
        alt="Every question asked in this conversation has been addressed by a "
            "later reply",
        why="The single most useful signal in a real meeting. Phrased positively "
            "and watched for collapse, because Jev mishandles negation.",
        accuracy=0.80,
        polarity=Polarity.LOW,
        dwell_s=6.0,
        refractory_s=45.0,
    ),

    # -- bank tier: the full read, only when the gate opens ------------------
    Channel(
        id="confusion",
        kind="score",
        instructions="How lost the listener sounds right now",
        alt="How much difficulty the other participant is having following the "
            "explanation",
        criteria=[
            "Following along easily",
            "Hesitant; asking for repetition or clarification",
            "Visibly lost; misstating what was said",
        ],
        why="Fires on the derivative, not the level. Permanent confusion is not "
            "actionable; confusion *climbing* means you are losing them now.",
        accuracy=0.76,
        min_slope=0.04,
        refractory_s=30.0,
    ),
    Channel(
        id="objection",
        kind="choice",
        instructions="What kind of pushback the other participant is raising",
        alt="Which category best describes the other participant's current "
            "resistance",
        criteria={
            "price": "Cost, budget, or value for money",
            "timing": "Not now, bad quarter, other priorities",
            "authority": "Needs someone else to decide",
            "trust": "Doubts it works, or doubts the speaker",
            "none": "No resistance is being expressed",
        },
        watch=("price", "timing", "authority", "trust"),
        why="Four traces from one question. Each maps to a different prepared "
            "card, so the useful response is retrieval, not generation.",
        accuracy=0.78,
        dwell_s=1.5,
    ),
    Channel(
        id="commitment",
        kind="noul",
        instructions="A speaker has just committed to doing something specific",
        alt="Someone in this conversation has taken on an action or promised a "
            "next step",
        why="Rising edge writes a draft action item. Paired with "
            "commitment_bound to catch the vague ones.",
        accuracy=0.80,
        dwell_s=0.5,
        refractory_s=15.0,
    ),
    Channel(
        id="commitment_bound",
        kind="noul",
        instructions="The most recent commitment names both a specific date and "
                     "a specific owner",
        alt="The latest promised next step is pinned to a named person and a "
            "named deadline",
        why="The valuable half of the pair: a commitment with nobody's name and "
            "no date is the one worth interrupting about.",
        accuracy=0.76,
        polarity=Polarity.LOW,
        dwell_s=2.0,
    ),
    Channel(
        id="claim_checkable",
        kind="noul",
        instructions="A specific factual or numerical claim was just made that "
                     "a document could verify",
        alt="The last turn asserted a concrete figure or fact that could be "
            "looked up",
        why="Triggers retrieval, not an opinion. Jev decides there is something "
            "worth checking; code does the checking.",
        accuracy=0.78,
        dwell_s=1.0,
        refractory_s=20.0,
    ),
    Channel(
        id="interest",
        kind="score",
        instructions="How engaged the other participant sounds",
        alt="How much genuine enthusiasm the other party is showing",
        criteria=[
            "Disengaged; short, flat replies",
            "Politely attentive",
            "Leaning in; asking follow-up questions",
        ],
        why="Used as a *modifier* on other channels rather than a trigger. "
            "Falling interest plus rising confusion is a different situation "
            "from either alone.",
        accuracy=0.74,
        silent=True,
    ),
    Channel(
        id="repetition",
        kind="noul",
        instructions="The speaker is restating a point they have already made "
                     "earlier in this conversation",
        alt="The current turn repeats substance that appeared earlier in the "
            "transcript",
        why="The politest possible intervention, and one nobody can self-detect "
            "while talking.",
        accuracy=0.74,
        dwell_s=2.0,
        refractory_s=60.0,
    ),
    Channel(
        id="off_agenda",
        kind="noul",
        instructions="The current topic matches the stated agenda in the state",
        alt="The conversation is presently discussing the agenda item it set "
            "out to discuss",
        why="Drift is gradual, which is exactly what a smoothed trace is good "
            "at and a human in the conversation is bad at.",
        accuracy=0.76,
        polarity=Polarity.LOW,
        dwell_s=12.0,
        refractory_s=90.0,
    ),
    Channel(
        id="decision_ready",
        kind="noul",
        instructions="Enough has been discussed that a decision could be made "
                     "right now",
        alt="This conversation has reached a point where the participants could "
            "reasonably close the matter",
        why="The wrap-up nudge. Meetings overrun because nobody notices the "
            "moment the useful part ended.",
        accuracy=0.72,
        dwell_s=8.0,
        refractory_s=120.0,
    ),
)


def gate_channels() -> tuple[Channel, ...]:
    return tuple(c for c in BANK if c.tier is Tier.GATE)


def bank_channels() -> tuple[Channel, ...]:
    return BANK


def by_id(channel_id: str) -> Channel:
    for c in BANK:
        if c.id == channel_id:
            return c
    raise KeyError(channel_id)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def _question_body(ch: Channel, instructions: str) -> dict[str, Any]:
    q: dict[str, Any] = {"type": ch.kind, "instructions": instructions}
    if ch.kind == "choice":
        q["criteria"] = ch.criteria
    elif ch.kind == "score":
        q["criteria"] = list(ch.criteria or [])
    return q


def build_questions(channels: tuple[Channel, ...]) -> dict[str, dict[str, Any]]:
    """Flatten channels into one Jev ``questions`` map, both phrasings included.

    Everything goes in a single request on purpose: questions are evaluated in
    parallel against shared state, so a second phrasing is close to free in
    wall-clock terms even though it costs its own input tokens.
    """
    out: dict[str, dict[str, Any]] = {}
    for ch in channels:
        out[ch.id] = _question_body(ch, ch.instructions)
        if ch.alt:
            out[ch.alt_id] = _question_body(ch, ch.alt)
    return out


def _value_of(ch: Channel, answer: dict[str, Any], option: str | None) -> float:
    """Collapse a Jev answer into the single [0,1] signal we trace."""
    if ch.kind == "noul":
        return float(answer.get("noul", 0.0))
    if ch.kind == "choice":
        probs = answer.get("probabilities") or {}
        return float(probs.get(option, 0.0))
    # score: normalise position on the scale to [0,1]. `.score` can land between
    # levels, which is a feature -- it makes the trace continuous.
    levels = max(len(ch.criteria or []) - 1, 1)
    return min(max(float(answer.get("score", 0.0)) / levels, 0.0), 1.0)


def parse_answers(
    answers: dict[str, Any], channels: tuple[Channel, ...]
) -> dict[str, tuple[float, float | None]]:
    """Map a Jev response onto ``{trace_key: (value, alt_value)}``."""
    out: dict[str, tuple[float, float | None]] = {}
    for ch in channels:
        primary = answers.get(ch.id)
        if primary is None:
            continue
        alt_ans = answers.get(ch.alt_id) if ch.alt else None
        options = ch.watch if ch.kind == "choice" else (None,)
        for opt in options:
            key = f"{ch.id}:{opt}" if opt is not None else ch.id
            value = _value_of(ch, primary, opt)
            alt = _value_of(ch, alt_ans, opt) if alt_ans is not None else None
            out[key] = (value, alt)
    return out


GATE_OPEN_ASSUMPTION = 0.40
"""How often the gate is expected to open during an active conversation.
Bank channels sample only on those ticks, so this sets how much smoothing they
need. Run `thalamus demo` and read `gate opens` off the meter to check it
against your own traffic -- if it is far off, these time constants are wrong."""


def trace_specs(
    rate_hz: float, gate_open_rate: float = GATE_OPEN_ASSUMPTION
) -> dict[str, tuple[Channel, str | None, TriggerSpec]]:
    """Build a TriggerSpec per trace, derived from each channel's accuracy.

    Gate channels are sampled every tick; bank channels only when the gate
    opens, so they get a lower assumed rate and therefore more smoothing. The
    numbers are not tuned by hand anywhere -- they fall out of suggest_band.
    """
    out: dict[str, tuple[Channel, str | None, TriggerSpec]] = {}
    for ch in BANK:
        # A bank channel only sees the ticks on which the gate opened.
        eff_hz = rate_hz if ch.tier is Tier.GATE else max(rate_hz * gate_open_rate, 0.5)
        p_hit, p_miss = ch.mass()
        spec = suggest_band(
            ch.accuracy,
            rate_hz=eff_hz,
            p_hit=p_hit,
            p_miss=p_miss,
            polarity=ch.polarity,
            fire_at=ch.fire_at,
            dwell_s=ch.dwell_s,
            refractory_s=ch.refractory_s,
            min_slope=ch.min_slope,
        )
        options = ch.watch if ch.kind == "choice" else (None,)
        for opt in options:
            key = f"{ch.id}:{opt}" if opt is not None else ch.id
            out[key] = (ch, opt, spec)
    return out
