"""The signal layer.

This is where Thalamus earns its keep. Jev gives us a noisy, overconfident,
irregularly-sampled estimate of a semantic condition. Nothing here talks to a
model; it is pure arithmetic over a time series, which means it is fast, exactly
testable, and owns every notion of *time* in the system.

That division of labour is deliberate. Jev is documented to be unreliable at
counting, arithmetic and date ordering, so it is never asked "how long has this
been true" or "how many times". It is asked only "is this true *right now*",
five times a second, and the integral happens here.

Three ideas do the real work:

  smoothing    An EMA with a time constant (not a sample count), because ticks
               are event-driven and therefore irregularly spaced.
  hysteresis   A Schmitt trigger: fire high, release low. A single threshold on
               a noisy signal chatters; a band cannot.
  refractory   Once fired, a channel goes quiet for a while. This is the
               difference between an assistant and a nag.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Polarity", "TriggerSpec", "Sample", "Trace"]


class Polarity(str, Enum):
    """Which direction of travel is interesting."""

    HIGH = "high"  # fire when the smoothed value climbs (confusion rising)
    LOW = "low"    # fire when it falls (p(answered) collapsing => question hanging)


@dataclass(frozen=True)
class TriggerSpec:
    """How a channel decides it has something to say.

    fire/release form the hysteresis band. For ``HIGH`` polarity ``release``
    sits below ``fire``; for ``LOW`` it sits above. Both are expressed in
    calibrated probability, so they mean what they say -- see calibration.py.
    """

    fire: float
    release: float
    polarity: Polarity = Polarity.HIGH
    ema_tau_s: float = 1.6
    """Smoothing time constant in seconds. Roughly: how long a blip must
    persist before it moves the needle. Larger = calmer but later."""
    dwell_s: float = 0.0
    """The band must be held continuously for this long before firing."""
    refractory_s: float = 25.0
    """Silence after a fire, even if the condition stays true."""
    min_slope: float | None = None
    """If set, require the smoothed value to be *moving* at least this fast
    (probability per second, signed by polarity) at the moment of firing.
    This is the difference between "they are confused" -- which may be
    permanent and unhelpful to mention -- and "I am losing them right now"."""
    slope_window_s: float = 2.0
    warmup_s: float | None = None
    """Triggers stay disarmed for this long after the first sample. An EWMA is
    seeded from its first observation, so a freshly-created channel whose first
    sample happens to be a confident error sits inside the band until it decays
    -- a cold-start false positive that has nothing to do with the world.
    Defaults to 2 * ema_tau_s, which is enough for the seed to wash out."""
    unreliable_at: float = 0.30
    """Mean disagreement between a channel's two phrasings, above which the
    channel is treated as ill-posed and its triggers are suppressed."""

    def __post_init__(self) -> None:
        for name in ("fire", "release"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a probability, got {v}")
        if self.polarity is Polarity.HIGH and self.release >= self.fire:
            raise ValueError(
                "HIGH polarity needs release < fire to form a hysteresis band "
                f"(got fire={self.fire}, release={self.release})"
            )
        if self.polarity is Polarity.LOW and self.release <= self.fire:
            raise ValueError(
                "LOW polarity needs release > fire to form a hysteresis band "
                f"(got fire={self.fire}, release={self.release})"
            )
        if self.ema_tau_s <= 0:
            raise ValueError("ema_tau_s must be positive")
        if self.warmup_s is not None and self.warmup_s < 0:
            raise ValueError("warmup_s must be non-negative")

    @property
    def effective_warmup_s(self) -> float:
        return 2.0 * self.ema_tau_s if self.warmup_s is None else self.warmup_s


@dataclass
class Sample:
    t: float
    raw: float
    """What Jev returned, before calibration."""
    value: float
    """Calibrated probability."""
    ema: float
    """Smoothed calibrated probability -- the trace you should reason about."""
    alt: float | None = None
    """The same question asked a second way, calibrated. Output tokens are free,
    so we always ask twice; see channels.py."""


@dataclass
class Trace:
    """One semantic channel over time, plus its trigger state machine."""

    name: str
    spec: TriggerSpec
    history: int = 900

    samples: deque[Sample] = field(default_factory=lambda: deque(maxlen=900))
    ema: float | None = None
    disagreement: float | None = None
    """Smoothed |primary - alternate phrasing|. High means the *question* is
    bad, not that the world is ambiguous -- a signal you cannot get from a
    model that only answers once."""

    latched: bool = False
    """True while inside the hysteresis band (post-fire, pre-release)."""
    fire_count: int = 0
    last_fire_t: float | None = None
    _in_band_since: float | None = None
    _last_t: float | None = None
    _t0: float | None = None

    def __post_init__(self) -> None:
        if self.samples.maxlen != self.history:
            self.samples = deque(self.samples, maxlen=self.history)

    # ---- geometry of the band -------------------------------------------------

    def _in_band(self, v: float) -> bool:
        if self.spec.polarity is Polarity.HIGH:
            return v >= self.spec.fire
        return v <= self.spec.fire

    def _released(self, v: float) -> bool:
        if self.spec.polarity is Polarity.HIGH:
            return v <= self.spec.release
        return v >= self.spec.release

    # ---- derived quantities --------------------------------------------------

    @property
    def warm(self) -> bool:
        """Has the EWMA seed washed out enough to trust the trace?"""
        if self._t0 is None or self._last_t is None:
            return False
        return (self._last_t - self._t0) >= self.spec.effective_warmup_s

    @property
    def unreliable(self) -> bool:
        """Has this channel's two phrasings stopped agreeing?"""
        return (
            self.disagreement is not None
            and self.disagreement > self.spec.unreliable_at
        )

    def slope(self, window_s: float | None = None) -> float:
        """Least-squares slope of the smoothed trace, in probability/second.

        Least squares rather than a two-point difference because ticks are
        irregular and the last gap is not representative of anything.
        """
        window_s = window_s if window_s is not None else self.spec.slope_window_s
        if not self.samples:
            return 0.0
        t_end = self.samples[-1].t
        pts = [(s.t, s.ema) for s in self.samples if t_end - s.t <= window_s]
        n = len(pts)
        if n < 3:
            return 0.0
        mean_t = sum(t for t, _ in pts) / n
        mean_v = sum(v for _, v in pts) / n
        num = sum((t - mean_t) * (v - mean_v) for t, v in pts)
        den = sum((t - mean_t) ** 2 for t, _ in pts)
        if den <= 1e-12:
            return 0.0
        return num / den

    @property
    def signed_slope(self) -> float:
        """Slope oriented so that positive always means "moving toward firing"."""
        s = self.slope()
        return s if self.spec.polarity is Polarity.HIGH else -s

    def in_band_for(self) -> float:
        """Seconds the signal has held inside the band. 0.0 if outside."""
        if self._in_band_since is None or self._last_t is None:
            return 0.0
        return self._last_t - self._in_band_since

    # ---- the tick -------------------------------------------------------------

    def push(
        self,
        t: float,
        value: float,
        *,
        raw: float | None = None,
        alt: float | None = None,
    ) -> bool:
        """Absorb one sample. Returns True only on a *rising edge* -- the tick
        the channel decided to speak. Steady-state truth returns False, which
        is the whole point.
        """
        raw = value if raw is None else raw
        if self._t0 is None:
            self._t0 = t

        # Time-constant EMA. With irregular ticks a fixed alpha would weight a
        # 40ms gap the same as a 2s gap, which is wrong.
        if self.ema is None or self._last_t is None:
            self.ema = value
        else:
            dt = max(t - self._last_t, 0.0)
            alpha = 1.0 - math.exp(-dt / self.spec.ema_tau_s) if dt > 0 else 0.0
            self.ema += alpha * (value - self.ema)

        if alt is not None:
            d = abs(value - alt)
            if self.disagreement is None:
                self.disagreement = d
            else:
                # Disagreement is smoothed harder than the signal: one odd tick
                # should not condemn a channel.
                dt = max(t - (self._last_t or t), 0.0)
                alpha = 1.0 - math.exp(-dt / (self.spec.ema_tau_s * 3.0))
                self.disagreement += alpha * (d - self.disagreement)

        self.samples.append(
            Sample(t=t, raw=raw, value=value, ema=self.ema, alt=alt)
        )
        self._last_t = t

        v = self.ema
        fired = False

        if self._in_band(v):
            if self._in_band_since is None:
                self._in_band_since = t
            ready = (t - self._in_band_since) >= self.spec.dwell_s
            cool = (
                self.last_fire_t is None
                or (t - self.last_fire_t) >= self.spec.refractory_s
            )
            moving = (
                self.spec.min_slope is None
                or self.signed_slope >= self.spec.min_slope
            )
            armed = ready and cool and moving and self.warm
            if not self.latched and armed and not self.unreliable:
                self.latched = True
                self.fire_count += 1
                self.last_fire_t = t
                fired = True
            elif not self.latched and armed and self.unreliable:
                # Suppressed on purpose. Latch anyway so we do not re-evaluate
                # the same edge forever once the phrasings reconcile.
                self.latched = True
        else:
            self._in_band_since = None

        if self.latched and self._released(v):
            self.latched = False

        return fired

    # ---- serialisation for the HUD -------------------------------------------

    def window(self, seconds: float = 30.0) -> list[Sample]:
        if not self.samples:
            return []
        t_end = self.samples[-1].t
        return [s for s in self.samples if t_end - s.t <= seconds]


# ---------------------------------------------------------------------------
# Where to actually put your thresholds.
# ---------------------------------------------------------------------------
#
# This is the least obvious thing in the whole system and it cost a failing
# test to notice, so it lives in code rather than in a README.
#
# Intuition says a confident channel should be thresholded near 0.75. That is
# wrong for a smoothed imperfect sensor. Averaging a sensor of accuracy `a`
# that emits ~p_hit when it agrees with reality and ~p_miss when it does not
# drives the EMA toward a *fixed point*, not toward 1.0:
#
#     true  condition -> a * p_hit  + (1 - a) * p_miss
#     false condition -> a * p_miss + (1 - a) * p_hit
#
# At a = 0.70, p_hit = 0.90, p_miss = 0.10 those are 0.66 and 0.34. A threshold
# at 0.75 never fires at all; a threshold at 0.66 sits exactly on the attractor
# and coin-flips. The only sane place for the band is *between the two
# attractors*, which is what these helpers compute.


def attractors(
    accuracy: float, p_hit: float = 0.90, p_miss: float = 0.10
) -> tuple[float, float]:
    """The (false, true) fixed points a smoothed channel settles toward.

    ``accuracy`` is the channel's measured hit rate -- run scripts/fit_calibration.py
    against a few hundred labelled ticks to get a real number instead of a guess.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")
    true_fp = accuracy * p_hit + (1 - accuracy) * p_miss
    false_fp = accuracy * p_miss + (1 - accuracy) * p_hit
    return false_fp, true_fp


def sample_sigma(
    accuracy: float, p_hit: float = 0.90, p_miss: float = 0.10
) -> float:
    """Standard deviation of a single raw sample from a channel of this accuracy."""
    spread = p_hit - p_miss
    return math.sqrt(accuracy * (1 - accuracy)) * spread


def ema_sigma(alpha: float, accuracy: float, **kw) -> float:
    """Noise left in the smoothed trace after EWMA smoothing.

    An EWMA with smoothing factor alpha reduces variance by alpha / (2 - alpha).
    This is the quantity that decides your false-positive rate.
    """
    return sample_sigma(accuracy, **kw) * math.sqrt(alpha / (2 - alpha))


def required_tau(
    accuracy: float,
    rate_hz: float,
    *,
    sigmas: float = 3.0,
    fire_at: float = 0.65,
    p_hit: float = 0.90,
    p_miss: float = 0.10,
) -> float:
    """Smallest smoothing time constant that keeps the band `sigmas` away from noise.

    This is the design equation of the whole system:

        sigma_ema = sigma_sample * sqrt(alpha / (2 - alpha)),  alpha = 1 - exp(-1/(tau*f))

    and we need `gap >= sigmas * sigma_ema`, where gap is the distance from the
    false attractor up to the fire line.

    The consequence is the entire argument for a fast model. Reliability is
    bought with `tau * rate`, so **sample rate converts directly into response
    speed at fixed reliability**. A 70%-accurate channel needs ~2.8s of
    smoothing at 5Hz -- but only ~1.2s at 12Hz, which is what Jev's ~80ms
    round trip actually buys you. Accuracy is even stronger: it widens the gap
    *and* shrinks the noise, so better phrasing pays back quadratically.
    """
    lo, hi = attractors(accuracy, p_hit, p_miss)
    gap = fire_at * (hi - lo)
    sigma_s = sample_sigma(accuracy, p_hit, p_miss)
    if sigma_s <= 1e-9:
        return 1e-3
    k = (gap / (sigmas * sigma_s)) ** 2
    alpha = 2 * k / (1 + k)
    if alpha >= 1.0:
        return 1e-3
    dt = 1.0 / rate_hz
    return -dt / math.log(1.0 - alpha)


def disagreement_floor(
    accuracy: float, p_hit: float = 0.90, p_miss: float = 0.10
) -> float:
    """How much two phrasings disagree *from noise alone*.

    The two phrasings are independent draws, so they land on opposite sides
    whenever exactly one of them is wrong -- probability 2a(1-a) -- and when
    that happens they are a full band apart. At a = 0.78 that is already 0.27
    of expected disagreement with nothing whatsoever wrong with the question.
    Flagging a channel "ill-posed" at 0.30, as intuition suggests, therefore
    condemns perfectly good channels on their own noise.
    """
    delta = p_hit - p_miss
    return 2 * accuracy * (1 - accuracy) * delta


def disagreement_ceiling(
    accuracy: float, p_hit: float = 0.90, p_miss: float = 0.10
) -> float:
    """Expected disagreement when the two phrasings genuinely read different
    things -- they agree only when exactly one of them errs."""
    delta = p_hit - p_miss
    return (accuracy**2 + (1 - accuracy) ** 2) * delta


MIN_DISAGREEMENT_POWER = 0.25
"""Separation between the noise floor and the ill-posed ceiling below which the
two-phrasing test cannot tell the two apart, and is switched off."""


def disagreement_threshold(
    accuracy: float,
    p_hit: float = 0.90,
    p_miss: float = 0.10,
    *,
    at: float = 0.75,
) -> float:
    """Where to call a channel ill-posed -- or 1.0 to switch the test off.

    The test only has power when the noise floor and the contradictory ceiling
    are far apart, and that gap closes fast as accuracy falls: an 0.86 channel
    separates 0.19 from 0.61, but an 0.72 channel separates 0.32 from 0.48 and
    the estimator's own fluctuation covers most of the difference. Rather than
    run a test that cannot discriminate -- and permanently condemn every weak
    channel -- it is disabled below `MIN_DISAGREEMENT_POWER`.

    That is a real limitation, not a workaround: **the cheapest check on a
    channel's phrasing is least available exactly where phrasing is most likely
    to be the problem.** Weak channels have to be fixed by measurement instead.
    """
    lo = disagreement_floor(accuracy, p_hit, p_miss)
    hi = disagreement_ceiling(accuracy, p_hit, p_miss)
    if (hi - lo) < MIN_DISAGREEMENT_POWER:
        return 1.0  # never flags; |a-b| is bounded by 1
    return lo + at * max(hi - lo, 0.0)


def suggest_band(
    accuracy: float,
    *,
    rate_hz: float | None = None,
    sigmas: float = 3.0,
    p_hit: float = 0.90,
    p_miss: float = 0.10,
    fire_at: float = 0.65,
    release_at: float = 0.35,
    polarity: Polarity = Polarity.HIGH,
    **spec_kwargs,
) -> TriggerSpec:
    """Derive a complete TriggerSpec from a channel's measured accuracy.

    Place the band between the attractors, then -- if you say how fast you are
    sampling -- pick the smallest smoothing constant that keeps noise `sigmas`
    away from the fire line. Pass ``ema_tau_s`` explicitly to override.

    A channel near chance has no gap to straddle and this raises, which is the
    correct outcome: an uninformative channel should not ship with a
    plausible-looking threshold bolted onto it.
    """
    lo, hi = attractors(accuracy, p_hit, p_miss)
    if hi - lo < 0.08:
        raise ValueError(
            f"accuracy {accuracy:.2f} leaves only a {hi - lo:.3f} gap between "
            "attractors; this channel cannot be thresholded reliably. Rephrase "
            "it or drop it."
        )
    fire = lo + fire_at * (hi - lo)
    release = lo + release_at * (hi - lo)
    if polarity is Polarity.LOW:
        fire, release = 1.0 - fire, 1.0 - release

    if "unreliable_at" not in spec_kwargs:
        spec_kwargs["unreliable_at"] = round(
            disagreement_threshold(accuracy, p_hit, p_miss), 4
        )
    if rate_hz is not None and "ema_tau_s" not in spec_kwargs:
        spec_kwargs["ema_tau_s"] = round(
            required_tau(
                accuracy, rate_hz, sigmas=sigmas, fire_at=fire_at,
                p_hit=p_hit, p_miss=p_miss,
            ),
            3,
        )
    return TriggerSpec(
        fire=round(fire, 4),
        release=round(release, 4),
        polarity=polarity,
        **spec_kwargs,
    )
