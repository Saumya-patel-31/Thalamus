"""The signal layer is pure arithmetic, so it gets held to exact standards."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalamus.dsp import (
    Polarity, Trace, TriggerSpec, attractors, disagreement_ceiling,
    disagreement_floor, disagreement_threshold, required_tau, suggest_band,
)


def drive(trace: Trace, values, *, hz: float = 5.0, t0: float = 0.0) -> list[float]:
    """Push a series at a fixed rate; return the timestamps that fired."""
    fires = []
    for i, v in enumerate(values):
        t = t0 + i / hz
        if trace.push(t, v):
            fires.append(t)
    return fires


def test_band_must_be_oriented():
    for kwargs in (
        dict(fire=0.5, release=0.7, polarity=Polarity.HIGH),
        dict(fire=0.5, release=0.3, polarity=Polarity.LOW),
        dict(fire=0.7, release=0.5, ema_tau_s=0),
    ):
        try:
            TriggerSpec(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_hysteresis_kills_chatter():
    """A signal dithering across a single threshold must not machine-gun."""
    spec = TriggerSpec(fire=0.75, release=0.45, ema_tau_s=0.05, refractory_s=0.0)
    t = Trace("dither", spec)
    # Oscillate hard across 0.75 but never down to the release line.
    fires = drive(t, [0.95, 0.60, 0.95, 0.60] * 12)
    assert len(fires) == 1, f"chattered {len(fires)} times"


def test_release_rearms():
    spec = TriggerSpec(fire=0.75, release=0.45, ema_tau_s=0.05, refractory_s=0.0)
    t = Trace("rearm", spec)
    fires = drive(t, [0.95] * 8 + [0.05] * 12 + [0.95] * 8)
    assert len(fires) == 2, f"expected 2 distinct episodes, got {len(fires)}"


def test_refractory_silences_a_standing_truth():
    """Something that stays true for a minute is worth saying once."""
    spec = TriggerSpec(fire=0.7, release=0.4, ema_tau_s=0.1, refractory_s=25.0)
    t = Trace("standing", spec)
    fires = drive(t, [0.98] * 300)  # 60s at 5Hz
    assert len(fires) == 1, f"nagged {len(fires)} times"


def test_min_slope_separates_rising_from_merely_high():
    """'They are confused' is often not actionable. 'I am losing them' is."""
    flat = Trace("flat", TriggerSpec(fire=0.7, release=0.4, ema_tau_s=0.4, min_slope=0.05))
    # Starts high and stays there: EMA is level, slope ~0.
    assert drive(flat, [0.9] * 60) == [], "fired on a flat high signal"

    climbing = Trace("climb", TriggerSpec(fire=0.7, release=0.4, ema_tau_s=0.4, min_slope=0.05))
    ramp = [min(0.98, i / 60) for i in range(60)]
    assert drive(climbing, ramp), "failed to fire on a rising signal"


def test_low_polarity_detects_a_collapse():
    """p(question was answered) falling through the floor = a hanging thread."""
    spec = TriggerSpec(
        fire=0.25, release=0.55, polarity=Polarity.LOW, ema_tau_s=0.3, dwell_s=2.0
    )
    t = Trace("hanging", spec)
    # Answered, then the question goes ignored.
    fires = drive(t, [0.9] * 10 + [0.05] * 40)
    assert len(fires) == 1
    # dwell_s=2.0 means it must hang for 2s before we butt in.
    assert fires[0] >= 10 / 5.0 + 2.0


def test_dwell_blocks_a_brief_spike():
    spec = TriggerSpec(fire=0.7, release=0.4, ema_tau_s=0.05, dwell_s=3.0)
    t = Trace("spike", spec)
    assert drive(t, [0.99] * 5 + [0.0] * 10) == [], "fired on a 1s spike"


def test_disagreement_suppresses_an_ill_posed_channel():
    """Two phrasings that disagree means the question is bad. Stay quiet."""
    spec = TriggerSpec(fire=0.7, release=0.4, ema_tau_s=0.1, unreliable_at=0.30)
    t = Trace("ambiguous", spec)
    for i in range(40):
        t.push(i / 5.0, 0.95, alt=0.10)  # phrasings wildly apart
    assert t.unreliable
    assert t.fire_count == 0, "fired despite its own phrasings disagreeing"


def test_agreement_permits_firing():
    spec = TriggerSpec(fire=0.7, release=0.4, ema_tau_s=0.1, unreliable_at=0.30)
    t = Trace("clear", spec)
    fired = any(t.push(i / 5.0, 0.95, alt=0.92) for i in range(40))
    assert fired and not t.unreliable


def test_ema_respects_wall_clock_not_sample_count():
    """Two ticks 4s apart must move the EMA far more than two ticks 40ms apart."""
    spec = TriggerSpec(fire=0.9, release=0.1, ema_tau_s=1.0)
    fast, slow = Trace("fast", spec), Trace("slow", spec)
    fast.push(0.0, 0.0); fast.push(0.04, 1.0)
    slow.push(0.0, 0.0); slow.push(4.00, 1.0)
    assert fast.ema < 0.1 < 0.9 < slow.ema, (fast.ema, slow.ema)


def test_slope_is_signed_toward_firing():
    spec = TriggerSpec(fire=0.25, release=0.55, polarity=Polarity.LOW, ema_tau_s=0.2)
    t = Trace("falling", spec)
    drive(t, [1.0 - i / 40 for i in range(40)])
    assert t.slope() < 0 and t.signed_slope > 0


# ---------------------------------------------------------------------------
# The headline claim: oversampling rescues a bad sensor.
# ---------------------------------------------------------------------------

def _noisy_sensor(truth: bool, n: int, accuracy: float, rng: random.Random):
    """A ~70%-accurate, overconfident sensor -- Jev's measured character."""
    out = []
    for _ in range(n):
        correct = rng.random() < accuracy
        hit = truth if correct else (not truth)
        out.append(rng.uniform(0.82, 0.99) if hit else rng.uniform(0.01, 0.18))
    return out


def test_oversampling_beats_a_70_percent_sensor():
    """One sample from a 70% sensor is a coin flip with swagger. Fifty samples,
    smoothed behind a hysteresis band, is a reliable edge. This is the entire
    architectural bet, so it is asserted rather than asserted-in-prose.
    """
    rng = random.Random(7)
    # Thresholds derived from the channel's accuracy, not eyeballed. A naive
    # 0.75 band never fires at all against a 70% sensor; see suggest_band.
    spec = suggest_band(0.70, rate_hz=5.0, dwell_s=1.0, refractory_s=999)

    true_positives = 0
    false_positives = 0
    for trial in range(200):
        hot = Trace("hot", spec)
        drive(hot, _noisy_sensor(True, 100, 0.70, rng))   # 20s at 5Hz
        true_positives += hot.fire_count > 0

        cold = Trace("cold", spec)
        drive(cold, _noisy_sensor(False, 100, 0.70, rng))
        false_positives += cold.fire_count > 0

    recall = true_positives / 200
    fpr = false_positives / 200
    # A single raw sample would give recall ~0.70 / FPR ~0.30.
    assert recall >= 0.97, f"recall regressed to {recall:.3f}"
    assert fpr <= 0.03, f"false positive rate regressed to {fpr:.3f}"
    print(f"    oversampled 70% sensor -> recall {recall:.3f}, FPR {fpr:.3f}")


def test_oversampling_cannot_fix_bias():
    """Honest counterweight: averaging kills variance, not bias. A channel Jev
    systematically misreads is wrong on every tick, and no amount of sampling
    saves it. That is what the disagreement check is for.
    """
    rng = random.Random(11)
    spec = suggest_band(0.70, ema_tau_s=1.5, dwell_s=1.0)
    t = Trace("misread", spec)
    drive(t, _noisy_sensor(False, 50, 0.05, rng))  # confidently, consistently wrong
    assert t.fire_count == 1, "bias should sail straight through the filter"



def test_attractor_math_matches_simulation():
    """The closed-form fixed point should predict where the EMA actually lands."""
    rng = random.Random(3)
    lo, hi = attractors(0.70, p_hit=0.905, p_miss=0.095)
    spec = TriggerSpec(fire=0.99, release=0.01, ema_tau_s=2.0)  # never fires; just observing
    hot = Trace("hot", spec)
    drive(hot, _noisy_sensor(True, 400, 0.70, rng))
    cold = Trace("cold", spec)
    drive(cold, _noisy_sensor(False, 400, 0.70, rng))
    # The EMA is itself a random variable; compare its time-average over the
    # tail rather than one endpoint sample.
    def tail_mean(tr):
        tail = list(tr.samples)[len(tr.samples) // 2:]
        return sum(s.ema for s in tail) / len(tail)

    obs_hi, obs_lo = tail_mean(hot), tail_mean(cold)
    assert abs(obs_hi - hi) < 0.04, f"predicted {hi:.3f}, observed {obs_hi:.3f}"
    assert abs(obs_lo - lo) < 0.04, f"predicted {lo:.3f}, observed {obs_lo:.3f}"
    print(f"    predicted ({lo:.3f}, {hi:.3f})  observed ({obs_lo:.3f}, {obs_hi:.3f})")


def test_naive_threshold_wrecks_recall():
    """Quantifies the trap. The intuitive 0.75 band is not merely late against a
    70% channel -- it misses most true episodes outright, because the smoothed
    trace settles at 0.66 and only noise excursions ever reach 0.75.
    """
    rng = random.Random(5)
    naive_spec = TriggerSpec(fire=0.75, release=0.45, ema_tau_s=2.795, dwell_s=1.0,
                             refractory_s=999)
    derived_spec = suggest_band(0.70, rate_hz=5.0, dwell_s=1.0, refractory_s=999)

    naive_hits = derived_hits = 0
    for _ in range(200):
        series = _noisy_sensor(True, 100, 0.70, rng)
        n, d = Trace("n", naive_spec), Trace("d", derived_spec)
        drive(n, series)
        drive(d, series)          # same data, different band
        naive_hits += n.fire_count > 0
        derived_hits += d.fire_count > 0

    naive_recall, derived_recall = naive_hits / 200, derived_hits / 200
    assert naive_recall < 0.60, f"naive band recall {naive_recall:.2f}; docs stale"
    assert derived_recall > 0.95, f"derived band recall only {derived_recall:.2f}"
    print(f"    same data -- naive 0.75 band recall {naive_recall:.2f} "
          f"vs derived band {derived_recall:.2f}")


def test_sample_rate_converts_into_response_speed():
    """The core economic claim, as arithmetic: reliability depends on tau*rate,
    so sampling faster buys a shorter time constant at identical noise margin.
    This is why a ~80ms model is a capability and not just a saving.
    """
    taus = {hz: required_tau(0.70, hz) for hz in (3.0, 5.0, 12.0, 20.0)}
    assert taus[3.0] > taus[5.0] > taus[12.0] > taus[20.0]
    # tau * rate should be near-invariant: the reliability budget is conserved.
    budgets = [tau * hz for hz, tau in taus.items()]
    assert max(budgets) / min(budgets) < 1.15, budgets
    # 4x the sample rate should buy roughly 4x the responsiveness.
    assert 3.2 < taus[5.0] / taus[20.0] < 4.8, taus[5.0] / taus[20.0]
    print("    tau needed @0.70 acc: " +
          ", ".join(f"{hz:.0f}Hz={t:.2f}s" for hz, t in taus.items()))


def test_accuracy_pays_back_faster_than_rate():
    """Better phrasing widens the gap *and* shrinks the noise, so it beats
    throwing sample rate at the problem. Directive: fix the question first.
    """
    slow_but_sharp = required_tau(0.85, 5.0)
    fast_but_blunt = required_tau(0.70, 12.0)
    assert slow_but_sharp < fast_but_blunt, (slow_but_sharp, fast_but_blunt)
    print(f"    0.85 acc @5Hz needs {slow_but_sharp:.2f}s; "
          f"0.70 acc @12Hz still needs {fast_but_blunt:.2f}s")


def test_two_phrasings_disagree_from_noise_alone():
    """The trap behind the ill-posedness check.

    The phrasings are independent draws, so they land a full band apart
    whenever exactly one is wrong -- 2a(1-a) of the time. At a=0.78 that is
    0.27 of expected disagreement with nothing at all wrong with the question,
    so the intuitive 0.30 flag condemns healthy channels on their own noise.
    At a=0.72 the floor (0.32) is *above* 0.30 outright: every such channel
    would be permanently marked ill-posed.
    """
    assert disagreement_floor(0.78) > 0.25
    assert disagreement_floor(0.72) > 0.30, "the old hand-picked 0.30 was below the floor"
    assert disagreement_threshold(0.86) > disagreement_floor(0.86) + 0.2


def test_the_disagreement_check_switches_itself_off_when_powerless():
    """Floor and ceiling converge as accuracy falls, so below ~0.78 the test
    cannot separate a bad question from ordinary noise and is disabled."""
    assert disagreement_threshold(0.72) >= 1.0, "ran a test with no discriminating power"
    assert disagreement_threshold(0.74) >= 1.0
    assert disagreement_threshold(0.86) < 1.0, "disabled a test that does work"
    powers = [disagreement_ceiling(a) - disagreement_floor(a)
              for a in (0.70, 0.78, 0.86)]
    assert powers[0] < powers[1] < powers[2], powers


def test_derived_specs_never_flag_a_healthy_channel():
    """End-to-end: a well-behaved channel sampled with two honest phrasings
    must not condemn itself. This is the bug the HUD surfaced."""
    import random
    rng = random.Random(4)
    spec = suggest_band(0.86, rate_hz=5.0)
    t = Trace("healthy", spec)
    for i in range(300):
        truth = True
        a = 0.9 if rng.random() < 0.86 else 0.1
        b = 0.9 if rng.random() < 0.86 else 0.1
        t.push(i / 5.0, a + rng.uniform(-.08, .08), alt=b + rng.uniform(-.08, .08))
    assert not t.unreliable, f"healthy channel flagged; disagreement={t.disagreement:.3f}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
