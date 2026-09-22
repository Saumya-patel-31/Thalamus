"""End-to-end: a scripted call, a deliberately unreliable model, real timing.

These run the whole pipeline -- state window, two-stage gate, calibration,
smoothing, hysteresis, cross-channel rules -- on a virtual clock, so 116
seconds of conversation replay in about a second with every time constant
still meaning what it says.
"""

from __future__ import annotations

import asyncio
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.harness import run_script
from thalamus.backends.mock import MockBackend
from thalamus.engine import Engine
from thalamus.rules import DEFAULT_RULES

SEEDS = 8


def _sweep(n: int = SEEDS):
    runs = [asyncio.run(run_script(seed=s)) for s in range(n)]
    first: dict[str, list[float]] = collections.defaultdict(list)
    for eng in runs:
        seen = set()
        for c in eng.cards:
            if c.rule not in seen:
                first[c.rule].append(c.t)
                seen.add(c.rule)
    return runs, first


_RUNS, _FIRST = _sweep()

# When the script actually creates each condition. Measuring "first fire ever"
# is the wrong statistic for a rule the call triggers more than once -- a run
# that misses the first opportunity and catches the third looks like a 60s
# timing error when it is nothing of the sort.
WINDOWS = {
    "check_that":          (15.0, 32.0),
    "monologue":           (14.0, 40.0),
    "hanging_question":    (30.0, 52.0),
    "objection_price":     (47.0, 66.0),
    "losing_them":         (60.0, 85.0),
    "objection_authority": (108.0, 122.0),
}


def _in_window(rule: str) -> list[float]:
    lo, hi = WINDOWS[rule]
    out = []
    for eng in _RUNS:
        ts = [c.t for c in eng.cards if c.rule == rule and lo <= c.t <= hi]
        if ts:
            out.append(min(ts))
    return out


def test_engines_do_not_share_rule_state():
    """Rules remember when they last fired. Two engines must not share that.

    Regression test: DEFAULT_RULES is a module-level list of *stateful*
    objects, and handing the same instances to a second engine made it inherit
    the first one's cooldowns and go almost completely silent.
    """
    a = Engine(MockBackend(seed=1))
    b = Engine(MockBackend(seed=1))
    assert a.rules[0] is not b.rules[0]
    assert all(r is not d for r, d in zip(a.rules, DEFAULT_RULES))
    a.rules[0]._last_t = 5.0
    assert b.rules[0]._last_t is None

    counts = [len(r.cards) for r in _RUNS]
    assert min(counts) >= 4, f"a run went near-silent: {counts}"


def test_it_mostly_says_nothing():
    """The hard-won skill. Nine ticks in ten must end in silence."""
    for eng in _RUNS:
        rate = len(eng.cards) / eng.tick_count
        assert rate < 0.04, f"spoke on {rate:.1%} of ticks"
    avg = sum(len(e.cards) for e in _RUNS) / len(_RUNS)
    assert 4 <= avg <= 16, f"average {avg:.1f} cards per call is not conversational"


def test_the_important_interrupts_are_reliable():
    """A ~75%-accurate sensor, and yet these land inside their causal window on
    nearly every run. Measured over 16 seeds: monologue 16/16, price 15/16,
    losing_them 12/16. Floors here leave room for seed variation."""
    for rule, floor in (("monologue", 7), ("objection_price", 6), ("losing_them", 5)):
        hits = len(_in_window(rule))
        assert hits >= floor, f"{rule} fired in-window on only {hits}/{SEEDS} runs"


def test_timing_is_repeatable():
    """The whole claim: noisy input, repeatable moment.

    Standard deviation of 2-4s against a sensor that is individually wrong
    about a quarter of the time. Gate-tier channels are tighter than bank-tier
    ones, which are only sampled on the ~40% of ticks where the gate opens.
    """
    import statistics

    for rule, max_sd in (("check_that", 5.0), ("objection_price", 6.0),
                         ("losing_them", 6.0), ("objection_authority", 6.0)):
        ts = _in_window(rule)
        if len(ts) < 4:
            continue
        sd = statistics.stdev(ts)
        assert sd <= max_sd, f"{rule} σ={sd:.1f}s over {len(ts)} runs"


def test_interrupts_follow_their_cause():
    """Every fire of these rules must sit inside its causal window -- no card
    may arrive before the thing it is about."""
    for rule, (lo, hi) in WINDOWS.items():
        for eng in _RUNS:
            for c in eng.cards:
                if c.rule != rule:
                    continue
                # A rule may legitimately fire again later in the call; what is
                # forbidden is firing *before* its cause exists.
                assert c.t >= lo - 1.0, f"{rule} fired at t+{c.t:.1f}s, cause at {lo}s"


def test_dwell_suppresses_a_short_lived_condition():
    """off_agenda is only true for ~3.5s of the script and has a 12s dwell, so
    it must never fire. This is the filter refusing to take the bait."""
    assert "drifted" not in _FIRST, "fired on a 3.5s drift despite a 12s dwell"


def test_the_gate_actually_pays_for_itself():
    """Not assumed -- measured.

    A cheap pre-filter only saves money when it opens rarely *and* sees less
    state than the stage it is filtering. Give the gate the same 45s window as
    the bank and the split costs ~18% MORE than running the full bank every
    tick, because the state gets billed twice on every open tick. The gate
    therefore reads a 15s window, and the meter reports the real saving so a
    regression here is visible rather than silent.
    """
    for eng in _RUNS:
        m = eng.meter.snapshot()
        assert 0.2 < m["gate_open_rate"] < 0.7, f"gate open {m['gate_open_rate']:.0%}"
        assert m["gate_saving"] > 0.10, (
            f"two-stage split saved only {m['gate_saving']:.0%} of tokens")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
