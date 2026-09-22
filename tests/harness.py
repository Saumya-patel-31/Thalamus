"""Drive the full engine over a scripted call on a virtual clock.

The engine takes its clock as a parameter precisely so this is possible: a
116-second conversation replays deterministically in about a second, with every
time constant, dwell and refractory period still measured in the same units.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalamus.backends.mock import MockBackend
from thalamus.engine import Engine
from thalamus.state import RollingState, Utterance

SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "sales_call.jsonl"


class VClock:
    def __init__(self, t0: float = 10_000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t


async def run_script(
    *,
    seed: int = 0,
    rate_hz: float = 5.0,
    script: Path = SCRIPT,
    labels: list[dict] | None = None,
) -> Engine:
    """Replay the script. If ``labels`` is given, append one row per trace per
    tick as ``{key, raw, truth}``, pairing each sample with the ground truth
    *at that instant* -- which is the only way the pairs mean anything."""
    rows = [json.loads(l) for l in script.read_text().splitlines() if l.strip()]
    vc = VClock()
    engine = Engine(
        MockBackend(seed=seed, gate_latency_ms=0.0, bank_latency_ms=0.0),
        rate_hz=rate_hz,
        state=RollingState(agenda="Pricing walkthrough and next steps for the Q4 rollout."),
        clock=vc,
    )
    t0 = vc.t
    duration = rows[-1]["t"] + 6.0
    i = 0
    for step in range(int(duration * rate_hz)):
        elapsed = step / rate_hz
        vc.t = t0 + elapsed
        while i < len(rows) and rows[i]["t"] <= elapsed:
            engine.ingest(
                Utterance(vc.t, rows[i].get("speaker", "THEM"), rows[i].get("text", "")),
                rows[i].get("truth"),
            )
            i += 1
        await engine.tick()
        if labels is not None:
            _collect(engine, labels)
    return engine


def _collect(engine: Engine, sink: list[dict]) -> None:
    from thalamus.channels import BANK

    by_id = {c.id: c for c in BANK}
    for key, tr in engine.traces.items():
        if not tr.samples:
            continue
        cid, _, opt = key.partition(":")
        ch = by_id.get(cid)
        if ch is None or ch.kind == "score":
            continue  # a score needs a level, not a boolean
        truth = engine.oracle.get(cid)
        if ch.kind == "choice":
            truth = (truth == opt)
        if not isinstance(truth, bool):
            continue
        sink.append({"key": key, "raw": tr.samples[-1].raw, "truth": truth})


def summarise(engine: Engine) -> None:
    m = engine.meter.snapshot()
    print(f"\n  {engine.tick_count} ticks · {m['calls']} calls · "
          f"gate opened {m['gate_open_rate']*100:.0f}% of ticks")
    print(f"  {len(engine.cards)} interrupts over {engine.tick_count} ticks "
          f"({len(engine.cards)/engine.tick_count*100:.1f}% of ticks spoke)\n")
    for c in engine.cards:
        print(f"    t+{c.t:6.1f}s  [{c.severity:<5}] {c.rule:<18} {c.title}")


if __name__ == "__main__":
    eng = asyncio.run(run_script())
    summarise(eng)
