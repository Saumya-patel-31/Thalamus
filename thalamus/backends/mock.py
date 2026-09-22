"""A simulated decision model with Jev's measured character.

This exists so the engine, the signal layer and the HUD can be exercised
end-to-end with no API key -- but it is not a stub that returns tidy answers.
It deliberately reproduces the two properties that make the signal layer
necessary in the first place:

  * **accuracy around 0.7-0.8**, per channel, drawn independently each tick
  * **overconfidence** -- when it is wrong it is wrong at 0.9, not at 0.55

Watch the HUD with smoothing off and the raw traces flap all over the place.
Turn smoothing on and the same data reads cleanly. That contrast is the entire
argument of the project, and it is reproducible on a laptop with no key.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from . import Verdict
from ..channels import BANK, Tier

_BY_ID = {c.id: c for c in BANK}


class MockBackend:
    name = "mock"

    def __init__(
        self,
        seed: int | None = None,
        gate_latency_ms: float = 80.0,
        bank_latency_ms: float = 260.0,
        jitter: float = 0.35,
    ) -> None:
        self.rng = random.Random(seed)
        self.gate_latency_ms = gate_latency_ms
        self.bank_latency_ms = bank_latency_ms
        self.jitter = jitter

    # -- noise model ---------------------------------------------------------

    def _hit(self) -> float:
        return self.rng.uniform(0.82, 0.98)

    def _miss(self) -> float:
        return self.rng.uniform(0.02, 0.18)

    def _noul_clean(self, truth: bool, accuracy: float) -> float:
        """Emit p(yes) for a channel whose ground truth is `truth`."""
        believes = truth if self.rng.random() < accuracy else (not truth)
        return self._hit() if believes else self._miss()

    def _choice(self, truth: str, options: list[str], accuracy: float) -> dict[str, float]:
        believes = truth if self.rng.random() < accuracy else self.rng.choice(
            [o for o in options if o != truth] or [truth]
        )
        mass = self.rng.uniform(0.55, 0.92)
        rest = [o for o in options if o != believes]
        weights = [self.rng.random() + 0.05 for _ in rest]
        total = sum(weights) or 1.0
        probs = {believes: mass}
        for opt, w in zip(rest, weights):
            probs[opt] = (1.0 - mass) * w / total
        return probs

    def _score(self, truth: float, levels: int, accuracy: float) -> float:
        spread = 0.35 if self.rng.random() < accuracy else 1.4
        v = truth + self.rng.gauss(0, spread)
        return min(max(v, 0.0), float(levels - 1))

    # -- the call ------------------------------------------------------------

    async def evaluate(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        oracle: dict[str, Any] | None = None,
    ) -> Verdict:
        oracle = oracle or {}
        illposed = set(oracle.get("__illposed", ()))
        is_gate = all(
            _BY_ID.get(k.removesuffix("~alt"), None) is None
            or _BY_ID[k.removesuffix("~alt")].tier is Tier.GATE
            for k in questions
        )
        base = self.gate_latency_ms if is_gate else self.bank_latency_ms
        latency = base * (1 + self.rng.uniform(-self.jitter, self.jitter))
        await asyncio.sleep(latency / 1000.0)

        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            cid = qid.removesuffix("~alt")
            is_alt = qid.endswith("~alt")
            ch = _BY_ID.get(cid)
            if ch is None:
                continue
            truth = oracle.get(cid)
            acc = ch.accuracy
            # An ill-posed channel is one whose second phrasing reads the world
            # differently. We simulate that by flipping the truth for the alt.
            if is_alt and cid in illposed:
                truth = (not truth) if isinstance(truth, bool) else truth

            if ch.kind == "noul":
                answers[qid] = {"noul": round(self._noul_clean(bool(truth), acc), 4)}
            elif ch.kind == "choice":
                options = list((ch.criteria or {}).keys())
                target = truth if truth in options else (options[-1] if options else "none")
                probs = self._choice(str(target), options, acc)
                best = max(probs, key=probs.get)
                answers[qid] = {
                    "choice": best,
                    "probabilities": {k: round(v, 4) for k, v in probs.items()},
                    "confidence": round(probs[best], 4),
                }
            else:  # score
                levels = len(ch.criteria or [])
                target = float(truth) if isinstance(truth, (int, float)) else 0.0
                answers[qid] = {
                    "score": round(self._score(target, levels, acc), 4),
                    "confidence": round(self.rng.uniform(0.55, 0.95), 4),
                }

        # Token accounting mirrors the real thing closely enough for the cost
        # meter to be meaningful: state + every question's text is input.
        state_tokens = len(str(state)) // 4
        q_tokens = sum(len(str(q)) // 4 for q in questions.values())
        return Verdict(
            answers=answers,
            input_tokens=state_tokens + q_tokens,
            output_tokens=0,
            latency_ms=latency,
            model="mock-jev-1.13.0",
        )
