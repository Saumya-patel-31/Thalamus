"""Making the probabilities mean what they say.

Every threshold in this system is a number in probability space, so it only
works if the probabilities are honest. Independent audits of Jev put raw
expected calibration error somewhere around 0.06-0.16 and find it
*overconfident* -- one published bin states 0.97 and is right about 88.6% of
the time. Isotonic regression brings that to roughly 0.006-0.018.

So: measure once per channel, fit a monotone map, ship the map as JSON, and
threshold on calibrated values. Isotonic is implemented here with pool-adjacent-
violators in about thirty lines rather than pulling in scikit-learn, because the
whole project installs with no dependencies.

Uncalibrated is a supported mode -- ``Calibration.identity()`` -- and the HUD
says so, loudly, because an uncalibrated threshold is a guess wearing a
decimal point.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["isotonic_fit", "Calibration"]


def _pava(blocks: list[list[float]]) -> list[list[float]]:
    """Pool adjacent violators. Blocks are [weight, sum_y, right_edge_x]."""
    out: list[list[float]] = []
    for b in blocks:
        out.append(list(b))
        while len(out) >= 2 and (out[-2][1] / out[-2][0]) > (out[-1][1] / out[-1][0]):
            w2, s2, x2 = out.pop()
            w1, s1, x1 = out.pop()
            out.append([w1 + w2, s1 + s2, max(x1, x2)])
    return out


def isotonic_fit(
    pairs: list[tuple[float, bool]],
    *,
    max_points: int = 24,
    min_block: int | None = None,
) -> list[tuple[float, float]]:
    """Fit a monotone raw -> calibrated map by pool-adjacent-violators.

    ``pairs`` is ``(raw_probability, was_actually_true)``. Returns breakpoints
    for piecewise-linear interpolation.

    **Regularisation matters more than the fit.** Plain PAVA on a few thousand
    noisy labels overfits into a staircase that runs from 0 to 1 *inside* a
    region where the raw score carries no information at all -- which is exactly
    the situation for an overconfident model whose 0.83 and its 0.97 mean the
    same thing. Requiring a minimum number of observations per block collapses
    that staircase back onto the honest base rate. Without this the calibration
    map is worse than no map.
    """
    if not pairs:
        return [(0.0, 0.0), (1.0, 1.0)]
    pts = sorted(pairs, key=lambda p: p[0])
    n = len(pts)
    if min_block is None:
        # Enough samples per block that its mean is worth believing: the
        # standard error of a proportion at n=40 is about 0.08.
        min_block = max(40, n // 25)

    blocks = _pava([[1.0, 1.0 if y else 0.0, x] for x, y in pts])

    if min_block > 1 and len(blocks) > 1:
        merged: list[list[float]] = []
        for b in blocks:
            if merged and merged[-1][0] < min_block:
                merged[-1] = [merged[-1][0] + b[0], merged[-1][1] + b[1], b[2]]
            else:
                merged.append(list(b))
        while len(merged) >= 2 and merged[-1][0] < min_block:
            tail = merged.pop()
            merged[-1] = [merged[-1][0] + tail[0], merged[-1][1] + tail[1], tail[2]]
        blocks = _pava(merged)  # merging can break monotonicity; restore it

    curve = [(b[2], b[1] / b[0]) for b in blocks]

    if len(curve) > max_points:
        step = (len(curve) - 1) / (max_points - 1)
        idx = sorted({int(round(i * step)) for i in range(max_points)} | {0, len(curve) - 1})
        curve = [curve[i] for i in idx]

    if curve[0][0] > 0.0:
        curve.insert(0, (0.0, curve[0][1]))
    if curve[-1][0] < 1.0:
        curve.append((1.0, curve[-1][1]))
    return curve


@dataclass
class Calibration:
    """Per-trace monotone maps. Unknown traces pass through untouched."""

    maps: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    accuracy: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    @classmethod
    def identity(cls) -> "Calibration":
        return cls(maps={}, fitted=False)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(
            maps={k: [tuple(p) for p in v] for k, v in (data.get("maps") or {}).items()},
            accuracy=data.get("accuracy") or {},
            fitted=True,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"maps": {k: [list(p) for p in v] for k, v in self.maps.items()},
                 "accuracy": self.accuracy},
                indent=2,
            )
        )

    def apply(self, key: str, raw: float | None) -> float:
        if raw is None:
            return 0.0
        curve = self.maps.get(key)
        if not curve:
            return raw
        xs = [p[0] for p in curve]
        i = bisect_left(xs, raw)
        if i == 0:
            return curve[0][1]
        if i >= len(curve):
            return curve[-1][1]
        x0, y0 = curve[i - 1]
        x1, y1 = curve[i]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)


def expected_calibration_error(
    pairs: list[tuple[float, bool]], bins: int = 10
) -> float:
    """Standard ECE, for reporting before/after a fit."""
    if not pairs:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    total = len(pairs)
    err = 0.0
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1 for _, y in b if y) / len(b)
        err += len(b) / total * abs(conf - acc)
    return err
