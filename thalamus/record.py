"""Capturing a real session so it can be calibrated against.

The naive version of this feature is useless. Seventeen traces at 5Hz is about
eighty-five samples a second, so half an hour of conversation is a hundred and
fifty thousand rows, and nobody is going to label that. Worse, the overwhelming
majority of them are near 0.0 or 1.0 where the channel is already unambiguous,
so labelling them teaches the calibration map nothing.

What isotonic regression actually needs is **coverage across the probability
range**, especially through the middle where the channel is uncertain and the
map has real work to do. So the recorder bins each channel's output and keeps a
quota per bin, which turns a hundred and fifty thousand rows into a few hundred
that are worth a human's attention -- and which happen to be exactly the
ambiguous ones.

    thalamus mic --record session.jsonl      # capture
    thalamus label session.jsonl             # answer y/n on the samples
    python3 scripts/fit_calibration.py session.jsonl -o calibration.json

Rows carry transcript text. Treat the file as a recording of the conversation,
because that is what it is.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = ["Recorder"]


class Recorder:
    """Stratified sampler. Writes incrementally so a crash keeps what it had."""

    def __init__(
        self,
        path: str | Path,
        *,
        per_bin: int = 25,
        bins: int = 10,
        context_turns: int = 2,
    ) -> None:
        self.path = Path(path)
        self.per_bin = per_bin
        self.bins = bins
        self.context_turns = context_turns
        self.counts: Counter[tuple[str, int]] = Counter()
        self.kept = 0
        self.seen = 0
        self._fh = self.path.open("w", encoding="utf-8")

    def _bin(self, raw: float) -> int:
        return min(int(raw * self.bins), self.bins - 1)

    def observe(
        self,
        *,
        key: str,
        raw: float,
        alt: float | None,
        ema: float | None,
        t: float,
        tick: int,
        context: list[str],
        instructions: str = "",
    ) -> bool:
        """Offer one sample. Returns True if it was kept."""
        self.seen += 1
        slot = (key, self._bin(raw))
        if self.counts[slot] >= self.per_bin:
            return False
        self.counts[slot] += 1
        self.kept += 1
        row: dict[str, Any] = {
            "key": key,
            "raw": round(raw, 4),
            "truth": None,  # filled in by `thalamus label`
            "t": round(t, 2),
            "tick": tick,
            "ema": None if ema is None else round(ema, 4),
            "alt": None if alt is None else round(alt, 4),
            "instructions": instructions,
            "context": context[-self.context_turns:],
        }
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        return True

    def coverage(self) -> dict[str, tuple[int, int]]:
        """Per channel: (rows kept, distinct probability bins populated).

        The bin count matters more than the row count. A confident channel only
        ever emits near 0 and near 1, so it populates two or three bins and
        needs far fewer labels than a channel that spreads across the range --
        there are simply fewer distinct regions for the map to learn.
        """
        rows: Counter[str] = Counter()
        bins: Counter[str] = Counter()
        for (key, _), n in self.counts.items():
            rows[key] += n
            bins[key] += 1
        return {k: (rows[k], bins[k]) for k in rows}

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def summary(self) -> str:
        cov = self.coverage()
        if not cov:
            return "  recorded nothing"
        # Rows per populated bin is what decides whether a fit is worth doing.
        thin = sorted((rows / b, k, rows, b) for k, (rows, b) in cov.items())[:3]
        lines = [
            f"  recorded {self.kept} samples from {self.seen} observations "
            f"({self.kept / max(self.seen, 1) * 100:.1f}% kept) → {self.path}",
            f"  {len(cov)} channels; thinnest per populated bin: "
            + ", ".join(f"{k} ({rows} rows / {b} bins)" for _, k, rows, b in thin),
            f"  next:  thalamus label {self.path}",
        ]
        return "\n".join(lines)
