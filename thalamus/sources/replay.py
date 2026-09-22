"""Replay a scripted conversation from JSONL, in real time or faster.

Each line: ``{"t": 4.2, "speaker": "THEM", "text": "...", "truth": {...}}``

``truth`` is ground truth for the mock backend only -- which channels are
actually true at this moment in the script. It is how the demo can be honest
about a 75%-accurate sensor: the simulator knows the answer and then gets it
wrong a quarter of the time, exactly as the real thing would.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

from ..state import Utterance


async def replay_source(
    path: str | Path, *, speed: float = 1.0, loop: bool = False
) -> AsyncIterator[tuple[Utterance, dict | None]]:
    lines = [
        json.loads(ln)
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("//")
    ]
    if not lines:
        return
    while True:
        wall0 = time.monotonic()
        script0 = float(lines[0].get("t", 0.0))
        for row in lines:
            target = (float(row.get("t", 0.0)) - script0) / max(speed, 1e-6)
            delay = target - (time.monotonic() - wall0)
            if delay > 0:
                await asyncio.sleep(delay)
            yield (
                Utterance(
                    t=time.monotonic(),
                    speaker=row.get("speaker", "THEM"),
                    text=row.get("text", ""),
                ),
                row.get("truth"),
            )
        if not loop:
            return
