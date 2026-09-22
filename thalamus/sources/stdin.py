"""Type the conversation yourself.

Lines are ``SPEAKER: text``; a bare line is attributed to THEM. Useful for
poking a live Jev key without a microphone:

    ME: so the pricing is usage based
    THEM: that's going to be a problem for our finance team
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import AsyncIterator

from ..state import Utterance


async def stdin_source() -> AsyncIterator[tuple[Utterance, None]]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    while True:
        raw = await reader.readline()
        if not raw:
            return
        line = raw.decode(errors="replace").strip()
        if not line:
            continue
        speaker, _, text = line.partition(":")
        if text and speaker.strip().upper() in {"ME", "THEM", "SILENCE"}:
            yield Utterance(time.monotonic(), speaker.strip().upper(), text.strip()), None
        else:
            yield Utterance(time.monotonic(), "THEM", line), None
