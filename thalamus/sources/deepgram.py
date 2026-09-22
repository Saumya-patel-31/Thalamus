"""Live capture via Deepgram Flux, one socket per speaker.

Flux carries end-of-turn detection inside the model -- median EOT around 260ms,
with native barge-in -- which is why it is the right front end. A conventional
STT plus an external VAD adds 200-600ms before Thalamus has seen a word, and
this whole design is an argument about latency.

    pip install "thalamus[mic]"
    export DEEPGRAM_API_KEY=...
    thalamus devices                 # find the loopback device
    thalamus mic --system-device BlackHole

**Two sockets, not one.** The microphone is tagged ``ME``; a loopback device
carrying system output is tagged ``THEM``. That second socket is not a nicety:
``floor``, ``confusion``, ``interest`` and ``objection`` are all questions about
the *other* party, so a mic-only setup measures the wrong half of the call. See
``thalamus/sources/audio.py`` for the per-platform setup.

**Untested against a live key in this repo.** Written against the documented
Flux v2 schema (`wss://api.deepgram.com/v2/listen`, `Authorization: Token ...`,
`TurnInfo` messages carrying `transcript`). The replay and stdin sources are the
exercised ones. Check the frames you receive before trusting it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import AsyncIterator, Callable

from ..state import Utterance
from .audio import detect_loopback, resolve_device

FLUX_URL = "wss://api.deepgram.com/v2/listen?model={model}&encoding=linear16&sample_rate={rate}"
SAMPLE_RATE = 16_000
CHUNK_MS = 80  # Deepgram's recommended chunk size for latency
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_MS // 1000  # 1280 frames = 2560 bytes

Log = Callable[[str], None]


def preflight(api_key: str | None = None) -> str:
    """Check the optional extra and the key *before* the loop starts.

    ``flux_stream`` is an async generator, so nothing in its body runs until the
    first iteration -- a missing key would otherwise surface as a traceback from
    deep inside the engine rather than a clear message at startup.
    """
    try:
        import sounddevice  # noqa: F401
        import websockets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "live capture needs the optional extra: pip install 'thalamus[mic]'"
        ) from exc
    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    return key


async def flux_stream(
    *,
    speaker: str,
    device: int | str | None = None,
    model: str = "flux-general-en",
    api_key: str | None = None,
    eager: bool = False,
    log: Log = lambda _m: None,
) -> AsyncIterator[Utterance]:
    """One capture device -> one socket -> a stream of turns for one speaker.

    Reconnects with exponential backoff. A dropped socket must not end the
    session: the other speaker's stream keeps running, and the channels that
    depend on the dead half decay rather than freeze on a stale value.

    ``eager=True`` also emits on ``EagerEndOfTurn`` -- a few hundred ms sooner,
    but retractable by ``TurnResumed``. Worth taking for a reflex: a card that
    appears early and is superseded costs less than one that arrives after the
    moment has passed.
    """
    import sounddevice as sd
    import websockets

    key = preflight(api_key)
    dev = resolve_device(device)
    url = FLUX_URL.format(model=model, rate=SAMPLE_RATE)
    backoff = 0.5

    while True:
        audio: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()
        dropped = 0

        def on_audio(indata, _frames, _t, status) -> None:
            # PortAudio's thread. Hop to the loop without ever blocking here.
            nonlocal dropped
            if status:
                return
            try:
                loop.call_soon_threadsafe(audio.put_nowait, bytes(indata))
            except (asyncio.QueueFull, RuntimeError):
                dropped += 1  # drop rather than let the microphone back up

        try:
            async with websockets.connect(
                url, additional_headers={"Authorization": f"Token {key}"}
            ) as ws:
                log(f"{speaker}: connected")
                backoff = 0.5

                async def pump() -> None:
                    while True:
                        await ws.send(await audio.get())

                with sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_FRAMES,
                    dtype="int16",
                    channels=1,
                    device=dev,
                    callback=on_audio,
                ):
                    sender = asyncio.create_task(pump())
                    try:
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                            except (TypeError, ValueError):
                                continue
                            if msg.get("type") != "TurnInfo":
                                continue
                            text = (msg.get("transcript") or "").strip()
                            if not text:
                                continue
                            event = msg.get("event")
                            # No `event` field: treat any transcript as final.
                            if event in (None, "EndOfTurn") or (
                                eager and event == "EagerEndOfTurn"
                            ):
                                yield Utterance(time.monotonic(), speaker, text)
                    finally:
                        sender.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await sender
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure is a reconnect
            log(f"{speaker}: {type(exc).__name__}: {exc} — retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)
            continue
        else:
            log(f"{speaker}: stream ended")
            return


async def merge(*streams: AsyncIterator) -> AsyncIterator:
    """Interleave several async iterators, yielding items as they arrive.

    One stream ending or failing must not end the others -- in a call, losing
    the far-side socket is a degraded session, not the end of one.
    """
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    async def drain(src) -> None:
        try:
            async for item in src:
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass
        finally:
            await queue.put(done)

    tasks = [asyncio.create_task(drain(s)) for s in streams]
    finished = 0
    try:
        while finished < len(tasks):
            item = await queue.get()
            if item is done:
                finished += 1
                continue
            yield item
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def dual_source(
    *,
    mic_device: int | str | None = None,
    system_device: int | str | None = None,
    far_side: bool = True,
    model: str = "flux-general-en",
    api_key: str | None = None,
    eager: bool = False,
    log: Log = lambda _m: None,
) -> AsyncIterator[tuple[Utterance, None]]:
    """Both halves of the conversation, merged in arrival order.

    With ``far_side`` left on and ``system_device`` unset, a loopback device is
    guessed by name. ``far_side=False`` is the explicit opt-out for mic-only
    capture -- separate from "I did not name a device", because those mean very
    different things and only one of them deserves a warning.
    """
    preflight(api_key)
    if far_side and system_device is None:
        guess = detect_loopback()
        if guess is not None:
            system_device = guess.index
            log(f"far side: auto-detected [{guess.index}] {guess.name}")
    if not far_side:
        system_device = None

    streams = [
        flux_stream(speaker="ME", device=mic_device, model=model,
                    api_key=api_key, eager=eager, log=log)
    ]
    if system_device is not None:
        streams.append(
            flux_stream(speaker="THEM", device=system_device, model=model,
                        api_key=api_key, eager=eager, log=log)
        )
    elif far_side:
        log("far side: NOT FOUND — floor, confusion, interest and objection are "
            "all questions about the other party and will read as silence. Run "
            "`thalamus devices` and pass --system-device.")
    else:
        log("far side: disabled (--no-far-side); running microphone only")

    async for u in merge(*streams):
        yield u, None
