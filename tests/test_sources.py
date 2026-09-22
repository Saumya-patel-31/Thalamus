"""Source plumbing: merging two speakers, and device resolution.

No audio hardware and no network involved -- these exercise the parts that
decide whether a dropped far-side socket degrades the session or ends it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalamus.sources.audio import Device, _score
from thalamus.sources.deepgram import merge


async def _ticker(label: str, n: int, gap: float, fail_at: int | None = None):
    for i in range(n):
        await asyncio.sleep(gap)
        if fail_at is not None and i == fail_at:
            raise RuntimeError(f"{label} socket died")
        yield f"{label}{i}"


def _collect(*streams) -> list[str]:
    async def run():
        return [x async for x in merge(*streams)]

    return asyncio.run(run())


def test_merge_interleaves_in_arrival_order():
    got = _collect(_ticker("a", 3, 0.010), _ticker("b", 3, 0.015))
    assert sorted(got) == ["a0", "a1", "a2", "b0", "b1", "b2"]
    # The faster stream's first item must arrive before the slower one's.
    assert got.index("a0") < got.index("b0")


def test_one_speaker_dropping_does_not_end_the_session():
    """Losing the far-side socket is a degraded call, not the end of one."""
    got = _collect(_ticker("me", 5, 0.005), _ticker("them", 5, 0.005, fail_at=1))
    assert [g for g in got if g.startswith("me")] == ["me0", "me1", "me2", "me3", "me4"]
    assert "them0" in got and "them2" not in got


def test_merge_survives_a_stream_that_fails_immediately():
    got = _collect(_ticker("me", 3, 0.005), _ticker("them", 3, 0.005, fail_at=0))
    assert len([g for g in got if g.startswith("me")]) == 3


def test_merge_of_one_stream_is_a_passthrough():
    assert _collect(_ticker("solo", 4, 0.001)) == ["solo0", "solo1", "solo2", "solo3"]


def test_merge_cleans_up_its_tasks():
    async def run():
        before = len(asyncio.all_tasks())
        async for _ in merge(_ticker("a", 2, 0.001), _ticker("b", 2, 0.001)):
            pass
        await asyncio.sleep(0)
        return before, len(asyncio.all_tasks())

    before, after = asyncio.run(run())
    assert after <= before, f"leaked tasks: {before} -> {after}"


def test_loopback_names_are_recognised():
    """Name matching is a guess, so at least make it a well-ordered one."""
    for name in ("BlackHole 2ch", "Loopback Audio", "Stereo Mix (Realtek)",
                 "Monitor of Built-in Audio", "VB-Audio Virtual Cable"):
        assert _score(name) > 0, name
    for name in ("MacBook Pro Microphone", "Scarlett 2i2 USB", "AirPods Pro"):
        assert _score(name) == 0, name
    # BlackHole should outrank a generic ".monitor" match.
    assert _score("BlackHole 2ch") > _score("alsa_output.pci.monitor")


def test_device_str_flags_probable_loopback():
    loop = Device(3, "BlackHole 2ch", 2, 48000.0, loopback_score=_score("BlackHole 2ch"))
    mic = Device(1, "MacBook Pro Microphone", 1, 48000.0)
    assert loop.probable_loopback and "system audio?" in str(loop)
    assert not mic.probable_loopback and "system audio?" not in str(mic)


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
