"""Audio device discovery, including the loopback device that hears the far side.

A microphone hears one person. Every channel that matters -- ``floor``,
``confusion``, ``interest``, ``objection`` -- is about the *other* party, so a
single-mic setup is measuring the wrong half of the conversation.

The fix is a second capture device fed from the system's own output. Every
platform spells that differently, and macOS needs a third-party virtual device
because CoreAudio will not hand you system output directly:

    macOS    BlackHole (brew install --cask blackhole-2ch), Loopback, or
             Soundflower. Then make a Multi-Output Device in Audio MIDI Setup
             containing both your speakers and BlackHole, and select it as
             system output, so you still hear the call.
    Windows  WASAPI loopback, or "Stereo Mix" / "What U Hear" if the driver
             exposes one.
    Linux    PulseAudio/PipeWire expose a ".monitor" source per sink.

Wear headphones. With speakers the microphone also picks up the far side, both
sockets transcribe it, and every channel sees the same words twice.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Device", "list_devices", "detect_loopback", "resolve_device"]

# Ordered by confidence. Matched case-insensitively against the device name.
LOOPBACK_HINTS = (
    "blackhole",
    "loopback",
    "soundflower",
    "stereo mix",
    "what u hear",
    "wave out mix",
    ".monitor",
    "monitor of",
    "vb-audio",
    "voicemeeter",
)


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    channels: int
    default_samplerate: float
    hostapi: str = ""
    loopback_score: int = 0
    """0 = ordinary input. Higher = more likely to carry system output."""

    @property
    def probable_loopback(self) -> bool:
        return self.loopback_score > 0

    def __str__(self) -> str:
        tag = "  ← system audio?" if self.probable_loopback else ""
        return f"[{self.index:>2}] {self.name}  ({self.channels}ch){tag}"


def _score(name: str) -> int:
    low = name.lower()
    for i, hint in enumerate(LOOPBACK_HINTS):
        if hint in low:
            return len(LOOPBACK_HINTS) - i
    return 0


def list_devices() -> list[Device]:
    """Every input-capable device, ranked hint-first."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "audio device discovery needs: pip install 'thalamus[mic]'"
        ) from exc

    apis = sd.query_hostapis()
    out: list[Device] = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get("max_input_channels", 0)) < 1:
            continue
        name = str(d.get("name", f"device {i}"))
        api = ""
        try:
            api = str(apis[int(d.get("hostapi", 0))]["name"])
        except (IndexError, KeyError, TypeError, ValueError):
            pass
        out.append(
            Device(
                index=i,
                name=name,
                channels=int(d["max_input_channels"]),
                default_samplerate=float(d.get("default_samplerate", 0) or 0),
                hostapi=api,
                loopback_score=_score(name),
            )
        )
    return out


def detect_loopback() -> Device | None:
    """Best guess at the device carrying system output, or None.

    A guess, not a detection: it reads device *names*. Always confirm with
    ``thalamus devices`` before trusting which half of the call you captured.
    """
    candidates = [d for d in list_devices() if d.probable_loopback]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d.loopback_score, d.channels))


def resolve_device(spec: str | int | None) -> int | None:
    """Accept an index, a case-insensitive name fragment, or None.

    Names are preferred over indices in anything you keep: PortAudio device
    indices are not stable across reboots or when a USB interface is plugged in.
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, int):
        return spec
    text = str(spec).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    low = text.lower()
    matches = [d for d in list_devices() if low in d.name.lower()]
    if not matches:
        raise RuntimeError(
            f"no input device matching {text!r}. Run `thalamus devices` to list them."
        )
    if len(matches) > 1:
        names = ", ".join(f"[{m.index}] {m.name}" for m in matches[:5])
        raise RuntimeError(f"{text!r} matches several devices: {names}")
    return matches[0].index
