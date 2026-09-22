from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import webbrowser
from pathlib import Path

from .backends import get_backend
from .calibration import Calibration
from .channels import BANK, build_questions, gate_channels, trace_specs
from .cost import PRICE_PER_MTOK
from .engine import Engine
from .label import label_file
from .record import Recorder
from .hud.server import Broadcaster, make_server
from .state import RollingState

DEMO = Path(__file__).resolve().parents[1] / "demo" / "sales_call.jsonl"
AGENDA = "Pricing walkthrough and next steps for the Q4 rollout."


def _run(args) -> int:
    bus = Broadcaster()
    server = None
    if not args.no_hud:
        try:
            server = make_server(bus, port=args.port)
        except OSError as exc:
            print(f"\n  cannot serve the HUD on port {args.port}: {exc}"
                  f"\n  another instance may be running — try --port {args.port + 1}"
                  f" or --no-hud\n", file=sys.stderr)
            return 2
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{args.port}"
        print(f"  HUD    {url}")
        if args.open:
            webbrowser.open(url)

    try:
        backend = get_backend(args.backend)
    except RuntimeError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 2

    cal = Calibration.identity()
    if args.calibration and Path(args.calibration).exists():
        cal = Calibration.load(args.calibration)
        print(f"  cal    {args.calibration} ({len(cal.maps)} channels)")
    else:
        print("  cal    none — thresholds are nominal, not measured")

    recorder = None
    if getattr(args, "record", None):
        recorder = Recorder(args.record, per_bin=args.record_per_bin)
        print(f"  record {args.record} — up to {args.record_per_bin} samples per "
              f"probability decile, per channel")
        print("         rows contain transcript text; treat the file as a recording")

    engine = Engine(
        backend,
        rate_hz=args.rate,
        state=RollingState(agenda=AGENDA),
        calibration=cal,
        on_frame=lambda f: bus.publish(f.to_json()),
        recorder=recorder,
    )

    if args.source == "replay":
        from .sources.replay import replay_source

        src = replay_source(args.script, speed=args.speed, loop=args.loop)
        print(f"  source {Path(args.script).name} @ {args.speed}x")
        if args.speed != 1.0:
            print("         note: smoothing constants are in real seconds, so "
                  "replaying faster than 1x suppresses fires")
    elif args.source == "mic":
        from .sources.deepgram import dual_source, preflight

        try:
            preflight()
            src = dual_source(
                mic_device=args.mic_device,
                system_device=args.system_device,
                far_side=args.far_side,
                model=args.model,
                eager=args.eager,
                log=lambda m: print(f"  audio  {m}"),
            )
        except RuntimeError as exc:
            print(f"\n  {exc}\n", file=sys.stderr)
            return 2
        print(f"  source Deepgram Flux · {args.model}"
              f"{' · eager end-of-turn' if args.eager else ''}")
    else:
        from .sources.stdin import stdin_source

        src = stdin_source()
        print("  source stdin — type lines like 'THEM: that's too expensive'")

    print(f"  engine {args.rate}Hz · {len(BANK)} channels · "
          f"{len(build_questions(BANK))} questions · {len(engine.traces)} traces")
    print(f"  backend {backend.name}\n")

    try:
        asyncio.run(engine.run(src))
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.shutdown()
        if recorder:
            recorder.close()

    m = engine.meter.snapshot()
    print(f"\n  {engine.tick_count} ticks · {m['calls']} calls · "
          f"gate opened {m['gate_open_rate']*100:.0f}% · "
          f"p50 {m['p50_ms']}ms · spent ${m['spent']:.5f} "
          f"(${m['dollars_per_hour']:.2f}/hr)")
    if recorder:
        print("\n" + recorder.summary())
    print(f"  {len(engine.cards)} interrupts fired")
    for c in engine.cards[-8:]:
        print(f"    t+{c.t:6.1f}s  [{c.severity:<5}] {c.title}")
    return 0


def _devices() -> int:
    try:
        from .sources.audio import detect_loopback, list_devices
    except RuntimeError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 2
    try:
        devices = list_devices()
    except RuntimeError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 2

    print("\n  Audio inputs\n")
    for d in sorted(devices, key=lambda x: (-x.loopback_score, x.index)):
        api = f"  ({d.hostapi})" if d.hostapi else ""
        print(f"  {d}{api}")

    guess = detect_loopback()
    print()
    if guess:
        print(f"  Far side looks like [{guess.index}] {guess.name}\n"
              f"    thalamus mic --system-device \"{guess.name}\"")
    else:
        print("  No loopback device found. Thalamus can only hear you, and\n"
              "  floor / confusion / interest / objection are all questions about\n"
              "  the OTHER party. To capture them:\n"
              "    macOS    brew install --cask blackhole-2ch, then build a\n"
              "             Multi-Output Device in Audio MIDI Setup so you still\n"
              "             hear the call\n"
              "    Windows  enable Stereo Mix, or use a WASAPI loopback input\n"
              "    Linux    pick the .monitor source for your output sink")
    print("\n  Wear headphones. On speakers your microphone also picks up the far\n"
          "  side, both sockets transcribe it, and every channel sees it twice.\n")
    return 0


def _channels(args) -> int:
    specs = trace_specs(args.rate)
    gate_ids = {c.id for c in gate_channels()}
    print(f"\n  {len(BANK)} channels → {len(build_questions(BANK))} Jev questions "
          f"(each asked twice) → {len(specs)} traces\n")
    hdr = f"  {'trace':<26}{'tier':<7}{'fires on':<10}{'fire':>7}{'release':>9}{'τ (s)':>8}{'dwell':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for key, (ch, _opt, spec) in specs.items():
        tier = "gate" if ch.id in gate_ids else "bank"
        fires = "collapse" if spec.polarity.value == "low" else "climb"
        mark = " ·silent" if ch.silent else ""
        print(f"  {key:<26}{tier:<7}{fires:<10}{spec.fire:>7.3f}{spec.release:>9.3f}"
              f"{spec.ema_tau_s:>8.2f}{spec.dwell_s:>7.1f}{mark}")
    print("\n  Bands and time constants are derived from each channel's accuracy, "
          "not hand-tuned.\n  See thalamus/dsp.py: attractors(), required_tau(), suggest_band().\n")
    return 0


def _budget(args) -> int:
    gate_q = build_questions(gate_channels())
    bank_q = build_questions(BANK)
    gq = sum(len(str(q)) // 4 for q in gate_q.values())
    bq = sum(len(str(q)) // 4 for q in bank_q.values())
    bank_state = args.state_tokens
    gate_state = args.gate_state_tokens
    gate_tok, bank_tok = gate_state + gq, bank_state + bq

    print(f"\n  Per call: gate {gate_tok} tok ({gate_state} state + {gq} questions), "
          f"bank {bank_tok} tok ({bank_state} + {bq}).")
    print(f"  Input billed at ${PRICE_PER_MTOK}/MTok; output is free.\n")
    print(f"  {'rate':>6}{'gate open':>12}{'tok/s':>9}{'$/hour':>10}"
          f"{'$/8h day':>11}{'req/s':>8}{'vs 1-stage':>12}")
    print("  " + "─" * 68)
    for hz in (1.0, 2.0, 5.0, 10.0):
        for open_rate in (0.2, 0.4, 0.8):
            tps = hz * gate_tok + hz * open_rate * bank_tok
            single = hz * bank_tok
            per_hr = tps * 3600 / 1e6 * PRICE_PER_MTOK
            delta = (single - tps) / single
            tag = f"{delta:+.0%}"
            print(f"  {hz:>5.0f}H{open_rate:>11.0%}{tps:>9.0f}{per_hr:>10.2f}"
                  f"{per_hr*8:>11.2f}{hz*(1+open_rate):>8.1f}{tag:>12}")

    breakeven = (bank_tok - gate_tok) / bank_tok
    print(f"\n  A pre-filter is not automatically cheaper. This split only wins "
          f"while the\n  gate opens less than {breakeven:.0%} of the time -- above "
          f"that you are paying for\n  the state twice on every open tick and "
          f"would be better off running the\n  full bank every tick. Keep the "
          f"gate's state window narrow; that is what\n  moves the break-even, "
          f"and `thalamus demo` reports the realised saving.")
    print("\n  Documented ceilings: 250,000 tok/s and 1,200 req/min (= 20 req/s), "
          "so\n  latency is the binding constraint, not quota.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    # These commands run for the length of a call, so a piped or redirected
    # stdout must not sit in a block buffer -- you would see nothing at all
    # until the buffer filled, and conclude it had hung.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(
        prog="thalamus",
        description="A semantic interrupt controller. The model decides when; "
                    "your code decides what.",
    )
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--backend", choices=("mock", "jev"), default="mock")
        sp.add_argument("--rate", type=float, default=5.0, help="ticks per second")
        sp.add_argument("--port", type=int, default=8777)
        sp.add_argument("--no-hud", action="store_true")
        sp.add_argument("--open", action="store_true", help="open the HUD in a browser")
        sp.add_argument("--calibration", default="calibration.json")
        sp.add_argument("--record", metavar="PATH",
                        help="write a stratified sample of raw judgements for "
                             "labelling, so you can calibrate on a real session")
        sp.add_argument("--record-per-bin", type=int, default=25, metavar="N",
                        help="samples kept per probability decile per channel "
                             "(default 25 → ~250 rows per channel)")

    d = sub.add_parser("demo", help="replay a scripted call (no API key needed)")
    common(d)
    d.add_argument("--script", default=str(DEMO))
    d.add_argument("--speed", type=float, default=1.0)
    d.add_argument("--loop", action="store_true")
    d.set_defaults(source="replay")

    l = sub.add_parser("live", help="drive from typed lines on stdin")
    common(l)
    l.set_defaults(source="stdin", script=None, speed=1.0, loop=False)

    mic = sub.add_parser(
        "mic", help="live capture via Deepgram Flux — both speakers (untested)")
    common(mic)
    mic.add_argument("--mic-device", default=None,
                     help="input device index or name fragment for your voice")
    mic.add_argument("--system-device", default=None,
                     help="loopback device carrying the far side; auto-detected "
                          "by name if omitted (see `thalamus devices`)")
    mic.add_argument("--no-far-side", dest="far_side", action="store_false",
                     help="capture your microphone only, and accept that half "
                          "the channel bank is measuring a speaker it cannot hear")
    mic.add_argument("--model", default="flux-general-en")
    mic.add_argument("--eager", action="store_true",
                     help="fire on EagerEndOfTurn, ~200ms sooner but retractable")
    mic.set_defaults(source="mic", script=None, speed=1.0, loop=False, far_side=True)

    sub.add_parser("devices", help="list audio inputs and guess the loopback one")

    lb = sub.add_parser("label", help="answer y/n on a recorded session")
    lb.add_argument("path", help="the JSONL written by --record")
    lb.add_argument("--channel", default=None,
                    help="only label traces whose key contains this")

    c = sub.add_parser("channels", help="print the bank and its derived bands")
    c.add_argument("--rate", type=float, default=5.0)

    b = sub.add_parser("budget", help="what a continuous loop actually costs")
    b.add_argument("--state-tokens", type=int, default=900,
                   help="rolling state the bank sees")
    b.add_argument("--gate-state-tokens", type=int, default=260,
                   help="narrower slice the gate sees")

    args = p.parse_args(argv)
    if args.cmd == "label":
        return label_file(args.path, key_filter=args.channel)
    if args.cmd == "devices":
        return _devices()
    if args.cmd == "channels":
        return _channels(args)
    if args.cmd == "budget":
        return _budget(args)
    if args.cmd in ("demo", "live", "mic"):
        print(f"\n  thalamus — semantic interrupt controller")
        return _run(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
