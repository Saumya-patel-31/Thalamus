# Thalamus

**A semantic interrupt controller.** The model decides *when*; your code decides *what*.

Named after the brain's relay station — the part that gates which signals reach
conscious attention. Jev is the thalamus. Your LLM is the cortex, and it stays
asleep.

```bash
python3 -m thalamus demo --open
```

No API key. No dependencies. No build step.

---

## The idea

Everyone is using [Jev](https://docs.typesafe.ai) as *a faster classifier*. That's
the boring read.

At 70–500 ms, $0.042/MTok, output free, and *"adding questions barely changes the
response time"*, you can ask eighty semantic questions about the world five times
a second. That crosses a threshold where meaning stops being a **query** and
becomes a **sampled signal** — and once meaning is a signal you can do signal
processing on it: smoothing, derivatives, hysteresis, edge detection,
cross-channel correlation.

Which matters for one specific reason. **Jarvis was never a generation problem.**
Any decent model can write what Jarvis says. What was missing is *knowing when to
speak unprompted*, and that is a decision problem that has to run continuously,
over everything, forever. It has been economically and physically impossible.

Thalamus is the interrupt controller. It costs about a dollar an hour.

## What you'll see

A 116-second sales call replays against a deliberately unreliable model. Roughly
**1% of ticks produce a card**; the rest are silence, which is the hard part.

```
t+ 19.4s  [info ] check_that         Checkable claim
t+ 24.6s  [nudge] monologue          You've been talking a while
t+ 40.0s  [nudge] repeating          Already covered
t+ 49.2s  [alert] hanging_question   Unanswered question
t+ 59.8s  [alert] objection_price    Price objection
t+ 67.0s  [alert] losing_them        You're losing them
t+ 79.4s  [alert] awkward_silence    They're waiting on you
t+ 83.4s  [info ] clean_commitment   Action item captured
t+114.4s  [alert] wrap_up            You could close this
```

**Flip the `smoothing` toggle in the HUD.** The raw traces flap wildly — that is
the model's actual per-tick output, right about three times in four. Smoothed,
the same data reads cleanly and the cards land with a standard deviation of
**2–4 seconds** across runs. That contrast is the entire argument, and it runs on a laptop with no
key.

## Architecture

```
mic ──► STT ──► rolling 45s window (pruned in code, to fight context rot)
                          │
  TICK (5 Hz)             ▼
  ├─ STAGE 1  GATE   3 channels, every tick ......... ~80 ms
  │     └─ gate closed? stop here. ~20% of ticks.
  └─ STAGE 2  BANK   12 channels, 24 questions ...... ~260 ms
                          │
                          ▼
     SIGNAL LAYER   (pure Python, 0 ms — where the product lives)
       EMA with a time constant · Schmitt trigger · dwell · refractory
       least-squares slope · phrasing-disagreement suppression
                          │
                          ▼
     RULES ──► a pre-written card, a deterministic action, or (usually) silence
```

Two disciplines hold the whole thing together:

**The model is asked only what is true *right now*.** Every duration, count and
integral is computed in Python. "Have I been monologuing?" is not a question —
it is `floor:me` integrated over thirty seconds. Jev is documented to be
unreliable at counting, arithmetic and date ordering, so it is never given the
opportunity.

**Every channel is asked twice, in two phrasings.** Output tokens are free and
questions run in parallel, so the second phrasing is nearly free in wall-clock
terms. Agreement costs nothing; *disagreement is a free measurement that the
question is ill-posed*, and the channel suppresses itself. You cannot get that
from a model you only ask once.

## The design equation

The least obvious thing here, and it cost a failing test to find.

Intuition says threshold a confident channel near 0.75. **That is wrong**, and it
produces a channel that never fires. Smoothing a sensor of accuracy `a` that
emits ~`p_hit` when right and ~`p_miss` when wrong drives the EMA toward a *fixed
point*, not toward 1.0:

```
true  condition → a·p_hit  + (1−a)·p_miss
false condition → a·p_miss + (1−a)·p_hit
```

At `a = 0.70` those are **0.66 and 0.34**. A band at 0.75 is never reached; a
band at 0.66 sits exactly on the attractor and coin-flips. The band belongs
*between* the attractors. Measured, on identical data:

| band | recall |
|---|---|
| naive 0.75 | **0.28** |
| derived (`suggest_band`) | **1.00** |

Then noise. After EWMA smoothing, `σ_ema = σ_sample · √(α/(2−α))`, and you need
the gap to clear it by ~3σ. Reliability is bought with **τ × rate**, so:

> **Sample rate converts directly into response speed at fixed reliability.**
> A 70%-accurate channel needs ~2.8 s of smoothing at 5 Hz — but only ~1.2 s at
> 12 Hz. *That* is what an 80 ms model buys you. It is a capability, not a saving.

Accuracy pays even better, since it widens the gap *and* shrinks the noise: an
85% channel at 5 Hz needs 0.55 s, beating a 70% channel at 12 Hz. **Fix the
phrasing before you raise the tick rate.**

None of the thresholds in this repo are hand-tuned. `thalamus channels` prints
them; they all fall out of `attractors()`, `required_tau()` and `suggest_band()`
in [`thalamus/dsp.py`](thalamus/dsp.py).

### Oversampling rescues a bad sensor

The architectural bet, asserted in [`tests/test_dsp.py`](tests/test_dsp.py)
rather than in prose. A 70%-accurate, overconfident sensor, sampled at 5 Hz over
a 20-second condition:

```
recall 1.000   ·   false-positive rate 0.010
```

With the honest counterweight, also tested: **oversampling kills variance, not
bias.** A channel Jev systematically misreads is wrong on every tick and sails
straight through the filter. That is what the two-phrasing disagreement check is
for.

## The numbers

```bash
python3 -m thalamus budget
```

Measured over the demo, not estimated — the meter shares the engine's clock, so
a virtual-clock replay reports the rates the run would produce live:

| | |
|---|---|
| 5 Hz, gate opening 40% | **$0.48/hr** of continuous ambient awareness |
| Realised two-stage saving | **+21%** of tokens vs running the full bank every tick |
| Rate-limit headroom | 7 req/s of 20; ~6k tok/s of 250k |

### A pre-filter is not automatically cheaper

The most surprising thing I measured. A cheap gate in front of an expensive bank
*loses* money whenever the gate opens often, because **the state gets billed
twice on every open tick**. With the gate reading the same 45 s window as the
bank, the two-stage split cost **18% more** than just running the full bank.

The fix is architectural, not a constant: the gate reads a 15 s window, because
"did something just happen / who is talking / is a question hanging" does not
need the whole conversation. That moves the break-even from ~45% to ~72% gate-open.

`thalamus budget` prints the break-even for your own shape, and the meter reports
the *realised* saving so a regression is visible rather than silent.

## Running against real Jev

```bash
export TYPESAFE_API_KEY=sk-...
python3 -m thalamus live --backend jev --open
# then type:  THEM: honestly that's way more than we budgeted
```

`--backend jev` posts to `https://api.typesafe.ai/v1/systemone` via stdlib
`urllib` (no SDK needed), retrying 429/529 with exponential backoff.

### A real call — both speakers

```bash
pip install "thalamus[mic]"
export DEEPGRAM_API_KEY=... TYPESAFE_API_KEY=...
thalamus devices                              # find your loopback device
thalamus mic --system-device "BlackHole" --backend jev --eager
```

**Thalamus opens two Flux sockets, not one.** Your microphone is tagged `ME`; a
loopback device carrying system output is tagged `THEM`. That second socket is
not a nicety — `floor`, `confusion`, `interest` and `objection` are *all*
questions about the other party, so a mic-only setup measures the wrong half of
the call. `thalamus devices` lists your inputs and guesses which one carries
system audio:

```
  [ 3] BlackHole 2ch  (2ch)  ← system audio?  (Core Audio)
  [ 1] MacBook Pro Microphone  (1ch)  (Core Audio)

  Far side looks like [3] BlackHole 2ch
    thalamus mic --system-device "BlackHole 2ch"
```

| platform | how to capture system output |
|---|---|
| macOS | `brew install --cask blackhole-2ch`, then build a **Multi-Output Device** in Audio MIDI Setup containing both your speakers and BlackHole, and select it as system output — so you still hear the call |
| Windows | WASAPI loopback, or enable **Stereo Mix** / *What U Hear* |
| Linux | PulseAudio/PipeWire expose a `.monitor` source per sink |

**Wear headphones.** On speakers your microphone also picks up the far side,
both sockets transcribe it, and every channel sees the same words twice.

Deepgram Flux is the right front end because end-of-turn detection lives
*inside* the model — median EOT ~260 ms, native barge-in — where a conventional
STT plus an external VAD adds 200–600 ms before Thalamus sees a word. `--eager`
fires on `EagerEndOfTurn`, a few hundred ms sooner but retractable; worth taking
for a reflex, since a card that appears early and is superseded costs less than
one that arrives after the moment passed.

Each socket reconnects independently with exponential backoff, and **one
speaker's socket dying degrades the session rather than ending it** — the
surviving stream keeps running and the orphaned channels decay rather than
freezing on a stale value.

> **Untested against a live key in this repo.** The capture path is written
> against the documented Flux v2 schema but has never run against real audio
> here; the merge, device-resolution and reconnect logic *are* tested. Replay
> and stdin are the exercised sources.

## The HUD

It is a scope, so it behaves like one.

| key | |
|---|---|
| `space` | **Hold** — freeze the traces to read them. Data keeps arriving; cards keep landing. |
| `s` | Smoothing on/off. Off is the raw per-tick model output. |
| `r` | Raw overlay underneath each smoothed trace |
| `c` | Collapse/expand channel groups |
| `e` | Export the session as JSON — every trace's series, spec and fire count, plus the cards |
| `t` | Light / dark |
| `?` | This list |

Click any channel for its rationale and live state: fire and release lines, τ,
dwell, measured accuracy, current slope, phrasing disagreement, and whether it
is *warming*, *armed*, *latched* or *ill-posed*. Hovering a card dims every
channel that did not contribute to it, so you can see exactly what fired it.

Reloading mid-call keeps your history — cards are served separately from the
live stream at `/session`, so a refresh does not wipe the session.

## Adding a channel

```python
Channel(
    id="budget_signal",
    kind="noul",
    instructions="The speaker has revealed a specific budget figure",
    alt="A concrete number for available spend has been stated",
    why="Rising edge tells the deal model it can stop guessing.",
    accuracy=0.80,     # measure it; the band and τ are derived from this
    dwell_s=1.5,
)
```

Phrase it **positively** — Jev reads negations at face value, so
`question_answered` asks whether the question *was* answered and watches for the
probability collapsing (`Polarity.LOW`). Never ask about time or counts.

## Calibrating on a real session

Every threshold is a number in probability space, so it only works if the
probabilities are honest. Independent audits put Jev's raw ECE around
0.064–0.160 and find it **overconfident** — one published bin states 0.97 and is
right 88.6% of the time. Isotonic regression brings that to ~0.006–0.018.

```bash
thalamus mic --record session.jsonl      # 1. capture a real call
thalamus label session.jsonl             # 2. answer y/n on the samples
python3 scripts/fit_calibration.py session.jsonl -o calibration.json
```

### Why `--record` samples instead of logging everything

Seventeen traces at 5 Hz is ~85 rows a second, so half an hour of conversation
is **150,000 rows** — and nobody labels that. Worse, almost all of them sit near
0.0 or 1.0 where the channel is already unambiguous, so labelling them teaches
the map nothing.

What isotonic actually needs is **coverage across the probability range**,
especially through the uncertain middle. So the recorder bins each channel's
output into deciles and keeps a quota per bin:

```
recorded 1899 samples from 5942 observations (32.0% kept) → session.jsonl
17 channels; thinnest per populated bin: objection:trust (69 rows / 8 bins)
```

Rows per *populated bin* matters more than raw row count. A confident channel
only ever emits near 0 and near 1, so it populates two or three bins and needs
far fewer labels — there are simply fewer distinct regions for the map to learn.

### Labelling

`thalamus label` groups rows **by channel**, not chronologically. Answering
"was the speaker confused?" forty times in a row is a far faster task than
answering forty different questions, because you hold one definition in your
head instead of reloading a new one every row. It also surfaces your own
inconsistency: if the same-looking context gets different answers, the
*question* is ambiguous — worth knowing before you calibrate against it.

`y`/`n`/`s`kip/`b`ack/`q`uit, checkpointed every 20 rows, saved atomically.

### Fitting

The fit holds out half the data, measures ECE before and after per channel, and
**writes a map only where it wins** — calibrating an already-calibrated channel
makes it slightly worse:

```
confusion            182   0.77      0.1599   0.1469  keep
floor:me             188   0.76      0.0855   0.1380  drop
objection:trust       69                              too few
```

It also prints measured per-channel accuracy. **Feed those back into
`Channel.accuracy`** — the trigger bands and time constants are all derived from
it, so a channel you measured at 0.86 gets a tighter band and four times less
smoothing than one you guessed at 0.72.

Regularisation matters more than the fit itself: unregularised PAVA on a few
thousand noisy labels overfits into a staircase running 0→1 *inside* a region
where the raw score carries no information. Minimum block size collapses it back
onto the honest base rate.

## Honest limitations

- **~68–77% per channel.** TypeSafe's own workflow evals land ~68%; independent
  public-benchmark numbers run ~72–77%. The signal layer is what makes that
  usable, and it cannot fix systematic misreads.
- **Text only.** Audio, vision and prosody are out of scope; everything arrives
  as transcript. Real confusion lives in tone, and this cannot hear it.
- **Adversarial input works.** Text engineered to influence classification will
  succeed. Do not point this at untrusted speakers and then act on it.
- **Recording other people has rules.** Two-party-consent jurisdictions require
  everyone's agreement before you capture a call. Thalamus keeps no audio and
  writes no transcript to disk, but it does send text to two third parties
  (Deepgram and TypeSafe). That is your disclosure to make, not the library's.
- **The mock is a simulator, not Jev.** It reproduces the *character* (accuracy,
  overconfidence), not the judgement. Numbers from `demo` describe the mock.
- **Bank channels are slow by construction.** Sampled only on the ~40% of ticks
  where the gate opens, they need 2–4 s of smoothing and fire with roughly twice
  the timing spread of gate channels. Wrap-up-style channels are minutes-scale.
- **Conjunctions of two bank channels are the least reliable pattern.**
  `vague_commitment` needs `commitment` to fire *and* `commitment_bound` to be
  low, and lands on about 7 runs in 16 where single-channel rules land on 15.
- **Replaying faster than 1× suppresses fires**, because time constants are in
  real seconds while the script compresses.

## Tests

```bash
python3 tests/test_dsp.py          # 20 — the signal layer, exactly
python3 tests/test_integration.py  #  7 — the whole pipeline, on a virtual clock
python3 tests/test_sources.py      #  7 — speaker merge, reconnect, device naming
python3 tests/test_record.py       #  8 — stratified capture, atomic labelling
python3 tests/test_hud.py          #  5 — SSE transport under hard disconnects
```

The integration tests replay 116 seconds of conversation in about a second by
handing the engine a virtual clock, then assert that the important
interrupts fire inside their causal window on ≥7 of 8 seeds with σ ≤ 6 s, that
fewer than 4% of ticks speak, that the two-stage gate actually saves tokens, and
that a 3.5-second topic drift never defeats a 12-second dwell.

They also pin a bug worth knowing about: `DEFAULT_RULES` holds *stateful*
objects (each rule remembers when it last fired), so sharing those instances
between two engines makes the second silently inherit the first's cooldowns.
`Engine` deep-copies; `default_rules()` exists so you do too.

## Sources

[Jev docs](https://docs.typesafe.ai) ·
[API reference](https://docs.typesafe.ai/api.md) ·
[practical guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e) ·
[launch coverage](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/) ·
[awesome-typesafe-jev](https://github.com/AbdelStark/awesome-typesafe-jev) ·
calibration audits: [OOD](https://github.com/scienthoon/jev-ood-calibration),
[isotonic](https://github.com/AnthusAI/Jev-Calibration),
[Cyrillic](https://github.com/AHTOOOXA/jev-cyrillic-audit) ·
[Deepgram Flux](https://deepgram.com/learn/introducing-flux-conversational-speech-recognition)
