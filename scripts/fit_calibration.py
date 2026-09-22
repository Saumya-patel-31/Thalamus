#!/usr/bin/env python3
"""Fit per-channel calibration maps, and refuse to ship the ones that don't help.

Every threshold in Thalamus is a number in probability space, so it only means
something if the probabilities are honest. Jev's raw output is documented to be
overconfident; isotonic regression fixes most of that. But calibration is not
free of risk -- fitting a map on a channel that is *already* calibrated makes it
slightly worse. So this tool holds out half the data, measures expected
calibration error before and after, and writes a map only where it wins.

Input: JSONL of {"key": "<trace key>", "raw": 0.87, "truth": true}

    python3 scripts/fit_calibration.py labelled.jsonl -o calibration.json

To see the workflow without labelling anything yourself, generate pairs from
the scripted demo (the mock backend knows its own ground truth):

    python3 scripts/fit_calibration.py --from-demo -o calibration.json

That is a *demonstration*, not a calibration: the labels come from the
simulator, so the resulting map tells you nothing about the real model. Use it
to see the report format, then label a few hundred real ticks per channel.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalamus.calibration import Calibration, expected_calibration_error, isotonic_fit


def load_pairs(path: Path) -> dict[str, list[tuple[float, bool]]]:
    """Read {key, raw, truth} rows, ignoring any still awaiting a label."""
    out: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)
    unlabelled = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("truth") is None:
            unlabelled += 1
            continue
        out[row["key"]].append((float(row["raw"]), bool(row["truth"])))
    if unlabelled:
        print(f"\n  skipping {unlabelled} unlabelled rows — "
              f"run `thalamus label {path}` to finish them")
    return out


async def from_demo(runs: int = 24) -> dict[str, list[tuple[float, bool]]]:
    """Harvest (raw, truth) pairs by replaying the demo, labelled per tick."""
    from tests.harness import run_script

    rows: list[dict] = []
    for seed in range(runs):
        await run_script(seed=seed, labels=rows)
    out: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)
    for r in rows:
        out[r["key"]].append((float(r["raw"]), bool(r["truth"])))
    return out


def decision_boundary(key: str) -> float:
    """Where "it said yes" starts, for this channel's shape.

    A Noul splits at 0.5. A five-way Choice puts at most ~0.74 on its winner
    and ~0.065 on each loser, so 0.5 is far too high a bar and would score a
    perfectly good channel as near-zero accuracy.
    """
    from thalamus.channels import BANK

    cid = key.split(":")[0]
    for c in BANK:
        if c.id == cid:
            p_hit, p_miss = c.mass()
            return (p_hit + p_miss) / 2
    return 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labelled", nargs="?", help="JSONL of {key, raw, truth}")
    ap.add_argument("--from-demo", action="store_true",
                    help="generate pairs from the scripted demo (illustrative only)")
    ap.add_argument("-o", "--out", default="calibration.json")
    ap.add_argument("--min-samples", type=int, default=80,
                    help="a confident channel populates only two or three "
                         "probability bins, so it needs far fewer rows than "
                         "one that spreads across the range")
    args = ap.parse_args()

    if args.from_demo:
        data = asyncio.run(from_demo())
        print("\n  NOTE: labels come from the simulator, so this map describes the "
              "\n  mock, not Jev. It exists to show the workflow and the report.\n")
    elif args.labelled:
        data = load_pairs(Path(args.labelled))
    else:
        ap.error("pass a labelled JSONL or --from-demo")
        return 2

    rng = random.Random(0)
    maps: dict[str, list[tuple[float, float]]] = {}
    acc: dict[str, float] = {}

    print(f"  {'trace':<26}{'n':>7}{'acc':>7}{'ECE before':>12}{'after':>9}{'':>4}")
    print("  " + "─" * 66)
    kept = skipped = 0
    for key, pairs in sorted(data.items()):
        if len(pairs) < args.min_samples:
            print(f"  {key:<26}{len(pairs):>7}{'':>7}{'':>12}{'':>9}  too few")
            skipped += 1
            continue
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        train, test = shuffled[:half], shuffled[half:]

        curve = isotonic_fit(train)
        cal = Calibration(maps={key: curve}, fitted=True)
        before = expected_calibration_error(test)
        after = expected_calibration_error([(cal.apply(key, p), y) for p, y in test])
        bound = decision_boundary(key)
        accuracy = sum(1 for p, y in pairs if (p >= bound) == y) / len(pairs)
        acc[key] = round(accuracy, 4)

        verdict = "keep" if after < before else "drop"
        if verdict == "keep":
            maps[key] = curve
            kept += 1
        else:
            skipped += 1
        print(f"  {key:<26}{len(pairs):>7}{accuracy:>7.2f}{before:>12.4f}"
              f"{after:>9.4f}  {verdict}")

    Calibration(maps=maps, accuracy=acc, fitted=bool(maps)).save(args.out)
    print(f"\n  wrote {args.out}: {kept} maps kept, {skipped} skipped")
    if acc:
        print("\n  Measured accuracies -- feed these back into each Channel's "
              "`accuracy=`\n  field, because the trigger bands and time constants "
              "are derived from them:")
        for k, v in sorted(acc.items(), key=lambda kv: kv[1]):
            print(f"    {k:<26}{v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
