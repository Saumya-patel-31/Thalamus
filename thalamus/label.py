"""Label a recorded session, one channel at a time.

Grouped by channel on purpose. Answering "was the speaker confused?" forty
times in a row is a different and far faster task than answering forty
different questions in sequence, because you hold one definition in your head
instead of reloading a new one every row. It also surfaces your own
inconsistency: if you find yourself answering the same-looking context
differently, the *question* is ambiguous, which is worth knowing before you
calibrate against it.

    thalamus label session.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["label_file"]

HELP = """
  y / n    yes / no          s  skip this one
  b        back one row      q  save and quit
  ?        repeat the question
"""


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)  # atomic: a Ctrl-C mid-write must not eat the session


def label_file(path: str | Path, *, key_filter: str | None = None) -> int:
    p = Path(path)
    if not p.exists():
        print(f"\n  no such recording: {p}\n")
        return 2
    rows = _load(p)
    if not rows:
        print(f"\n  {p} is empty\n")
        return 2

    todo = [
        i for i, r in enumerate(rows)
        if r.get("truth") is None
        and (key_filter is None or key_filter in r.get("key", ""))
    ]
    done_already = sum(1 for r in rows if r.get("truth") is not None)
    if not todo:
        print(f"\n  nothing left to label ({done_already}/{len(rows)} done)\n")
        return 0

    # Channel-major, then by raw value so similar cases sit together.
    todo.sort(key=lambda i: (rows[i]["key"], rows[i]["raw"]))

    print(f"\n  {len(todo)} unlabelled of {len(rows)} in {p.name}")
    print("  Answer what was ACTUALLY true, not what the model said.")
    print(HELP)

    labelled = 0
    current_key = None
    i = 0
    try:
        while i < len(todo):
            row = rows[todo[i]]
            if row["key"] != current_key:
                current_key = row["key"]
                remaining = sum(1 for j in todo[i:] if rows[j]["key"] == current_key)
                print(f"\n{'─' * 66}\n  {current_key}   ({remaining} to go)")
                print(f"  {row.get('instructions') or '(no question recorded)'}\n")

            for line in row.get("context") or []:
                print(f"      {line}")
            prompt = f"    [model said {row['raw']:.2f}]  y/n/s/b/q > "

            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                break
            if answer == "q":
                break
            if answer == "?":
                print(f"  {row.get('instructions')}\n")
                continue
            if answer == "b":
                i = max(i - 1, 0)
                continue
            if answer in ("s", ""):
                i += 1
                continue
            if answer not in ("y", "n"):
                print(HELP)
                continue
            row["truth"] = answer == "y"
            labelled += 1
            i += 1
            if labelled % 20 == 0:
                _save(p, rows)  # checkpoint
    except KeyboardInterrupt:
        print()

    _save(p, rows)
    total = sum(1 for r in rows if r.get("truth") is not None)
    print(f"\n  labelled {labelled} this pass · {total}/{len(rows)} total")
    per_channel: dict[str, int] = {}
    for r in rows:
        if r.get("truth") is not None:
            per_channel[r["key"]] = per_channel.get(r["key"], 0) + 1
    ready = sum(1 for n in per_channel.values() if n >= 80)
    print(f"  {ready} of {len(per_channel)} channels have enough to fit (80+)")
    if ready:
        print(f"  next:  python3 scripts/fit_calibration.py {p} -o calibration.json")
    else:
        print("  keep going — a fit wants roughly 80 labelled rows per channel")
    print()
    return 0
