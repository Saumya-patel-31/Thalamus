"""Recording and labelling a real session."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalamus.label import _load, _save
from thalamus.record import Recorder


def _tmp(name: str = "s.jsonl") -> Path:
    return Path(tempfile.mkdtemp()) / name


def _observe(rec: Recorder, key: str, raw: float, t: float = 0.0) -> bool:
    return rec.observe(key=key, raw=raw, alt=None, ema=raw, t=t, tick=1,
                       context=["ME: hello"], instructions="q?")


def test_quota_caps_each_probability_bin():
    """The whole point: an unbounded recorder writes 150k rows nobody labels."""
    p = _tmp()
    rec = Recorder(p, per_bin=5, bins=10)
    for _ in range(500):
        _observe(rec, "confusion", 0.93)     # all land in the top bin
    rec.close()
    assert rec.kept == 5, rec.kept
    assert rec.seen == 500


def test_quota_is_per_bin_not_per_channel():
    """Coverage across the range is what isotonic needs, so each bin gets its
    own allowance rather than competing for one pool."""
    p = _tmp()
    rec = Recorder(p, per_bin=3, bins=10)
    for decile in range(10):
        for _ in range(20):
            _observe(rec, "confusion", decile / 10 + 0.05)
    rec.close()
    assert rec.kept == 30, rec.kept
    rows, bins = rec.coverage()["confusion"]
    assert (rows, bins) == (30, 10)


def test_channels_do_not_share_a_quota():
    p = _tmp()
    rec = Recorder(p, per_bin=2, bins=10)
    for key in ("a", "b", "c"):
        for _ in range(10):
            _observe(rec, key, 0.5)
    rec.close()
    assert rec.kept == 6
    assert set(rec.coverage()) == {"a", "b", "c"}


def test_rows_are_written_as_they_arrive():
    """A call that crashes at minute 40 must not lose minutes 1-39."""
    p = _tmp()
    rec = Recorder(p, per_bin=5)
    _observe(rec, "confusion", 0.9)
    assert p.exists() and p.read_text().strip(), "nothing on disk before close()"
    rec.close()


def test_recorded_rows_await_a_human():
    p = _tmp()
    rec = Recorder(p, per_bin=2)
    _observe(rec, "confusion", 0.77, t=3.5)
    rec.close()
    row = json.loads(p.read_text().splitlines()[0])
    assert row["truth"] is None, "a recording must not invent its own labels"
    assert row["raw"] == 0.77 and row["key"] == "confusion"
    assert row["instructions"] and row["context"]


def test_recorder_keeps_the_raw_value_not_the_calibrated_one():
    """Calibration is what we are fitting; labelling its output is circular."""
    p = _tmp()
    rec = Recorder(p, per_bin=2)
    rec.observe(key="c", raw=0.91, alt=0.88, ema=0.40, t=1.0, tick=2,
                context=[], instructions="q")
    rec.close()
    row = json.loads(p.read_text().splitlines()[0])
    assert row["raw"] == 0.91 and row["ema"] == 0.4


def test_label_save_is_atomic_and_round_trips():
    p = _tmp()
    rec = Recorder(p, per_bin=2)
    for v in (0.1, 0.9):
        _observe(rec, "confusion", v)
    rec.close()

    rows = _load(p)
    assert len(rows) == 2
    rows[0]["truth"] = True
    _save(p, rows)
    assert not p.with_suffix(p.suffix + ".tmp").exists(), "left a temp file behind"
    back = _load(p)
    assert back[0]["truth"] is True and back[1]["truth"] is None


def test_summary_reports_rows_per_bin():
    p = _tmp()
    rec = Recorder(p, per_bin=4, bins=10)
    for d in range(3):
        for _ in range(10):
            _observe(rec, "narrow", d / 10 + 0.05)
    rec.close()
    text = rec.summary()
    assert "12 rows / 3 bins" in text, text


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
