"""Print every strictly-partial D_feedback example for operator review.

Reads ``results/d_feedback_records.json`` (per-example {id, question,
gold_answer, predicted_answer, f1}) and ``results/difficulty_table.json``
(the frozen 150-id table). Prints:

  1. Overall counts (F1==0, F1==1, 0<F1<1) over 150.
  2. The two tercile boundary F1 values explicitly.
  3. Per-tercile breakdown (zeros / ones / strictly-partial per tercile).
  4. Every strictly-partial example sorted by F1 ascending, with id,
     F1, question, gold answer, predicted answer.
  5. min / median / max of the strictly-partial F1 set.

No API spend.

Run: ``.venv/bin/python -m src.inspect_partial``
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import SPLIT_ORDER, load_splits  # noqa: E402
from src.difficulty import BAND_NAMES, DifficultyTable  # noqa: E402

RECORDS_PATH = REPO / "results" / "d_feedback_records.json"
TABLE_PATH = REPO / "results" / "difficulty_table.json"


def _count_classes(scores) -> tuple[int, int, int]:
    n_zero = sum(1 for s in scores if s == 0.0)
    n_one = sum(1 for s in scores if s == 1.0)
    n_partial = sum(1 for s in scores if 0.0 < s < 1.0)
    return n_zero, n_one, n_partial


def main() -> int:
    if not RECORDS_PATH.exists():
        print(f"ERROR: {RECORDS_PATH} does not exist. Run src.score_d_feedback first.")
        return 1
    if not TABLE_PATH.exists():
        print(f"ERROR: {TABLE_PATH} does not exist. Run src.score_d_feedback first.")
        return 1

    records = json.loads(RECORDS_PATH.read_text())
    table = DifficultyTable.load(TABLE_PATH)

    # Align records with the d_feedback order so band lookups by list-index
    # match the table.
    splits = load_splits()
    d_feedback = splits[0]
    n = len(d_feedback)
    if n != table.n:
        print(f"ERROR: d_feedback size {n} != table.n {table.n}")
        return 1

    # Per-id F1 list, in d_feedback order (== table id order).
    all_scores = [records[ex["id"]]["f1"] for ex in d_feedback]

    # Overall counts
    n_zero, n_one, n_partial = _count_classes(all_scores)

    # Tercile boundaries
    hard_scores = sorted(table.scores[i] for i in table.ids("hard"))
    mid_scores = sorted(table.scores[i] for i in table.ids("mid"))
    easy_scores = sorted(table.scores[i] for i in table.ids("easy"))
    b1 = max(hard_scores)
    b2 = max(mid_scores)

    # Per-tercile breakdown
    per_tercile = {}
    for band in BAND_NAMES:
        band_scores = [table.scores[i] for i in table.ids(band)]
        per_tercile[band] = _count_classes(band_scores)

    print(f"=== D_feedback inspection (n={n}, distractor substrate) ===")
    print()
    print(f"1) Overall counts:")
    print(f"   F1 == 0.0:           {n_zero}")
    print(f"   F1 == 1.0:           {n_one}")
    print(f"   0 < F1 < 1 (strict): {n_partial}")
    print(f"   sum: {n_zero + n_one + n_partial}")
    print()
    print(f"2) Tercile boundaries (by F1 rank):")
    print(f"   hard | mid  boundary: F1 = {b1:.4f}   (= max F1 of hard tercile)")
    print(f"   mid  | easy boundary: F1 = {b2:.4f}   (= max F1 of mid tercile)")
    print()
    print(f"3) Per-tercile breakdown:")
    print(f"   {'band':<6} {'size':>5} {'F1==0':>7} {'F1==1':>7} {'0<F1<1':>8}")
    for band in BAND_NAMES:
        sz = len(table.ids(band))
        z, o, p = per_tercile[band]
        print(f"   {band:<6} {sz:>5} {z:>7} {o:>7} {p:>8}")
    print()

    # 4) Every strictly-partial example, sorted by F1 ascending.
    print(f"4) Every strictly-partial example (sorted by F1 ascending):")
    print()
    partials = [
        records[ex["id"]] for ex in d_feedback
        if 0.0 < records[ex["id"]]["f1"] < 1.0
    ]
    partials.sort(key=lambda r: (r["f1"], r["id"]))

    for r in partials:
        print(f"  id={r['id']}  F1={r['f1']:.4f}")
        print(f"    Q:    {r['question']}")
        print(f"    gold: {r['gold_answer']!r}")
        print(f"    pred: {r['predicted_answer']!r}")
        print()

    # 5) Partial-F1 spread
    partial_f1s = [r["f1"] for r in partials]
    if partial_f1s:
        print(f"5) Partial F1 spread (n={len(partial_f1s)}):")
        print(f"   min:    {min(partial_f1s):.4f}")
        print(f"   median: {statistics.median(partial_f1s):.4f}")
        print(f"   max:    {max(partial_f1s):.4f}")
    else:
        print("5) No strictly-partial examples found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
