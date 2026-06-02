"""Chunk 5 diagnostic and GO/NO-GO verdict (distractor substrate).

Reads ``results/difficulty_table.json`` (written by
``src.score_d_feedback``), plots the per-id F1 distribution, and prints
the GO/NO-GO verdict using the *distractor-substrate* criterion:

    The middle (frontier) tercile must contain a meaningful share of
    STRICTLY PARTIAL instances (0 < F1 < 1). Equal-tercile-by-rank
    binning guarantees a non-empty middle tercile by construction, so
    that is not the gate; the gate is content. The threat under the
    distractor substrate is domination by F1==1.0 (the task LM is strong
    enough to ace many questions), which would leave the frontier band
    with no improvement signal. The threat under F1==0.0 domination is
    smaller here (no retrieval gap) but still a no-go.

Verdict rule (chosen and disclosed below):
  - GO        if (# strictly-partial in mid) >= 0.50 * |mid|
  - NO-GO     if (# strictly-partial in mid) < 0.25 * |mid|
  - BORDERLINE in between (this script will not decide unilaterally;
                the user reviews).

Run: ``.venv/bin/python -m src.diagnostic_chunk5``

Outputs:
  - ``results/diagnostic_chunk5/f1_distribution.png`` (figure)
  - stdout: all counts the chunk-5 brief asks for, plus the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.difficulty import BAND_NAMES, DifficultyTable  # noqa: E402

TABLE_PATH = REPO / "results" / "difficulty_table.json"
FIG_DIR = REPO / "results" / "diagnostic_chunk5"
FIG_PATH = FIG_DIR / "f1_distribution.png"

# Verdict thresholds (fraction of |mid|).
GO_FRACTION = 0.50
NO_GO_FRACTION = 0.25


def _count_classes(scores: list[float]) -> tuple[int, int, int]:
    n_zero = sum(1 for s in scores if s == 0.0)
    n_one = sum(1 for s in scores if s == 1.0)
    n_partial = sum(1 for s in scores if 0.0 < s < 1.0)
    return n_zero, n_one, n_partial


def _tercile_boundaries(table: DifficultyTable) -> tuple[float, float]:
    """Return the two F1 boundary values that split hard/mid/easy."""
    hard_scores = sorted(table.scores[i] for i in table.ids("hard"))
    mid_scores = sorted(table.scores[i] for i in table.ids("mid"))
    easy_scores = sorted(table.scores[i] for i in table.ids("easy"))
    # boundary1 = max(hard), boundary2 = max(mid)
    return max(hard_scores), max(mid_scores)


def _plot(scores: list[float], boundaries: tuple[float, float], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_hist, ax_strip) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Histogram (top panel)
    bins = np.linspace(0.0, 1.0, 21)  # 20 bins of width 0.05
    ax_hist.hist(scores, bins=bins, color="#5B8FF9", edgecolor="black", alpha=0.85)
    ax_hist.axvline(boundaries[0], color="black", linestyle="--", linewidth=1)
    ax_hist.axvline(boundaries[1], color="black", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Base-system F1")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"F1 distribution on D_feedback (n={len(scores)}), with tercile boundaries"
    )
    ax_hist.set_xlim(-0.02, 1.02)

    # Strip plot (bottom panel)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=len(scores))
    ax_strip.scatter(scores, jitter, alpha=0.6, s=20, color="#5B8FF9")
    ax_strip.axvline(boundaries[0], color="black", linestyle="--", linewidth=1)
    ax_strip.axvline(boundaries[1], color="black", linestyle="--", linewidth=1)
    ax_strip.set_xlim(-0.02, 1.02)
    ax_strip.set_ylim(-0.15, 0.15)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Base-system F1")
    ax_strip.set_title("Per-instance scores (jittered)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def main() -> int:
    if not TABLE_PATH.exists():
        print(f"ERROR: {TABLE_PATH} does not exist.")
        print("Run `.venv/bin/python -m src.score_d_feedback` first.")
        return 1

    table = DifficultyTable.load(TABLE_PATH)
    all_scores = list(table.scores)
    mid_scores = [table.scores[i] for i in table.ids("mid")]
    easy_scores = [table.scores[i] for i in table.ids("easy")]
    hard_scores = [table.scores[i] for i in table.ids("hard")]

    # Overall counts
    overall_zero, overall_one, overall_partial = _count_classes(all_scores)

    # Tercile boundaries
    b1, b2 = _tercile_boundaries(table)

    # Mid-tercile content
    mid_zero, mid_one, mid_partial = _count_classes(mid_scores)
    n_mid = len(mid_scores)
    mid_partial_frac = mid_partial / n_mid if n_mid else 0.0

    print(f"=== D_feedback F1 distribution (distractor substrate, n={table.n}) ===")
    print()
    print(f"Overall: F1==0.0: {overall_zero}, F1==1.0: {overall_one}, "
          f"0<F1<1: {overall_partial}  (sum={overall_zero+overall_one+overall_partial})")
    print()
    print(f"Tercile boundaries (by F1 rank):")
    print(f"  hard | mid  boundary: F1 = {b1:.4f}  (max of hard)")
    print(f"  mid  | easy boundary: F1 = {b2:.4f}  (max of mid)")
    print()
    print(f"Band sizes: hard={len(hard_scores)}, mid={n_mid}, easy={len(easy_scores)}")
    print()
    print(f"Middle (frontier) tercile content:")
    print(f"  count F1 == 0.0:           {mid_zero}")
    print(f"  count F1 == 1.0:           {mid_one}")
    print(f"  count 0 < F1 < 1 (strict): {mid_partial}   "
          f"({mid_partial_frac * 100:.1f}% of mid)")
    print()
    print(f"Figure: {FIG_PATH}")
    _plot(all_scores, (b1, b2), FIG_PATH)

    # Verdict
    print()
    print("=== Verdict rule ===")
    print(
        f"GO        if strictly-partial fraction of mid >= {GO_FRACTION * 100:.0f}%\n"
        f"NO-GO     if strictly-partial fraction of mid <  {NO_GO_FRACTION * 100:.0f}%\n"
        f"BORDERLINE in between (script does not decide; operator reviews)"
    )
    print()
    print(f"Observed: {mid_partial}/{n_mid} = {mid_partial_frac * 100:.1f}% strictly partial")
    print()
    if mid_partial_frac >= GO_FRACTION:
        print("VERDICT: GO")
        rc = 0
    elif mid_partial_frac < NO_GO_FRACTION:
        print("VERDICT: NO-GO")
        rc = 4
    else:
        print("VERDICT: BORDERLINE -- operator review")
        rc = 5
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
