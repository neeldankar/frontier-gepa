"""Chunk 5 diagnostic and GO/NO-GO verdict (distractor substrate, value bins).

Reads ``results/difficulty_table.json`` (written by ``src.score_d_feedback``,
binned by F1 value per DEVIATIONS.md entry 4), plots the per-id F1
distribution, and prints the GO/NO-GO verdict.

Under value-based binning the frontier (mid) band is by construction the
set of strictly-partial instances; "% partial in mid" is now 100% by
definition and uninformative. The gate is instead the COUNT of frontier
instances: a frontier band thinner than ~b=3 minibatches over N=44
iterations (=132 trajectory-instance touches) would force each instance
to be revisited many times and starve the diversity the reflection
proposer needs.

Verdict rule:
  - GO     if |mid| >= 20  (frontier band has enough distinct instances
                            to sample b=3 minibatches over N=44 without
                            extreme repetition)
  - NO-GO  if |mid| <  5   (frontier band is too thin to support the
                            band-sampling design)
  - BORDERLINE in between (script does not decide; operator reviews).

Run: ``.venv/bin/python -m src.diagnostic_chunk5``

Outputs:
  - ``results/diagnostic_chunk5/f1_distribution.png`` (figure)
  - stdout: all counts plus the verdict.
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

# Verdict thresholds (count of frontier instances under value-based binning).
GO_MIN_FRONTIER_COUNT = 20
NO_GO_MAX_FRONTIER_COUNT = 5


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


def _plot(scores: list[float], _unused_boundaries, fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_hist, ax_strip) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Value-bin separators: F1=0 (between hard and mid) and F1=1 (between
    # mid and easy). The hard band sits AT 0, the easy band sits AT 1, and
    # the frontier (mid) sits strictly between.
    for ax in (ax_hist, ax_strip):
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)

    bins = np.linspace(0.0, 1.0, 21)  # 20 bins of width 0.05
    ax_hist.hist(scores, bins=bins, color="#5B8FF9", edgecolor="black", alpha=0.85)
    ax_hist.set_xlabel("Base-system F1")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"F1 distribution on D_feedback (n={len(scores)}), value-based band boundaries"
    )
    ax_hist.set_xlim(-0.02, 1.02)

    # Strip plot (bottom panel)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=len(scores))
    ax_strip.scatter(scores, jitter, alpha=0.6, s=20, color="#5B8FF9")
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

    # Mid-band content (value-based binning: mid is by construction the
    # strictly-partial set, so all three classes are now 0 / 0 / |mid|).
    mid_zero, mid_one, mid_partial = _count_classes(mid_scores)
    n_mid = len(mid_scores)

    print(f"=== D_feedback F1 distribution (distractor substrate, n={table.n}) ===")
    print()
    print(f"Overall: F1==0.0: {overall_zero}, F1==1.0: {overall_one}, "
          f"0<F1<1: {overall_partial}  (sum={overall_zero+overall_one+overall_partial})")
    print()
    print(f"Band boundaries (value-based bins):")
    print(f"  hard | mid: F1 > 0     (max of hard = {b1:.4f})")
    print(f"  mid  | easy: F1 < 1    (max of mid  = {b2:.4f})")
    print()
    print(f"Band sizes: hard={len(hard_scores)}, mid={n_mid}, easy={len(easy_scores)}")
    print()
    print(f"Middle (frontier) band content (value-based bin = 0 < F1 < 1):")
    print(f"  count F1 == 0.0:           {mid_zero}   (must be 0 under value bins)")
    print(f"  count F1 == 1.0:           {mid_one}   (must be 0 under value bins)")
    print(f"  count 0 < F1 < 1 (strict): {mid_partial}   (== |mid|)")
    print()
    print(f"Figure: {FIG_PATH}")
    _plot(all_scores, (b1, b2), FIG_PATH)

    # Verdict
    print()
    print("=== Verdict rule (value-based binning) ===")
    print(
        f"GO     if |mid| >= {GO_MIN_FRONTIER_COUNT}  (enough frontier instances for b=3 over N=44)\n"
        f"NO-GO  if |mid| <  {NO_GO_MAX_FRONTIER_COUNT}\n"
        f"BORDERLINE in between (script does not decide; operator reviews)"
    )
    print()
    print(f"Observed |mid| = {n_mid}")
    print()
    if n_mid >= GO_MIN_FRONTIER_COUNT:
        print("VERDICT: GO")
        rc = 0
    elif n_mid < NO_GO_MAX_FRONTIER_COUNT:
        print("VERDICT: NO-GO")
        rc = 4
    else:
        print("VERDICT: BORDERLINE -- operator review")
        rc = 5
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
