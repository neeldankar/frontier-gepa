"""Chunk 13 diagnostic + GO/NO-GO verdict for IFBench.

Reads ``results/ifbench/difficulty_table.json`` and
``results/ifbench/continuity_gate.json`` (both written by
``src.score_ifbench_d_feedback``), prints the per-band composition and
the gate diagnostics verbatim, plots the per-id score distribution with
the chosen band boundaries drawn, and writes a machine-readable summary.

The binning rule is selected by the continuity gate in
``src/difficulty.py`` per BUILD_PLAN §4 D3. This script does not
re-decide the binning -- it reports what the gate chose and produces the
operator-facing figure and summary.

Run: ``.venv/bin/python -m src.diagnostic_chunk13``

Outputs:
  - ``results/ifbench/diagnostic_chunk13/histogram.png``
  - ``results/ifbench/diagnostic_chunk13/summary.json``
  - stdout: gate diagnostics, per-band composition, verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.difficulty import (  # noqa: E402
    DifficultyTable,
    difficulty_table_sha256,
)

IFBENCH_DIR = REPO / "results" / "ifbench"
TABLE_PATH = IFBENCH_DIR / "difficulty_table.json"
GATE_PATH = IFBENCH_DIR / "continuity_gate.json"
DIAG_DIR = IFBENCH_DIR / "diagnostic_chunk13"
FIG_PATH = DIAG_DIR / "histogram.png"
SUMMARY_PATH = DIAG_DIR / "summary.json"


def _count_classes(scores: list[float]) -> tuple[int, int, int]:
    n_zero = sum(1 for s in scores if s == 0.0)
    n_one = sum(1 for s in scores if s == 1.0)
    n_partial = sum(1 for s in scores if 0.0 < s < 1.0)
    return n_zero, n_one, n_partial


def _chosen_boundaries(table: DifficultyTable, chosen: str) -> tuple[float, float]:
    """Return the two score values that separate hard|mid|easy under the
    chosen binning. For value bins these are 0.0 and 1.0 (exact). For
    rank terciles these are the max-of-hard and the max-of-mid score
    values, which are the data-driven cutoffs."""
    if chosen == "value_bins":
        return 0.0, 1.0
    hard_scores = [table.scores[i] for i in table.ids("hard")]
    mid_scores = [table.scores[i] for i in table.ids("mid")]
    b1 = max(hard_scores) if hard_scores else 0.0
    b2 = max(mid_scores) if mid_scores else b1
    return b1, b2


def _plot(
    scores: list[float],
    boundaries: tuple[float, float],
    chosen: str,
    fig_path: Path,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_hist, ax_strip) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    b1, b2 = boundaries
    label = (
        "value-bin boundaries (0.0, 1.0)"
        if chosen == "value_bins"
        else f"rank-tercile boundaries ({b1:.4f}, {b2:.4f})"
    )
    for ax in (ax_hist, ax_strip):
        ax.axvline(b1, color="black", linestyle="--", linewidth=1)
        ax.axvline(b2, color="black", linestyle="--", linewidth=1)

    bins = np.linspace(0.0, 1.0, 21)
    ax_hist.hist(scores, bins=bins, color="#5B8FF9", edgecolor="black", alpha=0.85)
    ax_hist.set_xlabel("Base-system score (fraction of constraints satisfied)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"IFBench score distribution on D_feedback (n={len(scores)})\n"
        f"chosen binning: {chosen}; {label}"
    )
    ax_hist.set_xlim(-0.02, 1.02)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=len(scores))
    ax_strip.scatter(scores, jitter, alpha=0.6, s=20, color="#5B8FF9")
    ax_strip.set_xlim(-0.02, 1.02)
    ax_strip.set_ylim(-0.15, 0.15)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Base-system score")
    ax_strip.set_title("Per-instance scores (jittered)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def main() -> int:
    if not TABLE_PATH.exists():
        print(f"ERROR: {TABLE_PATH} does not exist.")
        print("Run `.venv/bin/python -m src.score_ifbench_d_feedback` first.")
        return 1
    if not GATE_PATH.exists():
        print(f"ERROR: {GATE_PATH} does not exist.")
        print("Run the scorer; it writes both the table and the gate.")
        return 1

    table = DifficultyTable.load(TABLE_PATH)
    gate = json.loads(GATE_PATH.read_text())
    table_hash = difficulty_table_sha256(TABLE_PATH)
    chosen = gate["chosen_binning"]

    all_scores = list(table.scores)
    hard_scores = [table.scores[i] for i in table.ids("hard")]
    mid_scores = [table.scores[i] for i in table.ids("mid")]
    easy_scores = [table.scores[i] for i in table.ids("easy")]

    overall_zero, overall_one, overall_partial = _count_classes(all_scores)
    hard_zero, hard_one, hard_partial = _count_classes(hard_scores)
    mid_zero, mid_one, mid_partial = _count_classes(mid_scores)
    easy_zero, easy_one, easy_partial = _count_classes(easy_scores)

    b1, b2 = _chosen_boundaries(table, chosen)

    print(f"=== IFBench D_feedback score distribution (n={table.n}) ===")
    print()
    print(
        f"Overall: score==0.0: {overall_zero}, score==1.0: {overall_one}, "
        f"0<score<1: {overall_partial}  "
        f"(sum={overall_zero + overall_one + overall_partial})"
    )
    print()
    print("== Continuity gate (BUILD_PLAN §4 D3) ==")
    print(f"  middle-tercile partial fraction:  {gate['middle_tercile_partial_frac']:.4f}  "
          f"(threshold >= {gate['middle_tercile_partial_threshold']:.2f}; "
          f"ok={gate['middle_tercile_partial_ok']})")
    print(f"  extreme-mass fraction:            {gate['extreme_mass_frac']:.4f}  "
          f"(threshold <  {gate['extreme_mass_threshold']:.2f}; "
          f"ok={gate['extreme_mass_ok']})")
    print(f"  rank-terciles stand:              {gate['rank_terciles_stand']}")
    print(f"  -> chosen binning:                {chosen}")
    print()
    print("== Frontier sizes under each binning ==")
    print(f"  rank-tercile frontier:  {gate['frontier_size_under_rank_terciles']}")
    print(f"  value-bin frontier:     {gate['frontier_size_under_value_bins']}")
    print(f"  chosen frontier:        {gate['frontier_size_under_chosen']}  "
          f"(min for GO = {gate['frontier_min_for_go']})")
    print()
    print(f"Band boundaries (chosen={chosen}): {b1:.4f}, {b2:.4f}")
    print()
    print("== Band sizes (chosen binning) ==")
    print(f"  hard: {len(hard_scores)}   (composition: zero={hard_zero}, partial={hard_partial}, one={hard_one})")
    print(f"  mid : {len(mid_scores)}   (composition: zero={mid_zero}, partial={mid_partial}, one={mid_one})")
    print(f"  easy: {len(easy_scores)}   (composition: zero={easy_zero}, partial={easy_partial}, one={easy_one})")
    print()
    print(f"Figure: {FIG_PATH}")
    _plot(all_scores, (b1, b2), chosen, FIG_PATH)

    print()
    print(f"Difficulty table SHA-256: {table_hash}")
    print()
    print(f"=== VERDICT: {gate['verdict']} ===")
    if gate["verdict"] == "GO":
        print(f"  frontier has {gate['frontier_size_under_chosen']} instances "
              f"(>= {gate['frontier_min_for_go']}); table frozen, ready for Chunk 14.")
        rc = 0
    else:
        print(f"  frontier has {gate['frontier_size_under_chosen']} instances "
              f"(<  {gate['frontier_min_for_go']}); STOP for operator review per D3.")
        rc = 4

    summary = {
        "n": table.n,
        "chosen_binning": chosen,
        "verdict": gate["verdict"],
        "difficulty_table_sha256": table_hash,
        "boundaries": {"b1": b1, "b2": b2},
        "band_sizes": {
            "easy": len(easy_scores),
            "mid": len(mid_scores),
            "hard": len(hard_scores),
        },
        "band_composition": {
            "easy": {"zero": easy_zero, "partial": easy_partial, "one": easy_one},
            "mid": {"zero": mid_zero, "partial": mid_partial, "one": mid_one},
            "hard": {"zero": hard_zero, "partial": hard_partial, "one": hard_one},
        },
        "overall_composition": {
            "zero": overall_zero,
            "partial": overall_partial,
            "one": overall_one,
        },
        "gate": gate,
    }
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {SUMMARY_PATH}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
