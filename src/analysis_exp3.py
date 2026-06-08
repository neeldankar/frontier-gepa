"""Exp-3 analysis: aggregate the 12 cell summaries into the primary figure,
the primary/secondary tables, and RESULTS.md.

RESULTS.md states only what the curves and tables show. No interpretation, no
comparison against the pre-registered predictions -- that lives in the writeup.

Outputs (results/exp3_curriculum/):
  - cumulative_accepts.png   primary figure (full + phase-1 zoom)
  - analysis_summary.json    aggregated numbers
  - RESULTS.md               tables + factual figure description

Run: .venv/bin/python -m src.analysis_exp3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "results" / "exp3_curriculum"

ARMS = ("random", "easy_to_hard", "hard_to_easy", "static_medium")
SEEDS = (0, 1, 2)
T = 40
PHASE1_LAST = 13  # iters 0-13 phase 1, 14-26 phase 2, 27-39 phase 3
PHASE_BOUNDS = (13.5, 26.5)
COLORS = {
    "random": "#777777",
    "easy_to_hard": "#1f77b4",
    "hard_to_easy": "#d62728",
    "static_medium": "#2ca02c",
}


def _load_cells() -> dict[str, dict[int, dict]]:
    cells: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            p = OUT / arm / f"seed{seed}.json"
            cells[arm][seed] = json.loads(p.read_text())
    return cells


def _cumulative(records: list[dict]) -> list[int]:
    """Cumulative accepts at each iteration 0..T-1 (records are iter-ordered)."""
    by_iter = {r["iter"]: bool(r["accepted"]) for r in records}
    out, running = [], 0
    for i in range(T):
        running += 1 if by_iter.get(i, False) else 0
        out.append(running)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def make_figure(cells, path: Path) -> None:
    fig, (axf, axz) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)

    for ax, (xlo, xhi, title) in zip(
        (axf, axz),
        ((-0.5, T - 0.5, "Cumulative accepts vs iteration (full run, T=40)"),
         (-0.3, PHASE1_LAST + 0.3, "Phase 1 zoom (iters 0-13: easy_to_hard's easy phase)")),
    ):
        for arm in ARMS:
            curves = [_cumulative(cells[arm][s]["iterations"]) for s in SEEDS]
            xs = list(range(T))
            # per-seed thin lines
            for c in curves:
                ax.step(xs, c, where="post", color=COLORS[arm], alpha=0.30, linewidth=1.0)
            # mean bold line
            mean_curve = [_mean([c[i] for c in curves]) for i in range(T)]
            ax.step(xs, mean_curve, where="post", color=COLORS[arm], linewidth=2.6,
                    label=arm)
        for b in PHASE_BOUNDS:
            if xlo <= b <= xhi:
                ax.axvline(b, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cumulative accepts")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)

    axf.legend(title="arm (bold=mean of 3 seeds; thin=per-seed)", fontsize=8, loc="upper left")
    fig.suptitle("Exp-3 curriculum: cumulative accepts over iterations (12 cells)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def build() -> dict:
    cells = _load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure(cells, OUT / "cumulative_accepts.png")

    agg: dict = {"arms": {}}
    for arm in ARMS:
        first = {s: cells[arm][s]["first_accept_iter"] for s in SEEDS}
        total = {s: cells[arm][s]["cumulative_accepts"] for s in SEEDS}
        f1 = {s: cells[arm][s]["final_test_f1"] for s in SEEDS}
        cum13 = {s: _cumulative(cells[arm][s]["iterations"])[PHASE1_LAST] for s in SEEDS}
        agg["arms"][arm] = {
            "first_accept_iter": {"per_seed": first, "mean": _mean(list(first.values()))},
            "total_accepts": {"per_seed": total, "mean": _mean(list(total.values()))},
            "final_test_f1": {
                "per_seed": f1,
                "mean": _mean(list(f1.values())),
                "range": max(f1.values()) - min(f1.values()),
            },
            "cumulative_accepts_through_iter13": {
                "per_seed": cum13, "mean": _mean(list(cum13.values()))
            },
        }
    (OUT / "analysis_summary.json").write_text(json.dumps(agg, indent=2))

    # ---- Tables ----
    def seed_cols(d, fmt):
        return [fmt.format(d[s]) for s in SEEDS]

    first_rows, total_rows, f1_rows, cum13_rows = [], [], [], []
    for arm in ARMS:
        a = agg["arms"][arm]
        first_rows.append([arm, *seed_cols(a["first_accept_iter"]["per_seed"], "{}"),
                           f"{a['first_accept_iter']['mean']:.2f}"])
        total_rows.append([arm, *seed_cols(a["total_accepts"]["per_seed"], "{}"),
                           f"{a['total_accepts']['mean']:.2f}"])
        f1_rows.append([arm, *seed_cols(a["final_test_f1"]["per_seed"], "{:.4f}"),
                        f"{a['final_test_f1']['mean']:.4f}",
                        f"{a['final_test_f1']['range']:.4f}"])
        cum13_rows.append([arm, *seed_cols(a["cumulative_accepts_through_iter13"]["per_seed"], "{}"),
                           f"{a['cumulative_accepts_through_iter13']['mean']:.2f}"])

    t_first = _md_table(["arm", "seed0", "seed1", "seed2", "mean"], first_rows)
    t_total = _md_table(["arm", "seed0", "seed1", "seed2", "mean"], total_rows)
    t_cum13 = _md_table(["arm", "seed0", "seed1", "seed2", "mean"], cum13_rows)
    t_f1 = _md_table(["arm", "seed0", "seed1", "seed2", "mean", "range(max-min)"], f1_rows)

    # Factual figure description (numbers only).
    e2h = agg["arms"]["easy_to_hard"]
    e2h_phase1 = e2h["cumulative_accepts_through_iter13"]["per_seed"]
    e2h_first = e2h["first_accept_iter"]["per_seed"]

    md = f"""# Exp-3 Curriculum: Results

Aggregated from the 12 cell summaries in `results/exp3_curriculum/<arm>/seed<seed>.json`
(4 arms x seeds {{0,1,2}}, T=40). This file states only what the figure and tables
show. Comparison against the pre-registered predictions is in the writeup, separately.

Primary endpoint: accept dynamics (P1 iteration-to-first-accept, P2 cumulative accepts).
Secondary: held-out test F1.

## Primary figure

`cumulative_accepts.png` -- cumulative accepts vs iteration, one bold line per arm
(mean of 3 seeds) with the three per-seed curves shown as thin lines of the same
color. Dotted verticals mark the phase boundaries (iters 0-13 / 14-26 / 27-39). The
right panel zooms to phase 1 (iters 0-13).

Reading the curves at the end of phase 1 (iteration 13), cumulative accepts:

{t_cum13}

`easy_to_hard` is at 0 cumulative accepts through iteration 13 for all three seeds
(per-seed {list(e2h_phase1.values())}); its first accepted iteration per seed is
{list(e2h_first.values())}.

## Primary table 1: iteration-to-first-accept

{t_first}

## Primary table 2: total accepts (at iteration 40)

{t_total}

## Secondary table: final held-out test F1 (300-example test split)

Labeled secondary; per the pre-registration this endpoint is reported, not treated as
discriminating. Per-seed values and the within-arm range are shown because the spread
is the relevant context.

{t_f1}
"""
    (OUT / "RESULTS.md").write_text(md)
    return agg


def main() -> int:
    agg = build()
    print("Wrote:")
    print(f"  {OUT/'cumulative_accepts.png'}")
    print(f"  {OUT/'analysis_summary.json'}")
    print(f"  {OUT/'RESULTS.md'}")
    print("\nHeadline numbers (factual):")
    for arm in ARMS:
        a = agg["arms"][arm]
        print(f"  {arm:<14} first_accept mean={a['first_accept_iter']['mean']:.2f} "
              f"per_seed={list(a['first_accept_iter']['per_seed'].values())} | "
              f"total_accepts mean={a['total_accepts']['mean']:.2f} "
              f"{list(a['total_accepts']['per_seed'].values())} | "
              f"cum@13 mean={a['cumulative_accepts_through_iter13']['mean']:.2f} "
              f"{list(a['cumulative_accepts_through_iter13']['per_seed'].values())} | "
              f"testF1 mean={a['final_test_f1']['mean']:.4f} "
              f"range={a['final_test_f1']['range']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
