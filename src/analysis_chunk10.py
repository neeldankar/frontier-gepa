"""Chunk 10: HotpotQA isolation analysis.

Offline. Reads from results/logs/ (Experiment 1, 70/15/15) AND
results/logs_hotpot_100/ (Experiment 1b, 100/0/0 for the 3 static arms).
Writes to results/analysis_chunk10/. Deterministic: a single
BOOTSTRAP_SEED is used for every bootstrap RNG and printed in summary.json.

Random and vanilla arms are reused from Experiment 1 unchanged. The Chunk-8
regression test `test_random_arm_byte_identical_under_70_15_15_and_100_0_0`
proves the random sampler is bit-identical under both mixes for the same
seed, so the reused random and vanilla cells represent the 100/0/0 regime
exactly as they represented the 70/15/15 regime.

Deliverables:
  - results/analysis_chunk10/endpoint_test_f1_isolation.png   side-by-side
                                                              forest plot
  - results/analysis_chunk10/accept_collapse.png              the headline:
                                                              static_easy
                                                              5.3 -> ~0
  - results/analysis_chunk10/iteration_curves_isolation.png   per-arm
                                                              val-F1
                                                              trajectories
  - results/analysis_chunk10/summary.json                     every number
  - results/analysis_chunk10/FINDINGS_hotpot_100.md           narrative

Run: .venv/bin/python -m src.analysis_chunk10
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOGS_EXP1 = REPO / "results" / "logs"                 # 70/15/15
LOGS_EXP1B = REPO / "results" / "logs_hotpot_100"     # 100/0/0
OUT_DIR = REPO / "results" / "analysis_chunk10"
DIFFICULTY_TABLE_PATH = REPO / "results" / "difficulty_table.json"

ARMS_ALL: tuple[str, ...] = (
    "random",
    "static_easy",
    "static_frontier",
    "static_hard",
    "vanilla_coupled_gepa",
)
ARMS_STATIC = ("static_easy", "static_frontier", "static_hard")
ARMS_REUSED = ("random", "vanilla_coupled_gepa")
SEEDS = (0, 1, 2)
N_ITER = 44

BOOTSTRAP_SEED = 20260602   # same as Chunk 7 for cross-analysis consistency
N_BOOT = 10_000


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


@dataclass
class CellData:
    arm: str
    seed: int
    regime: str  # "70/15/15" or "100/0/0"
    cell_summary: dict
    test_eval: dict
    val_scores_by_candidate: list[float]
    new_program_idx_by_iter: list[int | None]

    @property
    def n_test(self) -> int:
        return int(self.cell_summary["n_test"])

    @property
    def avg_test_f1(self) -> float:
        return float(self.cell_summary["avg_test_f1"])

    @property
    def per_instance_test_f1(self) -> list[float] | None:
        v = self.test_eval.get("per_instance_f1")
        return list(v) if v is not None else None

    @property
    def accepts(self) -> int:
        return len(self.cell_summary["accepted_in_iters"])

    def best_so_far_trajectory(self) -> np.ndarray:
        """Iter 0..44 best-so-far val F1; cumulative max over accepted candidates."""
        traj = np.empty(N_ITER + 1, dtype=float)
        traj[0] = self.val_scores_by_candidate[0]
        current = traj[0]
        for i, new_idx in enumerate(self.new_program_idx_by_iter, start=1):
            if new_idx is not None:
                v = self.val_scores_by_candidate[new_idx]
                if v > current:
                    current = v
            traj[i] = current
        return traj


def _load_cell(logs_root: Path, arm: str, seed: int, regime: str) -> CellData:
    from gepa.core.state import GEPAState

    cell_dir = logs_root / f"{arm}_{seed}"
    cs = json.loads((cell_dir / "cell_summary.json").read_text())
    te = json.loads((cell_dir / "test_eval.json").read_text())
    state = GEPAState.load(str(cell_dir))
    trace = state.full_program_trace
    if len(trace) != N_ITER:
        raise AssertionError(
            f"{arm}_{seed} ({regime}): expected {N_ITER} iters in trace, got {len(trace)}"
        )
    new_idx_by_iter: list[int | None] = []
    for entry in trace:
        v = entry.get("new_program_idx")
        new_idx_by_iter.append(int(v) if v is not None else None)
    return CellData(
        arm=arm, seed=seed, regime=regime,
        cell_summary=cs, test_eval=te,
        val_scores_by_candidate=list(state.program_full_scores_val_set),
        new_program_idx_by_iter=new_idx_by_iter,
    )


def _difficulty_table_hash() -> str:
    return hashlib.sha256(DIFFICULTY_TABLE_PATH.read_bytes()).hexdigest()


def load_all_cells() -> dict[tuple[str, str, int], CellData]:
    """Return cells keyed by (regime, arm, seed).

    regime "70/15/15": all 5 arms x 3 seeds from results/logs/.
    regime "100/0/0":  3 static arms x 3 seeds from results/logs_hotpot_100/.
    """
    cells: dict[tuple[str, str, int], CellData] = {}
    for arm in ARMS_ALL:
        for s in SEEDS:
            cells[("70/15/15", arm, s)] = _load_cell(LOGS_EXP1, arm, s, "70/15/15")
    for arm in ARMS_STATIC:
        for s in SEEDS:
            cd = _load_cell(LOGS_EXP1B, arm, s, "100/0/0")
            # Verify provenance fields the orchestrator stamped in Chunk 9.
            prov = json.loads((LOGS_EXP1B / f"{arm}_{s}" / "provenance.json").read_text())
            assert prov["sampler"] == "100/0/0", f"{arm}_{s}: sampler={prov['sampler']!r}"
            assert prov["sampler_mix"] == [1.0, 0.0, 0.0], prov["sampler_mix"]
            expected_hash = _difficulty_table_hash()
            assert prov["difficulty_table_hash"] == expected_hash, (
                f"{arm}_{s}: table hash drift: "
                f"got {prov['difficulty_table_hash']}, expected {expected_hash}"
            )
            cells[("100/0/0", arm, s)] = cd
    return cells


# ----------------------------------------------------------------------
# Bootstrap helpers (mirror Chunk 7)
# ----------------------------------------------------------------------


def hierarchical_bootstrap_ci(
    per_seed_test_f1s: list[list[float] | float],
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    K = len(per_seed_test_f1s)
    has_per_instance = isinstance(per_seed_test_f1s[0], list)
    if has_per_instance:
        as_arrays = [np.asarray(x, dtype=float) for x in per_seed_test_f1s]
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        seed_picks = rng.integers(0, K, size=K)
        if has_per_instance:
            ms = []
            for sp in seed_picks:
                arr = as_arrays[sp]
                idx = rng.integers(0, len(arr), size=len(arr))
                ms.append(arr[idx].mean())
            means[b] = float(np.mean(ms))
        else:
            picked = np.asarray([per_seed_test_f1s[i] for i in seed_picks], dtype=float)
            means[b] = float(picked.mean())
    point = (
        float(np.mean([a.mean() for a in as_arrays]))
        if has_per_instance
        else float(np.mean(per_seed_test_f1s))
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_mean_diff(
    diffs: list[float], n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    K = len(diffs)
    arr = np.asarray(diffs, dtype=float)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, K, size=K)
        means[b] = arr[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------
# Analysis sections
# ----------------------------------------------------------------------


def _arm_color(arm: str) -> str:
    return {
        "random":                "#888888",
        "static_easy":           "#1f77b4",
        "static_frontier":       "#d62728",
        "static_hard":           "#2ca02c",
        "vanilla_coupled_gepa":  "#9467bd",
    }[arm]


def per_arm_endpoint(cells, regime, arm) -> dict:
    per_seed_inst: list[list[float] | float] = []
    seed_f1s: list[dict] = []
    for s in SEEDS:
        c = cells[(regime, arm, s)]
        seed_f1s.append({"seed": s, "avg_test_f1": c.avg_test_f1})
        pi = c.per_instance_test_f1
        per_seed_inst.append(pi if pi is not None else c.avg_test_f1)
    point, lo, hi = hierarchical_bootstrap_ci(per_seed_inst)
    return {
        "regime": regime,
        "arm": arm,
        "seed_test_f1s": seed_f1s,
        "mean_test_f1": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_accepts": statistics.mean(cells[(regime, arm, s)].accepts for s in SEEDS),
        "per_seed_accepts": [cells[(regime, arm, s)].accepts for s in SEEDS],
    }


# ----------------------------------------------------------------------
# Plotters
# ----------------------------------------------------------------------


def plot_forest(per_arm: dict[tuple[str, str], dict], out_path: Path) -> None:
    """Side-by-side forest plot. Each arm has up to two rows (70/15/15 and
    100/0/0 for the 3 static arms; just one row for random and vanilla)."""
    rows: list[tuple[str, str, dict]] = []  # (label, color, data)
    for arm in ARMS_ALL:
        if arm in ARMS_REUSED:
            d = per_arm[("70/15/15", arm)]
            rows.append((f"{arm}  (reused, 70/15/15)", _arm_color(arm), d))
        else:
            d_old = per_arm[("70/15/15", arm)]
            d_new = per_arm[("100/0/0", arm)]
            rows.append((f"{arm}  70/15/15", _arm_color(arm), d_old))
            rows.append((f"{arm}  100/0/0", _arm_color(arm), d_new))
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ys = list(range(len(rows)))
    for y, (label, color, d) in zip(ys, rows):
        marker = "o" if "100/0/0" not in label else "s"
        ax.errorbar(
            d["mean_test_f1"], y,
            xerr=[[d["mean_test_f1"] - d["ci_lo"]], [d["ci_hi"] - d["mean_test_f1"]]],
            fmt=marker, color=color, markersize=8, capsize=4, elinewidth=2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out test F1 on the 300-example test set")
    ax.set_title(
        "D2 iso - test F1 per arm, 70/15/15 vs 100/0/0 (3 seeds, 95% CI)"
    )
    ax.grid(axis="x", alpha=0.3)
    overall = np.mean([d[2]["mean_test_f1"] for d in [(0, 0, x[2]) for x in rows]])
    ax.axvline(
        overall, color="black", linestyle="--", linewidth=1, alpha=0.5,
        label=f"row mean = {overall:.4f}",
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_accept_collapse(per_arm: dict, out_path: Path) -> None:
    arms = list(ARMS_STATIC)
    x = np.arange(len(arms))
    width = 0.35
    old_means = [per_arm[("70/15/15", a)]["mean_accepts"] for a in arms]
    new_means = [per_arm[("100/0/0", a)]["mean_accepts"] for a in arms]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, old_means, width, label="70/15/15 (Experiment 1)",
                   color="#9999cc", edgecolor="black")
    bars2 = ax.bar(x + width / 2, new_means, width, label="100/0/0 (Experiment 1b)",
                   color="#cc7777", edgecolor="black")
    for bars, vals in [(bars1, old_means), (bars2, new_means)]:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=10,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("Mean accepts per cell over N=44 iterations")
    ax.set_title(
        "Accept-count collapse: removing the 30% off-band leakage zeroes "
        "static_easy's accepts"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_iteration_curves(cells, out_path: Path) -> None:
    iters = np.arange(N_ITER + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    # 70/15/15: solid lines for all 5 arms.
    for arm in ARMS_ALL:
        seed_trajs = np.stack(
            [cells[("70/15/15", arm, s)].best_so_far_trajectory() for s in SEEDS]
        )
        mean = seed_trajs.mean(axis=0)
        lo = seed_trajs.min(axis=0)
        hi = seed_trajs.max(axis=0)
        ax.plot(iters, mean, color=_arm_color(arm), linewidth=2,
                label=f"{arm} 70/15/15")
        ax.fill_between(iters, lo, hi, color=_arm_color(arm), alpha=0.08)
    # 100/0/0: dashed lines for the 3 static arms.
    for arm in ARMS_STATIC:
        seed_trajs = np.stack(
            [cells[("100/0/0", arm, s)].best_so_far_trajectory() for s in SEEDS]
        )
        mean = seed_trajs.mean(axis=0)
        ax.plot(iters, mean, color=_arm_color(arm), linewidth=2,
                linestyle="--", label=f"{arm} 100/0/0")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far val/selection F1 on D_pareto (n=75)")
    ax.set_title(
        "Best-so-far val F1 vs iteration: solid = 70/15/15, dashed = 100/0/0 (static arms)"
    )
    ax.set_xlim(0, N_ITER)
    ax.set_ylim(bottom=0.55)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# FINDINGS narrative
# ----------------------------------------------------------------------


def write_findings(
    per_arm: dict,
    contrasts: list[dict],
    accept_drop: dict,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Chunk 10 - HotpotQA isolation analysis (the leakage was the mechanism)")
    lines.append("")

    se_old = accept_drop["static_easy"]["before"]
    se_new = accept_drop["static_easy"]["after"]
    se_before_per_seed = accept_drop["static_easy"]["before_per_seed"]
    se_after_per_seed = accept_drop["static_easy"]["after_per_seed"]
    lines.append(
        f"**HEADLINE: the off-band leakage was the mechanism behind static_easy's "
        f"Experiment-1 activity.** Under 70/15/15, static_easy accepted "
        f"{se_old:.1f} prompts per cell on average (per-seed {se_before_per_seed}) "
        f"despite supposedly drawing from the \"nothing to correct\" F1==1 band. "
        f"Under 100/0/0, static_easy accepted **{se_new:.1f}** prompts per cell "
        f"(per-seed {se_after_per_seed}). The 30% off-band exposure was the "
        f"entire driver; remove it and the band-arm behaves exactly as its label "
        f"says it should."
    )
    lines.append("")
    lines.append("## D5 iso - accept-count comparison (static arms)")
    lines.append("")
    lines.append("| arm | 70/15/15 mean accepts | 100/0/0 mean accepts | per-seed 70/15/15 | per-seed 100/0/0 |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS_STATIC:
        old = accept_drop[arm]
        lines.append(
            f"| `{arm}` | {old['before']:.1f} | **{old['after']:.1f}** | "
            f"{old['before_per_seed']} | {old['after_per_seed']} |"
        )
    lines.append("")
    static_easy_drop = se_old - se_new
    sf_old = accept_drop["static_frontier"]["before"]
    sf_new = accept_drop["static_frontier"]["after"]
    sh_old = accept_drop["static_hard"]["before"]
    sh_new = accept_drop["static_hard"]["after"]
    lines.append(
        f"The static_easy collapse ({se_old:.1f} → {se_new:.1f}, a drop of "
        f"{static_easy_drop:.1f} accepts/cell, with per-seed counts going from "
        f"{accept_drop['static_easy']['before_per_seed']} to "
        f"{accept_drop['static_easy']['after_per_seed']}) is the direct confirmation that "
        "the 30% off-band exposure was the entire driver of static_easy's Experiment-1 "
        "activity."
    )
    lines.append("")
    lines.append(
        f"The pattern for the other static arms is not symmetric and is worth flagging "
        f"honestly: static_frontier's mean accepts stayed essentially flat "
        f"({sf_old:.1f} → {sf_new:.1f}), while **static_hard's accepts more than doubled** "
        f"({sh_old:.1f} → {sh_new:.1f}). Under 70/15/15, static_hard's minibatch averaged ~2.1 "
        "hard + 0.45 mid + 0.45 easy; under 100/0/0 it is 3 hard. More hard examples per "
        "minibatch evidently give the reflection LM more material to propose actionable edits "
        "against, even though hard examples are F1==0 (complete failures). That contradicts the "
        "spec's a-priori framing of static_hard as the reflection-starved arm. It is a side "
        "finding of this isolation chunk, not a frontier-band claim."
    )
    lines.append("")

    lines.append("## D2 iso - endpoint test F1 by arm, 70/15/15 vs 100/0/0")
    lines.append("")
    lines.append("| arm | regime | mean test F1 | 95% CI |")
    lines.append("|---|---|---|---|")
    for arm in ARMS_ALL:
        if arm in ARMS_REUSED:
            d = per_arm[("70/15/15", arm)]
            lines.append(
                f"| `{arm}` | reused (70/15/15) | {d['mean_test_f1']:.4f} | "
                f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] |"
            )
        else:
            d_old = per_arm[("70/15/15", arm)]
            d_new = per_arm[("100/0/0", arm)]
            lines.append(
                f"| `{arm}` | 70/15/15 | {d_old['mean_test_f1']:.4f} | "
                f"[{d_old['ci_lo']:.4f}, {d_old['ci_hi']:.4f}] |"
            )
            lines.append(
                f"| `{arm}` | **100/0/0** | **{d_new['mean_test_f1']:.4f}** | "
                f"[{d_new['ci_lo']:.4f}, {d_new['ci_hi']:.4f}] |"
            )
    lines.append("")

    # Paired contrasts (100/0/0 frontier vs everything else)
    lines.append(
        "## D3 iso - paired contrasts under 100/0/0 (seed-matched; paired bootstrap, "
        "10000 resamples, 95% percentile)"
    )
    lines.append("")
    any_significant = any(not c["ci_includes_zero"] for c in contrasts)
    for c in contrasts:
        sig = "**CI excludes 0**" if not c["ci_includes_zero"] else "CI includes 0"
        lines.append(
            f"- **{c['left']} − {c['right']}** ({c['regime_left']} vs "
            f"{c['regime_right']}): {c['mean_diff']:+.4f} "
            f"[95% CI {c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {sig}"
        )
    lines.append("")

    # Did removing leakage sharpen, flip, or leave unchanged?
    arm_means_old = {arm: per_arm[("70/15/15", arm)]["mean_test_f1"] for arm in ARMS_ALL}
    arm_means_new = {arm: per_arm[("70/15/15", arm)]["mean_test_f1"] for arm in ARMS_REUSED}
    for arm in ARMS_STATIC:
        arm_means_new[arm] = per_arm[("100/0/0", arm)]["mean_test_f1"]
    spread_old = max(arm_means_old.values()) - min(arm_means_old.values())
    spread_new = max(arm_means_new.values()) - min(arm_means_new.values())
    # Within-arm seed spreads in each regime
    within_arm_spreads_new = []
    for arm in ARMS_STATIC:
        seeds_f1 = [per_arm[("100/0/0", arm)]["seed_test_f1s"][i]["avg_test_f1"] for i in range(3)]
        within_arm_spreads_new.append(max(seeds_f1) - min(seeds_f1))
    for arm in ARMS_REUSED:
        seeds_f1 = [per_arm[("70/15/15", arm)]["seed_test_f1s"][i]["avg_test_f1"] for i in range(3)]
        within_arm_spreads_new.append(max(seeds_f1) - min(seeds_f1))
    largest_within_new = max(within_arm_spreads_new)
    lines.append(
        f"Between-arm range collapsed from {spread_old:.4f} F1 points "
        f"(70/15/15 across all 5 arms) to **{spread_new:.4f}** F1 points "
        f"(100/0/0 for the 3 static arms + reused random and vanilla baselines). "
        f"Largest within-arm seed spread in the new mix-of-regimes layout is "
        f"{largest_within_new:.4f}, which is {largest_within_new / max(spread_new, 1e-9):.1f}x the "
        f"between-arm range. The cross-arm contrast did not sharpen into anything "
        f"that beats seed noise; the cleaner null is still a null."
    )
    lines.append("")

    if any_significant:
        sig_contrasts = [c for c in contrasts if not c["ci_includes_zero"]]
        names = [f"{c['left']}−{c['right']}" for c in sig_contrasts]
        lines.append(
            f"({len(sig_contrasts)} of {len(contrasts)} paired contrasts have a 95% CI "
            f"that excludes zero: {', '.join(names)}. Magnitudes are similar to the "
            f"within-arm seed spread, so this is the same flavor of marginally-detectable "
            f"signal seen in Experiment 1, not the headline-grade frontier effect the "
            f"hypothesis predicted.)"
        )
        lines.append("")

    lines.append("## What this isolation chunk demonstrates")
    lines.append("")
    lines.append(
        "1. **The off-band-leakage mechanism is real and measurable.** The static_easy "
        f"accept count goes from {se_old:.1f}/cell (with 30% off-band leakage) to "
        f"{se_new:.1f}/cell (with the leakage removed). That is the direct mechanism test "
        "Experiment 1's Chunk-7 analysis hypothesized and Chunk 10 confirms."
    )
    lines.append("")
    lines.append(
        "2. **Removing the leakage does NOT manufacture a frontier effect on held-out F1.** "
        "The cross-arm endpoint contrast under 100/0/0 remains within seed noise; the "
        "frontier-band hypothesis is still not supported by these data at this scale, even "
        "with the cleaner sampling regime. That is a stronger null than Experiment 1 alone "
        "produced, because the obvious confound (leakage) is now ruled out."
    )
    lines.append("")
    lines.append(
        "3. **Read together with Experiment 1, the result narrows the design space.** "
        "Whatever explains the null, it is not the leakage. The remaining candidate from the "
        "Chunk-7 diagnosis is mechanism 1: bounded bidirectional formatting gains on a "
        "frontier dominated by verbosity / specificity mismatches. Verifying that mechanism "
        "needs a different substrate where the frontier is populated and the feedback is "
        "actionable — exactly the IFBench probe BUILD_PLAN §7 Chunks 11-15 propose."
    )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"FINDINGS -> {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"bootstrap_seed: {BOOTSTRAP_SEED}")
    print(f"difficulty_table_hash: {_difficulty_table_hash()}")

    cells = load_all_cells()
    print(f"loaded {len(cells)} cells "
          f"({len(ARMS_ALL) * len(SEEDS)} from 70/15/15 + "
          f"{len(ARMS_STATIC) * len(SEEDS)} from 100/0/0)")

    # D2 iso: per-arm endpoint test F1 with bootstrap CIs.
    per_arm: dict[tuple[str, str], dict] = {}
    for arm in ARMS_ALL:
        per_arm[("70/15/15", arm)] = per_arm_endpoint(cells, "70/15/15", arm)
    for arm in ARMS_STATIC:
        per_arm[("100/0/0", arm)] = per_arm_endpoint(cells, "100/0/0", arm)

    # D5 iso: accept-count comparison (static arms).
    accept_drop: dict[str, dict] = {}
    for arm in ARMS_STATIC:
        accept_drop[arm] = {
            "before": per_arm[("70/15/15", arm)]["mean_accepts"],
            "after":  per_arm[("100/0/0", arm)]["mean_accepts"],
            "before_per_seed": per_arm[("70/15/15", arm)]["per_seed_accepts"],
            "after_per_seed":  per_arm[("100/0/0", arm)]["per_seed_accepts"],
        }
    print()
    for arm in ARMS_STATIC:
        d = accept_drop[arm]
        print(f"  {arm}: {d['before']:.1f}/cell (70/15/15) -> {d['after']:.1f}/cell (100/0/0)")

    # D3 iso: paired contrasts. Frontier under 100/0/0 vs each other arm.
    # For the static arms, use 100/0/0; for random and vanilla, reuse 70/15/15.
    contrasts: list[dict] = []
    contrast_pairs = (
        (("static_frontier", "100/0/0"), ("static_hard",            "100/0/0")),
        (("static_frontier", "100/0/0"), ("random",                 "70/15/15")),
        (("static_frontier", "100/0/0"), ("static_easy",            "100/0/0")),
        (("static_frontier", "100/0/0"), ("vanilla_coupled_gepa",   "70/15/15")),
    )
    for (left_arm, left_regime), (right_arm, right_regime) in contrast_pairs:
        diffs = [
            cells[(left_regime, left_arm, s)].avg_test_f1
            - cells[(right_regime, right_arm, s)].avg_test_f1
            for s in SEEDS
        ]
        mean_diff, lo, hi = paired_bootstrap_mean_diff(diffs)
        ci_zero = lo <= 0.0 <= hi
        rec = {
            "left": left_arm, "regime_left": left_regime,
            "right": right_arm, "regime_right": right_regime,
            "per_seed_diffs": diffs,
            "mean_diff": mean_diff,
            "ci_lo": lo, "ci_hi": hi,
            "ci_includes_zero": ci_zero,
        }
        contrasts.append(rec)
        print(
            f"  paired: {left_arm} ({left_regime}) - {right_arm} ({right_regime}): "
            f"{mean_diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  includes 0? {ci_zero}"
        )

    # Figures
    plot_forest(per_arm, OUT_DIR / "endpoint_test_f1_isolation.png")
    print(f"-> {OUT_DIR / 'endpoint_test_f1_isolation.png'}")
    plot_accept_collapse(per_arm, OUT_DIR / "accept_collapse.png")
    print(f"-> {OUT_DIR / 'accept_collapse.png'}")
    plot_iteration_curves(cells, OUT_DIR / "iteration_curves_isolation.png")
    print(f"-> {OUT_DIR / 'iteration_curves_isolation.png'}")

    # summary.json: every number
    summary = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOT,
        "difficulty_table_hash": _difficulty_table_hash(),
        "logs_paths": {
            "70/15/15": str(LOGS_EXP1.relative_to(REPO)),
            "100/0/0": str(LOGS_EXP1B.relative_to(REPO)),
        },
        "per_seed_test_f1": {
            f"{regime}::{arm}": {s: cells[(regime, arm, s)].avg_test_f1 for s in SEEDS}
            for (regime, arm) in [
                *[("70/15/15", a) for a in ARMS_ALL],
                *[("100/0/0", a) for a in ARMS_STATIC],
            ]
        },
        "per_seed_accepts": {
            f"{regime}::{arm}": {s: cells[(regime, arm, s)].accepts for s in SEEDS}
            for (regime, arm) in [
                *[("70/15/15", a) for a in ARMS_ALL],
                *[("100/0/0", a) for a in ARMS_STATIC],
            ]
        },
        "endpoint_test_f1": {f"{r}::{a}": v for (r, a), v in per_arm.items()},
        "accept_drop": accept_drop,
        "contrasts": contrasts,
        "n_iter_per_cell": N_ITER,
        "n_test_per_cell": 300,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {OUT_DIR / 'summary.json'}")

    write_findings(per_arm, contrasts, accept_drop, OUT_DIR / "FINDINGS_hotpot_100.md")
    print("Chunk 10 analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
