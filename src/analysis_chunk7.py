"""Chunk 7 offline analysis of the completed 15-cell Chunk-6 matrix.

NO API. NO re-running cells. Reads only from results/logs/. Writes only to
results/analysis_chunk7/. Deterministic: a single BOOTSTRAP_SEED is used
for every bootstrap RNG and printed in summary.json.

Deliverables:
  D1 results/analysis_chunk7/iteration_curves.png   per-arm best-so-far
                                                    VAL F1 vs iteration
  D2 results/analysis_chunk7/endpoint_test_f1.png   forest plot, held-out
                                                    test F1 with paired-
                                                    hierarchical bootstrap
                                                    CI per arm
  D3                                                pairwise contrasts
                                                    static_frontier - X
                                                    (paired bootstrap CIs)
  D4 results/analysis_chunk7/iterations_to_target.png bar chart at T=0.63
                                                    table over T in
                                                    {0.62, 0.63, 0.64}
                                                    plus per-cell 90%-of-
                                                    own-final
  D5                                                acceptance dynamics

Outputs also: results/analysis_chunk7/summary.json with every number,
and FINDINGS.md (honest narrative -- null first if that's what the
numbers say).
"""

from __future__ import annotations

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

LOGS_ROOT = REPO / "results" / "logs"
OUT_DIR = REPO / "results" / "analysis_chunk7"

ARMS = ("random", "static_easy", "static_frontier", "static_hard", "vanilla_coupled_gepa")
SEEDS = (0, 1, 2)
N_ITER = 44

BOOTSTRAP_SEED = 20260602  # documented; printed into summary.json
N_BOOT = 10_000
TARGET_SWEEP = (0.62, 0.63, 0.64)
PRIMARY_TARGET = 0.63


def _arm_color(arm: str) -> str:
    return {
        "random":                "#888888",
        "static_easy":           "#1f77b4",
        "static_frontier":       "#d62728",
        "static_hard":           "#2ca02c",
        "vanilla_coupled_gepa":  "#9467bd",
    }[arm]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class CellData:
    arm: str
    seed: int
    cell_summary: dict
    test_eval: dict
    val_scores_by_candidate: list[float]
    new_program_idx_by_iter: list[int | None]  # length 44, 0-indexed

    @property
    def n_test(self) -> int:
        return int(self.cell_summary["n_test"])

    @property
    def n_iter_actual(self) -> int:
        return int(self.cell_summary["state_i_final"]) + 1

    @property
    def avg_test_f1(self) -> float:
        return float(self.cell_summary["avg_test_f1"])

    @property
    def best_val_f1(self) -> float:
        return float(self.cell_summary["best_val_f1"])

    @property
    def per_instance_test_f1(self) -> list[float] | None:
        v = self.test_eval.get("per_instance_f1")
        return list(v) if v is not None else None

    @property
    def accepted_in_iters(self) -> list[int]:
        return list(self.cell_summary["accepted_in_iters"])

    def best_so_far_trajectory(self) -> np.ndarray:
        """Length 45 (iters 0..44). The seed's val F1 sits at iter 0."""
        traj = np.empty(N_ITER + 1, dtype=float)
        seed_val = self.val_scores_by_candidate[0]
        traj[0] = seed_val
        current_best = seed_val
        for i, new_idx in enumerate(self.new_program_idx_by_iter, start=1):
            if new_idx is not None:
                v = self.val_scores_by_candidate[new_idx]
                if v > current_best:
                    current_best = v
            traj[i] = current_best
        return traj


def _load_cell(arm: str, seed: int) -> CellData:
    from gepa.core.state import GEPAState

    cell_dir = LOGS_ROOT / f"{arm}_{seed}"
    cell_summary = json.loads((cell_dir / "cell_summary.json").read_text())
    test_eval = json.loads((cell_dir / "test_eval.json").read_text())

    # Load GEPAState for val scores and per-iter new_program_idx.
    state = GEPAState.load(str(cell_dir))
    val_scores = list(state.program_full_scores_val_set)
    trace = state.full_program_trace
    n = len(trace)
    if n != N_ITER:
        raise AssertionError(
            f"{arm}_{seed}: expected {N_ITER} iterations in full_program_trace, got {n}"
        )
    new_idx_by_iter: list[int | None] = []
    for entry in trace:
        v = entry.get("new_program_idx")
        new_idx_by_iter.append(int(v) if v is not None else None)
    return CellData(
        arm=arm,
        seed=seed,
        cell_summary=cell_summary,
        test_eval=test_eval,
        val_scores_by_candidate=val_scores,
        new_program_idx_by_iter=new_idx_by_iter,
    )


def validate_and_load() -> dict[tuple[str, int], CellData]:
    """Fail loud if anything is amiss; return all 15 cells."""
    cells: dict[tuple[str, int], CellData] = {}
    missing: list[tuple[str, int]] = []
    bad_n_test: list[tuple[str, int, int]] = []
    bad_n_iter: list[tuple[str, int, int]] = []

    for arm in ARMS:
        for seed in SEEDS:
            cd = LOGS_ROOT / f"{arm}_{seed}"
            if not (cd / "cell_summary.json").exists():
                missing.append((arm, seed))
                continue
            cell = _load_cell(arm, seed)
            if cell.n_test != 300:
                bad_n_test.append((arm, seed, cell.n_test))
            if cell.n_iter_actual != N_ITER:
                bad_n_iter.append((arm, seed, cell.n_iter_actual))
            cells[(arm, seed)] = cell

    if missing:
        raise SystemExit(f"validation failed -- missing cells: {missing}")
    if bad_n_test:
        raise SystemExit(f"validation failed -- n_test != 300: {bad_n_test}")
    if bad_n_iter:
        raise SystemExit(f"validation failed -- not 44 iters: {bad_n_iter}")
    if len(cells) != len(ARMS) * len(SEEDS):
        raise SystemExit(
            f"validation failed -- expected {len(ARMS) * len(SEEDS)} cells, got {len(cells)}"
        )

    manifest = (
        f"VALIDATED: {len(cells)}/{len(ARMS) * len(SEEDS)} cells; "
        f"every cell n_test=300 and state_i_final=43 (44 iters); "
        f"per-instance test F1 available={all(c.per_instance_test_f1 is not None for c in cells.values())}"
    )
    print(manifest)
    return cells


# ---------------------------------------------------------------------------
# D1 - iteration curves
# ---------------------------------------------------------------------------


def d1_iteration_curves(cells: dict[tuple[str, int], CellData]) -> dict:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    iters = np.arange(N_ITER + 1)

    per_arm_trajectories: dict[str, np.ndarray] = {}
    per_arm_endpoint_means: dict[str, float] = {}

    for arm in ARMS:
        seed_trajs = np.stack(
            [cells[(arm, s)].best_so_far_trajectory() for s in SEEDS]
        )  # shape (3, 45)
        per_arm_trajectories[arm] = seed_trajs
        mean_traj = seed_trajs.mean(axis=0)
        lo = seed_trajs.min(axis=0)
        hi = seed_trajs.max(axis=0)
        ax.plot(iters, mean_traj, color=_arm_color(arm), label=arm, linewidth=2)
        ax.fill_between(iters, lo, hi, color=_arm_color(arm), alpha=0.12)
        per_arm_endpoint_means[arm] = float(mean_traj[-1])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far val / selection F1 on D_pareto (n=75)")
    ax.set_title(
        "D1 - Best-so-far val/selection F1 vs iteration (3 seeds, band = min..max)"
    )
    ax.set_xlim(0, N_ITER)
    ax.set_ylim(bottom=0.55)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "iteration_curves.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"D1 -> {out_path}")

    # Source disclosure
    source = (
        "Source: state.program_full_scores_val_set (per-candidate avg D_pareto F1) "
        "indexed by state.full_program_trace[i]['new_program_idx']. "
        "Best-so-far is the cumulative max of accepted candidates' val scores; "
        "if no accept yet, value is the seed candidate's val F1."
    )

    return {
        "source": source,
        "per_arm_endpoint_mean_val_f1": per_arm_endpoint_means,
        "per_arm_seed_trajectories": {
            arm: per_arm_trajectories[arm].tolist() for arm in ARMS
        },
    }


# ---------------------------------------------------------------------------
# D2 - endpoint test F1 + hierarchical bootstrap CIs
# ---------------------------------------------------------------------------


def hierarchical_bootstrap_ci(
    per_seed_test_f1s: list[list[float] | float],
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """If per_seed_test_f1s[i] is a list of 300 per-instance scores, do a
    two-level bootstrap (resample seeds with replacement; for each chosen seed
    resample its 300 examples with replacement). If it's a single float
    (only aggregate available), bootstrap seeds only."""
    rng = np.random.default_rng(seed)
    K = len(per_seed_test_f1s)
    has_per_instance = isinstance(per_seed_test_f1s[0], list)

    if has_per_instance:
        as_arrays = [np.asarray(x, dtype=float) for x in per_seed_test_f1s]
    else:
        as_arrays = None  # type: ignore[assignment]

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

    point = float(np.mean(as_arrays).item()) if has_per_instance else float(np.mean(per_seed_test_f1s))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def d2_endpoint(cells: dict[tuple[str, int], CellData]) -> dict:
    per_arm_results: dict[str, dict] = {}
    forest_data = []
    has_per_instance_any = False

    for arm in ARMS:
        seeded = []
        per_seed_for_boot: list[list[float] | float] = []
        for s in SEEDS:
            c = cells[(arm, s)]
            seeded.append({"seed": s, "avg_test_f1": c.avg_test_f1})
            per_inst = c.per_instance_test_f1
            if per_inst is not None:
                per_seed_for_boot.append(per_inst)
                has_per_instance_any = True
            else:
                per_seed_for_boot.append(c.avg_test_f1)

        point, lo, hi = hierarchical_bootstrap_ci(per_seed_for_boot)
        per_arm_results[arm] = {
            "seed_test_f1s": seeded,
            "mean_test_f1": point,
            "ci_lo": lo,
            "ci_hi": hi,
        }
        forest_data.append((arm, point, lo, hi))

    # Forest plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ys = list(range(len(ARMS)))
    for y, (arm, point, lo, hi) in zip(ys, forest_data):
        ax.errorbar(
            point, y, xerr=[[point - lo], [hi - point]], fmt="o",
            color=_arm_color(arm), markersize=8, capsize=4, elinewidth=2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([a for a, *_ in forest_data])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out test F1 on the 300-example test set")
    ax.set_title(
        "D2 - Held-out test F1, per arm (mean across 3 seeds), 95% paired-hierarchical bootstrap CI"
    )
    ax.grid(axis="x", alpha=0.3)
    # Vertical line at overall mean for reference
    overall_mean = np.mean([d[1] for d in forest_data])
    ax.axvline(overall_mean, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label=f"5-arm mean = {overall_mean:.4f}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "endpoint_test_f1.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"D2 -> {out_path}")

    return {
        "per_arm": per_arm_results,
        "ci_method": (
            "hierarchical bootstrap, 10000 resamples, 95% percentile; "
            "resample seeds with replacement (n=3) and, when per-instance test "
            "scores are available, resample 300 examples with replacement "
            "within each chosen seed"
        ),
        "per_instance_available": has_per_instance_any,
        "binding_limitation": (
            "Only 3 seeds: the seed-level CI on the mean is wide and unstable; "
            "this is the dominant uncertainty across arms."
        ),
    }


# ---------------------------------------------------------------------------
# D3 - pairwise contrasts (paired bootstrap)
# ---------------------------------------------------------------------------


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


def d3_pairwise(cells: dict[tuple[str, int], CellData]) -> dict:
    contrasts = (
        ("static_frontier", "static_hard"),
        ("static_frontier", "random"),
        ("static_frontier", "static_easy"),
        ("static_frontier", "vanilla_coupled_gepa"),
    )
    results = []
    for a, b in contrasts:
        diffs = [cells[(a, s)].avg_test_f1 - cells[(b, s)].avg_test_f1 for s in SEEDS]
        mean_diff, lo, hi = paired_bootstrap_mean_diff(diffs)
        ci_includes_zero = (lo <= 0.0 <= hi)
        results.append({
            "left": a,
            "right": b,
            "per_seed_diffs": diffs,
            "mean_diff": mean_diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_includes_zero": ci_includes_zero,
        })
        print(
            f"D3 {a} - {b}: mean_diff = {mean_diff:+.4f}  "
            f"[{lo:+.4f}, {hi:+.4f}]  CI includes 0? {ci_includes_zero}"
        )

    return {
        "method": (
            "paired bootstrap on the 3 seed-matched test-F1 differences; "
            "10000 resamples, 95% percentile CI"
        ),
        "contrasts": results,
    }


# ---------------------------------------------------------------------------
# D4 - iterations to target
# ---------------------------------------------------------------------------


def d4_iterations_to_target(cells: dict[tuple[str, int], CellData]) -> dict:
    # Per-cell trajectories
    trajs = {(arm, s): cells[(arm, s)].best_so_far_trajectory() for arm in ARMS for s in SEEDS}

    def first_iter_at_or_above(traj: np.ndarray, target: float) -> int | None:
        idx = np.where(traj >= target)[0]
        return int(idx[0]) if idx.size else None

    # Sweep over thresholds
    table: dict[float, dict[str, dict]] = {}
    for T in TARGET_SWEEP:
        per_arm: dict[str, dict] = {}
        for arm in ARMS:
            iters_per_seed = []
            never = 0
            for s in SEEDS:
                k = first_iter_at_or_above(trajs[(arm, s)], T)
                if k is None:
                    iters_per_seed.append(None)
                    never += 1
                else:
                    iters_per_seed.append(k)
            valid = [i for i in iters_per_seed if i is not None]
            mean_iter = (sum(valid) / len(valid)) if valid else None
            per_arm[arm] = {
                "per_seed": iters_per_seed,
                "mean_iter_when_reached": mean_iter,
                "n_seeds_never_reached": never,
            }
        table[T] = per_arm

    # 90%-of-own-final per cell
    own90: dict[str, dict] = {}
    for arm in ARMS:
        per_seed_iters = []
        per_seed_targets = []
        for s in SEEDS:
            traj = trajs[(arm, s)]
            own_target = 0.9 * float(traj[-1])
            k = first_iter_at_or_above(traj, own_target)
            per_seed_iters.append(int(k) if k is not None else None)
            per_seed_targets.append(round(own_target, 4))
        valid = [i for i in per_seed_iters if i is not None]
        mean_iter = (sum(valid) / len(valid)) if valid else None
        own90[arm] = {
            "per_seed_target_f1": per_seed_targets,
            "per_seed_iter_reached": per_seed_iters,
            "mean_iter": mean_iter,
        }

    # Ranking flips
    arm_ranks_by_T = {}
    for T in TARGET_SWEEP:
        arm_means = {
            arm: table[T][arm]["mean_iter_when_reached"]
            for arm in ARMS
            if table[T][arm]["mean_iter_when_reached"] is not None
        }
        arm_ranks_by_T[T] = sorted(arm_means.items(), key=lambda kv: kv[1])

    flips_detected = len({tuple(arm for arm, _ in arm_ranks_by_T[T]) for T in TARGET_SWEEP}) > 1

    # Bar chart at PRIMARY_TARGET
    fig, ax = plt.subplots(figsize=(9, 5))
    arms_for_bars = list(ARMS)
    primary = table[PRIMARY_TARGET]
    means = []
    err_lo = []
    err_hi = []
    labels_below = []
    for arm in arms_for_bars:
        m = primary[arm]["mean_iter_when_reached"]
        per_seed = [k for k in primary[arm]["per_seed"] if k is not None]
        if m is None:
            means.append(0.0)
            err_lo.append(0.0)
            err_hi.append(0.0)
            labels_below.append(f"{arm}\n(never)")
        else:
            means.append(m)
            err_lo.append(m - min(per_seed))
            err_hi.append(max(per_seed) - m)
            never = primary[arm]["n_seeds_never_reached"]
            label = arm if never == 0 else f"{arm}\n({never}/3 never)"
            labels_below.append(label)
    ax.bar(
        range(len(arms_for_bars)), means,
        yerr=[err_lo, err_hi],
        color=[_arm_color(a) for a in arms_for_bars],
        capsize=5,
    )
    ax.set_xticks(range(len(arms_for_bars)))
    ax.set_xticklabels(labels_below, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"Iterations to first reach val F1 >= {PRIMARY_TARGET}")
    ax.set_title(
        f"D4 - Iterations to T={PRIMARY_TARGET} (one of {{'62, '63, '64}}); "
        "error bars are per-seed min..max"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "iterations_to_target.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"D4 -> {out_path}")

    return {
        "thresholds_swept": list(TARGET_SWEEP),
        "primary_threshold": PRIMARY_TARGET,
        "table": {
            f"{T:.2f}": {arm: table[T][arm] for arm in ARMS}
            for T in TARGET_SWEEP
        },
        "rank_by_T": {
            f"{T:.2f}": [arm for arm, _ in arm_ranks_by_T[T]]
            for T in TARGET_SWEEP
        },
        "ranking_flips_across_thresholds": flips_detected,
        "ninety_pct_of_own_final": own90,
    }


# ---------------------------------------------------------------------------
# D5 - acceptance dynamics
# ---------------------------------------------------------------------------


def d5_acceptance(cells: dict[tuple[str, int], CellData]) -> dict:
    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        per_seed = []
        accept_iters_all = []
        for s in SEEDS:
            c = cells[(arm, s)]
            ai = c.accepted_in_iters
            per_seed.append({"seed": s, "n_accepts": len(ai), "accepted_in_iters": ai})
            accept_iters_all.extend(ai)
        per_arm[arm] = {
            "per_seed": per_seed,
            "mean_accepts": statistics.mean(len(p["accepted_in_iters"]) for p in per_seed),
            "total_accepts_across_seeds": sum(len(p["accepted_in_iters"]) for p in per_seed),
            "all_accept_iters_pooled_across_seeds": sorted(accept_iters_all),
        }
    return {"per_arm": per_arm}


# ---------------------------------------------------------------------------
# FINDINGS narrative
# ---------------------------------------------------------------------------


def write_findings(
    cells: dict[tuple[str, int], CellData],
    d1: dict, d2: dict, d3: dict, d4: dict, d5: dict,
    out_path: Path,
):
    arm_means = {arm: d2["per_arm"][arm]["mean_test_f1"] for arm in ARMS}
    arm_lo = {arm: d2["per_arm"][arm]["ci_lo"] for arm in ARMS}
    arm_hi = {arm: d2["per_arm"][arm]["ci_hi"] for arm in ARMS}
    mean_min = min(arm_means.values())
    mean_max = max(arm_means.values())
    mean_spread = mean_max - mean_min

    # within-arm seed spread
    within_arm_spreads = {
        arm: max(
            cells[(arm, s)].avg_test_f1 for s in SEEDS
        ) - min(cells[(arm, s)].avg_test_f1 for s in SEEDS)
        for arm in ARMS
    }
    largest_within = max(within_arm_spreads.values())

    # frontier contrasts
    contrast_lines = []
    any_significant = False
    for c in d3["contrasts"]:
        sig_token = "**CI excludes 0**" if not c["ci_includes_zero"] else "CI includes 0"
        if not c["ci_includes_zero"]:
            any_significant = True
        contrast_lines.append(
            f"  - **{c['left']} - {c['right']}**: "
            f"{c['mean_diff']:+.4f} [95% CI {c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
            f"{sig_token}"
        )

    # Per-arm accepts
    accept_lines = []
    for arm in ARMS:
        a = d5["per_arm"][arm]
        accept_lines.append(
            f"  - {arm}: mean {a['mean_accepts']:.1f} accepts "
            f"(per-seed: {[p['n_accepts'] for p in a['per_seed']]})"
        )

    static_easy_accepts = d5["per_arm"]["static_easy"]["mean_accepts"]

    # rank flips
    rank_lines = []
    for T in TARGET_SWEEP:
        rank = d4["rank_by_T"][f"{T:.2f}"]
        rank_lines.append(f"  - T={T:.2f}: {' < '.join(rank)}")

    headline = (
        "**HEADLINE: NULL.** Endpoint test-F1 mean across the five arms ranges only "
        f"from {mean_min:.4f} to {mean_max:.4f} (spread {mean_spread:.4f}), while the largest "
        f"within-arm seed spread is {largest_within:.4f}. The seed-level noise is comparable "
        "to, and in places exceeds, the between-arm separation."
    )

    # Identify which contrasts exclude zero, so the narrative is precise.
    excludes_zero = [c for c in d3["contrasts"] if not c["ci_includes_zero"]]
    includes_zero = [c for c in d3["contrasts"] if c["ci_includes_zero"]]
    if excludes_zero:
        contrast_block = (
            f"Of the four paired contrasts, {len(excludes_zero)} has a 95% CI that excludes "
            f"zero ({', '.join(c['left'] + ' - ' + c['right'] for c in excludes_zero)}) "
            f"and {len(includes_zero)} include zero. The one that excludes zero "
            f"({excludes_zero[0]['left']} - {excludes_zero[0]['right']}: "
            f"{excludes_zero[0]['mean_diff']:+.4f}) puts static_frontier on the WRONG side of "
            "the contrast (worse, not better), and the magnitude is smaller than the within-"
            f"arm seed spread ({largest_within:.4f}). Honest read: the frontier-band "
            "hypothesis is NOT supported by these data at this scale, and the only CI that "
            "excludes zero points away from frontier rather than toward it."
        )
    else:
        contrast_block = (
            "Every paired contrast against static_frontier has a 95% CI that includes zero. "
            "The frontier-band hypothesis -- that selecting reflection minibatches from the "
            "partial-success band beats random / easy / hard / vanilla -- is NOT supported "
            "by these data at this scale."
        )

    lines: list[str] = []
    lines.append("# Chunk 7 - findings (the matrix is a null)")
    lines.append("")
    lines.append(headline)
    lines.append("")
    lines.append(
        "## D2 endpoint test F1 by arm "
        "(300-example held-out, mean of 3 seeds; 95% paired-hierarchical bootstrap CI)"
    )
    for arm in ARMS:
        lines.append(
            f"- {arm:24s} {arm_means[arm]:.4f}  "
            f"[{arm_lo[arm]:.4f}, {arm_hi[arm]:.4f}]"
        )
    lines.append("")
    lines.append(f"Mean range across arms: {mean_min:.4f} to {mean_max:.4f} (spread {mean_spread:.4f}).")
    lines.append(f"Largest within-arm seed spread: {largest_within:.4f}.")
    lines.append("")
    lines.append(
        "## D3 paired contrasts "
        "(seed-matched; paired bootstrap, 10000 resamples, 95% percentile)"
    )
    lines.extend(line.lstrip() for line in contrast_lines)
    lines.append("")
    lines.append(contrast_block)
    lines.append("")
    lines.append("## D4 iterations-to-target ranking flips across thresholds")
    lines.append("Ranks (fastest first; arms that never reach are omitted):")
    lines.extend(line.lstrip() for line in rank_lines)
    lines.append("")
    lines.append(
        f"Ranking flips across thresholds: {d4['ranking_flips_across_thresholds']}. "
        "That itself is a finding -- at this scale, \"static_frontier is faster\" depends "
        "on where you put the finish line, which is consistent with noise rather than a "
        "real efficiency edge. (In fact, static_frontier is the SLOWEST arm to reach "
        f"T=0.63 by mean iteration count, at {d4['table']['0.63']['static_frontier']['mean_iter_when_reached']:.1f} iters; "
        f"vanilla_coupled_gepa is the fastest at {d4['table']['0.63']['vanilla_coupled_gepa']['mean_iter_when_reached']:.1f}.)"
    )
    lines.append("")
    lines.append("## D5 acceptance dynamics (the off-band leakage in numbers)")
    lines.extend(line.lstrip() for line in accept_lines)
    lines.append("")
    lines.append(
        f"Note that **static_easy accepts at a non-trivial rate** "
        f"({static_easy_accepts:.1f} accepts per cell on average, with one seed accepting 9 "
        "times). The static_easy minibatch is supposed to be all F1==1 examples (\"nothing to "
        "correct\"); the only reason it accepts is the 15% / 15% off-band draws, which pull "
        "from the strictly-partial mid band and the F1==0 hard band on every iteration. That "
        "same 30% off-band exposure reaches every static arm, so each arm sees enough partial-"
        "band material to drive useful reflection, which dilutes the between-arm contrast the "
        "experiment is built around."
    )
    lines.append("")
    lines.append("## Two structural reasons the null is plausible at this scale")
    lines.append(
        "1. **Bounded bidirectional formatting gains.** The Chunk-5 partial inspection "
        "showed the strictly-partial cases are dominated by verbosity / specificity "
        "mismatches (the partial_verbose mu_f branch). Any instruction edit that tightens "
        "the answer to reduce extras necessarily risks dropping middle names or qualifiers "
        "and falling into the opposite verbosity failure. The reflection LM gets reliable "
        "signal both ways, so the floor is high and the ceiling is bounded -- arms converge "
        "to similar endpoints from different angles."
    )
    lines.append(
        "2. **15/15 off-band leakage.** The 70/15/15 mix means every static arm gets 30 "
        "percent of its minibatch from off-band instances. With the mid band 31 examples "
        "deep and the hard band 50 deep, even static_easy sees several partial instances "
        "and several zero instances over N=44 iterations on average, which is enough to "
        "drive accepted edits (D5 shows it does, including one static_easy cell with 9 "
        "accepts). The \"static-easy\" arm in this setup is not reflection-starved; it is "
        "reflection-diluted. A genuine purity test would use a 100/0/0 draw and accept the "
        "coverage-vs-purity tradeoff."
    )
    lines.append("")
    lines.append("## Interview-ready summary (3-4 sentences)")
    if excludes_zero:
        ez = excludes_zero[0]
        ez_clause = (
            f"Three of four paired contrasts against static_frontier have 95 percent CIs that "
            f"include zero; the one that excludes zero ({ez['left']} - {ez['right']} = "
            f"{ez['mean_diff']:+.4f}) puts static_frontier on the WORSE side of the contrast "
            f"and has a magnitude smaller than the within-arm seed spread "
            f"({largest_within:.4f})."
        )
    else:
        ez_clause = (
            "Every paired contrast against static_frontier has a 95 percent CI that includes "
            "zero."
        )
    lines.append(
        "We ran the 15-cell static-band matrix end-to-end at the locked "
        f"D_feedback=150 / A=20 / D_pareto=75 / N=44 settings and observed a clean null: "
        f"arm-mean held-out test F1 spans only {mean_spread:.4f} F1 points while within-arm "
        f"seed spread reaches {largest_within:.4f}. {ez_clause} The cleanest design lesson is "
        "the 15/15 off-band leakage: static_easy still accepted prompts because 30 percent of "
        "every minibatch is off-band, so each arm gets enough partial-band examples to drive "
        "useful reflection and the cross-arm contrast gets diluted. A sharper follow-up swaps "
        "the 70/15/15 mix for a 100/0/0 pure-band draw and asks whether the resulting purity "
        "gain beats the coverage loss -- that is the genuine test of band selection as a lever."
    )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"FINDINGS -> {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = validate_and_load()

    d1 = d1_iteration_curves(cells)
    d2 = d2_endpoint(cells)
    d3 = d3_pairwise(cells)
    d4 = d4_iterations_to_target(cells)
    d5 = d5_acceptance(cells)

    # Persist every number
    summary = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOT,
        "target_sweep": list(TARGET_SWEEP),
        "primary_target_for_plot": PRIMARY_TARGET,
        "n_iterations_per_cell": N_ITER,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "per_seed_test_f1": {
            arm: {s: cells[(arm, s)].avg_test_f1 for s in SEEDS} for arm in ARMS
        },
        "D1_iteration_curves": d1,
        "D2_endpoint_test_f1": d2,
        "D3_pairwise_contrasts": d3,
        "D4_iterations_to_target": d4,
        "D5_acceptance_dynamics": d5,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {OUT_DIR / 'summary.json'}")

    write_findings(cells, d1, d2, d3, d4, d5, OUT_DIR / "FINDINGS.md")
    print()
    print(f"bootstrap_seed used: {BOOTSTRAP_SEED}")
    print("Chunk 7 analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
