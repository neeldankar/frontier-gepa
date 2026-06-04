"""Drift instrumentation for the IFBench matrix (BUILD_PLAN §4 D4).

D4 verbatim:
  - The frozen difficulty table is scored at the run temperature (0.6)
    and defines the bands the matrix uses. The drift measurement must
    NOT correlate that 0.6 table against a temp-0 re-score, because that
    mixes policy drift with the temperature change.
  - Instead: re-score the base/seed candidate AND each cell's final best
    candidate at temp 0, and compute the Spearman rank correlation
    between the temp-0 base scoring and the temp-0 final scoring over
    all 150 feedback-pool instances.
  - Also report the frontier-leaving rate computed temperature-
    consistently: bin the temp-0 base scores with the same rule, then
    measure how many of those frontier instances leave the middle band
    under the temp-0 final scores.

This module implements the two temp-0 passes + Spearman + the
temperature-consistent frontier-leaving rate. The temp-0 base pass is
shared across all 15 cells (one deterministic pass), so the orchestrator
caches it to `results/logs_ifbench/_drift/base_temp0_scores.json` and
reuses it on subsequent cells.

The matrix-time D_pareto eval and held-out test eval are unaffected;
this drift step is an *additional* deterministic pass that the BUILD_PLAN
calls out as "negligible cost".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import dspy
import numpy as np

from src.difficulty import build_chosen_table


# ---------------------------------------------------------------------------
# Temperature-controlled re-score
# ---------------------------------------------------------------------------


def _make_temp0_task_lm(task_lm_config: Any) -> dspy.LM:
    """Construct a temp-0 dspy.LM mirroring `task_lm_config` for everything
    else (model, api_key, api_base, max_tokens). top_p is forced to 1.0
    because we want deterministic / argmax output and don't want nucleus
    sampling to mask that intent."""
    kwargs: dict[str, Any] = dict(
        model=task_lm_config.model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=task_lm_config.max_tokens,
        num_retries=task_lm_config.num_retries,
    )
    if task_lm_config.api_key:
        kwargs["api_key"] = task_lm_config.api_key
    if task_lm_config.api_base:
        kwargs["api_base"] = task_lm_config.api_base
    return dspy.LM(**kwargs)


def rescore_at_temp0(
    candidate: Mapping[str, str],
    d_feedback: list[Any],
    task_lm_config: Any,
    substrate: Any,
) -> tuple[list[float], dict[str, int]]:
    """Run a deterministic (temp-0) pass of `candidate` over every example
    in `d_feedback`, scoring with `substrate.metric_fn`. Returns
    (per_example_scores, cost_meta) where cost_meta has approximate token
    counts pulled off the dspy.LM history.

    Uses `dspy.Evaluate(num_threads=substrate.num_threads)` so the drift
    pass shares the same parallelism as the matrix run. For IFBench this
    is 16; for HotpotQA the substrate sets None (serial), so this code
    path matches the existing HotpotQA behaviour byte-for-byte if a
    HotpotQA drift step is ever wired in.
    """
    task_lm = _make_temp0_task_lm(task_lm_config)
    dspy.settings.configure(lm=task_lm)
    program = substrate.build_program()
    program.apply_instructions(dict(candidate))

    evaluator = dspy.Evaluate(
        devset=list(d_feedback),
        metric=substrate.metric_fn,
        num_threads=substrate.num_threads,
        failure_score=0.0,
        provide_traceback=True,
        max_errors=len(d_feedback) * 100,
    )
    res = evaluator(program)
    # res.results is list[tuple[Example, Prediction, score]]; dspy.Evaluate
    # preserves devset order, so the score list aligns with d_feedback ids.
    scores = [float(r[2]) for r in res.results]

    # Best-effort token accounting via dspy history.
    n_calls = len(task_lm.history or [])
    tokens_in = sum(
        int(((h.get("usage") or {}).get("prompt_tokens", 0)))
        for h in (task_lm.history or [])
    )
    tokens_out = sum(
        int(((h.get("usage") or {}).get("completion_tokens", 0)))
        for h in (task_lm.history or [])
    )
    cost_meta = {
        "n_calls": n_calls,
        "task_tokens_in": tokens_in,
        "task_tokens_out": tokens_out,
    }
    return scores, cost_meta


# ---------------------------------------------------------------------------
# Spearman with tie-aware mid-ranks
# ---------------------------------------------------------------------------


def _rankdata_midrank(arr: Iterable[float]) -> np.ndarray:
    """1-based mid-rank ranking (scipy.stats.rankdata default 'average'
    behaviour). Implemented manually to avoid a scipy dependency."""
    a = np.asarray(list(arr), dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0  # 1-based mid-rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_corr(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two equal-length sequences with
    mid-rank tie handling. Returns float in [-1, 1]. If either series is
    constant (all ranks identical), Pearson is undefined; we return 0.0
    in that case, treating "no information" as no correlation."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: len(x)={len(x)} len(y)={len(y)}")
    if len(x) < 2:
        raise ValueError("spearman_corr requires at least 2 elements")
    rx = _rankdata_midrank(x)
    ry = _rankdata_midrank(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Temperature-consistent frontier-leaving rate (D4)
# ---------------------------------------------------------------------------


def temperature_consistent_frontier_leaving_rate(
    base_temp0_scores: list[float],
    final_temp0_scores: list[float],
) -> dict[str, Any]:
    """Apply the Chunk-13 continuity gate ('build_chosen_table') to BOTH
    the temp-0 base and the temp-0 final scores. Identify the frontier
    (mid-band) id sets under each. Return:
      - base_frontier_ids:   id list (under temp-0 base binning)
      - final_frontier_ids:  id list (under temp-0 final binning)
      - n_left:              base_frontier - final_frontier (set diff size)
      - frontier_leaving_rate: n_left / |base_frontier| (in [0,1]; nan if
        base frontier is empty)
      - base_binning, final_binning: 'rank_terciles' or 'value_bins',
        which the gate selected on each scoring.

    The "temperature-consistent" reading: both ends are at temp 0, so
    any movement is policy drift, not temperature change. The Chunk-13
    rule (rank terciles iff middle ≥80% partial AND extreme mass <50%;
    else value bins) is applied independently to each scoring -- this
    is "same rule, applied at temp 0 to both ends", which is exactly the
    D4 instruction."""
    if len(base_temp0_scores) != len(final_temp0_scores):
        raise ValueError(
            f"length mismatch: base={len(base_temp0_scores)} "
            f"final={len(final_temp0_scores)}"
        )
    base_table, base_gate = build_chosen_table(list(base_temp0_scores))
    final_table, final_gate = build_chosen_table(list(final_temp0_scores))
    base_frontier = set(base_table.ids("mid"))
    final_frontier = set(final_table.ids("mid"))
    left = base_frontier - final_frontier
    rate = (len(left) / len(base_frontier)) if base_frontier else float("nan")
    return {
        "base_binning": base_gate["chosen_binning"],
        "final_binning": final_gate["chosen_binning"],
        "n_base_frontier": len(base_frontier),
        "n_final_frontier": len(final_frontier),
        "n_left_frontier": len(left),
        "frontier_leaving_rate": rate,
        "base_frontier_ids": sorted(base_frontier),
        "final_frontier_ids": sorted(final_frontier),
        "ids_left_frontier": sorted(left),
    }


# ---------------------------------------------------------------------------
# Orchestrator-facing entrypoint
# ---------------------------------------------------------------------------


def compute_drift_for_cell(
    base_seed_candidate: Mapping[str, str],
    final_candidate: Mapping[str, str],
    d_feedback: list[Any],
    task_lm_config: Any,
    substrate: Any,
    drift_cache_dir: Path,
) -> dict[str, Any]:
    """Run (or load cached) temp-0 base scoring, run temp-0 final
    scoring, compute Spearman and the temperature-consistent
    frontier-leaving rate, and return a serialisable dict for the
    cell_summary.

    `drift_cache_dir` is shared across all cells of a run; the base pass
    is cached to `drift_cache_dir/base_temp0_scores.json`.
    """
    drift_cache_dir = Path(drift_cache_dir)
    drift_cache_dir.mkdir(parents=True, exist_ok=True)
    base_cache = drift_cache_dir / "base_temp0_scores.json"

    t0 = time.time()
    if base_cache.exists():
        cached = json.loads(base_cache.read_text())
        base_temp0_scores = [float(s) for s in cached["scores"]]
        base_cost = cached.get("cost_meta", {})
        base_reused = True
    else:
        base_temp0_scores, base_cost = rescore_at_temp0(
            base_seed_candidate, d_feedback, task_lm_config, substrate
        )
        base_cache.write_text(json.dumps(
            {"scores": base_temp0_scores, "cost_meta": base_cost}, indent=2
        ))
        base_reused = False
    t_base = time.time() - t0

    t1 = time.time()
    final_temp0_scores, final_cost = rescore_at_temp0(
        final_candidate, d_feedback, task_lm_config, substrate
    )
    t_final = time.time() - t1

    spearman = spearman_corr(base_temp0_scores, final_temp0_scores)
    leaving = temperature_consistent_frontier_leaving_rate(
        base_temp0_scores, final_temp0_scores
    )

    return {
        "n_d_feedback": len(d_feedback),
        "base_temp0_scores": base_temp0_scores,
        "final_temp0_scores": final_temp0_scores,
        "base_temp0_reused": base_reused,
        "spearman_base_vs_final_temp0": spearman,
        "frontier_leaving": leaving,
        "wall_s_base": round(t_base, 2),
        "wall_s_final": round(t_final, 2),
        "cost_meta_base": base_cost,
        "cost_meta_final": final_cost,
    }
