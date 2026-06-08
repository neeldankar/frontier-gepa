"""Exp-3 Chunk-4 smoke: one real (arm, seed) cell, short budget.

Runs ``hard_to_easy`` seed 0 with T=8 against the real HotpotQA substrate and
real LMs, writing to ``results/exp3_curriculum/smoke/``. This is the GO/NO-GO
gate before the full matrix: it validates the Chunk-3 plumbing under real
conditions and measures per-iteration wall so the matrix can be projected.

Run: ``.venv/bin/python -m src.smoke_exp3``
Required env: TASK_MODEL + TASK_MODEL_API_KEY + REFLECTION_MODEL + OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from gepa.core.callbacks import GEPACallback

from src.curriculum_runner import build_and_run
from src.schedule import bin_for_iteration

_REPO = Path(__file__).resolve().parents[1]
SMOKE_ROOT = _REPO / "results" / "exp3_curriculum" / "smoke"
ARM = "hard_to_easy"
SEED = 0
T = 8


class TimingCallback(GEPACallback):
    """Record pure per-iteration wall (on_iteration_start -> on_iteration_end)."""

    def __init__(self) -> None:
        self._starts: dict[int, float] = {}
        self.durations: list[tuple[int, float]] = []  # (1-based iter, seconds)

    def on_iteration_start(self, event) -> None:
        self._starts[event["iteration"]] = time.perf_counter()

    def on_iteration_end(self, event) -> None:
        it = event["iteration"]
        if it in self._starts:
            self.durations.append((it, time.perf_counter() - self._starts[it]))


def main() -> int:
    load_dotenv(_REPO / ".env")

    timing = TimingCallback()
    capture: dict = {}

    t0 = time.perf_counter()
    summary = build_and_run(
        ARM,
        SEED,
        out_root=SMOKE_ROOT,
        T=T,
        extra_callbacks=[timing],
        state_capture=capture,
        raise_on_exception=True,
    )
    total_wall = time.perf_counter() - t0

    state = capture["state"]
    engine_secs = capture["engine_run_seconds"]
    test_secs = capture["test_eval_seconds"]
    records = summary["iterations"]
    run_dir = SMOKE_ROOT / ARM / f"seed{SEED}"
    meta = json.loads((run_dir / "run_meta.json").read_text())

    # ---- Confirmation 1: exactly T iterations, state.i spanned 0..T-1 ----
    iters = [r["iter"] for r in records]
    c1_count = len(records) == T
    c1_span = iters == list(range(T))

    # ---- Confirmation 2: bin schedule for hard_to_easy at T=8 ----
    expected_bins = [bin_for_iteration(ARM, i, T) for i in range(T)]
    actual_bins = [r["bin_sampled"] for r in records]
    c2 = actual_bins == expected_bins

    # ---- Confirmation 3: decoupling -- A=20 averages drove acceptance ----
    A = meta["A"]
    c3_A_is_20 = A == 20
    # parent/candidate a20 are averages in [0,1]; accepted iff candidate>parent.
    c3_avg_range = all(
        (r["parent_a20"] is None or 0.0 <= r["parent_a20"] <= 1.0)
        and (r["candidate_a20"] is None or 0.0 <= r["candidate_a20"] <= 1.0)
        for r in records
    )
    c3_accept_decided_on_A = all(
        (r["accepted"] == (r["candidate_a20"] > r["parent_a20"]))
        for r in records
        if r["parent_a20"] is not None and r["candidate_a20"] is not None
    )

    # ---- Confirmation 4: best candidate by D_pareto aggregate; test on 300 ---
    val_scores = list(state.program_full_scores_val_set)
    argmax_idx = max(range(len(val_scores)), key=lambda k: val_scores[k])
    c4_best_is_dpareto_argmax = argmax_idx == summary["best_candidate_idx"]
    c4_test_n = len(state.program_candidates)  # candidates discovered

    # ---- Timing ----
    per_iter = [d for _, d in timing.durations]
    mean_iter = sum(per_iter) / len(per_iter) if per_iter else float("nan")
    base_eval_secs = engine_secs - sum(per_iter)  # seed eval on D_pareto, pre-loop

    report = {
        "arm": ARM, "seed": SEED, "T": T,
        "confirmations": {
            "c1_exactly_T_iters": c1_count,
            "c1_state_i_span_0_to_T-1": c1_span,
            "c2_bin_schedule_matches": c2,
            "c3_A_eq_20": c3_A_is_20,
            "c3_a20_are_averages_in_0_1": c3_avg_range,
            "c3_accept_decided_on_A20": c3_accept_decided_on_A,
            "c4_best_is_dpareto_argmax": c4_best_is_dpareto_argmax,
        },
        "first_accept_iter": summary["first_accept_iter"],
        "cumulative_accepts": summary["cumulative_accepts"],
        "final_test_f1": summary["final_test_f1"],
        "best_candidate_idx": summary["best_candidate_idx"],
        "n_candidates_discovered": c4_test_n,
        "val_aggregate_scores": val_scores,
        "expected_bins": expected_bins,
        "actual_bins": actual_bins,
        "timing": {
            "total_wall_seconds": total_wall,
            "engine_run_seconds": engine_secs,
            "base_eval_seconds_est": base_eval_secs,
            "test_eval_seconds": test_secs,
            "per_iter_seconds": [round(d, 3) for d in per_iter],
            "mean_iter_seconds": mean_iter,
            "min_iter_seconds": min(per_iter) if per_iter else None,
            "max_iter_seconds": max(per_iter) if per_iter else None,
        },
        "bin_to_a20": [
            {"iter": r["iter"], "bin": r["bin_sampled"],
             "parent_a20": r["parent_a20"], "candidate_a20": r["candidate_a20"],
             "accepted": r["accepted"]}
            for r in records
        ],
    }

    # ---- Full-matrix projection (40 iters x 12 cells + 12 test evals) ----
    FULL_T, N_CELLS = 40, 12
    per_cell_iter = base_eval_secs + FULL_T * mean_iter
    per_cell_total = per_cell_iter + test_secs
    serial_total = N_CELLS * per_cell_total
    projection = {
        "assumptions": (
            "per-cell = base_eval + 40*mean_iter + test_eval; "
            "concurrency is idealized wall = serial/N (ignores Together rate "
            "limits and shared-endpoint contention; real speedup will be lower)."
        ),
        "mean_iter_seconds": mean_iter,
        "base_eval_seconds_est": base_eval_secs,
        "test_eval_seconds": test_secs,
        "per_cell_seconds": per_cell_total,
        "serial_total_hours": serial_total / 3600.0,
        "concurrent_N4_hours": serial_total / 4 / 3600.0,
        "concurrent_N6_hours": serial_total / 6 / 3600.0,
        "concurrent_N12_hours": serial_total / 12 / 3600.0,
    }
    report["projection"] = projection

    (SMOKE_ROOT / "smoke_report.json").write_text(json.dumps(report, indent=2))

    # ---- Print ----
    print("=" * 64)
    print(f"Exp-3 smoke: {ARM} seed{SEED} T={T}")
    print("=" * 64)
    print("Confirmations:")
    for k, v in report["confirmations"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nbin schedule expected: {expected_bins}")
    print(f"bin schedule actual:   {actual_bins}")
    print(f"\nfirst_accept_iter={summary['first_accept_iter']}  "
          f"cumulative_accepts={summary['cumulative_accepts']}  "
          f"final_test_f1={summary['final_test_f1']:.4f}")
    print(f"best_candidate_idx={summary['best_candidate_idx']} "
          f"(D_pareto argmax={argmax_idx}); "
          f"candidates discovered={c4_test_n}; val_scores={[round(s,4) for s in val_scores]}")
    print("\nPer-iter (iter: bin, parent_a20 -> candidate_a20, accepted):")
    for r in records:
        pa = f"{r['parent_a20']:.4f}" if r["parent_a20"] is not None else "None"
        ca = f"{r['candidate_a20']:.4f}" if r["candidate_a20"] is not None else "None"
        print(f"  {r['iter']}: {r['bin_sampled']:<7} {pa} -> {ca}  accepted={r['accepted']}")
    print("\nTiming:")
    print(f"  total_wall={total_wall:.1f}s  engine={engine_secs:.1f}s  "
          f"base_eval_est={base_eval_secs:.1f}s  test_eval={test_secs:.1f}s")
    print(f"  per_iter(s)={[round(d,1) for d in per_iter]}")
    print(f"  mean_iter={mean_iter:.1f}s  min={min(per_iter):.1f}s  max={max(per_iter):.1f}s")
    print("\nFull-matrix projection (40 iters x 12 cells + 12 test evals):")
    print(f"  per_cell={per_cell_total/60:.1f} min")
    print(f"  serial         : {projection['serial_total_hours']:.2f} h")
    print(f"  concurrent N=4 : {projection['concurrent_N4_hours']:.2f} h")
    print(f"  concurrent N=6 : {projection['concurrent_N6_hours']:.2f} h")
    print(f"  concurrent N=12: {projection['concurrent_N12_hours']:.2f} h")
    print(f"\nwrote {SMOKE_ROOT / 'smoke_report.json'}")

    all_pass = all(report["confirmations"].values())
    print(f"\n{'GO (all confirmations pass)' if all_pass else 'CHECK (a confirmation failed)'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
