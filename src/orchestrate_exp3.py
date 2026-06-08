"""Exp-3 matrix orchestrator: 4 arms x 3 seeds = 12 cells, resumable.

Each cell runs in its OWN process (ProcessPoolExecutor). Process isolation is
required because dspy.settings.configure(...) is process-global; running cells
as threads in one process would race on the global LM config. The Chunk-3
runner was built for exactly this: each (arm, seed) writes only to its own
results/exp3_curriculum/<arm>/seed<seed>{,.json} and shares no mutable state.

Resumable: a cell whose summary JSON already exists and is complete
(arm/seed/T match and it has T iteration records and a final_test_f1) is
skipped. A crashed/partial cell has no summary and is re-run.

429 handling: each cell's adapter retries transient errors (litellm
num_retries + src.retry.retryable with exponential backoff up to 30s). If
Together rate-limits sustainedly a cell will exhaust retries and raise; the
orchestrator logs the failure (incl. the 429) and continues other cells, and
the operator can relaunch at lower --workers (resumability makes that safe).

Run:   .venv/bin/python -m src.orchestrate_exp3 --workers 4
Dry:   .venv/bin/python -m src.orchestrate_exp3 --dry-run
Monitor (separate terminal):
   caffeinate -dims is already applied if launched via nohup caffeinate ...;
   tail -f results/exp3_curriculum/matrix_run.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

ARMS: tuple[str, ...] = ("random", "easy_to_hard", "hard_to_easy", "static_medium")
SEEDS: tuple[int, ...] = (0, 1, 2)
T: int = 40
OUT_ROOT = _REPO / "experiments" / "exp3_curriculum"


def all_cells() -> list[tuple[str, int]]:
    return [(arm, seed) for arm in ARMS for seed in SEEDS]


def cell_summary_path(out_root: Path, arm: str, seed: int) -> Path:
    return Path(out_root) / arm / f"seed{seed}.json"


def cell_complete(out_root: Path, arm: str, seed: int, T: int = T) -> bool:
    """A cell is complete iff its summary exists, matches (arm, seed, T), has
    exactly T iteration records, and has a non-null final_test_f1."""
    path = cell_summary_path(out_root, arm, seed)
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        d.get("arm") == arm
        and d.get("seed") == seed
        and d.get("T") == T
        and isinstance(d.get("iterations"), list)
        and len(d["iterations"]) == T
        and d.get("final_test_f1") is not None
    )


def plan(out_root: Path = OUT_ROOT, T: int = T) -> tuple[list, list]:
    """Return (todo, skipped) cell lists given what's already complete."""
    todo, skipped = [], []
    for arm, seed in all_cells():
        (skipped if cell_complete(out_root, arm, seed, T) else todo).append((arm, seed))
    return todo, skipped


def _run_one_cell(arm: str, seed: int, out_root: str, T: int) -> dict:
    """Worker (top-level so it is picklable for spawn). Runs one cell and
    returns a small status dict. Imports happen inside so each spawned process
    sets up its own env + dspy global state."""
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
    from src.curriculum_runner import build_and_run

    t0 = time.perf_counter()
    summary = build_and_run(
        arm, seed, out_root=Path(out_root), T=T, raise_on_exception=True
    )
    return {
        "arm": arm,
        "seed": seed,
        "seconds": time.perf_counter() - t0,
        "first_accept_iter": summary["first_accept_iter"],
        "cumulative_accepts": summary["cumulative_accepts"],
        "final_test_f1": summary["final_test_f1"],
    }


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def run_matrix(workers: int = 4, out_root: Path = OUT_ROOT, T: int = T) -> int:
    todo, skipped = plan(out_root, T)
    _log(f"Exp-3 matrix: {len(all_cells())} cells, {len(skipped)} complete, "
         f"{len(todo)} to run, workers={workers}, T={T}")
    for arm, seed in skipped:
        _log(f"  SKIP (complete): {arm} seed{seed}")
    if not todo:
        _log("Nothing to run; all cells complete.")
        return 0
    for arm, seed in todo:
        _log(f"  TODO: {arm} seed{seed}")

    failures: list[tuple[str, int, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_cell, a, s, str(out_root), T): (a, s) for a, s in todo}
        for fut in as_completed(futs):
            arm, seed = futs[fut]
            try:
                r = fut.result()
                done += 1
                _log(
                    f"DONE {arm} seed{seed} in {r['seconds']:.0f}s "
                    f"(first_accept={r['first_accept_iter']}, "
                    f"accepts={r['cumulative_accepts']}, "
                    f"test_f1={r['final_test_f1']:.4f})  [{done}/{len(todo)}]"
                )
            except Exception as e:  # noqa: BLE001 -- log and continue other cells
                msg = f"{type(e).__name__}: {e}"
                failures.append((arm, seed, msg))
                _log(f"FAIL {arm} seed{seed}: {msg}")
                _log(traceback.format_exc())

    # Final accounting
    todo_after, skipped_after = plan(out_root, T)
    _log(f"Matrix finished. complete={len(skipped_after)}/{len(all_cells())}, "
         f"still_missing={len(todo_after)}, failures_this_run={len(failures)}")
    if "429" in " ".join(m for _, _, m in failures) or any(
        "RateLimit" in m for _, _, m in failures
    ):
        _log("NOTE: rate-limit (429) failures detected -- relaunch at lower "
             "--workers; resumability will skip completed cells.")
    if todo_after:
        for arm, seed in todo_after:
            _log(f"  MISSING: {arm} seed{seed}")
        return 1
    _log("All 12 cell summaries exist.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Exp-3 4x3 matrix.")
    ap.add_argument("--workers", type=int, default=4, help="process-pool size (default 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the enumeration + skip plan and exit (no model calls)")
    args = ap.parse_args(argv)

    if args.dry_run:
        todo, skipped = plan()
        _log(f"DRY RUN: {len(all_cells())} cells; {len(skipped)} complete, {len(todo)} to run.")
        for arm, seed in all_cells():
            status = "complete" if (arm, seed) in skipped else "TODO"
            _log(f"  {arm} seed{seed}: {status}")
        print("\nMonitor a real run with:")
        print("  caffeinate -dims nohup .venv/bin/python -m src.orchestrate_exp3 "
              "--workers 4 > results/exp3_curriculum/matrix_run.log 2>&1 &")
        print("  tail -f results/exp3_curriculum/matrix_run.log")
        return 0

    return run_matrix(workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
