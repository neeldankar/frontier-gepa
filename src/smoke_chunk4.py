"""Chunk 4 tiny-slice smoke (the only API spend in this chunk).

Builds a shrunk config (d_feedback=5, accept_batch=4, d_pareto=4, test=4,
N=2 iterations), takes a corresponding slice of the real HotpotQA distractor
splits, scores the base system on the 5 d_feedback to build a small
DifficultyTable, then runs `run_gepa.run("static_frontier", seed=0)` against
that table. The smoke passes iff the engine completes both iterations
without raising AND produces at least one accepted candidate (i.e. the new
program ends up with at least 2 candidates: the seed plus one new).

Estimated cost: ~50 rollouts, ~$0.01 at Qwen2.5-7B-Turbo serverless.

Run as:
  .venv/bin/python -m src.smoke_chunk4

Requires TASK_MODEL and the matching API key in .env. Aborts at startup if
missing.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
import time
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import load_splits  # noqa: E402
from src.difficulty import build_difficulty_table  # noqa: E402
from src.feedback import metric_fn as module_metric  # noqa: E402
from src.program import build_program  # noqa: E402
from src.run_gepa import LMConfig, load_lm_configs_from_env, run  # noqa: E402


SMOKE_D_FEEDBACK = 10  # need enough variance for a non-degenerate frontier band
SMOKE_ACCEPT = 8       # larger A -> less likely to tie under strict > acceptance
SMOKE_D_PARETO = 4
SMOKE_TEST = 4
SMOKE_N_ITER = 5


def _smoke_config(base_config: dict) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["splits"] = {
        "d_feedback": SMOKE_D_FEEDBACK,
        "accept_batch": SMOKE_ACCEPT,
        "d_pareto": SMOKE_D_PARETO,
        "test": SMOKE_TEST,
        "seed_splits": int(cfg["splits"].get("seed_splits", 0)),
    }
    cfg["stopping"]["n"] = SMOKE_N_ITER
    return cfg


def _score_base_system(d_feedback: list, task_lm_config: LMConfig) -> list[float]:
    """Score the base (seed) program on each d_feedback example and return
    a list of F1s aligned with the d_feedback order. Used only to build the
    smoke's difficulty table; Chunk 5 will do the full-scale version."""
    dspy.settings.configure(lm=task_lm_config.to_lm())
    program = build_program()
    scores: list[float] = []
    for ex in d_feedback:
        out = program(**dict(ex.inputs()))
        scores.append(module_metric(ex, out))
    return scores


def main() -> int:
    load_dotenv(REPO / ".env")
    if not os.environ.get("TASK_MODEL"):
        print("ERROR: TASK_MODEL is not set. Set it in .env (see .env.example).")
        return 1
    if not os.environ.get("REFLECTION_MODEL"):
        print("ERROR: REFLECTION_MODEL is not set.")
        return 1

    base_config = yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())
    config = _smoke_config(base_config)

    task_lm_config, refl_lm_config = load_lm_configs_from_env(config=config)

    print("Loading splits (smoke shape)...")
    splits = load_splits(config=config)
    d_feedback, accept_batch, d_pareto, test = splits
    print(
        f"  d_feedback={len(d_feedback)}, accept={len(accept_batch)}, "
        f"d_pareto={len(d_pareto)}, test={len(test)}"
    )

    print("Scoring base system on d_feedback to build the difficulty table...")
    t0 = time.time()
    base_scores = _score_base_system(list(d_feedback), task_lm_config)
    print(f"  base F1s: {[f'{s:.2f}' for s in base_scores]} (took {time.time()-t0:.1f}s)")
    difficulty_table = build_difficulty_table(base_scores)
    bands = {b: difficulty_table.ids(b) for b in ("easy", "mid", "hard")}
    print(f"  bands: {bands}")

    run_dir = REPO / "results" / "smoke_chunk4"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    print()
    print("Running gepa engine (arm=static_frontier, seed=0, N=2)...")
    t0 = time.time()
    state = run(
        arm="static_frontier",
        seed=0,
        splits=splits,
        difficulty_table=difficulty_table,
        run_dir=run_dir,
        task_lm_config=task_lm_config,
        reflection_lm_config=refl_lm_config,
        config=config,
        raise_on_exception=True,
    )
    print(f"  engine finished in {time.time()-t0:.1f}s")
    print(f"  total_num_evals: {state.total_num_evals}")
    print(f"  iterations completed (state.i+1): {state.i + 1}")
    print(f"  total candidates: {len(state.program_candidates)}")
    if len(state.program_candidates) > 1:
        print("  ACCEPTED at least one new candidate -- smoke PASSED.")
        return 0
    print("  No new candidates accepted in 2 iterations.")
    print("  Smoke FAILED: at this tiny scale we expect at least one accept.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
