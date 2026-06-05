"""De-risk pilot launcher.

A thin measurement script over the existing Experiment-2 IFBench wiring.
Does NOT change substrate / sampler / acceptance / metric. Skips the
held-out test eval and the D4 drift instrumentation because the
de-risk pilot only needs the loop-internal accept signal:

  - first accept iteration (or "none by N")
  - cumulative accepts at iteration N
  - parent A-batch score trajectory at each accept
  - per-iteration accept log

Outputs (under `results/derisk_pilot[/_smoke]/`):
  - `per_iter.json`       per-iter accept flag + parent_a_sum + new_a_sum
                          + parent / new program idx
  - `summary.json`        first-accept-iter, cumulative accepts, wall, etc.
  - `parent_a_trajectory.json`  parent A-batch sum + new A-batch sum at
                                each accept, read directly from
                                `state.full_program_trace[i]` fields
                                `accept_batch_parent_score` and
                                `accept_batch_new_score` (written by
                                `src/decoupled_proposer.py:275-276`).
  - the standard engine artefacts (`gepa_state.bin`, `shared_rng.pkl`,
    `run_log.txt`, etc.) live in the same dir for inspection.

Usage:
  .venv/bin/python -m src.run_derisk_pilot --n-iter 40
  .venv/bin/python -m src.run_derisk_pilot --n-iter 2 --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from gepa.core.callbacks import GEPACallback, IterationEndEvent

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ifbench_substrate import (
    ifbench_substrate,
    load_ifbench_splits,
    verify_ifbench_difficulty_table_hash,
)
from src.run_gepa import load_lm_configs_from_env, run


class _PerIterWriter(GEPACallback):
    """Flush per-iter rows to disk at the end of every iteration so a
    mid-run crash leaves iters 1..k recoverable on disk.

    Reads `event["state"].full_program_trace` (already populated by the
    DecoupledReflectiveMutationProposer); writes the same row shape the
    runner emits at exit, atomically via `os.replace`."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.out_dir / "per_iter.json"
        self._tmp = self.out_dir / "per_iter.json.tmp"

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        state = event["state"]
        rows: list[dict] = []
        for it_idx, entry in enumerate(state.full_program_trace, start=1):
            new_idx = entry.get("new_program_idx")
            parent_a = entry.get("accept_batch_parent_score")
            new_a = entry.get("accept_batch_new_score")
            rows.append({
                "iter": it_idx,
                "accepted": new_idx is not None,
                "new_program_idx": new_idx,
                "parent_program_idx": entry.get("parent_program_idx"),
                "parent_a_sum": float(parent_a) if parent_a is not None else None,
                "new_a_sum": float(new_a) if new_a is not None else None,
            })
        self._tmp.write_text(json.dumps(rows, indent=2))
        import os
        os.replace(self._tmp, self._path)


def main() -> int:
    parser = argparse.ArgumentParser(description="De-risk pilot: random arm, IFBench, N=N")
    parser.add_argument("--n-iter", type=int, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Write to results/derisk_pilot/_smoke/ instead of /derisk_pilot/")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")

    base = REPO / "results" / "derisk_pilot"
    out_dir = base / "_smoke" if args.smoke else base
    out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())
    config = copy.deepcopy(config)
    config["stopping"]["n"] = int(args.n_iter)

    print(f"De-risk pilot launcher (NO new GEPA wiring; thin runner over Experiment-2 IFBench)")
    print(f"  arm=random, seed=0, N={args.n_iter}, smoke={args.smoke}")
    print(f"  out_dir: {out_dir.relative_to(REPO)}")

    hash_verified = verify_ifbench_difficulty_table_hash()
    print(f"  IFBench frozen-table SHA-256 verified: {hash_verified}")

    splits = load_ifbench_splits(config=config)
    substrate = ifbench_substrate()
    task_lm_config, refl_lm_config = load_lm_configs_from_env(config=config)
    print(f"  task_lm: {task_lm_config.model} @ temp {task_lm_config.temperature}")
    print(f"  refl_lm: {refl_lm_config.model}")
    print(f"  pool sizes: D_feedback={len(splits[0])}, A={len(splits[1])}, "
          f"D_pareto={len(splits[2])}, test={len(splits[3])} (test NOT evaluated)")

    per_iter_writer = _PerIterWriter(out_dir)

    t0 = time.time()
    state = run(
        arm="random",
        seed=0,
        splits=splits,
        difficulty_table=None,  # random arm does not consume the table
        run_dir=out_dir,
        task_lm_config=task_lm_config,
        reflection_lm_config=refl_lm_config,
        config=config,
        raise_on_exception=True,
        substrate=substrate,
        mix=(1.0, 0.0, 0.0),  # ignored on random arm; matches locked Exp-2 variant
        extra_callbacks=[per_iter_writer],
    )
    wall = time.time() - t0

    per_iter: list[dict] = []
    accepted_iters: list[int] = []
    a_trajectory: list[dict] = []
    for it_idx, entry in enumerate(state.full_program_trace, start=1):
        new_idx = entry.get("new_program_idx")
        accepted = new_idx is not None
        parent_a_sum = entry.get("accept_batch_parent_score")
        new_a_sum = entry.get("accept_batch_new_score")
        per_iter.append({
            "iter": it_idx,
            "accepted": bool(accepted),
            "new_program_idx": new_idx,
            "parent_program_idx": entry.get("parent_program_idx"),
            "parent_a_sum": float(parent_a_sum) if parent_a_sum is not None else None,
            "new_a_sum": float(new_a_sum) if new_a_sum is not None else None,
        })
        if accepted:
            accepted_iters.append(it_idx)
            a_trajectory.append({
                "iter": it_idx,
                "parent_a_sum_before_accept": (
                    float(parent_a_sum) if parent_a_sum is not None else None
                ),
                "new_a_sum_at_accept": (
                    float(new_a_sum) if new_a_sum is not None else None
                ),
            })

    summary = {
        "arm": "random",
        "seed": 0,
        "n_iter_target": int(args.n_iter),
        "state_i_final": int(state.i),
        "total_num_evals": int(state.total_num_evals),
        "n_candidates": len(state.program_candidates),
        "first_accept_iter": accepted_iters[0] if accepted_iters else None,
        "cumulative_accepts_at_N": len(accepted_iters),
        "accepted_iters": accepted_iters,
        "wall_time_s": round(wall, 2),
        "smoke": bool(args.smoke),
        "parent_a_trajectory_events": len(a_trajectory),
    }
    (out_dir / "per_iter.json").write_text(json.dumps(per_iter, indent=2))
    (out_dir / "parent_a_trajectory.json").write_text(json.dumps(a_trajectory, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 56)
    print(json.dumps(summary, indent=2))
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
