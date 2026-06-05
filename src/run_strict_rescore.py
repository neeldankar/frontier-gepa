"""Strict-vs-fraction re-score on the same 75 test-split examples as
the v1 difficulty validation, plus a fresh verification of the raw
constraint_count=1 / =2 counts from the IFBench source.

Measurement only. Reuses Experiment-2 IFBench wiring unchanged
(ifbench_substrate + ifbench_program seed + ifbench_feedback.
score_and_feedback). NO new experimental wiring, NO loader change,
NO Experiment 3.

`per_example.json` from the v1 validation pass stores only the
aggregate fraction score (k / N) per id -- no per-verifier
satisfaction booleans. To compute the strict (all-or-nothing) metric
we therefore re-run the same 75 examples (bucket-sample seed 42)
with a recording metric that pulls `satisfied` and `violations` off
`score_and_feedback`. The recording metric returns the fraction
score so dspy.Evaluate's view is identical to the v1 pass.

Outputs under `results/difficulty_validation/`:
  - `per_example_detailed.json`  per-id breakdown (sat/vio ids + both scores)
  - `strict_per_bucket.json`     per-bucket fraction + strict aggregates
  - `strict_summary.json`        top-level summary with wall + provenance
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ifbench_data import load_ifbench_usable
from src.ifbench_feedback import score_and_feedback
from src.ifbench_substrate import (
    ifbench_substrate,
    load_ifbench_splits,
    verify_ifbench_difficulty_table_hash,
)
from src.run_gepa import load_lm_configs_from_env

OUT_DIR = REPO / "results" / "difficulty_validation"
PER_BUCKET_TARGET = 25
NUM_THREADS = 48
BUCKET_SAMPLE_SEED = 42  # IDENTICAL to v1 -- same 75 examples


def _population_std(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    m = sum(scores) / len(scores)
    return (sum((s - m) ** 2 for s in scores) / len(scores)) ** 0.5


def main() -> int:
    load_dotenv(REPO / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())
    h = verify_ifbench_difficulty_table_hash()
    print(f"frozen IFBench difficulty-table SHA-256 verified: {h}")

    # ------------------------------------------------------------------
    # Task 3 prelude: live verification of raw cc=1 / cc=2 counts.
    # `constraint_count_histogram_raw` is populated BEFORE the floor
    # check in `src/ifbench_data.py:252`, so the histogram captures
    # unfiltered counts regardless of the floor argument passed here.
    # We pass floor=1 anyway to make the intent explicit. NO loader
    # change is made.
    # ------------------------------------------------------------------
    print("[task 3] Verifying raw constraint-count histogram from "
          "load_ifbench_usable(constraint_count_floor=1)…")
    usable = load_ifbench_usable(constraint_count_floor=1)
    raw_hist = dict(usable.constraint_count_histogram_raw)
    print(f"  raw n_rows: {usable.n_rows_raw}")
    print(f"  raw histogram: {raw_hist}")
    raw_summary = {
        "n_rows_raw": int(usable.n_rows_raw),
        "constraint_count_histogram_raw": {int(k): int(v) for k, v in raw_hist.items()},
        "cc1_count": int(raw_hist.get(1, 0)),
        "cc2_count": int(raw_hist.get(2, 0)),
        "constraint_count_floor_in_loader": 3,
    }

    # ------------------------------------------------------------------
    # Task 2: strict re-score.
    # ------------------------------------------------------------------
    task_lm_config, _ = load_lm_configs_from_env(config=config)
    print(
        f"task LM: {task_lm_config.model} @ temp {task_lm_config.temperature}; "
        f"num_threads={NUM_THREADS}; cache=False"
    )
    task_lm_kwargs = dict(
        model=task_lm_config.model,
        temperature=task_lm_config.temperature,
        top_p=task_lm_config.top_p,
        max_tokens=task_lm_config.max_tokens,
        num_retries=task_lm_config.num_retries,
        cache=False,
    )
    if task_lm_config.api_key:
        task_lm_kwargs["api_key"] = task_lm_config.api_key
    if task_lm_config.api_base:
        task_lm_kwargs["api_base"] = task_lm_config.api_base
    dspy.settings.configure(lm=dspy.LM(**task_lm_kwargs))

    substrate = ifbench_substrate()
    program = substrate.build_program()

    print("Loading carved splits (test = constraint_count >= 3)…")
    splits = load_ifbench_splits(config=config)
    test_split = splits[3]
    print(f"test split: n={len(test_split)}")

    # Bucket and sample IDENTICAL to v1.
    test_by_cc: dict[int, list] = {3: [], 4: [], 5: []}
    for ex in test_split:
        cc = int(ex["constraint_count"])
        if cc in test_by_cc:
            test_by_cc[cc].append(ex)

    rng = random.Random(BUCKET_SAMPLE_SEED)

    def _take(pool: list, n: int) -> list:
        if len(pool) <= n:
            return list(pool)
        return rng.sample(pool, n)

    buckets: dict[int, list] = {
        cc: _take(test_by_cc[cc], PER_BUCKET_TARGET) for cc in (3, 4, 5)
    }
    for cc in sorted(buckets):
        print(f"  bucket {cc}: {len(buckets[cc])} (target {PER_BUCKET_TARGET})")
    all_examples = []
    for cc in (3, 4, 5):
        all_examples.extend(buckets[cc])
    print(f"total examples to score: {len(all_examples)}")

    # Recording metric: dspy.Evaluate may call this from many threads;
    # we serialise writes with a Lock to avoid torn updates of nested
    # data. Dict insertion under the GIL is fine for the per-id keys but
    # the nested lists below are constructed before the assignment, so
    # the assignment itself is the only contended write -- a Lock just
    # adds safety margin.
    detailed_by_id: dict[int, dict] = {}
    lock = threading.Lock()

    def recording_metric(example, prediction, trace=None):
        out = score_and_feedback(example, prediction)
        n_total = int(example["constraint_count"])
        n_sat = len(out["satisfied"])
        row = {
            "id": int(example["id"]),
            "source_key": str(example["source_key"]),
            "constraint_count": n_total,
            "n_satisfied": int(n_sat),
            "n_violated": int(len(out["violations"])),
            "satisfied_ids": [s["instruction_id"] for s in out["satisfied"]],
            "violated_ids": [v["instruction_id"] for v in out["violations"]],
            "fraction_score": float(out["score"]),
            "strict_score": 1.0 if n_sat == n_total else 0.0,
        }
        with lock:
            detailed_by_id[int(example["id"])] = row
        return float(out["score"])

    t0 = time.time()
    evaluator = dspy.Evaluate(
        devset=all_examples,
        metric=recording_metric,
        num_threads=NUM_THREADS,
        failure_score=0.0,
        provide_traceback=True,
        max_errors=len(all_examples) * 100,
    )
    res = evaluator(program)
    wall = time.time() - t0
    print(f"strict-rescore scoring wall: {wall:.1f}s")

    # Per-bucket aggregates from the detailed side-channel.
    per_bucket: list[dict] = []
    for cc in (3, 4, 5):
        rows = [d for d in detailed_by_id.values() if d["constraint_count"] == cc]
        fracs = [d["fraction_score"] for d in rows]
        strict = [d["strict_score"] for d in rows]
        per_bucket.append({
            "constraint_count": cc,
            "n": len(rows),
            "fraction_mean": (sum(fracs) / len(fracs)) if fracs else None,
            "fraction_std": _population_std(fracs) if fracs else None,
            "strict_mean": (sum(strict) / len(strict)) if strict else None,
            "strict_std": _population_std(strict) if strict else None,
            "strict_n_all_satisfied": int(sum(int(s == 1.0) for s in strict)),
        })

    summary = {
        "n_examples_total": len(all_examples),
        "per_bucket_target": PER_BUCKET_TARGET,
        "num_threads": NUM_THREADS,
        "bucket_sample_seed": BUCKET_SAMPLE_SEED,
        "wall_time_s": round(wall, 2),
        "frozen_table_sha256": h,
        "task_model": task_lm_config.model,
        "task_temperature": task_lm_config.temperature,
        "buckets": per_bucket,
        "raw_constraint_count_histogram": raw_summary,
    }

    detailed_sorted = sorted(detailed_by_id.values(), key=lambda r: r["id"])
    (OUT_DIR / "per_example_detailed.json").write_text(
        json.dumps(detailed_sorted, indent=2)
    )
    (OUT_DIR / "strict_per_bucket.json").write_text(
        json.dumps(per_bucket, indent=2)
    )
    (OUT_DIR / "strict_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("=" * 56)
    print(json.dumps(summary, indent=2))
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
